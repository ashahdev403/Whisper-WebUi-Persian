"""
A Whisper backend that runs HuggingFace Transformers checkpoints directly.

The `whisper` backend needs an OpenAI .pt checkpoint and the `faster-whisper` backend needs a
CTranslate2 directory, so both require a conversion step before a fine-tuned HF model can be used.
This backend loads `WhisperForConditionalGeneration` as-is, which means the Persian models produced
by `training/whisper_trainer.py` and `training/quantize_and_push.py` (fp32/bf16/fp16/int8) can be
selected in the UI without any intermediate artifacts.
"""

import os
from typing import List

import ffmpeg
import numpy as np
import torch

from src.config import ModelConfig
from src.hooks.progressListener import ProgressListener
from src.languages import get_language_from_name
from src.modelCache import ModelCache
from src.prompts.abstractPromptStrategy import AbstractPromptStrategy
from src.whisper.abstractWhisperContainer import AbstractWhisperCallback, AbstractWhisperContainer

SAMPLE_RATE = 16000

# Sentence-final punctuation used to group word timestamps back into subtitle-sized segments.
# Includes the Persian full stop, question mark and comma.
SENTENCE_ENDINGS = (".", "!", "?", "。", "؟", "۔", "!", "?")

# Fall back to a new segment after this many seconds even without punctuation
MAX_WORD_SEGMENT_DURATION = 8.0


class LoadedTransformersModel:
    """Bundles the objects a callback needs - the pipeline plus the processor for prompt encoding."""

    def __init__(self, pipe, processor, torch_dtype, device: str):
        self.pipe = pipe
        self.processor = processor
        self.torch_dtype = torch_dtype
        self.device = device


class TransformersWhisperContainer(AbstractWhisperContainer):
    def __init__(self, model_name: str, device: str = None, compute_type: str = "float16",
                 download_root: str = None,
                 cache: ModelCache = None, models: List[ModelConfig] = []):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_name, device, compute_type, download_root, cache, models)

    def _get_model_config(self) -> ModelConfig:
        for model in self.models:
            if model.name == self.model_name:
                return model
        return None

    def _get_model_path(self) -> str:
        model_config = self._get_model_config()

        if model_config is None:
            # Not in the config - assume the name is a HuggingFace repository or a local directory
            return self.model_name
        return model_config.path if model_config.path is not None else model_config.url

    def ensure_downloaded(self):
        """
        Ensure that the model is downloaded, so that a subprocess doesn't have to do it.
        """
        model_path = self._get_model_path()

        if os.path.isdir(model_path):
            return True

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(model_path, cache_dir=self.download_root)
            return True
        except Exception as e:
            print("Error pre-downloading model " + model_path + ": " + str(e))
            return False

    def _resolve_dtype(self):
        """
        Map the shared `compute_type` option onto a torch dtype. Returns (dtype, quantization_config).
        """
        compute_type = (self.compute_type or "auto").lower()
        is_cuda = str(self.device).startswith("cuda")

        if compute_type in ["int8", "int8_float16"]:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["proj_out"])
                return (torch.float16 if is_cuda else torch.float32), quantization_config
            except Exception as e:
                print("WARNING: could not enable int8 quantization (" + str(e) + ") - falling back to float16/float32.")
                return (torch.float16 if is_cuda else torch.float32), None

        if compute_type in ["float16", "fp16"]:
            if not is_cuda:
                # float16 matmuls are unimplemented or extremely slow on CPU
                print("WARNING: float16 is not supported on CPU - using float32 instead.")
                return torch.float32, None
            return torch.float16, None

        if compute_type in ["bfloat16", "bf16"]:
            return torch.bfloat16, None

        if compute_type in ["float32", "fp32", "int16"]:
            return torch.float32, None

        # "auto" / "default" / anything else
        return (torch.float16 if is_cuda else torch.float32), None

    def _create_model(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model_path = self._get_model_path()
        torch_dtype, quantization_config = self._resolve_dtype()
        dtype_kwarg = _dtype_kwarg_name()

        print("Loading transformers whisper model " + model_path + " on " + str(self.device) +
              " (" + str(torch_dtype).replace("torch.", "") + ")")

        model_kwargs = {
            dtype_kwarg: torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if self.download_root is not None:
            model_kwargs["cache_dir"] = self.download_root
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = self.device

        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            model_path, cache_dir=self.download_root
        )

        if quantization_config is None:
            model = model.to(self.device)
        model.eval()

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            # bitsandbytes models are already placed by accelerate
            device=None if quantization_config is not None else self.device,
            **{dtype_kwarg: torch_dtype},
        )

        return LoadedTransformersModel(pipe, processor, torch_dtype, str(self.device))

    def create_callback(self, language: str = None, task: str = None,
                        prompt_strategy: AbstractPromptStrategy = None,
                        **decodeOptions: dict) -> AbstractWhisperCallback:
        return TransformersWhisperCallback(self, language=language, task=task,
                                           prompt_strategy=prompt_strategy, **decodeOptions)


class TransformersWhisperCallback(AbstractWhisperCallback):
    def __init__(self, model_container: TransformersWhisperContainer, language: str = None, task: str = None,
                 prompt_strategy: AbstractPromptStrategy = None,
                 **decodeOptions: dict):
        self.model_container = model_container
        self.language = language
        self.task = task
        self.prompt_strategy = prompt_strategy
        self.decodeOptions = decodeOptions

        self._printed_word_timestamp_warning = False

    def invoke(self, audio, segment_index: int, prompt: str, detected_language: str,
               progress_listener: ProgressListener = None):
        loaded: LoadedTransformersModel = self.model_container.get_model()

        audio_array = _as_audio_array(audio)
        audio_duration = len(audio_array) / SAMPLE_RATE

        initial_prompt = self.prompt_strategy.get_segment_prompt(segment_index, prompt, detected_language) \
                           if self.prompt_strategy else prompt

        language_code = self._resolve_language(detected_language)
        decodeOptions = dict(self.decodeOptions)

        verbose = decodeOptions.pop("verbose", None)
        word_timestamps = bool(decodeOptions.pop("word_timestamps", False))
        model_config = self.model_container._get_model_config()

        return_timestamps = "word" if word_timestamps else True

        generate_kwargs = self._build_generate_kwargs(loaded, language_code, initial_prompt, decodeOptions)
        call_kwargs = self._build_call_kwargs(model_config)

        try:
            output = self._run_pipeline(loaded, audio_array, return_timestamps, generate_kwargs, call_kwargs)
        except Exception as e:
            if return_timestamps == "word":
                # Word timestamps need alignment heads, which not every fine-tune ships with
                if not self._printed_word_timestamp_warning:
                    print("WARNING: word timestamps are unavailable for this model (" + str(e) +
                          ") - falling back to segment timestamps.")
                    self._printed_word_timestamp_warning = True
                return_timestamps = True
                output = self._run_pipeline(loaded, audio_array, return_timestamps, generate_kwargs, call_kwargs)
            else:
                raise

        text = (output.get("text") or "").strip()
        chunks = output.get("chunks") or []

        if return_timestamps == "word":
            segments = _group_words_into_segments(chunks, audio_duration)
        else:
            segments = _chunks_to_segments(chunks, audio_duration)

        if len(segments) == 0 and len(text) > 0:
            segments = [{"text": text, "start": 0.0, "end": audio_duration, "words": []}]

        if verbose:
            for segment in segments:
                print("[{:.3f}->{:.3f}] {}".format(segment["start"], segment["end"], segment["text"]))

        result = {
            "segments": segments,
            "text": text,
            "language": language_code if language_code else detected_language,
            "duration": audio_duration,
        }

        if self.prompt_strategy:
            self.prompt_strategy.on_segment_finished(segment_index, prompt, detected_language, result)

        if progress_listener is not None:
            progress_listener.on_progress(audio_duration, audio_duration)
            progress_listener.on_finished()
        return result

    def _run_pipeline(self, loaded: LoadedTransformersModel, audio_array, return_timestamps,
                      generate_kwargs: dict, call_kwargs: dict):
        with torch.no_grad():
            return loaded.pipe(
                {"raw": audio_array, "sampling_rate": SAMPLE_RATE},
                return_timestamps=return_timestamps,
                generate_kwargs=generate_kwargs,
                **call_kwargs,
            )

    def _build_call_kwargs(self, model_config: ModelConfig) -> dict:
        """
        Chunking is a per-model choice: 0 (the default) uses Whisper's sequential long-form decoding,
        which is the most accurate, while a positive value trades a little accuracy for parallelism.
        """
        call_kwargs = {}

        chunk_length_s = getattr(model_config, "chunk_length_s", 0) if model_config else 0
        batch_size = getattr(model_config, "batch_size", 1) if model_config else 1

        if chunk_length_s and chunk_length_s > 0:
            call_kwargs["chunk_length_s"] = chunk_length_s
            if batch_size and batch_size > 1:
                call_kwargs["batch_size"] = batch_size

        return call_kwargs

    def _build_generate_kwargs(self, loaded: LoadedTransformersModel, language_code: str,
                               initial_prompt: str, decodeOptions: dict) -> dict:
        """
        Translate the OpenAI Whisper decoding options used by the UI into `generate()` arguments.
        Options without a Transformers equivalent (best_of, patience, fp16, punctuation lists) are
        dropped rather than passed through, since `generate()` rejects unknown keyword arguments.
        """
        generate_kwargs = {}

        if language_code:
            generate_kwargs["language"] = language_code
        if self.task:
            generate_kwargs["task"] = self.task

        temperature = decodeOptions.get("temperature", None)
        if temperature is not None:
            if isinstance(temperature, (list, tuple)):
                values = tuple(float(t) for t in temperature)
                generate_kwargs["temperature"] = values if len(values) > 1 else values[0]
            else:
                generate_kwargs["temperature"] = float(temperature)

        beam_size = decodeOptions.get("beam_size", None)
        if beam_size is not None and int(beam_size) > 1:
            generate_kwargs["num_beams"] = int(beam_size)

        length_penalty = decodeOptions.get("length_penalty", None)
        if length_penalty is not None:
            generate_kwargs["length_penalty"] = float(length_penalty)

        condition_on_previous_text = decodeOptions.get("condition_on_previous_text", None)
        if condition_on_previous_text is not None:
            generate_kwargs["condition_on_prev_tokens"] = bool(condition_on_previous_text)

        for source_name, target_name in [("compression_ratio_threshold", "compression_ratio_threshold"),
                                         ("logprob_threshold", "logprob_threshold"),
                                         ("no_speech_threshold", "no_speech_threshold")]:
            value = decodeOptions.get(source_name, None)
            if value is not None:
                generate_kwargs[target_name] = float(value)

        suppress_tokens = _parse_suppress_tokens(decodeOptions.get("suppress_tokens", None))
        if suppress_tokens is not None:
            generate_kwargs["suppress_tokens"] = suppress_tokens

        if initial_prompt:
            try:
                prompt_ids = loaded.processor.get_prompt_ids(initial_prompt, return_tensors="pt")
                generate_kwargs["prompt_ids"] = prompt_ids.to(loaded.device)
            except Exception as e:
                print("WARNING: could not encode the initial prompt (" + str(e) + ") - ignoring it.")

        return generate_kwargs

    def _resolve_language(self, detected_language: str) -> str:
        """
        The UI passes full language names ("Persian"); `generate()` wants a code ("fa").
        """
        name = self.language if self.language else detected_language

        if not name:
            return None

        language = get_language_from_name(name)
        return language.code if language is not None else name


def _dtype_kwarg_name() -> str:
    """
    Transformers 5 renamed the `torch_dtype` argument to `dtype` and deprecated the old spelling.
    """
    import transformers

    try:
        major = int(str(transformers.__version__).split(".")[0])
    except ValueError:
        return "torch_dtype"

    return "dtype" if major >= 5 else "torch_dtype"


def _as_audio_array(audio) -> np.ndarray:
    """
    The VAD hands over float32 numpy segments, while the no-VAD path hands over a file path.
    """
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32)
    if torch.is_tensor(audio):
        return audio.detach().cpu().numpy().astype(np.float32)
    return _load_audio(str(audio))


def _load_audio(file: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    try:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}")

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def _parse_suppress_tokens(suppress_tokens):
    """
    "-1" is Whisper's "use the default suppression list", which Transformers already does.
    """
    if suppress_tokens is None:
        return None
    if isinstance(suppress_tokens, list):
        return suppress_tokens if len(suppress_tokens) > 0 else None

    tokens = [token.strip() for token in str(suppress_tokens).split(",") if token.strip()]

    if len(tokens) == 0 or tokens == ["-1"]:
        return None

    try:
        return [int(token) for token in tokens]
    except ValueError:
        print("WARNING: could not parse suppress_tokens '" + str(suppress_tokens) + "' - ignoring it.")
        return None


def _chunks_to_segments(chunks: List[dict], audio_duration: float) -> List[dict]:
    """
    Convert pipeline chunks into Whisper segments, filling in the open-ended timestamps that
    long-form decoding can leave behind on the final chunk.
    """
    segments = []

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()

        if len(text) == 0:
            continue

        start, end = _chunk_timestamp(chunk)

        if start is None:
            start = segments[-1]["end"] if len(segments) > 0 else 0.0
        if end is None or end <= start:
            end = audio_duration if audio_duration > start else start + 5.0

        segments.append({
            "text": text,
            "start": float(start),
            "end": float(end),
            "words": [],
        })

    return segments


def _group_words_into_segments(chunks: List[dict], audio_duration: float) -> List[dict]:
    """
    With word timestamps the pipeline returns one chunk per word, which would produce a subtitle
    per word. Group them back into sentence-sized segments that still carry their word list.
    """
    segments = []
    current_words = []

    def flush():
        if len(current_words) == 0:
            return
        text = "".join(word["word"] for word in current_words).strip()

        if len(text) > 0:
            segments.append({
                "text": text,
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "words": list(current_words),
            })
        current_words.clear()

    for chunk in chunks:
        word_text = chunk.get("text") or ""

        if len(word_text.strip()) == 0:
            continue

        start, end = _chunk_timestamp(chunk)

        if start is None:
            start = current_words[-1]["end"] if len(current_words) > 0 else \
                    (segments[-1]["end"] if len(segments) > 0 else 0.0)
        if end is None or end < start:
            end = audio_duration if audio_duration > start else start

        current_words.append({
            "start": float(start),
            "end": float(end),
            "word": word_text,
            "probability": 1.0,
        })

        stripped = word_text.strip()
        segment_duration = current_words[-1]["end"] - current_words[0]["start"]

        if stripped.endswith(SENTENCE_ENDINGS) or segment_duration >= MAX_WORD_SEGMENT_DURATION:
            flush()

    flush()
    return segments


def _chunk_timestamp(chunk: dict):
    timestamp = chunk.get("timestamp") or (None, None)

    if len(timestamp) < 2:
        return None, None
    return timestamp[0], timestamp[1]
