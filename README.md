# Whisper Persian

Fine-tune Whisper for Persian speech recognition, then actually use it — on long recordings, video
files, YouTube links and batch jobs — from one repository.

The project has two halves:

| | What it does | Entry point |
|---|---|---|
| **Transcription** | Full-featured WebUI and CLI: VAD, long-form audio, video, URLs, speaker diarization, SRT/VTT/TXT/JSON output, multi-GPU | `app.py`, `cli.py` |
| **Training** | Fine-tune Whisper on the FLEURS Farsi dataset, evaluate WER, quantize and publish to the Hub | `training/` |

The transcription half is a fork of [whisper-webui](https://gitlab.com/aadnk/whisper-webui) by
Kristian S. Stangeland, extended with a **`transformers` backend** so the fine-tuned Persian models
load directly from HuggingFace — no `.pt` or CTranslate2 conversion step. See [NOTICE](NOTICE).

## 🚀 Quick Start

```bash
./setup.sh
python app.py
```

Then open <http://localhost:7860>, pick **Persian Small** or **Persian Large v3**, drop in a file,
and press Submit.

ffmpeg must be on your PATH — everything decodes audio through it.

### Manual setup

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
cp .env.example .env        # only needed for publishing models to the Hub
```

## 🎧 Transcribing

### WebUI

```bash
python app.py
```

Three tabs:

- **Simple** — model, language, file/URL/microphone, VAD, diarization.
- **Full** — everything above plus decoding parameters (beam size, temperature fallback, thresholds,
  initial prompt, word timestamps).
- **Extra** — re-run diarization or word highlighting over an existing JSON/SRT without transcribing again.

Variants: `app-local.py` (no length limit), `app-network.py` (binds `0.0.0.0`), `app-shared.py`
(public Gradio link).

### CLI

```bash
# One file
python cli.py meeting.mp4 --model "Persian Small" --output_dir ./out

# A whole directory, on the GPU, in bf16
python cli.py recordings/*.mp3 --model "Persian Large v3" --compute_type bfloat16 --vad silero-vad

# Straight from YouTube
python cli.py "https://www.youtube.com/watch?v=..." --model "Persian Small"
```

Each input produces `-subs.srt`, `-subs.vtt`, `-transcript.txt` and `-result.json`.

### Why long files work

Whisper's encoder takes a fixed 30-second window, so a naive `pipe(audio)` call silently truncates
anything longer. This project handles length in two layers:

1. **Silero VAD** splits the audio on speech boundaries and feeds Whisper one utterance at a time,
   carrying the previous text forward as a prompt. This is the `--vad silero-vad` default and is
   what you want for anything over a few minutes.
2. The `transformers` backend then uses **sequential long-form decoding** for whatever it receives,
   so even with VAD off a two-hour file is transcribed in full.

There is no upper bound on input length — `input_audio_max_duration` is set to `-1` in
`config.json5`.

## 🧠 Backends

Set with `--whisper_implementation` or `whisper_implementation` in `config.json5`.

| Backend | Model format | Install | Use it when |
|---|---|---|---|
| **`transformers`** (default) | HuggingFace checkpoints | `requirements.txt` | You want the fine-tuned Persian models, or any HF Whisper checkpoint, with no conversion |
| `whisper` | OpenAI `.pt` | `requirements-whisper.txt` | You want the reference implementation; HF models are auto-converted on first use |
| `faster-whisper` | CTranslate2 | `requirements-fasterWhisper.txt` | Maximum GPU throughput, and you don't mind converting first |

To use a Persian model with `faster-whisper`, convert it once:

```bash
ct2-transformers-converter --model AmirMohseni/whisper-small-persian --output_dir ./whisper-small-persian-ct2 --quantization float16
```

then add the output directory to `config.json5`.

### Adding your own model

Append an entry to `models` in `config.json5` — a Hub id or a local directory both work:

```json5
{
    "name": "Persian Small (mine)",
    "url": "./whisper-small-persian",   // or "your-username/whisper-small-persian"
    "type": "transformers",
    "language": "Persian",
    // 0 = sequential long-form decoding (most accurate).
    // Set to 30 with a batch_size for faster, slightly less accurate chunked decoding on GPU.
    "chunk_length_s": 0,
    "batch_size": 1,
}
```

### Precision

`--compute_type` accepts `auto` (float16 on GPU, float32 on CPU), `float32`, `float16`, `bfloat16`
and `int8`. The published Persian models are already bf16; `int8` loads through bitsandbytes and
needs a CUDA GPU.

## 🎯 Training

```bash
python training/whisper_trainer.py
```

Customize via `training/configs.yaml`:

```yaml
dataset:
  name: "MohammadGholizadeh/fleurs-farsi"
  train_split: "train[:50%]"
  test_split: "dev[:50%]"

training:
  learning_rate: 3e-5
  per_device_train_batch_size: 32
  max_steps: 500
  warmup_steps: 100

output:
  output_dir: "./whisper-small-persian"
  push_to_hub: true
```

**Key parameters:**
- `learning_rate` — controls training speed (try 1e-5 to 5e-5)
- `max_steps` — training duration (500 steps ≈ 2 hours on an L4)
- `per_device_train_batch_size` — memory vs. speed trade-off
- `push_to_hub` — set to `false` for local-only training

Publishing to the Hub needs a write token in `.env`:

```bash
HF_TOKEN=your_actual_token_here
```

Get one from [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens).

### Results

The model reaches **25.8% WER** on the FLEURS Farsi evaluation set.

| Step | Training Loss | Validation Loss | WER    | Epoch |
|------|---------------|-----------------|--------|-------|
| 50   | 0.3277       | 0.3894         | 34.89% | 2.0   |
| 100  | 0.1146       | 0.3268         | 28.63% | 4.0   |
| 150  | 0.0373       | **0.3289**     | 27.34% | 6.0   |
| 200  | 0.0142       | 0.3390         | 27.54% | 8.0   |
| 250  | 0.0036       | 0.3523         | 26.53% | 10.0  |
| 300  | 0.0024       | 0.3677         | 26.53% | 12.0  |
| 350  | 0.0010       | 0.3734         | 26.50% | 14.0  |
| 400  | 0.0007       | 0.3777         | 26.20% | 16.0  |
| 450  | 0.0007       | 0.3807         | 25.99% | 18.0  |
| **500** | **0.0006**   | **0.3818**     | **25.82%** | **20.0** |

## 🔧 Quantization

```bash
# 8-bit, saved locally
python training/quantize_and_push.py \
    --model_id "AmirMohseni/whisper-small-persian" \
    --precision "int8" \
    --output_dir "./whisper-small-persian-int8"

# bfloat16, pushed to the Hub
python training/quantize_and_push.py \
    --model_id "AmirMohseni/whisper-small-persian" \
    --precision "bf16" \
    --output_dir "./whisper-small-persian-bf16" \
    --push_to_hub \
    --hub_model_id "your-username/whisper-small-persian-bf16"
```

| Precision | Memory Reduction | Speed | Quality Loss | Use Case |
|-----------|------------------|-------|--------------|----------|
| **bf16** | ~50% | Fast | Medium | Best balance for most users |
| **int8** | ~75% | Fast | High | Maximum memory savings |

The output directory can be dropped straight into `config.json5` as a `"transformers"` model.

## 🗣️ Speaker diarization

Optional, and off by default. It needs extra libraries and a HuggingFace token that has accepted the
`pyannote/speaker-diarization` model terms:

```bash
pip install -r src/diarization/requirements.txt
python app.py --auth_token YOUR_TOKEN --diarization True
```

The UI disables the diarization checkboxes automatically when the libraries are missing.

## 📦 Docker

```bash
docker build -t whisper-persian .
docker run -d --gpus all -p 7860:7860 -v whisper-cache:/root/.cache whisper-persian
```

Mount `/root/.cache` so the models are downloaded only once.

## 🧩 Other entry points

- `colab_webui.ipynb` — run the whole thing on a free Colab GPU, no local install. See
  [docs/colab.md](docs/colab.md).
- `simple_app.py` — a single-file Gradio app with no VAD, no URL download and no diarization. Useful
  as a minimal example or on machines where you don't want the full dependency set.

## 🔍 Troubleshooting

**Only the first 30 seconds are transcribed**
You are running a bare `pipeline(...)` call somewhere instead of `app.py`/`cli.py`. Whisper's encoder
is fixed at 30 seconds; long audio needs VAD or long-form decoding, which both entry points do by default.

**CUDA out of memory during transcription**
Use a smaller model, `--compute_type int8`, or lower `vad_max_merge_size`.

**CUDA out of memory during training**
```yaml
training:
  per_device_train_batch_size: 16  # or 8
  gradient_accumulation_steps: 4    # increase accordingly
```

**Word timestamps produce a warning**
Not every fine-tuned checkpoint ships the cross-attention alignment heads that word-level timestamps
need. The backend falls back to segment-level timestamps and says so in the log.

**Poor accuracy**
- Try `Persian Large v3` instead of `Persian Small`
- Keep VAD enabled — it prevents the model from drifting on long silences
- Put a few domain words in the **Initial Prompt** field
- For training: more steps, more data (drop the `[:50%]` splits), different learning rates

## 📄 License

Apache License 2.0 — see [LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
