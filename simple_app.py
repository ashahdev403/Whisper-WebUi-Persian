import json
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from transformers import pipeline
import gradio as gr

MODELS = {
    "Persian Small": "AmirMohseni/whisper-small-persian-bf16",
    "Persian Large v3": "AmirMohseni/whisper-large-v3-persian-bf16",
}

MEDIA_EXTENSIONS = [
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".mpg", ".mpeg",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
]

OUTPUT_ROOT = Path(tempfile.gettempdir()) / "whisper-persian-output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

HISTORY_HEADERS = ["Time", "Source", "Model", "Duration", "Segments", "Characters"]

model_pipelines = {}


def get_pipeline(model_choice):
    if model_choice is None or model_choice not in MODELS:
        model_choice = "Persian Small"
    model_id = MODELS[model_choice]
    if model_id not in model_pipelines:
        print(f"Loading model: {model_id}...")
        model_pipelines[model_id] = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=True,
            generate_kwargs={"language": "persian", "task": "transcribe"},
        )
        print("Model loaded.")
    return model_pipelines[model_id]


def resolve_path(value):
    """Gradio has returned plain paths, dicts and file objects across versions."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "name", "video", "url"):
            if value.get(key):
                return str(value[key])
        return None
    for attr in ("path", "name"):
        if hasattr(value, attr):
            return getattr(value, attr)
    return str(value)


def probe_duration(media_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", media_path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio(media_path, dest_dir):
    """Pull a 16 kHz mono WAV out of any container ffmpeg understands."""
    audio_path = dest_dir / "audio.wav"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", media_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read this file:\n{result.stderr[-800:]}")
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("No audio track found in this file.")
    return audio_path


def format_hms(seconds, decimal_separator):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{decimal_separator}{millis:03d}"


def normalize_chunks(result, fallback_duration):
    """Turn pipeline output into (start, end, text) triples with no gaps in timing."""
    chunks = result.get("chunks") or []
    cues = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        timestamp = chunk.get("timestamp") or (None, None)
        start, end = timestamp[0], timestamp[1]
        if start is None:
            start = cues[-1][1] if cues else 0.0
        if end is None or end <= start:
            end = start + 5.0
        cues.append([float(start), float(end), text])

    if not cues:
        text = (result.get("text") or "").strip()
        if text:
            cues.append([0.0, fallback_duration or 5.0, text])
    return cues


def write_outputs(cues, full_text, dest_dir, base_name, model_choice, source_name):
    txt_path = dest_dir / f"{base_name}.txt"
    srt_path = dest_dir / f"{base_name}.srt"
    vtt_path = dest_dir / f"{base_name}.vtt"
    json_path = dest_dir / f"{base_name}.json"

    txt_path.write_text(full_text.strip() + "\n", encoding="utf-8")

    srt_blocks = []
    for index, (start, end, text) in enumerate(cues, 1):
        srt_blocks.append(
            f"{index}\n{format_hms(start, ',')} --> {format_hms(end, ',')}\n{text}\n"
        )
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    vtt_blocks = ["WEBVTT", ""]
    for start, end, text in cues:
        vtt_blocks.append(f"{format_hms(start, '.')} --> {format_hms(end, '.')}")
        vtt_blocks.append(text)
        vtt_blocks.append("")
    vtt_path.write_text("\n".join(vtt_blocks), encoding="utf-8")

    json_path.write_text(
        json.dumps(
            {
                "source": source_name,
                "model": model_choice,
                "model_id": MODELS.get(model_choice),
                "text": full_text.strip(),
                "segments": [
                    {"start": start, "end": end, "text": text}
                    for start, end, text in cues
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return [str(txt_path), str(srt_path), str(vtt_path), str(json_path)]


def collect_sources(uploaded_files, microphone):
    sources = []
    if uploaded_files:
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        for item in uploaded_files:
            path = resolve_path(item)
            if path:
                sources.append((path, Path(path).name))
    mic_path = resolve_path(microphone)
    if mic_path:
        sources.append((mic_path, "microphone-recording"))
    return sources


def safe_stem(name):
    stem = Path(name).stem
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return cleaned.strip("_") or "transcript"


def transcribe(uploaded_files, microphone, model_choice, history, progress=gr.Progress()):
    history = list(history or [])
    sources = collect_sources(uploaded_files, microphone)

    if not sources:
        return (
            "Add a video or audio file, or record from the microphone, then press Transcribe.",
            "",
            None,
            history,
            history_rows(history),
        )

    run_dir = OUTPUT_ROOT / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    pipe = get_pipeline(model_choice)

    all_files = []
    transcripts = []
    segment_previews = []
    errors = []

    for index, (path, display_name) in enumerate(sources):
        progress((index) / len(sources), desc=f"Transcribing {display_name}")

        item_dir = run_dir / f"{index + 1:02d}_{safe_stem(display_name)}"
        item_dir.mkdir(parents=True, exist_ok=True)

        try:
            duration = probe_duration(path)
            audio_path = extract_audio(path, item_dir)
            result = pipe(str(audio_path))
            os.remove(audio_path)

            cues = normalize_chunks(result, duration)
            full_text = (result.get("text") or "").strip()

            files = write_outputs(
                cues, full_text, item_dir, safe_stem(display_name),
                model_choice, display_name,
            )
            all_files.extend(files)

            header = display_name if len(sources) > 1 else ""
            transcripts.append(f"{header}\n{full_text}".strip() if header else full_text)
            segment_previews.append(
                "\n".join(
                    f"[{format_hms(start, '.')} -> {format_hms(end, '.')}] {text}"
                    for start, end, text in cues
                )
            )

            history.append([
                datetime.now().strftime("%H:%M:%S"),
                display_name,
                model_choice,
                format_hms(duration, "."),
                len(cues),
                len(full_text),
            ])
        except Exception as exc:
            errors.append(f"{display_name}: {exc}")

    if len(all_files) > 1:
        zip_path = run_dir / "all_transcripts.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file_path in all_files:
                archive.write(file_path, arcname=Path(file_path).name)
        all_files.insert(0, str(zip_path))

    progress(1.0, desc="Done")

    text_output = "\n\n".join(t for t in transcripts if t)
    if errors:
        text_output = (text_output + "\n\n" if text_output else "") + \
            "Problems:\n" + "\n".join(f"  - {e}" for e in errors)

    return (
        text_output or "Nothing was transcribed.",
        "\n\n".join(p for p in segment_previews if p),
        all_files or None,
        history,
        history_rows(history),
    )


def history_rows(history):
    return gr.update(value=history or [])


def clear_history():
    return [], gr.update(value=[]), None


with gr.Blocks(title="Whisper Farsi") as demo:
    gr.Markdown(
        "# Whisper Farsi\n"
        "Persian speech recognition for video and audio. Upload files or record from the "
        "microphone, then download the transcript as TXT, SRT, VTT or JSON."
    )

    history_state = gr.State([])

    model = gr.Radio(
        choices=list(MODELS.keys()),
        value="Persian Small",
        label="Model",
        info="Persian Small is faster. Persian Large v3 is more accurate.",
    )

    with gr.Tabs():
        with gr.TabItem("Transcribe"):
            with gr.Row():
                with gr.Column():
                    files_input = gr.File(
                        label="Video or audio files",
                        file_count="multiple",
                        file_types=MEDIA_EXTENSIONS,
                    )
                with gr.Column():
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Or record audio",
                    )

            transcribe_btn = gr.Button("Transcribe", variant="primary")

            output_text = gr.Textbox(label="Transcript", lines=10)

            with gr.Accordion("Timed segments", open=False):
                segments_output = gr.Textbox(label="Segments", lines=12)

            downloads = gr.File(
                label="Download transcripts (TXT / SRT / VTT / JSON)",
                file_count="multiple",
                interactive=False,
            )

        with gr.TabItem("Library"):
            gr.Markdown(
                "Everything transcribed in this session. Files stay available in the "
                "Transcribe tab until you clear them."
            )
            history_table = gr.Dataframe(
                headers=HISTORY_HEADERS,
                datatype=["str", "str", "str", "str", "number", "number"],
                value=[],
                interactive=False,
                wrap=True,
            )
            clear_btn = gr.Button("Clear library")

    transcribe_btn.click(
        fn=transcribe,
        inputs=[files_input, mic_input, model, history_state],
        outputs=[output_text, segments_output, downloads, history_state, history_table],
    )

    clear_btn.click(
        fn=clear_history,
        inputs=None,
        outputs=[history_state, history_table, downloads],
    )


if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("PORT", 7860)))
