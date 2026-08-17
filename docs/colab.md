# Running on Google Colab

Use **`colab_webui.ipynb`** in the repository root — open it directly:

```
https://colab.research.google.com/github/ashahdev403/Whisper-WebUi-Persian/blob/main/colab_webui.ipynb
```

Set the runtime to a GPU first (**Runtime → Change runtime type → Hardware accelerator → T4 GPU**),
then run the cells in order. The notebook checks the GPU, clones the project, installs the
dependencies, converts the Persian models to CTranslate2, writes a config and launches the WebUI on
a public Gradio link.

This page only records the decisions behind that notebook, so it does not drift out of sync with it.

## Why it does not run `pip install -r requirements.txt`

Colab already ships torch, torchaudio, numpy and ffmpeg, built against its own CUDA. Installing the
requirements file reinstalls torch and can leave the runtime without working CUDA. The notebook
installs only the packages Colab is missing.

## Why the models are converted

The notebook uses the `faster-whisper` backend, which reads **CTranslate2** models only. The Persian
models are HuggingFace checkpoints, so they are converted once per session with
`ct2-transformers-converter`. `float16` is the right quantization on a GPU; `int8` is smaller and
faster but less accurate.

To skip conversion entirely, set `"whisper_implementation": "transformers"` in the config cell and
point the model URLs back at the `AmirMohseni/...` repositories. That backend runs HuggingFace
checkpoints directly, at some cost in GPU speed.

## Why everything goes through a config file

`app-local.py`, `app-network.py` and `app-shared.py` are presets — they call `create_ui` directly and
**do not parse command line arguments**, so `python app-shared.py --compute_type float16` silently
ignores the flag. The notebook writes `/content/config.colab.json5`, points `WHISPER_WEBUI_CONFIG`
at it and runs `app.py`, which keeps the backend, compute type, model list and `share` setting in one
visible place. The repository's own `config.json5` is left untouched.

## Things that bite

- **Nothing persists.** `/content/` and the HuggingFace cache are wiped when the runtime resets, so
  the conversion runs again next session. Mount Drive and convert into a Drive path to avoid it.
- **The share link is public** for 72 hours or until the cell is stopped. Set `"share": False` if the
  audio is sensitive.
- **Idle runtimes get reclaimed.** Runtime → Manage sessions → terminate when you are done.
