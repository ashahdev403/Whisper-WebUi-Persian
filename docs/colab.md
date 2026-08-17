# Running on Google Colab

If you don't have a GPU locally, Colab is the easiest way to run the full WebUI against the Persian
models. Set the runtime to a GPU first: **Runtime → Change runtime type → Hardware accelerator → GPU**.

Then run these three cells.

**1. Check out the project**

```python
!git clone --branch feature/merge-whisper-webui https://github.com/ashahdev403/Whisper-WebUi-Persian.git
%cd Whisper-WebUi-Persian
```

(Drop the `--branch` flag once the merge has landed on `main`.)

**2. Install dependencies**

Do *not* run `pip install -r requirements.txt` here - it reinstalls torch and can break Colab's CUDA
build. Colab already ships torch, torchaudio, numpy and ffmpeg, so install only the rest:

```python
!pip install -q "transformers>=4.48.0" accelerate gradio json5 ffmpeg-python yt-dlp more-itertools altair intervaltree srt
```

**3. Launch the WebUI with a public link**

```python
!python app-shared.py
```

Click the `https://xxxxx.gradio.live` URL that appears next to "Running on public URL". The link
expires after 72 hours.

Notes:

- The first transcription downloads the selected model (~500 MB for Persian Small, ~3 GB for
  Persian Large v3), so it takes a minute before anything appears.
- `app-shared.py` removes the audio length limit, so long recordings work.
- Colab disconnects idle sessions. Go to **Runtime → Manage sessions** and terminate the session
  when you are done, otherwise it keeps consuming your free compute.

## Ready-made notebook

`colab_webui.ipynb` in the repository root already contains all of the above, plus a GPU check, a
config sanity check and a CLI test cell. Open it directly in Colab instead of pasting the cells by
hand.
