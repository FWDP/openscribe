# OpenScribe

Transcribe audio and video on your own machine with [OpenAI Whisper](https://github.com/openai/whisper). OpenScribe uses PyTorch and FFmpeg and includes a Gradio interface and a command-line tool. Inference runs on your hardware without an account, API key, or paid transcription service. Internet access is needed for installation and the first model download.

## Set up locally

Install Python 3.12 and FFmpeg, with `ffmpeg` on your PATH. On Debian or Ubuntu use `sudo apt install ffmpeg`, and on macOS use `brew install ffmpeg`. On Windows, install FFmpeg and activate the virtual environment with `.venv\Scripts\Activate.ps1`.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
# Linux CPU-only installation. For CUDA, use the PyTorch selector linked below instead.
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0+cpu"
python -m pip install -r requirements.txt
```

On macOS, skip the Linux CPU-wheel command. For an NVIDIA GPU, install a CUDA build of PyTorch using the [PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements. OpenScribe recommends Base on CPU and selects a model by available CUDA memory. Tiny is useful for quick smoke tests. Small or a larger multilingual model can improve accuracy, at higher compute and memory cost. Available models are Tiny, Base, Small, Medium, Large-v3 and Turbo.

## Transcribe the full recording

For a long recording, run the CLI directly so the job does not depend on a browser session or upload limits. Use the actual path and filename on your machine.

```sh
python transcribe.py "/path/to/11-04-2026 15.16.m4a" --model base --language auto --output "$HOME/openscribe-results"
```

If the run is interrupted, repeat the same command. OpenScribe keys the job by the recording's SHA-256 hash, model, backend, language and chunk size. It resumes at the last completed chunk and prints the paths of all six exports after verifying the processed duration. A completed job can regenerate its exports without audio decoding or model inference.

```sh
GRADIO_ANALYTICS_ENABLED=False python app.py
```

Open the local address printed by Gradio. Upload M4A, MP3, WAV, MP4, FLAC or WebM, select a model and language, then click **Transcribe**. The player previews the first audio stream. To resume in the interface, upload the same file and select the same settings. The interface binds to loopback by default. Keep it on a trusted machine because it has no login.

### Language, timestamps and long files

- Use **Automatically detect** for mixed Taglish. Whisper detects a likely language for each chunk and transcribes the original speech. Choose **Filipino / Tagalog** (`--language tl`) or **English** (`--language en`) if automatic detection is unreliable. Code-switching accuracy varies with the model and audio quality.
- FFmpeg decodes the first audio stream to mono 16 kHz PCM on disk, about 242 MB for 126 minutes. The engine feeds all samples to Whisper in consecutive 120-second chunks, including the final partial chunk. It submits each sample once, so there is no overlapping input to concatenate. Speech straddling a chunk boundary can be less accurate.
- Checkpoints record exact decoded frame ranges. The completed JSON reports both the total decoded and processed durations, per-chunk languages, segment timestamps and word timestamps where available. Full frame coverage measures input processed. Review the transcript against the recording when completeness or wording matters.
- Downloads include plain text, Markdown, JSON, SRT, WebVTT, and timestamped text. Speaker diarization is outside Phase 0.

### Optional Faster-Whisper backend

```sh
python -m pip install -r requirements-faster.txt
python transcribe.py "/path/to/recording.m4a" --backend faster --model base --language auto
```

The UI exposes Faster-Whisper when installed. It uses INT8 on CPU. For GPU use, install the CUDA and cuDNN versions required by [CTranslate2 and Faster-Whisper](https://github.com/SYSTRAN/faster-whisper#gpu). Its public model download uses Hugging Face. The default backend downloads the official weights from `openaipublic.azureedge.net` and Whisper verifies their checksums.

### Files and Docker

The CLI and UI default to `outputs/` in the current working directory. Set `OPENSCRIBE_OUTPUT` to store results and checkpoints elsewhere. Set `OPENSCRIBE_MODELS` to change the model cache from `~/.cache/openscribe`. Gradio also caches uploads, audio previews and downloads in its temporary directory. Keep recording, cache and output directories private and delete them when you no longer need the data.

```sh
docker build -t openscribe .
docker run --rm -p 127.0.0.1:7860:7860 \
  -v openscribe-results:/app/outputs \
  -v openscribe-models:/home/openscribe/.cache/openscribe openscribe
```

If the build runs behind a TLS-inspecting proxy, add `--secret id=build_ca,src=/path/to/trusted-ca-bundle.pem` to `docker build`. The bundle is used during dependency installation.

## Tests

```sh
python -m pip install pytest
python -m pytest -q
```

The suite decodes all six input formats, checks time offsets and exports, simulates interruption and restart, and verifies frame coverage on a synthetic recording longer than 126 minutes using a stand-in for the model. GitHub Actions also runs real official Whisper CPU inference against short synthesized speech and checks the words and timestamps.

## OpenScribe workspace

Start the browser workspace from the repository root after installing the requirements.

```sh
python -m uvicorn server:app --host 127.0.0.1 --port 7860
```

Docker starts this same workspace. The lightweight Gradio interface is also available with `python app.py`.

The workspace keeps uploaded recordings, listening previews, checkpoints, and edits in `outputs/workspace/` by default. My transcripts opens previous jobs on the same installation. Use one server process so the in-memory queue can track active jobs. Keep this no-login service on a trusted machine. Recordings and transcripts are accessible to anyone with access to the service.

- Upload from the file picker or by dropping your recording. The recording panel reports duration and size and provides playback speed, a waveform seek control, and time skips.
- Processing reports saved audio duration and segment count. Pause takes effect at the next completed chunk or earlier cancellation check. Resume with the same settings to reuse the checkpoint. Selecting new settings starts a separate run.
- Search, edit, and save segments while listening. Select a timestamp to seek. The editor shows when edits are unsaved, preserves the original text in JSON, and removes stale word timings from edited segments. Exports use the last saved edits.
- The interface supports light and dark themes, reduced motion, and segment or word timestamp display. Playback captions and subtitle exports retain segment timing.

Live partial transcript, estimated completion time, and speaker diarization are not available. For the largest recordings, the CLI avoids the browser upload and playback-preview preparation.

### Workspace screenshots

[Desktop upload](docs/screenshots/openscribe-desktop.png) · [Transcript review](docs/screenshots/openscribe-review.png) · [Mobile dark mode](docs/screenshots/openscribe-mobile-dark.png)

The review screenshot uses synthetic speech and test transcript text.
