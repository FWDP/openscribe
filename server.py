"""Single-process, local OpenScribe workspace."""
import importlib.util
import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from whisper.tokenizer import LANGUAGES

from intelligence_api import install_intelligence
from transcribe import (
    DEFAULT_OUTPUT,
    FORMAT_SUFFIXES,
    MODELS,
    SAMPLE_RATE,
    export,
    hardware,
    recommend_model,
    transcribe,
    write_json,
)

WEB = Path(__file__).with_name("web")
FORMATS = {".m4a", ".mp3", ".wav", ".mp4", ".flac", ".webm"}


class RunSettings(BaseModel):
    model: str
    language: str | None = None
    backend: str = "whisper"


class PauseRequested(Exception):
    pass


def create_app(root=None):
    root = Path(root) if root else DEFAULT_OUTPUT / "workspace"
    app = FastAPI(title="OpenScribe", docs_url=None, redoc_url=None)
    app.mount("/assets", StaticFiles(directory=WEB), name="assets")
    queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
    lock = threading.RLock()
    active = {}

    def directory(job_id):
        try:
            if UUID(job_id).hex != job_id:
                raise ValueError("Expected a canonical job ID")
            folder = root / job_id
        except ValueError as error:
            raise HTTPException(404, "Recording not found.") from error
        if not (folder / "job.json").is_file():
            raise HTTPException(404, "Recording not found.")
        return folder

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def update(folder, **changes):
        with lock:
            job = read(folder / "job.json")
            job.update(changes)
            write_json(folder / "job.json", job)

    def job_info(folder):
        with lock:
            job = read(folder / "job.json")
            if job["status"] in {"queued", "running", "pausing"} and job["id"] not in active:
                job.update(status="paused", stage="Interrupted. Resume from the last checkpoint.")
            return job

    def local_request(marker):
        if marker != "1":
            raise HTTPException(403, "Open this action from the OpenScribe workspace.")

    @app.get("/")
    def home():
        return FileResponse(WEB / "index.html")

    @app.get("/api/config")
    def config():
        return {
            "models": MODELS, "recommended": recommend_model(), "device": hardware()[0],
            "languages": sorted(LANGUAGES.items(), key=lambda item: item[1]),
            "faster": importlib.util.find_spec("faster_whisper") is not None,
        }

    @app.get("/api/jobs")
    def history():
        return sorted((job_info(path.parent) for path in root.glob("*/job.json")),
                      key=lambda job: job["created"], reverse=True)

    @app.post("/api/jobs", status_code=201)
    def upload(file: Annotated[UploadFile, File()], x_openscribe: str | None = Header(None)):
        local_request(x_openscribe)
        name = Path(file.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in FORMATS:
            raise HTTPException(415, "Choose M4A, MP3, WAV, MP4, FLAC, or WebM.")
        job_id = uuid4().hex
        folder = root / job_id
        folder.mkdir(mode=0o700, parents=True)
        source = folder / ("source" + suffix)
        try:
            with source.open("wb") as target:
                shutil.copyfileobj(file.file, target, 1024 * 1024)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=duration:format=duration", "-of", "json", str(source)],
                check=True, capture_output=True,
            )
            metadata = json.loads(probe.stdout)
            if not metadata.get("streams"):
                raise ValueError("Missing audio stream")
            duration = float(metadata["streams"][0].get("duration")
                             or metadata.get("format", {}).get("duration") or 0)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source),
                 "-map", "0:a:0", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "48k",
                 str(folder / "preview.mp3")], check=True, capture_output=True,
            )
            waveform = subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(folder / "preview.mp3"),
                 "-filter_complex", "aresample=1000,showwavespic=s=960x90:colors=0x7559e8",
                 "-frames:v", "1", "-threads", "1", str(folder / "waveform.png")],
                check=False, capture_output=True,
            )
            if waveform.returncode:
                (folder / "waveform.png").unlink(missing_ok=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            shutil.rmtree(folder)
            raise HTTPException(422, "Could not decode this recording. Check that it has a playable audio track and that disk space is available.") from error
        job = {"id": job_id, "name": name, "source": source.name, "size": source.stat().st_size,
               "duration": duration, "processed": 0, "stage": "Ready to transcribe",
               "status": "ready", "segment": 0, "created": time(),
               "waveform": (folder / "waveform.png").is_file()}
        write_json(folder / "job.json", job)
        return job

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        return job_info(directory(job_id))

    @app.get("/api/jobs/{job_id}/media/{asset}")
    def playback(job_id: str, asset: str):
        names = {"audio": "preview.mp3", "waveform": "waveform.png"}
        if asset not in names:
            raise HTTPException(404, "Media not found.")
        path = directory(job_id) / names[asset]
        if not path.is_file():
            raise HTTPException(404, "Media not found.")
        return FileResponse(path)

    def run_job(folder, job, settings, cancel):
        def progress(_fraction, stage):
            if cancel.is_set():
                raise PauseRequested
            update(folder, status="running", stage=stage)

        def checkpoint(state):
            duration = state["total_frames"]
            if duration is not None:
                update(folder, duration=duration / SAMPLE_RATE,
                       processed=state["processed_frames"] / SAMPLE_RATE,
                       segment=len(state["chunks"]))
            if cancel.is_set():
                raise PauseRequested

        try:
            result, _ = transcribe(folder / job["source"], folder / "engine",
                                   device=job["device"], **settings, progress=progress,
                                   checkpoint_update=checkpoint)
            write_json(folder / "result.json", result)
            final = {"status": "complete", "stage": "Ready to review",
                     "processed": result["processed_seconds"], "duration": result["duration_seconds"]}
        except PauseRequested:
            final = {"status": "paused", "stage": "Paused. Resume from the last checkpoint."}
        except Exception as error:  # noqa: BLE001 - Report background job failures to the workspace.
            reason = str(error)
            if "403" in reason or "urlopen" in reason:
                reason = "Model download failed. Allow the model host in your network settings, then resume."
            elif "out of memory" in reason.lower():
                reason = "Available memory was insufficient. Select a smaller model and start again."
            final = {"status": "failed", "stage": reason or "Processing failed. Check the recording and retry."}
        with lock:
            try:
                update(folder, **final)
            finally:
                active.pop(job["id"], None)

    @app.post("/api/jobs/{job_id}/run", status_code=202)
    def start(job_id: str, settings: RunSettings, x_openscribe: str | None = Header(None)):
        local_request(x_openscribe)
        if (settings.model not in MODELS or settings.backend not in {"whisper", "faster"}
                or (settings.language is not None and settings.language not in LANGUAGES)):
            raise HTTPException(422, "Select a supported model, backend and language.")
        if settings.backend == "faster" and importlib.util.find_spec("faster_whisper") is None:
            raise HTTPException(422, "Install requirements-faster.txt to use Faster-Whisper.")
        folder = directory(job_id)
        with lock:
            job = job_info(folder)
            if job_id in active or job["status"] == "complete":
                raise HTTPException(409, "This job is active or already complete.")
            values = settings.model_dump()
            same_settings = values == job.get("settings")
            update(folder, settings=values, device=hardware()[0], status="queued", stage="Queued",
                   processed=job["processed"] if same_settings else 0,
                   segment=job["segment"] if same_settings else 0)
            cancel = threading.Event()
            active[job_id] = cancel
            queue.submit(run_job, folder, job_info(folder), values, cancel)
            return job_info(folder)

    @app.post("/api/jobs/{job_id}/pause")
    def pause(job_id: str, x_openscribe: str | None = Header(None)):
        local_request(x_openscribe)
        folder = directory(job_id)
        with lock:
            if job_id not in active or job_info(folder)["status"] not in {"queued", "running"}:
                raise HTTPException(409, "This job is no longer running.")
            active[job_id].set()
            update(folder, status="pausing", stage="Pausing at the next checkpoint")
            return job_info(folder)

    def reviewed(folder):
        file = folder / "result.json"
        if not file.is_file():
            raise HTTPException(409, "Complete transcription first.")
        result = read(file)
        edits = folder / "edits.json"
        if edits.is_file():
            for segment, value in zip(result["segments"], read(edits), strict=True):
                if segment["text"] != value:
                    segment.update(original_text=segment["text"], text=value, words=[])
        return result

    @app.get("/api/jobs/{job_id}/result")
    def result(job_id: str):
        with lock:
            return reviewed(directory(job_id))

    @app.put("/api/jobs/{job_id}/result")
    def edit(job_id: str, texts: list[str], x_openscribe: str | None = Header(None)):
        local_request(x_openscribe)
        folder = directory(job_id)
        with lock:
            result = reviewed(folder)
            if len(texts) != len(result["segments"]):
                raise HTTPException(422, "Submit one edit per transcript segment.")
            write_json(folder / "edits.json", texts)
            return reviewed(folder)

    @app.get("/api/jobs/{job_id}/export/{suffix}")
    def download(job_id: str, suffix: str):
        if suffix not in FORMAT_SUFFIXES:
            raise HTTPException(404, "Export format not found.")
        folder = directory(job_id)
        with lock:
            path = export(reviewed(folder), folder)[suffix]
            return FileResponse(path, filename=path.name)

    install_intelligence(app, directory, reviewed, local_request)
    return app


app = create_app()
