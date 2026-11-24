import subprocess
from pathlib import Path
from urllib.request import urlopen

import gradio_client

import app
import transcribe as engine


def test_gradio_upload_transcription_and_download(tmp_path, monkeypatch):
    sample = tmp_path / "speech.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
         "-i", "anullsrc=r=16000:cl=mono", "-t", "1", str(sample)],
        check=True,
    )
    assert app.prepare_preview(str(sample)).endswith(".mp3")
    monkeypatch.setattr(app, "recommend_model", lambda: "tiny")
    monkeypatch.setattr(app, "transcribe", lambda path, **kwargs: engine.transcribe(
        path, tmp_path / "out", **kwargs,
    ))
    monkeypatch.setattr(engine, "load_backend", lambda *_: lambda *_: ("en", [
        {"start": 0.1, "end": 0.5, "text": "Hello from the test."},
    ]))
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    demo = app.build_app()
    try:
        _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True)
        with urlopen(url, timeout=10) as response:
            assert response.status == 200
        client = gradio_client.Client(url, download_files=tmp_path / "downloads", verbose=False)
        preview, status, *downloads = client.predict(
            gradio_client.handle_file(sample), "tiny", "", "whisper", api_name="/transcribe"
        )
        assert "[00:00:00.100 → 00:00:00.500] Hello from the test." in preview
        assert status.startswith("Processed ")
        assert len(downloads) == 6
        assert Path(downloads[0]).read_text().strip() == "Hello from the test."
    finally:
        demo.close()
