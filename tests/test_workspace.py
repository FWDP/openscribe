import subprocess
from threading import Event
from time import monotonic, sleep

from fastapi.testclient import TestClient

import server
import transcribe as engine

HEADERS = {"X-OpenScribe": "1"}


def wait_for(client, url, state):
    deadline = monotonic() + 10
    while monotonic() < deadline:
        job = client.get(url).json()
        if job["status"] == state:
            return job
        assert job["status"] != "failed", job["stage"]
        sleep(0.025)
    raise AssertionError(f"Never reached {state}: {job}")


def test_workspace_checkpoint_playback_review_and_exports(tmp_path, monkeypatch):
    sample = tmp_path / "sample.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:sample_rate=16000:duration=3", str(sample)],
        check=True,
    )
    entered = Event()
    release = Event()
    calls = []

    def infer(audio, language):
        calls.append(len(audio))
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return "tl", [{"start": .2, "end": .8, "text": "Kumusta", "words": [
            {"start": .2, "end": .8, "word": "Kumusta", "probability": .9},
        ]}]

    monkeypatch.setattr(engine, "load_backend", lambda *_: infer)
    monkeypatch.setattr(server, "hardware", lambda: ("cpu", 0))
    monkeypatch.setattr(server, "transcribe", lambda *args, **options: engine.transcribe(
        *args, **options, chunk_seconds=1,
    ))

    with TestClient(server.create_app(tmp_path / "jobs")) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/jobs").json() == []
        assert client.post("/api/jobs", files={"file": ("bad.txt", b"invalid")}, headers=HEADERS).status_code == 415
        with sample.open("rb") as stream:
            response = client.post("/api/jobs", files={"file": (sample.name, stream)}, headers=HEADERS)
        assert response.status_code == 201, response.text
        job = response.json()
        assert job["size"] == sample.stat().st_size
        url = "/api/jobs/" + job["id"]
        assert client.get(url + "/media/audio", headers={"Range": "bytes=0-9"}).status_code == 206
        assert client.get(url + "/media/waveform").status_code == 200
        settings = {"model": "tiny"}
        assert client.post(url + "/run", json=settings).status_code == 403
        assert client.post(url + "/run", json=settings, headers=HEADERS).status_code == 202
        assert entered.wait(5)
        assert client.post(url + "/pause", headers=HEADERS).status_code == 200
        release.set()
        paused = wait_for(client, url, "paused")
        assert paused["processed"] == 1 and paused["segment"] == 1
        assert client.post(url + "/run", json=settings, headers=HEADERS).status_code == 202
        final = wait_for(client, url, "complete")
        assert final["processed"] == final["duration"]
        assert sum(calls) == round(final["duration"] * engine.SAMPLE_RATE)
        original = client.get(url + "/result").json()
        edits = [segment["text"] for segment in original["segments"]]
        edits[0] = "Kumusta, mundo!"
        changed = client.put(url + "/result", json=edits, headers=HEADERS).json()
        assert changed["segments"][0]["original_text"] == "Kumusta"
        assert changed["segments"][0]["words"] == []
        assert changed["segments"][1]["words"]
        for suffix in engine.FORMAT_SUFFIXES:
            response = client.get(url + "/export/" + suffix)
            assert response.status_code == 200
            assert "Kumusta, mundo!" in response.text
        assert client.get("/api/jobs").json()[0]["status"] == "complete"
