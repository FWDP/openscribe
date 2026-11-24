import json
import subprocess

import pytest

import transcribe as engine


def render_audio(path, seconds, *args):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
         "-i", "anullsrc=r=16000:cl=mono", *args, "-t", str(seconds), str(path)],
        check=True,
    )


@pytest.mark.parametrize("suffix,options", [
    ("wav", []), ("m4a", []), ("mp3", []), ("flac", []),
    ("mp4", ["-c:a", "aac", "-c:v", "libx264"]),
    ("webm", ["-c:a", "libopus", "-c:v", "libvpx-vp9"]),
])
def test_ffmpeg_input_formats(tmp_path, monkeypatch, suffix, options):
    media = tmp_path / f"sample.{suffix}"
    if suffix in {"mp4", "webm"}:
        options = ["-f", "lavfi", "-i", "color=size=32x32:rate=2", *options]
    render_audio(media, 2, *options)
    monkeypatch.setattr(engine, "load_backend", lambda *_: lambda *_: ("en", []))
    result, _ = engine.transcribe(media, tmp_path / "out", model="tiny", device="cpu")
    assert result["duration_seconds"] == pytest.approx(2, abs=0.1)


def test_interruption_resume_and_exports(tmp_path, monkeypatch):
    media = tmp_path / "sample.m4a"
    render_audio(media, 5)
    output = tmp_path / "out"
    lengths = []

    def infer(samples, language):
        assert language is None
        lengths.append(len(samples))
        if len(lengths) == 2:
            raise RuntimeError("interrupted")
        word = "Kumusta" if len(lengths) == 1 else "hello"
        return "tl" if len(lengths) == 1 else "en", [{
            "text": word, "start": 0.25, "end": 0.75,
            "words": [{"word": word, "start": 0.25, "end": 0.75, "probability": 0.9}],
        }]

    monkeypatch.setattr(engine, "load_backend", lambda *_: infer)
    options = {"model": "tiny", "device": "cpu", "chunk_seconds": 2}
    with pytest.raises(RuntimeError, match="interrupted"):
        engine.transcribe(media, output, **options)

    checkpoint, = output.glob("*/checkpoint.json")
    first = json.loads(checkpoint.read_text())
    assert first["processed_frames"] == 32000
    result, paths = engine.transcribe(media, output, **options)
    assert lengths == [32000, 32000, 32000, int(result["duration_seconds"] * 16000) - 64000]
    assert result["complete"] and result["processed_seconds"] == result["duration_seconds"]
    assert [segment["start"] for segment in result["segments"]] == [0.25, 2.25, 4.25]
    assert result["segments"][1]["words"][0]["start"] == 2.25
    assert result["chunks"][0]["language"] == "tl"
    assert list(paths) == list(engine.FORMAT_SUFFIXES)
    assert paths["txt"].read_text().splitlines() == ["Kumusta", "hello", "hello"]
    assert "00:00:02,250 --> 00:00:02,750" in paths["srt"].read_text()
    assert paths["vtt"].read_text().startswith("WEBVTT\n\n")
    assert "**00:00:02.250** hello" in paths["md"].read_text()
    assert "[00:00:02.250 → 00:00:02.750] hello" in paths["timestamped.txt"].read_text()
    assert json.loads(paths["json"].read_text()) == result

    def unexpected(*args):
        pytest.fail("Completed jobs should reuse their checkpoints")

    monkeypatch.setattr(engine, "load_backend", unexpected)
    monkeypatch.setattr(engine, "decode_media", unexpected)
    assert engine.transcribe(media, output, **options)[0] == result


def test_two_hours_and_six_minutes_of_decoded_audio(tmp_path, monkeypatch):
    duration = 126 * 60 + 1
    media = tmp_path / "long.flac"
    render_audio(media, duration)
    submitted = []

    def infer(samples, _language):
        submitted.append(len(samples))
        return "en", []

    monkeypatch.setattr(engine, "load_backend", lambda *_: infer)
    result, _ = engine.transcribe(media, tmp_path / "out", model="tiny", device="cpu")
    assert result["duration_seconds"] == result["processed_seconds"] == duration
    assert sum(submitted) == duration * engine.SAMPLE_RATE
    assert submitted == [120 * engine.SAMPLE_RATE] * 63 + [engine.SAMPLE_RATE]
