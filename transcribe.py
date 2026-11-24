"""Local, checkpointed transcription and subtitle export."""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from filelock import FileLock

MODELS = ("tiny", "base", "small", "medium", "large-v3", "turbo")
FORMAT_SUFFIXES = ("txt", "md", "json", "srt", "vtt", "timestamped.txt")
SAMPLE_RATE = 16000
DEFAULT_OUTPUT = Path(os.getenv("OPENSCRIBE_OUTPUT", "outputs"))
MODEL_CACHE = Path(os.getenv("OPENSCRIBE_MODELS", "~/.cache/openscribe")).expanduser()


def hardware():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.cuda.mem_get_info()[0] / 2**30
    return "cpu", 0


def recommend_model():
    device, memory = hardware()
    if device == "cpu":
        return "base"
    for model, minimum in (("large-v3", 12), ("turbo", 8), ("medium", 6),
                           ("small", 3), ("base", 1.5), ("tiny", 0)):
        if memory >= minimum:
            return model


def load_backend(backend, model, device):
    threads = max(1, min(4, os.cpu_count() or 1))
    if backend == "faster":
        from faster_whisper import WhisperModel

        engine = WhisperModel(
            model, device=device, compute_type="int8" if device == "cpu" else "float16",
            cpu_threads=threads if device == "cpu" else 0,
            download_root=str(MODEL_CACHE / "faster"),
        )

        def infer(samples, language):
            segments, info = engine.transcribe(
                samples, language=language, task="transcribe", beam_size=5,
                word_timestamps=True, vad_filter=False,
            )
            result = []
            for segment in segments:
                result.append({
                    "start": segment.start, "end": segment.end, "text": segment.text,
                    "words": [{"start": w.start, "end": w.end, "word": w.word,
                               "probability": w.probability} for w in segment.words or []],
                })
            return info.language, result

        return infer

    import torch
    import whisper

    if device == "cpu":
        torch.set_num_threads(threads)
    cache = MODEL_CACHE / "whisper"
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / f"{model}.lock")):
        engine = whisper.load_model(model, device=device, download_root=str(cache))

    def infer(samples, language):
        result = engine.transcribe(
            samples, language=language, task="transcribe", word_timestamps=True,
            fp16=device == "cuda", verbose=None, beam_size=5,
        )
        return result["language"], result["segments"]

    return infer


def decode_media(source, target):
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
        "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"FFmpeg could not decode the audio: {error.stderr.strip()}") from error


def write_text(path, text):
    # Replace atomically so an interruption leaves the previous checkpoint readable.
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def timestamp(seconds, separator="."):
    minutes, remainder = divmod(round(seconds * 1000), 60000)
    hours, minutes = divmod(minutes, 60)
    whole, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole:02}{separator}{millis:03}"


def export(result, directory):
    segments = result["segments"]
    plain = "\n".join(s["text"] for s in segments)
    timed = "\n".join(
        f"[{timestamp(s['start'])} → {timestamp(s['end'])}] {s['text']}" for s in segments
    )
    markdown = "\n\n".join(
        f"**{timestamp(s['start'])}** {s['text']}" for s in segments
    )

    def cue(segment, separator):
        start = timestamp(segment["start"], separator)
        end = timestamp(max(segment["end"], segment["start"] + .001), separator)
        return f"{start} --> {end}\n{segment['text']}"

    srt = "\n\n".join(f"{i}\n{cue(s, ',')}" for i, s in enumerate(segments, 1))
    vtt = "\n\n".join(cue(s, ".") for s in segments)
    values = (plain + "\n", markdown + "\n",
              json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
              srt + "\n", "WEBVTT\n\n" + vtt + "\n", timed + "\n")
    paths = {}
    for suffix, value in zip(FORMAT_SUFFIXES, values, strict=True):
        path = directory / f"transcript.{suffix}"
        write_text(path, value)
        paths[suffix] = path.resolve()
    return paths


def normalize_segments(segments, offset, duration):
    def absolute(value):
        return round(offset + min(duration, max(0, float(value))), 6)

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        start = absolute(segment["start"])
        words = []
        for word in segment.get("words", []):
            word_start = absolute(word["start"])
            words.append({
                "word": word["word"], "start": word_start,
                "end": max(word_start, absolute(word["end"])),
                "probability": float(word["probability"]),
            })
        yield {"start": start, "end": max(start, absolute(segment["end"])),
               "text": text, "words": words}


def transcribe(source, output=DEFAULT_OUTPUT, *, model=None, language=None,
               backend="whisper", device="auto", chunk_seconds=120, progress=None, checkpoint_update=None):
    import numpy as np

    if model is None:
        model = "base" if device == "cpu" else recommend_model()
    if model not in MODELS or backend not in ("whisper", "faster"):
        raise ValueError("Unsupported model or backend")
    if not isinstance(chunk_seconds, int) or not 1 <= chunk_seconds <= 600:
        raise ValueError("Chunk length must be between 1 and 600 seconds")
    if device == "auto":
        device = hardware()[0]
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu or cuda")
    source = Path(source)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    settings = {"version": 1, "sha256": digest, "model": model, "language": language,
                "backend": backend, "chunk_seconds": chunk_seconds}
    job_id = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    directory = Path(output) / job_id
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.json"

    def report(fraction, message):
        if progress:
            progress(fraction, message)

    with FileLock(str(directory / ".lock")):
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if state["settings"] != settings:
                raise ValueError("Checkpoint settings differ from this job")
        else:
            state = {"settings": settings, "total_frames": None, "processed_frames": 0,
                     "chunks": []}

        if checkpoint_update:
            checkpoint_update(state)

        if state["processed_frames"] != state["total_frames"]:
            with tempfile.TemporaryDirectory(prefix="pcm-", dir=directory) as temporary:
                report(0, "Decoding the first audio stream with FFmpeg")
                pcm = Path(temporary) / "audio.wav"
                decode_media(source, pcm)
                with wave.open(str(pcm), "rb") as audio:
                    total = audio.getnframes()
                    if not total:
                        raise ValueError("The first audio stream is empty")
                    frames = state["processed_frames"]
                    if state["total_frames"] not in (None, total):
                        raise ValueError("Decoded duration differs from the checkpoint")
                    state["total_frames"] = total
                    audio.setpos(frames)
                    if checkpoint_update:
                        checkpoint_update(state)
                    report(frames / total, f"Loading {model} on {device}")
                    infer = load_backend(backend, model, device)
                    chunk_size = SAMPLE_RATE * chunk_seconds
                    while frames < total:
                        size = min(chunk_size, total - frames)
                        report(frames / total,
                               f"Transcribing {timestamp(frames / SAMPLE_RATE)} of {timestamp(total / SAMPLE_RATE)}")
                        raw = audio.readframes(size)
                        if len(raw) != size * 2:
                            raise ValueError("Decoded audio ended unexpectedly")
                        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                        detected, segments = infer(samples, language)
                        state["chunks"].append({
                            "start_frame": frames, "end_frame": frames + size, "language": detected,
                            "segments": list(normalize_segments(
                                segments, frames / SAMPLE_RATE, size / SAMPLE_RATE)),
                        })
                        frames += size
                        state["processed_frames"] = frames
                        write_json(checkpoint, state)
                        if checkpoint_update:
                            checkpoint_update(state)

        end = 0
        for chunk in state["chunks"]:
            if chunk["start_frame"] != end or chunk["end_frame"] <= end:
                raise ValueError("Checkpoint contains a gap or overlapping audio")
            end = chunk["end_frame"]
        if end != state["total_frames"] or end != state["processed_frames"]:
            raise ValueError("Transcription did not cover the complete decoded audio")
        segments = sorted(
            (s for c in state["chunks"] for s in c["segments"]), key=lambda s: s["start"]
        )
        result = {
            "settings": settings, "sample_rate": SAMPLE_RATE, "complete": True,
            "duration_seconds": end / SAMPLE_RATE, "processed_seconds": end / SAMPLE_RATE,
            "chunks": [{k: v for k, v in c.items() if k != "segments"} for c in state["chunks"]],
            "segments": segments,
        }
        paths = export(result, directory)
        report(1, f"Processed {timestamp(end / SAMPLE_RATE)}")
        return result, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--language", default="auto", help="auto, en, tl, or a Whisper language")
    parser.add_argument("--backend", choices=("whisper", "faster"), default="whisper")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chunk-seconds", type=int, default=120)
    args = parser.parse_args()
    _, paths = transcribe(
        args.input, args.output, model=args.model,
        language=None if args.language.lower() == "auto" else args.language,
        backend=args.backend, device=args.device, chunk_seconds=args.chunk_seconds,
        progress=lambda fraction, message: print(f"{fraction:6.1%} {message}", flush=True),
    )
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
