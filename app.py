"""Local Gradio interface for OpenScribe."""
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
from whisper.tokenizer import LANGUAGES

from transcribe import FORMAT_SUFFIXES, MODELS, recommend_model, timestamp, transcribe

MEDIA_TYPES = [".m4a", ".mp3", ".wav", ".mp4", ".flac", ".webm"]


def prepare_preview(source):
    if not source:
        return None
    # Leave the original video or audio untouched for transcription. Browser
    # codec support varies, so provide a broadly playable preview independently.
    parent = Path(source).parent
    with tempfile.NamedTemporaryFile(suffix=".mp3", prefix="preview-", dir=parent,
                                     delete=False) as temporary:
        preview = Path(temporary.name)
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", source,
             "-map", "0:a:0", "-vn", "-c:a", "libmp3lame", "-b:a", "48k", str(preview)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as error:
        preview.unlink(missing_ok=True)
        raise gr.Error("The selected file has no playable audio stream.") from error
    return str(preview)


def build_app():
    recommended = recommend_model()
    backends = [("OpenAI Whisper", "whisper")]
    if importlib.util.find_spec("faster_whisper") is not None:
        backends.append(("Faster-Whisper", "faster"))
    languages = [("Automatically detect", "")] + [
        ("Filipino / Tagalog" if code == "tl" else label.title(), code)
        for code, label in sorted(LANGUAGES.items(), key=lambda pair: pair[1])
    ]
    with gr.Blocks(title="OpenScribe", analytics_enabled=False) as demo:
        gr.Markdown("# OpenScribe")
        source = gr.File(label="Audio or video", file_types=MEDIA_TYPES, type="filepath")
        player = gr.Audio(label="Audio player", interactive=False, format="mp3")
        with gr.Row():
            model = gr.Dropdown(MODELS, value=recommended, label=f"Model (recommended: {recommended})")
            language = gr.Dropdown(languages, value="", label="Language")
            backend = gr.Dropdown(backends, value="whisper", label="Backend")
        button = gr.Button("Transcribe", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        preview = gr.Textbox(label="Transcript preview", lines=12, interactive=False)
        with gr.Row():
            downloads = [gr.DownloadButton(label=suffix.upper()) for suffix in FORMAT_SUFFIXES]
        source.change(prepare_preview, source, player, show_progress="minimal")

        def submit(source, model, language, backend, progress=gr.Progress()):  # noqa: B008
            if not source:
                raise gr.Error("Upload an audio or video file.")
            result, paths = transcribe(
                source, model=model, language=language or None, backend=backend,
                progress=lambda fraction, description: progress(fraction, desc=description),
            )
            preview_text = paths["timestamped.txt"].read_text(encoding="utf-8")
            message = f"Processed {timestamp(result['processed_seconds'])}."
            # Serve individual exports even when the job directory is outside cwd.
            cached = [
                button.move_resource_to_block_cache(paths[suffix])
                for button, suffix in zip(downloads, FORMAT_SUFFIXES, strict=True)
            ]
            return preview_text, message, *cached

        button.click(submit, [source, model, language, backend], [preview, status, *downloads],
                     api_name="transcribe")
    return demo.queue()


if __name__ == "__main__":
    build_app().launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"), share=False)
