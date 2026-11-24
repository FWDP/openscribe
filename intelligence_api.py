"""Intelligence routes, local storage, and resumable background jobs."""
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from time import time
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from intelligence import (
    DETAIL,
    TEMPLATES,
    ProviderError,
    batches,
    consolidate,
    instruction,
    providers,
    source_records,
    validate_findings,
    version,
)
from intelligence_exports import export_analysis
from transcribe import write_json

Marker = Annotated[str | None, Header(alias="X-OpenScribe")]
PREFIX = "/api/jobs/{job_id}/intelligence"


class Settings(BaseModel):
    provider: str = "ollama"
    model: str = Field(min_length=1, max_length=200)
    context: int = Field(default=8192, ge=2048, le=262144)
    template: str = "summary"
    detail: str = "standard"
    language: str = Field(default="English", max_length=60)
    instructions: str = Field(default="", max_length=2000)
    consent: bool = False
    section: str | None = None


class Edits(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    sections: list[list[str]]


def install_intelligence(app, directory, reviewed, local_request):
    lock = threading.RLock()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openscribe-ai")
    active = {}

    def shutdown():
        with lock:
            for cancel in active.values():
                cancel.set()
        pool.shutdown(wait=False, cancel_futures=True)

    app.router.add_event_handler("shutdown", shutdown)

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def folder_for(job_id, analysis_id):
        try:
            if UUID(analysis_id).hex != analysis_id:
                raise ValueError
        except ValueError as error:
            raise HTTPException(404, "Analysis not found.") from error
        folder = directory(job_id) / "intelligence" / analysis_id
        if not (folder / "analysis.json").exists():
            raise HTTPException(404, "Analysis not found.")
        return folder

    def get_provider(provider_id):
        provider = providers().get(provider_id)
        if provider is None:
            raise HTTPException(422, "Configure this provider on the server first.")
        return provider

    def consent_required(provider, consent):
        if provider.external and not consent:
            raise HTTPException(403, "Confirm external processing before sending this transcript.")

    def metadata(folder, current_version, *, full=False):
        with lock:
            value = read(folder / "analysis.json")
            if value["status"] in {"queued", "running", "pausing"} and folder not in active:
                value.update(status="paused", error="The server stopped. Resume from the last saved part.")
            if full and value["status"] == "complete":
                value["references"] = [
                    {key: source[key] for key in ("id", "start", "end", "speaker")}
                    for source in read(folder / "source.json")
                ]
            if not full:
                value.pop("sections", None)
                value.pop("sources", None)
            value["outdated"] = value["version"] != current_version
            return value

    @app.get("/api/intelligence/providers")
    def configurations():
        return {
            "providers": [
                {"id": item.id, "label": item.label, "external": item.external, "context": item.context}
                for item in providers().values()
            ],
            "templates": list(TEMPLATES),
            "details": list(DETAIL),
        }

    @app.get("/api/intelligence/providers/{provider_id}/models")
    def models(provider_id: str):
        try:
            provider = get_provider(provider_id)
            return {"models": provider.models(), "external": provider.external}
        except (ProviderError, KeyError, TypeError) as error:
            message = str(error) if isinstance(error, ProviderError) else "The endpoint returned an invalid model list."
            raise HTTPException(502, message) from error

    @app.get(PREFIX)
    def history(job_id: str):
        current = version(source_records(reviewed(directory(job_id))))
        parent = directory(job_id) / "intelligence"
        return sorted([metadata(item.parent, current) for item in parent.glob("*/analysis.json")],
                      key=lambda item: item["created"], reverse=True)

    def work(folder, provider, cancel):
        path = folder / "analysis.json"
        value = read(path)
        state_path = folder / "checkpoint.json"
        chunks = []
        saved = []
        covered = set()
        try:
            sources = read(folder / "source.json")
            settings = value["settings"]
            sections = (settings["section"],) if settings["section"] else TEMPLATES[settings["template"]]
            guide = instruction(settings)
            chunks = list(batches(sources, settings["context"], guide))
            saved = read(state_path) if state_path.exists() else []
            if len(saved) > len(chunks):
                raise ProviderError("Checkpoint does not match the source snapshot.")
            covered = {record["id"] for chunk in chunks[:len(saved)] for record in chunk}
            while len(saved) < len(chunks) and not cancel.is_set():
                chunk = chunks[len(saved)]
                with lock:
                    value.update(status="running", total=len(chunks), completed=len(saved), covered=len(covered), error="")
                    write_json(path, value)
                prompt = guide + "\nSource records (JSON data):\n" + json.dumps(chunk, ensure_ascii=False)
                result = provider.generate(settings["model"], settings["context"], prompt)
                saved.append(validate_findings(result, chunk, sections))
                write_json(state_path, saved)
                covered.update(record["id"] for record in chunk)
                value["completed"] = len(saved)
            if cancel.is_set():
                value.update(status="paused", error="Paused. Resume to process the remaining parts.")
            elif covered != {record["id"] for record in sources}:
                raise ProviderError("Coverage check failed. Some source segments were not processed.")
            else:
                result = consolidate(saved, sections)
                value.update(status="complete", sections=result, covered=len(covered),
                             error="" if result else "The transcript did not provide enough evidence for this request.")
        except Exception as error:  # noqa: BLE001 - Convert worker errors to actionable job status.
            value.update(status="failed", error=str(error) if isinstance(error, ProviderError)
                         else "Provider response or local checkpoint could not be read. Check the service and retry.")
        finally:
            value.update(completed=len(saved), total=len(chunks), covered=len(covered))
            if len(saved) < len(chunks):
                pending = chunks[len(saved)]
                value["next_sources"] = [pending[0]["id"], pending[-1]["id"]]
            else:
                value.pop("next_sources", None)
            with lock:
                try:
                    write_json(path, value)
                finally:
                    active.pop(folder, None)

    def launch(folder, value, provider):
        cancel = threading.Event()
        value.update(status="queued", error="")
        write_json(folder / "analysis.json", value)
        active[folder] = cancel
        pool.submit(work, folder, provider, cancel)

    @app.post(PREFIX, status_code=202)
    def start(job_id: str, settings: Settings, marker: Marker = None):
        local_request(marker)
        provider = get_provider(settings.provider)
        consent_required(provider, settings.consent)
        if settings.template not in TEMPLATES or settings.detail not in DETAIL:
            raise HTTPException(422, "Select a supported analysis template and detail level.")
        if settings.section and settings.section not in TEMPLATES[settings.template]:
            raise HTTPException(422, "Select a section in the chosen template.")
        if settings.template in {"ask", "custom"} and not settings.instructions.strip():
            raise HTTPException(422, "Enter a question or custom analysis instructions.")
        transcript = reviewed(directory(job_id))
        sources = source_records(transcript)
        if not sources:
            raise HTTPException(422, "This transcript has no spoken segments to analyze.")
        options = settings.model_dump(exclude={"consent"})
        options["duration"] = transcript.get("duration_seconds") or max(source["end"] for source in sources)
        try:
            total = sum(1 for _ in batches(sources, settings.context, instruction(options)))
        except ProviderError as error:
            raise HTTPException(422, str(error)) from error
        analysis_id = uuid4().hex
        folder = directory(job_id) / "intelligence" / analysis_id
        folder.mkdir(mode=0o700, parents=True)
        write_json(folder / "source.json", sources)
        value = {"id": analysis_id, "title": settings.section or settings.template.capitalize(),
                 "settings": options, "version": version(sources), "created": time(),
                 "completed": 0, "total": total, "covered": 0, "source_count": len(sources)}
        with lock:
            launch(folder, value, provider)
            return dict(value)

    @app.get(PREFIX + "/{analysis_id}")
    def status(job_id: str, analysis_id: str):
        current = version(source_records(reviewed(directory(job_id))))
        return metadata(folder_for(job_id, analysis_id), current, full=True)

    @app.post(PREFIX + "/{analysis_id}/pause")
    def pause(job_id: str, analysis_id: str, marker: Marker = None):
        local_request(marker)
        folder = folder_for(job_id, analysis_id)
        with lock:
            if folder not in active:
                raise HTTPException(409, "This analysis is no longer running.")
            active[folder].set()
            data = read(folder / "analysis.json")
            data["status"] = "pausing"
            write_json(folder / "analysis.json", data)
            return data

    @app.post(PREFIX + "/{analysis_id}/retry", status_code=202)
    def retry(job_id: str, analysis_id: str, consent: bool = False, marker: Marker = None):
        local_request(marker)
        folder = folder_for(job_id, analysis_id)
        with lock:
            value = read(folder / "analysis.json")
            if folder in active or value["status"] == "complete":
                raise HTTPException(409, "This analysis is running or complete.")
            provider = get_provider(value["settings"]["provider"])
            consent_required(provider, consent)
            launch(folder, value, provider)
            return dict(value)

    @app.put(PREFIX + "/{analysis_id}")
    def edit(job_id: str, analysis_id: str, edits: Edits, marker: Marker = None):
        local_request(marker)
        folder = folder_for(job_id, analysis_id)
        with lock:
            path = folder / "analysis.json"
            value = read(path)
            sections = value.get("sections", [])
            if value["status"] != "complete" or [len(s["findings"]) for s in sections] != list(map(len, edits.sections)):
                raise HTTPException(422, "Submit one edit for each finding in a completed analysis.")
            if not edits.title.strip() or any(not text.strip() for section in edits.sections for text in section):
                raise HTTPException(422, "A title and non-empty findings are required.")
            revisions = folder / "revisions"
            revisions.mkdir(exist_ok=True)
            write_json(revisions / (uuid4().hex + ".json"), value)
            value["title"] = edits.title.strip()
            for section, texts in zip(sections, edits.sections, strict=True):
                for finding, text in zip(section["findings"], texts, strict=True):
                    if finding["text"] != text:
                        finding.setdefault("generated_text", finding["text"])
                        finding["text"] = text
            write_json(path, value)
        return status(job_id, analysis_id)

    @app.delete(PREFIX + "/{analysis_id}", status_code=204)
    def remove(job_id: str, analysis_id: str, marker: Marker = None):
        local_request(marker)
        folder = folder_for(job_id, analysis_id)
        with lock:
            if folder in active:
                raise HTTPException(409, "Pause the analysis before deleting it.")
            shutil.rmtree(folder)

    @app.get(PREFIX + "/{analysis_id}/export/{suffix}")
    def download(job_id: str, analysis_id: str, suffix: str, attribution: bool = True,
                 selected: Annotated[list[int] | None, Query()] = None):
        folder = folder_for(job_id, analysis_id)
        with lock:
            value = read(folder / "analysis.json")
            sources = read(folder / "source.json")
        if value["status"] != "complete":
            raise HTTPException(409, "Complete the analysis first.")
        if selected is not None:
            value["sections"] = [section for index, section in enumerate(value["sections"]) if index in selected]
        try:
            data, media = export_analysis(value, sources, suffix, attribution, BytesIO())
        except KeyError as error:
            raise HTTPException(404, "Unsupported Intelligence export format.") from error
        return Response(data, media_type=media,
                        headers={"Content-Disposition": f'attachment; filename="openscribe-intelligence.{suffix}"'})
