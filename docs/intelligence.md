# OpenScribe Intelligence architecture

The FastAPI workspace enables Intelligence on completed transcripts. The CLI and Gradio interfaces continue to provide the original transcription workflow.

## Providers and privacy

The server discovers models with the selected provider's API and sends chat requests through a common provider interface. The browser receives the provider name, locality, default context and model IDs. Endpoint URLs and credentials are server configuration. HTTP redirects are refused. The app has no tools or autonomous actions attached to model output. Remote endpoints receive transcript text only after explicit confirmation. Their data policies still apply. OpenScribe does not train models on recordings or transcripts.

Use a single server process on a trusted machine. The workspace has no user accounts or multi-user authorization. The job directory contains recordings, transcript edits, and Intelligence snapshots. Intelligence does not log prompts or results. Standard server access logs include route paths and job IDs. Disable those logs with Uvicorn's `--no-access-log` option if needed.

## Pipeline and recovery

1. Take a snapshot of the saved transcript with stable segment IDs, times, and any available speaker labels. Hash the snapshot to track version changes.
2. Reserve space for instructions and response tokens based on the selected context. Budget input conservatively using UTF-8 byte counts. Split on segment boundaries, prefer pauses when near budget, and repeat a short boundary segment when space permits. Split unusually long segments into fragments with the original segment ID.
3. Analyze each chunk separately. Keep transcript content inside JSON data and give the model separate system instructions. Validate the response shape, allowed section names, classification, and source IDs before checkpointing a successful part.
4. Consolidate by template section, merge identical normalized findings with their citations, and preserve distinct statements even where they disagree. Verify the union of processed IDs against the snapshot. Every part must complete before a job is marked complete.
5. Store and display progress from completed parts. Pause after the current provider call. Resume against the same snapshot and skip completed parts. Failed or invalid responses leave the remaining parts unprocessed and retryable.

Outputs under `outputs/workspace/<job-id>/intelligence/<run-id>/` include the source snapshot, checkpoints, run metadata, and copies of prior versions when analysis edits are saved. Delete a run in the UI to remove all of those files. To remove an entire recording and its transcripts and analyses, stop OpenScribe and remove its job directory. A transcript edit changes the version hash so older runs show a warning. Their timestamp references continue to point to the original source positions.

## Known limitations

- Validated citation IDs establish that a source exists. They do not establish that a finding accurately describes it. Review citations, interpretations, and software requirements before acting on generated text. Prompt separation and schema checks reduce injection risk, but a model can still produce misleading content.
- Completed part counts verify submission, not semantic coverage. Exact duplicates are merged. Paraphrases and contradictory statements remain separate for review. A full transcript analysis or Ask AI run makes one provider call per part, which can take time or incur provider charges.
- Ask AI follow-ups carry the previous question during the current browser session. They use the saved transcript as evidence and do not treat earlier model answers as facts.
- The UI saves edited copies, supports repeat generation as separate runs, and shows references. It does not replace a section in an existing edited run or restore a saved revision from the browser.
- PDF uses a system font when available. Use DOCX or Markdown where the PDF font lacks glyphs for the chosen output language.
- Local HTTP integration was verified with an Ollama-protocol fixture. The cloud proxy blocked the Ollama model registry, so production local or cloud LLM inference was not verified here.

## Verification

The automated suite exercises the existing transcription workflow and Intelligence. A loopback HTTP fixture checks Ollama model discovery and request formatting, then analyzes a synthetic transcript with 253 segments spanning 2:06:28. It verifies pause and resume, source coverage, invalid citations, transcript version warnings, saved edits, five export formats, external-provider confirmation, and removal of run data. Synthetic prompt-injection text is kept inside source data in the outgoing request.

Chromium verified provider discovery, meeting-minute generation, saving edits, PDF download, citation-linked playback, and an Ask AI result with insufficient evidence. These checks used the loopback fixture. [Intelligence workspace](screenshots/openscribe-intelligence.png) · [Ask AI](screenshots/openscribe-ask.png)
