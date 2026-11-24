const $ = (id) => document.getElementById(id);
const jobURL = (id) => "/api/jobs/" + id;
const activeStates = new Set(["queued", "running", "pausing"]);
const processingNotes = [
  "Turning sound into words.",
  "Whisper is doing its thing.",
  "A good conversation deserves a good transcript.",
  "Making every word count.",
];
const formats = [
  ["txt", "TXT"],
  ["md", "Markdown"],
  ["json", "JSON"],
  ["srt", "SRT"],
  ["vtt", "WebVTT"],
  ["timestamped.txt", "Timestamped TXT"],
];

let job;
let transcript;
let drafts = [];
let currentConfig;
let poll;
let uploading = false;
let changesPending = false;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text) element.textContent = text;
  if (className) element.className = className;
  return element;
}

async function api(url, { method = "GET", body } = {}) {
  const options = { method, headers: { "X-OpenScribe": "1" } };
  if (body instanceof FormData) {
    options.body = body;
  } else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "OpenScribe could not complete this request. Try again.",
    );
  }
  return response.json();
}

function clock(seconds) {
  const whole = Math.max(0, Math.floor(seconds || 0));
  return [Math.floor(whole / 3600), Math.floor(whole / 60) % 60, whole % 60]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

function feedback(message, error = false) {
  $("upload-status").textContent = message;
  $("upload-status").classList.toggle("error", error);
}

function theme() {
  const choice = $("theme").value;
  const dark =
    choice === "dark" ||
    (choice === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}

$("theme").value = localStorage.getItem("openscribe-theme") || "system";
$("theme").addEventListener("change", () => {
  localStorage.setItem("openscribe-theme", $("theme").value);
  theme();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", theme);
theme();

for (const height of [
  9, 22, 35, 17, 47, 59, 29, 67, 81, 55, 92, 73, 96, 65, 46, 78, 58, 42, 62, 34,
  21, 31, 17, 9,
]) {
  const bar = node("i");
  bar.style.setProperty("--bar", height + "%");
  document.querySelector(".hero-wave").append(bar);
}

function option(value, label) {
  const element = node("option", label);
  element.value = value;
  return element;
}

async function configure() {
  try {
    currentConfig = await api("/api/config");
    const names = {
      tiny: "Tiny",
      base: "Base",
      small: "Small",
      medium: "Medium",
      "large-v3": "Large-v3",
      turbo: "Turbo",
    };
    $("model").replaceChildren(
      ...currentConfig.models.map((model) =>
        option(
          model,
          names[model] +
            (model === currentConfig.recommended ? " · Recommended" : ""),
        ),
      ),
    );
    $("model").value = currentConfig.recommended;
    $("language").replaceChildren(
      option("", "Automatically detect"),
      ...currentConfig.languages.map(([code, label]) =>
        option(
          code,
          code === "tl"
            ? "Filipino / Tagalog"
            : label[0].toUpperCase() + label.slice(1),
        ),
      ),
    );
    $("backend-field").hidden = !currentConfig.faster;
    $("device").textContent =
      currentConfig.device === "cuda"
        ? "Processing device · NVIDIA GPU"
        : "Processing device · CPU";
  } catch (error) {
    feedback(error.message, true);
  }
}

async function loadHistory() {
  const results = $("jobs");
  try {
    const history = await api("/api/jobs");
    results.replaceChildren();
    if (!history.length)
      results.append(
        node("li", "No recordings yet. Upload one from New transcription."),
      );
    for (const entry of history) {
      const row = node("li");
      const button = node("button", entry.name, "jobs-link");
      button.type = "button";
      button.append(
        node("span", clock(entry.duration) + " · " + entry.status, "mono"),
      );
      button.addEventListener("click", async () => {
        if (!mayDiscard()) return;
        try {
          await selectJob(await api(jobURL(entry.id)));
          location.hash = "new";
        } catch (error) {
          results.prepend(node("li", error.message, "error"));
        }
      });
      row.append(button);
      results.append(row);
    }
  } catch (error) {
    results.replaceChildren(node("li", error.message, "error"));
  }
}

function route() {
  const selected = location.hash.slice(1);
  const current = document.querySelector("[data-page][aria-current=page]")
    ?.dataset.page;
  const page = ["new", "history", "settings", "about"].includes(selected)
    ? selected
    : (selected === "main" ? current : null) || "new";
  for (const section of document.querySelectorAll(".page"))
    section.hidden = section.id !== page;
  for (const link of document.querySelectorAll("[data-page]")) {
    if (link.dataset.page === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (page === "history") loadHistory();
}

function mayDiscard() {
  return !changesPending || confirm("Discard your unsaved transcript edits?");
}

async function selectJob(entry) {
  clearTimeout(poll);
  job = entry;
  transcript = undefined;
  drafts = [];
  changesPending = false;
  $("search").value = "";
  $("audio").src = jobURL(job.id) + "/media/audio";
  $("caption-track").removeAttribute("src");
  $("waveform").hidden = !job.waveform;
  $("wave-image").src = job.waveform ? jobURL(job.id) + "/media/waveform" : "";
  $("seek").max = job.duration || 1;
  $("seek").value = 0;
  $("recording-info").hidden = false;
  if (job.settings && currentConfig) {
    $("model").value = job.settings.model;
    $("language").value = job.settings.language || "";
    $("backend").value = job.settings.backend;
  }
  feedback("");
  await renderJob();
}

async function upload(file) {
  if (!file || uploading || !mayDiscard()) return;
  if (!/\.(m4a|mp3|wav|mp4|flac|webm)$/i.test(file.name)) {
    feedback("Choose M4A, MP3, WAV, MP4, FLAC, or WebM.", true);
    return;
  }
  uploading = true;
  $("file").disabled = true;
  feedback(
    "Uploading and preparing the audio preview. Large recordings can take a moment.",
  );
  const body = new FormData();
  body.append("file", file);
  try {
    await selectJob(await api("/api/jobs", { method: "POST", body }));
  } catch (error) {
    feedback(error.message, true);
  } finally {
    uploading = false;
    $("file").disabled = false;
  }
}

$("file").addEventListener("change", (event) => {
  upload(event.target.files[0]);
  event.target.value = "";
});
for (const name of ["dragenter", "dragover"])
  $("drop").addEventListener(name, (event) => {
    event.preventDefault();
    $("drop").classList.add("over");
  });
for (const name of ["dragleave", "drop"])
  $("drop").addEventListener(name, (event) => {
    event.preventDefault();
    $("drop").classList.remove("over");
    if (name === "drop") upload(event.dataTransfer.files[0]);
  });

async function renderJob() {
  if (!job) return;
  $("filename").textContent = job.name;
  $("metadata").textContent =
    clock(job.duration) + " · " + (job.size / 1048576).toFixed(2) + " MiB";
  const active = activeStates.has(job.status);
  const complete = job.status === "complete";
  $("options").hidden = complete;
  $("workspace").classList.toggle("reviewing", complete);
  $("progress-panel").hidden = ["ready", "complete"].includes(job.status);
  $("progress-panel").classList.toggle("is-running", active);
  $("pause").hidden = !active;
  $("pause").disabled = job.status === "pausing";
  $("transcribe").disabled = active || !currentConfig;
  $("transcribe").textContent = ["failed", "paused"].includes(job.status)
    ? "Resume or start again"
    : "Let’s transcribe!";
  for (const field of ["model", "language", "backend"])
    $(field).disabled = active;
  const fraction = job.duration ? Math.min(1, job.processed / job.duration) : 0;
  $("progress").value = fraction;
  $("percent").textContent = Math.floor(fraction * 100) + "%";
  if ($("stage").textContent !== job.stage) $("stage").textContent = job.stage;
  $("stage").classList.toggle("error", job.status === "failed");
  const transcribing =
    job.status === "running" && job.stage.startsWith("Transcribing ");
  $("processing-note").textContent = transcribing
    ? processingNotes[Math.floor(Date.now() / 9000) % processingNotes.length]
    : "";
  const chunk = transcribing
    ? " · Processing segment " + (job.segment + 1)
    : "";
  $("progress-meta").textContent = job.settings
    ? clock(job.processed) +
      " / " +
      clock(job.duration) +
      " · " +
      job.segment +
      " segments saved · " +
      job.device.toUpperCase() +
      " · " +
      job.settings.model +
      chunk
    : "";
  $("review").hidden = !complete;
  if (complete && !transcript) {
    const id = job.id;
    const result = await api(jobURL(id) + "/result");
    if (job.id === id) showResult(result);
  }
  clearTimeout(poll);
  if (active) poll = setTimeout(() => refreshJob(job.id), 800);
}

async function refreshJob(id) {
  try {
    const latest = await api(jobURL(id));
    if (job.id !== id) return;
    job = latest;
    await renderJob();
  } catch (error) {
    if (job.id !== id) return;
    feedback(error.message, true);
    if (activeStates.has(job.status))
      poll = setTimeout(() => refreshJob(id), 2500);
  }
}

$("transcribe").addEventListener("click", async () => {
  $("transcribe").disabled = true;
  try {
    const id = job.id;
    const next = await api(jobURL(id) + "/run", {
      method: "POST",
      body: {
        model: $("model").value,
        language: $("language").value || null,
        backend: $("backend").value,
      },
    });
    if (job.id !== id) return;
    job = next;
    await renderJob();
  } catch (error) {
    feedback(error.message, true);
    $("transcribe").disabled = false;
  }
});
$("pause").addEventListener("click", async () => {
  try {
    const id = job.id;
    const next = await api(jobURL(id) + "/pause", { method: "POST" });
    if (job.id !== id) return;
    job = next;
    await renderJob();
  } catch (error) {
    feedback(error.message, true);
  }
});

const audio = $("audio");
function seek(wanted) {
  audio.currentTime = Math.max(
    0,
    Math.min(audio.duration || job.duration, wanted),
  );
}
$("back").addEventListener("click", () => seek(audio.currentTime - 10));
$("forward").addEventListener("click", () => seek(audio.currentTime + 10));
$("seek").addEventListener("input", (event) =>
  seek(Number(event.target.value)),
);
$("speed").addEventListener("change", (event) => {
  audio.playbackRate = Number(event.target.value);
});
audio.addEventListener("loadedmetadata", () => {
  $("seek").max = audio.duration;
});
audio.addEventListener("timeupdate", () => {
  $("position").textContent = clock(audio.currentTime);
  $("seek").value = audio.currentTime;
  $("seek").setAttribute("aria-valuetext", clock(audio.currentTime));
});

function showResult(result) {
  transcript = result;
  drafts = result.segments.map((segment) => segment.text);
  changesPending = false;
  $("save").disabled = true;
  $("edit-state").textContent = result.segments.some(
    (segment) => "original_text" in segment,
  )
    ? "Edits saved"
    : "Machine-generated";
  $("caption-track").src = jobURL(job.id) + "/export/vtt?t=" + Date.now();
  $("exports").replaceChildren(
    ...formats.map(([suffix, label]) => {
      const link = node("a", label);
      link.href = jobURL(job.id) + "/export/" + suffix;
      return link;
    }),
  );
  renderSegments();
}

function highlight(text, query) {
  const match = node("div", "", "match");
  const expression = new RegExp(
    query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
    "gi",
  );
  let end = 0;
  for (const part of text.matchAll(expression)) {
    match.append(text.slice(end, part.index), node("mark", part[0]));
    end = part.index + part[0].length;
  }
  match.append(text.slice(end));
  return match;
}

function timestampButton(seconds, label) {
  const button = node("button", label || clock(seconds));
  button.type = "button";
  button.setAttribute(
    "aria-label",
    "Seek to " + clock(seconds) + (label ? " for " + label : ""),
  );
  button.addEventListener("click", () => seek(seconds));
  return button;
}

function renderSegments() {
  const container = $("segments");
  const query = $("search").value.trim();
  const mode = $("timestamps").value;
  const rows = [];
  for (const [index, segment] of transcript.segments.entries()) {
    if (query && !drafts[index].toLowerCase().includes(query.toLowerCase()))
      continue;
    const row = node("div", "", "segment");
    if (mode !== "none") row.append(timestampButton(segment.start));
    if (query) row.append(highlight(drafts[index], query));
    const edit = node("textarea");
    edit.rows = 2;
    edit.value = drafts[index];
    edit.setAttribute("aria-label", "Segment " + (index + 1) + " transcript");
    const words = node("div", "", "word-times");
    if (mode === "words" && segment.words.length) {
      words.append(
        ...segment.words.map((word) =>
          timestampButton(word.start, word.word.trim()),
        ),
      );
      words.hidden = edit.value !== segment.text;
    }
    edit.addEventListener("input", () => {
      drafts[index] = edit.value;
      words.hidden = edit.value !== segment.text;
      changesPending = drafts.some(
        (text, n) => text !== transcript.segments[n].text,
      );
      $("save").disabled = !changesPending;
      $("edit-state").textContent = changesPending
        ? "Unsaved edits"
        : "No unsaved edits";
    });
    row.append(edit, words);
    rows.push(row);
  }
  container.replaceChildren(...rows);
  $("search-status").textContent = query
    ? rows.length + " matching segments"
    : !rows.length
      ? "No spoken segments detected."
      : "";
}

$("search").addEventListener("input", renderSegments);
$("timestamps").value =
  localStorage.getItem("openscribe-timestamps") || "segments";
$("timestamps").addEventListener("change", () => {
  localStorage.setItem("openscribe-timestamps", $("timestamps").value);
  renderSegments();
});
$("save").addEventListener("click", async () => {
  $("save").disabled = true;
  try {
    const id = job.id;
    const result = await api(jobURL(id) + "/result", {
      method: "PUT",
      body: drafts,
    });
    if (job.id === id) showResult(result);
  } catch (error) {
    $("edit-state").textContent = error.message;
    $("save").disabled = false;
  }
});
$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(drafts.join("\n"));
    $("edit-state").textContent = changesPending
      ? "Copied draft · unsaved edits"
      : "Copied transcript";
  } catch {
    $("edit-state").textContent =
      "Clipboard unavailable. Save and download TXT.";
  }
});
window.addEventListener("beforeunload", (event) => {
  if (changesPending) event.preventDefault();
});
window.addEventListener("hashchange", route);
route();
configure();
