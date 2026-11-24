const $ = (id) => document.getElementById(id);
const running = new Set(["queued", "running", "pausing"]);
const templates = {
  summary: "Detailed summary",
  minutes: "Meeting minutes",
  actions: "Action items",
  requirements: "Business requirements",
  development: "Software development",
  sales: "Sales & discovery",
  research: "Research interview",
  lecture: "Educational lecture",
  podcast: "Podcast",
  board: "Board meeting",
  strategy: "Strategic planning",
  conversation: "General conversation",
  custom: "Custom analysis",
};

export function createIntelligence({ api, clock, node, cite }) {
  let job;
  let mode = "transcript";
  let configuration;
  let data;
  let poll;
  let questionContext = "";
  let requestNumber = 0;
  let modelDiscovery = 0;

  const url = () => "/api/jobs/" + job + "/intelligence";
  const provider = () =>
    configuration?.providers.find((p) => p.id === $("ai-provider").value);
  const options = (element, values) =>
    element.replaceChildren(
      ...values.map(([value, name]) => {
        const item = node("option", name);
        item.value = value;
        return item;
      }),
    );

  function message(text, error = false) {
    $("ai-message").textContent = text;
    $("ai-message").classList.toggle("error", error);
  }

  function button(text, action) {
    const element = node("button", text);
    element.type = "button";
    element.addEventListener("click", async () => {
      element.disabled = true;
      try {
        await action();
      } catch (error) {
        message(error.message, true);
      } finally {
        element.disabled = false;
      }
    });
    return element;
  }

  async function checkConnection() {
    const selected = provider();
    const ticket = ++modelDiscovery;
    options($("ai-model"), []);
    $("ai-generate").disabled = true;
    const response = await api(
      "/api/intelligence/providers/" + selected.id + "/models",
    );
    if (ticket !== modelDiscovery) return;
    options(
      $("ai-model"),
      response.models.map((model) => [model, model]),
    );
    $("ai-generate").disabled = !response.models.length;
    message(
      response.models.length
        ? "Connected · " + response.models.length + " models available."
        : "Connected. Install a model on this provider and check again.",
    );
  }

  function updateProvider() {
    const selected = provider();
    $("ai-context").value = selected.context;
    $("ai-consent").checked = false;
    $("ai-consent-field").hidden = !selected.external;
    $("ai-privacy").textContent = selected.external
      ? "External endpoint · transcript data leaves this app"
      : "Local endpoint · loopback connection";
    checkConnection().catch((error) => message(error.message, true));
  }

  async function configure() {
    if (configuration) return;
    configuration = await api("/api/intelligence/providers");
    options(
      $("ai-provider"),
      configuration.providers.map((p) => [p.id, p.label]),
    );
    options($("ai-template"), Object.entries(templates));
    options(
      $("ai-detail"),
      configuration.details.map((detail) => [
        detail,
        detail[0].toUpperCase() + detail.slice(1),
      ]),
    );
    $("ai-detail").value = "standard";
    updateProvider();
  }

  async function history() {
    const path = url();
    const list = await api(path);
    if (path !== url()) return;
    const visible = list.filter(
      (item) => (item.settings.template === "ask") === (mode === "ask"),
    );
    options($("ai-history"), [
      ["", "Select a saved run"],
      ...visible.map((item) => [
        item.id,
        item.title +
          " · " +
          item.status +
          (item.outdated ? " · older transcript" : ""),
      ]),
    ]);
    if (data) $("ai-history").value = data.id;
  }

  function showTab(tab) {
    clearTimeout(poll);
    requestNumber++;
    const changed = mode !== tab;
    mode = tab;
    for (const item of document.querySelectorAll("[data-review]")) {
      item.setAttribute("aria-pressed", String(item.dataset.review === tab));
    }
    const ai = tab === "intelligence" || tab === "ask";
    $("ai-view").hidden = !ai;
    $("transcript-view").hidden = tab !== "transcript";
    $("exports-view").hidden = tab !== "exports";
    if (!ai || !job) return;
    const ask = tab === "ask";
    $("ai-heading").textContent = ask
      ? "Ask about this recording"
      : "More than words.";
    $("ai-description").textContent = ask
      ? "Ask a question and jump directly to the evidence."
      : "Make sense of every conversation with source-linked findings.";
    $("ai-prompt-label").textContent = ask
      ? "Your question"
      : "Custom instructions";
    $("ai-prompt").required = ask;
    $("ai-template-field").hidden = ask;
    $("ai-detail-field").hidden = ask;
    $("ai-generate").textContent = ask
      ? "Ask the transcript"
      : "Generate Intelligence";
    if (changed) {
      $("ai-prompt").value = "";
      data = undefined;
      $("ai-result").replaceChildren();
    }
    configure()
      .then(history)
      .catch((error) => message(error.message, true));
  }

  async function load(id) {
    const ticket = ++requestNumber;
    clearTimeout(poll);
    const response = await api(url() + "/" + id);
    if (ticket !== requestNumber || !["intelligence", "ask"].includes(mode))
      return;
    data = response;
    render();
    if (running.has(data.status))
      poll = setTimeout(() => {
        load(id).catch((error) => message(error.message, true));
      }, 900);
    else await history();
  }

  async function generate(settings) {
    const path = url();
    const response = await api(path, { method: "POST", body: settings });
    if (path !== url()) return;
    await history();
    await load(response.id);
  }

  function authorize(settings) {
    const selected = configuration.providers.find(
      (p) => p.id === settings.provider,
    );
    if (!selected?.external) return false;
    if (selected === provider() && $("ai-consent").checked) return true;
    return confirm("Send transcript data to " + selected.label + "?");
  }

  function render() {
    const root = $("ai-result");
    const item = data;
    const base = url() + "/" + item.id;
    root.replaceChildren();
    if (item.outdated)
      root.append(
        node(
          "p",
          "This run used a previous transcript version. Generate again to use saved edits.",
          "ai-warning",
        ),
      );
    if (item.settings.template === "ask")
      root.append(node("p", item.settings.instructions));
    const summary = node(
      "p",
      item.completed +
        " of " +
        item.total +
        " parts processed · " +
        item.covered +
        " of " +
        item.source_count +
        " source segments covered",
      "mono",
    );
    root.append(summary);
    const progress = node("progress");
    progress.max = item.total;
    progress.value = item.completed;
    progress.setAttribute("aria-label", "Analysis parts completed");
    root.append(progress);
    if (item.error)
      root.append(
        node("p", item.error, item.status === "failed" ? "error" : ""),
      );
    if (running.has(item.status)) {
      const pause = button("Pause after this part", async () => {
        await api(base + "/pause", { method: "POST" });
        await load(item.id);
      });
      pause.disabled = item.status === "pausing";
      root.append(pause);
      return;
    }
    if (item.status !== "complete" && item.next_sources) {
      root.append(
        node(
          "p",
          "Next part includes source segments " +
            item.next_sources.join("–") +
            ".",
          "mono",
        ),
      );
    }
    if (item.status !== "complete") {
      root.append(
        button("Resume", async () => {
          await api(base + "/retry?consent=" + authorize(item.settings), {
            method: "POST",
          });
          await load(item.id);
        }),
      );
    }
    const trash = button("Delete run", async () => {
      if (
        !confirm(
          "Permanently delete this run, its source snapshot, and its revisions?",
        )
      )
        return;
      await api(base, { method: "DELETE" });
      data = undefined;
      root.replaceChildren();
      await history();
    });
    if (item.status !== "complete") {
      root.append(trash);
      return;
    }

    const titleLabel = node("label", "Analysis title");
    const title = node("input");
    title.value = item.title;
    title.maxLength = 160;
    titleLabel.append(title);
    root.append(titleLabel);
    const drafts = [];
    const selections = [];
    const references = new Map(item.references.map((ref) => [ref.id, ref]));
    item.sections.forEach((section, index) => {
      const details = node("details", "", "ai-section");
      details.open = index === 0;
      details.append(node("summary", section.title));
      const group = [];
      drafts.push(group);
      for (const finding of section.findings) {
        const article = node("article", "", "ai-finding");
        article.append(
          node(
            "div",
            "generated_text" in finding
              ? "User edited · check cited evidence"
              : finding.kind + " · AI draft",
            "mono",
          ),
        );
        const editor = node("textarea");
        editor.rows = 3;
        editor.value = finding.text;
        editor.setAttribute(
          "aria-label",
          section.title + " finding " + (group.length + 1),
        );
        group.push(editor);
        article.append(editor);
        const citations = node("div", "", "ai-citations");
        for (const id of finding.sources) {
          const ref = references.get(id);
          citations.append(
            button("#" + id + " · " + clock(ref.start), () => {
              showTab("transcript");
              cite(ref);
            }),
          );
        }
        article.append(citations);
        details.append(article);
      }
      const select = node("label", "Include in export ", "ai-checkbox");
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.checked = true;
      selections.push(checkbox);
      select.append(checkbox);
      details.append(
        select,
        button("Copy section", () =>
          navigator.clipboard.writeText(
            group.map((field) => field.value).join("\n"),
          ),
        ),
        button("Regenerate section", () =>
          generate({
            ...item.settings,
            section: section.title,
            consent: authorize(item.settings),
          }),
        ),
      );
      root.append(details);
    });
    const actions = node("div", "", "review-actions");
    actions.append(
      button("Save analysis edits", async () => {
        await api(base, {
          method: "PUT",
          body: {
            title: title.value,
            sections: drafts.map((group) => group.map((field) => field.value)),
          },
        });
        await load(item.id);
        message("Analysis edits saved.");
      }),
      button("Generate again", () =>
        generate({ ...item.settings, consent: authorize(item.settings) }),
      ),
      trash,
    );
    root.append(actions);

    const attribution = node(
      "label",
      "Include OpenScribe and FWDP attribution ",
      "ai-checkbox",
    );
    const include = node("input");
    include.type = "checkbox";
    include.checked = true;
    attribution.append(include);
    root.append(attribution);
    const exports = node("div", "", "export-toolbar");
    for (const suffix of ["txt", "md", "json", "pdf", "docx"]) {
      const link = node("a", suffix.toUpperCase());
      link.href = base + "/export/" + suffix;
      link.addEventListener("click", (event) => {
        const query = new URLSearchParams({
          attribution: String(include.checked),
        });
        for (const [index, checkbox] of selections.entries())
          if (checkbox.checked) query.append("selected", index);
        if (selections.length && !query.has("selected")) {
          event.preventDefault();
          message("Select at least one section to export.", true);
          return;
        }
        link.href = base + "/export/" + suffix + "?" + query;
      });
      exports.append(link);
    }
    root.append(exports);
  }

  for (const tab of document.querySelectorAll("[data-review]"))
    tab.addEventListener("click", () => showTab(tab.dataset.review));
  $("ai-provider").addEventListener("change", updateProvider);
  $("ai-discover").addEventListener("click", () =>
    checkConnection().catch((error) => message(error.message, true)),
  );
  $("ai-history").addEventListener("change", (event) => {
    if (event.target.value)
      load(event.target.value).catch((error) => message(error.message, true));
  });
  $("ai-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const ask = mode === "ask";
    const question = $("ai-prompt").value.trim();
    const settings = {
      provider: provider().id,
      model: $("ai-model").value,
      context: Number($("ai-context").value),
      language: $("ai-language").value,
      template: ask ? "ask" : $("ai-template").value,
      detail: ask ? "standard" : $("ai-detail").value,
      instructions: question,
      consent: $("ai-consent").checked,
    };
    if (ask && questionContext)
      settings.instructions =
        "Previous question: " +
        questionContext.slice(0, 150) +
        "\nFollow-up: " +
        question;
    $("ai-generate").disabled = true;
    try {
      await generate(settings);
      if (ask) questionContext = question;
    } catch (error) {
      message(error.message, true);
    } finally {
      $("ai-generate").disabled = false;
    }
  });

  return {
    recording(id) {
      if (job !== id) {
        job = id;
        questionContext = "";
        showTab("transcript");
      } else if (["intelligence", "ask"].includes(mode)) {
        if (data) load(data.id).catch((error) => message(error.message, true));
        else history().catch((error) => message(error.message, true));
      }
    },
  };
}
