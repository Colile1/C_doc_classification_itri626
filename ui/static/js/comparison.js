// Comparison page: the three model families on one held-out split.
document.addEventListener("DOMContentLoaded", async () => {
  const banner = document.getElementById("page-banner");
  const status = document.getElementById("compare-status");
  try {
    const data = await apiCall("/api/models");
    const entries = data.models.filter((m) => m.role === "comparison" && m.metrics);
    entries.sort((a, b) => Number(b.metrics.macro_f1) - Number(a.metrics.macro_f1));

    status.textContent = `${data.families_complete} of ${data.families_required} families complete`;
    status.className = "pulse " + (data.families_complete >= 3 ? "idle" : "live");

    if (!entries.length) {
      throw new Error("No evaluated model found. Run run_compare.py first.");
    }

    const top = entries[0];
    const runnerUp = entries[1];

    // --- headline tiles -----------------------------------------------------
    const summary = document.getElementById("compare-summary");
    entries.forEach((m) => summary.append(
      statTile(fmt(Number(m.metrics.macro_f1), 4), `${m.label} macro-F1`)));
    summary.append(statTile("2,400", "held-out pages, identical for all"));

    // --- what this page is --------------------------------------------------
    document.getElementById("compare-intro").appendChild(paragraph(
      "Three architectural families, one fixed split. Each model was trained on the "
      + "same 11,200 pages, had its checkpoint chosen on the same 2,400 validation "
      + "pages, and was scored once on the same 2,400 held-out pages. All three "
      + "runs used one NVIDIA Tesla T4 at batch size 32, so no cross-machine "
      + "correction is needed.", "muted"));

    if (runnerUp) {
      const gap = Number(top.metrics.macro_f1) - Number(runnerUp.metrics.macro_f1);
      const topMs = top.efficiency ? Number(top.efficiency.median_ms_per_image) : null;
      const upMs = runnerUp.efficiency ? Number(runnerUp.efficiency.median_ms_per_image) : null;
      let verdict = `${top.label} has the highest macro-F1 at `
        + `${fmt(Number(top.metrics.macro_f1), 4)}, ahead of ${runnerUp.label} by `
        + `${gap.toFixed(4)}.`;
      if (topMs && upMs) {
        verdict += ` That lead costs ${(topMs / upMs).toFixed(1)}x the inference time `
          + `(${topMs.toFixed(2)} ms against ${upMs.toFixed(2)} ms per page).`;
      }
      document.getElementById("compare-caveat").appendChild(
        paragraph(verdict, "verdict positive"));
    }

    // --- the comparison table ----------------------------------------------
    document.getElementById("test-table").appendChild(buildTable(entries.map((m) => {
      const e = m.efficiency || {};
      return {
        model: m.label,
        family: m.family,
        "test accuracy": `${(Number(m.metrics.accuracy) * 100).toFixed(2)}%`,
        "test macro-F1": fmt(Number(m.metrics.macro_f1), 4),
        "ms/page": e.median_ms_per_image ? Number(e.median_ms_per_image).toFixed(2) : "-",
        parameters: e.total_parameters ? Number(e.total_parameters).toLocaleString() : "-",
        "size MB": e.size_mb ? Number(e.size_mb).toFixed(1) : "-",
      };
    })));

    // --- how the three differ ----------------------------------------------
    document.getElementById("changes-table").appendChild(buildTable(entries.map((m) => ({
      model: m.label,
      family: m.family,
      platform: m.platform,
      epochs: m.epochs_run || "-",
      "best epoch": m.best_epoch ?? "-",
      "best validation macro-F1": m.best_val_macro_f1 === null
        ? "-" : fmt(m.best_val_macro_f1, 4),
    }))));

    // --- strengths and limitations -----------------------------------------
    const notes = document.getElementById("improvement-notes");
    entries.forEach((m) => {
      const block = document.createElement("div");
      block.style.marginTop = "14px";
      const head = document.createElement("p");
      head.className = "card-title";
      head.style.fontSize = "0.95rem";
      head.textContent = `${m.label} — ${m.family}`;
      block.appendChild(head);
      block.appendChild(paragraph(m.summary, "muted"));
      const lists = document.createElement("div");
      lists.style.display = "grid";
      lists.style.gridTemplateColumns = "1fr 1fr";
      lists.style.gap = "12px";
      [["Strengths", m.pros], ["Limitations", m.cons]].forEach(([title, items]) => {
        const column = document.createElement("div");
        const label = document.createElement("p");
        label.className = "muted";
        label.style.fontWeight = "600";
        label.style.margin = "4px 0 2px";
        label.textContent = title;
        column.appendChild(label);
        const list = document.createElement("ul");
        list.style.margin = "0";
        list.style.paddingLeft = "18px";
        list.style.fontSize = "0.86rem";
        items.forEach((text) => {
          const li = document.createElement("li");
          li.textContent = text;
          list.appendChild(li);
        });
        column.appendChild(list);
        lists.appendChild(column);
      });
      block.appendChild(lists);
      notes.appendChild(block);
    });

    // --- per-class winners --------------------------------------------------
    // Map the raw run names onto the labels used everywhere else on the page.
    const naming = {};
    data.models.forEach((m) => { naming[m.name] = m.label; });
    const named = (key) => naming[key] || key || "-";

    const winners = document.getElementById("class-winners");
    if (winners && data.class_strengths && data.class_strengths.length) {
      winners.appendChild(buildTable(data.class_strengths.map((r) => ({
        class: r.class_name,
        "best model": named(r.best_model),
        F1: fmt(Number(r.best_f1), 4),
        "runner-up": named(r.runner_up),
        margin: r.margin ? Number(r.margin).toFixed(4) : "-",
      }))));
    }
  } catch (err) {
    status.textContent = "Failed to load";
    showBanner(banner, err.message, "error");
  }
});
