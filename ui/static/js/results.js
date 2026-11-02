// results.js - every training run, the three-family comparison and the curves.
document.addEventListener("DOMContentLoaded", async () => {
  const banner = document.getElementById("page-banner");
  try {
    const [runs, data, models] = await Promise.all([
      apiCall("/api/training-runs"),
      apiCall("/api/results"),
      apiCall("/api/models"),
    ]);

    // --- headline -----------------------------------------------------------
    const status = document.getElementById("run-status");
    status.textContent = `${models.families_complete} of ${models.families_required} families complete`;
    status.className = "pulse " + (models.families_complete >= 3 ? "idle" : "live");

    const comparison = (data.rows || []).filter((r) => r.macro_f1);
    comparison.sort((a, b) => Number(b.macro_f1) - Number(a.macro_f1));
    const summary = document.getElementById("training-summary");
    comparison.forEach((row) =>
      summary.append(statTile(fmt(Number(row.macro_f1), 4), `${row.model} macro-F1`)));
    summary.append(statTile(String(runs.count), "runs recorded"));

    document.getElementById("training-notes").appendChild(paragraph(
      "Every model was trained on the same 11,200 pages, selected on the same 2,400 "
      + "validation pages, and scored once on the same 2,400 held-out pages. The three "
      + "runs used one Tesla T4 at batch size 32, so the comparison needs no "
      + "cross-machine caveat.", "muted"));

    // --- the comparison table ----------------------------------------------
    if (comparison.length) {
      const best = Math.max(...comparison.map((r) => Number(r.macro_f1)));
      const winner = comparison.find((r) => Number(r.macro_f1) === best);
      document.getElementById("verdict").appendChild(paragraph(
        `${winner.model} has the highest macro-F1 (${fmt(best, 4)}). `
        + "Accuracy is not the whole recommendation: inference cost and model size "
        + "are compared below.", "verdict positive"));
      document.getElementById("table-wrap").appendChild(buildTable(comparison.map((r) => ({
        model: r.model,
        family: r.family || "-",
        accuracy: `${(Number(r.accuracy) * 100).toFixed(2)}%`,
        "macro-F1": fmt(Number(r.macro_f1), 4),
        "ms/page": r.median_ms_per_image ? Number(r.median_ms_per_image).toFixed(2) : "-",
        parameters: r.total_parameters ? Number(r.total_parameters).toLocaleString() : "-",
        "size MB": r.size_mb ? Number(r.size_mb).toFixed(1) : "-",
      }))));
    } else {
      showBanner(banner, "No evaluated model yet. Run run_compare.py.", "error");
    }

    // --- every run ----------------------------------------------------------
    document.getElementById("runs-table").appendChild(buildTable(runs.runs.map((r) => ({
      model: r.label,
      family: r.family,
      role: r.role,
      platform: r.platform,
      epochs: r.epochs || "-",
      "best epoch": r.best_epoch ?? "-",
      "val macro-F1": r.best_val_macro_f1 === null ? "-" : fmt(r.best_val_macro_f1, 4),
      "test macro-F1": r.test_macro_f1 === null ? "-" : fmt(r.test_macro_f1, 4),
    }))));

    // --- curves, every model on shared axes ---------------------------------
    const withHistory = models.models.filter((m) => m.history && m.history.length);
    if (withHistory.length) {
      document.getElementById("loss-chart").appendChild(lineChart(
        withHistory.map((m) => ({
          name: m.label,
          values: m.history.map((r) => ({ x: Number(r.epoch), y: Number(r.val_loss) })),
        })),
        { yLabel: "Validation loss", xLabel: "Epoch",
          caption: "Lower is better. Every model overfits the training split; the "
                 + "validation loss is what matters for generalisation." }));
      document.getElementById("score-chart").appendChild(lineChart(
        withHistory.map((m) => ({
          name: m.label,
          values: m.history.map((r) => ({ x: Number(r.epoch), y: Number(r.val_macro_f1) })),
        })),
        { yLabel: "Validation macro-F1", xLabel: "Epoch", min: 0.4, max: 1,
          caption: "Higher is better. Macro-F1 gives each of the eight classes equal "
                 + "weight, so a model cannot hide a weak class behind a strong one." }));
    }

    // --- per-class ----------------------------------------------------------
    const entries = models.models.filter((m) => m.role === "comparison" && m.per_class.length);
    if (entries.length) {
      const classes = {};
      entries.forEach((m) => m.per_class.forEach((r) => {
        classes[r.class_name] = classes[r.class_name] || { class: r.class_name };
        classes[r.class_name][m.label] = Number(r.f1);
      }));
      const ordered = Object.values(classes).sort((a, b) => {
        const low = (row) => Math.min(...entries.map((m) => row[m.label] ?? 1));
        return low(a) - low(b);
      });
      document.getElementById("class-table").appendChild(buildTable(ordered.map((row) => {
        const out = { class: row.class };
        entries.forEach((m) => { out[m.label] = row[m.label] === undefined ? "-" : fmt(row[m.label], 4); });
        return out;
      })));
    }

    // --- class winners ------------------------------------------------------
    if (models.class_strengths && models.class_strengths.length) {
      document.getElementById("artifacts").appendChild(buildTable(
        models.class_strengths.map((r) => ({
          class: r.class_name,
          "best model": r.best_model,
          F1: fmt(Number(r.best_f1), 4),
          "runner-up": r.runner_up || "-",
          margin: r.margin ? Number(r.margin).toFixed(4) : "-",
        }))));
    }
  } catch (err) {
    showBanner(banner, err.message, "error");
  }
});
