// index.js - overview page: the headline result and the dataset behind it.
document.addEventListener("DOMContentLoaded", () => {
  const banner = document.getElementById("page-banner");
  const verdict = document.getElementById("verdict");
  const strip = document.getElementById("overview");
  const pulseEl = document.getElementById("pulse");

  async function render() {
    clearChildren(banner);
    try {
      const [data, comparison] = await Promise.all([apiCall("/api/extraction"), apiCall("/api/models")]);
      const entries = (comparison.models || [])
        .filter((m) => m.role === "comparison" && m.metrics)
        .sort((a, b) => Number(b.metrics.macro_f1) - Number(a.metrics.macro_f1));
      const testPages = data.classes.reduce((sum, c) => sum + c.test, 0);

      clearChildren(verdict);
      if (!entries.length) {
        verdict.appendChild(paragraph(
          "No evaluated model was found. Run run_compare.py after training to build the comparison.",
          "verdict null"));
      } else {
        const top = entries[0];
        const last = entries[entries.length - 1];
        verdict.appendChild(paragraph(
          `${top.label} scores highest, at macro-F1 ${fmt(Number(top.metrics.macro_f1), 4)} `
          + `on ${testPages.toLocaleString()} held-out pages; ${last.label} is lowest at `
          + `${fmt(Number(last.metrics.macro_f1), 4)}. Higher is better.`,
          "verdict positive"));
      }

      clearChildren(strip);
      strip.append(
        statTile(data.prepared_total.toLocaleString(), "pages across 8 classes"),
        statTile(`${comparison.families_complete} of ${comparison.families_required}`, "model families evaluated"));
      entries.forEach((m) => strip.append(
        statTile(`${(Number(m.metrics.accuracy) * 100).toFixed(2)}%`, `${m.label} accuracy`)));
    } catch (err) {
      showBanner(banner, err.message, "error");
    }
  }

  watchRecord("/api/pulse", async (pulse) => {
    renderPulse(pulseEl, pulse, "pages");
    await render();
  });
});
