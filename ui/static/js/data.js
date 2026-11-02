// data.js - dataset page: extraction estimate and per-class counts.
document.addEventListener("DOMContentLoaded", () => {
  const banner = document.getElementById("page-banner");
  const verdict = document.getElementById("verdict");
  const strip = document.getElementById("overview");
  const chart = document.getElementById("chart");
  const wrap = document.getElementById("table-wrap");
  const pulseEl = document.getElementById("pulse");

  function barRow(name, pct, valueText) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = name;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = Math.min(100, pct).toFixed(1) + "%";
    track.appendChild(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = valueText;
    row.append(label, track, value);
    return row;
  }

  async function render() {
    clearChildren(banner);
    try {
      const data = await apiCall("/api/extraction");
      if (!data.labels_present) {
        showBanner(banner, "No split summary found in " + data.extracted_dir
          + ". Run run_prepare_data.py and run_split.py to create it.", "error");
        return;
      }
      const labelled = data.classes.reduce((sum, c) => sum + c.labelled, 0);
      const complete = data.labels_present && data.estimated_pct >= 99.5;

      clearChildren(verdict);
      verdict.appendChild(paragraph(
        complete
          ? data.prepared_total.toLocaleString() + " prepared examples are represented in the split files."
          : "Prepared split files are not available yet.",
        "verdict " + (complete ? "positive" : "unclear")));

      clearChildren(strip);
      strip.append(
        statTile(labelled.toLocaleString(), "prepared images, 8 classes"),
        statTile(data.classes.reduce((sum, c) => sum + c.train, 0).toLocaleString(), "training images"),
        statTile(data.classes.reduce((sum, c) => sum + c.validation, 0).toLocaleString(), "validation images"),
        statTile(data.classes.reduce((sum, c) => sum + c.test, 0).toLocaleString(), "held out test images")
      );

      clearChildren(chart);
      const bars = document.createElement("div");
      bars.className = "bar-chart";
      for (const c of data.classes) {
        const trainPct = c.labelled ? (100 * c.train) / c.labelled : 0;
        bars.appendChild(barRow(c.class, trainPct, c.train + " train / " + c.validation + " val / " + c.test + " test"));
      }
      chart.appendChild(bars);

      clearChildren(wrap);
      wrap.appendChild(buildTable(data.classes.map((c) => ({
        class: c.class,
        labelled: c.labelled,
        train: c.train,
        validation: c.validation,
        test: c.test,
      }))));
    } catch (err) {
      showBanner(banner, err.message, "error");
    }
  }

  watchRecord("/api/pulse", async (pulse) => {
    renderPulse(pulseEl, pulse, "images");
    await render();
  });
});
