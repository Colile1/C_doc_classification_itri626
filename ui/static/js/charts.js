// charts.js - optional shared pieces for data pages: auto-refresh, SVG
// charts, number formatting, expandable explanation blocks, CSV export.
// Load after common.js (it uses apiCall and clearChildren).
//
// Charts are hand-drawn SVG rather than a charting library, because the
// app has no build step and loads no third-party script. That also lets
// every chart carry a real axis label, a key and a caption, which most
// defaults do not.
//
// Everything here assumes a reader who may never have seen the subject
// before. Plain words first, jargon only once it has been defined, and
// no figure without a sentence saying what to look at.

// ---------------------------------------------------------------- refresh

/**
 * Re-run `onChange` whenever the underlying record changes.
 *
 * Polls a cheap pulse endpoint rather than the expensive views, backs
 * off when nothing is happening, and stops entirely while the tab is
 * hidden so a forgotten tab costs nothing.
 *
 * The pulse endpoint returns {exists, running, mtime, total}: `mtime`
 * and `total` together decide whether anything changed, `running`
 * picks the fast or slow interval.
 *
 * Inputs: pulseUrl - the endpoint to poll; onChange - called with the
 * pulse body when it differs.
 * Output: a stop() function.
 */
function watchRecord(pulseUrl, onChange) {
  const FAST = 3000;   // something is running
  const SLOW = 15000;  // nothing is running
  let last = null;
  let timer = null;
  let stopped = false;

  async function tick() {
    if (stopped) return;
    if (document.hidden) {
      timer = setTimeout(tick, SLOW);
      return;
    }
    let pulse = null;
    try {
      pulse = await apiCall(pulseUrl);
    } catch (err) {
      timer = setTimeout(tick, SLOW);
      return;
    }
    const key = `${pulse.mtime}:${pulse.total}`;
    if (key !== last) {
      last = key;
      try {
        await onChange(pulse);
      } catch (err) {
        /* a render failure must not kill the watcher */
      }
    }
    timer = setTimeout(tick, pulse.running ? FAST : SLOW);
  }

  tick();
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !stopped) {
      clearTimeout(timer);
      tick();
    }
  });
  return () => { stopped = true; clearTimeout(timer); };
}

/**
 * Show a live/idle badge so the reader knows the page is current.
 * Inputs: element - the .pulse span; pulse - the pulse body; noun - what
 * is being counted, plural ("runs", "orders").
 */
function renderPulse(element, pulse, noun = "records") {
  clearChildren(element);
  if (!pulse || !pulse.exists) {
    element.textContent = `No ${noun} yet`;
    element.className = "pulse idle";
    return;
  }
  element.className = pulse.running ? "pulse live" : "pulse idle";
  element.textContent = pulse.running
    ? `Updating live - ${pulse.total} ${noun} so far`
    : `Up to date - ${pulse.total} ${noun}`;
}

// ---------------------------------------------------------------- numbers

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return Number(value).toFixed(digits);
}

function fmtDuration(seconds) {
  if (!seconds && seconds !== 0) return "-";
  if (seconds < 60) return `${seconds.toFixed(1)} seconds`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} minutes`;
  return `${(seconds / 3600).toFixed(1)} hours`;
}

/**
 * Describe a p-value in words.
 * Readers who do not work with statistics read "p = 0.04" as a
 * measurement; it is closer to a decision rule, so say what it decided.
 */
function pInWords(p, alpha = 0.05) {
  if (p === null || p === undefined) return "not tested";
  if (p < 0.001) return "very strong evidence of a difference";
  if (p < 0.01) return "strong evidence of a difference";
  if (p < alpha) return "evidence of a difference";
  if (p < 0.1) return "a hint, but not enough to call it";
  return "no evidence of a difference";
}

// ------------------------------------------------------------- expandable

/**
 * A disclosure block: a one-line summary that opens to more depth.
 *
 * This is how one page serves a newcomer and a researcher at once. The
 * visible text never assumes prior knowledge; the detail behind it adds
 * the method, the numbers and the caveats.
 */
function explainer(summaryText, bodyNodes, open = false) {
  const details = document.createElement("details");
  details.className = "explainer";
  if (open) details.open = true;
  const summary = document.createElement("summary");
  summary.textContent = summaryText;
  details.appendChild(summary);
  const body = document.createElement("div");
  body.className = "explainer-body";
  (Array.isArray(bodyNodes) ? bodyNodes : [bodyNodes]).forEach((node) => {
    body.appendChild(typeof node === "string" ? paragraph(node) : node);
  });
  details.appendChild(body);
  return details;
}

function paragraph(text, className = "") {
  const p = document.createElement("p");
  if (className) p.className = className;
  p.textContent = text;
  return p;
}

function heading(text, level = 3, className = "") {
  const h = document.createElement(`h${level}`);
  h.textContent = text;
  if (className) h.className = className;
  return h;
}

function bulletList(items) {
  const ul = document.createElement("ul");
  ul.className = "plain-list";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    ul.appendChild(li);
  });
  return ul;
}

// ----------------------------------------------------------------- charts

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, String(v)));
  return node;
}

// A colour-blind-safe sequence. Colour is never the only channel: every
// series is also named in the key and labelled on hover.
const SERIES_COLOURS = [
  "#1a73e8", "#d93025", "#188038", "#b06000",
  "#8430ce", "#007b83", "#5f6368", "#c5221f",
];

/**
 * Grouped bar chart with axes, gridlines, a key and value labels.
 *
 * Inputs: series - {name: [{x, mean, n, low, high}]}; options - title,
 * xLabel, yLabel, higherIsBetter.
 * Output: a figure element ready to append.
 */
function barChart(series, options = {}) {
  const {
    xLabel = "", yLabel = "", higherIsBetter = true, caption = "",
  } = options;

  const names = Object.keys(series);
  const categories = [];
  names.forEach((name) => series[name].forEach((point) => {
    if (!categories.includes(point.x)) categories.push(point.x);
  }));

  const W = 760, H = 340;
  const pad = { top: 16, right: 16, bottom: 64, left: 72 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const values = names.flatMap((n) => series[n].map((p) => p.mean));
  if (!values.length) return paragraph("Nothing to plot.", "muted");
  let lo = Math.min(...values), hi = Math.max(...values);
  const span = hi - lo || Math.abs(hi) || 1;
  lo = lo - span * 0.15;
  hi = hi + span * 0.15;
  // Quantities that cannot be negative must not be drawn as if they
  // could: a time axis reaching -2.4 seconds reads as a bug and
  // undermines the chart beside it. Clamp to zero for those.
  if (Math.min(...values) >= 0) lo = Math.max(0, lo);

  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`, class: "chart",
    role: "img", "aria-label": `${yLabel} by ${xLabel}`,
  });

  const yOf = (v) => pad.top + plotH - ((v - lo) / (hi - lo)) * plotH;

  // Gridlines and y axis ticks.
  for (let i = 0; i <= 4; i += 1) {
    const value = lo + ((hi - lo) * i) / 4;
    const y = yOf(value);
    svg.appendChild(svgEl("line", {
      x1: pad.left, y1: y, x2: pad.left + plotW, y2: y,
      stroke: "#e8eaed", "stroke-width": 1,
    }));
    const label = svgEl("text", {
      x: pad.left - 8, y: y + 4, "text-anchor": "end", class: "tick",
    });
    label.textContent = fmt(value, Math.abs(value) >= 100 ? 0 : 3);
    svg.appendChild(label);
  }

  // Axis lines.
  svg.appendChild(svgEl("line", {
    x1: pad.left, y1: pad.top, x2: pad.left, y2: pad.top + plotH,
    stroke: "#5f6368", "stroke-width": 1,
  }));
  svg.appendChild(svgEl("line", {
    x1: pad.left, y1: pad.top + plotH, x2: pad.left + plotW, y2: pad.top + plotH,
    stroke: "#5f6368", "stroke-width": 1,
  }));

  const slot = plotW / categories.length;
  const barW = Math.min(46, (slot * 0.7) / names.length);

  categories.forEach((category, ci) => {
    const centre = pad.left + slot * ci + slot / 2;
    names.forEach((name, ni) => {
      const point = series[name].find((p) => p.x === category);
      if (!point) return;
      const offset = (ni - (names.length - 1) / 2) * barW;
      const x = centre + offset - barW / 2;
      const y = yOf(point.mean);
      const rect = svgEl("rect", {
        x, y, width: barW, height: Math.max(1, pad.top + plotH - y),
        fill: SERIES_COLOURS[ni % SERIES_COLOURS.length],
      });
      const title = svgEl("title");
      title.textContent =
        `${name} - ${category}: ${fmt(point.mean)}`
        + (point.n ? ` (averaged over ${point.n})` : "");
      rect.appendChild(title);
      svg.appendChild(rect);
    });
    const label = svgEl("text", {
      x: centre, y: pad.top + plotH + 18, "text-anchor": "middle", class: "tick",
    });
    label.textContent = category;
    svg.appendChild(label);
  });

  // Axis titles.
  const xTitle = svgEl("text", {
    x: pad.left + plotW / 2, y: H - 26, "text-anchor": "middle", class: "axis-title",
  });
  xTitle.textContent = xLabel;
  svg.appendChild(xTitle);

  const yTitle = svgEl("text", {
    x: 16, y: pad.top + plotH / 2, "text-anchor": "middle",
    transform: `rotate(-90 16 ${pad.top + plotH / 2})`, class: "axis-title",
  });
  yTitle.textContent = yLabel + (higherIsBetter ? " (higher is better)" : " (lower is better)");
  svg.appendChild(yTitle);

  const figure = document.createElement("figure");
  figure.className = "chart-figure";
  figure.appendChild(svg);
  figure.appendChild(chartKey(names));
  if (caption) {
    const cap = document.createElement("figcaption");
    cap.textContent = caption;
    figure.appendChild(cap);
  }
  return figure;
}

/** Draw epoch metrics as accessible SVG lines. */
function lineChart(series, options = {}) {
  const { xLabel = "Epoch", yLabel = "Value", caption = "", min = null, max = null } = options;
  const values = series.flatMap((item) => item.values.map((point) => point.y));
  if (!values.length) return paragraph("No epoch metrics are available yet.", "muted");
  let lo = min === null ? Math.min(...values) : min;
  let hi = max === null ? Math.max(...values) : max;
  if (lo === hi) { lo = Math.min(0, lo - 1); hi += 1; }
  if (min === null) { const pad = (hi - lo) * 0.12; lo = Math.max(0, lo - pad); hi += pad; }
  const W = 760, H = 310, pad = { top: 18, right: 22, bottom: 54, left: 68 };
  const plotW = W - pad.left - pad.right, plotH = H - pad.top - pad.bottom;
  const epochs = series.flatMap((item) => item.values.map((point) => point.x));
  const xMin = Math.min(...epochs), xMax = Math.max(...epochs);
  const xOf = (x) => pad.left + (xMax === xMin ? plotW / 2 : ((x - xMin) / (xMax - xMin)) * plotW);
  const yOf = (y) => pad.top + plotH - ((y - lo) / (hi - lo)) * plotH;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": `${yLabel} by ${xLabel}` });
  for (let i = 0; i <= 4; i += 1) {
    const value = lo + ((hi - lo) * i) / 4, y = yOf(value);
    svg.appendChild(svgEl("line", { x1: pad.left, y1: y, x2: pad.left + plotW, y2: y, stroke: "#e8eaed", "stroke-width": 1 }));
    const tick = svgEl("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", class: "tick" });
    tick.textContent = fmt(value, 2); svg.appendChild(tick);
  }
  svg.appendChild(svgEl("line", { x1: pad.left, y1: pad.top + plotH, x2: pad.left + plotW, y2: pad.top + plotH, stroke: "#5f6368" }));
  const yTitle = svgEl("text", { x: 16, y: pad.top + plotH / 2, "text-anchor": "middle", transform: `rotate(-90 16 ${pad.top + plotH / 2})`, class: "axis-title" });
  yTitle.textContent = yLabel; svg.appendChild(yTitle);
  const xTitle = svgEl("text", { x: pad.left + plotW / 2, y: H - 14, "text-anchor": "middle", class: "axis-title" });
  xTitle.textContent = xLabel; svg.appendChild(xTitle);
  series.forEach((item, index) => {
    const points = item.values.map((point) => `${xOf(point.x)},${yOf(point.y)}`).join(" ");
    svg.appendChild(svgEl("polyline", { points, fill: "none", stroke: SERIES_COLOURS[index % SERIES_COLOURS.length], "stroke-width": 2.5, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    item.values.forEach((point) => {
      const mark = svgEl("circle", { cx: xOf(point.x), cy: yOf(point.y), r: 3, fill: SERIES_COLOURS[index % SERIES_COLOURS.length] });
      const title = svgEl("title"); title.textContent = `${item.name}, epoch ${point.x}: ${fmt(point.y, 4)}`; mark.appendChild(title); svg.appendChild(mark);
    });
  });
  const figure = document.createElement("figure"); figure.className = "chart-figure";
  figure.append(svg, chartKey(series.map((item) => item.name)));
  if (caption) { const cap = document.createElement("figcaption"); cap.textContent = caption; figure.appendChild(cap); }
  return figure;
}

/** A named key. Colour alone never carries the meaning. */
function chartKey(names) {
  const key = document.createElement("div");
  key.className = "chart-key";
  names.forEach((name, i) => {
    const item = document.createElement("span");
    item.className = "key-item";
    const swatch = document.createElement("span");
    swatch.className = "key-swatch";
    swatch.style.background = SERIES_COLOURS[i % SERIES_COLOURS.length];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(name));
    key.appendChild(item);
  });
  return key;
}

/**
 * Horizontal "win/loss" bar: in how many cases an option helped.
 *
 * Reading a median difference needs practice; "better on 10 of 11
 * datasets" does not. This draws that directly.
 * Inputs: the three counts; noun - what a case is, plural.
 */
function winLossBar(improved, worsened, tied, noun = "cases") {
  const total = improved + worsened + tied || 1;
  const wrap = document.createElement("div");
  wrap.className = "winloss";
  const bar = document.createElement("div");
  bar.className = "winloss-bar";
  [
    ["better", improved, "#188038"],
    ["tied", tied, "#9aa0a6"],
    ["worse", worsened, "#d93025"],
  ].forEach(([label, count, colour]) => {
    if (!count) return;
    const segment = document.createElement("span");
    segment.className = "winloss-seg";
    segment.style.width = `${(count / total) * 100}%`;
    segment.style.background = colour;
    segment.title = `${count} ${label}`;
    bar.appendChild(segment);
  });
  wrap.appendChild(bar);
  const caption = document.createElement("span");
  caption.className = "winloss-caption";
  caption.textContent =
    `better on ${improved} of ${total} ${noun}`
    + (worsened ? `, worse on ${worsened}` : "")
    + (tied ? `, tied on ${tied}` : "");
  wrap.appendChild(caption);
  return wrap;
}

/** Download an array of row objects as a CSV, built in the browser. */
function downloadCsv(rows, filename) {
  if (!rows.length) return;
  const columns = Object.keys(rows[0]);
  const escape = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [
    columns.join(","),
    ...rows.map((row) => columns.map((c) => escape(row[c])).join(",")),
  ].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
