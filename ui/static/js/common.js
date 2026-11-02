// common.js - shared helpers for every page: fetch wrapper, nav-rail
// collapse toggle, and small DOM builders. No framework, no build step.

// What an empty table cell shows. Named rather than inlined because the
// column sorter has to recognise it to keep blanks out of the numeric
// comparison; two literals would silently drift apart.
const EMPTY_CELL = "-";

/**
 * Call a JSON API endpoint and return its parsed body.
 *
 * Inputs: path - the request path; options - fetch() options (method,
 * body, headers); the body, if given, is JSON-stringified automatically
 * unless it is already a FormData instance (file uploads).
 * Output: the parsed response body on 2xx.
 * Raises: an Error whose message is the API's own "detail" string (or the
 * raw body text if the response was not the expected JSON shape), so a
 * caller can show the engine's own reason instead of a generic failure.
 */
async function apiCall(path, options = {}) {
  const init = { ...options };
  if (init.body && !(init.body instanceof FormData)) {
    init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
    init.body = JSON.stringify(init.body);
  }
  const response = await fetch(path, init);
  const text = await response.text();
  let body;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : text || response.statusText;
    throw new Error(detail);
  }
  return body;
}

/** Remove every child of an element. Inputs: el - the element. */
function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

/**
 * Render a banner inside a container, replacing its content.
 * Inputs: container - element to render into; message - text to show;
 * kind - "error" or "info", used only for styling.
 */
function showBanner(container, message, kind = "error") {
  clearChildren(container);
  const div = document.createElement("div");
  div.className = `banner banner-${kind}`;
  div.textContent = message;
  container.appendChild(div);
}

/**
 * Build one headline number for a .stat-strip.
 * Inputs: value - the number or short text; label - what it counts.
 * Output: a .stat-tile element.
 */
function statTile(value, label) {
  const tile = document.createElement("div");
  tile.className = "stat-tile";
  const v = document.createElement("div");
  v.className = "stat-value";
  v.textContent = value;
  const l = document.createElement("div");
  l.className = "stat-label";
  l.textContent = label;
  tile.append(v, l);
  return tile;
}

/**
 * Build a sortable <table> from an array of row objects.
 * Inputs: rows - array of plain objects, all assumed to share (most of)
 * the same keys; the column list is taken from the union of keys across
 * rows, in first-seen order, so a column present on only some rows
 * (e.g. a blank catalogue cell) still gets a header.
 * Output: a <table> element, or null if rows is empty.
 */
function buildTable(rows) {
  if (!rows || rows.length === 0) return null;
  const columns = [];
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!columns.includes(key)) columns.push(key);
    }
  }
  const table = document.createElement("table");
  table.className = "data-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  columns.forEach((col, i) => {
    const th = document.createElement("th");
    th.textContent = col;
    th.dataset.col = String(i);
    th.addEventListener("click", () => sortTableByColumn(table, i));
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((col) => {
      const td = document.createElement("td");
      const value = row[col];
      td.textContent = value === null || value === undefined || value === "" ? EMPTY_CELL : String(value);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

/**
 * Sort a table's rows by one column, toggling ascending/descending on
 * repeated clicks of the same header. Numeric-looking values sort
 * numerically; everything else sorts as text.
 * Inputs: table - the <table> built by buildTable; colIndex - 0-based
 * column index.
 */
function sortTableByColumn(table, colIndex) {
  const tbody = table.querySelector("tbody");
  const rows = Array.from(tbody.querySelectorAll("tr"));
  const header = table.querySelectorAll("th")[colIndex];
  const ascending = header.dataset.sort !== "asc";
  table.querySelectorAll("th").forEach((th) => delete th.dataset.sort);
  header.dataset.sort = ascending ? "asc" : "desc";

  const valueOf = (tr) => tr.children[colIndex].textContent;
  const asNumber = (text) => (text === EMPTY_CELL ? NaN : Number(text));
  const allNumeric = rows.every((tr) => !Number.isNaN(asNumber(valueOf(tr))));

  rows.sort((a, b) => {
    const va = valueOf(a);
    const vb = valueOf(b);
    const cmp = allNumeric ? asNumber(va) - asNumber(vb) : va.localeCompare(vb);
    return ascending ? cmp : -cmp;
  });
  rows.forEach((tr) => tbody.appendChild(tr));
}

document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.querySelector(".nav-toggle");
  const rail = document.querySelector(".nav-rail");
  if (toggle && rail) {
    toggle.setAttribute("aria-expanded", "true");
    toggle.addEventListener("click", () => {
      rail.classList.toggle("collapsed");
      toggle.setAttribute("aria-expanded", String(!rail.classList.contains("collapsed")));
    });
  }
});
