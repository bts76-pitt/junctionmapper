/* junctionmapper report: state, data loading, views. Needs JMLogic and Plotly. */
(function () {
  "use strict";
  const L = window.JMLogic;
  const JM = (window.JM = window.JM || {});

  /*** CONSTANTS ***/

  const ROLE_ORDER = ["JM1", "JM2"];
  const METHOD_NAMES = {tcj: "Tricellular corners", dp: "Douglas–Peucker"};
  const METHOD_HELP = {
    tcj: "Junction ends at the corners where three cells meet.",
    dp: "Junction ends at corners from Douglas–Peucker simplification of the outline.",
  };
  const MAX_POINTS = 6000;   // per trace; past this the violin is drawn without its dots
  const ICONS = {
    good: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    warning: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5l7 13H1z" fill="currentColor"/><path d="M8 6v4M8 12v.5" stroke="#0b0b0b" stroke-width="1.6" stroke-linecap="round"/></svg>',
  };

  /*** RUN REGISTRY AND DATA LOADER ***/

  JM.runs = [];
  JM.registerRun = function (summary) { JM.runs.push(summary); };

  const loader = L.createLoader(function (src, onload, onerror) {
    const s = document.createElement("script");
    s.src = src;
    s.onload = function () { s.remove(); onload(); };
    s.onerror = function () { s.remove(); onerror(); };
    document.head.appendChild(s);
  });
  JM.loader = loader;
  JM.registerIndex = function (p) { loader.receive("Index", p); };
  JM.registerMetric = function (p) { loader.receive("Metric", p); };
  JM.registerField = function (p) { loader.receive("Field", p); };
  JM.registerThumbs = function (p) { loader.receive("Thumbs", p); };

  const decoded = new Map();   // src -> decoded payload

  function loadIndex(run, method) {
    const src = `runs/${run}/metrics/${method}/_index.js`;
    return loader.load(src, {run}).then(function (got) {
      if (!decoded.has(src)) {
        const p = got.payload;
        decoded.set(src, {
          run, method, fields: p.fields,
          jField: L.decodeU32(p.junction_field), jLo: L.decodeU32(p.junction_lo),
          jHi: L.decodeU32(p.junction_hi),
          cField: L.decodeU32(p.cell_field), cId: L.decodeU32(p.cell_id),
        });
      }
      return decoded.get(src);
    });
  }

  function loadMetric(run, method, role, key) {
    const src = `runs/${run}/metrics/${method}/${role}/${key}.js`;
    return loader.load(src, {run}).then(function (got) {
      if (!decoded.has(src)) {
        const p = got.payload;
        const dec = b => (b ? L.decodeF32(b) : null);
        decoded.set(src, {run, junction_px: dec(p.junction_px), junction_um: dec(p.junction_um),
                          cell_px: dec(p.cell_px), cell_um: dec(p.cell_um)});
      }
      return decoded.get(src);
    });
  }
  // {field_id: data URI} for one run; an older run without thumbs.js gives {}.
  function loadThumbs(run) {
    return loader.load(`runs/${run}/thumbs.js`, {run}).then(got => got.payload, () => ({}));
  }

  // Thumbnails arrive after the page: each .thumb[data-run][data-field] gets its image.
  function fillThumbs(root) {
    const wells = [...root.querySelectorAll(".thumb[data-field]")];
    [...new Set(wells.map(w => w.dataset.run))].forEach(run => loadThumbs(run).then(map => {
      wells.filter(w => w.dataset.run === run && map[w.dataset.field] && !w.firstChild)
        .forEach(w => w.append(h("img", {src: map[w.dataset.field], alt: "", decoding: "async"})));
    }));
  }

  JM.data = {loadIndex, loadMetric, loadThumbs};

  /*** STATE ***/

  const state = (JM.state = {
    run: "all",            // "all" or a run id
    role: "JM1",
    metric: "band_mean_intensity",
    method: "tcj",
    useUm: true,
    dot: "junction",       // "junction" | "cell"
    flags: {},             // QC flag -> highlighted; all off
    table: false,          // distribution card shows its table twin
  });

  const listeners = [];
  JM.onChange = function (fn) { listeners.push(fn); };
  function set(patch) {
    Object.assign(state, patch);
    listeners.forEach(fn => fn(state));
    render();
  }
  JM.set = set;

  function selectedRuns() {
    return state.run === "all" ? JM.runs : JM.runs.filter(r => r.run === state.run);
  }
  const runIndex = id => JM.runs.findIndex(r => r.run === id);
  const runById = id => JM.runs.find(r => r.run === id);
  const allCalibrated = runs => runs.length > 0 && runs.every(r => r.calibrated);
  // Definitions are the same in every run; search them all so a metric only a later run has resolves.
  const metricInfo = key => L.unionMetrics(JM.runs).find(m => m.key === key);

  const rank = role => (ROLE_ORDER.includes(role) ? ROLE_ORDER.indexOf(role) : ROLE_ORDER.length);

  function roles(runs) {
    const seen = new Map();
    runs.forEach(r => r.roles.forEach(x => {
      if (!seen.has(x.role)) seen.set(x.role, new Set());
      seen.get(x.role).add(x.marker);
    }));
    return [...seen.keys()]
      .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
      .map(role => ({role, markers: [...seen.get(role)]}));
  }
  // "P-cadherin (JM1)" from the markers the runs in view give this role.
  function roleLabel(role, runs) {
    const r = roles(runs).find(x => x.role === role);
    return L.channelLabel(role, r ? r.markers.join(" / ") : "");
  }
  const methods = runs => [...new Set(runs.flatMap(r => r.methods))].sort(
    (a, b) => (a === "tcj" ? -1 : b === "tcj" ? 1 : a.localeCompare(b)));

  function initState() {
    if (JM.runs.length === 1) state.run = JM.runs[0].run;
  }

  // The runs a view shows: the field's own run in a field view, else the run filter's.
  function viewRuns(r) {
    const run = r.view === "field" ? runById(r.run) : null;
    return run ? [run] : selectedRuns();
  }

  // Marker, corner method and metric must exist in the runs in view; a run filter can remove them.
  function normalizeState(r) {
    const runs = viewRuns(r);
    state.role = L.pickRole(roles(runs), state.role);
    const ms = methods(runs);
    if (ms.length && !ms.includes(state.method)) state.method = ms[0];
    const mets = L.unionMetrics(runs);
    if (mets.length && !mets.some(m => m.key === state.metric)) state.metric = mets[0].key;
  }

  function theme() {
    const forced = document.documentElement.getAttribute("data-theme");
    if (forced) return forced;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  JM.theme = theme;
  const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /*** DOM HELPERS ***/

  // Every label from data goes in through textContent, never innerHTML.
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "tip") el.dataset.tip = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  JM.h = h;

  function icon(name) {
    const span = document.createElement("span");
    span.style.display = "inline-flex";
    span.innerHTML = ICONS[name];   // static markup, no data
    return span;
  }

  function info(text) {
    return h("button", {type: "button", class: "info", tip: text, "aria-label": text}, "i");
  }

  function control(label, tip, el) {
    return h("div", {class: "control"},
      h("span", {class: "control-label"}, label, tip ? info(tip) : null), el);
  }

  function select(options, value, onchange, tip) {
    const el = h("select", {class: "select", tip, onchange: e => onchange(e.target.value)},
      options.map(o => h("option", {value: o.value, title: o.title, selected: o.value === value}, o.label)));
    return el;
  }

  function segmented(options, value, onpick, label) {
    return h("div", {class: "seg", role: "group", "aria-label": label},
      options.map(o => h("button", {
        type: "button", "aria-pressed": String(o.value === value), tip: o.tip, disabled: o.disabled,
        onclick: () => onpick(o.value)}, o.label)));
  }

  /*** TOOLTIP ***/

  function initTooltip() {
    const tip = document.getElementById("tooltip");
    let anchor = null;
    function show(el) {
      anchor = el;
      tip.textContent = el.dataset.tip;
      tip.hidden = false;
      const r = el.getBoundingClientRect(), t = tip.getBoundingClientRect();
      let x = r.left + r.width / 2 - t.width / 2, y = r.bottom + 8;
      x = Math.max(8, Math.min(x, window.innerWidth - t.width - 8));
      if (y + t.height > window.innerHeight - 8) y = r.top - t.height - 8;
      tip.style.left = `${x}px`;
      tip.style.top = `${y}px`;
    }
    function hide() { anchor = null; tip.hidden = true; }
    const target = e => (e.target.closest ? e.target.closest("[data-tip]") : null);
    document.addEventListener("pointerover", e => { const el = target(e); if (el && el !== anchor) show(el); else if (!el) hide(); });
    document.addEventListener("focusin", e => { const el = target(e); if (el) show(el); });
    document.addEventListener("focusout", hide);
    document.addEventListener("scroll", hide, true);
    document.addEventListener("keydown", e => { if (e.key === "Escape") hide(); });
  }

  /*** ROUTING ***/

  JM.views = JM.views || {};

  function route() {
    const m = location.hash.match(/^#\/field\/([^/]+)\/([^?]+)(?:\?(.*))?$/);
    if (m) {
      const params = new URLSearchParams(m[3] || "");
      return {view: "field", run: m[1], field: decodeURIComponent(m[2]),
              junction: params.get("j"), cell: params.get("c")};
    }
    if (/^#\/provenance\/?$/.test(location.hash)) return {view: "provenance"};
    return {view: "overview"};
  }
  JM.route = route;
  JM.fieldHref = (run, field, sel) => `#/field/${run}/${encodeURIComponent(field)}${sel ? `?${sel}` : ""}`;

  /*** FILTER ROW ***/

  function renderFilters(r) {
    const box = document.getElementById("filters");
    box.replaceChildren();
    box.hidden = r.view === "provenance";
    if (box.hidden) return;
    const runs = viewRuns(r);
    if (r.view === "overview" && JM.runs.length > 1) {
      box.append(control("Run", null, select(
        [{value: "all", label: `All runs (${JM.runs.length})`}]
          .concat(JM.runs.map(x => ({value: x.run, label: x.label}))),
        state.run, v => set({run: v}), "Runs shown below")));
    }
    box.append(control("Marker", "Measured junction channel", select(
      roles(runs).map(x => ({value: x.role, label: L.channelLabel(x.role, x.markers.join(" / "))})),
      state.role, v => set({role: v}))));
    const m = metricInfo(state.metric);
    box.append(control("Metric", m.help + (m.td ? " Depends on the marker threshold." : ""), select(
      L.unionMetrics(runs).map(x => ({value: x.key, label: x.label, title: x.help})),
      state.metric, v => set({metric: v}))));
    box.append(control("Corners", null, segmented(
      methods(runs).map(x => ({value: x, label: x, tip: `${METHOD_NAMES[x] || x}. ${METHOD_HELP[x] || ""}`})),
      state.method, v => set({method: v}), "Corner method")));
    if (allCalibrated(runs)) {
      // Never disabled: it also sets the units of every metric in the detail panel.
      const note = m.unit_um ? "" : " (this metric has no length unit)";
      box.append(control("Units", null, segmented([
        {value: true, label: "µm", tip: `Calibrated units${note}`},
        {value: false, label: "px", tip: `Image pixels${note}`},
      ], state.useUm, v => set({useUm: v}), "Units")));
    }
    box.append(control("Dots", null, segmented([
      {value: "junction", label: "Junction", tip: "One dot per junction: mean of its cell sides"},
      {value: "cell", label: "Cell", tip: "One dot per cell: median over its junctions"},
    ], state.dot, v => set({dot: v}), "One dot per")));
  }

  function renderCrumbs(r) {
    const nav = document.getElementById("crumbs");
    nav.replaceChildren();
    const label = JM.runs.length === 1 ? JM.runs[0].label : `Comparison of ${JM.runs.length} runs`;
    if (r.view === "overview") {
      nav.append(h("span", {class: "here"}, label));
    } else if (r.view === "provenance") {
      nav.append(h("a", {href: "#/"}, label), h("span", {class: "sep"}, "/"), h("span", {class: "here"}, "Provenance"));
    } else {
      const run = runById(r.run);
      nav.append(h("a", {href: "#/"}, label), h("span", {class: "sep"}, "/"),
                 h("span", {class: "here"}, `${run && JM.runs.length > 1 ? run.label + " / " : ""}${r.field}`));
    }
  }

  /*** OVERVIEW ***/

  // One chip per run-level warning; the detail is in the tooltip and in the provenance view.
  function warningChips(run) {
    return (run.warnings || []).map(w => h("a", {class: "warn-chip", href: "#/provenance", tip: w.detail,
                                                 "data-kind": w.kind}, icon("warning"), w.text));
  }

  function tile(label, tip, value, sub) {
    return h("div", {class: "card tile"},
      h("span", {class: "tile-label", tip, tabindex: "0"}, label),
      h("span", {class: "tile-value"}, value),
      sub ? h("span", {class: "tile-sub"}, sub) : null);
  }

  function renderTiles(runs) {
    const sum = k => runs.reduce((a, r) => a + r.tiles[k], 0);
    const cells = sum("cells");
    const across = runs.length > 1 ? `across ${runs.length} runs` : null;
    return h("div", {class: "tiles"},
      tile("Cells", "Cells with at least one measured interface", L.fmtCount(cells), across),
      tile("Junctions", "Distinct cell–cell junctions (tricellular corners)", L.fmtCount(sum("junctions")), across),
      tile("Fields", "Images in the run", L.fmtCount(sum("fields")), across),
      tile("Flagged cells", "Measured cells with at least one QC flag (TCJ/DP corner disagreement not counted)", L.fmtShare(sum("flagged_cells"), cells), across));
  }

  function qcBadge(qc, cells) {
    const n = qc.flagged_cells || 0;
    if (!n) return h("span", {class: "badge good", tip: "No QC flags"}, icon("good"), "Clean");
    const detail = Object.entries(qc).filter(([k, v]) => k !== "flagged_cells" && v > 0)
      .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(", ");
    return h("span", {class: "badge warning", tip: detail || `${n} flagged cells`}, icon("warning"), L.fmtShare(n, cells).replace(/^(\S+)/, "$1 flagged"));
  }

  function fieldCard(run, f) {
    return h("a", {class: "card field-card", href: JM.fieldHref(run.run, f.field_id),
                   "aria-label": `Open field ${f.field_id}`},
      h("div", {class: "thumb", "data-run": run.run, "data-field": f.field_id}),
      h("div", {class: "field-meta"},
        h("span", {class: "field-name", title: f.field_id}, f.field_id),
        h("span", {class: "field-row"}, `${L.fmtCount(f.n_cells)} cells, ${L.fmtCount(f.n_junctions)} junctions`),
        qcBadge(f.qc, f.n_cells)));
  }

  function renderFieldGrid(runs) {
    const total = runs.reduce((a, r) => a + r.fields.length, 0);
    const wrap = h("section", null,
      h("div", {class: "section-head"}, h("h2", null, "Fields"), h("span", {class: "count"}, L.fmtCount(total))));
    runs.forEach(run => {
      const group = h("div", {class: "run-group"});
      if (JM.runs.length > 1) {
        group.append(h("div", {class: "run-group-head"},
          h("span", {class: "swatch", style: `background:${L.runColour(runIndex(run.run), theme())}`}), run.label));
      }
      group.append(h("div", {class: "field-grid"}, run.fields.map(f => fieldCard(run, f))));
      wrap.append(group);
    });
    return wrap;
  }

  /*** DISTRIBUTION ***/

  let renderToken = 0;

  // One trace per run: values, hover text and click targets for the current state.
  function seriesFor(run, index, metric, m, calibrated) {
    const cell = state.dot === "cell";
    const picked = L.metricValues(metric, cell ? "cell" : "junction", m, calibrated, state.useUm);
    const {values, index: pos} = L.finite(picked.values);
    const text = pos.map(i => (cell
      ? `Cell ${index.cId[i]} in ${index.fields[index.cField[i]]}`
      : `Junction ${L.junctionKey(index.jLo[i], index.jHi[i])} in ${index.fields[index.jField[i]]}`));
    return {run, values, pos, text, index, unit: picked.unit};
  }

  function plotTraces(series, unit) {
    const surface = cssVar("--surface"), ink2 = cssVar("--ink-2"), t = theme();
    return series.map(s => {
      const colour = L.runColour(runIndex(s.run.run), t);
      const n = s.values.length;
      return {
        type: "violin", name: escapeHtml(s.run.label),
        x: new Array(n).fill(`${escapeHtml(s.run.label)}<br><span style="font-size:11px">n = ${L.fmtCount(n)}</span>`),
        y: s.values, text: s.text, customdata: s.pos,
        points: n <= MAX_POINTS ? "all" : false, jitter: 0.45, pointpos: 0, spanmode: "hard",
        marker: {size: 6, color: colour, opacity: 0.65, line: {width: 1, color: surface}},
        line: {color: colour, width: 1.5}, fillcolor: hexA(colour, 0.12),
        box: {visible: true, width: 0.08, fillcolor: surface, line: {color: ink2, width: 1}},
        meanline: {visible: false},
        hoveron: n <= MAX_POINTS ? "points" : "violins",
        hovertemplate: `<b>%{y:.4~g}${unit ? " " + unit : ""}</b><br>%{text}<extra>${escapeHtml(s.run.label)}</extra>`,
      };
    });
  }

  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  }

  function plotLayout(title) {
    const ink2 = cssVar("--ink-2"), muted = cssVar("--muted"), grid = cssVar("--grid"), axis = cssVar("--axis");
    return {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: {family: cssVar("--font"), size: 12, color: ink2},
      margin: {l: 64, r: 16, t: 8, b: 56},
      // Each run is its own x category, so the run names sit under the violins: no legend.
      showlegend: false,
      violinmode: "overlay", violingap: 0.3,
      xaxis: {type: "category", showgrid: false, linecolor: axis, tickfont: {color: ink2}, fixedrange: true},
      yaxis: {title: {text: title, font: {color: muted, size: 12}}, gridcolor: grid, zeroline: false,
              linecolor: axis, tickfont: {color: muted}},
      modebar: {bgcolor: "rgba(0,0,0,0)", color: muted, activecolor: cssVar("--accent")},
      hoverlabel: {bgcolor: cssVar("--ink"), bordercolor: cssVar("--ink"), font: {color: cssVar("--page"), family: cssVar("--font")}},
      dragmode: "zoom",
    };
  }

  const PLOT_CONFIG = {displaylogo: false, responsive: true, displayModeBar: "hover",
                       modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toImage"]};

  function statsTable(series, unit) {
    const head = ["Run", "n", "Min", "Q1", "Median", "Q3", "Max"];
    return h("table", {class: "data-table"},
      h("thead", null, h("tr", null, head.map((c, i) => h("th", {scope: "col"}, i > 1 && unit ? `${c} (${unit})` : c)))),
      h("tbody", null, series.map(s => {
        const st = L.summaryStats(s.values);
        return h("tr", null,
          h("td", null, h("span", {class: "swatch", style: `background:${L.runColour(runIndex(s.run.run), theme())}`}), s.run.label),
          h("td", null, L.fmtCount(st.n)),
          [st.min, st.q1, st.median, st.q3, st.max].map(v => h("td", null, L.fmtValue(v))));
      })));
  }

  function renderDistribution(runs) {
    const m = metricInfo(state.metric);
    const calibrated = allCalibrated(runs);
    let unit = L.unitFor(m, calibrated, state.useUm);
    // The plot element outlives re-renders, so a reload holds the previous chart, dimmed.
    const plot = JM._lastPlot || h("div", {class: "plot", id: "dist-plot"});
    const note = h("div", {class: "chart-note", hidden: true});
    const tableBox = h("div", {hidden: !state.table});
    plot.hidden = state.table;
    const dotWord = state.dot === "cell" ? "cell" : "junction";
    const subText = () => `${roleLabel(state.role, runs)}, ${state.method}, one dot per ${dotWord}${unit ? ", " + unit : ""}`;
    const sub = h("span", {class: "chart-sub"}, subText());
    const card = h("section", {class: "card chart-card"},
      h("div", {class: "chart-head"},
        h("h2", {class: "chart-title", tip: m.help}, m.label),
        sub,
        segmented([{value: false, label: "Chart"}, {value: true, label: "Table"}], state.table,
                  v => set({table: v}), "Chart or table")),
      plot, tableBox, note);

    JM._lastPlot = plot;
    const token = ++renderToken;
    plot.classList.add("is-loading");
    Promise.all(runs.map(run => Promise.all([
      loadIndex(run.run, state.method),
      loadMetric(run.run, state.method, state.role, state.metric),
    ]).then(([index, metric]) => seriesFor(run, index, metric, m, calibrated), () => null)))
      .then(all => {
        if (token !== renderToken) return;
        plot.classList.remove("is-loading");
        const series = all.filter(Boolean);
        // The unit of the arrays actually plotted (one rule for every run in view).
        if (series.length) { unit = series[0].unit; sub.textContent = subText(); }
        const missing = runs.filter((r, i) => !all[i]).map(r => r.label);
        if (missing.length) {
          note.hidden = false;
          note.textContent = `No ${roleLabel(state.role, runs)} ${state.method} values: ${missing.join(", ")}`;
        }
        tableBox.replaceChildren(statsTable(series, unit));
        if (!series.length) { Plotly.purge(plot); plot.hidden = true; return; }
        if (state.table) return;
        const title = `${m.label}${unit ? ` (${unit})` : ""}`;
        Plotly.react(plot, plotTraces(series, unit), plotLayout(title), PLOT_CONFIG);
        plot.removeAllListeners("plotly_click");
        plot.on("plotly_click", ev => {
          const pt = ev.points && ev.points[0];
          if (!pt || pt.customdata === undefined) return;
          const s = series[pt.curveNumber];
          const t = L.pointTarget(s.index, pt.customdata, state.dot);
          location.hash = JM.fieldHref(s.run.run, t.field, t.sel);
        });
      });
    return card;
  }

  /*** COMPARISON ***/

  // Several runs side by side: one row per run (click to show only that run), then one violin per run.
  function renderRunTable(runs) {
    const head = [["Run"], ["Fields"], ["Cells", "Cells with at least one measured interface"],
                  ["Junctions", "Distinct cell–cell junctions (tricellular corners)"],
                  ["Flagged cells", "Measured cells with at least one QC flag (TCJ/DP corner disagreement not counted)"],
                  ["Warnings", "Run-level problems; hover a warning for detail"]];
    return h("section", {class: "card run-card"},
      h("table", {class: "data-table run-table"},
        h("thead", null, h("tr", null, head.map(([c, tip]) => h("th", {scope: "col"},
          tip ? h("span", {tip, tabindex: "0"}, c) : c)))),
        h("tbody", null, L.runRows(runs).map(row => h("tr", null,
          h("td", null, h("button", {type: "button", class: "run-link", tip: "Show only this run",
                                     onclick: () => set({run: row.run})},
            h("span", {class: "swatch", style: `background:${L.runColour(runIndex(row.run), theme())}`}), row.label)),
          h("td", null, L.fmtCount(row.fields)),
          h("td", null, L.fmtCount(row.cells)),
          h("td", null, L.fmtCount(row.junctions)),
          h("td", null, L.fmtShare(row.flagged, row.cells)),
          h("td", {class: "warn-cell"}, warnOrDash(runById(row.run))))))));
  }

  function warnOrDash(run) {
    const chips = warningChips(run);
    return chips.length ? chips : h("span", {class: "muted"}, "–");
  }

  JM.views.overview = function (root) {
    const runs = selectedRuns();
    const compare = JM.runs.length > 1 && state.run === "all";
    const chips = compare ? [] : runs.flatMap(warningChips);
    if (chips.length) root.append(h("div", {class: "warn-row", "aria-label": "Run warnings"}, chips));
    root.append(compare ? renderRunTable(runs) : renderTiles(runs), renderDistribution(runs),
                renderFieldGrid(runs));
    fillThumbs(root);
  };

  /*** FIELD VIEW ***/

  // Additive tints per channel role; the image well is always dark.
  const TINT = {segment: [1, 1, 1], JM1: [0.25, 1, 0.35], JM2: [1, 0.3, 1], nucleus: [0.35, 0.55, 1]};
  const TINT_OTHER = [1, 1, 1];
  const LINE_PX = 3;              // junction stroke, screen px
  const HIT_PX = 6;               // junction hit tolerance, screen px
  const FLAG_COLOUR = "#ec835a";  // status orange: not on the viridis junction ramp
  // A light casing under every junction keeps the ramp's dark end readable on a dark image; the
  // selected or hovered junction adds a dark outer ring, so it never looks like the others.
  const CASING = "rgba(255,255,255,0.85)";
  const HOT_RING = "rgba(11,11,11,0.9)";
  const NO_JUNCTIONS = "No junctions measured in this field.";

  // Viewer settings outlive re-renders: channel layers by role, overlays, and per-field view/selection.
  const layers = {};               // role -> {on, gain}
  const overlay = {junctions: true, outlines: true};
  let fieldUi = {key: null, xf: null, sel: null};
  // "run/field" -> {src, promise of the decoded field}. Each holds preview-size planes, a composite and
  // paths (~15 MB), so only the last FIELD_CACHE opened stay; the loader forgets the evicted file too.
  const FIELD_CACHE = 4;
  const fieldData = L.createLru(FIELD_CACHE, (key, v) => {
    loader.forget(v.src);
    v.promise.then(fd => { if (fd.composite) fd.composite.width = 0; fd.planes = null; fd.composite = null; },
                   () => {});
  });
  JM._cleanup = [];

  const pretty = s => s.replace(/_/g, " ");
  const jLabel = key => key.replace("-", "–");

  function loadField(run, fieldId) {
    const key = `${run}/${fieldId}`;
    const hit = fieldData.get(key);
    if (hit) return hit.promise;
    const src = `runs/${run}/fields/${encodeURIComponent(fieldId)}.js`;
    const promise = loader.load(src, {run}).then(got => decodeField(got.payload));
    fieldData.set(key, {src, promise});
    promise.catch(() => { if (fieldData.get(key)) fieldData.delete(key); });
    return promise;
  }

  function greyPlane(src, w, hgt) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = w; c.height = hgt;
        const g = c.getContext("2d");
        g.drawImage(img, 0, 0, w, hgt);
        const d = g.getImageData(0, 0, w, hgt).data, out = new Uint8ClampedArray(w * hgt);
        for (let i = 0; i < out.length; i++) out[i] = d[i * 4];
        resolve(out);
      };
      img.onerror = () => reject(new Error("preview did not decode"));
      img.src = src;
    });
  }

  function pathOf(lines) {
    const p = new Path2D();
    lines.forEach(line => line.forEach(([x, y], k) => (k ? p.lineTo(x, y) : p.moveTo(x, y))));
    return p;
  }

  function decodeField(f) {
    // Planes are decoded at the preview size, the space every outline and junction is in.
    const {width, height} = L.planeSize(f);
    return Promise.all(f.channels.map(c => greyPlane(c.src, width, height))).then(planes => {
      const cellPath = {};
      for (const id in f.cells) { const p = pathOf([f.cells[id]]); p.closePath(); cellPath[id] = p; }
      const allOutlines = new Path2D();
      Object.values(cellPath).forEach(p => allOutlines.addPath(p));
      const junctionPath = {};
      for (const m in f.junctions) {
        junctionPath[m] = {};
        for (const k in f.junctions[m]) junctionPath[m][k] = pathOf(f.junctions[m][k]);
      }
      return {raw: f, width, height, planes,
              channels: f.channels.map(c => ({role: c.role, marker: c.marker})),
              cellPath, allOutlines, junctionPath, composite: null, compositeSig: null};
    });
  }

  function layerFor(role) {
    if (!layers[role]) layers[role] = {on: role !== "nucleus", gain: 1};
    return layers[role];
  }

  // Additive composite of the channels that are on, each tinted and scaled by its gain.
  function composite(fd) {
    const sig = fd.channels.map(c => { const l = layerFor(c.role); return `${l.on}:${l.gain}`; }).join("|");
    if (fd.compositeSig === sig) return fd.composite;
    const n = fd.width * fd.height;
    const rgb = new Float32Array(n * 3);
    fd.channels.forEach((c, ci) => {
      const l = layerFor(c.role);
      if (!l.on) return;
      const t = TINT[c.role] || TINT_OTHER, g = fd.planes[ci];
      for (let i = 0; i < n; i++) {
        const v = g[i] * l.gain;
        rgb[i * 3] += v * t[0]; rgb[i * 3 + 1] += v * t[1]; rgb[i * 3 + 2] += v * t[2];
      }
    });
    const canvas = fd.composite || document.createElement("canvas");
    canvas.width = fd.width; canvas.height = fd.height;
    const g = canvas.getContext("2d"), img = g.createImageData(fd.width, fd.height), d = img.data;
    for (let i = 0; i < n; i++) {
      d[i * 4] = rgb[i * 3]; d[i * 4 + 1] = rgb[i * 3 + 1]; d[i * 4 + 2] = rgb[i * 3 + 2]; d[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    fd.composite = canvas;
    fd.compositeSig = sig;
    return canvas;
  }

  // Everything the view needs for the current marker / metric / corner method.
  function fieldModel(run, fd, index, metric, m) {
    const fieldPos = index.fields.indexOf(fd.raw.field_id);
    const slice = L.fieldSlice({junction_field: index.jField, junction_lo: index.jLo, junction_hi: index.jHi,
                                cell_field: index.cField, cell_id: index.cId}, fieldPos);
    const jPick = L.metricValues(metric, "junction", m, run.calibrated, state.useUm);
    const cPick = L.metricValues(metric, "cell", m, run.calibrated, state.useUm);
    const jArr = jPick.values, cArr = cPick.values;
    const jValue = new Map(slice.junctions.map(j => [j.key, jArr ? jArr[j.i] : NaN]));
    const cValue = new Map(slice.cells.map(c => [String(c.cell_id), cArr ? cArr[c.i] : NaN]));
    const [lo, hi] = L.quantileRange([...jValue.values()]);
    const byCell = new Map();
    slice.junctions.forEach(j => j.key.split("-").forEach(c => {
      if (!byCell.has(c)) byCell.set(c, []);
      byCell.get(c).push(j.key);
    }));
    return {slice, jValue, cValue, lo, hi, byCell, unit: jPick.unit, cUnit: cPick.unit,
            measured: new Set(slice.cells.map(c => String(c.cell_id))), missing: !metric,
            polylines: fd.raw.junctions[state.method] || {}};
  }

  // The junction keys a selection lights up, or null when nothing is selected.
  function emphasis(sel, model) {
    if (sel.brush) {
      if (state.dot !== "cell") return new Set(sel.brush);
      return new Set(sel.brush.flatMap(c => model.byCell.get(c) || []));
    }
    // A junction this field does not have (a stale link) highlights nothing rather than dims everything.
    if (sel.junction) return model.jValue.has(sel.junction) ? new Set([sel.junction]) : null;
    if (sel.cell) return new Set(model.byCell.get(sel.cell) || []);
    return null;
  }

  /*** FIELD VIEW: CANVAS ***/

  function createViewer(fd, model, sel, hooks) {
    const wrap = h("div", {class: "viewer", style: `--ar:${fd.width} / ${fd.height}`});
    const canvas = h("canvas", {tabindex: "0", "aria-label": "Field image. Scroll to zoom, drag to pan, double-click to fit."});
    const readout = h("div", {class: "viewer-readout", hidden: true});
    wrap.append(canvas, readout);
    const ctx = canvas.getContext("2d");
    let xf = fieldUi.xf;          // {k, tx, ty}; null until the first fit
    let hover = null, frame = 0;

    function fit() {
      const w = wrap.clientWidth, hh = wrap.clientHeight;
      const k = Math.min(w / fd.width, hh / fd.height);
      xf = {k, tx: (w - fd.width * k) / 2, ty: (hh - fd.height * k) / 2, fitted: true};
      fieldUi.xf = xf;
    }

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(wrap.clientWidth * dpr));
      canvas.height = Math.max(1, Math.round(wrap.clientHeight * dpr));
      if (!xf || xf.fitted) fit();
      draw();
    }

    function draw() { if (!frame) frame = requestAnimationFrame(paint); }

    function paint() {
      frame = 0;
      if (!xf) return;
      const dpr = window.devicePixelRatio || 1, k = xf.k;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = cssVar("--image-well") || "#0d0d0d";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * xf.tx, dpr * xf.ty);
      ctx.imageSmoothingEnabled = k < 2;
      ctx.drawImage(composite(fd), 0, 0);
      ctx.lineJoin = "round"; ctx.lineCap = "round";

      if (overlay.outlines) {
        ctx.lineWidth = 1 / k;
        ctx.strokeStyle = "rgba(255,255,255,0.28)";
        ctx.stroke(fd.allOutlines);
      }
      const flagOn = Object.keys(state.flags).filter(n => state.flags[n]);
      if (flagOn.length) {
        ctx.setLineDash([5 / k, 3 / k]);
        ctx.lineWidth = 2 / k;
        ctx.strokeStyle = FLAG_COLOUR;
        for (const id in fd.raw.flags) {
          if (model.measured.has(id) && fd.cellPath[id] && fd.raw.flags[id].some(n => flagOn.includes(n))) {
            ctx.stroke(fd.cellPath[id]);
          }
        }
        ctx.setLineDash([]);
      }
      const cellHi = state.dot === "cell" ? [sel.cell, hover && hover.cell].concat(sel.brush || []) : [];
      cellHi.filter(id => id && fd.cellPath[id]).forEach(id => {
        ctx.lineWidth = 2 / k; ctx.strokeStyle = "#ffffff"; ctx.stroke(fd.cellPath[id]);
      });

      if (overlay.junctions) {
        const paths = fd.junctionPath[state.method] || {};
        const lit = emphasis(sel, model);
        const sorted = Object.keys(paths).sort((a, b) => (lit ? lit.has(a) - lit.has(b) : 0));
        for (const key of sorted) {
          const v = model.jValue.has(key) ? model.jValue.get(key) : NaN;
          const hot = key === sel.junction || (hover && hover.junction === key);
          ctx.globalAlpha = L.junctionAlpha(sel, lit, key);
          if (hot) {   // selected or hovered: dark ring outside a wider light halo
            ctx.lineWidth = (LINE_PX + 9) / k; ctx.strokeStyle = HOT_RING; ctx.stroke(paths[key]);
            ctx.lineWidth = (LINE_PX + 5) / k; ctx.strokeStyle = "#ffffff"; ctx.stroke(paths[key]);
          } else {
            ctx.lineWidth = (LINE_PX + 2) / k; ctx.strokeStyle = CASING; ctx.stroke(paths[key]);
          }
          ctx.lineWidth = (hot ? LINE_PX + 1 : LINE_PX) / k;
          ctx.strokeStyle = L.colourFor(v, model.lo, model.hi);
          ctx.stroke(paths[key]);
        }
        ctx.globalAlpha = 1;
      }
    }

    const toImage = e => {
      const r = canvas.getBoundingClientRect();
      return [(e.clientX - r.left - xf.tx) / xf.k, (e.clientY - r.top - xf.ty) / xf.k];
    };
    const measuredOutlines = Object.fromEntries(
      Object.entries(fd.raw.cells).filter(([id]) => model.measured.has(id)));

    function pick(e) {
      const [x, y] = toImage(e);
      if (state.dot === "cell") {
        const cell = L.cellAt(measuredOutlines, x, y);
        return cell ? {cell} : null;
      }
      const junction = L.hitTest(model.polylines, x, y, HIT_PX / xf.k);
      return junction ? {junction} : null;
    }

    function showReadout(e, target) {
      if (!target) { readout.hidden = true; return; }
      const v = target.junction ? model.jValue.get(target.junction) : model.cValue.get(target.cell);
      const unit = target.junction ? model.unit : model.cUnit;
      readout.replaceChildren(
        h("b", null, `${L.fmtValue(v)}${unit && Number.isFinite(v) ? " " + unit : ""}`),
        h("span", null, target.junction ? `Junction ${jLabel(target.junction)}` : `Cell ${target.cell}`));
      const r = wrap.getBoundingClientRect();
      readout.hidden = false;
      readout.style.left = `${Math.min(e.clientX - r.left + 12, r.width - readout.offsetWidth - 8)}px`;
      readout.style.top = `${Math.max(8, e.clientY - r.top - readout.offsetHeight - 8)}px`;
    }

    let drag = null;
    canvas.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      drag = {x: e.clientX, y: e.clientY, tx: xf.tx, ty: xf.ty, moved: false};
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", e => {
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        if (drag.moved) {
          xf = fieldUi.xf = {k: xf.k, tx: drag.tx + dx, ty: drag.ty + dy};
          wrap.classList.add("is-panning");
          readout.hidden = true;
          draw();
          return;
        }
      }
      const t = pick(e);
      const same = (t && hover) ? t.junction === hover.junction && t.cell === hover.cell : t === hover;
      if (!same) { hover = t; draw(); }
      wrap.classList.toggle("is-hot", !!t);
      showReadout(e, t);
    });
    canvas.addEventListener("pointerup", e => {
      const d = drag;
      drag = null;
      wrap.classList.remove("is-panning");
      if (d && !d.moved) hooks.select(pick(e));
    });
    canvas.addEventListener("pointerleave", () => { hover = null; readout.hidden = true; draw(); });
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      const r = canvas.getBoundingClientRect(), cx = e.clientX - r.left, cy = e.clientY - r.top;
      const fitK = Math.min(wrap.clientWidth / fd.width, wrap.clientHeight / fd.height);
      const k = Math.min(fitK * 40, Math.max(fitK * 0.5, xf.k * Math.exp(-e.deltaY * 0.0015)));
      xf = fieldUi.xf = {k, tx: cx - ((cx - xf.tx) * k) / xf.k, ty: cy - ((cy - xf.ty) * k) / xf.k};
      readout.hidden = true;
      draw();
    }, {passive: false});
    canvas.addEventListener("dblclick", () => { fit(); draw(); });
    canvas.addEventListener("keydown", e => {
      if (e.key === "Escape") hooks.select(null);
      if (e.key === "0") { fit(); draw(); }
    });

    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    JM._cleanup.push(() => ro.disconnect());
    return {el: wrap, draw, fit: () => { fit(); draw(); }};
  }

  /*** FIELD VIEW: CONTROLS ***/

  function toggle(label, pressed, onclick, tip, swatch) {
    return h("button", {type: "button", class: "chip", "aria-pressed": String(pressed), tip, onclick},
      swatch || null, label);
  }

  function channelControls(fd, viewer) {
    return fd.channels.map(c => {
      const l = layerFor(c.role), t = TINT[c.role] || TINT_OTHER;
      const sw = h("span", {class: "chip-dot", style: `background:rgb(${t.map(v => Math.round(v * 255)).join(",")})`});
      const name = L.channelLabel(c.role, c.marker);
      const btn = toggle(name, l.on, () => {
        l.on = !l.on;
        btn.setAttribute("aria-pressed", String(l.on));
        viewer.draw();
      }, `Show or hide ${name}`, sw);
      const slider = h("input", {type: "range", class: "gain", min: "0.2", max: "4", step: "0.05", value: String(l.gain),
                                 "aria-label": `${name} brightness`, tip: `${name} brightness`});
      slider.addEventListener("input", () => {
        l.gain = Number(slider.value);
        viewer.draw();
        if (JM._redrawCloseup) JM._redrawCloseup();
      });
      return h("div", {class: "layer"}, btn, slider);
    });
  }

  function overlayControls(viewer) {
    return [["junctions", "Junctions", "Junction lines, coloured by the metric"],
            ["outlines", "Outlines", "Cell outlines"]].map(([k, label, tip]) => {
      const btn = toggle(label, overlay[k], () => {
        overlay[k] = !overlay[k];
        btn.setAttribute("aria-pressed", String(overlay[k]));
        viewer.draw();
      }, tip);
      return btn;
    });
  }

  // One toggle per QC flag present anywhere in the run, with this field's count.
  function flagControls(run, f) {
    const names = [];
    run.fields.forEach(x => Object.entries(x.qc).forEach(([k, v]) => {
      if (k !== "flagged_cells" && v > 0 && !names.includes(k)) names.push(k);
    }));
    return names.map(n => toggle(`${pretty(n)} ${L.fmtCount(f.qc[n] || 0)}`, !!state.flags[n],
      () => set({flags: Object.assign({}, state.flags, {[n]: !state.flags[n]})}),
      `Outline cells flagged ${pretty(n)} (count: measured cells in this field)`,
      h("span", {class: "chip-flag"})));
  }

  function colourBar(model) {
    const stops = Array.from({length: 13}, (_, i) => L.colourFor(i / 12, 0, 1)).join(",");
    const nan = [...model.jValue.values()].some(v => !Number.isFinite(v));
    return h("div", {class: "colourbar", tip: "Colour: 2nd–98th percentile of this field's junctions; values beyond are clamped"},
      h("span", {class: "cb-num"}, L.fmtValue(model.lo)),
      h("span", {class: "cb-ramp", style: `background:linear-gradient(90deg,${stops})`}),
      h("span", {class: "cb-num"}, `${L.fmtValue(model.hi)}${model.unit ? " " + model.unit : ""}`),
      nan ? h("span", {class: "cb-nan"}, h("span", {class: "chip-dot", style: `background:${L.NAN_GREY}`}), "no value") : null);
  }

  /*** FIELD VIEW: DETAIL PANEL ***/

  // Profile line colours per channel role: the viewer tints, darkened for a light surface.
  const ROLE_LINE = {
    light: {JM1: "#1e9a3c", JM2: "#b23ab2", segment: "#898781"},
    dark: {JM1: "#4cc764", JM2: "#d872d8", segment: "#898781"},
  };
  const PROFILE_ROLES = ["JM1", "JM2", "segment"];
  const TILE_ROLES = ["segment", "nucleus"];   // after JM1, JM2 and the merge
  let stripScope = "field";                     // position strips compare against "field" | "run"

  const markerOf = (fd, role) => (fd.channels.find(c => c.role === role) || {}).marker;
  const tintCss = t => `rgb(${t.map(v => Math.round(v * 255)).join(",")})`;
  const withUnit = (v, unit) => `${L.fmtValue(v)}${unit && Number.isFinite(v) ? " " + unit : ""}`;

  function ordinal(n) {
    const s = n % 100 >= 11 && n % 100 <= 13 ? "th" : ({1: "st", 2: "nd", 3: "rd"}[n % 10] || "th");
    return `${n}${s}`;
  }

  // About `count` round tick values covering [lo, hi].
  function niceTicks(lo, hi, count) {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= raw);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
    return out;
  }

  /* Close-up: one crop per channel at the junction's window, drawn from the decoded previews. */

  function tileSpecs(fd) {
    const ci = role => fd.channels.findIndex(c => c.role === role);
    const grey = [1, 1, 1], specs = [];
    const single = role => ({label: L.channelLabel(role, markerOf(fd, role)), dots: [TINT[role] || TINT_OTHER],
                             parts: [{ci: ci(role), role, tint: grey}]});
    ["JM1", "JM2"].forEach(role => { if (ci(role) >= 0) specs.push(single(role)); });
    const merged = L.mergeRoles(fd.channels.map(c => c.role));
    if (merged.length) {
      const tints = L.mergeTints(merged), names = ["green", "magenta", "cyan"];
      specs.push({label: "Merge", tip: `Merge of ${merged.map((r, i) => `${L.channelLabel(r, markerOf(fd, r))} in ${names[i]}`).join(", ")}; where they overlap reads white`,
                  dots: tints,
                  parts: merged.map((role, i) => ({ci: ci(role), role, tint: tints[i]}))});
    }
    TILE_ROLES.forEach(role => { if (ci(role) >= 0) specs.push(single(role)); });
    fd.channels.forEach((c, i) => {
      if (!["JM1", "JM2", ...TILE_ROLES].includes(c.role)) {
        specs.push({label: L.channelLabel(c.role, c.marker), dots: [TINT_OTHER], parts: [{ci: i, role: c.role, tint: grey}]});
      }
    });
    return specs;
  }

  // focus: the selected junction's polylines (band outlined, centre left clear) or a cell outline.
  function drawTile(canvas, fd, win, spec, focus, others) {
    const css = canvas.clientWidth || 140, dpr = window.devicePixelRatio || 1;
    const size = Math.max(1, Math.round(css * dpr));
    canvas.width = size; canvas.height = size;
    const src = document.createElement("canvas");
    src.width = win.w; src.height = win.h;
    const sg = src.getContext("2d"), img = sg.createImageData(win.w, win.h), d = img.data;
    const parts = spec.parts.map(p => ({plane: fd.planes[p.ci], tint: p.tint, gain: layerFor(p.role).gain}));
    for (let y = 0; y < win.h; y++) {
      for (let x = 0; x < win.w; x++) {
        const i = (win.y + y) * fd.width + win.x + x, o = (y * win.w + x) * 4;
        let r = 0, g = 0, b = 0;
        for (const p of parts) { const v = p.plane[i] * p.gain; r += v * p.tint[0]; g += v * p.tint[1]; b += v * p.tint[2]; }
        d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
      }
    }
    sg.putImageData(img, 0, 0);
    const g = canvas.getContext("2d");
    g.imageSmoothingEnabled = false;
    g.drawImage(src, 0, 0, size, size);
    // Overlay on its own layer so the band's centre can be cleared of every line, cell outlines too.
    const k = size / win.w;
    const ov = document.createElement("canvas");
    ov.width = size; ov.height = size;
    const o = ov.getContext("2d");
    o.setTransform(k, 0, 0, k, -win.x * k, -win.y * k);
    o.lineJoin = "round"; o.lineCap = "round";
    o.lineWidth = dpr / k; o.strokeStyle = "rgba(255,255,255,0.4)";
    others.forEach(p => o.stroke(p));
    o.strokeStyle = cssVar("--accent");
    if (focus.cell) {
      o.lineWidth = (2 * dpr) / k; o.stroke(focus.path);
    } else {
      const gap = Math.max(8 * dpr, 5 * k);   // clear centre: screen px, or ~5 preview px when zoomed in
      o.lineWidth = (gap + 3 * dpr) / k; o.stroke(focus.path);
      o.globalCompositeOperation = "destination-out";
      o.lineWidth = gap / k; o.stroke(focus.path);
      o.globalCompositeOperation = "source-over";
      if (focus.start) {   // where the profile's x = 0 sits
        o.beginPath(); o.arc(focus.start[0], focus.start[1], (3.5 * dpr) / k, 0, 2 * Math.PI);
        o.fillStyle = cssVar("--accent"); o.fill();
        o.lineWidth = (1.5 * dpr) / k; o.strokeStyle = "rgba(0,0,0,0.85)"; o.stroke();
      }
    }
    g.drawImage(ov, 0, 0);
  }

  function closeupFigure(fd, sel, model) {
    const lines = sel.junction ? (model.polylines[sel.junction] || []) : [fd.raw.cells[sel.cell] || []];
    if (!lines.length || !lines[0].length) return null;
    const win = L.cropWindow(lines, fd.width, fd.height);
    const cells = sel.junction ? sel.junction.split("-") : [sel.cell];
    const ps = fd.raw.profile_start && fd.raw.profile_start[state.method];
    const focus = sel.junction ? {path: fd.junctionPath[state.method][sel.junction], start: ps ? ps[sel.junction] : null}
                               : {cell: true, path: fd.cellPath[sel.cell]};
    const others = sel.junction ? cells.map(id => fd.cellPath[id]).filter(Boolean) : [];
    const tiles = tileSpecs(fd).map(spec => {
      const canvas = h("canvas", {class: "tile-img", "aria-hidden": "true"});
      return {spec, canvas, el: h("figure", {class: "tile-fig"}, canvas,
        h("figcaption", {tip: spec.tip}, spec.dots.map(t => h("span", {class: "chip-dot", style: `background:${tintCss(t)}`})),
          h("span", {class: "tile-cap"}, spec.label)))};
    });
    const el = h("section", {class: "detail-sec"},
      h("h4", {class: "detail-sec-head"}, h("span", {tip: "The selected junction's band is outlined; both cells' outlines faint. The dot marks where the intensity profile starts. Brightness follows the channel sliders."}, "Close-up")),
      h("div", {class: "tiles-grid", role: "img",
                "aria-label": `Close-up of ${sel.junction ? "junction " + jLabel(sel.junction) : "cell " + sel.cell} in each channel`},
        tiles.map(t => t.el)));
    return {el, draw() { tiles.forEach(t => drawTile(t.canvas, fd, win, t.spec, focus, others)); }};
  }

  /* Intensity along the junction: inline SVG, raw intensity units. */

  // One panel per channel, stacked on a shared x axis, each with its own y axis: the channels'
  // raw intensities differ by an order of magnitude, so a shared y would flatten one of them.
  function profileFigure(fd, key) {
    const prof = fd.raw.profiles && fd.raw.profiles[state.method] && fd.raw.profiles[state.method][key];
    if (!prof) return null;
    const roles = PROFILE_ROLES.filter(r => prof[r] && prof[r].x && prof[r].x.length);
    if (!roles.length) return null;
    const thresholds = fd.raw.thresholds || {};
    const unit = fd.raw.profile_unit || "px";
    const colours = ROLE_LINE[theme() === "dark" ? "dark" : "light"];
    const label = r => L.channelLabel(r, markerOf(fd, r));
    let xmax = 0;
    const panels = roles.map(r => {
      const sc = L.profileScale(prof[r], thresholds, r);
      xmax = Math.max(xmax, sc.xmax);
      // Round ticks from 0; the top tick sits on or above the peak, so the line stays in its panel.
      const ticks = niceTicks(0, sc.ymax > 0 ? sc.ymax : 1, r === "segment" ? 2 : 3);
      const step = ticks.length > 1 ? ticks[1] - ticks[0] : (sc.ymax || 1);
      while (ticks[ticks.length - 1] < sc.ymax) ticks.push(Number((ticks[ticks.length - 1] + step).toPrecision(12)));
      return {role: r, ticks, ymax: ticks[ticks.length - 1] || 1, height: r === "segment" ? 52 : 88};
    });
    const plot = h("div", {class: "profile-plot"});
    const readout = h("div", {class: "viewer-readout profile-readout", hidden: true});
    const wrap = h("div", {class: "profile-wrap"}, plot, readout);
    const hasT = roles.some(r => Number.isFinite(thresholds[r]));
    const el = h("section", {class: "detail-sec"},
      h("div", {class: "detail-sec-head"},
        h("h4", null, h("span", {tip: "Mean marker intensity across the band, corner to corner along the junction, in raw intensity units, starting at the dot in the close-up. Dotted verticals mark corners between the junction's pieces. Each channel has its own y axis."},
          "Intensity along the junction")),
        hasT ? h("span", {class: "legend-item detail-sec-sub", tip: "The applied marker threshold for this field"},
          h("span", {class: "key-line dashed"}), "threshold") : null),
      wrap);
    const NS = "http://www.w3.org/2000/svg";
    const s = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
    const text = (attrs, str) => { const t = s("text", attrs); t.textContent = str; return t; };

    function draw() {
      const W = Math.max(160, plot.clientWidth || 300), m = {l: 40, r: 8, t: 18, b: 22}, gap = 34;
      let y0 = 0;
      panels.forEach((p, i) => { p.top = y0 + m.t; y0 = p.top + p.height + (i < panels.length - 1 ? gap - m.t : 0); });
      const H = y0 + m.b;
      const X = x => m.l + (xmax > 0 ? (x / xmax) * (W - m.l - m.r) : 0);
      const Y = (p, y) => p.top + (1 - y / p.ymax) * p.height;
      const svg = s("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "profile-svg", role: "img",
                            "aria-label": `Intensity along junction ${jLabel(key)}: ${roles.map(label).join(", ")}`});
      const muted = cssVar("--muted"), ink2 = cssVar("--ink-2"), grid = cssVar("--grid"), axis = cssVar("--axis");
      // A join between two pieces of the junction is a null at the x where the last piece ended:
      // a dotted vertical marks that corner in every panel, under the lines.
      const p0 = prof[roles[0]];
      p0.x.forEach((x, i) => {
        if (i > 0 && i < p0.x.length - 1 && p0.y[i] === null && x === p0.x[i - 1] && x === p0.x[i + 1]) {
          panels.forEach(p => svg.append(s("line", {x1: X(x), x2: X(x), y1: p.top, y2: p.top + p.height, stroke: axis,
                                                   "stroke-width": 1, "stroke-dasharray": "1 3", class: "profile-corner"})));
        }
      });
      // The "threshold" label sits by the threshold line where no segment of the drawn line
      // crosses it: nearest the line first (above, then below), rightmost first. With no clear
      // place it goes to the right end of the panel's title row, which the line never reaches.
      const LABEL_W = 54, LABEL_H = 12;
      function thresholdLabelAt(p, t) {
        const yt = Y(p, t), pr = prof[p.role], runs = [];
        let run = [];
        pr.x.forEach((x, i) => {
          if (!Number.isFinite(pr.y[i]) || !Number.isFinite(x)) { if (run.length) runs.push(run); run = []; return; }
          run.push([X(x), Y(p, pr.y[i])]);
        });
        if (run.length) runs.push(run);
        const xs = [];   // right to left in small steps, so a clear gap anywhere along the line is found
        for (let x = W - m.r - LABEL_W; x >= m.l + 2; x -= 6) xs.push(x);
        const ys = [2, 10, 18].flatMap(d => [yt - LABEL_H - d, yt + d])
          .filter(y => y >= p.top - 2 && y + LABEL_H <= p.top + p.height + 2);
        const boxes = [];
        ys.forEach(y => xs.forEach(x => boxes.push({x: x - 2, y: y - 2, w: LABEL_W + 4, h: LABEL_H + 4, lx: x, ly: y})));
        const best = L.labelBoxAt(runs, boxes);
        if (!best || best.crossings > 0) return {x: W - m.r, y: p.top - 5, "text-anchor": "end", fill: muted, class: "tick halo"};
        return {x: best.box.lx, y: best.box.ly + LABEL_H - 2, fill: muted, class: "tick halo"};
      }
      panels.forEach(p => {
        // Panel name: a line key in the series colour, the text in a text colour.
        svg.append(s("line", {x1: m.l, x2: m.l + 14, y1: p.top - 9, y2: p.top - 9, stroke: colours[p.role],
                              "stroke-width": p.role === "segment" ? 1 : 2}));
        svg.append(text({x: m.l + 20, y: p.top - 5, fill: ink2, class: "panel-label"}, label(p.role)));
        p.ticks.forEach(t => {
          svg.append(s("line", {x1: m.l, x2: W - m.r, y1: Y(p, t), y2: Y(p, t), stroke: t ? grid : axis, "stroke-width": 1}));
          svg.append(text({x: m.l - 6, y: Y(p, t) + 4, "text-anchor": "end", fill: muted, class: "tick"}, L.fmtValue(t)));
        });
        const t = thresholds[p.role];
        if (Number.isFinite(t)) {
          svg.append(s("line", {x1: m.l, x2: W - m.r, y1: Y(p, t), y2: Y(p, t), stroke: colours[p.role], "stroke-width": 1,
                                "stroke-dasharray": "4 3"}));
          svg.append(text(thresholdLabelAt(p, t), "threshold"));
        }
        const pr = prof[p.role];
        let dpath = "", pen = false;
        pr.x.forEach((x, i) => {
          const y = pr.y[i];
          if (!Number.isFinite(y) || !Number.isFinite(x)) { pen = false; return; }   // null breaks the line
          dpath += `${pen ? "L" : "M"}${X(x).toFixed(1)},${Y(p, y).toFixed(1)}`;
          pen = true;
        });
        svg.append(s("path", {d: dpath, fill: "none", stroke: colours[p.role], "stroke-width": p.role === "segment" ? 1.25 : 2,
                              "stroke-linejoin": "round", "stroke-linecap": "round"}));
      });
      const last = panels[panels.length - 1], base = last.top + last.height;
      const xt = niceTicks(0, xmax, 4);
      xt.forEach((t, i) => svg.append(text({x: X(t), y: base + 15, "text-anchor": i === 0 ? "start" : "middle", fill: muted, class: "tick"},
        i === xt.length - 1 ? `${L.fmtValue(t)} ${unit}` : L.fmtValue(t))));
      const cross = s("line", {y1: panels[0].top, y2: base, stroke: muted, "stroke-width": 1, visibility: "hidden"});
      const dots = panels.map(p => s("circle", {r: 4, fill: colours[p.role], stroke: cssVar("--surface"), "stroke-width": 2, visibility: "hidden"}));
      svg.append(cross, ...dots);
      const hit = s("rect", {x: m.l, y: 0, width: Math.max(1, W - m.l - m.r), height: H, fill: "transparent"});
      svg.append(hit);
      const nearest = (xs, x) => {
        let best = 0, bd = Infinity;
        xs.forEach((v, i) => { const dd = Math.abs(v - x); if (dd < bd) { bd = dd; best = i; } });
        return best;
      };
      hit.addEventListener("pointermove", e => {
        const r = svg.getBoundingClientRect();
        const x = Math.max(0, Math.min(xmax, ((e.clientX - r.left - m.l) / (W - m.l - m.r)) * xmax));
        const at = panels.map(p => { const i = nearest(prof[p.role].x, x); return {p, x: prof[p.role].x[i], y: prof[p.role].y[i]}; });
        const sx = X(at[0].x);
        cross.setAttribute("x1", sx); cross.setAttribute("x2", sx); cross.setAttribute("visibility", "visible");
        at.forEach((a, i) => {
          const ok = Number.isFinite(a.y);
          dots[i].setAttribute("visibility", ok ? "visible" : "hidden");
          if (ok) { dots[i].setAttribute("cx", X(a.x)); dots[i].setAttribute("cy", Y(a.p, a.y)); }
        });
        readout.replaceChildren(h("span", null, `${L.fmtValue(at[0].x)} ${unit} along`),
          ...at.map(a => h("span", {class: "readout-row"},
            h("span", {class: "key-line", style: `background:${colours[a.p.role]}`}),
            h("b", null, L.fmtValue(Number.isFinite(a.y) ? a.y : NaN)), label(a.p.role))));
        readout.hidden = false;
        // Beside the crosshair where it fits, else under the chart, so it never hides the line read.
        const rw = readout.offsetWidth;
        const side = sx + 12 + rw <= W ? sx + 12 : sx - 12 - rw >= 0 ? sx - 12 - rw : null;
        readout.style.left = `${side !== null ? side : Math.max(0, Math.min(W - rw, sx - rw / 2))}px`;
        readout.style.top = `${side !== null ? 4 : H + 4}px`;
      });
      hit.addEventListener("pointerleave", () => {
        readout.hidden = true;
        cross.setAttribute("visibility", "hidden");
        dots.forEach(d => d.setAttribute("visibility", "hidden"));
      });
      plot.replaceChildren(svg);
    }
    return {el, draw};
  }

  /* Where it sits: one strip per metric, every junction (or cell) of the field or run as faint dots. */

  function positionFigure(run, fd, model, sel, pos, kind, token, box) {
    const isJ = kind === "junction";
    const cells = isJ ? sel.junction.split("-") : [sel.cell];
    const sub = () => (stripScope === "run" ? `percentile in the run` : `percentile in this field`);
    const subEl = h("span", {class: "detail-sec-sub"}, sub());
    const rows = run.metrics.map(m => {
      const strip = h("canvas", {class: "strip", "aria-hidden": "true"});
      const pct = h("span", {class: "pct"}, "…");
      const row = h("div", {class: m.key === state.metric ? "pos-row is-current" : "pos-row", tabindex: "0", tip: m.label},
        h("button", {type: "button", class: "metric-link", tip: `${m.label}. ${m.help}`, onclick: () => set({metric: m.key})}, m.label),
        strip, pct);
      return {m, row, strip, pct, data: null};
    });
    function drawStrip(r) {
      const c = r.strip, dpr = window.devicePixelRatio || 1;
      const W = c.clientWidth || 120, H = c.clientHeight || 20;
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
      const g = c.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      g.fillStyle = cssVar("--grid");
      g.fillRect(0, H / 2 - 0.5, W, 1);
      if (!r.data) return;
      const vals = stripScope === "run" ? r.data.run : r.data.field;
      const fin = L.finite(vals).values;
      if (!fin.length) return;
      let lo = Infinity, hi = -Infinity;
      fin.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      const pad = 5, X = v => pad + (hi > lo ? ((v - lo) / (hi - lo)) * (W - 2 * pad) : (W - 2 * pad) / 2);
      g.fillStyle = cssVar("--muted");
      g.globalAlpha = fin.length > 400 ? 0.12 : 0.35;
      fin.forEach((v, i) => {
        g.beginPath();
        g.arc(X(v), H / 2 + (((i * 0.6180339887) % 1) - 0.5) * (H - 8), 2, 0, 2 * Math.PI);
        g.fill();
      });
      g.globalAlpha = 1;
      if (Number.isFinite(r.data.value)) {
        const x = X(r.data.value);
        g.fillStyle = cssVar("--accent");
        g.fillRect(x - 1, 1, 2, H - 2);
        g.beginPath(); g.arc(x, H / 2, 4, 0, 2 * Math.PI);
        g.lineWidth = 2; g.strokeStyle = cssVar("--surface"); g.stroke(); g.fill();
      }
    }
    function fill(r) {
      const vals = stripScope === "run" ? r.data.run : r.data.field;
      const p = L.percentileOf(vals, r.data.value);
      r.pct.textContent = p === null ? "–" : ordinal(p);
      drawStrip(r);
    }
    const toggle = segmented([{value: "field", label: "Field", tip: `Compare with every ${isJ ? "junction" : "cell"} in this field`},
                              {value: "run", label: "Run", tip: `Compare with every ${isJ ? "junction" : "cell"} in the run`}],
      stripScope, v => {
        stripScope = v;
        toggle.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.textContent.toLowerCase() === v)));
        subEl.textContent = sub();
        rows.forEach(r => { if (r.data) fill(r); });
      }, "Compare with");
    const el = h("section", {class: "detail-sec"},
      h("div", {class: "detail-sec-head"},
        h("h4", null, h("span", {tip: `Faint dots: every ${isJ ? "junction" : "cell"} of the field or run. Accent: this one. Hover a row for its values.`}, "Where it sits")),
        subEl, toggle),
      h("div", {class: "pos-rows"}, rows.map(r => r.row)));
    rows.forEach(r => {
      loadMetric(run.run, state.method, state.role, r.m.key).then(v => {
        if (token !== box._token) return;
        const picked = L.metricValues(v, kind, r.m, run.calibrated, state.useUm);
        const arr = picked.values || [];
        const idx = isJ ? model.slice.junctions : model.slice.cells;
        const value = pos ? arr[pos.i] : NaN;
        r.data = {run: arr, field: idx.map(x => arr[x.i]), value};
        const lines = isJ
          ? cells.map(id => `Cell ${id}: ${withUnit(L.sideValue(fd.raw.sides, state.method, sel.junction, state.role, id, r.m.key, picked.um), picked.unit)}`)
            .concat(`Mean: ${withUnit(value, picked.unit)}`)
          : [`Median over its junctions: ${withUnit(value, picked.unit)}`];
        r.row.dataset.tip = `${r.m.label}\n${lines.join("\n")}`;
        fill(r);
      }, () => {
        if (token !== box._token) return;
        r.pct.textContent = "–";
      });
    });
    return {el, draw() { rows.forEach(r => drawStrip(r)); }};
  }

  function detailPanel(run, f, fd, model, sel, box) {
    const token = (box._token = (box._token || 0) + 1);
    if (box._ro) { box._ro.disconnect(); box._ro = null; }
    JM._redrawCloseup = null;
    if (!sel.junction && !sel.cell) {
      box.replaceChildren(h("p", {class: "detail-empty"},
        state.dot === "cell" ? "Click a cell in the image or chart." : "Click a junction in the image or chart."));
      return;
    }
    const isJ = !!sel.junction;
    const cells = isJ ? sel.junction.split("-") : [sel.cell];
    const pos = isJ ? model.slice.junctions.find(j => j.key === sel.junction)
                    : model.slice.cells.find(c => String(c.cell_id) === sel.cell);
    const flagTags = id => (fd.raw.flags[id] || []).map(n => h("span", {class: "flag-tag", tip: "QC flag"}, pretty(n)));
    const nJ = isJ ? 0 : (model.byCell.get(sel.cell) || []).length;
    const head = h("div", {class: "detail-head"},
      h("h3", null, isJ ? `Junction ${jLabel(sel.junction)}` : `Cell ${sel.cell}`),
      h("span", {class: "detail-sub"}, `${L.channelLabel(state.role, markerOf(fd, state.role))}, ${state.method}`
        + (isJ ? "" : `, ${nJ} junction${nJ === 1 ? "" : "s"}`)));
    const flagsBox = isJ
      ? h("div", {class: "detail-cells"}, cells.map(id => h("div", {class: "detail-cell"},
          h("span", {class: "detail-cell-id"}, `Cell ${id}`), flagTags(id))))
      : h("div", {class: "detail-cell detail-cells"}, flagTags(sel.cell));
    const figs = [closeupFigure(fd, sel, model), isJ ? profileFigure(fd, sel.junction) : null,
                  positionFigure(run, fd, model, sel, pos, isJ ? "junction" : "cell", token, box)].filter(Boolean);
    box.replaceChildren(head, flagsBox, ...figs.map(x => x.el));
    const drawAll = () => figs.forEach(x => x.draw());
    drawAll();
    JM._redrawCloseup = figs[0] && figs[0].el.querySelector(".tiles-grid") ? () => figs[0].draw() : null;
    let lastW = box.clientWidth;
    box._ro = new ResizeObserver(() => { if (box.clientWidth !== lastW) { lastW = box.clientWidth; drawAll(); } });
    box._ro.observe(box);
    JM._cleanup.push(() => { if (box._ro) box._ro.disconnect(); });
  }

  /*** FIELD VIEW: LINKED CHART ***/

  function fieldChart(run, model, sel, plot, hooks) {
    const cell = state.dot === "cell";
    const rows = cell
      ? model.slice.cells.map(c => ({id: String(c.cell_id), v: model.cValue.get(String(c.cell_id))}))
      : model.slice.junctions.map(j => ({id: j.key, v: model.jValue.get(j.key)}));
    const pts = rows.filter(r => Number.isFinite(r.v));
    const m = metricInfo(state.metric);
    if (!pts.length) { Plotly.purge(plot); return null; }
    const at = new Map(pts.map((p, i) => [p.id, i]));
    const ink2 = cssVar("--ink-2"), unit = cell ? model.cUnit : model.unit, ys = pts.map(p => p.v);
    // Violin for the shape; the dots are a jittered scatter so each can carry its ramp colour.
    const violin = {
      type: "violin", x0: 0, y: ys, points: false, spanmode: "hard", width: 0.9, hoverinfo: "skip",
      line: {color: cssVar("--axis"), width: 1}, fillcolor: "rgba(0,0,0,0)",
      box: {visible: true, width: 0.06, fillcolor: cssVar("--surface"), line: {color: ink2, width: 1}},
      meanline: {visible: false}, showlegend: false,
    };
    const dots = {
      type: "scatter", mode: "markers", y: ys, customdata: pts.map(p => p.id),
      x: pts.map((p, i) => (((i * 0.6180339887) % 1) - 0.5) * 0.5),
      text: pts.map(p => (cell ? `Cell ${p.id}` : `Junction ${jLabel(p.id)}`)),
      marker: {size: 8, color: pts.map(p => L.colourFor(p.v, model.lo, model.hi)), line: {width: 0.75, color: ink2}},
      selected: {marker: {opacity: 1, size: 11}}, unselected: {marker: {opacity: 0.15}},
      hovertemplate: `<b>%{y:.4~g}${unit ? " " + unit : ""}</b><br>%{text}<extra></extra>`,
      showlegend: false,
    };
    const layout = plotLayout(`${m.label}${unit ? ` (${unit})` : ""}`);
    layout.margin = {l: 56, r: 8, t: 8, b: 16};
    layout.xaxis = {type: "linear", range: [-0.55, 0.55], visible: false, fixedrange: true};
    layout.dragmode = "lasso";
    layout.hovermode = "closest";
    Plotly.react(plot, [violin, dots], layout, {displaylogo: false, responsive: true, displayModeBar: true,
      modeBarButtonsToRemove: ["autoScale2d", "toImage", "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d"]});
    plot.removeAllListeners("plotly_selected");
    plot.removeAllListeners("plotly_deselect");
    plot.removeAllListeners("plotly_click");
    plot.on("plotly_selected", ev => {
      if (!ev) return;
      hooks.brush(ev.points.filter(p => p.curveNumber === 1).map(p => p.customdata));
    });
    plot.on("plotly_deselect", () => hooks.brush(null));
    plot.on("plotly_click", ev => {
      const p = ev.points && ev.points[0];
      if (p && p.customdata !== undefined) hooks.select(cell ? {cell: p.customdata} : {junction: p.customdata});
    });
    return {
      highlight() {
        const ids = sel.brush || (cell ? (sel.cell ? [sel.cell] : []) : (sel.junction ? [sel.junction] : []));
        const idx = ids.map(id => at.get(id)).filter(i => i !== undefined);
        if (!sel.brush && plot.layout && plot.layout.selections && plot.layout.selections.length) {
          Plotly.relayout(plot, {selections: []});
        }
        Plotly.restyle(plot, {selectedpoints: [ids.length ? idx : null]}, [1]);
      },
    };
  }

  /*** FIELD VIEW: ASSEMBLY ***/

  JM.views.field = function (root, r) {
    const run = runById(r.run);
    const f = run && run.fields.find(x => x.field_id === r.field);
    if (!f) {
      root.append(h("p", null, "Field not found. ", h("a", {href: "#/"}, "Back to overview")));
      return;
    }
    if (f.assets === false) {
      root.append(h("div", {class: "field-head"}, h("a", {class: "back", href: "#/"}, "← Overview"),
                    h("h1", null, f.field_id)),
                  h("p", {class: "empty-note"}, "Could not load this field: the run wrote no image for it. ",
                    h("a", {href: "#/provenance"}, "Provenance")));
      return;
    }
    const key = `${run.run}/${f.field_id}`;
    // The route's ?j= / ?c= is the selection (sync() keeps it current); a brush lives only here.
    const sameField = fieldUi.key === key;
    fieldUi = {key, xf: sameField ? fieldUi.xf : null,
               sel: L.selectionFromRoute(r, sameField ? fieldUi.sel : null)};
    if (!sameField) {
      // A link to a cell opens in cell mode.
      if (r.cell && !r.junction && state.dot !== "cell") { setTimeout(() => set({dot: "cell"})); return; }
    }
    const sel = fieldUi.sel;
    // A selection only makes sense in its own dot mode.
    if (state.dot === "cell") sel.junction = null; else sel.cell = null;
    if (sel.dot !== state.dot) { sel.brush = null; sel.dot = state.dot; }

    root.append(h("div", {class: "field-head"},
      h("a", {class: "back", href: "#/"}, "← Overview"),
      h("h1", null, f.field_id), qcBadge(f.qc, f.n_cells),
      h("span", {class: "field-sub"}, `${L.fmtCount(f.n_cells)} cells, ${L.fmtCount(f.n_junctions)} junctions`
        + (JM.runs.length > 1 ? ` in ${run.label}` : ""))));
    const body = h("div", {class: "field-layout"}, h("div", {class: "card viewer-card is-loading"}, h("div", {class: "viewer"})));
    root.append(body);

    const m = metricInfo(state.metric);
    Promise.all([
      loadField(run.run, f.field_id),
      loadIndex(run.run, state.method).catch(() => null),
      loadMetric(run.run, state.method, state.role, state.metric).catch(() => null),
    ]).then(([fd, index, metric]) => {
      if (!body.isConnected) return;
      const model = index ? fieldModel(run, fd, index, metric, m)
        : {slice: {junctions: [], cells: []}, jValue: new Map(), cValue: new Map(), lo: 0, hi: 1,
           byCell: new Map(), unit: "", cUnit: "", measured: new Set(), missing: true, polylines: {}};
      const empty = !model.slice.junctions.length;
      const detail = h("div", {class: "detail"});
      const plot = h("div", {class: "plot field-plot"});
      let chart = null;

      function sync() {
        viewer.draw();
        if (chart) chart.highlight();
        detailPanel(run, f, fd, model, sel, detail);
        const q = sel.junction ? `j=${sel.junction}` : sel.cell ? `c=${sel.cell}` : "";
        history.replaceState(null, "", JM.fieldHref(run.run, f.field_id, q));
      }
      const hooks = {
        select(t) {
          sel.junction = t && t.junction ? t.junction : null;
          sel.cell = t && t.cell ? String(t.cell) : null;
          sel.brush = null;
          sync();
        },
        brush(ids) {
          sel.brush = ids && ids.length ? ids.map(String) : null;
          if (sel.brush) { sel.junction = null; sel.cell = null; }
          sync();
        },
      };
      const viewer = createViewer(fd, model, sel, hooks);
      const fitBtn = h("button", {type: "button", class: "chip", tip: "Fit the image (double-click or 0)", onclick: viewer.fit}, "Fit");
      const flags = flagControls(run, f);
      const card = h("div", {class: "card viewer-card"},
        h("div", {class: "viewer-bar"},
          h("div", {class: "bar-group", role: "group", "aria-label": "Channels"}, channelControls(fd, viewer)),
          h("div", {class: "bar-group", role: "group", "aria-label": "Overlays"}, overlayControls(viewer), fitBtn)),
        flags.length ? h("div", {class: "viewer-bar flags-bar", role: "group", "aria-label": "QC flags"},
          h("span", {class: "bar-label", tip: "QC flags; flagged cells get a dashed orange outline"}, "QC"), flags) : null,
        viewer.el,
        h("div", {class: "viewer-foot"},
          empty ? h("span", {class: "empty-note"}, NO_JUNCTIONS)
            : model.missing ? h("span", {class: "empty-note"}, `No ${roleLabel(state.role, [run])} ${state.method} values in this run`)
              : colourBar(model)));
      body.replaceChildren(card);
      if (!empty) {
        body.append(h("div", {class: "field-side"},
          h("section", {class: "card detail-card"}, detail),
          h("section", {class: "card chart-card"},
            h("div", {class: "chart-head"},
              h("h2", {class: "chart-title", tip: m.help}, m.label),
              h("span", {class: "chart-sub", tip: "Lasso or box-select to highlight in the image; double-click to clear"},
                `${roleLabel(state.role, [run])}, ${state.method}, per ${state.dot}`)),
            plot)));
        chart = fieldChart(run, model, sel, plot, hooks);
        JM._fieldPlot = plot;
      }
      sync();
    }, err => {
      body.replaceChildren(h("p", {class: "empty-note"}, `Could not load this field: ${err.message}`));
    });
  };

  /*** PROVENANCE ***/

  let provQuery = "";

  // Warnings first, then the run's provenance rows, as one searchable key-value table.
  function provRows(run) {
    return (run.warnings || []).map(w => ({group: "Warnings", key: w.text, value: w.detail}))
      .concat(run.provenance_rows || []);
  }

  function provTable(rows) {
    const body = h("tbody");
    let group = null;
    rows.forEach(r => {
      if (r.group !== group) {
        group = r.group;
        body.append(h("tr", {class: "prov-group"}, h("th", {scope: "colgroup", colspan: "2"}, group)));
      }
      body.append(h("tr", {class: r.group === "Warnings" ? "is-warning" : null},
        h("th", {scope: "row"}, r.key), h("td", null, r.value)));
    });
    return h("table", {class: "data-table prov-table"}, body);
  }

  function provSection(run, query) {
    const all = provRows(run), rows = L.filterRows(all, query);
    return h("section", {class: "card prov-card"},
      h("div", {class: "section-head"},
        JM.runs.length > 1 ? h("span", {class: "swatch", style: `background:${L.runColour(runIndex(run.run), theme())}`}) : null,
        h("h2", null, run.label),
        h("span", {class: "count"}, query ? `${L.fmtCount(rows.length)} of ${L.fmtCount(all.length)}` : L.fmtCount(all.length))),
      rows.length ? provTable(rows) : h("p", {class: "detail-empty"}, "No match"));
  }

  JM.views.provenance = function (root) {
    const search = h("input", {type: "search", class: "search", value: provQuery, placeholder: "Search",
                               "aria-label": "Search provenance", tip: "Every word must match a group, key or value"});
    const body = h("div", {class: "prov-body"});
    const draw = () => body.replaceChildren(...JM.runs.map(run => provSection(run, provQuery)));
    search.addEventListener("input", () => { provQuery = search.value; draw(); });
    root.append(h("div", {class: "prov-head"}, h("h1", null, "Provenance"), search), body);
    draw();
  };

  /*** HELP ***/

  // One collapsed panel: each metric's label and definition, then the few terms the views use.
  function initHelp() {
    const panel = document.getElementById("help"), btn = document.getElementById("help-toggle");
    const run = JM.runs[0];
    const item = (term, def, tag) => [h("dt", null, term, tag ? h("span", {class: "help-tag"}, tag) : null), h("dd", null, def)];
    const unit = m => [m.unit_px, run.calibrated ? m.unit_um : ""].filter(Boolean).join(" / ");
    panel.append(
      h("div", {class: "help-inner"},
        h("div", {class: "help-col"},
          h("h2", null, "Metrics"),
          h("dl", {class: "help-list"}, run.metrics.map(m => item(m.label, m.help + (unit(m) ? ` Unit: ${unit(m)}.` : ""),
                                                                   m.td ? "threshold" : null)))),
        h("div", {class: "help-col"},
          h("h2", null, "Terms"),
          h("dl", {class: "help-list"},
            Object.keys(METHOD_NAMES).map(k => item(`${k} (${METHOD_NAMES[k]})`, METHOD_HELP[k])),
            item("Junction value", "Mean of its two cell sides."),
            item("Cell value", "Median over its junctions."),
            item("threshold", "Metric depends on the marker threshold."),
            item("Flagged cell", "Measured cell with a QC flag. Corner-method disagreement not counted."),
            item("Junction colour", "2nd–98th percentile of the field, dark purple to yellow. Grey: no value."),
            item("Warnings", "Run-level problems. Hover for detail; full record under Provenance.")))));
    btn.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      btn.setAttribute("aria-expanded", String(!panel.hidden));
    });
  }

  /*** RENDER ***/

  function render() {
    const r = route();
    normalizeState(r);
    renderCrumbs(r);
    renderFilters(r);
    const root = document.getElementById("view");
    if (JM._lastPlot && r.view !== "overview") { Plotly.purge(JM._lastPlot); JM._lastPlot = null; }
    if (JM._fieldPlot) { Plotly.purge(JM._fieldPlot); JM._fieldPlot = null; }
    if (r.view !== "field") fieldUi = {key: null, xf: null, sel: null};
    JM._cleanup.splice(0).forEach(fn => fn());
    root.replaceChildren();
    (JM.views[r.view] || JM.views.overview)(root, r);
  }
  JM.render = render;

  /*** THEME ***/

  function initTheme() {
    try {
      const saved = localStorage.getItem("jm-theme");
      if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
    } catch (e) { /* storage blocked: follow the system */ }
    document.getElementById("theme-toggle").addEventListener("click", () => {
      const next = theme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("jm-theme", next); } catch (e) { /* not persisted */ }
      render();
    });
    if (window.matchMedia) {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      if (mq.addEventListener) mq.addEventListener("change", () => { if (!document.documentElement.hasAttribute("data-theme")) render(); });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (!JM.runs.length) return;
    initState();
    initTheme();
    initTooltip();
    initHelp();
    window.addEventListener("hashchange", render);
    render();
  });
})();
