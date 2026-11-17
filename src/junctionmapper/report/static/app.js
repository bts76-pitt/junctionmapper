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
  const metricInfo = key => JM.runs[0].metrics.find(m => m.key === key);
  const useUm = () => state.useUm && allCalibrated(selectedRuns());

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
  const methods = runs => [...new Set(runs.flatMap(r => r.methods))].sort(
    (a, b) => (a === "tcj" ? -1 : b === "tcj" ? 1 : a.localeCompare(b)));

  function initState() {
    if (JM.runs.length === 1) state.run = JM.runs[0].run;
    const rs = roles(JM.runs);
    if (rs.length && !rs.some(r => r.role === state.role)) state.role = rs[0].role;
    const ms = methods(JM.runs);
    if (ms.length && !ms.includes(state.method)) state.method = ms[0];
    if (!metricInfo(state.metric)) state.metric = JM.runs[0].metrics[0].key;
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
    for (const kid of kids.flat()) {
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
    return {view: "overview"};
  }
  JM.route = route;
  JM.fieldHref = (run, field, sel) => `#/field/${run}/${encodeURIComponent(field)}${sel ? `?${sel}` : ""}`;

  /*** FILTER ROW ***/

  function renderFilters(r) {
    const box = document.getElementById("filters");
    box.replaceChildren();
    const runs = selectedRuns();
    if (r.view === "overview" && JM.runs.length > 1) {
      box.append(control("Run", null, select(
        [{value: "all", label: `All runs (${JM.runs.length})`}]
          .concat(JM.runs.map(x => ({value: x.run, label: x.label}))),
        state.run, v => set({run: v}), "Runs shown below")));
    }
    box.append(control("Marker", "Measured junction channel", select(
      roles(runs).map(x => ({value: x.role, label: `${x.role} · ${x.markers.join(" / ")}`})),
      state.role, v => set({role: v}))));
    const m = metricInfo(state.metric);
    box.append(control("Metric", m.help + (m.td ? " Depends on the marker threshold." : ""), select(
      JM.runs[0].metrics.map(x => ({value: x.key, label: x.label, title: x.help})),
      state.metric, v => set({metric: v}))));
    box.append(control("Corners", null, segmented(
      methods(runs).map(x => ({value: x, label: x, tip: `${METHOD_NAMES[x] || x}. ${METHOD_HELP[x] || ""}`})),
      state.method, v => set({method: v}), "Corner method")));
    if (allCalibrated(runs)) {
      const unitless = !m.unit_um;
      box.append(control("Units", null, segmented([
        {value: true, label: "µm", tip: unitless ? "This metric has no length unit" : "Calibrated units", disabled: unitless},
        {value: false, label: "px", tip: unitless ? "This metric has no length unit" : "Image pixels", disabled: unitless},
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
    const label = JM.runs.length === 1 ? JM.runs[0].label : `${JM.runs.length} runs`;
    if (r.view === "overview") {
      nav.append(h("span", {class: "here"}, label));
    } else {
      const run = runById(r.run);
      nav.append(h("a", {href: "#/"}, label), h("span", {class: "sep"}, "/"),
                 h("span", {class: "here"}, `${run && JM.runs.length > 1 ? run.label + " · " : ""}${r.field}`));
    }
  }

  /*** OVERVIEW ***/

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
      .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(" · ");
    return h("span", {class: "badge warning", tip: detail || `${n} flagged cells`}, icon("warning"), L.fmtShare(n, cells).replace(/^(\S+)/, "$1 flagged"));
  }

  function fieldCard(run, f) {
    return h("a", {class: "card field-card", href: JM.fieldHref(run.run, f.field_id),
                   "aria-label": `Open field ${f.field_id}`},
      h("div", {class: "thumb", "data-run": run.run, "data-field": f.field_id}),
      h("div", {class: "field-meta"},
        h("span", {class: "field-name", title: f.field_id}, f.field_id),
        h("span", {class: "field-row"}, `${L.fmtCount(f.n_cells)} cells · ${L.fmtCount(f.n_junctions)} junctions`),
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
  function seriesFor(run, index, metric) {
    const cell = state.dot === "cell";
    const um = useUm() && metric[cell ? "cell_um" : "junction_um"];
    const raw = um || metric[cell ? "cell_px" : "junction_px"];
    const {values, index: pos} = L.finite(raw);
    const text = pos.map(i => (cell
      ? `${index.fields[index.cField[i]]} · cell ${index.cId[i]}`
      : `${index.fields[index.jField[i]]} · junction ${L.junctionKey(index.jLo[i], index.jHi[i])}`));
    return {run, values, pos, text, index};
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

  function plotLayout(title, multi) {
    const ink2 = cssVar("--ink-2"), muted = cssVar("--muted"), grid = cssVar("--grid"), axis = cssVar("--axis");
    return {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: {family: cssVar("--font"), size: 12, color: ink2},
      margin: {l: 64, r: 16, t: 8, b: 56},
      showlegend: multi, legend: {orientation: "h", x: 0, y: 1.08, font: {color: ink2}},
      violinmode: "group", violingap: 0.3,
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
    const unit = L.unitFor(m, allCalibrated(runs), state.useUm);
    // The plot element outlives re-renders, so a reload holds the previous chart, dimmed.
    const plot = JM._lastPlot || h("div", {class: "plot", id: "dist-plot"});
    const note = h("div", {class: "chart-note", hidden: true});
    const tableBox = h("div", {hidden: !state.table});
    plot.hidden = state.table;
    const dotWord = state.dot === "cell" ? "cell" : "junction";
    const card = h("section", {class: "card chart-card"},
      h("div", {class: "chart-head"},
        h("h2", {class: "chart-title", tip: m.help}, m.label),
        h("span", {class: "chart-sub"}, `${state.role} · ${state.method} · one dot per ${dotWord}${unit ? " · " + unit : ""}`),
        segmented([{value: false, label: "Chart"}, {value: true, label: "Table"}], state.table,
                  v => set({table: v}), "Chart or table")),
      plot, tableBox, note);

    JM._lastPlot = plot;
    const token = ++renderToken;
    plot.classList.add("is-loading");
    Promise.all(runs.map(run => Promise.all([
      loadIndex(run.run, state.method),
      loadMetric(run.run, state.method, state.role, state.metric),
    ]).then(([index, metric]) => seriesFor(run, index, metric), () => null)))
      .then(all => {
        if (token !== renderToken) return;
        plot.classList.remove("is-loading");
        const series = all.filter(Boolean);
        const missing = runs.filter((r, i) => !all[i]).map(r => r.label);
        if (missing.length) {
          note.hidden = false;
          note.textContent = `No ${state.role} ${state.method} values: ${missing.join(", ")}`;
        }
        tableBox.replaceChildren(statsTable(series, unit));
        if (!series.length) { Plotly.purge(plot); plot.hidden = true; return; }
        if (state.table) return;
        const title = `${m.label}${unit ? ` (${unit})` : ""}`;
        Plotly.react(plot, plotTraces(series, unit), plotLayout(title, series.length > 1), PLOT_CONFIG);
        plot.removeAllListeners("plotly_click");
        plot.on("plotly_click", ev => {
          const pt = ev.points && ev.points[0];
          if (!pt || pt.customdata === undefined) return;
          const s = series[pt.curveNumber], i = pt.customdata;
          const sel = state.dot === "cell"
            ? `c=${s.index.cId[i]}`
            : `j=${L.junctionKey(s.index.jLo[i], s.index.jHi[i])}`;
          const field = state.dot === "cell" ? s.index.fields[s.index.cField[i]] : s.index.fields[s.index.jField[i]];
          location.hash = JM.fieldHref(s.run.run, field, sel);
        });
      });
    return card;
  }

  JM.views.overview = function (root) {
    const runs = selectedRuns();
    root.append(renderTiles(runs), renderDistribution(runs), renderFieldGrid(runs));
    fillThumbs(root);
  };

  /*** FIELD VIEW ***/

  // Additive tints per channel role; the image well is always dark.
  const TINT = {segment: [1, 1, 1], JM1: [0.25, 1, 0.35], JM2: [1, 0.3, 1], nucleus: [0.35, 0.55, 1]};
  const TINT_OTHER = [1, 1, 1];
  const LINE_PX = 2.5;            // junction stroke, screen px
  const HIT_PX = 6;               // junction hit tolerance, screen px
  const FLAG_COLOUR = "#fab219";  // amber: never on the blue ramp
  const NO_JUNCTIONS = "No junctions measured in this field.";

  // Viewer settings outlive re-renders: channel layers by role, overlays, and per-field view/selection.
  const layers = {};               // role -> {on, gain}
  const overlay = {junctions: true, outlines: true};
  let fieldUi = {key: null, xf: null, sel: null};
  const fieldData = new Map();     // "run/field" -> decoded field
  JM._cleanup = [];

  const pretty = s => s.replace(/_/g, " ");
  const jLabel = key => key.replace("-", "–");

  function loadField(run, fieldId) {
    const key = `${run}/${fieldId}`;
    if (fieldData.has(key)) return fieldData.get(key);
    const p = loader.load(`runs/${run}/fields/${encodeURIComponent(fieldId)}.js`, {run})
      .then(got => decodeField(got.payload));
    fieldData.set(key, p);
    p.catch(() => fieldData.delete(key));
    return p;
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
    return Promise.all(f.channels.map(c => greyPlane(c.src, f.width, f.height))).then(planes => {
      const cellPath = {};
      for (const id in f.cells) { const p = pathOf([f.cells[id]]); p.closePath(); cellPath[id] = p; }
      const allOutlines = new Path2D();
      Object.values(cellPath).forEach(p => allOutlines.addPath(p));
      const junctionPath = {};
      for (const m in f.junctions) {
        junctionPath[m] = {};
        for (const k in f.junctions[m]) junctionPath[m][k] = pathOf(f.junctions[m][k]);
      }
      return {raw: f, width: f.width, height: f.height, planes,
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
    const um = run.calibrated && state.useUm && m.unit_um;
    const arr = kind => (metric ? (um && metric[`${kind}_um`]) || metric[`${kind}_px`] : null);
    const jArr = arr("junction"), cArr = arr("cell");
    const jValue = new Map(slice.junctions.map(j => [j.key, jArr ? jArr[j.i] : NaN]));
    const cValue = new Map(slice.cells.map(c => [String(c.cell_id), cArr ? cArr[c.i] : NaN]));
    const [lo, hi] = L.quantileRange([...jValue.values()]);
    const byCell = new Map();
    slice.junctions.forEach(j => j.key.split("-").forEach(c => {
      if (!byCell.has(c)) byCell.set(c, []);
      byCell.get(c).push(j.key);
    }));
    return {slice, jValue, cValue, lo, hi, byCell, unit: L.unitFor(m, run.calibrated, state.useUm),
            measured: new Set(slice.cells.map(c => String(c.cell_id))), missing: !metric,
            polylines: fd.raw.junctions[state.method] || {}};
  }

  // The junction keys a selection lights up, or null when nothing is selected.
  function emphasis(sel, model) {
    if (sel.brush) {
      if (state.dot !== "cell") return new Set(sel.brush);
      return new Set(sel.brush.flatMap(c => model.byCell.get(c) || []));
    }
    if (sel.junction) return new Set([sel.junction]);
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
          const dim = lit && !lit.has(key);
          const hot = key === sel.junction || (hover && hover.junction === key);
          ctx.globalAlpha = dim ? 0.22 : 1;
          if (hot) {
            ctx.lineWidth = (LINE_PX + 8) / k; ctx.strokeStyle = "rgba(255,255,255,0.95)"; ctx.stroke(paths[key]);
            ctx.lineWidth = (LINE_PX + 4) / k; ctx.strokeStyle = "#0b0b0b"; ctx.stroke(paths[key]);
          } else {
            ctx.lineWidth = (LINE_PX + 2) / k; ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.stroke(paths[key]);
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
      readout.replaceChildren(
        h("b", null, `${L.fmtValue(v)}${model.unit && Number.isFinite(v) ? " " + model.unit : ""}`),
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
      const btn = toggle(c.role, l.on, () => {
        l.on = !l.on;
        btn.setAttribute("aria-pressed", String(l.on));
        viewer.draw();
      }, `${c.role} · ${c.marker}`, sw);
      const slider = h("input", {type: "range", class: "gain", min: "0.2", max: "4", step: "0.05", value: String(l.gain),
                                 "aria-label": `${c.role} brightness`, tip: `${c.role} brightness`});
      slider.addEventListener("input", () => { l.gain = Number(slider.value); viewer.draw(); });
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

  function detailPanel(run, f, fd, model, sel, box) {
    const token = (box._token = (box._token || 0) + 1);
    if (!sel.junction && !sel.cell) {
      box.replaceChildren(h("p", {class: "detail-empty"},
        state.dot === "cell" ? "Click a cell in the image or chart." : "Click a junction in the image or chart."));
      return;
    }
    const isJ = !!sel.junction;
    const cells = isJ ? sel.junction.split("-") : [sel.cell];
    const pos = isJ ? model.slice.junctions.find(j => j.key === sel.junction)
                    : model.slice.cells.find(c => String(c.cell_id) === sel.cell);
    const flagRow = id => h("div", {class: "detail-cell"},
      h("span", {class: "detail-cell-id"}, `Cell ${id}`),
      ((fd.raw.flags[id] || []).map(n => h("span", {class: "flag-tag", tip: "QC flag"}, pretty(n)))));
    const head = h("div", {class: "detail-head"},
      h("h3", {tip: isJ ? "Junction value: mean of its two cell sides" : "Cell value: median over its junctions"},
        isJ ? `Junction ${jLabel(sel.junction)}` : `Cell ${sel.cell}`),
      h("span", {class: "detail-sub"}, `${state.role} · ${state.method}`
        + (isJ ? "" : ` · ${L.fmtCount((model.byCell.get(sel.cell) || []).length)} junctions`)));
    const table = h("table", {class: "data-table detail-table"}, h("tbody", null,
      run.metrics.map(m => h("tr", {class: m.key === state.metric ? "is-current" : null},
        h("th", {scope: "row"}, h("button", {type: "button", class: "metric-link", tip: m.help,
                                              onclick: () => set({metric: m.key})}, m.label)),
        h("td", {"data-metric": m.key}, "…"),
        h("td", {class: "unit"}, L.unitFor(m, run.calibrated, state.useUm))))));
    const cellFlags = isJ ? cells.map(flagRow)
      : (fd.raw.flags[sel.cell] || []).map(n => h("span", {class: "flag-tag", tip: "QC flag"}, pretty(n)));
    box.replaceChildren(head, h("div", {class: isJ ? "detail-cells" : "detail-cell detail-cells"}, cellFlags), table);
    if (!pos) {
      table.querySelectorAll("td[data-metric]").forEach(td => { td.textContent = "–"; });
      return;
    }
    const kind = isJ ? "junction" : "cell";
    run.metrics.forEach(m => {
      loadMetric(run.run, state.method, state.role, m.key).then(v => {
        if (token !== box._token) return;
        const um = run.calibrated && state.useUm && m.unit_um && v[`${kind}_um`];
        const arr = um || v[`${kind}_px`];
        const td = table.querySelector(`td[data-metric="${m.key}"]`);
        if (td) td.textContent = L.fmtValue(arr[pos.i]);
      }, () => {
        const td = table.querySelector(`td[data-metric="${m.key}"]`);
        if (td) td.textContent = "–";
      });
    });
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
    const ink2 = cssVar("--ink-2"), unit = model.unit, ys = pts.map(p => p.v);
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
    const layout = plotLayout(`${m.label}${unit ? ` (${unit})` : ""}`, false);
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
    const key = `${run.run}/${f.field_id}`;
    if (fieldUi.key !== key) {
      fieldUi = {key, xf: null, sel: {junction: r.junction, cell: r.cell, brush: null}};
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
      h("span", {class: "field-sub"}, `${L.fmtCount(f.n_cells)} cells · ${L.fmtCount(f.n_junctions)} junctions`
        + (JM.runs.length > 1 ? ` · ${run.label}` : ""))));
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
           byCell: new Map(), unit: "", measured: new Set(), missing: true, polylines: {}};
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
          h("span", {class: "bar-label", tip: "QC flags; flagged cells get a dashed amber outline"}, "QC"), flags) : null,
        viewer.el,
        h("div", {class: "viewer-foot"},
          empty ? h("span", {class: "empty-note"}, NO_JUNCTIONS)
            : model.missing ? h("span", {class: "empty-note"}, `No ${state.role} ${state.method} values in this run`)
              : colourBar(model)));
      body.replaceChildren(card);
      if (!empty) {
        body.append(h("div", {class: "field-side"},
          h("section", {class: "card detail-card"}, detail),
          h("section", {class: "card chart-card"},
            h("div", {class: "chart-head"},
              h("h2", {class: "chart-title", tip: m.help}, m.label),
              h("span", {class: "chart-sub", tip: "Lasso or box-select to highlight in the image; double-click to clear"},
                `${state.role} · ${state.method} · per ${state.dot}`)),
            plot)));
        chart = fieldChart(run, model, sel, plot, hooks);
        JM._fieldPlot = plot;
      }
      sync();
    }, err => {
      body.replaceChildren(h("p", {class: "empty-note"}, `Could not load this field: ${err.message}`));
    });
  };

  /*** RENDER ***/

  function render() {
    const r = route();
    renderCrumbs(r);
    renderFilters(r);
    const root = document.getElementById("view");
    if (JM._lastPlot && r.view !== "overview") { Plotly.purge(JM._lastPlot); JM._lastPlot = null; }
    if (JM._fieldPlot) { Plotly.purge(JM._fieldPlot); JM._fieldPlot = null; }
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
    window.addEventListener("hashchange", render);
    render();
  });
})();
