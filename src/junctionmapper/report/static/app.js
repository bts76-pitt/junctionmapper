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
  JM.data = {loadIndex, loadMetric};

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
    const flagged = sum("flagged_cells");
    const across = runs.length > 1 ? `across ${runs.length} runs` : null;
    return h("div", {class: "tiles"},
      tile("Cells", "Cells with at least one measured interface", L.fmtCount(sum("cells")), across),
      tile("Junctions", "Distinct cell–cell junctions (tricellular corners)", L.fmtCount(sum("junctions")), across),
      tile("Fields", "Images in the run", L.fmtCount(sum("fields")), across),
      tile("Flagged cells", "Segmented cells with at least one QC flag, measured or not", L.fmtCount(flagged), across));
  }

  function qcBadge(qc) {
    const n = qc.flagged_cells || 0;
    if (!n) return h("span", {class: "badge good", tip: "No QC flags"}, icon("good"), "Clean");
    const detail = Object.entries(qc).filter(([k, v]) => k !== "flagged_cells" && v > 0)
      .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(" · ");
    return h("span", {class: "badge warning", tip: detail || `${n} flagged cells`}, icon("warning"), `${n} flagged`);
  }

  function fieldCard(run, f) {
    return h("a", {class: "card field-card", href: JM.fieldHref(run.run, f.field_id),
                   "aria-label": `Open field ${f.field_id}`},
      h("div", {class: "thumb"}, f.thumb ? h("img", {src: f.thumb, alt: "", loading: "lazy", decoding: "async"}) : null),
      h("div", {class: "field-meta"},
        h("span", {class: "field-name", title: f.field_id}, f.field_id),
        h("span", {class: "field-row"}, `${L.fmtCount(f.n_cells)} cells · ${L.fmtCount(f.n_junctions)} junctions`),
        qcBadge(f.qc)));
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
  };

  // The field view proper (image, junctions, linked chart) replaces this.
  JM.views.field = JM.views.field || function (root, r) {
    const run = runById(r.run);
    const f = run && run.fields.find(x => x.field_id === r.field);
    if (!f) {
      root.append(h("p", null, "Field not found. ", h("a", {href: "#/"}, "Back to overview")));
      return;
    }
    root.append(
      h("div", {class: "field-head"}, h("a", {class: "back", href: "#/"}, "← Overview"), h("h1", null, f.field_id), qcBadge(f.qc)),
      h("div", {class: "tiles"},
        tile("Cells", "Cells with at least one measured interface", L.fmtCount(f.n_cells)),
        tile("Junctions", "Distinct cell–cell junctions (tricellular corners)", L.fmtCount(f.n_junctions)),
        tile("Flagged cells", "Segmented cells with at least one QC flag, measured or not", L.fmtCount(f.qc.flagged_cells || 0)),
        tile("Run", "The run this field belongs to", run.label)),
      h("div", {class: "card field-preview"}, h("div", {class: "thumb"}, f.thumb ? h("img", {src: f.thumb, alt: f.field_id}) : null)));
  };

  /*** RENDER ***/

  function render() {
    const r = route();
    renderCrumbs(r);
    renderFilters(r);
    const root = document.getElementById("view");
    if (JM._lastPlot && r.view !== "overview") { Plotly.purge(JM._lastPlot); JM._lastPlot = null; }
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
