/* junctionmapper report: pure functions, no DOM. Browser global JMLogic; CommonJS in node. */
(function (root) {
  "use strict";

  /*** DECODING ***/

  function bytes(b64) {
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return buf;
  }
  // Files are little-endian, as is every browser platform.
  const decodeF32 = b64 => new Float32Array(bytes(b64));
  const decodeU32 = b64 => new Uint32Array(bytes(b64));

  /*** NUMBERS ***/

  function finite(values) {
    const out = {values: [], index: []};
    for (let i = 0; i < values.length; i++) {
      if (Number.isFinite(values[i])) { out.values.push(values[i]); out.index.push(i); }
    }
    return out;
  }

  function sortedFinite(values) {
    return finite(values).values.sort((a, b) => a - b);
  }

  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  function quantileRange(values, q = [0.02, 0.98]) {
    const s = sortedFinite(values);
    if (!s.length) return [0, 1];
    return [quantile(s, q[0]), quantile(s, q[1])];
  }

  function summaryStats(values) {
    const s = sortedFinite(values);
    if (!s.length) return {n: 0, min: NaN, q1: NaN, median: NaN, q3: NaN, max: NaN};
    return {n: s.length, min: s[0], q1: quantile(s, 0.25), median: quantile(s, 0.5),
            q3: quantile(s, 0.75), max: s[s.length - 1]};
  }

  const fmtInt = new Intl.NumberFormat("en-US");
  const fmtCompact = new Intl.NumberFormat("en-US", {notation: "compact", maximumFractionDigits: 1});
  const fmtCount = n => (Math.abs(n) >= 10000 ? fmtCompact.format(n) : fmtInt.format(n));

  // "12 (6.5%)": a count with its share of total; whole percent from 10% up.
  function fmtShare(n, total) {
    if (!total) return fmtCount(n);
    const pct = (100 * n) / total;
    return `${fmtCount(n)} (${pct >= 10 || pct === 0 ? Math.round(pct) : pct.toFixed(1)}%)`;
  }

  function fmtValue(v) {
    if (!Number.isFinite(v)) return "–";
    if (v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(2);
    if (Math.abs(v) >= 1000) return fmtInt.format(Math.round(v));
    return String(Number(v.toPrecision(3)));
  }

  /*** COLOUR ***/

  // Junction ramp: perceptual sequential (viridis), dark purple -> yellow. Every step is
  // saturated, so with its dark casing a junction reads on bright or dark fluorescence alike.
  const RAMP = ["#440154", "#482475", "#414487", "#355f8d", "#2a788e", "#21918c", "#22a884",
                "#44bf70", "#7ad151", "#bddf26", "#fde725"]
    .map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)));
  const NAN_GREY = "rgb(150,150,150)";

  function colourFor(v, lo, hi) {
    if (!Number.isFinite(v)) return NAN_GREY;
    let t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
    t = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
    const c = RAMP[i].map((a, k) => Math.round(a + (RAMP[i + 1][k] - a) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  // Categorical slots in fixed order (validated adjacent-pair CVD-safe, light and dark).
  // Runs past the eighth share one neutral grey: hues are never cycled or generated.
  const RUN_COLOURS = {
    light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
  };
  const OVERFLOW_GREY = "#898781";
  const runColour = (i, theme) => RUN_COLOURS[theme === "dark" ? "dark" : "light"][i] || OVERFLOW_GREY;

  /*** KEYS AND UNITS ***/

  const junctionKey = (lo, hi) => `${lo}-${hi}`;

  // The one rule for "show this metric in µm?": the run is calibrated, the user asked, and the
  // metric has a µm form. Every view goes through it, so a number reads the same everywhere.
  const wantsUm = (m, calibrated, useUm) => !!(calibrated && useUm && m.unit_um);

  function unitFor(m, calibrated, useUm) {
    return wantsUm(m, calibrated, useUm) ? m.unit_um : m.unit_px;
  }

  // The values of one metric file for kind "junction" | "cell", and the unit of the array
  // actually used: a missing µm array falls back to px and is labelled px.
  function metricValues(metric, kind, m, calibrated, useUm) {
    const want = wantsUm(m, calibrated, useUm);
    if (!metric) return {values: null, unit: want ? m.unit_um : m.unit_px, um: want};
    const um = want && !!metric[`${kind}_um`];
    return {values: um ? metric[`${kind}_um`] : metric[`${kind}_px`], unit: um ? m.unit_um : m.unit_px, um};
  }

  /*** FILTERS ***/

  // roles: [{role}, ...] of the runs in view. The current role if they have it, else the first.
  function pickRole(roles, current) {
    return !roles.length || roles.some(r => r.role === current) ? current : roles[0].role;
  }

  // Every metric of the runs in view, each once, in the order first seen.
  function unionMetrics(runs) {
    const seen = new Map();
    runs.forEach(r => r.metrics.forEach(m => { if (!seen.has(m.key)) seen.set(m.key, m); }));
    return [...seen.values()];
  }

  /*** COMPARISON ***/

  // Where a distribution point leads: its field, and the junction (j=) or cell (c=) to select.
  function pointTarget(index, i, dot) {
    return dot === "cell"
      ? {field: index.fields[index.cField[i]], sel: `c=${index.cId[i]}`}
      : {field: index.fields[index.jField[i]], sel: `j=${junctionKey(index.jLo[i], index.jHi[i])}`};
  }

  // One row per run for the comparison table, in run order; flagged counts measured cells only.
  function runRows(runs) {
    return runs.map(r => ({
      run: r.run, label: r.label, fields: r.tiles.fields, cells: r.tiles.cells,
      junctions: r.tiles.junctions, flagged: r.tiles.flagged_cells,
    }));
  }

  /*** FIELD GEOMETRY ***/

  // Positions in the metric index arrays that belong to one field (fieldPos = its place in fields).
  function fieldSlice(index, fieldPos) {
    const junctions = [], cells = [];
    for (let i = 0; i < index.junction_field.length; i++) {
      if (index.junction_field[i] === fieldPos) {
        junctions.push({i, key: junctionKey(index.junction_lo[i], index.junction_hi[i])});
      }
    }
    for (let i = 0; i < index.cell_field.length; i++) {
      if (index.cell_field[i] === fieldPos) cells.push({i, cell_id: index.cell_id[i]});
    }
    return {junctions, cells};
  }

  function segmentDist2(px, py, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy;
    const t = len2 ? Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / len2)) : 0;
    const x = a[0] + t * dx - px, y = a[1] + t * dy - py;
    return x * x + y * y;
  }

  // polylines: {key: [[[x, y], ...], ...]}. The key of the nearest junction within tol, else null.
  function hitTest(polylines, x, y, tol) {
    let best = null, bestD = tol * tol;
    for (const key in polylines) {
      for (const line of polylines[key]) {
        for (let k = 0; k < line.length; k++) {
          const d = segmentDist2(x, y, line[k], line[Math.min(k + 1, line.length - 1)]);
          if (d <= bestD) { bestD = d; best = key; }
        }
      }
    }
    return best;
  }

  // Even-odd ray casting; outlines: {cell_id: [[x, y], ...]}.
  function inside(poly, x, y) {
    let hit = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  }

  // Width and height of the preview planes; every coordinate in the asset is in this space.
  // Assets from before raw_width held the raw size in width/height.
  function planeSize(f) {
    if (f.raw_width !== undefined) return {width: f.width, height: f.height};
    const s = f.scale || 1;
    return {width: Math.max(1, Math.round(f.width * s)), height: Math.max(1, Math.round(f.height * s))};
  }

  // Junction opacity: only a brush (lasso / box) dims what it did not catch. A click selection
  // halos its junction and leaves every other junction as it was.
  const DIM_ALPHA = 0.6;
  function junctionAlpha(sel, lit, key) {
    return sel && sel.brush && lit && !lit.has(key) ? DIM_ALPHA : 1;
  }

  function cellAt(outlines, x, y) {
    for (const id in outlines) if (inside(outlines[id], x, y)) return id;
    return null;
  }

  // The field view's selection on (re)entry: the route's ?j= / ?c= wins; a plain link selects
  // nothing. A brush is never in the route, so it survives only a re-render of the same field.
  function selectionFromRoute(route, prev) {
    const junction = route.junction || null, cell = route.cell || null;
    return {junction, cell, brush: prev && !junction && !cell ? prev.brush : null,
            dot: prev ? prev.dot : null};
  }

  // sides: the field asset's {method: {key: {role: {cell: {metric: [px, um|null]}}}}}.
  function sideValue(sides, method, key, role, cell, metric, useUm) {
    const v = sides && sides[method] && sides[method][key] && sides[method][key][role]
      && sides[method][key][role][cell] && sides[method][key][role][cell][metric];
    if (!v) return NaN;
    const x = useUm && v[1] !== null && v[1] !== undefined ? v[1] : v[0];
    return x === null || x === undefined ? NaN : x;
  }

  /*** DETAIL FIGURES ***/

  // Square crop around every point of polylines ({key: [[[x, y], ...], ...]} or [[[x, y], ...], ...]):
  // the bbox padded by margin x its longer side, at least min, clamped inside width x height.
  function cropWindow(polylines, width, height, margin = 0.4, min = 48) {
    const lines = Array.isArray(polylines) ? polylines : Object.values(polylines).flat();
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    lines.forEach(line => line.forEach(([x, y]) => {
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }));
    if (!Number.isFinite(x0)) { x0 = x1 = width / 2; y0 = y1 = height / 2; }
    const span = Math.max(x1 - x0, y1 - y0);
    const side = Math.max(1, Math.min(Math.round(Math.max(min, span * (1 + 2 * margin))), width, height));
    const clamp = (c, lim) => Math.max(0, Math.min(lim - side, Math.round(c - side / 2)));
    return {x: clamp((x0 + x1) / 2, width), y: clamp((y0 + y1) / 2, height), w: side, h: side};
  }

  // Share of the finite values <= v, in whole percent; null when v or every value is not finite.
  function percentileOf(values, v) {
    if (!Number.isFinite(v)) return null;
    let n = 0, below = 0;
    for (const x of values) if (Number.isFinite(x)) { n++; if (x <= v) below++; }
    return n ? Math.round((100 * below) / n) : null;
  }

  // Axis extent of one role's profile: x to its last finite point, y to its peak or threshold.
  function profileScale(profile, thresholds, role) {
    let xmax = 0, ymax = 0;
    (profile.x || []).forEach(x => { if (Number.isFinite(x)) xmax = Math.max(xmax, x); });
    (profile.y || []).forEach(y => { if (Number.isFinite(y)) ymax = Math.max(ymax, y); });
    const t = thresholds ? thresholds[role] : null;
    if (Number.isFinite(t)) ymax = Math.max(ymax, t);
    return {xmax, ymax};
  }

  // The channels the close-up's Merge tile composites: JM1, JM2 and outline (segment), in that
  // order, when at least two of them exist; never the nucleus. [] means no Merge tile.
  const MERGE_ROLES = ["JM1", "JM2", "segment"];
  function mergeRoles(roles) {
    const out = MERGE_ROLES.filter(r => roles.includes(r));
    return out.length >= 2 ? out : [];
  }

  // Merge tile colours by merge order, not by role: a colocalisation set where the first two
  // overlap to white and the third (cyan) stays distinct from both.
  const MERGE_TINTS = [[0, 1, 0], [1, 0, 1], [0, 1, 1]];
  function mergeTints(merged) {
    return merged.map((_, i) => MERGE_TINTS[i % MERGE_TINTS.length]);
  }

  // Does segment (ax,ay)-(bx,by) touch the box {x, y, w, h}? Liang-Barsky clip.
  function segmentHitsBox(ax, ay, bx, by, b) {
    let t0 = 0, t1 = 1;
    const dx = bx - ax, dy = by - ay;
    const edges = [[-dx, ax - b.x], [dx, b.x + b.w - ax], [-dy, ay - b.y], [dy, b.y + b.h - ay]];
    for (const [p, q] of edges) {
      if (p === 0) { if (q < 0) return false; continue; }
      const r = q / p;
      if (p < 0) { if (r > t1) return false; if (r > t0) t0 = r; }
      else { if (r < t0) return false; if (r < t1) t1 = r; }
    }
    return true;
  }

  // {box, crossings}: the candidate box the fewest line segments cross, the first on a tie;
  // null with no boxes. lines: [[[x, y], ...], ...], one array per unbroken run of the drawn line.
  function labelBoxAt(lines, boxes) {
    let best = null, bestN = Infinity;
    boxes.forEach(b => {
      let n = 0;
      lines.forEach(line => {
        if (line.length === 1) { const [x, y] = line[0]; if (segmentHitsBox(x, y, x, y, b)) n++; }
        for (let i = 1; i < line.length; i++) {
          if (segmentHitsBox(line[i - 1][0], line[i - 1][1], line[i][0], line[i][1], b)) n++;
        }
      });
      if (n < bestN) { bestN = n; best = b; }
    });
    return best ? {box: best, crossings: bestN} : null;
  }

  // "P-cadherin (JM1)": marker first, role in brackets; the segment channel reads "outline".
  function channelLabel(role, marker) {
    const name = role === "segment" ? "outline" : role;
    return marker ? `${marker} (${name})` : name;
  }

  /*** PROVENANCE ***/

  // Rows whose group, key or value contain every whitespace-separated term (any case).
  function filterRows(rows, query) {
    const terms = String(query || "").toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return rows.slice();
    return rows.filter(r => {
      const text = `${r.group} ${r.key} ${r.value}`.toLowerCase();
      return terms.every(t => text.includes(t));
    });
  }

  /*** CACHE ***/

  // Map with a size cap: get() and set() mark a key most recent; past the cap the least recent
  // entry goes (onEvict(key, value)). The entry just set is never the one evicted.
  function createLru(cap, onEvict) {
    const m = new Map();
    return {
      get(k) {
        if (!m.has(k)) return undefined;
        const v = m.get(k);
        m.delete(k); m.set(k, v);
        return v;
      },
      set(k, v) {
        m.delete(k); m.set(k, v);
        while (m.size > Math.max(1, cap)) {
          const old = m.keys().next().value;
          const ov = m.get(old);
          m.delete(old);
          if (onEvict) onEvict(old, ov);
        }
      },
      delete: k => m.delete(k),
      keys: () => m.keys(),
      get size() { return m.size; },
    };
  }

  /*** SCRIPT LOADER ***/

  // Data files are <script>s that call JM.register*(payload). The run id inside a copied file is
  // not authoritative, so requests run one at a time and each registration belongs to the request
  // in flight; the caller's ctx (e.g. {run: "r2"}) travels with it.
  // inject(src, onload, onerror) adds the script; receive(kind, payload) is wired to JM.register*.
  function createLoader(inject) {
    const cache = new Map();
    let chain = Promise.resolve();
    let inflight = null;

    function load(src, ctx) {
      if (cache.has(src)) return cache.get(src);
      const p = chain.then(() => new Promise((resolve, reject) => {
        inflight = {src, ctx, got: null};
        const done = ok => () => {
          const req = inflight;
          inflight = null;
          if (!ok) reject(new Error(`could not load ${src}`));
          else if (!req.got) reject(new Error(`${src} did not register`));
          else resolve(req.got);
        };
        inject(src, done(true), done(false));
      }));
      chain = p.catch(() => {});
      cache.set(src, p);
      p.catch(() => cache.delete(src));
      return p;
    }

    function receive(kind, payload) {
      if (!inflight) return false;
      inflight.got = {kind, payload, ctx: inflight.ctx, src: inflight.src};
      return true;
    }

    // Drop a finished file from the cache (memory); an in-flight request is left alone.
    function forget(src) {
      if (!inflight || inflight.src !== src) cache.delete(src);
    }

    return {load, receive, forget};
  }

  const api = {decodeF32, decodeU32, finite, quantileRange, summaryStats, fmtCount, fmtShare, fmtValue,
               colourFor, runColour, junctionKey, unitFor, wantsUm, metricValues, pickRole, unionMetrics,
               planeSize, junctionAlpha, fieldSlice, hitTest, cellAt,
               selectionFromRoute, sideValue, createLru, createLoader, pointTarget, runRows, filterRows,
               cropWindow, percentileOf, profileScale, channelLabel, mergeRoles, mergeTints, labelBoxAt, NAN_GREY};
  root.JMLogic = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
