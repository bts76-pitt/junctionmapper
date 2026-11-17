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

  function fmtValue(v) {
    if (!Number.isFinite(v)) return "–";
    if (v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(2);
    if (Math.abs(v) >= 1000) return fmtInt.format(Math.round(v));
    return String(Number(v.toPrecision(3)));
  }

  /*** COLOUR ***/

  // Sequential single-hue blue, light -> dark (steps 100..700 of the reference ramp).
  const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
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

  function unitFor(metric, calibrated, useUm) {
    return calibrated && useUm && metric.unit_um ? metric.unit_um : metric.unit_px;
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

    return {load, receive};
  }

  const api = {decodeF32, decodeU32, finite, quantileRange, summaryStats, fmtCount, fmtValue,
               colourFor, runColour, junctionKey, unitFor, createLoader, NAN_GREY};
  root.JMLogic = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
