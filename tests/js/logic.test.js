const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../../src/junctionmapper/report/static/logic.js");

test("decodeF32 round-trips", () => { assert.deepEqual([...L.decodeF32("AACAPwAAAEA=")], [1, 2]); });
test("decodeU32 round-trips", () => { assert.deepEqual([...L.decodeU32("AQAAAAcAAAA=")], [1, 7]); });
test("decodeF32 keeps NaN", () => {
  const b64 = Buffer.from(new Float32Array([NaN, 3]).buffer).toString("base64");
  const out = L.decodeF32(b64);
  assert.ok(Number.isNaN(out[0])); assert.equal(out[1], 3);
});

test("NaN is grey, never a ramp colour", () => { assert.equal(L.colourFor(NaN, 0, 1), "rgb(150,150,150)"); });
test("colourFor runs dark to light (viridis) and clamps", () => {
  const rgb = s => s.match(/\d+/g).map(Number);
  const lum = s => { const [r, g, b] = rgb(s); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
  assert.equal(L.colourFor(0, 0, 1), "rgb(68,1,84)");       // dark purple
  assert.equal(L.colourFor(1, 0, 1), "rgb(253,231,37)");    // yellow
  for (let i = 0; i < 10; i++) assert.ok(lum(L.colourFor(i / 10, 0, 1)) < lum(L.colourFor((i + 1) / 10, 0, 1)));
  assert.equal(L.colourFor(-5, 0, 1), L.colourFor(0, 0, 1));
  assert.equal(L.colourFor(9, 0, 1), L.colourFor(1, 0, 1));
  assert.match(L.colourFor(3, 3, 3), /^rgb\(\d+,\d+,\d+\)$/);
});
test("no ramp colour is the NaN grey", () => {
  for (let i = 0; i <= 20; i++) assert.notEqual(L.colourFor(i / 20, 0, 1), L.NAN_GREY);
});

test("finite drops NaN and keeps positions", () => {
  assert.deepEqual(L.finite([1, NaN, 3]), {values: [1, 3], index: [0, 2]}); });
test("finite accepts typed arrays", () => {
  assert.deepEqual(L.finite(new Float32Array([NaN, 2])), {values: [2], index: [1]}); });

test("quantileRange ignores NaN and interpolates", () => {
  assert.deepEqual(L.quantileRange([0, NaN, 10], [0, 1]), [0, 10]);
  assert.deepEqual(L.quantileRange([0, 10], [0.5, 0.5]), [5, 5]);
  const [lo, hi] = L.quantileRange(Array.from({length: 101}, (_, i) => i));
  assert.equal(lo, 2); assert.equal(hi, 98);
});
test("quantileRange of nothing finite is a unit range", () => {
  assert.deepEqual(L.quantileRange([NaN]), [0, 1]); });

test("junctionKey", () => { assert.equal(L.junctionKey(3, 12), "3-12"); });

test("unit hides µm when uncalibrated", () => {
  assert.equal(L.unitFor({unit_px: "px", unit_um: "µm"}, false, true), "px"); });
test("unit uses µm when calibrated and asked", () => {
  assert.equal(L.unitFor({unit_px: "px²", unit_um: "µm²"}, true, true), "µm²");
  assert.equal(L.unitFor({unit_px: "px²", unit_um: "µm²"}, true, false), "px²");
  assert.equal(L.unitFor({unit_px: "", unit_um: ""}, true, true), "");
});

test("summaryStats gives n, quartiles and range over finite values", () => {
  assert.deepEqual(L.summaryStats([4, NaN, 1, 3, 2, 5]),
    {n: 5, min: 1, q1: 2, median: 3, q3: 4, max: 5});
  assert.equal(L.summaryStats([NaN]).n, 0);
});

test("fmtCount compacts large counts only", () => {
  assert.equal(L.fmtCount(1284), "1,284");
  assert.equal(L.fmtCount(12900), "12.9K");
});
test("fmtShare gives the count and its percentage", () => {
  assert.equal(L.fmtShare(12, 185), "12 (6.5%)");
  assert.equal(L.fmtShare(60, 120), "60 (50%)");
  assert.equal(L.fmtShare(0, 20), "0 (0%)");
  assert.equal(L.fmtShare(0, 0), "0");
});
test("fmtValue keeps three significant figures and shows NaN as a dash", () => {
  assert.equal(L.fmtValue(0.123456), "0.123");
  assert.equal(L.fmtValue(1234.5), "1,235");
  assert.equal(L.fmtValue(NaN), "–");
});

test("runColour follows run order and never cycles", () => {
  assert.equal(L.runColour(0, "light"), "#2a78d6");
  assert.equal(L.runColour(0, "dark"), "#3987e5");
  assert.notEqual(L.runColour(8, "light"), L.runColour(0, "light"));
  assert.equal(L.runColour(8, "light"), L.runColour(9, "light"));
});

// The run id inside a copied file is not authoritative: a registration belongs to the
// run whose file was requested.
function fakeInject(files) {
  const calls = [];
  const inject = (src, onload, onerror) => {
    calls.push(src);
    setTimeout(() => {
      if (!(src in files)) return onerror();
      if (files[src]) loader.receive(files[src].kind, files[src].payload);
      onload();
    }, 0);
  };
  const loader = L.createLoader(inject);
  return {loader, calls};
}

test("loader attributes each registration to the requested run", async () => {
  const {loader} = fakeInject({
    "runs/r1/metrics/tcj/_index.js": {kind: "Index", payload: {run: "r1", fields: ["a"]}},
    "runs/r2/metrics/tcj/_index.js": {kind: "Index", payload: {run: "r1", fields: ["b"]}},
  });
  const [a, b] = await Promise.all([
    loader.load("runs/r1/metrics/tcj/_index.js", {run: "r1"}),
    loader.load("runs/r2/metrics/tcj/_index.js", {run: "r2"})]);
  assert.equal(a.ctx.run, "r1"); assert.deepEqual(a.payload.fields, ["a"]);
  assert.equal(b.ctx.run, "r2"); assert.deepEqual(b.payload.fields, ["b"]);
});

test("loader runs one request at a time and caches by src", async () => {
  const {loader, calls} = fakeInject({x: {kind: "Metric", payload: {}}, y: {kind: "Metric", payload: {}}});
  const px = loader.load("x", {run: "r1"});
  loader.load("y", {run: "r1"});
  assert.deepEqual(calls, []);           // nothing injected synchronously
  await px;
  assert.deepEqual(calls, ["x"]);        // y waits for x
  await loader.load("y", {run: "r1"});
  await loader.load("x", {run: "r1"});
  assert.deepEqual(calls, ["x", "y"]);   // cached
});

test("loader rejects a missing file and a file that does not register", async () => {
  const {loader} = fakeInject({silent: null});
  await assert.rejects(loader.load("missing", {}), /could not load missing/);
  await assert.rejects(loader.load("silent", {}), /silent did not register/);
});

test("a registration with no request in flight is ignored", () => {
  const loader = L.createLoader(() => {});
  assert.equal(loader.receive("Field", {}), false);
});

test("hitTest finds the junction under the cursor", () => {
  const p = {"2-4": [[[0, 0], [10, 0]]], "4-5": [[[0, 10], [10, 10]]]};
  assert.equal(L.hitTest(p, 5, 1, 3), "2-4"); assert.equal(L.hitTest(p, 5, 5, 3), null); });
test("hitTest takes the nearest of several and searches every polyline", () => {
  const p = {"1-2": [[[0, 0], [10, 0]]], "2-3": [[[50, 50], [60, 50]], [[0, 4], [10, 4]]]};
  assert.equal(L.hitTest(p, 5, 3, 3), "2-3");
  assert.equal(L.hitTest(p, 12, 0, 3), "1-2");   // past the end: distance to the endpoint
  assert.equal(L.hitTest({"1-2": [[[3, 3]]]}, 4, 3, 2), "1-2");   // single-point polyline
});
test("cellAt uses point-in-polygon", () => {
  assert.equal(L.cellAt({"7": [[0,0],[10,0],[10,10],[0,10]]}, 5, 5), "7"); });
test("cellAt is null outside every outline and handles concave outlines", () => {
  const u = {"3": [[0, 0], [10, 0], [10, 10], [7, 10], [7, 3], [3, 3], [3, 10], [0, 10]]};
  assert.equal(L.cellAt(u, 5, 8), null);
  assert.equal(L.cellAt(u, 1, 8), "3");
  assert.equal(L.cellAt(u, 20, 20), null);
});
test("fieldSlice picks only that field's rows", () => {
  const idx = {junction_field: [0, 0, 1], junction_lo: [2, 3, 2], junction_hi: [4, 5, 4],
               cell_field: [0, 1], cell_id: [2, 2]};
  assert.deepEqual(L.fieldSlice(idx, 1).junctions, [{i: 2, key: "2-4"}]);
  assert.deepEqual(L.fieldSlice(idx, 0).cells, [{i: 0, cell_id: 2}]);
  assert.deepEqual(L.fieldSlice(idx, 5), {junctions: [], cells: []});
});

test("selectionFromRoute: the route's junction or cell wins over the remembered selection", () => {
  const prev = {junction: "2-4", cell: null, brush: ["1-2"], dot: "junction"};
  assert.deepEqual(L.selectionFromRoute({junction: "7-9", cell: null}, prev),
    {junction: "7-9", cell: null, brush: null, dot: "junction"});
  assert.deepEqual(L.selectionFromRoute({junction: null, cell: "5"}, prev),
    {junction: null, cell: "5", brush: null, dot: "junction"});
});
test("selectionFromRoute: a plain link opens with nothing selected", () => {
  assert.deepEqual(L.selectionFromRoute({junction: null, cell: null}, null),
    {junction: null, cell: null, brush: null, dot: null});
});
test("selectionFromRoute: a re-render of the same field keeps its brush", () => {
  const prev = {junction: null, cell: null, brush: ["1-2", "3-4"], dot: "junction"};
  assert.deepEqual(L.selectionFromRoute({junction: null, cell: null}, prev).brush, ["1-2", "3-4"]);
});

test("createLru evicts the least recently used past its cap, never the newest", () => {
  const gone = [];
  const c = L.createLru(2, k => gone.push(k));
  c.set("a", 1); c.set("b", 2);
  assert.equal(c.get("a"), 1);          // a is now most recent
  c.set("c", 3);
  assert.deepEqual(gone, ["b"]);
  assert.equal(c.get("b"), undefined);
  assert.deepEqual([...c.keys()], ["a", "c"]);
  c.delete("a");
  assert.deepEqual([...c.keys()], ["c"]);
});
test("createLru with cap 1 keeps only the newest", () => {
  const c = L.createLru(1);
  c.set("a", 1); c.set("b", 2);
  assert.deepEqual([...c.keys()], ["b"]);
});

test("sideValue reads one cell side, px or µm, null as NaN", () => {
  const sides = {tcj: {"2-4": {JM1: {"2": {area: [10, 0.4], ratio: [0.5, null]}, "4": {area: [null, null]}}}}};
  assert.equal(L.sideValue(sides, "tcj", "2-4", "JM1", "2", "area", false), 10);
  assert.equal(L.sideValue(sides, "tcj", "2-4", "JM1", "2", "area", true), 0.4);
  assert.equal(L.sideValue(sides, "tcj", "2-4", "JM1", "2", "ratio", true), 0.5);   // no µm: px
  assert.ok(Number.isNaN(L.sideValue(sides, "tcj", "2-4", "JM1", "4", "area", false)));
  assert.ok(Number.isNaN(L.sideValue(sides, "dp", "2-4", "JM1", "2", "area", false)));
  assert.ok(Number.isNaN(L.sideValue(undefined, "tcj", "2-4", "JM1", "2", "area", false)));
});

test("loader.forget drops a cached file so the next load re-requests it", async () => {
  let n = 0;
  const ld = L.createLoader((src, onload) => { n++; ld.receive("Field", {src}); onload(); });
  await ld.load("a.js"); await ld.load("a.js");
  assert.equal(n, 1);
  ld.forget("a.js");
  await ld.load("a.js");
  assert.equal(n, 2);
});

// Comparison view
const IDX = {fields: ["f1", "f2"], jField: [0, 1], jLo: [3, 7], jHi: [5, 9], cField: [1, 0], cId: [4, 2]};
test("pointTarget opens the junction's field with the junction selected", () => {
  assert.deepEqual(L.pointTarget(IDX, 1, "junction"), {field: "f2", sel: "j=7-9"}); });
test("pointTarget opens the cell's field with the cell selected", () => {
  assert.deepEqual(L.pointTarget(IDX, 0, "cell"), {field: "f2", sel: "c=4"}); });
test("runRows gives one row per run in order", () => {
  const runs = [
    {run: "r1", label: "A", tiles: {fields: 2, cells: 40, junctions: 90, flagged_cells: 4}},
    {run: "r2", label: "B", tiles: {fields: 1, cells: 0, junctions: 0, flagged_cells: 0}},
  ];
  assert.deepEqual(L.runRows(runs), [
    {run: "r1", label: "A", fields: 2, cells: 40, junctions: 90, flagged: 4},
    {run: "r2", label: "B", fields: 1, cells: 0, junctions: 0, flagged: 0},
  ]);
});

test("filterRows matches every term in group, key or value, any case", () => {
  const rows = [{group: "Model", key: "model", value: "cpsam"},
                {group: "Thresholds", key: "JM1 in f1", value: "50 (own Otsu 48)"},
                {group: "Versions", key: "junctionmapper", value: "0.1.0"}];
  assert.deepEqual(L.filterRows(rows, ""), rows);
  assert.deepEqual(L.filterRows(rows, "CPSAM"), [rows[0]]);
  assert.deepEqual(L.filterRows(rows, "jm1 otsu"), [rows[1]]);
  assert.deepEqual(L.filterRows(rows, "jm1 cpsam"), []);
});

// ---- final review fixes ----

test("planeSize is the preview size the coordinates are in", () => {
  assert.deepEqual(L.planeSize({width: 1500, height: 1257, scale: 0.5025, raw_width: 2985, raw_height: 2500}),
                   {width: 1500, height: 1257});
  // an asset written before raw_width existed held the raw size: scale it down
  assert.deepEqual(L.planeSize({width: 2984, height: 2500, scale: 1500 / 2984}), {width: 1500, height: 1257});
  assert.deepEqual(L.planeSize({width: 800, height: 600, scale: 1}), {width: 800, height: 600});
});

test("metricValues: one decision for the array and its unit", () => {
  const px = new Float32Array([1]), um = new Float32Array([0.25]);
  const area = {unit_px: "px²", unit_um: "µm²"};
  const dens = {unit_px: "per px²", unit_um: "per µm²"};
  const ratio = {unit_px: "", unit_um: ""};
  const metric = {junction_px: px, junction_um: um, cell_px: px, cell_um: null};
  assert.deepEqual(L.metricValues(metric, "junction", area, true, true), {values: um, unit: "µm²", um: true});
  assert.deepEqual(L.metricValues(metric, "junction", dens, true, true), {values: um, unit: "per µm²", um: true});
  assert.deepEqual(L.metricValues(metric, "junction", area, true, false), {values: px, unit: "px²", um: false});
  assert.deepEqual(L.metricValues(metric, "junction", area, false, true), {values: px, unit: "px²", um: false});
  assert.deepEqual(L.metricValues(metric, "junction", ratio, true, true), {values: px, unit: "", um: false});
  // the µm array is missing: the px array is used and labelled px, never µm
  assert.deepEqual(L.metricValues(metric, "cell", area, true, true), {values: px, unit: "px²", um: false});
  assert.deepEqual(L.metricValues(null, "cell", area, true, true), {values: null, unit: "µm²", um: true});
  assert.equal(L.wantsUm(dens, true, true), true);
  assert.equal(L.wantsUm(ratio, true, true), false);
});

test("pickRole keeps the role when the runs have it, else the first", () => {
  const roles = [{role: "JM1"}, {role: "JM2"}];
  assert.equal(L.pickRole(roles, "JM2"), "JM2");
  assert.equal(L.pickRole([{role: "JM1"}], "JM2"), "JM1");
  assert.equal(L.pickRole([], "JM2"), "JM2");
});

test("unionMetrics lists every metric of the selected runs once, in first-seen order", () => {
  const runs = [{metrics: [{key: "a"}, {key: "b"}]}, {metrics: [{key: "b"}, {key: "c"}]}];
  assert.deepEqual(L.unionMetrics(runs).map(m => m.key), ["a", "b", "c"]);
  assert.deepEqual(L.unionMetrics([]), []);
});

test("junctionAlpha dims only for a brush, never for a click selection", () => {
  const lit = new Set(["1-2"]);
  assert.equal(L.junctionAlpha({junction: "1-2", brush: null}, lit, "3-4"), 1);
  assert.equal(L.junctionAlpha({cell: "1", brush: null}, lit, "3-4"), 1);
  assert.equal(L.junctionAlpha({brush: ["1-2"]}, lit, "1-2"), 1);
  const dimmed = L.junctionAlpha({brush: ["1-2"]}, lit, "3-4");
  assert.ok(dimmed >= 0.5 && dimmed < 1);
  assert.equal(L.junctionAlpha({brush: null}, null, "3-4"), 1);
});

test("cropWindow pads, enforces a minimum and clamps to the image", () => {
  assert.deepEqual(L.cropWindow({"2-4": [[[10, 10], [20, 10]]]}, 100, 100), {x: 0, y: 0, w: 48, h: 48});
  const w = L.cropWindow({"2-4": [[[500, 500], [600, 520]]]}, 1000, 1000);
  assert.ok(w.w >= 140 && w.x + w.w <= 1000 && w.y >= 0);
  assert.equal(w.w, w.h);
});
test("cropWindow accepts a bare list of polylines and never exceeds a small image", () => {
  const w = L.cropWindow([[[1, 1], [3, 3]]], 30, 20);
  assert.ok(w.x >= 0 && w.y >= 0 && w.x + w.w <= 30 && w.y + w.h <= 20);
});
test("percentileOf ignores NaN and is null for NaN input", () => {
  assert.equal(L.percentileOf([1, 2, NaN, 3, 4], 3), 75);
  assert.equal(L.percentileOf([1, 2], NaN), null);
  assert.equal(L.percentileOf([NaN], 1), null);
});
test("profileScale includes the threshold", () => {
  assert.equal(L.profileScale({x: [0, 5], y: [1, 2]}, {JM1: 10}, "JM1").ymax >= 10, true);
  assert.equal(L.profileScale({x: [0, 5], y: [1, 2]}, {JM1: 10}, "JM1").xmax, 5);
});
test("channelLabel puts the marker first and renames segment", () => {
  assert.equal(L.channelLabel("JM1", "P-cadherin"), "P-cadherin (JM1)");
  assert.equal(L.channelLabel("segment", "beta-catenin"), "beta-catenin (outline)");
  assert.equal(L.channelLabel("segment", ""), "outline");
  assert.equal(L.channelLabel("JM1"), "JM1");
});
test("profileScale ignores a missing threshold", () => {
  assert.equal(L.profileScale({x: [0, 5], y: [1, 2]}, {JM1: null}, "JM1").ymax < 10, true);
  assert.ok(L.profileScale({x: [0, 5], y: [1, null]}, {}, "JM1").ymax >= 1);
});
test("mergeRoles merges any two or more of JM1, JM2 and outline, never the nucleus", () => {
  assert.deepEqual(L.mergeRoles(["segment", "JM1", "nucleus"]), ["JM1", "segment"]);
  assert.deepEqual(L.mergeRoles(["JM1", "nucleus"]), []);
  assert.deepEqual(L.mergeRoles(["nucleus", "segment", "JM2", "JM1"]), ["JM1", "JM2", "segment"]);
});
