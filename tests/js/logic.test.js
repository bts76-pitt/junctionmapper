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
test("colourFor runs light to dark and clamps", () => {
  const lum = s => s.match(/\d+/g).map(Number).reduce((a, b) => a + b, 0);
  assert.ok(lum(L.colourFor(0, 0, 1)) > lum(L.colourFor(0.5, 0, 1)));
  assert.ok(lum(L.colourFor(0.5, 0, 1)) > lum(L.colourFor(1, 0, 1)));
  assert.equal(L.colourFor(-5, 0, 1), L.colourFor(0, 0, 1));
  assert.equal(L.colourFor(9, 0, 1), L.colourFor(1, 0, 1));
  assert.match(L.colourFor(3, 3, 3), /^rgb\(\d+,\d+,\d+\)$/);
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
