# Interactive Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static `report.html` with an offline, interactive dashboard that shows each field's cells and junctions, linked to per-junction metrics, for any IF panel.

**Architecture:** `run` writes `run.json`, one asset file per field (downsampled channel previews, cell outlines, junction polylines), and one data file per metric (junction-level and cell-level values as base64 Float32). `index.html` is a vanilla-JS app with a vendored Plotly bundle. It loads data files on demand through `<script>` tags, so it works from `file://`. `junctionmapper report` builds the same app over several runs.

**Tech Stack:** Python 3.11, numpy, pandas, Pillow, jinja2 (all already in the container). Vanilla JS (ES2020) + vendored `plotly.js-cartesian-dist-min@2.35.2`. JS unit tests with `node --test` (host module `nodejs/24.4.1`; not in the container).

**Spec:** `docs/superpowers/specs/2026-11-16-interactive-report-design.md`

## Global Constraints

- Opens offline from `file://` in current Chrome, Firefox, Safari, Edge. No `fetch()`; data arrives by `<script>` tags calling `JM.register*`.
- References nothing external: no CDN, no web fonts, no URLs. Libraries are vendored in `src/junctionmapper/report/static/`.
- Hundreds of fields per run: `index.html` must not embed per-field images or per-junction values.
- Generic: marker names, roles present (JM1, optional JM2, optional segment, nucleus), and metrics come from config and tables. No experiment-specific strings.
- Report generation never changes any table or measured value.
- Text on the main views: labels and numbers only. Definitions in hover tooltips and one collapsed help panel. No paragraph over two lines.
- Light and dark theme.
- Python tests run as `./scripts/jm pytest …`; JS tests as `module load nodejs/24.4.1 && node --test tests/js/`.
- Never write into the `--labels` directory.

## Review Focus

- Uncalibrated run (no pixel size): the µm/px toggle is hidden and px values are shown, with px units on axes.
- Run without JM2 or without a segment channel (`run --labels`): pickers list only roles present; missing previews are skipped, not errors.
- Field with zero interfaces: field view opens with an empty-state line; overview and metric files still build.
- NaN metric values (e.g. `linearity_index` of a closed loop): excluded from charts, drawn in neutral grey in the image, shown as "–" in the detail panel; never plotted as 0.
- Comparison of runs whose labels collide, or whose `field_id`s repeat across runs: duplicate labels are rejected with a clear error; repeated `field_id`s are fine because every asset path and index is namespaced by run.

---

## File Structure

```
src/junctionmapper/pipeline.py            FieldResult gains `paths` (interface_id -> pixel walk)
src/junctionmapper/report/assets.py       previews, outlines, polylines, field asset file
src/junctionmapper/report/data.py         junction/cell aggregation, metric + index files, run.json
src/junctionmapper/report/dashboard.py    build_dashboard(): copy/emit files, render index.html
src/junctionmapper/report/summary.py      QC/threshold/provenance helpers moved out of build.py
src/junctionmapper/report/templates/index.html
src/junctionmapper/report/static/app.css
src/junctionmapper/report/static/logic.js pure functions, testable in node
src/junctionmapper/report/static/app.js   DOM, views, image viewer, Plotly wiring
src/junctionmapper/report/static/plotly-cartesian.min.js   vendored, MIT
src/junctionmapper/cli.py                 run writes the dashboard; new `report` command
tests/test_paths.py  tests/test_report_assets.py  tests/test_report_data.py
tests/test_report_dashboard.py  tests/js/logic.test.js
```

`build.py` and `template.html` are deleted in Task 7.

## Shared data formats (every task reads these)

All coordinates are **preview pixels**, `[x, y]`, rounded to 0.5. `scale` = preview px per raw px.

**Field asset** — `report/runs/<run>/fields/<field_id>.js`:
```js
JM.registerField({
  run: "<run>", field_id: "<id>", width: W, height: H, scale: s,
  labels_sha256: "<hex of seg.cells.tobytes()>",            // detects stale geometry
  channels: [{role: "segment"|"JM1"|"JM2"|"nucleus", marker: "<name>", src: "data:image/jpeg;base64,..."}],
  cells: {"<cell_id>": [[x,y], ...]},                       // outer contour, closed
  junctions: {"tcj": {"<lo>-<hi>": [[[x,y],...], ...]},     // list of polylines per junction
              "dp":  {"<lo>-<hi>": [[[x,y],...], ...]}},
  sides: {"<method>": {"<lo>-<hi>": {"<role>": {"<cell_id>": {"<metric key>": [px, um|null], ...}}}}},
                                                            // each cell side's values; NaN -> null
  flags: {"<cell_id>": ["isolated", "corner_methods_disagree", ...]}   // only true flags
});
```

**Metric index** — `report/runs/<run>/metrics/<method>/_index.js`:
```js
JM.registerIndex({run, method, fields: ["<id>", ...],
  junction_field: "<b64 Uint32>", junction_lo: "<b64 Uint32>", junction_hi: "<b64 Uint32>",
  cell_field: "<b64 Uint32>", cell_id: "<b64 Uint32>"});
```

**Metric values** — `report/runs/<run>/metrics/<method>/<role>/<key>.js`:
```js
JM.registerMetric({run, method, role, key,
  junction_px: "<b64 Float32>", junction_um: "<b64 Float32>"|null,
  cell_px: "<b64 Float32>",     cell_um: "<b64 Float32>"|null});
```
Junction value = mean over that `pair_id`'s cell-side rows. Cell value = median over that cell's rows. Order matches the index. NaN stays NaN.

**Run summary** — inlined in `index.html` as `JM.registerRun({...})` per run:
`{run, label, roles: [{role, marker}], methods: ["tcj","dp"], calibrated: bool, metrics: [{key, label, unit_px, unit_um, td, help}], fields: [{field_id, n_cells, n_junctions, qc: {...}, thumb: "<data URI>"}], tiles: {cells, junctions, fields, flagged_cells}, provenance: {...}}`.

`metrics` comes from `PARAMETERS` (key, label, px, um, td). `help` is a one-line definition per metric, added to `PARAMETERS` in Task 3.

---

### Task 1: Record junction paths during measurement

**Files:**
- Modify: `src/junctionmapper/pipeline.py` (`FieldResult`, `measure_field` interface loop ~line 380)
- Test: `tests/test_paths.py`

**Interfaces:**
- Produces: `FieldResult.paths: dict[str, np.ndarray]` — `interface_id` → `(N, 2)` int array of `(row, col)`, equal to `Interface.pixels`. Default `{}` (existing constructors keep working).

- [ ] **Step 1: Write the failing tests**

```python
def test_every_interface_row_has_its_pixel_walk(setup):
    f, seg, cfg = setup
    r = measure_field(f, seg, cfg)
    assert set(r.interfaces.interface_id) == set(r.paths)
    for p in r.paths.values():
        assert p.ndim == 2 and p.shape[1] == 2 and len(p) >= 2

def test_paths_do_not_change_any_table(setup):
    f, seg, cfg = setup
    a, b = measure_field(f, seg, cfg), measure_field(f, seg, cfg)
    pd.testing.assert_frame_equal(a.interfaces, b.interfaces)
    assert len(a.paths) == len(b.paths)

def test_path_contour_matches_the_table(setup):
    f, seg, cfg = setup
    r = measure_field(f, seg, cfg)
    row = r.interfaces.iloc[0]
    assert interface_contour(r.paths[row.interface_id]) == pytest.approx(row.interface_contour_px)
```
Use the `setup` fixture pattern from `tests/test_pipeline.py` (`build() + (make_cfg(),)`).

- [ ] **Step 2: Run** `./scripts/jm pytest tests/test_paths.py -q` — Expected: FAIL (`FieldResult` has no `paths`).
- [ ] **Step 3: Implement.** Add `paths: dict = field(default_factory=dict)` to `FieldResult`. In the interface loop, store `pix` under `_interface_id(...)` once per interface (not per role).
- [ ] **Step 4: Run** the test file and `tests/test_pipeline.py` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(pipeline): keep each interface's pixel walk on FieldResult`

---

### Task 2: Field asset files

**Files:**
- Create: `src/junctionmapper/report/assets.py`
- Test: `tests/test_report_assets.py`

**Interfaces:**
- Consumes: `FieldResult.paths` (Task 1); `Field` (planes, markers, saturation); `Segmentation.cells`.
- Produces:
  - `preview(plane: np.ndarray, scale: float) -> str` — data URI (JPEG, quality 85). Display stretch: 0.5–99.5 percentile of finite values → 0–255. Constant or all-NaN plane → mid-grey image, no error.
  - `preview_scale(shape: tuple[int, int], max_side: int = 1500) -> float` — `min(1, max_side / max(shape))`.
  - `simplify(path: np.ndarray, tol: float) -> np.ndarray` — Ramer–Douglas–Peucker on an open polyline, endpoints kept.
  - `to_preview(path_rc: np.ndarray, scale: float) -> list[list[float]]` — `(row, col)` → `[x, y]` in preview px, rounded to 0.5.
  - `cell_outlines(labels: np.ndarray, scale: float, tol: float = 1.0) -> dict[str, list]` — outer contour per non-zero label.
  - `junction_polylines(result: FieldResult, scale: float, tol: float = 1.0) -> dict[str, dict[str, list]]` — `{"tcj"|"dp": {"<lo>-<hi>": [polyline, ...]}}`, grouped by `pair_id`.
  - `write_field_asset(field: Field, seg: Segmentation, result: FieldResult, run: str, out_dir: Path) -> Path` — writes `out_dir/<field_id>.js` in the shared format. Channels in config order; a role whose plane is absent is skipped.

- [ ] **Step 1: Write the failing tests**

```python
def test_preview_scale_caps_the_long_side():
    assert preview_scale((2500, 2984)) == pytest.approx(1500 / 2984)
    assert preview_scale((400, 300)) == 1.0

def test_preview_is_a_jpeg_data_uri_of_the_scaled_size():
    uri = preview(np.random.default_rng(0).random((300, 200)) * 255, 0.5)
    img = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert uri.startswith("data:image/jpeg;base64,") and img.size == (100, 150)

def test_preview_of_a_flat_or_nan_plane_does_not_fail():
    preview(np.zeros((10, 10)), 1.0); preview(np.full((10, 10), np.nan), 1.0)

def test_simplify_keeps_endpoints_and_drops_collinear_points():
    line = np.array([[0, i] for i in range(50)])
    assert simplify(line, 1.0).tolist() == [[0, 0], [0, 49]]

def test_to_preview_swaps_to_xy_and_scales():
    assert to_preview(np.array([[10, 20]]), 0.5) == [[10.0, 5.0]]

def test_junction_polylines_group_cell_sides_by_pair(setup):   # fixture: tests/fixtures13
    f, seg, cfg = setup
    j = junction_polylines(measure_field(f, seg, cfg), 1.0)
    pairs = {p.split("|")[-1] for p in measure_field(f, seg, cfg).interfaces.pair_id}
    assert set(j["tcj"]) <= pairs and set(j) == {"tcj", "dp"}

def test_field_asset_registers_one_field_with_every_configured_channel(tmp_path, setup):
    f, seg, cfg = setup
    p = write_field_asset(f, seg, measure_field(f, seg, cfg), "r1", tmp_path)
    d = _load(p, "registerField")   # helper: check the JM.registerField( ... ); wrapper, json-parse the inside
    assert d["run"] == "r1" and {c["role"] for c in d["channels"]} == {r.value for r in f.planes}
    assert d["labels_sha256"] == hashlib.sha256(seg.cells.tobytes()).hexdigest()
```

- [ ] **Step 2: Run** `./scripts/jm pytest tests/test_report_assets.py -q` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `assets.py`. Resize with `PIL.Image.resize(..., Image.BILINEAR)`. Contours: `skimage.measure.find_contours(labels == cid, 0.5)`, keep the longest. JSON via `json.dumps(..., separators=(",", ":"))`.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(report): per-field asset files with previews, outlines and junction paths`

---

### Task 3: Junction/cell aggregation, metric files, run.json

**Files:**
- Create: `src/junctionmapper/report/data.py`
- Modify: `src/junctionmapper/report/build.py` (`PARAMETERS`: add `"help"` per entry)
- Test: `tests/test_report_data.py`

**Interfaces:**
- Consumes: tables as written by `write_tables` (interfaces, cells, flags); `PARAMETERS`.
- Produces:
  - `junction_table(ifc: pd.DataFrame, method: str) -> pd.DataFrame` — one row per (`role`, `pair_id`); columns `field_id, lo, hi, role` + every `PARAMETERS` px/um column; value = mean over that pair's rows (NaN-skipping; all-NaN → NaN).
  - `cell_table(ifc: pd.DataFrame, method: str) -> pd.DataFrame` — one row per (`role`, `field_id`, `cell_id`); value = median over that cell's rows.
  - `b64_f32(a) -> str`, `b64_u32(a) -> str` — little-endian base64.
  - `write_metric_files(ifc: pd.DataFrame, run: str, out_dir: Path, calibrated: bool) -> None` — writes `_index.js` per method and one file per (method, role, metric) in the shared format. Index order: sorted by `field_id`, then `lo`, `hi` / `cell_id`; identical across all metric files of a method.
  - `write_run_json(out: Path, cfg: RunConfig, prov: dict, fields: list[str]) -> Path` — `{"config": ..., "provenance": ..., "fields": [...]}`.
  - `run_summary(run_dir: Path, run: str, label: str) -> dict` — the run summary object (shared format), read from `run.json`, the parquet tables and the field asset thumbnails.

- [ ] **Step 1: Write the failing tests**

```python
def test_junction_value_is_the_mean_of_its_cell_sides():
    ifc = _two_sided(pair="f|tcj|2-4", values=(10.0, 30.0))      # helper builds 2 rows, 1 pair
    j = junction_table(ifc, "tcj")
    assert len(j) == 1 and j.iloc[0].band_mean_intensity == 20.0 and (j.iloc[0].lo, j.iloc[0].hi) == (2, 4)

def test_cell_value_is_the_median_of_its_rows():
    ifc = _cell_rows(cell=3, values=(1.0, 2.0, 9.0))
    assert cell_table(ifc, "tcj").iloc[0].band_mean_intensity == 2.0

def test_nan_stays_nan_and_is_not_zero():
    ifc = _two_sided(pair="f|tcj|2-4", values=(np.nan, np.nan))
    assert np.isnan(junction_table(ifc, "tcj").iloc[0].band_mean_intensity)

def test_metric_files_align_with_the_index(tmp_path, setup_tables):
    write_metric_files(setup_tables.interfaces, "r1", tmp_path, calibrated=True)
    idx = _load(tmp_path / "tcj/_index.js"); m = _load(tmp_path / "tcj/JM1/band_mean_intensity.js")
    assert len(_f32(m["junction_px"])) == len(_u32(idx["junction_lo"]))
    assert len(_f32(m["cell_px"])) == len(_u32(idx["cell_id"]))

def test_uncalibrated_run_writes_no_um_arrays(tmp_path, setup_tables):
    write_metric_files(setup_tables.interfaces, "r1", tmp_path, calibrated=False)
    assert _load(tmp_path / "tcj/JM1/interface_contour.js")["junction_um"] is None

def test_run_without_jm2_writes_only_jm1(tmp_path, setup_tables_jm1_only):
    write_metric_files(setup_tables_jm1_only.interfaces, "r1", tmp_path, calibrated=True)
    assert not (tmp_path / "tcj/JM2").exists()

def test_field_with_no_interfaces_still_builds_a_summary(tmp_path): ...
    # a run dir whose interfaces table is empty: run_summary returns n_junctions == 0 for it

def test_every_parameter_has_a_one_line_help():
    assert all(p["help"] and "\n" not in p["help"] and len(p["help"]) <= 120 for p in PARAMETERS)
```
`_load` strips the `JM.registerX(` wrapper and parses JSON; `_f32`/`_u32` decode base64 with `np.frombuffer`. Put these helpers in the test file.

- [ ] **Step 2: Run** `./scripts/jm pytest tests/test_report_data.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** `data.py`; add `help` strings to `PARAMETERS`, one line each, taken from the formulas block in `tables.py`.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(report): junction- and cell-level metric files and run.json`

---

### Task 4: Dashboard shell and overview

**Files:**
- Create: `src/junctionmapper/report/dashboard.py`, `templates/index.html`, `static/app.css`, `static/logic.js`, `static/app.js`, `static/plotly-cartesian.min.js` (+ `static/LICENSE-plotly.txt`)
- Test: `tests/test_report_dashboard.py`, `tests/js/logic.test.js`

**Interfaces:**
- Consumes: `run_summary` (Task 3); the files from Tasks 2–3.
- Produces:
  - `build_dashboard(runs: list[tuple[Path, str]], out: Path) -> Path` — `runs` = (run directory, label). Copies each run's `report/runs/<run>/` tree to `out/runs/<run>/` (skips the copy when `out` is that run's own `report/`), inlines `app.css`, `logic.js`, `app.js`, Plotly and one `JM.registerRun(...)` per run into `out/index.html`. Raises `ConfigError` on duplicate labels. Run ids are `r1`, `r2`, … in argument order.
  - `logic.js` (browser global `JMLogic`, and `module.exports` when `module` exists):
    - `decodeF32(b64) -> Float32Array`, `decodeU32(b64) -> Uint32Array`
    - `finite(values) -> {values, index}` — drops NaN, keeps original positions
    - `colourFor(v, lo, hi) -> "rgb(...)"` — sequential single-hue ramp; NaN → neutral grey `"rgb(150,150,150)"`
    - `quantileRange(values, q=[0.02,0.98]) -> [lo, hi]` over finite values
    - `junctionKey(lo, hi) -> "lo-hi"`
    - `unitFor(metric, calibrated, useUm) -> string`
  - Defaults: marker JM1, metric `band_mean_intensity`, corner method `tcj`, units µm when calibrated, dot = junction, QC flag toggles all off.
  - Overview view in `app.js`: header (run picker, marker, metric, corner method, unit toggle — hidden when not calibrated, dot = junction/cell switch), tiles (cells, junctions, fields, flagged cells), field grid (thumbnail + QC badge, click → field view), pooled distribution (Plotly violin with points, one trace per run). Metric/index files load with `loadScript(src)` (inject `<script>`, resolve on `JM.register*`).

Get Plotly: `npm pack plotly.js-cartesian-dist-min@2.35.2` (host, `module load nodejs/24.4.1`), copy `plotly-cartesian.min.js` and its LICENSE into `static/`.

- [ ] **Step 1: Write the failing tests**

```js
// tests/js/logic.test.js
const L = require("../../src/junctionmapper/report/static/logic.js");
test("decodeF32 round-trips", () => { assert.deepEqual([...L.decodeF32("AACAPwAAAEA=")], [1, 2]); });
test("NaN is grey, never a ramp colour", () => { assert.equal(L.colourFor(NaN, 0, 1), "rgb(150,150,150)"); });
test("finite drops NaN and keeps positions", () => {
  assert.deepEqual(L.finite([1, NaN, 3]), {values: [1, 3], index: [0, 2]}); });
test("unit hides µm when uncalibrated", () => {
  assert.equal(L.unitFor({unit_px: "px", unit_um: "µm"}, false, true), "px"); });
```
```python
def test_index_html_references_nothing_external(tmp_path, built_run):
    html = build_dashboard([(built_run, "A")], tmp_path / "rep").read_text()
    assert not re.search(r"(src|href)\s*=\s*\"https?://", html) and "fetch(" not in html

def test_one_register_run_per_run_and_runs_are_namespaced(tmp_path, built_run, built_run2):
    html = build_dashboard([(built_run, "A"), (built_run2, "B")], tmp_path / "rep").read_text()
    assert html.count("JM.registerRun(") == 2
    assert (tmp_path / "rep/runs/r1/fields").is_dir() and (tmp_path / "rep/runs/r2/fields").is_dir()

def test_duplicate_labels_are_rejected(tmp_path, built_run):
    with pytest.raises(ConfigError, match="duplicate run label"):
        build_dashboard([(built_run, "A"), (built_run, "A")], tmp_path / "rep")
```
`built_run` fixture: run `measure_field` on `tests/fixtures13`, then `write_tables`, `write_run_json`, `write_metric_files`, `write_field_asset` into `tmp_path`.

- [ ] **Step 2: Run** both suites — Expected: FAIL.
- [ ] **Step 3: Implement.** Style: neutral greys with one blue accent, Inter-like system font stack (no web fonts), 8 px spacing grid, cards only for tiles and the chart/image panels. Tooltips on every control and metric name use `help`. Light/dark via `prefers-color-scheme` plus a toggle.
- [ ] **Step 4: Run** both suites — Expected: PASS. Then open `index.html` from a built fixture run in a browser: overview renders, metric change reloads the chart.
- [ ] **Step 5: Commit** `feat(report): dashboard shell and overview`

---

### Task 5: Field view — image, junctions, linked chart, detail panel

**Files:**
- Modify: `static/app.js`, `static/app.css`, `static/logic.js`
- Test: `tests/js/logic.test.js`

**Interfaces:**
- Consumes: field asset, index and metric files (shared formats); `JMLogic` (Task 4).
- Produces in `logic.js`:
  - `fieldSlice(index, fieldPos) -> {junctions: [{i, key}], cells: [{i, cell_id}]}` — positions in the index arrays belonging to one field.
  - `hitTest(polylines, x, y, tol) -> key|null` — nearest junction within `tol` preview px.
  - `cellAt(outlines, x, y) -> cell_id|null` — point-in-polygon.

Field view behaviour:
- Canvas viewer: channel layers composited additively, each with on/off and brightness; mouse-wheel zoom, drag pan, double-click reset.
- Junction lines drawn with `Path2D`, coloured by the selected metric via `colourFor` and `quantileRange` over the field's finite values; colour bar with min/max labels.
- Cell outlines thin and translucent; flagged cells marked when their flag toggle is on.
- Click junction → detail panel: every metric for that junction (both cell sides listed), units per toggle. Click cell (dot = cell mode) → its junctions highlighted.
- Linked chart: Plotly strip/violin of this field's values; lasso or box select highlights the matching junctions; selecting in the image highlights chart points.
- Empty state: a field with no junctions shows the image and one line, "No junctions measured in this field."

- [ ] **Step 1: Write the failing tests**

```js
test("hitTest finds the junction under the cursor", () => {
  const p = {"2-4": [[[0, 0], [10, 0]]], "4-5": [[[0, 10], [10, 10]]]};
  assert.equal(L.hitTest(p, 5, 1, 3), "2-4"); assert.equal(L.hitTest(p, 5, 5, 3), null); });
test("cellAt uses point-in-polygon", () => {
  assert.equal(L.cellAt({"7": [[0,0],[10,0],[10,10],[0,10]]}, 5, 5), "7"); });
test("fieldSlice picks only that field's rows", () => {
  const idx = {junction_field: [0, 0, 1], junction_lo: [2, 3, 2], junction_hi: [4, 5, 4],
               cell_field: [0, 1], cell_id: [2, 2]};
  assert.deepEqual(L.fieldSlice(idx, 1).junctions, [{i: 2, key: "2-4"}]); });
```

- [ ] **Step 2: Run** `node --test tests/js/` — Expected: FAIL.
- [ ] **Step 3: Implement** the three functions and the field view.
- [ ] **Step 4: Run** — Expected: PASS. Browser check on the fixture run and on one real AO0025 run (e.g. `jm-all/P2_B/out_abs/T47D_CDH1_KO` re-run with this branch): click a junction, lasso in the chart, toggle channels.
- [ ] **Step 5: Commit** `feat(report): field view with linked image, junctions and charts`

---

### Task 6: `run` writes the dashboard; `report` command; comparison view

**Files:**
- Modify: `src/junctionmapper/cli.py` (`run`; new `report` command)
- Modify: `static/app.js` (comparison view)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - `run`: inside `work()`, call `write_field_asset(field, seg, result, "r1", out / "report/runs/r1/fields")` (the field planes are only in memory there). After `write_tables`: `write_run_json(out, cfg, prov, fields)`, `write_metric_files(interfaces, "r1", out / "report/runs/r1/metrics", calibrated)`, `build_dashboard([(out, out.name)], out / "report")`. Message: `wrote tables and report/index.html to OUT`.
  - `junctionmapper report RUN_DIR... -o OUT [--label NAME ...]` — labels default to run directory names; count of `--label` must equal count of runs. Exit 1 with a clear message if a RUN_DIR lacks `run.json` ("made before the interactive report; re-run with `run --labels`").
  - Comparison view (shown when more than one run): one violin per run for the chosen marker and metric, junction or cell dots; clicking a point opens that field in that run's field view with the junction selected.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_writes_the_dashboard_and_no_report_html(tmp_path, labelled_input): ...
    # invoke `run --labels`; assert (out/"report/index.html").exists(), (out/"run.json").exists(),
    # one fields/*.js per field, and not (out/"report.html").exists()

def test_report_combines_runs(tmp_path, two_runs):
    res = CliRunner().invoke(app, ["report", *map(str, two_runs), "-o", str(tmp_path / "cmp")])
    assert res.exit_code == 0 and (tmp_path / "cmp/runs/r2/fields").is_dir()

def test_report_refuses_a_run_without_run_json(tmp_path):
    (tmp_path / "old").mkdir()
    res = CliRunner().invoke(app, ["report", str(tmp_path / "old"), "-o", str(tmp_path / "x")])
    assert res.exit_code == 1 and "re-run with `run --labels`" in res.output

def test_label_count_must_match(tmp_path, two_runs):
    res = CliRunner().invoke(app, ["report", *map(str, two_runs), "--label", "A", "-o", str(tmp_path / "x")])
    assert res.exit_code != 0
```

- [ ] **Step 2: Run** `./scripts/jm pytest tests/test_cli.py -q` — Expected: new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite `./scripts/jm pytest -q` — Expected: PASS except old report tests (removed in Task 7).
- [ ] **Step 5: Commit** `feat(cli): run writes the interactive report; add report command`

---

### Task 7: Provenance, help panel, retire the old report

**Files:**
- Create: `src/junctionmapper/report/summary.py`
- Delete: `src/junctionmapper/report/build.py`, `src/junctionmapper/report/template.html`
- Modify: `static/app.js` (provenance view, help panel), `tests/test_report.py`, `README.md`, `docs/decisions.md`

**Interfaces:**
- Produces: `summary.py` holds `PARAMETERS` and the QC/threshold/provenance helpers moved from `build.py`, renamed without the leading underscore: `qc_counts`, `qc`, `threshold_info`, `run_roles`, `provenance_rows`. `data.run_summary` uses them for field QC badges, tiles and the provenance view.

- [ ] **Step 1:** Move the helpers; retarget `tests/test_report.py` imports to `summary`; delete tests that assert on old HTML text or matplotlib figures. Run `./scripts/jm pytest -q` — Expected: PASS.
- [ ] **Step 2:** Provenance view: key–value table (inputs, model, parameters, thresholds per role, versions, code hash), searchable. Help panel: collapsed by default, lists each metric's `label` and `help`.
- [ ] **Step 3:** README: replace the report section with a short block: what `report/index.html` shows, how to open it, the `report` command. `decisions.md`: one entry for "dashboard replaces report.html" (why: the static report showed no cells; the user asked for an interactive dashboard).
- [ ] **Step 4: Run** `./scripts/jm pytest -q` and `node --test tests/js/` — Expected: all PASS. Browser check: provenance and help panel open; light and dark themes legible.
- [ ] **Step 5: Commit** `feat(report): provenance and help; retire the static report`
