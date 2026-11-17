# Junction Detail Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the field view's numeric detail table with a junction close-up, an intensity profile along the junction with threshold lines, and per-metric "where it sits" strips.

**Architecture:** junctionmapper samples each junction's marker profile at full resolution while it writes the field asset (new `profiles` and `thresholds` keys). The browser draws the close-up from the preview planes it already decodes, the profile from the asset, and the strips from the metric files it already loads.

**Tech Stack:** Python (numpy, scipy.ndimage), vanilla JS, inline SVG for the small figures (no new library).

**Spec:** `docs/superpowers/specs/2026-11-16-junction-detail-figures-design.md` (builds on `2026-11-16-interactive-report-design.md`).

## Global Constraints

- Opens offline from `file://`; no `fetch()`; nothing external; generic to any panel.
- Report generation never changes any table or measured value. Profiles go in the field asset only.
- Never write into the `--labels` directory.
- Profile: one per junction (`pair_id`) and per channel role among JM1, JM2 and segment (whichever exist), along the lower cell id's interface pixel walk; value = mean marker intensity over band pixels within the band half-width; ≤ 200 points; x in µm (px if uncalibrated).
- Threshold per role per field = the applied threshold (`resolve_thresholds`), written beside the profiles.
- Close-up = junction bbox + 40% margin, at least 48 preview px, one tile per channel present.
- Strips default to the field; one-click toggle to the whole run. No numeric table; values on hover.
- No "·" (middle dot) anywhere in user-facing text. Separate items with commas, "in", line breaks or layout instead (e.g. "JM1, tcj, per junction"; "32 cells, 109 junctions"). Task 2 also replaces the existing ones in app.js, index.html and summary.py, and adds a test that the built index.html and app.js contain no "·".
- Channel labels everywhere show the marker name first and its role second, from the config: e.g. "β-catenin (outline)", "P-cadherin (JM1)", "E-cadherin (JM2)", "Hoechst (nucleus)". The segment role is shown as "outline" (the channel the cells were outlined on), never "segment". Falls back to the role alone when no marker name is configured.
- Python tests: `./scripts/jm pytest …`; JS: `source /etc/profile.d/lmod.sh; module load nodejs/24.4.1; node --test tests/js/*.test.js`.

## Review Focus

- Junction shorter than the band (2–3 path pixels): profile still has ≥ 2 points, no division by zero.
- Junction at the image edge: close-up window clamps to the image; band window clamps at borders.
- Role with no threshold (e.g. JM2 absent, or threshold NaN): no threshold line, no error.
- Metric whose values are all NaN in the field: strip shows "–" and no percentile.
- Older field assets without `profiles`: panel hides the profile figure instead of failing.

---

## Field asset additions (shared format)

```js
profiles:   {"<method>": {"<lo-hi>": {"<role>": {"x": [..], "y": [..]}}}},   // x µm or px, y raw intensity; NaN → null
thresholds: {"<role>": t | null},
profile_unit: "µm" | "px"
```

### Task 1: Junction intensity profiles in the field asset

**Files:**
- Modify: `src/junctionmapper/report/assets.py` (new `junction_profile`, `junction_profiles`; `write_field_asset` gains `thresholds`)
- Modify: `src/junctionmapper/cli.py` (`work()` passes the resolved thresholds to `_write_asset` → `write_field_asset`)
- Modify: `docs/superpowers/plans/2026-11-16-interactive-report.md` (Shared data formats: field asset block)
- Test: `tests/test_report_assets.py`

**Interfaces:**
- Consumes: `FieldResult.paths`, `result.interfaces` (`interface_id, pair_id, corner_method, cell_id, role`), `result.cells.dilation_cycles`, `Field.planes`, `Field.pixel_size_um`, `resolve_thresholds(field, cfg, run_t) -> {Role: float}`.
- Produces:
  - `junction_profile(path_rc: np.ndarray, image: np.ndarray, band: np.ndarray, radius: int, px_um: float | None, max_points: int = 200) -> dict` — `{"x": list[float], "y": list[float|None]}`. y[i] = mean of `image` over `band` pixels in the `(2r+1)²` window centred on path pixel i (clamped at borders; None if no band pixel). x = cumulative path length (diagonal steps √2) × `px_um` if given, else px. Resample to ≤ `max_points` by linear interpolation on x; keep both endpoints; length-1 path → x [0, 0], y [v, v].
  - `junction_profiles(field: Field, result: FieldResult) -> dict` — `{method: {"lo-hi": {role: profile}}}` using `mi.dilated_band(path, shape, cycles)` and `radius = cycles`, for the side whose `cell_id` is the lower id.
  - `write_field_asset(field, seg, result, run, out_dir, thresholds: dict[str, float] | None = None)` — adds `profiles`, `thresholds` (role value → float or null), `profile_unit`.

- [ ] **Step 1: Write the failing tests**

```python
def test_profile_of_a_uniform_band_is_flat_at_the_image_value():
    img = np.full((20, 20), 7.0); path = np.array([[10, c] for c in range(3, 15)])
    band = mi.dilated_band(path, img.shape, 2)
    p = junction_profile(path, img, band, 2, None)
    assert p["y"] == pytest.approx([7.0] * len(p["y"])) and p["x"][0] == 0 and p["x"][-1] == pytest.approx(11)

def test_profile_x_is_in_micrometres_when_calibrated():
    path = np.array([[5, c] for c in range(10)]); img = np.ones((12, 12))
    p = junction_profile(path, img, mi.dilated_band(path, img.shape, 1), 1, 0.5)
    assert p["x"][-1] == pytest.approx(4.5)

def test_profile_is_resampled_to_at_most_200_points_keeping_the_ends():
    path = np.array([[1, c] for c in range(600)]); img = np.tile(np.arange(600.0), (3, 1))
    p = junction_profile(path, img, mi.dilated_band(path, img.shape, 0), 0, None)
    assert len(p["x"]) == 200 and p["y"][0] == 0 and p["y"][-1] == 599

def test_a_two_pixel_junction_still_has_a_profile():
    path = np.array([[4, 4], [4, 5]]); img = np.ones((9, 9))
    assert len(junction_profile(path, img, mi.dilated_band(path, img.shape, 2), 2, None)["y"]) >= 2

def test_profile_sees_a_gap_in_the_marker(setup):            # fixture from tests.fixtures13
    ...  # zero the JM1 plane over the middle third of one junction's path; its profile's middle
         # third is ≤ 10% of its ends' mean

def test_field_asset_carries_profiles_thresholds_and_unit(tmp_path, setup):
    f, seg, cfg = setup
    d = _load(write_field_asset(f, seg, measure_field(f, seg, cfg), "r1", tmp_path,
                                thresholds={"JM1": 12.5}), "Field")
    key = next(iter(d["junctions"]["tcj"]))
    assert set(d["profiles"]["tcj"][key]) >= {"JM1"} and d["thresholds"] == {"JM1": 12.5}
    assert d["profile_unit"] in ("µm", "px")
```

- [ ] **Step 2: Run** `./scripts/jm pytest tests/test_report_assets.py -q` — Expected: FAIL (`junction_profile` missing).
- [ ] **Step 3: Implement** in `assets.py`; window mean via `scipy.ndimage.uniform_filter` on `image*band` and `band` (ratio, guarded) rather than a Python loop. Pass thresholds from `cli.work()`; NaN threshold → null.
- [ ] **Step 4: Run** the assets, cli and dashboard test files — Expected: PASS. Then full suite.
- [ ] **Step 5: Commit** `feat(report): junction intensity profiles and thresholds in field assets`

---

### Task 2: Detail panel figures

**Files:**
- Modify: `src/junctionmapper/report/static/logic.js`, `app.js`, `app.css`
- Test: `tests/js/logic.test.js`

**Interfaces:**
- Consumes: field asset (`profiles`, `thresholds`, `profile_unit`, `junctions`, `cells`, `width`, `height`), decoded preview planes, `L.metricValues` / `L.wantsUm`, the field and run metric arrays the panel already loads, `sides`.
- Produces in `logic.js`:
  - `cropWindow(polylines, width, height, margin = 0.4, min = 48) -> {x, y, w, h}` — square window around the bbox of all polylines, padded by `margin × max(bboxW, bboxH)`, at least `min`, clamped inside the image.
  - `percentileOf(values, v) -> number | null` — share of finite values ≤ v, ×100, rounded; null when v is not finite or no finite values.
  - `profileScale(profile, thresholds, role) -> {xmax, ymax}` — y range covers the profile and the threshold.
  - `channelLabel(role, marker) -> string` — `"P-cadherin (JM1)"`; role `segment` reads `outline`; no marker → role alone (`"JM1"`, `"outline"`). Used by the channel toggles, close-up tiles, profile legend, marker picker and QC text.

Panel layout, top to bottom (no numeric table):
1. Header: junction key, marker, method; two cell chips with flags (as now).
2. **Close-up:** tiles in this order: JM1, JM2 (if present), then **Merge** (JM1 + JM2 composited in their channel colours, with the outline channel off), then the outline and nucleus channels. Each tile is labelled with marker name and role. Tiles are, each a canvas crop of that channel's decoded plane at `cropWindow`, ~140 px wide, junction polyline drawn in the accent colour, both cells' outlines faint.
3. **Intensity along the junction:** inline SVG line chart; JM1 and JM2 always both drawn when present (legend uses marker name and role), plus the outline channel's profile as a thin grey line when present in the run's fixed marker colours; dashed horizontal line at each role's threshold, labelled "threshold"; x axis in `profile_unit`; hover shows x and y. Hidden when the asset has no profile for this junction.
4. **Where it sits:** one row per metric (label with help tooltip, strip, percentile). Strip = faint dots of every finite junction value in the field (toggle "Field | Run"), this junction's value as an accent marker. Hover on a row: cell A / cell B / mean with unit. All-NaN metric → "–".

- [ ] **Step 1: Write the failing tests**

```js
test("cropWindow pads, enforces a minimum and clamps to the image", () => {
  assert.deepEqual(L.cropWindow({"2-4": [[[10, 10], [20, 10]]]}, 100, 100), {x: 0, y: 0, w: 48, h: 48});
  const w = L.cropWindow({"2-4": [[[500, 500], [600, 520]]]}, 1000, 1000);
  assert.ok(w.w >= 140 && w.x + w.w <= 1000 && w.y >= 0); });
test("percentileOf ignores NaN and is null for NaN input", () => {
  assert.equal(L.percentileOf([1, 2, NaN, 3, 4], 3), 75);
  assert.equal(L.percentileOf([1, 2], NaN), null); });
test("profileScale includes the threshold", () => {
  assert.equal(L.profileScale({x: [0, 5], y: [1, 2]}, {JM1: 10}, "JM1").ymax >= 10, true); });
test("channelLabel puts the marker first and renames segment", () => {
  assert.equal(L.channelLabel("JM1", "P-cadherin"), "P-cadherin (JM1)");
  assert.equal(L.channelLabel("segment", "beta-catenin"), "beta-catenin (outline)");
  assert.equal(L.channelLabel("segment", ""), "outline"); });
test("profileScale ignores a missing threshold", () => {
  assert.equal(L.profileScale({x: [0, 5], y: [1, 2]}, {JM1: null}, "JM1").ymax < 10, true); });
```

```python
def test_dashboard_text_has_no_middle_dot(tmp_path, built_run):   # tests/test_report_dashboard.py
    html = build_dashboard([(built_run, "A")], tmp_path / "rep").read_text()
    assert "\u00b7" not in html.replace(_vendored_plotly(), "")
```

- [ ] **Step 2: Run** the node tests — Expected: FAIL.
- [ ] **Step 3: Implement** the functions and the panel.
- [ ] **Step 4: Run** node tests and `node --check` — PASS. Rebuild the real-size comparison (P2_B HCC1500 vs MCF7_CDH1_KO, 2 fields each, `run --labels … --no-gpu` into scratch, then `report`) and screenshot in headless Chromium: panel with a junction selected, light and dark, plus phone width. Look at every screenshot; the profile must visibly dip where the close-up shows a gap.
- [ ] **Step 5: Commit** `feat(report): junction close-up, intensity profile and position strips in the detail panel`
