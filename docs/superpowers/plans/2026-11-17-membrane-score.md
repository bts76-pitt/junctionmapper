# Membrane Score Implementation Plan

> **Status (2026-11-20): implemented.** Later changes: `cutoff` defaults to none (`acd072f`), so `membrane_like` is null unless set. Too-dim reads the ridge pixels, and `min_ridge_intensity` was added. Where this plan says `cutoff: 1.5` or "False when too dim", the code and README win.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `run` writes an outline-free, per-patch "membrane-like" score for every stain channel, plus a per-field summary.

**Architecture:** A new pure module `measure/membrane.py` scores one field from its raw planes (no labels). `cli.run`'s `work()` calls it before segmentation and attaches two frames to `FieldResult`; `tables.py` registers them so `write_tables` writes them like the other five.

**Tech Stack:** Python 3.11, numpy, scipy.ndimage, scikit-image 0.26 (`skimage.filters.sato`, `threshold_otsu`), pandas, pydantic, typer. Tests: pytest via `./scripts/jm pytest`.

**Spec:** `docs/superpowers/specs/2026-11-17-membrane-score-design.md`

## Global Constraints

- Generic: no marker names, line names or biology in code or defaults (test fixtures may name markers).
- Channels scored: every configured role except `nucleus`.
- Lengths in µm, converted with `field.pixel_size_um`.
- Defaults: `line_width_um: [0.3, 0.6]`, `patch_um: 20`, `region_um: 6`, `ridge_fraction: 0.1`, `cutoff: 1.5`.
- Too dim: `(patch cell-region mean - background) < 3 * noise`; `background` = field 1st percentile of the channel; `noise` = MAD of the field's high-pass residual.
- Baseline re-runs the filter on one seeded shuffle of the patch's cell-region pixels; seed derived from `field_id`, role and patch (stable across processes: `zlib.crc32`, not `hash`).
- Region = Otsu on the smoothed raw nucleus plane, grown by `region_um`; never from label files, also under `--labels`.
- Cost: at most about 10 s per field per scored channel at 2984 x 2500, both filter passes.
- Field-atomic: zero cells is not a failure; existing tables and metrics unchanged.
- Never write into `--labels`. Never `rm -rf`. Never `git add -A`. Python only via `./scripts/jm`.
- No middle-dot separator (U+00B7) anywhere. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A field smaller than one patch (the 40 x 40 test fixtures): no crash; one clipped patch or none, field row with `n_patches` 0 and NaN `median_score` when none. Test in Task 2.
2. Coarse pixels where the line width is under 1 px (e.g. 1 µm/px): filter scale clamped to 1 px, no crash, scores finite. Test in Task 2.
3. A constant channel (all 0, or all 255): `noise` 0, every patch too dim, no division warnings, no NaN `membrane_like`. Test in Task 2.
4. NaN pixels in a float source plane: excluded from means, filter input and shuffle; scores finite. Test in Task 2.
5. A config with only JM1 + nucleus (no JM2, no segment): only JM1 rows. Test in Task 3.

---

### Task 1: Config block and table plumbing

**Files:**
- Modify: `src/junctionmapper/config.py` (new model above `RunConfig`; new field on `RunConfig`)
- Modify: `src/junctionmapper/tables.py` (two column lists, `SCHEMAS`, `_BOOL`, `_INT`)
- Modify: `src/junctionmapper/pipeline.py` (`FieldResult`)
- Test: `tests/test_config.py`, `tests/test_tables.py`

**Interfaces:**
- Produces: `MembraneScoreConfig` (pydantic, `extra="forbid"`) with `line_width_um: tuple[float, float] = (0.3, 0.6)`, `patch_um: float = 20.0`, `region_um: float = 6.0`, `ridge_fraction: float = 0.1`, `cutoff: float = 1.5`; `RunConfig.membrane_score: MembraneScoreConfig = MembraneScoreConfig()`.
- Produces: `MEMBRANE_PATCH_COLUMNS = ["field_id", "role", "marker", "patch_row", "patch_col", "y0_px", "x0_px", "size_px", "region_fraction", "mean_intensity", "background", "noise", "too_dim", "contrast", "baseline", "score", "membrane_like"]`, `MEMBRANE_FIELD_COLUMNS = ["field_id", "role", "marker", "n_patches", "n_scored", "n_too_dim", "median_score", "membrane_like_fraction"]`; `SCHEMAS` gains `"membrane_patches"` and `"membrane_fields"` after `"nuclei_distances"`.
- Produces: `FieldResult.membrane_patches` and `FieldResult.membrane_fields`, declared after `nuclei_distances` and before `paths`, defaulting to `make_frame(<name>, [])`, so positional `FieldResult(*(make_frame(n, []) for n in SCHEMAS))` still works.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
def test_membrane_score_defaults():
    m = make_cfg().membrane_score
    assert m.line_width_um == (0.3, 0.6) and m.patch_um == 20 and m.region_um == 6
    assert m.ridge_fraction == 0.1 and m.cutoff == 1.5

@pytest.mark.parametrize("bad", [
    {"line_width_um": [0.6, 0.3]}, {"line_width_um": [0, 0.5]}, {"patch_um": 0},
    {"region_um": -1}, {"ridge_fraction": 0}, {"ridge_fraction": 0.6}, {"cutoff": 1.0},
    {"typo": 1}])
def test_membrane_score_rejects_bad_values(bad):
    with pytest.raises((ValueError, ConfigError)):
        make_cfg(membrane_score=bad)

# tests/test_tables.py
def test_membrane_tables_are_written_even_when_empty(tmp_path):
    empty = FieldResult(*(make_frame(n, []) for n in SCHEMAS))
    write_tables([empty], tmp_path)
    for name in ("membrane_patches", "membrane_fields"):
        assert list(pd.read_parquet(tmp_path / f"{name}.parquet").columns) == SCHEMAS[name]
    assert make_frame("membrane_patches", []).dtypes["too_dim"] == bool
```

- [ ] **Step 2: Run them to verify they fail**

Run: `./scripts/jm pytest tests/test_config.py tests/test_tables.py -q -k membrane`
Expected: FAIL (`membrane_score` unknown / `KeyError: 'membrane_patches'`).

- [ ] **Step 3: Implement**

Validation: `0 < lo <= hi`, `patch_um > 0`, `region_um >= 0`, `0 < ridge_fraction <= 0.5`, `cutoff > 1`. Add `too_dim`, `membrane_like` to `_BOOL`; `patch_row, patch_col, y0_px, x0_px, size_px, n_patches, n_scored, n_too_dim` to `_INT`.

- [ ] **Step 4: Run the full suite** (existing tests iterate `SCHEMAS`)

Run: `./scripts/jm pytest -q`
Expected: all pass, 1 skipped (java).

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/config.py src/junctionmapper/tables.py src/junctionmapper/pipeline.py tests/test_config.py tests/test_tables.py
git commit -m "feat(membrane): config block and table schemas"
```

### Task 2: The score

**Files:**
- Create: `src/junctionmapper/measure/membrane.py`
- Test: `tests/test_membrane.py`

**Interfaces:**
- Consumes: `MembraneScoreConfig`, `make_frame("membrane_patches" | "membrane_fields", rows)`, `Field` (`planes: dict[Role, np.ndarray]`, `markers`, `pixel_size_um`, `field_id`).
- Produces: `score_field(field: Field, cfg: MembraneScoreConfig) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(membrane_patches, membrane_fields)` frames with exactly the Task 1 columns.
- Produces (module-level, tested directly): `cell_region(nucleus: np.ndarray, px_um: float, region_um: float) -> np.ndarray` (bool); `floor_and_noise(plane: np.ndarray) -> tuple[float, float]`.

Algorithm (spec steps 1 to 7):
- Patches: side `round(patch_um / px)` px on a grid from (0, 0); edge patches clipped to the image; scored when `region_fraction >= 0.5`.
- Filter: `sato(..., black_ridges=False)` with sigmas spanning `line_width_um / px / 2`, each clamped to `>= 1` px. Downsample before filtering only if needed for the cost budget; keep the smallest sigma `>= 1` px after downsampling.
- `contrast` = mean of plane over top `ridge_fraction` of filter response within the patch's region / mean over the rest of the region.
- `baseline`: shuffle the patch's finite region pixels (`np.random.default_rng(zlib.crc32(f"{field_id}|{role}|{r}|{c}".encode()))`), refilter, same contrast.
- `score = contrast / baseline`; `membrane_like = score >= cutoff`, False when too dim.
- `noise == 0` or `(mean - background) < 3 * noise` -> `too_dim`, `contrast`/`baseline`/`score` NaN.
- Field row: `membrane_like_fraction` = sum of `region_fraction` over membrane-like patches / sum over scored patches; NaN when none scored.

Test fixture (in the test file, px = 0.1 µm, 400 x 400 = 2 x 2 patches): nucleus plane with disks of radius 10 px, value 200, at the centre of each 50 px cell; JM1 = grid lines every 50 px, 5 px wide, value 100, on 10, plus Gaussian noise sd 2 (seed 0).
Helpers in the test file: `field_with(jm1=..., jm2=None, segment=None, nucleus=None, px=0.1) -> Field`, `lines()`, `noise()`, `filled()`, `sparse_ones()` (each 400 x 400 float32), `small_field()` (40 x 40).

- [ ] **Step 1: Write the failing tests**

```python
def test_thin_lines_are_membrane_like(): # JM1 = grid lines
    patches, fields = score_field(field_with(jm1=lines()), MembraneScoreConfig())
    s = patches[patches.role == "JM1"]
    assert len(s) == 4 and not s.too_dim.any() and (s.score >= 1.5).all() and s.membrane_like.all()
    assert fields.set_index("role").loc["JM1", "membrane_like_fraction"] == pytest.approx(1.0)

def test_uniform_noise_scores_near_one():   # 50 + noise sd 2
    s = score_field(field_with(jm1=noise()), MembraneScoreConfig())[0]
    assert s.score.between(0.8, 1.3).all() and not s.membrane_like.any()

def test_filled_cells_are_not_membrane_like():  # each cell interior 80, borders 10, smoothed sigma 2
    s = score_field(field_with(jm1=filled()), MembraneScoreConfig())[0]
    assert (s.score < 1.5).all()

def test_a_blank_channel_is_too_dim_not_diffuse():  # zeros with 0.1% ones
    s = score_field(field_with(jm1=sparse_ones()), MembraneScoreConfig())[0]
    assert s.too_dim.all() and s.score.isna().all() and not s.membrane_like.any()

def test_score_is_invariant_to_intensity_scale():
    a = score_field(field_with(jm1=lines()), MembraneScoreConfig())[0]
    b = score_field(field_with(jm1=lines() * 10), MembraneScoreConfig())[0]
    assert np.allclose(a.score, b.score, rtol=1e-6) and (a.too_dim == b.too_dim).all()

def test_same_input_gives_identical_scores():  # seeded shuffle
    a, b = (score_field(field_with(jm1=lines()), MembraneScoreConfig())[0] for _ in range(2))
    pd.testing.assert_frame_equal(a, b)

def test_nucleus_channel_is_never_scored_and_every_other_role_is():  # JM1 + JM2 + segment
    s = score_field(field_with(jm1=lines(), jm2=noise(), segment=lines()), MembraneScoreConfig())[0]
    assert set(s.role) == {"JM1", "JM2", "segment"}

def test_a_blank_nucleus_plane_leaves_every_patch_unscored():
    p, f = score_field(field_with(jm1=lines(), nucleus=np.zeros((400, 400))), MembraneScoreConfig())
    assert (p.region_fraction == 0).all() and p.score.isna().all() and (f.n_scored == 0).all()

# Review Focus 1 to 4
def test_a_field_smaller_than_a_patch_does_not_crash():   # 40 x 40 at 0.1 µm
    p, f = score_field(small_field(), MembraneScoreConfig())
    assert len(f) >= 1 and (f.n_patches <= 1).all()

def test_coarse_pixels_clamp_the_filter_scale():   # same arrays, pixel_size_um = 1.0
    p, _ = score_field(field_with(jm1=lines(), px=1.0), MembraneScoreConfig(patch_um=200))
    assert np.isfinite(p.loc[~p.too_dim, "score"]).all()

@pytest.mark.parametrize("value", [0.0, 255.0])
def test_a_constant_channel_is_too_dim_without_warnings(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p, _ = score_field(field_with(jm1=np.full((400, 400), value)), MembraneScoreConfig())
    assert p.too_dim.all() and (~p.membrane_like).all()

def test_nan_pixels_are_ignored():
    img = lines(); img[::7, ::7] = np.nan
    p, _ = score_field(field_with(jm1=img), MembraneScoreConfig())
    assert np.isfinite(p.score).all() and p.membrane_like.all()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `./scripts/jm pytest tests/test_membrane.py -q`
Expected: FAIL with `ModuleNotFoundError: junctionmapper.measure.membrane`.

- [ ] **Step 3: Implement `score_field`, `cell_region`, `floor_and_noise` in `measure/membrane.py`**

`floor_and_noise`: `background = nanpercentile(plane, 1)`; residual = plane minus `gaussian_filter(plane, 2)` over finite pixels; `noise = 1.4826 * MAD(residual)`.

- [ ] **Step 4: Run them to verify they pass, then time a full-size field**

Run: `./scripts/jm pytest tests/test_membrane.py -q`
Expected: all pass.
Run: `./scripts/jm python -c "<build a 2984 x 2500 lines field at 0.0749 µm, time score_field>"`
Expected: under 10 s per scored channel. Record the time in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/measure/membrane.py tests/test_membrane.py
git commit -m "feat(membrane): outline-free per-patch membrane score"
```

### Task 3: Wire into `run`, flag, provenance, docs

**Files:**
- Modify: `src/junctionmapper/cli.py` (`run` signature and `work()`; `prov` dict)
- Modify: `README.md` (table list: "five tables" -> seven; a section per new table), `docs/decisions.md` (one entry: why outline-free, why the structureless baseline)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `score_field(field, cfg.membrane_score) -> (patches, fields)`; `FieldResult.membrane_patches/.membrane_fields`.
- Produces: `run --no-membrane-score`; `run.json` provenance key `membrane_score` = `"on"` or `"off"`.

In `work()`: compute `score_field` first (before `_load_labels` / `segment_field`), assign both frames onto the result after `measure_field`. Off: leave the empty defaults.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_writes_membrane_tables(env, monkeypatch):
    stub_segment(monkeypatch, env["seg"])
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--out", env["out"])
    assert r.exit_code == 0, r.output
    f = pd.read_parquet(env["out"] / "membrane_fields.parquet")
    assert set(f.role) == {"JM1", "JM2"} and len(f.field_id.unique()) == 1
    assert json.loads((env["out"] / "run.json").read_text())["provenance"]["membrane_score"] == "on"

def test_run_no_membrane_score_writes_empty_tables(env, monkeypatch):
    stub_segment(monkeypatch, env["seg"])
    invoke("run", "--config", env["cfg"], "--input", env["tif"], "--out", env["out"], "--no-membrane-score")
    assert len(pd.read_parquet(env["out"] / "membrane_patches.parquet")) == 0
    assert json.loads((env["out"] / "run.json").read_text())["provenance"]["membrane_score"] == "off"

def test_run_with_labels_still_scores_membranes(env):   # labels dir from `segment`
    ...  # same pattern as test_run_with_labels_dir_skips_segmentation; assert membrane_fields non-empty

def test_a_field_with_no_cells_still_gets_membrane_rows(env, monkeypatch):
    stub_segment(monkeypatch, <segmentation with all-zero cells>)
    ...  # exit 0, membrane_fields has the field, cells table empty for it

def test_run_with_only_jm1_scores_only_jm1(env, monkeypatch):   # Review Focus 5
    ...  # config with JM1 + nucleus only; set(membrane_fields.role) == {"JM1"}
```

Read the provenance key path from how `report/data.py` writes `run.json`; adjust the two `run.json` asserts to it.

- [ ] **Step 2: Run them to verify they fail**

Run: `./scripts/jm pytest tests/test_cli.py -q -k membrane`
Expected: FAIL (no `membrane_fields.parquet` / unknown option).

- [ ] **Step 3: Implement**, then update README and `docs/decisions.md`.

- [ ] **Step 4: Run the full suites**

Run: `./scripts/jm pytest -q` and `source /etc/profile.d/lmod.sh; module load nodejs/24.4.1; node --test tests/js/*.test.js`
Expected: all pass, 1 skipped; JS unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/cli.py tests/test_cli.py README.md docs/decisions.md
git commit -m "feat(membrane): score in run, --no-membrane-score, provenance, docs"
```

### Task 4: Real-data check (outside the repo)

**Files (all outside the repo):**
- Create: `/ix3/alee/LO_LAB/Personal/Brent_Schlegel/scratch/jm-report/membrane_check/run.sbatch` (copy of `../run.sbatch`, output `runs/<row>_<slug>`)
- Create: `/ix3/alee/LO_LAB/Personal/Brent_Schlegel/scratch/jm-report/membrane_check/summarise.py` (experiment-specific: names the reference stains)

- [ ] **Step 1:** Submit the 79-task array (`--array=0-78`, htc, 4 CPUs, 16G, 3h) at the branch head; wait for it.
- [ ] **Step 2:** `summarise.py` writes `summary.tsv`: per row, line, role/marker, `membrane_like_fraction` and `median_score` (medians over fields), plus `n_too_dim` share; and `crops.png` in the style of `../membrane_spike/crops.png`, captioned with the new score.
- [ ] **Step 3:** Report to the user: reference stains (Na/K-ATPase, phalloidin) `membrane_like_fraction`, the spike's ±dox pairs side by side with the spike's numbers, the too-dim share, and seconds per field. The user decides whether `cutoff` and `ridge_fraction` hold. A miss reopens tuning (spec), it does not block code review.
