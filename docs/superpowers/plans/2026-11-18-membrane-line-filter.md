# Membrane Line Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The membrane score counts continuous and beaded lines as lines and scattered dots as no structure; `cutoff` becomes optional.

**Architecture:** In `measure/membrane.py`, replace Sato (max over a fine scale range) with an anisotropic Hessian line response at one coarse scale set by a new `dot_spacing_um`. Everything downstream of the response (ridge split, shuffle baseline, too-dim test, tables) is unchanged. `cutoff` defaults to `None`, which leaves `membrane_like` null.

**Tech Stack:** Python 3.11, numpy, scipy.ndimage, scikit-image 0.26 (`skimage.feature.hessian_matrix`, `hessian_matrix_eigvals`), pandas, pydantic. Tests: `./scripts/jm python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-11-18-membrane-line-filter-design.md` (amends `2026-11-17-membrane-score-design.md`)

## Global Constraints

- Generic: no marker, line or biology names in code or defaults (test fixtures may name markers).
- Filter: `sigma**2 * max(0, -l2 - |l1|)`, `l2` the most negative Hessian eigenvalue, one `sigma`.
- `sigma = dot_spacing_um / 2`, in downsampled pixels. No max over a range of scales.
- `dot_spacing_um` default `1.0`; must be finite and `> line_width_um[1]`.
- `line_width_um` sets only the downsampling factor (rule unchanged: largest integer keeping `line_width_um[0] / px / 2 >= 1` px after downsampling).
- Sigma passed to `hessian_matrix` stays `> 1.0` (`FILTER_MIN_SIGMA_PX = 1.01`): skimage 0.26 truncates at 100 sigma otherwise.
- `cutoff: float | None = None`; when set, still `> 1`.
- Unchanged: shuffle baseline (native-pixel shuffle, then downsample), too-dim test, `min_ridge_intensity`, table columns and dtypes.
- Hooks: Read is blocked on .py/.js (use `sed -n`); Bash is blocked on CamelCase identifiers (grep lowercase or `-i`).
- Never `rm -rf`. Never `git add -A`. No U+00B7. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Downsampled path (0.075 um/px, factor 2): scattered dots still score near 1. Test in Task 1.
2. Coarse pixels (1 um/px): sigma clamps to 1.01, no crash, finite scores. Existing `test_coarse_pixels_clamp_the_filter_scale`, kept in Task 1.
3. Cutoff unset: `membrane_like` all null with `boolean` dtype, `membrane_like_fraction` NaN, and both survive the parquet and CSV write. Test in Task 2.
4. `dot_spacing_um` at or below `line_width_um[1]`, or NaN: config error. Test in Task 1.
5. A field with no line structure but bright speckle (noise on a floor): still near 1 under the new filter. Existing `test_uniform_noise_scores_near_one`, kept in Task 1.

---

### Task 1: Anisotropic line filter at the dot-spacing scale

**Files:**
- Modify: `src/junctionmapper/config.py` (`MembraneScoreConfig`)
- Modify: `src/junctionmapper/measure/membrane.py` (`_filter_geometry`, `_ridge`, `_patch_rows`, `score_field`, module docstring, constants, imports)
- Test: `tests/test_membrane.py`, `tests/test_config.py`
- Docs: `README.md` (config key table, `membrane_patches` section), `docs/decisions.md` (new entry after the min_ridge_intensity one)

**Interfaces:**
- Produces: `MembraneScoreConfig.dot_spacing_um: float = 1.0`
- Produces: `_filter_geometry(px_um: float, line_width_um: tuple[float, float], dot_spacing_um: float) -> tuple[int, float]` (factor, sigma in downsampled px)
- Produces: `_ridge(img: np.ndarray, sigma: float) -> np.ndarray`
- `_patch_rows(..., size, factor, sigma)` takes the single sigma in place of the list.

- [ ] **Step 1: Write the failing tests** in `tests/test_membrane.py`

Fixtures (0.1 um/px, same 50 px grid and `_noise()` as `lines()`):
- `beaded_lines(spacing_px=10)`: dots every `spacing_px` (default 1 um) along the grid lines; each dot a single pixel blurred by `ndi.gaussian_filter(..., 1.5)` and scaled so the peak is 100 on a 10 floor, plus `_noise()`.
- `scattered_dots()`: the same number of dots, same blur and peak, at seeded random positions (`default_rng(3)`), plus `_noise()`.
- `dense_dots()`: as `scattered_dots` with 10x the count.

```python
def test_beaded_lines_score_like_lines():
    s = score_field(field_with(jm1=beaded_lines()), MembraneScoreConfig())[0]
    assert not s.too_dim.any() and (s.score >= 1.5).all()

@pytest.mark.parametrize("make", [scattered_dots, dense_dots])
def test_scattered_dots_score_near_one(make):
    s = score_field(field_with(jm1=make()), MembraneScoreConfig())[0]
    assert np.isfinite(s.score).all() and (s.score <= 1.2).all()

def test_scattered_dots_score_near_one_on_the_downsampled_path():
    s = score_field(field_with(jm1=scattered_dots(), px=0.075), MembraneScoreConfig())[0]
    full = s[(s.y0_px == 0) & (s.x0_px == 0)]
    assert np.isfinite(full.score).all() and (full.score <= 1.2).all()

def test_dots_spaced_wider_than_the_setting_read_as_dots():
    # 2 um beads at the default 1 um (sigma 5 px) do not merge; the synthetic sweep in the
    # spec measured 19 against 44 for 1 um beads.
    wide = score_field(field_with(jm1=beaded_lines(spacing_px=20)), MembraneScoreConfig())[0]
    near = score_field(field_with(jm1=beaded_lines()), MembraneScoreConfig())[0]
    assert (wide.score < near.score).all()
```

Replace `test_filter_sigmas_stay_above_one_pixel` with the same parametrization over `px` asserting `factor >= 1 and sigma > 1.0` from `_filter_geometry(px, (0.3, 0.6), 1.0)`, plus:

```python
def test_filter_sigma_is_half_the_dot_spacing():
    assert _filter_geometry(0.1, (0.3, 0.6), 1.0) == (1, pytest.approx(5.0))
    assert _filter_geometry(0.075, (0.3, 0.6), 1.0) == (2, pytest.approx(1.0 / 0.075 / 2 / 2))
```

In `tests/test_config.py`: defaults assert `m.dot_spacing_um == 1.0`; bad values add `{"dot_spacing_um": 0.6}`, `{"dot_spacing_um": float("nan")}`, `{"dot_spacing_um": 0.5, "line_width_um": [0.3, 0.6]}`.

- [ ] **Step 2: Run them to verify they fail**

Run: `./scripts/jm python -m pytest tests/test_membrane.py tests/test_config.py -q`
Expected: FAIL (`dot_spacing_um` unknown; scattered dots score above 1.2 under Sato).

- [ ] **Step 3: Implement**

- `config.py`: add `dot_spacing_um: float = 1.0` with a comment ("dots farther apart than this do not make a line"); a `model_validator(mode="after")` raising `ConfigError` unless finite and `> line_width_um[1]`. Update the `line_width_um` comment: "thinnest line resolved; sets the downsampling only".
- `membrane.py`:
  - `_filter_geometry`: factor as now from `line_width_um[0]`; `sigma = max(dot_spacing_um / px_um / 2 / factor, FILTER_MIN_SIGMA_PX)`.
  - `_ridge(img, sigma)`: `hessian_matrix(img, sigma, use_gaussian_derivatives=True)`, `hessian_matrix_eigvals` (returns decreasing order, so `e[1]` is the most negative); return `sigma**2 * np.clip(-e[1] - np.abs(e[0]), 0, None)`.
  - Drop the `sato` import; thread the single `sigma` through `_patch_rows` and `score_field`.
  - Module docstring: say the filter ignores structure that curves equally both ways, at one scale of half the dot spacing, so beaded lines merge into lines and scattered dots blur to haze.
- `README.md`: add the `membrane_score.dot_spacing_um | 1.0 |` row; reword the `line_width_um` row; in the `membrane_patches` section say the line filter suppresses dots and joins beads closer than `dot_spacing_um`.
- `docs/decisions.md`: entry "**The line filter ignores dots and works at one coarse scale.**" with *Why* (Sato gives a dot 71% of a line's response; at fine scales a dot's rim reads as a curved line) and *Cost if wrong* (two membranes closer than `dot_spacing_um` merge; beads wider apart than it read as dots).

- [ ] **Step 4: Re-band and run the whole membrane suite**

Run: `./scripts/jm python -m pytest tests/test_membrane.py tests/test_config.py -q`
Expected: PASS. Existing tests whose measured values move (`lines`, `filled`, `noise_on_floor`, the downsampled-path and fraction tests) may be re-banded only if the spec's goals still hold (thin lines >= 1.5, noise and filled near 1, blank too dim); record each old and new measured value in the task report.

- [ ] **Step 5: Run the full suite**

Run: `./scripts/jm python -m pytest -q`
Expected: all pass, 1 skipped (no java).

- [ ] **Step 6: Commit**

```bash
git add src/junctionmapper/config.py src/junctionmapper/measure/membrane.py tests/test_membrane.py tests/test_config.py README.md docs/decisions.md
git commit -m "feat(membrane): line filter that ignores dots, at the dot-spacing scale"
```

### Task 2: Optional cutoff

**Files:**
- Modify: `src/junctionmapper/config.py` (`cutoff`, `_check_cutoff`)
- Modify: `src/junctionmapper/measure/membrane.py` (`membrane_like` in `_patch_rows`, `_field_row`)
- Test: `tests/test_membrane.py`, `tests/test_config.py`, `tests/test_tables.py`
- Docs: `README.md`, `docs/superpowers/specs/2026-11-17-membrane-score-design.md` (defaults block and step 6)

**Interfaces:**
- Consumes: Task 1's `MembraneScoreConfig` and `score_field`.
- Produces: `MembraneScoreConfig.cutoff: float | None = None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_membrane.py
def test_no_cutoff_makes_no_membrane_like_call():
    p, f = score_field(field_with(jm1=lines()), MembraneScoreConfig())
    assert np.isfinite(p.score).all() and p.membrane_like.isna().all()
    assert str(p.membrane_like.dtype) == "boolean" and f.membrane_like_fraction.isna().all()
```

```python
# tests/test_tables.py
def test_an_all_null_membrane_like_column_round_trips(tmp_path):
    # write one membrane_patches frame with membrane_like all None through write_tables;
    # read the parquet back: dtype boolean, all NA; the CSV column is empty strings.
```

Existing tests that assert on `membrane_like` or `membrane_like_fraction` pass `MembraneScoreConfig(cutoff=1.5)` (add a module-level `CALL = MembraneScoreConfig(cutoff=1.5)` and use it). `tests/test_config.py`: defaults assert `m.cutoff is None`; `{"cutoff": 1.0}` stays rejected.

- [ ] **Step 2: Run them to verify they fail**

Run: `./scripts/jm python -m pytest tests/test_membrane.py tests/test_config.py tests/test_tables.py -q`
Expected: FAIL (`membrane_like` is set at the 1.5 default).

- [ ] **Step 3: Implement**

- `config.py`: `cutoff: float | None = None`; the validator accepts `None`.
- `membrane.py`: `membrane_like` is null when `cfg.cutoff is None` or the score is not finite. `_field_row` gives `membrane_like_fraction = NaN` when no scored patch has a non-null call.
- `README.md`: cutoff row default "none"; one line: "Set `cutoff` from the experiment's own controls; it depends on optics, pixel size, cell density and signal-to-noise."
- Spec `2026-11-17`: defaults block `cutoff: null`, step 6 says the call is optional.

- [ ] **Step 4: Run the tests and the full suite**

Run: `./scripts/jm python -m pytest -q`
Expected: all pass, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/config.py src/junctionmapper/measure/membrane.py tests/test_membrane.py tests/test_config.py tests/test_tables.py README.md docs/superpowers/specs/2026-11-17-membrane-score-design.md
git commit -m "feat(membrane): cutoff optional, no membrane_like call by default"
```

### Task 3: Real-data check (controller; no repo diff)

**Files (scratch, outside the repo):** `/ix3/alee/LO_LAB/Personal/Brent_Schlegel/scratch/jm-report/membrane_check/`

- [ ] **Step 1:** Archive `runs/` to `runs_v3/`, make an empty `runs/`, and `sbatch run.sbatch` (overlay config keeps `min_ridge_intensity: 5.0`; no cutoff). Keep `feat/membrane-score` checked out until the array finishes.
- [ ] **Step 2:** Run `cutoff.py`, `summarise.py` and the pair view. Pass condition: SUM44 -dox alpha-catenin median score drops toward 1; -dox vs +dox p120 and the Na/K-ATPase and phalloidin references keep their separation. Report the measured medians and the extra filter time.
- [ ] **Step 3:** Show the user the cutoff sweep; on their choice, add `cutoff` to the overlay config only (experiment config, never the code default).
