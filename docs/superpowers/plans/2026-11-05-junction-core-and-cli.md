# Junction Mapper Core + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless Python library and CLI that reads OME-TIFF/Zarr, segments cells with Cellpose-SAM, detects interfaces by two corner methods, and emits Junction Mapper's published parameters as tidy tables plus an HTML report.

**Architecture:** Pure functions over numpy arrays, layered: config → field reader → segmentation → geometry (boundary, corners, interfaces) → measurement → tables → report. No module imports napari. Every measurement function takes arrays and parameters, never a context object, so the measurement math is testable against analytically-known synthetic fixtures without any image file or model.

**Tech Stack:** Python 3.11+, numpy, scipy.ndimage, scikit-image, bioio (+bioio-ome-tiff, +bioio-ome-zarr), cellpose ≥4 (Cellpose-SAM), pandas + pyarrow, pydantic v2, jinja2 + matplotlib, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-junction-mapper-port-design.md`

## Global Constraints

- Package `junctionmapper`. README credits the original: Imperial College London, doi:10.7554/eLife.45413.
- **Three channel roles: `JM1`, `JM2`, `nucleus`.** No protein identity is ever hardcoded. Marker identity is configuration and travels with every output row.
- **JM1 is the segmentation source**, as in the original.
- Thresholds follow the paper formula `Σ if(g>T) g`. The threshold primitive returns **masked greyscale, not a binary mask** — surviving pixels keep their values.
- **8-connectivity** (`|dx|≤1 and |dy|≤1`) is the single connectivity rule everywhere. Never the Java's OR-test.
- Outputs carry **both** pixel and µm/µm² values. Fail loudly when calibration is neither in metadata nor configured.
- Dilation band default **2 px**, specifiable in µm.
- `numIntraCells` default **5**, configurable, with an all-pairs option.
- Vendor formats (ND2/CZI/LIF) out of scope. Non-OME TIFF accepted with explicit role mapping and pixel size.
- 2D only. Z/T/scene iterate as independent fields.
- Tests: exact equality for integer pixel counts; `rel=1e-9` only for ratios and Euclidean distances.

## Review Focus

Five input classes the spec implies that no task's happy path exercises. Each has its test pinned to the owning task below.

1. **A cell with no neighbours.** Discohesion is the biology under study, so isolated cells are signal, not error. Must emit a cell row with zero interfaces, not crash and not silently vanish. (Task 7)
2. **A cell touching the image border.** Its boundary is truncated, so every interface metric is meaningless. Must be flagged and excluded, never measured. (Task 7)
3. **Missing calibration or missing channel names.** Real exports frequently lack both. Must fail with a message naming what to supply, never default to 1.0 µm/px. (Tasks 1, 2)
4. **Multinucleate cells and orphaned nuclei.** The Java ignores both. Must flag, not drop silently or double-count. (Task 11)
5. **Degenerate Cellpose labels** — single-pixel regions, labels with interior holes, non-contiguous labels. Must be rejected with a flag rather than producing a malformed boundary. (Task 4)

---

### Task 1: Configuration and channel roles

**Files:**
- Create: `src/junctionmapper/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Role` (StrEnum: `JM1`, `JM2`, `NUCLEUS`); `ChannelSpec(role: Role, channel: str | int, marker: str)`; `RunConfig(channels: list[ChannelSpec], pixel_size_um: float | None, dilation_px: int = 2, dilation_um: float | None = None, num_intra_cells: int = 5, all_pairs: bool = False, threshold: dict[Role, float] | None = None)`; `RunConfig.role_for(r: Role) -> ChannelSpec`; `RunConfig.marker_for(r: Role) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_role_lookup_returns_marker_identity():
    cfg = RunConfig(channels=[ChannelSpec(role=Role.JM1, channel="NaK", marker="Na/K-ATPase")],
                    pixel_size_um=0.325)
    assert cfg.marker_for(Role.JM1) == "Na/K-ATPase"

def test_duplicate_role_rejected():
    with pytest.raises(ValidationError):
        RunConfig(channels=[ChannelSpec(role=Role.JM1, channel=0, marker="CDH1"),
                            ChannelSpec(role=Role.JM1, channel=1, marker="NaK")],
                  pixel_size_um=0.325)

def test_jm1_and_nucleus_are_required():
    with pytest.raises(ValidationError):
        RunConfig(channels=[ChannelSpec(role=Role.JM2, channel=0, marker="p120")],
                  pixel_size_um=0.325)

def test_dilation_um_converts_to_px_using_pixel_size():
    cfg = RunConfig(channels=[...], pixel_size_um=0.5, dilation_um=1.5)
    assert cfg.resolved_dilation_px() == 3
```

JM2 is optional; JM1 and NUCLEUS are required.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v` — Expected: FAIL, `config` module not found.

- [ ] **Step 3: Implement the pydantic v2 models in `src/junctionmapper/config.py`**

`resolved_dilation_px()` prefers `dilation_um` over `dilation_px` when both are set, rounding to nearest int, minimum 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/config.py tests/test_config.py
git commit -m "feat: run configuration with explicit channel roles and marker identity"
```

---

### Task 2: Field reader

**Files:**
- Create: `src/junctionmapper/io/reader.py`
- Test: `tests/test_reader.py`

**Interfaces:**
- Consumes: `RunConfig`, `Role` from Task 1.
- Produces: `Field` dataclass with `planes: dict[Role, np.ndarray]` (2D, float32), `pixel_size_um: float`, `markers: dict[Role, str]`, `source: Path`, `field_id: str`; and `read_fields(path: Path, cfg: RunConfig) -> Iterator[Field]` yielding one `Field` per Z/T/scene combination.

- [ ] **Step 1: Write the failing tests**

```python
def test_reads_ome_tiff_and_maps_channels_by_name(tmp_ome_tiff):
    fields = list(read_fields(tmp_ome_tiff, cfg_named))
    assert fields[0].planes[Role.JM1].shape == (64, 64)
    assert fields[0].markers[Role.JM1] == "Na/K-ATPase"

def test_zstack_yields_one_field_per_plane(tmp_ome_tiff_z3):
    assert len(list(read_fields(tmp_ome_tiff_z3, cfg_named))) == 3

def test_missing_calibration_raises_naming_the_fix(tmp_plain_tiff):
    cfg = cfg_named.model_copy(update={"pixel_size_um": None})
    with pytest.raises(CalibrationError, match="pixel_size_um"):
        list(read_fields(tmp_plain_tiff, cfg))

def test_config_pixel_size_overrides_absent_metadata(tmp_plain_tiff):
    assert list(read_fields(tmp_plain_tiff, cfg_named))[0].pixel_size_um == 0.325

def test_channel_name_not_found_raises_naming_available_names(tmp_ome_tiff):
    cfg = cfg_with_channel("NoSuchChannel")
    with pytest.raises(ChannelError, match="NoSuchChannel"):
        list(read_fields(tmp_ome_tiff, cfg))
```

Covers Review Focus item 3. Build the OME-TIFF fixtures with `tifffile.imwrite(..., ome=True)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reader.py -v` — Expected: FAIL

- [ ] **Step 3: Implement `read_fields` in `src/junctionmapper/io/reader.py`**

Use `bioio.BioImage`. Metadata pixel size wins when present; `cfg.pixel_size_um` fills the gap; neither raises `CalibrationError`. Resolve `ChannelSpec.channel` against `BioImage.channel_names` when it is a `str`, by position when `int`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reader.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/io tests/test_reader.py
git commit -m "feat: OME-TIFF/Zarr field reader with explicit calibration handling"
```

---

### Task 3: Threshold primitive

**Files:**
- Create: `src/junctionmapper/measure/threshold.py`
- Test: `tests/test_threshold.py`

**Interfaces:**
- Produces: `mask_above(image: np.ndarray, t: float) -> np.ndarray` returning an array the same dtype/shape where `image > t` keeps its value and everything else is 0.

This is a small task on purpose: it is the single highest-risk porting trap, since the same array is consumed both as a mask (`> 0`) and as the intensity source.

- [ ] **Step 1: Write the failing tests**

```python
def test_surviving_pixels_keep_their_greyscale_values():
    out = mask_above(np.array([[10, 60, 200]], dtype=np.float32), 50)
    assert out.tolist() == [[0.0, 60.0, 200.0]]

def test_is_not_a_binary_mask():
    assert mask_above(np.array([[200]], dtype=np.float32), 50).max() == 200.0

def test_threshold_is_exclusive():
    assert mask_above(np.array([[50]], dtype=np.float32), 50).tolist() == [[0.0]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_threshold.py -v` — Expected: FAIL

- [ ] **Step 3: Implement `mask_above` in `src/junctionmapper/measure/threshold.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_threshold.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/measure/threshold.py tests/test_threshold.py
git commit -m "feat: masked-greyscale threshold primitive"
```

---

### Task 4: Ordered boundary traversal

**Files:**
- Create: `src/junctionmapper/geometry/boundary.py`
- Test: `tests/test_boundary.py`

**Interfaces:**
- Produces: `trace_boundary(labels: np.ndarray, label: int) -> np.ndarray` returning an `(N, 2)` int array of `(row, col)` in contiguous order, each pixel 8-adjacent to the next and the last 8-adjacent to the first; `BoundaryError` for degenerate labels.

- [ ] **Step 1: Write the failing tests**

```python
def test_square_boundary_is_contiguous_and_closed():
    labels = np.zeros((10, 10), int); labels[2:8, 2:8] = 1
    b = trace_boundary(labels, 1)
    assert len(b) == 20                                  # perimeter of a 6x6 block
    assert np.all(np.abs(np.diff(b, axis=0)).max(axis=1) <= 1)
    assert np.abs(b[0] - b[-1]).max() <= 1

def test_single_pixel_label_rejected():
    labels = np.zeros((10, 10), int); labels[5, 5] = 1
    with pytest.raises(BoundaryError, match="degenerate"):
        trace_boundary(labels, 1)

def test_label_with_interior_hole_rejected():
    labels = np.zeros((12, 12), int); labels[2:10, 2:10] = 1; labels[5:7, 5:7] = 0
    with pytest.raises(BoundaryError, match="hole"):
        trace_boundary(labels, 1)

def test_noncontiguous_label_rejected():
    labels = np.zeros((12, 12), int); labels[2:4, 2:4] = 1; labels[8:10, 8:10] = 1
    with pytest.raises(BoundaryError, match="contiguous"):
        trace_boundary(labels, 1)
```

Covers Review Focus item 5.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_boundary.py -v` — Expected: FAIL

- [ ] **Step 3: Implement `trace_boundary` in `src/junctionmapper/geometry/boundary.py`**

Moore-neighbour tracing with a Jacob's stopping criterion. Validate first with `scipy.ndimage.label` (contiguity) and `scipy.ndimage.binary_fill_holes` (holes); reject areas below 4 px as degenerate.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_boundary.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/geometry/boundary.py tests/test_boundary.py
git commit -m "feat: ordered boundary tracing with degenerate-label rejection"
```

---

### Task 5: Douglas–Peucker corners

**Files:**
- Create: `src/junctionmapper/geometry/corners_dp.py`
- Test: `tests/test_corners_dp.py`

**Interfaces:**
- Consumes: boundary arrays from Task 4.
- Produces: `dp_corners(boundary: np.ndarray, epsilon: float) -> np.ndarray` of indices into `boundary`, sorted ascending.

- [ ] **Step 1: Write the failing tests**

```python
def test_square_yields_four_corners():
    b = trace_boundary(square_labels(), 1)
    assert len(dp_corners(b, epsilon=1.5)) == 4

def test_corner_indices_land_on_the_geometric_corners():
    b = trace_boundary(square_labels(), 1)         # 6x6 block at offset (2,2)
    pts = {tuple(p) for p in b[dp_corners(b, 1.5)]}
    assert pts == {(2, 2), (2, 7), (7, 7), (7, 2)}

def test_larger_epsilon_yields_no_more_corners():
    b = trace_boundary(hexagon_labels(), 1)
    assert len(dp_corners(b, 5.0)) <= len(dp_corners(b, 1.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_corners_dp.py -v` — Expected: FAIL

- [ ] **Step 3: Implement `dp_corners` in `src/junctionmapper/geometry/corners_dp.py`**

Closed-curve Douglas–Peucker: seed with the two mutually most-distant boundary points, then recurse on both arcs. Perpendicular distance to the chord, as in Douglas & Peucker (doi:10.3138/FM57-6770-U75U-7727).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_corners_dp.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/geometry/corners_dp.py tests/test_corners_dp.py
git commit -m "feat: Douglas-Peucker corner detection on closed boundaries"
```

---

### Task 6: Tricellular-junction corners

**Files:**
- Create: `src/junctionmapper/geometry/corners_tcj.py`
- Test: `tests/test_corners_tcj.py`

**Interfaces:**
- Produces: `neighbour_at(labels, boundary, label, search_radius=2) -> np.ndarray` of int neighbour label per boundary pixel (0 = background/free edge); `tcj_corners(neighbour_ids: np.ndarray) -> np.ndarray` of indices where neighbour identity changes, treating the array as circular.

- [ ] **Step 1: Write the failing tests**

```python
def test_neighbour_ids_identify_the_adjacent_label():
    labels = two_cells_sharing_a_vertical_edge()        # 1 left, 2 right
    b = trace_boundary(labels, 1)
    ids = neighbour_at(labels, b, 1)
    assert set(ids) == {0, 2}

def test_corners_land_where_neighbour_identity_changes():
    labels = three_cells_meeting_at_a_point()
    b = trace_boundary(labels, 1)
    assert len(tcj_corners(neighbour_at(labels, b, 1))) == 2

def test_isolated_cell_has_no_corners():
    labels = single_isolated_cell()
    b = trace_boundary(labels, 1)
    assert len(tcj_corners(neighbour_at(labels, b, 1))) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_corners_tcj.py -v` — Expected: FAIL

- [ ] **Step 3: Implement both functions in `src/junctionmapper/geometry/corners_tcj.py`**

For each boundary pixel, take the modal non-self, non-background label within `search_radius`; ties resolve to the nearest. `tcj_corners` wraps circularly so a change between last and first registers.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_corners_tcj.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/geometry/corners_tcj.py tests/test_corners_tcj.py
git commit -m "feat: tricellular-junction corners from label adjacency"
```

---

### Task 7: Interface extraction and cell-level QC flags

**Files:**
- Create: `src/junctionmapper/geometry/interfaces.py`, `src/junctionmapper/qc.py`
- Test: `tests/test_interfaces.py`

**Interfaces:**
- Consumes: Tasks 4–6.
- Produces: `Interface` dataclass (`cell_id`, `neighbour_id: int | None`, `start_idx`, `end_idx`, `pixels: np.ndarray`, `method: str`); `extract_interfaces(labels, label, method, epsilon=None) -> list[Interface]`; `CellFlags` dataclass (`border_touching`, `isolated`, `degenerate`, `corner_methods_disagree`, `n_corners_dp`, `n_corners_tcj`); `assess_cell(labels, label, epsilon) -> CellFlags`.

- [ ] **Step 1: Write the failing tests**

```python
def test_two_adjacent_cells_yield_one_shared_interface():
    labels = two_cells_sharing_a_vertical_edge()
    ifs = extract_interfaces(labels, 1, method="tcj")
    assert [i.neighbour_id for i in ifs] == [2]

def test_isolated_cell_yields_no_interfaces_and_is_flagged():
    labels = single_isolated_cell()
    assert extract_interfaces(labels, 1, method="tcj") == []
    assert assess_cell(labels, 1, epsilon=1.5).isolated is True

def test_border_touching_cell_is_flagged():
    labels = np.zeros((10, 10), int); labels[0:5, 0:5] = 1
    assert assess_cell(labels, 1, epsilon=1.5).border_touching is True

def test_corner_method_disagreement_is_flagged():
    labels = three_cells_meeting_at_a_point()          # 2 TCJ corners, 4+ DP corners
    assert assess_cell(labels, 1, epsilon=1.5).corner_methods_disagree is True

def test_interface_pixels_are_contiguous_and_shared_by_both_cells():
    labels = two_cells_sharing_a_vertical_edge()
    px = extract_interfaces(labels, 1, method="tcj")[0].pixels
    assert np.all(np.abs(np.diff(px, axis=0)).max(axis=1) <= 1)
```

Covers Review Focus items 1 and 2.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_interfaces.py -v` — Expected: FAIL

- [ ] **Step 3: Implement both modules**

`method="dp"` splits the boundary at `dp_corners`; `method="tcj"` splits at `tcj_corners` and carries the neighbour id. DP interfaces get `neighbour_id` from the modal neighbour over their pixel run. Runs whose neighbour is background are free edge and are not interfaces. `corner_methods_disagree` when the counts differ by more than 1. Border-touching and degenerate cells are flagged and excluded from measurement by callers.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_interfaces.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/geometry/interfaces.py src/junctionmapper/qc.py tests/test_interfaces.py
git commit -m "feat: interface extraction for both corner methods, with cell QC flags"
```

---

### Task 8: Interface geometry metrics

**Files:**
- Create: `src/junctionmapper/measure/interface.py`
- Test: `tests/test_measure_geometry.py`

**Interfaces:**
- Produces: `interface_contour(pixels) -> float`, `straight_line_length(pixels) -> float`, `linearity_index(pixels) -> float`. Parameters 1, 2, and their ratio from the user guide.

- [ ] **Step 1: Write the failing tests**

```python
def test_contour_of_a_straight_run_is_its_step_sum():
    px = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    assert interface_contour(px) == pytest.approx(3.0)

def test_contour_counts_diagonal_steps_as_sqrt2():
    px = np.array([[0, 0], [1, 1], [2, 2]])
    assert interface_contour(px) == pytest.approx(2 * math.sqrt(2))

def test_straight_line_length_is_first_to_last_euclidean():
    px = np.array([[0, 0], [1, 1], [0, 4]])
    assert straight_line_length(px) == pytest.approx(4.0)

def test_linearity_of_a_straight_run_is_one():
    assert linearity_index(np.array([[0, 0], [0, 1], [0, 2]])) == pytest.approx(1.0)

def test_single_pixel_interface_has_zero_length_and_nan_linearity():
    px = np.array([[3, 3]])
    assert interface_contour(px) == 0.0
    assert math.isnan(linearity_index(px))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_measure_geometry.py -v` — Expected: FAIL

- [ ] **Step 3: Implement the three functions**

`linearity_index` is `straight_line_length / interface_contour`, NaN when the contour is 0.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_measure_geometry.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/measure/interface.py tests/test_measure_geometry.py
git commit -m "feat: interface contour, straight-line length, linearity index"
```

---

### Task 9: Dilated band, area and intensity metrics

**Files:**
- Modify: `src/junctionmapper/measure/interface.py`
- Test: `tests/test_measure_intensity.py`

**Interfaces:**
- Consumes: `mask_above` (Task 3).
- Produces: `dilated_band(pixels, shape, cycles) -> np.ndarray` (bool mask); `band_metrics(band, marker_image, t) -> BandMetrics` with fields `interface_area_px`, `marker_area_px`, `interface_occupancy_pct`, `marker_intensity`, `intensity_per_area`, `cluster_density`. Parameters 4–7 from the user guide.

- [ ] **Step 1: Write the failing tests**

```python
def test_one_dilation_cycle_widens_a_line_to_three_pixels():
    band = dilated_band(np.array([[5, c] for c in range(10)]), (20, 20), cycles=1)
    assert band[:, 5].sum() == 3

def test_interface_area_is_the_band_pixel_count():
    assert band_metrics(band_of(12), img, t=0).interface_area_px == 12

def test_marker_area_counts_only_pixels_above_threshold():
    img = np.zeros((20, 20), np.float32); img[5, 0:4] = 200.0
    assert band_metrics(line_band(), img, t=50).marker_area_px == 4

def test_intensity_sums_greyscale_values_not_pixel_counts():
    img = np.zeros((20, 20), np.float32); img[5, 0:4] = 200.0
    assert band_metrics(line_band(), img, t=50).marker_intensity == pytest.approx(800.0)

def test_occupancy_is_marker_area_over_interface_area_as_percent():
    m = band_metrics(band_of(20, marker_px=5), img, t=50)
    assert m.interface_occupancy_pct == pytest.approx(25.0)

def test_zero_marker_area_gives_nan_cluster_density_not_zero_division():
    assert math.isnan(band_metrics(line_band(), np.zeros((20, 20), np.float32), t=50).cluster_density)
```

The intensity test is the one that catches the binary-mask trap.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_measure_intensity.py -v` — Expected: FAIL

- [ ] **Step 3: Implement both functions**

`dilated_band` rasterises the pixel run then applies `scipy.ndimage.binary_dilation` with a 3×3 structuring element, `cycles` iterations. `cluster_density` is `marker_intensity / marker_area_px`, `intensity_per_area` is `marker_intensity / interface_area_px`; both NaN on a zero denominator.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_measure_intensity.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/measure/interface.py tests/test_measure_intensity.py
git commit -m "feat: dilated interface band with area and intensity metrics"
```

---

### Task 10: Fragmented and junction contour metrics

**Files:**
- Modify: `src/junctionmapper/measure/interface.py`
- Test: `tests/test_measure_fragments.py`

**Interfaces:**
- Produces: `fragmented_contour(pixels, marker_image, t) -> float`, `junction_contour(pixels, marker_image, t) -> float`, `junction_straight_line(pixels, marker_image, t) -> float`, `coverage_index(pixels, marker_image, t) -> float`. Parameters 3, 8, 9 plus coverage.

This is the task carrying the spec's one deliberate divergence from the Java.

- [ ] **Step 1: Write the failing tests**

```python
def test_fragmented_contour_sums_only_connected_above_threshold_steps():
    # two 3-px fragments separated by a 4-px gap along a straight run
    assert fragmented_contour(straight_run(10), two_fragment_image(), t=50) == pytest.approx(4.0)

def test_fragmented_contour_excludes_gaps_the_java_or_test_would_admit():
    # last=(100,50) next=(101,80): one coordinate differs by 1, the other by 30
    assert fragmented_contour(l_shaped_run(), gap_image(), t=50) == pytest.approx(0.0)

def test_junction_contour_spans_gaps_between_first_and_last_marker_pixel():
    assert junction_contour(straight_run(10), two_fragment_image(), t=50) == pytest.approx(9.0)

def test_coverage_is_fragmented_over_interface_contour():
    assert coverage_index(straight_run(10), two_fragment_image(), t=50) == pytest.approx(4.0 / 9.0)

def test_fully_stained_interface_has_coverage_one():
    assert coverage_index(straight_run(10), fully_stained(), t=50) == pytest.approx(1.0)

def test_no_marker_above_threshold_gives_zero_coverage():
    assert coverage_index(straight_run(10), np.zeros((20, 20), np.float32), t=50) == 0.0
```

The second test is the regression guard for the corrected connectivity rule.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_measure_fragments.py -v` — Expected: FAIL

- [ ] **Step 3: Implement the four functions**

Walk `pixels` in order. `fragmented_contour` accumulates the step between consecutive above-threshold pixels **only when both coordinates differ by at most 1**. `junction_contour` accumulates every step between consecutive above-threshold pixels regardless of gaps. `junction_straight_line` is the Euclidean distance between the first and last above-threshold pixel. Both JM1 and JM2 use these same functions — no asymmetry.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_measure_fragments.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/measure/interface.py tests/test_measure_fragments.py
git commit -m "feat: fragmented and junction contour with corrected 8-connectivity"
```

---

### Task 11: Nuclei association and inter-nuclear distances

**Files:**
- Create: `src/junctionmapper/measure/nuclei.py`
- Test: `tests/test_nuclei.py`

**Interfaces:**
- Produces: `nucleus_centroids(nucleus_labels) -> dict[int, tuple[float, float]]`; `associate(cell_labels, nucleus_labels) -> tuple[dict[int, list[int]], NucleusFlags]`; `intra_distances(centroids, k=5, all_pairs=False) -> list[tuple[int, int, float]]`. `NucleusFlags` carries `multinucleate: list[int]`, `orphan_nuclei: list[int]`, `anucleate_cells: list[int]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_centroid_matches_the_java_midpoint_formula():
    # (max(x)-min(x))//2 + min(x), per the user guide
    assert nucleus_centroids(rect_nucleus(rows=(2, 8), cols=(4, 10)))[1] == (5, 7)

def test_each_nucleus_maps_to_its_containing_cell():
    cells, flags = associate(two_cells(), one_nucleus_in_each())
    assert cells == {1: [1], 2: [2]}

def test_two_nuclei_in_one_cell_flags_multinucleate():
    _, flags = associate(one_cell(), two_nuclei_inside())
    assert flags.multinucleate == [1]

def test_nucleus_outside_any_cell_flags_orphan():
    _, flags = associate(one_cell(), nucleus_in_background())
    assert flags.orphan_nuclei == [1]

def test_cell_without_a_nucleus_is_flagged():
    _, flags = associate(two_cells(), one_nucleus_in_each_but_one())
    assert flags.anucleate_cells == [2]

def test_k_nearest_returns_k_distances_per_nucleus_sorted_ascending():
    d = intra_distances(six_centroids(), k=3)
    assert len(d) == 6 * 3
    assert d[0][2] <= d[1][2] <= d[2][2]
```

Covers Review Focus item 4.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nuclei.py -v` — Expected: FAIL

- [ ] **Step 3: Implement the three functions**

Centroid uses the guide's midpoint-of-extent formula, not the centre of mass — the guide states the assumption that nuclei are roughly symmetrical. Association assigns a nucleus to the cell label holding the plurality of its pixels. `intra_distances` uses `scipy.spatial.cKDTree`; `all_pairs=True` ignores `k`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nuclei.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/measure/nuclei.py tests/test_nuclei.py
git commit -m "feat: nucleus-cell association and inter-nuclear distances"
```

---

### Task 12: Cellpose-SAM segmentation driver

**Files:**
- Create: `src/junctionmapper/segment/cellpose.py`
- Test: `tests/test_segment.py`

**Interfaces:**
- Consumes: `Field` (Task 2).
- Produces: `SegmentationParams(model: str = "cpsam", diameter: float | None = None, flow_threshold: float = 0.4, cellprob_threshold: float = 0.0, gpu: bool = True)`; `segment_field(field, params) -> Segmentation` with `cells: np.ndarray`, `nuclei: np.ndarray`, `params_used: dict`.

- [ ] **Step 1: Write the failing tests**

```python
def test_segment_field_calls_cellpose_on_jm1_and_nucleus_planes(monkeypatch):
    seen = record_eval_calls(monkeypatch)
    segment_field(synthetic_field(), SegmentationParams())
    assert [c["plane"] for c in seen] == ["JM1", "NUCLEUS"]

def test_params_used_records_everything_needed_to_reproduce(monkeypatch):
    stub_cellpose(monkeypatch)
    used = segment_field(synthetic_field(), SegmentationParams(diameter=30)).params_used
    assert used["model"] == "cpsam" and used["diameter"] == 30
    assert "cellpose_version" in used

def test_falls_back_to_cpu_when_gpu_unavailable(monkeypatch):
    stub_no_gpu(monkeypatch)
    assert segment_field(synthetic_field(), SegmentationParams(gpu=True)).params_used["gpu"] is False
```

Cellpose is stubbed throughout; no model weights in tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_segment.py -v` — Expected: FAIL

- [ ] **Step 3: Implement `segment_field` in `src/junctionmapper/segment/cellpose.py`**

Wrap `cellpose.models.CellposeModel`. Keep the Cellpose import inside the function so the rest of the library imports without it. Record the resolved parameters plus `cellpose.version` into `params_used` for provenance.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_segment.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/segment tests/test_segment.py
git commit -m "feat: Cellpose-SAM segmentation driver with provenance capture"
```

---

### Task 13: Pipeline assembly and tidy tables

**Files:**
- Create: `src/junctionmapper/pipeline.py`, `src/junctionmapper/tables.py`
- Test: `tests/test_tables.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–12.
- Produces: `measure_field(field, segmentation, cfg) -> FieldResult` with `interfaces: pd.DataFrame`, `cells: pd.DataFrame`, `nuclei: pd.DataFrame`, `flags: pd.DataFrame`; `write_tables(results, outdir) -> None` writing `interfaces`, `cells`, `nuclei`, `flags` as both `.parquet` and `.csv`.

- [ ] **Step 1: Write the failing tests**

```python
def test_interface_rows_carry_role_marker_and_method():
    df = measure_field(f, seg, cfg).interfaces
    assert {"field_id", "cell_id", "neighbour_id", "corner_method", "role", "marker"} <= set(df.columns)
    assert set(df["marker"]) == {"Na/K-ATPase", "p120"}

def test_every_metric_has_both_pixel_and_micron_columns():
    df = measure_field(f, seg, cfg).interfaces
    assert "interface_contour_px" in df and "interface_contour_um" in df

def test_both_corner_methods_appear_for_every_measured_cell():
    df = measure_field(f, seg, cfg).interfaces
    assert set(df["corner_method"]) == {"dp", "tcj"}

def test_isolated_cell_appears_in_cells_with_zero_interfaces():
    res = measure_field(field_with_isolated_cell(), seg, cfg)
    assert res.cells.loc[res.cells.cell_id == 3, "n_interfaces"].item() == 0

def test_border_touching_cells_are_flagged_and_absent_from_interfaces():
    res = measure_field(field_with_border_cell(), seg, cfg)
    assert 1 in set(res.flags.loc[res.flags.border_touching, "cell_id"])
    assert 1 not in set(res.interfaces["cell_id"])

def test_keys_join_interfaces_to_cells_to_nuclei():
    res = measure_field(f, seg, cfg)
    assert not res.interfaces.merge(res.cells, on=["field_id", "cell_id"]).empty
    assert not res.cells.merge(res.nuclei, on=["field_id", "cell_id"]).empty

def test_threshold_used_is_recorded_per_field_and_role():
    assert "threshold" in measure_field(f, seg, cfg).interfaces.columns
```

The join test is the spec's downstream requirement; the marker column is the CDH1/NaK pooling guard.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tables.py tests/test_pipeline.py -v` — Expected: FAIL

- [ ] **Step 3: Implement both modules**

`measure_field` iterates cells, skips border-touching and degenerate ones into `flags`, runs both corner methods, and measures JM1 and JM2 with the same functions. Long format: one row per (field, cell, interface, corner_method, role). µm columns derive from `field.pixel_size_um`, areas by its square.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tables.py tests/test_pipeline.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/pipeline.py src/junctionmapper/tables.py tests/test_tables.py tests/test_pipeline.py
git commit -m "feat: field measurement pipeline and tidy table output"
```

---

### Task 14: HTML report

**Files:**
- Create: `src/junctionmapper/report/build.py`, `src/junctionmapper/report/template.html`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `FieldResult` list, `RunConfig`, segmentation `params_used`.
- Produces: `build_report(results, cfg, provenance, out: Path) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
def test_report_is_a_single_self_contained_file(tmp_path):
    html = build_report(results, cfg, prov, tmp_path / "r.html").read_text()
    assert "<img src=\"data:image/png;base64," in html
    assert "http://" not in html and "https://cdn" not in html

def test_qc_section_reports_counts_and_exclusion_reasons(tmp_path):
    html = build_report(results, cfg, prov, tmp_path / "r.html").read_text()
    for s in ("cells found", "border", "isolated", "corner methods disagree"):
        assert s in html.lower()

def test_report_shows_distributions_not_only_means(tmp_path):
    html = build_report(results, cfg, prov, tmp_path / "r.html").read_text()
    assert "distribution" in html.lower()

def test_provenance_section_records_markers_model_and_versions(tmp_path):
    html = build_report(results, cfg, prov, tmp_path / "r.html").read_text()
    for s in ("Na/K-ATPase", "cpsam", "junctionmapper"):
        assert s in html

def test_varying_threshold_across_fields_is_called_out(tmp_path):
    html = build_report(results_with_varying_t, cfg, prov, tmp_path / "r.html").read_text()
    assert "threshold varied" in html.lower()
```

The last test is the spec's comparability warning.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_report.py -v` — Expected: FAIL

- [ ] **Step 3: Implement `build_report` and the Jinja2 template**

Three sections: QC, Quantification, Provenance. Matplotlib figures embedded as base64 PNG — no external assets. Quantification shows per-parameter distributions pooled and per-field; histograms or strip plots, never bare condition means.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/report tests/test_report.py
git commit -m "feat: self-contained HTML QC and quantification report"
```

---

### Task 15: CLI

**Files:**
- Create: `src/junctionmapper/cli.py`
- Modify: `pyproject.toml` (console script entry point)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 1–14.
- Produces: `junctionmapper run --config cfg.yaml --input PATH --out DIR [--labels DIR]`, and `junctionmapper segment --config cfg.yaml --input PATH --out DIR` which stops after writing labels.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_writes_tables_and_report(tmp_path):
    assert CliRunner().invoke(app, ["run", "--config", cfg_yaml, "--input", tif, "--out", str(tmp_path)]).exit_code == 0
    for f in ("interfaces.parquet", "interfaces.csv", "cells.parquet", "report.html"):
        assert (tmp_path / f).exists()

def test_run_with_labels_dir_skips_segmentation(tmp_path, monkeypatch):
    forbid_cellpose(monkeypatch)
    assert CliRunner().invoke(app, ["run", "--config", cfg_yaml, "--input", tif,
                                    "--labels", str(label_dir), "--out", str(tmp_path)]).exit_code == 0

def test_segment_writes_labels_and_no_tables(tmp_path):
    CliRunner().invoke(app, ["segment", "--config", cfg_yaml, "--input", tif, "--out", str(tmp_path)])
    assert (tmp_path / "labels").exists()
    assert not (tmp_path / "interfaces.parquet").exists()

def test_missing_calibration_exits_nonzero_with_actionable_message(tmp_path):
    r = CliRunner().invoke(app, ["run", "--config", cfg_no_pixel_size, "--input", plain_tif, "--out", str(tmp_path)])
    assert r.exit_code != 0 and "pixel_size_um" in r.output
```

`--labels` is the seam the napari plugin will write to, and the resume path for re-measuring approved labels without re-running Cellpose.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v` — Expected: FAIL

- [ ] **Step 3: Implement the Typer app and register the `junctionmapper` console script**

YAML config loads into `RunConfig`. Labels persist as one `.npy` per field per role, named by `field_id`, alongside a JSON provenance sidecar.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/junctionmapper/cli.py pyproject.toml tests/test_cli.py
git commit -m "feat: CLI for segmentation, measurement, and reporting"
```

---

### Task 16: Java spot-check harness

**Files:**
- Create: `tests/validation/test_java_spotcheck.py`, `tests/validation/README.md`, `scripts/edge_map_from_labels.py`
- Test: as above

**Interfaces:**
- Produces: `labels_to_edge_map(labels: np.ndarray, out: Path) -> Path` writing a 1-bit PNG the Java's "Load Existing" accepts; a pytest module comparing recorded Java values against Python, skipped when reference data is absent.

- [ ] **Step 1: Write the failing tests**

```python
def test_edge_map_png_is_one_bit_and_boundaries_are_set(tmp_path):
    p = labels_to_edge_map(two_cells_sharing_a_vertical_edge(), tmp_path / "e.png")
    assert Image.open(p).mode == "1"
    assert np.array(Image.open(p)).sum() > 0

@pytest.mark.skipif(not REFERENCE.exists(), reason="no recorded Java reference values")
def test_integer_metrics_match_java_exactly():
    for row in load_reference():
        got = measure_one(row)
        assert got["interface_area_px"] == row["interface_area_px"]
        assert got["marker_intensity"] == row["marker_intensity"]

@pytest.mark.skipif(not REFERENCE.exists(), reason="no recorded Java reference values")
def test_ratio_metrics_match_java_within_tolerance():
    for row in load_reference():
        assert measure_one(row)["linearity_index"] == pytest.approx(row["linearity_index"], rel=1e-9)

@pytest.mark.skipif(not REFERENCE.exists(), reason="no recorded Java reference values")
def test_coverage_divergence_is_recorded_not_asserted_equal():
    report_divergence("coverage_index", [...])     # writes tests/validation/divergence.md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/validation -v` — Expected: FAIL on the first test, the rest skipped.

- [ ] **Step 3: Implement the converter and the harness**

`tests/validation/README.md` documents the manual procedure: build the Java with `./build.sh run`, load an example TIFF, load the generated edge map at step 1, record the DP epsilon used, click through steps 4–7 for a handful of cells, and transcribe the `.xls` into `reference.csv`. Reference values use the Douglas–Peucker corner path only; the tricellular method has no Java counterpart.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/validation -v` — Expected: PASS (1 passed, 3 skipped without reference data)

- [ ] **Step 5: Commit**

```bash
git add tests/validation scripts/edge_map_from_labels.py
git commit -m "test: Java spot-check harness with edge-map injection"
```

---

## Self-Review

**Spec coverage.** Channel roles → T1; OME I/O and units → T2; thresholds → T3, T13; segmentation → T12; both corner methods → T5, T6, T7; the nine parameters → T8, T9, T10; nuclei → T11; tables and keys → T13; report → T14; batch → T15; validation → T16 plus the analytic fixtures throughout. Deferred to Plan 2, as scoped: the napari plugin, the review queue, bulk approve, resumable review state, and per-field threshold review. Task 15's `--labels` is the seam those will write to.

**Review Focus coverage.** Isolated cells T7 + T13; border-touching T7 + T13; missing calibration and channel names T1, T2, T15; multinucleate and orphan nuclei T11; degenerate labels T4.

**Type consistency.** `Role` is the single enum from T1 through T15. `Field.planes` is keyed by `Role` everywhere. `Interface.pixels` is `(N, 2)` int `(row, col)` in T7–T10. `trace_boundary` returns that same shape in T4. Metric functions take `pixels` first and return floats; `band_metrics` is the only one returning a dataclass.

**Proportion.** Sixteen tasks against a spec of comparable length. Code blocks are test assertions and signatures; no function bodies except the two-line connectivity rule in T10, which the signature does not determine and which is the spec's one deliberate divergence.
