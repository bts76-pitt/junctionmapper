"""Spot-check harness: compare Junction Mapper (Java) numbers, recorded by hand, against ours.

The procedure a person follows is in ``tests/validation/README.md``. This module holds

* the edge-map converter tests (always run),
* the loader / validator for ``tests/validation/reference.csv`` and the comparison logic,
* self-tests of that logic on a synthetic reference (always run; the "Java" numbers there are
  OUR numbers, so they prove the plumbing and say nothing about the port), and
* the four comparison tests against the real recorded reference (skipped, loudly, until
  ``reference.csv`` exists).

What is asserted, and how
-------------------------
exact (no tolerance)   interface_area_px, marker_area_px, marker_intensity: pixel counts and
                       integer intensity sums. If the port is right these are identical.
rel 1e-9               interface_contour_px, straight_line_length_px, linearity_index,
                       junction_contour_px, junction_straight_line_px.
                       linearity_index is compared as ``1 / java_distance_ratio`` against ours
                       (README.md, recorded divergence 1: ours is the reciprocal of the Java's
                       ``getDistanceRatio``).
recorded only          fragmented_contour_px, coverage_index (recorded divergence 3: the Java's
                       JM1 walk uses an OR connectivity rule that bridges gaps; its JM2 walk
                       over-counts). Written to ``divergence.md``, never asserted equal.

Reference cells must be interior (recorded divergence 2: the Java clips the dilation band at the
image border). A row whose band would touch the Java's dead zone, whose cell touches the border
(not measured by us), or whose operator-supplied ``interior`` flag is ``no`` is SKIPPED with a
reason, not failed.

Only the Douglas-Peucker corner path has a Java counterpart; a ``tcj`` row is rejected.
"""
from __future__ import annotations

import csv
import importlib.util
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from junctionmapper.config import ChannelSpec, Role, RunConfig
from junctionmapper.geometry.interfaces import extract_interfaces
from junctionmapper.io.reader import read_fields
from junctionmapper.measure import interface as mi
from junctionmapper.pipeline import measure_field
from junctionmapper.segment.cellpose import Segmentation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REFERENCE = HERE / "reference.csv"
DIVERGENCE_MD = HERE / "divergence.md"
SCRIPT = REPO / "scripts" / "edge_map_from_labels.py"
README = HERE / "README.md"

_MOD = "edge_map_from_labels"
if _MOD in sys.modules:
    edge_map = sys.modules[_MOD]
else:
    _spec = importlib.util.spec_from_file_location(_MOD, SCRIPT)
    edge_map = importlib.util.module_from_spec(_spec)
    sys.modules[_MOD] = edge_map
    _spec.loader.exec_module(edge_map)
labels_to_edge_map = edge_map.labels_to_edge_map


# =============================================================================================
# 1. Edge-map converter
# =============================================================================================

def two_cells_sharing_a_vertical_edge() -> np.ndarray:
    lab = np.zeros((12, 16), np.int32)
    lab[2:10, 2:8] = 1
    lab[2:10, 8:14] = 2
    return lab


def staircase_cells() -> np.ndarray:
    """Two cells meeting along a diagonal staircase, with background all round.

    A one-pixel-thick or one-sided line leaks here: the Java flood fill is 8-connected, so
    it slips between two pixels that touch only at a corner.
    """
    lab = np.zeros((24, 24), np.int32)
    for r in range(4, 20):
        for c in range(4, 20):
            lab[r, c] = 1 if c - 4 <= r - 4 else 2
    return lab


def _boundary_by_brute_force(lab: np.ndarray) -> np.ndarray:
    """Independent statement of the rule: set iff a 4-neighbour inside the image differs."""
    h, w = lab.shape
    out = np.zeros(lab.shape, bool)
    for r in range(h):
        for c in range(w):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and lab[rr, cc] != lab[r, c]:
                    out[r, c] = True
    return out


def test_edge_map_png_is_one_bit_and_boundaries_are_set(tmp_path):
    p = labels_to_edge_map(two_cells_sharing_a_vertical_edge(), tmp_path / "e.png")
    assert Image.open(p).mode == "1"
    assert np.array(Image.open(p)).sum() > 0


def test_edge_map_png_is_a_one_bit_grayscale_png_on_disk(tmp_path):
    """Pillow's mode is not the file: check the IHDR the Java's ImageIO will see."""
    p = labels_to_edge_map(two_cells_sharing_a_vertical_edge(), tmp_path / "e.png")
    raw = p.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert raw[12:16] == b"IHDR"
    width, height, bit_depth, colour_type = struct.unpack(">IIBB", raw[16:26])
    assert (width, height) == (16, 12)       # same size as the image the Java loaded
    assert bit_depth == 1 and colour_type == 0


@pytest.mark.parametrize("labels", [two_cells_sharing_a_vertical_edge(), staircase_cells()],
                         ids=["vertical", "staircase"])
def test_edge_map_round_trips_to_the_label_boundaries(tmp_path, labels):
    p = labels_to_edge_map(labels, tmp_path / "e.png")
    got = np.array(Image.open(p))
    assert got.dtype == bool and got.shape == labels.shape
    expected = _boundary_by_brute_force(labels)
    assert (got == expected).all(), np.argwhere(got != expected)[:5]


def test_edge_map_hand_checked_pixels(tmp_path):
    got = np.array(Image.open(labels_to_edge_map(two_cells_sharing_a_vertical_edge(),
                                                 tmp_path / "e.png")))
    # the line between the two cells is two pixels wide: the last column of cell 1 and the
    # first column of cell 2
    assert got[5, 7] and got[5, 8]
    assert not got[5, 6] and not got[5, 9]           # cell interiors are left clear
    assert not got[5, 4] and not got[5, 11]
    assert got[5, 2] and got[5, 13]                  # cell against background: cell side...
    assert got[5, 1] and got[5, 14]                  # ...and background side
    assert not got[5, 0] and not got[0, 0]           # far background is left clear
    assert got[2, 4] and got[9, 4] and got[1, 4] and got[10, 4]   # top/bottom rows, both sides


def test_edge_map_barrier_confines_an_8_connected_flood_fill(tmp_path):
    """Replays the Java's step 4 (8-connected flood of unset pixels) on the PNG.

    Every flooded region must be exactly one label's interior; none may leak into a
    neighbouring cell or the background.
    """
    for labels in (two_cells_sharing_a_vertical_edge(), staircase_cells()):
        edge = np.array(Image.open(labels_to_edge_map(labels, tmp_path / "e.png")))
        java_view = edge_map.java_view_labels(labels)
        comp, n = ndimage.label(~edge, structure=np.ones((3, 3)))
        for lab_id in np.unique(labels[labels > 0]):
            seed = np.argwhere(java_view == lab_id)[0]
            region = comp == comp[tuple(seed)]
            assert (region == (java_view == lab_id)).all(), f"label {lab_id} leaked or was cut"
        # and the labelled cells are never merged with each other
        assert len({comp[tuple(np.argwhere(java_view == i)[0])]
                    for i in np.unique(labels[labels > 0])}) == len(np.unique(labels[labels > 0]))


def test_java_view_is_the_labels_minus_the_barrier():
    lab = two_cells_sharing_a_vertical_edge()
    view = edge_map.java_view_labels(lab)
    assert (view[edge_map.edge_map_pixels(lab)] == 0).all()
    assert (view[~edge_map.edge_map_pixels(lab)] == lab[~edge_map.edge_map_pixels(lab)]).all()
    assert view[3:9, 3:7].min() == 1 and view[3:9, 9:13].min() == 2     # eroded by one pixel
    assert (view == 1).sum() == 6 * 4 and (view == 2).sum() == 6 * 4


@pytest.mark.parametrize("bad", [
    np.zeros((3, 3, 3), np.int32), np.zeros((4, 4), np.float64),
    np.array([[0, -1], [0, 0]]), np.zeros(5, np.int32),
])
def test_edge_map_rejects_things_that_are_not_a_label_image(tmp_path, bad):
    with pytest.raises(ValueError):
        labels_to_edge_map(bad, tmp_path / "e.png")
    assert not (tmp_path / "e.png").exists()


def test_edge_map_cli_writes_png_and_overview(tmp_path):
    lab = staircase_cells()
    np.save(tmp_path / "f.JM1.npy", lab)
    rc = edge_map.main([str(tmp_path / "f.JM1.npy"), str(tmp_path / "out" / "e.png"),
                        "--overview", str(tmp_path / "out" / "ov.png")])
    assert rc == 0
    assert Image.open(tmp_path / "out" / "e.png").mode == "1"
    ov = Image.open(tmp_path / "out" / "ov.png")
    assert ov.size == (24, 24) or ov.size[0] >= 24      # overview may be scaled up for legibility


# =============================================================================================
# 2. Reference loading, validation, measurement and comparison
# =============================================================================================

class ReferenceFormatError(ValueError):
    """reference.csv exists but is not valid. This FAILS the run; it is never a skip."""


class RowSkip(Exception):
    """A valid row that cannot be compared, for a stated reason."""


REQUIRED_COLUMNS = (
    "image", "labels", "jm1_channel", "jm2_channel", "nucleus_channel",
    "cell_id", "neighbour_id", "role", "corner_method", "epsilon", "dilation_cycles",
    "threshold", "interior",
    "interface_contour_px", "straight_line_length_px", "java_distance_ratio",
    "fragmented_contour_px", "coverage_index",
    "interface_area_px", "marker_area_px", "marker_intensity",
)
OPTIONAL_COLUMNS = ("junction_contour_px", "junction_straight_line_px",
                    "java_corner", "java_paired_with", "notes")
INT_COLUMNS = ("jm1_channel", "jm2_channel", "nucleus_channel", "cell_id", "neighbour_id")
FLOAT_COLUMNS = ("interface_contour_px", "straight_line_length_px", "java_distance_ratio",
                 "fragmented_contour_px", "coverage_index", "junction_contour_px",
                 "junction_straight_line_px")
INTEGER_VALUED_METRICS = ("interface_area_px", "marker_area_px", "marker_intensity")

EXACT_METRICS = ("interface_area_px", "marker_area_px", "marker_intensity")
TOLERANCE_METRICS = ("interface_contour_px", "straight_line_length_px", "linearity_index",
                     "junction_contour_px", "junction_straight_line_px")
RECORDED_METRICS = ("fragmented_contour_px", "coverage_index")
REL_TOL = 1e-9
ABS_TOL = 1e-12

EXPLANATIONS = {
    "fragmented_contour_px": (
        "Divergence 3 (README.md): the Java's JM1 walk joins steps with an OR rule and bridges "
        "gaps; its JM2 walk also counts the step into each fragment. Ours joins only "
        "8-connected steps."),
    "coverage_index": (
        "Divergence 3 (README.md): coverage is fragmented contour / interface contour, so it "
        "inherits the connectivity difference in the fragmented contour."),
}


def _ref_error(line: int, msg: str) -> ReferenceFormatError:
    return ReferenceFormatError(f"{msg} (data row {line - 1}, line {line} of the file)")


def _parse_int(raw: str, name: str, line: int) -> int:
    try:
        f = float(raw)
    except ValueError:
        raise _ref_error(line, f"column {name!r}: {raw!r} is not a number") from None
    if not math.isfinite(f) or f != int(f):
        raise _ref_error(line, f"column {name!r}: {raw!r} must be a whole number")
    return int(f)


def _parse_float(raw: str, name: str, line: int) -> float:
    try:
        f = float(raw)
    except ValueError:
        raise _ref_error(line, f"column {name!r}: {raw!r} is not a number") from None
    if not math.isfinite(f) or f < 0:
        raise _ref_error(line, f"column {name!r}: {raw!r} must be a finite number >= 0")
    return f


def load_reference(path: Path) -> list[dict]:
    """Parse and validate ``reference.csv``; raise ReferenceFormatError on any defect.

    Schema (see README.md): one row per (cell, interface, marker role). Paths in ``image`` and
    ``labels`` are resolved relative to the CSV and must exist.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReferenceFormatError(f"cannot read {path}: {exc}") from exc
    reader = csv.reader(text.splitlines())
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration:
        raise ReferenceFormatError(f"{path} is empty; it needs the header row from README.md") from None
    dupes = sorted({h for h in header if header.count(h) > 1})
    if dupes:
        raise ReferenceFormatError(f"duplicate column(s) in header: {dupes}")
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    unknown = [c for c in header if c not in REQUIRED_COLUMNS and c not in OPTIONAL_COLUMNS]
    if missing or unknown:
        raise ReferenceFormatError(
            f"header does not match the schema in tests/validation/README.md: "
            f"missing {missing}, unknown {unknown}")

    rows: list[dict] = []
    for lineno, cells in enumerate(reader, start=2):
        if not any(c.strip() for c in cells):
            continue
        if len(cells) != len(header):
            raise _ref_error(lineno, f"has {len(cells)} fields, header has {len(header)}")
        raw = {h: c.strip() for h, c in zip(header, cells)}
        row: dict = {"_line": lineno, "_csv_dir": path.parent}
        for name in REQUIRED_COLUMNS:
            if raw[name] == "":
                raise _ref_error(lineno, f"column {name!r} is empty")
        for name in INT_COLUMNS:
            row[name] = _parse_int(raw[name], name, lineno)
        for name in ("jm1_channel", "jm2_channel", "nucleus_channel"):
            if row[name] < 0:
                raise _ref_error(lineno, f"column {name!r} must be >= 0")
        if len({row["jm1_channel"], row["jm2_channel"], row["nucleus_channel"]}) != 3:
            raise _ref_error(lineno, "jm1_channel, jm2_channel and nucleus_channel must differ")
        if row["cell_id"] < 1 or row["neighbour_id"] < 1 or row["cell_id"] == row["neighbour_id"]:
            raise _ref_error(lineno, "cell_id and neighbour_id must be distinct labels >= 1")

        row["role"] = raw["role"]
        if row["role"] not in ("JM1", "JM2"):
            raise _ref_error(lineno, f"role must be JM1 or JM2, got {raw['role']!r}")
        row["corner_method"] = raw["corner_method"].lower()
        if row["corner_method"] == "tcj":
            raise _ref_error(lineno, "corner_method 'tcj' has no Java counterpart; reference "
                                     "values are Douglas-Peucker only ('dp')")
        if row["corner_method"] != "dp":
            raise _ref_error(lineno, f"corner_method must be 'dp', got {raw['corner_method']!r}")

        row["epsilon"] = _parse_int(raw["epsilon"], "epsilon", lineno)
        if not 1 <= row["epsilon"] <= 50:
            raise _ref_error(lineno, "epsilon is the Java slider value, an integer 1..50")
        row["dilation_cycles"] = _parse_int(raw["dilation_cycles"], "dilation_cycles", lineno)
        if not 1 <= row["dilation_cycles"] <= 9:
            raise _ref_error(lineno, "dilation_cycles is the Java slider value, an integer 1..9")
        row["threshold"] = _parse_int(raw["threshold"], "threshold", lineno)
        if not 0 <= row["threshold"] <= 255:
            raise _ref_error(lineno, "threshold is the Java slider value, an integer 0..255")
        if raw["interior"].lower() not in ("yes", "no"):
            raise _ref_error(lineno, f"interior must be 'yes' or 'no', got {raw['interior']!r}")
        row["interior"] = raw["interior"].lower() == "yes"

        for name in FLOAT_COLUMNS:
            if raw.get(name, "") == "":
                row[name] = None
            else:
                row[name] = _parse_float(raw[name], name, lineno)
        if row["java_distance_ratio"] < 1.0 - 1e-9:
            raise _ref_error(lineno, f"java_distance_ratio {raw['java_distance_ratio']} is < 1: "
                                     "that is contour / straight-line (>= 1) as the Java prints "
                                     "it; did you transcribe a linearity index (<= 1) instead?")
        for name in INTEGER_VALUED_METRICS:
            row[name] = _parse_float(raw[name], name, lineno)
            if row[name] != int(row[name]):
                raise _ref_error(lineno, f"column {name!r} is a pixel count or integer sum; "
                                         f"{raw[name]!r} is not a whole number")
            row[name] = int(row[name])
        row["notes"] = raw.get("notes", "")

        for name in ("image", "labels"):
            p = (path.parent / raw[name]).resolve()
            if not p.is_file():
                raise _ref_error(lineno, f"{name} file not found: {p}")
            row[name] = p
        rows.append(row)

    if not rows:
        raise ReferenceFormatError(f"{path} has a header but no data rows")
    seen: dict[tuple, int] = {}
    per_cell_eps: dict[tuple, tuple[int, int]] = {}
    for r in rows:
        key = (r["image"], r["labels"], r["cell_id"], r["neighbour_id"], r["role"])
        if key in seen:
            raise _ref_error(r["_line"], f"duplicates line {seen[key]}: one row per "
                                         "(image, cell, neighbour, role)")
        seen[key] = r["_line"]
        ekey = (r["image"], r["cell_id"])
        if ekey in per_cell_eps and per_cell_eps[ekey][0] != r["epsilon"]:
            raise _ref_error(r["_line"], f"cell {r['cell_id']} was recorded with epsilon "
                                         f"{per_cell_eps[ekey][0]} on line {per_cell_eps[ekey][1]} "
                                         f"but {r['epsilon']} here; epsilon is set once per cell")
        per_cell_eps.setdefault(ekey, (r["epsilon"], r["_line"]))
    return rows


# ---- measurement -----------------------------------------------------------------------------

#: radius used to assign neighbours on the Java's view of the labels. The injected edge map is
#: two pixels thick between cells, so the eroded cells' boundaries are 3 px apart.
JAVA_VIEW_NEIGHBOUR_RADIUS = 3


def java_dilation_is_unclipped(pixels: np.ndarray, shape: tuple[int, int], cycles: int) -> bool:
    """True when the Java's border-clipped dilation equals the uniform dilation used here.

    The Java's ``getDilatedImage`` refuses any new pixel whose 3x3 neighbourhood leaves
    ``[1, w-2] x [1, h-2]``, so growth is dead at indices 0, 1, w-2 and w-1. If our full band
    stays inside ``[2, h-3] x [2, w-3]`` the two bands are identical (README.md divergence 2).
    """
    band = mi.dilated_band(pixels, shape, cycles)
    rows, cols = np.nonzero(band)
    h, w = shape
    return bool(rows.min() >= 2 and rows.max() <= h - 3 and cols.min() >= 2 and cols.max() <= w - 3)


def _config(row: dict) -> RunConfig:
    return RunConfig(
        channels=[ChannelSpec(role=Role.JM1, channel=row["jm1_channel"], marker="JM1"),
                  ChannelSpec(role=Role.JM2, channel=row["jm2_channel"], marker="JM2"),
                  ChannelSpec(role=Role.NUCLEUS, channel=row["nucleus_channel"], marker="nucleus")],
        pixel_size_um=1.0, dilation_px=row["dilation_cycles"],
        threshold={Role(row["role"]): float(row["threshold"])},
        epsilon_px=float(row["epsilon"]), neighbour_radius_px=JAVA_VIEW_NEIGHBOUR_RADIUS)


_MEASURED: dict[tuple, tuple] = {}


def _measure_field_for(row: dict):
    """(FieldResult, java-view labels, cfg) for the row's image/labels/epsilon/dilation/threshold."""
    key = (row["image"], row["labels"], row["jm1_channel"], row["jm2_channel"],
           row["nucleus_channel"], row["epsilon"], row["dilation_cycles"], row["role"],
           row["threshold"])
    if key not in _MEASURED:
        cfg = _config(row)
        labels = np.load(row["labels"])
        view = edge_map.java_view_labels(labels)
        field_ = next(iter(read_fields(row["image"], cfg)))
        if field_.planes[Role.JM1].shape != labels.shape:
            raise ReferenceFormatError(
                f"line {row['_line']}: image is {field_.planes[Role.JM1].shape} but the labels "
                f"are {labels.shape}")
        seg = Segmentation(cells=view, nuclei=np.zeros_like(view), params_used={})
        res = measure_field(field_, seg, cfg, epsilon=float(row["epsilon"]))
        _MEASURED[key] = (res, view, cfg)
    return _MEASURED[key]


def measure_one(row: dict) -> dict:
    """Our metrics for the interface a reference row describes.

    Measured on the Java's view of the segmentation (labels with the injected barrier removed),
    with the row's epsilon, dilation cycles and threshold. Raises RowSkip for rows that cannot
    be compared and AssertionError if the Java has an interface that we do not.
    """
    if not row["interior"]:
        raise RowSkip("operator marked the cell as not interior (Java clips its dilation band "
                      "at the image border; README.md divergence 2)")
    res, view, cfg = _measure_field_for(row)
    cid, nid, role = row["cell_id"], row["neighbour_id"], row["role"]
    fl = res.flags[res.flags.cell_id == cid]
    if fl.empty:
        raise AssertionError(f"cell {cid} is not a label in {row['labels'].name}")
    if bool(fl.iloc[0].border_touching) or bool(fl.iloc[0].degenerate):
        raise RowSkip(f"cell {cid} touches the image border or is degenerate; it is not "
                      "measured (Java clips its dilation band at the border)")
    df = res.interfaces
    sel = df[(df.cell_id == cid) & (df.neighbour_id == nid) & (df.corner_method == "dp")
             & (df.role == role)]
    if sel.empty:
        raise AssertionError(
            f"the Java reports an interface between cells {cid} and {nid} at epsilon "
            f"{row['epsilon']}, but we find none (same labels, same epsilon). Compare the "
            f"corner positions on the Java's Step 6 overlay")
    if len(sel) > 1:
        raise RowSkip(f"cells {cid} and {nid} have {len(sel)} separate Douglas-Peucker "
                      "interfaces; the row cannot say which one was transcribed")
    ours = sel.iloc[0]
    pix = next(i.pixels for i in extract_interfaces(
        view, cid, "dp", float(row["epsilon"]), cfg.tcj_min_run, cfg.neighbour_radius_px)
        if i.start_idx == ours.start_idx and i.end_idx == ours.end_idx)
    if not java_dilation_is_unclipped(pix, view.shape, row["dilation_cycles"]):
        raise RowSkip("the interface band reaches the Java's border dead zone (within 2 px of "
                      "the image edge), where its dilation is clipped (README.md divergence 2)")
    out = {m: float(ours[m]) for m in
           EXACT_METRICS + TOLERANCE_METRICS + RECORDED_METRICS if m in ours.index}
    if role == "JM1":
        out["junction_contour_px"] = float(ours["junction_contour_px"])
        out["junction_straight_line_px"] = float(ours["junction_straight_line_px"])
    return out


# ---- comparison ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Comparison:
    line: int
    image: str
    cell_id: int
    neighbour_id: int
    role: str
    epsilon: int
    metric: str
    java: float
    ours: float

    @property
    def abs_diff(self) -> float:
        return abs(self.ours - self.java)

    @property
    def rel_diff(self) -> float:
        if self.abs_diff == 0 or (math.isnan(self.ours) and math.isnan(self.java)):
            return 0.0
        return self.abs_diff / abs(self.java) if self.java else math.inf

    @property
    def agrees(self) -> bool:
        if math.isnan(self.ours) or math.isnan(self.java):
            return math.isnan(self.ours) and math.isnan(self.java)
        if self.metric in EXACT_METRICS:
            return self.ours == self.java
        return math.isclose(self.ours, self.java, rel_tol=REL_TOL, abs_tol=ABS_TOL)


@dataclass
class CompareResult:
    comparisons: list[Comparison] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)     # (csv line, reason)
    errors: list[str] = field(default_factory=list)

    def mismatches(self, metrics) -> list[Comparison]:
        return [c for c in self.comparisons if c.metric in metrics and not c.agrees]


def java_values(row: dict) -> dict:
    """The row's Java numbers keyed by OUR metric names.

    ``linearity_index`` is the reciprocal of the Java's ``getDistanceRatio`` (README.md
    divergence 1), so it is converted here: we compare ``1 / java_distance_ratio`` with our
    linearity_index. A ratio of exactly 1 maps to 1, a straight interface.
    """
    vals = {m: row[m] for m in EXACT_METRICS + RECORDED_METRICS
            + ("interface_contour_px", "straight_line_length_px")}
    vals["linearity_index"] = 1.0 / row["java_distance_ratio"]
    if row["role"] == "JM1":
        for m in ("junction_contour_px", "junction_straight_line_px"):
            if row[m] is not None:
                vals[m] = row[m]
    return {k: float(v) for k, v in vals.items()}


def compare_reference(rows: list[dict]) -> CompareResult:
    result = CompareResult()
    for row in rows:
        try:
            ours = measure_one(row)
        except RowSkip as why:
            result.skipped.append((row["_line"], str(why)))
            continue
        except AssertionError as why:
            result.errors.append(f"line {row['_line']}: {why}")
            continue
        for metric, java in java_values(row).items():
            if metric not in ours:
                continue
            result.comparisons.append(Comparison(
                line=row["_line"], image=row["image"].stem, cell_id=row["cell_id"],
                neighbour_id=row["neighbour_id"], role=row["role"], epsilon=row["epsilon"],
                metric=metric, java=java, ours=ours[metric]))
    return result


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.10g}"


def report_divergence(metrics, comparisons, path: Path | None = None, skipped=()) -> Path:
    """Write ``divergence.md``: one row per recorded comparison, with the divergence it falls under.

    ``metrics`` are the recorded-only metrics to tabulate. All of ``comparisons`` feed the
    summary of asserted metrics. The file has no timestamp, so rerunning on the same reference
    reproduces it byte for byte.
    """
    path = DIVERGENCE_MD if path is None else Path(path)
    comparisons = list(comparisons)
    metrics = tuple(metrics)
    by_line_metric = {(c.line, c.role, c.cell_id, c.neighbour_id, c.metric): c for c in comparisons}
    out = [
        "# Java spot-check: where the Python port and the original differ",
        "",
        "Generated by `tests/validation/test_java_spotcheck.py` from `reference.csv`; do not edit.",
        "Java numbers are hand-transcribed from the original application's spreadsheets.",
        "Divergence numbers refer to the section of the repository `README.md` that records them.",
        "",
        "## Asserted metrics",
        "",
        "Exact metrics must be identical; the others must agree to a relative tolerance of "
        f"{REL_TOL:g}. `linearity_index` is `1 / java_distance_ratio`.",
        "",
        "| metric | rule | rows | worst abs diff | worst rel diff | result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for m in EXACT_METRICS + TOLERANCE_METRICS:
        cs = [c for c in comparisons if c.metric == m]
        if not cs:
            continue
        rule = "exact" if m in EXACT_METRICS else f"rel {REL_TOL:g}"
        ok = all(c.agrees for c in cs)
        out.append(f"| {m} | {rule} | {len(cs)} | {_fmt(max(c.abs_diff for c in cs))} | "
                   f"{_fmt(max(c.rel_diff for c in cs))} | {'agree' if ok else 'DISAGREE'} |")
    for m in metrics:
        cs = sorted((c for c in comparisons if c.metric == m),
                    key=lambda c: (c.line, c.role))
        out += ["", f"## Recorded, not asserted: `{m}`", "", EXPLANATIONS.get(m, ""), "",
                "| line | image | cell | neighbour | role | eps | Java | ours | abs diff | rel diff | explained by |",
                "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|"]
        if not cs:
            out.append("| | no comparable rows | | | | | | | | | |")
        for c in cs:
            if c.agrees:
                why = "agrees"
            elif m == "fragmented_contour_px":
                why = "divergence 3"
            else:
                frag = by_line_metric.get((c.line, c.role, c.cell_id, c.neighbour_id,
                                           "fragmented_contour_px"))
                why = ("divergence 3 (via fragmented contour)" if frag is not None
                       and not frag.agrees else "UNEXPLAINED: fragmented contour agrees")
            rel = "n/a (Java is 0)" if math.isinf(c.rel_diff) else _fmt(c.rel_diff)
            out.append(f"| {c.line} | {c.image} | {c.cell_id} | {c.neighbour_id} | {c.role} | "
                       f"{c.epsilon} | {_fmt(c.java)} | {_fmt(c.ours)} | {_fmt(c.abs_diff)} | "
                       f"{rel} | {why} |")
    if skipped:
        out += ["", "## Rows not compared", "", "| line | reason |", "|---:|---|"]
        out += [f"| {ln} | {why} |" for ln, why in sorted(skipped)]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


# ---- skip / fail policy ----------------------------------------------------------------------

def reference_rows_or_outcome(path: Path | None = None) -> list[dict]:
    """The validated rows; SKIP if the file is absent, FAIL if it exists but is malformed."""
    path = REFERENCE if path is None else Path(path)
    if not path.exists():
        pytest.skip(f"no recorded Java reference values: {path} does not exist; follow "
                    f"{README} to record some")
    try:
        return load_reference(path)
    except ReferenceFormatError as exc:
        pytest.fail(f"{path} exists but is not a valid reference file: {exc}", pytrace=False)


_RESULT_CACHE: dict[Path, CompareResult] = {}


def recorded_result(path: Path | None = None) -> CompareResult:
    path = REFERENCE if path is None else Path(path)
    rows = reference_rows_or_outcome(path)
    if path not in _RESULT_CACHE:
        _RESULT_CACHE[path] = compare_reference(rows)
    result = _RESULT_CACHE[path]
    if result.errors:
        pytest.fail("the Java has interfaces we cannot find:\n  " + "\n  ".join(result.errors),
                    pytrace=False)
    if not result.comparisons:
        pytest.skip("every reference row was skipped: "
                    + "; ".join(f"line {ln}: {why}" for ln, why in result.skipped))
    if result.skipped:
        import warnings
        warnings.warn("reference rows not compared: "
                      + "; ".join(f"line {ln}: {why}" for ln, why in result.skipped), stacklevel=2)
    return result


def _describe(cs: list[Comparison]) -> str:
    return "\n".join(f"line {c.line} cell {c.cell_id}|{c.neighbour_id} {c.role} {c.metric}: "
                     f"Java {_fmt(c.java)} vs ours {_fmt(c.ours)}" for c in cs)


# =============================================================================================
# 3. Tests against the real recorded reference (skipped until reference.csv exists)
# =============================================================================================

def test_reference_csv_is_well_formed_when_present():
    reference_rows_or_outcome()


def test_integer_metrics_match_java_exactly():
    result = recorded_result()
    bad = result.mismatches(EXACT_METRICS)
    assert not bad, "integer metrics must be identical for interior cells:\n" + _describe(bad)


def test_ratio_metrics_match_java_within_tolerance():
    result = recorded_result()
    bad = result.mismatches(TOLERANCE_METRICS)
    assert not bad, f"metrics differ beyond rel={REL_TOL}:\n" + _describe(bad)


def test_coverage_divergence_is_recorded_not_asserted_equal():
    result = recorded_result()
    path = report_divergence(RECORDED_METRICS, result.comparisons, skipped=result.skipped)
    text = path.read_text(encoding="utf-8")
    for m in RECORDED_METRICS:
        assert f"`{m}`" in text
    # deliberately no assertion that Java == ours for these metrics


# =============================================================================================
# 4. Self-tests of the harness (always run). The "Java" numbers here are OUR numbers.
# =============================================================================================

def synthetic_labels() -> np.ndarray:
    """60x60. A (left) and B (right) share a boundary that jogs by one pixel; C sits above A and
    touches the top border; D is a small interior cell below B."""
    lab = np.zeros((60, 60), np.int32)
    lab[15:45, 15:30] = 1
    lab[15:45, 30:45] = 2
    lab[25:45, 30] = 1                  # the jog: A owns column 30 for the lower rows
    lab[0:15, 15:30] = 3
    lab[45:55, 35:44] = 4
    return lab


def write_synthetic(tmp_path: Path) -> tuple[Path, Path]:
    import tifffile
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(60, 60, 3), dtype=np.uint8)
    tif = tmp_path / "synthetic.tif"
    tifffile.imwrite(tif, img, photometric="rgb")
    npy = tmp_path / "synthetic.JM1.npy"
    np.save(npy, synthetic_labels())
    return tif, npy


HEADER = list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)


def _row_from_ours(tif, npy, cell, nb, role, tmp_path, *, epsilon=2, cycles=2, threshold=100,
                   interior="yes"):
    """A reference row whose 'Java' numbers are produced by our own pipeline."""
    proto = {"image": tif.resolve(), "labels": npy.resolve(), "jm1_channel": 0, "jm2_channel": 1,
             "nucleus_channel": 2, "cell_id": cell, "neighbour_id": nb, "role": role,
             "epsilon": epsilon, "dilation_cycles": cycles, "threshold": threshold,
             "interior": True, "_line": 0}
    ours = measure_one(proto)
    lin = ours["linearity_index"]
    d = {"image": tif.name, "labels": npy.name, "jm1_channel": 0, "jm2_channel": 1,
         "nucleus_channel": 2, "cell_id": cell, "neighbour_id": nb, "role": role,
         "corner_method": "dp", "epsilon": epsilon, "dilation_cycles": cycles,
         "threshold": threshold, "interior": interior,
         "interface_contour_px": repr(ours["interface_contour_px"]),
         "straight_line_length_px": repr(ours["straight_line_length_px"]),
         "java_distance_ratio": repr(1.0 / lin),
         "fragmented_contour_px": repr(ours["fragmented_contour_px"] + 7.0),   # Java differs
         "coverage_index": repr(ours["coverage_index"] + 0.25),                 # Java differs
         "interface_area_px": int(ours["interface_area_px"]),
         "marker_area_px": int(ours["marker_area_px"]),
         "marker_intensity": int(ours["marker_intensity"]),
         "junction_contour_px": repr(ours.get("junction_contour_px", "")) if role == "JM1" else "",
         "junction_straight_line_px": repr(ours.get("junction_straight_line_px", ""))
         if role == "JM1" else "", "notes": ""}
    return d


def write_csv(path: Path, rows: list[dict], header=None):
    header = header or HEADER
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])
    return path


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synthetic")
    tif, npy = write_synthetic(tmp)
    return tmp, tif, npy


def _ab_rows(synth_):
    tmp, tif, npy = synth_
    return [_row_from_ours(tif, npy, 1, 2, "JM1", tmp, threshold=100),
            _row_from_ours(tif, npy, 1, 2, "JM2", tmp, threshold=60)]


def test_synthetic_reference_precondition_the_jogged_interface_is_not_straight(synth):
    tmp, tif, npy = synth
    row = _row_from_ours(tif, npy, 1, 2, "JM1", tmp)
    assert float(row["java_distance_ratio"]) > 1.0001      # otherwise the reciprocal is invisible
    assert row["interface_area_px"] > 0 and row["marker_area_px"] > 0


def test_harness_agrees_with_itself_on_a_consistent_reference(synth):
    tmp, tif, npy = synth
    rows = _ab_rows(synth) + [_row_from_ours(tif, npy, 1, 3, "JM1", tmp)]
    p = write_csv(tmp / "ok.csv", rows)
    result = compare_reference(load_reference(p))
    assert not result.errors and not result.skipped
    assert not result.mismatches(EXACT_METRICS + TOLERANCE_METRICS), _describe(
        result.mismatches(EXACT_METRICS + TOLERANCE_METRICS))
    metrics = {c.metric for c in result.comparisons}
    assert set(EXACT_METRICS + TOLERANCE_METRICS + RECORDED_METRICS) <= metrics
    # the planted divergences are visible to the recorded metrics and nowhere else
    assert len(result.mismatches(RECORDED_METRICS)) == 2 * len(rows)     # both planted, every row


def test_an_integer_off_by_one_is_caught(synth):
    tmp, tif, npy = synth
    for col in EXACT_METRICS:
        rows = _ab_rows(synth)[:1]
        rows[0][col] += 1
        res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
        bad = res.mismatches(EXACT_METRICS)
        assert [c.metric for c in bad] == [col]


def test_a_float_metric_off_by_more_than_the_tolerance_is_caught(synth):
    tmp, tif, npy = synth
    rows = _ab_rows(synth)[:1]
    rows[0]["interface_contour_px"] = repr(float(rows[0]["interface_contour_px"]) * (1 + 1e-6))
    res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
    assert [c.metric for c in res.mismatches(TOLERANCE_METRICS)] == ["interface_contour_px"]


def test_linearity_is_compared_as_the_reciprocal_of_the_java_ratio(synth):
    tmp, tif, npy = synth
    ok = compare_reference(load_reference(write_csv(tmp / "x.csv", _ab_rows(synth)[:1])))
    assert not ok.mismatches(("linearity_index",))
    # an operator who copied our-style linearity (<1) into the ratio column is refused at load
    rows = _ab_rows(synth)[:1]
    rows[0]["java_distance_ratio"] = repr(1.0 / float(rows[0]["java_distance_ratio"]))
    with pytest.raises(ReferenceFormatError, match="linearity"):
        load_reference(write_csv(tmp / "y.csv", rows))
    # and a Java ratio that is merely wrong (still >= 1) is a linearity mismatch
    rows = _ab_rows(synth)[:1]
    rows[0]["java_distance_ratio"] = repr(float(rows[0]["java_distance_ratio"]) + 0.01)
    bad = compare_reference(load_reference(write_csv(tmp / "z.csv", rows)))
    assert [c.metric for c in bad.mismatches(TOLERANCE_METRICS)] == ["linearity_index"]


def test_java_value_for_linearity_is_the_reciprocal_not_the_raw_ratio():
    row = {m: 1.0 for m in EXACT_METRICS + RECORDED_METRICS
           + ("interface_contour_px", "straight_line_length_px")}
    row.update(java_distance_ratio=1.25, role="JM2")
    assert java_values(row)["linearity_index"] == pytest.approx(0.8, rel=1e-12)


def test_interface_the_java_has_and_we_lack_is_an_error_not_a_skip(synth):
    tmp, tif, npy = synth
    rows = _ab_rows(synth)[:1]
    rows[0]["neighbour_id"] = 4          # A does not touch D
    res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
    assert res.errors and "we find none" in res.errors[0] and not res.comparisons


def test_border_touching_and_non_interior_rows_are_skipped_with_a_reason(synth):
    tmp, tif, npy = synth
    border = _row_from_ours(tif, npy, 1, 3, "JM1", tmp)
    border.update(cell_id=3, neighbour_id=1)             # C touches the top border
    flagged = _row_from_ours(tif, npy, 1, 2, "JM1", tmp, interior="no")
    res = compare_reference(load_reference(write_csv(tmp / "x.csv", [border, flagged])))
    assert not res.errors and not res.comparisons
    reasons = dict(res.skipped)
    assert "touches the image border" in reasons[2]
    assert "not interior" in reasons[3]


def test_band_reaching_the_java_dead_zone_is_skipped():
    pix = np.array([[10, 3], [11, 3]])
    assert not java_dilation_is_unclipped(pix, (40, 40), 2)       # band reaches column 1
    assert java_dilation_is_unclipped(np.array([[10, 4], [11, 4]]), (40, 40), 2)
    assert not java_dilation_is_unclipped(np.array([[10, 36], [11, 36]]), (40, 40), 2)
    assert java_dilation_is_unclipped(np.array([[10, 35], [11, 35]]), (40, 40), 2)


def test_each_cell_is_measured_with_the_epsilon_it_was_recorded_with(synth):
    tmp, tif, npy = synth
    a = _row_from_ours(tif, npy, 1, 2, "JM1", tmp, epsilon=2)
    b = _row_from_ours(tif, npy, 1, 2, "JM1", tmp, epsilon=1)
    # same interface recorded at a different epsilon: still consistent with itself
    for row in (a, b):
        res = compare_reference(load_reference(write_csv(tmp / "x.csv", [row])))
        assert not res.mismatches(EXACT_METRICS + TOLERANCE_METRICS)
    # rows for one cell with two epsilons are refused
    with pytest.raises(ReferenceFormatError, match="epsilon is set once per cell"):
        load_reference(write_csv(tmp / "y.csv", [a, dict(b, role="JM2")]))


# ---- measure_one against an independent oracle ------------------------------------------------
# The synthetic "Java" values above come from measure_one itself, so they prove the plumbing but
# would not notice measure_one measuring the wrong thing. These do.

def _oracle(tif, npy, cell, nb, role, eps, cycles, thr):
    """Band metrics restated from their definitions with scipy/tifffile, no junctionmapper.measure."""
    import tifffile
    view = edge_map.java_view_labels(np.load(npy))
    found = [i for i in extract_interfaces(view, cell, "dp", float(eps), 3, JAVA_VIEW_NEIGHBOUR_RADIUS)
             if i.neighbour_id == nb]
    assert len(found) == 1
    pix = found[0].pixels
    plane = tifffile.imread(tif)[..., 0 if role == "JM1" else 1].astype(np.float64)
    mask = np.zeros(view.shape, bool)
    mask[pix[:, 0], pix[:, 1]] = True
    band = ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=cycles)
    on = band & (plane > thr)
    return {"interface_area_px": float(band.sum()), "marker_area_px": float(on.sum()),
            "marker_intensity": float(plane[on].sum())}


@pytest.mark.parametrize("role, thr, cycles", [("JM1", 100, 2), ("JM2", 60, 2), ("JM1", 30, 3),
                                                ("JM2", 200, 1)])
def test_measure_one_agrees_with_an_independent_oracle(synth, role, thr, cycles):
    tmp, tif, npy = synth
    row = {"image": tif.resolve(), "labels": npy.resolve(), "jm1_channel": 0, "jm2_channel": 1,
           "nucleus_channel": 2, "cell_id": 1, "neighbour_id": 2, "role": role, "epsilon": 2,
           "dilation_cycles": cycles, "threshold": thr, "interior": True, "_line": 0}
    want = _oracle(tif, npy, 1, 2, role, 2, cycles, thr)
    got = measure_one(row)
    assert {k: got[k] for k in want} == want
    assert want["marker_area_px"] > 0           # the check has something to say


def test_measure_one_works_on_the_java_view_of_the_labels(synth):
    tmp, tif, npy = synth
    row = {"image": tif.resolve(), "labels": npy.resolve(), "jm1_channel": 0, "jm2_channel": 1,
           "nucleus_channel": 2, "cell_id": 1, "neighbour_id": 2, "role": "JM1", "epsilon": 2,
           "dilation_cycles": 2, "threshold": 100, "interior": True, "_line": 0}
    res, view, cfg = _measure_field_for(row)
    raw = np.load(npy)
    assert (view == edge_map.java_view_labels(raw)).all()
    area = res.cells.set_index("cell_id").area_px
    assert area[1] == (view == 1).sum() < (raw == 1).sum()
    assert cfg.neighbour_radius_px == JAVA_VIEW_NEIGHBOUR_RADIUS


# ---- the four reference tests, run against a synthetic reference ------------------------------

def _use_reference(monkeypatch, tmp_path, path):
    monkeypatch.setattr(sys.modules[__name__], "REFERENCE", path)
    monkeypatch.setattr(sys.modules[__name__], "DIVERGENCE_MD", tmp_path / "divergence.md")
    _RESULT_CACHE.clear()


def test_the_comparison_tests_pass_and_write_divergence_md_on_a_consistent_reference(
        synth, monkeypatch, tmp_path):
    tmp, tif, npy = synth
    _use_reference(monkeypatch, tmp_path, write_csv(tmp / "good.csv", _ab_rows(synth)))
    test_reference_csv_is_well_formed_when_present()
    test_integer_metrics_match_java_exactly()
    test_ratio_metrics_match_java_within_tolerance()
    test_coverage_divergence_is_recorded_not_asserted_equal()
    text = (tmp_path / "divergence.md").read_text()
    assert "fragmented_contour_px" in text and "coverage_index" in text


@pytest.mark.parametrize("edit, which", [
    (lambda r: r[0].update(interface_area_px=r[0]["interface_area_px"] + 1),
     test_integer_metrics_match_java_exactly),
    (lambda r: r[1].update(marker_intensity=r[1]["marker_intensity"] - 1),
     test_integer_metrics_match_java_exactly),
    (lambda r: r[0].update(marker_area_px=r[0]["marker_area_px"] + 1),
     test_integer_metrics_match_java_exactly),
    (lambda r: r[0].update(java_distance_ratio=repr(float(r[0]["java_distance_ratio"]) + 0.05)),
     test_ratio_metrics_match_java_within_tolerance),
    (lambda r: r[1].update(straight_line_length_px=repr(float(r[1]["straight_line_length_px"]) + 1)),
     test_ratio_metrics_match_java_within_tolerance),
    (lambda r: r[0].update(junction_contour_px=repr(float(r[0]["junction_contour_px"]) + 1)),
     test_ratio_metrics_match_java_within_tolerance),
])
def test_each_comparison_test_fails_on_a_reference_that_disagrees(
        synth, monkeypatch, tmp_path, edit, which):
    tmp, tif, npy = synth
    rows = _ab_rows(synth)
    edit(rows)
    _use_reference(monkeypatch, tmp_path, write_csv(tmp / "badvals.csv", rows))
    with pytest.raises(AssertionError):
        which()


def test_the_recorded_divergence_test_does_not_fail_when_fragmentation_differs(
        synth, monkeypatch, tmp_path):
    tmp, tif, npy = synth
    rows = _ab_rows(synth)
    rows[0]["fragmented_contour_px"] = "0.0"
    rows[0]["coverage_index"] = "0.0"
    _use_reference(monkeypatch, tmp_path, write_csv(tmp / "frag.csv", rows))
    test_coverage_divergence_is_recorded_not_asserted_equal()


@pytest.mark.parametrize("which", [
    test_reference_csv_is_well_formed_when_present, test_integer_metrics_match_java_exactly,
    test_ratio_metrics_match_java_within_tolerance,
    test_coverage_divergence_is_recorded_not_asserted_equal,
])
def test_every_comparison_test_fails_on_a_malformed_reference_and_skips_on_an_absent_one(
        synth, monkeypatch, tmp_path, which):
    bad = tmp_path / "reference.csv"
    bad.write_text("image,labels\nx,y\n")
    _use_reference(monkeypatch, tmp_path, bad)
    with pytest.raises(pytest.fail.Exception):
        which()
    _use_reference(monkeypatch, tmp_path, tmp_path / "absent.csv")
    with pytest.raises(pytest.skip.Exception):
        which()


def test_the_recorded_epsilon_reaches_the_corner_detector(synth):
    tmp, tif, npy = synth
    seen = {}
    for eps in (1, 30):
        row = {"image": tif.resolve(), "labels": npy.resolve(), "jm1_channel": 0, "jm2_channel": 1,
               "nucleus_channel": 2, "cell_id": 1, "neighbour_id": 2, "role": "JM1",
               "epsilon": eps, "dilation_cycles": 2, "threshold": 100, "interior": True,
               "_line": 0}
        res, _, _ = _measure_field_for(row)
        cell = res.cells.set_index("cell_id").loc[1]
        assert cell.epsilon == eps
        seen[eps] = int(cell.n_interfaces_dp)
    assert seen[1] != seen[30], seen        # epsilon changes the corner set on this cell


# ---- schema validation -----------------------------------------------------------------------

def _valid_rows(synth_):
    return _ab_rows(synth_)


@pytest.mark.parametrize("edit, match", [
    (lambda r: r[0].update(corner_method="tcj"), "no Java counterpart"),
    (lambda r: r[0].update(corner_method="rdp"), "must be 'dp'"),
    (lambda r: r[0].update(role="JM3"), "role must be"),
    (lambda r: r[0].update(epsilon=0), "epsilon"),
    (lambda r: r[0].update(epsilon=2.5), "whole number"),
    (lambda r: r[0].update(epsilon=51), "epsilon"),
    (lambda r: r[0].update(dilation_cycles=0), "dilation_cycles"),
    (lambda r: r[0].update(dilation_cycles=10), "dilation_cycles"),
    (lambda r: r[0].update(threshold=256), "threshold"),
    (lambda r: r[0].update(threshold=-1), "threshold"),
    (lambda r: r[0].update(interior="maybe"), "interior"),
    (lambda r: r[0].update(interface_area_px="12.5"), "whole number"),
    (lambda r: r[0].update(marker_intensity="abc"), "not a number"),
    (lambda r: r[0].update(marker_area_px=""), "is empty"),
    (lambda r: r[0].update(cell_id=2, neighbour_id=2), "distinct"),
    (lambda r: r[0].update(java_distance_ratio="0.5"), "linearity"),
    (lambda r: r[0].update(nucleus_channel=0), "must differ"),
    (lambda r: r[0].update(labels="missing.npy"), "not found"),
    (lambda r: r.append(dict(r[0])), "duplicates"),
])
def test_malformed_reference_rows_are_rejected(synth, edit, match):
    tmp, tif, npy = synth
    rows = _valid_rows(synth)
    edit(rows)
    with pytest.raises(ReferenceFormatError, match=match):
        load_reference(write_csv(tmp / "bad.csv", rows))


@pytest.mark.parametrize("header", [
    [h for h in HEADER if h != "epsilon"],                      # missing column
    [("epsilon_px" if h == "epsilon" else h) for h in HEADER],  # typo'd column
    HEADER + ["cell_id"],                                       # duplicate column
])
def test_malformed_reference_header_is_rejected(synth, header):
    tmp, tif, npy = synth
    with pytest.raises(ReferenceFormatError):
        load_reference(write_csv(tmp / "bad.csv", _valid_rows(synth), header=header))


def test_empty_and_header_only_references_are_rejected(synth):
    tmp, tif, npy = synth
    (tmp / "empty.csv").write_text("")
    with pytest.raises(ReferenceFormatError, match="empty"):
        load_reference(tmp / "empty.csv")
    with pytest.raises(ReferenceFormatError, match="no data rows"):
        load_reference(write_csv(tmp / "h.csv", []))


def test_short_row_is_rejected(synth):
    tmp, tif, npy = synth
    p = write_csv(tmp / "ok.csv", _valid_rows(synth))
    p.write_text(p.read_text().rstrip("\n") + "\n1,2,3\n")
    with pytest.raises(ReferenceFormatError, match="fields"):
        load_reference(p)


# ---- skip / fail policy ----------------------------------------------------------------------

def test_absent_reference_skips_naming_the_file_and_the_readme(tmp_path):
    with pytest.raises(pytest.skip.Exception) as info:
        reference_rows_or_outcome(tmp_path / "reference.csv")
    msg = str(info.value)
    assert "reference.csv" in msg and "README.md" in msg


def test_malformed_reference_fails_instead_of_skipping(tmp_path):
    (tmp_path / "reference.csv").write_text("not,the,schema\n1,2,3\n")
    with pytest.raises(pytest.fail.Exception, match="not a valid reference"):
        reference_rows_or_outcome(tmp_path / "reference.csv")


def test_unreadable_reference_fails_instead_of_skipping(tmp_path):
    (tmp_path / "reference.csv").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(pytest.fail.Exception):
        reference_rows_or_outcome(tmp_path / "reference.csv")


def test_a_tcj_row_fails_the_run_rather_than_being_compared(synth):
    tmp, tif, npy = synth
    rows = _ab_rows(synth)
    rows[1]["corner_method"] = "tcj"
    p = write_csv(tmp / "reference.csv", rows)
    with pytest.raises(pytest.fail.Exception, match="no Java counterpart"):
        reference_rows_or_outcome(p)


def test_a_reference_with_only_skippable_rows_skips_with_the_reasons(synth):
    tmp, tif, npy = synth
    p = write_csv(tmp / "allskip.csv", [_row_from_ours(tif, npy, 1, 2, "JM1", tmp, interior="no")])
    with pytest.raises(pytest.skip.Exception, match="not interior"):
        recorded_result(p)


def test_a_missing_interface_fails_the_run(synth):
    tmp, tif, npy = synth
    rows = _ab_rows(synth)[:2]
    rows[1]["neighbour_id"] = 4      # one good row, so "nothing comparable" cannot mask it
    with pytest.raises(pytest.fail.Exception, match="we find none"):
        recorded_result(write_csv(tmp / "noif.csv", rows))


# ---- divergence report -----------------------------------------------------------------------

def test_report_divergence_writes_a_legible_table(synth, tmp_path):
    tmp, tif, npy = synth
    res = compare_reference(load_reference(write_csv(tmp / "ok.csv", _ab_rows(synth))))
    out = report_divergence(RECORDED_METRICS, res.comparisons, tmp_path / "divergence.md",
                            skipped=[(9, "example reason")])
    text = out.read_text(encoding="utf-8")
    assert "| metric | rule |" in text and "| interface_area_px | exact |" in text
    assert "| linearity_index | rel 1e-09 |" in text
    assert "## Recorded, not asserted: `fragmented_contour_px`" in text
    assert "## Recorded, not asserted: `coverage_index`" in text
    assert "| line | image | cell | neighbour | role | eps | Java | ours | abs diff | rel diff |" in text
    assert text.count("| divergence 3 |") == 2          # both fragmented-contour rows
    assert text.count("| divergence 3 (via fragmented contour) |") == 2   # both coverage rows
    assert "| 7 |" in text                               # the planted +7 shows as the abs diff
    assert "example reason" in text
    assert "UNEXPLAINED" not in text
    # deterministic: no timestamp, so a rerun is byte-identical
    out2 = report_divergence(RECORDED_METRICS, res.comparisons, tmp_path / "d2.md",
                             skipped=[(9, "example reason")])
    assert out.read_text() == out2.read_text()


def test_report_divergence_flags_coverage_differences_that_fragmentation_does_not_explain(tmp_path):
    base = dict(line=2, image="i", cell_id=1, neighbour_id=2, role="JM1", epsilon=2)
    cs = [Comparison(metric="fragmented_contour_px", java=5.0, ours=5.0, **base),
          Comparison(metric="coverage_index", java=0.5, ours=0.4, **base)]
    text = report_divergence(RECORDED_METRICS, cs, tmp_path / "d.md").read_text()
    assert "UNEXPLAINED" in text


def test_report_divergence_reports_a_zero_java_value_without_dividing_by_zero(tmp_path):
    c = Comparison(line=2, image="i", cell_id=1, neighbour_id=2, role="JM2", epsilon=2,
                   metric="fragmented_contour_px", java=0.0, ours=3.0)
    text = report_divergence(RECORDED_METRICS, [c], tmp_path / "d.md").read_text()
    assert "n/a (Java is 0)" in text


def test_the_readme_documents_the_exact_header():
    text = README.read_text(encoding="utf-8")
    assert ",".join(HEADER) in text, "README.md must show the exact reference.csv header"
