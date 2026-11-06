"""Spot-check harness: compare Junction Mapper (Java) numbers against ours on a SHARED path.

The procedure a person follows is in ``tests/validation/README.md``; ``scripts/java_reference.py``
produces the reference automatically by running the Java classes headless. This module holds

* the edge-map converter tests (always run),
* the loader / validator for ``tests/validation/reference.csv`` and the comparison logic,
* self-tests of that logic on a synthetic reference (always run; the "Java" numbers there are
  OUR numbers, so they prove the plumbing and say nothing about the port),
* a frozen golden slice of REAL Java output on a rectilinear control grid (always run: this is
  what ties the port to the original without needing a JDK in the test environment), and
* the comparison tests against the recorded reference (skipped, loudly, until it exists).

Why the Java's corners are inputs
---------------------------------
The Java does not run Douglas-Peucker on the boundary: ``Cell_Membrane.getDouglasPeucker`` takes
the convex hull of the edge pixels first, runs the recursion over the hull vertices with a loop
bound that skips the last interior vertex of a range and an integer-truncated slope, so a
rectangle gets 3 corners and a regular hexagon 8-10 corners, none at a vertex (README.md,
recorded divergence 5). "The Java's interface at epsilon E" and "our interface at epsilon E" are
therefore different interfaces, and comparing them would turn every difference into an apparent
port bug. The harness validates the METRIC formulas instead: each reference row carries the
Java's two corner pixels, and our side is measured along our boundary between the nearest
boundary pixels to those corners (the snap distance is recorded and printed). Corner counts are
compared too, as information only, never as a failure.

What is asserted, and how
-------------------------
A row is STRAIGHT when its shared path is a single axis-aligned run on an axis-aligned outline
(``exactly_comparable``; derived from the geometry, never from a hand-typed flag). There the
Java's walk and our boundary visit the same pixels, and every compared metric is exact: the
three integer metrics with no tolerance, the lengths and ratios at rel 1e-9. ``linearity_index``
is compared as ``1 / java_distance_ratio`` against ours (recorded divergence 1: ours is the
reciprocal of the Java's ``getDistanceRatio``).

Every other row turns a corner or follows an oblique edge. The Java cuts convex corners
diagonally and its 8-neighbour edge set is two pixels thick on oblique edges, so its path
differs from ours by a pixel or two even with the corners snapped to ours. Those rows compare
areas and intensity at the stated OBLIQUE_* tolerances, linearity at an absolute bound, and the
straight-line length (which depends only on the two end pixels) exactly; contour and the two
junction lengths depend on which pixels the path visits and are only recorded. The numbers come
from the real Java over 396 such rows (see the comment at OBLIQUE_AREA_REL); they are an artefact
of the edge-map seam, NOT a tolerance on the port.

recorded only          fragmented_contour_px, coverage_index (recorded divergence 3: the Java's
                       JM1 walk uses an OR connectivity rule that bridges gaps, so the Java value
                       is expected to be >= ours; a Java value below ours is reported
                       UNEXPLAINED). Written to ``divergence.md``, never asserted equal.

Reference cells must be interior (recorded divergence 2: the Java clips the dilation band at the
image border). A row whose band would touch the Java's dead zone, whose cell touches the border,
or whose ``interior`` flag is ``no`` is SKIPPED with a reason. A reference that exists but yields
no comparison at all FAILS; so does one with any undeclared skip.

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
from junctionmapper.errors import BoundaryError
from junctionmapper.geometry.boundary import trace_boundary
from junctionmapper.geometry.corners_dp import dp_corners
from junctionmapper.io.reader import read_fields
from junctionmapper.measure import interface as mi

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

_GEN = "java_reference"
if _GEN in sys.modules:
    java_reference = sys.modules[_GEN]
else:
    _gspec = importlib.util.spec_from_file_location(_GEN, REPO / "scripts" / "java_reference.py")
    java_reference = importlib.util.module_from_spec(_gspec)
    sys.modules[_GEN] = java_reference
    _gspec.loader.exec_module(java_reference)


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


def disks() -> np.ndarray:
    """Round cells with a curved shared boundary and an acute pocket where three meet."""
    lab = np.zeros((40, 40), np.int32)
    rr, cc = np.mgrid[0:40, 0:40]
    d1 = (rr - 14) ** 2 + (cc - 14) ** 2
    d2 = (rr - 14) ** 2 + (cc - 27) ** 2
    d3 = (rr - 28) ** 2 + (cc - 20) ** 2
    inside = np.minimum(np.minimum(d1, d2), d3) <= 81
    lab[inside] = (np.argmin(np.stack([d1, d2, d3]), axis=0) + 1)[inside]
    return lab


def java_select_cell(edge: np.ndarray, row: int, col: int) -> set:
    """A literal port of ImageUtilities.selectCell (ImageUtilities.java:1356-1410).

    The flood VISITS all eight neighbours of a popped pixel but only EXPANDS from a clear pixel
    that has no set pixel among its eight neighbours; a clear pixel beside the line joins the
    cell as an edge point and stops. Set pixels are never part of the cell.
    """
    h, w = edge.shape
    nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    def touches_line(r, c):
        return any(0 <= r + dr < h and 0 <= c + dc < w and edge[r + dr, c + dc] for dr, dc in nbrs)

    todo, seen, cell = [(row, col)], set(), set()
    while todo:
        r, c = todo.pop(0)
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if edge[r, c]:
            continue
        cell.add((r, c))
        if not touches_line(r, c):
            todo += [(r + dr, c + dc) for dr, dc in nbrs if 0 <= r + dr < h and 0 <= c + dc < w]
    return cell


@pytest.mark.parametrize("make", [two_cells_sharing_a_vertical_edge, staircase_cells, disks],
                         ids=["vertical", "staircase", "disks"])
def test_edge_map_confines_the_javas_own_flood_and_java_view_models_it(make):
    """Replays the Java's step 4 on the edge map with a literal port of its flood.

    Each flooded region must lie inside one label (no leak into a neighbour or the background)
    and equal what ``java_view_labels`` reports for that label.
    """
    labels = make()
    edge = edge_map.edge_map_pixels(labels)
    view = edge_map.java_view_labels(labels)
    for lab_id in (int(i) for i in np.unique(labels[labels > 0])):
        free = ~edge & (labels == lab_id)
        interior = [(r, c) for r, c in zip(*np.nonzero(view == lab_id))
                    if not edge[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].any()]
        assert interior, f"label {lab_id} has no pixel the flood can expand from"
        region = java_select_cell(edge, *interior[0])
        assert region <= {(int(r), int(c)) for r, c in np.argwhere(labels == lab_id)}, "leak"
        assert region == {(int(r), int(c)) for r, c in np.argwhere(view == lab_id)}
        assert len(region) <= free.sum()


def test_a_one_sided_one_pixel_barrier_does_not_leak_either():
    """Why the two-pixel barrier is for symmetry and margin, not leak prevention.

    The Java's flood expands only from pixels with no set 8-neighbour, so even a line drawn
    on one side only (set iff a 4-neighbour has a HIGHER label), one pixel thick, confines it.
    """
    labels = staircase_cells()
    one = np.zeros(labels.shape, bool)
    one[1:, :] |= labels[:-1, :] > labels[1:, :]
    one[:-1, :] |= labels[1:, :] > labels[:-1, :]
    one[:, 1:] |= labels[:, :-1] > labels[:, 1:]
    one[:, :-1] |= labels[:, 1:] > labels[:, :-1]
    for lab_id in (1, 2):
        r, c = np.unravel_index(np.argmax(ndimage.distance_transform_edt(labels == lab_id)),
                                labels.shape)
        assert java_select_cell(one, int(r), int(c)) <= {(int(a), int(b)) for a, b in
                                                          np.argwhere(labels == lab_id)}


def test_java_view_is_the_labels_minus_the_barrier_for_a_rectilinear_cell():
    lab = two_cells_sharing_a_vertical_edge()
    view = edge_map.java_view_labels(lab)
    assert (view[edge_map.edge_map_pixels(lab)] == 0).all()
    assert (view[~edge_map.edge_map_pixels(lab)] == lab[~edge_map.edge_map_pixels(lab)]).all()
    assert view[3:9, 3:7].min() == 1 and view[3:9, 9:13].min() == 2     # eroded by one pixel
    assert (view == 1).sum() == 6 * 4 and (view == 2).sum() == 6 * 4


def test_java_view_omits_acute_tips_the_flood_cannot_reach():
    lab = staircase_cells()
    view = edge_map.java_view_labels(lab)
    free = ~edge_map.edge_map_pixels(lab) & (lab > 0)
    assert ((view > 0) <= free).all()               # never more than label minus barrier
    assert (view > 0).sum() < free.sum()            # the 45-degree tips are lost
    assert (view[view > 0] == lab[view > 0]).all()


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
    """A valid row that cannot be compared, for a stated reason.

    ``declared`` is True when the person who made the reference asked for the skip
    (``interior = no``); every other skip is a harness finding and fails the run.
    """

    def __init__(self, reason: str, declared: bool = False):
        super().__init__(reason)
        self.declared = declared


REQUIRED_COLUMNS = (
    "image", "labels", "jm1_channel", "jm2_channel", "nucleus_channel",
    "cell_id", "role", "corner_method", "epsilon", "dilation_cycles", "threshold", "interior",
    "java_corner_row", "java_corner_col", "java_paired_row", "java_paired_col", "java_n_corners",
    "interface_contour_px", "straight_line_length_px", "java_distance_ratio",
    "fragmented_contour_px", "coverage_index",
    "interface_area_px", "marker_area_px", "marker_intensity",
)
OPTIONAL_COLUMNS = ("junction_contour_px", "junction_straight_line_px",
                    "java_corner", "java_paired_with", "seed_row", "seed_col", "notes")
INT_COLUMNS = ("jm1_channel", "jm2_channel", "nucleus_channel", "cell_id", "java_corner_row",
               "java_corner_col", "java_paired_row", "java_paired_col", "java_n_corners")
FLOAT_COLUMNS = ("interface_contour_px", "straight_line_length_px", "fragmented_contour_px",
                 "junction_contour_px", "junction_straight_line_px")
NONFINITE_OK = ("java_distance_ratio", "coverage_index")     # Infinity / NaN are legitimate Java output
INTEGER_VALUED_METRICS = ("interface_area_px", "marker_area_px", "marker_intensity")

EXACT_METRICS = ("interface_area_px", "marker_area_px", "marker_intensity")
TOLERANCE_METRICS = ("interface_contour_px", "straight_line_length_px", "linearity_index",
                     "junction_contour_px", "junction_straight_line_px")
RECORDED_METRICS = ("fragmented_contour_px", "coverage_index")
REL_TOL = 1e-9
ABS_TOL = 1e-12

#: Tolerances for rows that are not a straight axis-aligned run (the Java's edge walk and our
#: boundary then differ by a pixel or two, and at convex corners the Java cuts diagonally).
#: Source: 396 non-straight rows from the real Java on two Voronoi tilings and a hexagonal tiling
#: with corners snapped to ours (0.00 px), dilation 2: interface area -15..+3 px (worst 11.3%),
#: marker area -12..+6 px (worst 14.1%), marker intensity -2130..+1229 (worst 14.0%), linearity
#: -0.29..+0.31; the Task 16 review's own hexagon run (area up to 7.9%, intensity +-1175) lies
#: inside these. They describe the edge-map seam, NOT the port, and apply ONLY to rows that are
#: not exactly comparable. Contour, junction contour and junction straight-line length depend on
#: WHICH pixels the path visits (the Java's differed by up to 17.8 px / 41% on 10-px interfaces), so on
#: non-straight rows they are recorded in divergence.md, not asserted; the straight-line length
#: depends only on the two end pixels and stays exact on every row.
OBLIQUE_AREA_REL = 0.15
OBLIQUE_AREA_ABS = 16
OBLIQUE_INTENSITY_REL = 0.15
OBLIQUE_LINEARITY_ABS = 0.35
PATH_DEPENDENT_METRICS = ("interface_contour_px", "junction_contour_px", "junction_straight_line_px")

#: A Java corner farther than this from our boundary is a path disagreement (not a metric one).
SNAP_TOLERANCE_PX = 2.0

CORNER_NOTE = ("The Java takes the convex hull of the edge pixels before Douglas-Peucker and "
               "skips the last interior vertex of each range (README.md divergence 5), so its "
               "corner count differs from ours by construction. Information only.")

EXPLANATIONS = {
    "fragmented_contour_px": (
        "Divergence 3 (README.md): the Java's JM1 walk joins steps with an OR rule and bridges "
        "gaps, which can only ADD length, so the Java value is expected to be >= ours. A Java "
        "value below ours is not explained by it and is marked UNEXPLAINED."),
    "coverage_index": (
        "Coverage is fragmented contour / interface contour. The table says which side of the "
        "ratio differs: a numerator difference is divergence 3 (when the Java is larger); a "
        "denominator difference is a path disagreement, not a measurement one."),
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


def _parse_float(raw: str, name: str, line: int, allow_nonfinite: bool = False) -> float:
    try:
        f = float(raw)
    except ValueError:
        raise _ref_error(line, f"column {name!r}: {raw!r} is not a number") from None
    if math.isnan(f):
        if allow_nonfinite:
            return f
        raise _ref_error(line, f"column {name!r}: {raw!r} must be a finite number >= 0")
    if f < 0 or (math.isinf(f) and not allow_nonfinite):
        raise _ref_error(line, f"column {name!r}: {raw!r} must be a finite number >= 0")
    return f


def _seed_labels(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def load_reference(path: Path) -> list[dict]:
    """Parse and validate ``reference.csv``; raise ReferenceFormatError on any defect.

    Schema (see README.md): one row per (cell, Java interface, marker role); the Java interface
    is identified by its two corner pixels. Paths in ``image`` and ``labels`` are resolved
    relative to the CSV and must exist.
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
        row: dict = {"_line": lineno}
        for name in REQUIRED_COLUMNS:
            if raw[name] == "":
                raise _ref_error(lineno, f"column {name!r} is empty")
        for name in INT_COLUMNS:
            row[name] = _parse_int(raw[name], name, lineno)
            if row[name] < 0:
                raise _ref_error(lineno, f"column {name!r} must be >= 0")
        if len({row["jm1_channel"], row["jm2_channel"], row["nucleus_channel"]}) != 3:
            raise _ref_error(lineno, "jm1_channel, jm2_channel and nucleus_channel must differ")
        if row["cell_id"] < 1 or row["java_n_corners"] < 1:
            raise _ref_error(lineno, "cell_id and java_n_corners must be >= 1")

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
            row[name] = None if raw.get(name, "") == "" else _parse_float(raw[name], name, lineno)
        for name in NONFINITE_OK:
            row[name] = _parse_float(raw[name], name, lineno, allow_nonfinite=True)
        ratio = row["java_distance_ratio"]
        if not math.isnan(ratio) and ratio < 1.0 - 1e-9:
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
        # The click that flooded the cell. Optional for a hand-transcribed row, but then the
        # row cannot be reproduced exactly: the Java's edge pixels, hull and corners depend
        # on where it was clicked (README.md "Why the click matters").
        sr, sc = raw.get("seed_row", ""), raw.get("seed_col", "")
        if sr == "" and sc != "":
            raise _ref_error(lineno, "seed_col is given without seed_row; give both or neither")
        if sr != "" and sc == "":
            raise _ref_error(lineno, "seed_row is given without seed_col; give both or neither")
        row["seed"] = None
        if sr != "":
            row["seed"] = (_parse_int(sr, "seed_row", lineno), _parse_int(sc, "seed_col", lineno))
            if min(row["seed"]) < 0:
                raise _ref_error(lineno, "seed_row and seed_col must be >= 0")

        for name in ("image", "labels"):
            p = (path.parent / raw[name]).resolve()
            if not p.is_file():
                raise _ref_error(lineno, f"{name} file not found: {p}")
            row[name] = p
        if row["seed"] is not None:
            # validated whenever present, whether or not the row is later compared
            lab = _seed_labels(row["labels"])
            r, c = row["seed"]
            if not (r < lab.shape[0] and c < lab.shape[1] and lab[r, c] == row["cell_id"]):
                raise _ref_error(lineno, f"cell {row['cell_id']}: the recorded click (row {r}, "
                                         f"col {c}) is not inside that cell")
        rows.append(row)

    if not rows:
        raise ReferenceFormatError(f"{path} has a header but no data rows")
    seen: dict[tuple, int] = {}
    per_cell: dict[tuple, tuple[int, int, int]] = {}
    seeds: dict[tuple, object] = {}
    for r in rows:
        key = (r["image"], r["labels"], r["cell_id"], r["role"], r["java_corner_row"],
               r["java_corner_col"], r["java_paired_row"], r["java_paired_col"])
        if key in seen:
            raise _ref_error(r["_line"], f"duplicates line {seen[key]}: one row per "
                                         "(image, cell, Java interface, role)")
        seen[key] = r["_line"]
        ckey = (r["image"], r["cell_id"])
        if ckey in per_cell:
            eps, nc, first = per_cell[ckey]
            if eps != r["epsilon"]:
                raise _ref_error(r["_line"], f"cell {r['cell_id']} was recorded with epsilon {eps} "
                                             f"on line {first} but {r['epsilon']} here; epsilon is "
                                             "set once per cell")
            if nc != r["java_n_corners"]:
                raise _ref_error(r["_line"], f"cell {r['cell_id']} has java_n_corners {nc} on line "
                                             f"{first} but {r['java_n_corners']} here")
            if seeds[ckey] != r["seed"]:
                raise _ref_error(r["_line"], f"cell {r['cell_id']} has click {seeds[ckey]} on line "
                                             f"{first} but {r['seed']} here; one cell is one click "
                                             "(different clicks give different corner sets)")
        else:
            per_cell[ckey] = (r["epsilon"], r["java_n_corners"], r["_line"])
            seeds[ckey] = r["seed"]
    return rows


# ---- measurement -----------------------------------------------------------------------------

#: neighbour-search radius passed to the run config. Nothing reads neighbours any more; kept so
#: the config is the one the pipeline would build for the Java's view (cells 3 px apart).
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
        epsilon_px=float(row["epsilon"]), neighbour_radius_px=JAVA_VIEW_NEIGHBOUR_RADIUS)


_LABELS: dict[Path, tuple] = {}
_PLANES: dict[tuple, dict] = {}


def _labels_for(row: dict):
    """(labels, java-view labels, edge map) for the row's label file."""
    key = row["labels"]
    if key not in _LABELS:
        labels = np.load(key)
        _LABELS[key] = (labels, edge_map.java_view_labels(labels), edge_map.edge_map_pixels(labels))
    return _LABELS[key]


def _planes_for(row: dict) -> dict:
    key = (row["image"], row["jm1_channel"], row["jm2_channel"], row["nucleus_channel"])
    if key not in _PLANES:
        field_ = next(iter(read_fields(row["image"], _config(row))))
        labels = _labels_for(row)[0]
        if field_.planes[Role.JM1].shape != labels.shape:
            raise ReferenceFormatError(
                f"line {row['_line']}: image is {field_.planes[Role.JM1].shape} but the labels "
                f"are {labels.shape}")
        _PLANES[key] = field_.planes
    return _PLANES[key]


@dataclass
class Geometry:
    """Our boundary of the cell and where the Java's two corners land on it."""
    view: np.ndarray
    edge: np.ndarray
    boundary: np.ndarray
    i: int                      # boundary position nearest the Java corner
    j: int                      # boundary position nearest the Java paired corner
    snap_i: float
    snap_j: float
    java_ends: tuple
    arcs: list                  # the two candidate corner-to-corner paths


@dataclass
class Measured:
    metrics: dict
    straight: bool              # exactly comparable (see exactly_comparable)
    why: str                    # one-line reason for the class
    context: str                # both endpoint pairs and snap distances, for messages
    our_corners: int


def _geometry(row: dict) -> Geometry:
    """Locate the Java's corner pair on our boundary. Raises RowSkip / AssertionError."""
    if not row["interior"]:
        raise RowSkip("operator marked the cell as not interior (Java clips its dilation band "
                      "at the image border; README.md divergence 2)", declared=True)
    labels, view, edge = _labels_for(row)
    cid = row["cell_id"]
    if not (labels == cid).any():
        raise AssertionError(f"cell {cid} is not a label in {row['labels'].name}")
    if any((labels[s] == cid).any() for s in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1])):
        raise RowSkip(f"cell {cid} touches the image border; it is not measured (the Java "
                      "clips its dilation band there; README.md divergence 2)")
    try:
        b = trace_boundary(view, cid)
    except BoundaryError as exc:
        raise RowSkip(f"cell {cid} is degenerate on the Java's view of the labels: {exc}") from exc
    ends = ((row["java_corner_row"], row["java_corner_col"]),
            (row["java_paired_row"], row["java_paired_col"]))
    snaps = []
    for rc in ends:
        d2 = ((b - np.array(rc)) ** 2).sum(1)
        k = int(d2.argmin())
        snaps.append((k, math.sqrt(float(d2[k]))))
    (i, di), (j, dj) = snaps
    n = len(b)
    arcs = []
    for a, z in ((i, j), (j, i)):
        length = (z - a) % n or n
        arcs.append(b[(a + np.arange(length + 1)) % n])
    geo = Geometry(view, edge, b, i, j, di, dj, ends, arcs)
    if max(di, dj) > SNAP_TOLERANCE_PX:
        raise AssertionError(
            f"cell {cid}: path disagreement, not a metric one: {_context(geo)}; the Java corner is "
            f"more than {SNAP_TOLERANCE_PX} px from our boundary")
    return geo


def _context(geo: Geometry, arc=None) -> str:
    a, b = geo.boundary[geo.i], geo.boundary[geo.j]
    s = (f"Java corners {tuple(geo.java_ends[0])} -> {tuple(geo.java_ends[1])}; ours snapped "
         f"{(int(a[0]), int(a[1]))} -> {(int(b[0]), int(b[1]))} (snap {geo.snap_i:.2f} / "
         f"{geo.snap_j:.2f} px)")
    return s if arc is None else s + f"; shared path {len(arc)} px"


def exactly_comparable(geo: Geometry, cid: int, arc: np.ndarray) -> tuple[bool, str]:
    """Is this path one on which the Java's edge walk equals ours, so metrics are exact?

    Derived from the geometry only. True when (1) the path is a single straight axis-aligned run
    (the Java's walk cuts convex corners diagonally, so any path that turns a corner differs),
    and (2) near the path (Chebyshev radius 2) the Java's edge pixels, the cell's pixels with any
    of 8 neighbours set in the edge map, equal our boundary pixels, the cell's pixels with any
    of 4 neighbours outside it. Axis-aligned outlines satisfy (2); oblique edges (2 px thick in
    the Java's 8-neighbour set) and reflex corners do not.
    """
    steps = np.diff(arc, axis=0)
    if len(steps) == 0 or not (steps == steps[0]).all() or np.abs(steps[0]).sum() != 1:
        return False, "path is not a single straight axis-aligned run (Java cuts convex corners)"
    cell = geo.view == cid
    ring8 = cell & ndimage.binary_dilation(geo.edge, structure=np.ones((3, 3), bool))
    ring4 = cell & ~ndimage.binary_erosion(cell, structure=ndimage.generate_binary_structure(2, 1))
    mask = np.zeros(cell.shape, bool)
    mask[arc[:, 0], arc[:, 1]] = True
    near = ndimage.binary_dilation(mask, structure=np.ones((5, 5), bool))
    if ((ring8 & near) != (ring4 & near)).any():
        return False, "outline near the path is not axis-aligned (Java edge set is thicker)"
    return True, "straight axis-aligned run on an axis-aligned outline"


def select_arc(geo: Geometry, java_contour: float | None):
    """Choose between the two corner-to-corner arcs by the Java's recorded contour.

    Raises RowSkip when both arcs are consistent with it (e.g. opposite corners of a rectangle)
    and the row cannot say which side was transcribed. ``java_contour=None`` (the harness's own
    synthetic rows) takes the shorter arc.
    """
    a, b = geo.arcs
    ca, cb = mi.interface_contour(a), mi.interface_contour(b)
    if java_contour is None:
        return a if ca <= cb else b
    ok = [math.isclose(c, java_contour, rel_tol=REL_TOL, abs_tol=ABS_TOL) for c in (ca, cb)]
    if all(ok) and not np.array_equal(a, b):
        raise RowSkip(f"both arcs between the Java corners have the recorded contour "
                      f"{java_contour:g}; the row cannot say which one was measured ({_context(geo)})")
    return a if abs(ca - java_contour) <= abs(cb - java_contour) else b


def measure_arc(row: dict, geo: Geometry, arc: np.ndarray) -> Measured:
    """Our metrics along ``arc`` with the row's dilation cycles and threshold."""
    cid, role = row["cell_id"], row["role"]
    if not java_dilation_is_unclipped(arc, geo.view.shape, row["dilation_cycles"]):
        raise RowSkip("the shared path's band reaches the Java's border dead zone (within 2 px of "
                      "the image edge), where its dilation is clipped (README.md divergence 2)")
    plane = _planes_for(row)[Role(role)]
    t = float(row["threshold"])
    band = mi.dilated_band(arc, geo.view.shape, row["dilation_cycles"])
    bm = mi.band_metrics(band, plane, t)
    metrics = {
        "interface_contour_px": mi.interface_contour(arc),
        "straight_line_length_px": mi.straight_line_length(arc),
        "linearity_index": mi.linearity_index(arc),
        "fragmented_contour_px": mi.fragmented_contour(arc, plane, t),
        "coverage_index": mi.coverage_index(arc, plane, t),
        "interface_area_px": float(bm.interface_area_px),
        "marker_area_px": float(bm.marker_area_px),
        "marker_intensity": float(bm.marker_intensity),
    }
    if role == "JM1":
        metrics["junction_contour_px"] = mi.junction_contour(arc, plane, t)
        metrics["junction_straight_line_px"] = mi.junction_straight_line(arc, plane, t)
    straight, why = exactly_comparable(geo, cid, arc)
    return Measured(metrics, straight, why, _context(geo, arc),
                    len(dp_corners(geo.boundary, float(row["epsilon"]))))


def measure_one(row: dict) -> Measured:
    """Our metrics along the path between the Java's two corner pixels.

    Raises RowSkip for rows that cannot be compared and AssertionError for a path disagreement.
    """
    geo = _geometry(row)
    return measure_arc(row, geo, select_arc(geo, row["interface_contour_px"]))


# ---- comparison ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Comparison:
    line: int
    image: str
    cell_id: int
    role: str
    epsilon: int
    metric: str
    java: float
    ours: float
    straight: bool = True       # exactly comparable row
    context: str = ""

    @property
    def abs_diff(self) -> float:
        return abs(self.ours - self.java)

    @property
    def rel_diff(self) -> float:
        if self.abs_diff == 0 or (math.isnan(self.ours) and math.isnan(self.java)):
            return 0.0
        return self.abs_diff / abs(self.java) if self.java else math.inf

    @property
    def rule(self) -> str:
        m = self.metric
        if m in RECORDED_METRICS:
            return "recorded"
        if self.straight or m == "straight_line_length_px":
            return "exact" if m in EXACT_METRICS else f"rel {REL_TOL:g}"
        if m in PATH_DEPENDENT_METRICS:
            return "recorded (path-dependent)"
        if m in ("interface_area_px", "marker_area_px"):
            return f"within {OBLIQUE_AREA_REL:.0%} or {OBLIQUE_AREA_ABS} px"
        if m == "marker_intensity":
            return f"within {OBLIQUE_INTENSITY_REL:.0%}"
        return f"within {OBLIQUE_LINEARITY_ABS}"

    @property
    def asserted(self) -> bool:
        return not self.rule.startswith("recorded")

    @property
    def agrees(self) -> bool:
        a, j = self.ours, self.java
        if math.isnan(a) or math.isnan(j):
            return math.isnan(a) and math.isnan(j)
        if math.isinf(a) or math.isinf(j):
            return a == j
        m, d = self.metric, abs(a - j)
        if self.rule in ("exact", f"rel {REL_TOL:g}") or not self.asserted:
            if m in EXACT_METRICS and self.asserted:
                return a == j
            return math.isclose(a, j, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        if m in ("interface_area_px", "marker_area_px"):
            return d <= max(OBLIQUE_AREA_ABS, OBLIQUE_AREA_REL * abs(j))
        if m == "marker_intensity":
            return d <= OBLIQUE_INTENSITY_REL * abs(j)
        return d <= OBLIQUE_LINEARITY_ABS


@dataclass
class CompareResult:
    comparisons: list[Comparison] = field(default_factory=list)
    skipped: list[tuple[int, str, bool]] = field(default_factory=list)   # (line, reason, declared)
    errors: list[str] = field(default_factory=list)
    corner_counts: list[tuple] = field(default_factory=list)   # (image, cell, eps, java_n, our_n)

    def mismatches(self, metrics) -> list[Comparison]:
        return [c for c in self.comparisons
                if c.metric in metrics and c.asserted and not c.agrees]


def java_values(row: dict) -> dict:
    """The row's Java numbers keyed by OUR metric names.

    ``linearity_index`` is the reciprocal of the Java's ``getDistanceRatio`` (README.md
    divergence 1), so it is converted here: we compare ``1 / java_distance_ratio`` with our
    linearity_index. A ratio of exactly 1 maps to 1, a straight interface; Infinity maps to 0.
    """
    vals = {m: row[m] for m in EXACT_METRICS + RECORDED_METRICS
            + ("interface_contour_px", "straight_line_length_px")}
    ratio = row["java_distance_ratio"]
    vals["linearity_index"] = float("nan") if math.isnan(ratio) else 1.0 / ratio
    if row["role"] == "JM1":
        for m in ("junction_contour_px", "junction_straight_line_px"):
            if row[m] is not None:
                vals[m] = row[m]
    return {k: float(v) for k, v in vals.items()}


def compare_reference(rows: list[dict]) -> CompareResult:
    result = CompareResult()
    counted = set()
    for row in rows:
        try:
            m = measure_one(row)
        except RowSkip as why:
            result.skipped.append((row["_line"], str(why), why.declared))
            continue
        except AssertionError as why:
            result.errors.append(f"line {row['_line']}: {why}")
            continue
        for metric, java in java_values(row).items():
            if metric not in m.metrics:
                continue
            result.comparisons.append(Comparison(
                line=row["_line"], image=row["image"].stem, cell_id=row["cell_id"],
                role=row["role"], epsilon=row["epsilon"], metric=metric, java=java,
                ours=m.metrics[metric], straight=m.straight, context=m.context))
        key = (row["image"].stem, row["cell_id"], row["epsilon"])
        if key not in counted:
            counted.add(key)
            result.corner_counts.append(key + (row["java_n_corners"], m.our_corners))
    return result


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.10g}"


def _same(c: Comparison) -> bool:
    return (math.isnan(c.ours) and math.isnan(c.java)) or math.isclose(
        c.ours, c.java, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _attribution(c: Comparison, index: dict) -> str:
    """Which recorded divergence, if any, explains a recorded-metric difference. Checks sign."""
    if _same(c):
        return "agrees"
    suffix = "" if c.straight else " (+ path difference on a non-straight row)"
    if c.metric == "fragmented_contour_px":
        if c.java > c.ours:
            return "divergence 3" + suffix
        return "UNEXPLAINED: Java < ours (divergence 3 only predicts Java >= ours)"
    frag = index.get((c.line, "fragmented_contour_px"))
    contour = index.get((c.line, "interface_contour_px"))
    if frag is not None and not _same(frag):
        if frag.java > frag.ours:
            return "numerator (fragmented contour) differs: divergence 3" + suffix
        return "UNEXPLAINED: numerator (fragmented contour) differs and Java < ours"
    if contour is not None and not _same(contour):
        return "denominator (interface contour) differs: path disagreement, not a metric"
    return "UNEXPLAINED: neither fragmented contour nor interface contour differs"


def report_divergence(metrics, comparisons, path: Path | None = None, skipped=(),
                      corner_counts=()) -> Path:
    """Write ``divergence.md``: asserted-metric summary, recorded divergences, corner counts.

    ``metrics`` are the recorded-only metrics to tabulate. All of ``comparisons`` feed the summary
    of asserted metrics. ``skipped`` holds (line, reason[, declared]) tuples; ``corner_counts``
    holds (image, cell, epsilon, java corners, our corners). The file has no timestamp, so
    rerunning on the same reference reproduces it byte for byte.
    """
    path = DIVERGENCE_MD if path is None else Path(path)
    comparisons = list(comparisons)
    metrics = tuple(metrics)
    index = {(c.line, c.metric): c for c in comparisons}
    out = [
        "# Java spot-check: where the Python port and the original differ",
        "",
        "Generated by `tests/validation/test_java_spotcheck.py` from `reference.csv`; do not edit.",
        "Java numbers come from the original application (`scripts/java_reference.py` or a",
        "hand transcription). Divergence numbers refer to the repository `README.md`.",
        "Each row is measured by our functions along our boundary between the pixels nearest the",
        "Java's two corner pixels, so the metric formulas are compared on a shared path.",
        "",
        "## Metric summary",
        "",
        "*straight* rows (a single axis-aligned run on an axis-aligned outline) must match exactly:",
        f"integer metrics identically, the rest to relative {REL_TOL:g}. *other* rows (a path that turns",
        "a corner or follows an oblique edge) are compared at the stated tolerance where the metric",
        "is asserted; contour, junction contour and junction straight-line length depend on which",
        "pixels the path visits, so on those rows they are recorded here and not asserted. These",
        "tolerances are an artefact of the edge-map seam, not a tolerance on the port.",
        "`linearity_index` is `1 / java_distance_ratio`.",
        "",
        "| metric | class | rule | n | agree | worst abs diff | worst rel diff | result |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for m in EXACT_METRICS + TOLERANCE_METRICS:
        for straight, label in ((True, "straight"), (False, "other")):
            cs = [c for c in comparisons if c.metric == m and c.straight == straight]
            if not cs:
                continue
            n_ok = sum(c.agrees for c in cs)
            result = ("recorded" if not cs[0].asserted
                      else "agree" if n_ok == len(cs) else "DISAGREE")
            worst_rel = max(c.rel_diff for c in cs)
            out.append(f"| {m} | {label} | {cs[0].rule} | {len(cs)} | {n_ok} | "
                       f"{_fmt(max(c.abs_diff for c in cs))} | {_fmt(worst_rel)} | {result} |")
    for m in metrics:
        cs = sorted((c for c in comparisons if c.metric == m), key=lambda c: (c.line, c.role))
        out += ["", f"## Recorded, not asserted: `{m}`", "", EXPLANATIONS.get(m, ""), "",
                "| line | image | cell | role | eps | Java | ours | abs diff | rel diff | explained by |",
                "|---:|---|---:|---|---:|---:|---:|---:|---:|---|"]
        if not cs:
            out.append("| | no comparable rows | | | | | | | | |")
        for c in cs:
            rel = "n/a (Java is 0)" if math.isinf(c.rel_diff) else _fmt(c.rel_diff)
            out.append(f"| {c.line} | {c.image} | {c.cell_id} | {c.role} | {c.epsilon} | "
                       f"{_fmt(c.java)} | {_fmt(c.ours)} | {_fmt(c.abs_diff)} | {rel} | "
                       f"{_attribution(c, index)} |")
    corner_counts = list(corner_counts)
    if corner_counts:
        out += ["", "## Corner counts (information, never a failure)", "", CORNER_NOTE, "",
                "| image | cell | eps | Java corners | our DP corners |", "|---|---:|---:|---:|---:|"]
        out += [f"| {i} | {c} | {e} | {jn} | {on} |" for i, c, e, jn, on in sorted(corner_counts)]
    skipped = list(skipped)
    if skipped:
        out += ["", "## Rows not compared", "", "| line | reason |", "|---:|---|"]
        out += [f"| {s[0]} | {s[1]} |" for s in sorted(skipped)]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


# ---- skip / fail policy ----------------------------------------------------------------------

def outcome_of(fn, *args):
    """Run ``fn`` and return ("passed" | "failed" | "skipped", message).

    Asserting on the OUTCOME, not the exception type, is what lets the policy tests see a
    failure that has degraded into a skip: ``pytest.raises(pytest.fail.Exception)`` does not
    catch ``Skipped``, so the test would be reported as skipped and stay green.
    """
    try:
        fn(*args)
    except pytest.skip.Exception as exc:
        return "skipped", str(exc)
    except (pytest.fail.Exception, AssertionError) as exc:
        return "failed", str(exc)
    return "passed", ""


def reference_rows_or_outcome(path: Path | None = None) -> list[dict]:
    """The validated rows; SKIP if the file is absent, FAIL if it exists but is malformed."""
    path = REFERENCE if path is None else Path(path)
    if not path.exists():
        pytest.skip(f"no recorded Java reference values: {path} does not exist; generate it "
                    f"with scripts/java_reference.py or follow {README}")
    try:
        return load_reference(path)
    except ReferenceFormatError as exc:
        pytest.fail(f"{path} exists but is not a valid reference file: {exc}", pytrace=False)


_RESULT_CACHE: dict[Path, CompareResult] = {}


def _skip_summary(result: CompareResult) -> str:
    reasons: dict[str, int] = {}
    for _, why, _ in result.skipped:
        reasons[why.split(" (")[0]] = reasons.get(why.split(" (")[0], 0) + 1
    return "; ".join(f"{n} x {why}" for why, n in sorted(reasons.items()))


def recorded_result(path: Path | None = None) -> CompareResult:
    """Compare every row of an EXISTING reference; a reference that compares nothing FAILS.

    Absent file: skip. Present but malformed: fail. Present but unable to produce a single
    comparison (all rows skipped): fail, naming the count and the reasons. Silence is the one
    outcome this harness must not produce.
    """
    path = REFERENCE if path is None else Path(path)
    rows = reference_rows_or_outcome(path)
    if path not in _RESULT_CACHE:
        _RESULT_CACHE[path] = compare_reference(rows)
    result = _RESULT_CACHE[path]
    if result.errors:
        pytest.fail("path disagreements between the Java and our boundary:\n  "
                    + "\n  ".join(result.errors), pytrace=False)
    if not result.comparisons:
        pytest.fail(f"{path} has {len(rows)} rows but none could be compared: all "
                    f"{len(result.skipped)} were skipped ({_skip_summary(result)}). A reference "
                    "that compares nothing is a defect, not an absent reference.", pytrace=False)
    return result


def _describe(cs: list[Comparison]) -> str:
    return "\n".join(f"line {c.line} cell {c.cell_id} {c.role} {c.metric} [{c.rule}]: Java "
                     f"{_fmt(c.java)} vs ours {_fmt(c.ours)}\n    {c.context}" for c in cs)


# =============================================================================================
# 3. Tests against the recorded reference (skipped until reference.csv exists)
# =============================================================================================

def test_reference_csv_is_well_formed_when_present():
    reference_rows_or_outcome()


def test_no_reference_row_was_silently_skipped():
    result = recorded_result()
    undeclared = [(ln, why) for ln, why, declared in result.skipped if not declared]
    assert not undeclared, (
        f"{len(undeclared)} of {len(result.skipped) + len({c.line for c in result.comparisons})} "
        f"reference rows were skipped without being declared (interior = no): "
        + "; ".join(f"line {ln}: {why}" for ln, why in undeclared))


def test_integer_metrics_match_java_exactly():
    result = recorded_result()
    bad = result.mismatches(EXACT_METRICS)
    assert not bad, "integer metrics disagree:\n" + _describe(bad)
    exact_rows = [c for c in result.comparisons if c.metric in EXACT_METRICS and c.straight]
    assert exact_rows, (
        "no straight axis-aligned interior row in the reference, so the exact (zero-tolerance) "
        "assertions were never applied; add the rectilinear control (scripts/java_reference.py "
        f"--control). {len(result.comparisons)} comparisons ran only at oblique tolerances.")


def test_ratio_metrics_match_java_within_tolerance():
    result = recorded_result()
    bad = result.mismatches(TOLERANCE_METRICS)
    assert not bad, "length / ratio metrics disagree:\n" + _describe(bad)


def test_coverage_divergence_is_recorded_not_asserted_equal():
    result = recorded_result()
    path = report_divergence(RECORDED_METRICS, result.comparisons, skipped=result.skipped,
                             corner_counts=result.corner_counts)
    text = path.read_text(encoding="utf-8")
    for m in RECORDED_METRICS:
        assert f"`{m}`" in text
    # deliberately no assertion that Java == ours for these metrics, nor on corner counts


# =============================================================================================
# 4. Self-tests of the harness (always run). The "Java" numbers here are OUR numbers.
# =============================================================================================

def synthetic_labels() -> np.ndarray:
    """60x60. A (left) and B (right) share a boundary that jogs by one pixel; C sits above A and
    touches the top border; D is a small cell below B; E sits within 3 px of the top edge (its
    band reaches the Java's dead zone); F is a 3x3 blob whose every pixel is barrier."""
    lab = np.zeros((60, 60), np.int32)
    lab[15:45, 15:30] = 1
    lab[15:45, 30:45] = 2
    lab[25:45, 30] = 1                  # the jog: A owns column 30 for the lower rows
    lab[0:15, 15:30] = 3
    lab[45:55, 35:44] = 4
    lab[2:9, 40:50] = 5
    lab[30:33, 50:53] = 6
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

# Java corner pairs on the synthetic field, as (row, col) on the Java's view of the labels.
A_TOP = (1, (16, 16), (16, 28))          # straight run along A's top side (13 px)
D_TOP = (4, (46, 36), (46, 42))          # straight run along D's top side
D_BENT = (4, (46, 36), (50, 42))         # D's top side then down its right side (turns a corner)
A_JOG = (1, (16, 28), (43, 29))          # A's right side, across the one-pixel jog
D_OPPOSITE = (4, (46, 36), (53, 42))     # opposite corners: both arcs have contour 13
C_BORDER = (3, (2, 16), (2, 28))         # C touches the image border
A_NEAR_JOG = (1, (22, 28), (24, 28))     # straight vertical run, but the jog's reflex corner is 2 px away
E_DEAD = (5, (3, 41), (3, 48))           # band reaches rows 1..5: the Java's dead zone
F_BLOB = (6, (31, 51), (31, 51))         # no cell survives the barrier


def _proto(tif, npy, cell, a, b, role, eps=2, cycles=2, thr=100, contour=None):
    return {"image": tif.resolve(), "labels": npy.resolve(), "jm1_channel": 0, "jm2_channel": 1,
            "nucleus_channel": 2, "cell_id": cell, "role": role, "epsilon": eps,
            "dilation_cycles": cycles, "threshold": thr, "interior": True, "_line": 0,
            "java_corner_row": a[0], "java_corner_col": a[1], "java_paired_row": b[0],
            "java_paired_col": b[1], "interface_contour_px": contour}


def make_row(tif, npy, pair, role, *, eps=2, cycles=2, thr=100, interior="yes"):
    """A reference row whose 'Java' numbers are produced by our own functions on the shorter arc.

    The fragmented contour and coverage are deliberately off (+7, +0.25) and the Java corner
    count is ours + 3, so the recorded metrics show a planted divergence and the corner counts
    differ, neither of which may fail the run.
    """
    cell, a, b = pair
    proto = _proto(tif, npy, cell, a, b, role, eps, cycles, thr)
    geo = _geometry(proto)
    m = measure_arc(proto, geo, select_arc(geo, None))
    x = m.metrics
    d = {"image": tif.name, "labels": npy.name, "jm1_channel": 0, "jm2_channel": 1,
         "nucleus_channel": 2, "cell_id": cell, "role": role, "corner_method": "dp",
         "epsilon": eps, "dilation_cycles": cycles, "threshold": thr, "interior": interior,
         "java_corner_row": a[0], "java_corner_col": a[1], "java_paired_row": b[0],
         "java_paired_col": b[1], "java_n_corners": m.our_corners + 3,
         "interface_contour_px": repr(x["interface_contour_px"]),
         "straight_line_length_px": repr(x["straight_line_length_px"]),
         "java_distance_ratio": repr(1.0 / x["linearity_index"]),
         "fragmented_contour_px": repr(x["fragmented_contour_px"] + 7.0),
         "coverage_index": repr(x["coverage_index"] + 0.25),
         "interface_area_px": int(x["interface_area_px"]),
         "marker_area_px": int(x["marker_area_px"]),
         "marker_intensity": int(x["marker_intensity"]),
         "junction_contour_px": repr(x["junction_contour_px"]) if role == "JM1" else "",
         "junction_straight_line_px": repr(x["junction_straight_line_px"]) if role == "JM1" else "",
         "notes": ""}
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


def _good_rows(synth_):
    tmp, tif, npy = synth_
    return [make_row(tif, npy, A_TOP, "JM1", thr=100), make_row(tif, npy, A_TOP, "JM2", thr=60),
            make_row(tif, npy, D_BENT, "JM1"), make_row(tif, npy, A_JOG, "JM1"),
            make_row(tif, npy, D_TOP, "JM1")]


def _measured(synth_, pair, role="JM1"):
    tmp, tif, npy = synth_
    proto = _proto(tif, npy, pair[0], pair[1], pair[2], role)
    geo = _geometry(proto)
    return measure_arc(proto, geo, select_arc(geo, None))


def test_synthetic_preconditions_straight_and_bent_rows_are_classified_as_such(synth):
    a, d, jog = _measured(synth, A_TOP), _measured(synth, D_BENT), _measured(synth, A_JOG)
    assert a.straight and a.metrics["interface_contour_px"] == 12.0
    assert a.metrics["interface_area_px"] == (2 * 2 + 1) * (13 + 2 * 2)       # 85, by hand
    assert not d.straight and "straight axis-aligned run" in d.why
    assert 1.0 / d.metrics["linearity_index"] > 1.3                           # reciprocal visible
    assert not jog.straight
    assert _measured(synth, D_TOP).straight


def test_a_straight_run_beside_a_reflex_corner_is_not_exactly_comparable(synth):
    near = _measured(synth, A_NEAR_JOG)
    assert not near.straight and "not axis-aligned" in near.why
    # the same straight run well away from the jog is
    assert _measured(synth, (1, (17, 28), (19, 28))).straight


def test_the_recorded_contour_selects_the_arc(synth):
    tmp, tif, npy = synth
    geo = _geometry(_proto(tif, npy, *A_TOP, "JM1"))
    short, long_ = sorted(geo.arcs, key=len)
    assert select_arc(geo, mi.interface_contour(long_)) is long_
    assert select_arc(geo, mi.interface_contour(short)) is short
    assert len(long_) > 3 * len(short)


def test_harness_agrees_with_itself_on_a_consistent_reference(synth):
    tmp, tif, npy = synth
    rows = _good_rows(synth)
    result = compare_reference(load_reference(write_csv(tmp / "ok.csv", rows)))
    assert not result.errors and not result.skipped
    bad = result.mismatches(EXACT_METRICS + TOLERANCE_METRICS)
    assert not bad, _describe(bad)
    assert set(EXACT_METRICS + TOLERANCE_METRICS + RECORDED_METRICS) <= {c.metric for c in result.comparisons}
    # the planted divergences are visible to the recorded metrics and nowhere else
    planted = [c for c in result.comparisons if c.metric in RECORDED_METRICS and not c.agrees]
    assert len(planted) == 2 * len(rows)
    assert any(c.straight for c in result.comparisons) and any(not c.straight for c in result.comparisons)


def test_corner_count_differences_are_reported_and_never_fail(synth):
    tmp, tif, npy = synth
    rows = _good_rows(synth)
    for r in rows:
        r["java_n_corners"] = 40            # absurdly different from our count
    result = compare_reference(load_reference(write_csv(tmp / "cc.csv", rows)))
    assert not result.errors and not result.skipped
    assert not result.mismatches(EXACT_METRICS + TOLERANCE_METRICS)
    assert result.corner_counts and all(jn == 40 and on != 40 for *_, jn, on in result.corner_counts)
    text = report_divergence(RECORDED_METRICS, result.comparisons, tmp / "d.md",
                             corner_counts=result.corner_counts).read_text()
    assert "Corner counts (information, never a failure)" in text and "convex hull" in text


def test_an_integer_off_by_one_is_caught_on_a_straight_row(synth):
    tmp, tif, npy = synth
    for col in EXACT_METRICS:
        rows = _good_rows(synth)[:1]
        rows[0][col] += 1
        res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
        bad = res.mismatches(EXACT_METRICS)
        assert [c.metric for c in bad] == [col] and bad[0].rule == "exact"


def test_oblique_rows_are_toleranced_not_exact_and_not_unbounded(synth):
    tmp, tif, npy = synth
    rows = [make_row(tif, npy, D_BENT, "JM1")]
    rows[0]["interface_area_px"] += 1                       # within 8 px: an injection artefact
    res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
    assert not res.mismatches(EXACT_METRICS) and res.comparisons[0].straight is False
    rows[0]["interface_area_px"] += 60                      # far beyond 10% / 8 px
    res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
    assert [c.metric for c in res.mismatches(EXACT_METRICS)] == ["interface_area_px"]
    rows = [make_row(tif, npy, D_BENT, "JM1")]
    rows[0]["marker_intensity"] = int(rows[0]["marker_intensity"] * 1.5)
    res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
    assert [c.metric for c in res.mismatches(EXACT_METRICS)] == ["marker_intensity"]


def test_a_float_metric_off_by_more_than_the_tolerance_is_caught_on_a_straight_row(synth):
    tmp, tif, npy = synth
    rows = _good_rows(synth)[:1]
    rows[0]["interface_contour_px"] = repr(float(rows[0]["interface_contour_px"]) * (1 + 1e-6))
    res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
    assert [c.metric for c in res.mismatches(TOLERANCE_METRICS)] == ["interface_contour_px"]
    # the same relative error on a non-straight row is only recorded (path-dependent)
    rows = [make_row(tif, npy, D_BENT, "JM1")]
    rows[0]["interface_contour_px"] = repr(float(rows[0]["interface_contour_px"]) * (1 + 1e-6))
    assert not compare_reference(load_reference(write_csv(tmp / "y.csv", rows))).mismatches(TOLERANCE_METRICS)


def test_linearity_is_compared_as_the_reciprocal_of_the_java_ratio(synth):
    tmp, tif, npy = synth
    ok = compare_reference(load_reference(write_csv(tmp / "x.csv", [make_row(tif, npy, D_BENT, "JM1")])))
    assert not ok.mismatches(("linearity_index",))
    # an operator who copied our-style linearity (<1) into the ratio column is refused at load
    rows = [make_row(tif, npy, D_BENT, "JM1")]
    rows[0]["java_distance_ratio"] = repr(1.0 / float(rows[0]["java_distance_ratio"]))
    with pytest.raises(ReferenceFormatError, match="linearity"):
        load_reference(write_csv(tmp / "y.csv", rows))
    # and a Java ratio that is merely wrong (still >= 1) is a linearity mismatch
    rows = [make_row(tif, npy, D_BENT, "JM1")]
    rows[0]["java_distance_ratio"] = repr(float(rows[0]["java_distance_ratio"]) + 2.0)
    bad = compare_reference(load_reference(write_csv(tmp / "z.csv", rows)))
    assert [c.metric for c in bad.mismatches(TOLERANCE_METRICS)] == ["linearity_index"]


def test_java_value_for_linearity_is_the_reciprocal_not_the_raw_ratio():
    row = {m: 1.0 for m in EXACT_METRICS + RECORDED_METRICS
           + ("interface_contour_px", "straight_line_length_px")}
    row.update(java_distance_ratio=1.25, role="JM2")
    assert java_values(row)["linearity_index"] == pytest.approx(0.8, rel=1e-12)
    row.update(java_distance_ratio=float("inf"))
    assert java_values(row)["linearity_index"] == 0.0
    row.update(java_distance_ratio=float("nan"))
    assert math.isnan(java_values(row)["linearity_index"])


def test_infinite_and_nan_java_values_are_accepted_and_recorded(synth):
    tmp, tif, npy = synth
    rows = [make_row(tif, npy, A_TOP, "JM1")]
    rows[0].update(java_distance_ratio="Infinity", coverage_index="NaN")
    loaded = load_reference(write_csv(tmp / "inf.csv", rows))
    assert math.isinf(loaded[0]["java_distance_ratio"]) and math.isnan(loaded[0]["coverage_index"])
    res = compare_reference(loaded)              # compares (and disagrees) instead of failing the load
    assert res.comparisons and not res.errors


def test_a_java_corner_off_our_boundary_is_a_path_disagreement_with_both_endpoints(synth):
    tmp, tif, npy = synth
    rows = _good_rows(synth)[:1]
    rows[0].update(java_corner_row=30, java_corner_col=20)        # the middle of cell A
    res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
    assert not res.comparisons and len(res.errors) == 1
    msg = res.errors[0]
    assert "path disagreement" in msg and "Java corners (30, 20)" in msg and "snapped" in msg


def test_a_mismatch_message_prints_both_endpoint_pairs_and_the_snap_distances(synth):
    tmp, tif, npy = synth
    rows = _good_rows(synth)[:1]
    rows[0]["interface_area_px"] += 1
    res = compare_reference(load_reference(write_csv(tmp / "x.csv", rows)))
    text = _describe(res.mismatches(EXACT_METRICS))
    assert "Java corners (16, 16) -> (16, 28)" in text
    assert "ours snapped (16, 16) -> (16, 28)" in text and "snap 0.00 / 0.00 px" in text
    assert "[exact]" in text


def _skips(synth_, rows, name="s.csv"):
    tmp = synth_[0]
    return compare_reference(load_reference(write_csv(tmp / name, rows)))


def test_border_touching_cell_is_skipped_with_a_reason(synth):
    tmp, tif, npy = synth
    row = make_row(tif, npy, A_TOP, "JM1")
    row.update(cell_id=C_BORDER[0], java_corner_row=2, java_corner_col=16, java_paired_row=2,
               java_paired_col=28)
    res = _skips(synth, [row])
    assert not res.errors and not res.comparisons
    (line, why, declared), = res.skipped
    assert "touches the image border" in why and declared is False


def test_operator_declared_non_interior_row_is_a_declared_skip(synth):
    tmp, tif, npy = synth
    res = _skips(synth, [make_row(tif, npy, A_TOP, "JM1", interior="no")])
    (line, why, declared), = res.skipped
    assert "not interior" in why and declared is True


def test_band_reaching_the_java_dead_zone_is_skipped_with_a_reason(synth):
    tmp, tif, npy = synth
    row = _good_rows(synth)[0]
    row.update(cell_id=E_DEAD[0], java_corner_row=3, java_corner_col=41, java_paired_row=3,
               java_paired_col=48)
    (line, why, declared), = _skips(synth, [row]).skipped
    assert "dead zone" in why and declared is False


def test_a_degenerate_cell_is_skipped_with_a_reason(synth):
    tmp, tif, npy = synth
    row = _good_rows(synth)[0]
    row.update(cell_id=F_BLOB[0], java_corner_row=31, java_corner_col=51, java_paired_row=31,
               java_paired_col=51)
    res = _skips(synth, [row])
    (line, why, declared), = res.skipped
    assert "degenerate" in why and declared is False and not res.errors


def test_a_pair_whose_two_arcs_both_fit_the_recorded_contour_is_skipped_as_ambiguous(synth):
    tmp, tif, npy = synth
    row = _good_rows(synth)[0]
    row.update(cell_id=D_OPPOSITE[0], java_corner_row=46, java_corner_col=36, java_paired_row=53,
               java_paired_col=42, interface_contour_px="13.0")
    res = _skips(synth, [row])
    (line, why, declared), = res.skipped
    assert "cannot say which one" in why and not res.comparisons


def test_band_reaching_the_dead_zone_helper():
    pix = np.array([[10, 3], [11, 3]])
    assert not java_dilation_is_unclipped(pix, (40, 40), 2)       # band reaches column 1
    assert java_dilation_is_unclipped(np.array([[10, 4], [11, 4]]), (40, 40), 2)
    assert not java_dilation_is_unclipped(np.array([[10, 36], [11, 36]]), (40, 40), 2)
    assert java_dilation_is_unclipped(np.array([[10, 35], [11, 35]]), (40, 40), 2)


def test_the_recorded_epsilon_reaches_the_corner_count_and_is_per_cell(synth):
    tmp, tif, npy = synth
    counts = {}
    for eps in (1, 30):
        p = _proto(tif, npy, *A_JOG, "JM1", eps=eps)
        geo = _geometry(p)
        counts[eps] = measure_arc(p, geo, select_arc(geo, None)).our_corners
    assert counts[1] != counts[30], counts
    a, b = make_row(tif, npy, A_TOP, "JM1", eps=2), make_row(tif, npy, A_TOP, "JM2", eps=1)
    with pytest.raises(ReferenceFormatError, match="epsilon is set once per cell"):
        load_reference(write_csv(tmp / "y.csv", [a, dict(b, java_corner_row=16, java_paired_col=27)]))


# ---- measure_arc against an independent oracle ------------------------------------------------
# The synthetic "Java" values above come from measure_arc itself, so they prove the plumbing but
# would not notice measure_arc measuring the wrong thing. The oracle restates the band metrics
# from their definitions with numpy only (an explicit shifted-OR dilation, no scipy, no
# junctionmapper.measure) on a HAND-LISTED pixel sequence.

A_TOP_PATH = [(16, c) for c in range(16, 29)]                                   # 13 px, contour 12
D_BENT_PATH = [(46, c) for c in range(36, 43)] + [(r, 42) for r in range(47, 51)]    # contour 10


def _dilate_by_shifts(mask: np.ndarray, cycles: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(cycles):
        padded = np.pad(out, 1)
        nxt = np.zeros_like(out)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                nxt |= padded[dr:dr + out.shape[0], dc:dc + out.shape[1]]
        out = nxt
    return out


def _oracle(tif, path, role, cycles, thr):
    import tifffile
    plane = tifffile.imread(tif)[..., 0 if role == "JM1" else 1].astype(np.int64)
    mask = np.zeros(plane.shape, bool)
    for r, c in path:
        mask[r, c] = True
    band = _dilate_by_shifts(mask, cycles)
    on = band & (plane > thr)
    steps = np.diff(np.array(path), axis=0)
    return {"interface_area_px": float(band.sum()), "marker_area_px": float(on.sum()),
            "marker_intensity": float(plane[on].sum()),
            "interface_contour_px": float(np.hypot(steps[:, 0], steps[:, 1]).sum())}


@pytest.mark.parametrize("pair, path, role, thr, cycles", [
    (A_TOP, A_TOP_PATH, "JM1", 100, 2), (A_TOP, A_TOP_PATH, "JM2", 60, 2),
    (A_TOP, A_TOP_PATH, "JM1", 30, 3), (D_BENT, D_BENT_PATH, "JM2", 200, 1),
    (D_BENT, D_BENT_PATH, "JM1", 100, 2),
])
def test_measure_arc_agrees_with_an_independent_oracle_on_a_hand_listed_path(
        synth, pair, path, role, thr, cycles):
    tmp, tif, npy = synth
    proto = _proto(tif, npy, *pair, role, cycles=cycles, thr=thr)
    got = measure_arc(proto, _geometry(proto), select_arc(_geometry(proto), None))
    want = _oracle(tif, path, role, cycles, thr)
    assert {k: got.metrics[k] for k in want} == want
    assert want["marker_area_px"] > 0           # the check has something to say


def test_the_oracle_itself_reproduces_a_hand_computed_band(synth):
    tmp, tif, npy = synth
    assert _oracle(tif, A_TOP_PATH, "JM1", 2, 100)["interface_area_px"] == 85.0
    assert _oracle(tif, A_TOP_PATH, "JM1", 1, 100)["interface_area_px"] == 3 * 15
    assert _oracle(tif, D_BENT_PATH, "JM1", 1, 100)["interface_contour_px"] == 10.0


# ---- frozen REAL Java output on the rectilinear control grid -----------------------------------
# Produced by scripts/java_reference.py (the real Java classes, headless) for cell 5 of
# java_reference.make_control(), epsilon 2, dilation 2, thresholds 100 / 100. Four Java
# interfaces x two roles. The straight side (82,56)->(47,56) is exact; the three that round a
# convex corner differ from ours by the Java's diagonal corner cut and compare at the stated
# tolerance. Running this needs no JDK: it ties the port to the original in every test run.
GOLDEN_JAVA_CONTROL_CELL5 = """\
image,labels,jm1_channel,jm2_channel,nucleus_channel,cell_id,role,corner_method,epsilon,dilation_cycles,threshold,interior,java_corner_row,java_corner_col,java_paired_row,java_paired_col,java_n_corners,interface_contour_px,straight_line_length_px,java_distance_ratio,fragmented_contour_px,coverage_index,interface_area_px,marker_area_px,marker_intensity,junction_contour_px,junction_straight_line_px,java_corner,java_paired_with,seed_row,seed_col,notes
control_grid.tif,control_grid.JM1.npy,0,1,2,5,JM1,dp,2,2,100,yes,47,56,47,98,4,43.41421356237309,42.0,1.0336717514850737,11.0,0.2533732410975573,244,142,25430,40.0,40.0,1,2,64,74,java_reference.py
control_grid.tif,control_grid.JM1.npy,0,1,2,5,JM2,dp,2,2,100,yes,47,56,47,98,4,43.41421356237309,42.0,1.0336717514850737,26.414213562373096,0.6084231728492296,244,150,27037,,,1,2,64,74,java_reference.py
control_grid.tif,control_grid.JM1.npy,0,1,2,5,JM1,dp,2,2,100,yes,47,98,83,97,4,36.41421356237309,36.013886210738214,1.0111159164909982,8.123105625617661,0.22307513552925634,209,116,20629,35.12310562561766,35.014282800023196,2,3,64,74,java_reference.py
control_grid.tif,control_grid.JM1.npy,0,1,2,5,JM2,dp,2,2,100,yes,47,98,83,97,4,36.41421356237309,36.013886210738214,1.0111159164909982,25.414213562373096,0.6979201546901914,209,124,22648,,,2,3,64,74,java_reference.py
control_grid.tif,control_grid.JM1.npy,0,1,2,5,JM1,dp,2,2,100,yes,83,97,82,56,4,41.41421356237309,41.012193308819754,1.0098024568090311,21.0,0.5070722873530444,234,159,27928,39.0,39.0,3,4,64,74,java_reference.py
control_grid.tif,control_grid.JM1.npy,0,1,2,5,JM2,dp,2,2,100,yes,83,97,82,56,4,41.41421356237309,41.012193308819754,1.0098024568090311,27.414213562373096,0.6619518084313037,234,137,24918,,,3,4,64,74,java_reference.py
control_grid.tif,control_grid.JM1.npy,0,1,2,5,JM1,dp,2,2,100,yes,82,56,47,56,4,35.0,35.0,1.0,7.0,0.2,200,114,19125,33.0,33.0,4,1,64,74,java_reference.py
control_grid.tif,control_grid.JM1.npy,0,1,2,5,JM2,dp,2,2,100,yes,82,56,47,56,4,35.0,35.0,1.0,22.0,0.6285714285714286,200,124,21602,,,4,1,64,74,java_reference.py
"""


def test_frozen_java_output_on_the_control_grid_matches_our_metrics(tmp_path):
    java_reference.make_control(tmp_path)
    p = tmp_path / "golden.csv"
    p.write_text(GOLDEN_JAVA_CONTROL_CELL5)
    rows = load_reference(p)
    assert {r["seed"] for r in rows} == {(64, 74)}      # (row, col) of the click; Java x=74, y=64
    result = compare_reference(rows)                    # also checks the click lies in cell 5
    assert not result.errors and not result.skipped
    assert len({c.line for c in result.comparisons}) == 8
    bad = result.mismatches(EXACT_METRICS + TOLERANCE_METRICS)
    assert not bad, _describe(bad)
    exact = [c for c in result.comparisons if c.straight and c.metric in EXACT_METRICS]
    assert {c.line for c in exact} == {8, 9}, "the straight left side, JM1 and JM2"
    assert all(c.ours == c.java for c in exact)
    # ...and the junction metrics (JM1 only) agree with the real Java
    assert {c.metric for c in result.comparisons} >= {"junction_contour_px", "junction_straight_line_px"}
    # the Java's corner count (4) is ours or not: information only
    assert result.corner_counts


# ---- the comparison tests, run against a synthetic reference ----------------------------------

def _use_reference(monkeypatch, tmp_path, path):
    monkeypatch.setattr(sys.modules[__name__], "REFERENCE", path)
    monkeypatch.setattr(sys.modules[__name__], "DIVERGENCE_MD", tmp_path / "divergence.md")
    _RESULT_CACHE.clear()


def test_the_comparison_tests_pass_and_write_divergence_md_on_a_consistent_reference(
        synth, monkeypatch, tmp_path):
    tmp, tif, npy = synth
    _use_reference(monkeypatch, tmp_path, write_csv(tmp / "good.csv", _good_rows(synth)))
    for fn in (test_reference_csv_is_well_formed_when_present, test_no_reference_row_was_silently_skipped,
               test_integer_metrics_match_java_exactly, test_ratio_metrics_match_java_within_tolerance,
               test_coverage_divergence_is_recorded_not_asserted_equal):
        assert outcome_of(fn) == ("passed", ""), fn.__name__
    text = (tmp_path / "divergence.md").read_text()
    assert "fragmented_contour_px" in text and "coverage_index" in text


@pytest.mark.parametrize("edit, which", [
    (lambda r: r[0].update(interface_area_px=r[0]["interface_area_px"] + 1),
     test_integer_metrics_match_java_exactly),
    (lambda r: r[1].update(marker_intensity=r[1]["marker_intensity"] - 1),
     test_integer_metrics_match_java_exactly),
    (lambda r: r[0].update(marker_area_px=r[0]["marker_area_px"] + 1),
     test_integer_metrics_match_java_exactly),
    (lambda r: r[0].update(java_distance_ratio=repr(float(r[0]["java_distance_ratio"]) + 0.5)),
     test_ratio_metrics_match_java_within_tolerance),
    (lambda r: r[2].update(java_distance_ratio=repr(float(r[2]["java_distance_ratio"]) + 2.0)),
     test_ratio_metrics_match_java_within_tolerance),
    (lambda r: r[4].update(straight_line_length_px=repr(float(r[4]["straight_line_length_px"]) + 1)),
     test_ratio_metrics_match_java_within_tolerance),
    (lambda r: r[0].update(junction_contour_px=repr(float(r[0]["junction_contour_px"]) + 1)),
     test_ratio_metrics_match_java_within_tolerance),
])
def test_each_comparison_test_fails_on_a_reference_that_disagrees(
        synth, monkeypatch, tmp_path, edit, which):
    tmp, tif, npy = synth
    rows = _good_rows(synth)
    edit(rows)
    _use_reference(monkeypatch, tmp_path, write_csv(tmp / "badvals.csv", rows))
    kind, msg = outcome_of(which)
    assert kind == "failed", (kind, msg)


def test_the_exact_assertions_fail_when_the_reference_has_no_straight_row(synth, monkeypatch, tmp_path):
    tmp, tif, npy = synth
    rows = [make_row(tif, npy, D_BENT, "JM1"), make_row(tif, npy, A_JOG, "JM1")]
    _use_reference(monkeypatch, tmp_path, write_csv(tmp / "nostraight.csv", rows))
    kind, msg = outcome_of(test_integer_metrics_match_java_exactly)
    assert kind == "failed" and "never applied" in msg and "--control" in msg


def test_the_recorded_divergence_test_does_not_fail_when_fragmentation_differs(
        synth, monkeypatch, tmp_path):
    tmp, tif, npy = synth
    rows = _good_rows(synth)
    rows[0]["fragmented_contour_px"] = "0.0"
    rows[0]["coverage_index"] = "0.0"
    _use_reference(monkeypatch, tmp_path, write_csv(tmp / "frag.csv", rows))
    assert outcome_of(test_coverage_divergence_is_recorded_not_asserted_equal)[0] == "passed"


@pytest.mark.parametrize("which", [
    test_reference_csv_is_well_formed_when_present, test_no_reference_row_was_silently_skipped,
    test_integer_metrics_match_java_exactly, test_ratio_metrics_match_java_within_tolerance,
    test_coverage_divergence_is_recorded_not_asserted_equal,
])
def test_every_comparison_test_fails_on_a_malformed_reference_and_skips_on_an_absent_one(
        synth, monkeypatch, tmp_path, which):
    bad = tmp_path / "reference.csv"
    bad.write_text("image,labels\nx,y\n")
    _use_reference(monkeypatch, tmp_path, bad)
    assert outcome_of(which)[0] == "failed"
    _use_reference(monkeypatch, tmp_path, tmp_path / "absent.csv")
    kind, msg = outcome_of(which)
    assert kind == "skipped" and "absent.csv" in msg and "README.md" in msg


# ---- schema validation -----------------------------------------------------------------------

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
    (lambda r: r[0].update(java_corner_row=""), "is empty"),
    (lambda r: r[0].update(java_corner_col=-1), "must be >= 0"),
    (lambda r: r[0].update(java_corner_row=1.5), "whole number"),
    (lambda r: r[0].update(java_n_corners=0), "java_n_corners"),
    (lambda r: r[0].update(java_distance_ratio="0.5"), "linearity"),
    (lambda r: r[0].update(nucleus_channel=0), "must differ"),
    (lambda r: r[0].update(labels="missing.npy"), "not found"),
    (lambda r: r[1].update(java_n_corners=r[1]["java_n_corners"] + 1, role="JM1",
                           java_corner_row=17), "java_n_corners"),
    (lambda r: r.append(dict(r[0])), "duplicates"),
])
def test_malformed_reference_rows_are_rejected(synth, edit, match):
    tmp, tif, npy = synth
    rows = _good_rows(synth)[:2]
    edit(rows)
    with pytest.raises(ReferenceFormatError, match=match):
        load_reference(write_csv(tmp / "bad.csv", rows))


@pytest.mark.parametrize("header", [
    [h for h in HEADER if h != "epsilon"],                      # missing column
    [h for h in HEADER if h != "java_corner_row"],              # the Java corner is required
    [("epsilon_px" if h == "epsilon" else h) for h in HEADER],  # typo'd column
    HEADER + ["cell_id"],                                       # duplicate column
])
def test_malformed_reference_header_is_rejected(synth, header):
    tmp, tif, npy = synth
    with pytest.raises(ReferenceFormatError):
        load_reference(write_csv(tmp / "bad.csv", _good_rows(synth)[:2], header=header))


def test_empty_and_header_only_references_are_rejected(synth):
    tmp, tif, npy = synth
    (tmp / "empty.csv").write_text("")
    with pytest.raises(ReferenceFormatError, match="empty"):
        load_reference(tmp / "empty.csv")
    with pytest.raises(ReferenceFormatError, match="no data rows"):
        load_reference(write_csv(tmp / "h.csv", []))


def test_short_row_is_rejected(synth):
    tmp, tif, npy = synth
    p = write_csv(tmp / "ok.csv", _good_rows(synth)[:2])
    p.write_text(p.read_text().rstrip("\n") + "\n1,2,3\n")
    with pytest.raises(ReferenceFormatError, match="fields"):
        load_reference(p)


# ---- skip / fail policy (asserted on the OUTCOME, so a failure that became a skip is seen) ----

def test_absent_reference_skips_naming_the_file_and_the_readme(tmp_path):
    kind, msg = outcome_of(reference_rows_or_outcome, tmp_path / "reference.csv")
    assert kind == "skipped" and "reference.csv" in msg and "README.md" in msg


def test_malformed_reference_fails_instead_of_skipping(tmp_path):
    (tmp_path / "reference.csv").write_text("not,the,schema\n1,2,3\n")
    kind, msg = outcome_of(reference_rows_or_outcome, tmp_path / "reference.csv")
    assert kind == "failed" and "not a valid reference" in msg


def test_unreadable_reference_fails_instead_of_skipping(tmp_path):
    (tmp_path / "reference.csv").write_bytes(b"\xff\xfe\x00bad")
    assert outcome_of(reference_rows_or_outcome, tmp_path / "reference.csv")[0] == "failed"


def test_a_tcj_row_fails_the_run_rather_than_being_compared(synth):
    tmp, tif, npy = synth
    rows = _good_rows(synth)[:2]
    rows[1]["corner_method"] = "tcj"
    p = write_csv(tmp / "reference.csv", rows)
    kind, msg = outcome_of(reference_rows_or_outcome, p)
    assert kind == "failed" and "no Java counterpart" in msg


def test_a_path_disagreement_fails_the_run_and_is_not_a_skip(synth):
    tmp, tif, npy = synth
    rows = _good_rows(synth)[:2]
    rows[1].update(java_corner_row=30, java_corner_col=20)
    kind, msg = outcome_of(recorded_result, write_csv(tmp / "noif.csv", rows))
    assert kind == "failed" and "path disagreement" in msg


def test_a_populated_reference_where_every_row_is_skipped_fails_naming_the_count(synth):
    tmp, tif, npy = synth
    rows = [make_row(tif, npy, A_TOP, "JM1", interior="no"),
            make_row(tif, npy, A_TOP, "JM2", interior="no")]
    kind, msg = outcome_of(recorded_result, write_csv(tmp / "allskip.csv", rows))
    assert kind == "failed"                       # NOT "skipped": that would look like no reference
    assert "2 rows" in msg and "none could be compared" in msg and "not interior" in msg


def test_a_reference_with_an_undeclared_skip_fails_the_silent_skip_check(synth, monkeypatch, tmp_path):
    tmp, tif, npy = synth
    border = _good_rows(synth)[0]
    border.update(cell_id=C_BORDER[0], java_corner_row=2, java_corner_col=16, java_paired_row=2,
                  java_paired_col=28, java_corner=9)
    rows = _good_rows(synth) + [border]
    _use_reference(monkeypatch, tmp_path, write_csv(tmp / "partial.csv", rows))
    kind, msg = outcome_of(test_no_reference_row_was_silently_skipped)
    assert kind == "failed" and "1 of" in msg and "touches the image border" in msg
    assert outcome_of(test_integer_metrics_match_java_exactly)[0] == "passed"   # others still run


def test_declared_skips_do_not_fail_the_silent_skip_check(synth, monkeypatch, tmp_path):
    tmp, tif, npy = synth
    rows = _good_rows(synth) + [make_row(tif, npy, D_TOP, "JM2", interior="no")]
    _use_reference(monkeypatch, tmp_path, write_csv(tmp / "declared.csv", rows))
    assert outcome_of(test_no_reference_row_was_silently_skipped)[0] == "passed"


# ---- divergence report -----------------------------------------------------------------------

def test_report_divergence_writes_a_legible_table(synth, tmp_path):
    tmp, tif, npy = synth
    res = compare_reference(load_reference(write_csv(tmp / "ok.csv", _good_rows(synth))))
    out = report_divergence(RECORDED_METRICS, res.comparisons, tmp_path / "divergence.md",
                            skipped=[(9, "example reason", False)],
                            corner_counts=res.corner_counts)
    text = out.read_text(encoding="utf-8")
    assert "| metric | class | rule |" in text
    assert "| interface_area_px | straight | exact |" in text
    assert "| interface_area_px | other | within 15% or 16 px |" in text
    assert "| linearity_index | straight | rel 1e-09 |" in text
    assert "## Recorded, not asserted: `fragmented_contour_px`" in text
    assert "## Recorded, not asserted: `coverage_index`" in text
    assert "| line | image | cell | role | eps | Java | ours | abs diff | rel diff |" in text
    assert text.count("| divergence 3 |") + text.count("| divergence 3 (+ path") == 5
    assert text.count("numerator (fragmented contour) differs: divergence 3") >= 5
    assert "| 7 |" in text                               # the planted +7 shows as the abs diff
    assert "example reason" in text and "UNEXPLAINED:" not in text
    out2 = report_divergence(RECORDED_METRICS, res.comparisons, tmp_path / "d2.md",
                             skipped=[(9, "example reason", False)], corner_counts=res.corner_counts)
    assert out.read_text() == out2.read_text()           # deterministic: no timestamp


def _cmp(metric, java, ours, line=2, straight=True):
    return Comparison(line=line, image="i", cell_id=1, role="JM1", epsilon=2, metric=metric,
                      java=java, ours=ours, straight=straight)


def test_fragmented_attribution_checks_the_sign(tmp_path):
    cs = [_cmp("fragmented_contour_px", 9.0, 5.0, line=2),     # Java > ours: divergence 3
          _cmp("fragmented_contour_px", 3.0, 5.0, line=3),     # Java < ours: not explained
          _cmp("fragmented_contour_px", 5.0, 5.0, line=4)]
    text = report_divergence(RECORDED_METRICS, cs, tmp_path / "d.md").read_text()
    rows = {int(l.split("|")[1]): l for l in text.splitlines() if l.startswith("| ") and l[2].isdigit()}
    assert rows[2].rstrip().endswith("| divergence 3 |")
    assert "UNEXPLAINED: Java < ours" in rows[3]
    assert rows[4].rstrip().endswith("| agrees |")


def test_coverage_attribution_names_the_side_of_the_ratio_that_differs(tmp_path):
    def table(cs):
        text = report_divergence(RECORDED_METRICS, cs, tmp_path / "d.md").read_text()
        cov = text.split("`coverage_index`")[-1]
        return [l for l in cov.splitlines() if l.startswith("| 2 ")][0]

    same_c = _cmp("interface_contour_px", 10.0, 10.0)
    assert "numerator (fragmented contour) differs: divergence 3" in table(
        [same_c, _cmp("fragmented_contour_px", 9.0, 5.0), _cmp("coverage_index", 0.9, 0.5)])
    assert "numerator (fragmented contour) differs and Java < ours" in table(
        [same_c, _cmp("fragmented_contour_px", 3.0, 5.0), _cmp("coverage_index", 0.3, 0.5)])
    assert "denominator (interface contour) differs: path disagreement" in table(
        [_cmp("interface_contour_px", 11.0, 10.0), _cmp("fragmented_contour_px", 5.0, 5.0),
         _cmp("coverage_index", 0.45, 0.5)])
    assert "UNEXPLAINED: neither" in table(
        [same_c, _cmp("fragmented_contour_px", 5.0, 5.0), _cmp("coverage_index", 0.9, 0.5)])


def test_report_divergence_reports_a_zero_java_value_without_dividing_by_zero(tmp_path):
    c = _cmp("fragmented_contour_px", 0.0, 3.0)
    text = report_divergence(RECORDED_METRICS, [c], tmp_path / "d.md").read_text()
    assert "n/a (Java is 0)" in text


# ---- the generator (scripts/java_reference.py) -------------------------------------------------

def test_generator_header_is_the_loaders_schema():
    assert java_reference.HEADER == HEADER


CANNED_JAVA = """\
cell,seedx,seedy,ncorners,ncellpx,nedgepx,minx,maxx,miny,maxy,corner,cx,cy,paired,px,py,contour,euclid,ratio,frag,iarea,m1area,m1int,junc_contour,junc_euclid,m2area,m2int,m2dist
1,30,100,4,100,40,10,30,10,30,1,10,12,2,30,12,20.0,20.0,1.0,5.0,100,60,6000.0,18.0,17.0,50,5000.0,9.0
1,30,100,4,100,40,10,30,10,30,2,30,12,3,30,28,0.0,0.0,Infinity,0.0,0,0,0.0,0.0,0.0,0,0.0,0.0
2,70,100,3,100,40,3,30,10,30,1,1,12,2,30,12,20.0,20.0,1.0,5.0,100,60,6000.0,18.0,17.0,50,5000.0,9.0
"""


def test_generator_turns_java_output_into_valid_reference_rows(tmp_path):
    import io
    jrows = list(csv.DictReader(io.StringIO(CANNED_JAVA)))
    java_reference.make_control(tmp_path)
    rows = java_reference.rows_from_java(jrows, [7, 8], "control_grid.tif", "control_grid.JM1.npy",
                                         jm1=0, jm2=1, nucleus=2, eps=2, dil=2, t1=100, t2=60,
                                         shape=(40, 40))
    assert len(rows) == 6                                          # JM1 and JM2 per Java interface
    p = tmp_path / "gen.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(java_reference.HEADER)
        for r in rows:
            w.writerow([r.get(h, "") for h in java_reference.HEADER])
    loaded = load_reference(p)                                     # also: Infinity/NaN accepted
    jm1, jm2 = loaded[0], loaded[1]
    assert (jm1["role"], jm2["role"]) == ("JM1", "JM2") and jm1["threshold"] == 100 and jm2["threshold"] == 60
    assert jm1["fragmented_contour_px"] == 5.0 and jm2["fragmented_contour_px"] == 9.0   # m2dist
    assert jm1["marker_area_px"] == 60 and jm2["marker_area_px"] == 50
    assert jm1["marker_intensity"] == 6000 and jm2["marker_intensity"] == 5000
    assert jm1["junction_contour_px"] == 18.0 and jm2["junction_contour_px"] is None
    assert (jm1["java_corner_row"], jm1["java_corner_col"]) == (12, 10)       # (y, x) -> (row, col)
    assert math.isclose(jm1["coverage_index"], 0.25) and math.isclose(jm2["coverage_index"], 0.45)
    assert jm1["seed"] == (100, 30) and loaded[4]["seed"] == (100, 70)        # (seedy, seedx)
    assert math.isinf(loaded[2]["java_distance_ratio"]) and math.isnan(loaded[2]["coverage_index"])
    assert [r["cell_id"] for r in loaded] == [7, 7, 7, 7, 8, 8]               # seeds map to cell ids
    assert loaded[0]["interior"] is True and loaded[4]["interior"] is False   # minx 3 - 2 < 2


def _seeded(synth_, **kw):
    tmp, tif, npy = synth_
    return [dict(_good_rows(synth_)[0], **kw)]


def test_seed_columns_are_optional(synth):
    tmp = synth[0]
    assert load_reference(write_csv(tmp / "n.csv", _seeded(synth)))[0]["seed"] is None
    assert load_reference(write_csv(tmp / "s.csv", _seeded(synth, seed_row=30, seed_col=20)))[0]["seed"] == (30, 20)


def test_seed_pair_mutant_row_blank_col_set_is_rejected(synth):
    with pytest.raises(ReferenceFormatError, match="seed_col is given without seed_row"):
        load_reference(write_csv(synth[0] / "b.csv", _seeded(synth, seed_row="", seed_col=3)))


def test_seed_pair_mutant_row_set_col_blank_is_rejected(synth):
    with pytest.raises(ReferenceFormatError, match="seed_row is given without seed_col"):
        load_reference(write_csv(synth[0] / "b.csv", _seeded(synth, seed_row=30, seed_col="")))


@pytest.mark.parametrize("bad, match", [({"seed_row": -1, "seed_col": 3}, ">= 0"),
                                        ({"seed_row": 1.5, "seed_col": 3}, "whole number")])
def test_malformed_seed_values_are_rejected(synth, bad, match):
    with pytest.raises(ReferenceFormatError, match=match):
        load_reference(write_csv(synth[0] / "b.csv", _seeded(synth, **bad)))


def test_seed_bounds_mutant_row_beyond_the_image_is_rejected_cleanly(synth):
    with pytest.raises(ReferenceFormatError, match="not inside that cell"):
        load_reference(write_csv(synth[0] / "b.csv", _seeded(synth, seed_row=500, seed_col=20)))


def test_seed_bounds_mutant_col_beyond_the_image_is_rejected_cleanly(synth):
    with pytest.raises(ReferenceFormatError, match="not inside that cell"):
        load_reference(write_csv(synth[0] / "b.csv", _seeded(synth, seed_row=30, seed_col=500)))


def test_seed_row_col_swap_mutant_is_pinned_by_a_seed_inside_in_one_order_only(synth):
    tmp, tif, npy = synth
    lab = np.load(npy)
    assert lab[40, 20] == 1 and lab[20, 40] == 2          # (row 40, col 20) is cell 1; swapped is cell 2
    ok = load_reference(write_csv(tmp / "ok.csv", _seeded(synth, seed_row=40, seed_col=20)))
    assert ok[0]["seed"] == (40, 20)
    with pytest.raises(ReferenceFormatError, match=r"\(row 20, col 40\) is not inside that cell"):
        load_reference(write_csv(tmp / "sw.csv", _seeded(synth, seed_row=20, seed_col=40)))


def test_a_seed_outside_the_cell_is_rejected_even_when_the_row_is_not_compared(synth):
    rows = _seeded(synth, seed_row=50, seed_col=50, interior="no")      # background; row would be skipped
    with pytest.raises(ReferenceFormatError, match="not inside that cell"):
        load_reference(write_csv(synth[0] / "b.csv", rows))


def test_rows_of_one_cell_must_carry_the_same_click(synth):
    tmp, tif, npy = synth
    a = dict(_good_rows(synth)[0], seed_row=40, seed_col=20)
    b = dict(_good_rows(synth)[1], seed_row=41, seed_col=20)           # same cell 1, different click
    with pytest.raises(ReferenceFormatError, match="one cell is one click"):
        load_reference(write_csv(tmp / "mix.csv", [a, b]))
    c = dict(_good_rows(synth)[1])                                      # same cell, click missing
    with pytest.raises(ReferenceFormatError, match="one cell is one click"):
        load_reference(write_csv(tmp / "mix2.csv", [a, c]))
    assert len(load_reference(write_csv(tmp / "same.csv", [a, dict(b, seed_row=40)]))) == 2


def test_generator_chooses_interior_cells_and_seeds_inside_them():
    lab = np.zeros((50, 50), np.int32)
    lab[10:30, 10:30] = 1
    lab[0:8, 0:8] = 2                              # touches the border
    lab[40:44, 40:44] = 3                          # too close to the border for dilation 2
    view = edge_map.java_view_labels(lab)
    assert java_reference.choose_cells(view, 2, None, None) == [1]
    x, y = java_reference.seed_for(view, 1)
    assert view[y, x] == 1
    with pytest.raises(java_reference.GeneratorError, match="too close"):
        java_reference.choose_cells(view, 2, [2], None)


def test_generator_control_field_is_deterministic_and_rectilinear(tmp_path):
    a = java_reference.make_control(tmp_path / "a")
    b = java_reference.make_control(tmp_path / "b")
    assert a[0].read_bytes() == b[0].read_bytes() and (np.load(a[1]) == np.load(b[1])).all()
    lab = np.load(a[1])
    view = edge_map.java_view_labels(lab)
    for cid in range(1, 10):
        rr, cc = np.nonzero(view == cid)
        assert (view[rr.min():rr.max() + 1, cc.min():cc.max() + 1] == cid).all()    # a rectangle


def test_generator_refuses_a_work_dir_inside_the_java_tree(tmp_path):
    java = tmp_path / "Junction_Mapper"
    (java / "uk").mkdir(parents=True)
    (java / "lib").mkdir()
    with pytest.raises(java_reference.GeneratorError, match="inside the Java checkout"):
        java_reference.compile_java(java, java / "scratch")
    assert not (java / "scratch").exists()          # nothing was written into the oracle


@pytest.mark.parametrize("missing, match", [("java", "java not found"), ("javac", "javac not found")])
def test_generator_names_a_missing_jdk_tool_in_one_clean_line(monkeypatch, tmp_path, missing, match):
    monkeypatch.setattr(java_reference.shutil, "which",
                        lambda t: None if t == missing else "/usr/bin/" + t)
    (tmp_path / "uk").mkdir()
    (tmp_path / "lib").mkdir()
    with pytest.raises(java_reference.GeneratorError, match=match):
        java_reference.check_prerequisites(tmp_path)


def test_generator_names_a_missing_java_checkout_or_jars(monkeypatch, tmp_path):
    monkeypatch.setattr(java_reference.shutil, "which", lambda t: "/usr/bin/" + t)
    with pytest.raises(java_reference.GeneratorError, match="no Java sources at"):
        java_reference.check_prerequisites(tmp_path)
    (tmp_path / "uk").mkdir()
    with pytest.raises(java_reference.GeneratorError, match="no jars in"):
        java_reference.check_prerequisites(tmp_path)


def test_generator_cli_reports_prerequisites_as_one_error_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(java_reference.shutil, "which", lambda t: None)
    rc = java_reference.main(["--work-dir", str(tmp_path / "w"), "--out", str(tmp_path / "r.csv"),
                              "--control", "--java-dir", str(tmp_path)])
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == 1 and len(err) == 1 and err[0].startswith("error:") and "not found on PATH" in err[0]


def _jdk_missing_reason():
    import shutil
    if shutil.which("java") is None or shutil.which("javac") is None:
        return ("java/javac not found on PATH (the Apptainer image has no JDK; run "
                "scripts/java_reference.py on the host)")
    if not (java_reference.DEFAULT_JAVA_DIR / "uk").is_dir() or \
            not list((java_reference.DEFAULT_JAVA_DIR / "lib").glob("*.jar")):
        return (f"the Java checkout with uk/ and lib/*.jar is not visible at "
                f"{java_reference.DEFAULT_JAVA_DIR} (the Apptainer image only mounts this repository)")
    return None


@pytest.mark.skipif(_jdk_missing_reason() is not None, reason=_jdk_missing_reason() or "")
def test_generator_runs_the_real_java_on_the_control_grid_and_the_port_agrees(tmp_path):
    out = tmp_path / "ref" / "reference.csv"
    rc = java_reference.main(["--work-dir", str(tmp_path / "work"), "--out", str(out), "--control"])
    assert rc == 0
    result = compare_reference(load_reference(out))
    assert not result.errors
    assert not result.mismatches(EXACT_METRICS + TOLERANCE_METRICS), _describe(
        result.mismatches(EXACT_METRICS + TOLERANCE_METRICS))
    assert any(c.straight for c in result.comparisons)


def test_the_readme_documents_the_exact_header():
    text = README.read_text(encoding="utf-8")
    assert ",".join(HEADER) in text, "README.md must show the exact reference.csv header"
