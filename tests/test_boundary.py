import numpy as np
from scipy import ndimage
import pytest

from junctionmapper.errors import BoundaryError, JunctionMapperError
from junctionmapper.geometry.boundary import trace_boundary


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


def test_error_is_package_error():
    assert issubclass(BoundaryError, JunctionMapperError)


def test_rows_cols_order_and_edge_touching_label():
    labels = np.zeros((10, 20), int); labels[0:3, 0:8] = 2     # touches image border
    b = trace_boundary(labels, 2)
    assert b.shape[1] == 2 and b[:, 0].max() == 2 and b[:, 1].max() == 7
    assert np.all(labels[b[:, 0], b[:, 1]] == 2)
    assert len(b) == 2 * 8 + 2 * 1                       # 3x8 block: all but 6 interior
    assert len({tuple(p) for p in b}) == len(b)


def test_diamond_and_neck_shapes_are_closed_walks():
    labels = np.zeros((12, 12), int)
    labels[2:5, 2:5] = 1; labels[5, 4] = 1; labels[6:9, 3:6] = 1   # one-pixel neck
    b = trace_boundary(labels, 1)
    d = np.abs(np.diff(np.vstack([b, b[:1]]), axis=0)).max(axis=1)
    assert np.all(d <= 1) and np.all(d >= 1)
    assert (5, 4) in {tuple(p) for p in b}               # the neck is on the walk
    assert np.all(labels[b[:, 0], b[:, 1]] == 1)


def test_other_labels_do_not_block_trace():
    labels = np.zeros((10, 10), int); labels[2:8, 2:5] = 1; labels[2:8, 5:8] = 2
    b = trace_boundary(labels, 1)
    assert len(b) == 6 * 3 - 4                           # 6x3 block minus 4 interior
    assert np.all(labels[b[:, 0], b[:, 1]] == 1)


def test_trace_covers_every_boundary_pixel():
    """Contour lengths are counted along this walk, so a pixel the trace cuts past
    would shorten the measurement with nothing in the output to show it."""
    labels = np.zeros((12, 14), int)
    labels[2:6, 2:6] = 1
    labels[2:6, 8:12] = 1
    labels[3, 6:9] = 1  # one-pixel-wide neck: the hard case for Moore tracing
    b = trace_boundary(labels, 1)
    mask = labels == 1
    edge = mask & ~ndimage.binary_erosion(mask, ndimage.generate_binary_structure(2, 1))
    assert {tuple(p) for p in np.argwhere(edge)} <= {tuple(p) for p in b}


def _assert_closed_walk(labels, b, label=1):
    assert b.ndim == 2 and b.shape[1] == 2
    assert np.all(labels[b[:, 0], b[:, 1]] == label)
    wrap = np.vstack([b, b[:1]])
    d = np.abs(np.diff(wrap, axis=0)).max(axis=1)
    assert np.all(d == 1)


def test_diagonal_band_traces():
    L = np.zeros((12, 14), int)
    for r in range(2, 9):
        L[r, 2 + (r - 2): 5 + (r - 2)] = 1
    _assert_closed_walk(L, trace_boundary(L, 1))


def test_start_pixel_with_only_east_and_southeast_neighbours():
    L = np.zeros((10, 10), int)
    L[2, 2:4] = 1                       # top row: start (2,2) has E; below is offset right
    L[3, 3:7] = 1; L[4, 3:7] = 1
    b = trace_boundary(L, 1)
    _assert_closed_walk(L, b)


def test_start_pixel_with_southwest_neighbour():
    L = np.zeros((10, 10), int)
    L[2, 4:6] = 1                       # start (2,4); rows below extend left
    L[3, 2:6] = 1; L[4, 2:5] = 1
    b = trace_boundary(L, 1)
    _assert_closed_walk(L, b)
    assert {tuple(p) for p in b} >= {(2, 4), (3, 2), (4, 2)}


def _invariants(mask, b):
    """Every entry on the mask, every step 8-adjacent including the wrap, and no
    boundary pixel left out — the contract the corner and interface tasks rely on."""
    assert all(mask[tuple(p)] for p in b)
    steps = np.abs(np.diff(np.vstack([b, b[:1]]), axis=0)).max(axis=1)
    assert steps.max() <= 1 and steps.min() >= 1
    cross = ndimage.generate_binary_structure(2, 1)
    edge = mask & ~ndimage.binary_erosion(mask, cross)
    assert not ({tuple(p) for p in np.argwhere(edge)} - {tuple(p) for p in b})


def test_invariants_hold_across_many_random_shapes():
    """A property sweep, not an example. The stopping criterion was once wrong in a
    way every rectangle-shaped test missed while rejecting ~4% of real cells, so the
    invariants are pinned over a population of shapes rather than three hand-picked ones."""
    rng = np.random.default_rng(7)
    eight = np.ones((3, 3), bool)
    checked = 0
    for _ in range(300):
        size = int(rng.integers(14, 40))
        blob = ndimage.gaussian_filter(rng.random((size, size)), rng.uniform(1.5, 4.0))
        labels, n = ndimage.label(blob > rng.uniform(0.45, 0.58), structure=eight)
        for k in range(1, n + 1):
            cell = labels == k
            cell[0, :] = cell[-1, :] = cell[:, 0] = cell[:, -1] = False
            if cell.sum() < 8:
                continue
            _, parts = ndimage.label(cell, structure=eight)
            if parts != 1 or (ndimage.binary_fill_holes(cell) & ~cell).any():
                continue  # rejected by a guard, not by the trace
            _invariants(cell, trace_boundary(cell.astype(int), 1))
            checked += 1
    assert checked > 400, f"sweep degenerated to {checked} shapes"
