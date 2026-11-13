"""snap_to_membrane: move Cellpose cell-cell boundaries onto the membrane ridge, then smooth."""
import numpy as np

from junctionmapper.segment.snap import snap_to_membrane

# Two cells side by side: cell 1 cols 20..49, cell 2 cols 50..79, rows 20..79.
SHAPE = (100, 100)


def two_cells():
    lab = np.zeros(SHAPE, np.uint16)
    lab[20:80, 20:50] = 1
    lab[20:80, 50:80] = 2
    return lab


def ridge_at(col, shape=SHAPE):
    cols = np.arange(shape[1])
    return np.tile(100.0 * np.exp(-0.5 * ((cols - col) / 1.5) ** 2), (shape[0], 1)).astype(np.float32)


def last_col_of_cell1(lab, rows=slice(35, 65)):
    """Per middle row, the last column of cell 1 (the boundary position)."""
    return np.array([np.nonzero(r == 1)[0].max() for r in lab[rows]])


def test_boundary_moves_onto_a_membrane_ridge_within_reach():
    out = snap_to_membrane(two_cells(), ridge_at(55))
    edge = last_col_of_cell1(out)
    assert np.all(np.abs(edge - 54.5) <= 1.5), edge     # boundary at the ridge, not at col 49/50


def test_boundary_already_on_the_ridge_stays_put():
    out = snap_to_membrane(two_cells(), ridge_at(49.5))
    assert np.all(np.abs(last_col_of_cell1(out) - 49) <= 1)


def test_boundary_never_moves_further_than_the_erosion():
    """A ridge 20 px away is out of reach: the boundary may move at most erode_px."""
    out = snap_to_membrane(two_cells(), ridge_at(70), erode_px=8)
    assert np.all(last_col_of_cell1(out) <= 49 + 8 + 1)


def test_foreground_and_background_are_unchanged():
    """Edges facing background are not cell-cell junctions; only cell-cell boundaries move."""
    lab = two_cells()
    out = snap_to_membrane(lab, ridge_at(55))
    assert np.array_equal(out > 0, lab > 0)


def test_labels_keep_dtype_ids_and_no_cell_is_lost():
    lab = two_cells()
    lab[40:44, 80:84] = 3                    # a 4x4 cell, smaller than the erosion window
    out = snap_to_membrane(lab, ridge_at(55))
    assert out.dtype == lab.dtype
    assert set(np.unique(out)) == set(np.unique(lab))
    assert (out == 3).sum() > 0


def test_boundaries_are_smoothed_not_traced_through_specks():
    """Punctate membrane: bright specks scattered either side of a straight junction must not
    make the boundary zig-zag. Smoothing keeps the boundary within a couple of px of straight."""
    rng = np.random.default_rng(0)
    mem = ridge_at(52) * 0.5
    for r, c in zip(rng.integers(20, 80, 40), rng.integers(44, 60, 40)):
        mem[r, c] += 300.0
    edge = last_col_of_cell1(snap_to_membrane(two_cells(), mem), rows=slice(30, 70))
    assert edge.max() - edge.min() <= 3, edge


def test_no_cells_is_returned_unchanged():
    lab = np.zeros(SHAPE, np.int32)
    out = snap_to_membrane(lab, ridge_at(50))
    assert np.array_equal(out, lab) and out.dtype == lab.dtype
