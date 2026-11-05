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
