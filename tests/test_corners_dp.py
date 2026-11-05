import numpy as np
import pytest
from skimage.draw import polygon

from junctionmapper.errors import ConfigError
from junctionmapper.geometry.boundary import trace_boundary
from junctionmapper.geometry.corners_dp import dp_corners


def square_labels():
    labels = np.zeros((10, 10), int)
    labels[2:8, 2:8] = 1
    return labels


def hexagon_labels(radius=16, size=40):
    """Regular hexagon (side == radius == 16 px) with vertices at 0, 60, ... degrees."""
    c = size / 2
    ang = np.deg2rad(np.arange(6) * 60)
    labels = np.zeros((size, size), int)
    rr, cc = polygon(c + radius * np.sin(ang), c + radius * np.cos(ang), labels.shape)
    labels[rr, cc] = 1
    return labels


def test_square_yields_four_corners():
    b = trace_boundary(square_labels(), 1)
    assert len(dp_corners(b, epsilon=1.5)) == 4


def test_corner_indices_land_on_the_geometric_corners():
    b = trace_boundary(square_labels(), 1)
    pts = {tuple(p) for p in b[dp_corners(b, 1.5)]}
    assert pts == {(2, 2), (2, 7), (7, 7), (7, 2)}


def test_larger_epsilon_yields_no_more_corners():
    b = trace_boundary(hexagon_labels(), 1)
    assert len(dp_corners(b, 5.0)) <= len(dp_corners(b, 1.0))


def test_hexagon_has_six_corners_at_moderate_epsilon():
    b = trace_boundary(hexagon_labels(), 1)
    assert len(dp_corners(b, 2.5)) == 6


def test_epsilon_monotone_over_a_sweep():
    b = trace_boundary(hexagon_labels(), 1)
    counts = [len(dp_corners(b, e)) for e in (0.5, 1, 2, 3, 5, 8, 20)]
    assert counts == sorted(counts, reverse=True)


def test_indices_sorted_unique_and_in_range():
    b = trace_boundary(hexagon_labels(), 1)
    idx = dp_corners(b, 1.0)
    assert np.issubdtype(idx.dtype, np.integer)
    assert np.all(np.diff(idx) > 0) and idx[0] >= 0 and idx[-1] < len(b)


def test_epsilon_larger_than_shape_returns_the_two_seeds():
    b = trace_boundary(square_labels(), 1)
    idx = dp_corners(b, 1000.0)
    assert len(idx) == 2
    d = np.hypot(*(b[idx[0]] - b[idx[1]]))
    assert d == pytest.approx(np.hypot(5, 5))      # a diagonal: the most distant pair


@pytest.mark.parametrize("eps", [0, -1.0, float("nan"), float("inf")])
def test_bad_epsilon_rejected(eps):
    b = trace_boundary(square_labels(), 1)
    with pytest.raises(ConfigError):
        dp_corners(b, eps)


def test_too_short_boundary_rejected():
    with pytest.raises(ConfigError):
        dp_corners(np.array([[0, 0], [0, 1]]), 1.0)


def test_bad_shape_rejected():
    with pytest.raises(ConfigError):
        dp_corners(np.zeros((5, 3)), 1.0)


def test_repeated_pixel_in_neck_boundary_is_handled():
    labels = np.zeros((12, 20), int)
    labels[2:9, 2:7] = 1
    labels[5, 7:12] = 1                      # one-pixel-wide neck
    labels[2:9, 12:17] = 1
    b = trace_boundary(labels, 1)
    assert len({tuple(p) for p in b}) < len(b)   # precondition: a pixel repeats
    idx = dp_corners(b, 1.5)
    assert len(idx) >= 4 and np.all(np.diff(idx) > 0)
