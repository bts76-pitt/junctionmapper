import numpy as np
import pytest
from scipy import ndimage
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


def polygon_labels(verts, shape=(30, 30)):
    v = np.array(verts)
    labels = np.zeros(shape, int)
    rr, cc = polygon(v[:, 0], v[:, 1], shape)
    labels[rr, cc] = 1
    return labels


def concave_labels():
    return polygon_labels([[20, 21], [8, 5], [13, 7], [7, 8], [17, 21]])


def l_labels():
    labels = np.zeros((40, 40), int)
    labels[5:35, 5:12] = 1
    labels[28:35, 5:35] = 1
    return labels


def necked_labels():
    labels = np.zeros((12, 20), int)
    labels[2:9, 2:7] = 1
    labels[5, 7:12] = 1                      # one-pixel-wide neck
    labels[2:9, 12:17] = 1
    return labels


def coords(b, idx):
    return {(int(r), int(c)) for r, c in b[idx]}


def test_square_yields_four_corners():
    b = trace_boundary(square_labels(), 1)
    assert len(dp_corners(b, epsilon=1.5)) == 4


def test_corner_indices_land_on_the_geometric_corners():
    b = trace_boundary(square_labels(), 1)
    pts = {tuple(p) for p in b[dp_corners(b, 1.5)]}
    assert pts == {(2, 2), (2, 7), (7, 7), (7, 2)}


def test_hexagon_has_six_corners_at_moderate_epsilon():
    b = trace_boundary(hexagon_labels(), 1)
    assert len(dp_corners(b, 2.5)) == 6


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


def test_epsilon_monotone_over_a_sweep_on_several_shapes():
    shapes = [square_labels(), hexagon_labels(), concave_labels(), l_labels(), necked_labels()]
    for labels in shapes:
        b = trace_boundary(labels, 1)
        counts = [len(dp_corners(b, e)) for e in (0.5, 1, 1.5, 2, 3, 5, 8, 20, 1000)]
        assert counts == sorted(counts, reverse=True)
        assert counts[-1] == 2


def test_necked_boundary_keeps_distinct_ascending_indices_and_both_tips():
    b = trace_boundary(necked_labels(), 1)
    assert len({tuple(p) for p in b}) < len(b)           # precondition: pixels repeat
    idx = dp_corners(b, 1.5)
    assert np.all(np.diff(idx) > 0)
    assert coords(b, idx) == {(2, 2), (2, 6), (5, 7), (2, 16), (8, 16), (8, 12), (5, 11), (8, 2)}
    # each neck tip is visited twice by the boundary; exactly one visit is a corner
    for tip in [(5, 7), (5, 11)]:
        visits = [k for k in range(len(b)) if tuple(b[k]) == tip]
        assert len(visits) == 2 and sum(k in set(idx) for k in visits) == 1


def test_three_collinear_points_return_only_the_two_seeds():
    idx = dp_corners(np.array([[0, 0], [0, 1], [0, 2]]), 0.5)
    assert coords(np.array([[0, 0], [0, 1], [0, 2]]), idx) == {(0, 0), (0, 2)}


# --- tests that pin specific wrong implementations (mutants) ---------------------

def test_kills_segment_distance_mutant_uses_distance_to_infinite_chord():
    # Concave shape: the point (11, 10) projects beyond the end of its sub-chord, so
    # distance-to-segment would measure to an endpoint and drop this corner.
    b = trace_boundary(concave_labels(), 1)
    assert coords(b, dp_corners(b, 1.5)) == {
        (7, 8), (8, 5), (10, 7), (10, 8), (11, 10), (13, 7), (17, 21), (20, 21)}


def test_kills_non_farthest_seed_mutants_seeds_are_the_most_distant_pair():
    b = trace_boundary(polygon_labels([[18, 9], [15, 11], [15, 2], [19, 5]]), 1)
    d2 = ((b[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    idx = dp_corners(b, 1000.0)                                  # huge epsilon: seeds only
    assert 0 not in idx                         # trace origin is not a seed
    assert coords(b, idx) == {(15, 2), (15, 11)}
    assert d2[idx[0], idx[1]] == d2.max()


def test_kills_unroll_drops_endpoint_i_mutant_wrap_arc_corner_is_returned():
    b = trace_boundary(polygon_labels([[7, 7], [11, 4], [14, 9], [8, 12], [8, 10]]), 1)
    idx = dp_corners(b, 2.5)
    assert idx[0] == 0 and tuple(b[0]) == (7, 7)                 # corner lies on the wrap arc
    assert coords(b, idx) == {(7, 7), (8, 12), (11, 4), (14, 9)}


# --- roll invariance --------------------------------------------------------------

@pytest.mark.parametrize("make", [square_labels, hexagon_labels, concave_labels, l_labels])
@pytest.mark.parametrize("eps", [1.0, 1.5, 3.0])
def test_corner_coordinates_are_invariant_to_where_the_boundary_starts(make, eps):
    b = trace_boundary(make(), 1)
    ref = coords(b, dp_corners(b, eps))
    for shift in (1, 3, 7, len(b) // 3, len(b) // 2, len(b) - 1):
        r = np.roll(b, shift, axis=0)
        assert coords(r, dp_corners(r, eps)) == ref


def _random_blob(rng):
    noise = ndimage.gaussian_filter(rng.random((40, 40)), 3)
    comp, n = ndimage.label(noise > np.quantile(noise, 0.6))
    if n == 0:
        return None
    biggest = np.argmax(np.bincount(comp.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(comp == biggest).astype(int)


def test_roll_invariance_on_random_blobs():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(120):
        labels = _random_blob(rng)
        try:
            b = trace_boundary(labels, 1)
        except Exception:
            continue
        if len({tuple(p) for p in b}) != len(b):
            continue                                              # necks: duplicate coords
        for eps in (1.0, 3.0):
            ref = coords(b, dp_corners(b, eps))
            for shift in (1, len(b) // 2, len(b) - 3):
                r = np.roll(b, shift, axis=0)
                assert coords(r, dp_corners(r, eps)) == ref
        checked += 1
    assert checked > 50
