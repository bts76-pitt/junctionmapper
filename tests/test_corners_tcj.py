import numpy as np
import pytest

from junctionmapper.errors import ConfigError
from junctionmapper.geometry.boundary import trace_boundary
from junctionmapper.geometry.corners_tcj import neighbour_at, tcj_corners


def two_cells_sharing_a_vertical_edge():
    """Label 1 (left) and 2 (right), each 8 x 5 px, in a background margin."""
    labels = np.zeros((14, 14), dtype=int)
    labels[3:11, 2:7] = 1
    labels[3:11, 7:12] = 2
    return labels


def three_cells_meeting_at_a_point():
    """Cell 1 is enclosed: label 2 fills the top half of the surround, 3 the bottom.

    Cell 1 therefore meets 2 along its upper half and 3 along its lower half, with
    tricellular junctions at its left and right sides: exactly two identity changes.
    """
    labels = np.zeros((20, 20), dtype=int)
    labels[:10, :] = 2
    labels[10:, :] = 3
    labels[6:14, 6:14] = 1
    return labels


def single_isolated_cell():
    labels = np.zeros((14, 14), dtype=int)
    labels[3:10, 3:10] = 1
    return labels


def _ids(labels, label=1, **kw):
    b = trace_boundary(labels, label)
    return b, neighbour_at(labels, b, label, **kw)


def test_neighbour_ids_identify_the_adjacent_label():
    labels = two_cells_sharing_a_vertical_edge()
    b, ids = _ids(labels)
    assert set(ids) == {0, 2}
    assert len(ids) == len(b)


def test_corners_land_where_neighbour_identity_changes():
    labels = three_cells_meeting_at_a_point()
    b, ids = _ids(labels)
    assert set(ids) == {2, 3}
    assert len(tcj_corners(ids)) == 2


def test_isolated_cell_has_no_corners():
    b, ids = _ids(single_isolated_cell())
    assert set(ids) == {0}
    assert len(tcj_corners(ids)) == 0


def test_partly_free_partly_shared_boundary_has_corners_at_both_transitions():
    labels = two_cells_sharing_a_vertical_edge()
    b, ids = _ids(labels)
    corners = tcj_corners(ids)
    assert len(corners) == 2
    # one corner starts the run of 2s, the other starts the run of 0s
    assert sorted(ids[corners]) == [0, 2]


def test_corners_are_ascending_int_indices_into_the_boundary():
    b, ids = _ids(three_cells_meeting_at_a_point())
    c = tcj_corners(ids)
    assert np.issubdtype(c.dtype, np.integer)
    assert np.all(np.diff(c) > 0)
    assert c.min() >= 0 and c.max() < len(b)


def test_corner_is_first_position_of_the_new_neighbour():
    ids = np.array([0, 0, 2, 2, 2, 0])
    assert tcj_corners(ids).tolist() == [2, 5]


def test_change_between_last_and_first_registers_circularly():
    ids = np.array([2, 2, 2, 0, 0, 0])
    assert tcj_corners(ids).tolist() == [0, 3]


def test_corner_set_is_independent_of_trace_start():
    labels = three_cells_meeting_at_a_point()
    b, ids = _ids(labels)
    base = {tuple(b[i]) for i in tcj_corners(ids)}
    for shift in (1, 7, 13):
        b2, ids2 = np.roll(b, shift, axis=0), np.roll(ids, shift)
        assert {tuple(b2[i]) for i in tcj_corners(ids2)} == base


def test_degenerate_arrays_have_no_corners():
    assert len(tcj_corners(np.array([], dtype=int))) == 0
    assert len(tcj_corners(np.array([3]))) == 0
    assert len(tcj_corners(np.array([3, 3, 3]))) == 0


def test_circular_sequence_never_has_exactly_one_change():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ids = rng.integers(0, 3, size=rng.integers(2, 12))
        assert len(tcj_corners(ids)) != 1


# --- search_radius=2 tie-breaking -------------------------------------------------

def _tie_fixture(a, b_):
    """Cell 1 = rows 3..6, cols 4..6. Single foreign pixels at (2,4) and (2,6) are
    equally frequent and equidistant from boundary pixel (3,5)."""
    labels = np.zeros((10, 12), dtype=int)
    labels[3:7, 4:7] = 1
    labels[2, 4] = a
    labels[2, 6] = b_
    return labels


def _id_at(labels, rc):
    b, ids = _ids(labels)
    idx = [k for k, p in enumerate(map(tuple, b)) if p == rc]
    assert len(idx) == 1
    return ids[idx[0]]


def test_equal_count_equal_distance_resolves_to_lowest_label_not_position():
    assert _id_at(_tie_fixture(2, 3), (3, 5)) == 2
    assert _id_at(_tie_fixture(3, 2), (3, 5)) == 2   # swapped sides, same winner


def test_equal_count_resolves_to_the_nearer_label():
    labels = np.zeros((10, 12), dtype=int)
    labels[3:7, 4:7] = 1
    labels[2, 5] = 3      # distance 1
    labels[1, 4] = 2      # distance sqrt(5); lower label but farther
    assert _id_at(labels, (3, 5)) == 3


def test_more_frequent_label_beats_nearer_label():
    labels = np.zeros((10, 12), dtype=int)
    labels[3:7, 4:7] = 1
    labels[2, 5] = 3                      # one pixel, near
    labels[1, 4] = 2
    labels[1, 6] = 2                      # two pixels, farther
    assert _id_at(labels, (3, 5)) == 2


def test_search_radius_one_ignores_labels_two_pixels_away():
    labels = np.zeros((10, 12), dtype=int)
    labels[3:7, 4:7] = 1
    labels[1, 5] = 2
    b = trace_boundary(labels, 1)
    assert 2 in set(neighbour_at(labels, b, 1, search_radius=2))
    assert 2 not in set(neighbour_at(labels, b, 1, search_radius=1))


def test_labels_other_than_self_and_background_only():
    b, ids = _ids(two_cells_sharing_a_vertical_edge())
    assert 1 not in set(ids)


def test_image_border_counts_as_free_edge():
    labels = np.zeros((8, 8), dtype=int)
    labels[0:6, 0:6] = 1                  # touches top and left border
    b, ids = _ids(labels)
    assert set(ids) == {0}


@pytest.mark.parametrize("radius", [0, -1, 1.5, True])
def test_bad_search_radius_is_a_config_error(radius):
    labels = single_isolated_cell()
    b = trace_boundary(labels, 1)
    with pytest.raises(ConfigError):
        neighbour_at(labels, b, 1, search_radius=radius)
