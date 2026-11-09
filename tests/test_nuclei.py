import numpy as np
import pytest

from junctionmapper.errors import ConfigError
from junctionmapper.measure.nuclei import associate, intra_distances, nucleus_centroids

SHAPE = (30, 30)


def blank():
    return np.zeros(SHAPE, dtype=np.int32)


def rect(arr, label, rows, cols):
    """Fill inclusive (row0, row1) x (col0, col1) with label."""
    arr[rows[0] : rows[1] + 1, cols[0] : cols[1] + 1] = label
    return arr


def rect_nucleus(rows, cols):
    return rect(blank(), 1, rows, cols)


def two_cells():
    a = blank()
    rect(a, 1, (0, 29), (0, 14))
    rect(a, 2, (0, 29), (15, 29))
    return a


def one_cell():
    return rect(blank(), 1, (0, 29), (0, 29))


def one_nucleus_in_each():
    a = blank()
    rect(a, 1, (5, 8), (5, 8))
    rect(a, 2, (5, 8), (20, 23))
    return a


def one_nucleus_in_each_but_one():
    return rect(blank(), 1, (5, 8), (5, 8))


def two_nuclei_inside():
    a = blank()
    rect(a, 1, (2, 5), (2, 5))
    rect(a, 2, (15, 18), (15, 18))
    return a


def nucleus_in_background():
    cells = rect(blank(), 1, (0, 9), (0, 9))
    nuc = rect(blank(), 1, (20, 25), (20, 25))
    return cells, nuc


def six_centroids():
    return {
        1: (0, 0), 2: (0, 3), 3: (4, 0), 4: (10, 10), 5: (11, 10), 6: (20, 1),
    }


def test_centroid_matches_the_java_midpoint_formula():
    # (max(x)-min(x))//2 + min(x), per the user guide
    assert nucleus_centroids(rect_nucleus(rows=(2, 8), cols=(4, 10)))[1] == (5, 7)


def test_centroid_uses_floored_midpoint_not_centre_of_mass():
    a = blank()
    a[2:6, 3] = 1  # rows 2..5 -> (5-2)//2+2 = 3
    a[2, 3:10] = 1  # cols 3..9 -> 3 + 3 = 6
    assert nucleus_centroids(a)[1] == (3, 6)  # centre of mass would differ


def test_centroids_skip_background_and_cover_every_label():
    a = rect(blank(), 4, (0, 2), (0, 2))
    rect(a, 9, (10, 12), (10, 14))
    assert nucleus_centroids(a) == {4: (1, 1), 9: (11, 12)}


def test_each_nucleus_maps_to_its_containing_cell():
    cells, flags = associate(two_cells(), one_nucleus_in_each())
    assert cells == {1: [1], 2: [2]}


def test_two_nuclei_in_one_cell_flags_multinucleate():
    _, flags = associate(one_cell(), two_nuclei_inside())
    assert flags.multinucleate == [1]


def test_nucleus_outside_any_cell_flags_orphan():
    c, n = nucleus_in_background()
    _, flags = associate(c, n)
    assert flags.orphan_nuclei == [1]


def test_cell_without_a_nucleus_is_flagged():
    _, flags = associate(two_cells(), one_nucleus_in_each_but_one())
    assert flags.anucleate_cells == [2]


def test_nucleus_mostly_over_background_is_still_assigned_to_its_best_cell():
    cells = rect(blank(), 1, (0, 9), (0, 9))
    mostly_out = rect(blank(), 1, (8, 15), (0, 5))  # 12 px in cell, 36 out
    m, flags = associate(cells, mostly_out)
    assert m == {1: [1]} and flags.orphan_nuclei == []


def test_six_six_eight_nucleus_goes_to_cell_a_not_orphan():
    cells = blank()
    rect(cells, 1, (0, 9), (0, 9))    # cell A
    rect(cells, 2, (0, 9), (10, 19))  # cell B
    nuc = rect(blank(), 1, (7, 11), (8, 11))  # 6 px in A, 6 in B, 8 background
    m, flags = associate(cells, nuc)
    assert m == {1: [1]}  # tie between cells -> lowest label
    assert flags.orphan_nuclei == []
    assert flags.anucleate_cells == [2]


def test_single_pixel_over_a_cell_is_enough_to_be_assigned():
    cells = rect(blank(), 1, (0, 4), (0, 4))
    nuc = rect(blank(), 1, (4, 12), (4, 12))  # 1 px in cell, 80 over background
    m, flags = associate(cells, nuc)
    assert m == {1: [1]} and flags.orphan_nuclei == []


def test_nucleus_entirely_over_background_is_orphan_even_with_cells_present():
    cells = rect(blank(), 1, (0, 4), (0, 4))
    nuc = rect(blank(), 1, (10, 14), (10, 14))
    m, flags = associate(cells, nuc)
    assert m == {} and flags.orphan_nuclei == [1] and flags.anucleate_cells == [1]


def test_tie_between_cells_goes_to_lowest_label_regardless_of_array_order():
    cells = blank()
    rect(cells, 7, (0, 9), (0, 4))
    rect(cells, 3, (0, 9), (5, 9))
    nuc = rect(blank(), 1, (2, 3), (3, 6))  # 2 px in 7, 2 px in 3
    m, _ = associate(cells, nuc)
    assert m == {3: [1]}
    m2, _ = associate(cells[:, ::-1].copy(), nuc[:, ::-1].copy())
    assert m2 == {3: [1]}


def test_flag_lists_are_sorted_and_anucleate_covers_all_cell_labels():
    cells = blank()
    for lab, c0 in ((5, 0), (2, 6), (9, 12), (4, 18)):
        rect(cells, lab, (0, 5), (c0, c0 + 5))
    nuc = blank()
    rect(nuc, 8, (1, 2), (1, 2))
    rect(nuc, 3, (1, 2), (4, 4))
    rect(nuc, 6, (1, 2), (19, 20))
    rect(nuc, 1, (20, 25), (20, 25))
    nuc[3, 8] = 2
    m, flags = associate(cells, nuc)
    assert m == {5: [3, 8], 4: [6], 2: [2]}
    assert flags.multinucleate == [5]
    assert flags.orphan_nuclei == [1]
    assert flags.anucleate_cells == [9]


def test_associate_rejects_mismatched_shapes():
    with pytest.raises(ConfigError):
        associate(np.zeros((4, 4), int), np.zeros((5, 5), int))


def test_k_nearest_returns_k_distances_per_nucleus_sorted_ascending():
    d = intra_distances(six_centroids(), k=3)
    assert len(d) == 6 * 3
    assert d[0][2] <= d[1][2] <= d[2][2]


def test_distances_exclude_self_and_are_ordered_by_nucleus_then_distance():
    d = intra_distances(six_centroids(), k=3)
    assert all(i != j for i, j, _ in d)
    assert [t[0] for t in d] == sorted(t[0] for t in d)
    for i in range(1, 7):
        ds = [x[2] for x in d if x[0] == i]
        assert ds == sorted(ds)
    assert d[0] == (1, 2, 3.0)


def test_coincident_centroids_of_distinct_nuclei_give_zero_distance():
    d = intra_distances({1: (2, 2), 2: (2, 2), 3: (5, 6)}, k=1)
    assert (1, 2, 0.0) in d


def test_k_larger_than_available_is_clamped_with_no_inf():
    d = intra_distances(six_centroids(), k=50)
    assert len(d) == 6 * 5
    assert all(np.isfinite(x[2]) for x in d)


def test_fewer_than_two_nuclei_gives_empty():
    assert intra_distances({}, k=3) == []
    assert intra_distances({1: (0, 0)}, k=3) == []


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.0])
def test_bad_k_raises(bad):
    with pytest.raises(ConfigError):
        intra_distances(six_centroids(), k=bad)


def test_centroids_are_python_ints():
    r, c = nucleus_centroids(rect_nucleus((2, 8), (4, 10)))[1]
    assert type(r) is int and type(c) is int


def test_all_pairs_ignores_k_and_emits_each_pair_once():
    d = intra_distances(six_centroids(), k=1, all_pairs=True)
    assert len(d) == 15
    assert all(i < j for i, j, _ in d)
    assert len({(i, j) for i, j, _ in d}) == 15
    assert d == sorted(d, key=lambda t: (t[0], t[1]))
