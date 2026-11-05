import numpy as np
import pytest

from junctionmapper.errors import ConfigError
from junctionmapper.geometry import interfaces as interfaces_mod
from junctionmapper.geometry.boundary import trace_boundary
from junctionmapper.geometry.interfaces import Interface, extract_interfaces
from junctionmapper.qc import CellFlags, assess_cell


def two_cells_sharing_a_vertical_edge():
    labels = np.zeros((14, 14), dtype=int)
    labels[3:11, 2:7] = 1
    labels[3:11, 7:12] = 2
    return labels


def single_isolated_cell():
    labels = np.zeros((14, 14), dtype=int)
    labels[3:10, 3:10] = 1
    return labels


def three_cells_meeting_at_a_point():
    """Cell 1 enclosed: label 2 above, 3 below -> 2 TCJ corners, 4 DP corners."""
    labels = np.zeros((20, 20), dtype=int)
    labels[:10, :] = 2
    labels[10:, :] = 3
    labels[6:14, 6:14] = 1
    return labels


def wrapping_cell():
    """Cell 1 meets 3 on its right and 2 on top/left/bottom, so the run of neighbour 2
    starts on the bottom edge and wraps through boundary index 0 (top-left)."""
    labels = np.full((14, 14), 2, dtype=int)
    labels[:, 11:] = 3
    labels[3:11, 3:11] = 1
    return labels


def _contiguous(px):
    return np.all(np.abs(np.diff(px, axis=0)).max(axis=1) <= 1)


def test_two_adjacent_cells_yield_one_shared_interface():
    labels = two_cells_sharing_a_vertical_edge()
    ifs = extract_interfaces(labels, 1, method="tcj")
    assert [i.neighbour_id for i in ifs] == [2]


def test_isolated_cell_yields_no_interfaces_and_is_flagged():
    labels = single_isolated_cell()
    assert extract_interfaces(labels, 1, method="tcj") == []
    assert assess_cell(labels, 1, epsilon=1.5).isolated is True


def test_border_touching_cell_is_flagged():
    labels = np.zeros((10, 10), int)
    labels[0:5, 0:5] = 1
    assert assess_cell(labels, 1, epsilon=1.5).border_touching is True


def test_corner_method_disagreement_is_flagged():
    labels = three_cells_meeting_at_a_point()
    f = assess_cell(labels, 1, epsilon=1.5)
    assert f.n_corners_tcj == 2
    assert f.n_corners_dp >= 4
    assert f.corner_methods_disagree is True


def test_interface_pixels_are_contiguous_and_shared_by_both_cells():
    labels = two_cells_sharing_a_vertical_edge()
    px = extract_interfaces(labels, 1, method="tcj")[0].pixels
    assert _contiguous(px)
    # every pixel belongs to cell 1 and is within the neighbour search radius (2) of cell 2
    assert all(labels[y, x] == 1 for y, x in px)
    assert all((labels[max(0, y - 2):y + 3, max(0, x - 2):x + 3] == 2).any() for y, x in px)


def test_wrap_around_interface_is_contiguous_and_start_after_end():
    labels = wrapping_cell()
    b = trace_boundary(labels, 1)
    ifs = extract_interfaces(labels, 1, method="tcj")
    wrap = [i for i in ifs if i.neighbour_id == 2]
    assert len(wrap) == 1
    w = wrap[0]
    assert w.start_idx > w.end_idx
    assert _contiguous(w.pixels)
    assert tuple(w.pixels[0]) == tuple(b[w.start_idx])
    assert (w.pixels == b[0]).all(axis=1).any()            # passes through index 0
    # no free edge here, so the interfaces partition the boundary exactly
    assert sum(len(i.pixels) for i in ifs) == len(b)


def test_interface_fields_and_method_recorded():
    labels = two_cells_sharing_a_vertical_edge()
    i = extract_interfaces(labels, 1, method="tcj")[0]
    assert isinstance(i, Interface)
    assert (i.cell_id, i.method) == (1, "tcj")


def test_free_edge_is_excluded_and_does_not_drop_interface_pixels():
    labels = two_cells_sharing_a_vertical_edge()
    b = trace_boundary(labels, 1)
    ids = interfaces_mod.neighbour_at(labels, b, 1)
    for method, eps in (("tcj", None), ("dp", 1.5)):
        ifs = extract_interfaces(labels, 1, method=method, epsilon=eps)
        assert all(i.neighbour_id not in (None, 0) for i in ifs)
        for i in ifs:
            assert _contiguous(i.pixels)
        # every position whose neighbour is 2 and not near a corner is retained
        got = {tuple(p) for i in ifs for p in i.pixels}
        core = {tuple(b[k]) for k in range(len(b)) if ids[k] == 2}
        assert len(core & got) >= len(core) - 4


def test_dp_interfaces_get_modal_neighbour():
    labels = three_cells_meeting_at_a_point()
    ifs = extract_interfaces(labels, 1, method="dp", epsilon=1.5)
    assert {i.neighbour_id for i in ifs} == {2, 3}
    assert all(i.method == "dp" for i in ifs)
    for i in ifs:
        assert _contiguous(i.pixels)


def test_dp_neighbour_ties_break_to_smallest_label_regardless_of_trace_start():
    labels = np.zeros((12, 12), int)
    labels[3:9, 3:9] = 1
    labels[:3, 3:9] = 5          # above
    labels[9:, 3:9] = 4          # below
    found = set()
    for arr in (labels, labels.T.copy(), labels[::-1].copy(), labels[:, ::-1].copy()):
        # force a single dp interface spanning the whole loop: one corner, tied 4 vs 5
        orig = interfaces_mod.dp_corners
        interfaces_mod.dp_corners = lambda b, e: np.array([0])
        try:
            ifs = extract_interfaces(arr, 1, method="dp", epsilon=1.5)
        finally:
            interfaces_mod.dp_corners = orig
        assert len(ifs) == 1
        found.add(ifs[0].neighbour_id)
    assert len(found) == 1


def test_single_corner_gives_one_closed_loop_interface(monkeypatch):
    labels = three_cells_meeting_at_a_point()
    monkeypatch.setattr(interfaces_mod, "dp_corners", lambda b, e: np.array([7]))
    b = trace_boundary(labels, 1)
    ifs = extract_interfaces(labels, 1, method="dp", epsilon=1.5)
    assert len(ifs) == 1
    i = ifs[0]
    assert i.start_idx == i.end_idx == 7
    assert len(i.pixels) == len(b)
    assert tuple(i.pixels[0]) == tuple(b[7])
    assert _contiguous(i.pixels)


def test_zero_dp_corners_on_isolated_cell_gives_no_interfaces(monkeypatch):
    monkeypatch.setattr(interfaces_mod, "dp_corners", lambda b, e: np.zeros(0, int))
    assert extract_interfaces(single_isolated_cell(), 1, "dp", 1.5) == []


def enclosed_cell():
    """Cell 1 (6x6) fully enclosed by a 4 px thick ring of label 2."""
    labels = np.zeros((20, 20), dtype=int)
    labels[3:17, 3:17] = 2
    labels[7:13, 7:13] = 1
    return labels


def test_enclosed_cell_has_one_whole_boundary_interface():
    labels = enclosed_cell()
    b = trace_boundary(labels, 1)
    ifs = extract_interfaces(labels, 1, method="tcj")
    assert len(ifs) == 1
    i = ifs[0]
    assert i.neighbour_id == 2
    assert i.start_idx == i.end_idx == 0
    assert len(i.pixels) == len(b)
    assert (i.pixels == b).all()
    assert _contiguous(i.pixels)


def test_enclosed_cell_under_dp_is_split_geometrically_but_loses_nothing():
    labels = enclosed_cell()
    b = trace_boundary(labels, 1)
    ifs = extract_interfaces(labels, 1, method="dp", epsilon=1.5)
    assert {i.neighbour_id for i in ifs} == {2}
    assert sum(len(i.pixels) for i in ifs) == len(b)


def test_zero_dp_corners_on_enclosed_cell_gives_whole_loop(monkeypatch):
    monkeypatch.setattr(interfaces_mod, "dp_corners", lambda b, e: np.zeros(0, int))
    ifs = extract_interfaces(enclosed_cell(), 1, "dp", 1.5)
    assert len(ifs) == 1 and ifs[0].neighbour_id == 2
    assert ifs[0].start_idx == ifs[0].end_idx == 0


def test_isolated_cell_does_not_flag_corner_disagreement():
    f = assess_cell(single_isolated_cell(), 1, epsilon=1.5)
    assert f.isolated is True
    assert f.n_corners_tcj == 0 and f.n_corners_dp >= 4
    assert f.corner_methods_disagree is False


def test_dp_requires_epsilon():
    with pytest.raises(ConfigError):
        extract_interfaces(two_cells_sharing_a_vertical_edge(), 1, method="dp")


def test_tcj_ignores_epsilon():
    labels = two_cells_sharing_a_vertical_edge()
    a = extract_interfaces(labels, 1, method="tcj")
    b = extract_interfaces(labels, 1, method="tcj", epsilon=1.5)
    assert [(i.start_idx, i.end_idx) for i in a] == [(i.start_idx, i.end_idx) for i in b]


def test_unknown_method_rejected():
    with pytest.raises(ConfigError):
        extract_interfaces(two_cells_sharing_a_vertical_edge(), 1, method="nope")


def test_assess_cell_flags_for_interior_shared_cell():
    f = assess_cell(two_cells_sharing_a_vertical_edge(), 1, epsilon=1.5)
    assert isinstance(f, CellFlags)
    assert (f.border_touching, f.isolated, f.degenerate) == (False, False, False)


def test_assess_cell_degenerate_tiny_cell_does_not_raise():
    labels = np.zeros((10, 10), int)
    labels[4, 4] = 1
    f = assess_cell(labels, 1, epsilon=1.5)
    assert f.degenerate is True
    assert f.n_corners_dp is None and f.n_corners_tcj is None
    assert f.corner_methods_disagree is False


def test_assess_cell_dp_failure_marks_degenerate(monkeypatch):
    def boom(b, e):
        raise ConfigError("tiny")
    monkeypatch.setattr("junctionmapper.qc.dp_corners", boom)
    f = assess_cell(two_cells_sharing_a_vertical_edge(), 1, epsilon=1.5)
    assert f.degenerate is True and f.n_corners_dp is None
    assert f.n_corners_tcj is not None and f.corner_methods_disagree is False


def test_assess_cell_bad_epsilon_still_raises():
    with pytest.raises(ConfigError):
        assess_cell(two_cells_sharing_a_vertical_edge(), 1, epsilon=0)
