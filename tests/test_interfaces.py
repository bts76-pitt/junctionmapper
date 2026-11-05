import numpy as np
import pytest

from junctionmapper.errors import BoundaryError, ConfigError
from junctionmapper.geometry import interfaces as interfaces_mod
from junctionmapper.geometry.boundary import trace_boundary
from junctionmapper.geometry.corners_dp import dp_corners
from junctionmapper.geometry.corners_tcj import neighbour_at, tcj_corners
from junctionmapper.geometry.interfaces import Interface, extract_interfaces
from junctionmapper.measure.interface import interface_contour
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


def _positions(i, n):
    return (i.start_idx + np.arange(len(i.pixels))) % n


def _coverage(ifs, n):
    counts = np.zeros(n, int)
    for i in ifs:
        np.add.at(counts, _positions(i, n), 1)
    return counts


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
    assert all((labels[max(0, y - 2):y + 3, max(0, x - 2):x + 3] == 2).any() for y, x in px[:-1])   # the end corner is the next run's first pixel


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
    # inclusive of both corners: the wrap run ends ON the first corner (lower index)
    assert tuple(w.pixels[-1]) == tuple(b[w.end_idx])
    # no free edge here: every position covered, corners exactly twice
    counts = _coverage(ifs, len(b))
    assert counts.min() == 1 and counts.max() == 2
    assert counts.sum() == sum(len(i.pixels) for i in ifs)


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
    assert len(i.pixels) == len(b) + 1          # start pixel repeated to close the loop
    assert tuple(i.pixels[0]) == tuple(i.pixels[-1]) == tuple(b[7])
    assert interface_contour(i.pixels) == pytest.approx(interface_contour(np.vstack([b, b[:1]])))
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
    assert len(i.pixels) == len(b) + 1          # closed contour = true perimeter
    assert (i.pixels[:-1] == b).all() and (i.pixels[-1] == b[0]).all()
    assert interface_contour(i.pixels) == pytest.approx(20.0)   # 6x6 block: 5+5+5+5 steps
    assert _contiguous(i.pixels)


def test_enclosed_cell_under_dp_is_split_geometrically_but_loses_nothing():
    labels = enclosed_cell()
    b = trace_boundary(labels, 1)
    ifs = extract_interfaces(labels, 1, method="dp", epsilon=1.5)
    assert {i.neighbour_id for i in ifs} == {2}
    counts = _coverage(ifs, len(b))
    assert counts.min() >= 1 and counts.max() <= 2
    # Each interface's contour is the walk between its own two corners, so the four
    # of them sum to the perimeter: nothing is lost by DP splitting the loop.
    assert sum(interface_contour(i.pixels) for i in ifs) == pytest.approx(20.0)
    assert {len(i.pixels) for i in ifs} == {6}  # 4 arcs of 5 steps, corner-inclusive


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


def test_label_zero_is_rejected_as_background():
    labels = np.zeros((10, 10), int)
    labels[2:8, 2:8] = 1
    with pytest.raises(ConfigError, match="background"):
        assess_cell(labels, 0, epsilon=1.5)


def test_engulfed_cell_does_not_report_corner_methods_disagree():
    """Zero TCJ corners means the neighbour identity never changes, so the two methods
    are not measuring the same thing and the comparison carries no signal. Previously a
    cell wholly enclosed by one neighbour raised the flag (DP 4 vs TCJ 0) even though
    its single whole-boundary interface is extracted correctly."""
    labels = np.zeros((30, 30), int)
    labels[3:27, 3:27] = 2
    labels[10:20, 10:20] = 1
    flags = assess_cell(labels, 1, epsilon=1.5)
    assert flags.n_corners_tcj == 0 and flags.n_corners_dp == 4
    assert flags.corner_methods_disagree is False
    assert flags.isolated is False
    assert flags.all_contacts_filtered is False  # the contact is the whole boundary


def test_a_cell_whose_only_contact_is_filtered_away_is_flagged():
    """The cell has a neighbour, but its single-pixel contact is below the TCJ min_run,
    so TCJ yields no interface. Without the flag the cell contributes nothing to the
    results with nothing in the output to say why."""
    labels = np.zeros((20, 20), int)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 2
    labels[9, 9] = 2  # lone pixel reaching toward cell 1
    flags = assess_cell(labels, 1, epsilon=1.5)
    assert flags.isolated is False
    assert flags.n_corners_tcj == 0
    assert flags.all_contacts_filtered is True
    assert extract_interfaces(labels, 1, method="tcj") == []


def test_a_cell_with_a_real_shared_edge_is_not_flagged_as_filtered():
    labels = two_cells_sharing_a_vertical_edge()
    assert assess_cell(labels, 1, epsilon=1.5).all_contacts_filtered is False


def test_shared_edge_has_exact_length_and_contour_including_both_corners():
    # Cell 1 = rows 3..10, cols 2..6; cell 2 = cols 7..11. Within radius 2 of cell 2,
    # cell 1 pixels are cols 5,6: (3,5),(3,6), (4..9,6), (10,6),(10,5) = 10 positions.
    # Corners are the first neighbour-2 pixel (3,5) and the first background pixel
    # (10,4); the inclusive walk is 11 pixels, 10 unit steps.
    labels = two_cells_sharing_a_vertical_edge()
    ifs = extract_interfaces(labels, 1, method="tcj")
    assert len(ifs) == 1
    px = ifs[0].pixels
    assert len(px) == 11
    assert tuple(px[0]) == (3, 5) and tuple(px[-1]) == (10, 4)
    assert interface_contour(px) == 10.0


def test_kept_interface_keeps_its_end_corner_when_next_run_is_free_edge():
    labels = two_cells_sharing_a_vertical_edge()
    b = trace_boundary(labels, 1)
    for method, eps in (("tcj", None), ("dp", 1.5)):
        for i in extract_interfaces(labels, 1, method=method, epsilon=eps):
            assert tuple(i.pixels[-1]) == tuple(b[i.end_idx % len(b)])
            assert tuple(i.pixels[0]) == tuple(b[i.start_idx])


def test_adjacent_corners_give_two_pixel_interface_with_nonzero_contour(monkeypatch):
    # Regression: end-exclusive slicing gave a 1-pixel interface with contour 0.0.
    labels = enclosed_cell()
    b = trace_boundary(labels, 1)
    monkeypatch.setattr(interfaces_mod, "dp_corners", lambda bb, e: np.array([0, 1, 10]))
    ifs = extract_interfaces(labels, 1, method="dp", epsilon=1.5)
    short = [i for i in ifs if i.start_idx == 0]
    assert len(short) == 1
    assert len(short[0].pixels) == 2
    step = np.hypot(*(b[1] - b[0]))
    assert interface_contour(short[0].pixels) == pytest.approx(step)
    assert interface_contour(short[0].pixels) != 0.0


def voronoi_field(seed, size=120, n_seeds=40):
    rng = np.random.default_rng(seed)
    pts = rng.integers(0, size, size=(n_seeds, 2))
    yy, xx = np.mgrid[0:size, 0:size]
    d = (yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2
    return d.argmin(axis=-1) + 1


@pytest.mark.parametrize("method", ["tcj", "dp"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_voronoi_field_corners_covered_exactly_twice_everything_else_once(method, seed):
    labels = voronoi_field(seed)
    checked = 0
    for lab in np.unique(labels):
        m = labels == lab
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            continue
        try:
            b = trace_boundary(labels, lab)
        except BoundaryError:
            continue
        corners = (tcj_corners(neighbour_at(labels, b, lab)) if method == "tcj"
                   else dp_corners(b, 1.5))
        if len(corners) < 2:
            continue
        ifs = extract_interfaces(labels, lab, method=method, epsilon=1.5)
        assert len(ifs) == len(corners)       # no background in a full tiling
        for i in ifs:
            assert (i.pixels == b[_positions(i, len(b))]).all()
            assert _contiguous(i.pixels)
        counts = _coverage(ifs, len(b))
        assert counts.min() >= 1 and counts.max() <= 2
        assert set(np.flatnonzero(counts == 2)) == set(int(c) for c in corners)
        checked += 1
    assert checked >= 10
