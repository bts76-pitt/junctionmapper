import math

import numpy as np
import pandas as pd
import pytest

from junctionmapper.config import Role
from junctionmapper.errors import ConfigError
from junctionmapper.pipeline import measure_field, resolve_thresholds
from tests.fixtures13 import (
    PX, build, field_with_border_cell, field_with_isolated_cell, grid3, make_cfg, zoo,
)


@pytest.fixture
def setup():
    return build() + (make_cfg(),)


def _row(df, **kw):
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    assert len(sel) == 1, (kw, len(sel))
    return sel.iloc[0]


# ---- brief tests -------------------------------------------------------------------

def test_interface_rows_carry_role_marker_and_method(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    assert {"field_id", "cell_id", "neighbour_id", "corner_method", "role", "marker"} <= set(df.columns)
    assert set(df["marker"]) == {"Na/K-ATPase", "p120"}
    assert set(zip(df.role, df.marker)) == {("JM1", "Na/K-ATPase"), ("JM2", "p120")}


def test_both_corner_methods_appear_for_every_measured_cell(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    assert set(df["corner_method"]) == {"dp", "tcj"}
    assert set(df.cell_id) == {1, 2, 3, 4}
    for cid, g in df.groupby("cell_id"):
        assert set(g["corner_method"]) == {"dp", "tcj"}, cid


def test_every_metric_has_both_pixel_and_micron_columns(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    assert "interface_contour_px" in df and "interface_contour_um" in df
    for stem in ("interface_contour", "straight_line_length", "fragmented_contour",
                 "junction_contour", "junction_straight_line"):
        assert np.allclose(df[f"{stem}_um"].astype(float), df[f"{stem}_px"].astype(float) * PX)
    for stem in ("interface_area", "marker_area"):
        assert np.allclose(df[f"{stem}_um2"].astype(float),
                           df[f"{stem}_px"].astype(float) * PX ** 2)
    assert "distance_ratio" not in df
    assert (df.linearity_index.dropna() <= 1.0 + 1e-9).all()


def test_isolated_cell_appears_in_cells_with_zero_interfaces():
    f, seg = field_with_isolated_cell()
    res = measure_field(f, seg, make_cfg())
    c = _row(res.cells, cell_id=3)
    assert c.n_neighbours == 0 and c.n_interfaces_dp == 0 and c.n_interfaces_tcj == 0
    assert c.measured
    assert _row(res.flags, cell_id=3).isolated
    assert 3 not in set(res.interfaces["cell_id"])


def test_border_touching_cells_are_flagged_and_absent_from_interfaces():
    f, seg = field_with_border_cell()
    res = measure_field(f, seg, make_cfg())
    assert 1 in set(res.flags.loc[res.flags.border_touching, "cell_id"])
    assert set(res.flags.loc[res.flags.border_touching, "cell_id"]) == {1}
    assert 1 not in set(res.interfaces["cell_id"])
    row = _row(res.cells, cell_id=1)
    assert not row.measured and row.n_neighbours == 0 and row.area_px == 36


def test_keys_join_interfaces_to_cells_to_nuclei(setup):
    f, seg, cfg = setup
    res = measure_field(f, seg, cfg)
    assert not res.interfaces.merge(res.cells, on=["field_id", "cell_id"]).empty
    assert not res.cells.merge(res.nuclei, on=["field_id", "cell_id"]).empty
    assert len(res.flags) == len(res.cells) == 4
    assert res.cells.epsilon.nunique() == 1


# ---- item 3: per-role thresholds ---------------------------------------------------

def test_threshold_used_is_recorded_per_field_and_role(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    assert set(df[df.role == "JM1"].threshold) == {50.0}
    assert set(df[df.role == "JM2"].threshold) == {20.0}


def test_each_role_is_measured_at_its_own_threshold(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    j1 = _row(df, cell_id=4, neighbour_id=2, corner_method="dp", role="JM1")
    j2 = _row(df, cell_id=4, neighbour_id=2, corner_method="dp", role="JM2")
    # JM1 at T=50: 100-valued cols 18..21 only (the 40-valued decoy on 22..23 is excluded)
    assert j1.marker_area_px == 4 and j1.marker_intensity == 400.0
    # JM2 at T=20: 30-valued cols 18..25 (would be empty at T=50)
    assert j2.marker_area_px == 8 and j2.marker_intensity == 240.0


def test_resolve_thresholds_config_wins_otsu_and_flat_fallback(setup):
    f, seg, cfg = setup
    assert resolve_thresholds(f, cfg) == {Role.JM1: 50.0, Role.JM2: 20.0}
    cfg2 = make_cfg(threshold={Role.JM1: 7.0})
    t = resolve_thresholds(f, cfg2)
    assert t[Role.JM1] == 7.0
    assert 0.0 <= t[Role.JM2] < 30.0            # Otsu separates the 0 and 30 populations
    f.planes[Role.JM2] = np.full((40, 40), 3.0, np.float32)
    assert resolve_thresholds(f, cfg2)[Role.JM2] == 0.0
    f.planes[Role.JM2] = np.full((40, 40), np.nan, np.float32)
    assert resolve_thresholds(f, cfg2)[Role.JM2] == 0.0


# ---- hand-computed straight interface ----------------------------------------------

def test_hand_computed_values_dp_cell4_vs_cell2(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    j1 = _row(df, cell_id=4, neighbour_id=2, corner_method="dp", role="JM1")
    j2 = _row(df, cell_id=4, neighbour_id=2, corner_method="dp", role="JM2")
    # interface = row 18, cols 18..25 inclusive, walked from corner 0 (boundary idx 0) to idx 7
    assert (j1.start_idx, j1.end_idx) == (0, 7)
    assert j1.interface_contour_px == pytest.approx(7.0)
    assert j1.interface_contour_um == pytest.approx(3.5)
    assert j1.straight_line_length_px == pytest.approx(7.0)
    assert j1.linearity_index == pytest.approx(1.0)
    # band: 2 cycles of a 1x8 row -> rows 16..20 x cols 16..27 = 5*12 px
    assert j1.dilation_cycles == 2
    assert j1.interface_area_px == 60
    assert j1.interface_area_um2 == pytest.approx(60 * 0.25)
    assert j1.marker_area_um2 == pytest.approx(1.0)
    assert j1.fragmented_contour_px == pytest.approx(3.0)
    assert j1.junction_contour_um == pytest.approx(1.5)
    assert j1.coverage_index == pytest.approx(3.0 / 7.0)
    assert j1.interface_occupancy_pct == pytest.approx(100 * 4 / 60)
    assert j1.intensity_per_px2 == pytest.approx(400 / 60)
    assert j1.intensity_per_um2 == pytest.approx(400 / 60 / 0.25)
    assert j1.cluster_density_per_px2 == pytest.approx(100.0)
    assert j1.cluster_density_per_um2 == pytest.approx(400.0)
    assert j2.coverage_index == pytest.approx(1.0)
    assert j2.cluster_density_per_px2 == pytest.approx(30.0)


def test_pixel_size_scales_um_not_px(setup):
    f, seg, cfg = setup
    f.pixel_size_um = 2.0
    df = measure_field(f, seg, cfg).interfaces
    r = _row(df, cell_id=4, neighbour_id=2, corner_method="dp", role="JM1")
    assert r.interface_contour_px == pytest.approx(7.0)
    assert r.interface_contour_um == pytest.approx(14.0)
    assert r.interface_area_um2 == pytest.approx(60 * 4.0)
    assert r.marker_area_um2 == pytest.approx(4 * 4.0)
    assert r.intensity_per_um2 == pytest.approx(400 / 60 / 4.0)
    assert r.cluster_density_per_um2 == pytest.approx(100.0 / 4.0)


# ---- item 6: bent interface, differing JM2 -------------------------------------------

def test_bent_tcj_interface_exact_values():
    f, seg = zoo()
    df = measure_field(f, seg, make_cfg()).interfaces
    j1 = _row(df, cell_id=1, corner_method="tcj", role="JM1")
    j2 = _row(df, cell_id=1, corner_method="tcj", role="JM2")
    # walk (10,18),(10,19), col 19 down to (19,19), row 19 left to (19,10), up to (17,10)
    # = 22 pixels, 1+9+9+2 = 21 orthogonal steps; boundary has 36 pixels, idx 8 -> 29
    for r in (j1, j2):
        assert (r.neighbour_id, r.start_idx, r.end_idx) == (2, 8, 29)
        assert r.interface_contour_px == pytest.approx(21.0)
        assert r.straight_line_length_px == pytest.approx(math.sqrt(7 ** 2 + 8 ** 2))
        assert r.linearity_index == pytest.approx(math.sqrt(113) / 21)
        assert r.linearity_index < 0.51
        assert r.dilation_cycles == 2
    # JM1 above T=50 along the walk: (10..12,19) then (19,12),(19,11); decoy 40 at (14,19) excluded
    assert j1.fragmented_contour_px == pytest.approx(3.0)
    assert j1.junction_contour_px == pytest.approx(3 + 7 * math.sqrt(2))
    assert j1.junction_straight_line_px == pytest.approx(math.sqrt(9 ** 2 + 8 ** 2))
    assert j1.coverage_index == pytest.approx(3 / 21)
    assert j1.marker_area_px == 5 and j1.marker_intensity == 500.0
    # JM2 above T=20: (13..15,19) then (19,17..14), all 30
    assert j2.fragmented_contour_px == pytest.approx(5.0)
    assert j2.junction_contour_px == pytest.approx(5 + math.sqrt(20))
    assert j2.junction_straight_line_px == pytest.approx(math.sqrt(61))
    assert j2.coverage_index == pytest.approx(5 / 21)
    assert j2.marker_area_px == 7 and j2.marker_intensity == 210.0
    assert j2.junction_contour_um == pytest.approx((5 + math.sqrt(20)) * PX)


def test_bent_dp_interfaces_are_two_straight_legs_with_exact_indices():
    f, seg = zoo()
    df = measure_field(f, seg, make_cfg()).interfaces
    d = df[(df.cell_id == 1) & (df.corner_method == "dp")]
    assert set(zip(d.start_idx, d.end_idx)) == {(9, 18), (18, 27)}
    leg1 = _row(d, start_idx=9, role="JM1")        # (10,19) .. (19,19)
    leg2 = _row(d, start_idx=18, role="JM1")       # (19,19) .. (19,10)
    for leg in (leg1, leg2):
        assert leg.interface_contour_px == pytest.approx(9.0)
        assert leg.linearity_index == pytest.approx(1.0)
    assert leg1.fragmented_contour_px == pytest.approx(2.0)      # (10..12,19)
    assert leg2.fragmented_contour_px == pytest.approx(1.0)      # (19,12),(19,11)
    j2leg1 = _row(d, start_idx=9, role="JM2")
    j2leg2 = _row(d, start_idx=18, role="JM2")
    assert j2leg1.fragmented_contour_px == pytest.approx(2.0)    # (13..15,19)
    assert j2leg2.fragmented_contour_px == pytest.approx(3.0)    # (19,17..14)
    assert j2leg2.junction_straight_line_px == pytest.approx(3.0)


# ---- item 1/5: cells table ----------------------------------------------------------

@pytest.mark.parametrize("px", [0.5, 2.0])
def test_cells_values_at_two_pixel_sizes(px):
    f, seg = build(px=px)
    res = measure_field(f, seg, make_cfg())
    c4 = _row(res.cells, cell_id=4)
    assert c4.area_px == 64
    assert c4.area_um2 == pytest.approx(64 * px * px)
    assert c4.pixel_size_um == px
    assert c4.measured
    assert (c4.n_neighbours, c4.n_interfaces_dp, c4.n_interfaces_tcj) == (2, 2, 2)
    assert c4.n_nuclei == 1 and _row(res.cells, cell_id=1).n_nuclei == 1
    assert _row(res.cells, cell_id=2).n_nuclei == 0
    assert c4.dilation_cycles == 2
    assert (c4.epsilon, c4.tcj_min_run, c4.neighbour_radius_px) == (2.0, 3, 2)


def test_n_neighbours_counts_distinct_neighbours_not_interfaces():
    f, seg = grid3()
    cells = measure_field(f, seg, make_cfg()).cells
    centre = _row(cells, cell_id=5)
    # up, down, left, right; diagonals only touch at corners
    assert centre.n_neighbours == 4
    assert centre.n_interfaces_dp == 4 and centre.n_interfaces_tcj == 4
    assert _row(cells, cell_id=1).n_neighbours == 2
    assert "n_interfaces" not in cells


def test_zoo_cells_table():
    f, seg = zoo()
    cells = measure_field(f, seg, make_cfg()).cells
    exp = {  # cell: (area, measured, n_neighbours, dp, tcj, n_nuclei)
        1: (100, True, 1, 2, 1, 2), 2: (300, True, 1, 2, 1, 1), 3: (32, False, 0, 0, 0, 0),
        6: (36, False, 0, 0, 0, 0), 7: (36, True, 0, 0, 0, 0),
    }
    for cid, (area, meas, nn, dp, tcj, nn_nuc) in exp.items():
        c = _row(cells, cell_id=cid)
        assert (c.area_px, c.measured, c.n_neighbours, c.n_interfaces_dp, c.n_interfaces_tcj,
                c.n_nuclei) == (area, meas, nn, dp, tcj, nn_nuc), cid
        assert c.area_um2 == pytest.approx(area * 0.25)


def test_dilation_cycles_follow_config(setup):
    f, seg, _ = setup
    res = measure_field(f, seg, make_cfg(dilation_px=3))
    r = _row(res.interfaces, cell_id=4, neighbour_id=2, corner_method="dp", role="JM1")
    assert r.dilation_cycles == 3
    assert r.interface_area_px == 7 * 14           # rows 15..21 x cols 15..28
    assert (res.cells.dilation_cycles == 3).all()
    res = measure_field(f, seg, make_cfg(dilation_um=1.5))   # 1.5 um / 0.5 um/px
    assert (res.cells.dilation_cycles == 3).all()


# ---- item 4: flags --------------------------------------------------------------------

def test_zoo_flags_exact_values():
    f, seg = zoo()
    res = measure_field(f, seg, make_cfg())
    fl = res.flags.set_index("cell_id")
    cols = ["border_touching", "isolated", "degenerate", "corner_methods_disagree",
            "all_contacts_filtered", "multinucleate", "anucleate"]
    exp = {
        1: (False, False, False, True, False, True, False),    # dp 4 vs tcj 2 corners
        2: (False, False, False, True, False, False, False),   # L-shape: 6 dp corners vs 2
        3: (False, True, True, False, False, False, True),     # two blobs, untraceable
        4: (False, False, False, False, True, False, True),    # only contact is 2 px away, diagonal
        5: (False, False, False, False, True, False, True),
        6: (True, True, False, False, False, False, True),
        7: (False, True, False, False, False, False, True),
    }
    for cid, vals in exp.items():
        assert tuple(bool(fl.loc[cid, c]) for c in cols) == vals, cid
    corners = {1: (4, 2), 2: (6, 2), 4: (4, 0), 5: (2, 0), 6: (4, 0), 7: (4, 0)}
    for cid, pair in corners.items():
        assert (fl.loc[cid, "n_corners_dp"], fl.loc[cid, "n_corners_tcj"]) == pair, cid
    assert pd.isna(fl.loc[3, "n_corners_dp"]) and pd.isna(fl.loc[3, "n_corners_tcj"])


def test_degenerate_cell_is_in_cells_and_flags_but_not_interfaces():
    f, seg = zoo()
    res = measure_field(f, seg, make_cfg())
    assert 3 not in set(res.interfaces.cell_id)
    assert not _row(res.cells, cell_id=3).measured
    assert set(res.interfaces.cell_id) == {1, 2, 5}


def test_epsilon_reaches_the_geometry_and_is_recorded():
    f, seg = zoo()
    base = measure_field(f, seg, make_cfg())
    assert _row(base.flags, cell_id=4).n_corners_dp == 4
    big = measure_field(f, seg, make_cfg(epsilon_px=100.0))
    assert _row(big.flags, cell_id=4).n_corners_dp == 2     # only the two seeds survive
    assert (big.cells.epsilon == 100.0).all()
    # the extracted dp interfaces change too: cell 1 is split only at the two seeds, the
    # opposite corners (10,10) = idx 0 and (19,19) = idx 18, each leg 2 * 9 px long
    dp1 = big.interfaces[(big.interfaces.cell_id == 1) & (big.interfaces.corner_method == "dp")
                         & (big.interfaces.role == "JM1")]
    assert set(zip(dp1.start_idx, dp1.end_idx)) == {(0, 18), (18, 0)}
    assert list(dp1.interface_contour_px) == pytest.approx([18.0, 18.0])
    assert _row(big.cells, cell_id=2).n_interfaces_dp == 1
    assert _row(base.cells, cell_id=2).n_interfaces_dp == 2
    # the explicit argument wins over the config
    arg = measure_field(f, seg, make_cfg(epsilon_px=100.0), epsilon=2.0)
    assert _row(arg.flags, cell_id=4).n_corners_dp == 4
    assert (arg.cells.epsilon == 2.0).all()


def test_tcj_min_run_reaches_geometry_and_cells():
    f, seg = zoo()
    base = measure_field(f, seg, make_cfg())
    live = measure_field(f, seg, make_cfg(tcj_min_run=1))
    # default: cell 4's single-pixel contact with cell 5 is filtered out of tcj
    assert (_row(base.cells, cell_id=4).n_interfaces_tcj, _row(base.flags, cell_id=4).n_corners_tcj) == (0, 0)
    c4 = _row(live.cells, cell_id=4)
    assert (c4.n_interfaces_tcj, c4.n_neighbours, c4.tcj_min_run) == (1, 1, 1)
    f4 = _row(live.flags, cell_id=4)
    assert (f4.n_corners_tcj, f4.all_contacts_filtered) == (2, False)
    row = _row(live.interfaces, cell_id=4, corner_method="tcj", role="JM1")
    assert (row.neighbour_id, row.start_idx, row.end_idx) == (5, 10, 11)
    assert (base.cells.tcj_min_run == 3).all()


def test_neighbour_radius_reaches_geometry_and_cells():
    f, seg = zoo()
    r1 = measure_field(f, seg, make_cfg(neighbour_radius_px=1))
    assert _row(r1.flags, cell_id=4).isolated and _row(r1.flags, cell_id=5).isolated
    assert (r1.cells.neighbour_radius_px == 1).all()
    r3 = measure_field(f, seg, make_cfg(neighbour_radius_px=3))
    row = _row(r3.interfaces, cell_id=4, corner_method="tcj", role="JM1")
    assert (row.neighbour_id, row.start_idx, row.end_idx) == (5, 9, 12)
    assert (r3.cells.neighbour_radius_px == 3).all()


def test_short_contact_cells_n_neighbours():
    f, seg = zoo()
    cells = measure_field(f, seg, make_cfg()).cells
    # n_neighbours is the set of distinct neighbour ids over BOTH corner methods. Cell 4's one
    # contact (a single pixel, 2 px diagonal from cell 5) is dropped by tcj (shorter than
    # min_run) and loses the dp modal vote to background, so neither method lists cell 5: 0.
    # Cell 5 is only 2x2: one boundary pixel (41,43) sees cell 4 and one sees background in
    # its dp run, a tie that goes to the nonzero label, so dp lists cell 4, while tcj filters
    # the same 1-pixel contact and lists nothing: union = 1. The two
    # methods genuinely disagree about whether cell 5 has a neighbour (and the relation is not
    # reciprocal); the flags row says all_contacts_filtered=True. Expected, not a bug.
    assert _row(cells, cell_id=4).n_neighbours == 0
    c5 = _row(cells, cell_id=5)
    assert (c5.n_neighbours, c5.n_interfaces_dp, c5.n_interfaces_tcj) == (1, 1, 0)


# ---- item 2: nuclei ---------------------------------------------------------------------

D13 = math.sqrt(208)    # n1 (13,13) -> n3 (25,21)
D14 = math.sqrt(1258)   # n1 -> n4 (46,26)
D24 = math.sqrt(1189)   # n2 (13,16) -> n4
D34 = math.sqrt(466)    # n3 -> n4
D23 = 13.0              # n2 (13,16) -> n3 (25,21)


def test_nuclei_table_assignment_and_orphan():
    f, seg = zoo()
    n = measure_field(f, seg, make_cfg()).nuclei.set_index("nucleus_id")
    assert n.loc[1, "cell_id"] == 1 and n.loc[2, "cell_id"] == 1 and n.loc[3, "cell_id"] == 2
    assert pd.isna(n.loc[4, "cell_id"])
    assert list(n.orphan) == [False, False, False, True]
    assert (n.loc[1, "centroid_row"], n.loc[1, "centroid_col"]) == (13, 13)
    assert (n.loc[4, "centroid_row"], n.loc[4, "centroid_col"]) == (46, 26)
    assert n.loc[1, "area_px"] == 9 and n.loc[1, "area_um2"] == pytest.approx(2.25)


def test_nuclei_distances_default_k_gives_all_ranks_sorted():
    f, seg = zoo()
    d = measure_field(f, seg, make_cfg(num_intra_cells=5)).nuclei_distances
    n1 = d[d.nucleus_id == 1].sort_values("rank_from_nucleus")
    assert list(n1["rank_from_nucleus"]) == [1, 2, 3]
    assert list(n1.neighbour_nucleus_id) == [2, 3, 4]
    assert list(n1.distance_px) == pytest.approx([3.0, D13, D14])
    assert list(n1.distance_um) == pytest.approx([1.5, D13 / 2, D14 / 2])
    n4 = d[d.nucleus_id == 4].sort_values("rank_from_nucleus")
    assert list(n4.neighbour_nucleus_id) == [3, 2, 1]
    assert list(n4.distance_px) == pytest.approx([D34, D24, D14])


def test_num_intra_cells_limits_rank_count_and_summary_matches_table():
    f, seg = zoo()
    res = measure_field(f, seg, make_cfg(num_intra_cells=2))
    d = res.nuclei_distances
    assert len(d) == 8 and d["rank_from_nucleus"].max() == 2
    n1 = d[d.nucleus_id == 1].sort_values("rank_from_nucleus")
    assert list(n1.distance_px) == pytest.approx([3.0, D13])
    n = res.nuclei.set_index("nucleus_id")
    for nid in (1, 2, 3, 4):
        rows = d[d.nucleus_id == nid]
        assert n.loc[nid, "n_intra_distances"] == len(rows) == 2
        assert n.loc[nid, "mean_intra_distance_px"] == pytest.approx(rows.distance_px.mean())
        assert n.loc[nid, "mean_intra_distance_um"] == pytest.approx(rows.distance_um.mean())
    assert n.loc[1, "mean_intra_distance_px"] == pytest.approx((3.0 + D13) / 2)


def test_all_pairs_emits_every_ordered_pair_so_each_distance_appears_from_both_ends():
    f, seg = zoo()
    res = measure_field(f, seg, make_cfg(all_pairs=True))
    d = res.nuclei_distances
    assert len(d) == 12                                   # 4 nuclei, 4 * 3 ordered pairs
    assert set(zip(d.nucleus_id, d.neighbour_nucleus_id)) == {
        (i, j) for i in (1, 2, 3, 4) for j in (1, 2, 3, 4) if i != j}
    # one copy of each distance is the lower-id half of the table
    half = d[d.nucleus_id < d.neighbour_nucleus_id]
    assert len(half) == 6
    n1 = d[d.nucleus_id == 1].sort_values("rank_from_nucleus")
    assert list(n1.distance_px) == pytest.approx([3.0, D13, D14])     # ranked nearest-first


def test_rank_is_each_nucleus_own_nearest_neighbour_order_not_lower_id_rows_only():
    """C1: with all_pairs, ``rank_from_nucleus == 1`` must be the nearest neighbour OF EVERY
    nucleus, the highest id included. Ranking over only the rows a nucleus held as the lower
    id made rank 1 the nearest *higher-id* neighbour and left the highest id out of the table
    altogether."""
    f, seg = zoo()
    d = measure_field(f, seg, make_cfg(all_pairs=True)).nuclei_distances
    truth = {1: (2, 3.0), 2: (1, 3.0), 3: (2, D23), 4: (3, D34)}
    assert set(d.nucleus_id) == set(truth)                # nucleus 4 is present at all
    for nid, (nearest, dist) in truth.items():
        rows = d[(d.nucleus_id == nid) & (d.rank_from_nucleus == 1)]
        assert len(rows) == 1, nid
        assert int(rows.iloc[0].neighbour_nucleus_id) == nearest, nid
        assert rows.iloc[0].distance_px == pytest.approx(dist), nid
        # and the whole ordering is that nucleus's own, nearest first
        own = d[d.nucleus_id == nid].sort_values("rank_from_nucleus")
        assert list(own.rank_from_nucleus) == [1, 2, 3]
        assert list(own.distance_px) == pytest.approx(sorted(own.distance_px))


def test_all_pairs_summary_covers_every_distance_of_every_nucleus():
    f, seg = zoo()
    res = measure_field(f, seg, make_cfg(all_pairs=True))
    n = res.nuclei.set_index("nucleus_id")
    assert [n.loc[i, "n_intra_distances"] for i in (1, 2, 3, 4)] == [3, 3, 3, 3]
    assert not n.mean_intra_distance_px.isna().any()
    # nucleus 4 (highest id, stored only as the second element of its pairs):
    assert n.loc[4, "mean_intra_distance_px"] == pytest.approx((D14 + D24 + D34) / 3)
    assert n.loc[4, "mean_intra_distance_um"] == pytest.approx((D14 + D24 + D34) / 6)
    assert n.loc[1, "mean_intra_distance_px"] == pytest.approx((3.0 + D13 + D14) / 3)
    # a plain groupby of the long table reproduces the summary
    d = res.nuclei_distances
    for nid in (1, 2, 3, 4):
        rows = d[d.nucleus_id == nid]
        assert len(rows) == n.loc[nid, "n_intra_distances"]
        assert rows.distance_px.mean() == pytest.approx(n.loc[nid, "mean_intra_distance_px"])


def test_no_nucleus_gives_empty_nuclei_and_anucleate_flags(setup):
    f, seg, cfg = setup
    seg.nuclei[:] = 0
    res = measure_field(f, seg, cfg)
    assert res.nuclei.empty and "nucleus_id" in res.nuclei.columns
    assert res.nuclei_distances.empty
    assert "rank_from_nucleus" in res.nuclei_distances.columns
    assert res.flags.anucleate.all() and not res.flags.multinucleate.any()


# ---- misc -------------------------------------------------------------------------------

def test_jm2_is_optional(setup):
    f, seg, cfg = setup
    del f.planes[Role.JM2]
    df = measure_field(f, seg, cfg).interfaces
    assert set(df.role) == {"JM1"}


def test_shape_mismatch_raises_config_error(setup):
    f, seg, cfg = setup
    f.planes[Role.JM1] = np.zeros((5, 5), np.float32)
    with pytest.raises(ConfigError):
        measure_field(f, seg, cfg)


def test_nucleus_shape_mismatch_raises_config_error(setup):
    f, seg, cfg = setup
    seg.nuclei = np.zeros((5, 5), np.int32)
    with pytest.raises(ConfigError):
        measure_field(f, seg, cfg)
