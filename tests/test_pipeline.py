from pathlib import Path

import numpy as np
import pytest

from junctionmapper.config import ChannelSpec, Role, RunConfig
from junctionmapper.errors import ConfigError
from junctionmapper.io.reader import Field
from junctionmapper.pipeline import measure_field, resolve_thresholds
from junctionmapper.segment.cellpose import Segmentation

PX = 0.5


def make_cfg(**kw):
    return RunConfig(
        channels=[ChannelSpec(role=Role.JM1, channel=0, marker="Na/K-ATPase"),
                  ChannelSpec(role=Role.JM2, channel=1, marker="p120"),
                  ChannelSpec(role=Role.NUCLEUS, channel=2, marker="DAPI")],
        pixel_size_um=PX, threshold={Role.JM1: 50.0, Role.JM2: 50.0}, **kw)


def build(field_id="f1"):
    """2x2 block of 8x8 squares: 1=(10:18,10:18) 2=(10:18,18:26) 3=(18:26,10:18) 4=(18:26,18:26).

    JM1 is 100 on row 18, cols 18..21 (four pixels of the dp interface of cell 4 vs 2), else 0.
    JM2 is 100 on row 18, cols 18..25 (the whole interface).
    """
    lab = np.zeros((40, 40), np.int32)
    lab[10:18, 10:18] = 1
    lab[10:18, 18:26] = 2
    lab[18:26, 10:18] = 3
    lab[18:26, 18:26] = 4
    nuc = np.zeros_like(lab)
    nuc[12:15, 12:15] = 1      # inside cell 1
    nuc[20:23, 20:23] = 2      # inside cell 4
    jm1 = np.zeros(lab.shape, np.float32)
    jm1[18, 18:22] = 100
    jm2 = np.zeros(lab.shape, np.float32)
    jm2[18, 18:26] = 100
    f = Field(planes={Role.JM1: jm1, Role.JM2: jm2, Role.NUCLEUS: np.zeros_like(jm1)},
              pixel_size_um=PX, markers={Role.JM1: "Na/K-ATPase", Role.JM2: "p120"},
              source=Path("x.tif"), field_id=field_id)
    return f, Segmentation(cells=lab, nuclei=nuc, params_used={})


def field_with_isolated_cell():
    f, seg = build()
    lab = seg.cells
    lab[lab == 3] = 0
    lab[32:36, 30:34] = 3          # far from everything, off the border
    return f, seg


def field_with_border_cell():
    f, seg = build()
    seg.cells[0:6, 0:6] = 1        # cell 1 touches the top-left border (old cell 1 is replaced)
    seg.cells[10:18, 10:18] = np.where(seg.cells[10:18, 10:18] == 1, 0, seg.cells[10:18, 10:18])
    return f, seg


@pytest.fixture
def setup():
    return build() + (make_cfg(),)


def test_interface_rows_carry_role_marker_and_method(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    assert {"field_id", "cell_id", "neighbour_id", "corner_method", "role", "marker"} <= set(df.columns)
    assert set(df["marker"]) == {"Na/K-ATPase", "p120"}


def test_both_corner_methods_appear_for_every_measured_cell(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    assert set(df["corner_method"]) == {"dp", "tcj"}
    for cid, g in df.groupby("cell_id"):
        assert set(g["corner_method"]) == {"dp", "tcj"}, cid


def test_hand_computed_values_dp_cell4_vs_cell2(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    sel = df[(df.cell_id == 4) & (df.neighbour_id == 2) & (df.corner_method == "dp")]
    assert set(sel.role) == {"JM1", "JM2"}
    j1 = sel[sel.role == "JM1"].iloc[0]
    j2 = sel[sel.role == "JM2"].iloc[0]
    # interface = row 18, cols 18..25 inclusive: 8 pixels, 7 unit steps
    assert j1.interface_contour_px == pytest.approx(7.0)
    assert j1.interface_contour_um == pytest.approx(3.5)
    assert j1.straight_line_length_px == pytest.approx(7.0)
    assert j1.linearity_index == pytest.approx(1.0)
    # band: 2 cycles of a 1x8 row -> rows 16..20 x cols 16..27 = 5*12 px
    assert j1.dilation_cycles == 2
    assert j1.interface_area_px == 60
    assert j1.interface_area_um2 == pytest.approx(60 * 0.25)
    # JM1 above threshold at cols 18..21 only: 4 px, 3 steps, intensity 400
    assert j1.marker_area_px == 4
    assert j1.marker_area_um2 == pytest.approx(1.0)
    assert j1.marker_intensity == pytest.approx(400.0)
    assert j1.fragmented_contour_px == pytest.approx(3.0)
    assert j1.junction_contour_um == pytest.approx(1.5)
    assert j1.coverage_index == pytest.approx(3.0 / 7.0)
    assert j1.threshold == 50.0
    # JM2 covers the whole interface
    assert j2.marker_area_px == 8
    assert j2.coverage_index == pytest.approx(1.0)


def test_every_metric_has_both_pixel_and_micron_columns(setup):
    f, seg, cfg = setup
    df = measure_field(f, seg, cfg).interfaces
    assert "interface_contour_px" in df and "interface_contour_um" in df
    for stem in ("straight_line_length", "fragmented_contour", "junction_contour",
                 "junction_straight_line"):
        assert f"{stem}_px" in df and f"{stem}_um" in df
        assert np.allclose(df[f"{stem}_um"].astype(float), df[f"{stem}_px"].astype(float) * PX)
    assert "distance_ratio" not in df
    assert (df.linearity_index.dropna() <= 1.0 + 1e-9).all()


def test_isolated_cell_appears_in_cells_with_zero_interfaces():
    f, seg = field_with_isolated_cell()
    res = measure_field(f, seg, make_cfg())
    assert res.cells.loc[res.cells.cell_id == 3, "n_interfaces"].item() == 0
    assert res.flags.loc[res.flags.cell_id == 3, "isolated"].item()
    assert 3 not in set(res.interfaces["cell_id"])


def test_border_touching_cells_are_flagged_and_absent_from_interfaces():
    f, seg = field_with_border_cell()
    res = measure_field(f, seg, make_cfg())
    assert 1 in set(res.flags.loc[res.flags.border_touching, "cell_id"])
    assert 1 not in set(res.interfaces["cell_id"])
    row = res.cells[res.cells.cell_id == 1].iloc[0]
    assert not row.measured and row.n_interfaces == 0


def test_keys_join_interfaces_to_cells_to_nuclei(setup):
    f, seg, cfg = setup
    res = measure_field(f, seg, cfg)
    assert not res.interfaces.merge(res.cells, on=["field_id", "cell_id"]).empty
    assert not res.cells.merge(res.nuclei, on=["field_id", "cell_id"]).empty
    assert len(res.flags) == len(res.cells) == 4
    assert res.cells.epsilon.nunique() == 1


def test_threshold_used_is_recorded_per_field_and_role(setup):
    f, seg, cfg = setup
    assert "threshold" in measure_field(f, seg, cfg).interfaces.columns


def test_nuclei_hand_values(setup):
    f, seg, cfg = setup
    n = measure_field(f, seg, cfg).nuclei.set_index("nucleus_id")
    assert n.loc[1, "cell_id"] == 1 and n.loc[2, "cell_id"] == 4
    assert n.loc[1, "area_px"] == 9 and n.loc[1, "area_um2"] == pytest.approx(2.25)
    # centroids (13,13) and (21,21): distance 8*sqrt(2) px
    assert n.loc[1, "mean_intra_distance_px"] == pytest.approx(8 * 2 ** 0.5)
    assert n.loc[1, "mean_intra_distance_um"] == pytest.approx(4 * 2 ** 0.5)


def test_no_nucleus_gives_empty_nuclei_and_anucleate_flags(setup):
    f, seg, cfg = setup
    seg.nuclei[:] = 0
    res = measure_field(f, seg, cfg)
    assert res.nuclei.empty and "nucleus_id" in res.nuclei.columns
    assert res.flags.anucleate.all()


def test_jm2_is_optional(setup):
    f, seg, cfg = setup
    del f.planes[Role.JM2]
    df = measure_field(f, seg, cfg).interfaces
    assert set(df.role) == {"JM1"}


def test_resolve_thresholds_config_wins_otsu_and_flat_fallback(setup):
    f, seg, cfg = setup
    assert resolve_thresholds(f, cfg) == {Role.JM1: 50.0, Role.JM2: 50.0}
    cfg2 = make_cfg()
    cfg2.threshold = {Role.JM1: 7.0}
    t = resolve_thresholds(f, cfg2)
    assert t[Role.JM1] == 7.0
    assert 0.0 <= t[Role.JM2] < 100.0           # Otsu separates the 0 and 100 populations
    f.planes[Role.JM2] = np.full((40, 40), 3.0, np.float32)
    assert resolve_thresholds(f, cfg2)[Role.JM2] == 0.0


def test_pixel_size_scales_um_not_px(setup):
    f, seg, cfg = setup
    f.pixel_size_um = 2.0
    df = measure_field(f, seg, cfg).interfaces
    r = df[(df.cell_id == 4) & (df.neighbour_id == 2) & (df.corner_method == "dp")
           & (df.role == "JM1")].iloc[0]
    assert r.interface_contour_px == pytest.approx(7.0)
    assert r.interface_contour_um == pytest.approx(14.0)
    assert r.interface_area_um2 == pytest.approx(60 * 4.0)


def test_shape_mismatch_raises_config_error(setup):
    f, seg, cfg = setup
    f.planes[Role.JM1] = np.zeros((5, 5), np.float32)
    with pytest.raises(ConfigError):
        measure_field(f, seg, cfg)
