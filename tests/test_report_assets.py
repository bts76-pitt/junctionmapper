import base64
import dataclasses
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from junctionmapper.io.reader import Field
from junctionmapper.pipeline import measure_field
from junctionmapper.config import Role
from junctionmapper.measure import interface as mi
from junctionmapper.report.assets import (cell_outlines, junction_polylines, junction_profile, preview,
                                          preview_scale, simplify, to_preview,
                                          write_field_asset)
from junctionmapper.segment.cellpose import Segmentation
from tests.fixtures13 import build, make_cfg


@pytest.fixture
def setup():
    return build() + (make_cfg(),)


def _load(path, name):
    text = path.read_text()
    head, tail = f"JM.register{name}(", ");"
    assert text.startswith(head) and text.rstrip("\n").endswith(tail)
    assert text[len(text.rstrip("\n")):] in ("", "\n")
    return json.loads(text[len(head):text.rstrip("\n").rindex(tail)])


def test_preview_scale_caps_the_long_side():
    assert preview_scale((2500, 2984)) == pytest.approx(1500 / 2984)
    assert preview_scale((400, 300)) == 1.0


def test_preview_is_a_jpeg_data_uri_of_the_scaled_size():
    uri = preview(np.random.default_rng(0).random((300, 200)) * 255, 0.5)
    img = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert uri.startswith("data:image/jpeg;base64,") and img.size == (100, 150)


def test_preview_of_a_flat_or_nan_plane_does_not_fail():
    preview(np.zeros((10, 10)), 1.0)
    preview(np.full((10, 10), np.nan), 1.0)


def test_simplify_keeps_endpoints_and_drops_collinear_points():
    line = np.array([[0, i] for i in range(50)])
    assert simplify(line, 1.0).tolist() == [[0, 0], [0, 49]]


def test_to_preview_swaps_to_xy_and_puts_a_pixel_at_its_centre():
    # raw pixel (row 10, col 20) covers x 20..21, y 10..11 of the drawn image
    assert to_preview(np.array([[10, 20]]), 1.0) == [[20.5, 10.5]]
    assert to_preview(np.array([[9, 20]]), 0.25) == [[5.0, 2.5]]


def test_a_field_longer_than_the_preview_is_sized_and_drawn_at_the_preview(tmp_path, setup):
    f, seg, cfg = setup
    r = measure_field(f, seg, cfg)
    h, w = 1600, 3000
    labels = np.ones((h, w), np.int32)                      # one cell filling the field
    big = Field(planes={role: np.zeros((h, w), np.float32) for role in f.planes},
                pixel_size_um=f.pixel_size_um, markers=f.markers, source=f.source, field_id="big")
    empty = dataclasses.replace(r, interfaces=r.interfaces.iloc[0:0], flags=r.flags.iloc[0:0], paths={})
    d = _load(write_field_asset(big, Segmentation(labels, np.zeros_like(labels), {}), empty, "r1",
                                tmp_path), "Field")
    assert d["scale"] == pytest.approx(0.5)
    assert (d["width"], d["height"]) == (1500, 800)            # the planes the viewer draws
    assert (d["raw_width"], d["raw_height"]) == (3000, 1600)
    for c in d["channels"]:
        img = Image.open(io.BytesIO(base64.b64decode(c["src"].split(",", 1)[1])))
        assert img.size == (1500, 800)
    xy = np.array(d["cells"]["1"])                         # the outline spans exactly the preview
    assert (xy[:, 0].min(), xy[:, 0].max()) == (0, 1500)
    assert (xy[:, 1].min(), xy[:, 1].max()) == (0, 800)


def test_cell_outlines_cover_every_label(setup):
    _, seg, _ = setup
    out = cell_outlines(seg.cells, 1.0)
    assert set(out) == {"1", "2", "3", "4"} and all(len(v) >= 4 for v in out.values())


def test_junction_polylines_group_cell_sides_by_pair(setup):
    f, seg, cfg = setup
    j = junction_polylines(measure_field(f, seg, cfg), 1.0)
    pairs = {p.split("|")[-1] for p in measure_field(f, seg, cfg).interfaces.pair_id}
    assert set(j["tcj"]) <= pairs and set(j) == {"tcj", "dp"}
    assert j["dp"]


def test_field_asset_registers_one_field_with_every_configured_channel(tmp_path, setup):
    f, seg, cfg = setup
    p = write_field_asset(f, seg, measure_field(f, seg, cfg), "r1", tmp_path)
    d = _load(p, "Field")
    assert d["run"] == "r1" and {c["role"] for c in d["channels"]} == {r.value for r in f.planes}
    assert d["labels_sha256"] == hashlib.sha256(seg.cells.tobytes()).hexdigest()


def test_field_asset_lists_each_cell_side_with_the_table_values(tmp_path, setup):
    from junctionmapper.report.summary import PARAMETERS
    f, seg, cfg = setup
    r = measure_field(f, seg, cfg)
    d = _load(write_field_asset(f, seg, r, "r1", tmp_path), "Field")
    itf = r.interfaces
    groups = [g for _, g in itf.groupby(["corner_method", "role", "pair_id"]) if len(g) == 2]
    pair = groups[0]
    method, role, pid = pair.iloc[0][["corner_method", "role", "pair_id"]]
    side = d["sides"][method][pid.split("|")[-1]][role]
    assert set(side) == {str(int(c)) for c in pair.cell_id}
    for _, row in pair.iterrows():
        got = side[str(int(row.cell_id))]
        assert set(got) == {p["key"] for p in PARAMETERS}
        for p in PARAMETERS:
            px, um = got[p["key"]]
            want = row[p["px"]]
            assert (px is None) if np.isnan(want) else px == pytest.approx(want, rel=1e-6)
            assert um is None if not p["um"] or np.isnan(row[p["um"]]) else um == pytest.approx(row[p["um"]], rel=1e-6)


def test_field_asset_sides_serialise_nan_as_null(tmp_path, setup):
    from junctionmapper.report.summary import PARAMETERS
    f, seg, cfg = setup
    r = measure_field(f, seg, cfg)
    key = PARAMETERS[0]["px"]
    r.interfaces.loc[r.interfaces.index[0], key] = np.nan
    row = r.interfaces.iloc[0]
    p = write_field_asset(f, seg, r, "r1", tmp_path)
    assert "NaN" not in p.read_text()
    d = _load(p, "Field")
    got = d["sides"][row.corner_method][row.pair_id.split("|")[-1]][row.role][str(int(row.cell_id))]
    assert got[PARAMETERS[0]["key"]][0] is None


def test_profile_of_a_uniform_band_is_flat_at_the_image_value():
    img = np.full((20, 20), 7.0)
    path = np.array([[10, c] for c in range(3, 15)])
    band = mi.dilated_band(path, img.shape, 2)
    p = junction_profile(path, img, band, 2, None)
    assert p["y"] == pytest.approx([7.0] * len(p["y"])) and p["x"][0] == 0
    assert p["x"][-1] == pytest.approx(11)


def test_profile_x_is_in_micrometres_when_calibrated():
    path = np.array([[5, c] for c in range(10)])
    img = np.ones((12, 12))
    p = junction_profile(path, img, mi.dilated_band(path, img.shape, 1), 1, 0.5)
    assert p["x"][-1] == pytest.approx(4.5)


def test_profile_is_resampled_to_at_most_200_points_keeping_the_ends():
    path = np.array([[1, c] for c in range(600)])
    img = np.tile(np.arange(600.0), (3, 1))
    p = junction_profile(path, img, mi.dilated_band(path, img.shape, 0), 0, None)
    assert len(p["x"]) == 200 and p["y"][0] == 0 and p["y"][-1] == 599


def test_a_two_pixel_junction_still_has_a_profile():
    path = np.array([[4, 4], [4, 5]])
    img = np.ones((9, 9))
    assert len(junction_profile(path, img, mi.dilated_band(path, img.shape, 2), 2, None)["y"]) >= 2


def test_a_one_pixel_junction_has_two_equal_points():
    path = np.array([[4, 4]])
    img = np.full((9, 9), 3.0)
    assert junction_profile(path, img, mi.dilated_band(path, img.shape, 1), 1, None) == \
        {"x": [0.0, 0.0], "y": [3.0, 3.0]}


def test_profile_sees_a_gap_in_the_marker(tmp_path, setup):
    f, seg, cfg = setup
    probe = measure_field(f, seg, cfg)
    iid = next(i for i in probe.paths if i.startswith("f1|tcj|1@"))
    path = probe.paths[iid]
    n = len(path)
    planes = dict(f.planes)
    jm1 = planes[Role.JM1].copy()
    jm1[:] = 100.0
    for r, c in path[n // 3: 2 * n // 3 + 1]:
        jm1[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4] = 0.0
    planes[Role.JM1] = jm1
    f2 = dataclasses.replace(f, planes=planes)
    res = measure_field(f2, seg, cfg)
    key = str(res.interfaces[res.interfaces.interface_id == iid].pair_id.iloc[0]).split("|")[-1]
    d = _load(write_field_asset(f2, seg, res, "r1", tmp_path), "Field")
    y = d["profiles"]["tcj"][key]["JM1"]["y"]
    k = len(y)
    mid = np.mean(y[k * 2 // 5: k * 3 // 5])
    ends = np.mean(y[: k // 5] + y[-(k // 5):])
    assert mid <= 0.1 * ends


def test_field_asset_carries_profiles_thresholds_and_unit(tmp_path, setup):
    f, seg, cfg = setup
    d = _load(write_field_asset(f, seg, measure_field(f, seg, cfg), "r1", tmp_path,
                                thresholds={"JM1": 12.5}), "Field")
    key = next(iter(d["junctions"]["tcj"]))
    assert set(d["profiles"]["tcj"][key]) >= {"JM1"} and d["thresholds"] == {"JM1": 12.5}
    assert d["profile_unit"] in ("µm", "px")


def test_a_nan_threshold_becomes_null(tmp_path, setup):
    f, seg, cfg = setup
    d = _load(write_field_asset(f, seg, measure_field(f, seg, cfg), "r1", tmp_path,
                                thresholds={"JM1": float("nan")}), "Field")
    assert d["thresholds"] == {"JM1": None}
