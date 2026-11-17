import base64
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from junctionmapper.pipeline import measure_field
from junctionmapper.report.assets import (cell_outlines, junction_polylines, preview,
                                          preview_scale, simplify, to_preview,
                                          write_field_asset)
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


def test_to_preview_swaps_to_xy_and_scales():
    assert to_preview(np.array([[10, 20]]), 0.5) == [[10.0, 5.0]]


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
    from junctionmapper.report.build import PARAMETERS
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
    from junctionmapper.report.build import PARAMETERS
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
