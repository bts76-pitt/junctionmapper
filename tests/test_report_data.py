import base64
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from junctionmapper.config import Role
from junctionmapper.pipeline import measure_field
from junctionmapper.report.assets import write_field_asset
from junctionmapper.report.build import PARAMETERS
from junctionmapper.report.data import (b64_f32, b64_u32, cell_table, junction_table,
                                        run_summary, write_metric_files, write_run_json,
                                        write_thumbs)
from junctionmapper.tables import make_frame, write_tables
from tests.fixtures13 import build, field_with_border_cell, make_cfg


def _load(path):
    text = path.read_text().rstrip("\n")
    head = text.index("(")
    assert text.endswith(");")
    return json.loads(text[head + 1:-2])


def _f32(s):
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def _u32(s):
    return np.frombuffer(base64.b64decode(s), dtype="<u4")


def _rows(specs, col="band_mean_intensity"):
    """specs: (field, cell, pair_id, value) per row; tcj, JM1."""
    df = make_frame("interfaces", [])
    rows = []
    for i, (f, c, pair, v) in enumerate(specs):
        rows.append({"field_id": f, "cell_id": c, "corner_method": "tcj", "role": "JM1",
                     "marker": "m", "pair_id": pair, "interface_id": f"i{i}", col: v})
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if rows else df


def _two_sided(pair, values):
    lo, hi = (int(x) for x in pair.split("|")[-1].split("-"))
    return _rows([("f", lo, pair, values[0]), ("f", hi, pair, values[1])])


def _cell_rows(cell, values):
    return _rows([("f", cell, f"f|tcj|{cell}-{10 + k}", v) for k, v in enumerate(values)])


def _make_run(tmp_path, jm1_only=False, empty=False):
    f, seg, cfg = build() + (make_cfg(),)
    if jm1_only:
        f.planes.pop(Role.JM2)
        cfg = cfg.model_copy(update={"channels": [c for c in cfg.channels if c.role != Role.JM2],
                                     "threshold": {Role.JM1: 50.0}})
    r = measure_field(f, seg, cfg)
    write_field_asset(f, seg, r, "r1", tmp_path / "report/runs/r1/fields")
    if empty:
        r = dataclasses.replace(r, interfaces=make_frame("interfaces", []))
    write_tables(
        [r], tmp_path)
    write_run_json(tmp_path, cfg, {"tool": "x"}, [f.field_id])
    return SimpleNamespace(dir=tmp_path, interfaces=pd.read_parquet(tmp_path / "interfaces.parquet"),
                           cfg=cfg)


@pytest.fixture
def setup_tables(tmp_path):
    return _make_run(tmp_path)


@pytest.fixture
def setup_tables_jm1_only(tmp_path):
    return _make_run(tmp_path, jm1_only=True)


def test_junction_value_is_the_mean_of_its_cell_sides():
    j = junction_table(_two_sided("f|tcj|2-4", (10.0, 30.0)), "tcj")
    assert len(j) == 1 and j.iloc[0].band_mean_intensity == 20.0
    assert (j.iloc[0].lo, j.iloc[0].hi) == (2, 4)


def test_cell_value_is_the_median_of_its_rows():
    assert cell_table(_cell_rows(3, (1.0, 2.0, 9.0)), "tcj").iloc[0].band_mean_intensity == 2.0


def test_nan_stays_nan_and_is_not_zero():
    j = junction_table(_two_sided("f|tcj|2-4", (np.nan, np.nan)), "tcj")
    assert np.isnan(j.iloc[0].band_mean_intensity)


def test_nan_is_skipped_when_another_side_has_a_value():
    j = junction_table(_two_sided("f|tcj|2-4", (np.nan, 6.0)), "tcj")
    assert j.iloc[0].band_mean_intensity == 6.0


def test_tables_are_filtered_by_method_and_sorted():
    ifc = _rows([("b", 1, "b|tcj|1-2", 1.0), ("a", 5, "a|tcj|5-9", 1.0), ("a", 2, "a|tcj|2-3", 1.0)])
    ifc.loc[0, "corner_method"] = "dp"
    j = junction_table(ifc, "tcj")
    assert list(j.field_id) == ["a", "a"] and list(j.lo) == [2, 5]


def test_b64_round_trip_is_little_endian():
    assert _f32(b64_f32([1.5, np.nan]))[0] == 1.5 and np.isnan(_f32(b64_f32([1.5, np.nan]))[1])
    assert _u32(b64_u32([1, 258])).tolist() == [1, 258]
    assert base64.b64decode(b64_u32([1])) == b"\x01\x00\x00\x00"


def test_metric_files_align_with_the_index(tmp_path, setup_tables):
    write_metric_files(setup_tables.interfaces, "r1", tmp_path, calibrated=True)
    idx = _load(tmp_path / "tcj/_index.js")
    m = _load(tmp_path / "tcj/JM1/band_mean_intensity.js")
    assert len(_f32(m["junction_px"])) == len(_u32(idx["junction_lo"])) > 0
    assert len(_f32(m["cell_px"])) == len(_u32(idx["cell_id"])) > 0
    assert idx["fields"] == ["f1"] and idx["run"] == "r1" and m["role"] == "JM1"
    assert m["method"] == "tcj" and m["key"] == "band_mean_intensity"


def test_calibrated_run_writes_um_arrays(tmp_path, setup_tables):
    write_metric_files(setup_tables.interfaces, "r1", tmp_path, calibrated=True)
    m = _load(tmp_path / "tcj/JM1/interface_contour.js")
    assert len(_f32(m["junction_um"])) == len(_f32(m["junction_px"]))
    assert _load(tmp_path / "tcj/JM1/linearity_index.js")["junction_um"] is None


def test_uncalibrated_run_writes_no_um_arrays(tmp_path, setup_tables):
    write_metric_files(setup_tables.interfaces, "r1", tmp_path, calibrated=False)
    m = _load(tmp_path / "tcj/JM1/interface_contour.js")
    assert m["junction_um"] is None and m["cell_um"] is None


def test_every_metric_of_a_method_shares_one_index_order(tmp_path, setup_tables):
    write_metric_files(setup_tables.interfaces, "r1", tmp_path, calibrated=True)
    n = {len(_f32(_load(tmp_path / f"dp/JM{k}/{p['key']}.js")["junction_px"]))
         for k in (1, 2) for p in PARAMETERS}
    assert n == {len(_u32(_load(tmp_path / "dp/_index.js")["junction_lo"]))}


def test_run_without_jm2_writes_only_jm1(tmp_path, setup_tables_jm1_only):
    write_metric_files(setup_tables_jm1_only.interfaces, "r1", tmp_path, calibrated=True)
    assert (tmp_path / "tcj/JM1").exists() and not (tmp_path / "tcj/JM2").exists()


def test_run_summary_of_a_real_run(setup_tables):
    s = run_summary(setup_tables.dir, "r7", "my run")
    assert s["run"] == "r7" and s["label"] == "my run" and s["calibrated"] is True
    assert s["methods"] == ["dp", "tcj"]
    assert [r["role"] for r in s["roles"]] == ["JM1", "JM2"] and s["roles"][0]["marker"] == "Na/K-ATPase"
    assert [m["key"] for m in s["metrics"]] == [p["key"] for p in PARAMETERS]
    units = {m["key"]: (m["unit_px"], m["unit_um"]) for m in s["metrics"]}
    assert units["interface_contour"] == ("px", "µm") and units["interface_area"] == ("px²", "µm²")
    assert units["linearity_index"] == ("", "") and units["intensity_per_area"] == ("", "")
    assert all(m["help"] for m in s["metrics"])
    f = s["fields"][0]
    assert f["field_id"] == "f1" and "thumb" not in f
    assert f["qc"]["flagged_cells"] >= 0 and f["n_junctions"] > 0
    assert s["tiles"]["fields"] == 1 and s["tiles"]["cells"] == f["n_cells"]
    assert s["tiles"]["junctions"] == f["n_junctions"]
    assert s["provenance"] == {"tool": "x"}


def test_flagged_cells_count_only_measured_cells(tmp_path):
    f, seg = field_with_border_cell()
    cfg = make_cfg()
    write_tables([measure_field(f, seg, cfg)], tmp_path)
    write_run_json(tmp_path, cfg, {}, [f.field_id])
    flags = pd.read_parquet(tmp_path / "flags.parquet")
    ifc = pd.read_parquet(tmp_path / "interfaces.parquet")
    measured = set(zip(ifc.field_id, ifc.cell_id.astype(int)))
    border = flags[flags["border_touching"]]
    assert len(border) and not set(zip(border.field_id, border.cell_id.astype(int))) & measured
    s = run_summary(tmp_path, "r1", "x")
    bools = [c for c in flags.columns if pd.api.types.is_bool_dtype(flags[c])]
    want = sum(1 for _, r in flags.iterrows()
               if (r.field_id, int(r.cell_id)) in measured and r[bools].any())
    assert s["tiles"]["flagged_cells"] == want <= s["tiles"]["cells"]
    for f in s["fields"]:
        assert f["qc"]["flagged_cells"] <= f["n_cells"]
        assert all(v <= f["n_cells"] for v in f["qc"].values())


def test_write_thumbs_registers_one_jpeg_per_field(setup_tables):
    path = write_thumbs(setup_tables.dir)
    assert path == setup_tables.dir / "report/runs/r1/thumbs.js"
    text = path.read_text()
    assert text.startswith("JM.registerThumbs(")
    thumbs = _load(path)
    assert list(thumbs) == ["f1"] and thumbs["f1"].startswith("data:image/jpeg;base64,")


def test_field_with_no_interfaces_still_builds_a_summary(tmp_path):
    run = _make_run(tmp_path, empty=True)
    s = run_summary(run.dir, "r1", "x")
    assert s["fields"][0]["n_junctions"] == 0 and s["fields"][0]["n_cells"] == 0
    assert s["methods"] == [] and s["tiles"]["junctions"] == 0
    write_metric_files(run.interfaces, "r1", tmp_path / "m", calibrated=True)


def test_every_parameter_has_a_one_line_help():
    assert all(p["help"] and "\n" not in p["help"] and len(p["help"]) <= 120 for p in PARAMETERS)
