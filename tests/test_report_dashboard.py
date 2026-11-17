import json
import re

import pytest

from junctionmapper.config import Role
from junctionmapper.errors import ConfigError
from junctionmapper.pipeline import measure_field
from junctionmapper.report.assets import write_field_asset
from junctionmapper.report.dashboard import _plotly, build_dashboard
from junctionmapper.report.data import write_metric_files, write_run_json, write_thumbs
from junctionmapper.tables import write_tables
from tests.fixtures13 import build, make_cfg


def _make_run(d, jm1_only=False):
    f, seg = build()
    cfg = make_cfg()
    if jm1_only:
        f.planes.pop(Role.JM2)
        cfg = cfg.model_copy(update={"channels": [c for c in cfg.channels if c.role != Role.JM2],
                                     "threshold": {Role.JM1: 50.0}})
    r = measure_field(f, seg, cfg)
    write_tables([r], d)
    write_run_json(d, cfg, {"tool": "x"}, [f.field_id])
    write_metric_files(r.interfaces, "r1", d / "report/runs/r1/metrics", calibrated=True)
    write_field_asset(f, seg, r, "r1", d / "report/runs/r1/fields")
    write_thumbs(d)
    return d


@pytest.fixture
def built_run(tmp_path):
    return _make_run(tmp_path / "runA")


@pytest.fixture
def built_run2(tmp_path):
    return _make_run(tmp_path / "runB")


def _runs(html):
    """The JSON argument of every inlined JM.registerRun(...) call."""
    return [json.loads(m) for m in re.findall(r"JM\.registerRun\((.*?)\);\n", html)]


def test_index_html_references_nothing_external(tmp_path, built_run):
    html = build_dashboard([(built_run, "A")], tmp_path / "rep").read_text()
    assert not re.search(r"(src|href)\s*=\s*\"https?://", html) and "fetch(" not in html


def test_one_register_run_per_run_and_runs_are_namespaced(tmp_path, built_run, built_run2):
    html = build_dashboard([(built_run, "A"), (built_run2, "B")], tmp_path / "rep").read_text()
    assert html.count("JM.registerRun(") == 2
    assert (tmp_path / "rep/runs/r1/fields").is_dir() and (tmp_path / "rep/runs/r2/fields").is_dir()


def test_duplicate_labels_are_rejected(tmp_path, built_run):
    with pytest.raises(ConfigError, match="duplicate run label"):
        build_dashboard([(built_run, "A"), (built_run, "A")], tmp_path / "rep")


def test_run_ids_follow_argument_order(tmp_path, built_run, built_run2):
    html = build_dashboard([(built_run2, "B"), (built_run, "A")], tmp_path / "rep").read_text()
    assert [(r["run"], r["label"]) for r in _runs(html)] == [("r1", "B"), ("r2", "A")]
    assert (tmp_path / "rep/runs/r2/metrics/tcj/_index.js").is_file()


def test_build_inside_own_report_dir_keeps_files_in_place(built_run):
    asset = built_run / "report/runs/r1/fields/f1.js"
    before = asset.stat().st_mtime_ns
    out = build_dashboard([(built_run, "A")], built_run / "report")
    assert out == built_run / "report/index.html" and out.is_file()
    assert asset.stat().st_mtime_ns == before
    assert sorted(p.name for p in (built_run / "report/runs").iterdir()) == ["r1"]


def test_own_report_dir_of_a_later_run_is_refused(built_run, built_run2):
    with pytest.raises(ConfigError, match="first run"):
        build_dashboard([(built_run2, "B"), (built_run, "A")], built_run / "report")


def test_run_without_report_assets_is_a_config_error(tmp_path, built_run):
    with pytest.raises(ConfigError, match="no report assets"):
        build_dashboard([(tmp_path / "nothing", "A")], tmp_path / "rep")


def test_labels_cannot_break_out_of_the_script(tmp_path, built_run):
    html = build_dashboard([(built_run, "</script><b>x")], tmp_path / "rep").read_text()
    assert "</script><b>" not in html
    assert _runs(html)[0]["label"] == "</script><b>x"


def test_index_inlines_the_app_and_no_per_junction_values(tmp_path, built_run):
    html = build_dashboard([(built_run, "A")], tmp_path / "rep").read_text()
    for marker in ("JMLogic", "Plotly", "registerRun", "--accent"):
        assert marker in html
    summary = json.dumps(_runs(html))
    assert "junction_px" not in summary and "cells" in summary


def test_index_html_embeds_no_images_and_each_run_has_thumbs(tmp_path, built_run, built_run2):
    html = build_dashboard([(built_run, "A"), (built_run2, "B")], tmp_path / "rep").read_text()
    assert html.count(_plotly()) == 1           # vendored Plotly carries its own icon URIs
    assert "data:image" not in html.replace(_plotly(), "")
    for k in (1, 2):
        assert (tmp_path / f"rep/runs/r{k}/thumbs.js").read_text().startswith("JM.registerThumbs(")


def test_rebuilding_into_the_same_folder_leaves_no_stale_files(tmp_path, built_run):
    out = tmp_path / "rep"
    build_dashboard([(built_run, "A")], out)
    assert (out / "runs/r1/metrics/tcj/JM2").is_dir()
    other = _make_run(tmp_path / "jm1only", jm1_only=True)
    build_dashboard([(other, "B")], out)
    assert not (out / "runs/r1/metrics/tcj/JM2").exists()
    assert (out / "runs/r1/metrics/tcj/JM1").is_dir()
