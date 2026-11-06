"""CLI tests. Cellpose is not installed here, so measurement tests go through ``--labels`` and
``segment`` is tested with ``segment_field`` stubbed. Inputs are real TIFFs and real YAML."""
import json

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml
from typer.testing import CliRunner

from junctionmapper import cli
from junctionmapper.cli import app
from junctionmapper.errors import SegmentationError
from junctionmapper.segment.cellpose import Segmentation
from tests.fixtures13 import build

FID = "img_t0_z0"
CFG = {
    "channels": [
        {"role": "JM1", "channel": 0, "marker": "Na/K-ATPase"},
        {"role": "JM2", "channel": 1, "marker": "p120"},
        {"role": "nucleus", "channel": 2, "marker": "DAPI"},
    ],
    "pixel_size_um": 0.5,
    "threshold": {"JM1": 50.0, "JM2": 20.0},
}


def invoke(*args):
    return CliRunner().invoke(app, [str(a) for a in args])


def write_cfg(path, **over):
    d = {**CFG, **over}
    d = {k: v for k, v in d.items() if v is not ...}
    path.write_text(yaml.safe_dump(d))
    return path


def write_tif(path, planes=None):
    f, seg = build()
    if planes is None:
        planes = np.stack([f.planes[r] for r in list(f.planes)]).astype(np.float32)
    tifffile.imwrite(path, planes, photometric="minisblack")
    return seg


def save_labels(d, fid, seg):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{fid}.JM1.npy", seg.cells)
    np.save(d / f"{fid}.nucleus.npy", seg.nuclei)


@pytest.fixture
def env(tmp_path):
    cfg = write_cfg(tmp_path / "cfg.yaml")
    tif = tmp_path / "img.tif"
    seg = write_tif(tif)
    labels = tmp_path / "labs"
    save_labels(labels, FID, seg)
    return {"cfg": cfg, "tif": tif, "seg": seg, "labels": labels, "out": tmp_path / "out", "tmp": tmp_path}


def run_ok(env, **kw):
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    return r


def stub_segment(monkeypatch, seg):
    def fake(field, params):
        return Segmentation(cells=seg.cells.copy(), nuclei=seg.nuclei.copy(),
                            params_used={"model": params.model, "field_id": field.field_id,
                                         "flow_threshold": params.flow_threshold})
    monkeypatch.setattr(cli, "segment_field", fake)


def forbid_cellpose(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("segmentation must not run")
    monkeypatch.setattr(cli, "segment_field", boom)


# ---- run ----------------------------------------------------------------------------

def test_run_writes_tables_and_report(env, monkeypatch):
    stub_segment(monkeypatch, env["seg"])
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--out", env["out"])
    assert r.exit_code == 0, r.output
    for name in ("interfaces", "cells", "nuclei", "nuclei_distances", "flags"):
        for ext in ("parquet", "csv"):
            assert (env["out"] / f"{name}.{ext}").exists()
    assert (env["out"] / "report.html").exists()
    assert len(pd.read_parquet(env["out"] / "interfaces.parquet")) > 0
    # fresh segmentation is saved so the run can be resumed / corrected
    assert (env["out"] / "labels" / f"{FID}.JM1.npy").exists()


def test_run_with_labels_dir_skips_segmentation(env, monkeypatch):
    forbid_cellpose(monkeypatch)
    r = run_ok(env)
    assert r.exit_code == 0, r.output
    assert (env["out"] / "report.html").exists()
    assert set(pd.read_parquet(env["out"] / "cells.parquet").field_id) == {FID}


def test_run_without_cellpose_gives_actionable_message_not_traceback(env):
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--out", env["out"])
    assert r.exit_code != 0
    assert "segmentation" in r.output and "pip install" in r.output
    assert "Traceback" not in r.output
    assert isinstance(r.exception, SystemExit)
    assert not (env["out"] / "interfaces.parquet").exists()


def test_report_is_deterministic_and_has_no_timestamp(env):
    run_ok(env)
    a = (env["out"] / "report.html").read_bytes()
    env2 = dict(env, out=env["tmp"] / "out2")
    run_ok(env2)
    assert a == (env2["out"] / "report.html").read_bytes()


def test_labels_run_provenance_records_disk_and_directory(env):
    run_ok(env)
    html = (env["out"] / "report.html").read_text()
    assert "labels_source" in html and "disk" in html
    assert "labels_dir" in html and str(env["labels"]) in html


def test_fresh_run_provenance_records_segmentation_params(env, monkeypatch):
    stub_segment(monkeypatch, env["seg"])
    invoke("run", "--config", env["cfg"], "--input", env["tif"], "--out", env["out"],
           "--flow-threshold", "0.7")
    html = (env["out"] / "report.html").read_text()
    assert "flow_threshold" in html and "0.7" in html
    assert "disk" not in html


# ---- labels contract ----------------------------------------------------------------

def test_missing_label_file_is_an_error_naming_file_and_dir_no_fallback(env, monkeypatch):
    forbid_cellpose(monkeypatch)       # a silent fallback to segmentation would trip this
    (env["labels"] / f"{FID}.JM1.npy").unlink()
    r = run_ok(env)
    assert r.exit_code != 0
    assert f"{FID}.JM1.npy" in r.output and str(env["labels"]) in r.output
    assert not (env["out"] / "interfaces.parquet").exists()
    assert not isinstance(r.exception, AssertionError)


def test_missing_nucleus_label_file_is_an_error(env):
    (env["labels"] / f"{FID}.nucleus.npy").unlink()
    r = run_ok(env)
    assert r.exit_code != 0 and f"{FID}.nucleus.npy" in r.output


def test_labels_dir_that_does_not_exist(env):
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--labels",
               env["tmp"] / "nope", "--out", env["out"])
    assert r.exit_code != 0 and "nope" in r.output and "Traceback" not in r.output


def test_label_shape_mismatch_names_both_shapes(env):
    np.save(env["labels"] / f"{FID}.JM1.npy", np.zeros((10, 12), np.int32))
    r = run_ok(env)
    assert r.exit_code != 0
    assert "(10, 12)" in r.output and "(40, 40)" in r.output
    assert f"{FID}.JM1.npy" in r.output            # our own check, not only the pipeline's


def test_nucleus_label_shape_mismatch_is_an_error(env):
    np.save(env["labels"] / f"{FID}.nucleus.npy", np.zeros((7, 7), np.int32))
    r = run_ok(env)
    assert r.exit_code != 0 and "(7, 7)" in r.output and f"{FID}.nucleus.npy" in r.output


def test_non_integer_labels_rejected(env):
    np.save(env["labels"] / f"{FID}.JM1.npy", env["seg"].cells.astype(np.float32))
    r = run_ok(env)
    assert r.exit_code != 0 and "integer" in r.output


def test_non_contiguous_label_ids_round_trip_exactly(env, monkeypatch):
    cells = np.zeros((40, 40), np.int32)
    cells[10:18, 10:18] = 1
    cells[10:18, 18:26] = 5
    cells[18:26, 10:18] = 9000
    nuc = np.zeros((40, 40), np.uint16)
    nuc[12:15, 12:15] = 7
    seg = Segmentation(cells=cells, nuclei=nuc, params_used={"model": "stub"})
    monkeypatch.setattr(cli, "segment_field", lambda f, p: seg)
    r = invoke("segment", "--config", env["cfg"], "--input", env["tif"], "--out", env["out"])
    assert r.exit_code == 0, r.output
    got = np.load(env["out"] / "labels" / f"{FID}.JM1.npy")
    assert got.dtype == cells.dtype and np.array_equal(got, cells)
    gotn = np.load(env["out"] / "labels" / f"{FID}.nucleus.npy")
    assert gotn.dtype == nuc.dtype and np.array_equal(gotn, nuc)
    # and the run path reads them back unrelabelled
    forbid_cellpose(monkeypatch)
    out2 = env["tmp"] / "out2"
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--labels",
               env["out"] / "labels", "--out", out2)
    assert r.exit_code == 0, r.output
    ids = set(pd.read_parquet(out2 / "cells.parquet").cell_id)
    assert ids == {1, 5, 9000}
    assert set(pd.read_parquet(out2 / "nuclei.parquet").nucleus_id) == {7}


# ---- segment ------------------------------------------------------------------------

def test_segment_writes_labels_and_no_tables(env, monkeypatch):
    stub_segment(monkeypatch, env["seg"])
    r = invoke("segment", "--config", env["cfg"], "--input", env["tif"], "--out", env["out"],
               "--flow-threshold", "0.6")
    assert r.exit_code == 0, r.output
    assert (env["out"] / "labels").is_dir()
    assert not (env["out"] / "interfaces.parquet").exists()
    assert not (env["out"] / "report.html").exists()
    assert not list(env["out"].glob("*.csv"))
    side = json.loads((env["out"] / "labels" / f"{FID}.provenance.json").read_text())
    assert side["params_used"]["flow_threshold"] == 0.6
    assert side["config"]["pixel_size_um"] == 0.5
    assert side["config_as_given"]["channels"][0]["marker"] == "Na/K-ATPase"
    assert "junctionmapper_version" in side
    assert side["field_id"] == FID
    assert side["files"] == {"JM1": f"{FID}.JM1.npy", "nucleus": f"{FID}.nucleus.npy"}


def test_segment_output_feeds_run_labels(env, monkeypatch):
    stub_segment(monkeypatch, env["seg"])
    assert invoke("segment", "--config", env["cfg"], "--input", env["tif"],
                  "--out", env["out"]).exit_code == 0
    forbid_cellpose(monkeypatch)
    out2 = env["tmp"] / "out2"
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--labels",
               env["out"] / "labels", "--out", out2)
    assert r.exit_code == 0, r.output


def test_segment_without_cellpose_is_clean(env):
    r = invoke("segment", "--config", env["cfg"], "--input", env["tif"], "--out", env["out"])
    assert r.exit_code != 0 and "pip install" in r.output and "Traceback" not in r.output


# ---- calibration / config errors ----------------------------------------------------

def test_missing_calibration_exits_nonzero_with_actionable_message(env):
    cfg = write_cfg(env["tmp"] / "nocal.yaml", pixel_size_um=None)
    r = invoke("run", "--config", cfg, "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code != 0 and "pixel_size_um" in r.output
    assert "Traceback" not in r.output


def test_omitted_pixel_size_key_is_same_as_null(env):
    cfg = write_cfg(env["tmp"] / "nocal.yaml", pixel_size_um=...)
    r = invoke("run", "--config", cfg, "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code != 0 and "pixel_size_um" in r.output
    assert "Field required" not in r.output


def _assert_clean(r, cfg_path, *needles):
    assert r.exit_code != 0
    assert str(cfg_path) in r.output
    for n in needles:
        assert n in r.output, r.output
    assert "Traceback" not in r.output
    assert isinstance(r.exception, SystemExit)


def test_malformed_yaml(env):
    bad = env["tmp"] / "bad.yaml"
    bad.write_text("channels: [unclosed\n  pixel_size_um: : 3\n")
    r = invoke("run", "--config", bad, "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    _assert_clean(r, bad, "YAML")


def test_yaml_that_is_not_a_mapping(env):
    bad = env["tmp"] / "list.yaml"
    bad.write_text("- a\n- b\n")
    r = invoke("run", "--config", bad, "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    _assert_clean(r, bad, "mapping")


def test_unknown_config_key_is_an_error_not_silently_ignored(env):
    cfg = write_cfg(env["tmp"] / "typo.yaml", treshold={"JM1": 1})
    r = invoke("run", "--config", cfg, "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    _assert_clean(r, cfg, "treshold", "unknown")


def test_missing_required_role(env):
    cfg = write_cfg(env["tmp"] / "norole.yaml", channels=[CFG["channels"][0], CFG["channels"][1]])
    r = invoke("run", "--config", cfg, "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    _assert_clean(r, cfg, "nucleus")


def test_misnamed_role(env):
    chans = [dict(c) for c in CFG["channels"]]
    chans[0]["role"] = "JM_1"
    cfg = write_cfg(env["tmp"] / "badrole.yaml", channels=chans)
    r = invoke("run", "--config", cfg, "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    _assert_clean(r, cfg, "role")


def test_unreadable_config_path(env):
    missing = env["tmp"] / "does_not_exist.yaml"
    r = invoke("run", "--config", missing, "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    _assert_clean(r, missing, "cannot read")


def test_config_path_is_a_directory(env):
    r = invoke("run", "--config", env["tmp"], "--input", env["tif"], "--labels", env["labels"],
               "--out", env["out"])
    _assert_clean(r, env["tmp"], "cannot read")


def test_missing_input_path_is_clean(env):
    missing = env["tmp"] / "gone.tif"
    r = invoke("run", "--config", env["cfg"], "--input", missing, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code != 0 and "gone.tif" in r.output and "Traceback" not in r.output


def test_input_that_is_not_a_tiff_is_clean(env):
    junk = env["tmp"] / "junk.tif"
    junk.write_bytes(b"this is not a tiff")
    r = invoke("run", "--config", env["cfg"], "--input", junk, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code != 0 and "junk.tif" in r.output and "Traceback" not in r.output
    assert isinstance(r.exception, SystemExit)


# ---- batch failure policy -----------------------------------------------------------

def make_batch(env, good=("a", "b")):
    d = env["tmp"] / "batch"
    d.mkdir()
    for n in ("a", "b"):
        write_tif(d / f"{n}.tif")
    for n in good:
        save_labels(env["labels"], f"{n}_t0_z0", env["seg"])
    return d


def test_one_bad_field_does_not_lose_the_batch(env):
    d = make_batch(env, good=("a",))                  # b has no label files
    r = invoke("run", "--config", env["cfg"], "--input", d, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code == 3, r.output                  # partial: unambiguously not 0
    assert (env["out"] / "report.html").exists()
    assert set(pd.read_parquet(env["out"] / "cells.parquet").field_id) == {"a_t0_z0"}
    assert "1 succeeded" in r.output and "1 failed" in r.output
    assert "b_t0_z0" in r.output and "b_t0_z0.JM1.npy" in r.output


def test_all_good_batch_exits_zero(env):
    d = make_batch(env)
    r = invoke("run", "--config", env["cfg"], "--input", d, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code == 0, r.output
    assert set(pd.read_parquet(env["out"] / "cells.parquet").field_id) == {"a_t0_z0", "b_t0_z0"}
    assert "2 succeeded" in r.output and "0 failed" in r.output


def test_all_fields_failed_writes_nothing_and_says_so(env):
    d = make_batch(env, good=())
    r = invoke("run", "--config", env["cfg"], "--input", d, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code == 1
    assert "every field failed" in r.output
    assert not (env["out"] / "report.html").exists()
    assert not (env["out"] / "interfaces.parquet").exists()


def test_failure_reasons_are_grouped_not_repeated(env):
    d = make_batch(env, good=())
    r = invoke("run", "--config", env["cfg"], "--input", d, "--out", env["out"])  # no cellpose
    assert r.exit_code == 1
    assert r.output.count("pip install") == 1
    assert "a_t0_z0" in r.output and "b_t0_z0" in r.output


def test_file_level_failure_does_not_stop_other_files(env):
    d = make_batch(env)
    write_tif(d / "c.tif", planes=np.zeros((2, 40, 40), np.float32))   # only 2 channels
    r = invoke("run", "--config", env["cfg"], "--input", d, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code == 3, r.output
    assert "c.tif" in r.output
    assert set(pd.read_parquet(env["out"] / "cells.parquet").field_id) == {"a_t0_z0", "b_t0_z0"}


def test_duplicate_field_id_across_files_is_a_failure_not_a_crash(env):
    d = make_batch(env, good=("a",))
    write_tif(d / "a.tiff")                           # same stem -> same field_id as a.tif
    r = invoke("run", "--config", env["cfg"], "--input", d, "--labels", env["labels"],
               "--out", env["out"])
    assert "duplicate field_id" in r.output and "a.tiff" in r.output
    assert r.exit_code == 3 and "Traceback" not in r.output
    assert set(pd.read_parquet(env["out"] / "cells.parquet").field_id) == {"a_t0_z0"}


def test_segment_partial_failure_exit_code(env, monkeypatch):
    d = make_batch(env)
    calls = []

    def flaky(field, params):
        calls.append(field.field_id)
        if field.field_id.startswith("b"):
            raise SegmentationError("model blew up")
        return Segmentation(env["seg"].cells, env["seg"].nuclei, {"model": "stub"})
    monkeypatch.setattr(cli, "segment_field", flaky)
    r = invoke("segment", "--config", env["cfg"], "--input", d, "--out", env["out"])
    assert r.exit_code == 3 and "model blew up" in r.output
    assert (env["out"] / "labels" / "a_t0_z0.JM1.npy").exists()
    assert not (env["out"] / "labels" / "b_t0_z0.JM1.npy").exists()
