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
from junctionmapper.errors import ConfigError, SegmentationError
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


def test_default_run_proposes_one_threshold_carried_across_every_field(tmp_path):
    """C2: with no threshold in the config the proposal is ONE value per marker role for the
    whole run, so the report's red "threshold varied / not valid" banner does not fire on the
    default path. Per-field Otsu returned 40.234375 and 40.341797 on two near-identical fields
    and lit that banner, and marked most panels not comparable, on every default run."""
    cfg = write_cfg(tmp_path / "cfg.yaml", threshold=...)        # no threshold key at all
    d = tmp_path / "in"
    d.mkdir()
    labels = tmp_path / "labs"
    for name, shift in (("a", 0.0), ("b", 0.3)):
        f, seg = build()
        planes = np.stack([f.planes[r] for r in list(f.planes)]).astype(np.float32)
        planes[:2] += shift                                      # near-identical, not identical
        tifffile.imwrite(d / f"{name}.tif", planes, photometric="minisblack")
        save_labels(labels, f"{name}_t0_z0", seg)
    out = tmp_path / "out"
    r = invoke("run", "--config", cfg, "--input", d, "--labels", labels, "--out", out)
    assert r.exit_code == 0, r.output
    ifc = pd.read_parquet(out / "interfaces.parquet")
    assert set(ifc.field_id) == {"a_t0_z0", "b_t0_z0"}
    for role in ("JM1", "JM2"):
        g = ifc[ifc.role == role]
        assert g.threshold.nunique() == 1, role                  # one applied value for the run
        assert g.threshold_field_otsu.nunique() == 2, role       # each field's own value kept
    html = (out / "report.html").read_text()
    assert 'id="threshold-warning"' not in html
    assert "THRESHOLD VARIED" not in html
    assert "median of Otsu" in html                              # the proposal is on the page


def test_one_unsupported_file_fails_its_field_and_keeps_the_good_field(tmp_path):
    """I4: UnsupportedFileFormatError was neither a JunctionMapperError nor in _iter_fields's
    except tuple, so a directory holding one vendor file printed a traceback, exited 1 and wrote
    nothing - losing the readable field's tables and report."""
    cfg = write_cfg(tmp_path / "cfg.yaml")
    d = tmp_path / "in"
    d.mkdir()
    seg = write_tif(d / "f1.tif")
    (d / "x.czi").write_bytes(b"not an image")
    labels = tmp_path / "labs"
    save_labels(labels, "f1_t0_z0", seg)
    out = tmp_path / "out"
    r = invoke("run", "--config", cfg, "--input", d, "--labels", labels, "--out", out)
    assert r.exit_code == 3, r.output                       # PARTIAL, not 1
    assert "Traceback" not in r.output and "x.czi" in r.output
    assert "out of scope" in r.output
    assert "1 succeeded, 1 failed" in r.output
    # the good field's outputs survived
    assert set(pd.read_parquet(out / "cells.parquet").field_id) == {"f1_t0_z0"}
    html = (out / "report.html").read_text()
    assert (out / "report.html").exists() and "x.czi" in html


def test_a_tif_no_reader_can_open_is_a_per_field_failure_not_a_crash(tmp_path):
    cfg = write_cfg(tmp_path / "cfg.yaml")
    d = tmp_path / "in"
    d.mkdir()
    seg = write_tif(d / "f1.tif")
    (d / "broken.tif").write_bytes(b"II*\x00 not really a tiff")
    labels = tmp_path / "labs"
    save_labels(labels, "f1_t0_z0", seg)
    out = tmp_path / "out"
    r = invoke("run", "--config", cfg, "--input", d, "--labels", labels, "--out", out)
    assert r.exit_code == 3, r.output
    assert "Traceback" not in r.output and "broken.tif" in r.output
    assert (out / "report.html").exists()


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


# ---- fix round 1 --------------------------------------------------------------------
import os
import tomllib
from pathlib import Path

import junctionmapper


def counting_segment(monkeypatch, seg):
    calls = []

    def fake(field, params):
        calls.append(field.field_id)
        return Segmentation(seg.cells.copy(), seg.nuclei.copy(), {"model": "stub"})
    monkeypatch.setattr(cli, "segment_field", fake)
    return calls


def corrected_labels(env):
    """Hand-corrected labels already sitting in OUT/labels."""
    d = env["out"] / "labels"
    cells = np.zeros((40, 40), np.int32)
    cells[5:9, 5:9] = 42
    save_labels(d, FID, Segmentation(cells, np.zeros_like(cells), {}))
    return d, {p.name: p.read_bytes() for p in d.iterdir()}


@pytest.mark.parametrize("cmd", ["run", "segment"])
def test_existing_labels_are_never_silently_overwritten(env, monkeypatch, cmd):
    calls = counting_segment(monkeypatch, env["seg"])
    d, before = corrected_labels(env)
    r = invoke(cmd, "--config", env["cfg"], "--input", env["tif"], "--out", env["out"])
    assert r.exit_code == 1, r.output
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before      # nothing modified
    assert calls == []                                                   # refused BEFORE segmenting
    assert str(d) in r.output and f"{FID}.JM1.npy" in r.output
    assert f"--labels {d}" in r.output and "--overwrite-labels" in r.output
    assert "Traceback" not in r.output
    assert not (env["out"] / "report.html").exists()


@pytest.mark.parametrize("cmd", ["run", "segment"])
def test_overwrite_labels_flag_replaces_them(env, monkeypatch, cmd):
    counting_segment(monkeypatch, env["seg"])
    d, before = corrected_labels(env)
    r = invoke(cmd, "--config", env["cfg"], "--input", env["tif"], "--out", env["out"],
               "--overwrite-labels")
    assert r.exit_code == 0, r.output
    assert np.array_equal(np.load(d / f"{FID}.JM1.npy"), env["seg"].cells)


def test_partially_populated_labels_dir_refuses_rather_than_half_writing(env, monkeypatch):
    calls = counting_segment(monkeypatch, env["seg"])
    batch = make_batch(env, good=())
    save_labels(env["out"] / "labels", "a_t0_z0", env["seg"])           # a exists, b does not
    r = invoke("segment", "--config", env["cfg"], "--input", batch, "--out", env["out"])
    assert r.exit_code == 1 and "a_t0_z0" in r.output
    assert calls == []
    assert not (env["out"] / "labels" / "b_t0_z0.JM1.npy").exists()


def test_run_with_labels_never_writes_labels_even_into_out_labels(env, monkeypatch):
    forbid_cellpose(monkeypatch)
    before = {p.name: p.read_bytes() for p in env["labels"].iterdir()}
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--labels", env["labels"],
               "--out", env["tmp"])
    assert r.exit_code == 0
    assert {p.name: p.read_bytes() for p in env["labels"].iterdir()} == before


# ---- output directory -----------------------------------------------------------------

@pytest.mark.parametrize("cmd", ["run", "segment"])
def test_out_is_an_existing_file(env, cmd):
    f = env["tmp"] / "afile"
    f.write_text("x")
    r = invoke(cmd, "--config", env["cfg"], "--input", env["tif"], "--labels", env["labels"],
               "--out", f) if cmd == "run" else \
        invoke(cmd, "--config", env["cfg"], "--input", env["tif"], "--out", f)
    assert r.exit_code == 1 and str(f) in r.output and "not a directory" in r.output
    assert "Traceback" not in r.output and isinstance(r.exception, SystemExit)


def test_out_directory_not_writable(env):
    ro = env["tmp"] / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("permissions are not enforceable for this user")
        r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--labels", env["labels"],
                   "--out", ro / "sub")
        assert r.exit_code == 1 and str(ro) in r.output and "Traceback" not in r.output
        r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--labels", env["labels"],
                   "--out", ro)
        assert r.exit_code == 1 and "not writable" in r.output
    finally:
        ro.chmod(0o700)


# ---- version --------------------------------------------------------------------------

def test_dunder_version_matches_pyproject():
    py = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert junctionmapper.__version__ == py["project"]["version"]


def test_version_resolves_and_reaches_report_without_installed_metadata(env, monkeypatch):
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(cli.metadata, "version", missing)       # the container: nothing installed
    assert cli._version() == junctionmapper.__version__ != "unknown"
    assert run_ok(env).exit_code == 0
    html = (env["out"] / "report.html").read_text()
    assert junctionmapper.__version__ in html
    # the version row itself must not read "unknown". (The MODEL may legitimately be unknown
    # here: these labels carry no provenance sidecar, and the report now says so per field.)
    import re
    row = re.search(r'<tr><td class="l">junctionmapper version</td><td class="l">([^<]*)</td>', html)
    assert row and row.group(1) == junctionmapper.__version__
    assert "code_identity" in html                   # and the code is identified regardless


# ---- napari-seam properties -----------------------------------------------------------

def distinct_labels(offset):
    cells = np.zeros((40, 40), np.int32)
    cells[10:18, 10:18] = offset + 1
    cells[10:18, 18:26] = offset + 2
    cells[18:26, 10:18] = offset + 3
    cells[18:26, 18:26] = offset + 4
    nuc = np.zeros_like(cells)
    nuc[12:15, 12:15] = offset + 1
    return Segmentation(cells, nuc, {})


def test_each_field_is_measured_with_its_own_label_file(env):
    d = env["tmp"] / "batch"
    d.mkdir()
    for n in ("a", "b"):
        write_tif(d / f"{n}.tif")
    save_labels(env["labels"], "a_t0_z0", distinct_labels(0))
    save_labels(env["labels"], "b_t0_z0", distinct_labels(100))
    r = invoke("run", "--config", env["cfg"], "--input", d, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code == 0, r.output
    cells = pd.read_parquet(env["out"] / "cells.parquet")
    assert set(cells[cells.field_id == "a_t0_z0"].cell_id) == {1, 2, 3, 4}
    assert set(cells[cells.field_id == "b_t0_z0"].cell_id) == {101, 102, 103, 104}
    nuc = pd.read_parquet(env["out"] / "nuclei.parquet")
    assert set(nuc[nuc.field_id == "b_t0_z0"].nucleus_id) == {101}


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int64, np.uint64])
def test_label_dtype_reaches_measure_field_unchanged(env, monkeypatch, dtype):
    seen = []
    real = cli.measure_field

    def spy(field, seg, cfg, *a, **k):
        seen.append((seg.cells.dtype, seg.nuclei.dtype))
        return real(field, seg, cfg, *a, **k)
    monkeypatch.setattr(cli, "measure_field", spy)
    save_labels(env["labels"], FID, Segmentation(env["seg"].cells.astype(dtype),
                                                 env["seg"].nuclei.astype(dtype), {}))
    assert run_ok(env).exit_code == 0
    assert seen == [(np.dtype(dtype), np.dtype(dtype))]


def test_bad_file_sorted_first_does_not_abort_the_batch(env):
    d = make_batch(env)
    write_tif(d / "0bad.tif", planes=np.zeros((2, 40, 40), np.float32))   # sorts before a, b
    r = invoke("run", "--config", env["cfg"], "--input", d, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code == 3, r.output
    assert "0bad.tif" in r.output
    assert set(pd.read_parquet(env["out"] / "cells.parquet").field_id) == {"a_t0_z0", "b_t0_z0"}


def test_report_provenance_carries_thresholds_version_and_failures(env):
    d = make_batch(env, good=("a",))                      # b fails
    r = invoke("run", "--config", env["cfg"], "--input", d, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code == 3
    html = (env["out"] / "report.html").read_text()
    assert junctionmapper.__version__ in html
    assert "fields_failed" in html
    assert "a_t0_z0&#34;: {&#34;JM1&#34;: 50.0" in html          # per-field thresholds record
    assert "labels_dir" in html


# ---- I3: a figure must name the code and the segmentation behind it -------------------

def test_labels_run_report_names_the_model_parameters_device_and_code(env, monkeypatch):
    """I3: on the supported --labels path the report said only "labels loaded from disk" and
    stamped a constant 0.1.0.dev0, so two runs a month apart were indistinguishable. The
    sidecar's params_used, the code identity and the library versions must all be on the page."""
    def fake(field, params):
        return Segmentation(cells=env["seg"].cells.copy(), nuclei=env["seg"].nuclei.copy(),
                            params_used={"model": params.model, "diameter": 37.5,
                                         "flow_threshold": params.flow_threshold,
                                         "device": "cuda:0", "cellpose_version": "4.0.1"})
    monkeypatch.setattr(cli, "segment_field", fake)
    seg_out = env["tmp"] / "seg"
    assert invoke("segment", "--config", env["cfg"], "--input", env["tif"], "--out", seg_out,
                  "--flow-threshold", "0.55").exit_code == 0
    side = json.loads((seg_out / "labels" / f"{FID}.provenance.json").read_text())
    assert side["code_identity"] and side["created"] and side["library_versions"]

    forbid_cellpose(monkeypatch)
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"],
               "--labels", seg_out / "labels", "--out", env["out"])
    assert r.exit_code == 0, r.output
    html = (env["out"] / "report.html").read_text()
    for s in ("cpsam", "0.55", "cuda:0", "37.5", "4.0.1",      # the segmentation behind it
              "labels_created", side["created"],                # when the labels were made
              "code_identity", "library_versions", "numpy"):    # and the code behind the figure
        assert s in html, s
    import junctionmapper.provenance as pv
    from tests.test_provenance import RESOLVED_PREFIXES
    ident = pv.code_identity()
    assert ident.split(" (")[0] in html                         # git sha, or installed version
    assert "0.1.0.dev0" != ident
    # and what reached the page is a RESOLVED identity: "unresolvable: ... no installed
    # distribution ..." also satisfies the substring check above.
    assert ident.startswith(RESOLVED_PREFIXES), ident
    assert "unresolvable" not in html


def test_labels_without_a_sidecar_are_reported_per_field_not_globally(env):
    r = run_ok(env)                                  # env's labels are written without a sidecar
    assert r.exit_code == 0, r.output
    html = (env["out"] / "report.html").read_text()
    assert "sidecar" in html and "absent" in html
    assert FID in html
    assert "labels_without_provenance" in html
    # and the code identity is still recorded, because that does not depend on the labels
    assert "code_identity" in html and "library_versions" in html


# ---- sidecar --------------------------------------------------------------------------

def _write_sidecar(env, source):
    side = env["labels"] / f"{FID}.provenance.json"
    side.write_text(json.dumps({"source": source, "field_id": FID}))


def test_sidecar_source_mismatch_warns_and_is_recorded_but_does_not_refuse(env):
    _write_sidecar(env, "/elsewhere/other_experiment.tif")
    r = run_ok(env)
    assert r.exit_code == 0, r.output
    assert "WARNING" in r.stderr and "/elsewhere/other_experiment.tif" in r.stderr
    assert str(env["tif"].resolve()) in r.stderr
    html = (env["out"] / "report.html").read_text()
    assert "labels_source_mismatch" in html and "other_experiment.tif" in html


def test_sidecar_matching_source_is_silent(env, monkeypatch):
    _write_sidecar(env, str(env["tif"].resolve()))
    r = run_ok(env)
    assert r.exit_code == 0 and "WARNING" not in r.stderr
    assert "labels_source_mismatch" not in (env["out"] / "report.html").read_text()


def test_missing_sidecar_is_not_an_error(env):
    assert not list(env["labels"].glob("*.json"))
    r = run_ok(env)
    assert r.exit_code == 0 and "WARNING" not in r.stderr


def test_segment_then_run_produces_no_mismatch_warning(env, monkeypatch):
    stub_segment(monkeypatch, env["seg"])
    assert invoke("segment", "--config", env["cfg"], "--input", env["tif"],
                  "--out", env["out"]).exit_code == 0
    side = json.loads((env["out"] / "labels" / f"{FID}.provenance.json").read_text())
    assert side["source"] == str(env["tif"].resolve())
    r = invoke("run", "--config", env["cfg"], "--input", env["tif"], "--labels",
               env["out"] / "labels", "--out", env["tmp"] / "o2")
    assert r.exit_code == 0 and "WARNING" not in r.stderr


# ---- minor items ----------------------------------------------------------------------

def test_negative_label_ids_rejected(env):
    bad = env["seg"].cells.copy()
    bad[0, 0] = -3
    np.save(env["labels"] / f"{FID}.JM1.npy", bad)
    r = run_ok(env)
    assert r.exit_code != 0 and "negative" in r.output and f"{FID}.JM1.npy" in r.output


def test_summary_goes_to_stderr(env):
    r = run_ok(env)
    assert "1 succeeded, 0 failed" in r.stderr
    assert "succeeded" not in r.stdout


@pytest.mark.parametrize("cmd", ["run", "segment"])
def test_help_documents_partial_exit_code(cmd):
    r = invoke(cmd, "--help")
    assert r.exit_code == 0 and "PARTIAL" in r.output and "3" in r.output


def test_labels_dir_recorded_as_absolute_path(env, monkeypatch):
    monkeypatch.chdir(env["tmp"])
    r = invoke("run", "--config", "cfg.yaml", "--input", "img.tif", "--labels", "labs",
               "--out", "out")
    assert r.exit_code == 0, r.output
    html = (env["out"] / "report.html").read_text()
    assert str(env["labels"].resolve()) in html


def test_plain_tiff_read_does_not_emit_bioio_failure_warning(env, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        assert run_ok(env).exit_code == 0
    assert not [rec for rec in caplog.records if "Attempted file" in rec.getMessage()]


@pytest.mark.parametrize("which", ["nucleus", "sidecar"])
def test_collision_on_any_one_label_file_refuses(env, monkeypatch, which):
    calls = counting_segment(monkeypatch, env["seg"])
    d = env["out"] / "labels"
    d.mkdir(parents=True)
    name = {"nucleus": f"{FID}.nucleus.npy", "sidecar": f"{FID}.provenance.json"}[which]
    (d / name).write_bytes(b"precious")
    r = invoke("segment", "--config", env["cfg"], "--input", env["tif"], "--out", env["out"])
    assert r.exit_code == 1 and name in r.output
    assert (d / name).read_bytes() == b"precious" and calls == []


# ---- fix round 2 ---------------------------------------------------------------------

@pytest.mark.parametrize("cmd", ["segment", "run"])
@pytest.mark.parametrize("field", ["a", "b", "c"])
@pytest.mark.parametrize("suffix", ["JM1.npy", "nucleus.npy", "provenance.json"])
def test_collision_anywhere_in_a_three_field_batch_refuses_and_writes_nothing(
        env, monkeypatch, cmd, field, suffix):
    """The napari case: one hand-corrected later field in a batch. Every (field, file)
    position must be caught, with nothing written for any field."""
    calls = counting_segment(monkeypatch, env["seg"])
    batch = make_batch(env, good=())
    write_tif(batch / "c.tif")
    d = env["out"] / "labels"
    d.mkdir(parents=True)
    victim = d / f"{field}_t0_z0.{suffix}"
    victim.write_bytes(b"hand-corrected")
    r = invoke(cmd, "--config", env["cfg"], "--input", batch, "--out", env["out"])
    assert r.exit_code == 1, r.output
    assert victim.name in r.output and "--overwrite-labels" in r.output
    assert victim.read_bytes() == b"hand-corrected"
    assert [p.name for p in d.iterdir()] == [victim.name]     # nothing else appeared
    assert calls == []
    assert not (env["out"] / "report.html").exists()


# ---- bioio logger scoping --------------------------------------------------------------

def _spy_reader(monkeypatch, seen, exc=None):
    import logging

    def fake(path, cfg):
        seen.append(logging.getLogger("bioio.bio_image").level)
        if exc:
            raise exc
        return
        yield
    monkeypatch.setattr(cli, "read_fields", fake)


@pytest.mark.parametrize("name,quiet", [("x.tif", True), ("x.tiff", True), ("x.zarr", False)])
def test_bioio_logger_level_restored_after_read(env, monkeypatch, name, quiet):
    import logging
    log = logging.getLogger("bioio.bio_image")
    log.setLevel(logging.INFO)
    try:
        seen = []
        _spy_reader(monkeypatch, seen)
        cfg, _ = cli._load_config(env["cfg"])
        assert list(cli._iter_fields([env["tmp"] / name], cfg, [])) == []
        assert seen == [logging.ERROR if quiet else logging.INFO]   # only TIFF reads are quieted
        assert log.level == logging.INFO                             # and always restored
    finally:
        log.setLevel(logging.NOTSET)


def test_bioio_logger_level_restored_after_a_read_that_raises(env, monkeypatch):
    import logging
    log = logging.getLogger("bioio.bio_image")
    log.setLevel(logging.INFO)
    try:
        _spy_reader(monkeypatch, [], exc=RuntimeError("boom"))
        cfg, _ = cli._load_config(env["cfg"])
        with pytest.raises(RuntimeError):
            list(cli._iter_fields([env["tmp"] / "x.tif"], cfg, []))
        assert log.level == logging.INFO
    finally:
        log.setLevel(logging.NOTSET)


# ---- unreadable sidecar ----------------------------------------------------------------

def test_unreadable_sidecar_warns_and_the_run_proceeds(env):
    (env["labels"] / f"{FID}.provenance.json").write_bytes(b"\x00not json{{")
    r = run_ok(env)
    assert r.exit_code == 0, r.output
    assert "unreadable" in r.stderr and f"{FID}.provenance.json" in r.stderr
    assert set(pd.read_parquet(env["out"] / "cells.parquet").field_id) == {FID}
    assert "labels_source_mismatch" not in (env["out"] / "report.html").read_text()


# ---- N1: an OME-Zarr store is a DIRECTORY, so the single entry point must accept one ---

def write_zarr(path, planes=None):
    """A minimal OME-Zarr store (a directory) holding the fixture's three channels."""
    import zarr
    f, seg = build()
    if planes is None:
        planes = np.stack([f.planes[r] for r in list(f.planes)]).astype(np.float32)
    group = zarr.open_group(path, mode="w", zarr_format=2)
    group.create_array("0", data=planes, chunks=(1, *planes.shape[1:]))
    axes = [{"name": "c", "type": "channel"}] + [
        {"name": n, "type": "space", "unit": "micrometer"} for n in "yx"]
    group.attrs["multiscales"] = [
        {"version": "0.4", "name": path.name, "axes": axes,
         "datasets": [{"path": "0", "coordinateTransformations": [
             {"type": "scale", "scale": [1, 0.5, 0.5]}]}]}]
    group.attrs["omero"] = {"channels": [{"label": n} for n in ("Na/K-ATPase", "p120", "DAPI")]}
    return seg


def test_a_zarr_store_is_a_leaf_input_not_a_directory_to_scan(tmp_path):
    """N1: README and the design claim OME-Zarr, the reader supports it, but _input_files
    filtered on is_file() and scanned INSIDE the store, so no invocation could reach it."""
    store = tmp_path / "img.ome.zarr"
    write_zarr(store)
    assert store.is_dir() and store.suffix == ".zarr"
    assert cli._input_files(store) == [store]


def test_a_directory_of_stores_and_tifs_is_scanned_in_sorted_order(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_zarr(d / "b.ome.zarr")
    write_zarr(d / "d.ome.zarr")
    write_tif(d / "a.tif")
    write_tif(d / "c.tif")
    (d / "notes.txt").write_text("not an image")
    (d / "sub").mkdir()                             # a plain directory is not an input
    assert [p.name for p in cli._input_files(d)] == ["a.tif", "b.ome.zarr", "c.tif", "d.ome.zarr"]


def test_a_directory_holding_only_stores_is_accepted(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    write_zarr(d / "s2.ome.zarr")
    write_zarr(d / "s1.ome.zarr")
    assert [p.name for p in cli._input_files(d)] == ["s1.ome.zarr", "s2.ome.zarr"]


def test_a_directory_with_neither_stores_nor_images_still_raises(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "notes.txt").write_text("x")
    (d / "sub").mkdir()
    with pytest.raises(ConfigError) as exc:
        cli._input_files(d)
    assert "no image files" in str(exc.value) and ".zarr" in str(exc.value)


def test_a_vendor_suffix_is_still_admitted_so_it_can_be_refused_by_name(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "x.czi").write_bytes(b"not an image")
    assert [p.name for p in cli._input_files(d)] == ["x.czi"]
    assert cli._input_files(d / "x.czi") == [d / "x.czi"]


def test_a_zarr_path_that_is_a_file_is_a_clean_failure_not_a_crash(env):
    bogus = env["tmp"] / "f.ome.zarr"
    bogus.write_bytes(b"not a zarr store")
    assert cli._input_files(bogus) == [bogus]
    r = invoke("run", "--config", env["cfg"], "--input", bogus, "--labels", env["labels"],
               "--out", env["out"])
    assert r.exit_code != 0 and "Traceback" not in r.output
    assert "f.ome.zarr" in r.output


def test_a_missing_zarr_store_is_reported_as_a_missing_input(tmp_path):
    with pytest.raises(ConfigError) as exc:
        cli._input_files(tmp_path / "gone.ome.zarr")
    assert "input path not found" in str(exc.value)


def test_run_reads_an_ome_zarr_store_end_to_end(tmp_path, monkeypatch):
    """The claim demonstrated, not asserted: the real reader, on a store written by zarr,
    driven from the CLI."""
    cfg = write_cfg(tmp_path / "cfg.yaml")
    store = tmp_path / "img.ome.zarr"
    seg = write_zarr(store)
    stub_segment(monkeypatch, seg)
    out = tmp_path / "out"
    r = invoke("run", "--config", cfg, "--input", store, "--out", out)
    assert r.exit_code == 0, r.output
    cells = pd.read_parquet(out / "cells.parquet")
    assert len(cells) and all(str(f).startswith("img.ome") for f in cells.field_id)
    assert len(pd.read_parquet(out / "interfaces.parquet")) > 0
    assert (out / "report.html").exists()


def test_run_reads_a_directory_of_zarr_stores_end_to_end(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path / "cfg.yaml")
    d = tmp_path / "in"
    d.mkdir()
    seg = write_zarr(d / "s1.ome.zarr")
    write_zarr(d / "s2.ome.zarr")
    stub_segment(monkeypatch, seg)
    out = tmp_path / "out"
    r = invoke("run", "--config", cfg, "--input", d, "--out", out)
    assert r.exit_code == 0, r.output
    fids = set(pd.read_parquet(out / "cells.parquet").field_id)
    assert len(fids) == 2, fids
    assert all(str(f).startswith(("s1.ome", "s2.ome")) for f in fids)


# ---- N8: WHICH files the calibration pass opens ---------------------------------------

def test_calibration_files_are_spread_over_the_sorted_inputs_not_the_first_five(tmp_path):
    """N8: nothing tested _calibration_files, so `return list(files[:n])` passed the whole
    suite. With one field per file the library's field-id rule gives the same answer, which
    is exactly why a CLI-side regression would be invisible; this pins the FILE selection."""
    files = [tmp_path / f"f{i:02d}.tif" for i in range(20)]
    picked = cli._calibration_files(files)
    assert [p.name for p in picked] == ["f00.tif", "f05.tif", "f10.tif", "f14.tif", "f19.tif"]
    assert picked != files[:cli.CALIBRATION_FIELDS]          # not the first five
    assert picked == sorted(picked)                          # and in input order
    assert picked[0] == files[0] and picked[-1] == files[-1]  # spanning the run


@pytest.mark.parametrize("n", [1, 2, 5])
def test_five_or_fewer_inputs_are_all_calibration_files(tmp_path, n):
    files = [tmp_path / f"f{i}.tif" for i in range(n)]
    assert cli._calibration_files(files) == files


def test_calibration_file_count_is_bounded_at_five(tmp_path):
    for total in (6, 13, 20, 300):
        files = [tmp_path / f"f{i:03d}.tif" for i in range(total)]
        picked = cli._calibration_files(files)
        assert len(picked) == cli.CALIBRATION_FIELDS == 5, total
        assert len(set(picked)) == len(picked), total


# ---- N3: a failed calibration must try other files, and must not blame a human --------

def _spread_corrupt_batch(tmp_path, total=13, corrupt=(0, 3, 6, 9, 12)):
    """``total`` inputs of which the files the spread picks are unreadable."""
    d = tmp_path / "in"
    d.mkdir()
    for i in range(total):
        p = d / f"f{i:02d}.tif"
        if i in corrupt:
            p.write_bytes(b"II*\x00 not really a tiff")
        else:
            write_tif(p)
    return d


def test_calibration_walks_past_unreadable_files_to_the_next_candidates(tmp_path):
    """N3: the five chosen files failing left propose_run_thresholds with nothing, so every
    field silently fell back to its own Otsu. The spread order is now walked on."""
    from junctionmapper.config import Role
    cfg, _ = cli._load_config(write_cfg(tmp_path / "cfg.yaml", threshold=...))
    d = _spread_corrupt_batch(tmp_path)
    files = cli._input_files(d)
    assert [p.name for p in cli._calibration_files(files)] == [
        "f00.tif", "f03.tif", "f06.tif", "f09.tif", "f12.tif"]       # all five corrupt
    run_t, note = cli._calibrate(files, cfg)
    assert len(run_t.calibration_fields) == cli.CALIBRATION_FIELDS
    assert run_t.source[Role.JM1] == "calibration" and run_t.applied[Role.JM1] > 0
    assert note and "could not be read" in note and "f00.tif" in note


def test_calibration_attempts_are_bounded_when_every_file_is_unreadable(tmp_path, monkeypatch):
    cfg, _ = cli._load_config(write_cfg(tmp_path / "cfg.yaml", threshold=...))
    d = _spread_corrupt_batch(tmp_path, total=13, corrupt=tuple(range(13)))
    files = cli._input_files(d)
    opened = []
    real = cli._iter_fields

    def counting(fs, c, fl):
        opened.append([p.name for p in fs])
        return real(fs, c, fl)
    monkeypatch.setattr(cli, "_iter_fields", counting)
    run_t, note = cli._calibrate(files, cfg)
    assert len(opened) == cli._CALIBRATION_ATTEMPTS == 10        # not all 13, not 13 twice
    assert not run_t.applied and run_t.calibration_fields == ()
    assert note and note.startswith("degraded")


def test_degraded_note_counts_the_files_actually_opened_not_the_candidate_list(tmp_path, monkeypatch):
    """Thirteen inputs, only f00 corrupt: ten candidates are listed but the walk stops once
    five read cleanly, so six files are opened. The note used to say "of the 10"."""
    cfg, _ = cli._load_config(write_cfg(tmp_path / "cfg.yaml", threshold=...))
    d = _spread_corrupt_batch(tmp_path, total=13, corrupt=(0,))
    files = cli._input_files(d)
    opened = []
    real = cli._iter_fields

    def counting(fs, c, fl):
        opened.append([p.name for p in fs])
        return real(fs, c, fl)
    monkeypatch.setattr(cli, "_iter_fields", counting)
    _, note = cli._calibrate(files, cfg)
    assert len(opened) == 6                                      # 1 unreadable + 5 read
    assert note and "1 of the 6 calibration file(s) opened" in note, note
    assert "of the 10" not in note
    assert "5 file(s) were read for it" in note


def test_a_run_whose_calibration_resolved_nothing_does_not_blame_a_human(tmp_path):
    """N3, the load-bearing half: when every calibration candidate fails, no run value is
    resolved, every field falls back to Otsu on its own plane, the applied values then differ
    across fields -- and the red banner used to say the threshold "was overridden somewhere",
    which is false and contradicts the Provenance section's "no marker role resolved".

    Reachable with more inputs than the calibration pass may open: the first ten candidates in
    the spread order are corrupt, so the proposal gets nothing, while the measurement pass
    still reads the three readable files.
    """
    cfg = write_cfg(tmp_path / "cfg.yaml", threshold=...)       # no threshold: propose one
    d = tmp_path / "in"
    d.mkdir()
    labels = tmp_path / "labs"
    candidates = (0, 1, 2, 3, 4, 5, 6, 7, 9, 12)                # the ten _calibrate may open
    for i in range(13):
        p = d / f"f{i:02d}.tif"
        if i in candidates:
            p.write_bytes(b"II*\x00 not really a tiff")
            continue
        f, seg = build()
        planes = np.stack([f.planes[r] for r in list(f.planes)]).astype(np.float32)
        planes[:2] += 30.0 * i                                  # so the own-Otsu values differ
        tifffile.imwrite(p, planes, photometric="minisblack")
        save_labels(labels, f"f{i:02d}_t0_z0", seg)
    out = tmp_path / "out"
    r = invoke("run", "--config", cfg, "--input", d, "--labels", labels, "--out", out)
    assert r.exit_code == 3, r.output                           # the corrupt files failed
    ifc = pd.read_parquet(out / "interfaces.parquet")
    assert ifc[ifc.role == "JM1"].threshold.nunique() == 3      # per-field fallback, not one T
    html = (out / "report.html").read_text()
    assert "threshold_calibration" in html and "degraded" in html
    assert "no marker role resolved" in html
    w = html[html.index('id="threshold-warning"'):]
    w = w[:w.index("</section>")]
    assert 'data-attribution="fallback"' in w, w
    assert "Not an override" in w
    assert "overridden per field" not in w
