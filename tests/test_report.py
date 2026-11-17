"""The report's run-level facts (summary.py), computed from real ``measure_field`` output:
QC accounting, threshold comparability, run warnings and the provenance table."""
import numpy as np
import pandas as pd
import pytest

from junctionmapper.config import Role
from junctionmapper.pipeline import FieldResult, measure_field
from junctionmapper.report.summary import (PARAMETERS, provenance_rows, qc, qc_counts, run_roles,
                                           run_warnings, threshold_info)
from tests.fixtures13 import T1, T2, build, make_cfg, zoo

PROV = {
    "junctionmapper_version": "9.9.9-test",
    "params_used": {"model": "cpsam", "diameter": None, "flow_threshold": 0.4,
                    "cellprob_threshold": 0.0, "device": "cpu", "cellpose_version": "4.0.1"},
}


# ---- fixtures ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return make_cfg()


@pytest.fixture(scope="module")
def results(cfg):
    fz, sz = zoo("zoo")
    f1, s1 = build("f1")
    return [measure_field(fz, sz, cfg), measure_field(f1, s1, cfg)]


@pytest.fixture(scope="module")
def results_varying_t(cfg):
    """f1 at the config thresholds (JM1=50, JM2=20); f2 re-measured with JM1=60 only."""
    f1, s1 = build("f1")
    f2, s2 = build("f2")
    cfg2 = make_cfg(threshold={Role.JM1: 60.0, Role.JM2: T2})
    return [measure_field(f1, s1, cfg), measure_field(f2, s2, cfg2)]


def _cat(results, name):
    return pd.concat([getattr(r, name) for r in results], ignore_index=True)


def _qc(results):
    fields = sorted({str(f) for r in results for f in r.cells.field_id})
    return qc(fields, *(_cat(results, n) for n in ("cells", "flags", "nuclei", "interfaces")))


def _expected_qc(results):
    """Independent recount from the tables."""
    out = dict.fromkeys(("cells_found", "measured", "with_interfaces", "contrib_dp", "contrib_tcj",
                         "not_contributing", "excl_degenerate", "excl_border", "excl_isolated",
                         "excl_filtered", "excl_other"), 0)
    for r in results:
        cells, flags, ifc = r.cells, r.flags.set_index("cell_id"), r.interfaces
        out["cells_found"] += len(cells)
        out["measured"] += int(cells.measured.sum())
        dp = set(ifc.loc[ifc.corner_method == "dp", "cell_id"])
        tcj = set(ifc.loc[ifc.corner_method == "tcj", "cell_id"])
        out["contrib_dp"] += len(dp)
        out["contrib_tcj"] += len(tcj)
        out["with_interfaces"] += len(dp | tcj)
        for cid in cells.cell_id:
            if cid in dp | tcj:
                continue
            f = flags.loc[cid]
            out["not_contributing"] += 1
            key = ("excl_degenerate" if f.degenerate else "excl_border" if f.border_touching else
                   "excl_isolated" if f.isolated else "excl_filtered" if f.all_contacts_filtered
                   else "excl_other")
            out[key] += 1
    return out


def _two_fields_differing_by(base, delta):
    other = base.interfaces.assign(field_id="f2")
    other["threshold"] = np.where(other["role"] == "JM1", T1 + delta, T2)
    tabs = {n: getattr(base, n).assign(field_id="f2") for n in
            ("cells", "nuclei", "flags", "nuclei_distances")}
    return [base, FieldResult(interfaces=other, **tabs)]


# ---- QC accounting -----------------------------------------------------------------

def test_qc_per_field_counts_match_tables_and_reasons_are_the_zoo_ones(results):
    per, _ = _qc(results)
    z = per["zoo"]
    assert {k: z[k] for k in _expected_qc(results[:1])} == _expected_qc(results[:1])
    # hand-checked against ZOO_NOTES: 7 cells; 3 untraceable; 6 touches the border; 7 isolated;
    # 4 has only a too-short contact (no interface at all); 5 keeps a dp interface but no tcj one
    assert z["cells_found"] == 7
    assert (z["excl_degenerate"], z["excl_border"], z["excl_isolated"],
            z["excl_filtered"], z["excl_other"]) == (1, 1, 1, 1, 0)
    assert z["multinucleate"] == 1 and z["orphan_nuclei"] == 1
    assert z["contrib_dp"] == 3 and z["contrib_tcj"] == 2
    f1 = per["f1"]
    assert f1["cells_found"] == 4 and f1["not_contributing"] == 0


def test_qc_exclusion_reasons_add_up_to_the_shortfall(results):
    for fid, r in _qc(results)[0].items():
        assert sum(r[k] for k in r if k.startswith("excl_")) == r["not_contributing"], fid
        assert r["cells_found"] == r["with_interfaces"] + r["not_contributing"], fid


def test_qc_counts_of_an_empty_field_are_zero(results):
    r = results[1]
    empty = qc_counts(*(getattr(r, n).iloc[0:0] for n in ("cells", "flags", "nuclei", "interfaces")))
    assert set(empty.values()) == {0}


def test_field_losing_most_cells_is_called_out(results, cfg):
    f, s = zoo("zoo")
    keep = {3, 6, 7}        # degenerate, border-touching, isolated: nothing contributes
    for lab in range(1, 8):
        if lab not in keep:
            s.cells[s.cells == lab] = 0
    s.nuclei[:] = 0
    lost = measure_field(f, s, cfg)
    assert lost.interfaces.empty
    _, warn = _qc([lost] + results[1:])
    assert [w["field"] for w in warn] == ["zoo"]
    assert (warn[0]["found"], warn[0]["usable"]) == (3, 0)
    for reason in ("1 degenerate", "1 border-touching", "1 isolated"):
        assert reason in warn[0]["why"], reason
    assert _qc(results[1:])[1] == []


# ---- thresholds --------------------------------------------------------------------

def test_varying_threshold_is_detected_per_role(results_varying_t):
    t = threshold_info(_cat(results_varying_t, "interfaces"), {"JM1", "JM2"})
    assert t["JM1"]["varied"] and not t["JM2"]["varied"]
    assert t["JM1"]["by_field"] == {"f1": [50.0], "f2": [60.0]}
    assert t["JM1"]["differing"] == ["f1", "f2"] and t["JM1"]["markers"] == ["Na/K-ATPase"]


def test_no_threshold_variation_when_constant(results):
    assert not any(t["varied"] for t in threshold_info(_cat(results, "interfaces")).values())


def test_threshold_variation_needs_more_than_a_grey_level(cfg):
    base = measure_field(*build("f1"), cfg)
    small = threshold_info(_cat(_two_fields_differing_by(base, 0.1), "interfaces"))
    big = threshold_info(_cat(_two_fields_differing_by(base, 10.0), "interfaces"))
    assert not small["JM1"]["varied"] and big["JM1"]["varied"]


@pytest.mark.parametrize("delta", [1.01, 1.5, 2.0, 2.5, 10.0])
def test_a_varied_threshold_always_names_the_fields_that_establish_the_span(cfg, delta):
    """N2: for a span in (1.0, 2.0] every field is within one grey level of the median; the
    fields named must still be the two that establish the span."""
    base = measure_field(*build("f1"), cfg)
    t = threshold_info(_cat(_two_fields_differing_by(base, delta), "interfaces"))
    assert t["JM1"]["differing"] == ["f1", "f2"], delta


@pytest.mark.parametrize("raw,want", [(None, None), ("none", set()), ("JM1, JM2", {"JM1", "JM2"})])
def test_run_roles(raw, want):
    assert run_roles({} if raw is None else {"threshold_run_roles": raw}) == want


def test_threshold_pairs_each_value_with_its_field():
    res = []
    for fid, t in (("fa", 40.0), ("fb", 50.0), ("fc", 60.0)):
        f, s = build(fid)
        res.append(measure_field(f, s, make_cfg(threshold={Role.JM1: t, Role.JM2: T2})))
    t = threshold_info(_cat(res, "interfaces"))
    assert t["JM1"]["by_field"] == {"fa": [40.0], "fb": [50.0], "fc": [60.0]}
    assert t["JM2"]["by_field"] == {f: [20.0] for f in ("fa", "fb", "fc")}


# ---- run warnings ------------------------------------------------------------------

@pytest.mark.parametrize("roles,attribution,phrase", [
    ("JM1, JM2", "override", "per-field override"),
    ("none", "fallback", "calibration resolved nothing"),
    (None, "unknown", "cause not recorded"),
])
def test_threshold_warning_names_the_right_cause(results_varying_t, roles, attribution, phrase):
    """N3: with no run value resolved each field falls back to its own Otsu; that is not an
    override and the warning must not say it is."""
    prov = {} if roles is None else {"threshold_run_roles": roles}
    tinfo = threshold_info(_cat(results_varying_t, "interfaces"), run_roles(prov))
    [w] = run_warnings(prov, tinfo, [])
    assert w["kind"] == "threshold_varied" and w["role"] == "JM1" and w["attribution"] == attribution
    assert w["text"] == f"JM1 threshold varied: {phrase}"
    assert "50 vs 60" in w["detail"] and "not comparable" in w["detail"]
    if attribution == "fallback":
        assert "Not an override" in w["detail"] and "set per field" not in w["detail"]
    if attribution == "override":
        assert "f1, f2" in w["detail"]


def test_no_warnings_for_a_clean_run(results):
    assert run_warnings(PROV, threshold_info(_cat(results, "interfaces"), {"JM1", "JM2"}), []) == []


def test_run_warnings_from_provenance():
    prov = {"fields_failed": "2", "fields_failed_detail": "a.tif: bad; b.tif: worse",
            "labels_without_provenance": "3 field(s) whose labels carry no provenance sidecar: x, y, z",
            "labels_source_mismatch": '[{"applied_to": "/p/b.tif", "field_id": "b", "labels_made_from": "/p/a.tif"}]'}
    lost = [{"field": "z", "found": 3, "usable": 0, "why": "1 degenerate"}]
    w = {x["kind"]: x for x in run_warnings(prov, {}, lost)}
    assert w["fields_failed"]["text"] == "2 fields failed" and "b.tif: worse" in w["fields_failed"]["detail"]
    assert w["labels_without_provenance"]["text"] == "3 fields: labels without provenance"
    assert w["labels_source_mismatch"]["text"] == "1 field: labels from another image"
    assert "/p/a.tif" in w["labels_source_mismatch"]["detail"]
    assert w["cells_lost"]["text"] == "1 field: most cells unmeasured"
    assert "z: 0 of 3 cells measured (1 degenerate)" == w["cells_lost"]["detail"]
    assert all("\n" not in x["text"] and len(x["text"]) <= 60 for x in w.values())


# ---- provenance --------------------------------------------------------------------

def _prov_rows(results, cfg, prov=PROV):
    cells = _cat(results, "cells")
    return provenance_rows(cfg, prov, cells, threshold_info(_cat(results, "interfaces")), 2)


def test_provenance_lists_markers_model_versions_and_parameters(results, cfg):
    rows = _prov_rows(results, cfg)
    kv = {(r["group"], r["key"]): r["value"] for r in rows}
    assert kv[("Markers", "JM1")].startswith("Na/K-ATPase") and ("Markers", "nucleus") in kv
    assert kv[("Model", "model")] == "cpsam" and kv[("Model", "flow_threshold")] == "0.4"
    assert kv[("Versions", "junctionmapper")] == "9.9.9-test"
    for k in ("epsilon (px)", "tcj_min_run", "neighbour_radius_px", "dilation_cycles", "pixel size (µm/px)"):
        assert ("Parameters", k) in kv, k
    assert kv[("Parameters", "pixel size (µm/px)")] == "0.5"
    assert kv[("Thresholds", "JM1 · f1")].startswith("50") and kv[("Thresholds", "JM2 · f1")].startswith("20")
    assert kv[("Thresholds", "JM1 applied")] == "50"
    assert all(set(r) == {"group", "key", "value"} and isinstance(r["value"], str) for r in rows)


def test_provenance_scrubs_urls(results, cfg):
    prov = {**PROV, "params_used": {**PROV["params_used"], "weights": "https://x.org/m.pt"}}
    vals = " ".join(r["value"] for r in _prov_rows(results, cfg, prov))
    assert "://" not in vals and "x.org/m.pt" in vals


def test_provenance_has_no_timestamp_unless_supplied(results, cfg):
    assert not any(r["key"] == "timestamp" for r in _prov_rows(results, cfg))
    rows = _prov_rows(results, cfg, {**PROV, "timestamp": "2026-01-02T03:04:05"})
    assert {"group": "Inputs", "key": "timestamp", "value": "2026-01-02T03:04:05"} in rows


def test_every_parameter_has_a_label_and_help():
    assert all(p["label"] and p["help"] for p in PARAMETERS)
