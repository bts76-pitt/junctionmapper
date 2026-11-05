"""Task 14: self-contained HTML report, built from real ``measure_field`` output."""
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from junctionmapper.config import ChannelSpec, Role, RunConfig
from junctionmapper.errors import ConfigError, JunctionMapperError
from junctionmapper.pipeline import FieldResult, measure_field
from junctionmapper.report.build import PARAMETERS, build_report
from tests.fixtures13 import PX, T1, T2, build, make_cfg, zoo

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


@pytest.fixture(scope="module")
def html(results, cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("rep") / "r.html"
    return build_report(results, cfg, PROV, out).read_text()


def _row_cells(html, row_id):
    m = re.search(rf'<tr id="{re.escape(row_id)}"[^>]*>(.*?)</tr>', html, re.S)
    assert m, f"row {row_id} missing"
    return {k: int(v) for k, v in re.findall(r'<td data-col="(\w+)">(\d+)</td>', m.group(1))}


def _dist_rows(html, scope="pooled"):
    rows = {}
    for m in re.finditer(r'<tr class="dist"([^>]*)>', html):
        a = dict(re.findall(r'data-(\w+)="([^"]*)"', m.group(1)))
        if a["scope"] == scope:
            rows[(a["param"], a["method"], a["role"], a["marker"])] = int(a["n"])
    return rows


def _expected_qc(results):
    cells = pd.concat([r.cells for r in results])
    flags = pd.concat([r.flags for r in results])
    nuc = pd.concat([r.nuclei for r in results])
    ifc = pd.concat([r.interfaces for r in results])
    return {
        "cells_found": len(cells),
        "measured": int(cells.measured.sum()),
        "border_touching": int(flags.border_touching.sum()),
        "isolated": int(flags.isolated.sum()),
        "degenerate": int(flags.degenerate.sum()),
        "corner_methods_disagree": int(flags.corner_methods_disagree.sum()),
        "all_contacts_filtered": int(flags.all_contacts_filtered.sum()),
        "multinucleate": int(flags.multinucleate.sum()),
        "anucleate": int(flags.anucleate.sum()),
        "orphan_nuclei": int(nuc.orphan.sum()),
        "with_interfaces": len(ifc[["field_id", "cell_id"]].drop_duplicates()),
    }


def _empty_like(res, field_id):
    return FieldResult(**{n: getattr(res, n).iloc[0:0].copy() for n in
                          ("interfaces", "cells", "nuclei", "flags", "nuclei_distances")})


# ---- brief tests -------------------------------------------------------------------

def test_report_is_a_single_self_contained_file(html):
    assert '<img src="data:image/png;base64,' in html
    assert "http://" not in html and "https://cdn" not in html
    # stronger: no URL of any kind, no external resource references
    assert "://" not in html
    assert "<link" not in html.lower() and "<script src" not in html.lower()
    assert "url(" not in html.lower() and "@import" not in html.lower()
    assert not re.search(r'<img[^>]+src="(?!data:image/png;base64,)', html)


def test_qc_section_reports_counts_and_exclusion_reasons(html):
    low = html.lower()
    for s in ("cells found", "border", "isolated", "degenerate", "corner methods disagree",
              "all contacts filtered", "multinucleate", "anucleate", "orphan"):
        assert s in low, s


def test_report_shows_distributions_not_only_means(html):
    assert "distribution" in html.lower()
    assert "q1" in html.lower() and "median" in html.lower() and "q3" in html.lower()
    # figures are images; no stray "mean of" headline
    assert html.count('<img src="data:image/png;base64,') >= 2 * len(PARAMETERS)


def test_provenance_section_records_markers_model_and_versions(html):
    for s in ("Na/K-ATPase", "p120", "cpsam", "junctionmapper", "9.9.9-test"):
        assert s in html


def test_varying_threshold_across_fields_is_called_out(results_varying_t, cfg, tmp_path):
    html = build_report(results_varying_t, cfg, PROV, tmp_path / "r.html").read_text()
    assert "threshold varied" in html.lower()


# ---- QC numbers --------------------------------------------------------------------

def test_qc_pooled_counts_match_tables(results, html):
    assert _row_cells(html, "qc-pooled") == _expected_qc(results)


def test_qc_per_field_counts_match_tables_and_reasons_are_the_zoo_ones(results, html):
    zoo_row = _row_cells(html, "qc-field-zoo")
    assert zoo_row == _expected_qc(results[:1])
    # hand-checked against ZOO_NOTES: 7 cells; 6 touches border; 3 is untraceable; 7 isolated
    assert zoo_row["cells_found"] == 7
    assert zoo_row["border_touching"] == 1
    assert zoo_row["degenerate"] == 1
    assert zoo_row["isolated"] >= 1
    assert zoo_row["multinucleate"] == 1
    assert zoo_row["orphan_nuclei"] == 1
    f1_row = _row_cells(html, "qc-field-f1")
    assert f1_row["cells_found"] == 4 and f1_row["border_touching"] == 0
    assert f1_row["isolated"] == 0 and f1_row["degenerate"] == 0
    # pooled is the sum of the fields, not a recount
    for k in zoo_row:
        assert _row_cells(html, "qc-pooled")[k] == zoo_row[k] + f1_row[k]


def test_field_losing_most_cells_is_called_out(results, cfg, tmp_path):
    f, s = zoo("zoo")
    s.cells[:] = 0
    s.cells[0:6, 0:6] = 1       # one cell, touching the border: everything excluded
    s.cells[0:6, 10:16] = 2
    s.nuclei[:] = 0
    lost = measure_field(f, s, cfg)
    html = build_report([lost] + results[1:], cfg, PROV, tmp_path / "r.html").read_text()
    m = re.search(r'<section id="qc-warnings".*?</section>', html, re.S)
    assert m and "zoo" in m.group(0)
    assert "f1" not in m.group(0)
    assert 'id="qc-warnings"' not in build_report(results[1:], cfg, PROV,
                                                  tmp_path / "ok.html").read_text()


# ---- quantification: never pooled across method or marker ---------------------------

def test_distributions_are_split_by_corner_method_role_and_marker(results, html):
    ifc = pd.concat([r.interfaces for r in results])
    rows = _dist_rows(html)
    for p in PARAMETERS:
        col = p["um"] if p.get("um") else p["px"]
        for (method, role, marker), g in ifc.groupby(["corner_method", "role", "marker"]):
            assert rows[(col, method, role, marker)] == int(g[col].notna().sum()), (col, method)
    assert {k[1] for k in rows} == {"dp", "tcj"}          # no combined group
    assert {(k[2], k[3]) for k in rows} == {("JM1", "Na/K-ATPase"), ("JM2", "p120")}
    assert not any("," in k[1] or "+" in k[1] or k[1] == "all" for k in rows)


def test_per_field_distributions_are_present_and_per_field(results, html):
    ifc = pd.concat([r.interfaces for r in results])
    seen = {a for a in re.findall(r'data-scope="(field:[^"]*)"', html)}
    assert seen == {"field:zoo", "field:f1"}
    n = 0
    for fid, g in ifc.groupby("field_id"):
        sel = g[(g.corner_method == "dp") & (g.role == "JM1")]
        key = f"interface_contour_um"
        match = [v for k, v in _dist_rows(html, f"field:{fid}").items()
                 if k == (key, "dp", "JM1", "Na/K-ATPase")]
        assert match == [len(sel)], fid
        n += 1
    assert n == 2


def test_um_columns_used_when_pixel_size_known(html):
    assert "interface_contour_um" in {k[0] for k in _dist_rows(html)}
    assert "interface_contour_px" not in {k[0] for k in _dist_rows(html)}
    assert "µm" in html


def test_markers_never_hardcoded_come_from_config(tmp_path):
    cfg = RunConfig(channels=[ChannelSpec(role=Role.JM1, channel=0, marker="Zonulin-X"),
                              ChannelSpec(role=Role.JM2, channel=1, marker="Q<&>9"),
                              ChannelSpec(role=Role.NUCLEUS, channel=2, marker="DAPI")],
                    pixel_size_um=PX, threshold={Role.JM1: T1, Role.JM2: T2})
    f, s = build("f1")
    res = measure_field(f, s, cfg)
    html = build_report([res], cfg, PROV, tmp_path / "r.html").read_text()
    assert "Zonulin-X" in html
    assert "Na/K-ATPase" not in html and "p120" not in html and "E-cadherin" not in html
    assert "Q&lt;&amp;&gt;9" in html and "Q<&>9" not in html       # escaped, not injected


def test_marker_names_follow_the_table_not_the_config(results, cfg, tmp_path):
    """Per-row marker column is authoritative for distribution labels."""
    renamed = []
    for r in results:
        d = r.interfaces.copy()
        d["marker"] = d["marker"].replace({"Na/K-ATPase": "TableMarkerA"})
        renamed.append(replace(r, interfaces=d))
    html = build_report(renamed, cfg, PROV, tmp_path / "r.html").read_text()
    assert {(k[2], k[3]) for k in _dist_rows(html)} == {("JM1", "TableMarkerA"), ("JM2", "p120")}


def test_figure_count_depends_on_parameters_not_fields(results, cfg, tmp_path):
    def n_img(res):
        h = build_report(res, cfg, PROV, tmp_path / "c.html").read_text()
        return h.count('<img src="data:image/png;base64,')
    f3, s3 = build("f3")
    one = n_img(results[1:])
    three = n_img(results + [measure_field(f3, s3, cfg)])
    assert one == three
    assert one == 1 + 2 * len(PARAMETERS) + 3


# ---- threshold comparability -------------------------------------------------------

def test_threshold_warning_names_fields_values_role_and_explains(results_varying_t, cfg, tmp_path):
    html = build_report(results_varying_t, cfg, PROV, tmp_path / "r.html").read_text()
    m = re.search(r'<section id="threshold-warning".*?</section>', html, re.S)
    assert m
    w = m.group(0)
    assert 'data-role="JM1"' in w and 'data-role="JM2"' not in w     # only JM1 varied
    assert "f1" in w and "f2" in w and "50" in w and "60" in w
    assert "not valid" in w.lower() and "absolute intensity" in w.lower()
    assert "Na/K-ATPase" in w
    # the cost is visible where the numbers are: threshold-dependent pooled panels are marked
    assert 'data-threshold-varied="1"' in html
    marked = re.findall(r'<tr class="dist"[^>]*data-threshold-varied="1"[^>]*>', html)
    assert marked and all('data-role="JM1"' in t for t in marked)
    assert not re.search(r'data-param="interface_contour_um"[^>]*data-threshold-varied="1"', html)


def test_no_threshold_warning_when_constant(results, cfg, tmp_path):
    html = build_report(results, cfg, PROV, tmp_path / "r.html").read_text()
    assert "threshold varied" not in html.lower()
    assert 'id="threshold-warning"' not in html


# ---- provenance --------------------------------------------------------------------

def test_provenance_lists_resolved_parameters(html, results):
    prov = html[html.index('id="provenance"'):]
    for s in ("JM1", "JM2", "nucleus", "DAPI", "tcj_min_run", "neighbour_radius_px",
              "dilation_cycles", "epsilon", "pixel size", "0.5", "flow_threshold", "0.4",
              "50", "20"):
        assert s in prov, s
    # resolved thresholds come from the table
    assert 'data-threshold-role="JM1"' in prov
    rows = re.findall(r'<tr data-threshold-role="(\w+)"><td class="l">(\w+)</td><td>([^<]*)</td>', prov)
    assert ("JM1", "f1", "50") in rows and ("JM2", "f1", "20") in rows
    assert "built-in" not in html


# ---- empty / degenerate inputs -----------------------------------------------------

def test_field_where_every_cell_is_excluded_renders(cfg, tmp_path):
    f, s = build("f1")
    s.cells[:] = 0
    s.cells[0:6, 0:6] = 1
    s.cells[0:6, 10:16] = 2
    res = measure_field(f, s, cfg)
    assert res.interfaces.empty and len(res.cells) == 2
    html = build_report([res], cfg, PROV, tmp_path / "r.html").read_text()
    assert _row_cells(html, "qc-pooled")["cells_found"] == 2
    assert _row_cells(html, "qc-pooled")["border_touching"] == 2
    assert "no interfaces" in html.lower()
    assert "Traceback" not in html


def test_empty_interfaces_table_renders_with_message(results, cfg, tmp_path):
    r = replace(results[1], interfaces=results[1].interfaces.iloc[0:0].copy())
    html = build_report([r], cfg, PROV, tmp_path / "r.html").read_text()
    assert "no interfaces" in html.lower()
    assert 'class="dist"' not in html
    assert _row_cells(html, "qc-pooled")["cells_found"] == 4


def test_no_results_at_all_renders(cfg, tmp_path):
    html = build_report([], cfg, PROV, tmp_path / "r.html").read_text()
    assert "no fields" in html.lower() and "provenance" in html.lower()


def test_run_without_jm2_renders_and_says_so(tmp_path):
    cfg = RunConfig(channels=[ChannelSpec(role=Role.JM1, channel=0, marker="Na/K-ATPase"),
                              ChannelSpec(role=Role.NUCLEUS, channel=2, marker="DAPI")],
                    pixel_size_um=PX, threshold={Role.JM1: T1})
    f, s = build("f1")
    del f.planes[Role.JM2]
    res = measure_field(f, s, cfg)
    assert set(res.interfaces.role) == {"JM1"}
    html = build_report([res], cfg, PROV, tmp_path / "r.html").read_text()
    assert {k[2] for k in _dist_rows(html)} == {"JM1"}
    assert "JM2" in html and "not configured" in html.lower()


def test_run_without_pixel_size_reports_pixels_only(tmp_path):
    cfg = make_cfg()
    cfg = cfg.model_copy(update={"pixel_size_um": None})
    f, s = build("f1", px=float("nan"))
    res = measure_field(f, s, cfg)
    html = build_report([res], cfg, PROV, tmp_path / "r.html").read_text()
    params = {k[0] for k in _dist_rows(html)}
    assert "interface_contour_px" in params
    assert not any(p.endswith(("_um", "_um2")) for p in params)
    assert "no pixel size" in html.lower() and "pixels" in html.lower()
    assert "µm" not in html


# ---- determinism, errors -----------------------------------------------------------

def test_same_input_gives_byte_identical_html(results, cfg, tmp_path):
    a = build_report(results, cfg, PROV, tmp_path / "a.html").read_bytes()
    b = build_report(results, cfg, PROV, tmp_path / "b.html").read_bytes()
    assert a == b


def test_result_order_does_not_change_the_report(results, cfg, tmp_path):
    a = build_report(results, cfg, PROV, tmp_path / "a.html").read_bytes()
    b = build_report(results[::-1], cfg, PROV, tmp_path / "b.html").read_bytes()
    assert a == b


def test_no_timestamp_unless_supplied(results, cfg, tmp_path):
    h = build_report(results, cfg, PROV, tmp_path / "a.html").read_text()
    assert not re.search(r"20\d\d-\d\d-\d\d", h)
    h2 = build_report(results, cfg, {**PROV, "timestamp": "2026-01-02T03:04:05"},
                      tmp_path / "b.html").read_text()
    assert "2026-01-02T03:04:05" in h2


def test_duplicate_field_ids_raise_config_error(results, cfg, tmp_path):
    with pytest.raises(ConfigError) as e:
        build_report([results[0], results[0]], cfg, PROV, tmp_path / "r.html")
    assert isinstance(e.value, JunctionMapperError)
    assert not (tmp_path / "r.html").exists()


def test_returns_path_and_creates_parent(results, cfg, tmp_path):
    out = build_report(results, cfg, PROV, tmp_path / "sub" / "r.html")
    assert out == tmp_path / "sub" / "r.html" and out.is_file()
    assert isinstance(out, Path)


def test_provenance_scrubs_urls(results, cfg, tmp_path):
    prov = {**PROV, "params_used": {**PROV["params_used"], "weights": "https://x.org/m.pt"}}
    h = build_report(results, cfg, prov, tmp_path / "r.html").read_text()
    assert "://" not in h and "x.org/m.pt" in h


def test_qc_text_distinguishes_disagreement_from_filtered_contacts(html):
    qc = html[html.index('id="qc"'):html.index('id="quantification"')]
    assert "found corners and reached different answers" in qc
    assert "too short to measure" in qc
    assert "not raised" in qc


def test_every_embedded_image_decodes_to_a_png(html):
    import base64
    imgs = re.findall(r'<img src="data:image/png;base64,([^"]*)"', html)
    assert len(imgs) == 1 + 2 * len(PARAMETERS) + 3
    for b in imgs:
        assert base64.b64decode(b, validate=True)[:8] == b"\x89PNG\r\n\x1a\n"
