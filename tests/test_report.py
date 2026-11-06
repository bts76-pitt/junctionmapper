"""Task 14: self-contained HTML report, built from real ``measure_field`` output."""
import base64
import io
import re
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from junctionmapper.config import ChannelSpec, Role, RunConfig
from junctionmapper.errors import ConfigError, JunctionMapperError
from junctionmapper.pipeline import FieldResult, measure_field
import junctionmapper.report.build as build_mod
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


def _expected_flags(results):
    flags = pd.concat([r.flags for r in results])
    nuc = pd.concat([r.nuclei for r in results])
    out = {c: int(flags[c].sum()) for c in (
        "border_touching", "isolated", "degenerate", "corner_methods_disagree",
        "all_contacts_filtered", "multinucleate", "anucleate")}
    out["orphan_nuclei"] = int(nuc.orphan.sum())
    return out


def _expected_qc(results):
    """Independent recount from the tables (per field, then summed)."""
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
    assert _row_cells(html, "flags-pooled") == _expected_flags(results)


def test_qc_per_field_counts_match_tables_and_reasons_are_the_zoo_ones(results, html):
    zoo_row = _row_cells(html, "qc-field-zoo")
    assert zoo_row == _expected_qc(results[:1])
    # hand-checked against ZOO_NOTES: 7 cells; 3 untraceable; 6 touches the border; 7 isolated;
    # 4 has only a too-short contact (no interface at all); 5 keeps a dp interface but no tcj one
    assert zoo_row["cells_found"] == 7
    assert (zoo_row["excl_degenerate"], zoo_row["excl_border"], zoo_row["excl_isolated"],
            zoo_row["excl_filtered"], zoo_row["excl_other"]) == (1, 1, 1, 1, 0)
    flags = _row_cells(html, "flags-field-zoo")
    assert flags["multinucleate"] == 1 and flags["orphan_nuclei"] == 1
    f1_row = _row_cells(html, "qc-field-f1")
    assert f1_row["cells_found"] == 4 and f1_row["not_contributing"] == 0
    for k in zoo_row:       # pooled is the sum of the fields
        assert _row_cells(html, "qc-pooled")[k] == zoo_row[k] + f1_row[k]


def test_qc_exclusion_reasons_add_up_to_the_shortfall(html):
    """Cells found = contributing + sum of the one-reason exclusion columns, per row."""
    for rid in ("qc-pooled", "qc-field-zoo", "qc-field-f1"):
        r = _row_cells(html, rid)
        excl = sum(r[k] for k in r if k.startswith("excl_"))
        assert excl == r["not_contributing"], rid
        assert r["cells_found"] == r["with_interfaces"] + r["not_contributing"], rid
    z = _row_cells(html, "qc-field-zoo")
    assert z["measured"] - z["with_interfaces"] == 2      # the two measured cells that yield nothing
    assert "exactly one reason" in html and "degenerate, then border-touching, then isolated" in html


def test_qc_reports_contributing_cells_per_corner_method(results, html):
    z = _row_cells(html, "qc-field-zoo")
    assert z["contrib_dp"] == 3 and z["contrib_tcj"] == 2      # cell 5: dp rows, no tcj rows
    assert z["contrib_dp"] != z["contrib_tcj"]
    low = html.lower()
    assert "different sets of cells" in low and "too short" in low
    # and under every figure and in the summary rows, per group, recomputed from the tables
    ifc = pd.concat([r.interfaces for r in results])
    for (m, role, marker), g in ifc.groupby(["corner_method", "role", "marker"]):
        n = len(g[["field_id", "cell_id"]].drop_duplicates())
        assert f"{role} {marker} {m} = {n}" in html
        row = re.search(rf'<tr class="dist" data-scope="pooled" data-param="interface_contour_um" '
                        rf'data-method="{m}" data-role="{role}"[^>]*data-cells="(\d+)"', html)
        assert row and int(row.group(1)) == n, (m, role)


def test_usable_percentage_is_over_cells_found(results, html):
    z = _expected_qc(results[:1])
    assert z["measured"] != z["cells_found"]            # the denominators differ on this field
    row = re.search(r'<tr id="qc-field-zoo".*?</tr>', html, re.S).group(0)
    assert row.rstrip().endswith(f"<td>{100 * z['with_interfaces'] / z['cells_found']:.0f}%</td></tr>")


def test_field_losing_most_cells_is_called_out(results, cfg, tmp_path):
    f, s = zoo("zoo")
    keep = {3, 6, 7}        # degenerate, border-touching, isolated: nothing contributes
    for lab in range(1, 8):
        if lab not in keep:
            s.cells[s.cells == lab] = 0
    s.nuclei[:] = 0
    lost = measure_field(f, s, cfg)
    assert lost.interfaces.empty
    html = build_report([lost] + results[1:], cfg, PROV, tmp_path / "r.html").read_text()
    m = re.search(r'<section id="qc-warnings".*?</section>', html, re.S)
    assert m
    w = m.group(0)
    assert "zoo" in w and "f1" not in w
    assert "0 of 3 cells" in w and "the 3 that did not" in w
    for reason in ("1 degenerate", "1 border-touching", "1 isolated"):
        assert reason in w, reason
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
    assert _row_cells(html, "qc-pooled")["excl_border"] == 2
    assert _row_cells(html, "flags-pooled")["border_touching"] == 2
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


def test_qc_text_states_the_disagreement_rule_and_the_filtered_meaning(html):
    qc = html[html.index('id="qc"'):html.index('id="quantification"')]
    low = " ".join(qc.split())
    assert "differ by more than one" in low and "found at least one corner" in low
    assert "silent" in low and "enclosed" in low and "isolated" in low
    assert "can still contribute dp interfaces" in low
    assert "overlapping; not exclusion reasons" in low


def test_every_embedded_image_decodes_to_a_png(html):
    import base64
    imgs = re.findall(r'<img src="data:image/png;base64,([^"]*)"', html)
    assert len(imgs) == 1 + 2 * len(PARAMETERS) + 3
    for b in imgs:
        assert base64.b64decode(b, validate=True)[:8] == b"\x89PNG\r\n\x1a\n"


# =====================================================================================
# Fix round 1: tests that guard what the report SAYS, not just that it renders.
#
# How figure content is asserted: ``_png`` is wrapped (never replaced) so that, for every
# figure the real build produces, we capture (a) the live matplotlib axes - titles, x-limits,
# face colours, tick labels, histogram bar heights - and (b) the encoded PNG, which we decode
# with Pillow and measure (count of pixels at the "threshold varied" red). No byte hashes: pixel
# statistics and axis state are robust to platform font differences.
# =====================================================================================

RED_RGB = np.array([0xC0, 0x39, 0x2B])


class FigLog:
    def __init__(self):
        self.figs = []

    def wrap(self, orig):
        def inner(fig):
            axes = []
            for ax in fig.axes:
                axes.append({
                    "title": ax.get_title(), "xlim": tuple(ax.get_xlim()),
                    "facecolor": ax.get_facecolor(),
                    "xticks": [t.get_text() for t in ax.get_xticklabels()],
                    "bars": [p.get_height() for p in ax.patches],
                    "n_lines": len(ax.lines),
                })
            png = orig(fig)
            self.figs.append({"axes": axes, "png": base64.b64decode(png)})
            return png
        return inner


def _build_logged(res, cfg, out, monkeypatch):
    log = FigLog()
    monkeypatch.setattr(build_mod, "_png", log.wrap(build_mod._png))
    text = build_report(res, cfg, PROV, out).read_text()
    return text, log.figs


def _pooled_fig(figs, k):          # figures in creation order: (pooled, strip) x P, 3 extras, qc
    return figs[2 * k]


def _strip_fig(figs, k):
    return figs[2 * k + 1]


def _red_pixels(png):
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    return int((np.abs(a - RED_RGB).sum(axis=2) < 30).sum())


def _col(k, um=True):
    p = PARAMETERS[k]
    return p["um"] if um and p["um"] else p["px"]


@pytest.fixture(scope="module")
def logged(results, cfg, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        yield _build_logged(results, cfg, tmp_path_factory.mktemp("lg") / "r.html", mp)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def logged_varied(results_varying_t, cfg, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        yield _build_logged(results_varying_t, cfg, tmp_path_factory.mktemp("lv") / "r.html", mp)
    finally:
        mp.undo()


def test_figure_log_has_the_expected_layout(logged):
    _, figs = logged
    assert len(figs) == 1 + 2 * len(PARAMETERS) + 3


def test_histogram_panels_sit_in_dp_then_tcj_order_and_hold_their_own_data(results, logged):
    """Catches dp/tcj panels swapped: title method, n and bar mass must match the table."""
    _, figs = logged
    ifc = pd.concat([r.interfaces for r in results])
    seen_unequal = False
    for k in range(len(PARAMETERS)):
        col = _col(k)
        axes = _pooled_fig(figs, k)["axes"]
        assert len(axes) == 4                                  # 2 markers x 2 methods
        for idx, ax in enumerate(axes):
            m = ("dp", "tcj")[idx % 2]                         # column 0 is dp, column 1 is tcj
            role, marker = (("JM1", "Na/K-ATPase"), ("JM2", "p120"))[idx // 2]
            assert ax["title"].startswith(f"{role} {marker} | {m}"), ax["title"]
            g = ifc[(ifc.role == role) & (ifc.corner_method == m)]
            n = int(g[col].notna().sum())
            assert f"n={n}" in ax["title"]
            assert sum(ax["bars"]) == n, (col, idx)
        dp_n = int(ifc[(ifc.role == "JM1") & (ifc.corner_method == "dp")][col].notna().sum())
        tcj_n = int(ifc[(ifc.role == "JM1") & (ifc.corner_method == "tcj")][col].notna().sum())
        seen_unequal |= dp_n != tcj_n
    assert seen_unequal        # the fixture really distinguishes the two methods


def test_summary_statistics_match_independent_percentiles(results, html):
    """Catches median reported as Q1 and px values labelled as the um column."""
    ifc = pd.concat([r.interfaces for r in results])
    f = lambda x: f"{x:.4g}"
    checked = distinct = 0
    for m in re.finditer(r'<tr class="dist" data-scope="pooled" data-param="(\w+)" data-method="(\w+)" '
                         r'data-role="(\w+)".*?<td>(\d+)</td><td>\d+</td><td>([^<]*)</td><td>([^<]*)</td>'
                         r'<td>([^<]*)</td><td>([^<]*)</td><td>([^<]*)</td></tr>', html):
        col, method, role, n, mn, q1, med, q3, mx = m.groups()
        v = ifc[(ifc.corner_method == method) & (ifc.role == role)][col].dropna().to_numpy(float)
        if not v.size:
            continue
        exp = [f(v.min()), f(np.percentile(v, 25)), f(np.median(v)), f(np.percentile(v, 75)), f(v.max())]
        assert [mn, q1, med, q3, mx] == exp, (col, method, role)
        checked += 1
        distinct += q1 != med
    assert checked >= 40 and distinct >= 1


def test_quartiles_of_a_hand_computed_distribution(results, cfg, tmp_path):
    """f1 has 8 dp JM1 interfaces; give them the contours 1..8 um. Q1=2.75, median=4.5, Q3=6.25."""
    r = results[1]
    d = r.interfaces.copy()
    sel = (d.corner_method == "dp") & (d.role == "JM1")
    assert sel.sum() == 8
    d.loc[sel, "interface_contour_um"] = np.arange(1.0, 9.0)
    html = build_report([replace(r, interfaces=d)], cfg, PROV, tmp_path / "r.html").read_text()
    row = re.search(r'<tr class="dist" data-scope="pooled" data-param="interface_contour_um" '
                    r'data-method="dp" data-role="JM1".*?</tr>', html).group(0)
    assert re.findall(r"<td>([^<]*)</td>", row)[-5:] == ["1", "2.75", "4.5", "6.25", "8"]


def test_um_labelled_values_are_micrometres_not_pixels(results, html):
    """px and um differ by the pixel size (0.5): a um row holding px numbers has the wrong median."""
    ifc = pd.concat([r.interfaces for r in results])
    med = lambda col: f"{np.median(ifc[(ifc.corner_method == 'tcj') & (ifc.role == 'JM1')][col].dropna()):.4g}"
    row = re.search(r'<tr class="dist" data-scope="pooled" data-param="interface_contour_um" '
                    r'data-method="tcj" data-role="JM1".*?</tr>', html).group(0)
    assert re.findall(r"<td>([^<]*)</td>", row)[-3] == med("interface_contour_um")
    assert med("interface_contour_um") != med("interface_contour_px")


def test_threshold_pairing_with_fields_in_banner_and_provenance(tmp_path):
    """Three fields, three distinct thresholds: a banner or provenance row pairing a value with
    the wrong field is visible (zoo/f1 both using 50 hid it)."""
    res = []
    for fid, t in (("fa", 40.0), ("fb", 50.0), ("fc", 60.0)):
        f, s = build(fid)
        res.append(measure_field(f, s, make_cfg(threshold={Role.JM1: t, Role.JM2: T2})))
    html = build_report(res, make_cfg(), PROV, tmp_path / "r.html").read_text()
    w = re.search(r'<section id="threshold-warning".*?</section>', html, re.S).group(0)
    for fid, t in (("fa", "40"), ("fb", "50"), ("fc", "60")):
        assert f"<code>{fid}</code> = {t}" in w, fid
    assert "thresholds 40 vs 50 vs 60" in w
    prov = html[html.index('id="provenance"'):]
    rows = re.findall(r'<tr data-threshold-role="(\w+)"><td class="l">(\w+)</td><td>([^<]*)</td>', prov)
    assert {r for r in rows if r[0] == "JM1"} == {("JM1", "fa", "40"), ("JM1", "fb", "50"), ("JM1", "fc", "60")}
    assert {r for r in rows if r[0] == "JM2"} == {("JM2", f, "20") for f in ("fa", "fb", "fc")}


def test_strip_tick_labels_are_the_sorted_field_ids_in_order(logged):
    _, figs = logged
    for k in range(len(PARAMETERS)):
        for ax in _strip_fig(figs, k)["axes"]:
            assert ax["xticks"] == ["f1", "zoo"], (k, ax["xticks"])


def test_many_fields_drop_tick_labels_and_banner_is_capped(results, cfg, tmp_path, monkeypatch):
    base = results[1]
    res = []
    for i in range(41):
        fid = f"g{i:02d}"
        tabs = {n: getattr(base, n).assign(field_id=fid) for n in
                ("interfaces", "cells", "nuclei", "flags", "nuclei_distances")}
        tabs["interfaces"]["threshold"] = np.where(tabs["interfaces"]["role"] == "JM1", 50.0 + i, 20.0)
        res.append(FieldResult(**tabs))
    html, figs = _build_logged(res, cfg, tmp_path / "r.html", monkeypatch)
    assert len(figs) == 1 + 2 * len(PARAMETERS) + 3                       # still bounded
    for k in range(len(PARAMETERS)):
        for ax in _strip_fig(figs, k)["axes"]:
            assert ax["xticks"] == [] or set(ax["xticks"]) == {""}        # labels dropped beyond 40
    w = re.search(r'<section id="threshold-warning".*?</section>', html, re.S).group(0)
    assert w.count("<code>") == 10 and "and 31 more fields" in w
    prov = html[html.index('id="provenance"'):]
    assert len(re.findall(r'<tr data-threshold-role="JM1">', prov)) == 41     # complete list kept
    assert "omit field labels beyond 40 fields" in " ".join(html.split())


def test_threshold_varied_marking_is_drawn_in_the_figures(logged, logged_varied):
    """Catches the red marking being dropped: panel colour, title and decoded pixels."""
    _, fv = logged_varied
    _, fc = logged
    for k, p in enumerate(PARAMETERS):
        for fig in (_pooled_fig(fv, k), _strip_fig(fv, k)):
            jm1 = fig["axes"][:2]
            jm2 = fig["axes"][2:4]
            if p["td"]:
                assert all("THRESHOLD VARIED" in a["title"] for a in jm1), p["key"]
                assert all(abs(a["facecolor"][0] - 253 / 255) < 0.01 for a in jm1)
                assert not any("THRESHOLD VARIED" in a["title"] for a in jm2)   # JM2 did not vary
                assert _red_pixels(fig["png"]) > 30, p["key"]
            else:
                assert not any("THRESHOLD VARIED" in a["title"] for a in fig["axes"]), p["key"]
                assert _red_pixels(fig["png"]) == 0, p["key"]
        assert _red_pixels(_pooled_fig(fc, k)["png"]) == 0
        assert not any("THRESHOLD VARIED" in a["title"] for a in _pooled_fig(fc, k)["axes"])


def test_histogram_axes_are_per_marker_and_shared_within_a_marker(cfg, tmp_path, monkeypatch):
    """JM2 is scaled x20 (threshold too), so its marker_intensity is an order of magnitude larger."""
    f, s = zoo("zoo")
    f.planes[Role.JM2] = f.planes[Role.JM2] * 20
    cfg20 = make_cfg(threshold={Role.JM1: T1, Role.JM2: T2 * 20})
    _, figs = _build_logged([measure_field(f, s, cfg20)], cfg20, tmp_path / "r.html", monkeypatch)
    k = next(i for i, p in enumerate(PARAMETERS) if p["key"] == "marker_intensity")
    ax = _pooled_fig(figs, k)["axes"]
    jm1_dp, jm1_tcj, jm2_dp, jm2_tcj = (a["xlim"] for a in ax)
    assert jm1_dp == jm1_tcj and jm2_dp == jm2_tcj             # methods comparable within a marker
    assert jm2_dp[1] > 5 * jm1_dp[1]                           # a weak marker keeps its own range
    assert "per marker" in " ".join(tmp_path.joinpath("r.html").read_text().split())


def _png_chunks(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    i, out = 8, []
    while i < len(png):
        n, typ = struct.unpack(">I4s", png[i:i + 8])
        out.append((typ.decode(), png[i + 8:i + 8 + n]))
        i += 12 + n
    return out


def test_embedded_pngs_carry_no_text_chunks_url_or_version(logged):
    _, figs = logged
    for fig in figs:
        types = [t for t, _ in _png_chunks(fig["png"])]
        assert not {"tEXt", "iTXt", "zTXt"} & set(types), types
        assert set(types) <= {"IHDR", "pHYs", "IDAT", "IEND"}, types      # nothing but pixels


def test_mixed_pixel_size_names_the_fields_missing_from_um_panels(cfg, tmp_path):
    f1, s1 = build("f1")
    fn, sn = build("fnopx", px=float("nan"))
    res = [measure_field(f1, s1, cfg), measure_field(fn, sn, cfg.model_copy(update={"pixel_size_um": None}))]
    html = build_report(res, cfg, PROV, tmp_path / "r.html").read_text()
    m = re.search(r'<p class="warn">Fields without a pixel size \(([^)]*)\)', html)
    assert m and m.group(1) == "fnopx"
    n = {int(x) for x in re.findall(r'data-scope="pooled" data-param="interface_contour_um" data-method="dp" data-role="JM1"[^>]*data-n="(\d+)"', html)}
    assert n == {8}                           # only the field with a pixel size contributes
    assert "Fields without a pixel size" not in build_report(res[:1], cfg, PROV, tmp_path / "o.html").read_text()
