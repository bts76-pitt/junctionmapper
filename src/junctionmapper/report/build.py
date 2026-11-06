"""Build the single-file HTML report from ``FieldResult`` objects.

Three sections: QC (can I trust this run), Quantification (what does it show) and Provenance.
Every figure is an inline base64 PNG; the page references nothing external and contains no URL.

Figure count (independent of the number of fields and of markers):
    1 (QC bar) + P (pooled distributions) + P (per-field strip panels) + 3 (cell / nucleus size)
with P = len(PARAMETERS) = 13. Each figure is a small-multiples grid whose panels are the
(role, marker) x corner-method groups; groups are never pooled together. Per-field detail is
also available as summary rows (n, quartiles) rather than as extra figures, so the page grows
in rows, not in images, as fields are added.
"""
from __future__ import annotations

import base64
import html as _html
import io
import json
import re
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

from junctionmapper.config import Role, RunConfig  # noqa: E402
from junctionmapper.errors import ConfigError  # noqa: E402
from junctionmapper.pipeline import THRESHOLD_SAME_TOL  # noqa: E402

METHODS = ("dp", "tcj")
ROLE_ORDER = {"JM1": 0, "JM2": 1}
_TABLES = ("interfaces", "cells", "nuclei", "flags", "nuclei_distances")

# key, label, px column, um column (None: unit-free / intensity), threshold-dependent
PARAMETERS: list[dict[str, Any]] = [
    {"key": "interface_contour", "label": "Interface contour", "px": "interface_contour_px", "um": "interface_contour_um", "td": False},
    {"key": "straight_line_length", "label": "Straight-line interface length", "px": "straight_line_length_px", "um": "straight_line_length_um", "td": False},
    {"key": "linearity_index", "label": "Linearity index (straight / contour)", "px": "linearity_index", "um": None, "td": False},
    {"key": "interface_area", "label": "Interface area", "px": "interface_area_px", "um": "interface_area_um2", "td": False},
    {"key": "fragmented_contour", "label": "Fragmented junction contour", "px": "fragmented_contour_px", "um": "fragmented_contour_um", "td": True},
    {"key": "junction_contour", "label": "Junction contour", "px": "junction_contour_px", "um": "junction_contour_um", "td": True},
    {"key": "junction_straight_line", "label": "Straight-line junction length", "px": "junction_straight_line_px", "um": "junction_straight_line_um", "td": True},
    {"key": "coverage_index", "label": "Coverage index (fragmented / interface contour)", "px": "coverage_index", "um": None, "td": True},
    {"key": "marker_area", "label": "Marker area", "px": "marker_area_px", "um": "marker_area_um2", "td": True},
    {"key": "marker_intensity", "label": "Marker intensity (sum above threshold)", "px": "marker_intensity", "um": None, "td": True},
    {"key": "interface_occupancy_pct", "label": "Interface occupancy (%)", "px": "interface_occupancy_pct", "um": None, "td": True},
    {"key": "intensity_per_area", "label": "Intensity per interface area", "px": "intensity_per_px2", "um": "intensity_per_um2", "td": True},
    {"key": "cluster_density", "label": "Cluster density (intensity per marker area)", "px": "cluster_density_per_px2", "um": "cluster_density_per_um2", "td": True},
]

# Exclusion accounting: every cell that contributes no interface is assigned to exactly one
# reason, in this precedence order, so the reason columns add up to "Not contributing".
EXCL_COLUMNS = [
    ("excl_degenerate", "Degenerate"), ("excl_border", "Border-touching"),
    ("excl_isolated", "Isolated"), ("excl_filtered", "All contacts filtered"),
    ("excl_other", "Other (no interface)"),
]
QC_COLUMNS = [
    ("cells_found", "Cells found"), ("measured", "Measured"),
    ("with_interfaces", "Contributing (any method)"),
    ("contrib_dp", "Contributing: dp"), ("contrib_tcj", "Contributing: tcj"),
    ("not_contributing", "Not contributing"),
] + EXCL_COLUMNS
# Overlapping raw flag counts (a cell can carry several); not exclusion reasons.
FLAG_COLUMNS = [
    ("border_touching", "Border-touching"), ("isolated", "Isolated"), ("degenerate", "Degenerate"),
    ("corner_methods_disagree", "Corner methods disagree"),
    ("all_contacts_filtered", "All contacts filtered"),
    ("multinucleate", "Multinucleate"), ("anucleate", "Anucleate"), ("orphan_nuclei", "Orphan nuclei"),
]
_FLAG_COLS = ("border_touching", "isolated", "degenerate", "corner_methods_disagree",
              "all_contacts_filtered", "multinucleate", "anucleate")
BANNER_CAP = 10        # fields listed inline in the threshold banner; the full list is in Provenance
LOST_FRACTION = 0.5     # a field with fewer than this fraction of cells yielding interfaces is flagged


# ---- small helpers -----------------------------------------------------------------

def _scrub(v: Any) -> str:
    """Text for display; URLs are defused so the page carries none."""
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return re.sub(r"(?i)\b[a-z][a-z0-9+.-]*://", "", s)


def _fmt(x: float) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.4g}"


def _unit(col: str) -> str:
    for suffix, unit in (("_um2", "µm²"), ("_um", "µm"), ("_px2", "per px²"), ("_px", "px")):
        if col.endswith(suffix):
            if suffix == "_um2" and col.startswith(("intensity", "cluster")):
                return "per µm²"
            return "px²" if suffix == "_px" and "area" in col else unit
    return ""


def _mpl(s: str) -> str:
    return s.replace("$", r"\$")


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, metadata={"Software": None})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cat(results: list, name: str) -> pd.DataFrame:
    frames = [getattr(r, name) for r in results if len(getattr(r, name))]
    return pd.concat(frames, ignore_index=True) if frames else getattr(results[0], name).iloc[0:0] if results else pd.DataFrame()


def _fids(res) -> set[str]:
    return {str(f) for n in _TABLES for f in getattr(res, n)["field_id"].dropna()}


def _stats(vals: np.ndarray, cells: int = 0) -> dict[str, Any]:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return {"n": 0, "cells": 0, "min": "–", "q1": "–", "median": "–", "q3": "–", "max": "–"}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "cells": cells, "min": _fmt(v.min()), "q1": _fmt(q1), "median": _fmt(med),
            "q3": _fmt(q3), "max": _fmt(v.max())}


def _finite(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _has_pixel_size(cells: pd.DataFrame, cfg: RunConfig) -> bool:
    if len(cells):
        ps = pd.to_numeric(cells["pixel_size_um"], errors="coerce")
        return bool(((ps > 0) & np.isfinite(ps)).any())
    ps = cfg.pixel_size_um
    return bool(ps) and np.isfinite(ps) and ps > 0


def _fields_without_pixel_size(cells: pd.DataFrame, fields: list[str]) -> list[str]:
    """Fields that have cells but no usable pixel size (absent from the micrometre panels)."""
    out = []
    for fid in fields:
        ps = pd.to_numeric(cells.loc[cells["field_id"] == fid, "pixel_size_um"], errors="coerce")
        if len(ps) and not ((ps > 0) & np.isfinite(ps)).any():
            out.append(fid)
    return out


# ---- QC ----------------------------------------------------------------------------

def _qc_counts(cells: pd.DataFrame, flags: pd.DataFrame, nuclei: pd.DataFrame,
               ifc: pd.DataFrame) -> dict[str, int]:
    """Counts for ONE field (cell ids are only unique within a field)."""
    out = {k: 0 for k, _ in QC_COLUMNS}
    out.update({c: 0 for c in _FLAG_COLS})
    out["orphan_nuclei"] = int(nuclei["orphan"].sum()) if len(nuclei) else 0
    ids = [int(c) for c in cells["cell_id"].dropna()]
    out["cells_found"] = len(ids)
    out["measured"] = int(cells["measured"].sum()) if len(cells) else 0
    contrib = {m: {int(c) for c in ifc.loc[ifc["corner_method"] == m, "cell_id"].dropna()}
               for m in METHODS} if len(ifc) else {m: set() for m in METHODS}
    anyc = contrib["dp"] | contrib["tcj"]
    out["contrib_dp"], out["contrib_tcj"] = len(contrib["dp"] & set(ids)), len(contrib["tcj"] & set(ids))
    out["with_interfaces"] = len(anyc & set(ids))
    for c in _FLAG_COLS:
        out[c] = int(flags[c].sum()) if len(flags) else 0
    fl = {int(r["cell_id"]): r for _, r in flags.iterrows()} if len(flags) else {}
    for cid in ids:
        if cid in anyc:
            continue
        r = fl.get(cid)
        if r is not None and r["degenerate"]:
            out["excl_degenerate"] += 1
        elif r is not None and r["border_touching"]:
            out["excl_border"] += 1
        elif r is not None and r["isolated"]:
            out["excl_isolated"] += 1
        elif r is not None and r["all_contacts_filtered"]:
            out["excl_filtered"] += 1
        else:
            out["excl_other"] += 1
        out["not_contributing"] += 1
    return out


def _qc(fields: list[str], cells, flags, nuclei, ifc):
    def sel(df, fid):
        return df[df["field_id"] == fid]
    per = {fid: _qc_counts(sel(cells, fid), sel(flags, fid), sel(nuclei, fid), sel(ifc, fid))
           for fid in fields}
    pooled = {k: sum(c[k] for c in per.values()) for k in [k for k, _ in QC_COLUMNS + FLAG_COLUMNS]}
    rows = [{"id": "qc-pooled", "flag_id": "flags-pooled", "label": "All fields (pooled)", "counts": pooled}]
    warnings = []
    for fid in fields:
        counts = per[fid]
        rows.append({"id": f"qc-field-{fid}", "flag_id": f"flags-field-{fid}", "label": fid, "counts": counts})
        found = counts["cells_found"]
        if found and counts["with_interfaces"] < LOST_FRACTION * found:
            why = [f"{counts[k]} {lab.lower()}" for k, lab in EXCL_COLUMNS if counts[k]]
            warnings.append({"field": fid, "found": found, "usable": counts["with_interfaces"],
                             "lost": counts["not_contributing"], "why": ", ".join(why)})
    for r in rows:
        c = r["counts"]
        r["usable_pct"] = "–" if not c["cells_found"] else f"{100 * c['with_interfaces'] / c['cells_found']:.0f}%"
    return rows, warnings


def _qc_figure(pooled: dict[str, int]) -> str | None:
    if not pooled["cells_found"]:
        return None
    keys = [k for k, _ in QC_COLUMNS] + [k for k, _ in FLAG_COLUMNS]
    labels = dict(QC_COLUMNS + FLAG_COLUMNS)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    vals = [pooled[k] for k in keys]
    ypos = np.arange(len(keys))[::-1]
    ax.barh(ypos, vals, color="#4c78a8")
    ax.set_yticks(ypos, [labels[k] for k in keys])
    for y, v in zip(ypos, vals):
        ax.text(v, y, f" {v}", va="center", fontsize=8)
    ax.set_xlabel("cells (nuclei for orphan); flag counts overlap")
    ax.set_title("QC counts, all fields pooled", fontsize=10)
    fig.tight_layout()
    return _png(fig)


# ---- quantification ----------------------------------------------------------------

def _groups(ifc: pd.DataFrame) -> list[tuple[str, str]]:
    if not len(ifc):
        return []
    pairs = {(str(r), str(m)) for r, m in zip(ifc["role"], ifc["marker"])}
    return sorted(pairs, key=lambda p: (ROLE_ORDER.get(p[0], 9), p[1]))


def _threshold_info(ifc: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per role: the APPLIED threshold per field, and whether it genuinely varied.

    ``varied`` means the applied values span more than ``THRESHOLD_SAME_TOL`` grey levels,
    i.e. somebody overrode T for some field. The default path proposes one value for the
    whole run (``pipeline.propose_run_thresholds``), so it does not trip this; a bare
    difference in the last bits of a float never did mean "not comparable".
    ``field_otsu`` carries, per field, what Otsu on that field's own plane would have given,
    and ``deviating`` the fields where the two are far enough apart to be worth a look.
    """
    info: dict[str, dict[str, Any]] = {}
    if not len(ifc):
        return info
    has_otsu = "threshold_field_otsu" in ifc.columns
    has_dev = "threshold_deviates_from_run" in ifc.columns
    for role in sorted(set(ifc["role"]), key=lambda r: ROLE_ORDER.get(r, 9)):
        g = ifc[ifc["role"] == role]
        by_field = {str(f): sorted({float(t) for t in gf["threshold"].dropna()})
                    for f, gf in g.groupby("field_id")}
        own = {str(f): sorted({float(t) for t in gf["threshold_field_otsu"].dropna()})
               for f, gf in g.groupby("field_id")} if has_otsu else {}
        deviating = sorted({str(f) for f, gf in g.groupby("field_id")
                            if bool(gf["threshold_deviates_from_run"].any())}) if has_dev else []
        distinct = sorted({t for ts in by_field.values() for t in ts})
        varied = bool(distinct) and (distinct[-1] - distinct[0]) > THRESHOLD_SAME_TOL
        run_value = float(np.median([v for ts in by_field.values() for v in ts])) if distinct else 0.0
        differing = [f for f, ts in sorted(by_field.items())
                     if any(abs(t - run_value) > THRESHOLD_SAME_TOL for t in ts)]
        info[str(role)] = {
            "role": str(role), "markers": sorted(set(map(str, g["marker"]))),
            "by_field": by_field, "distinct": distinct, "varied": varied,
            "distinct_txt": " vs ".join(_fmt(x) for x in distinct),
            "shown": [(f, ", ".join(_fmt(x) for x in v)) for f, v in sorted(by_field.items())[:BANNER_CAP]],
            "more": max(0, len(by_field) - BANNER_CAP),
            "run_value": _fmt(run_value), "tol": _fmt(THRESHOLD_SAME_TOL),
            "differing": differing[:BANNER_CAP],
            "differing_more": max(0, len(differing) - BANNER_CAP),
            "n_differing": len(differing),
            "field_otsu": own, "deviating": deviating,
        }
    return info


RED = "#c0392b"
BLUE = "#4c78a8"


def _hist_figure(sub: dict, groups, col, label, unit, varied_roles, td) -> str:
    """Rows = (role, marker), columns = dp / tcj.

    The bin range is computed per (role, marker) and shared by dp and tcj within it, so the
    two methods of one marker are directly comparable while a weak marker is not crushed by a
    strong one.
    """
    rows = groups
    fig, axes = plt.subplots(len(rows), len(METHODS), figsize=(3.4 * len(METHODS), 2.5 * len(rows)),
                             squeeze=False)
    for i, (role, marker) in enumerate(rows):
        vs = [sub[(role, marker, m)] for m in METHODS]
        vs = [v[np.isfinite(v)] for v in vs]
        allv = np.concatenate(vs)
        lo, hi = (float(allv.min()), float(allv.max())) if allv.size else (0.0, 1.0)
        if hi <= lo:
            lo, hi = lo - 0.5, lo + 0.5
        bins = np.linspace(lo, hi, 31)
        for j, m in enumerate(METHODS):
            ax = axes[i][j]
            v = vs[j]
            bad = td and role in varied_roles
            title = _mpl(f"{role} {marker} | {m}  n={v.size}")
            if v.size:
                ax.hist(v, bins=bins, color=RED if bad else BLUE, edgecolor="white", linewidth=0.3)
                ax.axvline(float(np.median(v)), color="black", lw=1, ls="--")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(lo, hi)
            if bad:
                title += "\nTHRESHOLD VARIED: not comparable"
                ax.set_facecolor("#fdecea")
            ax.set_title(title, fontsize=8, color="#a93226" if bad else "black")
            ax.tick_params(labelsize=7)
            ax.set_xlabel(_mpl(f"{label} [{unit}]" if unit else label), fontsize=7)
    fig.tight_layout(h_pad=1.5)
    return _png(fig)


TICK_LABEL_MAX_FIELDS = 40


def _strip_figure(per_field: dict, fields, groups, label, unit, varied_roles, td) -> str:
    n = len(fields)
    width = min(max(3.4, 0.28 * n + 1.5), 14.0)
    labelled = n <= TICK_LABEL_MAX_FIELDS
    fig, axes = plt.subplots(len(groups), len(METHODS),
                             figsize=(width * len(METHODS) / 2 + 1.5, (3.6 if labelled else 3.0) * len(groups)),
                             squeeze=False, sharey="row")
    rng = np.random.default_rng(0)
    for i, (role, marker) in enumerate(groups):
        for j, m in enumerate(METHODS):
            ax = axes[i][j]
            bad = td and role in varied_roles
            for k, fid in enumerate(fields):
                v = per_field[(role, marker, m, fid)]
                v = np.sort(v[np.isfinite(v)])
                if not v.size:
                    continue
                pts = v if v.size <= 60 else v[np.linspace(0, v.size - 1, 60).astype(int)]
                ax.plot(k + rng.uniform(-0.25, 0.25, pts.size), pts, ".", ms=2.5, alpha=0.5,
                        color=RED if bad else BLUE)
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.vlines(k, q1, q3, color="black", lw=2)
                ax.plot([k], [med], "o", color="white", mec="black", ms=3)
            ax.set_xlim(-0.8, max(n - 1, 0) + 0.8)
            if labelled:
                ax.set_xticks(range(n), fields, rotation=90, fontsize=6)
            else:
                ax.set_xticks([])
                ax.set_xlabel(f"{n} fields (sorted by id; labels omitted)", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            t = _mpl(f"{role} {marker} | {m}")
            if bad:
                t += "\nTHRESHOLD VARIED"
                ax.set_facecolor("#fdecea")
            ax.set_title(t, fontsize=8, color="#a93226" if bad else "black", pad=6)
            if j == 0:
                ax.set_ylabel(_mpl(f"{label} [{unit}]" if unit else label), fontsize=7)
    fig.tight_layout(h_pad=3.0, w_pad=1.5)
    return _png(fig)


def _ncells(g: pd.DataFrame, col: str) -> int:
    """Distinct (field, cell) pairs with a finite value of ``col``."""
    if not len(g):
        return 0
    ok = np.isfinite(_finite(g, col))
    return len(g.loc[ok, ["field_id", "cell_id"]].drop_duplicates())


def _quantification(ifc: pd.DataFrame, fields: list[str], varied_roles: set[str], use_um: bool):
    groups = _groups(ifc)
    out = []
    if not groups:
        return out, groups
    for p in PARAMETERS:
        col = p["um"] if (use_um and p["um"]) else p["px"]
        unit = _unit(col)
        sub, per_field = {}, {}
        rows = []
        for role, marker in groups:
            g0 = ifc[(ifc["role"] == role) & (ifc["marker"] == marker)]
            for m in METHODS:
                g = g0[g0["corner_method"] == m]
                sub[(role, marker, m)] = _finite(g, col)
                rows.append({"scope": "pooled", "param": col, "method": m, "role": role,
                             "marker": marker, "varied": bool(p["td"] and role in varied_roles),
                             **_stats(sub[(role, marker, m)], _ncells(g, col))})
                for fid in fields:
                    gf = g[g["field_id"] == fid]
                    v = _finite(gf, col)
                    per_field[(role, marker, m, fid)] = v
                    rows.append({"scope": f"field:{fid}", "param": col, "method": m, "role": role,
                                 "marker": marker, "varied": bool(p["td"] and role in varied_roles),
                                 **_stats(v, _ncells(gf, col))})
        out.append({
            "key": p["key"], "label": p["label"], "col": col, "unit": unit, "td": p["td"],
            "varied": any(r["varied"] for r in rows),
            "pooled_img": _hist_figure(sub, groups, col, p["label"], unit, varied_roles, p["td"]),
            "field_img": _strip_figure(per_field, fields, groups, p["label"], unit, varied_roles, p["td"]),
            "cells_caption": [{"role": r["role"], "marker": r["marker"], "method": r["method"],
                               "cells": r["cells"]} for r in rows if r["scope"] == "pooled"],
            "pooled_rows": [r for r in rows if r["scope"] == "pooled"],
            "field_rows": [r for r in rows if r["scope"] != "pooled"],
        })
    return out, groups


def _simple_hist(values: np.ndarray, label: str) -> str | None:
    v = values[np.isfinite(values)]
    if not v.size:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.hist(v, bins=30, color="#4c78a8", edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.median(v)), color="black", lw=1, ls="--")
    ax.set_xlabel(_mpl(label), fontsize=8)
    ax.set_title(_mpl(f"{label}  n={v.size}"), fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _png(fig)


def _extras(cells, nuclei, use_um):
    specs = [("Cell area", cells, "area_um2" if use_um else "area_px"),
             ("Nucleus area", nuclei, "area_um2" if use_um else "area_px"),
             ("Mean intra-nuclear distance (per nucleus)", nuclei,
              "mean_intra_distance_um" if use_um else "mean_intra_distance_px")]
    out = []
    for label, df, col in specs:
        unit = _unit(col)
        v = _finite(df, col) if len(df) else np.array([])
        out.append({"label": label, "col": col, "unit": unit,
                    "img": _simple_hist(v, f"{label} [{unit}]"), **_stats(v)})
    return out


# ---- provenance --------------------------------------------------------------------

def _distinct(df: pd.DataFrame, col: str) -> list[str]:
    if not len(df):
        return []
    return sorted({_fmt(float(x)) for x in pd.to_numeric(df[col], errors="coerce").dropna()})


def _provenance(cfg: RunConfig, prov: dict, cells, ifc, tinfo, use_um):
    roles = []
    for spec in sorted(cfg.channels, key=lambda c: ({"JM1": 0, "JM2": 1}.get(c.role.value, 2))):
        roles.append({"role": spec.role.value, "marker": spec.marker, "channel": str(spec.channel)})
    if not any(c.role == Role.JM2 for c in cfg.channels):
        roles.append({"role": "JM2", "marker": None, "channel": None})
    params = prov.get("params_used") or {}
    thresholds = []
    for role, t in tinfo.items():
        thresholds.append({"role": role, "marker": ", ".join(t["markers"]), "varied": t["varied"],
                           "by_field": [{"field": f, "vals": ", ".join(_fmt(x) for x in v) or "–",
                                         "otsu": ", ".join(_fmt(x) for x in t["field_otsu"].get(f, ())) or "–",
                                         "deviates": f in t["deviating"]}
                                        for f, v in sorted(t["by_field"].items())],
                           "n_deviating": len(t["deviating"]),
                           "summary": ", ".join(_fmt(x) for x in t["distinct"])})
    cfg_t = {r.value: _fmt(float(v)) for r, v in sorted((cfg.threshold or {}).items(), key=lambda kv: kv[0].value)}
    version = prov.get("junctionmapper_version")
    if version is None:
        try:
            version = metadata.version("junctionmapper")
        except metadata.PackageNotFoundError:
            version = "unknown"
    scalars = [
        ("epsilon", _distinct(cells, "epsilon"), _fmt(cfg.epsilon_px)),
        ("tcj_min_run", _distinct(cells, "tcj_min_run"), str(cfg.tcj_min_run)),
        ("neighbour_radius_px", _distinct(cells, "neighbour_radius_px"), str(cfg.neighbour_radius_px)),
        ("dilation_cycles", _distinct(cells, "dilation_cycles"), "–"),
        ("pixel size (µm per px)" if use_um else "pixel size", _distinct(cells, "pixel_size_um"),
         _fmt(cfg.pixel_size_um) if cfg.pixel_size_um else "not set"),
    ]
    return {
        "version": _scrub(str(version)), "model": _scrub(params.get("model", "not recorded")),
        "params": [{"key": str(k), "value": _scrub(params[k])} for k in sorted(params, key=str)],
        "roles": roles, "thresholds": thresholds, "cfg_thresholds": cfg_t,
        "scalars": [{"name": n, "observed": ", ".join(o) or "–", "cfg": c} for n, o, c in scalars],
        "timestamp": _scrub(prov["timestamp"]) if prov.get("timestamp") else None,
        "threshold_proposal": _scrub(prov["threshold_proposal"]) if prov.get("threshold_proposal") else None,
        "extra": [{"key": str(k), "value": _scrub(prov[k])} for k in sorted(prov, key=str)
                  if k not in ("params_used", "junctionmapper_version", "timestamp",
                               "threshold_proposal")],
        "num_intra_cells": cfg.num_intra_cells, "all_pairs": cfg.all_pairs,
        "dilation_px": cfg.dilation_px, "dilation_um": cfg.dilation_um,
    }


# ---- entry point -------------------------------------------------------------------

def build_report(results: Iterable, cfg: RunConfig, provenance: dict | None, out: Path) -> Path:
    """Write the report to ``out`` (parents created) and return ``out``.

    ``provenance`` keys (all optional): ``params_used`` (segmentation parameters, including
    ``model``), ``junctionmapper_version``, ``timestamp`` (embedded only if given), plus any
    extra scalar entries. Raises ConfigError if two results share a ``field_id``.
    """
    results = list(results)
    provenance = provenance or {}
    out = Path(out)
    owner: dict[str, int] = {}
    for i, r in enumerate(results):
        for f in _fids(r):
            if f in owner and owner[f] != i:
                raise ConfigError(f"duplicate field_id across results: {f!r}")
            owner[f] = i
    fields = sorted(owner)

    if results:
        cells, flags = _cat(results, "cells"), _cat(results, "flags")
        nuclei, ifc = _cat(results, "nuclei"), _cat(results, "interfaces")
    else:
        cells = flags = nuclei = ifc = pd.DataFrame(columns=["field_id"])
    for df in (cells, flags, nuclei, ifc):
        if len(df):
            df["field_id"] = df["field_id"].astype(str)

    use_um = _has_pixel_size(cells, cfg)
    tinfo = _threshold_info(ifc)
    varied_roles = {r for r, t in tinfo.items() if t["varied"]}
    qc_rows, qc_warnings = _qc(fields, cells, flags, nuclei, ifc) if results else ([], [])
    quant, groups = _quantification(ifc, fields, varied_roles, use_um)
    no_px = _fields_without_pixel_size(cells, fields) if use_um else []
    extras = _extras(cells, nuclei, use_um) if len(cells) else []

    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)), autoescape=True,
                      trim_blocks=True, lstrip_blocks=True)
    page = env.get_template("template.html").render(
        n_fields=len(fields), fields=fields, qc_columns=QC_COLUMNS, qc_rows=qc_rows,
        qc_warnings=qc_warnings, qc_img=_qc_figure(qc_rows[0]["counts"]) if qc_rows else None,
        lost_pct=int(LOST_FRACTION * 100), quant=quant, groups=groups, extras=extras,
        use_um=use_um, no_px_fields=no_px, flag_columns=FLAG_COLUMNS, excl_columns=EXCL_COLUMNS,
        tick_max=TICK_LABEL_MAX_FIELDS, has_interfaces=bool(len(ifc)), tinfo=list(tinfo.values()),
        varied=[t for t in tinfo.values() if t["varied"]], doi="10.7554/eLife.45413",
        prov=_provenance(cfg, provenance, cells, ifc, tinfo, use_um),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
