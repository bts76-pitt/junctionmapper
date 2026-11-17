"""Metric definitions and the run-level facts the report shows: QC counts, thresholds,
run warnings and the provenance table.

Everything here reads the tables and the run's provenance; nothing changes a measured value.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from junctionmapper.config import RunConfig
from junctionmapper.pipeline import THRESHOLD_SAME_TOL

METHODS = ("dp", "tcj")
ROLE_ORDER = {"JM1": 0, "JM2": 1}

# key, label, px column, um column (None: unit-free / intensity), threshold-dependent, one-line help
PARAMETERS: list[dict[str, Any]] = [
    {"key": "interface_contour", "label": "Interface contour", "px": "interface_contour_px", "um": "interface_contour_um", "td": False, "help": "Length of the interface pixel walk (1 per orthogonal step, sqrt 2 per diagonal)."},
    {"key": "straight_line_length", "label": "Straight-line interface length", "px": "straight_line_length_px", "um": "straight_line_length_um", "td": False, "help": "Straight-line distance between the interface's two end corners."},
    {"key": "linearity_index", "label": "Linearity index (straight / contour)", "px": "linearity_index", "um": None, "td": False, "help": "Straight-line length / contour length. At most 1; 1 is a straight interface."},
    {"key": "interface_area", "label": "Interface area", "px": "interface_area_px", "um": "interface_area_um2", "td": False, "help": "Pixel count of the dilated band around the interface."},
    {"key": "fragmented_contour", "label": "Fragmented junction contour", "px": "fragmented_contour_px", "um": "fragmented_contour_um", "td": True, "help": "Walk length over connected above-threshold interface pixels; gaps are not bridged."},
    {"key": "junction_contour", "label": "Junction contour", "px": "junction_contour_px", "um": "junction_contour_um", "td": True, "help": "Walk length from first to last above-threshold pixel, gaps included."},
    {"key": "junction_straight_line", "label": "Straight-line junction length", "px": "junction_straight_line_px", "um": "junction_straight_line_um", "td": True, "help": "Straight-line distance from the first to the last above-threshold interface pixel."},
    {"key": "coverage_index", "label": "Coverage index (fragmented / interface contour)", "px": "coverage_index", "um": None, "td": True, "help": "Fragmented junction contour / interface contour."},
    {"key": "marker_area", "label": "Marker area", "px": "marker_area_px", "um": "marker_area_um2", "td": True, "help": "Pixel count of band pixels above the marker threshold."},
    {"key": "marker_intensity", "label": "Marker intensity (sum above threshold)", "px": "marker_intensity", "um": None, "td": True, "help": "Sum of above-threshold values in the band, in raw intensity units."},
    {"key": "interface_occupancy_pct", "label": "Interface occupancy (%)", "px": "interface_occupancy_pct", "um": None, "td": True, "help": "100 * marker area / interface area, from pixel counts."},
    {"key": "intensity_per_area", "label": "Intensity per interface area", "px": "intensity_per_px2", "um": "intensity_per_um2", "td": True, "help": "Marker intensity / interface area."},
    {"key": "cluster_density", "label": "Cluster density (intensity per marker area)", "px": "cluster_density_per_px2", "um": "cluster_density_per_um2", "td": True, "help": "Marker intensity / marker area."},
    {"key": "band_mean_intensity", "label": "Band mean intensity (no threshold)", "px": "band_mean_intensity", "um": None, "td": False, "help": "Mean of all band pixels, no threshold, in raw intensity units."},
]

# Exclusion accounting: every cell that contributes no interface is assigned to exactly one
# reason, in this precedence order, so the reason counts add up to "not_contributing".
EXCL_COLUMNS = [
    ("excl_degenerate", "degenerate"), ("excl_border", "border-touching"),
    ("excl_isolated", "isolated"), ("excl_filtered", "all contacts filtered"),
    ("excl_other", "other (no interface)"),
]
QC_KEYS = ["cells_found", "measured", "with_interfaces", "contrib_dp", "contrib_tcj",
           "not_contributing"] + [k for k, _ in EXCL_COLUMNS]
# Overlapping raw flag counts (a cell can carry several); not exclusion reasons.
FLAG_KEYS = ["border_touching", "isolated", "degenerate", "corner_methods_disagree",
             "all_contacts_filtered", "dp_low_contact", "dp_missed_tcj_contact",
             "closed_loop_interface", "multinucleate", "anucleate"]
LOST_FRACTION = 0.5     # a field where fewer than this share of cells yield interfaces is a warning
IDS_SHOWN = 5           # field ids named in a warning line; the full list is in the detail

# Provenance keys shown in their own group, or not at all: ``config`` and ``thresholds`` are
# shown as resolved rows, and the failure / label keys as run warnings (shown first in the view).
_PROV_SHOWN = {"junctionmapper_version", "code_identity", "library_versions", "params_used",
               "thresholds", "threshold_proposal", "threshold_run_roles", "threshold_calibration",
               "config", "input", "timestamp", "fields_failed", "fields_failed_detail",
               "labels_without_provenance", "labels_source_mismatch", "assets_missing"}


# ---- small helpers -----------------------------------------------------------------

def scrub(v: Any) -> str:
    """Text for display; URLs are defused so the page carries none."""
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return re.sub(r"(?i)\b[a-z][a-z0-9+.-]*://", "", s)


def fmt(x: float | None) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.4g}"


def has_pixel_size(cells: pd.DataFrame, cfg: RunConfig) -> bool:
    if len(cells):
        ps = pd.to_numeric(cells["pixel_size_um"], errors="coerce")
        return bool(((ps > 0) & np.isfinite(ps)).any())
    ps = cfg.pixel_size_um
    return bool(ps) and np.isfinite(ps) and ps > 0


def _ids(ids: list[str]) -> str:
    more = len(ids) - IDS_SHOWN
    return ", ".join(ids[:IDS_SHOWN]) + (f" (+{more} more)" if more > 0 else "")


# ---- QC ----------------------------------------------------------------------------

def qc_counts(cells: pd.DataFrame, flags: pd.DataFrame, nuclei: pd.DataFrame,
              ifc: pd.DataFrame) -> dict[str, int]:
    """Counts for ONE field (cell ids are only unique within a field)."""
    out = dict.fromkeys(QC_KEYS + FLAG_KEYS, 0)
    out["orphan_nuclei"] = int(nuclei["orphan"].sum()) if len(nuclei) else 0
    ids = {int(c) for c in cells["cell_id"].dropna()}
    out["cells_found"] = len(ids)
    out["measured"] = int(cells["measured"].sum()) if len(cells) else 0
    contrib = {m: {int(c) for c in ifc.loc[ifc["corner_method"] == m, "cell_id"].dropna()}
               if len(ifc) else set() for m in METHODS}
    anyc = contrib["dp"] | contrib["tcj"]
    out["contrib_dp"], out["contrib_tcj"] = len(contrib["dp"] & ids), len(contrib["tcj"] & ids)
    out["with_interfaces"] = len(anyc & ids)
    for c in FLAG_KEYS:
        out[c] = int(flags[c].sum()) if len(flags) and c in flags else 0
    fl = {int(r["cell_id"]): r for _, r in flags.iterrows()} if len(flags) else {}
    for cid in sorted(ids - anyc):
        r = fl.get(cid)
        reason = next((k for k, col in (("excl_degenerate", "degenerate"), ("excl_border", "border_touching"),
                                        ("excl_isolated", "isolated"), ("excl_filtered", "all_contacts_filtered"))
                       if r is not None and bool(r[col])), "excl_other")
        out[reason] += 1
        out["not_contributing"] += 1
    return out


def qc(fields: list[str], cells, flags, nuclei, ifc) -> tuple[dict[str, dict[str, int]], list[dict]]:
    """Per-field counts, and the fields where fewer than LOST_FRACTION of the cells contributed."""
    def sel(df, fid):
        return df[df["field_id"] == fid] if len(df) else df
    per = {fid: qc_counts(sel(cells, fid), sel(flags, fid), sel(nuclei, fid), sel(ifc, fid))
           for fid in fields}
    lost = []
    for fid, c in per.items():
        found = c["cells_found"]
        if found and c["with_interfaces"] < LOST_FRACTION * found:
            why = ", ".join(f"{c[k]} {lab}" for k, lab in EXCL_COLUMNS if c[k])
            lost.append({"field": fid, "found": found, "usable": c["with_interfaces"], "why": why})
    return per, lost


# ---- thresholds --------------------------------------------------------------------

def run_roles(provenance: dict) -> "set[str] | None":
    """The marker roles a single run-level threshold was carried for, if the run recorded it.

    ``None`` means nothing was recorded (a library consumer). That is NOT "nothing resolved".
    """
    raw = provenance.get("threshold_run_roles")
    if raw is None:
        return None
    return {s for s in (part.strip() for part in str(raw).split(",")) if s and s != "none"}


def threshold_info(ifc: pd.DataFrame, roles: "set[str] | None" = None) -> dict[str, dict[str, Any]]:
    """Per role: the APPLIED threshold per field, and whether it genuinely varied.

    varied       the applied values span more than ``THRESHOLD_SAME_TOL`` grey levels
    differing    the fields that establish the span: holders of an extreme value, plus any
                 field more than the tolerance from the run value (never empty when varied)
    attribution  "override": a run value WAS carried for the role, so a differing field was set
                 per field; "fallback": it was not, every field fell back to Otsu on its own
                 plane and nobody overrode anything; "unknown": the run did not record which
    field_otsu   per field, Otsu on that field's own plane
    deviating    fields where the two are far enough apart to be worth a look
    """
    info: dict[str, dict[str, Any]] = {}
    if not len(ifc):
        return info
    has_otsu = "threshold_field_otsu" in ifc.columns
    has_dev = "threshold_deviates_from_run" in ifc.columns
    for role in sorted(set(ifc["role"]), key=lambda r: (ROLE_ORDER.get(r, 9), r)):
        g = ifc[ifc["role"] == role]
        by_field = {str(f): sorted({float(t) for t in gf["threshold"].dropna()})
                    for f, gf in g.groupby("field_id")}
        own = {str(f): sorted({float(t) for t in gf["threshold_field_otsu"].dropna()})
               for f, gf in g.groupby("field_id")} if has_otsu else {}
        deviating = sorted({str(f) for f, gf in g.groupby("field_id")
                            if bool(gf["threshold_deviates_from_run"].any())}) if has_dev else []
        distinct = sorted({t for ts in by_field.values() for t in ts})
        varied = bool(distinct) and (distinct[-1] - distinct[0]) > THRESHOLD_SAME_TOL
        run_value = float(np.median([v for ts in by_field.values() for v in ts])) if distinct else None
        extremes = {distinct[0], distinct[-1]} if varied else set()
        differing = [f for f, ts in sorted(by_field.items())
                     if varied and (any(abs(t - run_value) > THRESHOLD_SAME_TOL for t in ts)
                                    or any(t in extremes for t in ts))]
        attribution = ("unknown" if roles is None
                       else "override" if str(role) in roles else "fallback")
        info[str(role)] = {
            "role": str(role), "markers": sorted(set(map(str, g["marker"]))),
            "by_field": by_field, "distinct": distinct, "varied": varied, "run_value": run_value,
            "differing": differing, "attribution": attribution,
            "field_otsu": own, "deviating": deviating,
        }
    return info


# ---- run warnings ------------------------------------------------------------------

_CAUSE = {
    "override": ("per-field override", "A run value was carried for {role}; these fields were set per field: {ids}."),
    "fallback": ("calibration resolved nothing",
                 "No run value was resolved for {role}: each field used Otsu on its own plane. "
                 "Not an override. Fields involved: {ids}."),
    "unknown": ("cause not recorded", "The run did not record whether a run value was carried for "
                                      "{role}. Fields involved: {ids}."),
}


def run_warnings(prov: dict, tinfo: dict[str, dict[str, Any]], lost: list[dict]) -> list[dict[str, str]]:
    """One line per run-level problem: ``{kind, text, detail}`` (plus ``role``/``attribution``
    for a threshold warning). The text is a label; the detail goes in a tooltip."""
    out: list[dict[str, str]] = []
    for t in tinfo.values():
        if not t["varied"]:
            continue
        short, why = _CAUSE[t["attribution"]]
        values = " vs ".join(fmt(x) for x in t["distinct"])
        out.append({
            "kind": "threshold_varied", "role": t["role"], "attribution": t["attribution"],
            "text": f"{t['role']} threshold varied: {short}",
            "detail": (f"Applied {values} ({', '.join(t['markers'])}). "
                       + why.format(role=t["role"], ids=_ids(t["differing"]))
                       + " Threshold-dependent metrics are not comparable across fields."),
        })
    if prov.get("fields_failed"):
        n = int(prov["fields_failed"])
        out.append({"kind": "fields_failed", "text": f"{n} field{'s' if n != 1 else ''} failed",
                    "detail": scrub(prov.get("fields_failed_detail", ""))})
    if lost:
        n = len(lost)
        out.append({"kind": "cells_lost",
                    "text": f"{n} field{'s' if n != 1 else ''}: most cells unmeasured",
                    "detail": "; ".join(f"{w['field']}: {w['usable']} of {w['found']} cells measured"
                                        + (f" ({w['why']})" if w["why"] else "") for w in lost)})
    if prov.get("labels_without_provenance"):
        raw = str(prov["labels_without_provenance"])
        m = re.match(r"(\d+)", raw)
        n = m.group(1) if m else "Some"
        out.append({"kind": "labels_without_provenance",
                    "text": f"{n} field{'s' if n != '1' else ''}: labels without provenance",
                    "detail": scrub(raw)})
    if prov.get("labels_source_mismatch"):
        try:
            items = json.loads(prov["labels_source_mismatch"])
        except (TypeError, ValueError):
            items = []
        n = len(items) or 1
        detail = "; ".join(f"{i.get('field_id')}: labels made from {i.get('labels_made_from')}, "
                           f"applied to {i.get('applied_to')}" for i in items if isinstance(i, dict))
        out.append({"kind": "labels_source_mismatch",
                    "text": f"{n} field{'s' if n != 1 else ''}: labels from another image",
                    "detail": scrub(detail or prov["labels_source_mismatch"])})
    if prov.get("assets_missing"):
        try:
            missing = json.loads(prov["assets_missing"])
        except (TypeError, ValueError):
            missing = {}
        missing = missing if isinstance(missing, dict) else {}
        n = len(missing) or 1
        out.append({"kind": "assets_missing",
                    "text": f"{n} field{'s' if n != 1 else ''}: no image in the report",
                    "detail": scrub("; ".join(f"{k}: {v}" for k, v in sorted(missing.items()))
                                    or prov["assets_missing"])
                              + ". Their measurements are in the tables."})
    return out


# ---- provenance --------------------------------------------------------------------

def _distinct(df: pd.DataFrame, col: str) -> str:
    if not len(df) or col not in df:
        return ""
    return ", ".join(sorted({fmt(float(x)) for x in pd.to_numeric(df[col], errors="coerce").dropna()}))


def provenance_rows(cfg: RunConfig, prov: dict, cells: pd.DataFrame,
                    tinfo: dict[str, dict[str, Any]], n_fields: int) -> list[dict[str, str]]:
    """The provenance view as ``{group, key, value}`` rows, all strings, URLs defused."""
    rows: list[dict[str, str]] = []

    def add(group: str, key: str, value: Any) -> None:
        if value is None or value == "":
            return
        rows.append({"group": group, "key": str(key), "value": scrub(value)})

    params = prov.get("params_used") if isinstance(prov.get("params_used"), dict) else {}
    add("Inputs", "input", prov.get("input"))
    add("Inputs", "fields measured", str(n_fields))
    add("Inputs", "labels", params.get("labels_dir") and f"{params.get('labels_source', 'disk')}: {params['labels_dir']}")
    add("Inputs", "timestamp", prov.get("timestamp"))
    for spec in sorted(cfg.channels, key=lambda c: (ROLE_ORDER.get(c.role.value, 2), c.role.value)):
        add("Markers", spec.role.value, f"{spec.marker} (channel {spec.channel})")

    add("Model", "model", params.get("model", "not recorded"))
    for k in sorted(params, key=str):
        if k not in ("model", "labels_dir", "labels_source"):
            add("Model", k, params[k])

    add("Parameters", "epsilon (px)", _distinct(cells, "epsilon") or fmt(cfg.epsilon_px))
    add("Parameters", "tcj_min_run", _distinct(cells, "tcj_min_run") or str(cfg.tcj_min_run))
    add("Parameters", "neighbour_radius_px", _distinct(cells, "neighbour_radius_px") or str(cfg.neighbour_radius_px))
    add("Parameters", "dilation_cycles", _distinct(cells, "dilation_cycles"))
    add("Parameters", "dilation", f"{fmt(cfg.dilation_um)} µm" if cfg.dilation_um else f"{cfg.dilation_px} px")
    add("Parameters", "pixel size (µm/px)", _distinct(cells, "pixel_size_um") or (fmt(cfg.pixel_size_um) if cfg.pixel_size_um else "not set"))
    add("Parameters", "num_intra_cells", str(cfg.num_intra_cells))
    add("Parameters", "all_pairs", str(cfg.all_pairs).lower())

    for role, v in sorted((cfg.threshold or {}).items(), key=lambda kv: kv[0].value):
        add("Thresholds", f"{role.value} configured", fmt(float(v)))
    add("Thresholds", "proposal", prov.get("threshold_proposal"))
    add("Thresholds", "run value carried for", prov.get("threshold_run_roles"))
    add("Thresholds", "calibration", prov.get("threshold_calibration"))
    for role, t in tinfo.items():
        add("Thresholds", f"{role} applied", " vs ".join(fmt(x) for x in t["distinct"])
            + (" (varied)" if t["varied"] else ""))
        if t["deviating"]:
            add("Thresholds", f"{role} own-Otsu deviates", _ids(t["deviating"]))
        for f, vals in sorted(t["by_field"].items()):
            own = ", ".join(fmt(x) for x in t["field_otsu"].get(f, ()))
            add("Thresholds", f"{role} · {f}", ", ".join(fmt(x) for x in vals) + (f" (own Otsu {own})" if own else ""))

    add("Versions", "junctionmapper", prov.get("junctionmapper_version"))
    add("Versions", "libraries", prov.get("library_versions"))
    add("Code", "code identity", prov.get("code_identity"))

    for k in sorted(prov, key=str):
        if k not in _PROV_SHOWN:
            add("Other", k, prov[k])
    return rows
