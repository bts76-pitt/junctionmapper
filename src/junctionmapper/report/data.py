"""Junction- and cell-level metric files, ``run.json`` and the run summary for the interactive report.

A junction is one (role, pair_id) for one corner method. Its value is the mean over the cell-side
rows that share the pair_id. A cell value is the median over that cell's rows. NaN is skipped, and
a group that is all NaN stays NaN (never 0).
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from junctionmapper.config import RunConfig
from junctionmapper.report.build import PARAMETERS, _has_pixel_size

RUN = "r1"          # the run id inside a run directory
THUMB_PX = 240
_VALUE_COLS = [c for p in PARAMETERS for c in (p["px"], p["um"]) if c]
_J_KEYS = ["role", "field_id", "lo", "hi"]
_C_KEYS = ["role", "field_id", "cell_id"]
# Records that the TCJ and DP corners differ, not a failure; it fires on nearly every cell, so it
# is counted but does not make a cell "flagged".
NOT_FLAGGING = {"corner_methods_disagree"}
# Only unmeasured cells carry these, and the summary counts measured cells only.
UNMEASURED_ONLY = {"border_touching", "isolated", "degenerate"}


def b64_f32(a) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode("ascii")


def b64_u32(a) -> str:
    return base64.b64encode(np.asarray(a, dtype="<u4").tobytes()).decode("ascii")


def _values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in _VALUE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce") if c in out else np.nan
    return out


def junction_table(ifc: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per (role, pair_id): ``field_id, lo, hi, role`` and the mean of every metric column."""
    d = _values(ifc[ifc["corner_method"] == method])
    cols = ["field_id", "lo", "hi", "role"] + _VALUE_COLS
    if d.empty:
        return pd.DataFrame(columns=cols)
    span = d["pair_id"].astype(str).str.split("|").str[-1].str.split("-", n=1, expand=True)
    d["lo"], d["hi"] = span[0].astype(int), span[1].astype(int)
    g = d.groupby(["role", "pair_id"], sort=False)
    out = g[_VALUE_COLS].mean()
    first = g[["field_id", "lo", "hi"]].first()
    out = pd.concat([first, out], axis=1).reset_index()
    return out.sort_values(["role", "field_id", "lo", "hi"], kind="stable").reset_index(drop=True)[cols]


def cell_table(ifc: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per (role, field_id, cell_id): the median of every metric column."""
    d = _values(ifc[ifc["corner_method"] == method])
    cols = ["field_id", "cell_id", "role"] + _VALUE_COLS
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["cell_id"] = d["cell_id"].astype(int)
    out = d.groupby(["role", "field_id", "cell_id"], sort=True)[_VALUE_COLS].median().reset_index()
    return out[cols]


def _register(path: Path, kind: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"JM.register{kind}(" + json.dumps(payload, separators=(",", ":")) + ");\n")


def _aligned(table: pd.DataFrame, role: str, index: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The role's rows laid out in the method-wide index order; absent rows are NaN."""
    mine = table[table["role"] == role].drop(columns="role").set_index(keys)
    return mine.reindex(pd.MultiIndex.from_frame(index[keys]))


def write_metric_files(ifc: pd.DataFrame, run: str, out_dir: Path, calibrated: bool) -> None:
    """``out_dir/<method>/_index.js`` and ``out_dir/<method>/<role>/<key>.js`` for every metric.

    Row order, shared by all files of a method: junctions by (field_id, lo, hi), cells by
    (field_id, cell_id). The um arrays are ``null`` for an uncalibrated run or a unit-free metric.
    """
    out_dir = Path(out_dir)
    for method in sorted(ifc["corner_method"].dropna().unique()):
        jt, ct = junction_table(ifc, method), cell_table(ifc, method)
        jidx = jt[["field_id", "lo", "hi"]].drop_duplicates().sort_values(
            ["field_id", "lo", "hi"], kind="stable").reset_index(drop=True)
        cidx = ct[["field_id", "cell_id"]].drop_duplicates().sort_values(
            ["field_id", "cell_id"], kind="stable").reset_index(drop=True)
        fields = sorted(set(jidx.field_id) | set(cidx.field_id))
        pos = {f: i for i, f in enumerate(fields)}
        _register(out_dir / method / "_index.js", "Index", {
            "run": run, "method": method, "fields": fields,
            "junction_field": b64_u32([pos[f] for f in jidx.field_id]),
            "junction_lo": b64_u32(jidx.lo), "junction_hi": b64_u32(jidx.hi),
            "cell_field": b64_u32([pos[f] for f in cidx.field_id]),
            "cell_id": b64_u32(cidx.cell_id)})
        for role in sorted(jt["role"].unique(), key=lambda r: (r != "JM1", r)):
            jr = _aligned(jt, role, jidx, ["field_id", "lo", "hi"])
            cr = _aligned(ct, role, cidx, ["field_id", "cell_id"])
            for p in PARAMETERS:
                with_um = calibrated and p["um"] is not None
                _register(out_dir / method / role / f"{p['key']}.js", "Metric", {
                    "run": run, "method": method, "role": role, "key": p["key"],
                    "junction_px": b64_f32(jr[p["px"]]),
                    "junction_um": b64_f32(jr[p["um"]]) if with_um else None,
                    "cell_px": b64_f32(cr[p["px"]]),
                    "cell_um": b64_f32(cr[p["um"]]) if with_um else None})


def write_run_json(out: Path, cfg: RunConfig, prov: dict, fields: list[str]) -> Path:
    """``<out>/run.json``: config, provenance and the field ids of the run."""
    path = Path(out) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"config": cfg.model_dump(mode="json"), "provenance": prov, "fields": list(fields)}
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    return path


def _unit(col: str | None) -> str:
    """Unit string of a metric column; unit-free and intensity metrics have none."""
    if col is None or col.startswith(("intensity", "cluster")):
        return ""
    if col.endswith("_um2"):
        return "µm²"
    if col.endswith("_um"):
        return "µm"
    if col.endswith("_px"):
        return "px²" if "area" in col else "px"
    return ""


def _thumb(asset: Path) -> str:
    """First channel preview of a field asset, re-encoded as a JPEG data URI of 240 px long side."""
    if not asset.exists():
        return ""
    text = asset.read_text()
    data = json.loads(text[text.index("(") + 1:text.rindex(");")])
    if not data.get("channels"):
        return ""
    src = data["channels"][0]["src"]
    img = Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1]))).convert("L")
    k = THUMB_PX / max(img.size)
    img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def write_thumbs(run_dir: Path) -> Path:
    """``<run_dir>/report/runs/r1/thumbs.js``: one JPEG thumbnail per field of the run.

    Kept out of ``index.html`` so the page stays small however many fields a run has.
    """
    run_dir = Path(run_dir)
    fields = json.loads((run_dir / "run.json").read_text())["fields"]
    base = run_dir / "report" / "runs" / RUN
    thumbs = {fid: _thumb(base / "fields" / f"{fid}.js") for fid in fields}
    path = base / "thumbs.js"
    _register(path, "Thumbs", thumbs)
    return path


def run_summary(run_dir: Path, run: str, label: str) -> dict:
    """The run summary object: everything ``index.html`` needs, none of the per-junction values."""
    run_dir = Path(run_dir)
    doc = json.loads((run_dir / "run.json").read_text())
    cfg = RunConfig.model_validate(doc["config"])
    ifc = pd.read_parquet(run_dir / "interfaces.parquet")
    cells = pd.read_parquet(run_dir / "cells.parquet")
    flags = pd.read_parquet(run_dir / "flags.parquet")
    calibrated = _has_pixel_size(cells, cfg)
    # Only measured cells (at least one interface row) count, the denominator of tiles.cells.
    measured_cells = ifc[["field_id", "cell_id"]].drop_duplicates().astype({"cell_id": "int64"})
    flags = flags.dropna(subset=["cell_id"]).astype({"cell_id": "int64"}).merge(
        measured_cells, on=["field_id", "cell_id"], how="inner")
    bool_cols = [c for c in flags.columns
                 if pd.api.types.is_bool_dtype(flags[c]) and c not in UNMEASURED_ONLY]
    flagging = [c for c in bool_cols if c not in NOT_FLAGGING]
    any_flag = flags[flagging].any(axis=1) if flagging else pd.Series(False, index=flags.index)
    tcj = ifc[ifc["corner_method"] == "tcj"]
    fields = []
    for fid in doc["fields"]:
        fl = flags[flags["field_id"] == fid]
        qc = {"flagged_cells": int(any_flag[fl.index].sum())}
        qc.update({c: int(fl[c].sum()) for c in bool_cols})
        fields.append({
            "field_id": fid,
            "n_cells": int(ifc.loc[ifc["field_id"] == fid, "cell_id"].nunique()),
            "n_junctions": int(tcj.loc[tcj["field_id"] == fid, "pair_id"].nunique()),
            "qc": qc,
        })
    measured = {"JM1", "JM2"}
    return {
        "run": run, "label": label,
        "roles": [{"role": c.role.value, "marker": c.marker}
                  for c in cfg.channels if c.role.value in measured],
        "methods": sorted(ifc["corner_method"].dropna().unique().tolist()),
        "calibrated": calibrated,
        "metrics": [{"key": p["key"], "label": p["label"], "unit_px": _unit(p["px"]),
                     "unit_um": _unit(p["um"]), "td": p["td"], "help": p["help"]}
                    for p in PARAMETERS],
        "fields": fields,
        "tiles": {"cells": int(ifc[["field_id", "cell_id"]].drop_duplicates().shape[0]),
                  "junctions": int(tcj["pair_id"].nunique()),
                  "fields": len(fields),
                  "flagged_cells": int(any_flag.sum())},
        "provenance": doc["provenance"],
    }
