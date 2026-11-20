"""Per-field asset files for the interactive report.

Each field becomes one ``<field_id>.js`` calling ``JM.registerField(...)`` with downsampled
channel previews, cell outlines and junction polylines, all in preview pixels ``[x, y]``.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import find_contours

from junctionmapper.config import Role
from junctionmapper.io.reader import Field
from junctionmapper.measure import interface as mi
from junctionmapper.pipeline import FieldResult
from junctionmapper.report.summary import PARAMETERS
from junctionmapper.segment.cellpose import Segmentation


def preview_scale(shape: tuple[int, int], max_side: int = 1500) -> float:
    return min(1.0, max_side / max(shape))


def preview_size(shape: tuple[int, int], scale: float) -> tuple[int, int]:
    """``(width, height)`` of the preview of a ``(rows, cols)`` plane: the size the viewer draws."""
    h, w = shape
    return max(1, round(w * scale)), max(1, round(h * scale))


def preview(plane: np.ndarray, scale: float) -> str:
    """JPEG data URI of ``plane``, stretched 0.5-99.5 percentile to 0-255."""
    a = np.asarray(plane, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size:
        lo, hi = np.percentile(finite, [0.5, 99.5])
    else:
        lo = hi = 0.0
    if finite.size and hi > lo:
        u8 = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1) * 255
    else:
        u8 = np.full(a.shape, 128.0)
    img = Image.fromarray(u8.round().astype(np.uint8), "L")
    size = preview_size(a.shape, scale)
    if size != img.size:
        img = img.resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def simplify(path: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker on an open polyline; endpoints are kept."""
    p = np.asarray(path)
    n = len(p)
    if n < 3:
        return p.copy()
    keep = np.zeros(n, bool)
    keep[[0, n - 1]] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i].astype(float), p[j].astype(float)
        seg = p[i + 1:j].astype(float)
        d = b - a
        norm = np.hypot(*d)
        if norm == 0:
            dist = np.hypot(*(seg - a).T)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - a[1]) - d[1] * (seg[:, 0] - a[0])) / norm
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return p[keep]


def to_preview(path_rc: np.ndarray, scale: float) -> list[list[float]]:
    """``(row, col)`` pixel indices to ``[x, y]`` preview pixels, rounded to 0.5.

    Pixel ``(r, c)`` covers ``[c, c+1] x [r, r+1]`` of the drawn image, so its centre is
    ``(c + 0.5, r + 0.5)``; without the half pixel every line sits up and left of its membrane.
    """
    xy = (np.asarray(path_rc, dtype=float)[:, ::-1] + 0.5) * scale
    return (np.round(xy * 2) / 2).tolist()


def cell_outlines(labels: np.ndarray, scale: float, tol: float = 1.0) -> dict[str, list]:
    """Longest outer contour of every non-zero label, keyed by cell id."""
    out: dict[str, list] = {}
    for cid, sl in enumerate(ndimage.find_objects(labels.astype(np.int64)), start=1):
        if sl is None:
            continue
        mask = np.pad(labels[sl] == cid, 1)
        contours = find_contours(mask.astype(float), 0.5)
        if not contours:
            continue
        c = max(contours, key=len) - 1 + np.array([sl[0].start, sl[1].start])
        out[str(cid)] = to_preview(simplify(c, tol), scale)
    return out


def junction_polylines(result: FieldResult, scale: float, tol: float = 1.0
                       ) -> dict[str, dict[str, list]]:
    """``{"tcj"|"dp": {"<lo>-<hi>": [polyline, ...]}}`` grouped by pair_id."""
    out: dict[str, dict[str, list]] = {"tcj": {}, "dp": {}}
    itf = result.interfaces
    if len(itf):
        rows = itf.drop_duplicates("interface_id")
        for iid, pair, method in zip(rows.interface_id, rows.pair_id, rows.corner_method):
            path = result.paths.get(iid)
            if path is None or method not in out:
                continue
            key = str(pair).split("|")[-1]
            out[method].setdefault(key, []).append(to_preview(simplify(path, tol), scale))
    return out


def _num(v) -> float | None:
    v = float(v)
    return None if not np.isfinite(v) else v


def junction_sides(result: FieldResult) -> dict:
    """``{method: {"<lo>-<hi>": {role: {cell_id: {metric: [px, um|None]}}}}}``, one entry per cell side.

    A side is the mean of that cell's rows for the pair (NaN-skipping); NaN becomes None (JSON null).
    """
    out: dict = {}
    itf = result.interfaces
    if not len(itf):
        return out
    cols = [c for p in PARAMETERS for c in (p["px"], p["um"]) if c and c in itf]
    vals = itf[["corner_method", "role", "pair_id", "cell_id"]].copy()
    for c in cols:
        vals[c] = itf[c].astype(float)
    grouped = vals.groupby(["corner_method", "role", "pair_id", "cell_id"], sort=True)[cols].mean()
    for (method, role, pair, cell), row in grouped.iterrows():
        key = str(pair).split("|")[-1]
        side = out.setdefault(str(method), {}).setdefault(key, {}).setdefault(str(role), {})
        side[str(int(cell))] = {
            p["key"]: [_num(row[p["px"]]) if p["px"] in row else None,
                       _num(row[p["um"]]) if p["um"] and p["um"] in row else None]
            for p in PARAMETERS}
    return out


def junction_profile(path_rc: np.ndarray, image: np.ndarray, band: np.ndarray, radius: int,
                     px_um: float | None, max_points: int = 200) -> dict:
    """Marker brightness along a junction: ``{"x": [...], "y": [... | None]}``.

    ``y[i]`` is the mean of ``image`` over the ``band`` pixels inside the ``(2*radius+1)`` square
    window centred on path pixel ``i`` (the window is cut at the image border); None where the
    window holds no band pixel. ``x`` is the path length so far, a diagonal step counting
    sqrt(2), in micrometres when ``px_um`` is given, else in pixels. A path longer than
    ``max_points`` is resampled by linear interpolation, ends kept.
    """
    path = np.asarray(path_rc, dtype=int)
    img = np.asarray(image, dtype=np.float64)
    # A NaN or inf pixel is no sample: left out of both the sum and the count.
    b = np.asarray(band, dtype=bool) & np.isfinite(img)
    img = np.where(b, img, 0.0)
    size = 2 * int(radius) + 1
    num = ndimage.uniform_filter(img, size=size, mode="constant")
    den = ndimage.uniform_filter(b.astype(np.float64), size=size, mode="constant")
    r, c = path[:, 0], path[:, 1]
    d = den[r, c]
    y = np.where(d > 1e-9, num[r, c] / np.where(d > 1e-9, d, 1.0), np.nan)
    step = np.hypot(*np.diff(path, axis=0).T) if len(path) > 1 else np.zeros(0)
    x = np.concatenate([[0.0], np.cumsum(step)])
    if len(path) == 1:
        x = np.array([0.0, 0.0])
        y = np.array([y[0], y[0]])
    if px_um:
        x = x * float(px_um)
    if len(x) > max_points:
        xs = np.linspace(x[0], x[-1], max_points)
        ok = np.isfinite(y)
        if ok.any():
            ys = np.interp(xs, x[ok], y[ok])
            nearest = np.abs(xs[:, None] - x[None, :]).argmin(axis=1) if len(x) * max_points < 4e6 \
                else np.clip(np.searchsorted(x, xs), 0, len(x) - 1)
            ys[~ok[nearest]] = np.nan
        else:
            ys = np.full(max_points, np.nan)
        x, y = xs, ys
    return {"x": [round(float(v), 4) for v in x],
            "y": [None if not np.isfinite(v) else round(float(v), 4) for v in y]}


def band_crop(path_rc: np.ndarray, shape: tuple[int, int], cycles: int):
    """``(slices, path_in_crop, band_in_crop)``: the junction's band built on a sub-array.

    The crop is the path's bounding box grown by ``2*cycles`` (band dilation plus profile window
    radius) and clamped to the image, so profiles read from it equal those from the full image.
    """
    path = np.asarray(path_rc, dtype=int)
    m = 2 * int(cycles)
    lo = np.maximum(path.min(axis=0) - m, 0)
    hi = np.minimum(path.max(axis=0) + m + 1, shape)
    sl = (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])))
    local = path - lo
    return sl, local, mi.dilated_band(local, (int(hi[0] - lo[0]), int(hi[1] - lo[1])), int(cycles))


def walk_order(pieces: list) -> list:
    """``pieces`` (``(start_idx, end_idx, ...)`` tuples of one cell side) in boundary walk order.

    Sorted by ``start_idx``, then rotated to begin at the first piece that does not continue
    another (its start is no piece's end), so a run of pieces through boundary index 0 stays
    in order.
    """
    ordered = sorted(pieces, key=lambda p: p[0])
    ends = {p[1] for p in ordered}
    first = next((i for i, p in enumerate(ordered) if p[0] not in ends), 0)
    return ordered[first:] + ordered[:first]


def junction_profiles(field: Field, result: FieldResult, scale: float = 1.0,
                      max_points: int = 200) -> tuple[dict, dict]:
    """``(profiles, starts)`` for every junction, along one cell side's walk.

    ``profiles``: ``{method: {"<lo>-<hi>": {role: profile}}}``; roles are every plane the field
    has among JM1, JM2 and segment. ``starts``: ``{method: {"<lo>-<hi>": [x, y]}}``, the walk's
    first pixel in preview px (``scale``), i.e. where the profile's x = 0 sits.

    The side is the lower-id cell's, or the higher-id cell's when the lower has no measured
    interface for the pair. All of that side's interfaces for the pair (dp splits one contact
    into several) are joined in walk order into one profile: x runs on cumulatively and a None y
    at each join breaks the line at the corner. Points are shared out by piece length.
    """
    profiles: dict = {}
    starts: dict = {}
    itf = result.interfaces
    if not len(itf):
        return profiles, starts
    roles = [r for r in (Role.JM1, Role.JM2, Role.SEGMENT) if r in field.planes]
    if not roles:
        return profiles, starts
    shape = field.planes[roles[0]].shape
    cycles = dict(zip(result.cells["cell_id"].astype(int), result.cells["dilation_cycles"]))
    px = field.pixel_size_um if field.pixel_size_um and np.isfinite(field.pixel_size_um) else None
    rows = itf.drop_duplicates("interface_id")
    sides: dict = {}
    for iid, pair, method, cid, s0, e0 in zip(rows.interface_id, rows.pair_id, rows.corner_method,
                                              rows.cell_id, rows.start_idx, rows.end_idx):
        path = result.paths.get(iid)
        cyc = cycles.get(int(cid))
        if path is None or len(path) == 0 or cyc is None or not np.isfinite(cyc):
            continue
        key = (str(method), str(pair).split("|")[-1])
        sides.setdefault(key, {}).setdefault(int(cid), []).append((int(s0), int(e0), np.asarray(path)))
    for (method, key), by_cell in sides.items():
        lo, hi = (int(v) for v in key.split("-"))
        cid = lo if lo in by_cell else hi
        pieces = walk_order(by_cell[cid])
        cyc = int(cycles[cid])
        total = sum(len(p[2]) for p in pieces)
        joined = {role.value: {"x": [], "y": []} for role in roles}
        offset = 0.0
        for n, (_, _, path) in enumerate(pieces):
            sl, local, band = band_crop(path, shape, cyc)
            budget = max(2, round(max_points * len(path) / total))
            end = offset
            for role in roles:
                prof = junction_profile(local, field.planes[role][sl], band, cyc, px, budget)
                out = joined[role.value]
                if n:
                    out["x"].append(round(offset, 4))
                    out["y"].append(None)
                out["x"].extend(round(offset + v, 4) for v in prof["x"])
                out["y"].extend(prof["y"])
                end = offset + prof["x"][-1]
            offset = end
        profiles.setdefault(method, {})[key] = joined
        starts.setdefault(method, {})[key] = to_preview(pieces[0][2][:1], scale)[0]
    return profiles, starts


def write_field_asset(field: Field, seg: Segmentation, result: FieldResult, run: str,
                      out_dir: Path, thresholds: dict[str, float] | None = None) -> Path:
    labels = seg.cells
    h, w = labels.shape
    scale = preview_scale((h, w))
    pw, ph = preview_size((h, w), scale)
    channels = [{"role": role.value, "marker": field.markers.get(role, ""),
                 "src": preview(plane, scale)} for role, plane in field.planes.items()]
    flags: dict[str, list[str]] = {}
    fl = result.flags
    bool_cols = [c for c in fl.columns if fl[c].dtype == bool]
    for _, row in fl.iterrows():
        on = [c for c in bool_cols if row[c]]
        if on:
            flags[str(int(row["cell_id"]))] = on
    try:
        profiles, starts = junction_profiles(field, result, scale)
    except Exception as e:  # a profile is a figure, never worth losing the field's images for
        print(f"junctionmapper report: warning: field {field.field_id}: junction profiles "
              f"skipped ({type(e).__name__}: {e})", file=sys.stderr)
        profiles, starts = {}, {}
    data = {
        # width/height: the preview, which every coordinate is in; raw_*: the label image.
        "run": run, "field_id": field.field_id, "width": pw, "height": ph, "scale": scale,
        "raw_width": w, "raw_height": h,
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "channels": channels, "cells": cell_outlines(labels, scale),
        "junctions": junction_polylines(result, scale), "sides": junction_sides(result),
        "flags": flags,
        "profiles": profiles, "profile_start": starts,
        "thresholds": {str(k): _num(v) for k, v in (thresholds or {}).items()},
        "profile_unit": "µm" if field.pixel_size_um and np.isfinite(field.pixel_size_um) else "px",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{field.field_id}.js"
    path.write_text("JM.registerField(" + json.dumps(data, separators=(",", ":"), allow_nan=False) + ");\n")
    return path
