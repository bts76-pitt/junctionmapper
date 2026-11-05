"""Run the measurement over one field and emit tidy tables.

Core library functions return pixels; this module is where scaling to micrometres happens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from junctionmapper.config import Role, RunConfig
from junctionmapper.errors import ConfigError
from junctionmapper.geometry.interfaces import extract_interfaces
from junctionmapper.io.reader import Field
from junctionmapper.measure import interface as mi
from junctionmapper.measure.nuclei import associate, intra_distances, nucleus_centroids
from junctionmapper.qc import assess_cell
from junctionmapper.segment.cellpose import Segmentation
from junctionmapper.tables import make_frame

# Geometry-layer constants that RunConfig records but the geometry functions do not yet accept.
_GEOMETRY_TCJ_MIN_RUN = 3
_GEOMETRY_NEIGHBOUR_RADIUS = 2
METHODS = ("dp", "tcj")


@dataclass
class FieldResult:
    interfaces: pd.DataFrame
    cells: pd.DataFrame
    nuclei: pd.DataFrame
    flags: pd.DataFrame
    nuclei_distances: pd.DataFrame


def resolve_thresholds(field: Field, cfg: RunConfig) -> dict[Role, float]:
    """Threshold T per marker role present in ``field``.

    ``cfg.threshold[role]`` wins. Otherwise Otsu on that role's plane; a flat, empty or
    non-finite plane (Otsu cannot run) falls back to 0.0.
    """
    out: dict[Role, float] = {}
    for role in (Role.JM1, Role.JM2):
        plane = field.planes.get(role)
        if plane is None:
            continue
        if cfg.threshold and role in cfg.threshold:
            out[role] = float(cfg.threshold[role])
            continue
        vals = np.asarray(plane, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or vals.min() == vals.max():
            out[role] = 0.0
            continue
        try:
            out[role] = float(threshold_otsu(vals))
        except ValueError:
            out[role] = 0.0
    return out


def measure_field(field: Field, seg: Segmentation, cfg: RunConfig,
                  epsilon: float | None = None) -> FieldResult:
    """Measure every non-border, non-degenerate cell of ``field``.

    Both corner methods are run; JM1 and (if present) JM2 are measured identically.
    The Douglas-Peucker tolerance is ``cfg.epsilon_px`` unless ``epsilon`` is passed, and the
    value used is recorded in ``cells.epsilon``. ``cfg.tcj_min_run`` and
    ``cfg.neighbour_radius_px`` are recorded too; the geometry layer fixes them at 3 and 2, so
    other values raise ConfigError rather than being silently ignored.
    """
    if (cfg.tcj_min_run, cfg.neighbour_radius_px) != (_GEOMETRY_TCJ_MIN_RUN, _GEOMETRY_NEIGHBOUR_RADIUS):
        raise ConfigError(
            "tcj_min_run/neighbour_radius_px other than 3/2 are not supported by the geometry "
            f"layer yet (got {cfg.tcj_min_run}/{cfg.neighbour_radius_px})")
    if epsilon is None:
        epsilon = cfg.epsilon_px
    labels = np.asarray(seg.cells)
    nuc_labels = np.asarray(seg.nuclei)
    if labels.ndim != 2:
        raise ConfigError(f"cell labels must be 2-D, got shape {labels.shape}")
    for role, plane in field.planes.items():
        if np.shape(plane) != labels.shape:
            raise ConfigError(f"{role.value} plane shape {np.shape(plane)} != label shape {labels.shape}")
    px = float(field.pixel_size_um)
    fid = field.field_id
    cycles = cfg.resolved_dilation_px(px)
    thresholds = resolve_thresholds(field, cfg)
    roles = [r for r in (Role.JM1, Role.JM2) if r in field.planes and r in thresholds
             and any(c.role == r for c in cfg.channels)]

    if nuc_labels.shape != labels.shape:
        raise ConfigError(f"nucleus labels shape {nuc_labels.shape} != cell labels {labels.shape}")
    mapping, nflags = associate(labels, nuc_labels)
    nuc_of = {n: c for c, ns in mapping.items() for n in ns}

    iface_rows, cell_rows, flag_rows = [], [], []
    for cid in (int(c) for c in np.unique(labels) if c != 0):
        fl = assess_cell(labels, cid, epsilon)
        area = int((labels == cid).sum())
        counts = {m: 0 for m in METHODS}
        neighbours: set[int] = set()
        measured = not (fl.border_touching or fl.degenerate)
        if measured:
            for method in METHODS:
                for itf in extract_interfaces(labels, cid, method, epsilon):
                    counts[method] += 1
                    neighbours.add(int(itf.neighbour_id))
                    pix = itf.pixels
                    contour = mi.interface_contour(pix)
                    straight = mi.straight_line_length(pix)
                    lin = mi.linearity_index(pix)
                    band = mi.dilated_band(pix, labels.shape, cycles)
                    for role in roles:
                        img, t = field.planes[role], thresholds[role]
                        bm = mi.band_metrics(band, img, t)
                        frag = mi.fragmented_contour(pix, img, t)
                        jc = mi.junction_contour(pix, img, t)
                        jsl = mi.junction_straight_line(pix, img, t)
                        iface_rows.append({
                            "field_id": fid, "cell_id": cid, "neighbour_id": itf.neighbour_id,
                            "corner_method": itf.method, "role": role.value,
                            "marker": cfg.marker_for(role),
                            "start_idx": itf.start_idx, "end_idx": itf.end_idx,
                            "threshold": t, "dilation_cycles": cycles,
                            "interface_contour_px": contour, "interface_contour_um": contour * px,
                            "straight_line_length_px": straight,
                            "straight_line_length_um": straight * px,
                            "linearity_index": lin,
                            "fragmented_contour_px": frag, "fragmented_contour_um": frag * px,
                            "junction_contour_px": jc, "junction_contour_um": jc * px,
                            "junction_straight_line_px": jsl,
                            "junction_straight_line_um": jsl * px,
                            "coverage_index": mi.coverage_index(pix, img, t),
                            "interface_area_px": bm.interface_area_px,
                            "interface_area_um2": bm.interface_area_px * px * px,
                            "marker_area_px": bm.marker_area_px,
                            "marker_area_um2": bm.marker_area_px * px * px,
                            "marker_intensity": bm.marker_intensity,
                            "interface_occupancy_pct": bm.interface_occupancy_pct,
                            "intensity_per_px2": bm.intensity_per_area,
                            "intensity_per_um2": bm.intensity_per_area / (px * px),
                            "cluster_density_per_px2": bm.cluster_density,
                            "cluster_density_per_um2": bm.cluster_density / (px * px),
                        })
        cell_rows.append({
            "field_id": fid, "cell_id": cid, "area_px": area, "area_um2": area * px * px,
            "measured": measured, "n_neighbours": len(neighbours),
            "n_interfaces_dp": counts["dp"], "n_interfaces_tcj": counts["tcj"],
            "n_nuclei": len(mapping.get(cid, [])), "epsilon": float(epsilon),
            "tcj_min_run": cfg.tcj_min_run, "neighbour_radius_px": cfg.neighbour_radius_px,
            "dilation_cycles": cycles, "pixel_size_um": px,
        })
        flag_rows.append({
            "field_id": fid, "cell_id": cid, "border_touching": fl.border_touching,
            "isolated": fl.isolated, "degenerate": fl.degenerate,
            "corner_methods_disagree": fl.corner_methods_disagree,
            "n_corners_dp": fl.n_corners_dp, "n_corners_tcj": fl.n_corners_tcj,
            "all_contacts_filtered": fl.all_contacts_filtered,
            "multinucleate": cid in nflags.multinucleate,
            "anucleate": cid in nflags.anucleate_cells,
        })

    # Nuclei: intra-nuclear distances are summarised per nucleus (mean over its pairs).
    cents = nucleus_centroids(nuc_labels)
    pairs: dict[int, list[tuple[float, int]]] = {}
    for i, j, d in intra_distances(cents, cfg.num_intra_cells, cfg.all_pairs):
        pairs.setdefault(i, []).append((d, j))
    dists: dict[int, list[float]] = {}
    dist_rows = []
    for i in sorted(pairs):
        for rank, (d, j) in enumerate(sorted(pairs[i]), start=1):     # nearest first
            dists.setdefault(i, []).append(d)
            dist_rows.append({
                "field_id": fid, "nucleus_id": i, "neighbour_nucleus_id": j,
                "rank": rank, "distance_px": d, "distance_um": d * px,
            })
    nuc_rows = []
    for nid, (r, c) in cents.items():
        ds = dists.get(nid, [])
        mean = float(np.mean(ds)) if ds else float("nan")
        area = int((nuc_labels == nid).sum())
        nuc_rows.append({
            "field_id": fid, "nucleus_id": nid, "cell_id": nuc_of.get(nid),
            "orphan": nid not in nuc_of, "area_px": area, "area_um2": area * px * px,
            "centroid_row": r, "centroid_col": c, "n_intra_distances": len(ds),
            "mean_intra_distance_px": mean, "mean_intra_distance_um": mean * px,
        })

    return FieldResult(
        interfaces=make_frame("interfaces", iface_rows), cells=make_frame("cells", cell_rows),
        nuclei=make_frame("nuclei", nuc_rows), flags=make_frame("flags", flag_rows),
        nuclei_distances=make_frame("nuclei_distances", dist_rows),
    )
