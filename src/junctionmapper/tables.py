"""Tidy table schemas and writers.

Five tables per run, all keyed by ``field_id`` (+ ``cell_id`` / ``nucleus_id``) so several
fields concatenate into one table each. Schemas live here so an empty table still has its
full column set.

Units and formulas (``s`` = ``pixel_size_um``):
  * ``*_um``  = ``*_px * s`` for lengths; ``*_um2`` = ``*_px * s**2`` for areas.
  * ``linearity_index``  = straight-line length / contour, <= 1 (1 = straight).
  * ``coverage_index``   = fragmented contour / interface contour.
  * ``interface_occupancy_pct`` = 100 * marker area / interface area (pixel counts, unitless).
  * ``marker_intensity`` = sum of above-threshold values in the band (raw intensity units).
  * ``intensity_per_px2``         = marker_intensity / interface_area_px   (per px^2).
  * ``intensity_per_um2``         = intensity_per_px2 / s**2.
  * ``cluster_density_per_px2``   = marker_intensity / marker_area_px     (per px^2).
  * ``cluster_density_per_um2``   = cluster_density_per_px2 / s**2.
  * ``n_neighbours`` = distinct non-zero neighbour ids over both corner methods.
  * nuclei summary: ``n_intra_distances`` and ``mean_intra_distance_*`` are the count and
    mean of that nucleus's rows in ``nuclei_distances`` (with ``all_pairs`` each pair is
    emitted once, under the lower nucleus id).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from junctionmapper.errors import ConfigError

# Length columns have _px and _um twins; areas have _px and _um2 (pixel size squared).
# Intensity sums, per-pixel intensity ratios and dimensionless ratios are unscaled.
INTERFACE_COLUMNS = [
    "field_id", "cell_id", "neighbour_id", "corner_method", "role", "marker",
    "start_idx", "end_idx", "threshold", "dilation_cycles",
    "interface_contour_px", "interface_contour_um",
    "straight_line_length_px", "straight_line_length_um",
    "linearity_index",  # straight / contour, <= 1 (inverse of the Java distance ratio)
    "fragmented_contour_px", "fragmented_contour_um",
    "junction_contour_px", "junction_contour_um",
    "junction_straight_line_px", "junction_straight_line_um",
    "coverage_index",
    "interface_area_px", "interface_area_um2",
    "marker_area_px", "marker_area_um2",
    "marker_intensity", "interface_occupancy_pct",
    "intensity_per_px2", "intensity_per_um2",
    "cluster_density_per_px2", "cluster_density_per_um2",
]
CELL_COLUMNS = [
    "field_id", "cell_id", "area_px", "area_um2", "measured",
    "n_neighbours", "n_interfaces_dp", "n_interfaces_tcj",
    "n_nuclei", "epsilon", "tcj_min_run", "neighbour_radius_px",
    "dilation_cycles", "pixel_size_um",
]
NUCLEUS_COLUMNS = [
    "field_id", "nucleus_id", "cell_id", "orphan", "area_px", "area_um2",
    "centroid_row", "centroid_col",
    "n_intra_distances", "mean_intra_distance_px", "mean_intra_distance_um",
]
NUCLEUS_DISTANCE_COLUMNS = [
    "field_id", "nucleus_id", "neighbour_nucleus_id", "rank", "distance_px", "distance_um",
]
FLAG_COLUMNS = [
    "field_id", "cell_id", "border_touching", "isolated", "degenerate",
    "corner_methods_disagree", "n_corners_dp", "n_corners_tcj", "all_contacts_filtered",
    "multinucleate", "anucleate",
]
SCHEMAS = {
    "interfaces": INTERFACE_COLUMNS,
    "cells": CELL_COLUMNS,
    "nuclei": NUCLEUS_COLUMNS,
    "flags": FLAG_COLUMNS,
    "nuclei_distances": NUCLEUS_DISTANCE_COLUMNS,
}
_NULLABLE_INT = ["neighbour_id", "n_corners_dp", "n_corners_tcj", "cell_id"]
_BOOL = ["border_touching", "isolated", "degenerate", "corner_methods_disagree",
         "all_contacts_filtered", "multinucleate", "anucleate", "measured", "orphan"]
_INT = ["start_idx", "end_idx", "dilation_cycles", "interface_area_px", "marker_area_px",
        "area_px", "n_neighbours", "n_interfaces_dp", "n_interfaces_tcj", "n_nuclei", "tcj_min_run",
        "neighbour_radius_px", "neighbour_nucleus_id", "rank",
        "nucleus_id", "centroid_row", "centroid_col", "n_intra_distances"]
_STR = ["field_id", "corner_method", "role", "marker"]


def make_frame(name: str, rows: list[dict]) -> pd.DataFrame:
    """DataFrame with exactly ``SCHEMAS[name]`` columns and stable dtypes, even if empty."""
    cols = SCHEMAS[name]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c in _STR:
            df[c] = df[c].astype("object")
        elif c in _BOOL:
            df[c] = df[c].astype("bool")
        elif c in _NULLABLE_INT or c in _INT:
            df[c] = df[c].astype("Int64")
        else:
            df[c] = df[c].astype("float64")
    return df


def write_tables(results: Iterable, outdir) -> None:
    """Write every table in ``SCHEMAS`` as ``<name>.parquet`` and ``<name>.csv`` in ``outdir``.

    Raises ConfigError if two results share a ``field_id`` (their rows would merge silently).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = list(results)
    seen: set = set()
    for r in results:
        mine = {f for name in SCHEMAS for f in getattr(r, name)["field_id"].dropna()}
        if seen & mine:
            raise ConfigError(f"duplicate field_id across results: {sorted(seen & mine)}")
        seen |= mine
    for name, cols in SCHEMAS.items():
        frames = [getattr(r, name) for r in results if len(getattr(r, name))]
        df = pd.concat(frames, ignore_index=True) if frames else make_frame(name, [])
        df = df[cols]
        df.to_parquet(outdir / f"{name}.parquet", index=False)
        df.to_csv(outdir / f"{name}.csv", index=False)
