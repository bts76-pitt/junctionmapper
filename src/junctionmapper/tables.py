"""Tidy table schemas and writers.

Four tables per run, all keyed by ``field_id`` (+ ``cell_id``) so several fields
concatenate into one table each. Schemas live here so an empty table still has
its full column set.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

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
    "marker_intensity", "interface_occupancy_pct", "intensity_per_area", "cluster_density",
]
CELL_COLUMNS = [
    "field_id", "cell_id", "area_px", "area_um2", "measured",
    "n_interfaces", "n_interfaces_dp", "n_interfaces_tcj",
    "n_nuclei", "epsilon", "dilation_cycles", "pixel_size_um",
]
NUCLEUS_COLUMNS = [
    "field_id", "nucleus_id", "cell_id", "orphan", "area_px", "area_um2",
    "centroid_row", "centroid_col",
    "n_intra_distances", "mean_intra_distance_px", "mean_intra_distance_um",
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
}
_NULLABLE_INT = ["neighbour_id", "n_corners_dp", "n_corners_tcj", "cell_id"]
_BOOL = ["border_touching", "isolated", "degenerate", "corner_methods_disagree",
         "all_contacts_filtered", "multinucleate", "anucleate", "measured", "orphan"]
_INT = ["start_idx", "end_idx", "dilation_cycles", "interface_area_px", "marker_area_px",
        "area_px", "n_interfaces", "n_interfaces_dp", "n_interfaces_tcj", "n_nuclei",
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
    """Write interfaces/cells/nuclei/flags as ``<name>.parquet`` and ``<name>.csv`` in ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = list(results)
    for name, cols in SCHEMAS.items():
        frames = [getattr(r, name) for r in results if len(getattr(r, name))]
        df = pd.concat(frames, ignore_index=True) if frames else make_frame(name, [])
        df = df[cols]
        df.to_parquet(outdir / f"{name}.parquet", index=False)
        df.to_csv(outdir / f"{name}.csv", index=False)
