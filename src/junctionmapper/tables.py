"""Tidy table schemas and writers.

Five tables per run. Each is keyed by ``field_id`` (plus ``cell_id`` or
``nucleus_id``), so tables from several fields concatenate cleanly. The schemas
live here so an empty table still carries its full column set.


UNITS
-----
``s`` is ``pixel_size_um``.

    *_um     = *_px * s         lengths
    *_um2    = *_px * s**2      areas


FORMULAS
--------
    linearity_index           straight-line length / contour  (<= 1; 1 = straight)
    coverage_index            fragmented contour / interface contour
    interface_occupancy_pct   100 * marker area / interface area  (pixel counts)
    marker_intensity          sum of above-threshold values in the band (raw units)
    intensity_per_px2         marker_intensity / interface_area_px
    intensity_per_um2         intensity_per_px2 / s**2
    cluster_density_per_px2   marker_intensity / marker_area_px
    cluster_density_per_um2   cluster_density_per_px2 / s**2
    n_neighbours              distinct non-zero neighbour ids, both corner methods


WHAT ONE ``interfaces`` ROW IS
------------------------------
One row is ONE CELL-SIDE interface: the view from a single cell, with its own
band and its own numbers.

A row count is NOT a junction count.

    interface_id    one cell-side row
    pair_id         the cell pair it belongs to  (``field|method|lo-hi``)

Count junctions with ``pair_id.nunique()``.

How many rows share a ``pair_id`` depends on the corner method:

    tcj    at most 2, one per side
    dp     usually more -- on a real Cellpose run, median 3 and maximum 10

Douglas-Peucker corners are not neighbour-aware, so a single run of contact is
split wherever a corner happens to fall. Pooling dp rows as though each junction
contributed two of them weights junctions wrongly.

(Why cell-side at all: the original Java tool measures the interfaces of one
selected cell at a time, and this keeps that unit.)


THRESHOLD COLUMNS
-----------------
    threshold                      the value actually applied
    threshold_field_otsu           what Otsu on this field's own plane would give
                                   (NaN where Otsu cannot run)
    threshold_deviates_from_run    set when those two are further apart than the
                                   tolerance in ``pipeline.threshold_deviates``

One threshold is applied per role for the whole run, unless it was overridden.


FLAGS ON AN INTERFACE
---------------------
    contact_fraction        fraction of the interface's boundary positions that
                            are actually adjacent to ``neighbour_id``
    low_contact_fraction    set when that falls below
                            ``interfaces.LOW_CONTACT_FRACTION``
    closed_loop             a whole-boundary interface (``start_idx == end_idx``)
    splits_nucleus          the boundary between ``cell_id`` and ``neighbour_id`` cuts a
                            nucleus: one nucleus label has at least
                            ``pipeline.SPLIT_NUCLEUS_FRACTION`` of its pixels in each
                            cell. Usually a segmentation error, a false junction drawn
                            along membrane lying over the nucleus; exclude these rows.

``contact_fraction`` is at or near 1 for tcj by construction, and can be well
below 1 for dp.

A ``closed_loop`` interface has ``linearity_index`` near 0 because of its
geometry, not because it is tortuous.


``nuclei_distances`` IS LONG
----------------------------
One row per ORDERED (``nucleus_id``, ``neighbour_nucleus_id``) pair.

``rank_from_nucleus`` is that nucleus's own ordering of every distance involving
it (1 = its nearest neighbour).

Under ``all_pairs`` each unordered pair appears twice, once from each end. To
keep one copy of each distance, filter to
``nucleus_id < neighbour_nucleus_id``.

In the nuclei summary, ``n_intra_distances`` and ``mean_intra_distance_*`` are
the count and mean of that nucleus's own rows above -- a plain groupby on
``nucleus_id``. Under ``all_pairs`` that is n - 1 rows for every nucleus.
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
    "pair_id", "interface_id",
    "start_idx", "end_idx", "contact_fraction", "low_contact_fraction", "closed_loop",
    "splits_nucleus",
    "threshold", "threshold_field_otsu",
    "threshold_deviates_from_run", "dilation_cycles",
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
    "field_id", "nucleus_id", "neighbour_nucleus_id", "rank_from_nucleus",
    "distance_px", "distance_um",
]
FLAG_COLUMNS = [
    "field_id", "cell_id", "border_touching", "isolated", "degenerate",
    "corner_methods_disagree", "n_corners_dp", "n_corners_tcj", "all_contacts_filtered",
    "dp_low_contact", "dp_missed_tcj_contact", "closed_loop_interface",
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
_BOOL = ["threshold_deviates_from_run", "low_contact_fraction", "closed_loop", "splits_nucleus",
         "border_touching", "isolated", "degenerate", "corner_methods_disagree",
         "all_contacts_filtered", "dp_low_contact", "dp_missed_tcj_contact",
         "closed_loop_interface", "multinucleate", "anucleate", "measured", "orphan"]
_INT = ["start_idx", "end_idx", "dilation_cycles", "interface_area_px", "marker_area_px",
        "area_px", "n_neighbours", "n_interfaces_dp", "n_interfaces_tcj", "n_nuclei", "tcj_min_run",
        "neighbour_radius_px", "neighbour_nucleus_id", "rank_from_nucleus",
        "nucleus_id", "centroid_row", "centroid_col", "n_intra_distances"]
_STR = ["field_id", "corner_method", "role", "marker", "pair_id", "interface_id"]


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
