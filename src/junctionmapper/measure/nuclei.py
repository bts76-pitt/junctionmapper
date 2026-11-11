"""Nucleus centroids, nucleus-to-cell association and inter-nuclear distances.

Coordinates are ``(row, col)`` throughout, i.e. ``(y, x)``. Label arrays use 0
for background; the cell and nucleus label arrays are independent labellings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from junctionmapper.errors import ConfigError


@dataclass
class NucleusFlags:
    """Field-level summary: sorted lists of ids (not per-cell flags)."""

    multinucleate: list[int] = field(default_factory=list)  # cell ids, >1 nucleus
    orphan_nuclei: list[int] = field(default_factory=list)  # nucleus ids, no cell
    anucleate_cells: list[int] = field(default_factory=list)  # cell ids, 0 nuclei


def nucleus_centroids(nucleus_labels) -> dict[int, tuple[int, int]]:
    """Return ``{label: (row, col)}`` using the Java tool's midpoint formula.

    ``(max - min) // 2 + min`` over each axis of the nucleus's extent, with integer
    division, as in ``Cell_Nucleus.getGeometricMean``.

    This is the floored midpoint of the bounding extent, NOT the centre of mass.

    Values are ints on purpose: they are pixel indices produced by integer
    division, as in the Java original.
    """
    labels = np.asarray(nucleus_labels)
    out: dict[int, tuple[int, int]] = {}
    rows, cols = np.nonzero(labels)
    if rows.size == 0:
        return out
    vals = labels[rows, cols]
    order = np.argsort(vals, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]
    ids, starts = np.unique(vals, return_index=True)
    ends = list(starts[1:]) + [vals.size]
    for lab, s, e in zip(ids, starts, ends):
        r, c = rows[s:e], cols[s:e]
        out[int(lab)] = (
            int((r.max() - r.min()) // 2 + r.min()),
            int((c.max() - c.min()) // 2 + c.min()),
        )
    return out


def associate(cell_labels, nucleus_labels):
    """Assign each nucleus to the cell holding the most of its pixels.

    A nucleus is an orphan if and only if it overlaps no cell at all -- zero
    non-background pixels under it.

    Otherwise it goes to the cell label with the most pixels under it, ignoring
    background in that comparison. Ties go to the lowest cell label, independent of
    array order.

    Returns ``({cell_id: sorted [nucleus_ids]}, NucleusFlags)``. The mapping holds
    only cells that received a nucleus, while ``anucleate_cells`` enumerates every
    cell label present in ``cell_labels``. All flag lists are sorted.

    Divergence from the Java tool

        Java associates a nucleus with the membrane containing its centroid point
        (``setCellMembrane`` -> ``getSelectedMembrane(getGeometricMean())``).

        This uses pixel overlap instead, because that is robust to irregular
        nuclei and to nuclei straddling junctions, where the centroid can fall on
        background or on the wrong cell.

        Consequence: a nucleus with a single pixel over a cell is assigned to it,
        rather than orphaned.
    """
    cells = np.asarray(cell_labels)
    nuclei = np.asarray(nucleus_labels)
    if cells.shape != nuclei.shape:
        raise ConfigError(
            f"cell_labels shape {cells.shape} != nucleus_labels shape {nuclei.shape}"
        )
    mapping: dict[int, list[int]] = {}
    orphans: list[int] = []
    mask = nuclei != 0
    nuc_vals = nuclei[mask]
    cell_vals = cells[mask]
    for nid in np.unique(nuc_vals):
        under = cell_vals[nuc_vals == nid]
        under = under[under != 0]  # background is not a candidate
        if under.size == 0:
            orphans.append(int(nid))
            continue
        labs, counts = np.unique(under, return_counts=True)  # labs ascending
        winner = int(labs[counts == counts.max()].min())
        mapping.setdefault(winner, []).append(int(nid))
    mapping = {c: sorted(v) for c, v in sorted(mapping.items())}
    all_cells = [int(c) for c in np.unique(cells) if c != 0]
    flags = NucleusFlags(
        multinucleate=[c for c, v in mapping.items() if len(v) > 1],
        orphan_nuclei=sorted(orphans),
        anucleate_cells=[c for c in all_cells if c not in mapping],
    )
    return mapping, flags


def intra_distances(
    centroids: dict[int, tuple[int, int]], k: int = 5, all_pairs: bool = False
) -> list[tuple[int, int, float]]:
    """Euclidean distances between nucleus centroids, as ``(i, j, dist)``.

    By default, each nucleus ``i`` gets its ``k`` nearest other nuclei.

        self excluded    by index, not by zero distance -- so distinct nuclei
                         with identical centroids still give 0
        k                clamped to ``n - 1``; ``k < 1`` raises ConfigError
        fewer than 2     returns ``[]``
        nuclei
        ordering         by ``i`` ascending, then distance ascending, ties by ``j``

    Never emits inf.

    ``all_pairs=True`` ignores ``k`` and emits each unordered pair once with
    ``i < j``, ordered by ``(i, j)``.

    (Why an exact per-row distance sort rather than cKDTree: ties at the k-th
    neighbour then resolve deterministically, to the lowest id.)
    """
    if not all_pairs and (
        isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1
    ):
        raise ConfigError(f"k must be a positive integer, got {k!r}")
    ids = sorted(centroids)
    n = len(ids)
    if n < 2:
        return []
    pts = np.array([centroids[i] for i in ids], dtype=float)
    out: list[tuple[int, int, float]] = []
    kk = min(int(k), n - 1)
    for a, i in enumerate(ids):
        d = np.hypot(*(pts - pts[a]).T)
        if all_pairs:
            out.extend((i, ids[b], float(d[b])) for b in range(a + 1, n))
            continue
        d[a] = np.inf  # exclude self by index
        order = np.lexsort((np.arange(n), d))[:kk]  # distance, then id order
        out.extend((i, ids[b], float(d[b])) for b in order)
    return out
