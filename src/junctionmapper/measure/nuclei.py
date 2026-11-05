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


def nucleus_centroids(nucleus_labels) -> dict[int, tuple[float, float]]:
    """Return ``{label: (row, col)}`` using the Java tool's midpoint formula.

    ``(max - min) // 2 + min`` over each axis of the nucleus's extent (integer
    division, as in ``Cell_Nucleus.getGeometricMean``). This is the floored
    midpoint of the bounding extent, NOT the centre of mass. Values are ints.
    """
    labels = np.asarray(nucleus_labels)
    out: dict[int, tuple[float, float]] = {}
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
    """Assign each nucleus to the cell holding the plurality of its pixels.

    Background (0) is a candidate: a nucleus mostly over background is an
    orphan. Ties are broken deterministically: any cell beats background, and
    among cells the lowest label wins (independent of array order).

    Returns ``({cell_id: sorted [nucleus_ids]}, NucleusFlags)``. The mapping
    holds only cells that received a nucleus; ``anucleate_cells`` enumerates
    every cell label present in ``cell_labels``. All flag lists are sorted.
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
        labs, counts = np.unique(under, return_counts=True)  # labs ascending
        best = counts.max()
        winners = labs[counts == best]
        nonzero = winners[winners != 0]
        winner = int(nonzero.min()) if nonzero.size else 0
        if winner == 0:
            orphans.append(int(nid))
        else:
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
    centroids: dict[int, tuple[float, float]], k: int = 5, all_pairs: bool = False
) -> list[tuple[int, int, float]]:
    """Euclidean distances between nucleus centroids, as ``(i, j, dist)``.

    Default: for each nucleus ``i`` its ``k`` nearest other nuclei. Self is
    excluded by index (not by zero distance, so distinct nuclei with identical
    centroids still give 0). ``k`` is clamped to ``n - 1``; fewer than two
    nuclei gives ``[]``; ``k < 1`` raises ConfigError. Never emits inf.
    Ordering: by ``i`` ascending, then distance ascending (ties by ``j``).

    ``all_pairs=True`` ignores ``k`` and emits each unordered pair once with
    ``i < j``, ordered by ``(i, j)``.

    Computed with an exact per-row distance sort rather than cKDTree so that
    ties at the k-th neighbour are resolved deterministically (lowest id).
    """
    if not all_pairs and (not isinstance(k, (int, np.integer)) or k < 1):
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
