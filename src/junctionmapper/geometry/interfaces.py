"""Split a cell's boundary into interfaces (one per neighbouring cell contact)."""
from dataclasses import dataclass

import numpy as np

from junctionmapper.errors import ConfigError
from junctionmapper.geometry.boundary import trace_boundary
from junctionmapper.geometry.corners_dp import dp_corners
from junctionmapper.geometry.corners_tcj import merge_short_runs, neighbour_at, tcj_corners

TCJ_MIN_RUN = 3     # must match the default of tcj_corners so ids and corners agree


@dataclass
class Interface:
    """A run of boundary positions of ``cell_id`` against one neighbour.

    ``start_idx`` is the first boundary position of the run; ``end_idx`` is the first
    position of the NEXT run (exclusive). The run covers positions
    ``start_idx, start_idx+1, ..., end_idx-1`` modulo the boundary length, so the
    interface that wraps through index 0 has ``start_idx > end_idx``. With a single
    corner the run is the whole loop and ``start_idx == end_idx``. ``pixels`` is the
    (row, col) walk in that order, hence contiguous across the wrap.
    """
    cell_id: int
    neighbour_id: "int | None"
    start_idx: int
    end_idx: int
    pixels: np.ndarray
    method: str


def _modal_neighbour(ids: np.ndarray) -> int:
    """Most frequent id; ties go to the smallest NONZERO label (0 wins only outright).

    Depends only on the multiset of ids, never on position or trace start.
    """
    vals, counts = np.unique(ids, return_counts=True)
    top = counts.max()
    winners = vals[counts == top]
    nonzero = winners[winners != 0]
    return int(nonzero.min()) if nonzero.size else 0


def extract_interfaces(labels: np.ndarray, label: int, method: str,
                       epsilon: "float | None" = None) -> "list[Interface]":
    """Interfaces of ``label`` split by ``method`` ('dp' or 'tcj').

    ``epsilon`` is required for 'dp' (ConfigError if None) and ignored for 'tcj'.
    Runs against background (free edge) are dropped. With zero corners the decision is
    made on the neighbour ids: an isolated cell (all background) gives ``[]``, while a
    cell engulfed by a single neighbour gives one whole-boundary interface
    (``start_idx == end_idx == 0``). A whole-boundary interface is a legitimate result,
    not a degenerate one. A single corner likewise yields one closed-loop interface.
    Neighbour symmetry is not guaranteed: A may list B while B does not list A.
    """
    if method not in ("dp", "tcj"):
        raise ConfigError(f"method must be 'dp' or 'tcj', got {method!r}")
    if method == "dp" and epsilon is None:
        raise ConfigError("epsilon is required for method='dp'")
    boundary = trace_boundary(labels, label)
    n = len(boundary)
    ids = neighbour_at(labels, boundary, label)
    if method == "tcj":
        corners = tcj_corners(ids, min_run=TCJ_MIN_RUN)
        run_ids = merge_short_runs(ids, TCJ_MIN_RUN)
    else:
        corners = np.asarray(dp_corners(boundary, epsilon), dtype=int)
        run_ids = ids
    if len(corners) == 0:
        # No corner anywhere: the whole boundary is one run. Whether it is an interface
        # is decided by the neighbour ids (all background -> isolated -> none; one
        # non-zero identity -> engulfed cell -> one whole-boundary interface), not by
        # the corner count. Same representation as the single-corner case, anchored at 0.
        corners = np.array([0], dtype=int)
    m = len(corners)
    out = []
    for k in range(m):
        s = int(corners[k])
        e = int(corners[(k + 1) % m])
        length = (e - s) % n or n          # m == 1 -> e == s -> whole loop
        idx = (s + np.arange(length)) % n
        nb = int(run_ids[s]) if method == "tcj" else _modal_neighbour(ids[idx])
        if nb == 0:
            continue                       # free edge: consumes positions, not an interface
        out.append(Interface(cell_id=int(label), neighbour_id=nb, start_idx=s,
                             end_idx=e, pixels=boundary[idx], method=method))
    return out
