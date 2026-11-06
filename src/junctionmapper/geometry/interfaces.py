"""Split a cell's boundary into interfaces (one per neighbouring cell contact)."""
from dataclasses import dataclass

import numpy as np

from junctionmapper.errors import ConfigError
from junctionmapper.geometry.boundary import trace_boundary
from junctionmapper.geometry.corners_dp import dp_corners
from junctionmapper.geometry.corners_tcj import merge_short_runs, neighbour_at, tcj_corners

from junctionmapper.config import DEFAULT_NEIGHBOUR_RADIUS_PX, DEFAULT_TCJ_MIN_RUN

TCJ_MIN_RUN = DEFAULT_TCJ_MIN_RUN   # single definition lives on RunConfig's default

LOW_CONTACT_FRACTION = 0.8
"""Below this fraction of an interface's positions actually touching the labelled neighbour,
the interface carries enough non-junction to be worth flagging.

Chosen at four fifths, not at a half. The modal rule keeps a run only when the neighbour wins
the plurality, so on a run that holds just the neighbour and free edge the fraction is above a
half by construction: a half would never fire in the commonest case, which is the one that
matters -- a neighbour touching 8 of a 20-pixel side yields fraction 0.53 and a dp interface
whose contour is 19 px against a true contact of 11. By contrast tcj, which cuts exactly where
the identity changes, is at 1.0. A fifth of the band not lying against the labelled neighbour
already moves the band metrics (interface area, marker area, intensity, occupancy, coverage) by
a comparable fraction, since they scale with the length of the band.

It is a reporting threshold on an interface the modal-neighbour rule has already produced,
never a filter: the row is still emitted and still measured.
"""


@dataclass
class Interface:
    """An interface of ``cell_id`` against one neighbour, as the walk between two corners.

    ``pixels`` is authoritative and INCLUSIVE of both corner pixels (start corner first,
    end corner last), in walk order and contiguous, as in the reference method, so
    contour and straight-line length are measured corner to corner. ``start_idx`` is the
    boundary position of the start corner; ``end_idx`` is the boundary position of the
    end corner, i.e. the first position of the next run. The walk is
    ``start_idx, start_idx+1, ..., end_idx`` modulo the boundary length, so the interface
    that wraps through index 0 has ``start_idx > end_idx``. Adjacent interfaces share
    their common corner pixel, so a corner position is covered by exactly two interfaces
    and every other position by one. When one run is the whole boundary (single corner,
    or an engulfed cell) ``start_idx == end_idx`` and the start pixel is repeated as the
    last pixel to close the loop, so the contour equals the true perimeter -- a CLOSED LOOP,
    whose straight-line length is ~0 and whose linearity index is therefore ~0 by geometry,
    not by tortuosity.

    ``contact_fraction`` is the fraction of the interface's boundary positions (the run, not
    counting the repeated closing pixel) whose nearest non-self label actually IS
    ``neighbour_id``. For 'tcj' it is at or near 1 by construction, since the run was cut where
    that identity changes. For 'dp' the corners know nothing about neighbours, so a run is
    labelled with its plurality neighbour and the fraction can be well below 1: the rest of the
    run is free edge or another cell that nevertheless sits inside the measured band. Reading a
    dp row's band metrics without reading its ``contact_fraction`` is reading a number that may
    be half non-junction.
    """
    cell_id: int
    neighbour_id: "int | None"
    start_idx: int
    end_idx: int
    pixels: np.ndarray
    method: str
    contact_fraction: float = 1.0


def _modal_neighbour(ids: np.ndarray) -> int:
    """Most frequent id; ties go to the smallest NONZERO label (0 wins only outright).

    Depends only on the multiset of ids, never on position or trace start.
    """
    vals, counts = np.unique(ids, return_counts=True)
    top = counts.max()
    winners = vals[counts == top]
    nonzero = winners[winners != 0]
    return int(nonzero.min()) if nonzero.size else 0


def _check_int_ge1(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
        raise ConfigError(f"{name} must be an integer >= 1, got {v!r}")
    return int(v)


def extract_interfaces(labels: np.ndarray, label: int, method: str,
                       epsilon: "float | None" = None, min_run: int = TCJ_MIN_RUN,
                       search_radius: int = DEFAULT_NEIGHBOUR_RADIUS_PX) -> "list[Interface]":
    """Interfaces of ``label`` split by ``method`` ('dp' or 'tcj').

    ``epsilon`` is required for 'dp' (ConfigError if None) and ignored for 'tcj'.
    ``min_run`` (TCJ only, shortest contact run kept) and ``search_radius`` (window used to
    assign each boundary pixel a neighbour; both methods) must be integers >= 1.
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
    min_run = _check_int_ge1("min_run", min_run)
    search_radius = _check_int_ge1("search_radius", search_radius)
    boundary = trace_boundary(labels, label)
    n = len(boundary)
    ids = neighbour_at(labels, boundary, label, search_radius=search_radius)
    if method == "tcj":
        corners = tcj_corners(ids, min_run=min_run)
        run_ids = merge_short_runs(ids, min_run)
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
        # modular, inclusive of the end corner (which is the start again for a loop)
        idx = (s + np.arange(length + 1)) % n
        run_idx = idx[:-1]
        nb = int(run_ids[s]) if method == "tcj" else _modal_neighbour(ids[run_idx])
        if nb == 0:
            continue                       # free edge: consumes positions, not an interface
        # measured on the RAW ids, so it says how much of the run really touches nb
        frac = float(np.mean(ids[run_idx] == nb)) if run_idx.size else 0.0
        out.append(Interface(cell_id=int(label), neighbour_id=nb, start_idx=s,
                             end_idx=e, pixels=boundary[idx], method=method,
                             contact_fraction=frac))
    return out
