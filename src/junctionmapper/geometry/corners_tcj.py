"""Tricellular-junction (TCJ) corner detection from label adjacency.

Parameter-free alternative to Douglas-Peucker: a corner is a boundary position where the
identity of the neighbouring cell changes (including changes to or from a free edge).
"""
import numpy as np

from junctionmapper.config import DEFAULT_NEIGHBOUR_RADIUS_PX, DEFAULT_TCJ_MIN_RUN
from junctionmapper.errors import ConfigError


def neighbour_at(labels: np.ndarray, boundary: np.ndarray, label: int,
                 search_radius: int = DEFAULT_NEIGHBOUR_RADIUS_PX) -> np.ndarray:
    """Neighbour label for every boundary *position* (0 = free edge / background).

    ``boundary`` is the (N, 2) (row, col) array from ``trace_boundary``. The result
    has length N and is index-aligned with it. A necked pixel that appears twice
    gets two entries, identical because they share coordinates.

    For each pixel, the square window of Chebyshev radius ``search_radius`` is
    scanned for labels that are neither 0 nor ``label``. The window is clipped to
    the image, and outside the image counts as background.

    The winner is chosen by, in order:

        1. highest pixel count in the window (modal label)
        2. smallest Euclidean distance to any pixel of that label
        3. smallest label value

    Rule 3 makes ties fully deterministic, independent of array order or trace
    start.
    """
    if (isinstance(search_radius, bool) or not isinstance(search_radius, (int, np.integer))
            or search_radius < 1):
        raise ConfigError(f"search_radius must be an integer >= 1, got {search_radius!r}")
    h, w = labels.shape
    r = int(search_radius)
    cache = {}
    out = np.zeros(len(boundary), dtype=int)
    for k, (y, x) in enumerate(boundary):
        key = (int(y), int(x))
        if key not in cache:
            y0, y1 = max(0, key[0] - r), min(h, key[0] + r + 1)
            x0, x1 = max(0, key[1] - r), min(w, key[1] + r + 1)
            win = labels[y0:y1, x0:x1]
            yy, xx = np.nonzero((win != 0) & (win != label))
            best = 0
            if yy.size:
                vals = win[yy, xx]
                d2 = (yy + y0 - key[0]) ** 2 + (xx + x0 - key[1]) ** 2
                best_key = None
                for v in np.unique(vals):
                    m = vals == v
                    cand = (-int(m.sum()), int(d2[m].min()), int(v))
                    if best_key is None or cand < best_key:
                        best_key, best = cand, int(v)
            cache[key] = best
        out[k] = cache[key]
    return out


def _check_min_run(min_run) -> int:
    if (isinstance(min_run, bool) or not isinstance(min_run, (int, np.integer))
            or min_run < 1):
        raise ConfigError(f"min_run must be an integer >= 1, got {min_run!r}")
    return int(min_run)


def merge_short_runs(neighbour_ids: np.ndarray, min_run: int) -> np.ndarray:
    """Copy of ``neighbour_ids`` with every circular run shorter than ``min_run`` absorbed.

    Each sub-threshold run is absorbed into an adjacent one, by this rule, applied
    repeatedly:

        take the shortest sub-threshold run    ties: smaller label, then earlier
                                               position
        give its positions to the longer of    ties: smaller label
        its two neighbouring runs

    Runs that become adjacent with the same label fuse.

    The array is circular, so a run may span the last and first positions.

    If every run is sub-threshold, the process continues until a single identity
    remains. ``min_run=1`` returns an unchanged copy.
    """
    min_run = _check_min_run(min_run)
    ids = np.asarray(neighbour_ids)
    if ids.ndim != 1:
        raise ConfigError("neighbour_ids must be 1-D")
    n = ids.size
    out = ids.copy()
    if min_run == 1 or n < 2:
        return out
    while True:
        change = np.flatnonzero(out != np.roll(out, 1))
        if change.size == 0:
            return out                            # single identity left
        starts = change
        lens = np.diff(np.append(starts, starts[0] + n))
        m = len(starts)
        short = [i for i in range(m) if lens[i] < min_run]
        if not short:
            return out
        k = min(short, key=lambda i: (int(lens[i]), int(out[starts[i]]), int(starts[i])))
        prev, nxt = (k - 1) % m, (k + 1) % m
        # longer neighbour wins; equal length -> smaller label
        tgt = min((prev, nxt), key=lambda i: (-int(lens[i]), int(out[starts[i]])))
        pos = (starts[k] + np.arange(lens[k])) % n
        out[pos] = out[starts[tgt]]


def tcj_corners(neighbour_ids: np.ndarray, min_run: int = DEFAULT_TCJ_MIN_RUN) -> np.ndarray:
    """Ascending indices where the neighbour identity changes, treating the array as circular.

    Convention: index ``i`` is returned when ``ids[i] != ids[i-1]``, with
    ``ids[-1]`` preceding ``ids[0]``. The corner is therefore the FIRST position of
    the new neighbour.

    Slicing ``[c_k, c_{k+1})`` between consecutive corners yields a run with a
    single neighbour identity. The final interface wraps from the last corner
    through the end of the array and around to the first.

    Runs shorter than ``min_run`` positions are absorbed into a neighbouring run
    first (see ``merge_short_runs``).

    ``min_run`` is a filter on the measurement, not a property of the
    segmentation. ``min_run=1`` disables it, giving raw identity changes.

    (Why the default is 3: at a four-way vertex a diagonal-only neighbour can win
    one or two positions, and an interface spanning fewer than 3 boundary pixels
    cannot support a meaningful contour measurement. Every spurious run observed
    was 1-2 pixels.)

    A closed curve cannot have exactly one change, so counts are 0 or >= 2. Zero
    means an all-same boundary, such as an isolated cell. Arrays shorter than 2
    give no corners.

    Raises ConfigError for non-1-D input or an invalid ``min_run``.
    """
    min_run = _check_min_run(min_run)
    ids = np.asarray(neighbour_ids)
    if ids.ndim != 1:
        raise ConfigError(f"neighbour_ids must be 1-D, got shape {ids.shape}")
    if ids.size < 2:
        return np.zeros(0, dtype=int)
    ids = merge_short_runs(ids, min_run)
    return np.flatnonzero(ids != np.roll(ids, 1)).astype(int)
