"""Tricellular-junction (TCJ) corner detection from label adjacency.

Parameter-free alternative to Douglas-Peucker: a corner is a boundary position where the
identity of the neighbouring cell changes (including changes to or from a free edge).
"""
import numpy as np

from junctionmapper.errors import ConfigError


def neighbour_at(labels: np.ndarray, boundary: np.ndarray, label: int,
                 search_radius: int = 2) -> np.ndarray:
    """Neighbour label for every boundary *position* (0 = free edge / background).

    ``boundary`` is the (N, 2) (row, col) array from ``trace_boundary``; the result has
    length N and is index-aligned with it (a necked pixel that appears twice gets two
    entries, identical because they share coordinates).

    For each pixel, the square window of Chebyshev radius ``search_radius`` (clipped to
    the image; outside the image counts as background) is scanned for labels that are
    neither 0 nor ``label``. The winner is chosen by, in order:
      1. highest pixel count in the window (modal label),
      2. smallest Euclidean distance to any pixel of that label,
      3. smallest label value.
    Rule 3 makes ties fully deterministic, independent of array order or trace start.
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


def tcj_corners(neighbour_ids: np.ndarray) -> np.ndarray:
    """Ascending indices where the neighbour identity changes, treating the array as circular.

    Convention: index ``i`` is returned when ``ids[i] != ids[i-1]`` (with ``ids[-1]``
    preceding ``ids[0]``), i.e. the corner is the FIRST position of the new neighbour.
    Slicing ``[c_k, c_{k+1})`` between consecutive corners therefore yields a run with a
    single neighbour identity; the final interface wraps from the last corner through the
    end of the array and around to the first corner.

    A closed curve cannot have exactly one change, so counts are 0 or >= 2 (0 for an
    all-same boundary such as an isolated cell). Arrays shorter than 2 give no corners.
    """
    ids = np.asarray(neighbour_ids)
    if ids.size < 2:
        return np.zeros(0, dtype=int)
    return np.flatnonzero(ids != np.roll(ids, 1)).astype(int)
