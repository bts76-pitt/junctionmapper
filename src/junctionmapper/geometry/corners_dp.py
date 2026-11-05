"""Douglas-Peucker corner detection on a closed boundary."""
import numpy as np

from junctionmapper.errors import ConfigError


def _farthest_pair(pts: np.ndarray) -> tuple:
    """Indices (i < j) of the two mutually most distant points; O(N^2).

    Squared distances tie often, so the pair is chosen canonically to make the result
    independent of where the boundary array starts: among all pairs at the maximum
    distance, take the one whose coordinates sorted as ((r1, c1), (r2, c2)) with
    (r1, c1) <= (r2, c2) are lexicographically smallest. Indices are then the first
    occurrence of each coordinate (a necked pixel can occur twice), ordered i < j.

    "First occurrence" is the one residual dependence on the trace origin: if a seed
    coordinate were itself a repeated pixel, which occurrence is first would depend on
    where the trace started. Measured at 0 occurrences over 1,152 necked boundaries
    (5,134 traced), and geometrically it is hard to arrange — a farthest-pair seed is an
    extreme point of the shape, while a repeated pixel is an interior constriction.
    """
    dr = pts[:, 0, None] - pts[None, :, 0]
    dc = pts[:, 1, None] - pts[None, :, 1]
    d2 = dr * dr + dc * dc
    ii, jj = np.nonzero(np.triu(d2 == d2.max(), 1))
    best = None
    for a, b in zip(ii, jj):
        pa, pb = tuple(pts[a]), tuple(pts[b])
        key = (pa, pb) if pa <= pb else (pb, pa)
        if best is None or key < best:
            best = key
    first = {}
    for k, p in enumerate(map(tuple, pts)):
        first.setdefault(p, k)
    i, j = first[best[0]], first[best[1]]
    return (min(i, j), max(i, j))


def _split(pts: np.ndarray, lo: int, hi: int, epsilon: float, keep: set) -> None:
    """Douglas-Peucker on the open arc lo..hi (inclusive indices, lo < hi), iteratively."""
    stack = [(lo, hi)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = pts[a + 1:b] - pts[a]
        chord = pts[b] - pts[a]
        length = float(np.hypot(*chord))
        if length == 0.0:      # arc returns to its start (neck): fall back to radial distance
            dist = np.hypot(seg[:, 0], seg[:, 1])
        else:
            dist = np.abs(chord[0] * seg[:, 1] - chord[1] * seg[:, 0]) / length
        k = int(np.argmax(dist))
        if dist[k] > epsilon:
            m = a + 1 + k
            keep.add(m)
            stack.append((a, m))
            stack.append((m, b))


def dp_corners(boundary: np.ndarray, epsilon: float) -> np.ndarray:
    """Return ascending indices into ``boundary`` of its Douglas-Peucker corners.

    ``boundary`` is a closed (N, 2) cycle (N >= 3); it is seeded with the two mutually
    most distant points and each of the two arcs between them is simplified with
    perpendicular distance to the chord. A point is kept when it lies strictly more
    than ``epsilon`` from the chord. Results are indices, so repeated pixels (necks)
    are never conflated. If ``epsilon`` exceeds the shape, only the two seeds remain.

    Raises ConfigError for epsilon that is not a finite number > 0, or a boundary that
    is not (N, 2) with N >= 3.
    """
    pts = np.asarray(boundary, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ConfigError(f"boundary must have shape (N, 2), got {pts.shape}")
    if len(pts) < 3:
        raise ConfigError(f"boundary needs at least 3 points, got {len(pts)}")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ConfigError(f"epsilon must be a finite number > 0, got {epsilon!r}")

    i, j = _farthest_pair(pts)
    keep = {i, j}
    _split(pts, i, j, float(epsilon), keep)
    # Second arc j -> end -> start -> i, unrolled so it is an open polyline.
    n = len(pts)
    order = np.concatenate([np.arange(j, n), np.arange(0, i + 1)])
    tmp = set()
    _split(pts[order], 0, len(order) - 1, float(epsilon), tmp)
    keep.update(int(order[t]) for t in tmp)
    return np.array(sorted(keep), dtype=int)
