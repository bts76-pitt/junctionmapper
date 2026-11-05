"""Douglas-Peucker corner detection on a closed boundary."""
import numpy as np

from junctionmapper.errors import ConfigError


def _farthest_pair(pts: np.ndarray) -> tuple:
    """Indices (i < j) of the two mutually most distant points; O(N^2), first max wins."""
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    return (int(min(i, j)), int(max(i, j)))


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
