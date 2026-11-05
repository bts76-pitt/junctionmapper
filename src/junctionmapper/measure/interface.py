"""Interface geometry: contour length, straight-line length, linearity.

All lengths returned here are in PIXELS, not micrometres. Conversion to
physical units (multiplying by ``Field.pixel_size_um``) happens downstream.
"""

import math

import numpy as np

from junctionmapper.errors import ConfigError


def _as_path(pixels) -> np.ndarray:
    """Validate ``pixels`` and return it as an ``(N, 2)`` float64 array."""
    arr = np.asarray(pixels)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ConfigError(
            f"interface pixels must have shape (N, 2) of (row, col); got {arr.shape}"
        )
    return arr.astype(np.float64, copy=False)


def _steps(path: np.ndarray) -> np.ndarray:
    steps = np.diff(path, axis=0)
    if steps.size and np.abs(steps).max() > 1:
        raise ConfigError(
            "interface pixels are not a contiguous 8-connected walk "
            "(consecutive entries more than one pixel apart)"
        )
    return steps


def interface_contour(pixels) -> float:
    """Interface Contour (parameter 1), in pixels.

    Sum of Euclidean distances between consecutive pixels along the walk:
    1 for an orthogonal step, sqrt(2) for a diagonal step. Accumulated in
    float64 with ``math.fsum``. Zero for an empty or single-pixel interface.

    Raises ConfigError if ``pixels`` is not ``(N, 2)`` or consecutive pixels
    are not 8-adjacent.
    """
    path = _as_path(pixels)
    steps = _steps(path)
    if len(steps) == 0:
        return 0.0
    return math.fsum(np.hypot(steps[:, 0], steps[:, 1]))


def straight_line_length(pixels) -> float:
    """Straight-line Interface Length (parameter 2), in pixels.

    Euclidean distance between the first and last pixel. Zero for an empty
    or single-pixel interface. Uses endpoints only, so it does not check
    walk contiguity (that is enforced by ``interface_contour``).
    """
    path = _as_path(pixels)
    if len(path) < 2:
        return 0.0
    d = path[-1] - path[0]
    return float(math.hypot(d[0], d[1]))


def linearity_index(pixels) -> float:
    """Straight-line length / contour; dimensionless, NaN when undefined.

    For an open arc this lies in [0, 1] (1 = perfectly straight). NaN when
    the contour is 0 (empty or single-pixel interface). For the
    whole-boundary interface of a cell enclosed by one neighbour, first and
    last pixel are adjacent (~1 px apart) while the contour is the whole
    perimeter, so the value is near 0; that is a closed loop, not a very
    crooked line.
    """
    contour = interface_contour(pixels)
    if contour == 0.0:
        return float("nan")
    # Triangle inequality guarantees <= 1; clamp away last-ulp rounding.
    return min(1.0, straight_line_length(pixels) / contour)
