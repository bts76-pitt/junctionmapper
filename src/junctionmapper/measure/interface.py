"""Interface geometry: contour length, straight-line length, linearity.

All lengths returned here are in PIXELS, not micrometres. Conversion to
physical units (multiplying by ``Field.pixel_size_um``) happens downstream.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from junctionmapper.errors import ConfigError
from junctionmapper.measure.threshold import mask_above


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


_SQUARE_3X3 = np.ones((3, 3), dtype=bool)


def dilated_band(pixels, shape, cycles: int) -> np.ndarray:
    """Dilated Interface band (parameter 3, Dilation Cycles), as a bool mask.

    ``pixels`` is an ``(N, 2)`` int array of ``(row, col)``; repeated pixels
    are harmless because the run is rasterised into a mask. ``shape`` is the
    ``(rows, cols)`` of the image. The mask is dilated ``cycles`` times with the
    full 3x3 (8-connected) square, so one cycle widens a straight line to 3 px.
    ``cycles`` is a pixel count of iterations (not micrometres); 0 returns the
    undilated run. A band near the image edge is truncated at the edge.

    Divergence from the Java original (deliberate, do not "fix" back): the
    Java ``getDilatedImage`` suppresses a newly dilated pixel whenever any of
    its 3x3 neighbours falls outside ``[1, w-2] x [1, h-2]``, so dilation is
    dead at indices 0, 1, w-2 and w-1 -- a 2-px dead zone that compounds with
    each iteration. Cells within a couple of pixels of the border therefore
    get a narrower band there than the documented formula gives. That is an
    implementation border artefact, not the published algorithm, so uniform
    dilation is used here.

    Raises ConfigError for a bad ``pixels`` shape, a negative or non-integer
    ``cycles``, or pixels outside ``shape``.
    """
    if isinstance(cycles, bool) or not isinstance(cycles, (int, np.integer)) or cycles < 0:
        raise ConfigError(f"cycles must be a non-negative integer; got {cycles!r}")
    arr = np.asarray(pixels)
    if arr.size == 0:
        arr = arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ConfigError(
            f"interface pixels must have shape (N, 2) of (row, col); got {arr.shape}"
        )
    try:
        rows, cols = shape
    except (TypeError, ValueError):
        raise ConfigError(f"shape must be a (rows, cols) pair; got {shape!r}") from None
    mask = np.zeros((rows, cols), dtype=bool)
    if len(arr):
        r, c = arr[:, 0].astype(np.intp), arr[:, 1].astype(np.intp)
        if r.min() < 0 or c.min() < 0 or r.max() >= rows or c.max() >= cols:
            raise ConfigError(f"interface pixels fall outside the image shape {shape}")
        mask[r, c] = True
    # scipy treats iterations < 1 as "repeat until stable" (would flood the
    # image), so 0 cycles must be handled explicitly.
    if cycles == 0:
        return mask
    return ndimage.binary_dilation(mask, structure=_SQUARE_3X3, iterations=int(cycles))


@dataclass(frozen=True)
class BandMetrics:
    """Band measurements for one interface and one marker channel.

    Areas are pixel counts (convert with ``Field.pixel_size_um`` squared
    downstream); ``marker_intensity`` is in raw image intensity units;
    ``interface_occupancy_pct`` is a percentage; ``intensity_per_area`` and
    ``cluster_density`` are intensity per pixel. Ratios are NaN when undefined.
    """

    interface_area_px: int
    marker_area_px: int
    interface_occupancy_pct: float
    marker_intensity: float
    intensity_per_area: float
    cluster_density: float


def band_metrics(band, marker_image, t) -> BandMetrics:
    """Interface Area, Marker Area, Marker Intensity and derived ratios.

    ``band`` is a bool mask (from ``dilated_band``); ``marker_image`` is the
    greyscale marker channel of the same shape. The channel is thresholded with
    ``mask_above(marker_image, t)``, which keeps surviving values, so:

    - ``interface_area_px``: band pixel count (pixels).
    - ``marker_area_px``: band pixels with masked value > 0 (pixels).
    - ``marker_intensity``: sum of masked values over the band, in raw image
      units (float64 accumulation). Sub-threshold pixels contribute 0.
    - ``interface_occupancy_pct``: 100 * marker area / interface area.
    - ``intensity_per_area``: intensity / interface area (per pixel).
    - ``cluster_density``: intensity / marker area (per pixel).

    Ratios are NaN on a zero denominator. NaN pixels and pixels <= ``t`` are
    treated as below threshold; negative pixels (e.g. after background
    subtraction) are excluded whenever ``t >= 0``, but with a negative ``t``
    they pass and subtract from the intensity sum. A pixel above ``t`` whose
    value is not positive is not counted in the marker area (Java parity).

    Raises ConfigError for a non-bool ``band``, a shape mismatch or NaN ``t``.
    """
    band = np.asarray(band)
    marker_image = np.asarray(marker_image)
    if band.dtype != bool or band.ndim != 2:
        raise ConfigError("band must be a 2-D boolean mask (see dilated_band)")
    if marker_image.shape != band.shape:
        raise ConfigError(
            f"marker_image shape {marker_image.shape} does not match band shape {band.shape}"
        )
    masked = mask_above(marker_image, t)[band]
    interface_area = int(band.sum())
    marker_area = int(np.count_nonzero(masked > 0))
    intensity = float(masked.astype(np.float64).sum())

    def ratio(num, den):
        return num / den if den else float("nan")

    return BandMetrics(
        interface_area_px=interface_area,
        marker_area_px=marker_area,
        interface_occupancy_pct=100.0 * ratio(marker_area, interface_area),
        marker_intensity=intensity,
        intensity_per_area=ratio(intensity, interface_area),
        cluster_density=ratio(intensity, marker_area),
    )
