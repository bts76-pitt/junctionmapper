"""Outline-free membrane score: does a stain sit in thin lines, or spread through the cell?

No segmentation is used. The field is cut into square patches; within each patch's cell
region (nuclei grown outward, see ``cell_region``) a multi-scale line filter ranks pixels
by how ridge-like they are, and

    contrast = mean intensity on the top ``ridge_fraction`` of ridge pixels
               / mean intensity on the rest of the region
    baseline = the same, filter included, on one seeded shuffle of the region's pixels
    score    = contrast / baseline        (1 = no line structure)

Dividing by the shuffled baseline removes what any texture would give by chance (noise
alone puts the brightest pixels on the "ridge"), so a score means the same for any stain.

A patch whose signal is not clearly above the field's floor is ``too_dim`` and gets no
score: a blank channel reads as too dim, never as diffuse.

Lengths in the config are um; they are converted with ``field.pixel_size_um``.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import sato, threshold_otsu

from junctionmapper.config import MembraneScoreConfig, Role
from junctionmapper.io.reader import Field
from junctionmapper.tables import make_frame

# --- CONSTANTS ---------------------------------------------------------------------------
NUCLEUS_SMOOTH_SIGMA_PX = 2.0    # Gaussian smoothing of the nucleus plane before Otsu
BACKGROUND_PERCENTILE = 1.0      # field floor: confluent fields have no cell-free pixels
NOISE_HIGHPASS_SIGMA_PX = 2.0    # residual = plane - gaussian(plane, this); its MAD is noise
MAD_TO_SD = 1.4826               # MAD of a Gaussian times this = its standard deviation
DIM_NOISE_MULTIPLE = 3.0         # too dim when (region mean - floor) < this x noise
MIN_REGION_FRACTION = 0.5        # a patch is scored when at least this much is cell region
MIN_SIDE_FRACTION = 0.5          # a clipped edge patch is scored only when both its sides are
                                 # at least this fraction of a full side
MIN_SIGMA_PX = 1.0               # filter scales never go below one pixel


def cell_region(nucleus: np.ndarray, px_um: float, region_um: float) -> np.ndarray:
    """Where cells are (not where they meet): Otsu nuclei grown outward by ``region_um``.

    Always from the raw nucleus plane, never from labels, so the region does not depend on
    segmentation. A plane with no Otsu foreground (blank, constant or all NaN) gives an
    all-False region. NaN pixels never count as nucleus.
    """
    plane = np.asarray(nucleus, np.float64)
    finite = np.isfinite(plane)
    if not finite.any():
        return np.zeros(plane.shape, bool)
    smooth = ndi.gaussian_filter(np.where(finite, plane, np.nanmin(plane)),
                                 NUCLEUS_SMOOTH_SIGMA_PX)
    if smooth.max() <= smooth.min():
        return np.zeros(plane.shape, bool)
    nuclei = (smooth > threshold_otsu(smooth)) & finite
    if not nuclei.any():
        return nuclei
    return ndi.distance_transform_edt(~nuclei) * px_um <= region_um


def floor_and_noise(plane: np.ndarray) -> tuple[float, float]:
    """The field's floor and noise for one channel, in the channel's own units.

        background   1st percentile of the finite pixels
        noise        1.4826 x MAD of the high-pass residual (plane minus its Gaussian
                     smoothing at 2 px) over finite pixels; a robust noise SD, 0 for a
                     constant or quantisation-level plane

    Both scale with the plane, so the too-dim test built on them is scale-invariant. NaN
    pixels are ignored; an all-NaN plane gives ``(nan, 0.0)``.
    """
    plane = np.asarray(plane, np.float64)
    finite = np.isfinite(plane)
    if not finite.any():
        return math.nan, 0.0
    vals = plane[finite]
    background = float(np.percentile(vals, BACKGROUND_PERCENTILE))
    filled = np.where(finite, plane, np.median(vals))
    residual = (filled - ndi.gaussian_filter(filled, NOISE_HIGHPASS_SIGMA_PX))[finite]
    noise = MAD_TO_SD * float(np.median(np.abs(residual - np.median(residual))))
    return background, noise


def _filter_geometry(px_um: float, line_width_um: tuple[float, float]) -> tuple[int, list[float]]:
    """Downsampling factor and Sato sigmas (in downsampled px) covering ``line_width_um``.

    A line of width w is matched by sigma w / 2. The factor is the largest integer that keeps
    the smallest sigma at least 1 px after downsampling (cost: about 4x fewer pixels at 2),
    so the choice depends on pixel size only, never on image size. Sigmas are clamped to
    1 px and spaced at most 1 px apart.
    """
    lo, hi = (max(w / px_um / 2, MIN_SIGMA_PX) for w in line_width_um)
    factor = max(1, int(lo // MIN_SIGMA_PX))
    lo, hi = max(lo / factor, MIN_SIGMA_PX), max(hi / factor, MIN_SIGMA_PX)
    n = max(1, math.ceil(hi - lo) + 1) if hi > lo else 1
    return factor, [float(s) for s in np.linspace(lo, hi, n)]


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Mean over f x f blocks, ignoring NaN (NaN where a block has no finite pixel).

    Trailing rows and columns that do not fill a block are dropped.
    """
    if f == 1:
        return a
    h, w = a.shape[0] // f, a.shape[1] // f
    blocks = a[:h * f, :w * f].reshape(h, f, w, f)
    finite = np.isfinite(blocks)
    total = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    count = finite.sum(axis=(1, 3))
    return np.divide(total, count, out=np.full((h, w), np.nan), where=count > 0)


def _ridge(img: np.ndarray, sigmas: list[float]) -> np.ndarray:
    """Bright-line response (Sato), largest over ``sigmas``."""
    return sato(img, sigmas=sigmas, black_ridges=False)


def _contrast(values: np.ndarray, response: np.ndarray, ridge_fraction: float) -> float:
    """Mean of ``values`` on the top ``ridge_fraction`` of ``response`` / mean on the rest.

    Ties are broken by position (stable sort), so the cut does not move under a change of
    intensity scale. NaN when either side is empty or the rest has no positive mean.
    """
    n = values.size
    k = max(1, round(ridge_fraction * n))
    if n - k < 1:
        return math.nan
    order = np.argsort(-response, kind="stable")
    ridge, rest = values[order[:k]].mean(), values[order[k:]].mean()
    return float(ridge / rest) if rest > 0 else math.nan


def _patch_rows(field: Field, role: Role, region: np.ndarray, cfg: MembraneScoreConfig,
                size: int, factor: int, sigmas: list[float]) -> list[dict]:
    plane = np.asarray(field.planes[role], np.float64)
    finite = np.isfinite(plane)
    background, noise = floor_and_noise(plane)
    median = float(np.median(plane[finite])) if finite.any() else 0.0
    small = _block_mean(plane, factor)
    small = np.where(np.isfinite(small), small, median)              # filter needs no NaN
    small_region = _block_mean(region.astype(np.float64), factor) >= 0.5
    small_finite = _block_mean(np.where(finite, 1.0, np.nan), factor) > 0
    small_use = small_region & small_finite
    response = None                                                  # filtered on first need

    rows = []
    h, w = plane.shape
    for r, y0 in enumerate(range(0, h, size)):
        for c, x0 in enumerate(range(0, w, size)):
            y1, x1 = min(y0 + size, h), min(x0 + size, w)
            reg = region[y0:y1, x0:x1]
            region_fraction = float(reg.mean())
            vals = plane[y0:y1, x0:x1][reg & finite[y0:y1, x0:x1]]
            mean = float(vals.mean()) if vals.size else math.nan
            scored = (region_fraction >= MIN_REGION_FRACTION
                      and min(y1 - y0, x1 - x0) >= MIN_SIDE_FRACTION * size)
            too_dim = scored and not (
                noise > 0 and np.isfinite(mean) and mean - background >= DIM_NOISE_MULTIPLE * noise)
            contrast = baseline = score = math.nan
            # The patch in filter pixels, and the same window snapped to the block grid in
            # native pixels. An edge patch narrower than a block holds none: left unscored.
            sy, sx = slice(y0 // factor, y1 // factor), slice(x0 // factor, x1 // factor)
            ny = slice(sy.start * factor, sy.stop * factor)
            nx = slice(sx.start * factor, sx.stop * factor)
            use = small_use[sy, sx]
            if scored and not too_dim and use.sum() > 1:
                if response is None:
                    response = _ridge(small, sigmas)
                contrast = _contrast(small[sy, sx][use], response[sy, sx][use], cfg.ridge_fraction)
                # Shuffle in native pixels, then downsample as the original was: shuffling
                # after downsampling makes lone bright pixels that the 1 px filter scale
                # reads as ridges, inflating the baseline.
                shuffled = plane[ny, nx].copy()
                movable = region[ny, nx] & finite[ny, nx]
                seed = zlib.crc32(f"{field.field_id}|{role}|{r}|{c}".encode())
                shuffled[movable] = np.random.default_rng(seed).permutation(shuffled[movable])
                shuffled = _block_mean(shuffled, factor)
                shuffled = np.where(np.isfinite(shuffled), shuffled, median)
                baseline = _contrast(shuffled[use], _ridge(shuffled, sigmas)[use],
                                     cfg.ridge_fraction)
                if np.isfinite(contrast) and np.isfinite(baseline) and baseline > 0:
                    score = contrast / baseline
            rows.append(dict(
                field_id=field.field_id, role=str(role), marker=field.markers.get(role, ""),
                patch_row=r, patch_col=c, y0_px=y0, x0_px=x0, size_px=size,
                region_fraction=region_fraction, mean_intensity=mean,
                background=background, noise=noise, too_dim=too_dim,
                contrast=contrast, baseline=baseline, score=score,
                membrane_like=bool(np.isfinite(score) and score >= cfg.cutoff),
                _region_px=int(reg.sum())))     # weight for the field row; make_frame drops it
    return rows


def _field_row(field: Field, role: Role, rows: list[dict]) -> dict:
    """One channel's summary. ``n_scored`` counts patches that got a score.

    ``membrane_like_fraction`` is the share of scored cell-region pixels that lie in
    membrane-like patches, so a clipped edge patch weighs by the region it actually holds.
    """
    scored = [x for x in rows if np.isfinite(x["score"])]
    weight = sum(x["_region_px"] for x in scored)
    like = sum(x["_region_px"] for x in scored if x["membrane_like"])
    return dict(
        field_id=field.field_id, role=str(role), marker=field.markers.get(role, ""),
        n_patches=len(rows), n_scored=len(scored), n_too_dim=sum(x["too_dim"] for x in rows),
        median_score=float(np.median([x["score"] for x in scored])) if scored else math.nan,
        membrane_like_fraction=like / weight if weight > 0 else math.nan)


def score_field(field: Field, cfg: MembraneScoreConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every non-nucleus channel of ``field``: ``(membrane_patches, membrane_fields)``.

    Patches of ``round(patch_um / px)`` px tile the field from (0, 0); edge patches are
    clipped. A patch is scored when at least half of it is cell region, both its sides are
    at least half a full side, it is not too dim, and it spans at least one filter block
    (true at half a side; kept as a guard). Otherwise ``contrast``, ``baseline`` and
    ``score`` are NaN and ``membrane_like`` is False. ``too_dim`` is only ever True for a
    patch that passes the region and side tests.
    """
    px = field.pixel_size_um
    region = cell_region(field.planes[Role.NUCLEUS], px, cfg.region_um)
    size = max(1, round(cfg.patch_um / px))
    factor, sigmas = _filter_geometry(px, cfg.line_width_um)
    patches, fields = [], []
    for role in Role:
        if role is Role.NUCLEUS or role not in field.planes:
            continue
        rows = _patch_rows(field, role, region, cfg, size, factor, sigmas)
        patches += rows
        fields.append(_field_row(field, role, rows))
    return make_frame("membrane_patches", patches), make_frame("membrane_fields", fields)
