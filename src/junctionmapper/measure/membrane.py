"""Outline-free membrane score: does a stain sit in thin lines, or spread through the cell?

No segmentation is used. The field is cut into square patches; within each patch's cell
region (nuclei grown outward, see ``cell_region``) a multi-scale line filter ranks pixels
by how ridge-like they are, and

    contrast = (mean intensity on the top ``ridge_fraction`` of ridge pixels - floor)
               / (mean intensity on the rest of the region - floor)
    baseline = the same, filter included, on one seeded shuffle of the region's pixels
    score    = contrast / baseline        (1 = no line structure)

Dividing by the shuffled baseline removes what any texture would give by chance (noise
alone puts the brightest pixels on the "ridge"), so a score means the same for any stain.
Subtracting the field's floor makes it independent of an additive offset as well as of gain.

A patch whose ridge pixels are not clearly above the field's floor is ``too_dim`` and gets
no score: a blank channel reads as too dim, never as diffuse. The test reads the ridge
pixels, not the whole region, because thin membranes barely move a region mean. "Clearly
above" is 5 x the field's noise, or ``min_ridge_intensity`` raw units when that is set: the
noise of a dim stain grows with its own signal, so a ratio cannot tell a near-blank channel
from a faint real one, and only the experiment knows the level that separates them.

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
NOISE_ZERO_RTOL = 1e-9           # noise at or below this x max |pixel| is roundoff: treated as 0
DIM_NOISE_MULTIPLE = 5.0         # too dim when (ridge mean - floor) < this x noise. Pure
                                 # noise puts it ~4 sd up: 2.33 from the 1st-percentile
                                 # floor to the mean, plus up to 1.75 for picking the
                                 # top tenth. Assumes ridge_fraction near 0.1: at 0.01 the
                                 # selection term is ~2.7 and pure noise nears 5
MIN_REGION_FRACTION = 0.5        # a patch is scored when at least this much is cell region
MIN_SIDE_FRACTION = 0.5          # a clipped edge patch is scored only when both its sides are
                                 # at least this fraction of a full side
MIN_SIGMA_PX = 1.0               # filter scales never go below one pixel
FILTER_MIN_SIGMA_PX = 1.01       # floor on the sigmas passed to sato: skimage 0.26
                                 # hessian_matrix truncates kernels at 8 sigma only when every
                                 # sigma is > 1, else at 100 sigma (about 10x slower)


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


def _mad_sd(x: np.ndarray) -> float:
    """1.4826 x MAD: a robust SD, 0 for an empty or mostly constant sample."""
    if not x.size:
        return 0.0
    return MAD_TO_SD * float(np.median(np.abs(x - np.median(x))))


def floor_and_noise(plane: np.ndarray, region: np.ndarray | None = None) -> tuple[float, float]:
    """The field's floor and noise for one channel, in the channel's own units.

        background   1st percentile of the finite pixels
        noise        1.4826 x MAD of the high-pass residual over the finite pixels above
                     the floor. The residual is the plane minus its Gaussian smoothing at
                     2 px, the smoothing taken over above-floor pixels only, so a large
                     exact-constant floor (zeroed or clipped pixels) neither zeroes the
                     MAD nor puts its edges into it. If that is 0, the robust spread of
                     the finite ``region`` pixels (all finite pixels without a region).
                     A residual MAD at roundoff level (<= 1e-9 x max |pixel|) counts as 0.
                     0 for a constant or quantisation-level plane

    Both scale with the plane and shift-cancel under an offset, so the too-dim test built
    on them is scale- and offset-invariant. NaN pixels are ignored; an all-NaN plane gives
    ``(nan, 0.0)``.
    """
    plane = np.asarray(plane, np.float64)
    finite = np.isfinite(plane)
    if not finite.any():
        return math.nan, 0.0
    vals = plane[finite]
    background = float(np.percentile(vals, BACKGROUND_PERCENTILE))
    above = finite & (plane > background)
    if above.any():
        weight = ndi.gaussian_filter(above.astype(np.float64), NOISE_HIGHPASS_SIGMA_PX)
        total = ndi.gaussian_filter(np.where(above, plane, 0.0), NOISE_HIGHPASS_SIGMA_PX)
        noise = _mad_sd(plane[above] - total[above] / weight[above])
    else:
        noise = 0.0
    # Isolated pixels of one value (hot or saturated specks) leave a residual of roundoff
    # (~1e-14), not exactly 0; it must not pass for noise.
    if noise <= NOISE_ZERO_RTOL * float(np.max(np.abs(vals))):
        noise = 0.0
    if noise == 0.0:
        use = finite if region is None or not (region & finite).any() else region & finite
        noise = _mad_sd(plane[use])
    return background, noise


def _filter_geometry(px_um: float, line_width_um: tuple[float, float]) -> tuple[int, list[float]]:
    """Downsampling factor and Sato sigmas (in downsampled px) covering ``line_width_um``.

    A line of width w is matched by sigma w / 2. The factor is the largest integer that keeps
    the smallest sigma at least 1 px after downsampling (cost: about 4x fewer pixels at 2),
    so the choice depends on pixel size only, never on image size. Sigmas are clamped to
    ``FILTER_MIN_SIGMA_PX`` (just over 1 px, see there) and spaced at most 1 px apart.
    """
    lo, hi = (max(w / px_um / 2, MIN_SIGMA_PX) for w in line_width_um)
    factor = max(1, int(lo // MIN_SIGMA_PX))
    lo, hi = max(lo / factor, FILTER_MIN_SIGMA_PX), max(hi / factor, FILTER_MIN_SIGMA_PX)
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


def _ridge_split(values: np.ndarray, response: np.ndarray,
                 ridge_fraction: float) -> tuple[float, float]:
    """(Mean of ``values`` on the top ``ridge_fraction`` of ``response``, mean on the rest).

    Ties are broken by position (stable sort), so the cut does not move under a change of
    intensity scale. Both NaN when either side would be empty.
    """
    n = values.size
    k = max(1, round(ridge_fraction * n))
    if n - k < 1:
        return math.nan, math.nan
    order = np.argsort(-response, kind="stable")
    return float(values[order[:k]].mean()), float(values[order[k:]].mean())


def _contrast(ridge: float, rest: float, background: float) -> float:
    """(``ridge`` - ``background``) / (``rest`` - ``background``); NaN unless the rest is
    above the floor."""
    rest -= background
    return float((ridge - background) / rest) if rest > 0 else math.nan


def _patch_rows(field: Field, role: Role, region: np.ndarray, cfg: MembraneScoreConfig,
                size: int, factor: int, sigmas: list[float]) -> list[dict]:
    plane = np.asarray(field.planes[role], np.float64)
    finite = np.isfinite(plane)
    background, noise = floor_and_noise(plane, region)
    dim_cut = DIM_NOISE_MULTIPLE * noise if cfg.min_ridge_intensity is None else cfg.min_ridge_intensity
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
            ridge = contrast = baseline = score = math.nan
            too_dim = False
            # The patch in filter pixels, and the same window snapped to the block grid in
            # native pixels. An edge patch narrower than a block holds none: left unscored.
            sy, sx = slice(y0 // factor, y1 // factor), slice(x0 // factor, x1 // factor)
            ny = slice(sy.start * factor, sy.stop * factor)
            nx = slice(sx.start * factor, sx.stop * factor)
            use = small_use[sy, sx]
            if scored and use.sum() > 1:
                if response is None:
                    response = _ridge(small, sigmas)
                ridge, rest = _ridge_split(small[sy, sx][use], response[sy, sx][use],
                                           cfg.ridge_fraction)
                # noise 0 means no signal above the floor at all (blank, or constant
                # specks on it): too dim under either cut.
                too_dim = not (noise > 0 and np.isfinite(ridge) and ridge - background >= dim_cut)
            if scored and not too_dim and np.isfinite(ridge):
                contrast = _contrast(ridge, rest, background)
                # Shuffle in native pixels, then downsample as the original was: shuffling
                # after downsampling makes lone bright pixels that the 1 px filter scale
                # reads as ridges, inflating the baseline.
                shuffled = plane[ny, nx].copy()
                movable = region[ny, nx] & finite[ny, nx]
                seed = zlib.crc32(f"{field.field_id}|{role}|{r}|{c}".encode())
                shuffled[movable] = np.random.default_rng(seed).permutation(shuffled[movable])
                shuffled = _block_mean(shuffled, factor)
                shuffled = np.where(np.isfinite(shuffled), shuffled, median)
                baseline = _contrast(*_ridge_split(shuffled[use], _ridge(shuffled, sigmas)[use],
                                                   cfg.ridge_fraction), background)
                if np.isfinite(contrast) and np.isfinite(baseline) and baseline > 0:
                    score = contrast / baseline
            rows.append(dict(
                field_id=field.field_id, role=str(role), marker=field.markers.get(role, ""),
                patch_row=r, patch_col=c, y0_px=y0, x0_px=x0, size_px=size,
                region_fraction=region_fraction, mean_intensity=mean, ridge_intensity=ridge,
                background=background, noise=noise, too_dim=too_dim,
                contrast=contrast, baseline=baseline, score=score,
                membrane_like=(bool(score >= cfg.cutoff)
                               if cfg.cutoff is not None and np.isfinite(score) else None),
                _region_px=int(reg.sum())))     # weight for the field row; make_frame drops it
    return rows


def _field_row(field: Field, role: Role, rows: list[dict]) -> dict:
    """One channel's summary. ``n_scored`` counts patches that got a score.

    ``membrane_like_fraction`` is the share of called cell-region pixels that lie in
    membrane-like patches, so a clipped edge patch weighs by the region it actually holds.
    NaN when no patch has a call (no cutoff set, or nothing scored).
    """
    scored = [x for x in rows if np.isfinite(x["score"])]
    called = [x for x in scored if x["membrane_like"] is not None]
    weight = sum(x["_region_px"] for x in called)
    like = sum(x["_region_px"] for x in called if x["membrane_like"])
    return dict(
        field_id=field.field_id, role=str(role), marker=field.markers.get(role, ""),
        n_patches=len(rows), n_scored=len(scored), n_too_dim=sum(x["too_dim"] for x in rows),
        median_score=float(np.median([x["score"] for x in scored])) if scored else math.nan,
        membrane_like_fraction=like / weight if weight > 0 else math.nan)


def score_field(field: Field, cfg: MembraneScoreConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every non-nucleus channel of ``field``: ``(membrane_patches, membrane_fields)``.

    Patches of ``round(patch_um / px)`` px tile the field from (0, 0); edge patches are
    clipped. A patch is scored when at least half of it is cell region, both its sides are
    at least half a full side, it spans at least two filter pixels of region (true at half
    a side; kept as a guard), and it is not too dim. Otherwise ``contrast``, ``baseline``
    and ``score`` are NaN and ``membrane_like`` is null. ``ridge_intensity`` (the mean on
    the ridge pixels, which the too-dim test reads) is set for every patch that passes the
    first three tests, too dim or not, and ``too_dim`` is only ever True for such a patch: a
    patch with no finite pixels is unscored, not too dim.
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
