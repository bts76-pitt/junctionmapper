"""Move cell-cell boundaries onto the membrane ridge, then smooth them.

Cellpose draws smooth outlines that can sit a few pixels off the membrane. This step keeps
every cell where Cellpose put it, but lets the boundary BETWEEN two cells move onto the
brightest membrane within reach:

1. Seeds: every cell shrunk by ``erode_px``. A boundary can move at most this far.
2. Flood: watershed of the Gaussian-smoothed (``sigma_px``) segment plane from those seeds,
   inside the original foreground. Boundaries settle on bright ridges.
3. Smooth: each cell's mask is Gaussian-smoothed (``smooth_px``) and every foreground pixel
   goes to the cell whose smoothed mask is strongest there. Without this, boundaries hop
   between bright specks of a punctate membrane stain (over-fit, extra length).
   Seed pixels are never reassigned: smoothing favours the wider of two cells near
   background, and unclamped it would push boundaries past ``erode_px``.

Guarantees:
- The foreground is unchanged. Edges facing background are not junctions and do not move.
- Every input cell id survives. A cell too small to keep a seed keeps its own pixels as seed.
- dtype and ids are preserved; no new ids.

Defaults were picked by eye on plate-1 row E (Na/K-ATPase, 0.075 um px). Across its 120 fields
boundary length changed by at most 3% per line, and membrane signal on boundaries rose ~33%.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, maximum_filter, minimum_filter
from skimage.segmentation import watershed

ERODE_PX = 8
SIGMA_PX = 3.0
SMOOTH_PX = 20.0


def _seeds(cells: np.ndarray, erode_px: int) -> np.ndarray:
    """Each cell shrunk by ``erode_px``: a pixel is a seed if its (2k+1)^2 window holds only its
    own cell. Cells left with no seed keep all their pixels as seed (they cannot move)."""
    lab = cells.astype(np.int64)
    big = int(lab.max()) + 1
    k = 2 * erode_px + 1
    alone = minimum_filter(np.where(lab > 0, lab, big), size=k) == maximum_filter(lab, size=k)
    seeds = np.where(alone & (lab > 0), lab, 0)
    lost = np.setdiff1d(np.unique(lab[lab > 0]), np.unique(seeds[seeds > 0]))
    if lost.size:
        keep = np.isin(lab, lost)
        seeds[keep] = lab[keep]
    return seeds


def _smooth_labels(lab: np.ndarray, smooth_px: float) -> np.ndarray:
    """Reassign each foreground pixel to the cell with the strongest smoothed mask there."""
    out = np.zeros_like(lab)
    best = np.zeros(lab.shape, np.float32)
    pad = int(3 * smooth_px) + 1
    for cid, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        r = slice(max(sl[0].start - pad, 0), sl[0].stop + pad)
        c = slice(max(sl[1].start - pad, 0), sl[1].stop + pad)
        g = gaussian_filter((lab[r, c] == cid).astype(np.float32), smooth_px)
        win = g > best[r, c]
        best[r, c][win] = g[win]
        out[r, c][win] = cid
    out[lab == 0] = 0
    # A cell the smoothing swallowed whole gets its pre-smoothing pixels back.
    for cid in np.setdiff1d(np.unique(lab[lab > 0]), np.unique(out[out > 0])):
        out[lab == cid] = cid
    return out


def snap_to_membrane(cells: np.ndarray, membrane: np.ndarray, erode_px: int = ERODE_PX,
                     sigma_px: float = SIGMA_PX, smooth_px: float = SMOOTH_PX) -> np.ndarray:
    """Cell labels with cell-cell boundaries moved onto the ``membrane`` ridge and smoothed."""
    cells = np.asarray(cells)
    if not (cells > 0).any():
        return cells.copy()
    lab = cells.astype(np.int64)
    seeds = _seeds(lab, erode_px)
    flooded = watershed(gaussian_filter(np.asarray(membrane, np.float32), sigma_px),
                        markers=seeds, mask=lab > 0)
    smoothed = _smooth_labels(flooded.astype(np.int64), smooth_px)
    return np.where(seeds > 0, seeds, smoothed).astype(cells.dtype)


def snap_params(erode_px: int = ERODE_PX, sigma_px: float = SIGMA_PX,
                smooth_px: float = SMOOTH_PX) -> dict:
    """The parameters as recorded in label provenance."""
    return {"erode_px": int(erode_px), "sigma_px": float(sigma_px), "smooth_px": float(smooth_px)}
