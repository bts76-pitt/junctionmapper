"""Threshold primitive shared by every junction measurement.

The output keeps the greyscale value of each surviving pixel rather than
collapsing it to a boolean mask. Downstream code uses the same array both as a
mask (``> 0``) and as the intensity source, so the values must be preserved.
"""

import numpy as np


def mask_above(image: np.ndarray, t: float) -> np.ndarray:
    """Return a copy of ``image`` where pixels ``<= t`` are zeroed.

    Pixels strictly greater than ``t`` keep their original value. The result has
    the same dtype and shape as the input.
    """
    out = np.zeros_like(image)
    keep = image > t
    out[keep] = image[keep]
    return out
