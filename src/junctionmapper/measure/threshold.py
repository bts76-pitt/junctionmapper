"""Threshold primitive shared by every junction measurement.

The output keeps the greyscale value of each surviving pixel rather than
collapsing it to a boolean mask. Downstream code uses the same array both as a
mask (``> 0``) and as the intensity source, so the values must be preserved.
"""

import numpy as np

from junctionmapper.errors import ConfigError


def mask_above(image: np.ndarray, t: float) -> np.ndarray:
    """Return a copy of ``image`` where pixels ``<= t`` are zeroed.

    Pixels strictly greater than ``t`` keep their original value. The result has
    the same dtype and shape as the input.

    NaN pixels count as not above the threshold and are zeroed, since ``NaN > t``
    is false.

    A NaN threshold would zero the whole image and report every metric as zero, so
    it is rejected rather than propagated.
    """
    # np.isnan rather than math.isnan(float): isinstance(t, float) is False for
    # np.float32, so a float32 NaN would have slipped through and zeroed the image.
    if np.isscalar(t) and not isinstance(t, (bool, np.bool_)) and np.isnan(t):
        raise ConfigError("threshold is NaN; every pixel would be zeroed")
    out = np.zeros_like(image)
    keep = image > t
    out[keep] = image[keep]
    return out
