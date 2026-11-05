import numpy as np
import pytest

from junctionmapper.errors import ConfigError
from junctionmapper.measure.threshold import mask_above


def test_surviving_pixels_keep_their_greyscale_values():
    out = mask_above(np.array([[10, 60, 200]], dtype=np.float32), 50)
    assert out.tolist() == [[0.0, 60.0, 200.0]]


def test_is_not_a_binary_mask():
    assert mask_above(np.array([[200]], dtype=np.float32), 50).max() == 200.0


def test_threshold_is_exclusive():
    assert mask_above(np.array([[50]], dtype=np.float32), 50).tolist() == [[0.0]]


def test_integer_dtype_is_preserved():
    """Downstream code sums these values; an unexpected float/bool cast would
    change the intensity metrics without changing any shape."""
    out = mask_above(np.array([[10, 60]], dtype=np.uint16), 50)
    assert out.dtype == np.uint16
    assert out.tolist() == [[0, 60]]


def test_the_input_is_not_modified():
    image = np.array([[10, 60]], dtype=np.float32)
    mask_above(image, 50)
    assert image.tolist() == [[10.0, 60.0]]


def test_nan_pixels_count_as_below_the_threshold():
    out = mask_above(np.array([[np.nan, 60.0]], dtype=np.float32), 50)
    assert out.tolist() == [[0.0, 60.0]]


def test_a_nan_threshold_is_rejected():
    """It would zero the whole image and report every metric as zero, which is
    indistinguishable from a field that genuinely has no signal."""
    with pytest.raises(ConfigError, match="NaN"):
        mask_above(np.array([[10.0, 60.0]], dtype=np.float32), float("nan"))


def test_a_float32_nan_threshold_is_also_rejected():
    """isinstance(t, float) is False for np.float32, so a float32 NaN once slipped
    through the guard and silently zeroed the whole image."""
    with pytest.raises(ConfigError, match="NaN"):
        mask_above(np.array([[10.0, 60.0]], dtype=np.float32), np.float32("nan"))
