import numpy as np

from junctionmapper.measure.threshold import mask_above


def test_surviving_pixels_keep_their_greyscale_values():
    out = mask_above(np.array([[10, 60, 200]], dtype=np.float32), 50)
    assert out.tolist() == [[0.0, 60.0, 200.0]]


def test_is_not_a_binary_mask():
    assert mask_above(np.array([[200]], dtype=np.float32), 50).max() == 200.0


def test_threshold_is_exclusive():
    assert mask_above(np.array([[50]], dtype=np.float32), 50).tolist() == [[0.0]]
