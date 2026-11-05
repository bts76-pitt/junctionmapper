import math

import numpy as np
import pytest

from junctionmapper.errors import ConfigError
from junctionmapper.measure.interface import (
    coverage_index,
    fragmented_contour,
    junction_contour,
    junction_straight_line,
)


def straight_run(n):
    """n pixels along row 5, cols 0..n-1, as (row, col)."""
    return np.array([(5, c) for c in range(n)])


def _img(cols_on, shape=(20, 20), row=5):
    img = np.zeros(shape, np.float32)
    for c in cols_on:
        img[row, c] = 100.0
    return img


def two_fragment_image():
    # two 3-px fragments separated by a 4-px gap
    return _img([0, 1, 2, 7, 8, 9])


def fully_stained():
    return _img(range(10))


def l_shaped_run():
    # last=(100,50), next=(101,80): one coordinate differs by 1, other by 30
    return np.array([(100, 50), (101, 80)])


def gap_image():
    img = np.zeros((200, 200), np.float32)
    img[100, 50] = 100.0
    img[101, 80] = 100.0
    return img


def test_fragmented_contour_sums_only_connected_above_threshold_steps():
    assert fragmented_contour(straight_run(10), two_fragment_image(), t=50) == pytest.approx(4.0)


def test_fragmented_contour_excludes_gaps_the_java_or_test_would_admit():
    assert fragmented_contour(l_shaped_run(), gap_image(), t=50) == pytest.approx(0.0)


def test_junction_contour_spans_gaps_between_first_and_last_marker_pixel():
    assert junction_contour(straight_run(10), two_fragment_image(), t=50) == pytest.approx(9.0)


def test_junction_contour_of_the_or_case_spans_the_gap():
    assert junction_contour(l_shaped_run(), gap_image(), t=50) == pytest.approx(math.hypot(1, 30))


def test_junction_straight_line_is_first_to_last_above_threshold_pixel():
    img = _img([2, 3, 7])  # walk starts/ends on unstained pixels
    assert junction_straight_line(straight_run(10), img, t=50) == pytest.approx(5.0)
    assert junction_straight_line(l_shaped_run(), gap_image(), t=50) == pytest.approx(
        math.hypot(1, 30)
    )


def test_jm2_entry_step_from_unstained_pixel_is_not_counted():
    # pixels 0..3 below, 4..6 above: only the 2 steps inside the fragment count
    img = _img([4, 5, 6])
    assert fragmented_contour(straight_run(10), img, t=50) == pytest.approx(2.0)


def test_diagonal_step_counts_sqrt2():
    path = np.array([(0, 0), (1, 1), (2, 2)])
    img = np.zeros((5, 5), np.float32)
    img[0, 0] = img[1, 1] = img[2, 2] = 9.0
    assert fragmented_contour(path, img, t=1) == pytest.approx(2 * math.sqrt(2))


def test_coverage_is_fragmented_over_interface_contour():
    assert coverage_index(straight_run(10), two_fragment_image(), t=50) == pytest.approx(4.0 / 9.0)


def test_fully_stained_interface_has_coverage_one():
    assert coverage_index(straight_run(10), fully_stained(), t=50) == pytest.approx(1.0)


def test_no_marker_above_threshold_gives_zero_coverage():
    assert coverage_index(straight_run(10), np.zeros((20, 20), np.float32), t=50) == 0.0


def test_zero_or_one_marker_pixel_gives_zero_lengths():
    none = np.zeros((20, 20), np.float32)
    one = _img([4])
    for img in (none, one):
        assert fragmented_contour(straight_run(10), img, t=50) == 0.0
        assert junction_contour(straight_run(10), img, t=50) == 0.0
        assert junction_straight_line(straight_run(10), img, t=50) == 0.0
    assert coverage_index(straight_run(10), one, t=50) == 0.0


def test_coverage_is_nan_for_single_pixel_interface():
    img = _img([0])
    assert math.isnan(coverage_index(np.array([(5, 0)]), img, t=50))


def test_pixel_at_threshold_and_nan_pixels_are_below():
    img = _img([0, 1, 2])
    img[5, 1] = 50.0
    img[5, 2] = np.nan
    assert junction_contour(straight_run(10), img, t=50) == 0.0


def test_repeated_closed_loop_pixel_is_harmless():
    loop = np.array([(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)])
    img = np.full((5, 5), 9.0, np.float32)
    perimeter = 4.0
    assert fragmented_contour(loop, img, t=1) == pytest.approx(perimeter)
    assert junction_contour(loop, img, t=1) == pytest.approx(perimeter)
    assert junction_straight_line(loop, img, t=1) == 0.0
    assert coverage_index(loop, img, t=1) == pytest.approx(1.0)


def test_coverage_never_exceeds_one_on_random_walks():
    rng = np.random.default_rng(0)
    for _ in range(200):
        steps = rng.integers(-1, 2, size=(30, 2))
        path = np.cumsum(steps, axis=0) + 40
        img = (rng.random((80, 80)) > 0.4).astype(np.float32)
        c = coverage_index(path, img, t=0.5)
        assert 0.0 <= c <= 1.0 + 1e-12 or math.isnan(c)


def test_bad_inputs_raise_config_error():
    img = two_fragment_image()
    with pytest.raises(ConfigError):
        fragmented_contour(np.zeros((3, 3)), img, t=50)
    with pytest.raises(ConfigError):
        junction_contour(straight_run(10), img, t=float("nan"))
    with pytest.raises(ConfigError):
        junction_straight_line(np.array([(50, 50)]), img, t=50)
    with pytest.raises(ConfigError):
        coverage_index(straight_run(10), np.zeros(5), t=50)
