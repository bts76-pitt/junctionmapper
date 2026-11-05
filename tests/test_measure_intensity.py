import math

import numpy as np
import pytest

from junctionmapper.errors import ConfigError
from junctionmapper.measure.interface import band_metrics, dilated_band

LINE = np.array([[5, c] for c in range(10)])


def line_band():
    """Undilated horizontal run along row 5, columns 0-9, on a 20x20 image."""
    return dilated_band(LINE, (20, 20), cycles=0)


def band_of(n, marker_px=0):
    """A band of exactly n pixels (row 0); returns (band, image) with the first
    ``marker_px`` pixels bright (200.0)."""
    band = np.zeros((20, 20), bool)
    band[0, :n] = True
    img = np.zeros((20, 20), np.float32)
    img[0, :marker_px] = 200.0
    return band, img


def test_one_dilation_cycle_widens_a_line_to_three_pixels():
    band = dilated_band(LINE, (20, 20), cycles=1)
    assert band[:, 5].sum() == 3


def test_two_cycles_widen_to_five_pixels():
    assert dilated_band(LINE, (20, 20), cycles=2)[:, 5].sum() == 5


def test_zero_cycles_is_the_undilated_run():
    band = dilated_band(LINE, (20, 20), cycles=0)
    assert band.dtype == bool
    assert band.sum() == 10
    assert band[5, :10].all()


def test_duplicate_pixels_are_not_double_counted():
    px = np.array([[5, 0], [5, 1], [5, 2], [5, 1], [5, 0]])
    assert dilated_band(px, (20, 20), cycles=0).sum() == 3


def test_diagonal_run_dilates_with_full_3x3_element():
    band = dilated_band(np.array([[5, 5]]), (20, 20), cycles=1)
    assert band.sum() == 9


def test_band_clipped_by_image_edge_is_truncated():
    band = dilated_band(np.array([[0, 0], [0, 1]]), (10, 10), cycles=1)
    # rows 0-1, cols 0-2 survive; the rest falls off the image
    assert band.sum() == 6
    assert band.shape == (10, 10)


def test_empty_run_gives_empty_band():
    band = dilated_band(np.empty((0, 2), int), (10, 10), cycles=2)
    assert band.shape == (10, 10) and band.sum() == 0


def test_out_of_bounds_pixels_raise():
    with pytest.raises(ConfigError):
        dilated_band(np.array([[0, 10]]), (10, 10), cycles=1)
    with pytest.raises(ConfigError):
        dilated_band(np.array([[-1, 0]]), (10, 10), cycles=1)


@pytest.mark.parametrize("cycles", [-1, 1.5, True])
def test_bad_cycles_raise(cycles):
    with pytest.raises(ConfigError):
        dilated_band(LINE, (20, 20), cycles=cycles)


def test_bad_pixel_shape_raises():
    with pytest.raises(ConfigError):
        dilated_band(np.array([1, 2, 3]), (20, 20), cycles=1)


def test_interface_area_is_the_band_pixel_count():
    band, img = band_of(12)
    assert band_metrics(band, img, t=0).interface_area_px == 12


def test_marker_area_counts_only_pixels_above_threshold():
    img = np.zeros((20, 20), np.float32)
    img[5, 0:4] = 200.0
    img[5, 4:8] = 10.0  # below t
    assert band_metrics(line_band(), img, t=50).marker_area_px == 4


def test_intensity_sums_greyscale_values_not_pixel_counts():
    img = np.zeros((20, 20), np.float32)
    img[5, 0:4] = 200.0
    assert band_metrics(line_band(), img, t=50).marker_intensity == pytest.approx(800.0)


def test_intensity_ignores_pixels_outside_the_band():
    img = np.zeros((20, 20), np.float32)
    img[5, 0:4] = 200.0
    img[15, :] = 500.0
    assert band_metrics(line_band(), img, t=50).marker_intensity == pytest.approx(800.0)


def test_occupancy_is_marker_area_over_interface_area_as_percent():
    band, img = band_of(20, marker_px=5)
    m = band_metrics(band, img, t=50)
    assert m.interface_occupancy_pct == pytest.approx(25.0)


def test_cluster_density_and_intensity_per_area():
    band, img = band_of(20, marker_px=5)
    m = band_metrics(band, img, t=50)
    assert m.cluster_density == pytest.approx(200.0)  # 1000 / 5
    assert m.intensity_per_area == pytest.approx(50.0)  # 1000 / 20


def test_zero_marker_area_gives_nan_cluster_density_not_zero_division():
    m = band_metrics(line_band(), np.zeros((20, 20), np.float32), t=50)
    assert math.isnan(m.cluster_density)
    assert m.marker_area_px == 0
    assert m.marker_intensity == 0.0
    assert m.intensity_per_area == 0.0


def test_empty_band_gives_nan_ratios_not_errors():
    m = band_metrics(np.zeros((20, 20), bool), np.ones((20, 20), np.float32), t=0)
    assert m.interface_area_px == 0
    assert math.isnan(m.interface_occupancy_pct)
    assert math.isnan(m.intensity_per_area)
    assert math.isnan(m.cluster_density)


def test_marker_image_shape_mismatch_raises():
    with pytest.raises(ConfigError):
        band_metrics(line_band(), np.zeros((20, 21), np.float32), t=0)


def test_non_boolean_band_raises():
    with pytest.raises(ConfigError):
        band_metrics(np.ones((20, 20), np.uint8), np.zeros((20, 20), np.float32), t=0)


def test_negative_and_nan_pixels_never_count_above_a_nonnegative_threshold():
    img = np.zeros((20, 20), np.float32)
    img[5, 0] = -300.0
    img[5, 1] = np.nan
    img[5, 2] = 100.0
    m = band_metrics(line_band(), img, t=0)
    assert m.marker_area_px == 1
    assert m.marker_intensity == pytest.approx(100.0)


def test_integer_images_sum_without_overflow():
    img = np.zeros((20, 20), np.uint8)
    img[5, 0:10] = 255
    m = band_metrics(line_band(), img, t=0)
    assert m.marker_intensity == pytest.approx(2550.0)


def test_nan_threshold_raises():
    with pytest.raises(ConfigError):
        band_metrics(line_band(), np.zeros((20, 20), np.float32), t=float("nan"))
