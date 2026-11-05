import math

import numpy as np
import pytest

from junctionmapper.errors import ConfigError
from junctionmapper.measure.interface import (
    interface_contour,
    linearity_index,
    straight_line_length,
)


def test_contour_of_a_straight_run_is_its_step_sum():
    px = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    assert interface_contour(px) == pytest.approx(3.0)


def test_contour_counts_diagonal_steps_as_sqrt2():
    px = np.array([[0, 0], [1, 1], [2, 2]])
    assert interface_contour(px) == pytest.approx(2 * math.sqrt(2))


def test_straight_line_length_is_first_to_last_euclidean():
    px = np.array([[0, 0], [1, 1], [0, 4]])
    assert straight_line_length(px) == pytest.approx(4.0)


def test_linearity_of_a_straight_run_is_one():
    assert linearity_index(np.array([[0, 0], [0, 1], [0, 2]])) == pytest.approx(1.0)


def test_single_pixel_interface_has_zero_length_and_nan_linearity():
    px = np.array([[3, 3]])
    assert interface_contour(px) == 0.0
    assert math.isnan(linearity_index(px))


def test_two_pixel_interface_is_one_step_and_perfectly_linear():
    px = np.array([[0, 0], [1, 1]])
    assert interface_contour(px) == pytest.approx(math.sqrt(2))
    assert straight_line_length(px) == pytest.approx(math.sqrt(2))
    assert linearity_index(px) == pytest.approx(1.0)


def test_empty_interface_is_zero_length_and_nan_linearity():
    px = np.empty((0, 2), dtype=int)
    assert interface_contour(px) == 0.0
    assert straight_line_length(px) == 0.0
    assert math.isnan(linearity_index(px))


@pytest.mark.parametrize("bad", [np.zeros((3,)), np.zeros((3, 3)), np.zeros((2, 2, 2))])
def test_wrong_shape_raises_config_error(bad):
    for fn in (interface_contour, straight_line_length, linearity_index):
        with pytest.raises(ConfigError):
            fn(bad)


def test_non_adjacent_consecutive_pixels_raise_config_error():
    px = np.array([[0, 0], [0, 2]])
    with pytest.raises(ConfigError):
        interface_contour(px)


def test_long_straight_runs_have_linearity_exactly_one():
    horiz = np.stack([np.zeros(5000, int), np.arange(5000)], axis=1)
    diag = np.stack([np.arange(5000), np.arange(5000)], axis=1)
    assert linearity_index(horiz) == 1.0
    assert linearity_index(diag) == pytest.approx(1.0, rel=1e-12)


def test_linearity_never_exceeds_one():
    diag = np.stack([np.arange(777), np.arange(777)], axis=1)
    assert linearity_index(diag) <= 1.0


def test_closed_loop_has_linearity_near_zero():
    ring = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [1, 0]])
    assert straight_line_length(ring) == pytest.approx(1.0)
    assert interface_contour(ring) == pytest.approx(7.0)
    assert linearity_index(ring) == pytest.approx(1 / 7)


def test_results_are_python_floats_in_pixels():
    px = np.array([[0, 0], [0, 1]], dtype=np.int32)
    assert type(interface_contour(px)) is float
    assert type(straight_line_length(px)) is float
    assert type(linearity_index(px)) is float
