import pandas as pd
import pytest

from junctionmapper import tables
from junctionmapper.measure.interface import interface_contour
from junctionmapper.pipeline import measure_field
from tests.fixtures13 import build, make_cfg


@pytest.fixture
def setup():
    return build() + (make_cfg(),)


def test_every_interface_row_has_its_pixel_walk(setup):
    f, seg, cfg = setup
    r = measure_field(f, seg, cfg)
    assert set(r.interfaces.interface_id) == set(r.paths)
    for p in r.paths.values():
        assert p.ndim == 2 and p.shape[1] == 2 and len(p) >= 2


def test_paths_do_not_change_any_table(setup):
    f, seg, cfg = setup
    a, b = measure_field(f, seg, cfg), measure_field(f, seg, cfg)
    pd.testing.assert_frame_equal(a.interfaces, b.interfaces)
    assert list(a.interfaces.columns) == list(tables.INTERFACE_COLUMNS)
    assert len(a.paths) == len(b.paths)


def test_path_contour_matches_the_table(setup):
    f, seg, cfg = setup
    r = measure_field(f, seg, cfg)
    row = r.interfaces.iloc[0]
    assert interface_contour(r.paths[row.interface_id]) == pytest.approx(row.interface_contour_px)
