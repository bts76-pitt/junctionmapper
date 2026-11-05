from pathlib import Path

import numpy as np
import pytest
import tifffile

from junctionmapper.config import ChannelSpec, Role, RunConfig
from junctionmapper.errors import CalibrationError, ChannelError, JunctionMapperError
from junctionmapper.io.reader import read_fields

NAMES = ["DAPI", "NaK", "Ecad"]


def make_cfg(jm1="NaK", jm2="Ecad", nucleus="DAPI", pixel_size_um=0.325) -> RunConfig:
    channels = [
        ChannelSpec(role=Role.JM1, channel=jm1, marker="Na/K-ATPase"),
        ChannelSpec(role=Role.NUCLEUS, channel=nucleus, marker="DAPI"),
    ]
    if jm2 is not None:
        channels.append(ChannelSpec(role=Role.JM2, channel=jm2, marker="E-cadherin"))
    return RunConfig(channels=channels, pixel_size_um=pixel_size_um)


def write_ome(path: Path, data: np.ndarray, axes: str, size_um: float | None = 0.5) -> Path:
    meta = {"axes": axes, "Channel": {"Name": NAMES}}
    if size_um is not None:
        meta |= {"PhysicalSizeX": size_um, "PhysicalSizeY": size_um}
    tifffile.imwrite(path, data, ome=True, metadata=meta)
    return path


def planes_with_distinct_values(n_c: int = 3, lead: tuple[int, ...] = ()) -> np.ndarray:
    """Array (*lead, C, 64, 64) where each channel is filled with its index + 1."""
    arr = np.zeros((*lead, n_c, 64, 64), dtype=np.uint16)
    for c in range(n_c):
        arr[..., c, :, :] = c + 1
    return arr


@pytest.fixture
def tmp_ome_tiff(tmp_path):
    return write_ome(tmp_path / "img.ome.tif", planes_with_distinct_values(), "CYX")


@pytest.fixture
def tmp_ome_tiff_z3(tmp_path):
    return write_ome(tmp_path / "z3.ome.tif", planes_with_distinct_values(lead=(3,)), "ZCYX")


@pytest.fixture
def tmp_plain_tiff(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, planes_with_distinct_values(), photometric="minisblack")
    return path


def test_reads_ome_tiff_and_maps_channels_by_name(tmp_ome_tiff):
    fields = list(read_fields(tmp_ome_tiff, make_cfg()))
    assert len(fields) == 1
    f = fields[0]
    assert f.planes[Role.JM1].shape == (64, 64)
    assert f.markers[Role.JM1] == "Na/K-ATPase"
    # name -> channel mapping, not position: NaK is index 1, Ecad index 2, DAPI index 0
    assert f.planes[Role.JM1].mean() == 2
    assert f.planes[Role.JM2].mean() == 3
    assert f.planes[Role.NUCLEUS].mean() == 1
    assert all(p.dtype == np.float32 and p.ndim == 2 for p in f.planes.values())
    assert f.source == tmp_ome_tiff


def test_metadata_pixel_size_wins_over_config(tmp_ome_tiff):
    f = next(read_fields(tmp_ome_tiff, make_cfg(pixel_size_um=0.325)))
    assert f.pixel_size_um == pytest.approx(0.5)


def test_config_used_when_ome_metadata_has_no_pixel_size(tmp_path):
    path = write_ome(tmp_path / "nocal.ome.tif", planes_with_distinct_values(), "CYX", size_um=None)
    f = next(read_fields(path, make_cfg(pixel_size_um=0.325)))
    assert f.pixel_size_um == 0.325


def test_zstack_yields_one_field_per_plane(tmp_ome_tiff_z3):
    fields = list(read_fields(tmp_ome_tiff_z3, make_cfg()))
    assert len(fields) == 3
    assert len({f.field_id for f in fields}) == 3
    assert all(f.planes[Role.JM1].shape == (64, 64) for f in fields)


def test_z_and_t_iterate_independently(tmp_path):
    data = planes_with_distinct_values(lead=(2, 3))  # T=2, Z=3
    path = write_ome(tmp_path / "tz.ome.tif", data, "TZCYX")
    fields = list(read_fields(path, make_cfg()))
    assert len(fields) == 6
    assert len({f.field_id for f in fields}) == 6


def test_planes_follow_z_position(tmp_path):
    data = planes_with_distinct_values(lead=(3,))
    for z in range(3):
        data[z] += 10 * z
    path = write_ome(tmp_path / "zval.ome.tif", data, "ZCYX")
    means = [f.planes[Role.JM1].mean() for f in read_fields(path, make_cfg())]
    assert means == [2, 12, 22]


def test_config_without_jm2_yields_only_configured_roles(tmp_ome_tiff):
    f = next(read_fields(tmp_ome_tiff, make_cfg(jm2=None)))
    assert set(f.planes) == {Role.JM1, Role.NUCLEUS}
    assert set(f.markers) == {Role.JM1, Role.NUCLEUS}


def test_integer_channel_resolves_positionally(tmp_ome_tiff):
    f = next(read_fields(tmp_ome_tiff, make_cfg(jm1=2, jm2=None, nucleus=0)))
    assert f.planes[Role.JM1].mean() == 3
    assert f.planes[Role.NUCLEUS].mean() == 1


def test_missing_calibration_raises_naming_the_fix(tmp_plain_tiff):
    cfg = make_cfg(jm1=1, jm2=2, nucleus=0, pixel_size_um=None)
    with pytest.raises(CalibrationError, match="pixel_size_um"):
        list(read_fields(tmp_plain_tiff, cfg))


def test_config_pixel_size_overrides_absent_metadata(tmp_plain_tiff):
    cfg = make_cfg(jm1=1, jm2=2, nucleus=0)
    fields = list(read_fields(tmp_plain_tiff, cfg))
    assert fields[0].pixel_size_um == 0.325
    assert fields[0].planes[Role.JM1].mean() == 2


def test_channel_name_not_found_raises_naming_available_names(tmp_ome_tiff):
    cfg = make_cfg(jm1="NoSuchChannel")
    with pytest.raises(ChannelError, match="NoSuchChannel") as exc:
        list(read_fields(tmp_ome_tiff, cfg))
    assert "DAPI" in str(exc.value) and "Ecad" in str(exc.value)


def test_channel_index_out_of_range_raises(tmp_ome_tiff):
    with pytest.raises(ChannelError, match="7"):
        list(read_fields(tmp_ome_tiff, make_cfg(jm1=7)))


def test_channel_name_on_plain_tiff_raises_channel_error(tmp_plain_tiff):
    with pytest.raises(ChannelError, match="NaK"):
        list(read_fields(tmp_plain_tiff, make_cfg()))


def test_reader_errors_share_the_common_base():
    assert issubclass(CalibrationError, JunctionMapperError)
    assert issubclass(ChannelError, JunctionMapperError)
