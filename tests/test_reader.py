from pathlib import Path

import bioio_base.exceptions
import numpy as np
import pytest
import tifffile
import zarr

from junctionmapper.config import ChannelSpec, Role, RunConfig
from junctionmapper.errors import (
    CalibrationError,
    ChannelError,
    FormatError,
    JunctionMapperError,
)
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


def write_ome(
    path: Path,
    data: np.ndarray,
    axes: str,
    size_um: float | None = 0.5,
    size_y_um: float | None = None,
) -> Path:
    meta = {"axes": axes, "Channel": {"Name": NAMES}}
    if size_um is not None:
        meta |= {"PhysicalSizeX": size_um, "PhysicalSizeY": size_y_um or size_um}
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


def test_reader_errors_are_catchable_as_the_common_base(tmp_plain_tiff):
    cfg = make_cfg(jm1=1, jm2=2, nucleus=0, pixel_size_um=None)
    with pytest.raises(JunctionMapperError) as exc:
        list(read_fields(tmp_plain_tiff, cfg))
    assert isinstance(exc.value, CalibrationError)


def test_anisotropic_xy_pixels_raise_naming_both_values(tmp_path):
    path = write_ome(
        tmp_path / "aniso.ome.tif", planes_with_distinct_values(), "CYX", size_um=0.5, size_y_um=0.7
    )
    with pytest.raises(CalibrationError, match="isotropic") as exc:
        list(read_fields(path, make_cfg()))
    assert "0.5" in str(exc.value) and "0.7" in str(exc.value)


def test_equal_xy_pixels_are_accepted(tmp_path):
    path = write_ome(tmp_path / "iso.ome.tif", planes_with_distinct_values(), "CYX", size_um=0.4)
    assert next(read_fields(path, make_cfg())).pixel_size_um == pytest.approx(0.4)


def test_reads_ome_zarr_and_maps_channels_by_name(tmp_path):
    path = tmp_path / "img.ome.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=2)
    group.create_array("0", data=planes_with_distinct_values(), chunks=(1, 64, 64))
    axes = [{"name": "c", "type": "channel"}] + [
        {"name": n, "type": "space", "unit": "micrometer"} for n in "yx"
    ]
    scale = {"type": "scale", "scale": [1, 0.5, 0.5]}
    group.attrs["multiscales"] = [
        {"version": "0.4", "name": "img", "axes": axes,
         "datasets": [{"path": "0", "coordinateTransformations": [scale]}]}
    ]
    group.attrs["omero"] = {"channels": [{"label": n} for n in NAMES]}

    fields = list(read_fields(path, make_cfg()))
    assert len(fields) == 1
    f = fields[0]
    assert f.planes[Role.JM1].shape == (64, 64)
    assert f.markers[Role.JM1] == "Na/K-ATPase"
    assert f.planes[Role.JM1].mean() == 2
    assert f.planes[Role.NUCLEUS].mean() == 1
    assert f.pixel_size_um == pytest.approx(0.5)


def test_nanometer_zarr_is_converted_to_um(tmp_path):
    path = _write_zarr(tmp_path / "nm.ome.zarr", unit="nanometer", scale=500)
    assert next(read_fields(path, make_cfg())).pixel_size_um == pytest.approx(0.5)


def test_millimeter_zarr_is_converted_to_um(tmp_path):
    path = _write_zarr(tmp_path / "mm.ome.zarr", unit="millimeter", scale=0.0005)
    assert next(read_fields(path, make_cfg())).pixel_size_um == pytest.approx(0.5)


def test_unsupported_unit_raises_when_nothing_is_configured(tmp_path):
    path = _write_zarr(tmp_path / "px.ome.zarr", unit="parsec", scale=1)
    with pytest.raises(CalibrationError, match="parsec") as exc:
        list(read_fields(path, make_cfg(pixel_size_um=None)))
    assert "set pixel_size_um" in str(exc.value)


def test_configured_pixel_size_overrides_an_unconvertible_unit(tmp_path):
    """Metadata normally wins, but a unit we cannot convert must not make the file
    unanalysable when the operator has supplied a calibration explicitly."""
    path = _write_zarr(tmp_path / "px.ome.zarr", unit="parsec", scale=1)
    assert next(read_fields(path, make_cfg(pixel_size_um=0.325))).pixel_size_um == 0.325


def test_nanometer_ome_tiff_is_converted_to_um(tmp_path):
    path = tmp_path / "nm.ome.tif"
    meta = {
        "axes": "CYX", "Channel": {"Name": NAMES},
        "PhysicalSizeX": 500, "PhysicalSizeY": 500,
        "PhysicalSizeXUnit": "nm", "PhysicalSizeYUnit": "nm",
    }
    tifffile.imwrite(path, planes_with_distinct_values(), ome=True, metadata=meta)
    assert next(read_fields(path, make_cfg())).pixel_size_um == pytest.approx(0.5)


def _write_two_scene_ome(path: Path, sizes=(0.5, 0.25), names=(NAMES, NAMES)) -> Path:
    with tifffile.TiffWriter(path, ome=True) as w:
        for size, scene_names in zip(sizes, names):
            data = planes_with_distinct_values()
            w.write(
                data,
                metadata={
                    "axes": "CYX", "Channel": {"Name": scene_names},
                    "PhysicalSizeX": size, "PhysicalSizeY": size,
                },
            )
    return path


def test_scenes_iterate_as_independent_fields_with_own_calibration(tmp_path):
    path = _write_two_scene_ome(tmp_path / "scenes.ome.tif")
    fields = list(read_fields(path, make_cfg()))
    assert len(fields) == 2
    assert [f.pixel_size_um for f in fields] == [pytest.approx(0.5), pytest.approx(0.25)]
    assert len({f.field_id for f in fields}) == 2


def test_scenes_resolve_channel_names_per_scene(tmp_path):
    # Scene 1 orders its channels differently, so "NaK" is index 1 in scene 0 and 0 in scene 1.
    path = _write_two_scene_ome(
        tmp_path / "names.ome.tif", names=(NAMES, ["NaK", "DAPI", "Ecad"])
    )
    f0, f1 = list(read_fields(path, make_cfg()))
    # planes are filled with index + 1 in both scenes; only the name->index mapping differs
    assert f0.planes[Role.JM1].mean() == 2
    assert f1.planes[Role.JM1].mean() == 1


def test_field_id_scene_component_is_sanitised(tmp_path):
    path = _write_two_scene_ome(tmp_path / "s.ome.tif")
    for f in read_fields(path, make_cfg()):
        assert ":" not in f.field_id
    # direct check of the sanitiser on a hostile scene name
    from junctionmapper.io.reader import _SAFE_ID_CHARS

    assert _SAFE_ID_CHARS.sub("_", "a/b:c d") == "a_b_c_d"


def test_duplicated_tiff_axes_raise_format_error(tmp_path):
    path = tmp_path / "dup.tif"
    tifffile.imwrite(
        path,
        np.zeros((2, 3, 8, 8), np.uint16),
        photometric="minisblack",
        planarconfig="separate",
        metadata={"axes": "SQYX"},
    )
    with pytest.raises(FormatError, match="SQYX"):
        list(read_fields(path, make_cfg(jm1=0, jm2=1, nucleus=0)))


def test_uninterpretable_tiff_axes_raise_format_error(tmp_path):
    path = tmp_path / "bad.tif"
    tifffile.imwrite(path, np.zeros((2, 8, 8), np.uint16), metadata={"axes": "IYX"})
    with pytest.raises(FormatError):
        list(read_fields(path, make_cfg(jm1=0, jm2=1, nucleus=0)))


def test_only_one_xy_pixel_size_raises_naming_missing_axis():
    from types import SimpleNamespace

    from junctionmapper.io.reader import _metadata_pixel_size

    img = SimpleNamespace(physical_pixel_sizes=SimpleNamespace(X=0.5, Y=None, Z=None))
    with pytest.raises(CalibrationError, match="Y is missing"):
        _metadata_pixel_size(img)


def test_unreadable_non_tiff_becomes_a_format_error_not_a_bare_bioio_exception(tmp_path):
    """I4: bioio's UnsupportedFileFormatError is not a JunctionMapperError, so when it escaped
    the reader it escaped the CLI's per-field handling too and ended the whole run with a
    traceback, discarding the fields that had already been measured."""
    path = tmp_path / "junk.xyz"
    path.write_bytes(np.random.default_rng(0).bytes(256))
    with pytest.raises(FormatError) as exc:
        list(read_fields(path, make_cfg()))
    assert isinstance(exc.value, JunctionMapperError)
    assert "junk.xyz" in str(exc.value) and "OME-TIFF" in str(exc.value)
    assert not isinstance(exc.value, bioio_base.exceptions.UnsupportedFileFormatError)


def _write_zarr(path: Path, unit: str, scale: float) -> Path:
    group = zarr.open_group(path, mode="w", zarr_format=2)
    group.create_array("0", data=planes_with_distinct_values(), chunks=(1, 64, 64))
    axes = [{"name": "c", "type": "channel"}] + [
        {"name": n, "type": "space", "unit": unit} for n in "yx"
    ]
    group.attrs["multiscales"] = [
        {"version": "0.4", "name": "img", "axes": axes,
         "datasets": [{"path": "0", "coordinateTransformations": [
             {"type": "scale", "scale": [1, scale, scale]}]}]}
    ]
    group.attrs["omero"] = {"channels": [{"label": n} for n in NAMES]}
    return path
