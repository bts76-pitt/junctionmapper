"""Turn an image file plus a RunConfig into per-plane Field objects.

The analysis is strictly 2D: Z, T and scene are iterated as independent fields,
never analysed as volumes.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from bioio import BioImage
from bioio_base.exceptions import UnsupportedFileFormatError

from junctionmapper.config import ChannelSpec, Role, RunConfig
from junctionmapper.errors import CalibrationError, ChannelError


@dataclass
class Field:
    """One 2D plane set: a float32 plane per configured role."""

    planes: dict[Role, np.ndarray]
    pixel_size_um: float
    markers: dict[Role, str]
    source: Path
    field_id: str


def read_fields(path: Path, cfg: RunConfig) -> Iterator[Field]:
    """Yield one Field per scene/time/Z combination in ``path``.

    Pixel size comes from image metadata when present, otherwise from
    ``cfg.pixel_size_um``; if neither supplies one, CalibrationError is raised.
    """
    path = Path(path)
    try:
        yield from _read_bioio(path, cfg)
    except UnsupportedFileFormatError:
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise
        yield from _read_plain_tiff(path, cfg)


def _read_bioio(path: Path, cfg: RunConfig) -> Iterator[Field]:
    img = BioImage(path)
    names = [str(n) for n in img.channel_names]
    indices = _resolve_channels(cfg, names)
    pixel_size = _resolve_pixel_size(_metadata_pixel_size(img.physical_pixel_sizes), cfg)
    for scene in img.scenes:
        img.set_scene(scene)
        for t in range(img.dims.T):
            for z in range(img.dims.Z):
                data = img.get_image_data("CYX", T=t, Z=z)
                yield _build_field(
                    data, indices, cfg, pixel_size, path, f"{path.stem}_{scene}_t{t}_z{z}"
                )


def _read_plain_tiff(path: Path, cfg: RunConfig) -> Iterator[Field]:
    """Plain (non-OME) TIFF: no channel names or calibration, so cfg must supply them."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes, data = series.axes, series.asarray()
    # Generic stacks come back as Q (unknown) or S (samples); treat as channels.
    axes = axes.replace("S", "C").replace("Q", "C")
    unsupported = set(axes) - set("TZCYX")
    if unsupported or "Y" not in axes or "X" not in axes:
        raise ChannelError(f"cannot interpret TIFF axes {series.axes!r} in {path}")
    if "C" not in axes:
        data, axes = data[np.newaxis], "C" + axes
    n_c = data.shape[axes.index("C")]
    indices = _resolve_channels(cfg, [], n_c)
    pixel_size = _resolve_pixel_size(None, cfg)
    data = _expand(data, axes)  # -> TZCYX
    for t in range(data.shape[0]):
        for z in range(data.shape[1]):
            yield _build_field(
                data[t, z], indices, cfg, pixel_size, path, f"{path.stem}_t{t}_z{z}"
            )


def _expand(data: np.ndarray, axes: str) -> np.ndarray:
    for ax in "TZ":
        if ax not in axes:
            data, axes = data[np.newaxis], ax + axes
    return np.transpose(data, [axes.index(a) for a in "TZCYX"])


def _build_field(
    data: np.ndarray,
    indices: dict[Role, int],
    cfg: RunConfig,
    pixel_size: float,
    path: Path,
    field_id: str,
) -> Field:
    planes = {role: np.asarray(data[i], dtype=np.float32) for role, i in indices.items()}
    markers = {role: cfg.marker_for(role) for role in indices}
    return Field(planes, pixel_size, markers, path, field_id)


_ISOTROPY_RTOL = 1e-6


def _metadata_pixel_size(sizes) -> float | None:
    x, y = sizes.X, sizes.Y
    if x and y:
        if abs(x - y) > _ISOTROPY_RTOL * max(x, y):
            raise CalibrationError(
                f"image metadata gives anisotropic XY pixels (X={x} um, Y={y} um); "
                "the analysis assumes isotropic XY pixels"
            )
        return float(x)
    for value in (x, y):
        if value:
            return float(value)
    return None


def _resolve_pixel_size(metadata_um: float | None, cfg: RunConfig) -> float:
    if metadata_um is not None:
        return metadata_um
    if cfg.pixel_size_um is not None:
        return cfg.pixel_size_um
    raise CalibrationError(
        "no physical pixel size in the image metadata and none configured; "
        "set pixel_size_um in the run configuration"
    )


def _resolve_channels(
    cfg: RunConfig, names: list[str], n_channels: int | None = None
) -> dict[Role, int]:
    count = len(names) if n_channels is None else n_channels
    return {spec.role: _resolve_one(spec, names, count) for spec in cfg.channels}


def _resolve_one(spec: ChannelSpec, names: list[str], count: int) -> int:
    if isinstance(spec.channel, int):
        if not 0 <= spec.channel < count:
            raise ChannelError(
                f"channel index {spec.channel} for role {spec.role.value} is out of range; "
                f"image has {count} channel(s)"
            )
        return spec.channel
    if spec.channel not in names:
        raise ChannelError(
            f"channel name {spec.channel!r} for role {spec.role.value} not found; "
            f"available channel names: {names or 'none (use an integer index)'}"
        )
    return names.index(spec.channel)
