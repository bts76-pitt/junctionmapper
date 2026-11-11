"""Cellpose-SAM segmentation of the junction-marker (JM1) and nucleus channels.

This replaces the Java tool's click-every-cell step. The wrapper is deliberately
thin: it hands each 2D plane to Cellpose and returns the integer label arrays
(0 = background) exactly as Cellpose produced them. It does NOT validate, fill
holes in, relabel, size-filter or otherwise post-process the labels; geometry
that cannot be traced (tiny, fragmented or holed labels) is reported downstream
by ``trace_boundary`` as a ``BoundaryError``, which is the place to decide what
to do about it.

Reproducibility: the wrapper itself adds no randomness and no hidden state, but
the labels are only as reproducible as Cellpose. Results can differ between CPU
and GPU, between devices, and between cellpose/torch versions (GPU kernels are
not guaranteed bit-exact run to run). ``params_used`` records the versions and
the device actually used so a run can be re-created as closely as possible, but
bit-identical labels are not guaranteed.

Limitations: one ``diameter`` and one pair of thresholds is applied to both the
cell and nucleus planes (nuclei may want different values; a per-role field can be
added to ``SegmentationParams`` later). The ``CellposeModel(...)`` / ``.eval(...)``
signatures are pinned by stubs only, and cpsam may ignore ``diameter``; run one
smoke test against real cellpose before release.

Cellpose and torch are optional (the ``segmentation`` extra) and are imported
inside :func:`segment_field` so the rest of the library imports without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any

import numpy as np

from junctionmapper.config import Role
from junctionmapper.errors import SegmentationError
from junctionmapper.io.reader import Field

# JM2 is a measurement channel, not a segmentation target.
_SEGMENTED_ROLES = (Role.JM1, Role.NUCLEUS)


@dataclass(frozen=True)
class SegmentationParams:
    model: str = "cpsam"
    diameter: float | None = None  # None: let Cellpose estimate / use its default
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    gpu: bool = True


@dataclass
class Segmentation:
    cells: np.ndarray  # JM1 labels, 0 = background
    nuclei: np.ndarray  # nucleus labels; all zeros if the field has no nucleus plane
    params_used: dict[str, Any]


def _import_cellpose():
    try:
        import cellpose
        from cellpose import models
    except ImportError as exc:
        raise SegmentationError(
            "Cellpose is not installed, so segmentation cannot run. "
            "Install the optional dependency with: pip install junctionmapper[segmentation]"
        ) from exc
    return cellpose, models


def _cellpose_version(cellpose) -> str:
    try:
        return metadata.version("cellpose")
    except metadata.PackageNotFoundError:
        return str(getattr(cellpose, "__version__", "unknown"))


def _torch_info() -> tuple[str, str | None]:
    """Return (best backend: "cuda" | "mps" | "cpu", torch version) without requiring torch."""
    try:
        import torch
    except ImportError:
        return "cpu", None
    version = str(getattr(torch, "__version__", "unknown"))
    if torch.cuda.is_available():
        return "cuda", version
    try:
        if torch.backends.mps.is_available():
            return "mps", version
    except AttributeError:
        pass
    return "cpu", version


def _as_float(value: Any) -> float | None:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return float(arr[0]) if arr.size else None


def segment_field(field: Field, params: SegmentationParams) -> Segmentation:
    """Segment the JM1 and nucleus planes of ``field`` with Cellpose.

    A missing nucleus plane yields an all-zero ``nuclei`` array of the cell
    array's shape (rather than None) so downstream code can treat "no nuclei"
    as an empty label image. A missing JM1 plane raises SegmentationError.
    """
    if Role.JM1 not in field.planes:
        raise SegmentationError(f"Field {field.field_id!r} has no JM1 plane to segment.")

    cellpose, models = _import_cellpose()

    backend, torch_version = _torch_info()
    use_gpu = bool(params.gpu) and backend != "cpu"  # fall back to CPU, never fail
    selected = backend if use_gpu else "cpu"
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=params.model)

    # If the model reports its device, trust that over our prediction.
    device = getattr(model, "device", None)
    device_type = getattr(device, "type", None)
    if device_type is not None:
        use_gpu = device_type != "cpu"

    labels: dict[Role, np.ndarray] = {}
    diameters: dict[str, float] = {}
    for role in _SEGMENTED_ROLES:
        plane = field.planes.get(role)
        if plane is None:
            continue
        # Cellpose normalises a float32 input IN PLACE, so it must never be handed the
        # field's own plane: ``run`` measures the same Field after segmenting it, and a
        # mutated plane silently becomes the measurement input (thresholds are calibrated
        # on the raw planes beforehand, so every marker metric collapses to zero).
        out = model.eval(
            plane.copy(),
            diameter=params.diameter,
            flow_threshold=params.flow_threshold,
            cellprob_threshold=params.cellprob_threshold,
        )
        labels[role] = np.asarray(out[0])
        # cellpose < 4 returns (masks, flows, styles, diams); 4.x returns 3 items.
        if len(out) >= 4:
            d = _as_float(out[3])
            if d is not None:
                diameters[role.name] = d

    cells = labels[Role.JM1]
    nuclei = labels.get(Role.NUCLEUS)
    if nuclei is None:
        nuclei = np.zeros(cells.shape, dtype=cells.dtype)

    params_used: dict[str, Any] = {
        "model": params.model,
        "diameter": params.diameter,
        "diameter_used": (
            params.diameter if params.diameter is not None else (diameters or None)
        ),
        "flow_threshold": params.flow_threshold,
        "cellprob_threshold": params.cellprob_threshold,
        "gpu_requested": bool(params.gpu),
        "gpu": bool(use_gpu),
        # Model's own report when it has one; otherwise the backend this wrapper
        # selected (not confirmed by the model).
        "device": str(device) if device is not None else selected,
        "device_source": "model" if device is not None else "wrapper_selection",
        "cellpose_version": _cellpose_version(cellpose),
        "torch_version": torch_version,
        "roles_segmented": [r.name for r in _SEGMENTED_ROLES if r in labels],
        "input_shape": [int(n) for n in cells.shape],
        "input_dtype": str(field.planes[Role.JM1].dtype),
        "field_id": field.field_id,
        "pixel_size_um": field.pixel_size_um,
        "normalization": "cellpose default (per-image), planes passed as single 2D arrays",
    }
    return Segmentation(cells=cells, nuclei=nuclei, params_used=params_used)
