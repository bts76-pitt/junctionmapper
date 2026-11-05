"""Per-cell QC flags for operator review."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from junctionmapper.errors import BoundaryError, ConfigError
from junctionmapper.geometry.boundary import trace_boundary
from junctionmapper.geometry.corners_dp import dp_corners
from junctionmapper.geometry.corners_tcj import neighbour_at, tcj_corners

_NEIGHBOUR_RADIUS = 2       # same window as neighbour_at's default search_radius


@dataclass
class CellFlags:
    border_touching: bool
    isolated: bool
    degenerate: bool
    corner_methods_disagree: bool
    n_corners_dp: "int | None"
    n_corners_tcj: "int | None"


def assess_cell(labels: np.ndarray, label: int, epsilon: float) -> CellFlags:
    """QC flags for ``label``.

    ``degenerate`` is True when the boundary cannot be traced or either corner method
    raises; the affected count is None and ``corner_methods_disagree`` is then False
    (nothing to compare). An invalid ``epsilon`` is a configuration error and raises
    ConfigError rather than being reported as a bad cell.
    """
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, np.floating, np.integer))
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ConfigError(f"epsilon must be a finite number > 0, got {epsilon!r}")
    labels = np.asarray(labels)
    mask = labels == label
    border = bool(mask.any() and (mask[0].any() or mask[-1].any()
                                  or mask[:, 0].any() or mask[:, -1].any()))
    near = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool),
                                   iterations=_NEIGHBOUR_RADIUS)
    isolated = bool(mask.any() and not ((labels != 0) & (labels != label) & near).any())

    degenerate = False
    n_dp = n_tcj = None
    try:
        boundary = trace_boundary(labels, label)
    except BoundaryError:
        return CellFlags(border, isolated, True, False, None, None)
    try:
        n_dp = len(dp_corners(boundary, epsilon))
    except ConfigError:
        degenerate = True
    try:
        n_tcj = len(tcj_corners(neighbour_at(labels, boundary, label)))
    except ConfigError:
        degenerate = True
    disagree = n_dp is not None and n_tcj is not None and abs(n_dp - n_tcj) > 1
    return CellFlags(border, isolated, degenerate, disagree, n_dp, n_tcj)
