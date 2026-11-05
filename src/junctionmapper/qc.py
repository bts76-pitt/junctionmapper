"""Per-cell QC flags for operator review."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from junctionmapper.errors import BoundaryError, ConfigError
from junctionmapper.geometry.boundary import trace_boundary
from junctionmapper.geometry.corners_dp import dp_corners
from junctionmapper.geometry.corners_tcj import merge_short_runs, neighbour_at, tcj_corners

_NEIGHBOUR_RADIUS = 2       # same window as neighbour_at's default search_radius


@dataclass
class CellFlags:
    border_touching: bool
    isolated: bool
    degenerate: bool
    corner_methods_disagree: bool
    n_corners_dp: "int | None"
    n_corners_tcj: "int | None"
    all_contacts_filtered: bool = False


def assess_cell(labels: np.ndarray, label: int, epsilon: float, min_run: int = 3,
                search_radius: int = _NEIGHBOUR_RADIUS) -> CellFlags:
    """QC flags for ``label``.

    ``degenerate`` is True when the boundary cannot be traced or either corner method
    raises; the affected count is None and ``corner_methods_disagree`` is then False
    (nothing to compare). An invalid ``epsilon`` is a configuration error and raises
    ConfigError rather than being reported as a bad cell.

    ``corner_methods_disagree`` is also False whenever TCJ finds no corners at all.
    Zero TCJ corners means the neighbour identity never changes around the boundary --
    an isolated cell, or one wholly enclosed by a single neighbour -- so the two methods
    are not measuring the same thing and the comparison carries no signal. Both counts
    are still reported.

    ``all_contacts_filtered`` is True when the cell does have a neighbour but every
    contact is shorter than the TCJ ``min_run`` filter, so TCJ yields no interface for
    it. Without this flag such a cell would contribute nothing to the results with
    nothing to say why.
    """
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, np.floating, np.integer))
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ConfigError(f"epsilon must be a finite number > 0, got {epsilon!r}")
    for name, v in (("min_run", min_run), ("search_radius", search_radius)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ConfigError(f"{name} must be an integer >= 1, got {v!r}")
    if label == 0:
        raise ConfigError("label 0 is background, not a cell")
    labels = np.asarray(labels)
    mask = labels == label
    border = bool(mask.any() and (mask[0].any() or mask[-1].any()
                                  or mask[:, 0].any() or mask[:, -1].any()))
    near = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool),
                                   iterations=search_radius)
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
        n_tcj = len(tcj_corners(neighbour_at(labels, boundary, label, search_radius=search_radius), min_run=min_run))
    except ConfigError:
        degenerate = True
    disagree = (
        n_tcj is not None and n_tcj > 0
        and n_dp is not None and abs(n_dp - n_tcj) > 1
    )

    # A neighbour that every contact run is too short to keep: TCJ returns no
    # interface, so the cell would silently contribute nothing.
    all_filtered = False
    if not isolated and n_tcj == 0:
        raw = neighbour_at(labels, boundary, label, search_radius=search_radius)
        filtered = merge_short_runs(raw, min_run)
        all_filtered = bool((raw != 0).any() and not (filtered != 0).any())

    return CellFlags(border, isolated, degenerate, disagree, n_dp, n_tcj, all_filtered)
