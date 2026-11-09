"""Ordered boundary traversal of a single labelled region."""
import numpy as np
from scipy import ndimage

from junctionmapper.errors import BoundaryError

# Clockwise (in image coordinates) 8-neighbourhood as (drow, dcol), starting West.
_DIRS = ((0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1))
_DIR_INDEX = {d: i for i, d in enumerate(_DIRS)}
_EIGHT = np.ones((3, 3), bool)
_FOUR = ndimage.generate_binary_structure(2, 1)
_MIN_AREA = 4


def trace_boundary(labels: np.ndarray, label: int) -> np.ndarray:
    """Return the closed boundary of ``label`` as an (N, 2) int array of (row, col).

    Moore-neighbour tracing, clockwise from the topmost-leftmost pixel, stopped when
    a (pixel, backtrack) state repeats; the returned array is exactly that cycle.
    Each pixel is 8-adjacent to the next and the last to the first. A pixel is
    emitted more than once only where the shape is a one-pixel-wide neck, so a
    consumer walking corner-to-corner segments must expect repeated points.
    """
    mask = np.asarray(labels) == label
    area = int(mask.sum())
    if area < _MIN_AREA:
        raise BoundaryError(f"label {label} is degenerate ({area} px; need >= {_MIN_AREA})")
    _, n = ndimage.label(mask, structure=_EIGHT)
    if n != 1:
        raise BoundaryError(f"label {label} is not contiguous ({n} components)")
    if (ndimage.binary_fill_holes(mask) & ~mask).any():
        raise BoundaryError(f"label {label} has an interior hole")

    pad = np.pad(mask, 1)
    start = tuple(int(v) for v in np.argwhere(pad)[0])   # raster order: top, then left
    pos, back = start, 0                                  # backtrack = West (background)
    path = []
    seen = {}                                             # (pos, back) state -> index in path
    # Jacob's criterion in its general form: the walk is deterministic in (pos, back),
    # so the first repeated state closes the cycle. The seed state (backtrack = West)
    # is artificial and need not lie on that cycle, so any prefix before it is dropped.
    for _ in range(8 * area + 8):
        if (pos, back) in seen:
            path = path[seen[(pos, back)]:]
            break
        seen[(pos, back)] = len(path)
        path.append(pos)
        for k in range(1, 9):
            d = (back + k) % 8
            nxt = (pos[0] + _DIRS[d][0], pos[1] + _DIRS[d][1])
            if pad[nxt]:
                prev = _DIRS[(back + k - 1) % 8]
                bpix = (pos[0] + prev[0], pos[1] + prev[1])
                back = _DIR_INDEX[(bpix[0] - nxt[0], bpix[1] - nxt[1])]
                pos = nxt
                break
    else:  # defensive: at most 8 * area distinct states exist, so a repeat is guaranteed
        raise BoundaryError(f"label {label}: boundary trace did not close")
    traced = np.array(path, dtype=int) - 1

    # Moore tracing can in principle cut diagonally past a pixel that touches the
    # background, which would silently shorten every contour measured along this
    # walk. Measured at 0 occurrences over 300 Cellpose-like blobs and three
    # pathological shapes, but verify rather than assume: a short contour is
    # invisible in the output, whereas a rejected cell is not.
    edge = mask & ~ndimage.binary_erosion(mask, _FOUR)
    missed = int(edge.sum()) - len({tuple(p) for p in traced} & {
        tuple(p) for p in np.argwhere(edge)
    })
    if missed:
        raise BoundaryError(
            f"label {label}: boundary trace missed {missed} of {int(edge.sum())} "
            "boundary pixels, so contours measured along it would be short"
        )
    return traced
