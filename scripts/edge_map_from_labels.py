#!/usr/bin/env python3
"""Turn a cell-label image into the edge-map PNG the original Junction Mapper loads.

Why: step 1 of the Java application offers "Load Existing", which reads a binary edge-map PNG
and jumps to step 3. That is the only way to give the Java our segmentation without drawing it
by hand, so a human can click through the same cells and record the Java's numbers
(tests/validation/README.md).

    ./scripts/jm python scripts/edge_map_from_labels.py LABELS.npy EDGE_MAP.png [--overview OV.png]

LABELS.npy is a 2-D integer array, 0 = background, as written by ``junctionmapper segment``
(``<field_id>.JM1.npy``).

Which pixels are drawn
----------------------
A pixel is SET (white, value 1) when at least one of its four edge-adjacent neighbours inside
the image has a different label. That is the line *between* labels, drawn on both sides of
every contact, so:

* between two touching cells the line is two pixels wide (the last pixel of each cell);
* between a cell and background it is two pixels wide as well (the cell's outermost pixel and
  the background pixel beside it);
* pixels on the image border are not boundary merely for being on the border.

Everything else is clear (value 0).

Why this rule, and why it is not about leaks
--------------------------------------------
The Java fills a cell from a click with ``ImageUtilities.selectCell``. It VISITS all eight
neighbours of a pixel but EXPANDS only from pixels that have no set pixel among their eight
neighbours; a clear pixel next to the line is recorded as an "edge point" of the cell and never
expands. A line therefore cannot be crossed at all unless it has a hole, and a one-pixel
one-sided line works too (the Task 16 review ran the real flood on a one-sided 1-px barrier
over a staircase, a jogged grid and two Voronoi tilings and found no leak). Earlier versions of
this file claimed that a thin or one-sided line leaks; that came from modelling the flood as a
plain 8-connected labelling of the clear pixels, which it is not. Do not "simplify" the
barrier on the strength of either claim.

The two-sided, 4-neighbour line is kept for SYMMETRY and MARGIN: neither of two touching
cells is favoured by label order, the line has no holes even where labels meet at a corner, and
nothing depends on the exact expansion rule of one Java version.

Consequence the validation has to respect: the Java's cell is the label with that line removed
(each cell eroded by one pixel wherever it meets another label or background), minus any acute
tip the flood cannot reach. ``java_view_labels`` returns exactly that, modelling the flood as
the Java does it, and the harness measures our side on it.

The file is a 1-bit grayscale PNG (Pillow mode "1"). The Java's loader throws unless
``ImageIO.read`` returns ``TYPE_BYTE_BINARY``, and several of its code paths silently do nothing
on any other type, so an 8-bit "L" PNG is not an acceptable substitute.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _check_labels(labels) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"labels must be a 2-D array, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"labels must be an integer array, got dtype {arr.dtype}")
    if arr.size == 0 or arr.min() < 0:
        raise ValueError("labels must be non-empty and >= 0 (0 = background)")
    return arr


def edge_map_pixels(labels) -> np.ndarray:
    """Boolean image, True where the edge map is set (see the module docstring for the rule)."""
    lab = _check_labels(labels)
    edge = np.zeros(lab.shape, bool)
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]       # differs from the pixel above
    edge[:-1, :] |= lab[:-1, :] != lab[1:, :]      # differs from the pixel below
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]       # differs from the pixel to the left
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]      # differs from the pixel to the right
    return edge


def java_view_labels(labels) -> np.ndarray:
    """The labels as the Java's flood will see them: each cell's reachable clear pixels.

    Models ``ImageUtilities.selectCell`` (ImageUtilities.java:1356-1410): the flood expands only
    from clear pixels with no set pixel among their 8 neighbours ("interior" pixels) and adds
    every clear 8-neighbour of an expanding pixel to the cell. So a cell is its interior pixels
    plus their clear 8-neighbours (its "edge points"). Clear pixels that touch the line but are
    8-adjacent to no interior pixel, such as the tip of an acute corner, are NOT part of the
    Java's cell (1647 vs 1653 px on a 45-degree staircase in the review's run).

    One cell per label is assumed: a label whose interior pixels form two separate islands would
    need two clicks in the Java, and here both islands are returned under the one label (an upper
    bound; the harness then fails to trace it and skips the cell as degenerate).
    """
    lab = _check_labels(labels)
    edge = edge_map_pixels(lab)
    interior = ~ndimage.binary_dilation(edge, structure=np.ones((3, 3), bool))
    owner = ndimage.grey_dilation(np.where(interior, lab, 0), footprint=np.ones((3, 3), bool))
    return np.where(~edge & (owner > 0), owner, 0).astype(lab.dtype)


def labels_to_edge_map(labels: np.ndarray, out: Path) -> Path:
    """Write the edge map for ``labels`` to ``out`` (a 1-bit PNG) and return ``out``.

    The set pixels are exactly those of ``edge_map_pixels``: every pixel with a 4-neighbour of
    a different label, so the line between two touching cells is two pixels wide. Raises
    ValueError, and writes nothing, if ``labels`` is not a non-negative 2-D integer array.
    """
    edge = edge_map_pixels(labels)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(edge)                    # bool array -> mode "1"
    assert img.mode == "1"
    img.save(out, format="PNG")
    return out


def write_overview(labels: np.ndarray, out: Path, scale: int = 1) -> Path:
    """RGB picture of the labels with each id printed at its cell's centre.

    The operator needs the ids to fill in ``cell_id`` / ``neighbour_id`` in reference.csv.
    """
    lab = _check_labels(labels)
    rng = np.random.default_rng(0)
    palette = rng.integers(60, 230, size=(int(lab.max()) + 1, 3), dtype=np.uint8)
    palette[0] = 0
    rgb = palette[lab]
    rgb[edge_map_pixels(lab)] = 255
    img = Image.fromarray(rgb).resize((lab.shape[1] * scale, lab.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for i in np.unique(lab[lab > 0]):
        rows, cols = np.nonzero(lab == i)
        draw.text((cols.mean() * scale, rows.mean() * scale), str(int(i)), fill=(0, 0, 0),
                  anchor="mm")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", type=Path, help="2-D integer .npy label image (0 = background)")
    ap.add_argument("out", type=Path, help="edge-map PNG to write (1-bit)")
    ap.add_argument("--overview", type=Path, default=None,
                    help="also write a PNG with each cell's id printed on it")
    ap.add_argument("--overview-scale", type=int, default=2,
                    help="integer magnification of the overview (default 2)")
    args = ap.parse_args(argv)
    try:
        labels = np.load(args.labels, allow_pickle=False)
        path = labels_to_edge_map(labels, args.out)
        print(f"edge map written to {path} ({labels.shape[1]} x {labels.shape[0]} px, "
              f"{int(edge_map_pixels(labels).sum())} set pixels)")
        if args.overview is not None:
            print(f"overview written to {write_overview(labels, args.overview, args.overview_scale)}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
