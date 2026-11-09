"""Hand-built label fields for the pipeline tests.

``base``: 2x2 block of 8x8 squares, 1=(10:18,10:18) 2=(10:18,18:26) 3=(18:26,10:18)
4=(18:26,18:26) in a 40x40 field. The dp interface of cell 4 against cell 2 is row 18,
cols 18..25.

``zoo``: 60x60 field with one cell per situation (see ZOO_NOTES).
"""
from pathlib import Path

import numpy as np

from junctionmapper.config import ChannelSpec, Role, RunConfig
from junctionmapper.io.reader import Field
from junctionmapper.segment.cellpose import Segmentation

PX = 0.5
T1, T2 = 50.0, 20.0

ZOO_NOTES = """
1  square (10:20,10:20) beside 2: bent tcj interface; 2 nuclei (multinucleate)
2  L-shape wrapping cell 1's right and bottom sides; 1 nucleus
3  two separate blobs under one label: untraceable -> degenerate
4  6x6 square whose only 'contact' is a 2x2 cell 5 at a diagonal 2 px away -> filtered
5  the 2x2 cell
6  touches the top border
7  isolated 6x6 square
"""


def make_cfg(**kw):
    kw.setdefault("threshold", {Role.JM1: T1, Role.JM2: T2})
    return RunConfig(
        channels=[ChannelSpec(role=Role.JM1, channel=0, marker="Na/K-ATPase"),
                  ChannelSpec(role=Role.JM2, channel=1, marker="p120"),
                  ChannelSpec(role=Role.NUCLEUS, channel=2, marker="DAPI")],
        pixel_size_um=PX, **kw)


def _field(lab, nuc, jm1, jm2, field_id, px):
    f = Field(planes={Role.JM1: jm1, Role.JM2: jm2, Role.NUCLEUS: np.zeros_like(jm1)},
              pixel_size_um=px, markers={Role.JM1: "Na/K-ATPase", Role.JM2: "p120"},
              source=Path("x.tif"), field_id=field_id)
    return f, Segmentation(cells=lab, nuclei=nuc, params_used={})


def build(field_id="f1", px=PX):
    """Base fixture.

    JM1: 100 on row 18 cols 18..21, decoy 40 on cols 22..23 (passes T2=20, fails T1=50).
    JM2: 30 on row 18 cols 18..25 (passes T2, fails T1).
    """
    lab = np.zeros((40, 40), np.int32)
    lab[10:18, 10:18] = 1
    lab[10:18, 18:26] = 2
    lab[18:26, 10:18] = 3
    lab[18:26, 18:26] = 4
    nuc = np.zeros_like(lab)
    nuc[12:15, 12:15] = 1      # inside cell 1
    nuc[20:23, 20:23] = 2      # inside cell 4
    jm1 = np.zeros(lab.shape, np.float32)
    jm1[18, 18:22] = 100
    jm1[18, 22:24] = 40
    jm2 = np.zeros(lab.shape, np.float32)
    jm2[18, 18:26] = 30
    return _field(lab, nuc, jm1, jm2, field_id, px)


def field_with_isolated_cell():
    f, seg = build()
    seg.cells[seg.cells == 3] = 0
    seg.cells[32:36, 30:34] = 3          # far from everything, off the border
    return f, seg


def field_with_border_cell():
    f, seg = build()
    seg.cells[seg.cells == 1] = 0
    seg.cells[0:6, 0:6] = 1              # touches the top-left border
    seg.nuclei[:] = 0
    return f, seg


def grid3():
    """3x3 grid of 8x8 cells labelled 1..9; the centre cell 5 has exactly 4 neighbours."""
    lab = np.zeros((30, 30), np.int32)
    k = 1
    for r in range(3):
        for c in range(3):
            lab[3 + r * 8:11 + r * 8, 3 + c * 8:11 + c * 8] = k
            k += 1
    z = np.zeros(lab.shape, np.float32)
    return _field(lab, np.zeros_like(lab), z + 1, z + 1, "g", PX)


def zoo(field_id="z", px=PX):
    """See ZOO_NOTES.

    The tcj interface of cell 1 vs 2 walks (10,18),(10,19), down col 19 to (19,19), left along
    row 19 to (19,10), up to (17,10): 22 pixels, 21 orthogonal steps.
    JM1 (T=50) above at (10..12,19) and (19,11..12); decoy 40 at (14,19) (below T1, above T2).
    JM2 (T=20) value 30 at (13..15,19) and (19,14..17).
    Nuclei centroids (midpoint rule): n1 (13,13) and n2 (13,16) in cell 1, n3 (25,21) in
    cell 2, n4 (46,26) over background (orphan).
    """
    lab = np.zeros((60, 60), np.int32)
    lab[10:20, 10:20] = 1
    lab[10:20, 20:30] = 2
    lab[20:30, 10:30] = 2
    lab[40:44, 5:9] = 3
    lab[40:44, 15:19] = 3
    lab[34:40, 36:42] = 4
    lab[41:43, 43:45] = 5
    lab[0:6, 50:56] = 6
    lab[50:56, 10:16] = 7
    nuc = np.zeros_like(lab)
    nuc[12:15, 12:15] = 1
    nuc[12:15, 15:18] = 2
    nuc[24:27, 20:23] = 3
    nuc[45:48, 25:28] = 4
    jm1 = np.zeros(lab.shape, np.float32)
    for rc in ((10, 19), (11, 19), (12, 19), (19, 12), (19, 11)):
        jm1[rc] = 100
    jm1[14, 19] = 40
    jm2 = np.zeros(lab.shape, np.float32)
    for rc in ((13, 19), (14, 19), (15, 19), (19, 17), (19, 16), (19, 15), (19, 14)):
        jm2[rc] = 30
    return _field(lab, nuc, jm1, jm2, field_id, px)
