# junctionmapper

Cell–cell junction quantification from fluorescence microscopy, as a Python
library and CLI.

This is a reimplementation of the measurement method published as **Junction
Mapper** (Imperial College London, [doi:10.7554/eLife.45413](https://doi.org/10.7554/eLife.45413)).
The original is a Java Swing desktop application; this project reimplements its
published parameter definitions with modern image I/O and automated
segmentation. All credit for the method belongs to the original authors.

## What it does

Reads OME-TIFF and OME-Zarr, segments cells with Cellpose-SAM, decomposes each
cell boundary into interfaces with neighbouring cells, and measures the
published junction parameters for up to two junction markers plus nuclei.
Emits tidy tables and a self-contained HTML report.

Marker identity is configuration, never an assumption: the protein occupying
each of the `JM1`, `JM2`, and `nucleus` roles is declared per sample and
travels with every row of output.

## Status

Under active development. See `docs/superpowers/specs/` for the design and
`docs/superpowers/plans/` for the implementation plan.

## Install

```bash
pip install -e '.[dev]'              # library and tests
pip install -e '.[dev,segmentation]' # adds Cellpose-SAM
```

## Differences from the original

Documented in the design, but the one that changes numbers: the fragmented
junction contour uses proper 8-connectivity, where the Java admits a gap
whenever a single coordinate differs by one. This follows the formula in the
original's own user guide rather than its implementation.
