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

## Output tables

Five tidy tables per run (`interfaces`, `cells`, `nuclei`, `nuclei_distances`, `flags`), each
written as both `.parquet` and `.csv`, all keyed by `field_id` so several fields concatenate.

### `interfaces` — one row per measured *cell-side* interface

The unit of measurement is a **cell-side interface**, as in the original, which measures the
interfaces of the one cell the operator selected. A junction between two cells that are both
included is therefore measured **twice**, once from each side, each with its own dilated band
and its own numbers; a junction whose neighbour touches the field border or is degenerate is
measured once, because that neighbour is never measured. The tables are not deduplicated: each
side is a real measurement of that cell's junction, and discarding one would discard the cell's
own view of it.

What that means downstream:

- An `n` over rows counts cell-side interfaces, not junctions, and interior junctions carry
  twice the weight of edge-adjacent ones in any pooled distribution.
- `pair_id` (`field|method|lo-hi`, the two cell ids sorted) is the canonical key of the cell
  pair and groups the two sides. A junction count is `df.pair_id.nunique()`; one row per
  junction is `df.groupby("pair_id").mean()` or `.first()`, whichever the question wants.
- `interface_id` is the unique key of one cell-side interface, shared by its JM1 and JM2 rows.
- The report states this wherever it gives an `n`, and its QC section gives the junction count
  beside the side count.

Also per row: `contact_fraction`, the fraction of the interface's boundary positions actually
adjacent to the neighbour it is labelled with (see divergence 5 below), `low_contact_fraction`
when that is under 0.8, and `closed_loop` for a whole-boundary interface, whose
`linearity_index` is ~0 by geometry rather than by tortuosity.

### Intensity thresholds: one value per marker, carried across the run

Four of the measured parameters depend on an intensity threshold T per marker channel. Unless
`threshold` is set in the run configuration, T is **proposed once for the whole run** and
applied to every field: Otsu is computed on each of up to five fields chosen deterministically
and spread across the run (sorted `field_id`, evenly spaced; all of them when there are five or
fewer), and the run value is the **median** of those values, so one saturated or empty field
cannot drag it. Fields whose plane is flat or non-finite contribute nothing, and if every
calibration field is like that the run value falls back to 0.0 and the report says so.

`threshold` in `interfaces` is the value applied; `threshold_field_otsu` is what Otsu on that
field's own plane would have given, and `threshold_deviates_from_run` is true when the two are
further apart than 10% of the applied value (never less than one grey level) — information for
diagnosing a bad field, not an error. A `threshold` in the configuration still wins outright,
per role, and the report's red *threshold varied, not comparable* banner fires only when the
**applied** values differ across fields by more than one grey level, i.e. when T was genuinely
overridden somewhere; it names the fields that differ.

### `nuclei_distances` — one row per *ordered* nucleus pair

The table is long: a row is `(nucleus_id, neighbour_nucleus_id, rank_from_nucleus,
distance_px, distance_um)`, and `rank_from_nucleus` is **that** nucleus's own ordering of
every distance involving it, nearest first. So `rank_from_nucleus == 1` is the nearest
neighbour of `nucleus_id`, for every nucleus in the field, and
`df[df.rank_from_nucleus == 1]` is the nearest-neighbour distance distribution — the readout
discohesion scoring needs.

With `all_pairs: true` an unordered pair therefore appears **twice**, once from each end, each
copy carrying the rank it has for its own nucleus. That is deliberate: a single row per pair
cannot give both endpoints a correct rank, and the k-nearest mode (the default) already repeats
a mutual nearest pair. For one copy of each distance, keep the rows with
`nucleus_id < neighbour_nucleus_id`. The `n_intra_distances` and `mean_intra_distance_*`
columns of `nuclei` are the count and mean of a nucleus's own rows, i.e. a plain
`groupby("nucleus_id")` of this table.

## Differences from the original

The method is the original's. These are the places where this implementation
deliberately does not reproduce the original *code*, each verified by reading
the Java source and each recorded in the code that diverges.

### 1. `linearity_index` is the reciprocal of the Java's ratio

The Java's `getDistanceRatio` is contour over straight-line length, so it is
≥ 1 and grows as an interface gets more tortuous. This project reports
`linearity_index` = straight-line over contour, so it is ≤ 1 and 1.0 means a
perfectly straight interface. Same information, inverted. Converting an
older spreadsheet means taking a reciprocal; the two conventions never appear
in one table.

### 2. The dilation band is uniform; the Java's thins at the image border

Both implementations grow the interface band by repeated 8-connected 3×3
dilation, and both count cycles as iterations. But the Java's
`ImageUtilities.getDilatedImage` zeroes the outermost pixel of every row and
column on each pass, so near an edge the band is clipped and the loss
compounds with each cycle: at the default 2 cycles the band is up to 2 px
narrower along the border than in the field interior. Interface area, marker
area and marker intensity all read off that band, so in the Java they are
systematically low for cells near the edge of the field. This project dilates
uniformly. Border-touching cells are flagged and excluded from the interface
table anyway, which is the more conservative handling of the same problem.

### 3. Marker connectivity is symmetric and the same for both markers

The Java's two marker walks disagree with each other, and its JM1 walk
disagrees with the formula in its own user guide:

- **JM1** accepts a step between two above-threshold pixels when *either*
  coordinate differs by one (an OR). A step of (Δrow 1, Δcol 30) therefore
  counts as "connected" and contributes its full ~30 px to the fragmented
  junction contour, bridging a gap the formula says should break it.
- **JM2** advances its running position on *every* contour pixel, not only
  above-threshold ones, so it also counts the single step from an unstained
  pixel into each fragment — an over-count proportional to the number of
  fragments. It computes no fragmented metric and no straight-line junction
  length at all.

This project applies one rule to both markers: a step joins two consecutive
above-threshold pixels only when both coordinate differences are at most 1,
which is 8-connectivity and is what the user guide's formula describes. JM1
and JM2 are then measured by the same functions, so the two markers are
directly comparable.

### 4. Nuclei are assigned to cells by pixel plurality, not centroid containment

The Java associates a nucleus with whichever cell contains its centroid point
(`Cell_Nucleus.setCellMembrane` → `getSelectedMembrane(getGeometricMean())`),
where that "centroid" is the floored midpoint of the bounding box rather than
a centre of mass. A nucleus whose centroid falls on a junction or in a gap is
associated with nothing. This project keeps the Java's midpoint formula for
the reported centroid, but assigns each nucleus to the cell holding the most
of its pixels, ties going to the lowest label, and calls a nucleus an orphan
only when it overlaps no cell at all.

This matters for the biology the tool is aimed at. In discohesive tissue,
nuclei straddle junctions and gaps far more often, so a centroid rule inflates
the orphan and anucleate counts exactly in the condition under study.

### 5. Corners are found by Douglas–Peucker on the boundary, not on the convex hull

The original's own user guide cites Douglas & Peucker (1973) and describes
corner detection as that algorithm applied to the cell outline. Its code does
something else. `Cell_Membrane.getDouglasPeucker` takes the **convex hull** of
the cell's edge pixels first and then runs Douglas–Peucker over the hull
vertices, with a loop bound that never considers the last interior vertex of
each range and a point-to-line distance computed from an integer-truncated
slope.

Three consequences, measured by driving the original's own classes over 148
interfaces across rectilinear, jogged, staircase, Voronoi and hexagonal
geometries:

- A reflex (concave) vertex can never be a corner, because the hull discards
  it. Cells in discohesive tissue are frequently concave.
- A plain rectangle yields **three** corners rather than four, so one
  "interface" spans two sides of the cell.
- A regular hexagon yields eight to ten corners, **none of them at a vertex**.

This project runs Douglas–Peucker on the traced boundary itself, as the guide
describes, so a corner can fall on a concave turn and a rectangle has four of
them. Interface decompositions therefore differ from the original's for any
non-convex or oblique cell, and corner counts are not comparable between the
two implementations.

The tricellular (`tcj`) method, which has no counterpart in the original, cuts the boundary
exactly where the neighbouring identity changes. Douglas–Peucker corners know nothing about
neighbours, so a corner-to-corner run is assigned the neighbour holding the plurality of its
pixels, and two things follow that a reader has to be able to see:

- A dp interface can be mostly free edge. A neighbour touching 8 of a cell's 20-pixel side
  still wins the plurality, so dp reports one interface of 19 px contour against it where tcj
  reports 11 px: about 11 px of non-junction sit inside the measured band, depressing coverage
  and occupancy. `contact_fraction` gives the fraction of the interface actually against that
  neighbour (1.0 for tcj by construction), and `low_contact_fraction` plus the per-cell
  `dp_low_contact` flag mark it below 0.8.
- A short real contact can vanish. At 4 pixels of contact the plurality goes to background and
  dp emits no interface at all, while tcj finds one; the per-cell `dp_missed_tcj_contact` flag
  says so. Partial free edges are the phenotype in discohesive tissue, so this is not an
  academic case.

The rule itself stays: it is dp's published partition, and dp without it would need neighbour
information the method does not have. What is new here is the evidence, in the tables and in the
report's QC section.

This divergence is about *which* interface gets measured, not about how it is
measured. Given the same interface path, every metric in this project matches
the original exactly — verified on all 148 interfaces above for both junction
markers, including the integer pixel counts and intensity sums. The validation
harness in `tests/validation/` is built on that distinction: it takes the
original's corner coordinates as input and compares the measurements, while
reporting corner-count differences as information rather than as failures.
