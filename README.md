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
cannot drag it. (A field id does not exist until its file has been read, so the CLI opens up to
five input files, spread over the sorted input list, to offer that choice; with one field per
file the two selections are the same.) Fields whose plane is flat or non-finite contribute nothing, and if every
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
`ImageUtilities.getDilatedImage` (lines 660–690) suppresses a *newly* dilated
pixel whenever any of its 3×3 neighbours falls outside
`[1, w−2] × [1, h−2]` — so dilation is dead at indices 0, 1, w−2 and w−1, a
2 px dead frame, while pixels that were already set are left untouched. The
loss compounds with each cycle. Interface area, marker area and marker
intensity all read off that band, so in the Java they are systematically low
for cells within a couple of pixels of the field border. This project dilates
uniformly (`measure/interface.py`, `dilated_band`). Border-touching cells are flagged and
excluded from the interface table anyway, which is the more conservative handling of the same
problem.

### 3. Marker connectivity is symmetric and the same for both markers

The Java's two marker walks disagree with each other, and its JM1 walk
disagrees with the formula in its own user guide:

- **JM1** accepts a step between two above-threshold pixels when *either*
  coordinate differs by one (an OR). A step of (Δrow 1, Δcol 30) therefore
  counts as "connected" and contributes its full ~30 px to the fragmented
  junction contour, bridging a gap the formula says should break it.
- **JM2** tests the same step with an AND (both coordinate differences at most
  one) but advances its running position on *every* contour pixel, not only
  above-threshold ones, so it also counts the single step from an unstained
  pixel into each fragment — an over-count proportional to the number of
  fragments. It does write a fragmented metric: `Membrane_SpreadSheet.java`
  lines 150 and 169–170 fill a "JM2 Fragmented Junction contour" column from
  `measurementDistanceToNeighbour`, and a "JM2 Coverage index" from it. What the
  Java's JM2 genuinely lacks is the junction contour, the straight-line junction
  length and the junction linearity: `getMeasurementPoints` computes no
  `measurementEuclidianDistance`, and the JM2 table has no columns for them.

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

Consequences, from the code and from driving the original's own classes
headless over the rectilinear control grid that `scripts/java_reference.py`
builds:

- A reflex (concave) vertex can never be a corner, because the hull discards
  it. Cells in discohesive tissue are frequently concave.
- The Java's corner count for a cell is **click-dependent**, because its flood
  fill starts from the clicked pixel and the edge-pixel set it reaches decides
  the hull. On the nine rectangular cells of the control grid, clicked at the
  seed the generator records, the Java gave **four** corners, matching our
  Douglas–Peucker count; a different click inside the same centre cell (one of
  the 14 pixels sharing the maximum distance to the boundary) gives **three**,
  so one "interface" then spans two sides of the cell. Neither count is a
  property of rectangles: the click is part of the input, which is why
  `reference.csv` records it (`seed_row`, `seed_col`). See
  `tests/validation/README.md`, "Why the click matters".
- Because the Java's epsilon slider has no published default (it starts at 0,
  where only hull vertices are candidates), dp corner counts and therefore dp
  interface decompositions here are not comparable to *any* published Junction
  Mapper run, not merely different in principle.

A review-time measurement over 148 interfaces across jogged, staircase, Voronoi
and hexagonal geometries also found that a regular hexagon yields eight to ten
corners, none of them at a vertex. That driver is not committed and the
repository cannot regenerate the number; it is cited here as a one-off
observation, not as evidence this project ships.

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
measured, and the validation harness in `tests/validation/` is built on that
distinction: it takes the original's corner pixels as **input** and compares the
measurements along the shared path, reporting corner-count differences as
information rather than as failures.

## What the validation harness demonstrates

`scripts/java_reference.py` drives the original's own classes headless and
writes `tests/validation/reference.csv`; `./scripts/jm pytest tests/validation`
then compares our functions along our boundary between the Java's two corner
pixels. The numbers below are the ones recorded in
`tests/validation/README.md`, from a real-Java run of 208 interfaces (416 rows,
one per interface and marker) over two Voronoi tilings and a hexagonal tiling:

- **Where the two paths are the same pixel for pixel, the metrics agree
  exactly.** On the *straight* rows — our shared path is a single axis-aligned
  run on an axis-aligned outline — `interface_area_px`, `marker_area_px` and
  `marker_intensity` are **identical** to the Java with no tolerance, and
  `interface_contour_px`, `straight_line_length_px`, `linearity_index`,
  `junction_contour_px` and `junction_straight_line_px` agree to a relative
  1e-9.
- **Few rows are exactly comparable**: 10 of those 208 interfaces (20 of 416
  rows, about 5%). The harness refuses a reference containing no straight row,
  so the exact assertions can never silently not run.
- The other 396 rows agree only **within tolerances** (areas within 15% or
  16 px, intensity within 15%, linearity within 0.35; contour and the junction
  lengths are recorded, not asserted). Worst observed: interface area −15 to
  +3 px (11.3%), marker area −12 to +6 px, intensity 14.0%, linearity 0.31.
  Those differences are an artefact of the **edge-map seam**, not of the
  measurement: the Java's 8-neighbour edge set is two pixels thick on oblique
  edges and its walk follows the inner row, and it cuts convex corners
  diagonally, so the two paths are not the same pixels. They are not a
  tolerance on the port.
- `fragmented_contour_px` and `coverage_index` are **recorded and never
  asserted**, by divergence 3 above: they are deliberately different.

So the honest claim is: *given the same interface path, the formulas reproduce
the original's numbers, exactly where the paths coincide pixel for pixel and
within a measured envelope where the edge-map seam makes them differ.* Not
"every metric matches exactly on every interface" — divergence 3 alone makes
that false. Running the comparison needs a JDK and the Java checkout; without
`reference.csv` those tests skip, naming what is missing, and
`tests/validation/README.md` has the procedure.
