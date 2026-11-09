# junctionmapper

Measures cell–cell junctions in fluorescence microscopy images, as a Python
library and a CLI.

This reimplements the measurement method published as **Junction Mapper**
(Imperial College London, [doi:10.7554/eLife.45413](https://doi.org/10.7554/eLife.45413)).
The original is a Java Swing desktop application. This project implements the
same published parameter definitions, with modern image I/O and automated
segmentation. The method is the original authors'.

## What it does

Reads OME-TIFF and OME-Zarr, segments cells with Cellpose-SAM, splits each cell
boundary into interfaces with its neighbours, and measures the published
junction parameters for up to two junction markers plus nuclei. Writes tidy
tables and a standalone HTML report.

`--input` takes one image or a directory of them. An OME-Zarr store is a
*directory* whose name ends `.zarr`, so it is taken as one input in its own
right rather than scanned for files inside it. A directory can contain stores,
`.tif`/`.tiff` files, or both, and is read in sorted order. Vendor formats
(`.czi`, `.nd2`, `.lif`, and so on) are out of scope, and are rejected by name
instead of being opened and then failing somewhere inside a reader.

You declare which protein occupies each of the `JM1`, `JM2` and `nucleus` roles,
per sample. Nothing is inferred from channel order, and the declaration is
written into every row of output.

## Status

Under active development. The design is in `docs/superpowers/specs/` and the
implementation plan in `docs/superpowers/plans/`.

## Install

```bash
pip install -e '.[dev]'              # library and tests
pip install -e '.[dev,segmentation]' # adds Cellpose-SAM
```

## Output tables

Each run writes five tidy tables (`interfaces`, `cells`, `nuclei`,
`nuclei_distances`, `flags`) as both `.parquet` and `.csv`. All are keyed by
`field_id`, so tables from several fields concatenate.

### `interfaces` — one row per measured *cell-side* interface

The unit of measurement is a cell-side interface, as in the original, which
measures the interfaces of the single cell the operator selected. A junction
between two cells that are both included is therefore measured from both sides,
each side with its own dilated band and its own numbers. A junction whose
neighbour touches the field border or is degenerate is measured from one side
only, because that neighbour never gets measured at all. The tables are not
deduplicated: each side is a real measurement of that cell's junction, and
dropping one would drop that cell's own view of it.

**One side is not always one row, and under `dp` it usually is not.** The `tcj`
method cuts the boundary where the neighbouring identity changes, so a cell has
exactly one interface per neighbour and a junction between two included cells is
exactly two rows. Douglas-Peucker corners know nothing about neighbours, so a
single run of contact can be split across several corner-to-corner intervals:
under `dp` one cell's contact with one neighbour routinely yields two to five
rows, and a single junction ten or more. Count junctions with `pair_id`, never
by counting rows or by assuming a factor of two.

What this means downstream:

- An `n` over rows counts cell-side interfaces, not junctions, and the ratio
  between the two is not a constant: it is 2 for an interior `tcj` junction but
  varies per junction under `dp`, so interior junctions are over-weighted in a
  pooled distribution by an amount that differs from junction to junction.
  Collapse on `pair_id` before pooling if each junction should count once.
- `pair_id` (`field|method|lo-hi`, the two cell ids sorted) identifies the cell
  pair and groups its two sides. Count junctions with `df.pair_id.nunique()`.
  For one row per junction, use `df.groupby("pair_id").mean()` or `.first()`,
  depending on the question.
- `interface_id` identifies one cell-side interface, and is shared by its JM1
  and JM2 rows.
- The report repeats this wherever it prints an `n`, and its QC section prints
  the junction count next to the side count.

Each row also has `contact_fraction`, the fraction of the interface's boundary
positions actually adjacent to the neighbour it is labelled with (see divergence
5 below); `low_contact_fraction`, when that drops under 0.8; and `closed_loop`,
for an interface spanning a whole boundary, whose `linearity_index` is near 0
for geometric reasons rather than from tortuosity.

### Intensity thresholds: one value per marker for the whole run

Four of the measured parameters depend on an intensity threshold T per marker
channel. Unless `threshold` is set in the run configuration, T is proposed once
for the whole run and applied to every field: Otsu runs on each of up to five
fields chosen deterministically and spread across the run (sorted `field_id`,
evenly spaced, or all of them when there are five or fewer), and the run value
is the median of those. One saturated or empty field therefore cannot drag it.

Some details of that choice:

- A field id does not exist until its file has been read, so the CLI opens up to
  five input files, spread over the sorted input list, to offer the choice. With
  one field per file the two selections are the same.
- A multi-scene or multi-Z file offers all of its fields, but Otsu runs only on
  the up-to-five that are finally chosen, so the cost stays bounded at five
  fields however much those files contain.
- A calibration file that cannot be read is skipped and another tried, up to ten
  opens in all, and the report records that the proposal was degraded.
- Fields whose plane is flat or non-finite contribute nothing. If every
  calibration field is like that, the run value falls back to 0.0 and the report
  says so.

`threshold` in `interfaces` is the value that was applied.
`threshold_field_otsu` is what Otsu on that field's own plane would have given,
and `threshold_deviates_from_run` is true when the two differ by more than 10%
of the applied value, with a floor of one grey level. It is information for
diagnosing a bad field, not an error.

Note what that floor does on a low-valued plane. When the run value is, say,
2.28, one grey level is already a large relative disagreement, so the flag sets
on the outer fields of an ordinary *default* run where nothing was overridden.
That is correct and intended: the floor exists so a 16-bit plane does not trip
on noise. The flag stays informational, so read it as "this field's own
plane would have been thresholded elsewhere", never as an error.

A `threshold` in the configuration overrides all of this, per role. The report's
red *threshold varied, not comparable* banner appears only when the applied
values differ between fields by more than one grey level. That has two possible
causes, and the banner states which one per marker role: either T was overridden
for some field, or no single run value could be resolved for that role at all,
because every calibration file was unreadable, leaving each field on its own
Otsu. In both cases it names the fields that establish the difference.

### `nuclei_distances` — one row per *ordered* nucleus pair

This table is long. A row is `(nucleus_id, neighbour_nucleus_id,
rank_from_nucleus, distance_px, distance_um)`, and `rank_from_nucleus` is that
nucleus's own ordering of every distance involving it, nearest first. So
`rank_from_nucleus == 1` is the nearest neighbour of `nucleus_id` for every
nucleus in the field, and `df[df.rank_from_nucleus == 1]` is the
nearest-neighbour distance distribution, which is what discohesion scoring
needs.

With `all_pairs: true`, an unordered pair appears twice, once from each end,
each copy with the rank it has for its own nucleus. That is on purpose: a single
row per pair cannot give both endpoints a correct rank, and the default
k-nearest mode already repeats a mutual nearest pair. For one copy of each
distance, keep the rows where `nucleus_id < neighbour_nucleus_id`. The
`n_intra_distances` and `mean_intra_distance_*` columns of `nuclei` are the
count and mean of a nucleus's own rows here, which is a plain
`groupby("nucleus_id")` of this table.

## Differences from the original

The method is the original's. Below are the places where this implementation
knowingly does not reproduce the original code. Each was verified by
reading the Java source, and each is recorded in the code that diverges.

### 1. `linearity_index` is the reciprocal of the Java's ratio

The Java's `getDistanceRatio` divides contour by straight-line length, so it is
at least 1 and grows as an interface becomes more tortuous. This project reports
`linearity_index` as straight-line over contour, so it is at most 1, and 1.0
means a perfectly straight interface. Same information, inverted. To convert an
older spreadsheet, take the reciprocal. The two conventions never appear in one
table.

### 2. The dilation band is uniform; the Java's thins at the image border

Both implementations grow the interface band by repeated 8-connected 3×3
dilation, and both count cycles as iterations. But the Java's
`ImageUtilities.getDilatedImage` (lines 660–690) suppresses a newly dilated
pixel whenever any of its 3×3 neighbours falls outside `[1, w−2] × [1, h−2]`.
Dilation is therefore dead at indices 0, 1, w−2 and w−1, a 2 px dead frame,
while pixels that were already set are left untouched. The loss compounds with
each cycle. Interface area, marker area and marker intensity are all read off
that band, so in the Java all three are systematically low for cells within a
couple of pixels of the field border. This project dilates uniformly
(`measure/interface.py`, `dilated_band`). Border-touching cells are flagged and
excluded from the interface table in any case, which handles the same problem
more conservatively.

### 3. Marker connectivity is symmetric and the same for both markers

The Java's two marker walks disagree with each other, and its JM1 walk disagrees
with the formula in its own user guide.

- JM1 accepts a step between two above-threshold pixels when either coordinate
  differs by one, an OR. A step of (Δrow 1, Δcol 30) therefore counts as
  connected and contributes its full ~30 px to the fragmented junction contour,
  bridging a gap the formula says should break it.
- JM2 tests the same step with an AND, both coordinate differences at most one,
  but advances its running position on every contour pixel rather than only
  the above-threshold ones. It therefore also counts the single step from an
  unstained pixel into each fragment, an over-count proportional to the number
  of fragments. JM2 does write a fragmented metric:
  `Membrane_SpreadSheet.java` lines 150 and 169–170 fill a "JM2 Fragmented
  Junction contour" column from `measurementDistanceToNeighbour`, and a "JM2
  Coverage index" from that. What the Java's JM2 genuinely lacks is the junction
  contour, the straight-line junction length and the junction linearity:
  `getMeasurementPoints` computes no `measurementEuclidianDistance`, and the JM2
  table has no columns for any of the three.

This project applies one rule to both markers: a step joins two consecutive
above-threshold pixels only when both coordinate differences are at most 1. That
is 8-connectivity, and it is what the user guide's formula describes. JM1 and
JM2 then go through the same functions, so the two markers are directly
comparable.

### 4. Nuclei are assigned to cells by pixel plurality, not centroid containment

The Java associates a nucleus with whichever cell contains its centroid point
(`Cell_Nucleus.setCellMembrane` → `getSelectedMembrane(getGeometricMean())`),
where that "centroid" is the floored midpoint of the bounding box rather than a
centre of mass. A nucleus whose centroid lands on a junction or in a gap is
associated with nothing. This project keeps the Java's midpoint formula for the
reported centroid, but assigns each nucleus to the cell containing most of its
pixels, with ties going to the lowest label, and calls a nucleus an orphan only
when it overlaps no cell at all.

This matters for the biology the tool is aimed at. In discohesive tissue, nuclei
straddle junctions and gaps far more often, so a centroid rule inflates the
orphan and anucleate counts in exactly the condition under study.

### 5. Corners are found by Douglas–Peucker on the boundary, not on the convex hull

The original's own user guide cites Douglas & Peucker (1973) and describes corner
detection as that algorithm applied to the cell outline. Its code does something
else. `Cell_Membrane.getDouglasPeucker` first takes the convex hull of the
cell's edge pixels, then runs Douglas–Peucker over the hull vertices, with a
loop bound that never considers the last interior vertex of each range and a
point-to-line distance computed from an integer-truncated slope.

What follows from that, both from reading the code and from driving the
original's own classes headless over the rectilinear control grid that
`scripts/java_reference.py` builds:

- A reflex (concave) vertex can never be a corner, because the hull discards it.
  Cells in discohesive tissue are frequently concave.
- The Java's corner count for a cell is **click-dependent**, because its flood
  fill starts from the clicked pixel and the edge-pixel set it reaches decides
  the hull. On the nine rectangular cells of the control grid, clicked at the
  seed the generator records, the Java gave four corners, matching this
  project's Douglas–Peucker count. A different click inside the same centre cell,
  one of the 14 pixels tied for maximum distance to the boundary, gives
  three, so one "interface" then spans two sides of the cell. Neither count
  is a property of rectangles: the click is part of the input, which is why
  `reference.csv` records it as `seed_row` and `seed_col`. See "Why the click
  matters" in `tests/validation/README.md`.
- The Java's epsilon slider has no published default; it starts at 0, where only
  hull vertices are candidates. So dp corner counts, and the dp interface
  decompositions that follow from them, are not comparable to any published
  Junction Mapper run, not merely different in principle.

A review-time measurement over 148 interfaces across jogged, staircase, Voronoi
and hexagonal geometries also found that a regular hexagon yields eight to ten
corners, none of them at a vertex. That driver is not committed and this
repository cannot regenerate the number. Treat it as a one-off observation, not
as evidence this project ships.

This project runs Douglas–Peucker on the traced boundary itself, as the guide
describes, so a corner can land on a concave turn and a rectangle has four of
them. Interface decompositions therefore differ from the original's for any
non-convex or oblique cell, and corner counts are not comparable between the two
implementations.

The tricellular (`tcj`) method has no counterpart in the original. It cuts the
boundary exactly where the neighbouring identity changes. Douglas–Peucker
corners know nothing about neighbours, so a corner-to-corner run is assigned to
the neighbour that covers the most of its pixels. Two consequences are visible
in the output:

- A dp interface can be mostly free edge. A neighbour touching 8 pixels of a
  cell's 20-pixel side still wins the plurality, so dp reports one interface of
  19 px contour against it where tcj reports 11 px. About 11 px of non-junction
  then sit inside the measured band, which depresses coverage and occupancy.
  `contact_fraction` gives the fraction of the interface actually against that
  neighbour, 1.0 for tcj by construction, and `low_contact_fraction` plus the
  per-cell `dp_low_contact` flag mark it when it falls below 0.8.
- A short real contact can disappear. At 4 pixels of contact the plurality goes
  to background and dp emits no interface at all, while tcj finds one; the
  per-cell `dp_missed_tcj_contact` flag records that. Partial free edges are the
  phenotype in discohesive tissue, so this is not an academic case.

The rule stays as it is. It is dp's published partition, and dp without it
would need neighbour information the method does not have. What is new here is
the evidence for its cost, in the tables and in the report's QC section.

This divergence is about which interface gets measured, not about how it is
measured, and the validation harness in `tests/validation/` is built on that
distinction. It takes the original's corner pixels as input and compares the
measurements along the shared path, reporting corner-count differences as
information rather than as failures.

## What the validation harness shows

`scripts/java_reference.py` drives the original's own classes headless and
writes `tests/validation/reference.csv`. `./scripts/jm pytest tests/validation`
then compares this project's functions along its own boundary, between the
Java's two corner pixels. The numbers below are the ones recorded in
`tests/validation/README.md`, from a real-Java run of 208 interfaces (416 rows,
one per interface and marker) over two Voronoi tilings and a hexagonal tiling.

- Where the two paths are the same pixel for pixel, the metrics agree
  exactly. On the straight rows, where the shared path is a single
  axis-aligned run on an axis-aligned outline, `interface_area_px`,
  `marker_area_px` and `marker_intensity` are identical to the Java with no
  tolerance, and `interface_contour_px`, `straight_line_length_px`,
  `linearity_index`, `junction_contour_px` and `junction_straight_line_px` agree
  to a relative 1e-9.
- Few rows are exactly comparable: 10 of those 208 interfaces, 20 of 416
  rows, about 5%. The harness rejects a reference containing no straight row, so
  the exact assertions can never silently fail to run.
- The other 396 rows agree only within tolerances: areas within 15% or
  16 px, intensity within 15%, linearity within 0.35, with contour and the
  junction lengths recorded rather than asserted. Worst observed: interface area
  −15 to +3 px (11.3%), marker area −12 to +6 px, intensity 14.0%, linearity
  0.31. Those differences come from the edge-map seam, not from the
  measurement. The Java's 8-neighbour edge set is two pixels thick on oblique
  edges, its walk follows the inner row, and it cuts convex corners diagonally,
  so the two paths are not the same pixels. They are not a tolerance on the
  port.
- `fragmented_contour_px` and `coverage_index` are recorded and never asserted, because of divergence 3 above. They are
  different on purpose.

So the accurate claim is this: *given the same interface path, the formulas
reproduce the original's numbers, exactly where the paths coincide pixel for
pixel and within a measured envelope where the edge-map seam makes them differ.*
Not "every metric matches exactly on every interface", which divergence 3 alone
makes false. Running the comparison needs a JDK and the Java checkout. Without
`reference.csv` those tests skip and name what is missing, and
`tests/validation/README.md` has the procedure.
