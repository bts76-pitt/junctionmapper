# Java spot-check validation

This directory checks the Python port against the original Junction Mapper (Java). The tests
compare **metric formulas**, not corner detection: for every Java interface the reference
carries the Java's two corner pixels and the Java's numbers, and the harness measures our
functions along our boundary between those same two pixels.

Two ways to make `tests/validation/reference.csv`:

* **Automatic (use this).** `scripts/java_reference.py` runs the original Java classes headless
  (no display needed), clicks cells and writes the file: hundreds of interfaces in seconds.
* **Manual (the cross-check).** A person drives the Java GUI for one cell and transcribes its
  spreadsheet. On the same cell it must give the same numbers as the generator; that is how the
  generator itself is checked. Both are documented below.

What the tests do without `reference.csv`: the converter, the generator's helpers, a frozen slice
of real Java output (below) and the harness's own self-tests run; the comparison tests **skip**,
naming the file and this README. If `reference.csv` exists but is malformed they **fail**; if it
exists but yields no comparison at all (every row skipped) they **fail**, naming the count and
reasons; an undeclared skip fails `test_no_reference_row_was_silently_skipped`. Silence is never
an outcome.

## Read this before you start

* **The Java does not find the corners we do.** `Cell_Membrane.getDouglasPeucker` takes the
  *convex hull* of the edge pixels first, runs Douglas-Peucker over the hull vertices with a loop
  bound that never considers the last interior vertex of a range, and an integer-truncated slope
  (recorded divergence 5 in the repository `README.md`). A rectangle gets 3 corners, a regular
  hexagon 8 to 10 corners none of which is at a vertex. So "the Java's interface at epsilon E"
  and "our interface at epsilon E" are different interfaces. We keep our Douglas-Peucker as the
  published method describes it; the harness therefore takes the Java's corner pixels as
  **inputs** and never treats a difference in corner count as a failure. The Java's epsilon
  slider (an integer 0 to 50 that **starts at 0**, where only hull vertices are candidates) is
  recorded with each cell because it determines the Java's corners.
* **Measure persists nothing.** In the GUI only **"Save as Spreadsheet"** writes a file, and it
  writes **only the cell currently selected**. (The generator does not use the GUI.)
* **Douglas-Peucker only.** Our tricellular-junction method has no Java counterpart, so
  `reference.csv` cannot contain `tcj` rows (the loader rejects them).
* **Interior cells only.** The Java's dilation band is clipped near the image border, ours is
  not (recorded divergence 2). The generator marks a cell interior only if its whole band stays
  inside indices `[2, size-3]`; rows that are not interior, or whose cell touches the border,
  are skipped with a reason.
* **Settings are integers.** Dilation slider 1 to 9, thresholds 0 to 255 (the Java keeps pixels
  strictly above the threshold, as we do).

## What is compared

Each row is classified from the geometry, never from a hand-typed flag:

| class | definition | what is asserted |
|---|---|---|
| **straight** | our shared path is one axis-aligned run, on an outline that is axis-aligned near the path | **everything exactly**: `interface_area_px`, `marker_area_px`, `marker_intensity` identical (no tolerance); `interface_contour_px`, `straight_line_length_px`, `linearity_index`, `junction_contour_px`, `junction_straight_line_px` to relative 1e-9 |
| **other** | the path turns a corner or follows an oblique edge | areas within 15% or 16 px, intensity within 15%, linearity within 0.35, straight-line length to 1e-9; contour and the two junction lengths are **recorded** only |
| any | | `fragmented_contour_px` and `coverage_index` are **recorded, never asserted** (divergence 3) |

* `linearity_index` is compared as `1 / java_distance_ratio`: the Java prints contour over
  straight-line length (at least 1); this project reports its reciprocal (divergence 1).
* On *other* rows the Java and the port differ because the Java's 8-neighbour edge set is two
  pixels thick on oblique edges (its walk follows the inner row) and because it cuts convex
  corners diagonally. The tolerances are measured from the real Java (396 rows over two
  Voronoi tilings and a hexagonal tiling: interface area -15 to +3 px, worst 11.3%; marker area
  -12 to +6 px; intensity worst 14.0%; linearity within 0.31) and are an **artefact of the
  edge-map seam, not a tolerance on the port**. They apply only to rows that are not exactly
  comparable. Do not widen them to make a run pass: the *straight* rows are the evidence.
* **Straight rows are rare in real segmentations**, so the generator's `--control` adds a
  synthetic grid of rectangles. A reference with no straight row fails
  (`test_integer_metrics_match_java_exactly`: "the exact assertions were never applied").
* A **corner-count comparison** (the Java's count against our Douglas-Peucker count on the same
  boundary) is written to `divergence.md` as information. The Java hulls first, so the counts
  differ by construction; it is never a failure.
* A mismatch message prints, for the row, both endpoint pairs (the Java's corner pixels and the
  pixels of our boundary they snapped to) and the snap distances, so you can tell at once
  whether you are looking at a metric disagreement or a path disagreement. A Java corner more
  than 2 px from our boundary is reported as a **path disagreement** and fails.

## Automatic procedure

Needs a JDK (`java` and `javac`, 11 or later) and the Java checkout (its `uk/` sources and
`lib/*.jar`). The Apptainer image has neither, and does not see the Java checkout, so run the
generator on the host with a plain `python3` that has numpy, scipy and Pillow, and run the
tests in the image. The Java checkout is a read-only oracle: the generator compiles into the
scratch directory you give it and writes nothing into the Java tree.

1. **Make the labels.** For a real image, run `./scripts/jm junctionmapper segment ...`
   (the file you want is `OUT/labels/<field_id>.JM1.npy`) or use hand-corrected labels. The
   image must be the RGB TIFF the Java loads (band 0 = Junction 1, band 1 = Junction 2,
   band 2 = nuclei). Copy the TIFF and `.npy` into `tests/validation/reference/`. Skip this
   step to validate on the control grid only.

2. **Generate the reference** (on the host; scratch outside the Java tree):

   ```bash
   python3 scripts/java_reference.py \
       --java-dir /ix1/alee/LO_LAB/Personal/Brent_Schlegel/tools/Junction_Mapper \
       --work-dir .superpowers/java_work \
       --out tests/validation/reference.csv \
       --image tests/validation/reference/image_1.tif \
       --labels tests/validation/reference/image_1.JM1.npy \
       --control --n-cells 20 --epsilon 2 --dilation 2 \
       --threshold-jm1 100 --threshold-jm2 100
   ```

   `--cells 3,5,8` clicks named cells instead. Paths in the CSV are relative to it. A missing
   JDK, Java checkout or jar ends the run with one `error:` line saying what is missing.

3. **Run the comparison** in the image:

   ```bash
   ./scripts/jm pytest tests/validation -v
   ```

   This writes `tests/validation/divergence.md`.

4. **Read `divergence.md`.** The summary table says which classes agreed. A recorded
   `fragmented_contour_px` difference is attributed to divergence 3 only when the Java value is
   at least ours (it predicts the OR rule can only add length); otherwise it is `UNEXPLAINED`.
   A coverage difference says whether the numerator (fragmented contour) or the denominator
   (interface contour) differs.

5. **Commit** `reference.csv`, the TIFF and label files it points at, and `divergence.md`
   together. The file is deterministic, so a rerun on the same reference does not change it. We
   chose **not** to git-ignore `divergence.md`: it is the artefact a reader of the paper wants,
   it is only written when a reference exists, and committing it causes no churn.

## Manual procedure (the cross-check)

Do this for **one** cell and confirm the generator gives the same numbers for it. You need a
machine with a display (or X forwarding / VNC), a JDK 17 and the Java checkout (`$JAVA`).

1. **Prepare.** Put the RGB TIFF and the `.npy` labels in `tests/validation/reference/`
   (the `$JAVA/examples/image_*.tif` files are 800 x 600 RGB).

2. **Make the edge map and an overview.**

   ```bash
   ./scripts/jm python scripts/edge_map_from_labels.py \
       tests/validation/reference/image_1.JM1.npy \
       tests/validation/reference/image_1_edge_map.png \
       --overview tests/validation/reference/image_1_overview.png
   ```

   The PNG is 1-bit and white where a pixel has a 4-neighbour with a different label, so the
   line between two cells is two pixels wide and **each cell is eroded by one pixel** where it
   meets another cell or the background (a few acute-tip pixels more; the tests model the Java's
   flood exactly). The overview prints each cell's label id, which is `cell_id`.

3. **Choose the settings** and write them down: epsilon (say 2), dilation cycles (2), integer
   thresholds for Junction 1 and 2 (say 100 and 100). Pick a cell well inside the field.

4. **Start the Java** from a writable directory (its output directory `./yyyy_MM_dd_HHmm` is
   created under the current directory): `cd $JAVA && ./build.sh run`, or
   `java -jar dist/junction_mapper.jar`.

5. **Load the image and assign the bands.** File menu, open the TIFF; set band 0 to "Junction 1
   Channel", band 1 to "Junction 2 Channel", band 2 to "Nucleus". These are `jm1_channel`,
   `jm2_channel`, `nucleus_channel`.

6. **Inject the segmentation (step 1).** Press **Load Existing** and choose the edge-map PNG. If
   it reports "Edge map image is not binary type", the PNG is not 1-bit: regenerate it with the
   script (never re-save it in an image editor). You are at step 3; do not edit; press
   **Finish**.

7. **Select the cell (steps 4 and 5).** Click inside the cell, **Finish**, click it again.

8. **Step 6.** Set **Epsilon** to your epsilon (it returns to 0 each time you re-enter step 6),
   **# Dilations applied to edge** to your cycles, and the two thresholds. Release each slider.
   Do **not** add or remove corners with the mouse.

9. **Step 7.** Press **Measure**, then **Show Edge** repeatedly: each press prints one corner's
   console line, `# n :: java.awt.Point[x=X,y=Y] :: neighbour m :: java.awt.Point[x=X2,y=Y2] :: ...`.
   `x` is the column and `y` the row. Then press **Save as Spreadsheet** (Measure alone saves
   nothing, and only the selected cell is written).

10. **Transcribe from the `.xls`**, sheet `Cell # n`. Excel rows are 1-based; read the formula
    bar for full precision (the grid rounds, and rounding breaks the 1e-9 comparison).
    * Summary block, rows 1 to 8: `B4` is `# Corners` (`java_n_corners`); `B5` and `B7` are the
      thresholds (must equal what you set).
    * **JM1 table**: title row 11, header row 12, one row per corner from row 13.
    * **JM2 table**: starts five rows after the last JM1 row (title, header, then a row per
      corner), same columns up to `N`, no Junction Contour or Junction Straight-line columns.

    | reference.csv column | JM1 table | JM2 table |
    |---|---|---|
    | `java_corner`, `java_paired_with` (optional) | A "Corner #", B "Paired With #" | A, B |
    | `java_corner_row/col`, `java_paired_row/col` | from the Show Edge lines: corner A's `y`, `x`; its paired corner's `y`, `x` | same |
    | `interface_contour_px` | C "Interface Contour" | C |
    | `straight_line_length_px` | D "Interface Straight-line Length" | D |
    | `java_distance_ratio` | E "Interface Linearity Index" **as printed** (may be `Infinity` or `NaN`) | E |
    | `fragmented_contour_px` | F "JM1 Fragmented Junction Contour" | F "JM2 Fragmented Junction contour" |
    | `coverage_index` | G "JM1 Coverage Index" | G |
    | `dilation_cycles` | H (equals what you set) | H |
    | `interface_area_px` | I "Interface Area" | I |
    | `marker_area_px` | J "JM1 Area" | J "JM2 Area" |
    | `marker_intensity` | L "JM1 Intensity" | L "JM2 Intensity" |
    | `junction_contour_px` | O "JM1 Junction Contour" | blank |
    | `junction_straight_line_px` | P "JM1 Junction Straight-line Length" | blank |

    The Java's "Interface Linearity Index" is contour divided by straight-line length, at least 1.
    This project's `linearity_index` is its reciprocal. Copy the Java's number unchanged; the
    loader refuses a value below 1.

11. **Write `reference.csv`**: one row per (cell, Java interface, role), so each interface appears
    twice (`JM1` from the JM1 table, `JM2` from the JM2 table); geometry columns repeat. Exact
    header:

    ```
    image,labels,jm1_channel,jm2_channel,nucleus_channel,cell_id,role,corner_method,epsilon,dilation_cycles,threshold,interior,java_corner_row,java_corner_col,java_paired_row,java_paired_col,java_n_corners,interface_contour_px,straight_line_length_px,java_distance_ratio,fragmented_contour_px,coverage_index,interface_area_px,marker_area_px,marker_intensity,junction_contour_px,junction_straight_line_px,java_corner,java_paired_with,notes
    ```

12. **Compare with the generator.** Run the generator for the same cell
    (`--cells <id> --epsilon ... --dilation ... --threshold-jm1 ... --threshold-jm2 ...`, no
    `--control`) and diff the two files: the Java corner coordinates and every metric column for
    that cell must be identical. Then run `./scripts/jm pytest tests/validation -v`.

## Schema

| column | meaning |
|---|---|
| `image`, `labels` | paths (relative to `reference.csv`) to the TIFF and the `.npy` label file; both must exist |
| `jm1_channel`, `jm2_channel`, `nucleus_channel` | band indices assigned in the Java; three distinct integers |
| `cell_id` | label id of the cell the Java flooded |
| `role` | `JM1` or `JM2` (which marker table the numbers came from) |
| `corner_method` | always `dp`; `tcj` is rejected |
| `epsilon` | the Java Epsilon slider, integer 1 to 50; one value per cell |
| `dilation_cycles` | the Java dilation slider, integer 1 to 9 |
| `threshold` | the Java threshold slider for this row's role, integer 0 to 255 |
| `interior` | `yes` or `no`; `no` rows are skipped with that stated reason |
| `java_corner_row/col`, `java_paired_row/col` | the Java interface's two corner pixels (row = Java `y`, col = Java `x`) |
| `java_n_corners` | the Java's corner count for the cell (`# Corners`), consistent across a cell's rows; information only |
| `interface_contour_px` ... `marker_intensity` | the Java's numbers; the last three must be whole numbers; none may be blank |
| `junction_contour_px`, `junction_straight_line_px` | JM1 rows only, may be blank |
| `java_corner`, `java_paired_with`, `notes` | optional, never compared |

The loader validates every row (types, ranges, paths, one row per image/cell/Java interface/role,
one epsilon and one corner count per cell) and **fails** the run on any defect.
`java_distance_ratio` and `coverage_index` may be `Infinity` or `NaN` (a closed loop, a zero
contour); those are legitimate Java output and are recorded, not rejected.

## Frozen real Java output

`test_frozen_java_output_on_the_control_grid_matches_our_metrics` embeds the generator's output
for the centre cell of the control grid (four Java interfaces, both roles), so every test run,
with no JDK, compares the port with real Java numbers: the straight side exactly, the three
corner-rounded sides at their stated tolerances. The test that runs the generator itself
(`test_generator_runs_the_real_java_on_the_control_grid_and_the_port_agrees`) skips, naming
what is missing, when there is no JDK or no Java checkout visible.

## If numbers differ

* **Look at the message first.** It prints the Java corners, the pixels of our boundary they
  snapped to, and the snap distance. "path disagreement" means a Java corner is not on our
  boundary (check the label file, the edge map and that you loaded the same image); a metric
  disagreement on a path with 0.00 snap is a measurement difference.
* **Corner counts differ**: expected, the Java hulls first. Not a defect.
* **A non-straight row differs in contour or junction length**: recorded, not asserted; the
  Java's walk visits different pixels. Only the straight rows prove exactness.
* **Straight rows differ**: this is the signal. Check that epsilon, dilation and thresholds in
  the CSV match the sliders, then report it.

## Why the edge-map line is two pixels wide

The Java's flood (`ImageUtilities.selectCell`) *visits* all eight neighbours of a pixel but only
*expands* from pixels with no set pixel among their eight neighbours, so a line cannot be crossed
unless it has a hole; a one-sided one-pixel line works too (the Task 16 review ran the real
flood on one over a staircase, a jogged grid and two Voronoi tilings: no leak). The two-sided,
4-neighbour line is kept for **symmetry and margin**: neither of two touching cells is favoured
by label order, there are no holes where labels meet at a corner, and nothing depends on the
exact expansion rule of one Java version. It is *not* kept to prevent leaks. Its price, each
cell eroded by a pixel, is compensated by measuring our side on the Java's view of the labels
(`java_view_labels`).
