# Java spot-check validation

This directory checks the Python port against the original Junction Mapper (Java) on a handful
of cells. Nobody can run the Java without a display, so the check is a procedure a person
follows once, plus tests that skip until they have.

**Idea.** The Java's step 1 has a "Load Existing" button that reads a binary edge-map PNG and
jumps to step 3. We write our own segmentation as such a PNG, load it in the Java, click
through a few cells, and copy the Java's numbers into `tests/validation/reference.csv`. The
tests then recompute the same cells with the Python code and compare.

What the tests do without `reference.csv`: the converter and the harness's own self-tests run;
the four comparison tests **skip**, naming the file and this README. If `reference.csv` exists
but is malformed, they **fail** (a typo in a header must not silently disable the validation).

## Read this before you start

* **"Measure" persists nothing.** Only **"Save as Spreadsheet"** writes a file, and it writes
  **only the cell currently selected**. For each cell you want to record you must select it,
  set its sliders, press Measure (to look at it) and then press **Save as Spreadsheet**. If you
  skip the save, those numbers are gone.
* **Douglas-Peucker only.** The Java has one corner method. Our tricellular-junction method has
  no Java counterpart, so `reference.csv` cannot contain `tcj` rows (the loader rejects them).
* **Interior cells only.** The Java's dilation band is clipped near the image border, ours is
  not (repository `README.md`, recorded divergence 2). Choose cells whose whole interface is
  at least `dilation_cycles + 2` pixels from every image edge, and which do not touch the
  border. Rows that are not interior are skipped with a reason, never failed.
* **The Java is integer-valued.** Its Epsilon slider is an integer 0 to 50 and **starts at 0**
  (every corner is kept), its dilation slider is 1 to 9, and its two threshold sliders are 0 to
  255. Record the values you actually set. A different epsilon is a different set of corners
  and therefore different interfaces, so the epsilon travels with every row.

## Procedure

You need a machine with a display (or X forwarding / VNC), a JDK 17, and this repository in the
Apptainer image (`./scripts/jm`). `$JAVA` below is the original checkout,
`/ix1/alee/LO_LAB/Personal/Brent_Schlegel/tools/Junction_Mapper`; nothing here modifies it.

1. **Choose the image and a label file.** Use one of `$JAVA/examples/image_1.tif` to
   `image_4.tif` (800 x 600 RGB; band 0 = Junction 1, band 1 = Junction 2, band 2 = nuclei).
   Get cell labels for it, for example with `./scripts/jm junctionmapper segment` (the file you
   want is `OUT/labels/<field_id>.JM1.npy`), or hand-corrected labels from the napari plugin.
   Make a working directory `tests/validation/reference/` and put copies of the TIFF and the
   `.npy` in it. Keep the label file: the tests re-read exactly it.

2. **Make the edge map and an overview.**

   ```bash
   ./scripts/jm python scripts/edge_map_from_labels.py \
       tests/validation/reference/image_1.JM1.npy \
       tests/validation/reference/image_1_edge_map.png \
       --overview tests/validation/reference/image_1_overview.png
   ```

   The PNG is 1-bit, white where a pixel has a 4-neighbour with a different label, so the line
   between two cells is two pixels wide. **Each cell is therefore eroded by one pixel where it
   meets another cell or the background** (the Java floods the clear pixels). The tests measure
   our side on exactly that eroded image, so both implementations trace the same boundary.
   Open the overview: it prints each cell's label id, which is what `cell_id` and
   `neighbour_id` mean in `reference.csv`.

3. **Choose cells.** Pick 4 to 8 cells from the overview that are well inside the field (see
   "Interior cells only") and, for each, 1 to 3 interfaces that run along **one** neighbour
   from a corner at one triple junction to the corner at the next. Skip interfaces that bend
   round a triple junction: the Java and the port may place the corner on either side.
   Prefer a mix of straight, jogged and curved interfaces; check at least one cell with a
   mostly horizontal or vertical outline first (see "If numbers differ").

4. **Choose and write down the settings** you will use for every cell:
   `epsilon = 2` (our default `epsilon_px` is 2.0), `dilation_cycles = 2`, and a Junction 1 and
   Junction 2 threshold (integers; look at a few pixel values in the TIFF and take a value that
   splits stained from unstained, for example 40 and 20). You will type these into the Java
   sliders and into the CSV.

5. **Start the Java** (needs a display). Either

   ```bash
   cd $JAVA && ./build.sh run                      # compile from source and launch, or
   cd $JAVA && java -jar dist/junction_mapper.jar  # the released binary
   ```

   Start it from a directory you can write to: the Java creates a timestamped output directory
   `./yyyy_MM_dd_HHmm` under the current directory on the first image load, and every
   spreadsheet lands there.

6. **Load the image and assign the bands.** File menu, open the same TIFF as in step 1. In the
   parameters dialog set band 0 to "Junction 1 Channel", band 1 to "Junction 2 Channel" and
   band 2 to "Nucleus". These indices are the `jm1_channel`, `jm2_channel` and
   `nucleus_channel` columns (0, 1, 2 for the sample images).

7. **Inject the segmentation (step 1).** On the Membrane tab press **Load Existing** and choose
   the `_edge_map.png` from step 2. If it reports "Edge map image is not binary type", the PNG
   is not 1-bit; regenerate it with the script (never save it from an image editor). You are
   now at step 3 with our boundaries shown; do not edit them. Press **Finish**.

8. **Select the cells (step 4).** Click inside each cell you chose, and inside its neighbours
   if you like; each fills with a colour. Press **Finish**.

9. **Measure one cell (steps 5 to 7).** Repeat 9a to 9e for **each** chosen cell.
   a. Step 5: click the cell.
   b. Step 6: set **Epsilon** to your epsilon, **# Dilations applied to edge** to your
      dilation cycles, **Junction 1 Threshold** and **Junction 2 Threshold** to your
      thresholds. Release each slider (the Java acts on release). The Epsilon slider returns
      to 0 every time you re-enter step 6, so set it again for every cell. Do **not** add or
      remove corners with the mouse: a manually edited corner set is not Douglas-Peucker.
   c. Check that the corner pair you want exists on the overlay.
   d. Step 7: press **Measure**, then press **Save as Spreadsheet**. (Measure alone saves
      nothing.) The console area prints the path; there is one `.xls` per cell.
   e. "Back to Cell State" to select the next cell.

10. **Transcribe from each `.xls`**, sheet `Cell # n`. Excel row numbers below are 1-based;
    open the cell and read the formula bar, which shows the full-precision value (the grid
    rounds, and rounding breaks the 1e-9 comparison).
    * Summary block, rows 1 to 8: the thresholds in `B5` (Junction Marker 1) and `B7`
      (Junction Marker 2) must equal what you set. This block is context, not transcribed.
    * **JM1 table**: title in row 11, header in row 12, one row per corner from row 13. Pick the
      row for your interface (its `Corner #` and `Paired With #` identify it).
    * **JM2 table**: starts five rows after the last JM1 row (title, then header, then one row
      per corner). Its columns are the same up to `N`; it has no Junction Contour or
      Junction Straight-line columns.

    | reference.csv column | JM1 table | JM2 table |
    |---|---|---|
    | `interface_contour_px` | C "Interface Contour" | C |
    | `straight_line_length_px` | D "Interface Straight-line Length" | D |
    | `java_distance_ratio` | E "Interface Linearity Index" **as printed** | E |
    | `fragmented_contour_px` | F "JM1 Fragmented Junction Contour" | F "JM2 Fragmented Junction contour" |
    | `coverage_index` | G "JM1 Coverage Index" | G |
    | `dilation_cycles` | H "Dilation Cycles" (check it equals what you set) | H |
    | `interface_area_px` | I "Interface Area" | I |
    | `marker_area_px` | J "JM1 Area" | J "JM2 Area" |
    | `marker_intensity` | L "JM1 Intensity" | L "JM2 Intensity" |
    | `junction_contour_px` | O "JM1 Junction Contour" | leave blank |
    | `junction_straight_line_px` | P "JM1 Junction Straight-line Length" | leave blank |
    | `java_corner`, `java_paired_with` | A, B (audit only) | A, B |

    The Java's "Interface Linearity Index" is contour divided by straight-line length, so it is
    1 or more. This project's `linearity_index` is its reciprocal. Copy the Java's number
    unchanged into `java_distance_ratio`; the loader refuses a value below 1.

11. **Write `tests/validation/reference.csv`**, one row per (cell, interface, role): the same
    interface appears twice, once with `role = JM1` (from the JM1 table) and once with
    `role = JM2` (from the JM2 table). The geometry columns repeat. Use exactly this header:

    ```
    image,labels,jm1_channel,jm2_channel,nucleus_channel,cell_id,neighbour_id,role,corner_method,epsilon,dilation_cycles,threshold,interior,interface_contour_px,straight_line_length_px,java_distance_ratio,fragmented_contour_px,coverage_index,interface_area_px,marker_area_px,marker_intensity,junction_contour_px,junction_straight_line_px,java_corner,java_paired_with,notes
    ```

    | column | meaning |
    |---|---|
    | `image`, `labels` | paths to the TIFF and the `.npy` label file, relative to `reference.csv` (both must exist) |
    | `jm1_channel`, `jm2_channel`, `nucleus_channel` | band indices assigned in step 6; three distinct integers |
    | `cell_id`, `neighbour_id` | label ids from the overview; the interface of `cell_id` against `neighbour_id` |
    | `role` | `JM1` or `JM2`; selects which marker table the row came from |
    | `corner_method` | always `dp`; `tcj` is rejected |
    | `epsilon` | the Java Epsilon slider, integer 1 to 50; one value per cell |
    | `dilation_cycles` | the Java dilation slider, integer 1 to 9 |
    | `threshold` | the Java threshold slider **for this row's role**, integer 0 to 255 |
    | `interior` | `yes` or `no`: whole band at least 2 px inside the image edge and the cell off the border. `no` rows are skipped with that reason |
    | `interface_contour_px` ... `marker_intensity` | numbers from the table above; the last three must be whole numbers; none may be blank |
    | `junction_contour_px`, `junction_straight_line_px` | JM1 rows only, may be blank |
    | `java_corner`, `java_paired_with`, `notes` | optional, never compared |

    The loader validates every row (types, ranges, paths, one row per image/cell/neighbour/role,
    one epsilon per cell) and **fails** the run on any defect.

    Example (values are illustrative):

    ```
    reference/image_1.tif,reference/image_1.JM1.npy,0,1,2,17,23,JM1,dp,2,2,40,yes,31.65685425,29.0,1.0915,12.0,0.379,215,160,18452,25.3,24.1,3,7,
    ```

12. **Run the comparison.**

    ```bash
    ./scripts/jm pytest tests/validation -v
    ```

    It writes `tests/validation/divergence.md`. Commit `reference.csv`, the TIFF and label files
    it points at, and `divergence.md` together; the file is deterministic, so a rerun on the
    same reference does not change it.

## What is asserted

| metric | rule | why |
|---|---|---|
| `interface_area_px`, `marker_area_px`, `marker_intensity` | **exactly equal** | pixel counts and integer sums; any difference is a real bug for an interior cell |
| `interface_contour_px`, `straight_line_length_px` | relative 1e-9 | also confirms the same interface was matched |
| `linearity_index` | relative 1e-9, as `1 / java_distance_ratio` against ours | recorded divergence 1: reciprocal convention |
| `junction_contour_px`, `junction_straight_line_px` (JM1) | relative 1e-9 | Java's `eCadherinDistanceToNeighbour` spans gaps, as ours does |
| `fragmented_contour_px`, `coverage_index` | **recorded, never asserted** | recorded divergence 3: the Java's JM1 walk uses an OR connectivity rule that bridges gaps, and its JM2 walk over-counts |

`divergence.md` lists every recorded comparison with Java value, our value, absolute and
relative difference, and the divergence that explains it. A coverage difference that the
fragmented contour does not explain is printed as `UNEXPLAINED`. We chose **not** to git-ignore
`divergence.md`: it is the artefact a reader of the paper wants, it is only written when a
reference exists, and it is byte-for-byte reproducible, so committing it causes no churn.

## If numbers differ

* **Interface not found** (the run fails with "we find none"): the Java and the port chose
  different corners at that epsilon. Look at the Java's step 6 overlay and the cell's
  boundary; pick an interface that does not bend round a triple junction, or a different
  epsilon recorded for the whole cell.
* **Integer metrics differ by a few pixels on diagonal or concave outlines.** The Java marks
  a cell's edge pixels as the cell pixels with *any of eight* neighbours set in the edge map;
  our boundary uses four. They coincide on straight and convex outlines and differ at inner
  corners of staircases. Confirm by checking a cell with a rectilinear outline first: if
  that one matches exactly, the port is right and the difference belongs to the outline, so
  record it and prefer outlines that agree.
* **Everything differs for one cell:** check epsilon, dilation and thresholds in the sheet
  against the CSV (`B5`, `B7`, column `H`); a slider left at its start value is the usual cause.
