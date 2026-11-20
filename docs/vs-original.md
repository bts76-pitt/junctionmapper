# Differences from the original

This package follows the published method, but in five places it deliberately differs from the
original Java code. Each one was checked against the Java source and is commented in the code.
The second half of this page summarises the validation against the Java.

| # | What | Difference |
|---|---|---|
| 1 | `linearity_index` | Inverted, so 1 is straight. |
| 2 | Dilation band | Even width everywhere. The Java band gets thinner near the image edge. |
| 3 | Marker connectivity | Same rule for both markers. |
| 4 | Nucleus assignment | By most pixels instead of by centre point. |
| 5 | Corners | Found on the outline instead of its convex hull. |

## 1. `linearity_index` is inverted

The Java reports contour ÷ straight-line length, which is at least 1 and grows as the junction
gets more tortuous. This package reports straight-line length ÷ contour, which is at most 1, with
1 for a straight junction. Take the reciprocal to compare with older spreadsheets.

## 2. The band has even width

The Java dilation (`ImageUtilities.getDilatedImage`, lines 660–690) skips any new pixel close to
the image edge, and the loss grows with each dilation cycle. Areas and intensities therefore come
out low for cells near the border. Here the band grows evenly. Cells touching the border are
flagged and excluded anyway.

## 3. Both markers use the same connectivity rule

In the Java, JM1 treats two pixels as connected when either coordinate differs by one, so a jump
of 1 row and 30 columns counts as connected and bridges a real gap. JM2 checks both coordinates,
which is correct, but adds one extra step per fragment. Here both markers use 8-connectivity, as
in the user guide's formula, so JM1 and JM2 values can be compared directly.

## 4. Nuclei are assigned by pixel count

The Java gives a nucleus to the cell that contains its centre point. A nucleus whose centre falls
on a junction or a gap goes to no cell. Here a nucleus goes to the cell that holds most of its
pixels, with ties going to the lowest label, and is an orphan only if it doesn't overlap any cell.
The reported centre point still uses the Java formula.

This matters most in loosely packed tissue, where nuclei often sit across junctions and the
centre-point rule produces extra orphan nuclei and cells without a nucleus.

## 5. Corners are found on the outline

The user guide describes Douglas–Peucker run on the cell outline. The Java code first takes the
convex hull of the cell and runs Douglas–Peucker on that. From reading the code and running the
Java classes headless:

- Concave corners are never found, because the hull removes them. Cells in loose tissue are often concave.
- The corner count is click-dependent. The Java flood fill starts at the clicked pixel, and the pixels it reaches determine the hull. On the test grid, the recorded click gave four corners per rectangle, the same as this package, while another click equally close to the centre gave three. `reference.csv` therefore stores the click as `seed_row` and `seed_col`; see "Why the click matters" in `tests/validation/README.md`.
- No epsilon was published. The Java slider starts at 0, so `dp` corner counts can't be compared with published results.

In a one-off check during code review, a regular hexagon gave eight to ten corners across 148 interfaces, none at a vertex. The script for that check is not committed, and the number can't be reproduced from this repository.

This package runs Douglas–Peucker on the traced outline, as the guide describes, so a rectangle
has four corners. Results differ from the Java for any concave or slanted cell.

### The `tcj` method

`tcj` is new in this package. It splits the outline wherever the neighbouring cell changes. The
`dp` corners don't take neighbours into account, so each `dp` segment is assigned to the
neighbour that touches most of it. This has two side effects:

| Effect | Example | Flag |
|---|---|---|
| A `dp` interface can be mostly free edge. | A neighbour touching 8 px of a 20 px side is still assigned the side; `dp` reports 19 px of contour and `tcj` 11 px. | `contact_fraction`, `low_contact_fraction`, `dp_low_contact` |
| A short contact can be missed. | With 4 px of contact, background wins and `dp` reports no interface; `tcj` finds it. | `dp_missed_tcj_contact` |

`dp` is the published method, so its rule is unchanged; the flags record where it goes wrong.
These differences change which stretch of outline is measured. The measurement formulas are the
same, which is what the validation relies on.

## Validation

`scripts/java_reference.py` runs the Java classes headless and writes
`tests/validation/reference.csv`. `./scripts/jm pytest tests/validation` then measures the same
paths with this package, using the Java's own corner pixels.

The figures below come from `tests/validation/README.md`: 208 interfaces (416 rows, one per
interface and marker) on two Voronoi tilings and a hexagonal tiling.

| Comparison | Rows | Result |
|---|---|---|
| Identical paths (straight rows) | 20 of 416 rows (10 interfaces) | Areas and intensities identical; lengths within 1e-9. |
| Paths that differ along the edge map | the other 396 | Areas within 15% or 16 px, intensity within 15%, linearity within 0.35. |
| `fragmented_contour_px`, `coverage_index` | all | Differ by design (see 3). |

The differences on the 396 rows come from the edge map. The Java edges are two pixels thick on
diagonals, so the two packages walk different pixels. The largest differences seen were −15 to
+3 px (11.3%) in interface area, −12 to +6 px in marker area, 14.0% in intensity and 0.31 in
linearity. `fragmented_contour_px` and `coverage_index` are recorded and never asserted, and the
lengths on the 396 rows are recorded but not asserted. The test fails if the reference has no
straight rows, so the exact comparisons can't be skipped silently.

On the same path, the formulas give the Java's numbers, exactly where the paths match
and within the limits above where they don't. Difference 3 means some metrics will never match.

Running the validation needs a JDK and the Java source. Without `reference.csv` the tests are
skipped with a message saying what's missing.
