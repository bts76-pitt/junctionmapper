# Differences from the original

The method is the original's. Below are the five places where this code knowingly does not copy
the original Java code, then what the validation shows. Each difference was checked against the
Java source and is commented in the code.

| # | What | In short |
|---|---|---|
| 1 | `linearity_index` | Inverted: 1 = straight. |
| 2 | Dilation band | Even everywhere; the Java's thins near the image border. |
| 3 | Marker connectivity | One rule for both markers. |
| 4 | Nucleus assignment | By most pixels, not by centre point. |
| 5 | Corners | Found on the outline, not on its convex hull. |

---

## 1. `linearity_index` is inverted

- Java: contour ÷ straight-line. ≥ 1, grows with wiggliness.
- Here: straight-line ÷ contour. ≤ 1, where 1 is straight.
- To compare with an old spreadsheet, take the reciprocal.

## 2. The band is even; the Java's thins at the image border

- The Java's dilation drops any new pixel near the image edge (`ImageUtilities.getDilatedImage`, lines 660–690). The loss grows each cycle.
- So in the Java, areas and intensity read low for cells near the border.
- Here the band grows evenly. Border-touching cells are flagged and excluded anyway.

## 3. One connectivity rule for both markers

The Java's two marker walks disagree with each other.

- **JM1** links two pixels when *either* coordinate differs by one. A jump of (1 row, 30 columns) counts as connected and bridges a real gap.
- **JM2** checks both coordinates (correct), but adds one extra step per fragment.

Here both use 8-connectivity, as the user guide's formula describes. JM1 and JM2 are directly comparable.

## 4. Nuclei go to the cell holding most of their pixels

- Java: the cell containing the nucleus's centre point. A centre on a junction or gap goes to no cell.
- Here: the cell with most of the nucleus's pixels (ties to the lowest label). Orphan only if it touches no cell.
- The reported centre point still uses the Java formula.

(Why: in loose tissue nuclei often straddle junctions. The centre-point rule inflates orphan and no-nucleus counts in exactly that condition.)

## 5. Corners are found on the outline, not the convex hull

The user guide says Douglas–Peucker on the cell outline. The Java code takes the **convex hull** first, then runs Douglas–Peucker on the hull.

What that does, from the code and from running the Java classes headless:

- **A concave corner can never be found.** The hull removes it. Cells in loose tissue are often concave.
- **The corner count is click-dependent.** The Java flood fill starts where you click, and that decides the hull. On the test grid, the recorded click gave four corners per rectangle, as here; another equally central click gave three. So `reference.csv` records the click as `seed_row` and `seed_col`. See "Why the click matters" in `tests/validation/README.md`.
- **No published epsilon.** The Java slider starts at 0. `dp` corner counts cannot be compared with any published run.

A one-off review measurement over 148 interfaces found that a regular hexagon gave eight to ten corners, none at a vertex. That driver is not committed, so this repository cannot regenerate the number. Treat it as an observation, not as evidence.

Here Douglas–Peucker runs on the traced outline, as the guide describes. A rectangle has four corners. Results differ from the original for any concave or slanted cell.

### The `tcj` method is new

`tcj` cuts the outline where the neighbouring cell changes. `dp` corners ignore neighbours, so each `dp` stretch goes to whichever neighbour touches most of it. That has two costs:

| Effect | Example | Flag |
|---|---|---|
| A `dp` interface can be mostly free edge. | A neighbour touching 8 px of a 20 px side still wins: `dp` reports 19 px, `tcj` 11 px. | `contact_fraction`, `low_contact_fraction`, `dp_low_contact` |
| A short real contact can vanish. | At 4 px of contact, background wins and `dp` reports nothing; `tcj` finds it. | `dp_missed_tcj_contact` |

`dp` keeps its rule: it is the published method. The cost is now measured and flagged.

This difference is about *which* interface is measured, not *how*. The validation relies on that.

---

## What the validation shows

`scripts/java_reference.py` runs the original Java classes headless and writes `tests/validation/reference.csv`. `./scripts/jm pytest tests/validation` then measures the same path, between the Java's own corner pixels.

The numbers come from `tests/validation/README.md`: 208 interfaces (416 rows, one per interface × marker) over two Voronoi tilings and a hexagonal tiling.

| Comparison | Rows | Result |
|---|---|---|
| Same pixels exactly (straight rows) | 20 of 416 rows (10 interfaces) | Areas and intensity **identical**. Lengths within 1e-9. |
| Paths differ at the edge-map seam | the other 396 | Areas within 15% or 16 px, intensity within 15%, linearity within 0.35. |
| `fragmented_contour_px`, `coverage_index` | all | Different on purpose (difference 3). |

- The differences on the 396 come from the edge map, not the formulas. The Java's edges are two pixels thick on slants, so the paths are not the same pixels.
- Worst seen on the 396: interface area −15 to +3 px (11.3%), marker area −12 to +6 px, intensity 14.0%, linearity 0.31.
- `fragmented_contour_px` and `coverage_index` are recorded and never asserted. Lengths on the 396 are recorded, not asserted.
- The test refuses a reference with no straight row, so the exact checks never silently skip.

**What this shows:** given the same path, the formulas reproduce the original's numbers. Exactly where the paths match; within the stated limits where they don't. It does not show that every metric matches on every interface; difference 3 alone rules that out.

Running it needs a JDK and the Java checkout. Without `reference.csv` those tests skip and say what is missing.
