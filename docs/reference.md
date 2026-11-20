# Reference

Settings, inputs, outputs and every column. Start with the [README](../README.md).

**Contents**
[All settings](#all-settings) |
[Inputs](#inputs) |
[Segmentation options](#segmentation-options) |
[Measurements](#measurements) |
[Interfaces](#interfaces-table) |
[Intensity thresholds](#intensity-thresholds) |
[Nuclei distances](#nuclei_distances-table) |
[Membrane score](#membrane-score) |
[The report](#the-report)

---

## All settings

| Key | Default | What it does |
|---|---|---|
| `channels` | — | Which image channel holds each role, and which protein. |
| `pixel_size_um` | — | Pixel size, used only if the image has none. |
| `dilation_px` | 2 | Band half-width around each interface, in dilation steps. |
| `dilation_um` | — | Band width in µm instead. |
| `epsilon_px` | 2.0 | Corner sensitivity for the `dp` method. |
| `tcj_min_run` | 3 | Shortest contact kept by the `tcj` method, in pixels. |
| `neighbour_radius_px` | 2 | Window for deciding which neighbour each outline pixel touches. |
| `threshold` | — | Fixed intensity cutoff per role. Replaces the automatic one. |
| `num_intra_cells` | 5 | Nearest nuclei recorded per nucleus. |
| `all_pairs` | false | Record every nucleus pair, not just the nearest. |

Membrane score settings, under `membrane_score:`:

| Key | Default | What it does |
|---|---|---|
| `line_width_um` | [0.3, 0.6] | Membrane widths the line filter looks for, µm. |
| `patch_um` | 20 | Side of one scored square, µm. |
| `region_um` | 6 | How far the cell region reaches out from nuclei, µm. |
| `ridge_fraction` | 0.1 | Share of pixels treated as "on a line". |
| `cutoff` | none | Score at or above which a square is called membrane-like. None: no call. |
| `min_ridge_intensity` | none | Too-dim cutoff in raw intensity units. None: 5 × noise. |

---

## Inputs

- One image, or a folder of them, read in sorted order.
- `.tif`/`.tiff` files and OME-Zarr stores, mixed or not.
- An OME-Zarr store is a *directory* whose name ends `.zarr`. It is taken as one input in its own right, not searched inside.
- Vendor formats (`.czi`, `.nd2`, `.lif`, ...) are refused by name, up front.
- Each scene × time point × Z slice is one **field**.

---

## Segmentation options

**`snap`** moves cell–cell edges onto the membrane ridge of the `segment` channel, then smooths them.
- A boundary moves at most `--erode-px` (default 8).
- Smoothing (`--smooth-px`, default 20) stops it chasing specks of a dotted stain.
- Edges facing background, nuclei and cell ids do not change.

(Tested on 12 lines, 120 fields: boundary length changed by at most 3%. Membrane signal on boundaries rose in every line, median about +33%.)

**`--with-nucleus`** gives Cellpose the nucleus channel too.
- It cuts through fewer nuclei.
- Outlines can leave the membrane. On our test data, membrane-only outlines followed junctions better.
- Interfaces that cut a nucleus are flagged `splits_nucleus`. Some are real: in one optical section a neighbour's membrane can lie over a nucleus.

**The `segment` channel is never measured as a marker.** Its brightness in each band is still reported, as `segment_band_mean_intensity`.
- It reads high, because the outline was drawn on that signal.

**`run` will not overwrite label files** unless you pass `--overwrite-labels`. They may be hand-corrected.

---

## Measurements

Each interface is widened into a **band**. The band is measured for each marker.

| Column | What it is |
|---|---|
| `interface_contour` | Length along the interface. |
| `straight_line_length` | Distance between its two corners. |
| `linearity_index` | straight-line ÷ contour. 1 = perfectly straight. |
| `interface_area` | Area of the band. |
| `marker_area` | Band area above the threshold. |
| `marker_intensity` | Sum of above-threshold intensity in the band. |
| `interface_occupancy_pct` | 100 × marker area ÷ interface area. |
| `fragmented_contour` | Length of marker-positive stretches; gaps break it. |
| `junction_contour` | Length through marker-positive pixels; gaps bridged. |
| `junction_straight_line` | First to last marker-positive pixel. |
| `coverage_index` | fragmented contour ÷ interface contour. |
| `intensity_per_*` | marker intensity ÷ interface area. |
| `cluster_density_per_*` | marker intensity ÷ marker area. |

- Lengths come in `_px` and `_um`. Areas in `_px` and `_um2`.
- All tables carry `field_id`, so tables from many runs stack.

---

## `interfaces` table

**A row is one cell's side of one junction, for one marker.** It is not a junction.

A junction between two measured cells is seen from both sides. Both sides are kept: each is a real measurement.

How many rows a junction gives depends on the corner method:

| Method | Rows per junction | Why |
|---|---|---|
| `tcj` | 2 (1 if the neighbour is not measured) | Cuts where the neighbour changes. |
| `dp` | often 2–5 per side, sometimes 10+ | Corners ignore neighbours, so one contact is split wherever a corner falls. |

**So count junctions with `pair_id`, never by counting rows** or by assuming a factor of two.

| Column | Identifies |
|---|---|
| `pair_id` | The cell pair. Groups both sides. |
| `interface_id` | One cell side. Shared by its JM1 and JM2 rows. |

Pooling rows without grouping by `pair_id` over-weights some junctions. The report plots one dot per junction or per cell, never per row.

How far to trust a row:

| Column | Meaning |
|---|---|
| `contact_fraction` | Share of the interface that actually touches the neighbour. 1.0 for `tcj`; can be much lower for `dp`. |
| `low_contact_fraction` | `contact_fraction` < 0.8. The band partly measures free edge. |
| `closed_loop` | The interface is a whole outline. Its `linearity_index` is near 0 for geometric reasons. |

---

## Intensity thresholds

Four measurements use an intensity threshold, one per marker for the whole run.

How it is chosen, unless you set `threshold`:
1. Pick up to five fields, spread across the run.
2. Run Otsu on each.
3. Take the **median**. One saturated or empty field cannot drag it.

Three columns record it:

| Column | Meaning |
|---|---|
| `threshold` | The value used. |
| `threshold_field_otsu` | What this field alone would have given. |
| `threshold_deviates_from_run` | The two differ by more than 10% (and more than one grey level). |

The deviation flag is informational, not an error.
- On a low-valued image the one-grey-level floor dominates, so the flag sets on the outer fields of an ordinary *default* run. That is expected.

Edge cases:
- Unreadable files are skipped and others tried, up to ten. The report notes it.
- Blank images contribute nothing. If all are blank, the threshold is 0 and the report says so.
- The report warns *threshold varied* only when fields used thresholds more than one grey level apart.

---

## `nuclei_distances` table

One row per *ordered* pair: `nucleus_id`, `neighbour_nucleus_id`, `rank_from_nucleus`, `distance_px`, `distance_um`.

- `rank_from_nucleus` is 1 for that nucleus's nearest neighbour.
- With `all_pairs: true`, each pair appears twice, once from each end, each with its own rank.

```python
df[df.rank_from_nucleus == 1]                  # nearest-neighbour distances
df[df.nucleus_id < df.neighbour_nucleus_id]    # one copy of each pair
```

---

## Membrane score

Answers one question per channel: **does the stain sit in thin lines, or is it spread out?**

- It uses no cell outlines, so it does not change with `--labels`.
- Every channel except `nucleus` is scored.
- `run --no-membrane-score` turns it off.

### How it works

1. The **cell region** is the nuclei, grown outward by `region_um`.
2. The image is cut into squares of `patch_um`.
3. In each square, a line filter picks the brightest-looking line pixels (the top `ridge_fraction`).
4. **contrast** = brightness on those line pixels ÷ brightness on the rest (both above the image floor).
5. **baseline** = the same contrast after shuffling the square's pixels, which destroys any lines.
6. **score** = contrast ÷ baseline.

### Reading the score

| Score | Meaning |
|---|---|
| about 1 | No line structure. |
| well above 1 | Line-like. |
| blank | Not scored (too dim, or not enough cell region). |

- `membrane_like` is true when `score` ≥ `cutoff`. With no `cutoff` set, it is blank.
- Set `cutoff` from your own controls. It depends on the microscope, pixel size, cell density and signal.

### Known limits

- **Dots score like lines.** A dotted stain can score well above 1. A high score means line-like, not proof of membrane.
- **Compare scores only at one pixel size.** Membranes about 1 px wide can score near 1.
- **`region_um` must reach the cell borders.** If it doesn't, squares score near 1 or read as too dim. Check `region_fraction`.

### Too dim

A square is **too dim** when its line pixels are less than 5 × `noise` above the floor. It then gets no score; it is never called "diffuse".

- **For dim data, set `min_ridge_intensity`** (raw units). It replaces the 5 × noise rule. That rule cannot tell a near-blank channel from a faint real one.
- `background` = the channel's 1st percentile in the image.
- `noise` = how much neighbouring pixels differ, as a robust standard deviation.

(Detail: noise is the MAD of neighbour-pixel differences ÷ √2, over pixels above the floor. Quantised data get a ± half-step dither first. If that gives 0, it falls back to the spread of the region's pixels.)

### Columns

`membrane_patches`, one row per square per channel:

| Column | Meaning |
|---|---|
| `score`, `contrast`, `baseline` | As above. Blank when not scored. |
| `membrane_like` | `score` ≥ `cutoff`; blank if no score or no cutoff. |
| `too_dim` | True if too dim to score. |
| `ridge_intensity` | Mean on the line pixels. |
| `mean_intensity` | Mean over the whole cell region in the square. |
| `background`, `noise` | The image floor and noise. |
| `region_fraction` | Share of the square that is cell region. |

`membrane_fields`, one row per image per channel: `n_patches`, `n_scored`, `n_too_dim`, `median_score`, `membrane_like_fraction`.

- Squares neither scored nor too dim were skipped for geometry (edge slivers), too little cell region, or no signal.
- An image with no cells still gets rows.

---

## The report

`OUT/report/index.html`: one page per run, works offline.

- **Overview:** totals, one plot per measurement, a card per image.
- **Field view:** the image with junctions coloured by a measurement. Click a junction or cell for its values.
- **Warnings:** one chip per run-level problem.
- **Provenance:** inputs, model, settings, thresholds, versions, code hash.
- **Help (`?`):** each measurement's definition.

To open it:
- Double-click `index.html`. Nothing is downloaded.
- On the cluster, copy the whole `report/` folder to your computer, or use OnDemand's file browser.
- Keep the folder together: `index.html` loads `runs/` beside it.

`junctionmapper report OUT_A OUT_B -o compare` puts several runs on one page.
- It re-measures nothing.
- Labels default to the folder names (`--label` to set them).
- It warns if a run's labels changed since it was measured.
