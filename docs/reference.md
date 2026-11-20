# Reference

Settings, inputs, outputs and columns. For setup and running, see the [README](../README.md).

[All settings](#all-settings) |
[Inputs](#inputs) |
[Segmentation options](#segmentation-options) |
[Measurements](#measurements) |
[Interfaces](#interfaces-table) |
[Intensity thresholds](#intensity-thresholds) |
[Nuclei distances](#nuclei_distances-table) |
[Membrane score](#membrane-score) |
[Report](#report)

## All settings

| Key | Default | Meaning |
|---|---|---|
| `channels` | | Image channel, role and protein for each channel used. |
| `pixel_size_um` | | Pixel size, used if the image has none. |
| `dilation_px` | 2 | Band half-width around each interface, in dilation steps. |
| `dilation_um` | | Band width in µm. |
| `epsilon_px` | 2.0 | Corner tolerance for the `dp` method. |
| `tcj_min_run` | 3 | Shortest contact the `tcj` method keeps, in pixels. |
| `neighbour_radius_px` | 2 | Window used to decide which neighbour each outline pixel touches. |
| `threshold` | | Fixed intensity threshold per role. |
| `num_intra_cells` | 5 | Nearest nuclei recorded per nucleus. |
| `all_pairs` | false | Record every pair of nuclei instead of the nearest ones. |

Membrane score settings go under `membrane_score:`.

| Key | Default | Meaning |
|---|---|---|
| `line_width_um` | [0.3, 0.6] | Range of membrane widths the line filter looks for, µm. |
| `patch_um` | 20 | Side of each scored square, µm. |
| `region_um` | 6 | Distance the cell region extends from the nuclei, µm. |
| `ridge_fraction` | 0.1 | Fraction of pixels counted as line pixels. |
| `cutoff` | | Score at or above which a square counts as membrane-like. Unset by default. |
| `min_ridge_intensity` | | Too-dim threshold in raw intensity units. If unset, 5 × noise is used. |

## Inputs

`-i` takes one image or a folder. A folder is read in sorted order and can contain `.tif`/`.tiff`
files, OME-Zarr stores, or both. An OME-Zarr store is a *directory* whose name ends `.zarr`, and
it is taken as one input in its own right rather than searched for files.

Vendor formats (`.czi`, `.nd2`, `.lif` and so on) aren't supported and are rejected at the start
of the run.

Each scene, time point and Z slice is treated as a separate field.

## Segmentation options

`snap` moves cell–cell edges onto the membrane in the `segment` channel and then smooths them.
Edges move by at most `--erode-px` (default 8). The smoothing (`--smooth-px`, default 20) keeps
edges from following individual specks in a dotted stain. Edges that face background or nuclei
are left alone, and cell ids don't change. In a test on 12 cell lines and 120 fields, boundary
length changed by 3% at most and median membrane signal on the boundaries went up by about 33%.

`--with-nucleus` also passes the nucleus channel to Cellpose. Fewer nuclei get cut through, but
outlines can move off the membrane, and in our test data the membrane-only outlines followed the
junctions more closely. Interfaces that cut through a nucleus are flagged `splits_nucleus`. Some
of these are genuine, because a neighbour's membrane can overlie a nucleus in a single optical
section.

The `segment` channel isn't measured as a marker, but its mean intensity in each band is reported
as `segment_band_mean_intensity`. It reads high, because the outlines were drawn on that signal.

`run` won't overwrite existing label files unless you pass `--overwrite-labels`.

## Measurements

Each interface is widened into a band, and each marker is measured in that band.

| Column | Definition |
|---|---|
| `interface_contour` | Length along the interface. |
| `straight_line_length` | Distance between its two corners. |
| `linearity_index` | straight-line length ÷ contour. 1 is perfectly straight. |
| `interface_area` | Area of the band. |
| `marker_area` | Area of the band above the threshold. |
| `marker_intensity` | Sum of the above-threshold intensities in the band. |
| `interface_occupancy_pct` | 100 × marker area ÷ interface area. |
| `fragmented_contour` | Total length of marker-positive stretches. Gaps break it. |
| `junction_contour` | Length through the marker-positive pixels, with gaps bridged. |
| `junction_straight_line` | Distance from the first to the last marker-positive pixel. |
| `coverage_index` | fragmented contour ÷ interface contour. |
| `intensity_per_*` | marker intensity ÷ interface area. |
| `cluster_density_per_*` | marker intensity ÷ marker area. |

Lengths are given in `_px` and `_um`, areas in `_px` and `_um2`. Every table has a `field_id`
column, so tables from different runs can be concatenated.

## `interfaces` table

Each row is one cell's side of one junction, for one marker. A junction between two measured cells
is measured from both sides, and both rows are kept.

The number of rows per junction depends on the corner method:

| Method | Rows per junction | Reason |
|---|---|---|
| `tcj` | 2, or 1 if the neighbour isn't measured | It splits the outline where the neighbour changes. |
| `dp` | often 2–5 per side, sometimes 10+ | Its corners don't depend on neighbours, so one contact can be split at several corners. |

Count junctions with `pair_id`, never by counting rows or dividing by two.

| Column | Identifies |
|---|---|
| `pair_id` | The pair of cells. Both sides of a junction share it. |
| `interface_id` | One side. Its JM1 and JM2 rows share it. |

If you pool rows without grouping by `pair_id`, junctions split into many rows get extra weight.
The report plots one point per junction or per cell.

These columns help judge a row:

| Column | Meaning |
|---|---|
| `contact_fraction` | Fraction of the interface touching the neighbour. Always 1.0 for `tcj`; can be much lower for `dp`. |
| `low_contact_fraction` | `contact_fraction` below 0.8. Part of the band is measuring free edge. |
| `closed_loop` | The interface is the whole outline. `linearity_index` is close to 0 for these. |

## Intensity thresholds

Four of the measurements use one intensity threshold per marker, shared by the whole run. Unless
`threshold` is set in the config, it's chosen like this:

1. Up to five fields are picked, spread evenly across the run.
2. Otsu's method is run on each.
3. The median of those values is used, so one saturated or blank field has little effect.

| Column | Meaning |
|---|---|
| `threshold` | The threshold used. |
| `threshold_field_otsu` | The Otsu threshold for this field on its own. |
| `threshold_deviates_from_run` | The two differ by more than 10% and by more than one grey level. |

The deviation flag is informational. On a low-valued image the one-grey-level minimum dominates,
so the flag sets on the outer fields of an ordinary *default* run; this is expected.

If a file can't be read during calibration, another one is tried, up to ten in all, and the report
notes it. Blank images are ignored; if every image is blank the threshold is 0 and the report says
so. The report shows a "threshold varied" warning only when fields used thresholds more than one
grey level apart.

## `nuclei_distances` table

Columns: `nucleus_id`, `neighbour_nucleus_id`, `rank_from_nucleus`, `distance_px`, `distance_um`.
Each row is an ordered pair. `rank_from_nucleus` is 1 for the nearest neighbour of `nucleus_id`.
With `all_pairs: true` every pair appears twice, once from each end, with each end's own rank.

```python
df[df.rank_from_nucleus == 1]                  # nearest-neighbour distances
df[df.nucleus_id < df.neighbour_nucleus_id]    # each pair once
```

## Membrane score

The membrane score measures, for each channel, how much of the signal lies in thin lines. It
doesn't use the cell outlines, so `--labels` doesn't affect it. All channels except `nucleus` are
scored. `run --no-membrane-score` turns it off.

How it's computed:

1. The cell region is the nuclei grown outward by `region_um`.
2. The image is divided into squares with sides of `patch_um`.
3. In each square, a line filter selects the most line-like pixels (the top `ridge_fraction`).
4. contrast = mean intensity of those pixels ÷ mean intensity of the rest, both measured above the image's background level.
5. baseline = the same contrast after the square's pixels are shuffled, which removes any line structure.
6. score = contrast ÷ baseline.

A score near 1 means no line structure, and higher scores mean more line-like signal. Squares
that are too dim, or have too little cell region, get no score. `membrane_like` is true when the
score is at least `cutoff`, and empty when `cutoff` isn't set. A good cutoff depends on the
microscope, pixel size, cell density and signal strength, so set it from your own controls.

Limitations:

- Dotted staining can also score well above 1.
- Scores are only comparable between images with the same pixel size. Membranes about 1 px wide can score close to 1.
- `region_um` has to reach the cell borders. If it doesn't, squares score near 1 or come out too dim; `region_fraction` shows how much of each square is cell region.

A square is too dim when its line pixels are less than 5 × `noise` above `background`. It gets no
score. For dim data, set `min_ridge_intensity` (in raw units) instead, because the 5 × noise rule
can't distinguish a nearly blank channel from a faint real stain.

`background` is the channel's 1st percentile in the image. `noise` is a robust estimate of pixel
noise: the median absolute deviation of differences between neighbouring pixels, divided by √2,
using only pixels above `background`. Quantised images first get a small seeded dither of up to
half a quantisation step. If the estimate is 0, the spread of the cell-region pixels is used.

`membrane_patches` has one row per square per channel:

| Column | Meaning |
|---|---|
| `score`, `contrast`, `baseline` | As above. Empty when the square isn't scored. |
| `membrane_like` | `score` ≥ `cutoff`. Empty with no score or no cutoff. |
| `too_dim` | The square was too dim to score. |
| `ridge_intensity` | Mean intensity of the line pixels. |
| `mean_intensity` | Mean intensity of the cell region in the square. |
| `background`, `noise` | The image's background level and noise. |
| `region_fraction` | Fraction of the square that is cell region. |

`membrane_fields` has one row per image per channel, with `n_patches`, `n_scored`, `n_too_dim`,
`median_score` and `membrane_like_fraction`. Squares that are neither scored nor too dim were
skipped because they were thin slivers at the image edge, had less than half cell region, or had
no signal above background. Images with no cells still get rows.

## Report

`out/report/index.html` is a single page per run and works offline. It has:

- an overview with totals, a plot for each measurement and a card for each image
- an image view with junctions coloured by a chosen measurement; clicking a junction or cell shows its values
- run-level warnings
- provenance: inputs, model, settings, thresholds, software versions and code hash
- definitions of each measurement, under `?`

Open it by double-clicking `index.html`. On the cluster, copy the whole `report/` folder to your
computer or open it through OnDemand's file browser. The page loads the `runs/` folder next to
it, so keep them together.

`junctionmapper report OUT_A OUT_B -o compare` combines several runs in one report without
re-measuring anything. Each run is labelled with its folder name unless you pass `--label`. It
warns if a run's labels have changed since it was measured.
