# Interactive report: design

Date: 2026-11-16. Status: draft for the user's approval.

## Intent

The report shows the cells. You see each field with its outlines and junctions drawn on it.
You click a junction and get its numbers. Charts and image stay linked.

It replaces today's `report.html`: static histograms, prose and tables, with no images.

It works for any IF panel. Everything comes from the run config and the output tables.

(Why now: the AO0025 analysis needed a custom page of tables to read the results. Nobody
could see whether a number came from a real junction. The user called the result "a wall
of text". The original design already asked for a readable report. It never said
"dashboard", so the first build was static.)

## Goals

- An interactive dashboard in an industry-standard style. Think of a modern analytics
  product, not a paper supplement.
- The image is central: the field, with channel toggles and pan/zoom. Outlines and junctions
  are drawn on top. Junctions are coloured by a chosen metric.
- Click a junction or a cell: a side panel shows every metric for it.
- Charts and image are linked both ways. Select points in a chart: the matching junctions
  light up in the image. Select in the image: the chart highlights them.
- Show distributions per cell, never averages alone. Graded loss can be a subpopulation.
- One run = one condition. A comparison view puts several runs side by side.
- QC is visible at a glance: badges per field, flagged cells marked in the image.
- Provenance (inputs, model, parameters, versions) is one click away, not on the main view.

## Non-goals

- Correcting segmentation. That is the napari plugin (original design), a separate project.
- Full-resolution pixel inspection. Previews are downsampled (see Approaches).
- New metrics. The dashboard shows what junctionmapper already computes.
  The junction ÷ interior ratio from the AO0025 analysis stays in analysis scripts,
  because the original design ruled membrane-vs-cytoplasm indices out of scope.
- Statistics between conditions (tests, p-values). Downstream analysis, not this report.

## Anti-goals (a failure even if it works)

- Walls of text. Definitions live in hover tooltips and one collapsed help panel.
  No paragraph is longer than two lines on the main view.
- Anything specific to one experiment, stain, marker name or cell line.
- Needing a server, an internet connection or a Python process to open it.
- Hiding the per-cell spread behind a mean.

## Constraints

- Opens offline from `file://` in a current Chrome, Firefox, Safari or Edge.
  - Chrome blocks `fetch()` on `file://`. Lazy-loaded data must arrive by `<script>` tags,
    each file calling a registration function.
- References nothing external: no CDN, no fonts from the web. Libraries are vendored.
  (Same rule as today's report: it is archived next to data.)
- Scale: hundreds of fields per run (original design). Opening the report must stay fast.
  Each field's image and geometry load only when that field is opened.
- Fields are large: about 2500 × 3000 px, 3+ channels, 8 or 16 bit.
- Must work for 1 or 2 measured markers (JM1, optional JM2), an optional segment channel,
  and a nucleus channel. Channel count and names come from the config.
- Runs inside the existing Apptainer container. New Python dependencies must be added to
  `container/junctionmapper.def`.
- Report generation must not change any measured number or table.

## Approaches

**A. Static report folder (chosen by the user).**
`report/index.html` plus per-field asset files. Offline, no server, zip-and-send.
Downsampled previews with vector junction overlays.
Trade-off: no full-resolution pixels.

B. Local web app (Panel or Dash). Full resolution and dynamic comparison.
Rejected: needs a running Python server to view; cannot be emailed or archived as a file.

C. napari plugin. Full resolution on a desktop.
Rejected for the report: a viewer, not a dashboard; not shareable. It remains the plan
for correction.

## What gets built

### Output layout

```
<out>/report/
  index.html          the app: layout, styles, vendored JS, run summary data inline
  fields/<field_id>.js   one per field: preview images + junction/cell geometry + metrics
```

`index.html` alone must render the run overview, field list, QC and pooled distributions.
Field files load on demand when a field is opened.

### Data the report needs that does not exist yet

- **Junction geometry.** The tables hold `start_idx`/`end_idx`, not the pixel walk.
  The measurement pass already has `Interface.pixels`. *(My call:)* write a simplified
  polyline per interface during measurement, to a new per-field geometry file beside the
  labels. Recomputing walks inside the report would duplicate geometry code.
- **Cell outlines** as polygons, from the label image.
- **Previews.** One downsampled image per channel per field, longest side about 1500 px,
  contrast-stretched for display only. *(My call:)* WebP or JPEG, embedded as data URIs
  inside the field file.

### Views

1. **Run overview** (landing): cells, junctions and QC summary as compact tiles.
   A field grid with thumbnails and QC badges. Pooled per-cell distribution of the chosen
   metric, per marker.
2. **Field view**: the image viewer, the metric picker, a linked distribution chart for
   this field, and a detail panel for the selected junction or cell.
3. **Comparison**: built by `junctionmapper report RUN_DIR [RUN_DIR ...]`. One distribution
   per run for the chosen metric and marker, side by side. Clicking a point opens that cell
   in its field view.
4. **Provenance**: inputs, Cellpose model and parameters, thresholds, versions, code hash.

### Controls (all views)

- Marker (JM1 / JM2), metric (every column in `report.build.PARAMETERS`), corner method
  (`tcj` / `dp`), units (µm / px where calibrated).
- Channel toggles and per-channel brightness in the image viewer.
- Filter: QC-passing cells only (default on).

### Commands

- `junctionmapper run ...` writes the per-run report as today, in the new form.
- New: `junctionmapper report RUN_DIR [RUN_DIR ...] -o OUT` builds a comparison report from
  existing run outputs, without re-measuring. Run labels default to the run directory names.

### Calls made after the builder check

- **What a run directory holds.** `run` now also writes `run.json` (config + provenance)
  and `report/fields/*.js` (previews + geometry) into its output directory. *(My call.)*
  `report` reads only run directories made by this version. Older runs: re-run with
  `run --labels`, which is CPU-only and cheap.
- **Comparison folders are self-contained.** `report RUN_DIR ...` copies each run's field
  files into its own output. It can be zipped and sent. *(My call.)*
- **Geometry lives in the run's output directory**, never in the `--labels` directory
  (that is the hand-correction seam and must not be written by a measurement run). The
  geometry file records the hash of the label file it came from, so stale geometry is
  detectable. *(My call.)*
- **What the image draws.** One line per junction (`pair_id`), coloured by the mean of its
  cell sides. The detail panel lists each side's values. Default corner method `tcj`
  (at most two sides per junction). *(My call.)*
- **Default filter.** Same cells as the tables already measure (border-touching and
  degenerate cells have no interfaces). Each QC flag is a toggle and a marker in the image,
  off by default. *(My call.)*
- **One dot in a distribution = one junction (`pair_id`)** by default, the value being the
  mean of its cell sides, as in the image. A one-click switch shows one dot per cell instead
  (median over that cell's junctions); clicking a cell dot highlights all its junctions.
  Raw cell-side rows are never plotted (they double-count). *(User's decision.)*

## Left to the builder

- The JS stack: chart library, component styling, and the image viewer. Must be vendored
  and work from `file://`. The look should be clean and modern; a dark and a light theme.
- Polyline simplification tolerance (must stay visually faithful at preview scale).
- Preview size and encoding, within the "opens fast" constraint.
- How the old `report.html` tests migrate. The old template is retired.
- Exact QC badge rules, reusing the existing QC counts in `report/build.py`.
