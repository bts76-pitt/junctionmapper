# Junction Mapper, modernized — design

Date: 2026-11-05
Status: approved for planning

## Intent

Junction Mapper (Imperial College London, doi:10.7554/eLife.45413) quantifies cell–cell
junction morphology from fluorescence images of cell monolayers. Its measurement definitions
are sound and well documented. Its implementation is a 2019 Java Swing desktop application
driven entirely by mouse clicks: one image at a time, one cell at a time, no batch mode, no
scripting surface, 8-bit RGB TIFF only, no physical units, and an interactive segmentation
pipeline that takes an operator several minutes per field.

This project reimplements the published quantification in Python with modern image I/O and
automated segmentation, so the same measurements can be made across many fields without a
human driving every step.

### Why now

The user has a predicted junction-disruption score derived from the localization of adherens
junction genes under oncogenic mutations, signaling, and compensatory mechanisms. They need a
*measured* junction phenotype to place alongside that prediction.

The biological setting is invasive lobular carcinoma. E-cadherin is lost in some samples but
present in others, so **no protein can be hardcoded to any slot**. CDH1 fills JM1 where it is
present; an optimized Na/K-ATPase membrane marker substitutes where it is not. The tool's job is
to make the marker occupying each role a per-sample configuration item, not an assumption.

The scientific question is narrow and should stay that way: **are the junctions forming properly
or not.**

The original tool cannot support this. At its interaction cost, the number of fields needed to
resolve a graded phenotype is not reachable.

## Goals

1. Reproduce Junction Mapper's published parameter set — the per-interface junction metrics and
   the nucleus tool's inter-nuclear distances — computing each as the paper's formulas describe.
2. Read OME-TIFF and OME-Zarr, taking channel identity and pixel size from metadata.
3. Segment with Cellpose-SAM instead of the interactive threshold/skeletonise/draw pipeline.
4. Detect corners automatically, by two independent methods, and report both.
5. Keep a human in the loop for reviewing and correcting segmentation. This is non-negotiable
   per the user.
6. Emit tidy tables as the primary artifact, plus a readable HTML report per run.
7. Run batches unattended once segmentation is approved.

## Non-goals

- **New metrics.** The user explicitly scoped this to "their method." Earlier proposals for
  membrane-versus-cytoplasm partitioning indices and free-edge fraction were rejected as out of
  scope. Do not reintroduce them. (A later project may; this one does not.)
- **Model fine-tuning.** v1 is stock Cellpose-SAM plus human correction. Corrections must be
  persisted in a form that could feed Cellpose's human-in-the-loop training later without
  rework, but no training code ships in v1.
- **Tuning to one acquisition.** The user was explicit: "this isn't a bespoke program." Handle
  what OME metadata can describe; do not assume a channel order, bit depth, or magnification.
- **3D junction topology.** See constraints.
- **Replacing or wrapping the Java application.** It has no scripting surface. It is reference
  material and a validation oracle, not a dependency.

## Anti-goals

Things that would make this a failure even if it ran:

- Numbers that silently diverge from the published method with no record of how or why.
- A pipeline that produces confident output from mis-segmented cells. A wrong segmentation
  yields plausible junction metrics with no obvious error; this is the main scientific risk and
  the reason human review is mandatory.
- Quantification that cannot be run or tested without launching a GUI.
- A napari plugin so central that batch processing becomes impossible.

## Constraints

- **2D only.** Junction Mapper's method is planar. Z, time, and scene dimensions are iterated as
  independent 2D fields, not analysed as volumes. *(My call, stated to the user and not
  rejected.)*
- **Human-in-the-loop is mandatory**, not an optional QC mode. *(User.)*
- The measurement core must be importable and testable without napari. *(My call.)*
- Corrections must be reproducible: a measurement run starts from saved, approved labels, and
  re-measuring must not require re-reviewing. *(My call.)*

## Background: verified facts about the original

These were established by reading the source and the official release in this repository. A
builder should not need to re-derive them.

### The measurement core is nearly free-standing

- No `javax.swing` import exists anywhere in `uk/ac/imperial/cisbio/cell_metric/om/` or in
  `ImageUtilities`. The model layer uses `java.awt.Point`, `Color`, and `BufferedImage` only.
- The entire model→GUI coupling is 12 call sites, all parameter reads: four images
  (JM1 greyscale, JM1 threshold, JM2 threshold, JM2 greyscale), two ints (`dilatedEdgeWidth`,
  `numIntraCells`), and six logging calls to `setTextArea`.
- Of ~40 static methods in `ImageUtilities`, three take a GUI type, all for live preview.
  `getSkeletonImage` already null-guards its panel argument.

Consequence: `Cell_Membrane` and `Cell_Membrane_Corner` are readable as a specification of the
measurement math. Read them alongside the user guide.

### `getThresholdImage()` does not return a binary image

`ImageUtilities.getThresholdImage(image, low, high)` copies the greyscale and zeros pixels
outside the range; surviving pixels keep their original values. The result is used **both** as a
binary mask (`pixel > 0` → area) **and** as the intensity source (`+= pixel`). The implementation
does match the guide's `Σ if(g>T) g`. The name is the trap, not the math. A port that treats it
as binary will silently turn the intensity metrics into area metrics.

### JM1 and JM2 are not symmetric

`Membrane_SpreadSheet` writes 17 columns for JM1 and 14 for JM2. JM2 has no
`measurementEuclidianDistance`, so Junction Contour, Straight-line Junction Length, and Junction
Linearity Index do not exist for it.

`Cell_Membrane_Corner.getECadherinPoints()` and `getMeasurementPoints()` differ in two ways
despite the comment claiming "same principle":

| | JM1 | JM2 |
|---|---|---|
| `last` updated | only on above-threshold pixels | on every contour pixel |
| connectivity test | `x==last.x±1 \|\| y==last.y±1` (OR) | `x` and `y` both within ±1 (AND) |

JM1's OR test passes whenever a single coordinate differs by 1, so a jump such as
(100,50)→(101,80) is counted as connected and contributes 30 px to the fragmented junction
contour. This propagates into the JM1 Coverage Index.

**Decision (user):** compute this the way the paper's formula describes — proper 8-connectivity —
rather than reproducing the Java behaviour. JM1 and JM2 use one consistent rule.

### What the original cannot do

Single 8-bit RGB TIFF via `ImageIO`; no calibration field anywhere, so all outputs are in pixels
and incomparable across magnifications; results saved one cell at a time; `Measure` computes but
does not persist. Note that `loci_tools.jar` (Bio-Formats) sits on the release classpath and is
imported by nothing — the author intended better I/O and never built it.

## Decisions

### Fidelity

Implement the parameter definitions from `docs/user_guide.pdf`, which gives the formula, units,
and a diagram for each of the 9 primary per-edge parameters. Where the Java diverges from its own
documented formula, follow the document. Record every such divergence explicitly.

### Channel roles — configurable, never assumed

Three roles exist, matching the original: **JM1**, **JM2**, and **nucleus**. A per-sample
configuration maps channel (by OME name, or by index where names are absent) to role, and records
**which protein occupies each role**.

This is a hard requirement, not a convenience. The same experiment contains CDH1-positive and
CDH1-null samples, with Na/K-ATPase standing in for JM1 in the latter. Nothing in the code may
assume a marker identity.

Two consequences the builder must handle:

- The recorded marker identity travels with every row of output. Pooling a CDH1-JM1 sample with a
  NaK-JM1 sample is not an apples-to-apples comparison, and the schema must make that visible
  rather than let it happen silently.
- JM1 is also the segmentation source, as in the original. Where a continuous membrane marker
  fills JM1, its coverage metrics will sit near saturation — expected, not a bug.

### Segmentation

Cellpose-SAM on the JM1 channel produces labeled cells; the nucleus channel produces labeled
nuclei. This replaces the entire interactive edge-map pipeline — no adaptive threshold, no
skeletonisation, no manual line drawing, no region-growing by click.

### Intensity thresholds

Four of the nine primary parameters (marker area, marker intensity, coverage, fragmented contour)
depend on an intensity threshold T per marker channel. Cellpose supplies labels, not T, so T is a
separate first-class input. In the Java it was an operator-dragged slider set per cell.

**Decision (user): T is reviewable per field in napari**, alongside the labels.

The builder should implement this as: a value proposed automatically, a human able to inspect and
override it per field, and every value recorded in provenance. *(My call, to make that workable at
the stated scale:)* the proposal defaults to a single value derived from a calibration subset and
is carried across the run, so the common path is "accept", with per-field override where review
shows it is needed. Overriding T per field costs cross-field comparability of the absolute
intensity metrics; the report must show where T varied within a run so that cost is visible.

### Interface dilation band

The measured interface is the cell edge dilated by N cycles — `dilatedEdgeWidth`, a slider in the
Java, default 2. *(My call:)* accept it in pixels or µm, defaulting to 2 px to match the Java, with
µm recommended for any comparison spanning magnifications. Record the value used, in both units.

### Corners — both methods, always

The user asked for both, reasoning that cellular geometry varies and the comparison will catch
edge cases.

1. **Douglas–Peucker**, as the original: simplify the cell outline, treat remaining vertices as
   corners, and the contour between consecutive corners as one interface. Preserves comparability
   with published results. Takes an epsilon parameter.
2. **Tricellular junctions from label adjacency**: every boundary pixel lies between labels A and
   B, so an interface is the run of pixels sharing a neighbour and a corner is where neighbour
   identity changes. Deterministic, parameter-free, and closer to what a corner biologically is.

Both run on every cell. Results carry a method key. Disagreement between them is a per-cell QC
flag surfaced in the report.

### Nuclei

In scope: the nucleus tool's inter-nuclear distances, and **nucleus↔cell association**
(`Nucleus_Collection.associateMembranes()` in the Java). Association is required, not optional —
it is what lets junctional and nuclear statistics join, which is the user's stated downstream
need.

*(My calls:)* `numIntraCells`, the number of nearest neighbours measured per nucleus, defaults to
5 as in the Java and is configurable, with an all-pairs option. Association needs its own
correctness rules and QC flags for the cases the Java ignores: multinucleate cells, nuclei with
no containing cell, and cells with no nucleus. Those flags feed the report alongside the
corner-disagreement flag.

### Human in the loop

Correction operates on **labels**, not pixels — merge, split, delete, repaint a cell — which is
far faster than the original's pixel-level line editing. This happens in a napari plugin.

Corrected labels are a first-class saved artifact with provenance. Measurement runs from saved
labels, so re-measuring never requires re-reviewing, and the corrections remain available as
future training data.

**Scale: hundreds of fields per run (user).** This rules out a linear walk through every field.
The review surface needs:

- a review queue ordered by QC flag severity, so the worst fields surface first;
- bulk approve across a selection;
- resumable review state, since a run will not be reviewed in one sitting;
- an explicit per-field approval record distinguishing human-approved from machine-only.

Approval is therefore sampled rather than exhaustive, and the saved-labels artifact must encode
approval state per field. Whether approval is also tracked per cell is left to the builder, but
the report already distinguishes corrected from machine-only cells, so the data must support it.

Note the tension the user accepted: per-field threshold review plus hundreds of fields is a real
cost. The mitigation above — propose once, carry forward, review by exception — is what makes the
two answers compatible. If it proves too slow in practice, the fallback is a single global T per
run, which is a configuration change rather than a rebuild.

### Architecture

Three pieces, in dependency order:

1. **Core library** — headless, importable. I/O, segmentation driver, corner detection,
   measurement, table emission, report generation. Never imports napari.
2. **CLI** — batch over many fields, unattended, for use after labels are approved.
3. **napari plugin** — the interactive surface: load, run segmentation, review, correct, approve,
   and launch measurement. Depends on the core.

The user's words: "building this all as a napari plugin could be really powerful." Agreed as the
interactive surface; the core/plugin split is my call, so that quantification is testable without
driving a viewer and batches do not need a GUI.

### I/O and units

OME-TIFF and OME-Zarr. Channels selected by name from OME metadata where names exist, by index
where they do not, with the channel→role mapping always explicit in configuration.

Pixel size read from metadata; outputs carry µm and µm² alongside pixel values. Fail loudly when
calibration is **neither** present in metadata **nor** supplied explicitly — a configuration
override is always permitted, and is how the uncalibrated example TIFFs are processed.

*(My call:)* vendor formats (ND2, CZI, LIF) are **out of scope**. The user asked for OME-TIFF and
OME-Zarr specifically. Conversion is an upstream step via `bioio` or `bfconvert`, documented but
not shipped. Plain non-OME TIFF is accepted as input provided role mapping and pixel size are
supplied by configuration; this is what the validation images require.

### Outputs

Tidy tables (parquet plus CSV) are the primary artifact, keyed so interface, cell, nucleus, and
field join cleanly. The user will combine junctional and nuclear statistics downstream; that
downstream analysis is explicitly not part of this project, but the keys must make it possible.

One self-contained HTML report per run:

- **QC** — cells found per field, cells flagged by corner-method disagreement, cells excluded and
  why, and which cells a human corrected versus machine-only.
- **Quantification** — distributions per parameter, per-field and pooled, showing per-cell spread
  rather than condition means. Graded loss may present as a subpopulation rather than a shift, so
  the report must not hide the distribution behind an average.
- **Provenance** — input paths, Cellpose model and parameters, software versions, corrections
  applied.

The report is for reading. Tables are for analysis.

### Validation

**Primary oracle: synthetic fixtures** with analytically known geometry — shapes whose contour
length, straight-line length, dilated-band area, and thresholded overlap can be computed by hand.
These carry the main testing burden, test the measurement math independently of segmentation, and
run in CI.

**Secondary: a Java spot-check on a handful of cells (user's decision).** Enough to catch
systematic errors, not a full reference dataset. The constraints that force this are real and the
builder should not try to automate around them: the app has no scripting surface, it is one
`switch(state)` over `mouseClicked`, `Measure` does not persist, and "Save as Spreadsheet" writes
one cell at a time. Reference numbers cost manual clicks on a machine with a display.

Three things the spot-check has to respect:

- **Use the Douglas–Peucker corner path only.** Interfaces are defined by their corners, so
  Python and Java results correspond only when corner sets do. The tricellular-junction method
  has no Java counterpart and cannot be compared this way; it is validated against synthetic
  fixtures instead.
- **Segmentation must be injected, not redrawn.** The Java's step-1 "Load Existing" accepts any
  `TYPE_BYTE_BINARY` PNG edge map. Feed it one derived from the same Cellpose labels Python used.
  Note this still leaves the operator flood-filling cells and setting the DP epsilon slider by
  hand, so "identical segmentation" is close but not exact — record the epsilon used.
- **Tolerance:** *(my call)* exact equality for integer pixel counts (areas, intensity sums,
  pixel-length counts); a small relative tolerance only for derived ratios and Euclidean
  distances. "Floating-point tolerance" was wrong for the integer quantities and is corrected
  here — a one-pixel boundary offset changes those answers outright, so a mismatch means either
  the segmentation differed or the port is wrong, and both need investigating rather than
  absorbing into a tolerance band.

Expect and quantify divergence on the fragmented-contour and Coverage Index metrics, per the
connectivity decision above. Document it; do not hide it. Disagreement in any other parameter is
a bug in the port.

## Left to the builder

- Python packaging, project layout, dependency and environment management.
- Repository location. The port likely wants its own repo rather than living in this clone of the
  upstream Java project; set that up at implementation time.
- Choice of array and table libraries, and the plotting stack behind the report.
- How Cellpose-SAM is invoked and parameterised, and how GPU-versus-CPU execution is handled.
- Design of the synthetic fixture set: which shapes, how their expected values are derived, and
  how they cover the nine parameters plus the tricellular corner path.
- Douglas–Peucker epsilon: default value and whether it is exposed per-run or per-cell.
- How "excluded" cells are defined (border-touching, too few neighbours, size outliers).
- Report layout and styling. The user declined to review mockups; use judgment.
- Whether interfaces are stored once per cell pair or once per cell (each interface is shared by
  two cells, so the schema must take a position on double-counting).

## Reference material in this repository

| Path | What it is |
|---|---|
| `docs/user_guide.pdf` | **The specification.** 20pp, formulas, units, and diagrams for all 9 primary parameters; algorithm notes for every control. Authoritative where the Java disagrees. |
| `uk/.../om/Cell_Membrane_Corner.java` | The per-interface measurement math. |
| `uk/.../om/Cell_Membrane.java` | Boundary ordering, chain code, Douglas–Peucker. |
| `uk/.../output/Membrane_SpreadSheet.java` | The exact output column set and what feeds each. |
| `examples/image_1..4.tif` | Validation inputs, 800×600 8-bit RGB. |
| `CLAUDE.md` | Architecture notes, build instructions, provenance. |

Cited algorithms: skeletonisation after Arcelli, Cordella & Levialdi (doi:10.1049/el:19750113);
corner detection after Douglas & Peucker (doi:10.3138/FM57-6770-U75U-7727).
