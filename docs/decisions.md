# Decisions

This file records the judgment calls made while reimplementing the Junction Mapper method
as this library and CLI. The original is a Java Swing application driven entirely by mouse
interaction; a batch reimplementation has to settle questions the original answered by
having an operator in the chair. Each entry below is one such question, what was decided,
why, and **what it costs if the decision is wrong** — so that a reader who disagrees knows
exactly what to change and what it would affect.

Entries are grouped by consequence: decisions that move published numbers come first, then
output schema, provenance, validation against the Java, the QC report, and finally
conventions that affect no measurement. Divergences from the Java's observable behaviour
are listed separately in `README.md`; this file is about choices, not differences.

---

## 1. Decisions that move a published number

**Pixel size comes from image metadata, with config as fallback, and the µm→px conversion
happens at call time.** `resolved_dilation_px(pixel_size_um=None)` takes the argument as
authoritative rather than reading only `RunConfig`, and the error moved from config-time
validation to call time.
*Why:* metadata is the authority for calibration, so a dilation width given in µm would
otherwise be rejected for metadata-calibrated images, or converted at the wrong scale.
*Cost if wrong:* the dilation band width is wrong, which moves interface area, marker area,
marker intensity and occupancy — 4 of the 9 published parameters.

**Anisotropic XY calibration is refused, not silently halved.** The reader raises
`CalibrationError` when the X and Y pixel sizes differ by more than 1e-6 relative, instead
of using X and ignoring Y.
*Why:* anisotropic data would corrupt contour, area and intensity along one axis with
nothing in the output to show it.
*Cost if wrong:* genuinely anisotropic data a user wanted analysed is rejected and needs an
explicit override.

**Non-micrometer metadata units are converted.** The reader maps the declared spatial unit
rather than assuming µm.
*Why:* a nanometre-declared store would otherwise make `pixel_size_um` silently 1000×
wrong, corrupting the same 4 of 9 parameters with nothing in the output to show it.
*Cost if wrong:* a unit table to correct in one function. The lookup covers OME-TIFF and
OME-Zarr only; no other bioio reader is reachable in the container image.

**The boundary trace must cover every boundary pixel or the cell is rejected.** A coverage
check raises `BoundaryError` when the Moore trace misses any 4-connected boundary pixel.
*Why:* contours are counted along this walk, so a skipped pixel silently shortens Interface
Contour, Fragmented Junction Contour and Junction Contour. One erosion per cell is free next
to Cellpose.
*Cost if wrong:* a pathological cell is rejected instead of mismeasured — deliberately the
safer direction.

**The farthest-pair seed is made roll-invariant.** Among pairs at the maximum squared
distance, the lexicographically smallest `(row, col)` pair wins.
*Why:* integer squared distances tie often and `argmax` takes the first maximum, so rolling
the boundary changed the corner count on 15 of 400 test blobs — same cell, same pixels,
different interface count and therefore different published numbers. It was deterministic
per image only because the trace always starts topmost-leftmost, which a crop, flip or
rotation moves.
*Cost if wrong:* nil for the tie-break itself; without it, irreproducibility across crops.

**Short TCJ runs are merged (`min_run = 3`).** A deterministic `merge_short_runs` helper
plus `min_run: int = 3` on `tcj_corners`, with `min_run=1` recovering the raw behaviour.
*Why:* each spurious run adds a corner *and* a per-interface table row with a near-zero
contour length, polluting every per-interface statistic invisibly. 3 px because an interface
spanning fewer than 3 boundary pixels cannot support a contour measurement, and every
observed spurious run was 1–2 px.
*Cost if wrong:* a genuine but very short interface is merged away. Recoverable with
`min_run=1`, and the raw path is kept and tested for exactly that reason.

**A cell fully engulfed by one neighbour gets one whole-boundary interface, not zero.** The
0-corner cases are distinguished by neighbour identity: all-background stays empty, a single
non-zero identity returns one whole-boundary interface.
*Why:* an engulfed cell has no identity change along its boundary, so it previously got 0 TCJ
corners and `extract_interfaces` returned `[]` — silently dropping a real interface. Nothing
in the output showed the loss, and an engulfed cell is exactly what appears in the
discohesive tissue this tool is for.
*Cost if wrong:* nil; it restores a measurement that was being lost.

**`Interface.pixels` includes both corner pixels, so adjacent interfaces share a corner.**
The partition invariant changed from "every boundary pixel exactly once" to "a corner
position is covered by exactly two interfaces".
*Why:* the old exactly-once invariant was this plan's construction, not the published
method's. The reference dilates and measures each edge's own point list with the corner
included.
*Cost if wrong:* corner pixels are double-counted in area and intensity sums — but that is
precisely what the reference does, so matching it is the fidelity-preserving choice.

**Dilation is uniform (scipy), accepting a border divergence from the Java.**
*Why:* the spec says to follow the documented formula where the Java departs from it, and
the Java's border behaviour is an artefact of its implementation, not the published
algorithm. Cells that actually touch the border are already flagged `border_touching` and
excluded by callers.
*Cost if wrong:* band width differs by one pixel for cells within 2 px of the image border —
a population already treated as suspect.

**A nucleus is an orphan only if it overlaps no cell at all.** Otherwise it is assigned to
the cell holding the most of its pixels, with background excluded from the comparison and
the existing lowest-label tie-break retained.
*Why:* background was previously a plurality candidate, so a nucleus 6 px in cell A, 6 px in
cell B and 8 px over background became an orphan and both cells could be flagged anucleate,
despite 60% of it lying over cells. In discohesive tissue nuclei straddle junctions and gaps
far more often, so the old rule inflated `orphan_nuclei` and `anucleate_cells` precisely in
the condition under study — a systematic bias in the biological readout, not a tidiness
point.
*Cost if wrong:* a nucleus barely clipping a cell is assigned rather than orphaned. The tests
document that consequence openly, and no tunable threshold was introduced.

**Intra-nuclear distances are ranked per nucleus over both directions.** With
`all_pairs=True` the per-nucleus summary columns are computed over every distance involving
that nucleus in either direction, while `nuclei_distances` stores each unordered pair once.
*Why:* storing under the lower id made `n_intra_distances` and the mean depend on
segmentation label order — the highest-id nucleus got 0 and NaN — so a per-field plot of
`mean_intra_distance_um` would show a spurious NaN and a gradient with label id.
*Cost if wrong:* the summary counts each pair for both of its nuclei, which is the
conventional reading of a per-nucleus mean anyway.

**Threshold T is proposed once per run, not per field.** Otsu on up to 5 deterministically
spread calibration fields, median taken as the run value; `cfg.threshold` still wins per
role; each field records its own Otsu next to the applied value, plus a per-field deviation
flag; the override banner fires only on a genuine human override.
*Why:* a per-field threshold makes marker area and intensity incomparable across the fields
of one experiment, which is the comparison this tool exists to support.
*Cost if wrong:* a run whose fields genuinely need different T gets one value — which the
per-field Otsu column and the deviation flag make visible rather than silent.

**Otsu is computed only on the ≤5 calibration fields, not on every field.**
*Why:* the chosen field ids do not change, and computing it everywhere was redundant work
that also made a stated cost claim untrue.
*Cost if wrong:* the same fields are chosen; only the Otsu count differs.

**Calibration walks past unreadable files, bounded at 5 + 5 candidates.**
*Why:* a single corrupt file otherwise collapses the whole run-level proposal to per-field
Otsu.
*Cost if wrong:* a run with corrupt files calibrates from slightly different fields than even
spacing would pick, which the recorded calibration field ids make visible.

**Exact per-row distances, not cKDTree.** Nucleus neighbour ranking uses an exact per-row
numpy distance sort.
*Why:* cKDTree leaves ties at the k-th neighbour in arbitrary order, and deterministic output
is also required — the two requirements conflict, and determinism wins for a measurement tool
whose numbers go in a paper.
*Cost:* O(n²) time and O(n) memory per row. At the stated scale (hundreds of fields, hundreds
to low thousands of nuclei per field) this is seconds; cKDTree is the recorded swap if a field
ever has tens of thousands of nuclei.

**Douglas–Peucker epsilon defaults to 2.0 px, carried in `RunConfig`.**
*Why:* the Java has no published default to port — its Epsilon slider is an integer 0–50
starting at 0, which makes every chain-code point a corner, and the operator drags it per
cell. Any default is therefore this port's own convention and belongs with the other
reproducibility parameters.
*Cost if wrong:* corner counts shift with epsilon; the value is recorded per cell and
overridable per call, so a re-run with a different epsilon is one config change.

**Two measurement decisions deliberately left as-is, with the disagreement surfaced in a
column rather than resolved.** Both-cell-side rows are kept (it is what the original
measures) with a canonical `pair_id` and a junction count beside the side count; and the
modal-neighbour rule for DP interfaces is kept, with a `contact_fraction` column plus flags
for a low-contact DP interface and for a contact DP interface that TCJ found and DP dropped.
*Why:* deduplicating the tables would have broken every existing consumer and diverged from
the original.
*Cost if wrong:* a downstream analysis that ignores `pair_id` double-weights two-sided
junctions, and a reader who ignores `contact_fraction` over-reads a mostly-free-edge DP
interface.

**A short-interface-count edge case parked with its measured number.** The farthest-pair seed
is not given a neighbour-based tie-break for repeated boundary pixels at one-pixel necks:
measured 0 occurrences over 1,152 necked boundaries (5,134 traced), and no roll of those
changed the corner set.
*Why:* geometrically hard to arrange — a farthest-pair seed is an extreme point, a repeated
pixel is an interior constriction.
*Cost if wrong:* a pathological cell's corner count shifts by one between crops. Visible only
as irreproducibility, so the number is recorded in the docstring to be re-checked if it ever
shows up.

**Diagonal-only neighbour contacts are not chased.** 9 of 200 cells lack a neighbour that
3×3 dilation would see.
*Why:* a 1–2 px diagonal contact is not an interface worth measuring.
*Cost if wrong:* a real neighbour pair is absent from the TCJ output. The DP corner path is
unaffected and the two are compared downstream.

**Cells are segmented on a dedicated `segment` channel that is never measured, and a config
whose `segment` channel is also `JM1` or `JM2` is refused.** Segmentation without a `segment`
plane is refused too; there is no fallback to `JM1`. Supplied labels (`run --labels`) need no
`segment` channel.
*Why:* a boundary drawn on a marker's own signal makes that marker's occupancy ~100% by
construction. Seen on E-cadherin (AJ staining, AO0025) and afadin (MCF7 test, 2026-11-11). The
Java measures its boundary marker by design, because the operator drew the boundary; with
Cellpose drawing it, the same choice is circular.
*Cost if wrong:* a two-stain field yields one measured marker rather than two, and configs
written for segmentation on `JM1` must add a `segment` channel.

---

## 2. Output schema

**`n_neighbours` replaces `n_interfaces`,** keeping `n_interfaces_dp` and `n_interfaces_tcj`.
*Why:* DP and TCJ decompose the same boundary, so their sum double-counts while reading like
a neighbour count. A method-independent neighbour count is the column actually worth having.
*Cost if wrong:* one renamed column.

**A fifth table, `nuclei_distances`** (`field_id`, `nucleus_id`, `neighbour_nucleus_id`,
`rank`, `distance_px`, `distance_um`), deviating from the planned four. Stored in long form,
one row per ordered pair, with `rank_from_nucleus`.
*Why:* the original publishes the sorted per-nucleus distance list, and discohesion is a claim
about individual distances — a mean cannot distinguish one far neighbour from several
moderately far ones, which is the distinction the biology turns on. Per-rank columns on
`nuclei` would be ragged because the count depends on `num_intra_cells`.
*Cost if wrong:* one extra pair of output files nobody reads; the summary columns stay.

**Density columns are named for their real denominator.** `intensity_per_area` →
`intensity_per_px2`, `cluster_density` → `cluster_density_per_px2`, each with a `_per_um2`
twin.
*Why:* both denominators really are px², so the values are calibration-dependent while the
old names read as intensive — and a cross-calibration comparison is exactly what this tool
supports. This overrode a reviewer who judged the unscaled values defensible; the override is
on naming, not on the maths.
*Cost if wrong:* two renamed columns and two unused twins.

**Every reproducibility parameter lives in `RunConfig`:** `epsilon_px=2.0`, `tcj_min_run=3`,
`neighbour_radius_px=2`, at today's values, with an explicit `epsilon` argument still winning,
and `tcj_min_run`/`neighbour_radius_px` plumbed through `extract_interfaces` and `assess_cell`.
*Why:* a saved `RunConfig` must reproduce a run, and two of the three were unrecorded module
constants. Plumbing them through was chosen over leaving a `ConfigError` guard, which would
have left `RunConfig` advertising two knobs that error on every non-default value.
*Cost if wrong:* three config fields and two keyword arguments whose defaults change nothing.

**No per-role Cellpose diameter or threshold knobs.** The limitation is documented instead.
*Why:* `cpsam` is largely scale-agnostic (hence `diameter=None` and no estimate returned), and
a per-role knob cannot be validated against stubs — only against real Cellpose, which the
container cannot run.
*Cost if wrong:* a `nucleus_diameter` field is added later defaulting to today's behaviour, and
nothing already recorded becomes wrong.

---

## 3. Provenance and operator work

**The report identifies the code by git commit (with a dirty marker) or installed distribution
version, not by a constant `__version__`** — alongside the result-moving library versions and
the label sidecar's `params_used`, in the report's Segmentation section. This reopened an
earlier decision that a constant was sufficient.
*Why:* a constant `__version__` does not identify the code behind a figure.
*Cost if wrong:* provenance strings slightly longer than needed.

**Writing labels refuses by default on a collision, before any field is segmented,** with
`--overwrite-labels` to force it.
*Why:* `OUT/labels` is the napari plugin's output, so a stray re-run would discard
hand-corrected segmentations. Refusing *before* segmentation also means a long Cellpose run
cannot be thrown away at write time.
*Cost if wrong:* one flag, and users re-running into the same directory must pass it.

**A labels sidecar whose recorded source image differs from the field's warns loudly and is
recorded in provenance, but does not refuse.**
*Why:* labels from a different image of the same shape previously loaded silently, which is
plausible when fields are named systematically across experiments — but renames and
reorganised directories are normal work, so refusing would block legitimate use.
*Cost if wrong:* a noisy warning on reorganised data.

**`--labels` is a contract:** a missing file and a shape mismatch are errors naming the paths
and shapes, and provenance records that labels came from disk and from where.
*Why:* a published figure must not be mistakable for fresh Cellpose output.
*Cost if wrong:* nil; it is a recording requirement.

**A failed run-level calibration must not blame a human.** The override banner distinguishes
"a human set this per field" from "the run-level proposal failed, so each field fell back to
its own Otsu", agreeing with the Provenance section's "no marker role resolved".
*Why:* the report was stating something untrue about a human having overridden the value.
*Cost if wrong:* nil; it is a correction.

**Per-field failures are recorded and the run continues; exit 0 only when every field
succeeded.**
*Why:* one bad input should not lose a long run, but a partially failed run must not look
successful to a script.
*Cost if wrong:* nil.

**Unknown config keys are rejected in the CLI, not by `extra="forbid"` on `RunConfig`** —
against a reviewer's suggestion.
*Why:* `RunConfig` is also the library API, where a caller may legitimately construct one
programmatically; typo-catching belongs where YAML is parsed.
*Cost if wrong:* the check moves one layer down.

**The CLI accepts an OME-Zarr store directly,** rather than retracting the claim that it reads
them.
*Why:* the library genuinely reads OME-Zarr, the README says so, and only the entry point
could not be handed a store — retracting would have removed a working capability from the
user's reach.
*Cost if wrong:* a zarr CLI path exercised only by tests, rather than one that is unreachable.

**The container definition's reconstructibility claim was softened, not made true by pinning
the transitive set.** The def now states that its direct pins reproduce the direct set only,
and names `zarr`, `dask`, `numcodecs` and `bioio-base` as unpinned and able to drift. `zarr`
is declared explicitly (`pyproject.toml` dev extra `zarr>=3.0`; the def at `zarr==3.1.6`,
the version the live image already holds) because it had been load-bearing for tests while
present only transitively.
*Why:* pinning versions that could not be verified without a rebuild would have swapped an
overstated claim for an unverified one.
*Cost if wrong:* a rebuild can still drift in the transitive set — now visibly, because the
def says so, rather than behind a promise of reconstructibility.

---

## 4. Validation against the Java

**The spot-check harness validates the metric formulas, not the Java's corner algorithm.** The
Java's corner coordinates become inputs; our side is measured between those same pixels;
corner-count differences are reported as information with the hull explanation, never as
failures.
*Why:* the ask was the published method computed as the paper's formula describes, and exact
metric agreement was proved on a shared path — so a shared-path comparison isolates what
actually needs validating.
*Cost if wrong:* the harness never checks our corner algorithm against the Java's — which is
the point, since the two are documented to differ.

**Comparison rows must be interior cells; Douglas–Peucker only.** TCJ rows are rejected rather
than silently compared (TCJ has no Java counterpart), border rows skip with a reason, and the
recorded divergences are handled explicitly: `linearity_index` is compared after reciprocal
conversion, `fragmented_contour` and `coverage_index` are recorded rather than asserted.
*Why:* the Java's dilation band clips at the border.
*Cost if wrong:* a class of row is excluded from validation, which the skip reasons make
visible.

**`interface_area_px`, `marker_area_px` and `marker_intensity` must match exactly, with no
tolerance, for an interior cell** — restricted to rows established rectilinear from the
geometry, with oblique rows compared at a stated injection tolerance (~10% or 8 px for areas,
~15% for intensity).
*Why:* exact agreement on rectilinear interfaces is the strongest evidence this project has
and must stay sharp, while an oblique mismatch is an artefact of edge-map injection rather
than a port defect.
*Cost if wrong:* oblique rows catch only gross errors.

**A missing `reference.csv` skips with a reason; a malformed one fails.**
*Why:* a silent skip on a typo'd header would hide the entire validation.
*Cost if wrong:* nil.

**`reference.csv` is generated, never committed.** `scripts/java_reference.py` drives the real
Java headlessly, compiling into a scratch directory and never into the read-only Java tree;
the manual 12-step procedure stays documented as a cross-check.
*Why:* it is real-Java output that needs a host JDK and would go stale silently against the
Java checkout; the frozen golden slice already ties the port to the original on every test
run. Hand transcription is also slower, error-prone, and caps validation at a handful of cells.
*Cost if wrong:* a fresh checkout's five comparison tests keep skipping until someone runs the
generator, which the validation README tells them to do.

**The operator's click is part of the reference data.** The Java's output depends on where the
operator clicks to flood a cell — the same edge map and cell gives 155 edge pixels and 4
corners at (74,64) but 158 and 3 at (77,65). The generator clicks the pixel farthest from the
boundary, tie-broken smallest row then smallest column. `reference.csv` carries an optional
`seed_row`/`seed_col` pair, the CSV is named as the authoritative source, and the rule is
documented as "how to recompute it".
Three sub-decisions: the seed pair stays **optional** (a hand-transcribed row from a human's
own Java run is still useful without it, and the README states that such a row cannot be
reproduced exactly); seeds must **agree across all rows of one (image, cell_id)**, as a load
error (a CSV mixing two clicks for one cell mixes two corner sets — the exact
silent-wrong-numbers case the seed exists to exclude); and a present seed is **validated even
on an `interior=no` row** that is never compared.
*Cost if wrong:* a contributed row silently unreproducible, capped by the pair-and-containment
validation; a legitimate multi-run CSV refused and needing to be split per run, which is
visible and trivially recoverable; nothing beyond a stricter loader for the third.

**The 2-px injection barrier stays, with its justification corrected to symmetry and margin
rather than leak prevention.**
*Why:* a design justified by a false mechanism gets "simplified" by the next person on a wrong
premise.
*Cost if wrong:* a 1-px barrier would also work and would change no metric artefact.

**The README's click-variation table (154–158 edge pixels, 3–4 corners) stands as measured by
review rather than being re-measured.**
*Cost if wrong:* an illustrative number in a prose section is off. The load-bearing claim —
that the click changes the result at all — was observed directly by driving the real Java.

---

## 5. The QC report

**Per-field detail stays in the HTML, uncapped** (~16.7 KB per field of rows, ~8.5 MB at 500
fields × markers); no pooled-only mode.
*Why:* the rows sit in collapsed `details`, a local file that size opens fine, and the
per-field detail is the point — the alternative hides exactly the variation the operator
reviews.
*Cost if wrong:* a pooled-only switch added later, with no change to any computation.

**The threshold-disagreement flag's behaviour stands; its documentation was what was wrong.**
A one-grey-level difference against a threshold of 2.28 really is a large relative
disagreement, and the 1.0 floor exists so a 16-bit plane does not trip on noise. The tolerance
was explicitly *not* retuned.
*Why:* the claim "the default path does not trip the flag" was false; the behaviour was not.
*Cost if wrong:* a reader of a low-signal run sees an informational flag with documentation
explaining it, instead of a tolerance tuned to hide it.

**The override banner, when it fires, must name at least two fields with different applied
values.** The fired condition is the span of applied values, so the named set is the fields
that establish it.
*Why:* the banner could previously fire and then say "(0 fields in all)".
*Cost if wrong:* nil; it is a correction.

**The strip-title overlap fix ships without an automated test.**
*Why:* a re-review rendered both layouts and its own bounding-box detector reported zero
overlaps for the old, visibly broken layout too — so a naive overlap test would not have
caught the original defect, and any minimum-gap threshold would be arbitrary. The title text
is now structurally shorter, which is the real fix.
*Cost if wrong:* a cosmetic regression that no number depends on.

**The degraded-calibration note reports the number of files actually opened.** It previously
reported the length of the candidate list (up to 10), not the opens — so 13 inputs with one
corrupt file opened 6 and said "1 of the 10 calibration file(s) opened".
*Why:* the same standard applied to the override banner — a report must not state something
untrue — and the count is in the same report.
*Cost if wrong:* nil; it is a correction.

**The memory cost of narrowing before Otsu is accepted and documented, not engineered away.**
Computing Otsu only on the ≤5 calibration fields cannot be done while streaming, because the
subset is defined over the full sorted field id set — so `propose_run_thresholds` holds every
field it is handed until the subset is known, where it previously held one. The bound is in
*files* (≤5 readable ones on the CLI path), not in fields or bytes, so a multi-scene file
raises it: roughly 600 MB at 5 files × 10 scenes × 1024² × 3 float32.
*Why:* it is the direct consequence of narrowing before Otsu, trades CPU for memory exactly as
that decision requires, moves no measured value, and changes nothing in the supported
one-field-per-file use.
*Cost if wrong:* a library consumer streaming a multi-scene run meets memory pressure the
docstring under-describes. Recoverable by passing a calibration pass's own handful of files,
which the docstring already instructs.

**A no-pixel-size fallback path is kept even though `read_fields` raises `CalibrationError`
first and makes it unreachable today.**
*Why:* a napari-written labels path could bypass the reader.
*Cost if wrong:* one defensive branch and its test.

---

## 6. Conventions that affect no measurement

**One exception base.** `src/junctionmapper/errors.py` defines `JunctionMapperError`; every
module raises a subclass (`CalibrationError`, `ChannelError`, `BoundaryError`, `FormatError`,
`ConfigError`) so the CLI catches one type at each command top and exits non-zero with one
clean line. `ConfigError` inherits from both `JunctionMapperError` and `ValueError` so that
callers catching `ValueError` still work.
*Cost if wrong:* a module move and one raise site.

**Both `Interface.method` and the `corner_method` column keep their names,** with a mapping
between them.
*Why:* renaming either would contradict a task's own tests.
*Cost if wrong:* cosmetic.

**`rect_nucleus` bounds are inclusive,** matching the guide's `(max−min)//2 + min` centroid.
*Cost if wrong:* one fixture and one assertion.

**Tests always run as `./scripts/jm pytest …`** inside Apptainer, never bare `pytest`.
*Cost if wrong:* none; the wrapper is a pass-through.

**Commit trailers name this session's model.** Subagents are this session's tools, not
separate authors.
*Cost if wrong:* nil.

**Label arrays round-trip exactly via `np.save` with no relabelling,** tested with
non-contiguous ids 1 / 5 / 9000.
*Cost if wrong:* nil; it is a fidelity requirement.

**The collision pre-scan re-reads every image.**
*Why:* cheap next to Cellpose, and the alternative is discovering a collision after a long
segmentation run.
*Cost if wrong:* a few seconds per run.

**Four test-only gaps were fixed rather than parked,** despite being Minor: a test gap stops
being cosmetic when it leaves the central claim of a correctness fix unpinned. Two of these
did — code identity, and the CLI half of the run-level threshold, where `files[:n]` passed the
entire suite.
*Cost if wrong:* four tests' worth of work on a branch that was otherwise finishable.

**Cellpose-SAM and torch live in the one image, not a second one.** The container definition
originally excluded them: the library imports cellpose lazily and the suite stubs it, so a
~340 MB image ran the whole suite, and the def told the reader to build a separate image for
GPU segmentation. That separate image was written and then abandoned, because it had to repeat
all fifteen runtime pins and keep the two lists in step by hand.
*Why:* the duplication is a worse risk than the size. If the two pin lists ever drifted, a
label image and the measurements taken from it would come from different numpy and
scikit-image versions, which is precisely the failure the exact pinning exists to prevent. One
image makes the two steps share a runtime by construction instead of by discipline.
*Cost if wrong:* every `pytest` run and every CPU node pulls a ~3 GB CUDA stack it cannot use,
and the image takes appreciably longer to build and to stage. Reversible by splitting the def
again, at which point the pin duplication has to be solved properly — layering the second
image on the first with `Bootstrap: localimage` rather than copying the list.
