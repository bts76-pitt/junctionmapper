# Membrane score: design

Date: 2026-11-17. Status: draft after builder check, awaiting user approval. Builds on the run pipeline
(`cli.py run`, `pipeline.py`, `tables.py`).

## Intent

Say whether a stain sits in thin membrane-like lines or is spread through the cell, without
drawing cell outlines.

- Every junction metric today depends on outlines, and outlines come from one stain.
- When the stain under test is also what the outline depends on, the metric is circular.
- Example that prompted this: a panel whose only edge-marking stains depend on the protein being
  knocked out. Outlines fail exactly where the answer is interesting.
- The score must work on any panel. junctionmapper knows roles and marker names, never biology.

## Goals

- One score per patch per stain channel, needing no segmentation.
- A per-field summary: typical score, and the share of cell area that is membrane-like.
- The same number means the same thing for any stain (normalised to a structureless baseline).
- A blank or near-blank channel is reported as too dim, never as diffuse.
- Produced even for fields where segmentation finds no cells.

## Non-goals

- No dashboard changes. The selector redesign (by line, marker, group) is the next project and
  will display these tables.
- No per-cell or per-junction score.
- No change to existing tables or metrics.

## Anti-goals

- Any stain-, line- or experiment-specific logic or default.
- A score that reads "diffuse" on a channel with no signal.
- Outlines or nucleus-based territories used to place membranes.

## Approaches

- **A. A step inside `run`, before segmentation (chosen by the user).** One command; tables land
  in the run folder beside the others; provenance shared. Scored before segmentation, so it does
  not depend on cells. `--no-membrane-score` turns it off.
- B. Separate `junctionmapper localize` command. Faster to rerun while tuning, but a second
  command and a manual join to `run` outputs. Rejected; can be added on top of A later.
- C. Both. Most surface to maintain. Rejected.

## What gets built

### Inputs

- Channels scored: every configured role except `nucleus` (JM1, JM2, segment, whichever exist).
- Cell region (where cells are, not where they meet): nuclei found by Otsu on the smoothed raw
  nucleus plane, grown outward by `region_um`.
  - Always from the raw plane, also under `--labels`: the score must not depend on any labels,
    so a fresh run and a re-measure give the same region.
  - A nucleus plane with no Otsu foreground: every patch of that field is unscored
    (`region_fraction` 0), the field still succeeds.
  - No fallback for configs without a nucleus role: `RunConfig` requires one today.

### Score, per patch and channel

1. Square patches of `patch_um`, on a fixed grid. A patch is scored when at least half of it is
   cell region.
2. Ridge response: a multi-scale line filter (Sato or equivalent) at scales covering
   `line_width_um`.
3. Contrast = (mean intensity on the patch's strongest-ridge pixels (top `ridge_fraction` of
   ridge response within this patch's cell region) - `background`) / (mean on the rest of the
   patch's cell region - `background`), with `background` the field floor of step 7. The same
   subtraction applies to the baseline, so the score is invariant to an additive offset as well
   as to gain. Contrast is null when the background-subtracted rest mean is <= 0.
   The cut is per patch, so a dim patch in a bright field is scored on its own terms.
4. Baseline = steps 2 and 3 re-run, filter included, on a structureless copy of the patch
   (its cell-region pixels shuffled once, seeded). Reusing the original ridge mask on shuffled
   values would give about 1 always and normalise nothing.
5. `score` = contrast / baseline. 1 = no line structure.
6. `membrane_like` = `score >= cutoff`; null where `score` is null.
7. Too dim: the patch's ridge pixels are not clearly above the field's floor; the patch gets no score
   (`too_dim = true`, `score` null).
   - `background` = the field's 1st percentile of that channel (no cell-free pixels needed;
     confluent fields are normal).
   - `noise` = robust spread (MAD) of the field's high-pass residual, over the pixels above
     `background` (the smoothing also uses only those), so a large exact-constant floor
     (zeroed or clipped pixels) does not make it 0. If it is still 0, the robust spread of the
     cell-region pixels; 0 only when there is no signal above the floor.
   - too dim when (ridge-pixel mean - background) < 5 x noise. The ridge-pixel mean is the
     mean on the top `ridge_fraction` of line-filter pixels, as in step 3.
   - Amended after the real-data check: the first version used the patch cell-region mean at
     3 x noise. On dim 8-bit fields thin membranes barely move the region mean (1 to 4
     counts), so it rejected about 60% of patches, reference stains included. Pure noise
     puts the ridge-pixel mean about 4 x noise up (2.33 from the 1st-percentile floor, plus
     up to 1.75 for picking the top tenth), hence 5.
   - Scale-invariant and stain-independent; a channel at quantisation level (e.g. 0 to 2 on
     8-bit) fails it.

### Outputs

- `membrane_patches.{parquet,csv}`: one row per patch and channel.
  - `field_id, role, marker, patch_row, patch_col, y0_px, x0_px, size_px, region_fraction,
    mean_intensity, ridge_intensity, background, noise, too_dim, contrast, baseline, score, membrane_like`.
- `membrane_fields.{parquet,csv}`: one row per field and channel.
  - `field_id, role, marker, n_patches, n_scored, n_too_dim, median_score,
    membrane_like_fraction` (share of scored patches, weighted by region area).
- `run.json` provenance: the parameters used and `membrane_score: on|off`.
- Both tables exist (empty) when the step is off, so downstream readers need no special case.

### Config

A new optional block; every key has a default; `extra="forbid"` as elsewhere.

```yaml
membrane_score:
  line_width_um: [0.3, 0.6]   # range of membrane widths the line filter looks for
  patch_um: 20                # about one cell across
  region_um: 6                # how far the cell region extends from nuclei
  ridge_fraction: 0.1
  cutoff: 1.5
```

CLI: `--no-membrane-score` on `run`.

## Constraints

- Generic: no marker names, line names or biology anywhere in code or defaults.
- Lengths in µm, converted with the field's pixel size; a field with no calibration fails with
  the existing calibration error, as other µm settings do.
- Cost: at most about 10 s per field per scored channel at 2984 x 2500, both filter passes
  included. Downsampling for the filter is allowed if the smallest scale stays at least 1 px.
- Field-atomic as today: if any later step fails for a field, that field's rows are dropped with
  the rest. (A field fails only on read errors or exceptions; zero cells is not a failure.)
- Deterministic: the shuffle uses a seed derived from `field_id`, role and patch.
- Never write into `--labels` directories.

## My calls (user may override)

- Defaults above. `cutoff` 1.5 is provisional, to be checked on real data (below).
- Region from Otsu nuclei on the raw plane, never from label files.
- Baseline re-runs the filter on one seeded shuffle per patch.
- Too-dim test from field 1st percentile and high-pass noise (above).
- Weighted-by-area fraction, so a patch half outside the region counts half.

## Validation

- Synthetic tests: thin rings on a uniform background score well above cutoff; uniform noise
  and smooth blobs score near 1; a blank channel is all too dim; results invariant to intensity
  scale (x10).
- Real-data check (a recorded run, not a test; it fails nothing automatically):
  - Data: the AO0025 79-line dataset, inputs and configs listed in
    `/ix3/alee/LO_LAB/Personal/Brent_Schlegel/scratch/jm-all/tasks_abs.tsv`.
  - Spike for comparison: `/ix3/alee/LO_LAB/Personal/Brent_Schlegel/scratch/jm-report/membrane_spike/`
    (`ridge_score.py`, `ridge.tsv`, `crops.png`).
  - Which channels count as membrane references (there: Na/K-ATPase, phalloidin) and which
    pairs should separate live in an analysis script outside the repo, never in junctionmapper.
  - The builder reports per-channel `membrane_like_fraction` and crops; the user decides whether
    `cutoff` and `ridge_fraction` hold. A miss reopens tuning, it does not block code review.
- Known risk from the spike: punctate signal can score as line-like. The structureless baseline
  does not remove it. If real data shows this, the builder adds a blob-vs-line check.

## Left to the builder

- Exact line filter and scales, downsampling, and the too-dim test (must be scale-invariant
  and stain-independent).
- Code layout (suggested: `measure/membrane.py`, called from `cli.run`'s `work()` before
  segmentation; result carried on `FieldResult`).
