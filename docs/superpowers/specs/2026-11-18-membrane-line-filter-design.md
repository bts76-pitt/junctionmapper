# Membrane score: lines, not dots

Amends `2026-11-17-membrane-score-design.md`. Small change; short design.

## Why

- Sato in 2D ignores the second Hessian eigenvalue: a round dot gets 71% of a line's response.
- The pixel shuffle destroys dots, so the baseline cannot cancel them.
- Result: dotted cytoplasmic stains score as membrane. SUM44 -dox alpha-catenin (dotted) scores
  2.35, its junctional +dox field 2.23.

## Goals

- Continuous lines score as lines.
- Beaded lines (dots strung along a border) score as lines. Junctions can be punctate.
- Scattered dots score near 1, like a diffuse stain.

## Non-goals

- Measuring fragmentation. `coverage_index` (outline-based, from the original paper) does that
  wherever an independent outline exists.

## Constraints

- Generic: no marker, line or biology names in code.
- Scale- and offset-invariant, as now.

## Approaches

1. **Chosen: anisotropic line filter at one coarse scale.** Suppresses structure that curves
   equally both ways; the coarse scale merges beads into a line and blurs scattered dots to haze.
2. Second statistic, orientation coherence (structure tensor). Rejected: says dots align, not how
   much intensity sits on lines; two numbers and more parameters.
3. Keep Sato, baseline from relocated dots. Dropped: needs a threshold to find dots, and Sato
   scores beads and scattered dots the same at every scale.

## Design

- Filter: `sigma**2 * max(0, -l2 - |l1|)`, Hessian eigenvalues at one `sigma`; `l2` is the most
  negative. Replaces Sato.
- `sigma = dot_spacing_um / 2`, converted to (downsampled) pixels.
- Single scale, no max over a range: at fine scales a dot's rim reads as a short curved line.
- New config `dot_spacing_um`, default 1.0. Must exceed `line_width_um` hi.
  - Meaning: dots farther apart than this do not make a line.
- `line_width_um` now sets only the downsampling (the thinnest line resolved). This keeps the
  filter about 3x a dot's width, the regime where dots drop out.
- `cutoff` becomes optional, default unset.
  - Unset: `membrane_like` and `membrane_like_fraction` are null; `score` is the readout.
  - Each experiment sets its cutoff from its own controls.
- Unchanged: shuffle baseline, too-dim test, table columns.

## Evidence (synthetic, 0.1 um/px, dots and line blurred sigma 1.5 px)

Mean image value on the top 2% of filter response, over the image mean:

| sigma (px)       | 1.5 | 2.5 | 3.5 | 5   | 7   | 10  |
|------------------|-----|-----|-----|-----|-----|-----|
| continuous line  | 45  | 46  | 47  | 45  | 43  | 40  |
| beads 1 um apart | 42  | 48  | 46  | 44  | 43  | 40  |
| beads 2 um apart | 38  | 49  | 50  | 19  | 23  | 38  |
| scattered dots   | 38  | 48  | 22  | 2.0 | 0.4 | 7.8 |
| dense scattered  | 11  | 9   | 8   | 6   | 5   | 4   |

Sato gives scattered dots 48 at both 1.5 and 5 px.

## Tests

- Continuous line, 1 um beads: score well above 1.
- Scattered dots, dense dots: score near 1.
- Cutoff unset: `membrane_like` all null; set: as now.
- `dot_spacing_um` validation.
- Existing tests re-banded where the new filter moves them.

## Validation

- Rerun the 79 lines with `min_ridge_intensity: 5`.
- SUM44 -dox alpha-catenin drops toward 1.
- p120 -dox / +dox and the reference stains keep their split.
- Then choose the AO0025 cutoff, in the experiment config.

## Left to the builder

- The exact downsampling guard.
- Re-banding of existing tests.
