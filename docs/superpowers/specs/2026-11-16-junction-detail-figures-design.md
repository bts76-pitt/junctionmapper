# Junction detail figures: design

Date: 2026-11-16. Status: approved by the user. Builds on
`2026-11-16-interactive-report-design.md` (PR #8).

## Intent

The detail panel shows the right numbers, but a column of 14 values per side does not explain
a junction. Replace it with figures that show the junction and why its numbers came out.

## What gets built (approved)

1. **Close-up.** One crop per channel of the selected junction, from the field previews.
   The junction's band is outlined; both cells' outlines faint; a dot marks where the profile
   starts. Merge colours follow merge order (green, magenta, cyan), not channel roles.
2. **Intensity along the junction.** Marker brightness from corner to corner, JM1 and JM2
   (whichever exist), with each marker's applied threshold as a line.
3. **Where it sits.** One strip per metric: every junction of the field (toggle: whole run)
   as faint dots, this junction marked, its percentile beside it. Per-side values and the mean
   move to the hover tooltip.

## Approaches

- **A. Profile computed by junctionmapper during measurement (chosen).** Full-resolution
  pixels, real intensity units, so the threshold line is meaningful. Adds a few KB per junction
  to the field file.
- B. Profile sampled in the browser from the previews. No pipeline change, but previews are
  contrast-stretched: no real units, no threshold line. Rejected.

## Constraints

- All global constraints of the interactive-report design hold (offline, no external refs,
  generic, tables unchanged, never write into `--labels`).
- Profiles are written to the field asset only. No table changes.

## My calls

- Profile = one per junction (`pair_id`) and marker role, sampled along the interface pixel
  walk of the lower cell id's side (the higher id's when the lower has none). A dp junction's
  pieces are joined in walk order, the line broken at each corner. Value at each path pixel = mean marker intensity over
  band pixels within the band half-width of it (the same dilated band the metrics use).
  Resampled to at most 200 points. x axis = distance along the path in µm (px if uncalibrated).
- Threshold per role per field = the applied threshold already recorded in provenance
  (`thresholds`), written into the field asset beside the profiles.
- Close-up window = junction bounding box plus 40% margin, at least 48 preview px, drawn from
  the decoded preview planes already in memory. Tiles for every channel present.
- Strips default to the field; a one-click toggle switches to the whole run.
- Values stay one hover away; the panel has no table.

## Left to the builder

- Profile smoothing (none unless it reads badly).
- Exact layout and sizes within the panel width; light and dark themes.
