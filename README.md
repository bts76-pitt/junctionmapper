# junctionmapper

Measures cell–cell junctions in fluorescence microscopy images.

It reimplements **Junction Mapper** (Imperial College London,
[doi:10.7554/eLife.45413](https://doi.org/10.7554/eLife.45413)). The original is a Java app
where you click every cell and corner. This version keeps the published measurements and
automates the clicking. The method is the original authors'.

**More detail:** [Reference: settings, outputs, columns](docs/reference.md) |
[Differences from the original](docs/vs-original.md) |
[Design decisions](docs/decisions.md)

---

## What it does

1. Reads your images (OME-TIFF or OME-Zarr).
2. Finds cells and nuclei with Cellpose-SAM, or uses outlines you supply.
3. Splits each cell's outline into **interfaces**: one stretch per neighbouring cell.
4. Measures the published junction parameters on each interface, for one or two markers.
5. Writes tables (`.parquet` and `.csv`) and a report you open in a browser.

You say which channel holds which protein. Nothing is guessed from channel order.

---

## Run it

Everything runs inside one Apptainer image. Build it once:

```bash
./scripts/build-image          # makes container/junctionmapper.sif
./scripts/jm pytest -q         # optional: run the tests
```

Then run on your images. On a GPU node:

```bash
apptainer exec --nv --cleanenv --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/project/src \
  --bind "$PWD":/project --bind /ix1:/ix1 --bind /ix3:/ix3 --pwd /project \
  container/junctionmapper.sif \
  python -m junctionmapper.cli run -c config.yaml -i images/ -o out/ --diameter 250
```

- Leave out `--nv` and it still runs, but on CPU, very slowly.
- Bind the folders that hold your images and output (`/ix1`, `/ix3` on CRC).
- `--input` takes one image or a folder of them.

Below, `junctionmapper` is short for `python -m junctionmapper.cli`.

### Check the outlines first (optional)

```bash
junctionmapper segment -c config.yaml -i images/ -o seg/ --diameter 250
#   look at or fix seg/labels/ (for example in napari)
junctionmapper run -c config.yaml -i images/ -o out/ --labels seg/labels
```

`run --labels` uses your outlines as they are and skips Cellpose.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | All fields worked. |
| 1 | Failed. Nothing written. |
| 2 | Bad command-line option. |
| 3 | **Some fields failed.** The rest were written. Not a success. |

---

## Configure it

```yaml
channels:
  - {role: segment, channel: 3, marker: "Na/K-ATPase"}   # cells are outlined on this
  - {role: JM1,     channel: 2, marker: "E-cadherin"}    # junction marker 1 (required)
  - {role: JM2,     channel: 1, marker: "p120"}          # junction marker 2 (optional)
  - {role: nucleus, channel: 0, marker: "Hoechst"}       # required
pixel_size_um: 0.0749      # only used if the image has no pixel size
```

- `segment` is needed whenever Cellpose runs.
- `segment` cannot be the same channel as `JM1` or `JM2`.
- A misspelled key is an error.

Common settings:

| Key | Default | What it does |
|---|---|---|
| `dilation_px` | 2 | How far each interface is widened into a band. |
| `dilation_um` | — | Same, in µm. |
| `threshold` | — | Fixed intensity cutoff per marker. Default: chosen from the data. |
| `epsilon_px` | 2.0 | Corner sensitivity for the `dp` method. |

The full list, including the membrane score settings, is in [docs/reference.md](docs/reference.md#all-settings).

---

## Before you run

Three things decide whether the numbers mean anything.

**1. Set `--diameter`** (cell diameter in pixels).
- Without it, large cells break into fragments, with no error.
- Measure it from your own cells.

**2. Pick the `segment` channel with care.**
- Cells are outlined on it, and it is never measured as a marker.
- Best: a membrane stain your experiment does not change (Na/K-ATPase, phalloidin).
- If the stain depends on your experiment (for example a cadherin in a cadherin knockout), cells that lost it may not be found. Read missing cells as lost coverage, not as a result.

**3. Check the intensity threshold.** Four measurements depend on it.
See [thresholds](docs/reference.md#intensity-thresholds).

### Optional: move outlines onto the membrane

Cellpose outlines can sit a few pixels off the membrane. `snap` moves them onto it:

```bash
junctionmapper snap -c config.yaml -i images/ --labels seg/labels -o snapped/
junctionmapper run  -c config.yaml -i images/ --labels snapped/labels -o out/
```

- `--snap-to-membrane` on `segment` or `run` does the same in one step.
- Only cell–cell edges move, by at most 8 px.

`--with-nucleus` also shows Cellpose the nuclei. It splits fewer nuclei, but outlines can drift off the membrane. Off by default.

---

## What you get

| File | One row per |
|---|---|
| `interfaces` | cell side of a junction, per marker |
| `cells` | cell |
| `nuclei` | nucleus |
| `nuclei_distances` | pair of nearby nuclei |
| `flags` | cell (quality checks) |
| `membrane_patches` | ~20 µm square, per channel |
| `membrane_fields` | image, per channel |
| `report/index.html` | the whole run, to view |

**Count junctions with `pair_id`, not rows.** One junction can give anywhere from 1 to 10+ rows.

```python
df.pair_id.nunique()             # number of junctions
df.groupby("pair_id").mean()     # one row per junction
```

**The report** opens offline in any browser. Copy the whole `report/` folder; it needs the files beside it.

Compare runs on one page:

```bash
junctionmapper report OUT_A OUT_B -o compare --label control --label treated
```

Every column is explained in [docs/reference.md](docs/reference.md).

---

## Status

Under active development. Design notes are in `docs/superpowers/`; decisions in `docs/decisions.md`.
