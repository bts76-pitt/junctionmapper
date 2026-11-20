# junctionmapper

Measures cell–cell junctions in fluorescence microscopy images.

This is a Python reimplementation of Junction Mapper (Imperial College London,
[doi:10.7554/eLife.45413](https://doi.org/10.7554/eLife.45413)). The original is a Java app in
which you click every cell and corner by hand. This version uses the published measurements and
finds the cells and corners automatically.

See also: [settings, outputs and columns](docs/reference.md) |
[differences from the original](docs/vs-original.md) |
[design decisions](docs/decisions.md)

## What it does

1. Reads OME-TIFF or OME-Zarr images.
2. Segments cells and nuclei with Cellpose-SAM, or loads label images you supply.
3. Splits each cell outline into interfaces, one per neighbouring cell.
4. Measures the published junction parameters on each interface, for one or two markers.
5. Writes the results as `.parquet` and `.csv` tables, plus an HTML report.

The config file maps each image channel to a role and names the protein in it. Channel order
doesn't matter.

## Running it

Everything runs inside one Apptainer image. Build it once:

```bash
./scripts/build-image          # makes container/junctionmapper.sif
./scripts/jm pytest -q         # optional: run the tests
```

To run on a GPU node:

```bash
apptainer exec --nv --cleanenv --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/project/src \
  --bind "$PWD":/project --bind /ix1:/ix1 --bind /ix3:/ix3 --pwd /project \
  container/junctionmapper.sif \
  python -m junctionmapper.cli run -c config.yaml -i images/ -o out/ --diameter 250
```

- Without `--nv` it runs on the CPU, which is very slow.
- Bind whichever folders hold your images and output (`/ix1` and `/ix3` on CRC).
- `-i` takes one image or a folder of images.

The rest of this page writes `junctionmapper` for `python -m junctionmapper.cli`.

To check or fix the segmentation before measuring, run it in two steps:

```bash
junctionmapper segment -c config.yaml -i images/ -o seg/ --diameter 250
# check or edit seg/labels/, e.g. in napari
junctionmapper run -c config.yaml -i images/ -o out/ --labels seg/labels
```

With `--labels`, `run` uses those labels and skips Cellpose.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | All fields succeeded. |
| 1 | The run failed and wrote nothing. |
| 2 | Bad command-line option. |
| 3 | Some fields failed; the others were written. Check the log. |

## Config

```yaml
channels:
  - {role: segment, channel: 3, marker: "Na/K-ATPase"}   # cells are outlined on this
  - {role: JM1,     channel: 2, marker: "E-cadherin"}    # junction marker 1, required
  - {role: JM2,     channel: 1, marker: "p120"}          # junction marker 2, optional
  - {role: nucleus, channel: 0, marker: "Hoechst"}       # required
pixel_size_um: 0.0749      # used if the image has no pixel size
```

- Cellpose needs a `segment` channel.
- The `segment` channel can't also be `JM1` or `JM2`.
- Unknown keys raise an error.

The settings you're most likely to change:

| Key | Default | Meaning |
|---|---|---|
| `dilation_px` | 2 | How far each interface is widened into a band, in pixels. |
| `dilation_um` | | The same, in µm. |
| `threshold` | | Fixed intensity threshold per marker. By default it's set from the data. |
| `epsilon_px` | 2.0 | Corner tolerance for the `dp` corner method. |

All settings, including the membrane score ones, are listed in [docs/reference.md](docs/reference.md#all-settings).

## Before you run

Set `--diameter` to the typical cell diameter in pixels. Cellpose-SAM doesn't estimate it, and
without it large cells get split into fragments. Measure it from your own images.

Choose the `segment` channel carefully. Cells are outlined on it and it is never measured as a
marker. A membrane stain your experiment doesn't change works best, such as Na/K-ATPase or
phalloidin. If the stain does depend on the experiment (a cadherin in a cadherin knockout, say),
some cells may not be found at all, which shows up as lower coverage.

Check the intensity threshold. Four of the measurements depend on it; see
[thresholds](docs/reference.md#intensity-thresholds).

Cellpose outlines can sit a few pixels off the membrane. `snap` moves cell–cell edges onto it, by
at most 8 px:

```bash
junctionmapper snap -c config.yaml -i images/ --labels seg/labels -o snapped/
junctionmapper run  -c config.yaml -i images/ --labels snapped/labels -o out/
```

`--snap-to-membrane` on `segment` or `run` does the same thing in one step.

`--with-nucleus` gives Cellpose the nucleus channel as well. It splits fewer nuclei, but the
outlines can drift off the membrane. It's off by default.

## Output

| Table | One row per |
|---|---|
| `interfaces` | side of a junction, per marker |
| `cells` | cell |
| `nuclei` | nucleus |
| `nuclei_distances` | nucleus and one of its near neighbours |
| `flags` | cell (quality checks) |
| `membrane_patches` | ~20 µm square, per channel |
| `membrane_fields` | image, per channel |

A single junction can produce anywhere from 1 to 10+ rows in `interfaces`, so count junctions
with `pair_id`:

```python
df.pair_id.nunique()             # number of junctions
df.groupby("pair_id").mean()     # one row per junction
```

The report is `out/report/index.html` and opens in any browser without a network connection.
Copy the whole `report/` folder, since the page loads files next to it. To put several runs in
one report:

```bash
junctionmapper report OUT_A OUT_B -o compare --label control --label treated
```

The columns are described in [docs/reference.md](docs/reference.md).

## Status

Under active development. Design notes are in `docs/superpowers/`.
