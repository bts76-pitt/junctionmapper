"""Command line interface: ``junctionmapper run`` and ``junctionmapper segment``.

Error policy
------------
Everything the user can cause -- bad config, bad input, bad labels, a missing
optional dependency -- surfaces as a ``JunctionMapperError``.

Each command catches that, prints one clean line to stderr, and exits non-zero.

Unexpected exceptions are real bugs, and propagate.

Exit codes
----------
0  every field succeeded
1  the command failed: bad config or input, or every field failed (nothing written)
2  usage error (missing or unknown option; raised by typer itself)
3  partial: some fields failed, and the successful fields' outputs were written

The labels directory
--------------------
``segment`` writes three files per field:

    <field_id>.JM1.npy             cell labels
    <field_id>.nucleus.npy         nucleus labels
    <field_id>.provenance.json     what produced them

``run --labels DIR`` reads the ``.npy`` files back unchanged.

This directory is the seam a human-in-the-loop editor writes to.
"""
from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
from datetime import datetime, timezone
from collections.abc import Callable, Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import typer
import yaml
from pydantic import ValidationError

from junctionmapper import __version__
from junctionmapper.config import Role, RunConfig
from junctionmapper.errors import ConfigError, JunctionMapperError, SegmentationError
from junctionmapper.io.reader import Field, read_fields
from junctionmapper.provenance import code_provenance
from junctionmapper.pipeline import (
    CALIBRATION_FIELDS, FieldResult, RunThresholds, measure_field, propose_run_thresholds,
    resolve_thresholds,
)
from junctionmapper.report.build import build_report
from junctionmapper.segment.cellpose import Segmentation, SegmentationParams, segment_field
from junctionmapper.segment.snap import ERODE_PX, SIGMA_PX, SMOOTH_PX, snap_params, snap_to_membrane
from junctionmapper.tables import write_tables

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Cell-cell junction quantification (Junction Mapper, in Python).")

EXIT_FAILED, EXIT_PARTIAL = 1, 3
# What the design supports. Vendor formats are explicitly out of scope (spec, "File formats"),
# so they are recognised in order to be REFUSED by name rather than opened and failed deep
# inside a reader; a file with neither suffix is not an image and is ignored.
_INPUT_SUFFIXES = {".tif", ".tiff", ".zarr"}
_OUT_OF_SCOPE_SUFFIXES = {".czi", ".nd2", ".lif", ".oib", ".oif", ".ims", ".lsm", ".vsi"}
_MAX_IDS_SHOWN = 5

ConfigOpt = typer.Option(..., "--config", "-c", help="YAML run configuration.")
InputOpt = typer.Option(..., "--input", "-i", help="Image file, or a directory of image files.")
OutOpt = typer.Option(..., "--out", "-o", help="Output directory (created if absent).")


# ---- small helpers -------------------------------------------------------------------

# Cellpose 4 does not estimate a diameter: None means no rescaling at all, and cells much
# larger than the ~30 px the model was trained on break into fragments with no error.
_SNAP_HELP = (
    "After Cellpose, move cell-cell boundaries onto the segment-plane membrane ridge and "
    "smooth them (erode 8 px, sigma 3, smooth 20 px). To snap existing labels, use `snap`.")
_WITH_NUCLEUS_HELP = (
    "Also give Cellpose the nucleus plane when drawing cells (channel 2). Stops cell "
    "boundaries cutting through nuclei. Needs a 'nucleus' channel in the config.")
_DIAMETER_HELP = (
    "Cell diameter in px. Set it: with no value Cellpose does not rescale, "
    "and cells far above ~30 px fragment."
)


def _version() -> str:
    """Installed distribution metadata, else the package's own ``__version__``.

    This does NOT identify the code. See ``provenance.code_identity``, which is
    recorded beside it in every sidecar and report.
    """
    try:
        return metadata.version("junctionmapper")
    except metadata.PackageNotFoundError:
        return __version__ or "unknown"


def _oneline(text: object) -> str:
    return " ".join(str(text).split())


def _guard(fn: Callable) -> Callable:
    """Turn any JunctionMapperError into one stderr line and exit code 1."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except JunctionMapperError as exc:
            typer.echo(f"error: {_oneline(exc)}", err=True)
            raise typer.Exit(EXIT_FAILED) from None
    return wrapper


def _load_config(path: Path) -> tuple[RunConfig, dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        reason = getattr(exc, "strerror", None) or exc
        raise ConfigError(f"{path}: cannot read config file ({_oneline(reason)})") from None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML ({_oneline(exc)})") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config must be a YAML mapping of settings, "
                          f"got {type(raw).__name__}")
    unknown = sorted(set(map(str, raw)) - set(RunConfig.model_fields))
    if unknown:
        raise ConfigError(f"{path}: unknown setting(s) {unknown}; "
                          f"valid settings are {sorted(RunConfig.model_fields)}")
    # An omitted pixel size means "take it from the image metadata", the same as null; if the
    # image has none either, the reader raises a CalibrationError that names pixel_size_um.
    given = {"pixel_size_um": None, **raw}
    try:
        return RunConfig(**given), raw
    except ValidationError as exc:
        parts = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"])
            msg = err["msg"].removeprefix("Value error, ")
            parts.append(f"{where}: {msg}" if where else msg)
        raise ConfigError(f"{path}: invalid config: {_oneline('; '.join(parts))}") from None


def _is_input(path: Path) -> bool:
    """True if ``path`` names an input in its own right, decided by suffix alone.

    An OME-Zarr store is a DIRECTORY whose name ends ``.zarr``, so "is this an
    input" cannot be decided with ``is_file()``.

    Out-of-scope suffixes count here too, so a vendor file is picked up and then
    refused BY NAME in ``_iter_fields``, rather than ignored.

    (Deciding on ``is_file()`` left every zarr invocation unreachable from the CLI,
    even though the reader supports the format.)
    """
    return path.suffix.lower() in (_INPUT_SUFFIXES | _OUT_OF_SCOPE_SUFFIXES)


def _input_files(path: Path) -> list[Path]:
    """The input leaves of ``path``: ``path`` itself, or the recognised entries inside it.

    A recognised suffix wins over ``is_dir()``, so ``--input store.ome.zarr`` is
    taken as the store, not scanned for files inside it.

    A path with no recognised suffix behaves as it always has:

        a directory    scanned, sorted, so deterministic -- and now admits a zarr
                       store alongside files
        anything else  returned as a single leaf
    """
    if _is_input(path):                 # a .zarr store is a directory; a .tif file is not
        if not path.exists():
            raise ConfigError(f"input path not found: {path}")
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if _is_input(p))
        if not files:
            raise ConfigError(f"{path}: no image files ({', '.join(sorted(_INPUT_SUFFIXES))}) found")
        return files
    if not path.exists():
        raise ConfigError(f"input path not found: {path}")
    return [path]


def _duplicate(seen: set[str], field_id: str) -> bool:
    """True if ``field_id`` was already used (and records it otherwise)."""
    if field_id in seen:
        return True
    seen.add(field_id)
    return False


def _ensure_out_dir(out: Path) -> None:
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:      # FileExistsError, NotADirectoryError, PermissionError, ...
        reason = "exists and is not a directory" if isinstance(exc, (FileExistsError, NotADirectoryError)) \
            else (exc.strerror or str(exc))
        raise ConfigError(f"output path {out}: cannot use as output directory ({reason})") from None
    if not os.access(out, os.W_OK | os.X_OK):
        raise ConfigError(f"output path {out}: cannot use as output directory (not writable)")


def _refuse_label_overwrite(labels_dir: Path, files: list[Path], cfg: RunConfig) -> None:
    """Refuse, before any segmentation, if writing would replace existing label files.

    Existing labels may be hand-corrected work -- the napari seam -- and that must
    never be lost silently.

    Field ids are enumerated by reading the inputs, which is cheap next to Cellpose.
    """
    if not labels_dir.is_dir():
        return
    colliding: list[Path] = []
    for field in _iter_fields(files, cfg, []):
        for name in (_label_file(labels_dir, field.field_id, Role.JM1),
                     _label_file(labels_dir, field.field_id, Role.NUCLEUS),
                     labels_dir / f"{field.field_id}.provenance.json"):
            if name.exists():
                colliding.append(name)
    if colliding:
        raise ConfigError(
            f"refusing to overwrite {len(colliding)} existing label file(s) in {labels_dir} "
            f"(e.g. {colliding[0].name}); they may be hand-corrected. Nothing was changed. "
            f"Use --labels {labels_dir} to re-measure the existing labels without re-segmenting, "
            f"or pass --overwrite-labels to replace them")


# ---- labels on disk -------------------------------------------------------------------

def _label_file(directory: Path, field_id: str, role: Role) -> Path:
    return directory / f"{field_id}.{role.value}.npy"


def _save_labels(directory: Path, field: Field, seg: Segmentation, cfg: RunConfig, raw: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cells_path = _label_file(directory, field.field_id, Role.JM1)
    nuc_path = _label_file(directory, field.field_id, Role.NUCLEUS)
    # np.save keeps dtype and values exactly; labels are never renumbered.
    np.save(cells_path, np.asarray(seg.cells), allow_pickle=False)
    np.save(nuc_path, np.asarray(seg.nuclei), allow_pickle=False)
    sidecar = {
        **code_provenance(segmentation=True),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "field_id": field.field_id,
        "source": str(Path(field.source).resolve()),
        "files": {Role.JM1.value: cells_path.name, Role.NUCLEUS.value: nuc_path.name},
        "shape": [int(n) for n in np.shape(seg.cells)],
        "params_used": seg.params_used,
        "config": cfg.model_dump(mode="json"),
        "config_as_given": json.loads(json.dumps(raw, default=str)),
    }
    side = directory / f"{field.field_id}.provenance.json"
    side.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")  # written last


def _load_label(directory: Path, field: Field, role: Role) -> np.ndarray:
    path = _label_file(directory, field.field_id, role)
    if not path.is_file():
        raise SegmentationError(f"no label file {path.name} for field {field.field_id!r} "
                                f"in labels directory {directory}")
    try:
        arr = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise SegmentationError(f"cannot read label file {path}: {_oneline(exc)}") from None
    expected = tuple(np.shape(field.planes[Role.JM1]))
    if arr.dtype.kind not in "iu":
        raise SegmentationError(f"label file {path.name} must hold an integer array, got dtype {arr.dtype}")
    if arr.dtype.kind == "i" and arr.size and int(arr.min()) < 0:
        raise SegmentationError(f"label file {path.name} contains negative label ids "
                                f"(min {int(arr.min())}); labels must be >= 0, 0 = background")
    if arr.shape != expected:
        raise SegmentationError(f"label file {path.name} has shape {arr.shape} but field "
                                f"{field.field_id!r} planes have shape {expected}")
    return arr


def _read_sidecar(directory: Path, field: Field) -> dict | None:
    """The label sidecar as a dict, or None when it is absent or unreadable."""
    side = directory / f"{field.field_id}.provenance.json"
    if not side.is_file():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        typer.echo(f"warning: {side} is unreadable; cannot verify which image these labels came "
                   f"from, or what produced them", err=True)
        return None
    return data if isinstance(data, dict) else None


def _check_sidecar(directory: Path, field: Field, side: dict | None) -> dict | None:
    """Compare the sidecar's recorded source image with this field's. Warn, never refuse."""
    if side is None:
        return None
    recorded = side.get("source")
    here = str(Path(field.source).resolve())
    if recorded is None or str(Path(recorded).resolve()) == here:
        return None
    typer.echo(f"WARNING: labels for field {field.field_id!r} were made from {recorded} but are being "
               f"applied to {here}. If the image was only renamed or moved this is fine; otherwise "
               f"these labels belong to a different image. Recorded in provenance.", err=True)
    return {"field_id": field.field_id, "labels_made_from": recorded, "applied_to": here}


def _load_labels(directory: Path, field: Field) -> Segmentation:
    """Load one field's labels, and the provenance of whatever produced them.

    The sidecar written by ``segment``, or by an editor that keeps it, holds the
    model, its parameters, the device, when the labels were made, and what code
    made them.

    (Reading only its ``source`` field, as this once did, left the supported
    ``--labels`` path unable to say what segmentation is behind a figure.)
    """
    cells = _load_label(directory, field, Role.JM1)
    nuclei = _load_label(directory, field, Role.NUCLEUS)
    params: dict[str, Any] = {"labels_source": "disk", "labels_dir": str(directory)}
    side = _read_sidecar(directory, field)
    if side is None:
        name = f"{field.field_id}.provenance.json"
        why = "unreadable" if (directory / name).exists() else "absent"
        params["model"] = f"unknown: label provenance sidecar {why}"
        params["labels_provenance"] = (
            f"the label provenance sidecar {name} is {why}, so the model, parameters and code "
            f"that produced these labels are not recorded")
    else:
        used = side.get("params_used") if isinstance(side.get("params_used"), dict) else {}
        params["model"] = used.get("model", "not recorded in the label sidecar")
        for k in sorted(used):
            if k != "model":
                params[f"labels_{k}"] = used[k]
        params["labels_created"] = side.get("created", "not recorded")
        params["labels_code_identity"] = side.get("code_identity", "not recorded")
        params["labels_junctionmapper_version"] = side.get("junctionmapper_version", "not recorded")
    mismatch = _check_sidecar(directory, field, side)
    if mismatch:
        params["source_mismatch"] = mismatch
    return Segmentation(cells=cells, nuclei=nuclei, params_used=params)


# ---- per-field driver -----------------------------------------------------------------

@contextlib.contextmanager
def _quiet_bioio(path: Path):
    """bioio logs a warning when it declines a plain TIFF; the reader's fallback then succeeds,
    so for .tif/.tiff the message only misleads. Silenced for the duration of the read only."""
    log = logging.getLogger("bioio.bio_image")
    old = log.level
    if path.suffix.lower() in (".tif", ".tiff"):
        log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        log.setLevel(old)


def _iter_fields(files: list[Path], cfg: RunConfig, failures: list[tuple[str, str]]) -> Iterator[Field]:
    """Yield fields; a file that cannot be read is recorded as a failure and skipped."""
    for f in files:
        if f.suffix.lower() in _OUT_OF_SCOPE_SUFFIXES:
            failures.append((f.name, f"{f}: {f.suffix.lower()} is a vendor format and is out of "
                                     f"scope; convert it to OME-TIFF or OME-Zarr first"))
            continue
        it = read_fields(f, cfg)
        while True:
            try:
                with _quiet_bioio(f):
                    field = next(it)
            except StopIteration:
                break
            except (JunctionMapperError, OSError, tifffile.TiffFileError, ValueError) as exc:
                failures.append((f.name, f"cannot read {f}: {_oneline(exc)}"))
                break
            yield field


def _process(files: list[Path], cfg: RunConfig, work: Callable[[Field], Any]
             ) -> tuple[list[tuple[Field, Any]], list[tuple[str, str]]]:
    """Run ``work`` on every field. A JunctionMapperError fails only that field."""
    ok: list[tuple[Field, Any]] = []
    failures: list[tuple[str, str]] = []
    seen: set[str] = set()
    for field in _iter_fields(files, cfg, failures):
        if _duplicate(seen, field.field_id):
            failures.append((field.field_id, f"duplicate field_id {field.field_id!r} "
                                             f"(from {field.source}); rename or move one input"))
            continue
        try:
            ok.append((field, work(field)))
        except JunctionMapperError as exc:
            failures.append((field.field_id, _oneline(exc)))
    return ok, failures


def _report_outcome(n_ok: int, failures: list[tuple[str, str]]) -> None:
    typer.echo(f"{n_ok} succeeded, {len(failures)} failed", err=True)
    grouped: dict[str, list[str]] = {}
    for who, why in failures:
        grouped.setdefault(why, []).append(who)
    for why, who in grouped.items():
        shown = ", ".join(who[:_MAX_IDS_SHOWN])
        more = f" (+{len(who) - _MAX_IDS_SHOWN} more)" if len(who) > _MAX_IDS_SHOWN else ""
        typer.echo(f"  - {why}", err=True)
        typer.echo(f"      [{len(who)}] {shown}{more}", err=True)


def _finish(n_ok: int, failures: list[tuple[str, str]]) -> None:
    if failures:
        raise typer.Exit(EXIT_FAILED if n_ok == 0 else EXIT_PARTIAL)


def _require_segment_channel(cfg: RunConfig) -> None:
    """Refuse a segmenting run whose config names no ``segment`` channel, before any field is read."""
    if not any(c.role == Role.SEGMENT for c in cfg.channels):
        raise ConfigError(
            "segmentation needs a channel with role 'segment' (a membrane marker that is not JM1 "
            "or JM2); add one to the config, or pass --labels to measure existing labels")


def _snapped(field: Field, seg: Segmentation, params: dict, **extra: Any) -> Segmentation:
    """``seg`` with its cell boundaries snapped onto the segment plane; nuclei untouched."""
    cells = snap_to_membrane(seg.cells, field.planes[Role.SEGMENT], **params)
    return Segmentation(cells=cells, nuclei=seg.nuclei,
                        params_used={**seg.params_used, "snap_to_membrane": params, **extra})


def _seg_params(model: str, diameter: float | None, flow: float, cellprob: float, gpu: bool,
                with_nucleus: bool = False):
    return SegmentationParams(model=model, diameter=diameter, flow_threshold=flow,
                              cellprob_threshold=cellprob, gpu=gpu, with_nucleus=with_nucleus)


def _calibration_files(files: list[Path], n: int = CALIBRATION_FIELDS) -> list[Path]:
    """Up to ``n`` input files spread evenly over ``files`` (which ``_input_files`` sorted).

    This picks which files the calibration pass has to open. The threshold proposal
    itself is over up to ``n`` FIELDS chosen by sorted ``field_id``, in
    ``pipeline.calibration_subset``.

    It is done per FILE because a field id does not exist until the file has been
    read -- the alternative is reading every input twice.

    One file usually holds one field, and then the two selections coincide. A
    multi-scene file contributes all its fields to the subset, which
    ``calibration_subset`` then narrows to ``n``.
    """
    if len(files) <= n:
        return list(files)
    return [files[int(round(i))] for i in np.linspace(0, len(files) - 1, n)]


_CALIBRATION_ATTEMPTS = 2 * CALIBRATION_FIELDS
"""How many files the calibration pass may OPEN before giving up (see ``_calibration_fields``).

Twice the subset size: enough to replace every one of the chosen files once, and small enough
that a run of entirely unreadable inputs is not read twice over before the measurement pass
reports them properly.
"""


def _calibration_candidates(files: list[Path], n: int = CALIBRATION_FIELDS) -> list[Path]:
    """The spread picks first, then the remaining files in sorted order as replacements."""
    chosen = _calibration_files(files, n)
    picked = {str(p) for p in chosen}
    return chosen + [f for f in files if str(f) not in picked]


def _calibration_fields(files: list[Path], cfg: RunConfig,
                        n: int = CALIBRATION_FIELDS) -> tuple[list[Field], str | None]:
    """Fields of up to ``n`` READABLE calibration files, and a note when that was degraded.

    The spread order is walked past a failure to the next candidate, until ``n``
    files have been read or the candidates run out, bounded at
    ``_CALIBRATION_ATTEMPTS`` opens.

    A file counts as read only if it yielded fields and raised nothing, so a
    multi-scene file that dies part way through is skipped whole rather than
    contributing a truncated set.

    Failures are not reported here beyond the note. The measurement pass reads the
    same files and names each one.

    (Why walk past: a file that could not be read used to shrink the calibration
    set silently, and if all ``n`` failed the run reverted to per-field Otsu while
    the report blamed a human override.)
    """
    candidates = _calibration_candidates(files, n)[:_CALIBRATION_ATTEMPTS]
    fields: list[Field] = []
    unreadable: list[str] = []
    used = 0
    for f in candidates:
        failed: list[tuple[str, str]] = []
        got = list(_iter_fields([f], cfg, failed))
        if failed or not got:
            unreadable.append(f.name)
            continue
        fields.extend(got)
        used += 1
        if used >= n:
            break
    if not unreadable:
        return fields, None
    shown = ", ".join(unreadable[:_MAX_IDS_SHOWN])
    more = f" (+{len(unreadable) - _MAX_IDS_SHOWN} more)" if len(unreadable) > _MAX_IDS_SHOWN else ""
    return fields, (
        f"degraded: {len(unreadable)} of the {len(unreadable) + used} calibration file(s) opened could "
        f"not be read cleanly ({shown}{more}), so the proposal used the next candidate(s) in the "
        f"spread order instead; {used} file(s) were read for it")


def _calibrate(files: list[Path], cfg: RunConfig) -> tuple[RunThresholds, str | None]:
    """Propose one threshold per marker role for the whole run, and note any degradation.

    Reads the marker planes of the calibration files only.

    Skipped entirely when the configuration already fixes a threshold for every
    marker role, since then nothing would be proposed.
    """
    roles = [c.role for c in cfg.channels if c.role in (Role.JM1, Role.JM2)]
    if cfg.threshold and all(r in cfg.threshold for r in roles):
        return propose_run_thresholds([], cfg), None
    fields, note = _calibration_fields(files, cfg)
    return propose_run_thresholds(fields, cfg), note


def _merge_params(all_params: list[dict], sidecar_dir: Path) -> dict:
    """Keep the parameters every field agrees on; name the ones that varied per field.

    ``sidecar_dir`` is where the per-field provenance sidecars actually are:

        OUT/labels              when this run segmented
        the --labels directory  when it did not

    (The text used to say ``labels/<field_id>.provenance.json`` unconditionally,
    naming a directory that does not exist on the supported path.)
    """
    if not all_params:
        return {}
    keys = list(all_params[0])
    same = {k: all_params[0][k] for k in keys
            if all(k in p and p[k] == all_params[0][k] for p in all_params)}
    varying = sorted({k for p in all_params for k in p} - set(same))
    if varying:
        same["varies_by_field"] = (", ".join(varying)
                                   + f" (see {sidecar_dir}/<field_id>.provenance.json)")
    return same


# ---- commands -------------------------------------------------------------------------

@app.command()
@_guard
def snap(
    config: Path = ConfigOpt, input: Path = InputOpt, out: Path = OutOpt,
    labels: Path = typer.Option(..., "--labels", help="Directory of label files to snap."),
    erode_px: int = typer.Option(ERODE_PX, help="How far (px) a boundary may move."),
    sigma_px: float = typer.Option(SIGMA_PX, help="Smoothing (px) of the membrane plane."),
    smooth_px: float = typer.Option(SMOOTH_PX, help="Smoothing (px) of the snapped boundaries."),
    overwrite_labels: bool = typer.Option(False, "--overwrite-labels", help="Replace existing label "
                                          "files in OUT/labels (default: refuse)."),
) -> None:
    """Move existing cell-cell boundaries onto the membrane ridge; write labels to OUT/labels.

    The membrane is the config's segment channel. Foreground, nuclei and cell ids are
    unchanged. Measure the result with `run --labels OUT/labels`.

    \b
    Exit codes: 0 every field succeeded; 1 command failed, or every field failed
    (nothing written); 3 PARTIAL - some fields failed and the rest were written;
    2 usage error. Do not treat 3 as success.
    """
    cfg, raw = _load_config(config)
    _require_segment_channel(cfg)
    files = _input_files(input)
    if not labels.is_dir():
        raise ConfigError(f"labels directory not found: {labels}")
    labels = labels.resolve()
    labels_dir = out / "labels"
    if labels_dir.resolve() == labels:
        raise ConfigError("snap would overwrite its own input labels; choose a different --out")
    _ensure_out_dir(out)
    if not overwrite_labels:
        _refuse_label_overwrite(labels_dir, files, cfg)
    params = snap_params(erode_px, sigma_px, smooth_px)

    def work(field: Field) -> None:
        seg = _load_labels(labels, field)
        side = _read_sidecar(labels, field) or {}
        used = side.get("params_used") if isinstance(side.get("params_used"), dict) else {}
        seg = Segmentation(cells=seg.cells, nuclei=seg.nuclei, params_used=dict(used))
        _save_labels(labels_dir, field, _snapped(field, seg, params, snapped_from=str(labels)),
                     cfg, raw)

    ok, failures = _process(files, cfg, work)
    _report_outcome(len(ok), failures)
    if ok:
        typer.echo(f"snapped labels written to {labels_dir}")
    elif not failures:
        raise ConfigError(f"no fields found in {input}")
    _finish(len(ok), failures)


@app.command()
@_guard
def segment(
    config: Path = ConfigOpt, input: Path = InputOpt, out: Path = OutOpt,
    model: str = typer.Option("cpsam", help="Cellpose model."),
    diameter: float = typer.Option(None, help=_DIAMETER_HELP),
    flow_threshold: float = typer.Option(0.4),
    cellprob_threshold: float = typer.Option(0.0),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use a GPU if one is available."),
    with_nucleus: bool = typer.Option(False, "--with-nucleus", help=_WITH_NUCLEUS_HELP),
    snap: bool = typer.Option(False, "--snap-to-membrane", help=_SNAP_HELP),
    overwrite_labels: bool = typer.Option(False, "--overwrite-labels", help="Replace existing label "
                                          "files in OUT/labels (default: refuse)."),
) -> None:
    """Segment cells and nuclei and write label files to OUT/labels; measure nothing.

    \b
    Exit codes: 0 every field succeeded; 1 command failed, or every field failed
    (nothing written); 3 PARTIAL - some fields failed and the rest were written;
    2 usage error. Do not treat 3 as success.
    """
    cfg, raw = _load_config(config)
    _require_segment_channel(cfg)
    files = _input_files(input)
    params = _seg_params(model, diameter, flow_threshold, cellprob_threshold, gpu,
                         with_nucleus)
    labels_dir = out / "labels"
    _ensure_out_dir(out)
    if not overwrite_labels:
        _refuse_label_overwrite(labels_dir, files, cfg)

    def work(field: Field) -> None:
        seg = segment_field(field, params)
        if snap:
            seg = _snapped(field, seg, snap_params())
        _save_labels(labels_dir, field, seg, cfg, raw)

    ok, failures = _process(files, cfg, work)
    _report_outcome(len(ok), failures)
    if ok:
        typer.echo(f"labels written to {labels_dir}")
    elif not failures:
        raise ConfigError(f"no fields found in {input}")
    _finish(len(ok), failures)


@app.command()
@_guard
def run(
    config: Path = ConfigOpt, input: Path = InputOpt, out: Path = OutOpt,
    labels: Path = typer.Option(None, "--labels", help="Directory of label files written by "
                                "`segment` or an editor; skips Cellpose entirely."),
    model: str = typer.Option("cpsam", help="Cellpose model."),
    diameter: float = typer.Option(None, help=_DIAMETER_HELP),
    flow_threshold: float = typer.Option(0.4),
    cellprob_threshold: float = typer.Option(0.0),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use a GPU if one is available."),
    with_nucleus: bool = typer.Option(False, "--with-nucleus", help=_WITH_NUCLEUS_HELP),
    snap: bool = typer.Option(False, "--snap-to-membrane", help=_SNAP_HELP),
    overwrite_labels: bool = typer.Option(False, "--overwrite-labels", help="Replace existing label "
                                          "files in OUT/labels (default: refuse)."),
) -> None:
    """Segment (or load labels), measure, write the five tables and report.html to OUT.

    \b
    Exit codes: 0 every field succeeded; 1 command failed, or every field failed
    (nothing written); 3 PARTIAL - some fields failed and the rest were written;
    2 usage error. Do not treat 3 as success.
    """
    cfg, raw = _load_config(config)
    if labels is None:
        _require_segment_channel(cfg)
    files = _input_files(input)
    if labels is not None and not labels.is_dir():
        raise ConfigError(f"labels directory not found: {labels}")
    if labels is not None and snap:
        raise ConfigError("--snap-to-membrane applies to a fresh segmentation; to snap existing "
                          "labels, run `junctionmapper snap` and pass its labels to --labels")
    if labels is not None:
        labels = labels.resolve()
    _ensure_out_dir(out)
    if labels is None and not overwrite_labels:
        _refuse_label_overwrite(out / "labels", files, cfg)
    params = _seg_params(model, diameter, flow_threshold, cellprob_threshold, gpu,
                         with_nucleus)

    run_t, calib_note = _calibrate(files, cfg)

    def work(field: Field) -> tuple[FieldResult, dict, dict]:
        if labels is not None:
            seg = _load_labels(labels, field)
        else:
            seg = segment_field(field, params)
            if snap:
                seg = _snapped(field, seg, snap_params())
            _save_labels(out / "labels", field, seg, cfg, raw)
        result = measure_field(field, seg, cfg, run_thresholds=run_t)
        thresholds = {r.value: t for r, t in resolve_thresholds(field, cfg, run_t).items()}
        return result, seg.params_used, thresholds

    ok, failures = _process(files, cfg, work)
    _report_outcome(len(ok), failures)
    if not ok:
        typer.echo("every field failed: no tables or report were written" if failures
                   else f"no fields found in {input}", err=True)
        raise typer.Exit(EXIT_FAILED)

    results = [w[0] for _, w in ok]
    used = _merge_params([w[1] for _, w in ok], labels if labels is not None else out / "labels")
    prov: dict[str, Any] = {
        **code_provenance(segmentation=labels is None),
        "params_used": used,
        "thresholds": json.dumps({f.field_id: w[2] for f, w in ok}, sort_keys=True),
        "threshold_proposal": run_t.describe(),
        # Which roles a single run-level value was carried for. The report needs it to say why
        # the applied threshold varied: a per-field override, or a proposal that resolved
        # nothing (every calibration file unreadable) and left each field on its own Otsu.
        "threshold_run_roles": ", ".join(sorted(r.value for r in run_t.applied)) or "none",
        "config": cfg.model_dump_json(),
        "input": str(input),
    }
    if calib_note:
        prov["threshold_calibration"] = calib_note
    no_side = [f.field_id for f, w in ok if "labels_provenance" in w[1]]
    if no_side:
        prov["labels_without_provenance"] = (
            f"{len(no_side)} field(s) whose labels carry no provenance sidecar: "
            f"{', '.join(no_side[:_MAX_IDS_SHOWN])}"
            + (f" (+{len(no_side) - _MAX_IDS_SHOWN} more)" if len(no_side) > _MAX_IDS_SHOWN else ""))
    mism = [w[1]["source_mismatch"] for _, w in ok if "source_mismatch" in w[1]]
    if mism:
        prov["labels_source_mismatch"] = json.dumps(mism, sort_keys=True)
    if failures:
        prov["fields_failed"] = str(len(failures))
        prov["fields_failed_detail"] = "; ".join(f"{who}: {why}" for who, why in failures)
    write_tables(results, out)
    build_report(results, cfg, prov, out / "report.html")
    typer.echo(f"wrote tables and report.html to {out}")
    _finish(len(ok), failures)


if __name__ == "__main__":  # pragma: no cover
    app()
