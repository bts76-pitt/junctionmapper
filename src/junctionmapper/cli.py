"""Command line interface: ``junctionmapper run`` and ``junctionmapper segment``.

Error policy
------------
Everything the user can cause (bad config, bad input, bad labels, missing optional
dependency) surfaces as a ``JunctionMapperError``. Each command catches that, prints one
clean line to stderr and exits non-zero. Unexpected exceptions are real bugs and propagate.

Exit codes
----------
0  every field succeeded
1  the command failed: bad config / input, or every field failed (nothing was written)
2  usage error (missing or unknown option; raised by typer itself)
3  partial: some fields failed, the successful fields' outputs were written

The labels directory
--------------------
``segment`` writes, per field, ``<field_id>.JM1.npy`` (cell labels), ``<field_id>.nucleus.npy``
(nucleus labels) and ``<field_id>.provenance.json``. ``run --labels DIR`` reads the ``.npy``
files back unchanged. This directory is the seam a human-in-the-loop editor writes to.
"""
from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import typer
import yaml
from pydantic import ValidationError

from junctionmapper.config import Role, RunConfig
from junctionmapper.errors import ConfigError, JunctionMapperError, SegmentationError
from junctionmapper.io.reader import Field, read_fields
from junctionmapper.pipeline import FieldResult, measure_field, resolve_thresholds
from junctionmapper.report.build import build_report
from junctionmapper.segment.cellpose import Segmentation, SegmentationParams, segment_field
from junctionmapper.tables import write_tables

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Cell-cell junction quantification (Junction Mapper, in Python).")

EXIT_FAILED, EXIT_PARTIAL = 1, 3
_INPUT_SUFFIXES = {".tif", ".tiff", ".czi", ".nd2", ".lif"}
_MAX_IDS_SHOWN = 5

ConfigOpt = typer.Option(..., "--config", "-c", help="YAML run configuration.")
InputOpt = typer.Option(..., "--input", "-i", help="Image file, or a directory of image files.")
OutOpt = typer.Option(..., "--out", "-o", help="Output directory (created if absent).")


# ---- small helpers -------------------------------------------------------------------

def _version() -> str:
    try:
        return metadata.version("junctionmapper")
    except metadata.PackageNotFoundError:
        return "unknown"


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


def _input_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _INPUT_SUFFIXES)
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
        "junctionmapper_version": _version(),
        "field_id": field.field_id,
        "source": str(field.source),
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
    if arr.shape != expected:
        raise SegmentationError(f"label file {path.name} has shape {arr.shape} but field "
                                f"{field.field_id!r} planes have shape {expected}")
    return arr


def _load_labels(directory: Path, field: Field) -> Segmentation:
    cells = _load_label(directory, field, Role.JM1)
    nuclei = _load_label(directory, field, Role.NUCLEUS)
    return Segmentation(cells=cells, nuclei=nuclei,
                        params_used={"labels_source": "disk", "labels_dir": str(directory)})


# ---- per-field driver -----------------------------------------------------------------

def _iter_fields(files: list[Path], cfg: RunConfig, failures: list[tuple[str, str]]) -> Iterator[Field]:
    """Yield fields; a file that cannot be read is recorded as a failure and skipped."""
    for f in files:
        it = read_fields(f, cfg)
        while True:
            try:
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
    typer.echo(f"{n_ok} succeeded, {len(failures)} failed")
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


def _seg_params(model: str, diameter: float | None, flow: float, cellprob: float, gpu: bool):
    return SegmentationParams(model=model, diameter=diameter, flow_threshold=flow,
                              cellprob_threshold=cellprob, gpu=gpu)


def _merge_params(all_params: list[dict]) -> dict:
    """Keep the parameters every field agrees on; name the ones that varied per field."""
    if not all_params:
        return {}
    keys = list(all_params[0])
    same = {k: all_params[0][k] for k in keys
            if all(k in p and p[k] == all_params[0][k] for p in all_params)}
    varying = sorted({k for p in all_params for k in p} - set(same))
    if varying:
        same["varies_by_field"] = ", ".join(varying) + " (see labels/<field_id>.provenance.json)"
    return same


# ---- commands -------------------------------------------------------------------------

@app.command()
@_guard
def segment(
    config: Path = ConfigOpt, input: Path = InputOpt, out: Path = OutOpt,
    model: str = typer.Option("cpsam", help="Cellpose model."),
    diameter: float = typer.Option(None, help="Cell diameter in px (default: Cellpose decides)."),
    flow_threshold: float = typer.Option(0.4),
    cellprob_threshold: float = typer.Option(0.0),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use a GPU if one is available."),
) -> None:
    """Segment cells and nuclei and write label files to OUT/labels; measure nothing."""
    cfg, raw = _load_config(config)
    files = _input_files(input)
    params = _seg_params(model, diameter, flow_threshold, cellprob_threshold, gpu)
    labels_dir = out / "labels"

    def work(field: Field) -> None:
        _save_labels(labels_dir, field, segment_field(field, params), cfg, raw)

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
    diameter: float = typer.Option(None, help="Cell diameter in px (default: Cellpose decides)."),
    flow_threshold: float = typer.Option(0.4),
    cellprob_threshold: float = typer.Option(0.0),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use a GPU if one is available."),
) -> None:
    """Segment (or load labels), measure, write the five tables and report.html to OUT."""
    cfg, raw = _load_config(config)
    files = _input_files(input)
    if labels is not None and not labels.is_dir():
        raise ConfigError(f"labels directory not found: {labels}")
    params = _seg_params(model, diameter, flow_threshold, cellprob_threshold, gpu)

    def work(field: Field) -> tuple[FieldResult, dict, dict]:
        if labels is not None:
            seg = _load_labels(labels, field)
        else:
            seg = segment_field(field, params)
            _save_labels(out / "labels", field, seg, cfg, raw)
        result = measure_field(field, seg, cfg)
        thresholds = {r.value: t for r, t in resolve_thresholds(field, cfg).items()}
        return result, seg.params_used, thresholds

    ok, failures = _process(files, cfg, work)
    _report_outcome(len(ok), failures)
    if not ok:
        typer.echo("every field failed: no tables or report were written" if failures
                   else f"no fields found in {input}", err=True)
        raise typer.Exit(EXIT_FAILED)

    results = [w[0] for _, w in ok]
    if labels is not None:
        used = {"model": "labels loaded from disk (no segmentation run)",
                "labels_source": "disk", "labels_dir": str(labels)}
    else:
        used = _merge_params([w[1] for _, w in ok])
    prov: dict[str, Any] = {
        "junctionmapper_version": _version(),
        "params_used": used,
        "thresholds": json.dumps({f.field_id: w[2] for f, w in ok}, sort_keys=True),
        "config": cfg.model_dump_json(),
        "input": str(input),
    }
    if failures:
        prov["fields_failed"] = str(len(failures))
    write_tables(results, out)
    build_report(results, cfg, prov, out / "report.html")
    typer.echo(f"wrote tables and report.html to {out}")
    _finish(len(ok), failures)


if __name__ == "__main__":  # pragma: no cover
    app()
