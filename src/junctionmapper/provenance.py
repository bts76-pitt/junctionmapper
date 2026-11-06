"""What code produced a figure.

A figure that cannot be traced to the code behind it is not reproducible, and a static
``__version__`` string does not identify code: every commit of this branch stamps the same
``0.1.0.dev0``. What identifies it is the commit (when the working tree is a checkout) or the
installed distribution version (when it is not), plus the versions of the libraries whose
numbers can move a result.
"""
from __future__ import annotations

import subprocess
from importlib import metadata
from pathlib import Path

from junctionmapper import __version__

#: Libraries whose version can change a measured number. scikit-image supplies Otsu and the
#: morphology, scipy the labelling and distance transforms, numpy the arithmetic, pandas the
#: tables and matplotlib the figures.
CORE_LIBRARIES = ("numpy", "scipy", "scikit-image", "pandas", "matplotlib", "pyarrow")
#: Added when segmentation actually ran: they decide the labels.
SEGMENTATION_LIBRARIES = ("cellpose", "torch")

_GIT_TIMEOUT_S = 5


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                             timeout=_GIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None                     # no git binary, or it misbehaved
    if out.returncode != 0:
        return None                     # not a repository, or no commit yet
    return out.stdout.strip()


def git_commit(start: Path | None = None) -> str | None:
    """``<short sha>`` for the checkout holding this package, ``<short sha>-dirty`` when the
    working tree has uncommitted changes, or None when it is not a git checkout."""
    cwd = Path(start) if start is not None else Path(__file__).resolve().parent
    sha = _git(["rev-parse", "--short", "HEAD"], cwd)
    if not sha:
        return None
    dirty = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    return f"{sha}-dirty" if dirty else sha


def distribution_version() -> str | None:
    try:
        return metadata.version("junctionmapper")
    except metadata.PackageNotFoundError:
        return None


def code_identity() -> str:
    """One resolvable string: the commit when there is one, else the installed version.

    The container runs the package off ``PYTHONPATH`` without installing it, so the
    distribution version is frequently absent; the commit is what survives there.
    """
    sha = git_commit()
    dist = distribution_version()
    if sha and dist:
        return f"git {sha} (installed distribution {dist})"
    if sha:
        return f"git {sha} (not installed as a distribution; run from the source tree)"
    if dist:
        return f"installed distribution {dist} (no git checkout)"
    return f"unresolvable: no git checkout and no installed distribution; package says {__version__}"


def library_versions(segmentation: bool = False) -> dict[str, str]:
    """Versions of the libraries that can move a number, ``"not installed"`` when absent."""
    names = CORE_LIBRARIES + (SEGMENTATION_LIBRARIES if segmentation else ())
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def code_provenance(segmentation: bool = False) -> dict[str, str]:
    """The provenance block both the report and the label sidecar carry."""
    return {
        "junctionmapper_version": __version__ or "unknown",
        "code_identity": code_identity(),
        "library_versions": "; ".join(f"{k} {v}" for k, v in library_versions(segmentation).items()),
    }
