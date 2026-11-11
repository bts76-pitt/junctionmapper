"""What code produced a figure.

A figure that cannot be traced to the code behind it is not reproducible.

A static ``__version__`` string does not identify code: every commit of this
branch stamps the same ``0.1.0.dev0``.

What does identify it:

    the commit                        when the working tree is a checkout
    the installed distribution        when it is not
    version

plus the versions of the libraries whose numbers can move a result.
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
    """Run git, or None when there is no usable git here (the container has no git binary)."""
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                             timeout=_GIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None                     # no git binary, or it misbehaved
    if out.returncode != 0:
        return None                     # not a repository, or no commit yet
    return out.stdout.strip()


def _git_dir(start: Path) -> Path | None:
    """The ``.git`` directory above ``start``, resolving a ``.git`` file (a worktree)."""
    for d in (start, *start.parents):
        g = d / ".git"
        if g.is_dir():
            return g
        if g.is_file():
            try:
                line = g.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            if line.startswith("gitdir:"):
                p = Path(line.split(":", 1)[1].strip())
                return p if p.is_absolute() else (d / p)
    return None


def _head_sha(git_dir: Path) -> str | None:
    """HEAD's commit sha read straight off the filesystem, with no git binary.

    The shipped container installs no git -- its %post is pure pip, on purpose --
    so the binary is exactly unavailable on the path this project actually runs.

    Reading .git is what makes the commit recoverable there, which is the whole
    point of recording it.
    """
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None                             # detached HEAD holds the sha itself
    ref = head.split(":", 1)[1].strip()
    try:
        return (git_dir / ref).read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    try:                                                # packed refs
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha
    except OSError:
        return None
    return None


def git_state(start: Path | None = None) -> tuple[str | None, bool | None]:
    """``(short sha, dirty)``. ``dirty`` is None when it cannot be determined."""
    cwd = Path(start) if start is not None else Path(__file__).resolve().parent
    git_dir = _git_dir(cwd)
    sha = _head_sha(git_dir) if git_dir else None
    if sha is None:
        sha = _git(["rev-parse", "HEAD"], cwd)
    if not sha:
        return None, None
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    dirty = None if status is None else bool(status)
    return sha[:12], dirty


def git_commit(start: Path | None = None) -> str | None:
    """``<short sha>``, with ``-dirty`` appended when the working tree is known to be dirty
    and ``-dirty?`` when that cannot be checked (no git binary). None outside a checkout."""
    sha, dirty = git_state(start)
    if sha is None:
        return None
    return sha if dirty is False else f"{sha}-dirty" if dirty else f"{sha}-dirty?"


def distribution_version() -> str | None:
    try:
        return metadata.version("junctionmapper")
    except metadata.PackageNotFoundError:
        return None


def code_identity() -> str:
    """One resolvable string: the commit when there is one, else the installed version.

    The container runs the package off ``PYTHONPATH`` without installing it, so the
    distribution version is frequently absent. The commit is what survives there.

    Suffixes:

        -dirty     uncommitted changes
        -dirty?    no git binary was available to check (the image has none), so
                   the commit identifies the checkout but not the edits in it
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
