"""I3: a figure has to be traceable to the code that made it.

The branch stamped a constant ``0.1.0.dev0`` on every one of its 49 commits and recorded no
commit and no library versions, so two runs of different code were indistinguishable.
"""
from pathlib import Path

import pytest

from junctionmapper import __version__
from junctionmapper import provenance as pv


#: A resolved identity STARTS with one of these. Substring matching is not enough: the
#: total-failure string "unresolvable: no git checkout and no installed distribution; package
#: says 0.1.0.dev0" contains "installed distribution", so `"installed distribution" in ident`
#: was satisfied by the one string that means nothing was identified at all.
RESOLVED_PREFIXES = ("git ", "installed distribution ")


def test_code_identity_is_not_just_the_static_version_string():
    ident = pv.code_identity()
    assert ident != __version__ and "0.1.0.dev0" != ident
    assert ident.startswith(RESOLVED_PREFIXES), ident
    assert not ident.startswith("unresolvable"), ident


def test_an_identity_with_no_commit_and_no_distribution_is_not_mistaken_for_resolved(monkeypatch):
    """The negative assertion above is only meaningful if the degraded string is reachable."""
    monkeypatch.setattr(pv, "git_commit", lambda *a, **k: None)
    monkeypatch.setattr(pv, "distribution_version", lambda: None)
    ident = pv.code_identity()
    assert ident.startswith("unresolvable")
    assert not ident.startswith(RESOLVED_PREFIXES), ident
    assert "installed distribution" in ident       # the substring that used to pass the pin


def test_the_commit_is_recovered_without_the_git_binary(monkeypatch):
    """The shipped container installs no git, which is exactly the supported path, so the sha
    must come off the filesystem; only the dirty check needs the binary."""
    monkeypatch.setattr(pv, "_git", lambda args, cwd: None)     # as if git were absent
    sha, dirty = pv.git_state()
    assert sha and len(sha) == 12 and all(c in "0123456789abcdef" for c in sha)
    assert dirty is None                                        # unknowable without the binary
    assert pv.git_commit().endswith("-dirty?")


@pytest.mark.parametrize("head,extra", [
    ("ref: refs/heads/main", {"refs/heads/main": "a" * 40}),
    ("ref: refs/heads/packed", {"packed-refs": "# pack\n" + "b" * 40 + " refs/heads/packed\n"}),
    ("c" * 40, {}),                                             # detached HEAD
])
def test_head_sha_handles_loose_packed_and_detached_refs(tmp_path, head, extra):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head)
    for name, text in extra.items():
        (git / name).write_text(text)
    sha = pv._head_sha(git)
    assert sha and len(sha) == 40


def test_outside_a_checkout_there_is_no_commit_but_still_an_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "_git", lambda args, cwd: None)
    assert pv.git_state(tmp_path) == (None, None)
    assert pv.git_commit(tmp_path) is None
    monkeypatch.setattr(pv, "git_commit", lambda *a, **k: None)
    monkeypatch.setattr(pv, "distribution_version", lambda: None)
    assert "unresolvable" in pv.code_identity()                 # says so, never silently "fine"


def test_a_git_file_worktree_is_followed(tmp_path):
    real = tmp_path / "real.git"
    (real / "refs" / "heads").mkdir(parents=True)
    (real / "HEAD").write_text("ref: refs/heads/wt")
    (real / "refs" / "heads" / "wt").write_text("d" * 40)
    work = tmp_path / "wt"
    work.mkdir()
    (work / ".git").write_text(f"gitdir: {real}\n")
    assert pv._git_dir(work) == real
    assert pv._head_sha(real) == "d" * 40


def test_library_versions_cover_what_can_move_a_number():
    core = pv.library_versions()
    assert set(core) == set(pv.CORE_LIBRARIES)
    assert core["numpy"][0].isdigit() and core["scikit-image"][0].isdigit()
    assert "cellpose" not in core                               # segmentation did not run
    seg = pv.library_versions(segmentation=True)
    assert "cellpose" in seg and "torch" in seg                 # it did
    assert all(v for v in seg.values())                         # never blank; "not installed"


def test_code_provenance_block_has_the_three_keys_a_report_needs():
    block = pv.code_provenance()
    assert set(block) == {"junctionmapper_version", "code_identity", "library_versions"}
    assert "numpy" in block["library_versions"]


def test_a_broken_git_checkout_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "_git", lambda args, cwd: None)
    (tmp_path / ".git").mkdir()
    assert pv.git_state(tmp_path) == (None, None)               # no HEAD at all
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/gone")
    assert pv.git_state(tmp_path) == (None, None)               # ref missing, no packed-refs
