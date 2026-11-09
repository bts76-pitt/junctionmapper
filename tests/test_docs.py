"""The root README must not claim more than tests/validation/ can demonstrate.

C3: the README claimed "every metric in this project matches the original exactly - verified
on all 148 interfaces", while tests/validation/README.md said the opposite ("Few rows are
exactly comparable ... 416 rows of agreement is therefore not 416 exact comparisons"). The
148-interface driver is not in the repository, so no reader could check either statement. These
tests are cheap and they fail the moment the two documents disagree again.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
VALIDATION = (ROOT / "tests" / "validation" / "README.md").read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_readme_does_not_claim_exactness_on_every_interface():
    flat = _flat(README).lower()
    assert "every metric in this project matches the original exactly" not in flat
    assert "verified on all 148 interfaces" not in flat
    # divergence 3 makes any blanket "every metric" claim false
    assert "recorded and never\nasserted" in README or "recorded and never asserted" in _flat(README)


@pytest.mark.parametrize("number", ["208", "416", "20 of 416", "15%", "16 px", "0.35"])
def test_every_validation_number_in_the_readme_comes_from_the_validation_readme(number):
    assert number in _flat(README), number
    assert number.replace("20 of 416", "20 of 416") in _flat(VALIDATION), number


def test_the_readme_attributes_the_uncommitted_148_interface_run_as_such():
    flat = _flat(README)
    if "148" in flat:
        m = re.search(r"[^.]*148[^.]*\.[^.]*\.[^.]*\.", flat)
        assert m and "not committed" in m.group(0), m.group(0) if m else "no sentence found"


def test_the_readme_records_the_click_dependence_of_the_java_corner_count():
    flat = _flat(README)
    assert "click-dependent" in flat
    assert "seed_row" in flat and "Why the click matters" in flat
    # the old unconditional claim is gone
    assert "A plain rectangle yields **three** corners rather than four" not in flat


# ---- claims about the threshold machinery ---------------------------------------------

DEF = (ROOT / "container" / "junctionmapper.def").read_text(encoding="utf-8")
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_the_readme_says_the_deviation_flag_can_fire_on_a_default_low_valued_run():
    """N9: "the default path does not trip the flag" is not true for a low-valued plane, where
    the one-grey-level floor dominates. The behaviour is correct; the claim was not."""
    flat = _flat(README)
    assert "low-valued" in flat.lower()
    assert "the flag sets on the outer fields of an ordinary *default* run" in flat
    assert "informational" in flat


def test_the_readme_says_a_zarr_store_is_a_directory_and_a_valid_input():
    flat = _flat(README)
    assert "An OME-Zarr store is a *directory* whose name ends `.zarr`" in flat
    assert "taken as one input in its own right" in flat


def test_the_container_def_does_not_claim_more_than_its_pins_give():
    """N10: the 18 direct pins do match the live image, but transitive dependencies are
    unpinned, so a rebuild can drift and "reconstructible" overstated it."""
    flat = _flat(DEF)
    assert "reconstructible" not in flat
    for name in ("zarr", "dask", "numcodecs", "bioio-base"):
        assert name in flat, name


def test_zarr_is_declared_as_a_test_dependency_not_only_present_transitively():
    """N10: tests/test_reader.py imports zarr and the N1 tests add more, so it cannot stay an
    undeclared transitive dependency of bioio-ome-zarr."""
    assert "zarr>=" in PYPROJECT.replace("bioio-ome-zarr>=", "")
    assert '"zarr==' in DEF


def test_the_readme_does_not_claim_a_junction_is_always_two_rows():
    """The README said a junction "is therefore measured twice, once from each side".

    True for tcj, false for dp: Douglas-Peucker corners are not neighbour-aware, so one run
    of contact is split wherever a corner falls. On a real Cellpose run, dp gave a median of
    3 rows per pair_id and a maximum of 10, while tcj gave at most 2. The factor-of-two
    claim would make anyone pooling dp rows weight junctions wrongly.
    """
    flat = _flat(README)
    assert "measured twice, once from each side" not in flat
    assert "junctions therefore count twice" not in flat
    # and it must say what is actually true
    assert "pair_id" in flat
    assert "never\nby counting rows" in README or "never by counting rows" in flat
