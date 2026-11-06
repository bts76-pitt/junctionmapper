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
