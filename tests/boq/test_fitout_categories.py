"""The three fit-out trades of part 2c: which BOQ lines they claim, and which they leave alone."""

import csv

import pytest
from openpyxl import load_workbook

from secure_rag.boq.quotes import line_category
from secure_rag.core import config

# Category of every sample line before part 2c: only glass block and GRC lines had one.
_ISSUES_CATEGORIES = {"1.4": "glass_block", "2.2": "grc"}


@pytest.mark.parametrize(
    "description, category",
    [
        ("Hardwood joinery door frame", "joinery"),
        ("Aluminium sliding window, double glazed", "aluminium_glazing"),
        ("Polished marble floor tiles", "stone_cladding"),
        ("Granite cladding to entrance facade", "stone_cladding"),
        # Words the tests and the catalogue use stay in no category.
        ("Bronze door handle", ""),
        ("Teak bench seat", ""),
        ("Timber window board", ""),
        ("Bitumen waterproofing membrane", ""),
        # The first trade that fits wins: the two old ones come first.
        ("Aluminium framed glass block screen", "glass_block"),
        ("GRC panel with granite finish", "grc"),
    ],
)
def test_fitout_line_category(description, category):
    assert line_category(description) == category


def test_the_old_samples_keep_their_categories():
    with (config.SAMPLES_DIR / "boq_with_issues.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    found = {row["Item"]: line_category(row["Description"]) for row in rows if line_category(row["Description"])}
    assert found == _ISSUES_CATEGORIES

    sheet = load_workbook(config.SAMPLES_DIR / "boq_villa_blockwork.xlsx").active
    descriptions = [row[1] for row in sheet.iter_rows(values_only=True) if len(row) > 1 and isinstance(row[1], str)]
    assert descriptions and all(line_category(text) == "" for text in descriptions)
