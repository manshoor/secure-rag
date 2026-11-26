"""The invented vendor master: the three fit-out trades of part 2c, and the rules every record keeps.

    trade               approved           not approved
    joinery             V-JNR-1..5         V-JNR-9 (licence expired)
    aluminium_glazing   V-ALG-1..5         V-ALG-9
    stone_cladding      V-STN-1..5         V-STN-9

Per trade, of the five approved: every licence valid on both test dates, exactly
one expiring within 60 days of both, and reply records of all three kinds.
"""

import copy
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from secure_rag.core import config
from secure_rag.workflow import systems
from secure_rag.workflow.systems import ReplyRecord

_FILE = config.SYSTEMS_DIR / "vendors.json"
_SRC = Path(systems.__file__).resolve().parents[1]
_COMPANY = "alpha-construction"
_TRADES = {"JNR": "joinery", "ALG": "aluminium_glazing", "STN": "stone_cladding"}
_OLD_IDS = ["V-GLB-1", "V-GLB-2", "V-GLB-3", "V-GRC-1", "V-GRC-2", "V-GRC-3", "V-GRC-9"]
# The test date and the day this data was written: both must see the same picture.
_DATES = (date(2026, 11, 18), date(2026, 11, 26))
_EXPIRING_DAYS = 60
# Placeholder bank accounts: country code, then check digits 00, never valid.
_PLACEHOLDER_BANK = re.compile(r"^AE00\d{19}$")


def _rows() -> list[dict]:
    return json.loads(_FILE.read_text(encoding="utf-8"))["vendors"]


def _new_rows(prefix: str) -> list[dict]:
    return [row for row in _rows() if row["id"].startswith(f"V-{prefix}-")]


def test_every_vendor_has_one_category():
    """_receive takes the first package that lists a sender: a vendor in two packages breaks quote receipt."""
    for row in _rows():
        assert len(row["categories"]) == 1, row["id"]


def test_ids_are_unique_and_the_old_records_stay_first():
    ids = [row["id"] for row in _rows()]
    assert len(set(ids)) == len(ids)
    assert ids[: len(_OLD_IDS)] == _OLD_IDS
    assert all("reply_record" not in row for row in _rows()[: len(_OLD_IDS)])


@pytest.mark.parametrize("prefix", sorted(_TRADES))
def test_each_new_trade_has_five_approved_and_one_not(prefix):
    rows = _new_rows(prefix)
    assert [row["id"] for row in rows] == [f"V-{prefix}-{n}" for n in (1, 2, 3, 4, 5, 9)]
    for row in rows:
        assert row["company"] == _COMPANY and row["categories"] == [_TRADES[prefix]]
        assert row["approved"] is (not row["id"].endswith("-9"))


@pytest.mark.parametrize("prefix", sorted(_TRADES))
@pytest.mark.parametrize("today", _DATES)
def test_licences_of_a_new_trade(prefix, today):
    rows = _new_rows(prefix)
    approved = [row for row in rows if row["approved"]]
    expiry = {row["id"]: date.fromisoformat(row["licence_expiry"]) for row in rows}

    # Every approved licence valid; exactly one runs out within 60 days.
    assert all(expiry[row["id"]] >= today for row in approved)
    soon = [row["id"] for row in approved if expiry[row["id"]] - today <= timedelta(days=_EXPIRING_DAYS)]
    assert len(soon) == 1

    # The record that is not approved has an expired licence.
    assert expiry[f"V-{prefix}-9"] < today


@pytest.mark.parametrize("prefix", sorted(_TRADES))
def test_reply_records_are_mixed(prefix):
    records = {row["reply_record"] for row in _new_rows(prefix) if row["approved"]}
    assert records == {record.value for record in ReplyRecord}


def test_new_records_use_reserved_emails_and_placeholder_banks():
    rows = [row for prefix in _TRADES for row in _new_rows(prefix)]
    assert len(rows) == 18
    for row in rows:
        assert row["email"].endswith(".invalid") and _PLACEHOLDER_BANK.match(row["bank_account"]), row["id"]
    names = [row["name"] for row in _rows()]
    assert len(set(names)) == len(names)
    banks = [row["bank_account"] for row in _rows()]
    assert len(set(banks)) == len(banks)


def test_a_vendor_reads_its_reply_record_and_absent_means_new():
    by_id = {vendor.vendor_id: vendor for vendor in systems.vendors()}
    assert all(by_id[vendor_id].reply_record is ReplyRecord.NEW for vendor_id in _OLD_IDS)
    for row in _rows():
        if "reply_record" in row:
            assert by_id[row["id"]].reply_record is ReplyRecord(row["reply_record"])


@pytest.mark.parametrize("prefix, category", sorted(_TRADES.items()))
def test_vendors_for_a_new_trade(prefix, category):
    assert [vendor.vendor_id for vendor in systems.vendors_for(category, _COMPANY)] == [f"V-{prefix}-{n}" for n in range(1, 6)]
    assert systems.vendors_for(category, "beta-properties") == []


def test_the_old_trades_keep_their_vendors():
    assert [vendor.vendor_id for vendor in systems.vendors_for("glass_block", _COMPANY)] == ["V-GLB-1", "V-GLB-2", "V-GLB-3"]
    assert [vendor.vendor_id for vendor in systems.vendors_for("grc", _COMPANY)] == ["V-GRC-1", "V-GRC-2", "V-GRC-3"]


def test_no_money_path_reads_the_reply_record():
    """Only the vendor master and the sourcing specialist may name it."""
    readers = sorted(str(path.relative_to(_SRC)) for path in _SRC.rglob("*.py") if "reply_record" in path.read_text(encoding="utf-8"))
    assert readers == ["workflow/specialists/sourcing.py", "workflow/systems.py"]



@pytest.mark.parametrize("value", [None, "Reliable", "unknown", 3, ["mixed"]])
def test_a_bad_reply_record_reads_as_new_and_never_stops_a_read(monkeypatch, value):
    """The field is a sourcing hint; every money path reads vendors, so it must not raise."""
    data = json.loads(_FILE.read_text(encoding="utf-8"))
    data["vendors"][0]["reply_record"] = value
    real = systems._load
    monkeypatch.setattr(systems, "_load", lambda name: copy.deepcopy(data) if name == "vendors.json" else real(name))
    found = systems.vendors()
    assert len(found) == len(data["vendors"])
    assert found[0].reply_record is ReplyRecord.NEW
    assert [vendor.vendor_id for vendor in systems.vendors_for("glass_block", _COMPANY)] == ["V-GLB-1", "V-GLB-2", "V-GLB-3"]
