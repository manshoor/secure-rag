"""Vendor quotes: reading the files, checking them against the request, picking the lowest valid one."""

from datetime import date
from decimal import Decimal

import pytest

from secure_rag.core import config

from secure_rag.workflow import systems
from secure_rag.boq.boq_parser import BoqParseError, parse_quote
from secure_rag.boq.quotes import Problem, Rfq, check_quote, compare_quotes, line_category, quotes_needed

QUOTES = config.SAMPLES_DIR / "quotes"
TODAY = date(2026, 11, 18)
GLASS = Rfq("glass_block", {"1.4": Decimal("12")}, frozenset({"V-GLB-1", "V-GLB-2", "V-GLB-3"}))
GRC = Rfq("grc", {"2.2": Decimal("45")}, frozenset({"V-GRC-1", "V-GRC-2", "V-GRC-3"}))
HEADER_BLOCK = [
    "Vendor,Lumen Glassblock Trading LLC", "Vendor ID,V-GLB-1", "Quote ref,T-1", "Date,2026-11-18",
    "Valid until,2026-12-18", "Currency,AED", "",
]


def _write(tmp_path, lines: list[str], header: str = "Item,Description,Unit,Qty,Unit price,Amount"):
    path = tmp_path / "quote.csv"
    path.write_text("\n".join([*HEADER_BLOCK, header, *lines]) + "\n", encoding="utf-8")
    return path


def _check(path, rfq: Rfq = GLASS, sender: str = "V-GLB-1", today: date = TODAY):
    return check_quote(parse_quote(path), sender, rfq, today)


# ---- reading -------------------------------------------------------------------------


def test_quote_file_reads_header_block_and_lines():
    quote = parse_quote(QUOTES / "round1" / "V-GLB-1.csv")
    assert (quote.vendor_name, quote.vendor_id, quote.ref, quote.currency) == ("Lumen Glassblock Trading LLC", "V-GLB-1", "LGT-Q-2611", "AED")
    assert (quote.issued, quote.valid_until) == (date(2026, 11, 18), date(2026, 12, 18))
    [line] = quote.lines
    assert (line.item_no, line.unit, line.quantity, line.unit_price, line.amount) == ("1.4", "m2", Decimal("12"), Decimal("410.00"), Decimal("4920.00"))


def test_quote_xlsx_reads_like_csv(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "quote.xlsx"
    book = Workbook()
    for row in [*[cell.split(",") if cell else [] for cell in HEADER_BLOCK], ["Item", "Description", "Unit", "Qty", "Unit price", "Amount"]]:
        book.active.append(row)
    book.active.append(["1.4", "Glass block feature wall 100mm", "m2", 12, 410, 4920])
    book.save(path)
    assert parse_quote(path).lines[0].amount == Decimal("4920")


def test_quote_without_price_column_is_refused(tmp_path):
    with pytest.raises(BoqParseError, match="unit price"):
        parse_quote(_write(tmp_path, ["1.4,Glass block,m2,12,4920.00"], header="Item,Description,Unit,Qty,Amount"))


def test_quote_without_vendor_id_is_refused(tmp_path):
    path = tmp_path / "quote.csv"
    path.write_text("Vendor,X LLC\nItem,Description,Unit,Qty,Unit price,Amount\n1.4,Glass,m2,12,410,4920\n", encoding="utf-8")
    with pytest.raises(BoqParseError, match="vendor id"):
        parse_quote(path)


def test_quote_errors_hold_no_file_text(tmp_path):
    path = tmp_path / "quote.csv"
    path.write_text("Vendor,Secret Client Name\nnothing else\n", encoding="utf-8")
    with pytest.raises(BoqParseError) as raised:
        parse_quote(path)
    assert "Secret" not in str(raised.value)


# ---- checking ------------------------------------------------------------------------


def test_clean_quote_is_valid():
    check = _check(QUOTES / "round1" / "V-GLB-1.csv")
    assert check.valid and check.problems == () and check.flags == ()
    assert (check.vendor_id, check.package, check.total) == ("V-GLB-1", "glass_block", Decimal("4920.00"))


def test_wrong_arithmetic_is_invalid():
    check = _check(QUOTES / "round1" / "V-GRC-2.csv", GRC, "V-GRC-2")
    assert not check.valid and check.problems == (Problem.ARITHMETIC,)


def test_corrected_resend_is_valid():
    assert _check(QUOTES / "round2" / "V-GRC-2.csv", GRC, "V-GRC-2").valid


def test_unapproved_and_uninvited_vendor_is_invalid():
    check = _check(QUOTES / "round1" / "V-GRC-9.csv", GRC, "V-GRC-9")
    assert Problem.NOT_APPROVED in check.problems and Problem.NOT_REQUESTED in check.problems


def test_planted_instruction_is_flagged_but_judged_on_numbers():
    check = _check(QUOTES / "round1" / "V-GLB-2.csv", GLASS, "V-GLB-2")
    assert check.valid and check.flags == ("planted_instruction",)


def test_sender_must_match_the_vendor_in_the_file():
    assert _check(QUOTES / "round1" / "V-GLB-1.csv", GLASS, "V-GLB-3").problems == (Problem.WRONG_SENDER,)


def test_other_currency_is_invalid(tmp_path):
    path = _write(tmp_path, ["1.4,Glass block,m2,12,410.00,4920.00"])
    path.write_text(path.read_text().replace("Currency,AED", "Currency,USD"))
    assert _check(path).problems == (Problem.CURRENCY,)


def test_expired_quote_is_invalid():
    assert _check(QUOTES / "round1" / "V-GLB-1.csv", today=date(2026, 12, 19)).problems == (Problem.EXPIRED,)


def test_quantity_must_match_the_request(tmp_path):
    assert _check(_write(tmp_path, ["1.4,Glass block,m2,10,410.00,4100.00"])).problems == (Problem.QUANTITY,)


def test_line_not_requested_and_line_missing(tmp_path):
    check = _check(_write(tmp_path, ["9.9,Something else,m2,12,410.00,4920.00"]))
    assert check.problems == (Problem.UNKNOWN_LINE, Problem.MISSING_LINE)


def test_unreadable_price_is_invalid(tmp_path):
    assert _check(_write(tmp_path, ["1.4,Glass block,m2,12,TBC,4920.00"])).problems == (Problem.UNREADABLE,)


# ---- comparing -----------------------------------------------------------------------


def test_lowest_valid_quote_wins():
    checks = [_check(QUOTES / "round1" / f"{vendor}.csv", GLASS, vendor) for vendor in ("V-GLB-1", "V-GLB-2", "V-GLB-3")]
    assert compare_quotes(checks).vendor_id == "V-GLB-3"


def test_invalid_quote_never_wins_even_when_cheapest():
    cheap_but_wrong = _check(QUOTES / "round1" / "V-GRC-2.csv", GRC, "V-GRC-2")
    valid = _check(QUOTES / "round1" / "V-GRC-1.csv", GRC, "V-GRC-1")
    assert compare_quotes([cheap_but_wrong, valid]).vendor_id == "V-GRC-1"


def test_tie_goes_to_the_quote_received_first(tmp_path):
    first = _check(QUOTES / "round1" / "V-GLB-1.csv")
    second = _check(QUOTES / "round1" / "V-GLB-1.csv")
    assert compare_quotes([first, second]) is first


def test_no_valid_quote_means_no_winner():
    assert compare_quotes([_check(QUOTES / "round1" / "V-GRC-2.csv", GRC, "V-GRC-2")]) is None


# ---- policy and categories -----------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "needed"),
    [("9999.99", 1), ("10000", 3), ("100000", 3), ("100000.01", None)],
)
def test_quotes_needed_follows_the_policy_bands(total, needed):
    assert quotes_needed(Decimal(total)) == needed


@pytest.mark.parametrize(
    ("description", "category"),
    [("Glass block feature wall 100mm", "glass_block"), ("Decorative GRC cornice", "grc"), ("Internal cement plaster", "")],
)
def test_line_category(description, category):
    assert line_category(description) == category


def test_vendors_for_a_category_are_approved_and_same_company():
    assert [v.vendor_id for v in systems.vendors_for("grc", "alpha-construction")] == ["V-GRC-1", "V-GRC-2", "V-GRC-3"]
    assert systems.vendors_for("grc", "beta-properties") == []


def test_every_sample_quote_names_a_known_vendor():
    known = {vendor.vendor_id: vendor.name for vendor in systems.vendors()}
    for path in QUOTES.rglob("*.csv"):
        quote = parse_quote(path)
        assert known[quote.vendor_id] == quote.vendor_name and path.stem == quote.vendor_id


# ---- review fixes: units, repeated lines, odd header text ---------------------------------

GLASS_M2 = Rfq("glass_block", {"1.4": Decimal("12")}, frozenset({"V-GLB-1"}), {"1.4": "m2"})


def test_quote_in_another_unit_is_invalid(tmp_path):
    check = _check(_write(tmp_path, ["1.4,Glass block,pcs,12,9.00,108.00"]), rfq=GLASS_M2)
    assert Problem.UNIT in check.problems and not check.valid


def test_quote_unit_spelling_is_normalised(tmp_path):
    assert _check(_write(tmp_path, ["1.4,Glass block,sqm,12,410.00,4920.00"]), rfq=GLASS_M2).valid


def test_quote_naming_a_line_twice_is_invalid(tmp_path):
    check = _check(_write(tmp_path, ["1.4,Glass block,m2,12,300.00,3600.00", "1.4,Glass block,m2,12,300.00,3600.00"]))
    assert Problem.DUPLICATE_LINE in check.problems and not check.valid


def test_header_text_that_breaks_a_field_limit_is_a_parse_error(tmp_path):
    path = tmp_path / "quote.csv"
    block = [line if not line.startswith("Currency") else "Currency," + "ß" * 150 for line in HEADER_BLOCK]
    path.write_text("\n".join([*block, "Item,Description,Unit,Qty,Unit price,Amount", "1.4,Glass,m2,12,410.00,4920.00"]) + "\n", encoding="utf-8")
    with pytest.raises(BoqParseError):
        parse_quote(path)


@pytest.mark.parametrize("price", ["0.001", "0.0049"])
def test_quote_under_one_fils_is_invalid(tmp_path, price):
    """It would round to AED 0.00 and be ordered for nothing."""
    rfq = Rfq("glass_block", {"1.4": Decimal("1")}, GLASS.vendor_ids)
    check = _check(_write(tmp_path, [f"1.4,Glass block,m2,1,{price},{price}"]), rfq)
    assert check.problems == (Problem.ZERO_AMOUNT,)


def test_one_zero_line_spoils_a_quote_with_a_real_total(tmp_path):
    rfq = Rfq("glass_block", {"1.4": Decimal("1"), "1.5": Decimal("1")}, GLASS.vendor_ids)
    check = _check(_write(tmp_path, ["1.4,Glass block,m2,1,0.001,0.001", "1.5,Glass block,m2,1,50,50"]), rfq)
    assert check.problems == (Problem.ZERO_AMOUNT,)


def test_quote_of_half_a_fils_rounds_up_and_is_valid(tmp_path):
    rfq = Rfq("glass_block", {"1.4": Decimal("1")}, GLASS.vendor_ids)
    assert _check(_write(tmp_path, ["1.4,Glass block,m2,1,0.005,0.005"]), rfq).valid
