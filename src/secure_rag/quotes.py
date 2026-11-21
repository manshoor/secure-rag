"""Vendor quotes for BOQ lines the catalogue cannot price.

    RFQ (package, lines, invited vendors) ──► vendor replies (quote files, boq_parser.parse_quote)
        ──► check_quote: sender, approval, currency, validity, each line against the request
        ──► compare_quotes: lowest valid total wins; a tie goes to the quote received first

Prices come only from the quote files; a model never sets or edits one.
Problems and flags are codes, never text from a file.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum

from . import systems
from .boq import normalise_unit
from .boq_parser import Quote
from .guard import injection_findings
from .policy import EXECUTIVE_APPROVAL_AED, SINGLE_QUOTE_BELOW_AED, money

_CURRENCY = "AED"  # catalogue and budgets are in AED; no exchange rates
_PLANTED = "planted_instruction"

# Quotations the procurement policy asks for, by the purchase total (policy.quotes_required).
_ONE_QUOTE = 1
_THREE_QUOTES = 3

# Words in a BOQ description that put the line in a quotation category (vendors.json).
_CATEGORY_TERMS = {"glass_block": ("glass block",), "grc": ("grc",)}


class Problem(str, Enum):
    """Why a quote cannot be used. A quote with any problem never wins."""

    WRONG_SENDER = "wrong_sender"
    NOT_APPROVED = "vendor_not_approved"
    NOT_REQUESTED = "not_requested"
    CURRENCY = "currency_not_aed"
    EXPIRED = "expired"
    UNKNOWN_LINE = "unknown_line"
    UNREADABLE = "unreadable_number"
    QUANTITY = "quantity_differs"
    ARITHMETIC = "amount_differs"
    MISSING_LINE = "missing_line"
    UNIT = "unit_differs"
    DUPLICATE_LINE = "duplicate_line"


@dataclass(frozen=True)
class Rfq:
    """One request for quotation: a package of BOQ lines (item number -> quantity) and who was asked.

    `units` holds the canonical unit per item (boq.normalise_unit); an item without one is not unit-checked.
    """

    package: str
    lines: dict[str, Decimal]
    vendor_ids: frozenset[str]
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QuoteCheck:
    """A quote as checked: the sender, its total for the requested lines, problems and flags."""

    vendor_id: str
    package: str
    total: Decimal
    problems: tuple[Problem, ...]
    flags: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.problems


def check_quote(quote: Quote, sender_id: str, rfq: Rfq, today: date) -> QuoteCheck:
    """Check one reply against its request. The sender is who delivered it, not what the file claims."""
    problems: list[Problem] = []

    # Who sent it: the file must name the sender, who must be approved and invited.
    vendor = next((v for v in systems.vendors() if v.vendor_id == sender_id), None)
    if quote.vendor_id != sender_id:
        problems.append(Problem.WRONG_SENDER)
    if vendor is None or not vendor.approved:
        problems.append(Problem.NOT_APPROVED)
    if sender_id not in rfq.vendor_ids:
        problems.append(Problem.NOT_REQUESTED)

    # Terms: AED only, still valid today.
    if quote.currency != _CURRENCY:
        problems.append(Problem.CURRENCY)
    if quote.valid_until is None or quote.valid_until < today:
        problems.append(Problem.EXPIRED)

    # Lines: each one requested, readable, the requested quantity, and amount = qty x unit price.
    total = Decimal("0")
    quoted: set[str] = set()
    for line in quote.lines:
        if line.item_no not in rfq.lines:
            problems.append(Problem.UNKNOWN_LINE)
            continue
        if line.item_no in quoted:
            problems.append(Problem.DUPLICATE_LINE)
            continue
        quoted.add(line.item_no)
        expected_unit = rfq.units.get(line.item_no, "")
        if expected_unit and normalise_unit(line.unit) != expected_unit:
            problems.append(Problem.UNIT)
        if line.quantity is None or line.unit_price is None or line.amount is None:
            problems.append(Problem.UNREADABLE)
            continue
        if line.quantity != rfq.lines[line.item_no]:
            problems.append(Problem.QUANTITY)
        if money(line.quantity * line.unit_price) != money(line.amount):
            problems.append(Problem.ARITHMETIC)
        total += money(line.amount)
    if set(rfq.lines) - quoted:
        problems.append(Problem.MISSING_LINE)

    # Planted instructions are flagged for procurement; the numbers alone decide.
    texts = [quote.vendor_name, quote.ref, *(line.description for line in quote.lines)]
    flags = (_PLANTED,) if any(injection_findings(text) for text in texts) else ()
    return QuoteCheck(sender_id, rfq.package, total, tuple(dict.fromkeys(problems)), flags)


def compare_quotes(checks: list[QuoteCheck]) -> QuoteCheck | None:
    """The lowest valid total, or None. Checks come in the order received, so min() keeps the first of a tie."""
    valid = [check for check in checks if check.valid]
    if not valid:
        return None
    return min(valid, key=lambda check: check.total)


def quotes_needed(total: Decimal) -> int | None:
    """Valid quotations the policy asks for at this purchase total; None means an open tender."""
    if total < SINGLE_QUOTE_BELOW_AED:
        return _ONE_QUOTE
    if total <= EXECUTIVE_APPROVAL_AED:
        return _THREE_QUOTES
    return None


def line_category(description: str) -> str:
    """The quotation category a BOQ line belongs to, or '' when none fits."""
    text = description.lower()
    for category, terms in _CATEGORY_TERMS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms):
            return category
    return ""
