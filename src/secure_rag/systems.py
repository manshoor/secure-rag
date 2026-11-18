"""Adapter for the HR system, vendor catalogue and budget module.

    workflow.py ──► find_hires / catalogue / price_items
                    reserve_budget / spend_budget / release_budget
                       │
                       ├─► data/systems/*.json             (invented stand-ins, read only)
                       └─► data/runtime/budget_ledger.json (holds and spend per run)

The workflow calls only these domain functions. Replacing the JSON files with
real HR and ERP APIs changes this module and nothing else. Every amount is a
Decimal read from data, never a value the model produced.
"""

import fcntl
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path

from . import config
from .policy import money

_HR_FILE = "hr_new_hires.json"
_CATALOGUE_FILE = "vendor_catalogue.json"
_BUDGET_FILE = "budgets.json"
_ACTIVE_STATUS = "confirmed"

# Holds and spend per run, in the runtime folder (written), not the systems folder (read only).
_LEDGER_FILE = "budget_ledger.json"
_LEDGER_LOCK = "budget_ledger.lock"
_LEDGER_GUARD = threading.Lock()


class HoldState(str, Enum):
    """A run's claim on a budget: held while it waits for approval, spent once ordered."""

    HELD = "held"
    SPENT = "spent"


@dataclass(frozen=True)
class Hire:
    hire_id: str
    name: str
    role: str
    company: str
    start_date: date
    status: str


@dataclass(frozen=True)
class CatalogueItem:
    code: str
    description: str
    unit: str
    vendor: str
    unit_price: Decimal


@dataclass(frozen=True)
class PricedLine:
    code: str
    description: str
    unit: str
    vendor: str
    quantity: Decimal
    unit_price: Decimal
    total: Decimal


class SystemLookupError(LookupError):
    """A stand-in system has no record for the request (unknown item, budget line)."""


class BudgetShortError(Exception):
    """The amount does not fit what is left after other runs' holds and spend."""

    def __init__(self, shortfall: Decimal, available: Decimal) -> None:
        super().__init__(f"budget short by AED {shortfall:,.2f}")
        self.shortfall = shortfall
        self.available = available


# ---- HR -----------------------------------------------------------------------


def hire_by_id(hire_id: str) -> Hire | None:
    """One hire record by id, whatever its company or status."""
    for hire in _hires():
        if hire.hire_id == hire_id:
            return hire
    return None


def find_hires(company: str, role: str, start: date) -> list[Hire]:
    """Confirmed hires of one company, role and start date, sorted by id."""
    wanted = _singular(role)
    found = [
        hire
        for hire in _hires()
        if hire.company == company
        and _singular(hire.role) == wanted
        and hire.start_date == start
        and hire.status == _ACTIVE_STATUS
    ]
    return sorted(found, key=lambda hire: hire.hire_id)


def is_active(hire: Hire) -> bool:
    return hire.status == _ACTIVE_STATUS


# ---- catalogue ----------------------------------------------------------------


def catalogue() -> list[CatalogueItem]:
    """Every approved catalogue item."""
    raw = _load(_CATALOGUE_FILE)["items"]
    return [
        CatalogueItem(item["code"], item["description"], item["unit"], item["vendor"], Decimal(item["unit_price_aed"]))
        for item in raw
    ]


def approved_vendors() -> set[str]:
    """Vendors that may receive purchase orders (every vendor in the catalogue)."""
    return {item.vendor for item in catalogue()}


def price_items(quantities: dict[str, Decimal]) -> list[PricedLine]:
    """Price catalogue codes: quantity x catalogue rate, rounded per line. Raises for unknown codes."""
    by_code = {item.code: item for item in catalogue()}
    lines: list[PricedLine] = []
    for code, quantity in quantities.items():
        item = by_code.get(code)
        if item is None:
            raise SystemLookupError(f"{code} is not in the approved catalogue")
        lines.append(
            PricedLine(item.code, item.description, item.unit, item.vendor, quantity, item.unit_price, money(quantity * item.unit_price))
        )
    return lines


# ---- budget -------------------------------------------------------------------


def remaining_budget(company: str, department: str) -> Decimal:
    """Remaining budget in AED for one company department. Raises if there is no budget line."""
    for row in _load(_BUDGET_FILE)["budgets"]:
        if row["company"] == company and row["department"] == department:
            return Decimal(row["remaining_aed"])
    raise SystemLookupError(f"no budget for {company} / {department}")


def available_budget(company: str, department: str) -> Decimal:
    """Remaining budget less every run's hold and spend."""
    with _ledger() as ledger:
        return _available(ledger, company, department, exclude_run="")


def reserve_budget(run_id: str, company: str, department: str, amount: Decimal) -> Decimal:
    """Hold `amount` for one run. Returns what was available to it (other runs' holds taken off).

    Calling again for the same run re-checks against today's budget and replaces
    the hold. Raises BudgetShortError, and changes nothing, if it does not fit.
    """
    with _ledger() as ledger:
        available = _available(ledger, company, department, exclude_run=run_id)
        if amount > available:
            raise BudgetShortError(amount - available, available)

        # A spent run stays spent; a repeat call never turns it back into a hold.
        if ledger.get(run_id, {}).get("state") == HoldState.SPENT.value:
            return available
        ledger[run_id] = _entry(company, department, amount, HoldState.HELD)
        return available


def spend_budget(run_id: str, company: str, department: str, amount: Decimal) -> None:
    """Turn the run's hold into spend, after a last check. Raises BudgetShortError if it no longer fits."""
    with _ledger() as ledger:
        if ledger.get(run_id, {}).get("state") == HoldState.SPENT.value:
            return
        available = _available(ledger, company, department, exclude_run=run_id)
        if amount > available:
            raise BudgetShortError(amount - available, available)
        ledger[run_id] = _entry(company, department, amount, HoldState.SPENT)


def is_spent(run_id: str) -> bool:
    """True once a run's hold has become spend (its purchase order must still go out)."""
    with _ledger() as ledger:
        return ledger.get(run_id, {}).get("state") == HoldState.SPENT.value


def release_budget(run_id: str) -> None:
    """Drop a run's hold (rejected or stopped). Spend is never released."""
    with _ledger() as ledger:
        if ledger.get(run_id, {}).get("state") == HoldState.HELD.value:
            del ledger[run_id]


# ---- budget ledger (stand-in for ERP commitments) ------------------------------
#
#   budgets.json (read only)        data/runtime/budget_ledger.json
#   remaining_aed per department  - { run_id: {amount, state: held | spent} }
#                                 = available to the next run
#
# Every read-check-write runs under one file lock, so two processes (CLI and
# API) or two threads cannot both fit into the same budget.


def _available(ledger: dict, company: str, department: str, exclude_run: str) -> Decimal:
    used = sum(
        (
            Decimal(entry["amount_aed"])
            for run_id, entry in ledger.items()
            if run_id != exclude_run and entry["company"] == company and entry["department"] == department
        ),
        Decimal("0.00"),
    )
    return remaining_budget(company, department) - used


def _entry(company: str, department: str, amount: Decimal, state: HoldState) -> dict:
    return {"company": company, "department": department, "amount_aed": str(money(amount)), "state": state.value}


@contextmanager
def _ledger() -> Iterator[dict]:
    """The ledger under an exclusive lock; changes are written back atomically on a clean exit."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / _LEDGER_FILE
    with _LEDGER_GUARD, (config.DATA_DIR / _LEDGER_LOCK).open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            before = json.dumps(ledger, sort_keys=True)
            yield ledger
            if json.dumps(ledger, sort_keys=True) == before:
                return

            # Write a temp file, then rename: a crash never leaves half a ledger.
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# ---- file access (stand-in only) ---------------------------------------------


def _hires() -> list[Hire]:
    return [
        Hire(row["id"], row["name"], row["role"], row["company"], date.fromisoformat(row["start_date"]), row["status"])
        for row in _load(_HR_FILE)["hires"]
    ]


def _load(name: str) -> dict:
    path: Path = config.systems_dir() / name
    return json.loads(path.read_text(encoding="utf-8"))


def _singular(role: str) -> str:
    """'Site Engineers' -> 'site engineer', so a request's plural matches the HR record."""
    text = " ".join(role.lower().split())
    return text[:-1] if text.endswith("s") else text
