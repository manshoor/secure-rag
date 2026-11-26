"""Sourcing specialist: which approved vendors to ask per package, and whom to chase.

    pick_vendors, before the RFQ gate
      run state ─► vendor_facts (code: candidates per package, licence and reply enums) ─► agent
          packages              ids, line and candidate counts, the minimum to ask
          package_vendors(id)   one package's candidates: vendor id, licence, reply record
      ─► finish: per package {package, ask: vendor ids, note: template id} ─► _validate_ask (code)

    pick_chase, after procurement chose extend
      run state ─► chase_facts (code: quotes_apply's chase list, per short package) ─► agent
          chase_packages        ids, needed and valid counts, the minimum to chase
          chase_targets(id)     one package's targets: vendor id, replied, problem codes, reply record
      ─► finish: {chase: [{vendor_id, package}]} ─► _validate_chase (code)

Candidates are systems.vendors_for(category, company), less any vendor whose licence
has expired on config.today(). A model never adds a vendor: the vendor enum holds
today's candidates only, and the validator holds each package to its own.
The request note is a template id; code owns the wording (note_text). No model
words reach a vendor.

Handlers read only: no outbox, state, audit or `expected` write. Phase 3 applies
an outcome. Facts carry ids, enums and counts: never a name, email, bank account or date.

The rules (also the scripted stand-in and every fallback): ask every candidate with
the standard note, as _pick_vendors does; chase the whole list in quotes_apply's
order. reason_codes in an outcome are code's own, rebuilt from the accepted decision,
and so are left_out (per package) and skipped: the model never sends them.

reason: after _base's screen, sourcing drops it when it holds a vendor id this run
did not offer (any case) or the currency word AED. Still passes: vendor names,
approval claims ("no approval is needed"), offered ids in a misleading sentence,
and markup. So phase 3 escapes it and never shows it beside vendors.

A run state marked counts-only is private here, whatever the caller passes.
"""

import copy
import re
import unicodedata
from collections.abc import Callable
from datetime import date
from enum import Enum
from typing import Any

from ...boq.quotes import quotes_needed
from ...core import config
from ...core.agent_runtime import AgentResult, AgentSpec, EndReason, RunPrivacy, ScriptedClient, Tool
from .. import systems
from ..policy import EXECUTIVE_APPROVAL_AED
from ..procurement_agent import PkgStatus, _valid_by_package
from ..workflow import Privacy
from ._base import Status, outcome, run_specialist
from .evaluation import _KNOWN_PROBLEMS

_VENDORS_AGENT = "sourcing_vendors"
_CHASE_AGENT = "sourcing_chase"
_PACKAGES_TOOL = "packages"
_VENDORS_TOOL = "package_vendors"
_CHASE_PACKAGES_TOOL = "chase_packages"
_TARGETS_TOOL = "chase_targets"

# The most quotations the policy asks for below an open tender (three today). The rule
# itself is not known until quotes arrive, so a package asks at least this many when it can.
_MOST_QUOTES = quotes_needed(EXECUTIVE_APPROVAL_AED)

# A licence that runs out within this many days is shown as expiring soon; an expired one is left out.
_EXPIRING_DAYS = 60

# Limits: the script makes one call per package plus one; the rest is room for a live run.
_MAX_TURNS = 5
_SPARE_CALLS = 4
_MAX_TOKENS = 40_000
_MAX_COST_USD = 0.30

# reason is dropped when it names a vendor id not offered in this run, or a price word.
_VENDOR_ID_SHAPE = re.compile(r"(?<![A-Za-z0-9])V-[A-Za-z]+-[0-9]+", re.IGNORECASE)
_PRICE_WORD = re.compile(r"(?<![A-Za-z])AED(?![A-Za-z])", re.IGNORECASE)

_EMPTY_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}

_VENDORS_SYSTEM = (
    "You help procurement choose which approved vendors receive a request for quotation. "
    "For each package, pick the vendors to ask from its candidates, at least the package's minimum, "
    "and a note template for the request. Candidates come from the vendor master; you cannot add one. "
    "Use the tools to read each package's candidates, their licence status and reply record. "
    "Code sends nothing until procurement approves; you only propose."
)

_CHASE_SYSTEM = (
    "You help procurement after it extended a quote round. Code lists the invited vendors of short packages "
    "that have no valid quote. Pick whom to remind, in the order to remind them, at least each package's minimum. "
    "Use the tools to read who replied and the problem codes of their quotes. Code sends the reminders; you only propose."
)


class _Licence(str, Enum):
    VALID = "valid"
    EXPIRING_SOON = "expiring_soon"  # runs out within _EXPIRING_DAYS


class _Note(str, Enum):
    """Request note templates, by id. The wording is code's (_NOTE_WORDING)."""

    STANDARD = "standard"
    CONFIRM_LEAD_TIME = "confirm_lead_time"
    SITE_MEASURE = "site_measure"


# The words a vendor reads under the request's lines. No model text, no number.
_NOTE_WORDING = {
    _Note.STANDARD: "Please quote for the lines listed, in AED, and state how long your quote is valid.",
    _Note.CONFIRM_LEAD_TIME: "Please quote for the lines listed, in AED, state how long your quote is valid, and confirm the delivery lead time for each line.",
    _Note.SITE_MEASURE: "Please quote for the lines listed, in AED, and state how long your quote is valid. Quantities are to be confirmed by site measurement before fabrication; say if you need a site visit.",
}


class _AskCode(str, Enum):
    """Why these vendors, as codes. Listed in this order."""

    ASKED_ALL = "asked_all"  # a package asks every candidate
    ASKED_FEWER = "asked_fewer"  # a package leaves a candidate out
    LEFT_OUT_EXPIRING_LICENCE = "left_out_expiring_licence"
    LEFT_OUT_MIXED_RECORD = "left_out_mixed_record"
    LEFT_OUT_NEW_VENDOR = "left_out_new_vendor"
    LEFT_OUT_RELIABLE_VALID = "left_out_reliable_valid"  # a left-out candidate with a valid licence and a reliable record
    FEWER_THAN_THREE_CANDIDATES = "fewer_than_three_candidates"  # a package has fewer than _MOST_QUOTES


class _ChaseCode(str, Enum):
    """Why these targets, as codes. Listed in this order."""

    CHASED_ALL = "chased_all"  # a package chases every target
    CHASED_FEWER = "chased_fewer"  # a package skips a target
    SKIPPED_NO_REPLY = "skipped_no_reply"  # a skipped target sent nothing
    SKIPPED_PROBLEM_QUOTE = "skipped_problem_quote"  # a skipped target sent a quote with problems


# ---- public ---------------------------------------------------------------------------------------


def note_text(note: str) -> str:
    """The fixed wording of a note template id. Raises ValueError for any other string."""
    return _NOTE_WORDING[_Note(note)]


def pick_vendors(state: dict, client: Any | None, *, privacy: RunPrivacy, case_id: str | None = None) -> dict:
    """The outcome for whom to ask per package (see _base.outcome).

    client None means the scripted stand-in; a ScriptedClient given is scripted too.
    A counts-only state runs as PRIVATE. No package with a candidate: an empty
    decision, status nothing_to_do, and no model call.
    """
    facts = vendor_facts(state)
    decision, codes = _ask_rules(facts)
    ids = [package["package"] for package in facts["packages"]]
    if not ids:
        return outcome(_VENDORS_AGENT, Status.NOTHING_TO_DO, decision, codes, "", AgentResult(_VENDORS_AGENT, EndReason.PRIVATE))

    # The rules decide offline, privately and on any failure.
    script = [[(_PACKAGES_TOOL, {})], [(_VENDORS_TOOL, {"package": package}) for package in ids]]
    answer = {"packages": [{key: row[key] for key in ("package", "ask", "note")} for row in decision["packages"]]}
    task = {"packages": ids, "without_candidates": facts["without_candidates"]}
    offered = {row["vendor_id"] for package in facts["packages"] for row in package["candidates"]}
    found = _run(_ask_spec(facts), task, _ask_authorizer(set(ids)), client, state, privacy, case_id, (script, answer), decision, codes,
                 lambda output: _validate_ask(facts, output))
    return _screened(found, offered)


def pick_chase(state: dict, client: Any | None, *, privacy: RunPrivacy, case_id: str | None = None) -> dict:
    """The outcome for whom to remind after an extension (see _base.outcome).

    client None means the scripted stand-in; a ScriptedClient given is scripted too.
    A counts-only state runs as PRIVATE. An empty chase list: an empty decision,
    status nothing_to_do, and no model call. Never touches expected or the outbox.
    """
    facts = chase_facts(state)
    decision, codes = _chase_rules(facts)
    ids = [package["package"] for package in facts["packages"]]
    if not ids:
        return outcome(_CHASE_AGENT, Status.NOTHING_TO_DO, decision, codes, "", AgentResult(_CHASE_AGENT, EndReason.PRIVATE))

    script = [[(_CHASE_PACKAGES_TOOL, {})], [(_TARGETS_TOOL, {"package": package}) for package in ids]]
    answer = {"chase": decision["chase"]}
    offered = {row["vendor_id"] for package in facts["packages"] for row in package["targets"]}
    found = _run(_chase_spec(facts), {"short_packages": ids}, _chase_authorizer(set(ids)), client, state, privacy, case_id, (script, answer), decision, codes,
                 lambda output: _validate_chase(facts, output))
    return _screened(found, offered)


def vendor_facts(state: dict) -> dict:
    """Per package with a candidate: its id, line count, the minimum to ask, and each candidate as enums.

    Candidates are read from the vendor master now (category and company), never
    from package vendors already in the state. A licence expired today leaves the vendor out.
    """
    today = config.today()
    packages = []
    without = 0
    for package in state["packages"]:
        candidates = []
        for vendor in systems.vendors_for(package["category"], state["company"]):
            licence = _licence(vendor, today)
            if licence is not None:
                candidates.append({"vendor_id": vendor.vendor_id, "licence": licence.value, "reply_record": vendor.reply_record.value})
        if not candidates:
            without += 1
            continue
        packages.append({"package": package["id"], "lines": len(package["items"]), "min_ask": min(_MOST_QUOTES, len(candidates)), "candidates": candidates})
    return {"packages": packages, "without_candidates": without}


def chase_facts(state: dict) -> dict:
    """Per short package with a target: needed and valid counts, the minimum to chase, and each target.

    The targets are quotes_apply's chase list (_chase_list). replied and problems come
    from the target's counted quote for that package; problem strings code does not know are dropped.
    """
    valid = _valid_by_package(state["quotes"])
    needed = {row["package"]: row.get("needed") for row in state.get("comparison", [])}
    records = {vendor.vendor_id: vendor.reply_record.value for vendor in systems.vendors()}
    targets = _chase_list(state)

    # Per short package, in package order, only those with someone to chase.
    packages = []
    for package in state["packages"]:
        mine = [target["vendor_id"] for target in targets if target["package"] == package["id"]]
        if not mine:
            continue
        count = len(valid.get(package["id"], []))
        packages.append({
            "package": package["id"],
            "needed": needed.get(package["id"]),
            "valid_quotes": count,
            "min_chase": _min_chase(needed.get(package["id"]), count, len(mine)),
            "targets": [_target(state, package["id"], vendor, records) for vendor in mine],
        })
    return {"packages": packages}


# ---- facts helpers --------------------------------------------------------------------------------


def _licence(vendor: systems.Vendor, today: date) -> _Licence | None:
    """None when expired: the licence counts to the end of its expiry day."""
    days = (vendor.licence_expiry - today).days
    if days < 0:
        return None
    return _Licence.EXPIRING_SOON if days <= _EXPIRING_DAYS else _Licence.VALID


def _chase_list(state: dict) -> list[dict]:
    """As quotes_apply builds it: every invited vendor of a short package with no valid quote, in order."""
    valid = _valid_by_package(state["quotes"])
    short = [package for package in state["packages"] if package["status"] == PkgStatus.SHORT.value]
    return [
        {"vendor_id": vendor_id, "package": package["id"]}
        for package in short
        for vendor_id in package["vendors"]
        if vendor_id not in {quote["vendor_id"] for quote in valid.get(package["id"], [])}
    ]


def _min_chase(needed: int | None, valid: int, targets: int) -> int:
    """Enough to reach the count needed, if every one chased sends a valid quote; unknown needed: all."""
    if needed is None:
        return targets
    return min(max(needed - valid, 0), targets)


def _target(state: dict, package: str, vendor_id: str, records: dict[str, str]) -> dict:
    """One target: did it reply for this package, and its quote's known problem codes."""
    quote = next((q for q in state["quotes"] if q["counted"] and q["package"] == package and q["vendor_id"] == vendor_id), None)
    problems = [code for code in quote["problems"] if code in _KNOWN_PROBLEMS] if quote else []
    return {"vendor_id": vendor_id, "replied": quote is not None, "problems": problems, "reply_record": records.get(vendor_id, systems.ReplyRecord.NEW.value)}


# ---- the agents -----------------------------------------------------------------------------------


def _run(
    spec: AgentSpec,
    task: dict,
    authorize: Callable[[str, str, dict], bool],
    client: Any | None,
    state: dict,
    privacy: RunPrivacy,
    case_id: str | None,
    script: tuple[list[list[tuple[str, dict]]], dict],
    decision: dict,
    codes: list[str],
    validate: Callable[[dict], tuple[dict, list[str]] | None],
) -> dict:
    """Both entries: privacy from the state, the scripted stand-in, the run with the rules behind it."""
    # The state itself decides privacy too: this gate does not depend on the thread.
    if state.get("privacy") == Privacy.COUNTS_ONLY.value:
        privacy = RunPrivacy.PRIVATE
    # The stand-in answers in the model's shape: code adds left_out and skipped itself.
    if client is None:
        steps, answer = script
        client = ScriptedClient.from_steps(steps, {**answer, "reason_codes": codes, "reason": ""})
    success = Status.SCRIPTED if isinstance(client, ScriptedClient) else Status.AGENT
    return run_specialist(
        spec, task, authorize, client,
        privacy=privacy, success=success, rules=lambda: (copy.deepcopy(decision), list(codes)), validate=validate, case_id=case_id,
    )


def _screened(found: dict, offered: set[str]) -> dict:
    """The outcome with reason dropped if it names a vendor id not offered, or AED."""
    text = unicodedata.normalize("NFKC", found["reason"])
    named = {match.upper() for match in _VENDOR_ID_SHAPE.findall(text)}
    if named - {vendor.upper() for vendor in offered} or _PRICE_WORD.search(text):
        return {**found, "reason": ""}
    return found


def _package_arg(ids: list[str]) -> dict:
    return {"type": "object", "properties": {"package": {"type": "string", "enum": ids}}, "required": ["package"], "additionalProperties": False}


def _ask_spec(facts: dict) -> AgentSpec:
    """Schemas built for this moment: only today's packages and candidates are valid."""
    ids = [package["package"] for package in facts["packages"]]
    by_id = {package["package"]: package for package in facts["packages"]}
    vendor_ids = [row["vendor_id"] for package in facts["packages"] for row in package["candidates"]]
    tools = (
        Tool(_PACKAGES_TOOL, "Packages with candidates: line and candidate counts, the minimum to ask.", _EMPTY_ARGS, lambda args: _packages_view(facts)),
        Tool(_VENDORS_TOOL, "One package's candidates: vendor id, licence status, reply record.", _package_arg(ids), lambda args: copy.deepcopy(by_id[args["package"]])),
    )
    answer = {
        "type": "object",
        "properties": {
            "package": {"type": "string", "enum": ids},
            "ask": {"type": "array", "items": {"type": "string", "enum": vendor_ids}, "description": "Vendor ids of this package's candidates, each once."},
            "note": {"type": "string", "enum": [note.value for note in _Note]},
        },
        "required": ["package", "ask", "note"],
        "additionalProperties": False,
    }
    return AgentSpec(
        _VENDORS_AGENT, _VENDORS_SYSTEM, tools, _output_schema("packages", answer, _AskCode),
        max_turns=_MAX_TURNS, max_tool_calls=1 + len(ids) + _SPARE_CALLS, max_tokens=_MAX_TOKENS, max_cost_usd=_MAX_COST_USD,
    )


def _chase_spec(facts: dict) -> AgentSpec:
    """Schemas built for this moment: only today's short packages and targets are valid."""
    ids = [package["package"] for package in facts["packages"]]
    by_id = {package["package"]: package for package in facts["packages"]}
    vendor_ids = sorted({row["vendor_id"] for package in facts["packages"] for row in package["targets"]})
    tools = (
        Tool(_CHASE_PACKAGES_TOOL, "Short packages: needed and valid counts, targets, the minimum to chase.", _EMPTY_ARGS, lambda args: _chase_view(facts)),
        Tool(_TARGETS_TOOL, "One short package's targets: replied, problem codes, reply record.", _package_arg(ids), lambda args: copy.deepcopy(by_id[args["package"]])),
    )
    entry = {
        "type": "object",
        "properties": {"vendor_id": {"type": "string", "enum": vendor_ids}, "package": {"type": "string", "enum": ids}},
        "required": ["vendor_id", "package"],
        "additionalProperties": False,
    }
    return AgentSpec(
        _CHASE_AGENT, _CHASE_SYSTEM, tools, _output_schema("chase", entry, _ChaseCode),
        max_turns=_MAX_TURNS, max_tool_calls=1 + len(ids) + _SPARE_CALLS, max_tokens=_MAX_TOKENS, max_cost_usd=_MAX_COST_USD,
    )


def _output_schema(key: str, item: dict, codes: type[Enum]) -> dict:
    return {
        "type": "object",
        "properties": {
            key: {"type": "array", "items": item},
            "reason_codes": {"type": "array", "items": {"type": "string", "enum": [code.value for code in codes]}},
            "reason": {"type": "string", "description": "One short sentence for procurement. No prices."},
        },
        "required": [key, "reason_codes", "reason"],
        "additionalProperties": False,
    }


def _packages_view(facts: dict) -> dict:
    rows = [{"package": p["package"], "lines": p["lines"], "candidates": len(p["candidates"]), "min_ask": p["min_ask"]} for p in facts["packages"]]
    return {"packages": rows, "without_candidates": facts["without_candidates"]}


def _chase_view(facts: dict) -> dict:
    rows = [
        {"package": p["package"], "needed": p["needed"], "valid_quotes": p["valid_quotes"], "targets": len(p["targets"]), "min_chase": p["min_chase"]}
        for p in facts["packages"]
    ]
    return {"packages": rows}


def _ask_authorizer(ids: set[str]) -> Callable[[str, str, dict], bool]:
    """The vendor agent, its two tools, and only packages with candidates now."""
    return _authorizer(_VENDORS_AGENT, _PACKAGES_TOOL, _VENDORS_TOOL, ids)


def _chase_authorizer(ids: set[str]) -> Callable[[str, str, dict], bool]:
    """The chase agent, its two tools, and only short packages with targets now."""
    return _authorizer(_CHASE_AGENT, _CHASE_PACKAGES_TOOL, _TARGETS_TOOL, ids)


def _authorizer(agent_name: str, list_tool: str, package_tool: str, ids: set[str]) -> Callable[[str, str, dict], bool]:
    def authorize(agent: str, tool: str, args: dict) -> bool:
        if agent != agent_name:
            return False
        if tool == list_tool:
            return args == {}
        if tool == package_tool:
            return set(args) == {"package"} and args["package"] in ids
        return False

    return authorize


# ---- code's checks and rules: vendors -------------------------------------------------------------


def _validate_ask(facts: dict, output: dict) -> tuple[dict, list[str]] | None:
    """The decision if every package with candidates is answered once, each ask a subset of its own
    candidates with no repeats and at least its minimum, and every reason code holds; else None.
    Answers come back in package order, ask in id order."""
    # Every package once; reason codes once each.
    by_id = {package["package"]: package for package in facts["packages"]}
    answers = output["packages"]
    codes = output["reason_codes"]
    if sorted(answer["package"] for answer in answers) != sorted(by_id) or len(set(codes)) != len(codes):
        return None

    # Each ask: its own candidates, no repeats, at least the minimum; a known note.
    chosen = {}
    for answer in answers:
        package = by_id[answer["package"]]
        candidates = [row["vendor_id"] for row in package["candidates"]]
        ask = answer["ask"]
        if len(set(ask)) != len(ask) or not set(ask) <= set(candidates) or len(ask) < package["min_ask"]:
            return None
        if answer["note"] not in {note.value for note in _Note}:
            return None
        chosen[package["package"]] = _ask_row(package, ask, answer["note"])

    # The model's codes must hold for its own decision; the outcome carries every code that holds.
    decision = {"packages": [chosen[package] for package in by_id]}
    true_codes = _ask_codes(facts, decision)
    if not set(codes) <= {code.value for code in true_codes}:
        return None
    return decision, [code.value for code in _AskCode if code in true_codes]


def _ask_row(package: dict, ask: list[str], note: str) -> dict:
    """One decided row: ask and left_out in id order, both from code's candidate list."""
    candidates = [row["vendor_id"] for row in package["candidates"]]
    return {
        "package": package["package"],
        "ask": [vendor for vendor in candidates if vendor in ask],
        "note": note,
        "left_out": [vendor for vendor in candidates if vendor not in ask],
    }


def _ask_rules(facts: dict) -> tuple[dict, list[str]]:
    """Ask every candidate with the standard note: today's _pick_vendors."""
    rows = [_ask_row(p, [row["vendor_id"] for row in p["candidates"]], _Note.STANDARD.value) for p in facts["packages"]]
    decision = {"packages": rows}
    true_codes = _ask_codes(facts, decision)
    return decision, [code.value for code in _AskCode if code in true_codes]


def _ask_codes(facts: dict, decision: dict) -> set[_AskCode]:
    """The codes that hold for a decision: who was left out, and why they might have been."""
    asked = {row["package"]: set(row["ask"]) for row in decision["packages"]}
    found = set()
    for package in facts["packages"]:
        left = [row for row in package["candidates"] if row["vendor_id"] not in asked.get(package["package"], set())]
        found.add(_AskCode.ASKED_FEWER if left else _AskCode.ASKED_ALL)
        if any(row["licence"] == _Licence.EXPIRING_SOON.value for row in left):
            found.add(_AskCode.LEFT_OUT_EXPIRING_LICENCE)
        if any(row["reply_record"] == systems.ReplyRecord.MIXED.value for row in left):
            found.add(_AskCode.LEFT_OUT_MIXED_RECORD)
        if any(row["reply_record"] == systems.ReplyRecord.NEW.value for row in left):
            found.add(_AskCode.LEFT_OUT_NEW_VENDOR)
        if any(row["licence"] == _Licence.VALID.value and row["reply_record"] == systems.ReplyRecord.RELIABLE.value for row in left):
            found.add(_AskCode.LEFT_OUT_RELIABLE_VALID)
        if len(package["candidates"]) < _MOST_QUOTES:
            found.add(_AskCode.FEWER_THAN_THREE_CANDIDATES)
    return found


# ---- code's checks and rules: chase ---------------------------------------------------------------


def _validate_chase(facts: dict, output: dict) -> tuple[dict, list[str]] | None:
    """The decision if every entry is on code's list, none repeats, each short package gets at least
    its minimum, and every reason code holds; else None. The model's order is kept: it is the order to chase in."""
    entries = output["chase"]
    codes = output["reason_codes"]
    listed = {(row["vendor_id"], package["package"]) for package in facts["packages"] for row in package["targets"]}
    pairs = [(entry["vendor_id"], entry["package"]) for entry in entries]
    if len(set(pairs)) != len(pairs) or not set(pairs) <= listed or len(set(codes)) != len(codes):
        return None

    # Each short package: at least its minimum.
    for package in facts["packages"]:
        if sum(chosen == package["package"] for _, chosen in pairs) < package["min_chase"]:
            return None

    decision = _chase_decision(facts, pairs)
    true_codes = _chase_codes(facts, decision)
    if not set(codes) <= {code.value for code in true_codes}:
        return None
    return decision, [code.value for code in _ChaseCode if code in true_codes]


def _chase_rules(facts: dict) -> tuple[dict, list[str]]:
    """The whole list, in quotes_apply's order."""
    decision = _chase_decision(facts, [(row["vendor_id"], p["package"]) for p in facts["packages"] for row in p["targets"]])
    true_codes = _chase_codes(facts, decision)
    return decision, [code.value for code in _ChaseCode if code in true_codes]


def _chase_decision(facts: dict, pairs: list[tuple[str, str]]) -> dict:
    """chase in the given order; skipped is the rest of code's list, in code's order."""
    listed = [(row["vendor_id"], p["package"]) for p in facts["packages"] for row in p["targets"]]
    return {
        "chase": [{"vendor_id": vendor, "package": package} for vendor, package in pairs],
        "skipped": [{"vendor_id": vendor, "package": package} for vendor, package in listed if (vendor, package) not in pairs],
    }


def _chase_codes(facts: dict, decision: dict) -> set[_ChaseCode]:
    """The codes that hold for a decision: who was skipped, and whether they had replied."""
    chased = {(entry["vendor_id"], entry["package"]) for entry in decision["chase"]}
    found = set()
    for package in facts["packages"]:
        skipped = [row for row in package["targets"] if (row["vendor_id"], package["package"]) not in chased]
        found.add(_ChaseCode.CHASED_FEWER if skipped else _ChaseCode.CHASED_ALL)
        if any(not row["replied"] for row in skipped):
            found.add(_ChaseCode.SKIPPED_NO_REPLY)
        if any(row["replied"] for row in skipped):
            found.add(_ChaseCode.SKIPPED_PROBLEM_QUOTE)
    return found
