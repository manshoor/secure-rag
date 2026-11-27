"""Evaluation specialist: for each package short of valid quotes, recommend extend or drop.

    run state ─► evaluation_facts (code: ids, counts, codes) ─► agent, two read-only tools
        short_packages   the short packages with their counts
        package_quotes   one short package: invited vendors, and each quote's problem codes
    ─► finish: per short package {package, recommend, problems_that_matter} ─► _validate (code)

A recommendation shown beside the quotes gate; procurement still decides there.
Code alone decides quote validity, arithmetic and the winner (procurement_agent).

The rule (also the scripted stand-in and every fallback): extend when the round
was not extended yet, at least one invited vendor has no valid quote, and the
invited vendors are at least as many as the quotes needed (or needed is unknown);
else drop. It mirrors quotes_apply: an extension chases exactly those vendors,
once, and cannot help when too few were invited. The validator refuses an extend
the rule could not give.

A run state marked counts-only is private here, whatever the caller passes.
"""

import copy
from collections import Counter
from collections.abc import Callable
from enum import Enum
from typing import Any

from ...boq.quotes import Problem
from ...core.agent_runtime import AgentSpec, RunPrivacy, ScriptedClient, Tool
from ..procurement_agent import _UNREADABLE_FILE, PkgStatus, _valid_by_package
from ..workflow import Privacy
from ._base import _PACKAGE_ID, Status, _each, _holds, _one_of, _rows, _shaped, run_specialist

_AGENT_NAME = "evaluation"
_SHORT_TOOL = "short_packages"
_QUOTES_TOOL = "package_quotes"

# Limits: the script needs one call per short package plus one; the rest is room for a live run.
_MAX_TURNS = 5
_SPARE_CALLS = 4
_MAX_TOKENS = 40_000
_MAX_COST_USD = 0.30

# Problem codes a quote may carry, most basic first; anything else never reaches the model.
_KNOWN_PROBLEMS = (*(problem.value for problem in Problem), _UNREADABLE_FILE)

_SYSTEM = (
    "You help procurement at a gate. A quote round has closed and some packages have too few valid quotes. "
    "For each short package, recommend extend (one more round, chasing invited vendors without a valid quote) "
    "or drop (the lines stay unpriced on the RFQ list). A round can be extended only once. "
    "Rank all of the package's problem codes, the ones that matter most first. Use the tools to read the counts and codes. "
    "Code decides validity, prices and the winner; you only recommend."
)


class _Recommend(str, Enum):
    EXTEND = "extend"
    DROP = "drop"


class _ReasonCode(str, Enum):
    """Why, as codes. Listed in this order whenever the rules give them."""

    VENDORS_NOT_REPLIED = "vendors_not_replied"  # an invited vendor sent nothing
    FIXABLE_PROBLEMS = "fixable_problems"  # an invited vendor sent a quote with problems
    ALREADY_EXTENDED = "already_extended"  # the round was extended once already
    NO_VENDOR_LEFT = "no_vendor_left"  # every invited vendor has a valid quote: no one to chase
    ALL_INVALID = "all_invalid"  # not one valid quote
    TOO_FEW_VENDORS = "too_few_vendors"  # fewer vendors invited than quotes needed: extending cannot help


# ---- entry ----------------------------------------------------------------------------------------


def evaluate_shortfall(state: dict, client: Any | None, *, privacy: RunPrivacy, case_id: str | None = None) -> dict:
    """The outcome for the short packages of a closed round (see _base.outcome).

    client None means the scripted stand-in; a ScriptedClient given is scripted too.
    A counts-only state runs as PRIVATE: a caller can close an open state, never open
    a private one. Raises ValueError when no package is short.
    """
    facts = evaluation_facts(state)
    short = [package for package in facts["packages"] if package["status"] == PkgStatus.SHORT.value]
    if not short:
        raise ValueError("no package is short of valid quotes")

    # The state itself decides privacy too: this gate does not depend on the thread.
    if state.get("privacy") == Privacy.COUNTS_ONLY.value:
        privacy = RunPrivacy.PRIVATE

    # The rules decide offline, privately and on any failure.
    decision, codes = _rules(facts, short)
    if client is None:
        client = ScriptedClient.from_steps(_script(short), {**decision, "reason_codes": codes, "reason": ""})
    success = Status.SCRIPTED if isinstance(client, ScriptedClient) else Status.AGENT

    ids = [package["package"] for package in short]
    task = {"round": facts["round"], "extended": facts["extended"], "short_packages": ids}
    return run_specialist(
        _spec(facts, short),
        task,
        _authorizer(set(ids)),
        client,
        privacy=privacy,
        success=success,
        rules=lambda: (copy.deepcopy(decision), list(codes)),
        validate=lambda output: _validate(facts, output),
        case_id=case_id,
    )


def evaluation_facts(state: dict) -> dict:
    """Ids, counts, statuses and problem codes from the run state: never names, refs or file text.

    Figures come from what procurement_agent already computed: the needed count
    from _compare's comparison rows, valid quotes from _valid_by_package. Vendor
    ids come only from the invited list; a quote from anyone else is counted, not named.
    """
    # What code already computed: valid quotes now, needed counts from the last comparison.
    agent = state["agent"]
    valid = _valid_by_package(state["quotes"])
    needed = {row["package"]: row.get("needed") for row in state.get("comparison", [])}

    # Per package: counts, invited ids, and each counted quote as codes.
    packages = []
    for package in state["packages"]:
        invited = list(package["vendors"])
        counted = [quote for quote in state["quotes"] if quote["counted"] and quote["package"] == package["id"]]
        senders = {quote["vendor_id"] for quote in counted}
        valid_from = {quote["vendor_id"] for quote in valid.get(package["id"], [])}
        packages.append({
            "package": package["id"],
            "status": package["status"],
            "valid_quotes": len(valid.get(package["id"], [])),
            "needed": needed.get(package["id"]),
            "invited": invited,
            "replied": [vendor for vendor in invited if vendor in senders],
            "valid_from": [vendor for vendor in invited if vendor in valid_from],
            "quotes": [_quote_facts(quote, invited) for quote in counted],
        })
    return {"round": agent["round"], "extended": agent["extended"], "packages": packages}


def _quote_facts(quote: dict, invited: list[str]) -> dict:
    """One counted quote: who (invited or not), valid or not, and its known problem codes."""
    problems = [code for code in quote["problems"] if code in _KNOWN_PROBLEMS]
    return {"from_invited": quote["vendor_id"] in invited, "valid": bool(quote["valid"]), "problems": problems}


# ---- the agent ------------------------------------------------------------------------------------


def _spec(facts: dict, short: list[dict]) -> AgentSpec:
    """Schemas built for this moment: only today's short package ids are valid."""
    # The package argument may name only a package that is short now.
    ids = [package["package"] for package in short]
    by_id = {package["package"]: package for package in short}
    package_arg = {"type": "object", "properties": {"package": {"type": "string", "enum": ids}}, "required": ["package"], "additionalProperties": False}

    # Two read-only tools over facts code built; neither writes anything.
    tools = (
        Tool(_SHORT_TOOL, "The short packages with their counts.", {"type": "object", "properties": {}, "additionalProperties": False}, lambda args: _short_facts(facts, short)),
        Tool(_QUOTES_TOOL, "One short package: invited vendors and each quote's problem codes.", package_arg, lambda args: _package_facts(by_id[args["package"]])),
    )
    return AgentSpec(
        _AGENT_NAME, _SYSTEM, tools, _output_schema(ids),
        max_turns=_MAX_TURNS, max_tool_calls=len(short) + _SPARE_CALLS, max_tokens=_MAX_TOKENS, max_cost_usd=_MAX_COST_USD,
    )


def _output_schema(ids: list[str]) -> dict:
    answer = {
        "type": "object",
        "properties": {
            "package": {"type": "string", "enum": ids},
            "recommend": {"type": "string", "enum": [choice.value for choice in _Recommend]},
            "problems_that_matter": {
                "type": "array",
                "items": {"type": "string", "enum": list(_KNOWN_PROBLEMS)},
                "description": "Rank all of the package's problem codes, each once, most important first.",
            },
        },
        "required": ["package", "recommend", "problems_that_matter"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "packages": {"type": "array", "items": answer},
            "reason_codes": {"type": "array", "items": {"type": "string", "enum": [code.value for code in _ReasonCode]}},
            "reason": {"type": "string", "description": "One short sentence for procurement. No prices."},
        },
        "required": ["packages", "reason_codes", "reason"],
        "additionalProperties": False,
    }


def _short_facts(facts: dict, short: list[dict]) -> dict:
    """Counts only, per short package."""
    rows = [
        {
            "package": package["package"], "valid_quotes": package["valid_quotes"], "needed": package["needed"],
            "invited": len(package["invited"]), "replied": len(package["replied"]), "valid_from": len(package["valid_from"]),
        }
        for package in short
    ]
    return {"round": facts["round"], "extended": facts["extended"], "packages": rows}


def _package_facts(package: dict) -> dict:
    """One short package: each invited vendor's state, and each quote's codes."""
    invited = [
        {"vendor_id": vendor, "replied": vendor in package["replied"], "valid_quote": vendor in package["valid_from"]}
        for vendor in package["invited"]
    ]
    return {
        "package": package["package"], "valid_quotes": package["valid_quotes"], "needed": package["needed"],
        "invited": invited, "quotes": copy.deepcopy(package["quotes"]),
    }


def _authorizer(short_ids: set[str]) -> Callable[[str, str, dict], bool]:
    """This specialist, its two tools, and only packages that are short now."""

    def authorize(agent: str, tool: str, args: dict) -> bool:
        if agent != _AGENT_NAME:
            return False
        if tool == _SHORT_TOOL:
            return args == {}
        if tool == _QUOTES_TOOL:
            return set(args) == {"package"} and args["package"] in short_ids
        return False

    return authorize


def _script(short: list[dict]) -> list[list[tuple[str, dict]]]:
    """What the stand-in asks: the short list, then each short package."""
    return [[(_SHORT_TOOL, {})], [(_QUOTES_TOOL, {"package": package["package"]}) for package in short]]


# ---- code's checks and rules ----------------------------------------------------------------------


def _validate(facts: dict, output: dict) -> tuple[dict, list[str]] | None:
    """The decision if every short package is answered once, with its own problem codes; else None.

    Reason codes must each hold for some short package. problems_that_matter must be
    all of the package's codes, each once: the model picks only the order. extend
    must be one the rule could give. Answers come back in package order.
    """
    # Every short package once, no other; reason codes once each, and true.
    short = {package["package"]: package for package in facts["packages"] if package["status"] == PkgStatus.SHORT.value}
    answers = output["packages"]
    codes = output["reason_codes"]
    if sorted(answer["package"] for answer in answers) != sorted(short) or len(set(codes)) != len(codes):
        return None
    true_codes = {code.value for package in short.values() for code in _reasons(package, facts)}
    if not set(codes) <= true_codes:
        return None

    # Each answer: all its package's problem codes, once each; only an extend that can help.
    for answer in answers:
        package = short[answer["package"]]
        problems = answer["problems_that_matter"]
        if len(set(problems)) != len(problems) or set(problems) != _problems(package):
            return None
        if answer["recommend"] == _Recommend.EXTEND.value and not _can_extend(package, facts):
            return None

    # In package order, so the same decision always looks the same.
    by_id = {answer["package"]: answer for answer in answers}
    ordered = [_answer(package, by_id[package]["recommend"], by_id[package]["problems_that_matter"]) for package in short]
    return {"packages": ordered}, list(codes)


def _rules(facts: dict, short: list[dict]) -> tuple[dict, list[str]]:
    """Extend when an extension can help (_can_extend); else drop."""
    # One answer per short package; reason codes gathered over all of them.
    answers = []
    codes: set[_ReasonCode] = set()
    for package in short:
        choice = _Recommend.EXTEND if _can_extend(package, facts) else _Recommend.DROP
        answers.append(_answer(package["package"], choice.value, _ranked(package)))
        codes |= _reasons(package, facts)
    return {"packages": answers}, [code.value for code in _ReasonCode if code in codes]


def _can_extend(package: dict, facts: dict) -> bool:
    """Not extended yet, an invited vendor to chase, and enough vendors invited to reach the count needed."""
    if facts["extended"]:
        return False
    if not [vendor for vendor in package["invited"] if vendor not in package["valid_from"]]:
        return False
    return not _too_few_vendors(package)


def _too_few_vendors(package: dict) -> bool:
    return package["needed"] is not None and len(package["invited"]) < package["needed"]


def _reasons(package: dict, facts: dict) -> set[_ReasonCode]:
    """The reason codes that hold for one short package."""
    found = set()
    if facts["extended"]:
        found.add(_ReasonCode.ALREADY_EXTENDED)
    if len(package["replied"]) < len(package["invited"]):
        found.add(_ReasonCode.VENDORS_NOT_REPLIED)
    if any(vendor not in package["valid_from"] for vendor in package["replied"]):
        found.add(_ReasonCode.FIXABLE_PROBLEMS)
    if len(package["valid_from"]) == len(package["invited"]):
        found.add(_ReasonCode.NO_VENDOR_LEFT)
    if package["valid_quotes"] == 0:
        found.add(_ReasonCode.ALL_INVALID)
    if _too_few_vendors(package):
        found.add(_ReasonCode.TOO_FEW_VENDORS)
    return found


def _ranked(package: dict) -> list[str]:
    """Problem codes, those of invited vendors first; then by how many quotes have them; then the code order."""
    invited = Counter(code for quote in package["quotes"] if quote["from_invited"] for code in quote["problems"])
    others = Counter(code for quote in package["quotes"] if not quote["from_invited"] for code in quote["problems"])
    return sorted(_problems(package), key=lambda code: (-invited[code], -others[code], _KNOWN_PROBLEMS.index(code)))


def _problems(package: dict) -> set[str]:
    return {code for quote in package["quotes"] for code in quote["problems"]}


def _answer(package: str, recommend: str, problems: list[str]) -> dict:
    return {"package": package, "recommend": recommend, "problems_that_matter": list(problems)}


# ---- saved decisions ------------------------------------------------------------------------------


def decision_holds(decision: Any, reason_codes: Any) -> bool:
    """True if a saved decision has this specialist's shape: package ids, a recommend enum, known problem codes.

    Shape only: a decision that holds may still be wrong for the case (that is _validate's job).
    """
    packages = _rows({"package": _shaped(_PACKAGE_ID), "recommend": _one_of(_Recommend), "problems_that_matter": _each(_one_of(_KNOWN_PROBLEMS))})
    return _holds({"packages": packages}, _ReasonCode, decision, reason_codes)
