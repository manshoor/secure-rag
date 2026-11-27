"""Orchestrator agent: a plan of tasks for a BOQ case, made once and revised at triggers.

    make_plan (after matching) / revise_plan (at a trigger)
      facts (case_facts: ids, counts, statuses) ─► template (code: today's order from the
      case's next step; required kinds, optional kinds, their order) ─► agent, read-only tools
          case_facts           the facts; outcomes as statuses only
          task_kinds           this situation's template: order, required, optional, whys, paths
          outcome_codes(task)  one saved specialist outcome: status, decision ids, reason codes
      ─► finish: {tasks: [{kind, why}], paths: [{package, path}], reason} ─► _validated_plan (code)
      ─► plan: code issues ids, owners, statuses and depends_on (each new task waits for the one before);
         tasks that are not pending stay exactly as they were

    Nothing to choose (no optional task offered, no short package with two paths): no model
    call at all. Code's plan comes back at once, status nothing_to_do, by code. This wins over
    privacy: a counts-only run with nothing to choose is nothing_to_do, with a choice private.

Today's order (the code tables; "?" marks an optional task):

    review_lines? ─ price_catalogue ─ check_policy? ─ pick_vendors ─ draft_requests ─ ask_send_requests
      ─ collect_quotes ─ compare_quotes
      ─ [short: evaluate_shortfall? ─ ask_extend_or_drop
               ─ [extension: plan_chase? ─ chase_vendor (one per package) ─ collect_quotes ─ compare_quotes]]
      ─ finalize_pricing ─ hand_over ─ hold_budget ─ ask_procurement ─ ask_finance ─ write_orders

A template is that order cut at the step code says comes next (facts["next"]), with the
optional tasks just before it. The model chooses only which optional tasks run and, for
short_quotes, one path per short package. It never sets a price, a vendor, a total or an
approver, and it cannot drop, move or reassign a required task: owners come from _OWNERS.

The fixed plan (private runs, the offline stand-in and every fallback): the template with
every optional task, path chase where an extension can help, else recommend_drop.

A path is a recommendation for procurement's gate: it changes no task until part 3e maps
paths to tasks. Today the extension tasks enter whenever chase is offered for some package.

Skipping check_policy is a real choice: the wiring (3b, 3d) then sets code's quotation rule
itself, with no task, while the plan holds no pending check_policy. So at the policy step a
pending pick_vendors is allowed, and check_policy is not offered again once skipped.

Retry after a fallback: offered only when the failed task is the last task that started and
its step is still code's next step, at most once per failed task. The revision records the
answer (retry_planned or rules_accepted) with the failed id, so a decline is final.

Plans are made only for a BOQ run in agent mode (facts["entry"], facts["mode"]); a run that
ended (stopped, rejected, done) takes no trigger that plans from code's next step: code
cancels its pending tasks with CodeCause.RUN_ENDED. A person task is never replaced by nothing:
a loop gate code went past (no vendors to ask) is cancelled with why step_passed, and an
approval gate missing from the template raises ValueError (the plan is behind the state).

Saved outcomes reach the facts by a whitelist per specialist: known decision keys, each value
a code-issued id shape or a value of that specialist's enums. Any other entry is dropped and
counted in facts["outcomes_dropped"].

reason: after _base's screen, dropped when it names a vendor id not in the facts, or the
currency word AED (restated from sourcing). It can still hold names and approval claims,
so it is shown escaped and labelled, never beside a price, vendor or approver.

Facts with any privacy but "normal" run as PRIVATE here, whatever the caller passes.
"""

import copy
import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..boq.boq import LineStatus
from ..core.agent_runtime import AgentResult, AgentSpec, EndReason, RunPrivacy, ScriptedClient, Tool
from . import procurement_agent
from .procurement_agent import PkgStatus, RfqState, _allowed, _valid_by_package
from .specialists import boq_intake, evaluation, policy_check, sourcing
from .specialists._base import Status, outcome, run_specialist
from .specialists.boq_intake import weak_line_facts
from .specialists.sourcing import chase_facts
from .workflow import Entry, Privacy, RunMode, StepStatus, WfStatus

_AGENT_NAME = "orchestrator"
_FACTS_TOOL = "case_facts"
_KINDS_TOOL = "task_kinds"
_OUTCOME_TOOL = "outcome_codes"

# Limits: the script makes two calls plus one per saved outcome; the rest is room for a live run.
_MAX_TURNS = 5
_SPARE_CALLS = 4
_MAX_TOKENS = 40_000
_MAX_COST_USD = 0.30

# The most tasks a plan may hold, counting every task kept from earlier versions.
_MAX_TASKS = 40

_ID_PREFIX = "T"
_START = "start"  # the situation of make_plan
_BY_CODE = "code"  # who made a fixed plan or a code-only revision
_NO_PACKAGE = ""

# Steps and pauses of the run, as workflow.py and procurement_agent.py name them.
_LOOP_NODE = "agent"
_HOLD_STEP = "hold"
_PRICE_STEP = "boq_price"
_PROCUREMENT_GATE_STEP = "gate_procurement"
_FINANCE_GATE_STEP = "gate_finance"

# Saved outcomes: the shape of a task id (each specialist checks its own decision: decision_holds).
_TASK_ID = re.compile(rf"{_ID_PREFIX}[0-9]{{1,4}}")

# Shown with the paths in task_kinds.
_PATHS_NOTE = "A path is a recommendation for procurement's gate. It changes no task."

# reason is dropped when it names a vendor id the facts do not hold, or a price word.
_VENDOR_ID_SHAPE = re.compile(r"(?<![A-Za-z0-9])V-[A-Za-z]+-[0-9]+", re.IGNORECASE)
_PRICE_WORD = re.compile(r"(?<![A-Za-z])AED(?![A-Za-z])", re.IGNORECASE)

_EMPTY_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}

_SYSTEM = (
    "You plan a procurement case that turns a bill of quantities into approved purchase orders. "
    "Code lists the task kinds allowed now: which are required, which are optional, and their order. "
    "Send the tasks in that order: every required task, each optional task you think is worth running, "
    "and a why code from that kind's list. Each task waits for the one before it; code sets that. "
    "For each short package pick one path from those offered. Code issues ids, owners and statuses, runs "
    "the tasks, and keeps every gate for a person. You cannot set a price, a vendor, a total or an approver."
)


class TaskKind(str, Enum):
    """What a task does. A fixed list: the model can plan no other."""

    REVIEW_LINES = "review_lines"
    PRICE_CATALOGUE = "price_catalogue"
    CHECK_POLICY = "check_policy"
    PICK_VENDORS = "pick_vendors"
    DRAFT_REQUESTS = "draft_requests"
    ASK_SEND_REQUESTS = "ask_send_requests"
    COLLECT_QUOTES = "collect_quotes"
    COMPARE_QUOTES = "compare_quotes"
    EVALUATE_SHORTFALL = "evaluate_shortfall"
    ASK_EXTEND_OR_DROP = "ask_extend_or_drop"
    PLAN_CHASE = "plan_chase"
    CHASE_VENDOR = "chase_vendor"
    FINALIZE_PRICING = "finalize_pricing"
    HAND_OVER = "hand_over"
    HOLD_BUDGET = "hold_budget"
    ASK_PROCUREMENT = "ask_procurement"
    ASK_FINANCE = "ask_finance"
    WRITE_ORDERS = "write_orders"


class TaskStatus(str, Enum):
    """Where a task stands. Code sets it."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Trigger(str, Enum):
    """Why code asks the orchestrator to revise the plan."""

    SHORT_QUOTES = "short_quotes"  # a package is short of valid quotes
    REQUESTS_REFUSED = "requests_refused"  # procurement refused the quote requests
    BUDGET_SHORT = "budget_short"  # the budget cannot hold the total: the run stops
    SPECIALIST_FALLBACK = "specialist_fallback"  # a specialist ended in fallback
    TASK_NOT_ALLOWED = "task_not_allowed"  # the next planned task is not the step code allows now
    PLAN_USED_UP = "plan_used_up"  # no task is left and the case is not finished


class CodeCause(str, Enum):
    """Revisions code makes alone, with no model call: the quote loop never starts."""

    NOTHING_TO_SOURCE = "nothing_to_source"  # no line needs a quotation
    OPEN_TENDER = "open_tender"  # the catalogue total needs an open tender
    NOTHING_PRICED = "nothing_priced"  # nothing priced and nothing to quote: the run stops
    RUN_ENDED = "run_ended"  # the run stopped, was rejected or is done: nothing pending runs


class _Owner(str, Enum):
    INTAKE = "intake"
    POLICY = "policy"
    SOURCING = "sourcing"
    EVALUATION = "evaluation"
    CODE = "code"
    PROCUREMENT = "person:procurement"
    FINANCE = "person:finance"  # the finance gate: code picks finance or executive by the policy threshold


class _Why(str, Enum):
    """Why a task is in the plan, as codes. The model may send only the planning ones."""

    REQUIRED = "required"  # code requires this kind now
    WEAK_LINES = "weak_lines"  # review_lines: lines scored below the match threshold
    UNMATCHED_LINES = "unmatched_lines"  # review_lines: a weak line matched no catalogue item
    CITE_RULE = "cite_rule"  # check_policy: cite the quotation rule for the approvers
    SHORT_PACKAGE = "short_package"  # evaluate_shortfall: a package is short of valid quotes
    NO_VALID_QUOTE = "no_valid_quote"  # evaluate_shortfall: a short package has no valid quote at all
    VENDORS_TO_CHASE = "vendors_to_chase"  # plan_chase: invited vendors without a valid quote
    PROBLEM_QUOTES = "problem_quotes"  # plan_chase: a target replied with a quote that has problems
    RETRY_AFTER_FALLBACK = "retry_after_fallback"  # the specialist task again, once, after its fallback
    # Set by code when it cancels a task.
    NOTHING_TO_SOURCE = CodeCause.NOTHING_TO_SOURCE.value
    OPEN_TENDER = CodeCause.OPEN_TENDER.value
    NOTHING_PRICED = CodeCause.NOTHING_PRICED.value
    RUN_ENDED = CodeCause.RUN_ENDED.value
    REQUESTS_REFUSED = Trigger.REQUESTS_REFUSED.value
    BUDGET_SHORT = Trigger.BUDGET_SHORT.value
    STEP_PASSED = "step_passed"  # a loop gate code went past without asking (e.g. no vendors to ask)


class _Path(str, Enum):
    """A way forward for a short package, put to procurement as a recommendation."""

    CHASE = "chase"  # remind the invited vendors without a valid quote
    ASK_MORE = "ask_more"  # ask approved vendors left out at the RFQ gate (part 3e)
    RECOMMEND_DROP = "recommend_drop"


class _Need(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class _Asks(str, Enum):
    """What the model sends in a situation besides its tasks."""

    TASKS = "tasks"
    TASKS_AND_PATHS = "tasks_and_paths"  # one path per short package


class _Base(str, Enum):
    """Where a situation's template comes from."""

    REMAINING = "remaining"  # today's order from code's next step
    PENDING = "pending"  # the plan's pending tasks as they are, after an optional retry
    NOTHING = "nothing"  # no task is planned


class _PlanCode(str, Enum):
    """What the accepted plan did, as codes rebuilt by code. Listed in this order."""

    ALL_OPTIONAL_PLANNED = "all_optional_planned"
    SKIPPED_REVIEW_LINES = "skipped_review_lines"
    SKIPPED_CHECK_POLICY = "skipped_check_policy"
    SKIPPED_EVALUATE_SHORTFALL = "skipped_evaluate_shortfall"
    SKIPPED_PLAN_CHASE = "skipped_plan_chase"
    RETRY_PLANNED = "retry_planned"
    RULES_ACCEPTED = "rules_accepted"  # a fallback's rules answer stands: no retry
    PATH_CHASE = "path_chase"
    PATH_ASK_MORE = "path_ask_more"
    PATH_RECOMMEND_DROP = "path_recommend_drop"
    TASKS_CANCELLED = "tasks_cancelled"


# ---- code tables ----------------------------------------------------------------------------------

_OWNERS = {
    TaskKind.REVIEW_LINES: _Owner.INTAKE,
    TaskKind.PRICE_CATALOGUE: _Owner.CODE,
    TaskKind.CHECK_POLICY: _Owner.POLICY,
    TaskKind.PICK_VENDORS: _Owner.SOURCING,
    TaskKind.DRAFT_REQUESTS: _Owner.CODE,
    TaskKind.ASK_SEND_REQUESTS: _Owner.PROCUREMENT,
    TaskKind.COLLECT_QUOTES: _Owner.CODE,
    TaskKind.COMPARE_QUOTES: _Owner.CODE,
    TaskKind.EVALUATE_SHORTFALL: _Owner.EVALUATION,
    TaskKind.ASK_EXTEND_OR_DROP: _Owner.PROCUREMENT,
    TaskKind.PLAN_CHASE: _Owner.SOURCING,
    TaskKind.CHASE_VENDOR: _Owner.CODE,
    TaskKind.FINALIZE_PRICING: _Owner.CODE,
    TaskKind.HAND_OVER: _Owner.CODE,
    TaskKind.HOLD_BUDGET: _Owner.CODE,
    TaskKind.ASK_PROCUREMENT: _Owner.PROCUREMENT,
    TaskKind.ASK_FINANCE: _Owner.FINANCE,
    TaskKind.WRITE_ORDERS: _Owner.CODE,
}

# Specialist tasks: a fallback outcome on one of these is a trigger; it may run once more.
_SPECIALIST_KINDS = frozenset(kind for kind, owner in _OWNERS.items() if owner in {_Owner.INTAKE, _Owner.POLICY, _Owner.SOURCING, _Owner.EVALUATION})

# The whys a model may send, and per optional kind the ones it may carry (first: the fixed plan's).
_PLANNING_WHYS = (
    _Why.REQUIRED, _Why.WEAK_LINES, _Why.UNMATCHED_LINES, _Why.CITE_RULE, _Why.SHORT_PACKAGE,
    _Why.NO_VALID_QUOTE, _Why.VENDORS_TO_CHASE, _Why.PROBLEM_QUOTES, _Why.RETRY_AFTER_FALLBACK,
)
_OPTIONAL_WHYS = {
    TaskKind.REVIEW_LINES: (_Why.WEAK_LINES, _Why.UNMATCHED_LINES),
    TaskKind.CHECK_POLICY: (_Why.CITE_RULE,),
    TaskKind.EVALUATE_SHORTFALL: (_Why.SHORT_PACKAGE, _Why.NO_VALID_QUOTE),
    TaskKind.PLAN_CHASE: (_Why.VENDORS_TO_CHASE, _Why.PROBLEM_QUOTES),
}
_SKIPPED_CODES = {
    TaskKind.REVIEW_LINES: _PlanCode.SKIPPED_REVIEW_LINES,
    TaskKind.CHECK_POLICY: _PlanCode.SKIPPED_CHECK_POLICY,
    TaskKind.EVALUATE_SHORTFALL: _PlanCode.SKIPPED_EVALUATE_SHORTFALL,
    TaskKind.PLAN_CHASE: _PlanCode.SKIPPED_PLAN_CHASE,
}
_PATH_CODES = {_Path.CHASE: _PlanCode.PATH_CHASE, _Path.ASK_MORE: _PlanCode.PATH_ASK_MORE, _Path.RECOMMEND_DROP: _PlanCode.PATH_RECOMMEND_DROP}

# An optional task runs at the code step of the kind that follows it; every other kind is its own step.
_STEP_OF = {
    TaskKind.REVIEW_LINES: TaskKind.PRICE_CATALOGUE,
    TaskKind.EVALUATE_SHORTFALL: TaskKind.ASK_EXTEND_OR_DROP,
    TaskKind.PLAN_CHASE: TaskKind.CHASE_VENDOR,
}

# The loop's tools and pauses (procurement_agent) as the task kind each one runs.
_LOOP_KINDS = {
    procurement_agent.Tool.POLICY_RULE: TaskKind.CHECK_POLICY,
    procurement_agent.Tool.PICK_VENDORS: TaskKind.PICK_VENDORS,
    procurement_agent.Tool.DRAFT_RFQS: TaskKind.DRAFT_REQUESTS,
    procurement_agent.Tool.REQUEST_RFQ_APPROVAL: TaskKind.ASK_SEND_REQUESTS,
    procurement_agent.Tool.COLLECT_QUOTES: TaskKind.COLLECT_QUOTES,
    procurement_agent.Tool.COMPARE_QUOTES: TaskKind.COMPARE_QUOTES,
    procurement_agent.Tool.CHASE_VENDOR: TaskKind.CHASE_VENDOR,
    procurement_agent.Tool.ASK_PROCUREMENT: TaskKind.ASK_EXTEND_OR_DROP,
    procurement_agent.Tool.FINALIZE_PRICING: TaskKind.FINALIZE_PRICING,
    procurement_agent.Tool.FINISH: TaskKind.HAND_OVER,
}
_PAUSE_KINDS = {"gate_rfq": TaskKind.ASK_SEND_REQUESTS, "wait_quotes": TaskKind.COLLECT_QUOTES, "gate_quotes": TaskKind.ASK_EXTEND_OR_DROP}

# Today's order in segments; the short and extension segments enter only when they apply.
_R, _O = _Need.REQUIRED, _Need.OPTIONAL
_INTAKE = ((TaskKind.REVIEW_LINES, _O), (TaskKind.PRICE_CATALOGUE, _R))
_SOURCING = (
    (TaskKind.CHECK_POLICY, _O), (TaskKind.PICK_VENDORS, _R), (TaskKind.DRAFT_REQUESTS, _R),
    (TaskKind.ASK_SEND_REQUESTS, _R), (TaskKind.COLLECT_QUOTES, _R), (TaskKind.COMPARE_QUOTES, _R),
)
_SHORT = ((TaskKind.EVALUATE_SHORTFALL, _O), (TaskKind.ASK_EXTEND_OR_DROP, _R))
_EXTENSION = ((TaskKind.PLAN_CHASE, _O), (TaskKind.CHASE_VENDOR, _R), (TaskKind.COLLECT_QUOTES, _R), (TaskKind.COMPARE_QUOTES, _R))
_TAIL = (
    (TaskKind.FINALIZE_PRICING, _R), (TaskKind.HAND_OVER, _R), (TaskKind.HOLD_BUDGET, _R),
    (TaskKind.ASK_PROCUREMENT, _R), (TaskKind.ASK_FINANCE, _R), (TaskKind.WRITE_ORDERS, _R),
)
# Kinds that appear twice in today's order: the second time after an extension.
_ROUND_KINDS = frozenset({TaskKind.COLLECT_QUOTES, TaskKind.COMPARE_QUOTES})

# Cancelled when the quote loop never starts (it runs from the policy step to the hand-over).
_LOOP_TASKS = frozenset({
    TaskKind.CHECK_POLICY, TaskKind.PICK_VENDORS, TaskKind.DRAFT_REQUESTS, TaskKind.ASK_SEND_REQUESTS,
    TaskKind.COLLECT_QUOTES, TaskKind.COMPARE_QUOTES, TaskKind.EVALUATE_SHORTFALL, TaskKind.ASK_EXTEND_OR_DROP,
    TaskKind.PLAN_CHASE, TaskKind.CHASE_VENDOR, TaskKind.FINALIZE_PRICING, TaskKind.HAND_OVER,
})
# Cancelled when the quote requests are refused: nothing can arrive or be priced from quotes.
_QUOTE_TASKS = frozenset({
    TaskKind.COLLECT_QUOTES, TaskKind.COMPARE_QUOTES, TaskKind.EVALUATE_SHORTFALL, TaskKind.ASK_EXTEND_OR_DROP,
    TaskKind.PLAN_CHASE, TaskKind.CHASE_VENDOR, TaskKind.FINALIZE_PRICING,
})
_ALL_TASKS = frozenset(TaskKind)


@dataclass(frozen=True)
class _Situation:
    base: _Base
    cancels: frozenset[TaskKind] = frozenset()  # pending tasks of these kinds become cancelled
    cancel_why: _Why | None = None
    asks: _Asks = _Asks.TASKS


_SITUATIONS = {
    _START: _Situation(_Base.REMAINING),
    Trigger.SHORT_QUOTES: _Situation(_Base.REMAINING, asks=_Asks.TASKS_AND_PATHS),
    Trigger.REQUESTS_REFUSED: _Situation(_Base.REMAINING, _QUOTE_TASKS, _Why.REQUESTS_REFUSED),
    Trigger.BUDGET_SHORT: _Situation(_Base.NOTHING, _ALL_TASKS, _Why.BUDGET_SHORT),
    Trigger.SPECIALIST_FALLBACK: _Situation(_Base.PENDING),
    Trigger.TASK_NOT_ALLOWED: _Situation(_Base.REMAINING),
    Trigger.PLAN_USED_UP: _Situation(_Base.REMAINING),
}
_CODE_CANCELS = {
    CodeCause.NOTHING_TO_SOURCE: (_LOOP_TASKS, _Why.NOTHING_TO_SOURCE),
    CodeCause.OPEN_TENDER: (_LOOP_TASKS, _Why.OPEN_TENDER),
    CodeCause.NOTHING_PRICED: (_ALL_TASKS, _Why.NOTHING_PRICED),
    CodeCause.RUN_ENDED: (_ALL_TASKS, _Why.RUN_ENDED),
}

# Gate steps a person decides: a pending one is never replaced by nothing.
_PERSON_KINDS = frozenset(kind for kind, owner in _OWNERS.items() if owner in {_Owner.PROCUREMENT, _Owner.FINANCE})

# Optional kinds a revision offers only while the plan holds one pending: a skip is final.
_SKIP_IS_FINAL = frozenset({TaskKind.CHECK_POLICY})

# Situations whose first new task must be code's next step (or pick_vendors at the policy step).
_FIRST_STEP_SITUATIONS = frozenset({_START, Trigger.TASK_NOT_ALLOWED.value, Trigger.PLAN_USED_UP.value})

# An ignored approver (workflow._gate) sets these; the run still waits at that gate.
_AWAITING = {WfStatus.AWAITING_PROCUREMENT.value: TaskKind.ASK_PROCUREMENT, WfStatus.AWAITING_FINANCE.value: TaskKind.ASK_FINANCE}

# A task that has run, or runs now: kept exactly, and a task after it may wait for it.
_STARTED = frozenset({TaskStatus.DONE.value, TaskStatus.RUNNING.value, TaskStatus.SKIPPED.value, TaskStatus.FAILED.value})


@dataclass(frozen=True)
class _Slot:
    """A place in a template: a kind, required or optional, the packages it runs for, the whys it may carry."""

    kind: TaskKind
    need: _Need
    packages: tuple[str, ...]
    whys: tuple[_Why, ...]


@dataclass(frozen=True)
class _Context:
    """One make or revise: the plan before, the facts, and what code allows."""

    plan: dict | None
    facts: dict
    situation: str
    slots: tuple[_Slot, ...]
    kept: tuple[dict, ...]  # the plan's tasks that stay, cancelled ones included, in order
    replaced: tuple[str, ...]  # ids of pending tasks the new list replaces
    cancelled: tuple[str, ...]
    last_started: str  # the id a new first task waits for, or ''
    first_number: int  # the number of the first new id
    paths: dict[str, tuple[_Path, ...]]  # per short package, the paths offered (short_quotes only)
    retry_offer: str  # the failed task's id when a retry is offered, or ''


# ---- public ---------------------------------------------------------------------------------------


def case_facts(state: dict) -> dict:
    """Ids, counts and statuses from the run state: never descriptions, names, emails, amounts or file text.

    Works at every moment a plan is made or revised. After matching (before pricing)
    the state has no packages, agent, rfq or lines: only the matching counts exist.
    A saved outcome keeps its specialist, status, decision and reason codes; never its reason.
    """
    packages = _package_facts(state)
    outcomes, dropped = _outcome_facts(state)
    return {
        "entry": state.get("entry", ""),
        "mode": state.get("mode", ""),
        "privacy": state.get("privacy", ""),
        "status": state.get("status", ""),
        "next": _next_kind(state),
        "matching": _matching_facts(state),
        "pricing": _pricing_facts(state),
        "loop": _loop_facts(state),
        "packages": packages,
        "short_packages": [row["package"] for row in packages if row["status"] == PkgStatus.SHORT.value],
        "outcomes": outcomes,
        "outcomes_dropped": dropped,
    }


def make_plan(facts: dict, client: Any | None, *, privacy: RunPrivacy, case_id: str | None = None) -> dict:
    """The first plan: {status, plan, reason_codes, reason, audit, usage}.

    client None means the scripted stand-in; a ScriptedClient given is scripted too.
    Counts-only facts run as PRIVATE. Any failure or rejected plan gives the fixed plan, status fallback.
    Raises ValueError unless the facts are a BOQ run in agent mode.
    """
    _check_case(facts)
    return _plan_run(_context(None, facts, _START), client, privacy, case_id)


def revise_plan(plan: dict, facts: dict, trigger: Trigger, client: Any | None, *, privacy: RunPrivacy, case_id: str | None = None) -> dict:
    """The plan revised at a trigger (same shape as make_plan). Raises ValueError when the trigger does not hold."""
    trigger = Trigger(trigger)
    _check_case(facts)
    _check_trigger(plan, facts, trigger)
    return _plan_run(_context(plan, facts, trigger), client, privacy, case_id)


def fixed_plan(facts: dict) -> dict:
    """Today's order with every optional task, made by code. Raises ValueError like make_plan."""
    _check_case(facts)
    context = _context(None, facts, _START)
    return _assembled(context, _rules(context)[0], _BY_CODE)


def fixed_revision(plan: dict, facts: dict, trigger: Trigger) -> dict:
    """The revision code makes at a trigger: every optional task, path chase where it can help. Raises ValueError like revise_plan."""
    trigger = Trigger(trigger)
    _check_case(facts)
    _check_trigger(plan, facts, trigger)
    context = _context(plan, facts, trigger)
    return _assembled(context, _rules(context)[0], _BY_CODE)


def revise_by_code(plan: dict, cause: CodeCause) -> dict:
    """A revision with no model call: pending tasks of the cause's kinds are cancelled with its code."""
    kinds, why = _CODE_CANCELS[CodeCause(cause)]
    kept, cancelled = [], []
    for task in plan["tasks"]:
        if task["status"] == TaskStatus.PENDING.value and TaskKind(task["kind"]) in kinds:
            kept.append({**task, "status": TaskStatus.CANCELLED.value, "why": why.value})
            cancelled.append(task["id"])
            continue
        kept.append(copy.deepcopy(task))
    version = plan["version"] + 1
    revision = {"version": version, "trigger": CodeCause(cause).value, "by": _BY_CODE, "replaced": [], "cancelled": cancelled, "paths": [], "retry": None}
    return {"version": version, "by": _BY_CODE, "tasks": kept, "revisions": [*copy.deepcopy(plan["revisions"]), revision]}


# ---- facts ----------------------------------------------------------------------------------------


def _matching_facts(state: dict) -> dict | None:
    """Counts of the scored lines while they exist (pricing empties boq_raw)."""
    raw = state.get("boq_raw") or []
    if not raw:
        return None
    weak = weak_line_facts(state)
    unmatched = sum(row["status"] == LineStatus.UNMATCHED.value for row in weak["lines"])
    return {"lines": len(raw), "weak": len(weak["lines"]), "unmatched": unmatched, "held": weak["held"]}


def _pricing_facts(state: dict) -> dict | None:
    """Line counts from the price step; never its section amounts."""
    stats = state.get("boq_stats")
    if not stats:
        return None
    return {"priced": stats["priced"], "rfq": stats["rfq"], "review": stats["review"]}


def _loop_facts(state: dict) -> dict | None:
    """The quote loop's control state, as flags and counts."""
    agent = state.get("agent")
    if not agent:
        return None
    chased = {target["package"] for target in agent["chase"]}
    return {
        "policy_set": bool(agent["policy"]),
        "picked": agent["picked"],
        "rfq": agent["rfq"],
        "round": agent["round"],
        "round_closed": agent["round_closed"],
        "compared": agent["compared"],
        "extended": agent["extended"],
        "finalized": agent["finalized"],
        "chase_left": len(agent["chase"]),
        "chase_packages": [package["id"] for package in state.get("packages", []) if package["id"] in chased],
    }


def _package_facts(state: dict) -> list[dict]:
    """Per package: status and counts. Chase targets as code's chase list (sourcing.chase_facts) builds them."""
    packages = state.get("packages") or []
    if not packages:
        return []
    valid = _valid_by_package(state.get("quotes", []))
    needed = {row["package"]: row.get("needed") for row in state.get("comparison", [])}

    # Targets exist only for short packages.
    targets: dict[str, tuple[int, int]] = {}
    if any(package["status"] == PkgStatus.SHORT.value for package in packages):
        for row in chase_facts(state)["packages"]:
            targets[row["package"]] = (len(row["targets"]), sum(target["replied"] for target in row["targets"]))

    rows = []
    for package in packages:
        count, replied = targets.get(package["id"], (0, 0))
        rows.append({
            "package": package["id"],
            "status": package["status"],
            "lines": len(package["items"]),
            "invited": len(package["vendors"]),
            "left_out": len(package.get("left_out", [])),  # written by part 3e; absent until then
            "asked_again": bool(package.get("asked_again", False)),
            "valid_quotes": len(valid.get(package["id"], [])),
            "needed": needed.get(package["id"]),
            "targets": count,
            "targets_replied": replied,
        })
    return rows


def _outcome_facts(state: dict) -> tuple[dict, int]:
    """Saved specialist outcomes that pass the whitelist, and how many were dropped. Never reason, audit or usage."""
    found, dropped = {}, 0
    for task_id, saved in (state.get("outcomes") or {}).items():
        kept = _kept_outcome(saved) if isinstance(task_id, str) and _TASK_ID.fullmatch(task_id) else None
        if kept is None:
            dropped += 1
            continue
        found[task_id] = kept
    return found, dropped


def _kept_outcome(saved: Any) -> dict | None:
    """The outcome's whitelisted fields if the envelope holds and the specialist's decision_holds; else None."""
    if not isinstance(saved, dict):
        return None
    specialist = saved.get("specialist")
    if not isinstance(specialist, str) or specialist not in _HOLDS:
        return None
    status = saved.get("status")
    if not isinstance(status, str) or status not in {status.value for status in Status}:
        return None

    # The decision and its codes: the specialist's own shape check, and the shape that specialist name saves.
    decision, codes = saved.get("decision"), saved.get("reason_codes")
    keys, holds = _HOLDS[specialist]
    if not isinstance(decision, dict) or set(decision) != keys or not holds(decision, codes):
        return None
    return {"specialist": specialist, "status": saved["status"], "decision": copy.deepcopy(decision), "reason_codes": list(codes)}


# Per specialist (by the name its outcome carries): its decision keys and its own shape check.
# Sourcing saves two shapes under two names; its check tells them apart by the keys.
_HOLDS: dict[str, tuple[frozenset[str], Callable[[Any, Any], bool]]] = {
    "boq_intake": (frozenset({"lines"}), boq_intake.decision_holds),
    "policy_check": (frozenset({"cite"}), policy_check.decision_holds),
    "sourcing_vendors": (frozenset({"packages"}), sourcing.decision_holds),
    "sourcing_chase": (frozenset({"chase", "skipped"}), sourcing.decision_holds),
    "evaluation": (frozenset({"packages"}), evaluation.decision_holds),
}


def _next_kind(state: dict) -> str:
    """The kind of the step code runs next, or '' when the run is not running."""
    # An ignored approver: the run still waits at that gate.
    if state.get("status") in _AWAITING:
        return _AWAITING[state["status"]].value
    if state.get("status") != WfStatus.RUNNING.value:
        return ""
    if not state.get("boq_stats"):
        return TaskKind.PRICE_CATALOGUE.value

    # Inside the quote loop: the tool code allows first, or the pause it waits at.
    agent = state.get("agent")
    if agent and agent["next"] == _LOOP_NODE:
        return _LOOP_KINDS[procurement_agent.Tool(_allowed(state)[0])].value
    if agent and agent["next"] in _PAUSE_KINDS:
        return _PAUSE_KINDS[agent["next"]].value

    # After the loop (or with none): the hold, then the two gates, then the orders.
    done = {step["name"] for step in state.get("steps", []) if step["status"] == StepStatus.DONE.value}
    held = _HOLD_STEP in done if agent else _PRICE_STEP in done
    if not held:
        return TaskKind.HOLD_BUDGET.value
    if _FINANCE_GATE_STEP in done:
        return TaskKind.WRITE_ORDERS.value
    if _PROCUREMENT_GATE_STEP in done:
        return TaskKind.ASK_FINANCE.value
    return TaskKind.ASK_PROCUREMENT.value


# ---- situations and templates ---------------------------------------------------------------------


def _check_case(facts: dict) -> None:
    """The orchestrator plans only a BOQ run in agent mode."""
    if facts.get("entry") != Entry.BOQ.value or facts.get("mode") != RunMode.AGENT.value:
        raise ValueError("the orchestrator plans only a BOQ run in agent mode")


def _check_trigger(plan: dict, facts: dict, trigger: Trigger) -> None:
    """Code's own check that the trigger fired. budget_short cannot be checked from counts."""
    pending = [task for task in plan["tasks"] if task["status"] == TaskStatus.PENDING.value]

    # A run that ended has no next step to plan from: code cancels with RUN_ENDED instead.
    if trigger is not Trigger.BUDGET_SHORT and not facts["next"]:
        raise ValueError("the run ended: cancel its pending tasks with CodeCause.RUN_ENDED")
    if trigger is Trigger.SHORT_QUOTES and not facts["short_packages"]:
        raise ValueError("no package is short of valid quotes")
    if trigger is Trigger.REQUESTS_REFUSED and (facts["loop"] or {}).get("rfq") != RfqState.REJECTED.value:
        raise ValueError("the quote requests were not refused")
    if trigger is Trigger.SPECIALIST_FALLBACK and _failed_task(plan["tasks"], facts) is None:
        raise ValueError("no specialist task ended in fallback")
    if trigger is Trigger.TASK_NOT_ALLOWED and (not pending or _first_step_holds(pending[0]["kind"], facts["next"])):
        raise ValueError("the next planned task is allowed now")
    if trigger is Trigger.PLAN_USED_UP and pending:
        raise ValueError("the plan is not used up")


def _context(plan: dict | None, facts: dict, situation: str) -> _Context:
    """What code allows for this make or revise, and what happens to the plan's tasks."""
    rule = _SITUATIONS[situation]
    tasks = plan["tasks"] if plan else []
    paths = _offered_paths(facts) if rule.asks is _Asks.TASKS_AND_PATHS else {}
    offer = _retry_offer(plan, facts) if rule.base is _Base.PENDING else ""
    slots = tuple(_template(rule, plan, facts, offer))
    planned = {slot.kind for slot in slots}

    # Tasks that are not pending stay exactly; pending ones are cancelled or replaced.
    # A person task is never replaced by nothing: a loop gate code went past is cancelled,
    # and an approval gate missing from the template is code's bug.
    kept, replaced, cancelled = [], [], []
    for task in tasks:
        kind = TaskKind(task["kind"])
        if task["status"] != TaskStatus.PENDING.value:
            kept.append(copy.deepcopy(task))
            continue
        why = rule.cancel_why if kind in rule.cancels else None
        if why is None and kind in _PERSON_KINDS - planned:
            if kind not in _LOOP_TASKS:
                raise ValueError("an approval gate task would be replaced by nothing")
            why = _Why.STEP_PASSED
        if why is None:
            replaced.append(task["id"])
            continue
        kept.append({**task, "status": TaskStatus.CANCELLED.value, "why": why.value})
        cancelled.append(task["id"])
    started = [task["id"] for task in kept if task["status"] in _STARTED]

    # New ids continue past every id this plan ever issued.
    issued = [task["id"] for task in tasks] + [old for revision in (plan or {}).get("revisions", []) for old in revision["replaced"]]
    first = 1 + max((int(task_id[len(_ID_PREFIX):]) for task_id in issued), default=0)
    return _Context(
        plan, facts, situation, slots, tuple(kept), tuple(replaced), tuple(cancelled),
        started[-1] if started else "", first, paths, offer,
    )


def _template(rule: _Situation, plan: dict | None, facts: dict, offer: str) -> list[_Slot]:
    if rule.base is _Base.NOTHING:
        return []
    if rule.base is _Base.PENDING:
        return _pending_template(plan["tasks"], offer)
    return _remaining(facts, plan)


def _pending_template(tasks: list[dict], offer: str) -> list[_Slot]:
    """After a fallback: the failed kind once more if offered (optional), then the pending tasks as planned.
    A pending retry stays a retry."""
    slots = []
    for task in tasks:
        if task["id"] == offer:
            slots.append(_Slot(TaskKind(task["kind"]), _O, (task["package"],), (_Why.RETRY_AFTER_FALLBACK,)))
    for task in tasks:
        if task["status"] != TaskStatus.PENDING.value:
            continue
        why = _Why.RETRY_AFTER_FALLBACK if task["why"] == _Why.RETRY_AFTER_FALLBACK.value else _Why.REQUIRED
        slots.append(_Slot(TaskKind(task["kind"]), _R, (task["package"],), (why,)))
    return slots


def _failed_task(tasks: list[dict], facts: dict) -> dict | None:
    """The last task that started, if its saved outcome is a specialist's fallback."""
    started = [task for task in tasks if task["status"] in _STARTED]
    if not started or started[-1]["id"] not in facts["outcomes"]:
        return None
    task = started[-1]
    fell_back = facts["outcomes"][task["id"]]["status"] == Status.FALLBACK.value
    return task if fell_back and TaskKind(task["kind"]) in _SPECIALIST_KINDS else None


def _retry_offer(plan: dict, facts: dict) -> str:
    """The failed task's id if a retry may be offered: not itself a retry, its step still next, not decided before."""
    failed = _failed_task(plan["tasks"], facts)
    if failed is None or failed["why"] == _Why.RETRY_AFTER_FALLBACK.value:
        return ""
    if _step_of(failed["kind"]) != facts["next"]:
        return ""
    decided = {revision["retry"]["failed"] for revision in plan.get("revisions", []) if revision.get("retry")}
    return "" if failed["id"] in decided else failed["id"]


def _remaining(facts: dict, plan: dict | None) -> list[_Slot]:
    """Today's order from code's next step, with the optional tasks that run at that step."""
    if not facts["next"]:
        return []
    tasks = plan["tasks"] if plan else []
    order = _case_order(facts)
    kind = TaskKind(facts["next"])
    extended = bool((facts["loop"] or {}).get("extended"))
    places = [n for n, (entry, _) in enumerate(order) if entry is kind]
    if not places:
        raise ValueError("code's next step is not in today's order")

    # Round two: collect and compare again, after the extension.
    start = places[-1] if extended and kind in _ROUND_KINDS else places[0]
    while start and order[start - 1][1] is _O and _STEP_OF.get(order[start - 1][0]) is kind:
        start -= 1

    slots = []
    for entry, need in order[start:]:
        if need is _O:
            whys = _true_whys(entry, facts)
            skipped = plan is not None and entry in _SKIP_IS_FINAL and not _is_pending(entry, tasks)
            if whys and not _ran(entry, tasks) and not skipped:
                slots.append(_Slot(entry, _O, (_NO_PACKAGE,), whys))
            continue
        packages = _chase_packages(facts) if entry is TaskKind.CHASE_VENDOR else (_NO_PACKAGE,)
        if packages:
            slots.append(_Slot(entry, _R, packages, (_Why.REQUIRED,)))
    return slots


def _case_order(facts: dict) -> tuple:
    """Today's order for this case: the short segment once a package is short, the extension where it applies."""
    loop = facts["loop"] or {}
    extended = bool(loop.get("extended"))
    order = _INTAKE + _SOURCING
    if facts["short_packages"] or extended:
        order += _SHORT
        chase_helps = any(_Path.CHASE in offered for offered in _offered_paths(facts).values())
        if extended or chase_helps:
            order += _EXTENSION
    return order + _TAIL


def _chase_packages(facts: dict) -> tuple[str, ...]:
    """One chase task per package with someone to remind: after an extension, code's chase list; before, the short packages' targets."""
    loop = facts["loop"] or {}
    if loop.get("extended"):
        return tuple(loop["chase_packages"])
    return tuple(row["package"] for row in facts["packages"] if row["status"] == PkgStatus.SHORT.value and row["targets"])


def _ran(kind: TaskKind, tasks: list[dict]) -> bool:
    """An optional task is offered once: not when one of its kind has started or finished."""
    return any(task["kind"] == kind.value and task["status"] in _STARTED for task in tasks)


def _is_pending(kind: TaskKind, tasks: list[dict]) -> bool:
    return any(task["kind"] == kind.value and task["status"] == TaskStatus.PENDING.value for task in tasks)


def _true_whys(kind: TaskKind, facts: dict) -> tuple[_Why, ...]:
    """The kind's whys that hold by code's check; none means the optional kind is not offered."""
    return tuple(why for why in _OPTIONAL_WHYS[kind] if _WHY_HOLDS[why](facts))


def _short_rows(facts: dict) -> list[dict]:
    return [row for row in facts["packages"] if row["status"] == PkgStatus.SHORT.value]


_WHY_HOLDS: dict[_Why, Callable[[dict], bool]] = {
    _Why.WEAK_LINES: lambda facts: bool(facts["matching"] and facts["matching"]["weak"]),
    _Why.UNMATCHED_LINES: lambda facts: bool(facts["matching"] and facts["matching"]["unmatched"]),
    _Why.CITE_RULE: lambda facts: True,
    _Why.SHORT_PACKAGE: lambda facts: bool(_short_rows(facts)),
    _Why.NO_VALID_QUOTE: lambda facts: any(row["valid_quotes"] == 0 for row in _short_rows(facts)),
    _Why.VENDORS_TO_CHASE: lambda facts: bool(_chase_packages(facts)),
    _Why.PROBLEM_QUOTES: lambda facts: any(row["targets_replied"] for row in _short_rows(facts)),
}


def _offered_paths(facts: dict) -> dict[str, tuple[_Path, ...]]:
    """Per short package: chase when an extension can help (as evaluation's rule), ask_more when
    approved vendors were left out and not asked again, recommend_drop always."""
    extended = bool((facts["loop"] or {}).get("extended"))
    offered = {}
    for row in _short_rows(facts):
        enough = row["needed"] is None or row["invited"] >= row["needed"]
        found = {
            _Path.CHASE: not extended and row["targets"] > 0 and enough,
            _Path.ASK_MORE: row["left_out"] > 0 and not row["asked_again"],
            _Path.RECOMMEND_DROP: True,
        }
        offered[row["package"]] = tuple(path for path in _Path if found[path])
    return offered


def _step_of(kind: str) -> str:
    return _STEP_OF.get(TaskKind(kind), TaskKind(kind)).value


def _first_step_holds(kind: str, next_kind: str) -> bool:
    """The task runs at code's next step; at the policy step pick_vendors may come first (a skipped check_policy)."""
    if _step_of(kind) == next_kind:
        return True
    return next_kind == TaskKind.CHECK_POLICY.value and kind == TaskKind.PICK_VENDORS.value


# ---- the agent ------------------------------------------------------------------------------------


def _plan_run(context: _Context, client: Any | None, privacy: RunPrivacy, case_id: str | None) -> dict:
    """Make or revise: nothing to choose first, then privacy from the facts, the scripted stand-in,
    and the run with the fixed plan behind it."""
    # Nothing to choose: code's plan at once, no model call, whatever the privacy.
    if _nothing_to_choose(context):
        decision, codes = _rules(context)
        empty = AgentResult(_AGENT_NAME, EndReason.PRIVATE)
        found = outcome(_AGENT_NAME, Status.NOTHING_TO_DO, decision, codes, "", empty)
        return {"status": found["status"], "plan": _assembled(context, decision, _BY_CODE), "reason_codes": found["reason_codes"],
                "reason": "", "audit": found["audit"], "usage": found["usage"]}

    # The facts decide privacy too, failing closed: anything but normal runs private.
    if context.facts.get("privacy") != Privacy.NORMAL.value:
        privacy = RunPrivacy.PRIVATE

    # The fixed plan decides offline, privately and on any failure.
    decision, codes = _rules(context)
    if client is None:
        client = ScriptedClient.from_steps(_script(context), {**_fixed_answer(context), "reason": ""})
    success = Status.SCRIPTED if isinstance(client, ScriptedClient) else Status.AGENT
    found = run_specialist(
        _spec(context), _task_facts(context), _authorizer(context), client,
        privacy=privacy, success=success, rules=lambda: (copy.deepcopy(decision), list(codes)),
        validate=lambda output: _validated_plan(context, output), case_id=case_id,
    )
    return {
        "status": found["status"],
        "plan": _assembled(context, found["decision"], found["status"]),
        "reason_codes": found["reason_codes"],
        "reason": _screened(found["reason"], context.facts),
        "audit": found["audit"],
        "usage": found["usage"],
    }


def _nothing_to_choose(context: _Context) -> bool:
    """No optional slot is offered and no short package has more than one path."""
    optional = any(slot.need is _O for slot in context.slots)
    return not optional and all(len(offered) <= 1 for offered in context.paths.values())


def _script(context: _Context) -> list[list[tuple[str, dict]]]:
    steps = [[(_FACTS_TOOL, {}), (_KINDS_TOOL, {})]]
    if context.facts["outcomes"]:
        steps.append([(_OUTCOME_TOOL, {"task": task_id}) for task_id in context.facts["outcomes"]])
    return steps


def _task_facts(context: _Context) -> dict:
    """The first message: the situation and the plan so far, as ids, kinds and statuses."""
    tasks = context.plan["tasks"] if context.plan else []
    situation = context.situation.value if isinstance(context.situation, Trigger) else context.situation
    return {"situation": situation, "plan": [{key: task[key] for key in ("id", "kind", "status")} for task in tasks]}


def _spec(context: _Context) -> AgentSpec:
    """Schemas built for this moment: paths only for short_quotes, outcome_codes only for saved outcomes."""
    outcomes = context.facts["outcomes"]
    tools = [
        Tool(_FACTS_TOOL, "The case: ids, counts and statuses; saved outcomes as statuses.", _EMPTY_ARGS, lambda args: _facts_view(context.facts)),
        Tool(_KINDS_TOOL, "What may be planned now: order, required and optional kinds, whys, paths.", _EMPTY_ARGS, lambda args: _kinds_view(context)),
    ]
    if outcomes:
        schema = {"type": "object", "properties": {"task": {"type": "string", "enum": list(outcomes)}}, "required": ["task"], "additionalProperties": False}
        tools.append(Tool(_OUTCOME_TOOL, "One saved specialist outcome: status, decision ids, reason codes.", schema, lambda args: {"task": args["task"], **copy.deepcopy(outcomes[args["task"]])}))
    return AgentSpec(
        _AGENT_NAME, _SYSTEM, tuple(tools), _output_schema(context),
        max_turns=_MAX_TURNS, max_tool_calls=2 + len(outcomes) + _SPARE_CALLS, max_tokens=_MAX_TOKENS, max_cost_usd=_MAX_COST_USD,
    )


def _output_schema(context: _Context) -> dict:
    task = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": [kind.value for kind in TaskKind]},
            "why": {"type": "string", "enum": [why.value for why in _PLANNING_WHYS]},
        },
        "required": ["kind", "why"],
        "additionalProperties": False,
    }
    properties = {
        "tasks": {"type": "array", "items": task},
        "reason": {"type": "string", "description": "One short sentence for procurement. No prices."},
    }
    if context.paths:
        path = {
            "type": "object",
            "properties": {
                "package": {"type": "string", "enum": list(context.paths)},
                "path": {"type": "string", "enum": sorted({path.value for offered in context.paths.values() for path in offered})},
            },
            "required": ["package", "path"],
            "additionalProperties": False,
        }
        properties["paths"] = {"type": "array", "items": path}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _facts_view(facts: dict) -> dict:
    return {**copy.deepcopy(facts), "outcomes": {task_id: saved["status"] for task_id, saved in facts["outcomes"].items()}}


def _kinds_view(context: _Context) -> dict:
    """The template as the model reads it: consecutive tasks of one kind shown once, with their count."""
    order, required, whys = [], [], {}
    for slot in context.slots:
        whys[slot.kind.value] = [why.value for why in slot.whys]
        if order and order[-1] == slot.kind.value and slot.need is _R and required and required[-1]["kind"] == slot.kind.value:
            required[-1]["count"] += len(slot.packages)
            continue
        order.append(slot.kind.value)
        if slot.need is _R:
            required.append({"kind": slot.kind.value, "count": len(slot.packages)})
    view = {
        "situation": _task_facts(context)["situation"],
        "order": order,
        "required": required,
        "optional": [slot.kind.value for slot in context.slots if slot.need is _O],
        "whys": whys,
        "most_tasks": _MAX_TASKS - len(context.kept),
    }
    if context.paths:
        view["paths"] = {package: [path.value for path in offered] for package, offered in context.paths.items()}
        view["paths_note"] = _PATHS_NOTE
    return view


def _authorizer(context: _Context) -> Callable[[str, str, dict], bool]:
    """The orchestrator, its tools, and only saved outcomes' tasks."""
    outcomes = set(context.facts["outcomes"])

    def authorize(agent: str, tool: str, args: dict) -> bool:
        if agent != _AGENT_NAME:
            return False
        if tool in (_FACTS_TOOL, _KINDS_TOOL):
            return args == {}
        if tool == _OUTCOME_TOOL:
            return set(args) == {"task"} and args["task"] in outcomes
        return False

    return authorize


def _screened(reason: str, facts: dict) -> str:
    """reason dropped if it names a vendor id the facts do not hold, or AED."""
    text = unicodedata.normalize("NFKC", reason)
    known = {match.upper() for match in _VENDOR_ID_SHAPE.findall(json.dumps(facts))}
    named = {match.upper() for match in _VENDOR_ID_SHAPE.findall(text)}
    if named - known or _PRICE_WORD.search(text):
        return ""
    return reason


# ---- code's checks and the fixed plan -------------------------------------------------------------


def _validated_plan(context: _Context, output: dict) -> tuple[dict, list[str]] | None:
    """The decision if the tasks fit the template (kinds, order, counts), each waits for the one before it,
    each why fits its place and holds, the plan stays within _MAX_TASKS and the paths are offered; else None."""
    tasks = output.get("tasks")
    if not isinstance(tasks, list) or not all(_well_formed(task) for task in tasks):
        return None
    if len(context.kept) + len(tasks) > _MAX_TASKS:
        return None

    # Kinds in the template's order and counts; then whys.
    places = _walk(context.slots, tasks)
    if places is None:
        return None
    if context.situation in _FIRST_STEP_SITUATIONS and tasks and not _first_step_holds(tasks[0]["kind"], context.facts["next"]):
        return None
    if any(task["why"] not in {why.value for why in slot.whys} for task, (slot, _) in zip(tasks, places)):
        return None
    paths = _checked_paths(context, output.get("paths", []))
    if paths is None:
        return None
    decision = _decision(context, [(task, slot, package) for task, (slot, package) in zip(tasks, places)], paths)
    return decision, _plan_codes(context, places, paths)


def _well_formed(task: Any) -> bool:
    """A known kind and a planning why, nothing else: what the schema already holds, checked again."""
    if not isinstance(task, dict) or set(task) != {"kind", "why"}:
        return False
    if not isinstance(task["kind"], str) or not isinstance(task["why"], str):
        return False
    return task["kind"] in {kind.value for kind in TaskKind} and task["why"] in {why.value for why in _PLANNING_WHYS}


def _walk(slots: tuple[_Slot, ...], tasks: list[dict]) -> list[tuple[_Slot, str]] | None:
    """Each task's place in the template: optional slots taken or left, required ones filled exactly; else None."""
    places: list[tuple[_Slot, str]] = []
    n = 0
    for slot in slots:
        if slot.need is _O:
            if n < len(tasks) and tasks[n]["kind"] == slot.kind.value:
                places.append((slot, slot.packages[0]))
                n += 1
            continue
        for package in slot.packages:
            if n >= len(tasks) or tasks[n]["kind"] != slot.kind.value:
                return None
            places.append((slot, package))
            n += 1
    return places if n == len(tasks) else None


def _checked_paths(context: _Context, paths: Any) -> list[dict] | None:
    """One offered path per short package, back in package order; none outside short_quotes."""
    if not isinstance(paths, list) or not all(isinstance(row, dict) and set(row) == {"package", "path"} for row in paths):
        return None
    if not all(isinstance(row["package"], str) and isinstance(row["path"], str) for row in paths):
        return None
    chosen = {row["package"]: row["path"] for row in paths}
    if len(chosen) != len(paths) or set(chosen) != set(context.paths):
        return None
    if any(chosen[package] not in {path.value for path in offered} for package, offered in context.paths.items()):
        return None
    return [{"package": package, "path": chosen[package]} for package in context.paths]


def _decision(context: _Context, placed: list[tuple[dict, _Slot, str]], paths: list[dict]) -> dict:
    """New tasks with code's ids, owners and statuses. Each waits for the one before; the first for the last task that started."""
    ids = [f"{_ID_PREFIX}{context.first_number + n}" for n in range(len(placed))]
    tasks = []
    for n, (task, slot, package) in enumerate(placed):
        before = ids[n - 1] if n else context.last_started
        tasks.append({
            "id": ids[n],
            "kind": slot.kind.value,
            "owner": _OWNERS[slot.kind].value,
            "status": TaskStatus.PENDING.value,
            "depends_on": [before] if before else [],
            "why": task["why"],
            "package": package,
        })
    return {"tasks": tasks, "paths": paths}


def _plan_codes(context: _Context, places: list[tuple[_Slot, str]], paths: list[dict]) -> list[str]:
    """What the plan did with each optional slot and path, in enum order."""
    taken = {id(slot) for slot, _ in places}
    found = set()
    retry = (_Why.RETRY_AFTER_FALLBACK,)
    plain = [slot for slot in context.slots if slot.need is _O and slot.whys != retry]
    for slot in plain:
        if id(slot) not in taken:
            found.add(_SKIPPED_CODES[slot.kind])
    if plain and not found:
        found.add(_PlanCode.ALL_OPTIONAL_PLANNED)
    for slot in context.slots:
        if slot.need is _O and slot.whys == retry:
            found.add(_PlanCode.RETRY_PLANNED if id(slot) in taken else _PlanCode.RULES_ACCEPTED)
    found.update(_PATH_CODES[_Path(row["path"])] for row in paths)
    if context.cancelled:
        found.add(_PlanCode.TASKS_CANCELLED)
    return [code.value for code in _PlanCode if code in found]


def _fixed_answer(context: _Context) -> dict:
    """The fixed plan in the model's shape: every slot, the first why that holds."""
    tasks = []
    for slot in context.slots:
        for _ in slot.packages:
            tasks.append({"kind": slot.kind.value, "why": slot.whys[0].value})
    answer = {"tasks": tasks}
    if context.paths:
        answer["paths"] = [{"package": package, "path": _fixed_path(offered).value} for package, offered in context.paths.items()]
    return answer


def _fixed_path(offered: tuple[_Path, ...]) -> _Path:
    return _Path.CHASE if _Path.CHASE in offered else _Path.RECOMMEND_DROP


def _rules(context: _Context) -> tuple[dict, list[str]]:
    """The fixed decision, built directly: code's own plan never depends on passing the model's checks."""
    answer = _fixed_answer(context)
    places = _walk(context.slots, answer["tasks"])
    paths = answer.get("paths", [])
    decision = _decision(context, [(task, slot, package) for task, (slot, package) in zip(answer["tasks"], places)], paths)
    return decision, _plan_codes(context, places, paths)


def _assembled(context: _Context, decision: dict, by: str) -> dict:
    """The plan: kept tasks in place, then the new ones; a revision bumps the version and is recorded."""
    tasks = [*copy.deepcopy(list(context.kept)), *copy.deepcopy(decision["tasks"])]
    if len(tasks) > _MAX_TASKS:
        raise ValueError("code's own plan passes the task cap")
    if context.plan is None:
        return {"version": 1, "by": by, "tasks": tasks, "revisions": []}
    version = context.plan["version"] + 1
    revision = {
        "version": version, "trigger": context.situation.value, "by": by,
        "replaced": list(context.replaced), "cancelled": list(context.cancelled), "paths": copy.deepcopy(decision["paths"]),
        "retry": _retry_record(context, decision),
    }
    return {"version": version, "by": by, "tasks": tasks, "revisions": [*copy.deepcopy(context.plan["revisions"]), revision]}


def _retry_record(context: _Context, decision: dict) -> dict | None:
    """When a retry was offered: the failed id and the answer, so a decline is final."""
    if not context.retry_offer:
        return None
    planned = any(task["why"] == _Why.RETRY_AFTER_FALLBACK.value for task in decision["tasks"])
    answer = _PlanCode.RETRY_PLANNED if planned else _PlanCode.RULES_ACCEPTED
    return {"failed": context.retry_offer, "decision": answer.value}
