"""Cross-department workflow: HR -> procurement -> finance -> purchase order (LangGraph).

    request ─► understand ─► hr_check ─► procurement ──┐
                                                      │  (budget hold for the total:
    BOQ file ─► boq_parse ─► boq_match ─► boq_price ──┤   a shortfall stops here)
                                                      ▼
                 gate_procurement (human: procurement, same company, not the requester)
                                                      │
                 finance (hold re-checked; approver from policy.py thresholds)
                                                      │
                 gate_finance (human: finance up to AED 100,000, executive above)
                                                      │
                 execute ─► hold becomes spend ─► outbox: purchase order (+ IT ticket)

A hold keeps two open runs from spending the same budget. A rejection or a
stop after pricing releases it.

Every number (quantities, prices, totals, budget, thresholds, approver roles)
comes from systems.py, policy.py and the parsed BOQ, never from a model.
An ineligible approver is ignored and the run keeps waiting; a rejection ends
it with nothing in the outbox. Each step writes one audit line and one span.
"""

import fcntl
import functools
import json
import re
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import run_index, systems
from ..core import audit, config, db, tracing
from .agent import APPROVAL_CHOICES, Decision, open_checkpointer
from ..rag.answer import Composer, ask
from ..rag.guard import injection_findings
from ..core.identity import User, company_label, get_user
from .intake import Intake, Summarizer, TemplateSummarizer
from ..core.llm import Usage
from .policy import EXECUTIVE_APPROVAL_AED, budget_approver_roles, quotes_required
from ..rag.store import KnowledgeStore

_TRACE_TAG = "workflow"
_OUTBOX_FILE = "outbox.jsonl"
_POLICY_DOC = "alpha-procurement-policy"
_PROCUREMENT_ROLES = frozenset({"procurement"})
_QUOTE_PRICE_ROLES = frozenset({"procurement", "finance", "executive"})
_PRICE_KEYS = frozenset({"total", "lines"})
_QUOTE_LINE_PRICE_KEYS = frozenset({"unit_price", "total", "quote_ref"})


class Entry(str, Enum):
    """How a run started; each entry has its own allowed starter role and budget line."""

    REQUEST = "request"
    BOQ = "boq"


# Who may start each entry, and which department's budget pays.
_ENTRY_ROLE = {Entry.REQUEST: "hr", Entry.BOQ: "site_engineer"}
_ENTRY_DEPARTMENT = {Entry.REQUEST: "hr", Entry.BOQ: "projects"}


class RunMode(str, Enum):
    """Who picks the steps: fixed graph (classic) or the procurement agent loop (BOQ only)."""

    CLASSIC = "classic"
    AGENT = "agent"


class Gate(str, Enum):
    PROCUREMENT = "procurement"
    FINANCE = "finance"
    RFQ = "rfq"  # agent mode: send the quote requests
    QUOTES = "quotes"  # agent mode: too few valid quotes; extend once or drop the lines


class Event(str, Enum):
    """Agent mode: what a vendor mailbox or a scheduler reports while a run waits for quotes."""

    QUOTES_DELIVERED = "quotes_delivered"
    DEADLINE_PASSED = "deadline_passed"


class WfStatus(str, Enum):
    RUNNING = "running"
    AWAITING_PROCUREMENT = "awaiting_procurement"
    AWAITING_FINANCE = "awaiting_finance"
    AWAITING_RFQ = "awaiting_rfq"
    AWAITING_QUOTES = "awaiting_quotes"
    AWAITING_QUOTE_DECISION = "awaiting_quote_decision"
    STOPPED = "stopped"
    REJECTED = "rejected"
    DONE = "done"


class Privacy(str, Enum):
    """What audit lines and traces may hold for a run."""

    NORMAL = "normal"  # ids, amounts, statuses; text only with RAG_TRACE_CONTENT=full
    COUNTS_ONLY = "counts_only"  # private files: counts and ids only, whatever the setting


class StepStatus(str, Enum):
    DONE = "done"
    STOPPED = "stopped"
    REJECTED = "rejected"
    WAITING = "waiting"


_AWAITING = {
    Gate.PROCUREMENT: WfStatus.AWAITING_PROCUREMENT,
    Gate.FINANCE: WfStatus.AWAITING_FINANCE,
    Gate.RFQ: WfStatus.AWAITING_RFQ,
    Gate.QUOTES: WfStatus.AWAITING_QUOTE_DECISION,
}
_WAITING = frozenset(status.value for status in _AWAITING.values())
_QUOTE_CHOICES = frozenset({Decision.EXTEND, Decision.DROP})
_MAX_LIST_ROWS = 50  # run list length; the newest first

# Apply node -> the gate whose saved decision it finishes (wait_quotes_apply: procurement, no gate).
_APPLY_GATES = {
    "gate_procurement_apply": Gate.PROCUREMENT,
    "gate_finance_apply": Gate.FINANCE,
    "gate_rfq_apply": Gate.RFQ,
    "gate_quotes_apply": Gate.QUOTES,
}

# Approver ids that belong to an agent, never a person: they may not decide any gate.
AGENT_PREFIX = "agent:"
# Graph steps per invoke; an agent stage runs one planner step per graph step.
_RECURSION_LIMIT = 200
_NOTHING_PRICED = "no line could be priced from the catalogue"
_ZERO_TOTAL = "the run total is AED 0.00"


class WfState(TypedDict, total=False):
    run_id: str
    entry: str
    requester: str
    company: str
    department: str
    text: str  # the request sentence (request entry only)
    boq_file: str  # stored upload path (BOQ entry only)
    privacy: str
    fields: dict
    hires: list[dict]
    lines: list[dict]  # priced lines; amounts are strings so checkpoints keep them exact
    rfq: list[dict]  # lines nobody may price: need a quotation
    review: list[dict]  # lines procurement must look at before approving
    boq_raw: list[dict]  # parsed BOQ lines with match results
    boq_stats: dict
    total: str
    budget_remaining: str
    refused_vendors: list[str]
    flags: list[str]
    policy_question: str
    approver_roles: list[str]
    approvals: dict
    summaries: dict
    usage: dict
    steps: list[dict]
    status: str
    reason: str
    stopped_at: str

    # Agent mode only. Missing in classic runs and in runs saved before these fields existed.
    mode: str
    packages: list[dict]  # lines grouped by category for quote requests
    quotes: list[dict]  # checked vendor quotes
    comparison: list[dict]  # winner per package
    agent_steps: list[dict]  # planner proposals and their outcomes
    events: list[dict]  # deliveries and agent-gate decisions, with who made them
    agent: dict  # agent stage control: round, counters, next node
    reply: dict  # a pause's saved input, until its apply node has acted on it
    outcomes: dict  # specialist outcomes by key (procurement_agent._outcome_key); start_stage leaves them


@dataclass
class RunView:
    """A run as one viewer may see it."""

    run_id: str
    entry: Entry
    requester: str
    status: WfStatus
    reason: str
    stopped_at: str
    pending_gate: Gate | None
    steps: list[dict]
    hires: list[dict]
    lines: list[dict]
    rfq: list[dict]
    review: list[dict]
    boq_stats: dict
    total: Decimal
    budget_remaining: Decimal | None
    approver_roles: list[str]
    approvals: dict
    refused_vendors: list[str]
    flags: list[str]
    summaries: dict
    policy: dict
    cost_usd: float
    tokens: int
    private: bool = False  # counts-only run: show counts, not line text, outside the API
    mode: RunMode = RunMode.CLASSIC
    packages: list[dict] = field(default_factory=list)
    quotes: list[dict] = field(default_factory=list)  # prices only for procurement, finance, executive
    comparison: list[dict] = field(default_factory=list)
    agent_steps: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    decide_problem: str = ""  # why this viewer may not decide the waiting gate ('' if they may, or none waits)
    orders: list[dict] = field(default_factory=list)  # done runs: {vendor, lines, total} per PO, for anyone who may view
    vendor_names: dict[str, str] = field(default_factory=dict)  # vendor id -> name, for the vendors asked to quote

    @property
    def can_decide(self) -> bool:
        return self.pending_gate is not None and not self.decide_problem

    @property
    def budget_after(self) -> Decimal | None:
        """Budget left once this run's total is held or spent. budget_remaining is what was there before it.

        None when nothing is held: before the hold, or after a stop or rejection released it.
        """
        if self.budget_remaining is None or self.status in (WfStatus.STOPPED, WfStatus.REJECTED):
            return None
        return self.budget_remaining - self.total


class RunScope(str, Enum):
    """Run lists: cases I started, or gates I may decide now."""

    MINE = "mine"
    WAITING = "waiting"


@dataclass(frozen=True)
class RunSummary:
    """One row of a run list: ids, statuses and the total, never text."""

    run_id: str
    entry: str
    mode: str
    status: str
    pending_gate: str | None
    requester: str
    total: str
    updated: str


# ---- runner -------------------------------------------------------------------------


class WorkflowRunner:
    """Builds the graph once; starts, resumes and shows runs."""

    def __init__(
        self,
        store: KnowledgeStore,
        composer: Composer,
        intake: Intake,
        checkpointer: BaseCheckpointSaver | None = None,
        summarizer: Summarizer | None = None,
        boq_reader: Any = None,
        canonicalizer: Any = None,
        step_planner: Any = None,
        agent_client: Any = None,
    ) -> None:
        self._store = store
        self._composer = composer
        self._intake = intake
        self._summarizer = summarizer or TemplateSummarizer()
        self._boq_reader = boq_reader  # tests inject a reader; default is boq_parser.parse_boq
        self._canonicalizer = canonicalizer  # Claude mode only (boq.ClaudeCanonicalizer)
        self._step_planner = step_planner  # agent mode; None means procurement_agent.ScriptedPlanner
        self._agent_client = agent_client  # the specialists' tool-use client; None means each one's scripted stand-in
        self._graph = self._build(checkpointer or open_checkpointer())

        # One decision at a time per run; db.run_lock covers other processes.
        self._run_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

        # Policy citations per run and viewer access: a view must not call the model every time,
        # and a changed role or company must not be served text retrieved under the old one.
        self._policy_cache: dict[tuple, dict] = {}

    # ---- public API --------------------------------------------------------

    def start_request(self, user: User, text: str) -> RunView:
        """Start from a sentence. Raises PermissionError without the HR role."""
        return self._start(user, Entry.REQUEST, {"text": text, "privacy": Privacy.NORMAL.value})

    def start_boq(
        self, user: User, path: Path, privacy: Privacy = Privacy.NORMAL, mode: RunMode = RunMode.CLASSIC
    ) -> RunView:
        """Start from a BOQ file. Raises PermissionError without the site engineer role.

        Files under data/private/ always run counts-only, whatever the caller asked.
        """
        from ..boq.boq_parser import is_private

        if is_private(path):
            privacy = Privacy.COUNTS_ONLY
        return self._start(user, Entry.BOQ, {"boq_file": str(path), "privacy": privacy.value, "mode": mode.value})

    def decide(self, run_id: str, approver: User, decision: Decision, gate: Gate | None = None) -> RunView:
        """Feed a gate decision into a paused run.

        Raises ValueError if no gate is waiting, if `gate` is given and another
        gate is waiting (the approver saw an older state of the run), or if the
        decision is not an answer to the waiting gate (extend or drop at the
        quote gate, approve or reject everywhere else).
        """
        with self._lock_for(run_id), db.run_lock(f"wf:{run_id}"):
            waiting = _pending_gate(self._graph.get_state(_thread(run_id)))
            if waiting is None:
                raise ValueError(f"run {run_id} is not waiting for a decision")
            if gate is not None and gate is not waiting:
                raise ValueError(f"run {run_id} is waiting for {waiting.value}, not {gate.value}")
            if decision not in gate_choices(waiting):
                raise ValueError(f"the {waiting.value} gate takes {' or '.join(sorted(d.value for d in gate_choices(waiting)))}")
            command = Command(resume={"approver": approver.user_id, "decision": decision.value})
            with tracing.attributes(approver.user_id, [_TRACE_TAG], run_id), tracing.observe(
                "workflow_decide", as_type="chain", metadata={"run": run_id, "decision": decision.value}
            ) as span:
                self._graph.invoke(command, _thread(run_id))
                self._reindex(run_id)
                view = self._view(run_id, approver)
                reason = None if view.private else tracing.content(view.reason)  # a private run sends no text
                span.update(metadata={"status": view.status.value, "reason": reason})
                return view

    def resume(self, run_id: str, user: User) -> RunView:
        """Finish a run that stopped between steps (crash, failed write). Steps are idempotent.

        Raises LookupError if the user may not see the run, and ValueError if it is
        waiting for a person or has ended.
        """
        if not self.may_view(run_id, user):
            raise LookupError("run not found")
        with self._lock_for(run_id), db.run_lock(f"wf:{run_id}"):
            snapshot = self._graph.get_state(_thread(run_id))
            if _paused(snapshot) or not snapshot.next:
                raise ValueError(f"run {run_id} has nothing to resume")
            with tracing.attributes(user.user_id, [_TRACE_TAG], run_id), tracing.observe(
                "workflow_resume", as_type="chain", metadata={"run": run_id, "next": list(snapshot.next)}
            ):
                self._graph.invoke(None, _thread(run_id))
                self._reindex(run_id)
            return self._view(run_id, user)

    def deliver(self, run_id: str, event: Event, user: User, quotes: list[tuple[str, Path]] | None = None) -> RunView:
        """Feed vendor replies or a passed deadline into a run waiting for quotes.

        `quotes` pairs each file with the vendor id that sent it (the mailbox
        sender, not what the file claims). Raises LookupError if the user may not
        see the run, ValueError if it is not waiting for quotes, and PermissionError
        for anyone who may not act for procurement on it.
        """
        if not self.may_view(run_id, user):
            raise LookupError("run not found")
        with self._lock_for(run_id), db.run_lock(f"wf:{run_id}"):
            snapshot = self._graph.get_state(_thread(run_id))
            if not _pending_wait(snapshot):
                raise ValueError(f"run {run_id} is not waiting for quotes")
            problem = _gate_problem(snapshot.values, user, _PROCUREMENT_ROLES)
            if problem:
                raise PermissionError(problem)
            if event is Event.QUOTES_DELIVERED and not quotes:
                raise ValueError("a delivery needs at least one quote file")
            files = [{"sender": sender, "path": str(path)} for sender, path in (quotes or [])]
            command = Command(resume={"event": event.value, "by": user.user_id, "quotes": files})
            with tracing.attributes(user.user_id, [_TRACE_TAG], run_id), tracing.observe(
                "workflow_deliver", as_type="chain", metadata={"run": run_id, "event": event.value, "files": len(files)}
            ):
                self._graph.invoke(command, _thread(run_id))
                self._reindex(run_id)
            return self._view(run_id, user)

    def view(self, run_id: str, viewer: User) -> RunView:
        """The run as this viewer may see it. Raises LookupError for anyone else."""
        if not self.may_view(run_id, viewer):
            raise LookupError("run not found")
        return self._view(run_id, viewer)

    def requester_of(self, run_id: str) -> str:
        """Who started a run ('' for an unknown id). For the local CLI's default viewer."""
        return self._graph.get_state(_thread(run_id)).values.get("requester", "")

    def may_view(self, run_id: str, viewer: User) -> bool:
        """Requester, anyone who decided a gate, or an eligible approver of the waiting gate."""
        snapshot = self._graph.get_state(_thread(run_id))
        values = snapshot.values
        if not values:
            return False
        if viewer.user_id == values.get("requester") or viewer.user_id in values.get("approvals", {}).values():
            return True
        if viewer.user_id in {event.get("by") for event in values.get("events", [])}:
            return True

        # A saved decision or delivery still to be finished (a crash after the pause):
        # its author may see the run and resume it, while still eligible.
        saved = values.get("reply") or {}
        author = saved.get("approver") or saved.get("by")
        if author and viewer.user_id == author and snapshot.next:
            gate = _APPLY_GATES.get(snapshot.next[0])
            roles = _gate_roles(gate, values) if gate else _PROCUREMENT_ROLES
            return not _gate_problem(values, viewer, roles)
        gate = _pending_gate(snapshot)
        if gate is not None:
            return not _gate_problem(values, viewer, _gate_roles(gate, values))
        if _pending_wait(snapshot):
            return not _gate_problem(values, viewer, _PROCUREMENT_ROLES)
        return False

    def runs_for(self, viewer: User, scope: RunScope) -> list[RunSummary]:
        """Runs from the index this viewer may see: their own, or gates they may decide now.

        Each row is checked against the live run; a stale row is shown as the run is now.
        """
        found: list[RunSummary] = []
        for row in run_index.rows():
            # Cheap check on the row first: most rows cannot match and cost no checkpoint read.
            if not _row_may_match(row, viewer, scope):
                continue
            if len(found) >= _MAX_LIST_ROWS:
                break
            run_id = row["run_id"]
            if not self.may_view(run_id, viewer):
                continue
            snapshot = self._graph.get_state(_thread(run_id))
            values = snapshot.values
            gate = _pending_gate(snapshot)
            if scope is RunScope.MINE and values["requester"] != viewer.user_id:
                continue
            if scope is RunScope.WAITING and (gate is None or _gate_problem(values, viewer, _gate_roles(gate, values))):
                continue
            live = _index_row(snapshot)
            fields = {name: live[name] for name in RunSummary.__dataclass_fields__}
            found.append(RunSummary(**{**fields, "updated": row.get("updated", live["updated"])}))
        return found

    # ---- starting ------------------------------------------------------------

    def _start(self, user: User, entry: Entry, extra: dict) -> RunView:
        if _ENTRY_ROLE[entry] not in user.roles:
            raise PermissionError(f"starting a {entry.value} run needs the {_ENTRY_ROLE[entry]} role")
        company = sorted(user.companies)[0]
        run_id = uuid.uuid4().hex[:12]
        state: WfState = {
            "run_id": run_id,
            "entry": entry.value,
            "requester": user.user_id,
            "company": company,
            "department": _ENTRY_DEPARTMENT[entry],
            "steps": [],
            "approvals": {},
            "flags": [],
            "refused_vendors": [],
            "summaries": {},
            "usage": {"input": 0, "output": 0, "cost_usd": 0.0},
            "status": WfStatus.RUNNING.value,
            **extra,
        }
        # A row before the run starts: a run that crashes on its first step still shows in "my cases".
        self._record_row(_start_row(state))
        with tracing.attributes(user.user_id, [_TRACE_TAG], run_id), tracing.observe(
            "workflow_start", as_type="chain", metadata={"run": run_id, "entry": entry.value}
        ) as span:
            self._graph.invoke(state, _thread(run_id))
            self._reindex(run_id)
            view = self._view(run_id, user)
            span.update(metadata={"status": view.status.value, "total_aed": str(view.total)})
            return view

    # ---- graph ---------------------------------------------------------------

    def _build(self, checkpointer: BaseCheckpointSaver):
        graph = StateGraph(WfState)
        nodes = {
            "understand": self._understand,
            "hr_check": self._hr_check,
            "procurement": self._procurement,
            "boq_parse": self._boq_parse,
            "boq_match": self._boq_match,
            "boq_price": self._boq_price,
            "gate_procurement": _ask(lambda state: _gate_payload(state, Gate.PROCUREMENT)),
            "gate_procurement_apply": self._gate_procurement,
            "finance": self._finance,
            "gate_finance": _ask(lambda state: _gate_payload(state, Gate.FINANCE)),
            "gate_finance_apply": self._gate_finance,
            "execute": self._execute,
            "agent": self._agent,
            "agent_hold": self._agent_hold,
        }
        from . import procurement_agent

        # Agent pauses: an ask node that only waits, then an apply node (see _ask).
        for name, payload, apply in (
            ("gate_rfq", procurement_agent.rfq_payload, procurement_agent.rfq_apply),
            ("wait_quotes", procurement_agent.wait_payload, procurement_agent.wait_apply),
            ("gate_quotes", procurement_agent.quotes_payload, functools.partial(procurement_agent.quotes_apply, self)),
        ):
            nodes[name] = _ask(payload)
            nodes[f"{name}_apply"] = apply
        for name, node in nodes.items():
            graph.add_node(name, node)

        # Two entries converge on the procurement gate.
        graph.add_conditional_edges(START, _route_entry, ["understand", "boq_parse"])
        for before, after in (
            ("understand", "hr_check"),
            ("hr_check", "procurement"),
            ("procurement", "gate_procurement"),
            ("boq_parse", "boq_match"),
            ("boq_match", "boq_price"),
            ("finance", "gate_finance"),
            ("agent_hold", "gate_procurement"),
        ):
            graph.add_conditional_edges(before, _continue_or_end(after), [after, END])

        # Agent mode: after pricing, the procurement agent loop runs until it hands over to the hold.
        graph.add_conditional_edges("boq_price", _after_price, ["agent", "gate_procurement", END])
        agent_nodes = ["agent", "gate_rfq", "wait_quotes", "gate_quotes", "agent_hold", END]
        for name in ("agent", "gate_rfq_apply", "wait_quotes_apply", "gate_quotes_apply"):
            graph.add_conditional_edges(name, _agent_next, agent_nodes)

        # Every pause: the ask node always hands its reply to its apply node.
        for name in ("gate_procurement", "gate_finance", "gate_rfq", "wait_quotes", "gate_quotes"):
            graph.add_edge(name, f"{name}_apply")
        graph.add_conditional_edges("gate_procurement_apply", _after_gate("gate_procurement", "finance"), ["gate_procurement", "finance", END])
        graph.add_conditional_edges("gate_finance_apply", _after_gate("gate_finance", "execute"), ["gate_finance", "execute", END])
        graph.add_edge("execute", END)
        return graph.compile(checkpointer=checkpointer)

    # ---- request entry ---------------------------------------------------------

    def _understand(self, state: WfState) -> WfState:
        """Sentence -> structured fields. Planted instructions are flagged, never followed."""
        fields = self._intake.parse(state["text"])
        flags = [f"possible injection in request: {', '.join(found)}" for found in [injection_findings(state["text"])] if found]
        detail = {
            "hire_ids": fields.hire_ids,
            "count": fields.count,
            "start": fields.start.isoformat() if fields.start else None,
            "items": fields.items,
            "unknown_items": len(fields.unknown_items),
            "flags": len(flags),
        }
        update: WfState = {
            "fields": {**detail, "role": fields.role, "unknown_items": fields.unknown_items, "vendor_mentions": fields.vendor_mentions},
            "flags": state["flags"] + flags,
            "usage": _add_usage(state["usage"], fields.usage),
        }
        if not fields.items:
            return {**update, **_stop(state, "understand", "the request names no catalogue item", detail)}
        if fields.unknown_items:
            return {**update, **_stop(state, "understand", f"not in the catalogue: {', '.join(fields.unknown_items)}", detail)}
        if fields.count and not fields.start:
            return {**update, **_stop(state, "understand", "a count of hires needs a start date", detail)}
        if not fields.hire_ids and not (fields.count and fields.role and fields.start):
            return {**update, **_stop(state, "understand", "the request needs hire ids, or a count, role and start date", detail)}
        return {**update, **_step(state, "understand", StepStatus.DONE, detail)}

    def _hr_check(self, state: WfState) -> WfState:
        """Hires exist, belong to the requester's company, are confirmed, and have a start date."""
        fields = state["fields"]
        company = state["company"]
        problems: list[str] = []
        hires: list[systems.Hire] = []

        # Named hires: each must exist, be confirmed and work for the requester's company.
        for hire_id in fields["hire_ids"]:
            hire = systems.hire_by_id(hire_id)
            if hire is None:
                problems.append(f"{hire_id} is not in the HR system")
            elif hire.company != company:
                problems.append(f"{hire_id} belongs to {hire.company}, not {company}")
            elif not systems.is_active(hire):
                problems.append(f"{hire_id} is {hire.status}")
            else:
                hires.append(hire)

        # Counted hires: the HR system must hold at least that many confirmed people.
        if fields["count"]:
            from datetime import date

            start = date.fromisoformat(fields["start"])
            found = systems.find_hires(company, fields["role"], start)
            if len(found) < fields["count"]:
                problems.append(f"only {len(found)} {fields['role']} start on {start.isoformat()}, not {fields['count']}")
            named = {hire.hire_id for hire in hires}
            hires.extend([hire for hire in found if hire.hire_id not in named][: max(fields["count"] - len(hires), 0)])

        detail = {"hires": len(hires), "problems": len(problems)}
        if problems:
            return _stop(state, "hr_check", "; ".join(problems), detail)
        rows = [{"id": h.hire_id, "name": h.name, "role": h.role, "start": h.start_date.isoformat()} for h in hires]
        return {"hires": rows, **_step(state, "hr_check", StepStatus.DONE, detail)}

    def _procurement(self, state: WfState) -> WfState:
        """Catalogue prices x hires; vendors only from the catalogue; budget pre-check."""
        count = Decimal(len(state["hires"]))
        priced = systems.price_items({code: count for code in state["fields"]["items"]})
        lines = [_line_row(line) for line in priced]
        total = sum((line.total for line in priced), Decimal("0"))

        # A vendor named in the request is never used; one off the approved list is refused.
        approved = systems.approved_vendors()
        refused = [vendor for vendor in state["fields"]["vendor_mentions"] if vendor not in approved]
        return self._price_and_precheck(state, "procurement", lines, total, refused)

    # ---- BOQ entry (see boq.py) ------------------------------------------------

    def _boq_parse(self, state: WfState) -> WfState:
        from ..boq import boq

        return boq.parse_step(self, state)

    def _boq_match(self, state: WfState) -> WfState:
        from ..boq import boq

        return boq.match_step(self, state)

    def _boq_price(self, state: WfState) -> WfState:
        from ..boq import boq

        return boq.price_step(self, state)

    # ---- shared pricing tail ------------------------------------------------------

    def _price_and_precheck(
        self, state: WfState, step: str, lines: list[dict], total: Decimal, refused: list[str], extra: dict | None = None,
        nothing_priced: str = _NOTHING_PRICED,
    ) -> WfState:
        """Record the priced lines, hold the budget (or stop short), prepare the procurement gate."""
        roles = sorted(budget_approver_roles(total))
        update: WfState = {
            "lines": lines,
            "total": str(total),
            "refused_vendors": refused,
            "approver_roles": roles,
            "policy_question": _policy_question(total, Privacy(state["privacy"]), state["company"]),
        }
        detail = {"lines": len(lines), "total_aed": str(total), "refused_vendors": len(refused), **(extra or {})}

        # Nothing priced: nothing to approve (everything is on the RFQ or review list).
        if not lines:
            return {**update, **_stop(state, step, nothing_priced, detail)}
        if not total:
            return {**update, **_stop(state, step, _ZERO_TOTAL, detail)}

        # Hold the total now: nobody is asked to approve what the budget cannot pay,
        # and no other open run can count on the same money.
        try:
            remaining = systems.reserve_budget(state["run_id"], state["company"], state["department"], total)
        except systems.BudgetShortError as short:
            systems.release_budget(state["run_id"])  # a re-run may hold an older, larger amount
            detail["budget_aed"] = str(short.available)
            return {**update, "budget_remaining": str(short.available), **_stop(state, step, str(short), detail)}
        detail["budget_aed"] = str(remaining)
        update["budget_remaining"] = str(remaining)

        # Anything that fails before the gate (model call, audit write, Ctrl-C) drops the hold.
        with _release_on_failure(state["run_id"]):
            facts = _facts(state, lines, total, remaining, f"procurement approval, then {' or '.join(roles)}")
            note, usage = self._note(state, facts)
            return {
                **update,
                "summaries": {**state["summaries"], Gate.PROCUREMENT.value: note},
                "usage": _add_usage(state["usage"], usage),
                **_step(state, step, StepStatus.DONE, detail),
            }

    def _note(self, state: WfState, facts: dict) -> tuple[str, Usage]:
        """Approver note. The template is used for private runs, when the model fails,
        or when its note holds a number that is not in the facts."""
        template = TemplateSummarizer()
        if state["privacy"] == Privacy.COUNTS_ONLY.value:
            return template.summarize(facts)
        try:
            note, usage = self._summarizer.summarize(facts)
        except Exception:  # a failed note must not strand a run between gates
            return template.summarize(facts)
        if not _numbers(note) <= _numbers(json.dumps(facts)):
            fallback, _ = template.summarize(facts)
            return fallback, usage
        return note, usage

    # ---- agent mode (see procurement_agent.py) ------------------------------------

    def _agent(self, state: WfState) -> WfState:
        from . import procurement_agent

        return procurement_agent.agent_step(self, state)

    def _agent_hold(self, state: WfState) -> WfState:
        from . import procurement_agent

        return procurement_agent.hold_step(self, state)

    # ---- gates -------------------------------------------------------------------

    def _gate_procurement(self, state: WfState) -> WfState:
        return _gate(state, Gate.PROCUREMENT)

    def _finance(self, state: WfState) -> WfState:
        """Hold re-checked against today's budget; the approver role follows the policy threshold."""
        total = Decimal(state["total"])
        roles = sorted(budget_approver_roles(total))
        detail = {"total_aed": str(total), "approver_roles": roles}
        try:
            remaining = systems.reserve_budget(state["run_id"], state["company"], state["department"], total)
        except systems.BudgetShortError as short:
            systems.release_budget(state["run_id"])
            return _stop(state, "finance", str(short), {**detail, "budget_aed": str(short.available)})
        detail["budget_aed"] = str(remaining)
        with _release_on_failure(state["run_id"]):
            facts = _facts(state, state["lines"], total, remaining, " or ".join(roles))
            note, usage = self._note(state, facts)
            return {
                "approver_roles": roles,
                "budget_remaining": str(remaining),
                "summaries": {**state["summaries"], Gate.FINANCE.value: note},
                "usage": _add_usage(state["usage"], usage),
                **_step(state, "finance", StepStatus.DONE, detail),
            }

    def _gate_finance(self, state: WfState) -> WfState:
        return _gate(state, Gate.FINANCE)

    # ---- execute -------------------------------------------------------------------

    def _execute(self, state: WfState) -> WfState:
        """Write the purchase order (and IT ticket) to the outbox. Nothing leaves the machine."""
        entry = Entry(state["entry"])

        # Roles can change while a run waits: the requester must still hold the starter role.
        # A resumed run whose budget is already spent skips this: spent money must reach an order.
        spent = systems.is_spent(state["run_id"])
        if not spent and _ENTRY_ROLE[entry] not in get_user(state["requester"]).roles:
            systems.release_budget(state["run_id"])
            return _stop(state, "execute", f"requester no longer has the {_ENTRY_ROLE[entry]} role", {})

        # Agent mode: a quote may have expired or its vendor been struck off while approvals ran.
        if state.get("mode") == RunMode.AGENT.value and not spent:
            from .procurement_agent import stale_quotes

            stale = stale_quotes(state)
            if stale:
                systems.release_budget(state["run_id"])
                return _stop(state, "execute", "; ".join(stale), {"stale": len(stale)})

        # The hold becomes spend before anything is written; a last shortfall stops the run.
        try:
            systems.spend_budget(state["run_id"], state["company"], state["department"], Decimal(state["total"]))
        except systems.BudgetShortError as short:
            systems.release_budget(state["run_id"])
            return _stop(state, "execute", str(short), {"budget_aed": str(short.available)})

        order = {
            "company": state["company"],
            "department": state["department"],
            "lines": [{k: line[k] for k in ("code", "vendor", "quantity", "unit", "unit_price", "total")} for line in state["lines"]],
            "rfq_lines": len(state.get("rfq", [])),
            "total_aed": state["total"],
            "requested_by": state["requester"],
            "approved_by": state["approvals"],
        }
        if state.get("mode") == RunMode.AGENT.value:
            written = _write_vendor_orders(state, order)
        else:
            written = [_write_outbox(state["run_id"], "purchase_order", order)]
        if entry is Entry.REQUEST:
            ticket = {"company": state["company"], "hires": [h["id"] for h in state["hires"]], "items": state["fields"]["items"]}
            written.append(_write_outbox(state["run_id"], "it_ticket", ticket))
        detail = {"written": sum(written), "total_aed": state["total"]}
        return {**_step(state, "execute", StepStatus.DONE, detail), "status": WfStatus.DONE.value}

    # ---- views -------------------------------------------------------------------

    def _view(self, run_id: str, viewer: User) -> RunView:
        snapshot = self._graph.get_state(_thread(run_id))
        values: WfState = snapshot.values  # type: ignore[assignment]
        gate = _pending_gate(snapshot)
        waiting_quotes = _pending_wait(snapshot)
        status = _AWAITING[gate] if gate else WfStatus(values.get("status", WfStatus.RUNNING.value))
        steps = list(values.get("steps", []))
        if gate:
            steps.append({"name": f"gate_{gate.value}", "status": StepStatus.WAITING.value, "detail": {"roles": sorted(_gate_roles(gate, values))}})
        if waiting_quotes:
            status = WfStatus.AWAITING_QUOTES
            steps.append({"name": "wait_quotes", "status": StepStatus.WAITING.value, "detail": {"round": values.get("agent", {}).get("round", 1)}})
        prices = _sees_quote_prices(values, viewer)
        usage = values.get("usage", {})
        remaining = values.get("budget_remaining")
        return RunView(
            run_id=run_id,
            entry=Entry(values["entry"]),
            requester=values["requester"],
            status=status,
            reason=values.get("reason", ""),
            stopped_at=values.get("stopped_at", ""),
            pending_gate=gate,
            steps=steps,
            hires=values.get("hires", []),
            lines=[line if prices or line.get("code") != "quote" else _without_quote_price(line) for line in values.get("lines", [])],
            # Agent mode writes one PO per vendor; a classic run writes one PO for the run.
            orders=_vendor_orders(values.get("lines", [])) if status is WfStatus.DONE and values.get("mode") == RunMode.AGENT.value else [],
            vendor_names=_vendor_names(values.get("packages", [])),
            rfq=values.get("rfq", []),
            review=values.get("review", []),
            boq_stats=values.get("boq_stats", {}),
            total=Decimal(values.get("total", "0")),
            budget_remaining=Decimal(remaining) if remaining else None,
            approver_roles=values.get("approver_roles", []),
            approvals=values.get("approvals", {}),
            refused_vendors=values.get("refused_vendors", []),
            flags=values.get("flags", []),
            summaries=values.get("summaries", {}),
            policy=self._policy_for(run_id, values, viewer),
            cost_usd=round(usage.get("cost_usd", 0.0), 6),
            tokens=usage.get("input", 0) + usage.get("output", 0),
            private=values.get("privacy") == Privacy.COUNTS_ONLY.value,
            mode=RunMode(values.get("mode", RunMode.CLASSIC.value)),
            packages=values.get("packages", []),
            quotes=[quote if prices else _without_prices(quote) for quote in values.get("quotes", [])],
            comparison=[row if prices else _without_prices(row) for row in values.get("comparison", [])],
            agent_steps=values.get("agent_steps", []),
            events=values.get("events", []),
            decide_problem=_gate_problem(values, viewer, _gate_roles(gate, values)) if gate else "",
        )

    def _policy_for(self, run_id: str, values: WfState, viewer: User) -> dict:
        """The policy section behind the approval rule, retrieved AS THE VIEWER.

        The policy document is readable by procurement, finance and executives
        only, so the requester sees that a citation exists but not its text.
        """
        question = values.get("policy_question")
        if not question:
            return {}
        key = (run_id, viewer.user_id, tuple(sorted(viewer.roles)), tuple(sorted(viewer.companies)))
        if key not in self._policy_cache:
            with _trace_scope(values), tracing.attributes(viewer.user_id, [_TRACE_TAG], run_id):
                answer = ask(self._store, self._composer, viewer, question)
            cited = [c for c in answer.citations if c.doc_id == _POLICY_DOC]
            if answer.refused or not cited:
                self._policy_cache[key] = {"visible": False}
            else:
                self._policy_cache[key] = {
                    "visible": True,
                    "doc_id": cited[0].doc_id,
                    "title": cited[0].title,
                    "section": cited[0].section,
                    "text": answer.text,
                    "shown_docs": answer.shown_docs,
                }

        # Cached text is served only while every document the model saw is still readable now:
        # its citations can misname where a sentence came from.
        cached = self._policy_cache[key]
        if not cached["visible"]:
            return cached
        if not all(self._store.can_read_doc(viewer, doc_id) for doc_id in cached["shown_docs"]):
            return {"visible": False}
        return {name: value for name, value in cached.items() if name != "shown_docs"}

    def _reindex(self, run_id: str) -> None:
        """Refresh the run's row in the run list."""
        self._record_row(_index_row(self._graph.get_state(_thread(run_id))))

    @staticmethod
    def _record_row(row: dict) -> None:
        """A failed index write leaves a stale row, never a failed call: the list re-reads each run."""
        try:
            run_index.record(row)
        except (OSError, ValueError):
            pass

    def _lock_for(self, run_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._run_locks[run_id]


# ---- gate logic -----------------------------------------------------------------------


def _ask(payload: Any):
    """A node that only pauses and saves the reply; the next node (`<name>_apply`) acts on it.

    LangGraph keeps the first resume value of a node that raises and replays it.
    With nothing but interrupt() here, a crash while acting on a reply happens
    in the apply node: the reply is already saved, resume finishes it, and a
    later click cannot be swapped for it.
    """

    def node(state: WfState) -> WfState:
        return {"reply": interrupt(payload(state))}

    return node


def gate_choices(gate: Gate) -> frozenset[Decision]:
    """The answers a gate takes."""
    return _QUOTE_CHOICES if gate is Gate.QUOTES else APPROVAL_CHOICES


def _gate_payload(state: WfState, gate: Gate) -> dict:
    return {"gate": gate.value, "approver_roles": sorted(_gate_roles(gate, state)), "total_aed": state["total"]}


def _gate(state: WfState, gate: Gate) -> WfState:
    """Act on a saved gate decision (see _ask)."""
    roles = _gate_roles(gate, state)
    reply = state["reply"]

    # An ineligible approver is ignored: the run keeps waiting at this gate.
    approver, problem = _resolve_approver(state, reply["approver"], roles)
    if problem:
        _record(state, f"gate_{gate.value}", "ignored", {"approver": reply["approver"], "reason": problem})
        return {"reason": problem, "status": _AWAITING[gate].value, "reply": {}}

    # An eligible approver decides; a rejection ends the run.
    approvals = {**state["approvals"], gate.value: approver.user_id}
    detail = {"approver": approver.user_id, "decision": reply["decision"]}
    if reply["decision"] != Decision.APPROVE.value:
        systems.release_budget(state["run_id"])
        step = _step(state, f"gate_{gate.value}", StepStatus.REJECTED, detail)
        return {**step, "approvals": approvals, "status": WfStatus.REJECTED.value, "reason": f"rejected by {approver.user_id}", "reply": {}}
    step = _step(state, f"gate_{gate.value}", StepStatus.DONE, detail)
    return {**step, "approvals": approvals, "reason": "", "status": WfStatus.RUNNING.value, "reply": {}}


def _gate_roles(gate: Gate, state: Any) -> frozenset[str]:
    """Finance gate: roles from the policy threshold. Every other gate: procurement."""
    if gate is Gate.FINANCE:
        return frozenset(state.get("approver_roles", [])) or budget_approver_roles(Decimal(state.get("total", "0")))
    return _PROCUREMENT_ROLES


def _resolve_approver(state: Any, approver_id: str, roles: frozenset[str]) -> tuple[User | None, str]:
    """The approver and why they may not decide ('' if they may). Agent ids are never looked up."""
    if approver_id.startswith(AGENT_PREFIX):
        return None, "an agent cannot approve"
    approver = get_user(approver_id)
    return approver, _gate_problem(state, approver, roles)


def _gate_problem(state: Any, approver: User, roles: frozenset[str]) -> str:
    """Why this person may not decide this gate, or '' if they may.

    The company is the run's, fixed at start: a requester who later moves
    company does not widen who may approve.
    """
    if approver.user_id.startswith(AGENT_PREFIX):
        return "an agent cannot approve"
    if approver.user_id == state["requester"]:
        return "requester cannot approve their own request"
    if approver.user_id in state.get("approvals", {}).values():
        return "one person cannot approve both gates"
    if not roles & approver.roles:
        return f"approver needs one of {sorted(roles)}"
    if state["company"] not in approver.companies:
        return "approver works for a different company"
    return ""


def _pending_gate(snapshot: Any) -> Gate | None:
    for task in snapshot.tasks:
        for pending in task.interrupts:
            return Gate(pending.value["gate"]) if "gate" in pending.value else None
    return None


def _pending_wait(snapshot: Any) -> bool:
    """True while an agent-mode run waits for vendor replies or the deadline."""
    return any("wait" in pending.value for task in snapshot.tasks for pending in task.interrupts)


def _index_row(snapshot: Any) -> dict:
    """The run-list row: ids, status, gate and total. No text."""
    values = snapshot.values
    gate = _pending_gate(snapshot)
    if gate:
        status = _AWAITING[gate].value
    elif _pending_wait(snapshot):
        status = WfStatus.AWAITING_QUOTES.value
    else:
        status = values.get("status", WfStatus.RUNNING.value)
    return {
        "run_id": values["run_id"],
        "entry": values["entry"],
        "mode": values.get("mode", RunMode.CLASSIC.value),
        "status": status,
        "pending_gate": gate.value if gate else None,
        "requester": values["requester"],
        "company": values["company"],
        "approver_roles": sorted(_gate_roles(gate, values)) if gate else [],
        "total": f"{Decimal(values.get('total', '0')):.2f}",
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _start_row(state: WfState) -> dict:
    return {
        "run_id": state["run_id"], "entry": state["entry"], "mode": state.get("mode", RunMode.CLASSIC.value),
        "status": WfStatus.RUNNING.value, "pending_gate": None, "requester": state["requester"],
        "company": state["company"], "approver_roles": [], "total": "0.00",
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _row_may_match(row: dict, viewer: User, scope: "RunScope") -> bool:
    """False when an index row cannot be in this viewer's list; the live run decides the rest."""
    if scope is RunScope.MINE:
        return row.get("requester") == viewer.user_id
    if not row.get("pending_gate") or row.get("requester") == viewer.user_id:
        return False
    if row.get("company") not in viewer.companies:
        return False
    roles = set(row.get("approver_roles") or [])
    return not roles or bool(roles & viewer.roles)


def _paused(snapshot: Any) -> bool:
    return any(task.interrupts for task in snapshot.tasks)


def _sees_quote_prices(values: Any, viewer: User) -> bool:
    """Quote prices: procurement, finance and executives of the run's company only."""
    return bool(viewer.roles & _QUOTE_PRICE_ROLES) and values.get("company") in viewer.companies


def _vendor_orders(lines: list[dict]) -> list[dict]:
    """One PO per vendor with its total. Shown to the requester too: a known limit, see README."""
    by_vendor: dict[str, list[dict]] = defaultdict(list)
    for line in lines:
        by_vendor[line["vendor"]].append(line)
    return [
        {"vendor": vendor, "lines": len(rows), "total": f"{sum((Decimal(row['total']) for row in rows), Decimal('0')):.2f}"}
        for vendor, rows in sorted(by_vendor.items())
    ]


def _vendor_names(packages: list[dict]) -> dict[str, str]:
    asked = {vendor_id for package in packages for vendor_id in package.get("vendors", [])}
    if not asked:
        return {}
    return {vendor.vendor_id: vendor.name for vendor in systems.vendors() if vendor.vendor_id in asked}


def _without_quote_price(line: dict) -> dict:
    """A priced line from a quote, without what would show the quoted price."""
    return {key: value for key, value in line.items() if key not in _QUOTE_LINE_PRICE_KEYS}


def _without_prices(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in _PRICE_KEYS}


# ---- routing ----------------------------------------------------------------------------


def _route_entry(state: WfState) -> str:
    return "understand" if state["entry"] == Entry.REQUEST.value else "boq_parse"


def _continue_or_end(next_node: str):
    def route(state: WfState) -> str:
        return END if state["status"] == WfStatus.STOPPED.value else next_node

    return route


def _after_price(state: WfState) -> str:
    """Agent mode hands lines that need quotes to the agent loop; otherwise the classic path."""
    if state["status"] == WfStatus.STOPPED.value:
        return END
    return "agent" if state.get("agent", {}).get("next") == "agent" else "gate_procurement"


def _agent_next(state: WfState) -> str:
    if state["status"] == WfStatus.STOPPED.value:
        return END
    return state["agent"]["next"]


def _after_gate(this_gate: str, next_node: str):
    def route(state: WfState) -> str:
        if state["status"] in _WAITING:
            return this_gate
        if state["status"] == WfStatus.REJECTED.value:
            return END
        return next_node

    return route


# ---- steps, audit, traces -----------------------------------------------------------------


def _step(state: WfState, name: str, status: StepStatus, detail: dict) -> WfState:
    """Append a timeline step, write its audit line and span."""
    _record(state, name, status.value, detail)
    return {"steps": state["steps"] + [{"name": name, "status": status.value, "detail": detail}]}


def _stop(state: WfState, name: str, reason: str, detail: dict) -> WfState:
    step = _step(state, name, StepStatus.STOPPED, {**detail, "reason": reason})
    return {**step, "status": WfStatus.STOPPED.value, "reason": reason, "stopped_at": name}


def _record(state: WfState, name: str, status: str, detail: dict) -> None:
    """One audit line and one span per step: ids, counts, amounts and statuses."""
    safe = _counts_only(detail) if state.get("privacy") == Privacy.COUNTS_ONLY.value else detail
    audit.record("workflow_step", user=state["requester"], run=state["run_id"], step=name, status=status, **safe)
    traced = {**safe, "reason": tracing.content(safe["reason"])} if "reason" in safe else safe  # can quote the request
    with tracing.observe(f"workflow_{name}", metadata={"run": state["run_id"], "status": status, **_flat(traced)}):
        pass


def _trace_scope(state: WfState) -> Any:
    """A private run's traces hold no text, even with RAG_TRACE_CONTENT=full."""
    return tracing.counts_only() if state.get("privacy") == Privacy.COUNTS_ONLY.value else nullcontext()


def _counts_only(detail: dict) -> dict:
    """Keep numbers and booleans; drop strings and lists that could carry file text."""
    kept: dict = {}
    for key, value in detail.items():
        if isinstance(value, (int, float, bool)) or value is None:
            kept[key] = value
        elif key.endswith("_aed"):
            kept[key] = value
        elif isinstance(value, list):
            kept[f"{key}_count"] = len(value)
    return kept


def _flat(detail: dict) -> dict:
    """Trace metadata values must be simple: lists and dicts become JSON strings."""
    return {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in detail.items()}


# ---- helpers ------------------------------------------------------------------------------


def _line_row(line: systems.PricedLine, **extra: Any) -> dict:
    return {
        "code": line.code,
        "description": line.description,
        "unit": line.unit,
        "vendor": line.vendor,
        "quantity": str(line.quantity),
        "unit_price": str(line.unit_price),
        "total": str(line.total),
        **extra,
    }


def _policy_question(total: Decimal, privacy: Privacy, company: str) -> str:
    """The question whose cited answer justifies the approval rule (asked as each viewer).

    It names the run's company, so a viewer who works for two companies gets
    this company's rule. Private runs ask about the threshold band, not the exact total.
    """
    at = company_label(company)
    if privacy is Privacy.COUNTS_ONLY:
        band = "above" if total > EXECUTIVE_APPROVAL_AED else "up to"
        return f"Which approval and how many quotations does a purchase {band} AED {EXECUTIVE_APPROVAL_AED:,.0f} need at {at}?"
    return f"Which approval and how many quotations does a purchase of AED {total:,.2f} need at {at}?"


def _numbers(text: str) -> set[str]:
    """Numbers in a text, without thousands separators: '18,150.00' -> '18150.00'."""
    return {match.replace(",", "") for match in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def _facts(state: WfState, lines: list[dict], total: Decimal, remaining: Decimal, needs: str) -> dict:
    """Numbers for the approver note; the summarizer may reword them, not change them."""
    vendors = sorted({line["vendor"] for line in lines})
    what = f"{len(state.get('hires', []))} new-hire equipment sets" if state["entry"] == Entry.REQUEST.value else f"{len(lines)} BOQ lines"
    return {
        "what": what,
        "company": state["company"],
        "total": f"{total:,.2f}",
        "vendors": ", ".join(vendors) or "none",
        "budget_after": f"{remaining - total:,.2f}",
        "needs": f"{needs}; {quotes_required(total)}",
    }


def _add_usage(current: dict, usage: Usage) -> dict:
    return {
        "input": current.get("input", 0) + usage.input_tokens,
        "output": current.get("output", 0) + usage.output_tokens,
        "cost_usd": current.get("cost_usd", 0.0) + usage.cost_usd,
    }


def _write_outbox(run_id: str, kind: str, entry: dict, key: str = "") -> bool:
    """Stand-in for the ERP, IT service desk and vendor mail. One record per (run, kind, key).

    The key tells apart records of one kind in one run, such as one quote request
    per vendor. Without a key the record has the old shape, and a record written
    before keys existed counts as key "".
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / _OUTBOX_FILE

    # File lock around check and append: another process (CLI and API on SQLite)
    # may execute the same run; only one record may land.
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            for line in handle.read().splitlines():
                record = json.loads(line)
                if (record.get("run_id"), record.get("kind"), record.get("key", "")) == (run_id, kind, key):
                    return False

            # Keyed records carry the key; unkeyed ones keep the old shape.
            head = {"run_id": run_id, "kind": kind, **({"key": key} if key else {})}
            handle.write(json.dumps({**head, **entry}) + "\n")
            return True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_vendor_orders(state: WfState, order: dict) -> list[bool]:
    """Agent mode: one purchase order per vendor, keyed by vendor, each written once."""
    by_vendor: dict[str, list[dict]] = defaultdict(list)
    for line in order["lines"]:
        by_vendor[line["vendor"]].append(line)
    written = []
    for vendor, lines in sorted(by_vendor.items()):
        total = sum((Decimal(line["total"]) for line in lines), Decimal("0.00"))
        written.append(_write_outbox(state["run_id"], "purchase_order", {**order, "vendor": vendor, "lines": lines, "total_aed": str(total)}, key=vendor))
    return written


@contextmanager
def _release_on_failure(run_id: str) -> Iterator[None]:
    """Drop the run's hold if the block raises anything, Ctrl-C included, then re-raise."""
    try:
        yield
    except BaseException:
        systems.release_budget(run_id)
        raise


def _thread(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": _RECURSION_LIMIT}
