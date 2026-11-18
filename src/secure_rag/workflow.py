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
import json
import re
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import audit, config, db, systems, tracing
from .agent import Decision, open_checkpointer
from .answer import Composer, ask
from .guard import injection_findings
from .identity import User, company_label, get_user
from .intake import Intake, Summarizer, TemplateSummarizer
from .llm import Usage
from .policy import EXECUTIVE_APPROVAL_AED, budget_approver_roles, quotes_required
from .store import KnowledgeStore

_TRACE_TAG = "workflow"
_OUTBOX_FILE = "outbox.jsonl"
_POLICY_DOC = "alpha-procurement-policy"
_PROCUREMENT_ROLES = frozenset({"procurement"})


class Entry(str, Enum):
    """How a run started; each entry has its own allowed starter role and budget line."""

    REQUEST = "request"
    BOQ = "boq"


# Who may start each entry, and which department's budget pays.
_ENTRY_ROLE = {Entry.REQUEST: "hr", Entry.BOQ: "site_engineer"}
_ENTRY_DEPARTMENT = {Entry.REQUEST: "hr", Entry.BOQ: "projects"}


class Gate(str, Enum):
    PROCUREMENT = "procurement"
    FINANCE = "finance"


class WfStatus(str, Enum):
    RUNNING = "running"
    AWAITING_PROCUREMENT = "awaiting_procurement"
    AWAITING_FINANCE = "awaiting_finance"
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


_AWAITING = {Gate.PROCUREMENT: WfStatus.AWAITING_PROCUREMENT, Gate.FINANCE: WfStatus.AWAITING_FINANCE}


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
    ) -> None:
        self._store = store
        self._composer = composer
        self._intake = intake
        self._summarizer = summarizer or TemplateSummarizer()
        self._boq_reader = boq_reader  # tests inject a reader; default is boq_parser.parse_boq
        self._canonicalizer = canonicalizer  # Claude mode only (boq.ClaudeCanonicalizer)
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

    def start_boq(self, user: User, path: Path, privacy: Privacy = Privacy.NORMAL) -> RunView:
        """Start from a BOQ file. Raises PermissionError without the site engineer role.

        Files under data/private/ always run counts-only, whatever the caller asked.
        """
        from .boq_parser import is_private

        if is_private(path):
            privacy = Privacy.COUNTS_ONLY
        return self._start(user, Entry.BOQ, {"boq_file": str(path), "privacy": privacy.value})

    def decide(self, run_id: str, approver: User, decision: Decision, gate: Gate | None = None) -> RunView:
        """Feed a gate decision into a paused run.

        Raises ValueError if no gate is waiting, or if `gate` is given and another
        gate is waiting (the approver saw an older state of the run).
        """
        with self._lock_for(run_id), db.run_lock(f"wf:{run_id}"):
            waiting = _pending_gate(self._graph.get_state(_thread(run_id)))
            if waiting is None:
                raise ValueError(f"run {run_id} is not waiting for a decision")
            if gate is not None and gate is not waiting:
                raise ValueError(f"run {run_id} is waiting for {waiting.value}, not {gate.value}")
            command = Command(resume={"approver": approver.user_id, "decision": decision.value})
            with tracing.attributes(approver.user_id, [_TRACE_TAG]), tracing.observe(
                "workflow_decide", as_type="chain", metadata={"run": run_id, "decision": decision.value}
            ) as span:
                self._graph.invoke(command, _thread(run_id))
                view = self._view(run_id, approver)
                span.update(metadata={"status": view.status.value, "reason": view.reason})
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
            if _pending_gate(snapshot) is not None or not snapshot.next:
                raise ValueError(f"run {run_id} has nothing to resume")
            with tracing.attributes(user.user_id, [_TRACE_TAG]), tracing.observe(
                "workflow_resume", as_type="chain", metadata={"run": run_id, "next": list(snapshot.next)}
            ):
                self._graph.invoke(None, _thread(run_id))
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
        gate = _pending_gate(snapshot)
        if gate is None:
            return False
        return not _gate_problem(values, viewer, _gate_roles(gate, values))

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
        with tracing.attributes(user.user_id, [_TRACE_TAG]), tracing.observe(
            "workflow_start", as_type="chain", metadata={"run": run_id, "entry": entry.value}
        ) as span:
            self._graph.invoke(state, _thread(run_id))
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
            "gate_procurement": self._gate_procurement,
            "finance": self._finance,
            "gate_finance": self._gate_finance,
            "execute": self._execute,
        }
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
            ("boq_price", "gate_procurement"),
            ("finance", "gate_finance"),
        ):
            graph.add_conditional_edges(before, _continue_or_end(after), [after, END])
        graph.add_conditional_edges("gate_procurement", _after_gate("gate_procurement", "finance"), ["gate_procurement", "finance", END])
        graph.add_conditional_edges("gate_finance", _after_gate("gate_finance", "execute"), ["gate_finance", "execute", END])
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
        from . import boq

        return boq.parse_step(self, state)

    def _boq_match(self, state: WfState) -> WfState:
        from . import boq

        return boq.match_step(self, state)

    def _boq_price(self, state: WfState) -> WfState:
        from . import boq

        return boq.price_step(self, state)

    # ---- shared pricing tail ------------------------------------------------------

    def _price_and_precheck(
        self, state: WfState, step: str, lines: list[dict], total: Decimal, refused: list[str], extra: dict | None = None
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
            return {**update, **_stop(state, step, "no line could be priced from the catalogue", detail)}

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
        status = _AWAITING[gate] if gate else WfStatus(values.get("status", WfStatus.RUNNING.value))
        steps = list(values.get("steps", []))
        if gate:
            steps.append({"name": f"gate_{gate.value}", "status": StepStatus.WAITING.value, "detail": {"roles": sorted(_gate_roles(gate, values))}})
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
            lines=values.get("lines", []),
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
                }

        # Cached text is served only while the policy document is still readable now.
        cached = self._policy_cache[key]
        if cached["visible"] and not self._store.can_read_doc(viewer, cached["doc_id"]):
            return {"visible": False}
        return cached

    def _lock_for(self, run_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._run_locks[run_id]


# ---- gate logic -----------------------------------------------------------------------


def _gate(state: WfState, gate: Gate) -> WfState:
    """Pause for a person. Code before interrupt() re-runs on resume, so it has no side effects."""
    roles = _gate_roles(gate, state)
    reply = interrupt({"gate": gate.value, "approver_roles": sorted(roles), "total_aed": state["total"]})

    # An ineligible approver is ignored: the run keeps waiting at this gate.
    approver = get_user(reply["approver"])
    problem = _gate_problem(state, approver, roles)
    if problem:
        _record(state, f"gate_{gate.value}", "ignored", {"approver": approver.user_id, "reason": problem})
        return {"reason": problem, "status": _AWAITING[gate].value}

    # An eligible approver decides; a rejection ends the run.
    approvals = {**state["approvals"], gate.value: approver.user_id}
    detail = {"approver": approver.user_id, "decision": reply["decision"]}
    if reply["decision"] != Decision.APPROVE.value:
        systems.release_budget(state["run_id"])
        step = _step(state, f"gate_{gate.value}", StepStatus.REJECTED, detail)
        return {**step, "approvals": approvals, "status": WfStatus.REJECTED.value, "reason": f"rejected by {approver.user_id}"}
    return {**_step(state, f"gate_{gate.value}", StepStatus.DONE, detail), "approvals": approvals, "reason": "", "status": WfStatus.RUNNING.value}


def _gate_roles(gate: Gate, state: Any) -> frozenset[str]:
    if gate is Gate.PROCUREMENT:
        return _PROCUREMENT_ROLES
    return frozenset(state.get("approver_roles", [])) or budget_approver_roles(Decimal(state.get("total", "0")))


def _gate_problem(state: Any, approver: User, roles: frozenset[str]) -> str:
    """Why this person may not decide this gate, or '' if they may.

    The company is the run's, fixed at start: a requester who later moves
    company does not widen who may approve.
    """
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
            return Gate(pending.value["gate"])
    return None


# ---- routing ----------------------------------------------------------------------------


def _route_entry(state: WfState) -> str:
    return "understand" if state["entry"] == Entry.REQUEST.value else "boq_parse"


def _continue_or_end(next_node: str):
    def route(state: WfState) -> str:
        return END if state["status"] == WfStatus.STOPPED.value else next_node

    return route


def _after_gate(this_gate: str, next_node: str):
    def route(state: WfState) -> str:
        if state["status"] in (WfStatus.AWAITING_PROCUREMENT.value, WfStatus.AWAITING_FINANCE.value):
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
    with tracing.observe(f"workflow_{name}", metadata={"run": state["run_id"], "status": status, **_flat(safe)}):
        pass


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


def _write_outbox(run_id: str, kind: str, entry: dict) -> bool:
    """Stand-in for the ERP and IT service desk. One record per run and kind, never two."""
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
                if record.get("run_id") == run_id and record.get("kind") == kind:
                    return False
            handle.write(json.dumps({"run_id": run_id, "kind": kind, **entry}) + "\n")
            return True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _release_on_failure(run_id: str) -> Iterator[None]:
    """Drop the run's hold if the block raises anything, Ctrl-C included, then re-raise."""
    try:
        yield
    except BaseException:
        systems.release_budget(run_id)
        raise


def _thread(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}
