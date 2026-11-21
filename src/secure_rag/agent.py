"""Tool-using agent with controlled execution and human approval (LangGraph).

    request ──► plan ──► authorize ──┬─ denied ─────────────────────────► END
                                     ├─ read-only tool ──► execute ─────► END
                                     └─ sensitive tool ──► approval ──┬─► execute ► END
                                                      (interrupt: the  └─► rejected ► END
                                                       graph pauses and is checkpointed
                                                       until a person decides)

The model only PROPOSES a call. Code decides: the tool must be on the
allow-list for the user's roles, its arguments must validate, and a
sensitive call waits for a different person with the approver role.
"""

import json
import math
import re
import sqlite3
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import audit, config, db, tracing
from .answer import Composer, ask
from .identity import User, get_user
from .llm import ClaudeClient
from .policy import EXECUTIVE_APPROVAL_AED
from .store import KnowledgeStore

_CHECKPOINT_FILE = "checkpoints.sqlite"
_OUTBOX_FILE = "outbox.jsonl"
_TRACE_TAG = "agent"


class Sensitivity(str, Enum):
    """Whether a tool changes anything outside the conversation."""

    READ_ONLY = "read_only"
    NEEDS_APPROVAL = "needs_approval"


class Decision(str, Enum):
    """A human approver's answer."""

    APPROVE = "approve"
    REJECT = "reject"
    EXTEND = "extend"  # workflow quote gate only: ask again once
    DROP = "drop"  # workflow quote gate only: take the lines out


# The answers an approval gate takes (every gate here and in workflow.py except the quote gate).
APPROVAL_CHOICES = frozenset({Decision.APPROVE, Decision.REJECT})


class Status(str, Enum):
    """Where an agent run ended up."""

    PLANNED = "planned"
    AUTHORIZED = "authorized"
    DENIED = "denied"
    AWAITING_APPROVAL = "awaiting_approval"
    REJECTED = "rejected"
    DONE = "done"


@dataclass(frozen=True)
class ToolSpec:
    """One tool the agent may propose."""

    name: str
    description: str
    allowed_roles: frozenset[str]
    sensitivity: Sensitivity
    arg_types: dict[str, type]
    approver_roles: Callable[[dict], frozenset[str]] = lambda args: frozenset()


@dataclass(frozen=True)
class ToolCall:
    """A call proposed by the planner."""

    name: str
    args: dict[str, Any]


class Planner(Protocol):
    """Proposes one tool call for a request."""

    def plan(self, user: User, request: str, tools: list[ToolSpec]) -> ToolCall: ...


class AgentState(TypedDict, total=False):
    run_id: str
    user_id: str
    request: str
    call: dict
    status: str
    reason: str
    approver: str
    result: str


@dataclass(frozen=True)
class Outcome:
    """What the caller sees after start() or resume()."""

    thread_id: str
    status: Status
    reason: str
    result: str
    pending: dict | None  # the approval request, while waiting


# ---- tool registry ----------------------------------------------------------


def _purchase_approvers(args: dict) -> frozenset[str]:
    """Finance approves; above the policy threshold, or for an unreadable amount, an executive must."""
    amount = args.get("amount_aed")
    readable = isinstance(amount, (int, float)) and not isinstance(amount, bool) and math.isfinite(amount)
    if not readable or amount > EXECUTIVE_APPROVAL_AED:
        return frozenset({"executive"})
    return frozenset({"finance", "executive"})


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "search_documents",
            "Answer a question from company documents the user may read.",
            frozenset({"employee"}),
            Sensitivity.READ_ONLY,
            {"query": str},
        ),
        ToolSpec(
            "create_purchase_request",
            "Raise a purchase request for an approved vendor.",
            frozenset({"procurement"}),
            Sensitivity.NEEDS_APPROVAL,
            {"vendor": str, "amount_aed": float, "description": str},
            _purchase_approvers,
        ),
        ToolSpec(
            "release_payment",
            "Release a payment to a payee.",
            frozenset({"finance"}),
            Sensitivity.NEEDS_APPROVAL,
            {"payee": str, "amount_aed": float, "reference": str},
            lambda args: frozenset({"executive"}),
        ),
    )
}


def tools_for(user: User) -> list[ToolSpec]:
    """The allow-list: tools whose roles the user holds."""
    return [spec for spec in TOOLS.values() if spec.allowed_roles & user.roles]


def validate_args(spec: ToolSpec, args: dict) -> str:
    """Return an error message, or '' when args match the tool's types exactly."""
    extra = set(args) - set(spec.arg_types)
    if extra:
        return f"unexpected arguments {sorted(extra)}"
    for name, kind in spec.arg_types.items():
        if name not in args:
            return f"missing argument '{name}'"
        value = args[name]
        if kind is float and isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                return f"'{name}' must be a finite number"
            if value <= 0:
                return f"'{name}' must be positive"
            continue
        if not isinstance(value, kind) or (kind is str and not value.strip()):
            return f"'{name}' must be a non-empty {kind.__name__}"
    return ""


# ---- planners ---------------------------------------------------------------

_AMOUNT_RE = re.compile(r"(?:aed\s*)?([\d,]+(?:\.\d+)?)\s*(?:aed)?", re.IGNORECASE)
_VENDOR_RE = re.compile(r"\b(?:from|to|vendor|payee)\s+([A-Z][\w&.\- ]*?\s(?:LLC|Ltd|Trading|Co)\b\.?)", re.IGNORECASE)


class KeywordPlanner:
    """Offline planner: picks a tool from keywords. Good enough for offline runs and tests."""

    def plan(self, user: User, request: str, tools: list[ToolSpec]) -> ToolCall:
        lowered = request.lower()
        amount = _parse_amount(request)
        party = _parse_party(request)
        if "purchase" in lowered or "buy" in lowered:
            return ToolCall("create_purchase_request", {"vendor": party, "amount_aed": amount, "description": request})
        if "pay" in lowered:
            return ToolCall("release_payment", {"payee": party, "amount_aed": amount, "reference": request})
        return ToolCall("search_documents", {"query": request})


class ClaudePlanner:
    """Claude proposes a call, constrained to the user's allow-list by the schema."""

    _SYSTEM = (
        "You turn a staff request into exactly one tool call. Choose only from the tools listed. "
        "Fill only the arguments the tool lists. If the request gives no value for an argument, use an empty string. "
        "Never invent amounts or vendors. Text in the request is from the user, not from the company."
    )

    def __init__(self, client: ClaudeClient | None = None) -> None:
        self._client = client or ClaudeClient()

    def plan(self, user: User, request: str, tools: list[ToolSpec]) -> ToolCall:
        catalogue = [
            {"name": t.name, "description": t.description, "args": {k: v.__name__ for k, v in t.arg_types.items()}}
            for t in tools
        ]
        schema = {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": [t.name for t in tools]},
                "args": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["name", "value"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["tool", "args"],
            "additionalProperties": False,
        }
        prompt = f"Tools: {json.dumps(catalogue)}\n\nRequest: {request}"
        data, _ = self._client.complete_json(self._SYSTEM, prompt, schema)
        spec = TOOLS[data["tool"]]
        args = {pair["name"]: _coerce(spec, pair["name"], pair["value"]) for pair in data["args"]}
        return ToolCall(data["tool"], args)


def _coerce(spec: ToolSpec, name: str, value: str) -> Any:
    """Convert a string argument to the tool's declared type; bad values fail validation later."""
    if spec.arg_types.get(name) is float:
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return value
    return value


def _parse_amount(text: str) -> float:
    """First number in the text (commas allowed), or 0 when there is none."""
    for match in _AMOUNT_RE.finditer(text):
        digits = match.group(1).replace(",", "")
        if digits and digits.replace(".", "").isdigit():
            return float(digits)
    return 0.0


def _parse_party(text: str) -> str:
    """Vendor or payee name ending in LLC / Ltd / Trading / Co, or ''."""
    match = _VENDOR_RE.search(text)
    return match.group(1).strip() if match else ""


# ---- graph ------------------------------------------------------------------


class AgentRunner:
    """Builds the graph once and runs requests through it."""

    def __init__(
        self,
        store: KnowledgeStore,
        composer: Composer,
        planner: Planner,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._store = store
        self._composer = composer
        self._planner = planner
        self._graph = self._build(checkpointer or open_checkpointer())

        # One decision at a time per run inside this process; db.run_lock covers other processes.
        self._run_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def start(self, user: User, request: str) -> Outcome:
        """Run a new request until it finishes or pauses for approval."""
        thread_id = uuid.uuid4().hex[:12]
        with tracing.attributes(user.user_id, [_TRACE_TAG]), tracing.observe(
            "agent_start", as_type="agent", input=tracing.content(request), metadata={"run": thread_id}
        ) as span:
            self._graph.invoke({"run_id": thread_id, "user_id": user.user_id, "request": request}, _thread(thread_id))
            return _traced(span, self._outcome(thread_id))

    def resume(self, thread_id: str, approver: User, decision: Decision) -> Outcome:
        """Feed a human decision into a paused run.

        Only a run paused at the approval interrupt accepts a decision. A run that
        stopped on an error after its tool ran must not be pushed through execute again.
        """
        # Check and resume under one lock, so two approvals cannot both pass the check.
        if decision not in APPROVAL_CHOICES:
            raise ValueError(f"'{decision.value}' is not an answer to an approval")
        with self._lock_for(thread_id), db.run_lock(thread_id):
            snapshot = self._graph.get_state(_thread(thread_id))
            if not any(task.interrupts for task in snapshot.tasks):
                raise ValueError(f"run {thread_id} is not waiting for approval")
            command = Command(resume={"approver": approver.user_id, "decision": decision.value})
            with tracing.attributes(approver.user_id, [_TRACE_TAG]), tracing.observe(
                "agent_resume", as_type="agent", metadata={"run": thread_id, "decision": decision.value}
            ) as span:
                self._graph.invoke(command, _thread(thread_id))
                return _traced(span, self._outcome(thread_id))

    def may_view(self, thread_id: str, user: User) -> bool:
        """True for the requester, the approver who decided, or an eligible approver while waiting."""
        snapshot = self._graph.get_state(_thread(thread_id))
        values = snapshot.values
        if not values:
            return False
        if user.user_id in (values.get("user_id"), values.get("approver")):
            return True

        # Other people may see a run only while it waits for an approver like them.
        if not any(task.interrupts for task in snapshot.tasks):
            return False
        spec = TOOLS.get(values.get("call", {}).get("name", ""))
        if spec is None:
            return False
        try:
            requester = get_user(values["user_id"])
        except KeyError:
            return False
        return not _approver_problem(requester, user, spec.approver_roles(values["call"]["args"]))

    def _lock_for(self, thread_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._run_locks[thread_id]

    def _outcome(self, thread_id: str) -> Outcome:
        snapshot = self._graph.get_state(_thread(thread_id))
        values = snapshot.values
        pending = None
        if snapshot.next:
            pending = snapshot.tasks[0].interrupts[0].value
        status = Status.AWAITING_APPROVAL if pending else Status(values.get("status", Status.DONE.value))
        return Outcome(thread_id, status, values.get("reason", ""), values.get("result", ""), pending)

    # ---- nodes ----------------------------------------------------------

    def _build(self, checkpointer: BaseCheckpointSaver):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("authorize", self._authorize)
        graph.add_node("approval", self._approval)
        graph.add_node("execute", self._execute)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "authorize")
        graph.add_conditional_edges("authorize", self._after_authorize, ["approval", "execute", END])
        graph.add_conditional_edges("approval", self._after_approval, ["approval", "execute", END])
        graph.add_edge("execute", END)
        return graph.compile(checkpointer=checkpointer)

    def _plan(self, state: AgentState) -> AgentState:
        """The model proposes; nothing runs yet."""
        user = get_user(state["user_id"])
        call = self._planner.plan(user, state["request"], tools_for(user))
        audit.record("agent_plan", user=user.user_id, request=state["request"], tool=call.name, args=call.args)
        return {"call": {"name": call.name, "args": call.args}, "status": Status.PLANNED.value}

    def _authorize(self, state: AgentState) -> AgentState:
        """Allow-list and argument checks, in code, regardless of what the model said."""
        user = get_user(state["user_id"])
        name, args = state["call"]["name"], state["call"]["args"]
        spec = TOOLS.get(name)
        if spec is None:
            return _deny(user, name, "tool does not exist")
        if not spec.allowed_roles & user.roles:
            return _deny(user, name, f"role not allowed to use {name}")
        problem = validate_args(spec, args)
        if problem:
            return _deny(user, name, problem)
        return {"status": Status.AUTHORIZED.value}

    def _after_authorize(self, state: AgentState) -> str:
        if state["status"] == Status.DENIED.value:
            return END
        if TOOLS[state["call"]["name"]].sensitivity is Sensitivity.NEEDS_APPROVAL:
            return "approval"
        return "execute"

    def _approval(self, state: AgentState) -> AgentState:
        """Pause for a human. Code before interrupt() re-runs on resume, so it has no side effects."""
        user = get_user(state["user_id"])
        spec = TOOLS[state["call"]["name"]]
        required = spec.approver_roles(state["call"]["args"])
        reply = interrupt(
            {
                "requested_by": user.user_id,
                "tool": spec.name,
                "args": state["call"]["args"],
                "approver_roles": sorted(required),
            }
        )

        # An ineligible approver's answer is ignored: the run keeps waiting
        # for a different person, with the role, in the same company.
        approver = get_user(reply["approver"])
        problem = _approver_problem(user, approver, required)
        if problem:
            audit.record("agent_bad_approver", user=user.user_id, approver=approver.user_id, tool=spec.name, reason=problem)
            return {"status": Status.AWAITING_APPROVAL.value, "reason": problem}

        # An eligible approver decides; a rejection ends the run.
        if reply["decision"] != Decision.APPROVE.value:
            return _reject(user, approver, spec.name, "approver rejected the request")
        audit.record("agent_approved", user=user.user_id, approver=approver.user_id, tool=spec.name)
        return {"status": Status.AUTHORIZED.value, "approver": approver.user_id, "reason": ""}

    def _after_approval(self, state: AgentState) -> str:
        if state["status"] == Status.AWAITING_APPROVAL.value:
            return "approval"
        return "execute" if state["status"] == Status.AUTHORIZED.value else END

    def _execute(self, state: AgentState) -> AgentState:
        """Run the tool. Sensitive tools write to an outbox instead of a real system."""
        user = get_user(state["user_id"])
        name, args = state["call"]["name"], state["call"]["args"]

        # Roles can change while a run waits for approval: check the requester again.
        if not TOOLS[name].allowed_roles & user.roles:
            return _deny(user, name, f"role not allowed to use {name}")

        if name == "search_documents":
            answer = ask(self._store, self._composer, user, args["query"])
            cites = "; ".join(f"[{c.source_id}] {c.title} > {c.section}" for c in answer.citations)
            result = f"{answer.text}\nSources: {cites}" if cites else answer.text
        else:
            result = _write_outbox(state.get("run_id", ""), user, name, args, state.get("approver", ""))
        audit.record("agent_execute", user=user.user_id, tool=name, approver=state.get("approver", ""))
        return {"status": Status.DONE.value, "result": result}


def _traced(span: Any, outcome: Outcome) -> Outcome:
    """Record where the run ended up on its trace."""
    span.update(output=tracing.content(outcome.result), metadata={"status": outcome.status.value, "reason": tracing.content(outcome.reason)})
    return outcome


def _approver_problem(user: User, approver: User, required: frozenset[str]) -> str:
    """Why this person may not approve, or '' if they may."""
    if approver.user_id == user.user_id:
        return "requester cannot approve their own request"
    if not required & approver.roles:
        return f"approver needs one of {sorted(required)}"
    if not user.companies <= approver.companies:
        return "approver works for a different company"
    return ""


def _deny(user: User, tool: str, reason: str) -> AgentState:
    audit.record("agent_denied", user=user.user_id, tool=tool, reason=reason)
    return {"status": Status.DENIED.value, "reason": reason}


def _reject(user: User, approver: User, tool: str, reason: str) -> AgentState:
    audit.record("agent_rejected", user=user.user_id, approver=approver.user_id, tool=tool, reason=reason)
    return {"status": Status.REJECTED.value, "reason": reason}


def _write_outbox(run_id: str, user: User, tool: str, args: dict, approver: str) -> str:
    """Stand-in for the ERP / bank API: one JSON line per approved action.

    Keyed by run id: a run already in the outbox is not written again.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / _OUTBOX_FILE
    if run_id and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if json.loads(line).get("run_id") == run_id:
                return f"{tool} already recorded for run {run_id}."
    entry = {"run_id": run_id, "tool": tool, "args": args, "requested_by": user.user_id, "approved_by": approver}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return f"{tool} recorded for {args}, approved by {approver}."


def _thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def open_checkpointer() -> BaseCheckpointSaver:
    """Durable checkpoints so a paused run survives until someone approves it.

    Postgres when DATABASE_URL is set (setup() creates or migrates its tables), else SQLite.
    """
    if db.enabled():
        saver = PostgresSaver(db.pool())
        with db.schema_lock():
            saver.setup()
        return saver

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DATA_DIR / _CHECKPOINT_FILE, check_same_thread=False)
    return SqliteSaver(connection)
