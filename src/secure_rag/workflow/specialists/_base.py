"""The outcome every specialist returns, and the run that falls back to the rules.

    run_specialist
      ├─ run_agent ─► PRIVATE ───────────────────────► rules, status private
      │             ├─ not COMPLETED ────────────────► rules, status fallback
      │             └─ COMPLETED ─► validate (code) ─┬─► rejected ─► rules, status fallback
      │                                              └─► passed ───► agent or scripted
      └─ outcome: JSON-ready, so a graph can checkpoint it

The decision holds enums and code-issued ids only. reason is model text: screened
like reasoning, cut short, shown labelled; never part of the decision or the audit.
"""

import textwrap
from collections.abc import Callable
from enum import Enum

from ...core.agent_runtime import AgentResult, AgentSpec, EndReason, RunPrivacy, run_agent, screen_model_text

_REASON_KEY = "reason"  # the output schema's free-text field
_REASON_MAX_CHARS = 200

# A decision and its reason codes, as code checked or chose them.
_Checked = tuple[dict, list[str]]


class Status(str, Enum):
    """Who decided."""

    AGENT = "agent"  # Claude decided and code accepted it
    SCRIPTED = "scripted"  # the offline stand-in decided through the same loop
    PRIVATE = "private"  # a private run: the rules decided, not a failure
    FALLBACK = "fallback"  # the agent failed or its decision was rejected: the rules decided


# The statuses an accepted decision may carry.
_DECIDERS = frozenset({Status.AGENT, Status.SCRIPTED})


def run_specialist(
    spec: AgentSpec,
    task_facts: dict,
    authorize: Callable[[str, str, dict], bool],
    client: object,
    *,
    privacy: RunPrivacy,
    success: Status,
    rules: Callable[[], _Checked],
    validate: Callable[[dict], _Checked | None],
    case_id: str | None = None,
) -> dict:
    """Run the agent; on a private run, a failure or a rejected decision, the rules decide.

    success is the status of an accepted decision (AGENT or SCRIPTED). validate
    returns the checked decision and reason codes, or None to reject. A model
    failure never raises; a failure in rules is a bug in code and does.
    """
    if not isinstance(success, Status) or success not in _DECIDERS:
        raise ValueError("success must be AGENT or SCRIPTED")
    result = run_agent(spec, task_facts, authorize, client, privacy=privacy, case_id=case_id)

    # Private: the model was never asked.
    if result.end_reason is EndReason.PRIVATE:
        return outcome(spec.name, Status.PRIVATE, *rules(), "", result)

    # The run did not end with a result: the rules decide, flagged.
    checked = _validated(validate, result.output) if result.end_reason is EndReason.COMPLETED else None
    if checked is None:
        return outcome(spec.name, Status.FALLBACK, *rules(), "", result)

    # Accepted: the reason is the only model text kept, and only once screened.
    reason = screen_model_text(result.output.get(_REASON_KEY, ""), task_facts, result)
    return outcome(spec.name, success, *checked, _short(reason), result)


def outcome(specialist: str, status: Status, decision: dict, reason_codes: list[str], reason: str, result: AgentResult) -> dict:
    """The JSON-ready outcome. audit and usage come from the run, whoever decided."""
    usage = result.usage
    return {
        "specialist": specialist,
        "status": status.value,
        "decision": decision,
        "reason_codes": list(reason_codes),
        "reason": reason,
        "audit": result.audit_facts(),
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "seconds": usage.seconds, "cost_usd": usage.cost_usd},
    }


def _validated(validate: Callable[[dict], _Checked | None], output: dict | None) -> _Checked | None:
    """The validator's answer if it is a (dict, list) pair; anything else, an exception or no output rejects."""
    if output is None:
        return None
    try:
        answer = validate(output)
    except Exception:
        return None
    if not isinstance(answer, tuple) or len(answer) != 2:
        return None
    decision, codes = answer
    return answer if isinstance(decision, dict) and isinstance(codes, list) else None


def _short(reason: str) -> str:
    """Whole words only: a cut number would be a new one."""
    return textwrap.shorten(reason, _REASON_MAX_CHARS, placeholder=" ...") if reason else ""
