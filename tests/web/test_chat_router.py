"""Claude as the chat router, with fake clients: no paid calls.

Keyword rules run first and stay free. Only a message they miss goes to Claude,
which returns one intent from a fixed list. Typed text still never decides.
"""

import json
from dataclasses import replace

import anthropic
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config
from secure_rag.core.identity import get_user
from secure_rag.core.llm import ModelRefused, Usage
from secure_rag.web.chat import ClaudeRouter, Intent, route_message
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import RunMode, WfStatus, WorkflowRunner

ISSUES = config.SAMPLES_DIR / "boq_with_issues.csv"
UNMATCHED = "Is there anything on my plate today?"  # no keyword rule matches this


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


@pytest.fixture
def wf(system) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())


class Says:
    """A fake Claude client that returns one intent and records what it was sent."""

    def __init__(self, intent: str) -> None:
        self.intent = intent
        self.calls: list[tuple[str, dict]] = []

    def complete_json(self, system: str, prompt: str, schema: dict) -> tuple[dict, Usage]:
        self.calls.append((prompt, schema))
        return {"intent": self.intent}, Usage(80, 5)


class Fails:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def complete_json(self, system: str, prompt: str, schema: dict) -> tuple[dict, Usage]:
        raise self.error


def _with_router(system, client):
    return replace(system, router=ClaudeRouter(client))


def test_a_keyword_match_never_calls_the_model(system, wf):
    client = Says(Intent.HELP.value)
    reply = route_message(_with_router(system, client), wf, get_user("omar"), "what is waiting on me?")
    assert reply.intent is Intent.WAITING_ON_ME and client.calls == []


def test_a_message_the_rules_miss_goes_to_claude(system, wf):
    client = Says(Intent.WAITING_ON_ME.value)
    reply = route_message(_with_router(system, client), wf, get_user("omar"), UNMATCHED)
    assert reply.intent is Intent.WAITING_ON_ME
    prompt, schema = client.calls[0]
    assert json.loads(prompt)["message"] == UNMATCHED
    assert set(schema["properties"]["intent"]["enum"]) == {intent.value for intent in Intent}


def test_a_routed_decide_request_still_never_decides(system, wf):
    run = wf.start_boq(get_user("khalid"), ISSUES, mode=RunMode.AGENT)
    client = Says(Intent.DECIDE_REQUEST.value)
    reply = route_message(_with_router(system, client), wf, get_user("omar"), "Make it so, send them all out now.", run.run_id)
    assert reply.intent is Intent.DECIDE_REQUEST
    assert wf.view(run.run_id, get_user("omar")).status is WfStatus.AWAITING_RFQ


@pytest.mark.parametrize(
    "client",
    [Says("approve_everything"), Fails(ModelRefused("declined")), Fails(json.JSONDecodeError("cut off", "{", 1)),
     Fails(anthropic.APIConnectionError(request=__import__("httpx").Request("POST", "https://api.anthropic.com"))),
     Fails(TypeError("odd reply")), Fails(StopIteration())],
    ids=["invented_intent", "refused", "bad_json", "api_error", "type_error", "no_text_block"],
)
def test_a_bad_or_failed_routing_falls_back_to_documents(system, wf, client):
    reply = route_message(_with_router(system, client), wf, get_user("omar"), UNMATCHED)
    assert reply.intent is Intent.ASK_DOCUMENTS


def test_offline_has_no_router(system, wf):
    assert system.router is None
    assert route_message(system, wf, get_user("omar"), UNMATCHED).intent is Intent.ASK_DOCUMENTS


def test_claude_mode_builds_a_router(monkeypatch, tmp_path):
    from secure_rag.core.config import EmbedderKind, LlmKind
    from secure_rag.web.app import build_system

    monkeypatch.setattr(anthropic, "Anthropic", lambda *args, **kwargs: object())  # no key, no network
    built = build_system(embedder=EmbedderKind.HASH, llm=LlmKind.CLAUDE)
    assert isinstance(built.router, ClaudeRouter)
