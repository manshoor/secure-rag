"""Langfuse traces carry ids, scores and costs, and no text unless asked."""

import uuid

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from secure_rag import tracing
from secure_rag.answer import ask
from secure_rag.identity import get_user

_QUESTION = "How many leave days can I carry over?"


@pytest.fixture
def spans(monkeypatch):
    """A real Langfuse client that exports to memory, never the network."""
    exporter = InMemorySpanExporter()

    # Langfuse caches one client per public key, so each test needs its own key.
    client = tracing.make_client(
        public_key=f"pk-lf-{uuid.uuid4().hex}",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    tracing.configure(client)
    monkeypatch.delenv("RAG_TRACE_CONTENT", raising=False)
    yield lambda: (client.flush(), exporter.get_finished_spans())[1]


def _attrs(spans) -> str:
    return " ".join(str(dict(span.attributes)) for span in spans)


def test_off_by_default_is_a_no_op(system):
    tracing.configure(None)
    answer = ask(system.store, system.composer, get_user("sara"), _QUESTION)
    assert answer.trace_id == "" and not answer.refused


def test_ask_produces_nested_trace_without_text(system, spans):
    answer = ask(system.store, system.composer, get_user("sara"), _QUESTION)
    finished = spans()
    names = {span.name for span in finished}
    assert {"ask", "retrieve", "output_check"} <= names
    assert len({span.context.trace_id for span in finished}) == 1
    assert answer.trace_id

    text = _attrs(finished)
    assert "group-hr-annual-leave" in text  # chunk ids are traced
    assert "sara" in text  # user id is traced
    assert _QUESTION not in text and answer.text not in text  # text is not


def test_full_content_mode_adds_text(system, spans, monkeypatch):
    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    ask(system.store, system.composer, get_user("sara"), _QUESTION)
    assert _QUESTION in _attrs(spans())


def test_refused_question_is_traced_without_forbidden_chunks(system, spans):
    answer = ask(system.store, system.composer, get_user("omar"), "What does the board memo say?")
    finished = spans()
    assert answer.refused and any(span.name == "retrieve" for span in finished)
    assert "group-board-memo" not in _attrs(finished)


class _FakeMessages:
    """Mimics client.beta.messages with a fixed reply."""

    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason

    def create(self, **_):
        from types import SimpleNamespace

        block = SimpleNamespace(type="text", text='{"answer": "Ten days [S1].", "citations": ["S1"], "refused": false}')
        usage = SimpleNamespace(input_tokens=1000, output_tokens=50)
        return SimpleNamespace(model="claude-opus-5-5", usage=usage, stop_reason=self.stop_reason, content=[block])


def _fake_client(stop_reason: str):
    from types import SimpleNamespace

    from secure_rag.llm import ClaudeClient

    client = ClaudeClient.__new__(ClaudeClient)
    client._client = SimpleNamespace(beta=SimpleNamespace(messages=_FakeMessages(stop_reason)))
    return client


def test_generation_records_model_tokens_and_cost(spans):
    _, usage = _fake_client("end_turn").complete_json("system", "prompt", {})
    generation = next(span for span in spans() if span.name == "generate")
    attrs = str(dict(generation.attributes))
    assert "claude-opus-5-5" in attrs and "1000" in attrs
    assert str(round(usage.cost_usd, 3)) in attrs


def test_refusal_is_traced_as_warning(spans):
    from secure_rag.llm import ModelRefused

    with pytest.raises(ModelRefused):
        _fake_client("refusal").complete_json("system", "prompt", {})
    generation = next(span for span in spans() if span.name == "generate")
    assert "WARNING" in str(dict(generation.attributes))



def test_only_own_spans_are_exported():
    """The Anthropic SDK's own span would double the cost and add account ids."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    client = tracing.make_client(
        public_key=f"pk-lf-{uuid.uuid4().hex}",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    tracing.configure(client)
    with tracing.observe("ask"):
        # A gen_ai span from another library, as the Anthropic SDK emits.
        with provider.get_tracer("anthropic").start_as_current_span("anthropic.messages.create") as span:
            span.set_attribute("gen_ai.operation.name", "chat")
    client.flush()
    assert [span.name for span in exporter.get_finished_spans()] == ["ask"]
