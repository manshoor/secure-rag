"""Langfuse tracing, off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

    ask (trace: user id, tags)
      ├─ retrieve      span       chunk ids, scores, count
      ├─ generate      generation model, tokens, cost   (inside the Claude client)
      └─ output_check  guardrail  ok / reason

By default traces hold ids, counts, scores and costs, never question,
document or answer text: the people who read traces may not have the
document permissions the asking user had. RAG_TRACE_CONTENT=full adds text.
"""

import os
from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator

from langfuse import Langfuse, propagate_attributes
from langfuse.span_filter import is_langfuse_span

_KEY_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
_SERVICE_NAME = "secure-rag"  # shown as the service in Langfuse


class TraceContent(str, Enum):
    """How much text goes into traces."""

    IDS = "ids"
    FULL = "full"


class _NoOp:
    """Stands in for an observation when tracing is off."""

    def update(self, **_: Any) -> "_NoOp":
        return self


# One client per process. None means tracing is off.
_state: dict[str, Any] = {"client": None, "ready": False}


def configure(client: Langfuse | None) -> None:
    """Set the client explicitly (tests, or the API at startup)."""
    _state["client"] = client
    _state["ready"] = True


def make_client(**options: Any) -> Langfuse:
    """A Langfuse client that exports only this app's spans.

    The Anthropic SDK emits its own span per call. Exporting it would count
    each call's cost twice and add Anthropic account ids to every trace.
    """
    os.environ.setdefault("OTEL_SERVICE_NAME", _SERVICE_NAME)
    return Langfuse(should_export_span=is_langfuse_span, **options)


def _client() -> Langfuse | None:
    """Create the client from env on first use."""
    if not _state["ready"]:
        enabled = all(os.environ.get(name) for name in _KEY_VARS)
        configure(make_client() if enabled else None)
    return _state["client"]


def enabled() -> bool:
    return _client() is not None


def content(value: Any) -> Any:
    """Text for traces, or None unless RAG_TRACE_CONTENT=full."""
    mode = TraceContent(os.environ.get("RAG_TRACE_CONTENT", TraceContent.IDS.value))
    return value if mode is TraceContent.FULL else None


@contextmanager
def observe(name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
    """Open a nested observation; yields an object with update(**fields)."""
    client = _client()
    if client is None:
        yield _NoOp()
        return
    with client.start_as_current_observation(name=name, as_type=as_type, **fields) as observation:
        yield observation


@contextmanager
def attributes(user_id: str, tags: list[str]) -> Iterator[None]:
    """Tag every observation opened inside with the user and tags."""
    if _client() is None:
        yield
        return
    with propagate_attributes(user_id=user_id, tags=tags):
        yield


def current_trace_id() -> str:
    """Id of the active trace, for linking a response to Langfuse, or ''."""
    client = _client()
    if client is None:
        return ""
    return client.get_current_trace_id() or ""


def flush() -> None:
    """Send buffered spans (call before a short process exits)."""
    client = _client()
    if client is not None:
        client.flush()
