"""Tracing to Langfuse and LangSmith. Each is off unless its keys are set:
Langfuse needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, LangSmith needs LANGSMITH_KEY.
Both get the same observations, with the same fields.

    ask (trace: user id, tags)
      ├─ retrieve      span       chunk ids, scores, count
      ├─ generate      generation model, tokens, cost   (inside the Claude client)
      └─ output_check  guardrail  ok / reason

By default traces hold ids, counts, scores and costs, never question,
document or answer text: the people who read traces may not have the
document permissions the asking user had. RAG_TRACE_CONTENT=full adds text.

LangChain's own LangSmith tracing stays off, whatever LANGSMITH_TRACING says:
it would send whole graph state (BOQ lines, prices) past these rules.
"""

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, Iterator

import langsmith
from langfuse import Langfuse, propagate_attributes
from langfuse.span_filter import is_langfuse_span
from langsmith.run_trees import RunTree

_KEY_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
_LANGSMITH_KEY_VAR = "LANGSMITH_KEY"  # not LANGSMITH_API_KEY, so LangChain never picks it up
_SERVICE_NAME = "secure-rag"  # shown as the service in Langfuse, the project in LangSmith

_ERROR_LEVEL = "ERROR"  # Langfuse observation level
FLUSH_SECONDS = 5.0  # longest a CLI exit or server stop waits for LangSmith
_log = logging.getLogger(__name__)

# Langfuse observation types -> LangSmith run types; anything else is a chain.
_RUN_TYPES = {"generation": "llm", "retriever": "retriever"}

# Turn off LangChain's tracer for the whole process (it checks this switch first).
langsmith.configure(enabled=False)


class TraceContent(str, Enum):
    """How much text goes into traces."""

    IDS = "ids"
    FULL = "full"


class _Observation:
    """One observation in each backend that is on; update() goes to both."""

    def __init__(self, langfuse: Any, run: RunTree | None) -> None:
        self._langfuse = langfuse
        self._run = run
        self.level = ""  # a level the code set itself, such as WARNING for a refusal

    def update(self, **fields: Any) -> "_Observation":
        self.level = fields.get("level", self.level)
        if self._langfuse is not None:
            self._langfuse.update(**fields)
        if self._run is not None:
            _update_run(self._run, fields)
        return self


# One client per backend per process. None means that backend is off.
_state: dict[str, Any] = {"client": None, "ready": False, "langsmith": None, "langsmith_ready": False}

# The open LangSmith run, and the user and tags from attributes(). Our own
# variables, not LangSmith's: its run context would turn LangChain's tracer on.
_parent_run: ContextVar[RunTree | None] = ContextVar("parent_run", default=None)
_run_tags: ContextVar[dict[str, Any]] = ContextVar("run_tags", default={})

# Set inside a private run: content() sends no text, whatever RAG_TRACE_CONTENT says.
_counts_only: ContextVar[bool] = ContextVar("counts_only", default=False)


def configure(client: Langfuse | None) -> None:
    """Set the Langfuse client explicitly (tests, or the API at startup)."""
    _state["client"] = client
    _state["ready"] = True


def configure_langsmith(client: Any) -> None:
    """Set the LangSmith client explicitly (tests)."""
    _state["langsmith"] = client
    _state["langsmith_ready"] = True


def reset() -> None:
    """Forget both clients; the next use reads the keys from env again."""
    _state.update(client=None, ready=False, langsmith=None, langsmith_ready=False)


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


def make_langsmith_client(key: str) -> langsmith.Client:
    """A LangSmith client; LANGSMITH_ENDPOINT picks another region or a self-hosted server."""
    return langsmith.Client(api_key=key)


def _langsmith() -> Any:
    """Create the LangSmith client from env on first use."""
    if not _state["langsmith_ready"]:
        key = os.environ.get(_LANGSMITH_KEY_VAR)
        configure_langsmith(_try_langsmith(key) if key else None)
    return _state["langsmith"]


def _try_langsmith(key: str) -> Any:
    """The client, or None (tracing off) if a LANGSMITH_* setting is bad: a typo must not stop the app."""
    try:
        return make_langsmith_client(key)
    except Exception as error:  # the SDK raises several types for bad settings
        _log.warning("LangSmith tracing is off: %s", type(error).__name__)
        return None


def enabled() -> bool:
    return _client() is not None or _langsmith() is not None


def content(value: Any) -> Any:
    """Text for traces, or None unless RAG_TRACE_CONTENT=full (and never inside counts_only())."""
    if _counts_only.get():
        return None
    mode = TraceContent(os.environ.get("RAG_TRACE_CONTENT", TraceContent.IDS.value))
    return value if mode is TraceContent.FULL else None


@contextmanager
def counts_only() -> Iterator[None]:
    """No text in traces opened inside, for a private run."""
    token = _counts_only.set(True)
    try:
        yield
    finally:
        _counts_only.reset(token)


@contextmanager
def observe(name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
    """Open a nested observation; yields an object with update(**fields)."""
    client = _client()
    if client is None:
        with _langsmith_run(name, as_type, fields) as run:
            yield _Observation(None, run)
        return

    # The error is raised again outside the Langfuse block: inside it, OpenTelemetry
    # would record the message and stack trace, which may hold file text.
    error: BaseException | None = None
    ours: _Observation | None = None
    with client.start_as_current_observation(name=name, as_type=as_type, **fields) as observation:
        try:
            with _langsmith_run(name, as_type, fields) as run:
                ours = _Observation(observation, run)
                yield ours
        except BaseException as caught:
            error = caught
            if ours is None or not ours.level:  # keep a level and message the code set itself
                observation.update(level=_ERROR_LEVEL, status_message=type(caught).__name__)
    if error is not None:
        raise error


@contextmanager
def attributes(user_id: str, tags: list[str]) -> Iterator[None]:
    """Tag every observation opened inside with the user and tags."""
    token = _run_tags.set({"user_id": user_id, "tags": tags})
    try:
        if _client() is None:
            yield
            return
        with propagate_attributes(user_id=user_id, tags=tags):
            yield
    finally:
        _run_tags.reset(token)


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
    runs = _langsmith()
    if runs is not None:
        runs.flush(timeout=FLUSH_SECONDS)


# ---- LangSmith runs -------------------------------------------------------------------------
#
#   observe("ask")            -> RunTree "ask"        posted on open, patched on close
#     observe("retrieve")     ->   child "retrieve"   parent from _parent_run
#
# Fields map as: input -> inputs, output -> outputs, everything else -> metadata.


@contextmanager
def _langsmith_run(name: str, as_type: str, fields: dict[str, Any]) -> Iterator[RunTree | None]:
    """A LangSmith run for one observation, or None when LangSmith is off."""
    client = _langsmith()
    if client is None:
        yield None
        return

    # A child of the open run, or a new trace. Each run carries the user and tags
    # current here (the agent's lookup inside a person's run is the agent's).
    parent = _parent_run.get()
    run_type = _RUN_TYPES.get(as_type, "chain")
    inputs = _wrap("input", fields.get("input"))
    tags = _run_tags.get()
    if parent is not None:
        run = parent.create_child(name, run_type, inputs=inputs, tags=tags.get("tags"))
    else:
        run = RunTree(name=name, run_type=run_type, inputs=inputs, ls_client=client, project_name=_SERVICE_NAME, tags=tags.get("tags"))
    run.extra["metadata"] = {"user_id": tags.get("user_id", "")}  # create_child copies the parent's; drop them
    _update_run(run, {key: value for key, value in fields.items() if key != "input"})
    run.post()

    # Close the run; an error sends its type only, since the message may hold file text.
    token = _parent_run.set(run)
    try:
        yield run
    except BaseException as error:
        run.end(error=type(error).__name__)
        raise
    finally:
        _parent_run.reset(token)
        run.patch()


def _update_run(run: RunTree, fields: dict[str, Any]) -> None:
    """Apply Langfuse-style fields to a LangSmith run."""
    output = fields.get("output")
    if output is not None:
        run.outputs = _wrap("output", output)
    metadata = {key: value for key, value in fields.items() if key not in ("input", "output", "metadata")}
    metadata.update(fields.get("metadata") or {})
    if metadata:
        run.add_metadata(metadata)


def _wrap(key: str, value: Any) -> dict[str, Any]:
    """LangSmith wants a dict; None (no text allowed) sends an empty one."""
    return {} if value is None else {key: value}
