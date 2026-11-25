"""Thin Claude client: one JSON-schema call with usage, latency and refusal handling."""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from . import tracing

from .config import (
    ANSWER_MAX_TOKENS,
    CLAUDE_MODEL,
    FALLBACK_BETA,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
    TOKENS_PER_MTOK,
)

_REFUSAL = "refusal"
_WARNING = "WARNING"  # Langfuse observation level

# Reply block types, and those a declined model left that are sent back.
_TEXT_BLOCK = "text"
_TOOL_BLOCK = "tool_use"
_SWITCH_BLOCK = "fallback"  # marks where one model declined and the next took over
_REPLAYED = frozenset({_TEXT_BLOCK, _SWITCH_BLOCK})


@dataclass(frozen=True)
class Usage:
    """Tokens and time spent on one model call."""

    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def cost_usd(self) -> float:
        """Price of the call at the configured per-million-token rates."""
        return (
            self.input_tokens * PRICE_INPUT_PER_MTOK + self.output_tokens * PRICE_OUTPUT_PER_MTOK
        ) / TOKENS_PER_MTOK


class ModelRefused(RuntimeError):
    """Claude's safety layer declined the request."""


class ModelUnavailable(RuntimeError):
    """The call failed (network, API error, a reply cut off): the caller may fall back."""


@dataclass(frozen=True)
class ToolCall:
    """One tool the model asks for. Everything here is model text until code checks it."""

    id: str
    name: str
    arguments: Any = field(repr=False)


@dataclass(frozen=True)
class ToolTurn:
    """One model turn. echo is the turn as the API must get it back: never read, log or trace it."""

    text: str = field(repr=False)
    calls: tuple[ToolCall, ...]
    stop_reason: str | None
    usage: Usage
    echo: tuple[dict, ...] = field(repr=False)


class ClaudeClient:
    """Calls Claude with structured output so replies always parse."""

    def __init__(self) -> None:
        # Credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile.
        self._client = anthropic.Anthropic()

    def complete_json(self, system: str, prompt: str, schema: dict) -> tuple[dict, Usage]:
        """Return the parsed JSON reply and its usage."""
        with tracing.observe("generate", as_type="generation", model=CLAUDE_MODEL, input=tracing.content(prompt)) as span:
            started = time.perf_counter()

            # Server-side fallback reroutes a safety refusal to another model automatically.
            response = self._client.beta.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=ANSWER_MAX_TOKENS,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = Usage(response.usage.input_tokens, response.usage.output_tokens, time.perf_counter() - started)

            # Tokens and cost go to the trace whatever the outcome.
            span.update(
                model=response.model,
                usage_details={"input": usage.input_tokens, "output": usage.output_tokens},
                cost_details={"total": usage.cost_usd},
                metadata={"stop_reason": response.stop_reason},
            )
            if response.stop_reason == _REFUSAL:
                span.update(level=_WARNING, status_message="model declined")
                raise ModelRefused("model declined the request")
            text = next(block.text for block in response.content if block.type == "text")
            span.update(output=tracing.content(text))
            return json.loads(text), usage

    def tool_turn(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> ToolTurn:
        """One turn of a tool-use loop. A refusal is a turn with that stop reason, so its cost still counts."""
        # Only the newest message is traced: older ones repeat the model's calls, refused ones too.
        newest = tracing.content(messages[-1]["content"])
        with tracing.observe("generate", as_type="generation", model=CLAUDE_MODEL, input=newest) as span:
            started = time.perf_counter()

            # The model may not be forced to call a tool (the API rejects it), so it chooses.
            response = self._client.beta.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": "low"},
                system=system,
                messages=messages,
                tools=[{**tool, "strict": True} for tool in tools],
            )
            tokens_in, tokens_out = _billed_tokens(response.usage)
            usage = Usage(tokens_in, tokens_out, time.perf_counter() - started)

            # Tokens and cost go to the trace whatever the outcome.
            span.update(
                model=response.model,
                usage_details={"input": usage.input_tokens, "output": usage.output_tokens},
                cost_details={"total": usage.cost_usd},
                metadata={"stop_reason": response.stop_reason},
            )
            if response.stop_reason == _REFUSAL:
                span.update(level=_WARNING, status_message="model declined")
                return ToolTurn("", (), _REFUSAL, usage, ())

            # Code has not checked the calls yet, and text beside a call may repeat it: neither is traced.
            kept, switch = _kept_blocks(response.content)
            text = "\n".join(block.text for block in response.content[switch:] if block.type == _TEXT_BLOCK)
            if not any(block.type == _TOOL_BLOCK for block in response.content):
                span.update(output=tracing.content(text))
            calls = tuple(ToolCall(block.id, block.name, block.input) for block in kept if block.type == _TOOL_BLOCK)
            return ToolTurn(text, calls, response.stop_reason, usage, tuple(block.to_dict() for block in kept))


def _billed_tokens(usage: Any) -> tuple[int, int]:
    """Input and output tokens of every attempt: after a fallback the top-level counts hold the last one only."""
    counts = [(getattr(attempt, "input_tokens", None), getattr(attempt, "output_tokens", None)) for attempt in usage.iterations or []]

    # No attempts listed, or one of a kind without counts: the top-level counts are what is left.
    if not counts or not all(isinstance(count, int) for pair in counts for count in pair):
        return usage.input_tokens, usage.output_tokens
    return sum(pair[0] for pair in counts), sum(pair[1] for pair in counts)


def _kept_blocks(blocks: list) -> tuple[list, int]:
    """Blocks to send back, and where the answering model's own output starts.

    [thinking, text, tool_use, FALLBACK, tool_use] -> [text, FALLBACK, tool_use], 4
    """
    switches = [index for index, block in enumerate(blocks) if block.type == _SWITCH_BLOCK]
    start = switches[-1] + 1 if switches else 0
    earlier = [block for block in blocks[:start] if block.type in _REPLAYED]
    return earlier + blocks[start:], start
