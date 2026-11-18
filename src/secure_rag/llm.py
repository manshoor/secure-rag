"""Thin Claude client: one JSON-schema call with usage, latency and refusal handling."""

import json
import time
from dataclasses import dataclass

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
