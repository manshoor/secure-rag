"""Prompt-injection screening and output checks.

Defence in depth, in order:
  1. Ingestion: chunks that look like instructions to the model are quarantined
     (kept out of the index) and listed in the ingest report for review.
  2. Retrieval: permissions are enforced in the search filter, never by the model.
  3. Prompt: sources are wrapped in <source> tags and declared to be data.
  4. Output: every citation must point to a source the user was shown, and the
     answer must not contain a canary string from a document the user cannot read.
  5. Tools: an allow-list per role plus human approval (see agent.py).
"""

import html
import re
from dataclasses import dataclass

# Phrases that address the model rather than a human reader. A match
# quarantines the chunk; a human can release it after review.
_INJECTION_PATTERNS = [
    r"ignore (all )?(the )?(previous|prior|above) (instructions|rules)",
    r"disregard (all )?(the )?(previous|prior|above|system)",
    r"you are now (in )?\w+ mode",
    r"(reveal|print|show|output) (the )?(system prompt|hidden|full board memo|confidential)",
    r"call (the )?[a-z]+_[a-z_]+",  # a tool name such as create_purchase_request
    r"without (asking for )?approval",
    r"</?(system|source|instructions)>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Citation markers the model writes inline, e.g. [S2].
_CITATION_RE = re.compile(r"\[(S\d+)\]")


@dataclass(frozen=True)
class OutputCheck:
    """Result of checking a model answer before it reaches the user."""

    ok: bool
    reason: str = ""


def injection_findings(text: str) -> list[str]:
    """Return the suspicious phrases found in text (empty list if clean)."""
    return [match.group(0) for match in _INJECTION_RE.finditer(text)]


def looks_injected(text: str) -> bool:
    """True when text contains at least one instruction aimed at the model."""
    return bool(injection_findings(text))


def wrap_source(source_id: str, title: str, section: str, text: str) -> str:
    """Wrap one chunk so the model can tell document data from instructions."""
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    attrs = {name: html.escape(value, quote=True) for name, value in (("id", source_id), ("title", title), ("section", section))}
    return f'<source id="{attrs["id"]}" title="{attrs["title"]}" section="{attrs["section"]}">\n{safe}\n</source>'


def check_answer(answer: str, cited: list[str], shown: set[str], canaries: list[str]) -> OutputCheck:
    """Reject answers with missing, unknown or inline-only citations, or leaked canaries."""

    # 1. Every cited id must be a source the user was actually shown.
    unknown = [cid for cid in cited if cid not in shown]
    if unknown:
        return OutputCheck(False, f"cites sources that were not provided: {unknown}")

    # 2. Inline markers in the text must also be shown sources.
    inline = set(_CITATION_RE.findall(answer))
    if not inline <= shown:
        return OutputCheck(False, f"inline markers point outside the sources: {sorted(inline - shown)}")

    # 3. An answer with no citation cannot be verified by the reader.
    if not cited and not inline:
        return OutputCheck(False, "answer has no citation")

    # 4. Canary strings mark restricted documents; seeing one means a leak.
    leaked = [canary for canary in canaries if canary in answer]
    if leaked:
        return OutputCheck(False, "answer contains restricted content")
    return OutputCheck(True)
