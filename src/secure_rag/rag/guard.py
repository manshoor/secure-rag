"""Prompt-injection screening and output checks.

Defence in depth, in order:
  1. Ingestion: chunks that look like instructions to the model are quarantined
     (kept out of the index) and listed in the ingest report for review.
  2. Retrieval: permissions are enforced in the search filter, never by the model.
  3. Prompt: sources are wrapped in <source> tags, labelled with their company,
     and declared to be data.
  4. Output: every citation must point to a source the user was shown, and the
     answer must not contain a canary string from a document the user cannot read.
  5. Grounding: every sentence carries a citation and every number in it appears
     in the sources it cites (or in the question); the answer keeps to the
     company the question names, and names the company when two could apply.
  6. Tools: an allow-list per role plus human approval (see agent.py).

Reasons name the rule, never the text or a number: they go to audit lines and traces.
"""

import html
import re
from dataclasses import dataclass

from ..core.config import GROUP_COMPANY
from ..core.identity import companies_named

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
    r"ignore (the |all )?(policy|policies|rate card|catalogue|catalog|budget|approvals?|limits?)",
    r"\bpay (aed|usd|\$)\s*[\d,]+",
    r"\bprice (it |this |them |each |all )?at (aed|usd|\$)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Citation markers the model writes inline, e.g. [S2].
_CITATION_RE = re.compile(r"\[(S\d+)\]")

UNSUPPORTED_NUMBER = "unsupported number"
UNCITED_SENTENCE = "sentence without citation"
OTHER_COMPANY = "cites another company's document"
COMPANY_NEEDED = "question needs a company"

# A sentence ends at . ! ? before a space, or at a line break. A citation marker
# right after the full stop still belongs to it ("... retention.  [S1] Half ...").
# Decimal points ("0.1 percent") are not followed by a space, so they never split.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\s*\[S\d+\])|(?<=\])\s+(?=[A-Z])|\n+")
# A list number at the start of a line ("1. ", "2) ") is numbering, not a fact.
_LIST_NUMBER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
# Times count as one number ("12:30"); "18,150.00" and "18,150" are the same number.
_NUMBER_RE = re.compile(r"\d+(?::\d+)+|\d[\d,]*(?:\.\d+)?")
_TRAILING_ZEROS_RE = re.compile(r"\.0+$")
_WORD_RE = re.compile(r"[a-z]+")

# Number words count as their digits, so "three" in a source supports "3" in an answer.
_NUMBER_WORDS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
} | {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
# "one" is a pronoun, not a count, in "no one", "any one", "every one", "one of".
_PRONOUN_ONE_RE = re.compile(r"\b(?:no|any|every|some)\s+one\b|\bone\s+of\b")

# One- and two-word sentences ("Yes.", "In short:") need no citation.
_MIN_CLAIM_WORDS = 3
# The phrases the prompt asks for when part of a question is not covered.
_GAP_RE = re.compile(r"\bthe documents do not (?:cover|mention|say|state)\b", re.IGNORECASE)


@dataclass(frozen=True)
class OutputCheck:
    """Result of checking a model answer before it reaches the user."""

    ok: bool
    reason: str = ""
    detail: str = ""  # model text behind the failure; never in reasons or audit lines


def injection_findings(text: str) -> list[str]:
    """Return the suspicious phrases found in text (empty list if clean)."""
    return [match.group(0) for match in _INJECTION_RE.finditer(text)]


def looks_injected(text: str) -> bool:
    """True when text contains at least one instruction aimed at the model."""
    return bool(injection_findings(text))


def wrap_source(source_id: str, title: str, section: str, company: str, text: str) -> str:
    """Wrap one chunk so the model can tell document data from instructions, and whose rule it is."""
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    fields = (("id", source_id), ("title", title), ("section", section), ("company", company))
    attrs = {name: html.escape(value, quote=True) for name, value in fields}
    return (
        f'<source id="{attrs["id"]}" title="{attrs["title"]}" section="{attrs["section"]}" company="{attrs["company"]}">'
        f"\n{safe}\n</source>"
    )


def marker_ids(answer: str) -> set[str]:
    """Source ids cited inline, e.g. {'S1'} for '... [S1].'"""
    return set(_CITATION_RE.findall(answer))


def check_answer(answer: str, cited: list[str], shown: set[str], canaries: list[str]) -> OutputCheck:
    """Reject answers with missing, unknown or inline-only citations, or leaked canaries."""

    # 1. Every cited id must be a source the user was actually shown.
    unknown = [cid for cid in cited if cid not in shown]
    if unknown:
        return OutputCheck(False, "cites sources that were not provided")

    # 2. Inline markers in the text must also be shown sources.
    inline = set(_CITATION_RE.findall(answer))
    if not inline <= shown:
        return OutputCheck(False, "inline markers point outside the sources")

    # 3. An answer with no citation cannot be verified by the reader.
    if not cited and not inline:
        return OutputCheck(False, "answer has no citation")

    # 4. Canary strings mark restricted documents; seeing one means a leak.
    leaked = [canary for canary in canaries if canary in answer]
    if leaked:
        return OutputCheck(False, "answer contains restricted content")
    return OutputCheck(True)


def check_company_scope(answer: str, company_of: dict[str, str], named: set[str], retrieved: set[str]) -> OutputCheck:
    """Keep the answer to the companies the question is about.

    company_of maps each shown source id to its company id; named holds the
    companies the question names; retrieved the companies of every shown source
    (already narrowed to named ones by the search). Group-wide documents fit any company.
    """
    cited = {company_of[cid] for cid in marker_ids(answer) if cid in company_of} - {GROUP_COMPANY}

    # 1. A question about a company is answered from that company's documents only.
    if named and cited - named:
        return OutputCheck(False, OTHER_COMPANY)

    # 2. Two companies could apply: the answer must say whose rule each cited one is.
    if len(retrieved - {GROUP_COMPANY}) > 1 and not cited <= companies_named(answer):
        return OutputCheck(False, COMPANY_NEEDED)

    # 3. A sentence that names a company must cite that company's rule, not another's.
    for sentence in _sentences(answer):
        said = companies_named(sentence)
        sentence_cited = {company_of[cid] for cid in marker_ids(sentence) if cid in company_of} - {GROUP_COMPANY}
        if said and sentence_cited - said:
            return OutputCheck(False, OTHER_COMPANY)
    return OutputCheck(True)


def check_grounding(answer: str, source_texts: dict[str, str], question: str) -> OutputCheck:
    """Every claim is cited, and each number in it comes from a source it cites.

    source_texts maps each shown source id to its text. A number from the
    question may be repeated only next to a number from the cited source, so
    "up to AED 30,000 [S1]" fails when S1 says 25,000 and the question said 30,000.
    An uncited "The documents do not cover ..." sentence may repeat a number from
    any source the answer cites elsewhere.
    """
    asked = _numbers(question)
    cited_anywhere = marker_ids(answer)
    for sentence in _sentences(answer):
        cited = marker_ids(sentence)

        # A gap sentence carries no marker: check its numbers against the answer's cited sources.
        if not cited and _GAP_RE.search(sentence):
            cited = cited_anywhere

        # Numbers: each one must be in a cited source, or in the question beside a sourced one.
        claimed = _numbers(sentence)
        supported = set().union(*(_numbers(source_texts.get(cid, "")) for cid in cited))
        from_source = claimed & supported
        unsupported = claimed - supported - (asked if from_source else set())
        if unsupported:
            return OutputCheck(False, UNSUPPORTED_NUMBER, f"numbers {sorted(unsupported)} in: {sentence.strip()}")

        # Coverage: a claim carries a citation; only a stated gap or a fragment does not.
        words = _CITATION_RE.sub(" ", sentence).split()
        if marker_ids(sentence) or len(words) < _MIN_CLAIM_WORDS or _GAP_RE.search(sentence):
            continue
        return OutputCheck(False, UNCITED_SENTENCE, sentence.strip())
    return OutputCheck(True)


def _sentences(answer: str) -> list[str]:
    """Answer sentences with their markers, list numbering removed."""
    unnumbered = _LIST_NUMBER_RE.sub("", answer.strip())
    return [part for part in _SENTENCE_SPLIT_RE.split(unnumbered) if part.strip()]


def _numbers(text: str) -> set[str]:
    """Numbers as plain digits: '25,000' -> '25000', '18,150.00' -> '18150', 'three' -> '3'."""
    plain = _PRONOUN_ONE_RE.sub(" ", _CITATION_RE.sub(" ", text).lower())
    found = {_TRAILING_ZEROS_RE.sub("", match.replace(",", "")) for match in _NUMBER_RE.findall(plain)}
    found |= {str(_NUMBER_WORDS[word]) for word in _WORD_RE.findall(plain) if word in _NUMBER_WORDS}
    return found
