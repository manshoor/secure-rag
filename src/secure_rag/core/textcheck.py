"""Text checks that do not depend on any layer above core.

    injection_findings / looks_injected   phrases that address the model, not a human reader
    numbers_in                            the numbers a text states, as plain digits

rag.guard uses both for documents and answers; the agent runtime uses both to
screen a model's free text. Results name phrases and digits, never whole texts.
"""

import re

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

# Citation markers the model writes inline, e.g. [S2]. A marker is not a number.
CITATION_RE = re.compile(r"\[(S\d+)\]")

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


def injection_findings(text: str) -> list[str]:
    """Return the suspicious phrases found in text (empty list if clean)."""
    return [match.group(0) for match in _INJECTION_RE.finditer(text)]


def looks_injected(text: str) -> bool:
    """True when text contains at least one instruction aimed at the model."""
    return bool(injection_findings(text))


def numbers_in(text: str) -> set[str]:
    """Numbers as plain digits: '25,000' -> '25000', '18,150.00' -> '18150', 'three' -> '3'."""
    plain = _PRONOUN_ONE_RE.sub(" ", CITATION_RE.sub(" ", text).lower())
    found = {_TRAILING_ZEROS_RE.sub("", match.replace(",", "")) for match in _NUMBER_RE.findall(plain)}
    found |= {str(_NUMBER_WORDS[word]) for word in _WORD_RE.findall(plain) if word in _NUMBER_WORDS}
    return found
