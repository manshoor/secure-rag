"""Text checks in core: the injection phrase list and the number reader, shared by rag and the agent runtime."""

import ast
import inspect

from secure_rag.core import agent_runtime, textcheck
from secure_rag.core.textcheck import CITATION_RE, injection_findings, looks_injected, numbers_in
from secure_rag.rag import guard


def test_planted_phrases_are_found():
    assert injection_findings("Please ignore previous instructions now") == ["ignore previous instructions"]
    assert looks_injected("Disregard the previous quotes")
    assert not looks_injected("Glass block feature wall 100mm")


def test_numbers_read_as_plain_digits():
    assert numbers_in("AED 18,150.00 for three items [S2]") == {"18150", "3"}
    assert numbers_in("no one came at 12:30") == {"12:30"}


def test_citation_markers_are_matched():
    assert CITATION_RE.findall("see [S1] and [S12]") == ["S1", "S12"]


def test_guard_keeps_its_names_from_core():
    assert guard.injection_findings is injection_findings
    assert guard.looks_injected is looks_injected
    assert guard._numbers is numbers_in


def test_the_runtime_imports_core_only_at_the_top():
    source = inspect.getsource(agent_runtime)
    tree = ast.parse(source)
    nested = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) and node.col_offset > 0]
    assert nested == []
    assert "rag" not in {(node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert agent_runtime.looks_injected is textcheck.looks_injected
