"""Role-based access: no user can retrieve a document outside their companies and roles."""

import pytest

from secure_rag.answer import ask
from secure_rag.identity import DocAccess, can_read, get_user, list_users


@pytest.mark.parametrize(
    ("user_id", "company", "roles", "expected"),
    [
        ("sara", "group", {"employee"}, True),  # group-wide HR policy
        ("sara", "alpha-construction", {"procurement"}, False),  # wrong role
        ("lina", "alpha-construction", {"finance"}, False),  # right role, other company
        ("faisal", "beta-properties", {"finance"}, False),
        ("noura", "beta-properties", {"executive"}, True),
        ("omar", "group", {"executive"}, False),  # board memo
    ],
)
def test_access_rule(user_id, company, roles, expected):
    assert can_read(get_user(user_id), DocAccess(company, frozenset(roles))) is expected


def test_search_never_returns_unreadable_chunks(system):
    """Exhaustive: every user asks for every chunk's own text; nothing forbidden comes back."""
    for user in list_users():
        for chunk in system.report.chunks:
            for hit in system.store.search(user, f"{chunk.title} {chunk.section} {chunk.text}", 8):
                assert can_read(user, hit.chunk.access), (user.user_id, hit.chunk.chunk_id)


def test_vector_filter_alone_enforces_access(system):
    """The Qdrant payload filter blocks access even before the Python re-check."""
    omar = get_user("omar")
    readable = {i for i, c in enumerate(system.report.chunks) if can_read(omar, c.access)}
    for chunk in system.report.chunks:
        for index, _ in system.store._vector_search(omar, chunk.text):
            assert index in readable


def test_cross_company_question_is_refused(system):
    answer = ask(system.store, system.composer, get_user("lina"), "What retention applies under the steel subcontract?")
    assert answer.refused
    assert answer.citations == []


def test_hidden_canaries_are_only_for_unreadable_docs(system):
    assert system.store.hidden_canaries(get_user("omar")) == ["CANARY-BOARD-7731"]
    assert system.store.hidden_canaries(get_user("noura")) == []
