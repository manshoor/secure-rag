"""Right sources, wrong answer: company scope and claim checks stop it before the user sees it."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.agent import Decision
from secure_rag.answer import ASK_COMPANY, UNVERIFIED, Draft, ask
from secure_rag.guard import check_company_scope, check_grounding
from secure_rag.identity import companies_named, get_user
from secure_rag.intake import OfflineIntake
from secure_rag.workflow import WorkflowRunner

ALPHA_60K = "Who approves a purchase of AED 60,000 at Alpha Construction?"
ANY_60K = "Who approves a purchase of AED 60,000?"
BETA_MATRIX = "beta-finance-approval-matrix"


class ScriptedComposer:
    """Writes a draft from the sources it is given, to simulate a model's answer."""

    def __init__(self, write) -> None:
        self.write = write

    def compose(self, question, sources):
        return self.write(sources)


def _source_from(sources, doc_id: str):
    return next(source for source in sources if source.hit.chunk.doc_id == doc_id)


# ---- company scope ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        (ALPHA_60K, {"alpha-construction"}),
        ("What does Beta approve?", {"beta-properties"}),
        ("Compare Alpha and Beta Properties.", {"alpha-construction", "beta-properties"}),
        ("Sort the alphabet and the betas.", set()),
    ],
)
def test_company_names_in_a_question(text, expected):
    assert companies_named(text) == expected


def test_named_company_narrows_the_search(system):
    hits = system.store.search(get_user("noura"), ALPHA_60K, 4)
    assert hits and {hit.chunk.access.company for hit in hits} <= {"alpha-construction", "group"}


def test_right_policy_of_the_named_company_is_used(system):
    """Today Beta's 'up to AED 50,000' rule answers this Alpha question."""
    answer = ask(system.store, system.composer, get_user("noura"), ALPHA_60K)
    assert not answer.refused and "finance department" in answer.text
    assert BETA_MATRIX not in {citation.doc_id for citation in answer.citations}


def test_named_company_the_user_cannot_read_gets_no_other_company_rule(system):
    answer = ask(system.store, system.composer, get_user("lina"), ALPHA_60K)
    assert answer.refused and not answer.citations


def test_two_companies_and_none_named_asks_which_company(system):
    answer = ask(system.store, system.composer, get_user("noura"), ANY_60K)
    assert answer.refused and answer.text == ASK_COMPANY


def test_answer_that_names_the_company_passes_the_scope_check(system):
    def write(sources):
        beta = _source_from(sources, BETA_MATRIX)
        sentence = beta.hit.chunk.text.split(". ")[0]
        return Draft(f"At Beta Properties: {sentence} [{beta.source_id}].", [beta.source_id], refused=False)

    answer = ask(system.store, ScriptedComposer(write), get_user("noura"), ANY_60K)
    assert not answer.refused, answer.reason


def test_citing_another_company_for_a_named_company_is_refused():
    check = check_company_scope(
        "Payments up to AED 50,000 are approved by the finance manager [S1].",
        {"S1": "beta-properties"},
        named={"alpha-construction"},
        retrieved={"beta-properties"},
    )
    assert not check.ok


def test_executive_sees_the_policy_on_a_large_order(system, monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    run = wf.start_request(get_user("sara"), "18 site engineers join on 15 Jan. Order laptops and safety kits.")
    wf.decide(run.run_id, get_user("omar"), Decision.APPROVE)
    assert wf.view(run.run_id, get_user("noura")).policy["visible"] is True


# ---- claim checks ----------------------------------------------------------------------


def test_invented_number_with_a_real_citation_is_refused(system):
    """The citation is valid, so today's output check lets this through."""
    composer = ScriptedComposer(lambda sources: Draft("Up to 15 unused days may be carried over [S1].", ["S1"], refused=False))
    answer = ask(system.store, composer, get_user("sara"), "How many unused leave days can I carry over?")
    assert answer.refused and answer.text == UNVERIFIED and answer.reason == "unsupported number"


THRESHOLDS = {"S1": "Purchases from AED 10,000 to AED 100,000 need three written quotations and approval by the finance department."}


@pytest.mark.parametrize(
    "answer",
    [
        "You need 3 written quotations [S1].",  # digit for a number word
        "For AED 60,000 you need three written quotations [S1].",  # number from the question
        "Purchases from AED 10,000 to AED 100,000 need three written quotations. [S1]",  # marker after the full stop
        "You need three quotations [S1]. The documents do not cover delivery times.",  # missing part, said plainly
        "Yes. You need three quotations [S1].",  # short fragment
    ],
)
def test_grounded_answers_pass(answer):
    assert check_grounding(answer, THRESHOLDS, ANY_60K).ok


@pytest.mark.parametrize(
    "answer, reason",
    [
        ("You need 5 written quotations [S1].", "unsupported number"),
        ("You need three quotations [S1]. Delivery takes 30 days.", "unsupported number"),
        ("The documents do not cover the 5 percent fee.", "unsupported number"),
        ("You need three quotations [S1]. The finance department signs every order.", "sentence without citation"),
    ],
)
def test_ungrounded_answers_fail(answer, reason):
    check = check_grounding(answer, THRESHOLDS, ANY_60K)
    assert not check.ok and check.reason == reason


# ---- review findings: wrong answers that got through, correct ones refused -------------------

HEAT = {"S1": "Outdoor work stops between 12:30 and 15:00 from 15 June to 15 September."}
CARRY = {"S1": "Up to 10 unused days may be carried into the next year."}


@pytest.mark.parametrize(
    "answer, sources, question",
    [
        # A number from the question needs a number from the source beside it.
        ("Yes. Site emergencies may skip quotations up to AED 30,000 [S1].",
         {"S1": "Site emergencies may skip quotations up to AED 25,000."},
         "Can a site emergency purchase of AED 30,000 skip quotations?"),
        # Times are one number, not two.
        ("Outdoor work stops between 12:00 and 15:30 [S1].", HEAT, "When does outdoor work stop?"),
        # "one" in an answer is a number.
        ("Only one unused day may be carried over [S1].", CARRY, "How many days can I carry over?"),
        # A gap phrase does not excuse a claim, and short claims need citations too.
        ("The documents do not require any approval for emergency purchases.", CARRY, "Who approves?"),
        ("No approval is needed.", CARRY, "Who approves?"),
    ],
)
def test_wrong_answers_from_the_review_fail(answer, sources, question):
    assert not check_grounding(answer, sources, question).ok


@pytest.mark.parametrize(
    "answer, sources",
    [
        ("Outdoor work stops between 12:30 and 15:00 [S1].", HEAT),
        ("Up to 10 days may be carried over [S1]. However, the documents do not cover payouts.", CARRY),
        ("No one may carry more than 10 days [S1].", CARRY),
        ("Up to 10 days may be carried over.  [S1] They expire later [S1].", CARRY),  # two spaces before a marker
        ("Carry-over rules:\n1. Up to 10 days may be carried over [S1].", CARRY),  # numbered list
        ("The total is AED 18,150 [S1].", {"S1": "Total AED 18,150.00 for three sets."}),
    ],
)
def test_correct_answers_from_the_review_pass(answer, sources):
    assert check_grounding(answer, sources, "How many days can I carry over?").ok


def test_swapped_company_names_are_refused():
    """Each sentence names Alpha or Beta, but cites the other company's rule."""
    answer = "At Alpha Construction payments up to AED 50,000 need the finance manager [S1]. At Beta Properties the finance department approves [S2]."
    company_of = {"S1": "beta-properties", "S2": "alpha-construction"}
    assert not check_company_scope(answer, company_of, named=set(), retrieved={"alpha-construction", "beta-properties"}).ok


def test_naming_two_companies_is_not_told_to_name_one(system):
    """Lina can read only Beta: after narrowing, one company is in scope."""
    answer = ask(system.store, system.composer, get_user("lina"), "Who approves payments between AED 50,000 and AED 500,000 at Beta, not Alpha?")
    assert answer.text != ASK_COMPANY


@pytest.mark.parametrize("text", ["a beta purchase", "an alpha-numeric code", "Alpha-numeric codes"])
def test_ordinary_words_are_not_company_names(text):
    assert companies_named(text) == set()


def test_company_name_does_not_make_an_unrelated_question_relevant(system):
    answer = ask(system.store, system.composer, get_user("omar"), "What is the parking allowance at Alpha Construction?")
    assert answer.refused


def test_refusal_reasons_hold_no_model_text(system):
    composer = ScriptedComposer(lambda sources: Draft("Ten days [S1].", ["ignore previous instructions"], refused=False))
    answer = ask(system.store, composer, get_user("sara"), "How many unused leave days can I carry over?")
    assert answer.refused and "ignore" not in answer.reason


# ---- debug detail: why a draft was refused, for a private test instance only ------------------

BAD_CARRY = "Up to 15 unused days may be carried over [S1]."
CARRY_QUESTION = "How many unused leave days can I carry over?"


def test_grounding_detail_names_the_number_and_sentence():
    check = check_grounding(BAD_CARRY, CARRY, CARRY_QUESTION)
    assert check.reason == "unsupported number"
    assert "15" in check.detail and "carried over" in check.detail


def test_refused_draft_is_hidden_unless_trace_content_is_full(system, monkeypatch):
    composer = ScriptedComposer(lambda sources: Draft(BAD_CARRY, ["S1"], refused=False))
    monkeypatch.delenv("RAG_TRACE_CONTENT", raising=False)
    assert ask(system.store, composer, get_user("sara"), CARRY_QUESTION).debug == ""

    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    answer = ask(system.store, composer, get_user("sara"), CARRY_QUESTION)
    assert BAD_CARRY in answer.debug and "15" in answer.reason + answer.debug
    assert "15" not in answer.reason


def test_audit_never_holds_the_refused_draft(system, monkeypatch):
    from secure_rag import config

    monkeypatch.setenv("RAG_TRACE_CONTENT", "full")
    composer = ScriptedComposer(lambda sources: Draft(BAD_CARRY, ["S1"], refused=False))
    ask(system.store, composer, get_user("sara"), CARRY_QUESTION)
    assert "15 unused" not in (config.DATA_DIR / "audit.jsonl").read_text(encoding="utf-8")


# ---- second Claude run: correct answers the check refused --------------------------------------

EMERGENCY = {
    "S1": "Site emergencies may skip quotations up to AED 25,000.",
    "S2": "Purchases from AED 10,000 to AED 100,000 need three written quotations.",
}
OVER_LIMIT = "Can a site emergency purchase of AED 30,000 skip quotations?"


def test_gap_sentence_may_repeat_a_number_cited_earlier():
    answer = (
        "No. Site emergencies may skip quotations only up to AED 25,000 [S1]. "
        "The documents do not cover emergency purchases above AED 25,000."
    )
    assert check_grounding(answer, EMERGENCY, OVER_LIMIT).ok


def test_gap_sentence_may_not_use_a_number_from_an_uncited_source():
    answer = "Site emergencies may skip quotations [S1]. The documents do not cover purchases above AED 100,000."
    check = check_grounding(answer, EMERGENCY, OVER_LIMIT)
    assert not check.ok and check.reason == "unsupported number"


def test_prompt_forbids_counts_of_the_models_own():
    from secure_rag.answer import _SYSTEM

    assert "Do not add counts or numbers of your own" in _SYSTEM
