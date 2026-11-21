"""Budget holds: an open run reserves its total, so two runs cannot spend the same budget."""

import json
import shutil
import threading
from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config

from secure_rag.workflow import systems
from secure_rag.workflow.agent import Decision
from secure_rag.core.identity import get_user
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import WfStatus, WorkflowRunner

REQUEST = "3 site engineers join on 1 Dec. Order laptops and safety kits."
COMPANY, DEPARTMENT = "alpha-construction", "hr"
BUDGET = Decimal("30000.00")  # fits one request (18,150.00), not two


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


@pytest.fixture
def budget_file(tmp_path, monkeypatch):
    """A copy of the systems folder with alpha-construction / hr set to BUDGET."""
    folder = tmp_path / "systems"
    shutil.copytree(config.SYSTEMS_DIR, folder)
    monkeypatch.setenv("RAG_SYSTEMS_DIR", str(folder))
    path = folder / "budgets.json"
    _set_budget(path, BUDGET)
    return path


@pytest.fixture
def wf(system, budget_file) -> WorkflowRunner:
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())


def _set_budget(path, amount: Decimal) -> None:
    data = json.loads(path.read_text())
    data["budgets"][0]["remaining_aed"] = str(amount)  # alpha-construction / hr
    path.write_text(json.dumps(data))


def _start(wf):
    return wf.start_request(get_user("sara"), REQUEST)


# ---- an open run holds its total ------------------------------------------------------------


def test_second_open_run_cannot_use_the_held_budget(wf):
    first = _start(wf)
    assert first.status is WfStatus.AWAITING_PROCUREMENT
    second = _start(wf)
    assert second.status is WfStatus.STOPPED and second.pending_gate is None
    assert "budget short by AED 6,300.00" in second.reason  # 30,000 - 18,150 held = 11,850 left


def test_available_budget_counts_holds(wf):
    _start(wf)
    assert systems.available_budget(COMPANY, DEPARTMENT) == Decimal("11850.00")


def test_stopped_run_holds_nothing(wf):
    _start(wf)
    _start(wf)  # stops short
    assert systems.available_budget(COMPANY, DEPARTMENT) == Decimal("11850.00")


# ---- the hold ends with the run ---------------------------------------------------------------


def test_rejection_releases_the_hold(wf):
    first = _start(wf)
    wf.decide(first.run_id, get_user("omar"), Decision.REJECT)
    assert systems.available_budget(COMPANY, DEPARTMENT) == BUDGET
    assert _start(wf).status is WfStatus.AWAITING_PROCUREMENT


def test_finance_rejection_releases_the_hold(wf):
    first = _start(wf)
    wf.decide(first.run_id, get_user("omar"), Decision.APPROVE)
    wf.decide(first.run_id, get_user("faisal"), Decision.REJECT)
    assert systems.available_budget(COMPANY, DEPARTMENT) == BUDGET


def test_done_run_keeps_its_spend(wf):
    first = _start(wf)
    wf.decide(first.run_id, get_user("omar"), Decision.APPROVE)
    assert wf.decide(first.run_id, get_user("faisal"), Decision.APPROVE).status is WfStatus.DONE
    assert systems.available_budget(COMPANY, DEPARTMENT) == Decimal("11850.00")
    assert _start(wf).status is WfStatus.STOPPED


def test_finance_recheck_stop_releases_the_hold(wf, budget_file):
    first = _start(wf)
    _set_budget(budget_file, Decimal("10000.00"))  # the ERP cut the budget while the run waited
    after = wf.decide(first.run_id, get_user("omar"), Decision.APPROVE)
    assert after.status is WfStatus.STOPPED and after.stopped_at == "finance"
    assert systems.available_budget(COMPANY, DEPARTMENT) == Decimal("10000.00")


def test_own_hold_does_not_fail_the_finance_recheck(wf):
    first = _start(wf)
    assert wf.decide(first.run_id, get_user("omar"), Decision.APPROVE).status is WfStatus.AWAITING_FINANCE


# ---- the ledger itself ------------------------------------------------------------------------


def test_reserving_twice_for_one_run_holds_once(budget_file):
    systems.reserve_budget("run-a", COMPANY, DEPARTMENT, Decimal("18150.00"))
    systems.reserve_budget("run-a", COMPANY, DEPARTMENT, Decimal("18150.00"))
    assert systems.available_budget(COMPANY, DEPARTMENT) == Decimal("11850.00")


def test_parallel_reservations_never_overspend(budget_file):
    """Ten runs race for a budget that fits one: exactly one hold lands."""
    results: list[bool] = []
    barrier = threading.Barrier(10)

    def try_reserve(index: int) -> None:
        barrier.wait()
        try:
            systems.reserve_budget(f"run-{index}", COMPANY, DEPARTMENT, Decimal("18150.00"))
            results.append(True)
        except systems.BudgetShortError:
            results.append(False)

    threads = [threading.Thread(target=try_reserve, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1
    assert systems.available_budget(COMPANY, DEPARTMENT) == Decimal("11850.00")


def test_release_after_spend_keeps_the_spend(budget_file):
    systems.reserve_budget("run-a", COMPANY, DEPARTMENT, Decimal("18150.00"))
    systems.spend_budget("run-a", COMPANY, DEPARTMENT, Decimal("18150.00"))
    systems.release_budget("run-a")
    assert systems.available_budget(COMPANY, DEPARTMENT) == Decimal("11850.00")


# ---- review fixes: a failure after the hold neither loses money nor strands a run ---------------


class InterruptedSummarizer:
    """Stands in for Ctrl-C during the model call that writes the approver note."""

    def summarize(self, facts: dict):
        raise KeyboardInterrupt


def test_failure_after_the_hold_releases_it(system, budget_file):
    wf = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver(), InterruptedSummarizer())
    with pytest.raises(KeyboardInterrupt):
        _start(wf)
    assert systems.available_budget(COMPANY, DEPARTMENT) == BUDGET


def test_failed_outbox_write_resumes_to_one_order(wf, monkeypatch):
    from secure_rag.workflow import workflow

    first = _start(wf)
    wf.decide(first.run_id, get_user("omar"), Decision.APPROVE)
    real_write = workflow._write_outbox
    monkeypatch.setattr(workflow, "_write_outbox", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        wf.decide(first.run_id, get_user("faisal"), Decision.APPROVE)

    monkeypatch.setattr(workflow, "_write_outbox", real_write)
    done = wf.resume(first.run_id, get_user("sara"))
    assert done.status is WfStatus.DONE
    assert systems.available_budget(COMPANY, DEPARTMENT) == Decimal("11850.00")
    kinds = sorted(json.loads(line)["kind"] for line in (config.DATA_DIR / "outbox.jsonl").read_text().splitlines())
    assert kinds == ["it_ticket", "purchase_order"]


def test_spent_run_still_orders_after_requester_loses_role(wf, monkeypatch):
    """Money already spent must reach a purchase order, whatever changed since."""
    from secure_rag.core import identity
    from secure_rag.workflow import workflow
    from secure_rag.core.identity import User

    first = _start(wf)
    wf.decide(first.run_id, get_user("omar"), Decision.APPROVE)
    real_write = workflow._write_outbox
    monkeypatch.setattr(workflow, "_write_outbox", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        wf.decide(first.run_id, get_user("faisal"), Decision.APPROVE)

    monkeypatch.setattr(workflow, "_write_outbox", real_write)
    monkeypatch.setitem(identity._USERS, "sara", User("sara", "Sara", frozenset({COMPANY}), frozenset({"employee"})))
    assert wf.resume(first.run_id, get_user("faisal")).status is WfStatus.DONE


def test_resume_refuses_a_run_waiting_for_a_person(wf):
    first = _start(wf)
    with pytest.raises(ValueError):
        wf.resume(first.run_id, get_user("sara"))


def test_resume_needs_a_viewer_of_the_run(wf):
    first = _start(wf)
    with pytest.raises(LookupError):
        wf.resume(first.run_id, get_user("lina"))
