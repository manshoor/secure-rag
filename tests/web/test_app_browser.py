"""The /app chat page in a real browser: the demo path in two windows, text-only rendering,
keyboard approval, and the not-visible state.

Marked `ui`; skipped when Playwright's Chromium is not installed:

    uv run playwright install chromium
    uv run pytest -m ui
"""

import re
import socket
import threading
import time

import pytest
import uvicorn
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config
from secure_rag.web.api import create_app
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import WorkflowRunner

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.ui

ISSUES = config.SAMPLES_DIR / "boq_with_issues.csv"
WAIT_MS = 10_000  # list polls run every 5 s; case polls every 1.5 s
SYNC_MS = 2_000  # acceptance: a decision in one window shows in the other within 2 s (plus one poll of slack)


@pytest.fixture
def server(system, runner, monkeypatch):
    """The app on a free local port, offline, with demo controls on."""
    monkeypatch.setenv("RAG_DEMO", "on")
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    workflows = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app_server = uvicorn.Server(uvicorn.Config(create_app(system, runner, workflows), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=app_server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not app_server.started:
        if time.monotonic() > deadline:
            pytest.fail("server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    app_server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def browser():
    with sync_api.sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch()
        except sync_api.Error as error:
            pytest.skip(f"Chromium not installed: {error}")
        yield chromium
        chromium.close()


def _window(browser, base: str, user: str, case: str = ""):
    """A separate browser window (own storage) acting as one person."""
    page = browser.new_context(viewport={"width": 1440, "height": 1000}).new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{base}/app?as={user}{f'#case={case}' if case else ''}")
    page.get_by_text(f"Hello {user.capitalize()}").wait_for()
    page.errors = errors
    return page


def _start_sample(page) -> str:
    page.get_by_role("button", name="Price a BOQ").click()
    page.get_by_role("button", name="Use sample boq_with_issues.csv").click()
    started = page.get_by_text(re.compile(r"Case \w+ started"))
    started.wait_for()
    return re.search(r"Case (\w+) started", started.inner_text()).group(1)


def _upload(page, path) -> str:
    page.locator("#file").set_input_files(str(path))
    page.get_by_role("button", name="Send").click()
    started = page.get_by_text(re.compile(r"Case \w+ started"))
    started.wait_for()
    return re.search(r"Case (\w+) started", started.inner_text()).group(1)


def _click(page, name: str):
    button = page.get_by_role("button", name=name, exact=True)
    button.wait_for(timeout=WAIT_MS)
    button.click()


def _open_waiting(page, run_id: str):
    page.get_by_role("navigation", name="Cases").get_by_role("button", name=re.compile(run_id)).click(timeout=WAIT_MS)


# ---- the demo path ------------------------------------------------------------------------------


def test_demo_path_in_two_windows(server, browser):
    khalid = _window(browser, server, "khalid")
    run_id = _start_sample(khalid)
    khalid.get_by_text("Waiting: quote requests").first.wait_for()

    # Omar's window: the case shows up in "Waiting on you".
    omar = _window(browser, server, "omar")
    _open_waiting(omar, run_id)
    _click(omar, "Approve sending")

    # Khalid's open case follows within a poll.
    khalid.get_by_text("Waiting: vendor replies").first.wait_for(timeout=SYNC_MS + 1_500)

    _click(omar, "Simulate vendor replies")
    omar.get_by_text("Refused · vendor not approved").wait_for()
    _click(omar, "Extend once")
    _click(omar, "Simulate vendor replies")
    _click(omar, "Approve")

    faisal = _window(browser, server, "faisal")
    _open_waiting(faisal, run_id)
    _click(faisal, "Approve AED 33,015.00")
    faisal.get_by_text("4 purchase orders, one per vendor").wait_for()

    khalid.get_by_text("4 purchase orders, one per vendor").wait_for(timeout=SYNC_MS + 1_500)
    assert khalid.get_by_role("cell", name="18,920.00").is_visible()
    for page in (khalid, omar, faisal):
        assert page.errors == []


# ---- safety and access ----------------------------------------------------------------------------


def test_markup_in_a_boq_shows_as_text(server, browser, tmp_path):
    payload = '<img src=x onerror="window.__owned=1">'
    boq = tmp_path / "boq.csv"
    boq.write_text(ISSUES.read_text(encoding="utf-8").replace("Decorative GRC cornice", f"Decorative GRC cornice {payload}"), encoding="utf-8")
    khalid = _window(browser, server, "khalid")
    _upload(khalid, boq)
    khalid.get_by_role("tab", name="Lines").click()
    khalid.locator("#tab-panel").get_by_text(payload, exact=False).first.wait_for()
    assert khalid.locator("main img, aside img").count() == 0
    assert khalid.evaluate("window.__owned") is None


def test_keyboard_only_approval(server, browser):
    run_id = _start_sample(_window(browser, server, "khalid"))
    omar = _window(browser, server, "omar")
    _open_waiting(omar, run_id)
    omar.get_by_role("button", name="Approve sending").wait_for()

    # Tab from the top of the page to the button; focus must be visible on it.
    omar.locator("body").click(position={"x": 1, "y": 1})
    for _ in range(120):
        omar.keyboard.press("Tab")
        if omar.evaluate("document.activeElement?.textContent") == "Approve sending":
            break
    else:
        pytest.fail("Approve sending is not reachable with Tab")
    assert omar.evaluate("getComputedStyle(document.activeElement).outlineStyle") != "none"
    omar.keyboard.press("Enter")
    omar.get_by_text("Waiting: vendor replies").first.wait_for()


def test_other_company_sees_not_visible(server, browser):
    run_id = _start_sample(_window(browser, server, "khalid"))
    lina = _window(browser, server, "lina", case=run_id)
    lina.get_by_text("Not found or not visible to you").wait_for()
    assert lina.get_by_text(run_id).count() == 0


# ---- nothing priced from the catalogue -----------------------------------------------------------


def test_a_case_with_nothing_priced_is_not_shown_as_private(server, browser, tmp_path):
    boq = tmp_path / "quotes_only.csv"
    boq.write_text("Item,Description,Unit,Qty\n2.2,Decorative GRC cornice,LM,45\n")
    khalid = _window(browser, server, "khalid")
    run_id = _upload(khalid, boq)
    khalid.get_by_text("Waiting: quote requests").first.wait_for()
    assert khalid.get_by_text("Private file").count() == 0

    # Refused requests leave nothing to hold; the case says why it stopped.
    omar = _window(browser, server, "omar")
    _open_waiting(omar, run_id)
    omar.get_by_text("Rule: set by the quote total.").wait_for()
    _click(omar, "Reject")
    omar.get_by_text("needs a person: quote requests refused").first.wait_for()
    omar.get_by_text("Nothing held.").wait_for()
    assert omar.get_by_text("Private file").count() == 0
    assert khalid.errors == [] and omar.errors == []


def test_the_finance_card_shows_the_rule_for_the_quoted_total(server, browser, tmp_path):
    """4,620.00 + 7,155.00 = 11,775.00: three quotes, never the placeholder from before the quotes."""
    boq = tmp_path / "quotes_only.csv"
    boq.write_text("Item,Description,Unit,Qty\n1.4,Glass block feature wall 100mm,m2,12\n2.2,Decorative GRC cornice,LM,45\n")
    run_id = _upload(_window(browser, server, "khalid"), boq)
    omar = _window(browser, server, "omar")
    _open_waiting(omar, run_id)
    for button in ("Approve sending", "Simulate vendor replies", "Extend once", "Simulate vendor replies", "Approve"):
        _click(omar, button)

    faisal = _window(browser, server, "faisal")
    _open_waiting(faisal, run_id)
    faisal.get_by_role("button", name="Approve AED 11,775.00").wait_for()
    faisal.get_by_text("Rule: three written quotations.").wait_for()
    assert faisal.get_by_text("Rule: set by the quote total.").count() == 0
