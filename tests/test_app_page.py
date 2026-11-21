"""The /app chat page: served files, CSP header, a source scan, and the run fields the page needs."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag import config
from secure_rag.api import create_app
from secure_rag.intake import OfflineIntake
from secure_rag.workflow import WorkflowRunner

ISSUES = config.SAMPLES_DIR / "boq_with_issues.csv"
APP_DIR = Path(__file__).parent.parent / "src" / "secure_rag" / "static" / "app"


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")
    monkeypatch.setenv("RAG_DEMO", "on")


@pytest.fixture
def workflows(system):
    return WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())


@pytest.fixture
def client(system, runner, workflows):
    with TestClient(create_app(system, runner, workflows)) as test_client:
        yield test_client


def _as(user: str) -> dict:
    return {"X-User": user}


def _done_run(client) -> str:
    """The sample agent run, all the way to purchase orders."""
    files = {"file": ("boq.csv", ISSUES.read_bytes(), "text/csv")}
    run_id = client.post("/workflows/boq", files=files, data={"mode": "agent"}, headers=_as("khalid")).json()["run_id"]
    steps = [
        ("decision", "omar", {"decision": "approve", "gate": "rfq"}),
        ("events", "omar", {"event": "quotes_delivered", "sample": "round1"}),
        ("decision", "omar", {"decision": "extend", "gate": "quotes"}),
        ("events", "omar", {"event": "quotes_delivered", "sample": "round2"}),
        ("decision", "omar", {"decision": "approve", "gate": "procurement"}),
        ("decision", "faisal", {"decision": "approve", "gate": "finance"}),
    ]
    for route, user, body in steps:
        assert client.post(f"/workflows/{run_id}/{route}", json=body, headers=_as(user)).status_code == 200
    return run_id


# ---- the page and its files -------------------------------------------------------------


def test_app_page_has_a_strict_csp(client):
    response = client.get("/app")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    csp = response.headers["content-security-policy"]
    for directive in ("default-src 'none'", "script-src 'self'", "style-src 'self'", "connect-src 'self'", "frame-ancestors 'none'", "base-uri 'none'"):
        assert directive in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert response.headers["x-content-type-options"] == "nosniff"


def test_app_static_files_have_the_right_types(client):
    script = client.get("/app/static/app.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert "content-security-policy" in script.headers
    assert client.get("/app/static/app.css").headers["content-type"].startswith("text/css")


@pytest.mark.parametrize("path", ["/app/static/../workflow.html", "/app/static/%2e%2e/workflow.html", "/app/static/nope.js", "/app/static/"])
def test_app_static_refuses_other_paths(client, path):
    assert client.get(path).status_code == 404


def test_ui_page_is_unchanged(client):
    assert client.get("/ui").status_code == 200


# A page that only ever writes text: no HTML strings, no inline code or styles, nothing remote.
_FORBIDDEN = [
    r"\.innerHTML", r"\.outerHTML", r"insertAdjacentHTML", r"document\.write", r"\beval\(", r"new Function",
    r"\.cssText", r"setAttribute\(\s*['\"](style|on\w+)", r"https?://(?!www\.w3\.org/2000/svg\b)",
]
_FORBIDDEN_HTML = [r"<script(?![^>]*\bsrc=)", r"<style", r"\sstyle=", r"\son\w+=", r"https?://"]


def test_app_source_writes_text_only():
    scripts = sorted(APP_DIR.glob("*.js"))
    assert scripts, "no page scripts found"
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN:
            assert not re.search(pattern, source), f"{path.name}: {pattern}"
    page = (APP_DIR / "index.html").read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_HTML:
        assert not re.search(pattern, page), f"index.html: {pattern}"
    for path in APP_DIR.glob("*.css"):
        assert not re.search(r"https?://|@import", path.read_text(encoding="utf-8")), path.name


# ---- run fields the page shows ------------------------------------------------------------


def test_done_run_carries_orders_and_budget_after(client):
    run_id = _done_run(client)
    body = client.get(f"/workflows/{run_id}", headers=_as("khalid")).json()
    totals = {order["vendor"]: order["total"] for order in body["orders"]}
    assert totals == {
        "Al Noor Building Materials LLC": "6400.00", "Cobalt Dune Facades LLC": "7155.00",
        "Falcon Steel Fabrication LLC": "540.00", "Gulf Blocks Factory LLC": "18920.00",
    }
    assert body["budget_after"] == "566985.00"


def test_open_run_has_no_orders_and_names_its_vendors(client):
    files = {"file": ("boq.csv", ISSUES.read_bytes(), "text/csv")}
    body = client.post("/workflows/boq", files=files, data={"mode": "agent"}, headers=_as("khalid")).json()
    assert body["orders"] == []
    assert body["budget_after"] is None  # nothing is held before the quotes are in
    assert body["vendor_names"]["V-GLB-3"] == "Gulf Blocks Factory LLC"
    assert set(body["vendor_names"]) == {vendor for package in body["packages"] for vendor in package["vendors"]}


def _private_file(monkeypatch, tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    target = private / "boq.csv"
    target.write_bytes(ISSUES.read_bytes())
    monkeypatch.setattr(config, "PRIVATE_DIR", private)
    return target


def test_private_agent_run_hides_vendors_and_packages(client, workflows, monkeypatch, tmp_path):
    from secure_rag.identity import get_user
    from secure_rag.workflow import RunMode

    run_id = workflows.start_boq(get_user("khalid"), _private_file(monkeypatch, tmp_path), mode=RunMode.AGENT).run_id
    for viewer in ("khalid", "omar"):
        body = client.get(f"/workflows/{run_id}", headers=_as(viewer)).json()
        assert body["status"] == "awaiting_rfq"
        assert body["vendor_names"] == {} and body["packages"] == []


def test_classic_run_lists_no_vendor_orders(client):
    """A classic run writes one purchase order; per-vendor orders are agent mode only."""
    files = {"file": ("boq.csv", ISSUES.read_bytes(), "text/csv")}
    run_id = client.post("/workflows/boq", files=files, headers=_as("khalid")).json()["run_id"]
    client.post(f"/workflows/{run_id}/decision", json={"decision": "approve"}, headers=_as("omar"))
    body = client.post(f"/workflows/{run_id}/decision", json={"decision": "approve"}, headers=_as("faisal")).json()
    assert body["status"] == "done" and body["orders"] == []


def test_budget_after_is_empty_once_the_hold_is_released(client):
    files = {"file": ("boq.csv", ISSUES.read_bytes(), "text/csv")}
    run_id = client.post("/workflows/boq", files=files, headers=_as("khalid")).json()["run_id"]
    assert client.get(f"/workflows/{run_id}", headers=_as("khalid")).json()["budget_after"] is not None
    body = client.post(f"/workflows/{run_id}/decision", json={"decision": "reject"}, headers=_as("omar")).json()
    assert body["status"] == "rejected" and body["budget_after"] is None


def test_unknown_demo_setting_counts_as_off(client, monkeypatch):
    monkeypatch.setenv("RAG_DEMO", "1")
    response = client.get("/me", headers=_as("omar"))
    assert response.status_code == 200 and response.json()["demo"] is False


# ---- late replies in the page (Node with a fake DOM; skipped without Node) ---------------------------

_NODE = shutil.which("node")
_LATE_REPLY = Path(__file__).parent / "js" / "late_reply.mjs"


@pytest.mark.skipif(_NODE is None, reason="needs node")
@pytest.mark.parametrize("scenario", ["deliver_after_switch", "decide_after_switch", "wrong_case", "failed_poll"])
def test_page_drops_late_replies(scenario):
    result = subprocess.run([_NODE, str(_LATE_REPLY), scenario], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
