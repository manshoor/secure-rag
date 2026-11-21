"""Workflow over HTTP: start, upload, decide, view, and the /ui page."""

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from secure_rag.core import config
from secure_rag.web.api import MAX_UPLOAD_BYTES, create_app
from secure_rag.workflow.intake import OfflineIntake
from secure_rag.workflow.workflow import WorkflowRunner

REQUEST = "3 site engineers join on 1 Dec. Order laptops and safety kits."
VILLA = config.SAMPLES_DIR / "boq_villa_blockwork.xlsx"
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("RAG_TODAY", "2026-11-18")


@pytest.fixture
def client(system, runner):
    workflows = WorkflowRunner(system.store, system.composer, OfflineIntake(), InMemorySaver())
    with TestClient(create_app(system, runner, workflows)) as test_client:
        yield test_client


def _as(user: str) -> dict:
    return {"X-User": user}


def _start(client, user: str = "sara", text: str = REQUEST):
    return client.post("/workflows", json={"request": text}, headers=_as(user))


def _upload(client, name: str, content: bytes, user: str = "khalid", kind: str = XLSX_TYPE):
    return client.post("/workflows/boq", files={"file": (name, content, kind)}, headers=_as(user))


# ---- request path -------------------------------------------------------------------


def test_start_returns_priced_run_waiting_for_procurement(client):
    body = _start(client).json()
    assert body["status"] == "awaiting_procurement"
    assert body["total"] == "18150.00"
    assert {line["code"] for line in body["lines"]} == {"IT-LAPTOP-14", "PPE-KIT-STD"}


def test_user_without_hr_role_gets_403(client):
    assert _start(client, user="omar").status_code == 403


def test_both_gates_then_done(client):
    run_id = _start(client).json()["run_id"]
    path = f"/workflows/{run_id}/decision"
    assert client.post(path, json={"decision": "approve"}, headers=_as("omar")).json()["status"] == "awaiting_finance"
    assert client.post(path, json={"decision": "approve"}, headers=_as("faisal")).json()["status"] == "done"
    assert client.post(path, json={"decision": "approve"}, headers=_as("sara")).status_code == 409


def test_only_requester_and_eligible_approvers_can_view(client):
    run_id = _start(client).json()["run_id"]
    assert client.get(f"/workflows/{run_id}", headers=_as("sara")).status_code == 200
    assert client.get(f"/workflows/{run_id}", headers=_as("omar")).status_code == 200
    assert client.get(f"/workflows/{run_id}", headers=_as("lina")).status_code == 404
    assert client.post(f"/workflows/{run_id}/decision", json={"decision": "approve"}, headers=_as("lina")).status_code == 404
    assert client.get("/workflows/nope", headers=_as("sara")).status_code == 404


def test_policy_citation_follows_the_viewer(client):
    run_id = _start(client).json()["run_id"]
    assert client.get(f"/workflows/{run_id}", headers=_as("sara")).json()["policy"] == {"visible": False}
    assert client.get(f"/workflows/{run_id}", headers=_as("omar")).json()["policy"]["visible"] is True


# ---- BOQ upload ---------------------------------------------------------------------


def test_boq_upload_prices_the_sample_and_keeps_no_copy(client):
    body = _upload(client, "villa.xlsx", VILLA.read_bytes()).json()
    assert body["entry"] == "boq" and body["total"] == "215925.00"
    uploads = config.DATA_DIR / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_boq_upload_refuses_wrong_type_and_fake_content(client):
    assert _upload(client, "notes.txt", b"hello", kind="text/plain").status_code == 415
    assert _upload(client, "villa.xlsx", b"not a zip file").status_code == 415
    assert _upload(client, "villa.pdf", b"PK\x03\x04 not a pdf", kind="application/pdf").status_code == 415


def test_boq_upload_refuses_large_file(client):
    assert _upload(client, "big.csv", b"a" * (MAX_UPLOAD_BYTES + 1), kind="text/csv").status_code == 413


def test_boq_upload_needs_site_engineer_role(client):
    assert _upload(client, "villa.xlsx", VILLA.read_bytes(), user="sara").status_code == 403


# ---- ui page ------------------------------------------------------------------------


def test_ui_page_is_self_contained(client):
    assert client.get("/demo").status_code == 404
    response = client.get("/ui")
    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    page = response.text
    assert "<script src=\"http" not in page and "<link" not in page and "innerHTML" not in page


def test_ui_samples_are_allow_listed(client):
    assert client.get("/ui/samples/boq_villa_blockwork.xlsx").status_code == 200
    assert client.get("/ui/samples/pyproject.toml").status_code == 404
    assert client.get("/ui/samples/..%2F..%2Fpyproject.toml").status_code == 404


def test_ui_users_lists_sample_people(client):
    users = {user["user_id"] for user in client.get("/ui/users").json()}
    assert {"sara", "omar", "faisal", "noura", "lina", "khalid"} <= users
