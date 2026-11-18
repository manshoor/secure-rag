"""HTTP API over the same pipeline the CLI uses.

    POST /ask                          question -> cited answer or refusal
    POST /agent/runs                   request  -> run (done, denied, or awaiting approval)
    POST /agent/runs/{run_id}/decision approve | reject, as the calling user
    POST /workflows                    HR request -> HR check -> procurement -> finance
    POST /workflows/boq                BOQ upload (xlsx, csv, pdf) -> priced lines -> gates
    POST /workflows/{run_id}/decision  approve | reject the waiting gate, as the calling user
    GET  /workflows/{run_id}           the run as the caller may see it
    GET  /ui                           one static page that drives the workflow
    GET  /me                           who the API thinks you are
    GET  /health

Identity comes from the X-User header, resolved in current_user().
That is the one place to swap in real sign-in (an Entra ID token whose
groups map to companies and roles). Nothing in a request body can change
who the user is or what they may read.

    uv run uvicorn secure_rag.api:app --port 8010
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import config, db, tracing
from .agent import AgentRunner, Decision, Outcome
from .answer import Answer, ask
from .app import System, build_system, build_workflows
from .identity import User, get_user, list_users
from .workflow import Gate, RunView, WorkflowRunner

_USER_HEADER = "X-User"
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_HTTP_NOT_FOUND = 404
_HTTP_CONFLICT = 409
_HTTP_LENGTH_REQUIRED = 411
_HTTP_TOO_LARGE = 413
_HTTP_UNSUPPORTED = 415
_MAX_QUESTION_CHARS = 2000

# BOQ uploads: small files of known types only, checked by content, not by name.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
_UPLOAD_MAGIC = {".xlsx": b"PK\x03\x04", ".pdf": b"%PDF-"}  # .csv: UTF-8 text, checked below
_UPLOAD_SUFFIXES = {".xlsx", ".csv", ".pdf"}
_UPLOAD_DIR = "uploads"
_UPLOAD_PATH = "/workflows/boq"
_FORM_OVERHEAD_BYTES = 64 * 1024  # multipart boundaries and headers around the file

# The workflow page and the invented samples it can load. Nothing else is served.
_STATIC_DIR = Path(__file__).parent / "static"
_UI_SAMPLES = frozenset({"boq_villa_blockwork.xlsx", "boq_villa_blockwork.pdf", "boq_with_issues.csv"})


# ---- request and response bodies -------------------------------------------


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=_MAX_QUESTION_CHARS)


class CitationOut(BaseModel):
    source_id: str
    doc_id: str
    title: str
    section: str


class AnswerOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    refused: bool
    reason: str
    ms: int
    tokens: int
    cost_usd: float
    trace_id: str


class RunBody(BaseModel):
    request: str = Field(min_length=1, max_length=_MAX_QUESTION_CHARS)


class DecisionBody(BaseModel):
    decision: Decision


class RunOut(BaseModel):
    run_id: str
    status: str
    reason: str
    result: str
    pending: dict | None


class WorkflowBody(BaseModel):
    request: str = Field(min_length=1, max_length=_MAX_QUESTION_CHARS)


class WorkflowDecisionBody(BaseModel):
    decision: Decision
    gate: Gate | None = None  # the gate the approver saw; a different waiting gate gives 409


class WorkflowOut(BaseModel):
    """A run as the caller may see it. Money is a string so no float rounding creeps in."""

    run_id: str
    entry: str
    requester: str
    status: str
    reason: str
    stopped_at: str
    pending_gate: str | None
    steps: list[dict]
    hires: list[dict]
    lines: list[dict]
    rfq: list[dict]
    review: list[dict]
    boq_stats: dict
    total: str
    budget_remaining: str | None
    approver_roles: list[str]
    approvals: dict
    refused_vendors: list[str]
    flags: list[str]
    summaries: dict
    policy: dict
    cost_usd: float
    tokens: int


class MeOut(BaseModel):
    user_id: str
    name: str
    companies: list[str]
    roles: list[str]


# ---- app --------------------------------------------------------------------


def create_app(
    system: System | None = None, runner: AgentRunner | None = None, workflows: WorkflowRunner | None = None
) -> FastAPI:
    """Build the app. Tests pass an offline system; production builds from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ingest once per process; every request shares the index and the runner.
        app.state.system = system or build_system()
        app.state.runner = runner or AgentRunner(
            app.state.system.store, app.state.system.composer, app.state.system.planner
        )
        app.state.workflows = workflows or build_workflows(app.state.system)
        yield
        tracing.flush()

    app = FastAPI(title="Secure RAG", lifespan=lifespan)
    app.middleware("http")(_upload_size_guard)
    app.add_api_route("/health", _health, methods=["GET"])
    app.add_api_route("/me", _me, methods=["GET"], response_model=MeOut)
    app.add_api_route("/ask", _ask, methods=["POST"], response_model=AnswerOut)
    app.add_api_route("/agent/runs", _start_run, methods=["POST"], response_model=RunOut)
    app.add_api_route("/agent/runs/{run_id}/decision", _decide, methods=["POST"], response_model=RunOut)
    app.add_api_route("/workflows", _wf_start, methods=["POST"], response_model=WorkflowOut)
    app.add_api_route("/workflows/boq", _wf_boq, methods=["POST"], response_model=WorkflowOut)
    app.add_api_route("/workflows/{run_id}/decision", _wf_decide, methods=["POST"], response_model=WorkflowOut)
    app.add_api_route("/workflows/{run_id}", _wf_show, methods=["GET"], response_model=WorkflowOut)
    app.add_api_route("/ui", _ui_page, methods=["GET"], include_in_schema=False)
    app.add_api_route("/ui/users", _ui_users, methods=["GET"], include_in_schema=False)
    app.add_api_route("/ui/samples/{name}", _ui_sample, methods=["GET"], include_in_schema=False)
    return app


def current_user(x_user: Annotated[str | None, Header()] = None) -> User:
    """Resolve the caller. Local: a header naming a sample user; production: a verified token."""
    if config.auth_mode() is not config.AuthMode.HEADER:
        raise HTTPException(_HTTP_UNAUTHORIZED, "header sign-in is off; set RAG_AUTH=header for local use")
    if not x_user:
        raise HTTPException(_HTTP_UNAUTHORIZED, f"missing {_USER_HEADER} header")
    try:
        return get_user(x_user)
    except KeyError:
        raise HTTPException(_HTTP_UNAUTHORIZED, "unknown user") from None


CurrentUser = Annotated[User, Depends(current_user)]


# ---- routes -----------------------------------------------------------------
# Sync handlers: FastAPI runs them in a thread pool, so blocking model calls
# do not stall the event loop.


def _health() -> dict:
    """Also says whether Postgres and Qdrant are on: without them, users and access are the built-in sample set."""
    return {"status": "ok", "tracing": tracing.enabled(), "postgres": db.enabled(), "qdrant": bool(config.qdrant_url())}


def _me(user: CurrentUser) -> MeOut:
    return MeOut(user_id=user.user_id, name=user.name, companies=sorted(user.companies), roles=sorted(user.roles))


def _ask(body: AskBody, user: CurrentUser, request: Request) -> AnswerOut:
    system: System = request.app.state.system
    answer = ask(system.store, system.composer, user, body.question)
    return _answer_out(answer)


def _start_run(body: RunBody, user: CurrentUser, request: Request) -> RunOut:
    runner: AgentRunner = request.app.state.runner
    return _run_out(runner.start(user, body.request))


def _decide(run_id: str, body: DecisionBody, user: CurrentUser, request: Request) -> RunOut:
    """The caller is the approver. The graph ignores an ineligible one and keeps waiting.

    Only the requester and eligible approvers learn anything about a run; for
    everyone else it does not exist (same 404 as an unknown id).
    """
    runner: AgentRunner = request.app.state.runner
    if not runner.may_view(run_id, user):
        raise HTTPException(_HTTP_NOT_FOUND, "run not found")
    try:
        return _run_out(runner.resume(run_id, user, body.decision))
    except ValueError as error:
        raise HTTPException(_HTTP_CONFLICT, str(error)) from None


# ---- workflow routes ------------------------------------------------------------


def _wf_start(body: WorkflowBody, user: CurrentUser, request: Request) -> WorkflowOut:
    workflows: WorkflowRunner = request.app.state.workflows
    try:
        return _wf_out(workflows.start_request(user, body.request))
    except PermissionError as error:
        raise HTTPException(_HTTP_FORBIDDEN, str(error)) from None


def _wf_boq(user: CurrentUser, request: Request, file: Annotated[UploadFile, File()]) -> WorkflowOut:
    """Save the upload under a random name, run the BOQ path, then delete the file.

    The parse step is the only reader and runs before the first gate, so the
    file is gone before the response. Priced, RFQ and review lines (with their
    descriptions) stay in the run's checkpoint, like any run.
    """
    workflows: WorkflowRunner = request.app.state.workflows
    suffix = _checked_suffix(file.filename or "")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(_HTTP_TOO_LARGE, f"BOQ files are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    _check_content(suffix, content)

    folder = config.DATA_DIR / _UPLOAD_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(content)
    try:
        return _wf_out(workflows.start_boq(user, path))
    except PermissionError as error:
        raise HTTPException(_HTTP_FORBIDDEN, str(error)) from None
    finally:
        path.unlink(missing_ok=True)


async def _upload_size_guard(request: Request, call_next):
    """Refuse an oversized BOQ upload from its Content-Length, before the form is spooled to disk."""
    if request.url.path == _UPLOAD_PATH and request.method == "POST":
        length = request.headers.get("content-length")
        if length is None or not length.isdigit():
            return JSONResponse({"detail": "uploads need a Content-Length header"}, status_code=_HTTP_LENGTH_REQUIRED)
        if int(length) > MAX_UPLOAD_BYTES + _FORM_OVERHEAD_BYTES:
            return JSONResponse({"detail": f"BOQ files are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}, status_code=_HTTP_TOO_LARGE)
    return await call_next(request)


def _wf_decide(run_id: str, body: WorkflowDecisionBody, user: CurrentUser, request: Request) -> WorkflowOut:
    """Same rule as /agent/runs: a run you may not see does not exist for you."""
    workflows: WorkflowRunner = request.app.state.workflows
    if not workflows.may_view(run_id, user):
        raise HTTPException(_HTTP_NOT_FOUND, "run not found")
    try:
        return _wf_out(workflows.decide(run_id, user, body.decision, body.gate))
    except ValueError as error:
        raise HTTPException(_HTTP_CONFLICT, str(error)) from None


def _wf_show(run_id: str, user: CurrentUser, request: Request) -> WorkflowOut:
    workflows: WorkflowRunner = request.app.state.workflows
    try:
        return _wf_out(workflows.view(run_id, user))
    except LookupError:
        raise HTTPException(_HTTP_NOT_FOUND, "run not found") from None


def _checked_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(_HTTP_UNSUPPORTED, f"BOQ files must be one of {sorted(_UPLOAD_SUFFIXES)}")
    return suffix


def _check_content(suffix: str, content: bytes) -> None:
    """The bytes must match the extension: zip for xlsx, %PDF for pdf, UTF-8 text for csv."""
    if suffix in _UPLOAD_MAGIC:
        if content.startswith(_UPLOAD_MAGIC[suffix]):
            return
        raise HTTPException(_HTTP_UNSUPPORTED, f"the file content is not {suffix}")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(_HTTP_UNSUPPORTED, "a CSV BOQ must be UTF-8 text") from None
    if "\x00" in text:
        raise HTTPException(_HTTP_UNSUPPORTED, "a CSV BOQ must be plain text")


def _wf_out(view: RunView) -> WorkflowOut:
    """Runs on private files show counts only, like the CLI."""
    hidden = view.private
    return WorkflowOut(
        run_id=view.run_id,
        entry=view.entry.value,
        requester=view.requester,
        status=view.status.value,
        reason=view.reason,
        stopped_at=view.stopped_at,
        pending_gate=view.pending_gate.value if view.pending_gate else None,
        steps=view.steps,
        hires=view.hires,
        lines=[] if hidden else view.lines,
        rfq=[] if hidden else view.rfq,
        review=[] if hidden else view.review,
        boq_stats={key: value for key, value in view.boq_stats.items() if key != "sections"} if hidden else view.boq_stats,
        total=f"{view.total:.2f}",
        budget_remaining=f"{view.budget_remaining:.2f}" if view.budget_remaining is not None else None,
        approver_roles=view.approver_roles,
        approvals=view.approvals,
        refused_vendors=view.refused_vendors,
        flags=[] if hidden else view.flags,
        summaries=view.summaries,
        policy=view.policy,
        cost_usd=view.cost_usd,
        tokens=view.tokens,
    )


# ---- ui page --------------------------------------------------------------------------


def _ui_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "workflow.html", media_type="text/html")


def _ui_users() -> list[dict]:
    """The sample people the page can act as. Local header sign-in only."""
    if config.auth_mode() is not config.AuthMode.HEADER:
        raise HTTPException(_HTTP_UNAUTHORIZED, "header sign-in is off")
    return [{"user_id": u.user_id, "name": u.name, "roles": sorted(u.roles)} for u in list_users()]


def _ui_sample(name: str) -> FileResponse:
    """Only the invented samples on the allow-list; never a path from the request."""
    if name not in _UI_SAMPLES:
        raise HTTPException(_HTTP_NOT_FOUND, "no such sample")
    return FileResponse(config.SAMPLES_DIR / name, filename=name)


def _answer_out(answer: Answer) -> AnswerOut:
    return AnswerOut(
        answer=answer.text,
        citations=[CitationOut(**vars(c)) for c in answer.citations],
        refused=answer.refused,
        reason=answer.reason,
        ms=round(answer.seconds * 1000),
        tokens=answer.usage.input_tokens + answer.usage.output_tokens,
        cost_usd=round(answer.usage.cost_usd, 6),
        trace_id=answer.trace_id,
    )


def _run_out(outcome: Outcome) -> RunOut:
    return RunOut(
        run_id=outcome.thread_id,
        status=outcome.status.value,
        reason=outcome.reason,
        result=outcome.result,
        pending=outcome.pending,
    )


app = create_app()
