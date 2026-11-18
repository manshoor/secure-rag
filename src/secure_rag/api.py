"""HTTP API over the same pipeline the CLI uses.

    POST /ask                          question -> cited answer or refusal
    POST /agent/runs                   request  -> run (done, denied, or awaiting approval)
    POST /agent/runs/{run_id}/decision approve | reject, as the calling user
    GET  /me                           who the API thinks you are
    GET  /health

Identity comes from the X-User header, resolved in current_user().
That is the one place to swap in real sign-in (an Entra ID token whose
groups map to companies and roles). Nothing in a request body can change
who the user is or what they may read.

    uv run uvicorn secure_rag.api:app --port 8010
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import config, db, tracing
from .agent import AgentRunner, Decision, Outcome
from .answer import Answer, ask
from .app import System, build_system
from .identity import User, get_user

_USER_HEADER = "X-User"
_HTTP_UNAUTHORIZED = 401
_HTTP_NOT_FOUND = 404
_HTTP_CONFLICT = 409
_MAX_QUESTION_CHARS = 2000


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


class MeOut(BaseModel):
    user_id: str
    name: str
    companies: list[str]
    roles: list[str]


# ---- app --------------------------------------------------------------------


def create_app(system: System | None = None, runner: AgentRunner | None = None) -> FastAPI:
    """Build the app. Tests pass an offline system; production builds from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ingest once per process; every request shares the index and the runner.
        app.state.system = system or build_system()
        app.state.runner = runner or AgentRunner(
            app.state.system.store, app.state.system.composer, app.state.system.planner
        )
        yield
        tracing.flush()

    app = FastAPI(title="Secure RAG", lifespan=lifespan)
    app.add_api_route("/health", _health, methods=["GET"])
    app.add_api_route("/me", _me, methods=["GET"], response_model=MeOut)
    app.add_api_route("/ask", _ask, methods=["POST"], response_model=AnswerOut)
    app.add_api_route("/agent/runs", _start_run, methods=["POST"], response_model=RunOut)
    app.add_api_route("/agent/runs/{run_id}/decision", _decide, methods=["POST"], response_model=RunOut)
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
