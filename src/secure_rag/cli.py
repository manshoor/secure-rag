"""Command line: ask, agent, approve, eval, ingest report, users.

    uv run rag users
    uv run rag ingest
    uv run rag ask --user sara "How many leave days can I carry over?"
    uv run rag agent --user omar "Raise a purchase of AED 40,000 from Falcon Steel Fabrication LLC for beams"
    uv run rag approve <run id> --approver faisal [--reject]
    uv run rag eval
    uv run rag db setup | reset          (Postgres: create tables, seed users and access)
    uv run rag role revoke sara hr       (Postgres: takes effect on the next question)
"""

import argparse
import sys

from . import db, tracing
from .agent import AgentRunner, Decision, Outcome, Status
from .answer import ask
from .app import build_system
from .config import DOCS_DIR, GOLDEN_FILE
from .evaluate import format_report, run_eval, save_run
from .identity import RoleChange, get_user, list_users, seed_directory, set_role
from .ingest import load_folder

_EXIT_OK = 0
_EXIT_FAIL = 1

# rag db setup keeps live changes (safe to repeat); rag db reset restores the seed.
_DB_ACTIONS = {"setup": db.Overwrite.KEEP, "reset": db.Overwrite.RESTORE}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Secure RAG: cited answers with role-based access")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("users", help="list users, companies and roles")
    commands.add_parser("ingest", help="show indexed and quarantined chunks")

    ask_cmd = commands.add_parser("ask", help="ask a question as a user")
    ask_cmd.add_argument("--user", required=True)
    ask_cmd.add_argument("question")

    agent_cmd = commands.add_parser("agent", help="run a request through the tool agent")
    agent_cmd.add_argument("--user", required=True)
    agent_cmd.add_argument("request")

    approve_cmd = commands.add_parser("approve", help="approve or reject a paused agent run")
    approve_cmd.add_argument("run_id")
    approve_cmd.add_argument("--approver", required=True)
    approve_cmd.add_argument("--reject", action="store_const", const=Decision.REJECT, default=Decision.APPROVE, dest="decision")

    commands.add_parser("eval", help="run the golden evaluation set")

    db_cmd = commands.add_parser("db", help="Postgres: create tables and seed users and document access")
    db_cmd.add_argument("action", choices=sorted(_DB_ACTIONS))

    role_cmd = commands.add_parser("role", help="Postgres: grant or revoke a user's role")
    role_cmd.add_argument("change", type=RoleChange, choices=list(RoleChange), metavar="{grant,revoke}")
    role_cmd.add_argument("user")
    role_cmd.add_argument("role")

    args = parser.parse_args(argv)

    # The process is short-lived: send buffered traces before it exits.
    try:
        return _HANDLERS[args.command](args)
    finally:
        tracing.flush()


def _users(_: argparse.Namespace) -> int:
    for user in list_users():
        print(f"{user.user_id:<8} {', '.join(sorted(user.companies)):<40} {', '.join(sorted(user.roles))}")
    return _EXIT_OK


def _ingest(_: argparse.Namespace) -> int:
    report = build_system().report
    print(f"{len(report.documents)} documents, {len(report.chunks)} chunks indexed")
    for chunk, findings in report.quarantined:
        print(f"QUARANTINED {chunk.chunk_id}: {findings}")
    return _EXIT_OK


def _ask(args: argparse.Namespace) -> int:
    system = build_system()
    answer = ask(system.store, system.composer, get_user(args.user), args.question)
    print(answer.text)
    for citation in answer.citations:
        print(f"  [{citation.source_id}] {citation.title} > {citation.section}  ({citation.doc_id})")
    if answer.refused:
        print(f"  (refused: {answer.reason})")
    print(f"  {answer.seconds * 1000:.0f} ms, {answer.usage.input_tokens + answer.usage.output_tokens} tokens, ${answer.usage.cost_usd:.4f}")
    return _EXIT_OK


def _agent(args: argparse.Namespace) -> int:
    system = build_system()
    runner = AgentRunner(system.store, system.composer, system.planner)
    return _print_outcome(runner.start(get_user(args.user), args.request))


def _approve(args: argparse.Namespace) -> int:
    system = build_system()
    runner = AgentRunner(system.store, system.composer, system.planner)
    return _print_outcome(runner.resume(args.run_id, get_user(args.approver), args.decision))


def _print_outcome(outcome: Outcome) -> int:
    print(f"run {outcome.thread_id}: {outcome.status.value}")
    if outcome.pending:
        print(f"  waiting for approval: {outcome.pending}")
        print(f"  approve with: uv run rag approve {outcome.thread_id} --approver <user>")
    if outcome.reason:
        print(f"  reason: {outcome.reason}")
    if outcome.result:
        print(f"  {outcome.result}")
    return _EXIT_FAIL if outcome.status in (Status.DENIED, Status.REJECTED) else _EXIT_OK


def _eval(_: argparse.Namespace) -> int:
    report = run_eval(build_system(), GOLDEN_FILE)
    print(format_report(report))
    if save_run(report):
        print("saved to Postgres table eval_runs")
    return _EXIT_OK if report.passed == len(report.results) and not report.leaks else _EXIT_FAIL


def _db(args: argparse.Namespace) -> int:
    """setup adds missing users and documents; reset also restores seeded roles and access."""
    if not db.enabled():
        print("DATABASE_URL is not set")
        return _EXIT_FAIL
    documents = {d.doc_id: (d.title, d.access) for d in load_folder(DOCS_DIR).documents}
    seed_directory(_DB_ACTIONS[args.action], documents)
    print(f"{args.action}: {len(list_users())} users, {len(documents)} documents")
    return _EXIT_OK


def _role(args: argparse.Namespace) -> int:
    if not db.enabled():
        print("DATABASE_URL is not set; built-in users cannot change")
        return _EXIT_FAIL
    try:
        set_role(args.user, args.role, args.change)
    except KeyError as error:
        print(error.args[0])
        return _EXIT_FAIL
    user = get_user(args.user)
    print(f"{user.user_id}: {', '.join(sorted(user.roles)) or '(no roles)'}")
    return _EXIT_OK


_HANDLERS = {"users": _users, "ingest": _ingest, "ask": _ask, "agent": _agent, "approve": _approve, "eval": _eval, "db": _db, "role": _role}

if __name__ == "__main__":
    sys.exit(main())
