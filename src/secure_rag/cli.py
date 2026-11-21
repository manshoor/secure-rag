"""Command line: ask, agent, approve, eval, ingest report, users.

    uv run rag users
    uv run rag ingest
    uv run rag ask --user sara "How many leave days can I carry over?"
    uv run rag agent --user omar "Raise a purchase of AED 40,000 from Falcon Steel Fabrication LLC for beams"
    uv run rag approve <run id> --approver faisal [--reject]
    uv run rag eval
    uv run rag db setup | reset          (Postgres: create tables, seed users and access)
    uv run rag role revoke sara hr       (Postgres: takes effect on the next question)
    uv run rag workflow boq --user khalid --mode agent data/samples/boq_with_issues.csv
    uv run rag workflow deliver <run id> --user omar data/samples/quotes/round1/*.csv | --deadline
    uv run rag workflow approve <run id> --approver omar [--reject | --extend | --drop]
    RAG_DEMO=on uv run rag demo reset [--yes]
"""

import argparse
import json
import sys
from pathlib import Path

from . import db, demo, tracing
from .agent import AgentRunner, Decision, Outcome, Status
from .answer import ask
from .app import build_system, build_workflows
from .boq_parser import BoqParseError, ParserKind, parser_kind
from .config import DOCS_DIR, GOLDEN_FILE
from .evaluate import format_report, run_eval, save_run
from .identity import RoleChange, get_user, list_users, seed_directory, set_role
from .ingest import load_folder
from .workflow import Event, RunMode

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
    commands.add_parser("workflow-eval", help="run the workflow cases offline")

    workflow_cmd = commands.add_parser("workflow", help="cross-department workflow: HR -> procurement -> finance")
    flows = workflow_cmd.add_subparsers(dest="flow", required=True)
    flow_start = flows.add_parser("start", help="start from a request sentence")
    flow_start.add_argument("--user", required=True)
    flow_start.add_argument("request")
    flow_boq = flows.add_parser("boq", help="start from a BOQ file (xlsx, csv, pdf)")
    flow_boq.add_argument("--user", required=True)
    flow_boq.add_argument("file", type=Path)
    flow_boq.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.CLASSIC.value, help="agent: quote requests and checks")
    flow_approve = flows.add_parser("approve", help="decide the waiting gate (approve by default)")
    flow_approve.add_argument("run_id")
    flow_approve.add_argument("--approver", required=True)
    choice = flow_approve.add_mutually_exclusive_group()
    choice.add_argument("--reject", action="store_const", const=Decision.REJECT, default=Decision.APPROVE, dest="decision")
    choice.add_argument("--extend", action="store_const", const=Decision.EXTEND, dest="decision", help="quote gate: ask once more")
    choice.add_argument("--drop", action="store_const", const=Decision.DROP, dest="decision", help="quote gate: take the short lines out")
    flow_deliver = flows.add_parser("deliver", help="feed vendor replies (file name = sender vendor id) or a passed deadline")
    flow_deliver.add_argument("run_id")
    flow_deliver.add_argument("--user", required=True, help="who delivers: someone who may act for procurement")
    flow_deliver.add_argument("quotes", nargs="*", type=Path)
    flow_deliver.add_argument("--deadline", action="store_true", help="the reply deadline passed")
    flow_resume = flows.add_parser("resume", help="finish a run that stopped between steps (crash, failed write)")
    flow_resume.add_argument("run_id")
    flow_resume.add_argument("--user", help="run as this user (default: the requester)")
    flow_show = flows.add_parser("show", help="timeline of a run")
    flow_show.add_argument("run_id")
    flow_show.add_argument("--user", help="view as this user (default: the requester)")

    check_cmd = commands.add_parser("boq-check", help="parse a BOQ and report counts (and accuracy with --expected)")
    check_cmd.add_argument("file", type=Path)
    check_cmd.add_argument("--expected", type=Path)
    check_cmd.add_argument("--parser", choices=[kind.value for kind in ParserKind], help="default: RAG_BOQ_PARSER or tabular")

    db_cmd = commands.add_parser("db", help="Postgres: create tables and seed users and document access")
    db_cmd.add_argument("action", choices=sorted(_DB_ACTIONS))

    role_cmd = commands.add_parser("role", help="Postgres: grant or revoke a user's role")
    role_cmd.add_argument("change", type=RoleChange, choices=list(RoleChange), metavar="{grant,revoke}")
    role_cmd.add_argument("user")
    role_cmd.add_argument("role")

    demo_cmd = commands.add_parser("demo", help="demo helpers (RAG_DEMO=on)")
    demo_actions = demo_cmd.add_subparsers(dest="action", required=True)
    demo_reset = demo_actions.add_parser("reset", help="delete demo runs, holds, outbox and checkpoints (lists first)")
    demo_reset.add_argument("--yes", action="store_true", help="delete; without it, only list")

    args = parser.parse_args(argv)
    if args.command == "workflow" and args.flow == "deliver" and bool(args.quotes) == args.deadline:
        parser.error("workflow deliver needs quote files or --deadline, not both")

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


def _workflow(args: argparse.Namespace) -> int:
    from .workflow import WfStatus

    runner = build_workflows(build_system())
    try:
        if args.flow == "start":
            view = runner.start_request(get_user(args.user), args.request)
        elif args.flow == "boq":
            view = runner.start_boq(get_user(args.user), args.file, mode=RunMode(args.mode))
        elif args.flow == "deliver":
            # A mistyped path would count as an unreadable reply and use up the round.
            missing = [str(path) for path in args.quotes if not path.is_file()]
            if missing:
                print(f"error: no such file: {', '.join(missing)}")
                return _EXIT_FAIL
            event = Event.DEADLINE_PASSED if args.deadline else Event.QUOTES_DELIVERED
            quotes = [(path.stem, path) for path in args.quotes] or None
            view = runner.deliver(args.run_id, event, get_user(args.user), quotes)
        elif args.flow == "approve":
            view = runner.decide(args.run_id, get_user(args.approver), args.decision)
        elif args.flow == "resume":
            view = runner.resume(args.run_id, get_user(args.user or runner.requester_of(args.run_id) or "sara"))
        else:
            viewer = args.user or runner.requester_of(args.run_id)
            view = runner.view(args.run_id, get_user(viewer)) if viewer else runner.view(args.run_id, get_user("sara"))
    except (PermissionError, LookupError, ValueError, BoqParseError) as error:
        print(f"error: {error}")
        return _EXIT_FAIL
    _print_run(view)
    return _EXIT_FAIL if view.status in (WfStatus.STOPPED, WfStatus.REJECTED) else _EXIT_OK


def _demo(args: argparse.Namespace) -> int:
    """List the demo state, and delete it with --yes."""
    try:
        targets = demo.reset() if args.yes else demo.reset_targets()
    except demo.ResetRefused as refused:
        print(f"error: {refused}")
        return _EXIT_FAIL
    except OSError as error:
        print(f"error: reset stopped part way: {error}")
        return _EXIT_FAIL
    if not targets:
        print("nothing to reset")
        return _EXIT_OK
    verb = "deleted" if args.yes else "would delete"
    for path in targets:
        print(f"  {verb} {path}")
    if not args.yes:
        print("run again with --yes to delete (stop the server first)")
    return _EXIT_OK


def _print_run(view) -> None:
    """Timeline, money and queues. Counts only for runs on private files."""
    print(f"run {view.run_id}  [{view.entry.value}]  status: {view.status.value}")
    for step in view.steps:
        print(f"  {step['status']:<8} {step['name']:<18} {_short(step['detail'], view.private)}")
    if view.reason:
        print(f"  reason: {view.reason}")
    print(f"  total AED {view.total:,.2f}   budget left AED {view.budget_remaining or 0:,.2f}   approver: {' or '.join(view.approver_roles) or '-'}")
    if view.private:
        print(f"  private file: {len(view.lines)} priced, {len(view.rfq)} RFQ, {len(view.review)} in review (text not shown)")
    else:
        for line in view.lines:
            label = line.get("item_no", line["code"])
            print(f"    {label:<6} {line['code']:<14} {line['quantity']:>8} {line['unit']:<4} x {line.get('unit_price', '-'):>9} = {line.get('total', '-'):>11}  {line['vendor']}")
        for item in view.rfq:
            print(f"    RFQ    {item['item_no']:<6} {item['description'][:50]:<50}  {item['reason']}")
        for item in view.review:
            print(f"    REVIEW {item['item_no']:<6} {item['reason']}")
    for flag in view.flags if not view.private else []:
        print(f"  flag: {flag}")
    if view.refused_vendors:
        print(f"  refused vendors: {', '.join(view.refused_vendors)}")
    if view.policy:
        policy = view.policy
        print(f"  policy: {policy['title']} > {policy['section']}" if policy.get("visible") else "  policy: cited, not visible to you")
    for gate, note in view.summaries.items():
        print(f"  note for {gate}: {note.replace(chr(10), ' / ')}")
    if view.pending_gate:
        print(f"  waiting for {view.pending_gate.value}: uv run rag workflow approve {view.run_id} --approver <user>")
    print(f"  model cost ${view.cost_usd:.4f}, {view.tokens} tokens")


def _short(detail: dict, private: bool) -> str:
    keep = {key: value for key, value in detail.items() if not (private and isinstance(value, str) and not key.endswith("_aed"))}
    return ", ".join(f"{key}={value}" for key, value in keep.items())[:110]


def _boq_check(args: argparse.Namespace) -> int:
    from .boq import check_report

    kind = ParserKind(args.parser) if args.parser else parser_kind()
    try:
        report = check_report(args.file, kind, args.expected)
    except BoqParseError as error:
        print(f"error: {error}")
        return _EXIT_FAIL
    print(json.dumps(report, indent=2))
    return _EXIT_OK


def _workflow_eval(_: argparse.Namespace) -> int:
    from .workflow_eval import format_workflow_report, run_workflow_eval

    results = run_workflow_eval(build_system())
    print(format_workflow_report(results))
    return _EXIT_OK if all(result.passed for result in results) else _EXIT_FAIL


_HANDLERS = {"users": _users, "ingest": _ingest, "ask": _ask, "agent": _agent, "approve": _approve, "eval": _eval, "db": _db, "role": _role,
             "workflow": _workflow, "boq-check": _boq_check, "workflow-eval": _workflow_eval, "demo": _demo}

if __name__ == "__main__":
    sys.exit(main())
