"""Dependency-light ``agentctl`` command line client."""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from .sdk import AgentRuntimeClient


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("value must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentctl")
    parser.add_argument("--base-url", default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start a run on an existing thread")
    run.add_argument("--tenant-id", required=True)
    run.add_argument("--thread-id", required=True)
    run.add_argument("--task", required=True)

    resume = commands.add_parser("resume", help="resume a paused HITL run")
    resume.add_argument("--tenant-id", required=True)
    resume.add_argument("--thread-id", required=True)
    resume.add_argument("--decision", choices=("approve", "edit", "reject"), required=True)
    resume.add_argument("--edited-arguments", type=_json_object)
    resume.add_argument("--feedback")

    threads = commands.add_parser("threads")
    thread_commands = threads.add_subparsers(dest="threads_command", required=True)
    inspect = thread_commands.add_parser("inspect")
    inspect.add_argument("--tenant-id", required=True)
    inspect.add_argument("--thread-id", required=True)

    tools = commands.add_parser("tools")
    tool_commands = tools.add_subparsers(dest="tools_command", required=True)
    list_command = tool_commands.add_parser("list")
    list_command.add_argument("--tenant-id", required=True)
    list_command.add_argument(
        "--role", choices=("viewer", "analyst", "operator", "admin"), required=True
    )

    evaluation = commands.add_parser("eval")
    evaluation_commands = evaluation.add_subparsers(dest="eval_command", required=True)
    evaluation_run = evaluation_commands.add_parser("run")
    evaluation_run.add_argument("--run-config-identity", required=True)
    evaluation_run.add_argument("--fresh-isolated-database-confirmed", action="store_true")
    evaluation_run.add_argument("--cases")
    evaluation_run.add_argument("--output")
    evaluation_run.add_argument("--timeout-seconds", type=float)
    evaluation_run.add_argument("--overwrite", action="store_true")
    return parser


def _evaluation_args(args: argparse.Namespace) -> list[str]:
    output = [
        "--api-base-url",
        args.base_url or "http://127.0.0.1:8010",
        "--run-config-identity",
        args.run_config_identity,
    ]
    if args.fresh_isolated_database_confirmed:
        output.append("--fresh-isolated-database-confirmed")
    for flag, value in (
        ("--cases", args.cases),
        ("--output", args.output),
        ("--timeout-seconds", args.timeout_seconds),
    ):
        if value is not None:
            output.extend([flag, str(value)])
    if args.overwrite:
        output.append("--overwrite")
    return output


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[..., AgentRuntimeClient] = AgentRuntimeClient,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "eval":
        from .evaluation.run_evaluation import main as evaluation_main

        evaluation_main(_evaluation_args(args))
        return 0

    with client_factory(base_url=args.base_url) as client:
        if args.command == "run":
            result: Any = client.run(
                tenant_id=args.tenant_id,
                thread_id=args.thread_id,
                task=args.task,
            )
        elif args.command == "resume":
            result = client.resume(
                tenant_id=args.tenant_id,
                thread_id=args.thread_id,
                decision=args.decision,
                edited_arguments=args.edited_arguments,
                feedback=args.feedback,
            )
        elif args.command == "threads" and args.threads_command == "inspect":
            result = client.inspect_thread(
                tenant_id=args.tenant_id, thread_id=args.thread_id
            )
        elif args.command == "tools" and args.tools_command == "list":
            result = client.list_tools(tenant_id=args.tenant_id, role=args.role)
        else:  # pragma: no cover - argparse rejects incomplete command trees.
            raise RuntimeError("unsupported agentctl command")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
