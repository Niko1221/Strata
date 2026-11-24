from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import EXIT_BLOCKED, EXIT_INVALID, GateError
from .gate import evaluate
from .plan import load_plan
from .report import emit
from .selftest import selftest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only гейт мержа ours/upstream/hybrid/delta")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    check = subparsers.add_parser("check", help="проверить кандидата по merge-plan.json")
    check.add_argument("--repo", required=True, type=Path)
    check.add_argument("--plan", required=True, type=Path)
    check.add_argument("--run-checks", action="store_true", help="запустить доверенные команды из плана")
    check.add_argument("--format", choices=("human", "json", "both"), default="human")
    subparsers.add_parser("selftest", help="прогнать selftest на временном git-fixture")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "selftest":
        try:
            return selftest()
        except Exception as exc:
            print(f"Selftest: отказ: {exc}", file=sys.stderr)
            return EXIT_BLOCKED

    try:
        plan = load_plan(args.plan)
        report, code = evaluate(args.repo, plan, run_checks=args.run_checks)
        emit(report, args.format)
        return code
    except (GateError, OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        report = {"result": "invalid", "exitCode": EXIT_INVALID, "error": str(exc)}
        emit(report, args.format)
        return EXIT_INVALID
