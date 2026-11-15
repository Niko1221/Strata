"""Run one explicit PR00 verification command with the existing owned Job guard."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("H:/Strata/evidence/ram32gb-pr00-20261115")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an explicit verification argument vector is required")
    spec = importlib.util.spec_from_file_location(
        "pr00_guard", ROOT / "_docs/local-053-20261109/native/common.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    OUT.mkdir(exist_ok=True)
    for name in ("receipts", "temp", "cache"):
        (OUT / name).mkdir(exist_ok=True)
    guard.BASE = OUT
    guard.NODE = ROOT / "apps/desktop/src-tauri/resources/node/node.exe"
    env = guard.environment()
    env.update(
        CMAKE_BUILD_PARALLEL_LEVEL="5",
        CARGO_BUILD_JOBS="5",
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        STRATA_TEST_CONCURRENCY="1",
        PYTHONPATH=str(ROOT),
    )
    compiler_receipt = OUT / "cpu/compiler-environment.json"
    if compiler_receipt.is_file():
        env["Q36_CPU_COMPILE_RECEIPT"] = str(compiler_receipt)
    if command[0] == "python":
        command[0] = sys.executable
    elif command[0] == "node":
        command[0] = str(guard.NODE)
    result = guard.owned(ROOT, args.name, command, args.seconds, env=env)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
