"""Run one integration command with the established owned-Job resource floors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = Path("C:/Users/downl/Documents/ChatGPT/Strata")
BASE = Path("H:/Strata/evidence/upstream-sync-20261116")


def main() -> int:
    if sys.argv[1:2] == ["--child"]:
        os.chdir(ROOT)
        log = Path(sys.argv[2])
        if log.parent.resolve() != (BASE / "receipts").resolve():
            raise ValueError("Contained command log required")
        with log.open("xb") as stream:
            proc = subprocess.Popen(
                sys.argv[3:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            written = 0
            while block := proc.stdout.read1(64 * 1024):
                written += len(block)
                if written > 8 * 1024**2:
                    proc.terminate()
                    proc.wait(timeout=10)
                    raise RuntimeError("Command output limit exceeded")
                stream.write(block)
                stream.flush()
            return proc.wait(timeout=10)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--allow-low-ram", action="store_true",
                        help="Explicit user-authorized operation-only RAM exception; keep commit/disk/Job guards")
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if not command:
        parser.error("explicit command argument vector required")
    for folder in (BASE, BASE / "receipts", BASE / "temp", BASE / "cache"):
        folder.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "integration_guard", PRIMARY / "_docs/local-053-20261109/native/common.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.BASE = BASE
    guard.NODE = ROOT / "apps/desktop/src-tauri/resources/node/node.exe"
    with guard.NODE.open('rb') as node_stream:
        node_sha = hashlib.file_digest(node_stream, 'sha256').hexdigest()
    if node_sha != '86fc4785327db5f5c7558dc6156c543b433c313e813a222e088086222ab4afa6':
        raise ValueError('Verified Node 24.22.3 runtime required')
    env = guard.environment()
    # The production Windows shell is a trusted OS directory, not an inherited
    # user PATH. Several real shell fixtures need it under the filtered guard.
    powershell = Path(os.environ["SYSTEMROOT"]) / "System32/WindowsPowerShell/v1.0"
    if not (powershell / "powershell.exe").is_file():
        raise ValueError("Required Windows PowerShell executable unavailable")
    env["PATH"] = str(powershell) + ";" + env["PATH"]
    # The source-binding regression invokes PowerShell 7 explicitly. Use this
    # app runtime's verified executable, retaining the filtered PATH policy.
    pwsh = Path("C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-611-0_26.1112.611.0_x64__3k8sg7r9htsxt/dependencies/native/powershell")
    if not (pwsh / "pwsh.exe").is_file():
        raise ValueError("Required trusted PowerShell 7 executable unavailable")
    env["PATH"] = str(pwsh) + ";" + env["PATH"]
    env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
    env.update(
        PYTHONPATH=str(ROOT),
        CMAKE_BUILD_PARALLEL_LEVEL="5",
        CARGO_BUILD_JOBS="5",
        PYTHONUTF8="1",
        STRATA_TEST_CONCURRENCY="1",
    )
    if command[0] == "python":
        command[0] = sys.executable
    elif command[0] == "node":
        command[0] = str(guard.NODE)
    result = guard.owned(
        PRIMARY,
        args.name,
        [sys.executable, str(Path(__file__)), "--child",
         str(BASE / "receipts" / (args.name + ".command.txt")), *command],
        args.seconds,
        env=env,
        allow_low_ram=args.allow_low_ram,
    )
    log = BASE / "receipts" / (args.name + ".command.txt")
    if log.exists():
        with log.open("rb") as stream:
            result["command_log"] = {
                "path": str(log),
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
            }
        guard.save(BASE / "receipts" / (args.name + ".json"), result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
