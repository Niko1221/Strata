from __future__ import annotations

import os
import subprocess
from typing import Any

from . import GateError
from .gitops import git_stdout, git_text, repo_root


MAX_OUTPUT = 65536


def run_plan_checks(
    repo: Path,
    plan: dict[str, Any],
    expected_head: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    results: list[dict[str, Any]] = []
    blockers: list[str] = []
    not_run: list[str] = []
    root = repo_root(repo)
    for index, check in enumerate(plan.get("checks", [])):
        cwd = (root / check.get("cwd", ".")).resolve()
        try:
            cwd.relative_to(root)
        except ValueError as exc:
            raise GateError(f"checks.{check['name']}.cwd выходит за репозиторий") from exc
        if git_text(repo, ["rev-parse", "HEAD"]) != expected_head:
            blockers.append(f"check-state-changed:{check['name']}")
            not_run.extend(item["name"] for item in plan["checks"][index:])
            break
        command = check["command"]
        timeout_ms = check.get("timeoutMs", 120000)
        env = os.environ.copy()
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.pop("GIT_EXTERNAL_DIFF", None)
        env.pop("GIT_DIFF_OPTS", None)
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_ms / 1000,
                env=env,
                shell=False,
                check=False,
            )
            stdout = result.stdout[-MAX_OUTPUT:]
            stderr = result.stderr[-MAX_OUTPUT:]
            ok = result.returncode == 0
            results.append({
                "name": check["name"],
                "command": command,
                "cwd": check.get("cwd", "."),
                "timeoutMs": timeout_ms,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "ok": ok,
            })
            if not ok:
                blockers.append(f"check-failed:{check['name']}")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            results.append({
                "name": check["name"],
                "command": command,
                "cwd": check.get("cwd", "."),
                "timeoutMs": timeout_ms,
                "ok": False,
                "error": str(exc)[-2000:],
            })
            blockers.append(f"check-failed:{check['name']}")
        try:
            head_after = git_text(repo, ["rev-parse", "HEAD"])
            status_after = git_stdout(
                repo,
                ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
            )
        except GateError as exc:
            blockers.append(f"check-state-unreadable:{check['name']}")
            results[-1]["stateError"] = str(exc)[-2000:]
            not_run.extend(item["name"] for item in plan["checks"][index + 1:])
            break
        results[-1]["headBefore"] = expected_head
        results[-1]["headAfter"] = head_after
        if head_after != expected_head or status_after:
            blockers.append(f"check-state-changed:{check['name']}")
            not_run.extend(item["name"] for item in plan["checks"][index + 1:])
            break
    return results, blockers, not_run
