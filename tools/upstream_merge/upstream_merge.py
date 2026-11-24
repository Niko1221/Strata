#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from merge_gate import EXIT_BLOCKED, EXIT_INVALID, EXIT_OK, EXIT_REVIEW, GateError
from merge_gate.gate import evaluate
from merge_gate.gitops import (
    GIT_PREFIX,
    active_git_state,
    anchor_exists,
    git_stdout,
    git_text,
    is_descendant,
    repo_root,
    resolve_ref,
)
from merge_gate.plan import load_plan
from merge_gate.report import emit
from merge_gate.selftest import selftest as gate_selftest


SCHEMA_PATH = Path(__file__).with_name("merge-plan.schema.json")


def write_git(repo: Path, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_PAGER"] = "cat"
    env.pop("GIT_EXTERNAL_DIFF", None)
    try:
        return subprocess.run(
            [*GIT_PREFIX, "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            shell=False,
            check=False,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"git не выполнен: {exc}") from exc


def cleanup_git(root: Path, commands: list[list[str]]) -> list[str]:
    errors = []
    for command in commands:
        result = write_git(root, command)
        if result.returncode != 0:
            errors.append(f"{' '.join(command)}: {(result.stderr or result.stdout).strip()[-1000:]}")
    return errors


def restore_branch(root: Path, branch: str, fallback_sha: str) -> subprocess.CompletedProcess[str]:
    if branch:
        return write_git(root, ["switch", branch])
    return write_git(root, ["switch", "--detach", fallback_sha])


def require_clean(root: Path) -> tuple[str, list[str]]:
    status = git_stdout(
        root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
    )
    state = active_git_state(root)
    if status:
        raise GateError("рабочее дерево грязное")
    if state:
        raise GateError(f"активна Git-операция: {', '.join(state)}")
    return status, state


def ref_exists(root: Path, ref: str) -> bool:
    result = write_git(root, ["show-ref", "--verify", "--quiet", ref])
    return result.returncode == 0


def preflight(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    if args.fetch:
        result = write_git(root, ["fetch", "upstream"], timeout=300)
        if result.returncode != 0:
            raise GateError((result.stderr or result.stdout).strip()[-2000:])
    status, state = require_clean(root)
    ours = resolve_ref(root, args.ours)
    upstream = resolve_ref(root, args.upstream)
    base = git_text(root, ["merge-base", ours, upstream])
    counts = git_text(root, ["rev-list", "--left-right", "--count", f"{ours}...{upstream}"]).split()
    files = git_stdout(root, ["diff", "--name-only", base, upstream, "--"]).splitlines()
    report = {
        "result": "pass",
        "repo": str(root),
        "ours": {"ref": args.ours, "sha": ours},
        "upstream": {"ref": args.upstream, "sha": upstream},
        "mergeBase": base,
        "aheadBehind": {"oursOnly": int(counts[0]), "upstreamOnly": int(counts[1])},
        "changedFiles": files,
        "dirtyTree": bool(status),
        "activeState": state,
        "fetched": args.fetch,
    }
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Репозиторий: {root}")
        print(f"ours: {args.ours} {ours[:12]}")
        print(f"upstream: {args.upstream} {upstream[:12]}")
        print(f"merge-base: {base[:12]}")
        print(f"ahead/behind: {counts[0]}/{counts[1]}")
        print(f"Файлов upstream-дельты: {len(files)}")
        for path in files[:100]:
            print(f"  {path}")
    return EXIT_OK


def load_profile(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"профиль не читается: {exc}") from exc
    patches = data.get("patches") if isinstance(data, dict) else None
    if not isinstance(patches, list) or not patches:
        raise GateError("профиль должен содержать непустой patches")
    result = []
    ids: set[str] = set()
    for index, item in enumerate(patches):
        if not isinstance(item, dict):
            raise GateError(f"patches[{index}] должен быть объектом")
        patch_id = item.get("id")
        anchor = item.get("oursAnchor")
        if not isinstance(patch_id, str) or not patch_id.strip() or patch_id in ids:
            raise GateError(f"patches[{index}].id недопустим")
        if not isinstance(anchor, str) or not anchor.strip():
            raise GateError(f"patches[{index}].oursAnchor недопустим")
        ids.add(patch_id)
        result.append(item)
    return result


def matrix(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    status, state = require_clean(root)
    ours = resolve_ref(root, args.ours)
    upstream = resolve_ref(root, args.upstream)
    candidate = resolve_ref(root, args.candidate) if args.candidate else None
    rows = []
    for item in load_profile(args.profile):
        ours_found = anchor_exists(root, ours, item["oursAnchor"])
        upstream_anchor = item.get("upstreamAnchor")
        upstream_found = bool(upstream_anchor and anchor_exists(root, upstream, upstream_anchor))
        candidate_found = bool(candidate and anchor_exists(root, candidate, item["oursAnchor"]))
        upstream_status = item.get("upstreamStatus")
        if upstream_status not in {None, "full", "partial", "none"}:
            raise GateError(f"patches[{item['id']}].upstreamStatus недопустим")
        status_evidence = item.get("statusEvidence", [])
        if not isinstance(status_evidence, list) or not all(isinstance(x, str) and x.strip() for x in status_evidence):
            raise GateError(f"patches[{item['id']}].statusEvidence должен быть списком строк")
        status_reviewed = item.get("statusReviewed") is True
        ready = upstream_status is not None and bool(status_evidence) and status_reviewed
        rows.append({
            "id": item["id"],
            "oursAnchor": item["oursAnchor"],
            "upstreamAnchor": upstream_anchor,
            "inOurs": ours_found,
            "inUpstream": upstream_found,
            "inCandidate": candidate_found,
            "upstreamStatus": upstream_status,
            "statusEvidence": status_evidence,
            "statusReviewed": status_reviewed,
            "ready": ready,
            "reworkEffect": item.get("reworkEffect", "unknown"),
            "decision": item.get("decision", "defer"),
            "reason": item.get("reason", "Требуется ручное сравнение алгоритмов и контрактов."),
            "evidence": item.get("evidence", []),
            "notes": item.get("notes", ""),
        })
    ready_for_plan = all(row["ready"] for row in rows)
    report = {
        "result": "draft" if ready_for_plan else "review-required",
        "repo": str(root),
        "ours": args.ours,
        "upstream": args.upstream,
        "candidate": args.candidate,
        "dirtyTree": bool(status),
        "activeState": state,
        "matrix": rows,
        "readyForPlan": ready_for_plan,
        "warning": "Наличие якоря не определяет full/partial/none. readyForPlan подтверждает явный review, но не существование evidence-артефактов.",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    if args.format == "json":
        print(text, end="")
    else:
        print(f"Матрица: {len(rows)} строк")
        for row in rows:
            print(f"{row['id']}: ours={row['inOurs']} upstream={row['inUpstream']} status={row['upstreamStatus']} ready={row['ready']} decision={row['decision']}")
    return EXIT_OK if ready_for_plan else EXIT_REVIEW


def prepare(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    if not args.yes:
        print("Dry-run: будут созданы backup tag, sync-ветка и незавершённый merge. Повторите с --yes.")
        return EXIT_REVIEW
    require_clean(root)
    ours = resolve_ref(root, args.ours)
    upstream = resolve_ref(root, args.upstream)
    branch_ref = f"refs/heads/{args.branch}"
    if ref_exists(root, branch_ref):
        raise GateError(f"ветка уже существует: {args.branch}")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = args.backup_tag or f"pre-upstream-{stamp}"
    backup_ref = f"refs/tags/{backup}"
    if ref_exists(root, backup_ref):
        raise GateError(f"backup tag уже существует: {backup}")
    original_branch = git_text(root, ["branch", "--show-current"])
    original_head = git_text(root, ["rev-parse", "HEAD"])
    switched = write_git(root, ["switch", "-c", args.branch, args.ours])
    if switched.returncode != 0:
        raise GateError((switched.stderr or switched.stdout).strip()[-2000:])
    created = write_git(root, ["tag", backup, ours])
    if created.returncode != 0:
        cleanup_errors = cleanup_git(root, [
            ["switch", original_branch] if original_branch else ["switch", "--detach", original_head],
            ["branch", "-D", args.branch],
        ])
        detail = (created.stderr or created.stdout).strip()[-2000:]
        if cleanup_errors:
            detail += " | cleanup: " + "; ".join(cleanup_errors)
        raise GateError(detail)
    merged = write_git(root, ["merge", "--no-ff", "--no-commit", args.upstream], timeout=300)
    merge_state = active_git_state(root)
    if merged.returncode != 0 and "MERGE_HEAD" not in merge_state:
        cleanup_errors = cleanup_git(root, [
            ["switch", original_branch] if original_branch else ["switch", "--detach", original_head],
            ["branch", "-D", args.branch],
            ["tag", "-d", backup],
        ])
        detail = (merged.stderr or merged.stdout).strip()[-2000:]
        if cleanup_errors:
            detail += " | cleanup: " + "; ".join(cleanup_errors)
        raise GateError(detail)
    report = {
        "result": "merge-in-progress" if merged.returncode == 0 else "merge-conflicts",
        "repo": str(root),
        "branch": args.branch,
        "backupTag": backup,
        "originalBranch": original_branch or None,
        "originalHead": original_head,
        "ours": ours,
        "upstream": upstream,
        "returncode": merged.returncode,
        "stdout": merged.stdout[-10000:],
        "stderr": merged.stderr[-10000:],
        "recovery": {
            "abortArgv": ["python", "tools/upstream_merge/upstream_merge.py", "abort", "--repo", ".", "--yes"],
            "deleteBranchArgv": ["git", "branch", "-d", args.branch],
            "deleteBackupTagArgv": ["git", "tag", "-d", backup],
        },
        "next": "Разрешите конфликты по матрице, затем запустите check и обязательные проверки.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return EXIT_REVIEW if merged.returncode == 0 else EXIT_BLOCKED


def experiment(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    if not args.yes:
        print(f"Dry-run: будет создана ветка {args.branch} от {args.base}. Повторите с --yes.")
        return EXIT_REVIEW
    require_clean(root)
    base = resolve_ref(root, args.base)
    branch_ref = f"refs/heads/{args.branch}"
    if ref_exists(root, branch_ref):
        raise GateError(f"ветка уже существует: {args.branch}")
    result = write_git(root, ["switch", "-c", args.branch, args.base])
    if result.returncode != 0:
        raise GateError((result.stderr or result.stdout).strip()[-2000:])
    print(json.dumps({
        "result": "experiment-created",
        "repo": str(root),
        "branch": args.branch,
        "base": args.base,
        "baseSha": base,
    }, ensure_ascii=False, indent=2))
    return EXIT_OK


def write_promote_report(args: argparse.Namespace, report: dict[str, Any], must_write: bool = False, print_output: bool = True) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report_out:
        try:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(text, encoding="utf-8")
        except OSError as exc:
            if must_write:
                raise GateError(f"report-out не записывается: {exc}") from exc
            print(f"report-out не записывается: {exc}", file=sys.stderr)
    if print_output:
        print(text, end="")


def promote(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    if not args.yes:
        print("Dry-run: promote не меняет main, refs, remote и локальные ветки. Повторите с --yes.")
        return EXIT_REVIEW
    if not args.report_out:
        raise GateError("--report-out обязателен для фактического promote")
    if args.delete_local and not args.push:
        raise GateError("--delete-local требует --push")
    if args.pr and not args.push:
        raise GateError("--pr требует --push")
    if args.pr and not shutil.which("gh"):
        raise GateError("gh не найден; PR не будет создан")
    if not re.match(r"^\[(Engine|GUI|TUI|Benchmark|Docs|Other)\]", args.title):
        raise GateError("PR title должен начинаться с [Engine], [GUI], [TUI], [Benchmark], [Docs] или [Other]")
    body_path = (root / args.body_file).resolve()
    try:
        body_path.relative_to(root)
    except ValueError as exc:
        raise GateError("--body-file должен оставаться внутри репозитория") from exc
    if args.pr and not body_path.is_file():
        raise GateError("--body-file не найден")
    require_clean(root)
    current = git_text(root, ["branch", "--show-current"])
    if current != args.branch:
        raise GateError(f"сначала перейдите на ветку {args.branch}")
    plan = load_plan(args.plan, SCHEMA_PATH)
    gate_report, gate_code = evaluate(root, plan, run_checks=True)
    if gate_code != EXIT_OK:
        report = {"result": "blocked", "repo": str(root), "branch": args.branch, "gate": gate_report}
        write_promote_report(args, report, must_write=True)
        return EXIT_BLOCKED
    branch_sha = resolve_ref(root, args.branch)
    main_sha = resolve_ref(root, args.main)
    if branch_sha == main_sha or not is_descendant(root, args.main, args.branch):
        report = {
            "result": "blocked",
            "repo": str(root),
            "branch": args.branch,
            "error": "ветка эксперимента не содержит новых коммитов поверх main",
            "gate": gate_report,
        }
        write_promote_report(args, report, must_write=True)
        return EXIT_BLOCKED
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"pre-promote-{stamp}"
    if ref_exists(root, f"refs/tags/{backup}"):
        report = {
            "result": "blocked",
            "repo": str(root),
            "branch": args.branch,
            "error": f"backup tag уже существует: {backup}",
            "gate": gate_report,
        }
        write_promote_report(args, report, must_write=True)
        return EXIT_BLOCKED
    report = {
        "result": "promote-started",
        "repo": str(root),
        "branch": args.branch,
        "main": args.main,
        "backupTag": backup,
        "branchSha": branch_sha,
        "previousMainSha": main_sha,
        "gate": gate_report,
        "pushed": False,
        "prCreated": False,
        "localBranchDeleted": False,
        "recovery": {
            "manualRecoveryRequired": True,
            "previousMainSha": main_sha,
            "backupTag": backup,
            "instruction": "При необходимости восстановите main вручную из previousMainSha или backup tag; автоматический reset --hard не выполняется.",
        },
    }
    write_promote_report(args, report, must_write=True, print_output=False)
    tagged = write_git(root, ["tag", backup, args.main])
    if tagged.returncode != 0:
        report["result"] = "tag-failed"
        report["error"] = (tagged.stderr or tagged.stdout).strip()[-4000:]
        write_promote_report(args, report, must_write=True)
        return EXIT_BLOCKED
    switched = write_git(root, ["switch", args.main])
    if switched.returncode != 0:
        cleanup_errors = cleanup_git(root, [["tag", "-d", backup]])
        report["result"] = "switch-failed"
        report["error"] = (switched.stderr or switched.stdout).strip()[-4000:]
        if cleanup_errors:
            report["cleanupErrors"] = cleanup_errors
        write_promote_report(args, report, must_write=True)
        return EXIT_BLOCKED
    merged = write_git(root, ["merge", "--ff-only", args.branch])
    if merged.returncode != 0:
        cleanup_errors = cleanup_git(root, [
            ["switch", args.branch],
            ["tag", "-d", backup],
        ])
        report["result"] = "merge-failed"
        report["error"] = (merged.stderr or merged.stdout).strip()[-4000:]
        if cleanup_errors:
            report["cleanupErrors"] = cleanup_errors
        write_promote_report(args, report, must_write=True)
        return EXIT_BLOCKED
    report["result"] = "promoted"
    if args.push:
        pushed = write_git(root, ["push", "--atomic", args.remote, args.main, args.branch], timeout=300)
        if pushed.returncode != 0:
            report["result"] = "promoted-not-pushed"
            report["pushError"] = (pushed.stderr or pushed.stdout).strip()[-4000:]
            write_promote_report(args, report, must_write=True)
            return EXIT_BLOCKED
        report["pushed"] = True
    if args.pr:
        created = subprocess.run(
            ["gh", "pr", "create", "--base", args.main, "--head", args.branch,
             "--title", args.title, "--body-file", args.body_file],
            cwd=str(root), capture_output=True, text=True, encoding="utf-8",
            errors="replace", shell=False, check=False, timeout=120,
        )
        if created.returncode != 0:
            report["result"] = "promoted-not-pr"
            report["prError"] = (created.stderr or created.stdout).strip()[-4000:]
            write_promote_report(args, report, must_write=True)
            return EXIT_BLOCKED
        report["prCreated"] = True
        report["prUrl"] = created.stdout.strip()
    if args.delete_local:
        deleted = write_git(root, ["branch", "-d", args.branch])
        if deleted.returncode != 0:
            report["result"] = "promoted-not-cleaned"
            report["deleteError"] = (deleted.stderr or deleted.stdout).strip()[-4000:]
            write_promote_report(args, report, must_write=True)
            return EXIT_BLOCKED
        report["localBranchDeleted"] = True
    write_promote_report(args, report, must_write=True)
    return EXIT_OK


def status(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    report = {
        "repo": str(root),
        "branch": git_text(root, ["branch", "--show-current"]),
        "head": git_text(root, ["rev-parse", "HEAD"]),
        "activeState": active_git_state(root),
        "status": git_stdout(root, ["status", "--short", "--untracked-files=all"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return EXIT_OK


def abort(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    if not args.yes:
        print("Dry-run: будет выполнен git merge --abort. Повторите с --yes.")
        return EXIT_REVIEW
    if "MERGE_HEAD" not in active_git_state(root):
        raise GateError("активного merge нет")
    result = write_git(root, ["merge", "--abort"])
    if result.returncode != 0:
        raise GateError((result.stderr or result.stdout).strip()[-2000:])
    print("Merge отменён.")
    return EXIT_OK


def check(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan, SCHEMA_PATH)
    report, code = evaluate(args.repo, plan, run_checks=args.run_checks)
    emit(report, args.format)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Безопасный upstream merge: preflight, matrix, prepare, check, status, abort")
    sub = parser.add_subparsers(dest="mode", required=True)

    exp = sub.add_parser("experiment", help="создать отдельную ветку эксперимента")
    exp.add_argument("--repo", required=True, type=Path)
    exp.add_argument("--branch", required=True)
    exp.add_argument("--base", default="main")
    exp.add_argument("--yes", action="store_true")
    exp.set_defaults(func=experiment)

    pro = sub.add_parser("promote", help="перенести успешную ветку в main после зелёного gate")
    pro.add_argument("--repo", required=True, type=Path)
    pro.add_argument("--branch", required=True)
    pro.add_argument("--main", default="main")
    pro.add_argument("--plan", required=True, type=Path)
    pro.add_argument("--report-out", type=Path)
    pro.add_argument("--remote", default="origin")
    pro.add_argument("--push", action="store_true")
    pro.add_argument("--pr", action="store_true")
    pro.add_argument("--title", default="[Other] Strata optimization")
    pro.add_argument("--body-file", default="docs/UPSTREAM_MERGE.md")
    pro.add_argument("--delete-local", action="store_true")
    pro.add_argument("--yes", action="store_true")
    pro.set_defaults(func=promote)

    pre = sub.add_parser("preflight", help="read-only проверка refs и дельты")
    pre.add_argument("--repo", required=True, type=Path)
    pre.add_argument("--ours", default="main")
    pre.add_argument("--upstream", default="upstream/main")
    pre.add_argument("--fetch", action="store_true")
    pre.add_argument("--format", choices=("human", "json"), default="human")
    pre.set_defaults(func=preflight)

    mat = sub.add_parser("matrix", help="собрать черновую матрицу наших патчей")
    mat.add_argument("--repo", required=True, type=Path)
    mat.add_argument("--profile", required=True, type=Path)
    mat.add_argument("--ours", default="main")
    mat.add_argument("--upstream", default="upstream/main")
    mat.add_argument("--candidate")
    mat.add_argument("--out", type=Path)
    mat.add_argument("--format", choices=("human", "json"), default="human")
    mat.set_defaults(func=matrix)

    prep = sub.add_parser("prepare", help="создать backup tag, sync-ветку и начать merge --no-commit")
    prep.add_argument("--repo", required=True, type=Path)
    prep.add_argument("--ours", default="main")
    prep.add_argument("--upstream", default="upstream/main")
    prep.add_argument("--branch", required=True)
    prep.add_argument("--backup-tag")
    prep.add_argument("--yes", action="store_true")
    prep.set_defaults(func=prepare)

    chk = sub.add_parser("check", help="проверить готовый merge-plan.json")
    chk.add_argument("--repo", required=True, type=Path)
    chk.add_argument("--plan", required=True, type=Path)
    chk.add_argument("--run-checks", action="store_true")
    chk.add_argument("--format", choices=("human", "json", "both"), default="human")
    chk.set_defaults(func=check)

    st = sub.add_parser("status", help="показать ветку, HEAD и активную Git-операцию")
    st.add_argument("--repo", required=True, type=Path)
    st.set_defaults(func=status)

    ab = sub.add_parser("abort", help="отменить активный merge")
    ab.add_argument("--repo", required=True, type=Path)
    ab.add_argument("--yes", action="store_true")
    ab.set_defaults(func=abort)

    sub.add_parser("selftest", help="прогнать встроенные регрессии гейта")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "selftest":
        try:
            return gate_selftest()
        except Exception as exc:
            print(f"Selftest: отказ: {exc}", file=sys.stderr)
            return EXIT_BLOCKED
    try:
        return args.func(args)
    except (GateError, OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        report = {"result": "invalid", "exitCode": EXIT_INVALID, "error": str(exc)}
        exit_code = EXIT_INVALID
        if args.mode == "promote" and getattr(args, "report_out", None):
            try:
                args.report_out.parent.mkdir(parents=True, exist_ok=True)
                args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError as report_exc:
                report["reportWriteError"] = str(report_exc)
                report["exitCode"] = EXIT_BLOCKED
                exit_code = EXIT_BLOCKED
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
