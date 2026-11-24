from __future__ import annotations

from pathlib import Path
from typing import Any

from . import EXIT_BLOCKED, EXIT_OK, EXIT_REVIEW
from .checks import run_plan_checks
from .evidence import collect_evidence, collect_review
from .gitops import collect_git_facts, repo_root, reviewer_is_independent, snapshot_git_state
from .plan import is_legacy_plan


def evaluate(repo: Path, plan: dict[str, Any], run_checks: bool = False) -> tuple[dict[str, Any], int]:
    root = repo_root(repo)
    facts, status, _scan, git_blockers, git_warnings = collect_git_facts(root, plan)
    blockers = list(git_blockers)
    warnings = list(git_warnings)
    review, review_blockers, review_warnings, authors = collect_review(root, plan, facts["candidate"])
    blockers.extend(review_blockers)
    warnings.extend(review_warnings)
    if plan["decision"] == "defer" or any(item["decision"] == "defer" for item in plan["matrix"]):
        blockers.append("defer-not-ready")

    check_results: list[dict[str, Any]] = []
    not_run: list[str] = []
    not_run_reasons: dict[str, str] = {}
    check_state: dict[str, Any] = {"requested": run_checks}
    post_facts: dict[str, Any] | None = None
    checks = plan.get("checks", [])
    if run_checks and checks:
        check_state["before"] = snapshot_git_state(root, plan)
        if git_blockers:
            not_run = [check["name"] for check in checks]
            not_run_reasons = {name: "preflight-blocked" for name in not_run}
            blockers.append("checks-not-run:preflight")
        elif facts["head"] != facts["candidate"]:
            blockers.append("head-not-candidate")
            not_run = [check["name"] for check in checks]
            not_run_reasons = {name: "head-not-candidate" for name in not_run}
            blockers.append("checks-not-run:preflight")
        else:
            check_results, check_blockers, skipped = run_plan_checks(root, plan, facts["candidate"])
            blockers.extend(check_blockers)
            not_run.extend(skipped)
            not_run_reasons.update({name: "previous-check-changed-state" for name in skipped})
            try:
                check_state["after"] = snapshot_git_state(root, plan)
                before = check_state["before"]
                after = check_state["after"]
                for key in ("ours", "upstream", "candidate", "head"):
                    if before[key] != after[key]:
                        blockers.append(f"checks-{key}-changed")
                if before["status"] != after["status"]:
                    blockers.append("checks-worktree-changed")
                post_facts, post_status, _post_scan, post_blockers, post_warnings = collect_git_facts(root, plan)
                blockers.extend(post_blockers)
                warnings.extend(post_warnings)
                check_state["dirtyTreeAfter"] = bool(post_status)
            except GateError as exc:
                blockers.append("checks-state-unreadable")
                check_state["error"] = str(exc)[-2000:]
    elif run_checks:
        check_state["before"] = snapshot_git_state(root, plan)
        check_state["after"] = snapshot_git_state(root, plan)
    elif checks:
        not_run = [check["name"] for check in checks]
        not_run_reasons = {name: "run-checks-not-requested" for name in not_run}
        blockers.append("checks-not-run")

    try:
        evidence, evidence_blockers, evidence_warnings = collect_evidence(
            root,
            plan,
            facts["candidate"],
            check_results=check_results,
            run_checks=run_checks,
        )
    except GateError as exc:
        evidence = {}
        evidence_blockers = ["evidence-verification-failed"]
        evidence_warnings = []
        check_state["evidenceError"] = str(exc)[-2000:]
    blockers.extend(evidence_blockers)
    warnings.extend(evidence_warnings)

    acknowledgments = {item["code"]: item for item in plan.get("warningAcknowledgments", [])}
    warning_details: list[dict[str, Any]] = []
    unacknowledged: list[str] = []
    for code in sorted(set(warnings)):
        item = acknowledgments.get(code)
        acknowledged = item is not None
        rejected_reason: str | None = None
        if acknowledged and code.startswith("checks-not-run"):
            acknowledged = False
            rejected_reason = "обязательная проверка не подтверждается acknowledgment"
        if acknowledged and not reviewer_is_independent(item["reviewer"], authors):
            acknowledged = False
            rejected_reason = "reviewer не независим от автора"
            blockers.append(f"reviewer-not-independent:{code}")
        warning_details.append({
            "code": code,
            "acknowledged": acknowledged,
            "reason": item.get("reason") if item else None,
            "reviewer": item.get("reviewer") if item else None,
            "rejectedReason": rejected_reason,
        })
        if not acknowledged:
            unacknowledged.append(code)

    if blockers:
        exit_code = EXIT_BLOCKED
        result = "blocked"
    elif unacknowledged:
        exit_code = EXIT_REVIEW
        result = "review"
    else:
        exit_code = EXIT_OK
        result = "pass"

    report = {
        "result": result,
        "exitCode": exit_code,
        "repo": str(root),
        "decision": plan["decision"],
        "rationale": plan["rationale"],
        "author": plan.get("author"),
        "legacyPlan": is_legacy_plan(plan),
        "refs": {
            "ours": plan["oursRef"],
            "upstream": plan["upstreamRef"],
            "candidate": plan["candidateRef"],
        },
        "facts": facts,
        "postCheckFacts": post_facts,
        "dirtyTree": bool(status),
        "matrix": plan["matrix"],
        "evidence": evidence,
        "hybridContract": plan.get("hybridContract"),
        "deltaContract": plan.get("deltaContract"),
        "invariantsChanged": plan.get("invariantsChanged", []),
        "thresholds": plan["thresholds"],
        "thresholdRequests": plan.get("thresholdRequests", {}),
        "architectureReview": review,
        "warningAcknowledgments": plan.get("warningAcknowledgments", []),
        "acceptance": plan.get("acceptance"),
        "blockers": sorted(set(blockers)),
        "warnings": warning_details,
        "checks": check_results,
        "notRun": not_run,
        "notRunReasons": not_run_reasons,
        "checkState": check_state,
    }
    return report, exit_code
