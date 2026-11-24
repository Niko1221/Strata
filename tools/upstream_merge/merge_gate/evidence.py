from __future__ import annotations

from typing import Any

from . import GateError
from .gitops import candidate_author, path_in_ref, resolve_ref, reviewer_is_independent
from .plan import EVIDENCE_CATEGORIES, is_legacy_plan, looks_like_path, normalize_relative_path


def verify_artifact(repo: Path, candidate: str, artifact: str, legacy: bool, legacy_ids: set[str], cache: dict[tuple[str, str], bool]) -> tuple[bool, str]:
    if not looks_like_path(artifact):
        if legacy and artifact in legacy_ids:
            return True, "legacy-reference"
        return False, "missing"
    normalized = normalize_relative_path(artifact, "artifact")
    key = (candidate, normalized)
    if key not in cache:
        cache[key] = path_in_ref(repo, candidate, normalized)
    if cache[key]:
        return True, "candidate-object"
    return False, "missing"


def evidence_commands(item: dict[str, Any]) -> list[Any]:
    commands: list[Any] = []
    if "command" in item:
        commands.append(item["command"])
    commands.extend(item.get("commands", []))
    return commands


def collect_evidence(
    repo: Path,
    plan: dict[str, Any],
    candidate: str,
    check_results: list[dict[str, Any]] | None = None,
    run_checks: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}
    legacy = is_legacy_plan(plan)
    legacy_ids = {
        item["artifact"]
        for item in plan["evidence"].values()
        if isinstance(item.get("artifact"), str) and not looks_like_path(item["artifact"])
    }
    artifact_cache: dict[tuple[str, str], bool] = {}
    successful_commands = [result["command"] for result in (check_results or []) if result.get("ok") and isinstance(result.get("command"), list)]

    for key in EVIDENCE_CATEGORIES:
        if key not in plan["evidence"]:
            continue
        item = plan["evidence"][key]
        status = item["status"]
        verification: list[str] = []
        effective_revision = candidate
        revision_declared = "revision" in item
        if revision_declared:
            effective_revision = resolve_ref(repo, item["revision"])
            if effective_revision != candidate:
                blockers.append(f"evidence-revision-mismatch:{key}")
        elif not legacy:
            blockers.append(f"evidence-revision-missing:{key}")
        if effective_revision == candidate:
            verification.append(f"revision:{effective_revision}")

        if status == "failed":
            blockers.append(f"evidence-failed:{key}")
            if not legacy and (not item.get("reason") or not item.get("details")):
                blockers.append(f"evidence-incomplete:{key}")
        elif status == "not-run":
            blockers.append(f"evidence-not-run:{key}")
            if not legacy and (not item.get("reason") or not item.get("details")):
                blockers.append(f"evidence-incomplete:{key}")
        elif status == "not-applicable":
            if key in {"architectureReview", "fallback"}:
                blockers.append("architecture-review-missing" if key == "architectureReview" else "fallback-evidence-missing")
            else:
                warnings.append(f"evidence-not-applicable:{key}")
            if not legacy:
                if item.get("manualVerdict") != "accepted" or not item.get("reviewer") or not item.get("date"):
                    blockers.append(f"evidence-review-missing:{key}")
        elif status == "passed":
            artifacts = ([item["artifact"]] if "artifact" in item else []) + item.get("artifacts", [])
            for artifact in artifacts:
                exists, kind = verify_artifact(repo, candidate, artifact, legacy, legacy_ids, artifact_cache)
                verification.append(f"artifact:{artifact}:{kind}")
                if not exists:
                    blockers.append(f"evidence-artifact-missing:{key}")
            commands = evidence_commands(item)
            for command in commands:
                if not run_checks or not isinstance(command, list) or command not in successful_commands:
                    blockers.append(f"evidence-command-not-run:{key}")
                else:
                    verification.append("command:executed")
            if not artifacts and not commands:
                blockers.append(f"evidence-empty:{key}")
        details[key] = {
            **item,
            "effectiveRevision": effective_revision,
            "revisionDeclared": revision_declared,
            "verification": verification,
        }

    def verify_reference(ref: Any, label: str) -> None:
        if isinstance(ref, str):
            if looks_like_path(ref) or not legacy:
                exists, _kind = verify_artifact(repo, candidate, ref, legacy, legacy_ids, artifact_cache)
                if not exists:
                    blockers.append(f"matrix-evidence-missing:{label}")
            elif ref not in legacy_ids:
                blockers.append(f"matrix-evidence-missing:{label}")
            return
        category = ref["category"]
        if category not in plan["evidence"]:
            blockers.append(f"matrix-evidence-category-missing:{label}")
            return
        linked = plan["evidence"][category]
        if linked["status"] in {"failed", "not-run"}:
            blockers.append(f"matrix-evidence-not-passed:{label}")
        if "artifact" in ref:
            exists, _kind = verify_artifact(repo, candidate, ref["artifact"], legacy, legacy_ids, artifact_cache)
            if not exists:
                blockers.append(f"matrix-evidence-missing:{label}")
        if "revision" in ref:
            revision = resolve_ref(repo, ref["revision"])
            if revision != candidate:
                blockers.append(f"matrix-evidence-revision-mismatch:{label}")

    for item in plan["matrix"]:
        for index, ref in enumerate(item.get("anchorEvidence", [])):
            verify_reference(ref, f"{item['id']}:anchorEvidence:{index}")
        for index, ref in enumerate(item["evidence"]):
            verify_reference(ref, f"{item['id']}:{index}")
        for key in ("dropProof", "fullReplacementEvidence"):
            for index, ref in enumerate(item.get(key, [])):
                verify_reference(ref, f"{item['id']}:{key}:{index}")
    return details, blockers, warnings


def collect_review(
    repo: Path,
    plan: dict[str, Any],
    candidate: str,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    author = candidate_author(repo, candidate)
    authors = [author]
    evidence_review = plan["evidence"]["architectureReview"]
    reviewer = evidence_review.get("reviewer", "")
    if is_legacy_plan(plan):
        if not reviewer_is_independent(reviewer, authors):
            blockers.append("reviewer-not-independent")
        return {
            "legacy": True,
            "author": author,
            "reviewer": reviewer,
            "independent": reviewer_is_independent(reviewer, authors),
            "checkedAgainst": "candidate commit author",
        }, blockers, warnings, authors

    review = plan.get("architectureReview")
    plan_author = plan.get("author") or (review.get("author", "") if isinstance(review, dict) else "")
    if plan_author:
        authors.append(plan_author)
    else:
        blockers.append("plan-author-missing")
    if not isinstance(review, dict):
        blockers.append("architecture-review-missing")
        return {
            "legacy": False,
            "author": plan_author or author,
            "reviewer": reviewer,
            "independent": False,
        }, blockers, warnings, authors
    if review.get("status") != "completed":
        blockers.append("architecture-review-incomplete")
    if review.get("verdict") != "approved":
        blockers.append("architecture-review-not-approved")
    if review.get("independent") is not True:
        blockers.append("architecture-review-not-declared-independent")
    for key in ("author", "reviewer", "date", "signature", "artifact"):
        if not review.get(key):
            blockers.append(f"architecture-review-missing:{key}")
    review_author = review.get("author", "")
    review_reviewer = review.get("reviewer", "")
    if review_author:
        authors.append(review_author)
    if review_reviewer:
        reviewer = review_reviewer
    if evidence_review.get("reviewer") != reviewer:
        blockers.append("architecture-review-reviewer-mismatch")
    independent = reviewer_is_independent(reviewer, authors)
    if not independent:
        blockers.append("reviewer-not-independent")
    scope = review.get("scope", [])
    matrix_ids = {item["id"] for item in plan["matrix"]}
    if not isinstance(scope, list) or not matrix_ids.issubset(set(scope)):
        blockers.append("architecture-review-scope-missing")
    artifact = review.get("artifact")
    if artifact:
        exists, _kind = verify_artifact(repo, candidate, artifact, False, set(), {})
        if not exists:
            blockers.append("architecture-review-artifact-missing")
    return {
        **review,
        "legacy": False,
        "reviewer": reviewer,
        "independent": independent,
        "authorsChecked": sorted(set(authors)),
    }, blockers, warnings, authors
