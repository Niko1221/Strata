from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from . import GateError
from .plan import DEFAULT_THRESHOLDS
from .scan import structural_scan, scan_candidate_conflicts


GIT_STATE_FILES = (
    "MERGE_HEAD",
    "rebase-merge",
    "rebase-apply",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "sequencer",
)


GIT_PREFIX = [
    "git",
    "--no-optional-locks",
    "--literal-pathspecs",
    "-c", "core.fsmonitor=false",
    "-c", "core.untrackedCache=false",
    "-c", "core.preloadIndex=false",
    "-c", "status.showUntrackedFiles=all",
    "-c", "diff.noprefix=false",
    "-c", "core.quotePath=false",
    "-c", "diff.external=",
]


def run_git(repo: Path, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_PAGER"] = "cat"
    env.pop("GIT_EXTERNAL_DIFF", None)
    env.pop("GIT_DIFF_OPTS", None)
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


def git_text(repo: Path, args: list[str], timeout: int = 30) -> str:
    result = run_git(repo, args, timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:2000]
        raise GateError(f"git {' '.join(args)}: {detail or 'ошибка'}")
    return result.stdout.strip()


def git_stdout(repo: Path, args: list[str], timeout: int = 30) -> str:
    result = run_git(repo, args, timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:2000]
        raise GateError(f"git {' '.join(args)}: {detail or 'ошибка'}")
    return result.stdout


def repo_root(repo: Path) -> Path:
    if not repo.exists() or not repo.is_dir():
        raise GateError(f"репозиторий не найден: {repo}")
    root = git_text(repo, ["rev-parse", "--show-toplevel"])
    return Path(root).resolve()


def resolve_ref(repo: Path, ref: str) -> str:
    if not isinstance(ref, str) or not ref.strip() or "\0" in ref:
        raise GateError("ref не может быть пустым")
    return git_text(repo, ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"])


def is_descendant(repo: Path, ancestor: str, descendant: str) -> bool:
    result = run_git(repo, ["merge-base", "--is-ancestor", ancestor, descendant])
    if result.returncode not in (0, 1):
        raise GateError((result.stderr or result.stdout).strip()[:2000])
    return result.returncode == 0


def parse_numstat(text: str) -> tuple[list[str], int, int]:
    files: list[str] = []
    added = 0
    deleted = 0
    records = text.split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        parts = record.split("\t", 2)
        if len(parts) != 3:
            continue
        path = parts[2]
        if not path and index + 1 < len(records):
            _old_path = records[index]
            path = records[index + 1]
            index += 2
        if path:
            files.append(path)
        if parts[0].isdigit():
            added += int(parts[0])
        if parts[1].isdigit():
            deleted += int(parts[1])
    return files, added, deleted


def path_in_ref(repo: Path, ref: str, path: str) -> bool:
    return bool(git_stdout(repo, ["ls-tree", "-z", "--full-tree", ref, "--", path]))


def anchor_exists(repo: Path, ref: str, anchor: str) -> bool:
    if path_in_ref(repo, ref, anchor):
        return True
    result = run_git(repo, ["grep", "-I", "-F", "-q", "-e", anchor, ref, "--"])
    if result.returncode in (0, 1):
        return result.returncode == 0
    raise GateError((result.stderr or result.stdout).strip()[:2000])


def candidate_author(repo: Path, candidate: str) -> str:
    value = git_stdout(repo, ["show", "-s", "--format=%an%x00%ae", candidate])
    name, _email = (value.split("\0", 1) + [""])[:2]
    return name.strip()


def identity_tokens(value: str) -> set[str]:
    text = " ".join(value.strip().split()).casefold()
    tokens = {text} if text else set()
    match = re.search(r"<([^<>]+)>", text)
    if match:
        tokens.add(match.group(1).strip())
        name = text.split("<", 1)[0].strip()
        if name:
            tokens.add(name)
    return tokens


def same_identity(left: str, right: str) -> bool:
    return bool(identity_tokens(left) & identity_tokens(right))


def reviewer_is_independent(reviewer: str, authors: list[str]) -> bool:
    return bool(reviewer.strip()) and not any(same_identity(reviewer, author) for author in authors if author.strip())


def active_git_state(repo: Path) -> list[str]:
    git_dir = Path(git_text(repo, ["rev-parse", "--absolute-git-dir"]))
    return [name for name in GIT_STATE_FILES if (git_dir / name).exists()]


def collect_git_facts(repo: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    status = git_stdout(
        repo,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
    )
    if status:
        blockers.append("dirty-tree")

    active_state = active_git_state(repo)
    if active_state:
        blockers.append("merge-state:" + ",".join(active_state))

    ours = resolve_ref(repo, plan["oursRef"])
    upstream = resolve_ref(repo, plan["upstreamRef"])
    candidate = resolve_ref(repo, plan["candidateRef"])
    base = git_text(repo, ["merge-base", ours, upstream])
    if not is_descendant(repo, ours, candidate):
        blockers.append("candidate-not-descendant-of-ours")
    if not is_descendant(repo, upstream, candidate):
        blockers.append("candidate-not-descendant-of-upstream")

    left_right = git_text(repo, ["rev-list", "--left-right", "--count", f"{ours}...{upstream}"])
    counts = left_right.split()
    if len(counts) != 2:
        raise GateError("git rev-list вернул некорректные ahead/behind")
    ahead_behind = {"oursOnly": int(counts[0]), "upstreamOnly": int(counts[1])}

    numstat = git_stdout(repo, ["diff", "--numstat", "-z", "--no-renames", base, candidate, "--"])
    files, added, deleted = parse_numstat(numstat)
    diff_text = git_stdout(
        repo,
        [
            "diff", "--unified=3", "--no-color", "--no-ext-diff", "--no-textconv",
            "--no-renames", "--src-prefix=a/", "--dst-prefix=b/", base, candidate, "--",
        ],
        timeout=120,
    )
    scan = structural_scan(diff_text)
    scan["conflictMarkers"], scan["conflictMarkerFragments"] = scan_candidate_conflicts(repo, candidate)

    if scan["conflictMarkers"]:
        blockers.append("conflict-markers")
    if scan["conflictMarkerFragments"]:
        warnings.append("conflict-marker-fragments")

    anchors: list[dict[str, Any]] = []
    for item in plan["matrix"]:
        ours_exists = anchor_exists(repo, ours, item["oursAnchor"])
        candidate_exists = anchor_exists(repo, candidate, item["oursAnchor"])
        candidate_anchor = item.get("candidateAnchor")
        adapted_exists = bool(candidate_anchor and anchor_exists(repo, candidate, candidate_anchor))
        removed = item.get("anchorStatus") == "removed"
        removed_allowed = removed and item["decision"] == "drop" and item["upstreamStatus"] == "full"
        absent_allowed = item["decision"] == "drop" or removed_allowed or adapted_exists
        if removed and not removed_allowed:
            blockers.append(f"removed-anchor-decision-conflict:{item['id']}")
        anchors.append({
            "id": item["id"],
            "oursAnchor": item["oursAnchor"],
            "inOurs": ours_exists,
            "inCandidate": candidate_exists,
            "candidateAnchor": candidate_anchor,
            "candidateAnchorFound": adapted_exists,
        })
        if not ours_exists:
            blockers.append(f"ours-anchor-missing:{item['id']}")
        if not candidate_exists and not absent_allowed:
            blockers.append(f"ours-anchor-not-in-candidate:{item['id']}")
        if item.get("upstreamAnchor") and item["upstreamStatus"] in {"full", "partial"}:
            if not anchor_exists(repo, upstream, item["upstreamAnchor"]):
                blockers.append(f"upstream-anchor-missing:{item['id']}")
        if item["decision"] == "drop":
            replacement = item["replacementUpstreamAnchor"]
            if not anchor_exists(repo, upstream, replacement) or not anchor_exists(repo, candidate, replacement):
                blockers.append(f"drop-replacement-missing:{item['id']}")
        if item["upstreamStatus"] == "none" and item["decision"] in {"upstream", "delta", "hybrid"}:
            blockers.append(f"upstream-none-decision:{item['id']}")
        if item["decision"] == "upstream" and item["reworkEffect"] == "worse":
            blockers.append(f"upstream-worse-rework:{item['id']}")

    if plan["decision"] == "defer" and any(item["decision"] != "defer" for item in plan["matrix"]):
        blockers.append("defer-decision-conflict")
    if plan["decision"] == "drop" and not any(item["decision"] == "drop" for item in plan["matrix"]):
        blockers.append("drop-decision-conflict")

    thresholds = plan["thresholds"]
    if len(files) > thresholds["maxChangedFiles"]:
        warnings.append("large-diff-files")
    if added + deleted > thresholds["maxChangedLines"]:
        warnings.append("large-diff-lines")
    if scan["ifdefAdded"] > thresholds["maxIfdefAdded"]:
        warnings.append("ifdef-growth")
    if scan["globalState"]:
        warnings.append("global-state")
    if scan["offsetCoupling"]:
        warnings.append("offset-coupling")
    if scan["copyPasteWindows"]:
        warnings.append("copy-paste")
    if len(plan.get("invariantsChanged", [])) > thresholds["maxInvariantsChanged"]:
        warnings.append("invariants-changed")
    for key, requested in plan["thresholds"].items():
        if requested > DEFAULT_THRESHOLDS[key]:
            blockers.append(f"thresholds-relaxed:{key}")
    for key, requested in plan.get("thresholdRequests", {}).items():
        if requested > DEFAULT_THRESHOLDS[key]:
            blockers.append(f"thresholds-relaxed:{key}")

    facts = {
        "ours": ours,
        "upstream": upstream,
        "candidate": candidate,
        "mergeBase": base,
        "head": git_text(repo, ["rev-parse", "HEAD"]),
        "aheadBehind": ahead_behind,
        "changedFiles": len(files),
        "addedLines": added,
        "deletedLines": deleted,
        "structural": scan,
        "anchors": anchors,
        "activeState": active_state,
    }
    return facts, status, scan, blockers, warnings


def snapshot_git_state(repo: Path, plan: dict[str, Any]) -> dict[str, str]:
    return {
        "ours": resolve_ref(repo, plan["oursRef"]),
        "upstream": resolve_ref(repo, plan["upstreamRef"]),
        "candidate": resolve_ref(repo, plan["candidateRef"]),
        "head": git_text(repo, ["rev-parse", "HEAD"]),
        "status": git_stdout(
            repo,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
        ),
    }
