from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from . import EXIT_BLOCKED, EXIT_INVALID, EXIT_OK, EXIT_REVIEW, GateError
from .gate import evaluate
from .gitops import git_stdout, git_text, run_git
from .plan import DEFAULT_THRESHOLDS, REQUIRED_EVIDENCE, clone_plan, validate_plan
from .report import human_report


def fixture_git(repo: Path, args: list[str]) -> None:
    result = run_git(repo, args, timeout=120)
    if result.returncode != 0:
        raise GateError((result.stderr or result.stdout).strip()[:2000])


def make_fixture(root: Path) -> dict[str, Any]:
    fixture_git(root, ["init", "-q", "-b", "main"])
    fixture_git(root, ["config", "user.name", "Merge Gate Selftest"])
    fixture_git(root, ["config", "user.email", "selftest@example.invalid"])
    (root / "evidence").mkdir()
    (root / "base.txt").write_text("base\n", encoding="utf-8")
    (root / "evidence" / "selftest.txt").write_text("selftest evidence\n", encoding="utf-8")
    fixture_git(root, ["add", "base.txt", "evidence/selftest.txt"])
    fixture_git(root, ["commit", "-q", "-m", "base"])

    fixture_git(root, ["checkout", "-q", "-b", "ours"])
    (root / "ours.txt").write_text("ours\n", encoding="utf-8")
    fixture_git(root, ["add", "ours.txt"])
    fixture_git(root, ["commit", "-q", "-m", "ours"])

    fixture_git(root, ["checkout", "-q", "-b", "upstream", "main"])
    (root / "upstream.txt").write_text("upstream\n", encoding="utf-8")
    fixture_git(root, ["add", "upstream.txt"])
    fixture_git(root, ["commit", "-q", "-m", "upstream"])

    fixture_git(root, ["checkout", "-q", "ours"])
    fixture_git(root, ["merge", "-q", "--no-ff", "upstream", "-m", "candidate"])
    fixture_git(root, ["branch", "candidate"])

    artifact = "evidence/selftest.txt"
    evidence = {
        key: {"status": "passed", "artifact": artifact, "revision": "candidate"}
        for key in REQUIRED_EVIDENCE
    }
    evidence["architectureReview"]["reviewer"] = "Independent Reviewer"
    evidence["implementationCombinations"] = {
        "status": "passed",
        "artifact": artifact,
        "revision": "candidate",
        "combinations": [
            {"implementation": "upstream-only", "revision": "upstream", "status": "passed", "evidence": [artifact]},
            {"implementation": "ours-only", "revision": "ours", "status": "passed", "evidence": [artifact]},
            {"implementation": "hybrid", "revision": "candidate", "status": "passed", "evidence": [artifact]},
        ],
    }
    return {
        "version": 1,
        "author": "Plan Author",
        "oursRef": "ours",
        "upstreamRef": "upstream",
        "candidateRef": "candidate",
        "decision": "hybrid",
        "rationale": "selftest",
        "matrix": [{
            "id": "selftest",
            "oursAnchor": "ours.txt",
            "upstreamAnchor": "base.txt",
            "anchorStatus": "preserved",
            "anchorMethod": {
                "method": "source-review",
                "command": ["git", "grep", "-n", "ours.txt", "candidate", "--"],
                "checkedRefs": ["ours", "upstream", "candidate"],
            },
            "anchorEvidence": [artifact],
            "upstreamStatus": "partial",
            "upstreamAlgorithm": "fixture algorithm",
            "reworkEffect": "same",
            "decision": "hybrid",
            "reason": "selftest",
            "evidence": [artifact],
        }],
        "hybridContract": {
            "boundary": "fixture boundary",
            "inputs": "fixture inputs",
            "outputs": "fixture outputs",
            "ownership": "fixture ownership",
            "streamOrder": "fixture stream order",
            "fallback": "fixture fallback",
        },
        "architectureReview": {
            "status": "completed",
            "scope": ["selftest"],
            "author": "Plan Author",
            "reviewer": "Independent Reviewer",
            "independent": True,
            "date": "2026-11-24",
            "verdict": "approved",
            "signature": "selftest signature",
            "artifact": artifact,
        },
        "invariantsChanged": [],
        "evidence": evidence,
        "checks": [],
        "warningAcknowledgments": [],
        "thresholds": dict(DEFAULT_THRESHOLDS),
    }


def point_evidence_at(plan: dict[str, Any], ref: str) -> None:
    for item in plan["evidence"].values():
        if "revision" in item:
            item["revision"] = ref
        for combination in item.get("combinations", []):
            if combination.get("implementation") in {"hybrid", "delta"}:
                combination["revision"] = ref


def commit_fixture_file(root: Path, branch: str, path: str, content: str) -> None:
    fixture_git(root, ["checkout", "-q", "candidate"])
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    fixture_git(root, ["add", "--", path])
    fixture_git(root, ["commit", "-q", "-m", branch])
    fixture_git(root, ["branch", branch])


def restore_candidate(root: Path, candidate_sha: str) -> None:
    fixture_git(root, ["checkout", "-q", "--detach", candidate_sha])
    fixture_git(root, ["branch", "-f", "candidate", candidate_sha])
    fixture_git(root, ["checkout", "-q", "candidate"])


def selftest() -> int:
    passed = 0
    root = Path(tempfile.mkdtemp(prefix="merge-gate-selftest-"))

    def expect(condition: bool, name: str, detail: Any = "") -> None:
        nonlocal passed
        if not condition:
            raise RuntimeError(f"{name}: {detail}")
        passed += 1
        print(f"ok {passed}: {name}")

    def expect_invalid(call: Callable[[], Any], name: str) -> None:
        try:
            call()
        except GateError:
            expect(True, name)
        else:
            expect(False, name, "GateError не получен")

    try:
        plan = make_fixture(root)
        candidate_sha = git_text(root, ["rev-parse", "candidate"])
        head_before = git_text(root, ["rev-parse", "HEAD"])
        status_before = git_stdout(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"])

        report, code = evaluate(root, plan)
        expect(code == EXIT_OK, "read-only pass не меняет refs и дерево", human_report(report))
        expect(git_text(root, ["rev-parse", "HEAD"]) == head_before, "read-only pass сохраняет HEAD")
        expect(
            git_stdout(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"]) == status_before,
            "read-only pass сохраняет статус",
        )

        fixture_git(root, ["config", "status.showUntrackedFiles", "no"])
        (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        report, code = evaluate(root, plan)
        expect(code == EXIT_BLOCKED and "dirty-tree" in report["blockers"], "untracked файл виден при showUntrackedFiles=no")
        (root / "untracked.txt").unlink()

        bad = clone_plan(plan)
        bad["candidateRef"] = "main"
        report, code = evaluate(root, bad)
        expect(code == EXIT_BLOCKED and "candidate-not-descendant-of-ours" in report["blockers"], "плохая ancestry блокируется")

        commit_fixture_file(root, "bad-markers", "conflict.txt", "<<<<<<< HEAD\nx\n=======\ny\n>>>>>>> upstream\n")
        marked = clone_plan(plan)
        marked["candidateRef"] = "bad-markers"
        point_evidence_at(marked, "bad-markers")
        report, code = evaluate(root, marked)
        expect(code == EXIT_BLOCKED and "conflict-markers" in report["blockers"], "новые conflict markers блокируются")
        existing = clone_plan(marked)
        existing["oursRef"] = "bad-markers"
        existing["upstreamRef"] = "bad-markers"
        report, code = evaluate(root, existing)
        expect(code == EXIT_BLOCKED and "conflict-markers" in report["blockers"], "conflict markers находятся во всём candidate, а не только в diff")
        restore_candidate(root, candidate_sha)

        commit_fixture_file(
            root,
            "bad-diff3",
            "diff3.txt",
            "<<<<<<<<< ours\nbase\n||||||| ancestor\n========\ntheirs\n>>>>>>> upstream\n",
        )
        diff3 = clone_plan(plan)
        diff3["candidateRef"] = "bad-diff3"
        point_evidence_at(diff3, "bad-diff3")
        report, code = evaluate(root, diff3)
        expect(code == EXIT_BLOCKED and "conflict-markers" in report["blockers"], "diff3 и marker-size=8 блокируются")
        restore_candidate(root, candidate_sha)

        commit_fixture_file(root, "markdown", "README.md", "Заголовок\n=======\n")
        markdown = clone_plan(plan)
        markdown["candidateRef"] = "markdown"
        point_evidence_at(markdown, "markdown")
        report, code = evaluate(root, markdown)
        expect(code == EXIT_OK and "conflict-markers" not in report["blockers"], "Markdown Setext не считается conflict marker", human_report(report))
        restore_candidate(root, candidate_sha)

        fixture_git(root, ["config", "diff.noprefix", "true"])
        fixture_git(root, ["config", "core.quotePath", "true"])
        commit_fixture_file(root, "unicode-path", "unicode-é.txt", "static int global_cache = 0;\n")
        unicode_plan = clone_plan(plan)
        unicode_plan["candidateRef"] = "unicode-path"
        point_evidence_at(unicode_plan, "unicode-path")
        report, code = evaluate(root, unicode_plan)
        expect(
            code == EXIT_REVIEW and "global-state" in [item["code"] for item in report["warnings"]],
            "noprefix и quoted Unicode-путь не скрывают structural scan",
            human_report(report),
        )
        restore_candidate(root, candidate_sha)

        commit_fixture_file(
            root,
            "heuristics",
            "heuristics.txt",
            "void f() {\n"
            "    static int local_cache = 0;\n"
            "    std::mutex local_lock;\n"
            "    int i = 1;\n"
            "    int width = 2;\n"
            "    int value = i + 2 * width;\n"
            "    switch (i) {\n"
            "    case 1: value += 1; break;\n"
            "    default: break;\n"
            "    }\n"
            "    switch (i) {\n"
            "    case 1: value += 1; break;\n"
            "    default: break;\n"
            "    }\n"
            "}\n",
        )
        heuristics = clone_plan(plan)
        heuristics["candidateRef"] = "heuristics"
        point_evidence_at(heuristics, "heuristics")
        report, code = evaluate(root, heuristics)
        warning_codes = {item["code"] for item in report["warnings"]}
        expect(code == EXIT_OK, "эвристики не блокируют локальный state и арифметику", human_report(report))
        expect(not ({"global-state", "offset-coupling", "copy-paste"} & warning_codes), "повторные switch-блоки и локальный mutex не дают ложных warnings")

        incomplete = clone_plan(plan)
        incomplete["evidence"]["tests"] = {
            "status": "not-run",
            "reason": "selftest",
            "details": "not run",
            "revision": "candidate",
        }
        incomplete["warningAcknowledgments"] = [{
            "code": "evidence-not-run:tests",
            "reason": "selftest",
            "reviewer": "Independent Reviewer",
        }]
        report, code = evaluate(root, incomplete)
        expect(code == EXIT_BLOCKED and "evidence-not-run:tests" in report["blockers"], "not-run evidence нельзя погасить acknowledgment")

        fallback = clone_plan(plan)
        fallback["evidence"]["fallback"] = {
            "status": "not-applicable",
            "reason": "selftest",
            "reviewer": "Independent Reviewer",
            "date": "2026-11-24",
            "manualVerdict": "accepted",
            "revision": "candidate",
        }
        fallback["warningAcknowledgments"] = [{
            "code": "evidence-not-applicable:fallback",
            "reason": "selftest",
            "reviewer": "Independent Reviewer",
        }]
        report, code = evaluate(root, fallback)
        expect(code == EXIT_BLOCKED and "fallback-evidence-missing" in report["blockers"], "обязательный fallback нельзя объявить not-applicable")

        checks_not_run = clone_plan(plan)
        checks_not_run["checks"] = [{
            "name": "safe python",
            "command": [sys.executable, "-c", "print('ok')"],
            "cwd": ".",
            "timeoutMs": 10000,
        }]
        checks_not_run["warningAcknowledgments"] = [{
            "code": "checks-not-run",
            "reason": "selftest",
            "reviewer": "Independent Reviewer",
        }]
        report, code = evaluate(root, checks_not_run)
        expect(code == EXIT_BLOCKED and "checks-not-run" in report["blockers"], "checks-not-run нельзя погасить acknowledgment")

        deferred = clone_plan(plan)
        deferred["decision"] = "defer"
        deferred["matrix"][0]["decision"] = "defer"
        deferred["matrix"][0]["evidence"] = []
        deferred["defer"] = {
            "experimentRef": "experiment/defer",
            "reviewer": "Independent Reviewer",
            "missingEvidence": ["build"],
            "unblockConditions": ["зелёная сборка"],
        }
        report, code = evaluate(root, deferred)
        expect(code == EXIT_BLOCKED and "defer-not-ready" in report["blockers"], "defer остаётся экспериментом и блокируется")

        contradictory = clone_plan(plan)
        contradictory["matrix"][0]["upstreamStatus"] = "none"
        contradictory["matrix"][0]["decision"] = "upstream"
        report, code = evaluate(root, contradictory)
        expect(code == EXIT_BLOCKED and "upstream-none-decision:selftest" in report["blockers"], "upstream при upstreamStatus=none блокируется")

        bad_drop = clone_plan(plan)
        bad_drop["matrix"][0].update({
            "decision": "drop",
            "upstreamStatus": "partial",
            "replacementUpstreamAnchor": "ours.txt",
            "dropProof": ["evidence/selftest.txt"],
        })
        expect_invalid(lambda: validate_plan(bad_drop), "drop без полной upstream-замены отклоняется")

        bad_cwd = clone_plan(plan)
        bad_cwd["checks"] = [{
            "name": "outside",
            "command": [sys.executable, "-c", "print('ok')"],
            "cwd": "../outside",
            "timeoutMs": 10000,
        }]
        expect_invalid(lambda: validate_plan(bad_cwd), "cwd вне репозитория отклоняется при валидации")

        warning = clone_plan(plan)
        warning["thresholds"]["maxChangedFiles"] = 0
        report, code = evaluate(root, warning)
        expect(code == EXIT_REVIEW and "large-diff-files" in [item["code"] for item in report["warnings"]], "ужесточённый порог требует review")
        acknowledged = clone_plan(warning)
        acknowledged["warningAcknowledgments"] = [{
            "code": "large-diff-files",
            "reason": "selftest acknowledgment",
            "scope": ["selftest"],
            "mitigation": "selftest mitigation",
            "evidence": ["evidence/selftest.txt"],
            "reviewer": "Independent Reviewer",
            "date": "2026-11-24",
            "outcome": "accepted",
        }]
        validate_plan(acknowledged)
        report, code = evaluate(root, acknowledged)
        expect(code == EXIT_OK, "независимый reviewer подтверждает предупреждение", human_report(report))
        incomplete_ack = clone_plan(warning)
        incomplete_ack["warningAcknowledgments"] = [{
            "code": "large-diff-files",
            "reason": "selftest incomplete acknowledgment",
            "reviewer": "Independent Reviewer",
        }]
        expect_invalid(lambda: validate_plan(incomplete_ack), "неполный warning acknowledgment отклоняется")
        no_combinations = clone_plan(plan)
        del no_combinations["evidence"]["implementationCombinations"]
        expect_invalid(lambda: validate_plan(no_combinations), "hybrid без combination evidence отклоняется")

        relaxed = clone_plan(plan)
        relaxed["thresholds"]["maxChangedFiles"] = DEFAULT_THRESHOLDS["maxChangedFiles"] + 1
        report, code = evaluate(root, relaxed)
        expect(code == EXIT_BLOCKED and "thresholds-relaxed:maxChangedFiles" in report["blockers"], "план не может ослабить встроенные thresholds")

        injection = clone_plan(plan)
        injection["checks"] = [{
            "name": "invalid shell string",
            "command": "touch should-not-exist",
            "cwd": ".",
            "timeoutMs": 1000,
        }]
        expect_invalid(lambda: validate_plan(injection), "shell-строка вместо argv отклоняется")

        nul_argv = clone_plan(plan)
        nul_argv["checks"] = [{
            "name": "nul",
            "command": [sys.executable, "-c", "print('ok')", "bad\0arg"],
            "cwd": ".",
            "timeoutMs": 1000,
        }]
        expect_invalid(lambda: validate_plan(nul_argv), "NUL в argv отклоняется без traceback")

        bad_status = clone_plan(plan)
        bad_status["evidence"]["tests"]["status"] = []
        expect_invalid(lambda: validate_plan(bad_status), "status=[] отклоняется без traceback")

        invalid_plan_path = root.parent / "invalid-merge-plan.json"
        invalid_plan_path.write_bytes(b"\xff\xfe")
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "merge-gate.py"), "check", "--repo", str(root), "--plan", str(invalid_plan_path), "--format", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            check=False,
        )
        expect(process.returncode == EXIT_INVALID, "invalid UTF-8 даёт exit2")
        expect("Traceback" not in process.stderr, "invalid UTF-8 не даёт traceback", process.stderr)

        missing_artifact = clone_plan(plan)
        missing_artifact["evidence"]["build"]["artifact"] = "evidence/missing.txt"
        report, code = evaluate(root, missing_artifact)
        expect(code == EXIT_BLOCKED and "evidence-artifact-missing:build" in report["blockers"], "отсутствующий artifact блокируется")

        fixture_git(root, ["checkout", "-q", "-b", "other", "main"])
        (root / "foreign.txt").write_text("foreign evidence\n", encoding="utf-8")
        fixture_git(root, ["add", "foreign.txt"])
        fixture_git(root, ["commit", "-q", "-m", "foreign"])
        worktree_artifact = clone_plan(plan)
        for item in worktree_artifact["evidence"].values():
            item["artifact"] = "foreign.txt"
        worktree_artifact["matrix"][0]["evidence"] = ["foreign.txt"]
        worktree_artifact["architectureReview"]["artifact"] = "foreign.txt"
        report, code = evaluate(root, worktree_artifact)
        expect(code == EXIT_BLOCKED and "evidence-artifact-missing:build" in report["blockers"], "worktree artifact вне candidate не принимается")
        fixture_git(root, ["checkout", "-q", "candidate"])

        fixture_git(root, ["checkout", "-q", "candidate"])
        (root / "ours.txt").unlink()
        fixture_git(root, ["add", "-u"])
        fixture_git(root, ["commit", "-q", "-m", "remove ours"])
        fixture_git(root, ["branch", "candidate2"])
        removed_ours = clone_plan(plan)
        removed_ours["candidateRef"] = "candidate2"
        removed_ours["decision"] = "ours"
        removed_ours["matrix"][0].update({"decision": "ours", "anchorStatus": "removed"})
        point_evidence_at(removed_ours, "candidate2")
        report, code = evaluate(root, removed_ours)
        expect(code == EXIT_BLOCKED and "removed-anchor-decision-conflict:selftest" in report["blockers"], "removed anchor с решением ours блокируется")
        restore_candidate(root, git_text(root, ["rev-parse", "heuristics"]))

        wrong_revision = clone_plan(plan)
        wrong_revision["evidence"]["tests"]["revision"] = "upstream"
        report, code = evaluate(root, wrong_revision)
        expect(code == EXIT_BLOCKED and "evidence-revision-mismatch:tests" in report["blockers"], "evidence другого revision блокируется")

        command_evidence = clone_plan(plan)
        command = [sys.executable, "-c", "print('ok')"]
        command_evidence["evidence"]["build"]["command"] = command
        report, code = evaluate(root, command_evidence)
        expect(code == EXIT_BLOCKED and "evidence-command-not-run:build" in report["blockers"], "невыполненная evidence command блокируется")
        command_evidence["checks"] = [{"name": "build", "command": command, "cwd": ".", "timeoutMs": 10000}]
        report, code = evaluate(root, command_evidence, run_checks=True)
        expect(code == EXIT_OK and report["checks"][0]["ok"], "выполненная evidence command принимается", human_report(report))

        wrong_head = clone_plan(plan)
        marker = root / "check-ran.txt"
        wrong_head["checks"] = [{
            "name": "must not run",
            "command": [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"],
            "cwd": ".",
            "timeoutMs": 10000,
        }]
        fixture_git(root, ["checkout", "-q", "ours"])
        report, code = evaluate(root, wrong_head, run_checks=True)
        expect(code == EXIT_BLOCKED and "head-not-candidate" in report["blockers"], "checks на другом HEAD не запускаются")
        expect(not marker.exists() and not report["checks"], "команда при HEAD!=candidate не выполняется")
        fixture_git(root, ["checkout", "-q", "candidate"])

        moved_head = clone_plan(plan)
        moved_head["checks"] = [{
            "name": "move head",
            "command": [sys.executable, "-c", "import subprocess; subprocess.run(['git', 'symbolic-ref', 'HEAD', 'refs/heads/ours'], check=True)"],
            "cwd": ".",
            "timeoutMs": 10000,
        }]
        report, code = evaluate(root, moved_head, run_checks=True)
        expect(code == EXIT_BLOCKED and "checks-head-changed" in report["blockers"], "смена HEAD во время checks блокируется")
        fixture_git(root, ["checkout", "-q", "candidate"])

        dirty_check = clone_plan(plan)
        dirty_check["checks"] = [{
            "name": "dirty tree",
            "command": [sys.executable, "-c", "from pathlib import Path; Path('check-dirty.txt').write_text('dirty')"],
            "cwd": ".",
            "timeoutMs": 10000,
        }]
        report, code = evaluate(root, dirty_check, run_checks=True)
        expect(code == EXIT_BLOCKED and "checks-worktree-changed" in report["blockers"], "изменение дерева во время checks блокируется")
        (root / "check-dirty.txt").unlink()

        git_dir = Path(git_text(root, ["rev-parse", "--absolute-git-dir"]))
        sequencer = git_dir / "sequencer"
        sequencer.mkdir()
        report, code = evaluate(root, plan)
        expect(code == EXIT_BLOCKED and "merge-state:sequencer" in report["blockers"], "активный sequencer блокируется")
        sequencer.rmdir()

        same_reviewer = clone_plan(plan)
        same_reviewer["architectureReview"]["reviewer"] = "Plan Author"
        same_reviewer["evidence"]["architectureReview"]["reviewer"] = "Plan Author"
        report, code = evaluate(root, same_reviewer)
        expect(code == EXIT_BLOCKED and "reviewer-not-independent" in report["blockers"], "автор плана не может быть своим reviewer")

        print(f"Selftest: {passed} проверок пройдено.")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)
