#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(__file__).with_name("upstream_merge.py")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def run_cli(repo: Path, *args: str) -> tuple[int, dict | None, str]:
    result = subprocess.run(
        [sys.executable, str(CLI), *args, "--repo", str(repo)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        check=False,
    )
    data = None
    if result.stdout.strip():
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            data = None
    return result.returncode, data, result.stderr


def valid_plan(candidate: str = "opt/promote") -> dict:
    evidence = {
        key: {"status": "passed", "revision": candidate, "artifact": "base.txt"}
        for key in (
            "build", "tests", "correctness", "quality", "route", "fallback", "backends",
            "architectureReview", "benchmarksResources", "implementationCombinations",
            "targetCoverage", "regression", "bisect", "rollback",
        )
    }
    evidence["architectureReview"]["reviewer"] = "Independent Reviewer"
    evidence["benchmarksResources"].update({
        "baseline": {"label": "ours", "revision": "ours"},
        "experimentProfile": {
            "id": "fixture",
            "model": "fixture.gguf",
            "hardware": "fixture",
            "backend": "CUDA",
            "flags": {},
        },
        "repeats": 1,
        "results": [{"name": "fixture", "value": 1, "unit": "case"}],
        "tolerances": [{"metric": "fixture", "kind": "exact", "limit": 0, "unit": "case"}],
    })
    evidence["implementationCombinations"]["combinations"] = [
        {"implementation": name, "revision": ref, "status": "passed", "evidence": ["base.txt"]}
        for name, ref in (("upstream-only", "upstream"), ("ours-only", "ours"), ("hybrid", candidate))
    ]
    evidence["targetCoverage"].update({
        "targets": {
            "models": ["fixture"],
            "backends": [],
            "formats": [],
            "flags": [],
            "edgeCases": [],
        },
        "gaps": [],
    })
    questions = {
        key: {"verdict": "pass", "rationale": "fixture", "evidence": ["base.txt"]}
        for key in (
            "algorithmMeaning", "bufferOwnershipAndReadiness", "mtpDeterminismAndFallback",
            "partIndependence", "nextMergeCost", "separateAdaptation",
        )
    }
    return {
        "version": 1,
        "oursRef": "ours",
        "upstreamRef": "upstream",
        "candidateRef": candidate,
        "decision": "hybrid",
        "rationale": "fixture",
        "matrix": [{
            "id": "fixture",
            "oursAnchor": "ours.txt",
            "upstreamAnchor": "base.txt",
            "anchorLocations": {
                "ours": {"ref": "ours", "path": "ours.txt", "line": 1, "symbol": "ours"},
                "upstream": {"ref": "upstream", "path": "base.txt", "line": 1, "symbol": "base"},
            },
            "anchorStatus": "preserved",
            "anchorMethod": {
                "method": "source-review",
                "command": ["git", "grep", "-n", "ours.txt", candidate, "--"],
                "checkedRefs": ["ours", "upstream", candidate],
            },
            "anchorEvidence": ["base.txt"],
            "upstreamStatus": "partial",
            "oursAlgorithm": "fixture",
            "upstreamAlgorithm": "fixture",
            "contract": {
                "ours": "fixture",
                "upstream": "fixture",
                "equivalence": "compatible",
                "numericalTolerance": None,
                "determinism": "preserved",
                "boundaryBehavior": "fixture",
                "errorBehavior": "fixture",
                "sessionSemantics": "fixture",
            },
            "route": {
                "ours": "fixture",
                "upstream": "fixture",
                "candidate": "fixture",
                "gate": "fixture",
                "dispatch": "fixture",
                "fallback": "fixture",
                "defaultFlags": "fixture",
                "target": "fixture",
                "verified": True,
                "evidence": ["base.txt"],
            },
            "invariants": [],
            "resources": {
                "measurementStatus": "not-measured",
                "ours": {},
                "upstream": {},
                "comparison": "fixture",
                "evidence": ["base.txt"],
            },
            "coverage": {
                "status": "declared",
                "ours": {"models": [], "backends": [], "formats": [], "flags": [], "edgeCases": []},
                "upstream": {"models": [], "backends": [], "formats": [], "flags": [], "edgeCases": []},
                "candidate": {"models": [], "backends": [], "formats": [], "flags": [], "edgeCases": []},
                "gaps": [],
                "evidence": ["base.txt"],
            },
            "compatibility": {
                "ours": [],
                "upstream": [],
                "candidate": [],
                "assessment": "compatible",
                "conditions": [],
                "evidence": ["base.txt"],
            },
            "adaptationCost": {
                "effort": "low",
                "reversible": True,
                "nextMergeRisk": "low",
                "rationale": "fixture",
            },
            "reworkEffect": "same",
            "decision": "hybrid",
            "reason": "fixture",
            "evidence": ["base.txt"],
            "commitRef": "pending:fixture",
        }],
        "hybridContract": {
            "id": "fixture-hybrid",
            "matrixIds": ["fixture"],
            "boundary": "fixture",
            "inputs": "fixture",
            "outputs": "fixture",
            "ownership": "fixture",
            "streamOrder": "fixture",
            "fallback": "fixture",
            "responsibilitySplit": {"ours": "fixture", "upstream": "fixture"},
            "dataFlow": "fixture",
            "layout": "fixture",
            "units": "fixture",
            "invariants": ["fixture"],
            "targets": ["fixture"],
            "extensionPoint": "fixture",
            "errorIsolation": "fixture",
            "separateRuns": [
                {"implementation": name, "revision": ref, "status": "passed", "evidence": ["base.txt"]}
                for name, ref in (("upstream-only", "upstream"), ("ours-only", "ours"), ("hybrid", candidate))
            ],
            "adaptationCost": {
                "effort": "low",
                "reversible": True,
                "nextMergeRisk": "low",
                "rationale": "fixture",
            },
            "rollback": "fixture",
        },
        "invariantsChanged": [],
        "evidence": evidence,
        "checks": [],
        "warningAcknowledgments": [],
        "thresholds": {
            "maxChangedFiles": 60,
            "maxChangedLines": 3000,
            "maxIfdefAdded": 25,
            "maxInvariantsChanged": 1,
        },
        "architectureReview": {
            "status": "completed",
            "scope": ["fixture"],
            "author": "Plan Author",
            "questions": questions,
            "alternatives": [{"name": "fixture", "reasonRejected": "fixture"}],
            "reviewer": "Independent Reviewer",
            "independent": True,
            "date": "2026-11-24",
            "verdict": "approved",
            "signature": "fixture",
            "artifact": "base.txt",
        },
        "acceptance": {"status": "pending", "gateResult": "not-run", "notes": "fixture"},
    }


class UpstreamMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="upstream-merge-test-"))
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.name", "Merge Test")
        git(self.repo, "config", "user.email", "merge@example.invalid")
        (self.repo / "base.txt").write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "base.txt")
        git(self.repo, "commit", "-q", "-m", "base")
        git(self.repo, "checkout", "-q", "-b", "ours")
        (self.repo / "ours.txt").write_text("ours\n", encoding="utf-8")
        git(self.repo, "add", "ours.txt")
        git(self.repo, "commit", "-q", "-m", "ours")
        git(self.repo, "checkout", "-q", "-b", "upstream", "main")
        (self.repo / "upstream.txt").write_text("upstream\n", encoding="utf-8")
        git(self.repo, "add", "upstream.txt")
        git(self.repo, "commit", "-q", "-m", "upstream")
        git(self.repo, "checkout", "-q", "ours")
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps({
            "patches": [{
                "id": "ours-file",
                "oursAnchor": "ours.txt",
                "upstreamAnchor": "upstream.txt",
            }]
        }), encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_preflight(self) -> None:
        code, data, _ = run_cli(self.repo, "preflight", "--ours", "ours", "--upstream", "upstream", "--format", "json")
        self.assertEqual(code, 0)
        self.assertEqual(data["aheadBehind"], {"oursOnly": 1, "upstreamOnly": 1})
        self.assertFalse(data["dirtyTree"])

    def test_matrix(self) -> None:
        out = self.root / "matrix.json"
        code, data, _ = run_cli(
            self.repo, "matrix", "--profile", str(self.profile), "--ours", "ours",
            "--upstream", "upstream", "--out", str(out), "--format", "json",
        )
        self.assertEqual(code, 3)
        self.assertTrue(out.is_file())
        self.assertTrue(data["matrix"][0]["inOurs"])
        self.assertTrue(data["matrix"][0]["inUpstream"])
        self.assertIsNone(data["matrix"][0]["upstreamStatus"])
        self.assertFalse(data["matrix"][0]["ready"])
        self.assertEqual(data["result"], "review-required")

    def test_matrix_with_explicit_review(self) -> None:
        self.profile.write_text(json.dumps({
            "patches": [{
                "id": "ours-file",
                "oursAnchor": "ours.txt",
                "upstreamAnchor": "upstream.txt",
                "upstreamStatus": "full",
                "statusReviewed": True,
                "statusEvidence": ["review.md"],
            }]
        }), encoding="utf-8")
        code, data, _ = run_cli(
            self.repo, "matrix", "--profile", str(self.profile), "--ours", "ours",
            "--upstream", "upstream", "--format", "json",
        )
        self.assertEqual(code, 0)
        self.assertTrue(data["readyForPlan"])
        self.assertEqual(data["matrix"][0]["upstreamStatus"], "full")

    def test_prepare_dry_run(self) -> None:
        code, _, _ = run_cli(self.repo, "prepare", "--ours", "ours", "--upstream", "upstream", "--branch", "sync/test")
        self.assertEqual(code, 3)
        self.assertEqual(git(self.repo, "branch", "--show-current"), "ours")
        self.assertFalse((self.repo / ".git" / "refs" / "heads" / "sync" / "test").exists())

    def test_prepare_and_abort(self) -> None:
        code, data, _ = run_cli(
            self.repo, "prepare", "--ours", "ours", "--upstream", "upstream",
            "--branch", "sync/test", "--backup-tag", "pre-test", "--yes",
        )
        self.assertEqual(code, 3)
        self.assertEqual(data["result"], "merge-in-progress")
        self.assertEqual(git(self.repo, "branch", "--show-current"), "sync/test")
        self.assertTrue((self.repo / ".git" / "MERGE_HEAD").is_file())
        self.assertEqual(git(self.repo, "tag", "--list", "pre-test"), "pre-test")
        code, data, _ = run_cli(self.repo, "status")
        self.assertEqual(code, 0)
        self.assertIn("MERGE_HEAD", data["activeState"])
        code, _, _ = run_cli(self.repo, "abort", "--yes")
        self.assertEqual(code, 0)
        self.assertFalse((self.repo / ".git" / "MERGE_HEAD").exists())

    def test_experiment_dry_run_and_create(self) -> None:
        code, _, _ = run_cli(self.repo, "experiment", "--branch", "opt/test", "--base", "ours")
        self.assertEqual(code, 3)
        self.assertEqual(git(self.repo, "branch", "--show-current"), "ours")
        code, data, _ = run_cli(self.repo, "experiment", "--branch", "opt/test", "--base", "ours", "--yes")
        self.assertEqual(code, 0)
        self.assertEqual(data["result"], "experiment-created")
        self.assertEqual(git(self.repo, "branch", "--show-current"), "opt/test")

    def test_promote_requires_push_for_local_delete(self) -> None:
        code, data, _ = run_cli(
            self.repo, "promote", "--branch", "ours", "--plan", str(self.profile),
            "--delete-local", "--yes",
        )
        self.assertEqual(code, 2)
        self.assertEqual(data["result"], "invalid")

    def test_promote_requires_report_out(self) -> None:
        code, data, _ = run_cli(
            self.repo, "promote", "--branch", "ours", "--plan", str(self.profile), "--yes",
        )
        self.assertEqual(code, 2)
        self.assertIn("report-out", data["error"])

    def test_promote_rejects_title_without_area(self) -> None:
        code, data, _ = run_cli(
            self.repo, "promote", "--branch", "ours", "--plan", str(self.profile),
            "--report-out", str(self.root / "promote.json"), "--title", "Strata optimization", "--yes",
        )
        self.assertEqual(code, 2)
        self.assertIn("title", data["error"])

    def test_promote_success_includes_gate_report(self) -> None:
        git(self.repo, "switch", "-c", "opt/promote", "ours")
        git(self.repo, "merge", "-q", "--no-ff", "upstream", "-m", "candidate")
        plan = self.root / "merge-plan.json"
        plan.write_text(json.dumps(valid_plan()), encoding="utf-8")
        report_out = self.root / "promote-report.json"
        code, data, _ = run_cli(
            self.repo, "promote", "--branch", "opt/promote", "--plan", str(plan),
            "--report-out", str(report_out), "--yes",
        )
        self.assertEqual(code, 0, data)
        self.assertEqual(data["result"], "promoted")
        self.assertIn("gate", data)
        self.assertTrue(report_out.is_file())
        self.assertEqual(git(self.repo, "rev-parse", "main"), git(self.repo, "rev-parse", "opt/promote"))

    def test_promote_early_error_writes_report(self) -> None:
        plan = self.root / "invalid-plan.json"
        plan.write_text("{}", encoding="utf-8")
        report_out = self.root / "invalid-promote-report.json"
        code, data, _ = run_cli(
            self.repo, "promote", "--branch", "ours", "--plan", str(plan),
            "--report-out", str(report_out), "--yes",
        )
        self.assertEqual(code, 2)
        self.assertEqual(data["result"], "invalid")
        self.assertTrue(report_out.is_file())

    def test_prepare_recovery_cleans_created_refs(self) -> None:
        cases = (
            ("detached", "sync/detached", "bad..detached", "HEAD"),
            ("attached", "sync/attached", "bad..attached", "upstream"),
        )
        for label, branch, backup_tag, expected_ref in cases:
            with self.subTest(case=label):
                original_head = git(self.repo, "rev-parse", "upstream")
                if label == "detached":
                    git(self.repo, "checkout", "-q", "--detach", "upstream")
                else:
                    git(self.repo, "checkout", "-q", "upstream")
                code, data, _ = run_cli(
                    self.repo, "prepare", "--ours", "ours", "--upstream", "upstream",
                    "--branch", branch, "--backup-tag", backup_tag, "--yes",
                )
                self.assertEqual(code, 2, data)
                self.assertEqual(git(self.repo, "rev-parse", "HEAD"), original_head)
                self.assertEqual(git(self.repo, "rev-parse", "--abbrev-ref", "HEAD"), expected_ref)
                self.assertEqual(git(self.repo, "branch", "--list", branch), "")
                self.assertEqual(git(self.repo, "tag", "--list", backup_tag), "")
                self.assertFalse((self.repo / ".git" / "MERGE_HEAD").exists())
                self.assertNotIn("cleanup:", data["error"])

    def test_promote_report_write_failure_is_blocking(self) -> None:
        plan = self.root / "invalid-plan.json"
        plan.write_text("{}", encoding="utf-8")
        code, data, _ = run_cli(
            self.repo, "promote", "--branch", "ours", "--plan", str(plan),
            "--report-out", str(self.root), "--yes",
        )
        self.assertEqual(code, 1)
        self.assertIn("reportWriteError", data)

    def test_invalid_profile_and_schema(self) -> None:
        self.profile.write_text("{}", encoding="utf-8")
        code, data, _ = run_cli(self.repo, "matrix", "--profile", str(self.profile), "--format", "json")
        self.assertEqual(code, 2)
        self.assertEqual(data["result"], "invalid")

        def assert_invalid(mutate, label: str, expected: str) -> None:
            plan = valid_plan()
            mutate(plan)
            path = self.root / f"invalid-{label}.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            code, data, _ = run_cli(self.repo, "check", "--plan", str(path), "--format", "json")
            self.assertEqual(code, 2, data)
            self.assertEqual(data["result"], "invalid")
            self.assertIn(expected, data["error"])

        with self.subTest(case="all-14-evidence"):
            assert_invalid(lambda plan: plan["evidence"].pop("rollback"), "evidence", "rollback")

        with self.subTest(case="anchor-method"):
            assert_invalid(lambda plan: plan["matrix"][0].pop("anchorMethod"), "anchor-method", "anchorMethod")

        with self.subTest(case="anchor-evidence"):
            assert_invalid(lambda plan: plan["matrix"][0].pop("anchorEvidence"), "anchor-evidence", "anchorEvidence")

        with self.subTest(case="anchor-missing"):
            assert_invalid(lambda plan: plan["matrix"][0].update(anchorStatus="missing"), "anchor-missing", "anchorStatus")

        with self.subTest(case="combination-status"):
            assert_invalid(
                lambda plan: plan["evidence"]["implementationCombinations"]["combinations"][0].update(status="failed"),
                "combination-status",
                "status",
            )

        with self.subTest(case="combination-revision"):
            assert_invalid(
                lambda plan: plan["evidence"]["implementationCombinations"]["combinations"][0].pop("revision"),
                "combination-revision",
                "revision",
            )

        with self.subTest(case="combination-evidence"):
            assert_invalid(
                lambda plan: plan["evidence"]["implementationCombinations"]["combinations"][0].update(evidence=[]),
                "combination-evidence",
                "evidence",
            )

        def partial_drop(plan: dict) -> None:
            plan["matrix"][0].update({
                "decision": "drop",
                "upstreamStatus": "partial",
                "anchorStatus": "removed",
                "replacementUpstreamAnchor": "base.txt",
                "fullReplacementEvidence": ["base.txt"],
            })

        with self.subTest(case="removed-without-full-replacement"):
            assert_invalid(partial_drop, "removed-partial", "upstreamStatus")

        def removed_upstream(plan: dict) -> None:
            plan["matrix"][0].update({
                "decision": "upstream",
                "upstreamStatus": "full",
                "anchorStatus": "removed",
                "replacementUpstreamAnchor": "base.txt",
                "fullReplacementEvidence": ["base.txt"],
            })

        with self.subTest(case="removed-without-drop"):
            assert_invalid(removed_upstream, "removed-upstream", "decision")

        def removed_without_proof(plan: dict) -> None:
            plan["matrix"][0].update({
                "decision": "drop",
                "upstreamStatus": "full",
                "anchorStatus": "removed",
                "replacementUpstreamAnchor": "base.txt",
            })

        with self.subTest(case="removed-without-proof"):
            assert_invalid(removed_without_proof, "removed-no-proof", "fullReplacementEvidence")


if __name__ == "__main__":
    unittest.main()
