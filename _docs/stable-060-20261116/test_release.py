"""No-build boundary tests. These tests do not touch release/resource folders."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("release060", Path(__file__).with_name("release.py"))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class Boundaries(unittest.TestCase):
    def test_unit_only_tauri_override_does_not_cross_production_boundary(self) -> None:
        override = json.dumps({"bundle": {"resources": {}}})
        with patch.dict(os.environ, {"TAURI_CONFIG": override, "PATH": "test-only-path"}, clear=True):
            child_env = release.worker_environment(
                Path("H:/Strata/releases/test-only-no-write"), "test-token"
            )
            self.assertFalse("TAURI_CONFIG" in child_env, "Production child inherited the unit-only override")
            self.assertEqual(os.environ["TAURI_CONFIG"], override, "Caller environment must be preserved")
            self.assertEqual(child_env["CARGO_BUILD_JOBS"], "5")
            self.assertEqual(child_env["CARGO_NET_OFFLINE"], "true")

    def test_every_default_phase_is_non_mutating(self) -> None:
        for phase in release.PHASES:
            with self.subTest(phase=phase), patch.object(sys, "argv", ["release.py", phase]), patch.object(
                release, "reviewed", side_effect=AssertionError("execute reached")
            ), patch.object(Path, "mkdir", side_effect=AssertionError("mutation reached")), patch.object(
                release, "command", side_effect=AssertionError("child reached")
            ):
                self.assertEqual(release.main(), 0)

    def test_traversal_attempt_rejected_before_mutation(self) -> None:
        with patch.object(sys, "argv", ["release.py", "--attempt", "../r1"]), patch.object(
            Path, "mkdir", side_effect=AssertionError("mutation reached")
        ):
            with self.assertRaisesRegex(RuntimeError, "Attempt"):
                release.main()

    def test_worker_cannot_run_as_dry_run(self) -> None:
        with patch.object(sys, "argv", ["release.py", "--worker"]):
            with self.assertRaisesRegex(RuntimeError, "Worker requires execute"):
                release.main()

    def test_commit_must_be_full_sha(self) -> None:
        for commit in ("", "HEAD", "abcd", "0" * 39, "z" * 40):
            with self.subTest(commit=commit), patch.object(
                release, "git", side_effect=AssertionError("git reached")
            ):
                with self.assertRaisesRegex(RuntimeError, "full reviewed commit"):
                    release.reviewed(commit)

    def test_existing_payload_is_never_replaced(self) -> None:
        with patch.object(release, "no_links"), patch.object(Path, "exists", return_value=True), patch.object(
            release, "inventory", return_value={"python/python.exe": "x"}
        ), patch.object(Path, "mkdir", side_effect=AssertionError("mutation reached")):
            with self.assertRaisesRegex(RuntimeError, "Existing staged payload"):
                release.stage(Path("unused"), Path("unused-target"))

    def test_copy_refuses_existing_destination(self) -> None:
        with patch.object(release, "no_links"), patch.object(Path, "exists", return_value=True), patch.object(
            release, "inventory", return_value={"x": "hash"}
        ), patch.object(release.shutil, "copytree", side_effect=AssertionError("copy reached")):
            with self.assertRaisesRegex(RuntimeError, "Fresh copy destination"):
                release.copy_tree(Path("unused"), Path("unused-target"))


if __name__ == "__main__":
    unittest.main()
