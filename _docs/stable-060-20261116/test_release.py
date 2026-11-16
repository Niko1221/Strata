"""No-build boundary tests. These tests do not touch release/resource folders."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("release060", Path(__file__).with_name("release.py"))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class Boundaries(unittest.TestCase):
    def test_only_exact_checkout_branch_pairs_are_accepted(self) -> None:
        primary = Path("C:/Users/downl/Documents/ChatGPT/Strata")
        integration = Path("H:/Strata/worktrees/upstream-sync-20261116")
        with patch.dict(os.environ, {}, clear=True), patch.object(release, "no_links"), patch.object(
            Path, "resolve", autospec=True, side_effect=lambda path, **_kwargs: path
        ):
            release.validate_checkout(primary, "main")
            release.validate_checkout(integration, "codex/upstream-sync-20261116")
            for root, branch in (
                (primary, "codex/upstream-sync-20261116"),
                (integration, "main"),
                (Path("H:/Strata/worktrees/other"), "main"),
            ):
                with self.subTest(root=root, branch=branch), self.assertRaises(RuntimeError):
                    release.validate_checkout(root, branch)

    def test_checkout_alias_is_rejected(self) -> None:
        primary = Path("C:/Users/downl/Documents/ChatGPT/Strata")
        with patch.dict(os.environ, {}, clear=True), patch.object(release, "no_links"), patch.object(
            Path, "resolve", return_value=Path("H:/Strata/worktrees/other")
        ):
            with self.assertRaisesRegex(RuntimeError, "alias"):
                release.validate_checkout(primary, "main")

    def test_protected_sibling_key_allowed_but_source_and_release_keys_rejected(self) -> None:
        sibling = Path("H:/Strata/private-keys/synthetic/feed-key.json")
        forbidden = [
            Path("C:/Users/downl/Documents/ChatGPT/Strata/key.json"),
            Path("H:/Strata/worktrees/another/key.json"),
            Path("H:/Strata/releases/another/key.json"),
        ]
        with patch.dict(os.environ, {}, clear=True), patch.object(release, "no_links"), patch.object(
            Path, "resolve", autospec=True, side_effect=lambda path, **_kwargs: path
        ), patch.object(Path, "is_file", return_value=True), patch.object(
            Path, "open", side_effect=AssertionError("Key content must not be opened")
        ):
            self.assertEqual(release.signing_key_path(sibling), sibling)
            for key in forbidden:
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "outside source"):
                    release.signing_key_path(key)

    def test_linked_key_is_rejected_before_resolution_or_read(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            release, "no_links", side_effect=RuntimeError("Reparse path refused")
        ), patch.object(Path, "resolve", side_effect=AssertionError("Alias resolved too soon")):
            with self.assertRaisesRegex(RuntimeError, "Reparse"):
                release.signing_key_path(Path("H:/Strata/private-keys/synthetic/key.json"))

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


class PreparedInputs(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.output = patch.object(release, "OUT", self.root)
        self.output.start()
        self.addCleanup(self.output.stop)
        self.assets = self.root / "inputs/public-assets"
        self.assets.mkdir(parents=True)
        (self.assets / "icon.svg").write_text("synthetic icon", encoding="utf-8")
        self.receipt = self.root / "inputs/provenance.json"
        self.record = {
            "lucide_commit": "03f11e2d6adc319ea21e9a1d5820a99cd3767257",
            "files": {"icon.svg": release.sha(self.assets / "icon.svg")},
        }
        self.receipt.write_text(json.dumps(self.record), encoding="utf-8")
        self.args = release.arguments(
            [
                "dependencies",
                "--reviewed-commit",
                "a" * 40,
                "--public-assets",
                str(self.assets),
                "--public-assets-receipt",
                str(self.receipt),
                "--public-assets-receipt-sha256",
                release.sha(self.receipt),
            ]
        )

    def test_cli_identity_survives_parent_worker_and_node_boundary(self) -> None:
        worker = release.arguments(release.worker_command(self.args)[3:])
        self.assertTrue(worker.execute and worker.worker)
        self.assertEqual(release.public_asset_identity(worker), release.public_asset_identity(self.args))
        with patch.object(release, "command") as command:
            release.verify("dependencies", self.root, worker)
        self.assertEqual(
            command.call_args.args[0][-3:],
            [str(self.assets), str(self.receipt), self.args.public_assets_receipt_sha256],
        )

    def test_required_input_and_wrong_receipt_hash_refused(self) -> None:
        for phase in ("dependencies", "component", "app"):
            with self.subTest(phase=phase), self.assertRaisesRegex(RuntimeError, "Explicit public"):
                release.public_asset_identity(release.arguments([phase]))
        self.args.public_assets_receipt_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            release.public_asset_identity(self.args)

    def test_node_verifier_checks_explicit_receipt_and_actual_bytes(self) -> None:
        command = [
            str(release.NODE),
            str(release.HERE / "verify.mjs"),
            "public-assets",
            str(release.ROOT),
            str(self.root),
            "a" * 40,
            "-",
            str(self.assets),
            str(self.receipt),
            self.args.public_assets_receipt_sha256,
        ]
        # Windows Node needs its OS root for platform initialisation; no caller environment is inherited.
        child_env = {"SYSTEMROOT": "C:\\Windows"}
        valid = subprocess.run(command, env=child_env, capture_output=True, timeout=30)
        self.assertEqual(valid.returncode, 0, "Synthetic Node asset verification failed")
        wrong_hash = command[:-1] + ["0" * 64]
        refused = subprocess.run(wrong_hash, env=child_env, capture_output=True, timeout=30)
        self.assertNotEqual(refused.returncode, 0, "Node accepted a different receipt")
        (self.assets / "icon.svg").write_text("changed", encoding="utf-8")
        changed = subprocess.run(command, env=child_env, capture_output=True, timeout=30)
        self.assertNotEqual(changed.returncode, 0, "Node accepted changed asset bytes")

    def test_outside_and_reparse_inputs_refused(self) -> None:
        self.args.public_assets = self.root
        with self.assertRaisesRegex(RuntimeError, "inside release inputs"):
            release.public_asset_identity(self.args)
        self.args.public_assets = self.assets
        with patch.object(release, "no_links", side_effect=RuntimeError("Reparse path refused")):
            with self.assertRaisesRegex(RuntimeError, "Reparse"):
                release.public_asset_identity(self.args)

    def test_traversal_member_refused_even_with_matching_receipt_hash(self) -> None:
        self.record["files"] = {"../outside": "a" * 64}
        self.receipt.write_text(json.dumps(self.record), encoding="utf-8")
        self.args.public_assets_receipt_sha256 = release.sha(self.receipt)
        with self.assertRaisesRegex(RuntimeError, "Unsafe public asset"):
            release.public_asset_identity(self.args)

    def test_reserved_asset_byte_membership_and_receipt_changes_refused(self) -> None:
        identity = release.public_asset_identity(self.args)
        icon = self.assets / "icon.svg"
        icon.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "bytes changed"):
            release.unchanged_public_assets(self.args, identity)
        icon.write_text("synthetic icon", encoding="utf-8")
        extra = self.assets / "extra.svg"
        extra.write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "membership"):
            release.unchanged_public_assets(self.args, identity)
        extra.unlink()
        self.receipt.write_text(json.dumps(self.record) + "\n", encoding="utf-8")
        self.args.public_assets_receipt_sha256 = release.sha(self.receipt)
        with self.assertRaisesRegex(RuntimeError, "Reserved public"):
            release.unchanged_public_assets(self.args, identity)

    def resource_fixture(self) -> Path:
        resources = self.root / "resources"
        for name in ("harness", "local-backends"):
            folder = resources / name
            folder.mkdir(parents=True)
            (folder / "payload").write_text(name, encoding="utf-8")
        return resources

    def test_inplace_verifies_twice_without_staging_and_records_same_inventory(self) -> None:
        resources = self.resource_fixture()
        with patch.object(release, "PRIMARY_RES", resources), patch.object(
            release, "RES", resources
        ), patch.object(release, "verify") as verify, patch.object(
            release, "stage", side_effect=AssertionError("In-place copy attempted")
        ):
            result = release.dependencies(self.args, self.root)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(result["mode"], "verified-in-place")
        self.assertFalse(result["copied"])
        self.assertEqual(result["sourceFiles"], result["stagedFiles"])

    def test_inplace_detects_drift_during_second_verification(self) -> None:
        resources = self.resource_fixture()
        calls = []

        def verify(*_args) -> None:
            calls.append(True)
            if len(calls) == 2:
                (resources / "harness/payload").write_text("drift", encoding="utf-8")

        with patch.object(release, "PRIMARY_RES", resources), patch.object(
            release, "RES", resources
        ), patch.object(release, "verify", side_effect=verify):
            with self.assertRaisesRegex(RuntimeError, "Source dependency drift"):
                release.dependencies(self.args, self.root)

    def test_separate_existing_payload_still_refused(self) -> None:
        resources = self.resource_fixture()
        target = self.root / "target"
        (target / "harness").mkdir(parents=True)
        (target / "harness/payload").write_text("preserve", encoding="utf-8")
        with patch.object(release, "PRIMARY_RES", resources), patch.object(
            release, "RES", target
        ), patch.object(release, "verify"):
            with self.assertRaisesRegex(RuntimeError, "Existing staged payload"):
                release.dependencies(self.args, self.root)
        self.assertEqual((target / "harness/payload").read_text(), "preserve")

    def test_host_cache_requires_exact_node_modules_root_and_vite_version(self) -> None:
        expected = Path("H:/Strata/evidence/upstream-sync-20261116/host-deps-060/node_modules")
        with patch.object(Path, "resolve", return_value=expected), patch.object(
            release, "no_links"
        ), patch.object(release, "load", return_value={"version": "6.4.6"}):
            self.assertEqual(release.host_dependency_root(), expected)
        with patch.object(Path, "resolve", return_value=expected.parent):
            with self.assertRaisesRegex(RuntimeError, "Unexpected Host"):
                release.host_dependency_root()
        with patch.object(Path, "resolve", return_value=expected), patch.object(
            release, "no_links"
        ), patch.object(release, "load", return_value={"version": "6.4.3"}):
            with self.assertRaisesRegex(RuntimeError, "Vite 6.4.6"):
                release.host_dependency_root()


class PreservedNotices(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "source"
        self.resources = self.source / "resources"
        self.work = self.base / "notices-r1"
        self.prior = self.base / "inputs/notices"
        self.prior.mkdir(parents=True)
        self.work.mkdir()
        for context in (
            patch.dict(os.environ, {}, clear=True),
            patch.object(release, "ROOT", self.source),
            patch.object(release, "OUT", self.base),
            patch.object(release, "RES", self.resources),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.vendor = []
        for name in ("Microsoft-One.docx", "Microsoft-Two.docx", "Microsoft-Three.docx"):
            self.write(self.prior / name, "synthetic " + name)
            self.vendor.append({"file": name, "sha256": release.sha(self.prior / name)})
        self.write(self.source / "packaging/microsoft-2022-notice-sources.json", json.dumps(self.vendor))
        self.record = {"schema": 1, "files": {v["file"]: v["sha256"] for v in self.vendor}}
        self.write(self.prior / ".gitignore", "*\n")
        self.write(self.prior / "NOTICES.json", json.dumps(self.record))
        self.args = release.arguments(
            [
                "notices",
                "--reviewed-commit",
                "a" * 40,
                "--prior-notices",
                str(self.prior),
                "--prior-notices-manifest-sha256",
                release.sha(self.prior / "NOTICES.json"),
                "--cuda-dir",
                str(self.base / "cuda"),
            ]
        )

    def write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def repin_manifest(self) -> None:
        self.write(self.prior / "NOTICES.json", json.dumps(self.record))
        self.args.prior_notices_manifest_sha256 = release.sha(self.prior / "NOTICES.json")

    def prepare_current_licenses(self) -> Path:
        for path in (
            self.source / "LICENSE",
            self.source / "third_party/strata-ggml/LICENSE",
            self.source / "third_party/libstrix/NOTICE.md",
            self.source / "packaging/hakua-third-party-notices.md",
            self.args.cuda_dir / "EULA.txt",
            self.args.cuda_dir / "LICENSE",
            self.resources / "node/LICENSE",
            self.resources / "control/python/LICENSE",
        ):
            self.write(path, "current synthetic licence " + path.name)
        qwen = self.source / "data/experimental-speed-projection/LICENSE"
        self.write(qwen, "synthetic community licence")
        self.write(qwen.with_name("PROVENANCE.json"), json.dumps({"license": {"sha256": release.sha(qwen)}}))
        deps = self.base / "host-deps"
        for name in ("react", "react-dom", "scheduler", "katex", "prismjs", "pdfjs-dist", "html-to-image"):
            self.write(deps / name / "LICENSE", "synthetic " + name)
        return deps

    def test_preserved_input_and_fresh_target_complete_without_old_primary_notices(self) -> None:
        deps = self.prepare_current_licenses()
        identity = release.prior_notice_identity(self.args, self.work)
        with patch.object(release, "host_dependency_root", return_value=deps):
            result = release.notices(self.args, self.work)
        self.assertEqual(result["priorNotices"], identity)
        self.assertEqual(release.prior_notice_identity(self.args, self.work), identity)
        self.assertIn("Node-LICENSE.txt", result["noticeFiles"])
        self.assertEqual(
            (self.resources / "notices/Microsoft-One.docx").read_bytes(),
            (self.prior / "Microsoft-One.docx").read_bytes(),
        )

    def test_occupied_target_still_refused_and_preserved(self) -> None:
        deps = self.prepare_current_licenses()
        target = self.resources / "notices/old-payload"
        self.write(target, "preserve this")
        with patch.object(release, "host_dependency_root", return_value=deps):
            with self.assertRaisesRegex(RuntimeError, "Existing staged payload"):
                release.notices(self.args, self.work)
        self.assertEqual(target.read_text(), "preserve this")

    def test_prior_input_required_only_for_notices_unless_explicitly_passed(self) -> None:
        for phase in release.PHASES:
            args = release.arguments([phase])
            if phase == "notices":
                with self.assertRaisesRegex(RuntimeError, "Explicit prior notices"):
                    release.prior_notice_identity(args, self.work)
            else:
                with patch.object(Path, "resolve", side_effect=AssertionError("Input examined")):
                    self.assertIsNone(release.prior_notice_identity(args, self.work))

    def test_worker_receives_path_and_full_identity_including_scaffold(self) -> None:
        worker = release.arguments(release.worker_command(self.args)[3:])
        identity = release.prior_notice_identity(worker, self.work)
        self.assertEqual(identity, release.prior_notice_identity(self.args, self.work))
        self.assertIn(".gitignore", identity["files"])
        self.assertIn("NOTICES.json", identity["files"])
        self.assertEqual(len(identity["files"]), 5)

    def test_manifest_wrong_hash_and_traversal_rejected(self) -> None:
        self.args.prior_notices_manifest_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "manifest SHA-256 mismatch"):
            release.prior_notice_identity(self.args, self.work)
        self.record["files"]["../outside"] = "a" * 64
        self.repin_manifest()
        with self.assertRaisesRegex(RuntimeError, "Unsafe prior notice"):
            release.prior_notice_identity(self.args, self.work)

    def test_membership_and_payload_changes_rejected(self) -> None:
        extra = self.prior / "extra"
        self.write(extra, "extra")
        with self.assertRaisesRegex(RuntimeError, "membership or bytes"):
            release.prior_notice_identity(self.args, self.work)
        extra.unlink()
        self.write(self.prior / "Microsoft-One.docx", "changed")
        with self.assertRaisesRegex(RuntimeError, "membership or bytes"):
            release.prior_notice_identity(self.args, self.work)

    def test_vendor_change_rejected_even_with_new_manifest_pin(self) -> None:
        self.write(self.prior / "Microsoft-One.docx", "changed")
        self.record["files"]["Microsoft-One.docx"] = release.sha(self.prior / "Microsoft-One.docx")
        self.repin_manifest()
        with self.assertRaisesRegex(RuntimeError, "vendor notice pin mismatch"):
            release.prior_notice_identity(self.args, self.work)

    def test_reserved_scaffold_change_rejected(self) -> None:
        identity = release.prior_notice_identity(self.args, self.work)
        self.write(self.prior / ".gitignore", "changed scaffold")
        with self.assertRaisesRegex(RuntimeError, "Reserved prior notices changed"):
            release.unchanged_prior_notices(self.args, self.work, identity)

    def test_target_attempt_overlap_outside_and_reparse_inputs_rejected(self) -> None:
        for resource in (self.prior.parent, self.prior / "child"):
            with patch.object(release, "RES", resource):
                with self.assertRaisesRegex(RuntimeError, "overlap"):
                    release.prior_notice_identity(self.args, self.work)
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            release.prior_notice_identity(self.args, self.prior)
        self.args.prior_notices = self.base
        with self.assertRaisesRegex(RuntimeError, "inside release inputs"):
            release.prior_notice_identity(self.args, self.work)
        self.args.prior_notices = self.prior
        with patch.object(release, "no_links", side_effect=RuntimeError("Reparse path refused")):
            with self.assertRaisesRegex(RuntimeError, "Reparse"):
                release.prior_notice_identity(self.args, self.work)


if __name__ == "__main__":
    unittest.main()
