"""Synthetic boundaries only: no real profile, installer, model or network."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True


def module(name: str):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), Path(__file__).with_name(name + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def good_job() -> dict:
    return {
        "status": "FINISHED",
        "exit_code": 0,
        "native_exit_code": 0,
        "contained_before_resume": True,
        "termination_confirmed": True,
        "job_active_after": 0,
    }


class Boundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.delivery = module("delivery")
        self.release = self.delivery.release
        self.normal = module("normal-start")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.release.OUT = self.root / "release"
        self.release.OUT.mkdir()
        self.delivery.APP = self.root / "installed"
        self.delivery.PRIVATE = self.root / "private"
        self.delivery.USER_ROOTS = (self.root / "user", self.root / "absent")
        self.windows_root = os.environ.get("SYSTEMROOT", "C:/Windows")
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.commit = "a" * 40

    def write(self, path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def record(self, phase: str, extra: dict | None = None) -> tuple[Path, str]:
        path = self.release.OUT / (phase + "-r1") / "receipt.json"
        self.write(
            path,
            {
                "ok": True,
                "phase": phase,
                "sourceCommit": self.commit,
                "sourceRoot": str(self.delivery.native.PRIMARY),
                "releaseQualified": False,
                **(extra or {}),
            },
        )
        self.write(path.parent / "owned-job.json", good_job())
        return path, self.release.sha(path)

    def fixture_payload(self, root: Path) -> dict:
        root.mkdir(parents=True)
        (root / "strata-desktop.exe").write_bytes(b"synthetic shell")
        for name in ("control", "host", "node", "harness", "notices"):
            (root / name).mkdir()
            (root / name / "fixture.txt").write_bytes(b"fixture")
        return self.delivery.payload(root)

    def fixture_control_data(self):
        """Small synthetic tracked blobs; never inspect the real APP or source data."""
        self.release.ROOT = self.root / "trusted-source"
        for name in self.delivery.SHIPPED_CONTROL_DATA:
            data = ("fixture " + name).encode()
            for path in (self.delivery.APP / name, self.release.ROOT / name.removeprefix("control/")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        vector_name = next(
            n for n, role in self.delivery.SHIPPED_CONTROL_DATA.items() if role == "controlvector"
        )
        folder = Path(vector_name).parent
        vector = self.delivery.APP / vector_name
        licence = b"synthetic fixture licence"
        document = {
            "schema": 1,
            "vector": {
                "file": vector.name,
                "kind": "controlvector",
                "default": "off",
                "bytes": vector.stat().st_size,
                "sha256": self.release.sha(vector),
            },
            "license": {
                "file": "LICENSE",
                "name": "Qwen Community License 1.0",
                "bytes": len(licence),
                "sha256": hashlib.sha256(licence).hexdigest(),
            },
        }
        for base in (self.delivery.APP / folder, self.release.ROOT / folder.relative_to("control")):
            (base / "LICENSE").write_bytes(licence)
            self.write(base / "PROVENANCE.json", document)
        blobs = {}
        tracked = {}
        for path in self.release.ROOT.rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                tracked[path.relative_to(self.release.ROOT).as_posix()] = blob
                blobs[blob] = len(data)

        def git(*args):
            if args[0] == "ls-files":
                return args[-1] if args[-1] in tracked else ""
            if args[0] == "rev-parse":
                return tracked[args[1].split(":", 1)[1]]
            if args[0] == "cat-file":
                return str(blobs[args[-1]])
            if args[0] == "hash-object":
                data = Path(args[-1]).read_bytes()
                return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            raise AssertionError(args)

        for name, replacement in (
            ("git", git),
            ("reviewed", lambda commit: self.assertEqual(commit, self.commit)),
        ):
            handle = patch.object(self.release, name, side_effect=replacement)
            handle.start()
            self.addCleanup(handle.stop)
        return tracked, git, vector_name

    def test_eleven_named_control_roles_copy_identical_bytes(self):
        self.fixture_control_data()
        admitted = self.delivery.reject_app_weights(self.delivery.APP, self.commit)
        self.assertEqual(len(admitted), 11)
        self.assertEqual(
            sorted(row["role"] for row in admitted.values()),
            sorted(["vocabulary"] * 4 + ["routing-profile"] * 6 + ["controlvector"]),
        )
        rows = self.delivery.files(self.delivery.APP)
        target = self.root / "copied"
        self.delivery.copy_rows(
            self.delivery.APP, target, rows, reject_weights=True, shipped_control_data=admitted
        )
        self.assertEqual(self.delivery.files(target), rows)
        for name, identity in admitted.items():
            self.assertEqual((target / name).read_bytes(), (self.delivery.APP / name).read_bytes())
            self.assertEqual(identity["sourceCommit"], self.commit)
        self.assertEqual(admitted[next(n for n in admitted if n.endswith(".gguf"))]["default"], "off")

    def test_synthetic_backup_declares_component_data_and_retains_originals(self):
        self.fixture_control_data()
        self.write(self.delivery.APP / "host/package.json", {"version": "0.5.9"})
        original = self.delivery.files(self.delivery.APP)
        with patch.object(
            self.delivery, "extraction", return_value=({"installerSHA256": "a" * 64}, {}, self.root)
        ), patch.object(self.delivery, "quiescent"), patch.object(
            self.delivery, "model_identity", return_value={"synthetic": True}
        ), patch.object(
            self.delivery, "private_directory", side_effect=lambda path: path.mkdir(parents=True)
        ), patch.object(
            self.release, "command", return_value=""
        ), patch.object(
            self.delivery.shutil, "disk_usage", return_value=argparse.Namespace(free=100 * self.release.GIB)
        ):
            receipt = self.delivery.backup(
                argparse.Namespace(attempt="r1", reviewed_commit=self.commit), self.root
            )
        self.assertFalse(receipt["modelsCopied"])
        self.assertIn("inference-model", receipt["modelsCopiedScope"])
        self.assertEqual(len(receipt["shippedControlDataCopied"]), 11)
        self.assertTrue(receipt["rollbackRetained"])
        self.assertEqual(self.delivery.files(Path(receipt["backup"]) / "installed-app"), original)
        self.assertEqual(self.delivery.files(self.delivery.APP), original)

    def test_allowlist_is_exact_named_roles_not_suffix_or_tracked_growth(self):
        tracked, _, _ = self.fixture_control_data()
        self.assertEqual(
            set(self.delivery.SHIPPED_CONTROL_DATA),
            {
                "control/data/" + name
                for name in (
                    "draft_vocab.bin",
                    "draft_vocab_cyrillic.bin",
                    "draft_vocab_en.bin",
                    "draft_vocab_fr.bin",
                    "expert-profile.bin",
                    "expert-profile-coder.bin",
                    "expert-profile-sycl.bin",
                    "dsv41/profile-2.00.bin",
                    "dsv41/profile-2.25.bin",
                    "dsv41/profile-coder.bin",
                    "experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf",
                )
            },
        )
        for name in ("control/data/user.gguf", "control/data/user.bin", "other/draft_vocab.bin"):
            with self.subTest(name=name):
                candidate = self.delivery.APP / name
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_bytes(b"private model")
                tracked[name.removeprefix("control/")] = "b" * 40
                with patch.object(
                    self.release, "sha", side_effect=AssertionError("candidate hash")
                ), patch.object(
                    self.release, "reviewed", side_effect=AssertionError("late metadata refusal")
                ):
                    with self.assertRaisesRegex(RuntimeError, "weights refused before backup"):
                        self.delivery.reject_app_weights(self.delivery.APP, self.commit)
                candidate.unlink()

    def test_known_wrong_size_refuses_before_hash_and_same_size_edit_refuses_after_bounded_hash(self):
        self.fixture_control_data()
        name = "control/data/draft_vocab.bin"
        actual = self.delivery.APP / name
        original = actual.read_bytes()
        actual.write_bytes(original + b"extra")
        with patch.object(self.release, "sha", side_effect=AssertionError("content hash")):
            with self.assertRaisesRegex(RuntimeError, "before hashing"):
                self.delivery.bounded_component_match(actual, name.removeprefix("control/"), self.commit)
        actual.write_bytes(b"X" * len(original))
        with patch.object(self.release, "sha", wraps=self.release.sha) as hashed:
            with self.assertRaisesRegex(RuntimeError, "bytes differ"):
                self.delivery.reject_app_weights(self.delivery.APP, self.commit)
            self.assertIn(actual, [call.args[0] for call in hashed.call_args_list])
        self.assertFalse((self.root / "copied").exists())

    def test_untracked_or_oversized_source_refused_before_content_hash(self):
        tracked, _, _ = self.fixture_control_data()
        relative = "data/draft_vocab.bin"
        del tracked[relative]
        with patch.object(self.release, "sha", side_effect=AssertionError("content hash")):
            with self.assertRaisesRegex(RuntimeError, "not Git tracked"):
                self.delivery.reject_app_weights(self.delivery.APP, self.commit)
        (self.release.ROOT / relative).write_bytes(b"x" * (self.delivery.SHIPPED_DATA_LIMIT + 1))
        with self.assertRaisesRegex(RuntimeError, "bounded size"):
            self.delivery.reject_app_weights(self.delivery.APP, self.commit)

    def test_vector_provenance_or_license_change_refused(self):
        _, _, name = self.fixture_control_data()
        folder = self.delivery.APP / Path(name).parent
        for member in ("PROVENANCE.json", "LICENSE"):
            with self.subTest(member=member):
                path = folder / member
                original = path.read_bytes()
                path.write_bytes(b"X" * len(original))
                with self.assertRaisesRegex(RuntimeError, "bytes differ"):
                    self.delivery.reject_app_weights(self.delivery.APP, self.commit)
                path.write_bytes(original)
        real_load = self.release.load
        for field, value in (
            ("kind", "inference-model"),
            ("default", "on"),
            ("bytes", 999),
            ("sha256", "0" * 64),
        ):

            def changed(path):
                document = real_load(path)
                document["vector"][field] = value
                return document

            with self.subTest(field=field), patch.object(self.release, "load", side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, "provenance differs"):
                    self.delivery.reject_app_weights(self.delivery.APP, self.commit)

    def test_component_reparse_and_source_drift_refused(self):
        _, git, _ = self.fixture_control_data()
        relative = "data/draft_vocab.bin"
        source = self.release.ROOT / relative
        original_no_links = self.release.no_links

        def reparse(path):
            if path == source:
                raise RuntimeError("Synthetic reparse refused")
            return original_no_links(path)

        with patch.object(self.release, "no_links", side_effect=reparse):
            with self.assertRaisesRegex(RuntimeError, "reparse"):
                self.delivery.reject_app_weights(self.delivery.APP, self.commit)
        checks = 0

        def drift(*args):
            nonlocal checks
            if args[0] == "hash-object" and Path(args[-1]) == source:
                checks += 1
                if checks == 2:
                    return "b" * 40
            return git(*args)

        with patch.object(self.release, "git", side_effect=drift):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                self.delivery.reject_app_weights(self.delivery.APP, self.commit)

    def test_default_controllers_never_write_spawn_or_fetch(self) -> None:
        for controller, argv in (
            (self.delivery, ["extract"]),
            (self.normal, []),
            (self.normal, ["--capture-original"]),
        ):
            with self.subTest(controller=controller.__name__), patch.object(
                Path, "mkdir", side_effect=AssertionError("write")
            ), patch.object(
                controller.release, "reviewed", side_effect=AssertionError("read execution state")
            ), patch(
                "subprocess.run", side_effect=AssertionError("spawn")
            ), patch(
                "urllib.request.urlopen", side_effect=AssertionError("network")
            ):
                self.assertEqual(controller.main(argv), 0)

    def test_execution_requires_explicit_attempt_before_app_io(self) -> None:
        with patch.object(self.delivery, "app_build", side_effect=AssertionError("artifact read")):
            with self.assertRaisesRegex(RuntimeError, "Explicit fresh"):
                self.delivery.main(["extract", "--execute", "--reviewed-commit", self.commit])

    def test_bundle_is_not_exposed(self) -> None:
        self.assertNotIn("bundle", self.delivery.PHASES)

    def test_receipt_requires_external_digest(self) -> None:
        path, checksum = self.record("app")
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            self.delivery.receipt(path, None, "app", self.commit)
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            self.delivery.receipt(path, "0" * 64, "app", self.commit)
        self.assertTrue(self.delivery.receipt(path, checksum, "app", self.commit)["ok"])

    def test_receipt_rejects_other_source_or_phase(self) -> None:
        path, checksum = self.record("app")
        with self.assertRaises(RuntimeError):
            self.delivery.receipt(path, checksum, "extract", self.commit)
        with self.assertRaisesRegex(RuntimeError, "source or status"):
            self.delivery.receipt(path, checksum, "app", "b" * 40)

    def test_owned_job_cannot_be_partial_or_failed(self) -> None:
        path, checksum = self.record("app")
        for key, value in (
            ("status", "TIMEOUT"),
            ("exit_code", 1),
            ("native_exit_code", 1),
            ("contained_before_resume", False),
            ("termination_confirmed", False),
            ("job_active_after", 1),
        ):
            with self.subTest(key=key):
                self.write(path.parent / "owned-job.json", good_job() | {key: value})
                with self.assertRaisesRegex(RuntimeError, "drained"):
                    self.delivery.receipt(path, checksum, "app", self.commit)

    def test_app_artifacts_must_belong_to_receipt_attempt(self) -> None:
        folder = self.release.OUT / "app-r1"
        folder.mkdir()
        installer = folder / self.delivery.INSTALLER
        installer.write_bytes(b"installer")
        shell = folder / "cargo-target/release/strata-desktop.exe"
        shell.parent.mkdir(parents=True)
        shell.write_bytes(b"shell")
        fields = {
            "installer": str(installer),
            "installerSHA256": self.release.sha(installer),
            "installerBytes": installer.stat().st_size,
            "shell": str(shell),
            "shellSHA256": self.release.sha(shell),
            "installed": False,
            "published": False,
        }
        path, checksum = self.record("app", fields)
        args = argparse.Namespace(app_receipt=path, app_receipt_sha256=checksum, reviewed_commit=self.commit)
        self.assertEqual(self.delivery.app_build(args)["installerSHA256"], self.release.sha(installer))
        shell.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            self.delivery.app_build(args)
        fields["shell"] = str(self.root / "elsewhere.exe")
        path, checksum = self.record("app", fields)
        args.app_receipt_sha256 = checksum
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            self.delivery.app_build(args)

    def test_user_inventory_preserves_missing_roots_and_binary_preferences(self) -> None:
        root = self.delivery.USER_ROOTS[0]
        root.mkdir()
        (root / "private.bin").write_bytes(b"private binary state")
        (root / "model.gguf").write_bytes(b"weights")
        (root / "models").mkdir()
        (root / "models/pytorch_model.bin").write_bytes(b"weights")
        rows = self.delivery.user_files()
        self.assertEqual(set(rows[str(root)]), {"private.bin"})
        self.assertIsNone(rows[str(self.delivery.USER_ROOTS[1])])

    def test_copy_retains_previous_backup_and_detects_corruption(self) -> None:
        source, target = self.root / "src", self.root / "backup"
        source.mkdir()
        (source / "data").write_bytes(b"private fixture")
        rows = self.delivery.files(source)
        self.delivery.copy_rows(source, target, rows)
        with self.assertRaisesRegex(RuntimeError, "Fresh copy"):
            self.delivery.copy_rows(source, target, rows)
        self.assertEqual((target / "data").read_bytes(), b"private fixture")
        (source / "data").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "copy differs"):
            self.delivery.copy_rows(source, self.root / "new-attempt", rows)
        self.assertTrue((self.root / "new-attempt/data").exists())

    def test_copy_refuses_path_escape(self) -> None:
        source = self.root / "src"
        source.mkdir()
        for index, name in enumerate(
            (
                "../escape",
                "C:/escape",
                "/escape",
                "C:escape",
                "//server/share/escape",
                "data:ads",
                "x/../../escape",
            )
        ):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Unsafe"):
                self.delivery.copy_rows(
                    source, self.root / f"dst-{index}", {name: {"sha256": "0" * 64, "bytes": 0}}
                )
            self.assertFalse((self.root / f"dst-{index}").exists())
        self.assertFalse((self.root / "escape").exists())

    def test_inventory_rejects_reparse_before_hashing(self) -> None:
        root = self.root / "linked"
        root.mkdir()
        with patch.object(
            self.release, "no_links", side_effect=RuntimeError("Reparse refused")
        ), patch.object(self.release, "sha", side_effect=AssertionError("hash")):
            with self.assertRaisesRegex(RuntimeError, "Reparse"):
                self.delivery.files(root)

    def test_quiescence_rejects_installed_owner_without_stopping(self) -> None:
        process = Mock(pid=54321, info={"name": "strata-desktop.exe"})
        process.exe.return_value = str(self.delivery.APP / "strata-desktop.exe")
        process.cmdline.return_value = [process.exe.return_value]
        with patch("psutil.process_iter", return_value=[process]), self.assertRaisesRegex(
            RuntimeError, "still in use"
        ):
            self.delivery.quiescent()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_quiescence_refuses_unowned_model(self) -> None:
        process = Mock(pid=54321, info={"name": "llama-server.exe"})
        process.exe.return_value = str(self.root / "foreign/llama-server.exe")
        process.cmdline.return_value = [process.exe.return_value]
        with patch("psutil.process_iter", return_value=[process]), self.assertRaisesRegex(
            RuntimeError, "ownership"
        ):
            self.delivery.quiescent()
        process.terminate.assert_not_called()

    def test_quiescence_rejects_owned_helper_with_unlisted_name(self) -> None:
        process = Mock(pid=54321, info={"name": "rustxng.exe"})
        process.exe.return_value = str(self.delivery.APP / "harness/rustxng/rustxng.exe")
        process.cmdline.return_value = [process.exe.return_value]
        with patch("psutil.process_iter", return_value=[process]), self.assertRaisesRegex(
            RuntimeError, "still in use"
        ):
            self.delivery.quiescent()

    def test_private_acl_is_checked_before_accepting_backup_directory(self) -> None:
        with patch.dict(os.environ, {"SYSTEMROOT": str(self.root / "Windows")}), patch.object(
            self.release, "command", return_value='{"verified":true}'
        ):
            self.delivery.private_directory(self.delivery.PRIVATE / "backup-r1")
        with patch.dict(os.environ, {"SYSTEMROOT": str(self.root / "Windows")}), patch.object(
            self.release, "command", return_value='{"verified":false}'
        ), self.assertRaisesRegex(RuntimeError, "ACL verification"):
            self.delivery.private_directory(self.delivery.PRIVATE / "backup-r2")
        self.assertEqual(list((self.delivery.PRIVATE / "backup-r2").iterdir()), [])

    def test_private_acl_refuses_existing_nonempty_outside_and_reparse_before_command(self):
        target = self.delivery.PRIVATE / "backup-r1"
        target.mkdir(parents=True)
        original = target / "private.txt"
        original.write_bytes(b"preserve")
        with patch.object(self.release, "command", side_effect=AssertionError("ACL mutation")):
            for path in (target, self.root / "backup-r1", self.delivery.PRIVATE / "unexpected"):
                with self.subTest(path=path), self.assertRaises(RuntimeError):
                    self.delivery.private_directory(path)
            with patch.object(self.release, "no_links", side_effect=RuntimeError("Synthetic reparse")):
                with self.assertRaisesRegex(RuntimeError, "reparse"):
                    self.delivery.private_directory(self.delivery.PRIVATE / "backup-r2")
        self.assertEqual(original.read_bytes(), b"preserve")
        self.assertFalse((self.delivery.PRIVATE / "backup-r2").exists())

    def test_private_acl_live_h_empty_fixture_removes_explicit_broad_grant(self):
        # The only live ACL mutation is a fresh synthetic owned H: directory.
        with tempfile.TemporaryDirectory(
            prefix="delivery-acl-", dir="H:/Strata/evidence/upstream-sync-20261116/temp"
        ) as temp:
            self.delivery.PRIVATE = Path(temp).resolve()
            target = self.delivery.PRIVATE / "backup-r1"
            original_command = self.release.command

            def inject_broad_then_apply(args):
                subprocess.run(
                    [
                        str(Path(self.windows_root) / "System32/icacls.exe"),
                        str(target),
                        "/grant",
                        "*S-1-1-0:(OI)(CI)F",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                return original_command(args)

            with patch.dict(os.environ, {"SYSTEMROOT": self.windows_root}), patch.object(
                self.release, "command", side_effect=inject_broad_then_apply
            ):
                self.delivery.private_directory(target)
            self.assertEqual(list(target.iterdir()), [])
            # Success requires the production readback's exact two-identity DACL;
            # otherwise private_directory raises and no fixture payload is copied.
            (target / "synthetic.txt").write_bytes(b"fixture")
            self.assertEqual((target / "synthetic.txt").read_bytes(), b"fixture")

    def test_quiescence_refuses_access_denied(self) -> None:
        import psutil

        process = Mock(pid=54321, info={"name": "node.exe"})
        process.exe.side_effect = psutil.AccessDenied(process.pid)
        with patch("psutil.process_iter", return_value=[process]), self.assertRaisesRegex(
            RuntimeError, "Cannot establish"
        ):
            self.delivery.quiescent()

    def test_install_rejects_changed_user_state_before_spawn(self) -> None:
        expected = self.fixture_payload(self.delivery.APP)
        args = argparse.Namespace(
            candidate_qa_receipt=self.root / "candidate/receipt.json", reviewed_commit=self.commit
        )
        with patch.object(
            self.delivery,
            "extraction",
            return_value=(
                {"installer": "never.exe", "installerSHA256": "a" * 64},
                {"payload": expected},
                self.root,
            ),
        ), patch.object(
            self.delivery, "prior", return_value={"installerSHA256": "a" * 64, "resultSHA256": "b" * 64}
        ), patch.object(
            self.delivery, "pinned", return_value={}
        ), patch.object(
            self.delivery, "validate_qa"
        ), patch.object(
            self.delivery, "backup_identity", return_value=({}, {"user": {"old": {}}, "installed": {}})
        ), patch.object(
            self.delivery, "quiescent"
        ), patch.object(
            self.delivery.native, "logged", side_effect=AssertionError("spawn")
        ):
            with self.assertRaisesRegex(RuntimeError, "State changed"):
                self.delivery.install(args, self.root)

    def test_install_uses_standard_same_target_and_detects_payload_mismatch(self) -> None:
        expected = self.fixture_payload(self.delivery.APP)
        before = {
            "user": self.delivery.user_files(),
            "installed": self.delivery.files(self.delivery.APP),
            "selectedModel": {"sha256": "model"},
        }
        build = {"installer": "fixture.exe", "installerSHA256": "a" * 64}

        def corrupt(_argv, _work, _label, _timeout):
            (self.delivery.APP / "host/fixture.txt").write_bytes(b"broken install")

        with patch.object(
            self.delivery, "extraction", return_value=(build, {"payload": expected}, self.root)
        ), patch.object(
            self.delivery, "prior", return_value={"installerSHA256": "a" * 64, "resultSHA256": "b" * 64}
        ), patch.object(
            self.delivery, "pinned", return_value={}
        ), patch.object(
            self.delivery, "validate_qa"
        ), patch.object(
            self.delivery, "backup_identity", return_value=({}, before)
        ), patch.object(
            self.delivery, "quiescent"
        ), patch.object(
            self.release, "admit"
        ), patch.object(
            self.delivery.native, "logged", side_effect=corrupt
        ) as launch:
            with self.assertRaisesRegex(RuntimeError, "managed payload differs"):
                self.delivery.install(
                    argparse.Namespace(
                        candidate_qa_receipt=self.root / "candidate/receipt.json", reviewed_commit=self.commit
                    ),
                    self.root,
                )
        self.assertEqual(launch.call_args.args[0], ["fixture.exe", "/S", "/D=" + str(self.delivery.APP)])

    def test_extra_app_user_files_preserved_separately(self) -> None:
        rows = {
            "host/index.js": {},
            "strata-desktop.exe": {},
            "uninstall.exe": {},
            "user-note.txt": {"sha256": "abc"},
        }
        self.assertEqual(self.delivery.app_user_files(rows), {"user-note.txt": {"sha256": "abc"}})

    def test_qa_installed_requires_matching_installer(self) -> None:
        with patch.object(
            self.delivery, "extraction", return_value=({"installerSHA256": "a"}, {}, self.root)
        ), patch.object(self.delivery, "quiescent"), patch.object(
            self.delivery, "prior", return_value={"installerSHA256": "b", "exactPayloadMatch": True}
        ), patch.object(
            self.delivery.native, "logged", side_effect=AssertionError("spawn")
        ):
            with self.assertRaisesRegex(RuntimeError, "Current installed"):
                self.delivery.qa(argparse.Namespace(phase="installed-qa"), self.root)

    def test_normal_uses_get_only_and_retains_disabled_preferences(self) -> None:
        responses = {
            "/api/skills/learning": {"enabled": False, "autoApply": False},
            "/api/skills": [{"id": "existing", "enabled": False, "body": "private", "learning": None}],
            "/api/agent/tools": {"search": {"provider": "rustxng"}},
        }
        with patch.object(self.normal, "read_api", side_effect=lambda _base, route: responses[route]):
            selected = self.normal.choices("http://127.0.0.1:12345")
        self.assertFalse(selected["learning"]["enabled"])
        self.assertFalse(selected["learning"]["autoApply"])
        self.assertEqual(len(self.normal.choices_digest(selected)), 64)
        stream = Mock()
        response = Mock()
        stream.__enter__ = Mock(return_value=response)
        stream.__exit__ = Mock(return_value=False)
        response.read.return_value = b"{}"
        with patch("urllib.request.urlopen", return_value=stream) as request:
            self.normal.read_api("http://127.0.0.1:12345", "/api/health")
        self.assertEqual(request.call_args.args[0].get_method(), "GET")
        self.assertIsNone(request.call_args.args[0].data)

    def test_qa_gate_rejects_partial_or_misbound_results(self) -> None:
        report = {
            "ok": True,
            "mockModelOnly": True,
            "realRustxng": True,
            "consoleErrors": 0,
            "normalQuit": True,
            "actualWebView2": True,
            "isolatedHome": True,
            "target": "candidate",
            "actualInstalledExe": False,
            "sourceCommit": self.commit,
            "installerSHA256": "a" * 64,
            "checks": sorted(self.delivery.QA_CHECKS),
            "rustxngSources": [{"title": "fixture", "url": "https://example.invalid"}],
            "errors": [],
        }
        self.delivery.validate_qa(report, "candidate", self.commit, "a" * 64)
        for key, value in (
            ("checks", report["checks"][:-1]),
            ("normalQuit", False),
            ("sourceCommit", "b" * 40),
            ("installerSHA256", "b" * 64),
            ("mockModelOnly", False),
            ("rustxngSources", []),
            ("errors", ["fixture error"]),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "Complete source-bound"):
                self.delivery.validate_qa(report | {key: value}, "candidate", self.commit, "a" * 64)

    def test_normal_owner_rejects_reused_pid_before_environment_read(self) -> None:
        shell, host = Mock(), Mock()
        shell.create_time.return_value = 2.0
        host.create_time.return_value = 1.0
        args = argparse.Namespace(shell_pid=123, host_pid=456, shell_created=1.0, host_created=1.0)
        with patch("psutil.Process", side_effect=[shell, host]), self.assertRaisesRegex(
            RuntimeError, "PID was reused"
        ):
            self.normal.owned_processes(args)
        shell.environ.assert_not_called()

    def test_normal_rejects_missing_owner_and_choices(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "choice digest"):
            self.normal.main(["--execute", "--attempt", "r1"])
        with self.assertRaisesRegex(RuntimeError, "owners"):
            self.normal.main(["--execute", "--attempt", "r1", "--choices-sha256", "a" * 64])

    def test_app_weight_refusal_precedes_hash_backup_creation_and_copy(self) -> None:
        for index, name in enumerate(
            (
                "user.gguf",
                "cache/model.safetensors",
                "pack.q36epk/block",
                "weights.pack",
                "model.bin",
                "partial.gguf.part",
            )
        ):
            self.delivery.APP = self.root / f"weight-app-{index}"
            weight = self.delivery.APP / name
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"synthetic weight bytes to preserve")
            with self.subTest(name=name), patch.object(
                self.delivery, "extraction", return_value=({}, {}, self.root)
            ), patch.object(self.delivery, "quiescent"), patch.object(
                self.release, "sha", side_effect=AssertionError("weight read")
            ), patch.object(
                self.delivery, "private_directory", side_effect=AssertionError("backup creation")
            ), patch.object(
                self.delivery, "copy_rows", side_effect=AssertionError("payload copy")
            ):
                with self.assertRaisesRegex(RuntimeError, "weights refused before backup"):
                    self.delivery.backup(argparse.Namespace(attempt="r1"), self.root)
            self.assertEqual(weight.read_bytes(), b"synthetic weight bytes to preserve")

    def test_app_copy_rechecks_weight_member_list_before_destination_creation(self) -> None:
        src, dest = self.root / "app-race", self.root / "app-backup"
        src.mkdir()
        with self.assertRaisesRegex(RuntimeError, "before app copy"):
            self.delivery.copy_rows(src, dest, {"late-model.gguf": {}}, reject_weights=True)
        self.assertFalse(dest.exists())

    def profile_fixture(self, overrides: dict | None = None) -> dict:
        env = {
            "USERPROFILE": str(self.root / "user"),
            "APPDATA": str(self.root / "roaming"),
            "LOCALAPPDATA": str(self.root / "local"),
            **(overrides or {}),
        }
        return self.normal.effective_profile(
            env,
            env,
            str(self.root),
            str(self.root / "host"),
            {"dataDir": str(self.root / "user/.strata/data"), "modelsDir": str(self.root / "models")},
            str(self.root / "workspace"),
        )

    def test_original_profile_binding_rejects_isolated_copy_with_identical_choices(self) -> None:
        original = self.profile_fixture()
        isolated = self.profile_fixture({"STRATA_HOME": str(self.root / "isolated")})
        args = argparse.Namespace(port=12345, choices_sha256="a" * 64)
        with patch.object(
            self.normal, "original_baseline", return_value={"profileIdentity": original}
        ), patch.object(
            self.normal.delivery,
            "prior",
            return_value={"installed": True, "exactPayloadMatch": True, "payload": {}},
        ), patch.object(
            self.normal.delivery, "payload", return_value={}
        ), patch.object(
            self.normal, "owned_processes"
        ), patch.object(
            self.normal, "profile_identity", return_value=isolated
        ), patch.object(
            self.normal, "choices", return_value={"copied": "same choices"}
        ) as choices:
            with self.assertRaisesRegex(RuntimeError, "Original effective profile identity"):
                self.normal.verify(args, self.root)
        choices.assert_not_called()

    def test_legitimate_original_overrides_are_stable_and_preserved(self) -> None:
        env = {
            "STRATA_HOME": str(self.root / "custom/home"),
            "STRATA_SETTINGS_FILE": str(self.root / "custom/settings.json"),
            "STRATA_PROFILE_DIR": "~/custom/profiles",
            "STRATA_RUNTIME_DIR": str(self.root / "custom/runtime"),
            "STRATA_DESKTOP_DATA_DIR": str(self.root / "custom/desktop"),
        }
        original = self.profile_fixture(env)
        self.assertEqual(self.profile_fixture(env), original)
        self.assertEqual(
            original["effectiveHost"]["profiles"]["lexical"],
            os.path.normcase(str(self.root / "user/custom/profiles")),
        )
        for key in env:
            self.assertNotEqual(self.profile_fixture(env | {key: str(self.root / "different")}), original)

    def test_profile_snapshot_filters_secrets_and_tracks_actual_fallback_paths(self) -> None:
        env = {
            "USERPROFILE": str(self.root / "user"),
            "API_KEY": "NEVER_EXPORT_API_KEY",
            "STRATA_SHUTDOWN_TOKEN": "NEVER_EXPORT_SHUTDOWN",
            "OTHER_PASSWORD": "NEVER_EXPORT_PASSWORD",
        }
        snapshot = self.normal.filtered_paths(env, str(self.root))
        self.assertNotIn("NEVER_EXPORT", json.dumps(snapshot))
        original = self.profile_fixture()
        changed = self.normal.effective_profile(
            env,
            env,
            str(self.root),
            str(self.root / "host"),
            {"dataDir": str(self.root / "fallback/data"), "modelsDir": str(self.root / "other-models")},
            str(self.root / "other-workspace"),
        )
        self.assertNotEqual(original["effectiveHost"]["dataDir"], changed["effectiveHost"]["dataDir"])
        self.assertNotEqual(original["effectiveHost"]["modelsDir"], changed["effectiveHost"]["modelsDir"])
        self.assertNotEqual(original["effectiveHost"]["workspace"], changed["effectiveHost"]["workspace"])

    def test_process_and_api_capture_retains_only_allowed_path_fields(self) -> None:
        shell, host = Mock(), Mock()
        for process in (shell, host):
            process.environ.return_value = {
                "USERPROFILE": str(self.root / "user"),
                "SECRET": "DO_NOT_STORE_ENV",
                "STRATA_SHUTDOWN_TOKEN": "DO_NOT_STORE_TOKEN",
            }
            process.cwd.return_value = str(self.root)
        responses = {
            "/api/setup": {
                "app": {
                    "dataDir": str(self.root / "data"),
                    "modelsDir": str(self.root / "models"),
                    "unrelated": "DO_NOT_STORE_APP",
                },
                "privateState": "DO_NOT_STORE_STATE",
            },
            "/api/workspace": {"path": str(self.root / "workspace"), "entries": ["DO_NOT_STORE_LISTING"]},
        }
        with patch.object(self.normal, "owned_processes", return_value=(shell, host)), patch.object(
            self.normal, "read_api", side_effect=lambda _base, path: responses[path]
        ):
            result = self.normal.profile_identity(argparse.Namespace(), "http://127.0.0.1:12345")
        self.assertNotIn("DO_NOT_STORE", json.dumps(result))
        self.assertEqual(set(result["hostPaths"]), set(self.normal.PROFILE_PATH_ENV))

    def test_profile_path_case_and_alias_target_semantics(self) -> None:
        value = str(self.root / "Profile")
        self.assertEqual(
            self.normal.path_identity(value, str(self.root)),
            self.normal.path_identity(value.upper(), str(self.root)),
        )
        original = self.normal.path_identity(value, str(self.root))
        with patch.object(Path, "resolve", return_value=self.root / "different-target"):
            changed = self.normal.path_identity(value, str(self.root))
        self.assertEqual(original["lexical"], changed["lexical"])
        self.assertNotEqual(original["physical"], changed["physical"])
        for value in ("C:relative", "/drive-less", "\\drive-less"):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                self.normal.path_identity(value, str(self.root))

    def test_original_private_proof_is_digest_pinned(self) -> None:
        self.normal.delivery.PRIVATE = self.root / "private"
        proof_path = self.normal.delivery.PRIVATE / "original-profile-r1/original-profile.json"
        identity = self.profile_fixture()
        proof = {
            "observedVersion": "0.5.9",
            "profileIdentity": identity,
            "profileIdentitySHA256": self.normal.choices_digest(identity),
            "choicesSHA256": "a" * 64,
        }
        self.write(proof_path, proof)
        record = {"privateProof": str(proof_path), "privateProofSHA256": self.release.sha(proof_path)}
        with patch.object(self.normal.delivery, "prior", return_value=record):
            args = argparse.Namespace(choices_sha256="a" * 64)
            self.assertEqual(self.normal.original_baseline(args), proof)
            self.write(proof_path, proof | {"choicesSHA256": "b" * 64})
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                self.normal.original_baseline(args)


if __name__ == "__main__":
    unittest.main()
