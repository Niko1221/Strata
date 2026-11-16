"""Synthetic controller boundaries; no real keys, builds, release roots or GPU."""

import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from strata_control import signing  # noqa: E402


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
        self.native = module("native-input")
        self.feed = module("components-feed")
        self.release = self.feed.release
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.release.RES = self.root / "resources"
        self.git = patch.object(
            self.release, "git", return_value="apps/desktop/src-tauri/resources/control/.gitignore"
        )
        self.git_mock = self.git.start()
        self.addCleanup(self.git.stop)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.commit = "a" * 40

    def write(self, path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_default_plans_cannot_spawn_write_fetch_or_open_keys(self) -> None:
        for controller in (self.native, self.feed):
            release = controller.release
            with self.subTest(controller=controller.__name__), patch.object(
                release, "reviewed", side_effect=AssertionError("reviewed executed")
            ), patch.object(release, "save", side_effect=AssertionError("write")), patch.object(
                release, "command", side_effect=AssertionError("child")
            ), patch.object(
                Path, "open", side_effect=AssertionError("file read")
            ), patch.object(
                self.feed, "read_current", side_effect=AssertionError("network")
            ):
                self.assertEqual(controller.main([]), 0)

    def test_missing_execution_inputs_fail_before_runner(self) -> None:
        for controller, runner in ((self.native, self.native), (self.feed, self.feed.native)):
            with self.subTest(controller=controller.__name__), patch.object(
                runner, "run_controller", side_effect=AssertionError("runner started")
            ), self.assertRaises(RuntimeError):
                controller.main(["--execute", "--reviewed-commit", self.commit])

    def test_explicit_attempt_required_with_all_other_inputs(self) -> None:
        native_args = ["--execute", "--reviewed-commit", self.commit, "--min-driver", "nvidia=580.65"]
        feed_args = [
            "--execute",
            "--reviewed-commit",
            self.commit,
            "--no-dsv41-carry",
            "--key",
            str(self.feed.KEY),
            "--previous-source",
            self.feed.CURRENT,
            "--control-receipt",
            "control",
            "--component-receipt",
            "component",
            "--previous-feed",
            "feed",
            "--sequence",
            "1",
        ]
        for controller, runner, args in (
            (self.native, self.native, native_args),
            (self.feed, self.feed.native, feed_args),
        ):
            for flags in ([], ["--attempt", "r1", "--attempt=r2"], ["--attempt", "r0"]):
                with self.subTest(controller=controller.__name__, flags=flags), patch.object(
                    runner, "run_controller", side_effect=AssertionError("runner started")
                ), self.assertRaisesRegex(RuntimeError, "explicit valid --attempt"):
                    controller.main(args + flags)

    def test_native_and_feed_accept_both_explicit_attempt_forms(self) -> None:
        native_args = ["--execute", "--reviewed-commit", self.commit, "--min-driver", "nvidia=580.65"]
        feed_args = [
            "--execute",
            "--reviewed-commit",
            self.commit,
            "--no-dsv41-carry",
            "--key",
            str(self.feed.KEY),
            "--previous-source",
            self.feed.CURRENT,
            "--control-receipt",
            "control",
            "--component-receipt",
            "component",
            "--previous-feed",
            "feed",
            "--sequence",
            "1",
        ]
        for flags in (["--attempt", "r2"], ["--attempt=r2"]):
            with patch.object(self.native, "package_command"), patch.object(
                self.native, "run_controller", return_value=0
            ), patch.object(self.native, "finalize_binding"):
                self.assertEqual(self.native.main(native_args + flags), 0)
            with patch.object(self.feed.native, "run_controller", return_value=0):
                self.assertEqual(self.feed.main(feed_args + flags), 0)

    def test_native_production_selector_driver_and_cuda_path(self) -> None:
        args = argparse.Namespace(
            min_driver="nvidia=580.65", cuda_search=self.native.CUDA / "bin/x64", reviewed_commit=self.commit
        )
        with patch.object(self.native.release, "no_links"):
            command = list(map(str, self.native.package_command(args, self.root)))
            self.assertNotIn("--tree", command)
            self.assertNotIn("--allow-lib", command)
            self.assertEqual(command[command.index("--build-id") + 1], self.commit)
            self.assertEqual(
                [command[i + 1] for i, v in enumerate(command) if v == "--include"],
                list(self.native.PAYLOAD),
            )
            for bad in (None, "", "580.65", "nvidia=", "nvidia=580.65 --tree", "amd=580.65"):
                args.min_driver = bad
                with self.subTest(driver=bad), self.assertRaises(RuntimeError):
                    self.native.package_command(args, self.root)
            args.min_driver = "nvidia=580.65"
            args.cuda_search = self.root
            with self.assertRaisesRegex(RuntimeError, "CUDA 13.3"):
                self.native.package_command(args, self.root)

    def test_environment_filters_credentials_and_tauri_override(self) -> None:
        # The real production environment constructor is exercised with tool discovery stubbed only.
        with patch.dict(
            os.environ,
            {
                "GH_TOKEN": "synthetic",
                "OPENAI_API_KEY": "synthetic",
                "TAURI_CONFIG": "synthetic",
                "PYTHONPATH": "synthetic",
                "SYSTEMROOT": "C:/Windows",
                "PATH": "untrusted",
            },
        ), patch("shutil.which", return_value="C:/Program Files/Git/cmd/git.exe"):
            env = self.native.environment(self.root, "synthetic-owned-token")
        for name in ("GH_TOKEN", "OPENAI_API_KEY", "TAURI_CONFIG", "PYTHONPATH"):
            self.assertNotIn(name, env)
        self.assertNotIn("untrusted", env["PATH"])
        self.assertEqual(env["CMAKE_BUILD_PARALLEL_LEVEL"], "5")
        self.assertEqual(env["CTEST_PARALLEL_LEVEL"], "1")

    def test_failed_builder_preserves_both_output_snapshots(self) -> None:
        build = self.root / "build"
        (build / "Release").mkdir(parents=True)
        for name in ("strata.exe", "strata-q36.exe", "native-build.json"):
            (build / "Release" / name).write_bytes(b"synthetic-before")
        work = self.root / "attempt"
        work.mkdir()

        def fail(*_args):
            (build / "Release/strata.exe").write_bytes(b"synthetic-failed-build")
            raise RuntimeError("builder failed")

        with patch.object(self.native, "BUILD", build), patch.object(
            self.native, "package_command", return_value=[]
        ), patch.object(self.native, "configured_inputs", return_value={"fixture": True}), patch.object(
            self.native, "logged", side_effect=fail
        ), self.assertRaisesRegex(
            RuntimeError, "builder failed"
        ):
            self.native.native(argparse.Namespace(build=build), work)
        self.assertEqual((work / "build-before/strata.exe").read_bytes(), b"synthetic-before")
        self.assertEqual((work / "build-after/strata.exe").read_bytes(), b"synthetic-failed-build")
        self.assertTrue((work / "build-after/strata-q36.exe").is_file())
        self.assertFalse((work / "native-binding.json").exists())

    def binding_fixture(self) -> None:
        component = self.root / "unsigned"
        component.mkdir()
        (component / "strata.exe").write_bytes(b"synthetic")
        binding = {"sourceCommit": self.commit, "releaseQualified": False, "builder": {"exitCode": 0}}
        self.write(self.root / "native-binding.pending.json", binding)
        self.write(
            self.root / "receipt.json",
            {
                "ok": True,
                "phase": "native-input",
                "sourceCommit": self.commit,
                "nativeComponent": str(component),
                "files": self.native.release.inventory(component),
                "bindingPendingSHA256": self.native.release.sha(self.root / "native-binding.pending.json"),
            },
        )
        self.write(self.root / "owned-job.json", good_job())

    def test_binding_only_published_after_successful_drained_job(self) -> None:
        self.binding_fixture()
        self.native.finalize_binding(self.root)
        result = self.native.release.load(self.root / "native-binding.json")
        self.assertEqual(result["ownedJobSHA256"], self.native.release.sha(self.root / "owned-job.json"))
        with self.assertRaises(FileExistsError):
            self.native.finalize_binding(self.root)

    def test_binding_rejects_live_failed_job_and_changed_payload(self) -> None:
        self.binding_fixture()
        for key, value in (
            ("job_active_after", 1),
            ("exit_code", 2),
            ("native_exit_code", 2),
            ("contained_before_resume", False),
            ("termination_confirmed", False),
        ):
            self.write(self.root / "owned-job.json", good_job() | {key: value})
            with self.subTest(field=key), self.assertRaises(RuntimeError):
                self.native.finalize_binding(self.root)
            self.assertFalse((self.root / "native-binding.json").exists())
        self.write(self.root / "owned-job.json", good_job())
        (self.root / "unsigned/strata.exe").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "package changed"):
            self.native.finalize_binding(self.root)

    def control_fixture(self) -> tuple[Path, str]:
        work = self.root / "control-r1"
        assets = work / "assets"
        assets.mkdir(parents=True)
        (assets / "control.zip").write_bytes(b"synthetic archive")
        entry = {
            "version": "0.6.0",
            "platform": "windows-x86_64",
            "format": "zip",
            "url": "control.zip",
            "size": 17,
            "sha256": self.release.sha(assets / "control.zip"),
        }
        self.write(assets / "control.entry.json", entry)
        (work / "terminal").mkdir()
        (work / "terminal/source.zip").write_bytes(b"synthetic source archive")
        (work / "control").mkdir()
        (work / "control/COMMIT").write_text(self.commit)
        staged = self.release.RES / "control"
        staged.mkdir(parents=True)
        (staged / "COMMIT").write_text(self.commit)
        (staged / ".gitignore").write_text("synthetic tracked scaffold")
        self.write(work / "owned-job.json", good_job())
        record = {
            "ok": True,
            "phase": "control",
            "sourceCommit": self.commit,
            "sourceRoot": str(self.feed.native.PRIMARY),
            "releaseQualified": False,
            "assets": self.release.inventory(assets),
            "archive": str(work / "terminal/source.zip"),
            "archiveSHA256": self.release.sha(work / "terminal/source.zip"),
            "controlFiles": self.release.inventory(staged),
        }
        self.write(work / "receipt.json", record)
        return work / "receipt.json", self.release.sha(work / "receipt.json")

    def test_control_receipt_checks_real_temp_bytes_and_all_archives(self) -> None:
        path, checksum = self.control_fixture()
        with patch.object(self.release, "OUT", self.root):
            identity = self.feed.receipt_assets(path, checksum, "control", self.commit)
            self.assertEqual(identity["entry"]["url"], "control.zip")
            (path.parent / "terminal/source.zip").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "Original control archive"):
                self.feed.receipt_assets(path, checksum, "control", self.commit)

    def test_feed_rejects_wrong_receipt_source_hash_membership_and_job(self) -> None:
        path, checksum = self.control_fixture()
        with patch.object(self.release, "OUT", self.root):
            for bad_hash, bad_commit in (("0" * 64, self.commit), (checksum, "b" * 40)):
                with self.assertRaises(RuntimeError):
                    self.feed.receipt_assets(path, bad_hash, "control", bad_commit)
            self.write(path.parent / "owned-job.json", good_job() | {"job_active_after": 1})
            with self.assertRaisesRegex(RuntimeError, "Job"):
                self.feed.receipt_assets(path, checksum, "control", self.commit)
            self.write(path.parent / "owned-job.json", good_job())
            (path.parent / "assets/extra.txt").write_text("unexpected")
            with self.assertRaisesRegex(RuntimeError, "membership"):
                self.feed.receipt_assets(path, checksum, "control", self.commit)

    def test_control_scaffold_and_fresh_payload_are_independently_bound(self) -> None:
        path, checksum = self.control_fixture()
        staged = self.release.RES / "control"
        with patch.object(self.release, "OUT", self.root):
            self.feed.receipt_assets(path, checksum, "control", self.commit)
            (staged / ".gitignore").write_text("tampered scaffold")
            with self.assertRaisesRegex(RuntimeError, "Staged Control"):
                self.feed.receipt_assets(path, checksum, "control", self.commit)
            (staged / ".gitignore").write_text("synthetic tracked scaffold")
            self.git_mock.return_value = ""
            with self.assertRaisesRegex(RuntimeError, "scaffold must be tracked"):
                self.feed.receipt_assets(path, checksum, "control", self.commit)
            self.git_mock.return_value = "apps/desktop/src-tauri/resources/control/.gitignore"
            (path.parent / "control/COMMIT").write_text("tampered fresh payload")
            with self.assertRaisesRegex(RuntimeError, "Fresh Control"):
                self.feed.receipt_assets(path, checksum, "control", self.commit)
            (path.parent / "control/COMMIT").write_text(self.commit)
            (staged / "README.txt").write_text("not an allowed scaffold")
            record = self.release.load(path)
            record["controlFiles"] = self.release.inventory(staged)
            self.write(path, record)
            with self.assertRaisesRegex(RuntimeError, "Fresh Control"):
                self.feed.receipt_assets(path, self.release.sha(path), "control", self.commit)

    def predecessor_fixture(self) -> tuple[argparse.Namespace, bytes, bytes, dict]:
        key = signing.generate_key()  # Ephemeral fixture only; no production path is opened.
        payload = json.dumps(
            {
                "schema_version": 1,
                "feed": "strata-updates",
                "channel": "stable",
                "sequence": 900,
                "generated_at": "2026-11-16T00:00:00Z",
                "expires_at": None,
            }
        ).encode()
        signature = json.dumps(signing.sign_bytes(key, payload)).encode()
        path = self.root / "previous.json"
        path.write_bytes(payload)
        args = argparse.Namespace(
            previous_source=self.feed.CURRENT,
            previous_feed=path,
            previous_feed_sha256=self.release.sha(path),
            sequence=901,
        )
        trust = {key["key_id"]: base64.b64decode(key["public"])}
        return args, payload, signature, trust

    def test_predecessor_uses_real_signature_and_live_bytes(self) -> None:
        args, payload, signature, trust = self.predecessor_fixture()
        with patch.object(self.feed, "read_current", side_effect=[payload, signature]), patch.object(
            signing, "load_trusted_keys", return_value=trust
        ):
            result = self.feed.predecessor(args)
        self.assertEqual(result["sequence"], 900)
        self.assertEqual(result["sha256"], args.previous_feed_sha256)

    def test_predecessor_rejects_stale_source_replay_and_invalid_signature(self) -> None:
        args, payload, signature, trust = self.predecessor_fixture()
        cases = [(payload + b" ", signature, 901), (payload, signature, 900), (payload, b"{}", 901)]
        for remote, sig, sequence in cases:
            args.sequence = sequence
            with self.subTest(sequence=sequence, sig=sig[:2]), patch.object(
                self.feed, "read_current", side_effect=[remote, sig]
            ), patch.object(signing, "load_trusted_keys", return_value=trust), self.assertRaises(Exception):
                self.feed.predecessor(args)
        args.previous_source = "https://github.com/zapabob/Strata/releases/tag/v0.5.9"
        with patch.object(
            self.feed, "read_current", side_effect=AssertionError("network")
        ), self.assertRaises(RuntimeError):
            self.feed.predecessor(args)

    def test_feed_requires_explicit_scope_before_any_key_access(self) -> None:
        with patch.object(
            self.release, "signing_key_path", side_effect=AssertionError("key access")
        ), self.assertRaises(RuntimeError):
            self.feed.feed_phase(argparse.Namespace(no_dsv41_carry=False), self.root)

    def test_actual_runner_version_contract(self) -> None:
        for output in ("Strata 0.6.0", "strata-q38 own-native 0.6.0 CUDA13 SM120\n"):
            with self.subTest(output=output):
                self.native.validate_version(output)
        for output in ("strata-q38 own-native 0.6.0-dev CUDA13 SM120", "Strata 0.6.00", "Strata v0.6.0", "Strata 0.6.0+local", ""):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                self.native.validate_version(output)

    def test_native_success_probes_all_binaries_before_packaging(self) -> None:
        build = self.root / "build"
        (build / "Release").mkdir(parents=True)
        toolkit = self.root / "cuda"
        self.write(toolkit / "version.json", {"cuda": {"version": "13.3.1"}})
        work = self.root / "attempt"
        work.mkdir()
        inputs = {
            "sources": {"src/a.cpp": "0" * 64},
            "ggml": {"mode": "vendored"},
            "baseline": {"path": "profile.json", "sha256": "1" * 64},
        }
        probes = []

        def command(argv, **_kwargs):
            probes.append(Path(argv[0]).name)
            return "strata-q38 own-native 0.6.0 CUDA13 SM120\n" if Path(argv[0]).name == "strata-q38.exe" else "Strata 0.6.0"

        def logged(argv, _work, name, *_args):
            if name == "builder":
                self.assertEqual(str(argv[-1]), "5")
                for binary in ("strata.exe", "strata-q36.exe"):
                    (build / "Release" / binary).write_bytes(b"synthetic executable")
                self.write(
                    build / "Release/native-build.json",
                    {
                        "sources": inputs["sources"],
                        "recipe": {"ggml": inputs["ggml"], "cudaSpillBaseline": inputs["baseline"]},
                    },
                )
            if name == "package":
                self.assertEqual(probes, ["strata.exe", "strata-q36.exe", "strata-q38.exe"])
                import shutil

                shutil.copytree(work / "package-input", work / "unsigned" / self.native.release.NAME)
            return {"command": list(map(str, argv)), "exitCode": 0}

        def stage_q38(_release, _root, destination):
            (destination / "strata-q38.exe").write_bytes(b"synthetic q38")
            self.write(destination / "q38-build.json", {"sourceCommit": self.native.Q38.COMMIT})
            return {"sourceCommit": self.native.Q38.COMMIT}

        with patch.object(self.native.Q38, "stage", side_effect=stage_q38), patch.object(self.native, "BUILD", build), patch.object(
            self.native, "CUDA", toolkit
        ), patch.object(self.native, "package_command", return_value=["synthetic-packager"]), patch.object(
            self.native, "configured_inputs", return_value=inputs
        ), patch.object(
            self.native, "toolkit_libraries", return_value={}
        ), patch.object(
            self.native, "logged", side_effect=logged
        ), patch.object(
            self.native.release, "command", side_effect=command
        ), patch.object(
            self.native.release, "verify"
        ) as verify:
            result = self.native.native(
                argparse.Namespace(build=build, reviewed_commit=self.commit, cuda_search=toolkit), work
            )
        self.assertFalse(result["signed"])
        self.assertEqual(verify.call_args.args[0], "native")
        binding = self.native.release.load(work / "native-binding.pending.json")
        self.assertEqual(set(binding["binaries"]), {"strata.exe", "strata-q36.exe", "strata-q38.exe"})
        self.assertEqual(binding["builder"]["exitCode"], 0)
        self.assertFalse((work / "native-binding.json").exists())

    def test_feed_phase_calls_quiet_production_builder_and_verifies_outputs(self) -> None:
        # Real production feed encoding/signature verification, with an ephemeral in-memory key.
        tool = self.release.module("synthetic_feed_tool", ROOT / "packaging/feed_tool.py")
        key = signing.generate_key()
        trust = {key["key_id"]: base64.b64decode(key["public"])}
        identities = {}
        for phase in ("control", "component"):
            assets = self.root / phase
            assets.mkdir()
            archive = "control.zip" if phase == "control" else "engine.tar.gz"
            (assets / archive).write_bytes(b"synthetic archive")
            entry = {
                "version": "0.6.0",
                "platform": "windows-x86_64",
                "url": archive,
                "size": 17,
                "installed_bytes": 17,
                "sha256": self.release.sha(assets / archive),
                "format": "zip" if phase == "control" else "tar.gz",
            }
            if phase == "component":
                entry.update(
                    backend="cuda",
                    variant="cuda13",
                    archs=["sm_120"],
                    min_driver={"nvidia": "580.65"},
                    min_runtime=None,
                    build_id=self.commit,
                    manifest_sha256="1" * 64,
                )
            self.write(assets / "package.entry.json", entry)
            identities[phase] = {
                "assets": str(assets),
                "entry": entry,
                "files": self.release.inventory(assets),
            }
        args = argparse.Namespace(
            no_dsv41_carry=True,
            key=self.feed.KEY,
            sequence=901,
            control_receipt=self.root / "control-receipt",
            control_receipt_sha256="2" * 64,
            component_receipt=self.root / "component-receipt",
            component_receipt_sha256="3" * 64,
            reviewed_commit=self.commit,
        )
        work = self.root / "feed-r1"
        work.mkdir()
        commands = []

        def command(argv, **kwargs):
            commands.append((list(map(str, argv)), kwargs))
            if "build-feed" in argv:
                self.assertIs(kwargs.get("quiet"), True)
                self.assertIn("--no-dsv41-carry", argv)
                tool.build_feed(
                    work / "assets",
                    key=key,
                    sequence=901,
                    engine_dir=work / "component-entries",
                    control_dir=work / "control-entries",
                )
            else:
                from strata_control import feed

                checked = feed.fetch(str(work / "assets/strata-feed.json"), trust)
                self.assertEqual(checked.sequence, 901)
            return ""

        with patch.object(self.feed, "predecessor", return_value={"sequence": 900}), patch.object(
            self.feed, "receipt_assets", side_effect=lambda _p, _h, phase, _c: identities[phase]
        ), patch.object(
            self.release, "signing_key_path", return_value=self.root / "unused-key-path"
        ), patch.object(
            self.release, "command", side_effect=command
        ):
            result = self.feed.feed_phase(args, work)
        self.assertFalse(result["published"])
        self.assertTrue(result["signatureVerified"])
        self.assertEqual(len(commands), 2)
        self.assertIn("strata-feed.json.sig", result["assets"])
        self.assertFalse((self.root / "unused-key-path").exists())


if __name__ == "__main__":
    unittest.main()
