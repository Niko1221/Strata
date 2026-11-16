"""Synthetic publication boundaries only: no real app, GitHub, key, model or GPU."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("publication_test_subject", HERE / "publish.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
b, o = p.bundle, p.original
COMMIT = "a" * 40


def row(content: bytes) -> dict:
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def server(assets: dict) -> dict:
    return {
        "id": 10,
        "tag_name": p.TAG,
        "draft": True,
        "prerelease": False,
        "assets": [
            {
                "id": i + 1,
                "name": n,
                "size": r["bytes"],
                "digest": "sha256:" + r["sha256"],
                "state": "uploaded",
            }
            for i, (n, r) in enumerate(assets.items())
        ],
    }


def tar_stream(names: list[tuple[str, bytes, bytes]]) -> io.BytesIO:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, content, kind in names:
            item = tarfile.TarInfo(name)
            item.type = kind
            item.size = len(content) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                item.linkname = "somewhere"
            archive.addfile(item, io.BytesIO(content) if item.size else None)
    stream.seek(0)
    return stream


class Scratch(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()


class ArchiveBoundaries(Scratch):
    def test_full_stream_and_inner_installer_identity(self) -> None:
        expected = {b.delivery.INSTALLER: row(b"identical exe"), "SOURCE.json": row(b"{}")}
        stream = tar_stream(
            [
                (b.delivery.INSTALLER, b"identical exe", tarfile.REGTYPE),
                ("SOURCE.json", b"{}", tarfile.REGTYPE),
            ]
        )
        self.assertEqual(b.stream_members(stream, expected), expected)

    def test_duplicate_traversal_link_extra_and_changed_bytes_are_rejected(self) -> None:
        good = ("installer.exe", b"payload", tarfile.REGTYPE)
        cases = (
            [good, good],
            [("../installer.exe", b"payload", tarfile.REGTYPE)],
            [("installer.exe", b"", tarfile.SYMTYPE)],
            [good, ("secret.txt", b"secret", tarfile.REGTYPE)],
            [("installer.exe", b"changed", tarfile.REGTYPE)],
            [],
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                b.stream_members(tar_stream(case), {"installer.exe": row(b"payload")})

    def test_nonflat_or_oversized_public_assets_rejected(self) -> None:
        (self.root / "private").mkdir()
        with self.assertRaises(RuntimeError):
            b.table(self.root)

    def test_manifest_populations_do_not_self_hash(self) -> None:
        members = {"x.exe": row(b"x")}
        inner = b.sums(members)
        members["BUNDLE-SHA256SUMS.txt"] = row(inner)
        outer = b.sums({**members, b.ARCHIVE: row(b"archive")})
        self.assertNotIn(b"BUNDLE-SHA256SUMS.txt", inner)
        self.assertIn(b.ARCHIVE.encode(), outer)
        self.assertNotIn(b"  SHA256SUMS.txt", outer)
        self.assertEqual(len(b.PUBLIC_NAMES), 17)

    def test_public_projection_rejects_private_paths_and_credentials(self) -> None:
        for value in (
            "C:/Users/person/model.gguf",
            "H:\\private\\backup",
            "Bearer credential",
            "api_key=secret",
        ):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                b.safe_public({"detail": value})
        b.safe_public({"profile": "tools/baselines/reviewed.json", "sha256": "a" * 64})

    def test_source_scripts_require_reviewed_blob_with_only_complete_eol_conversion(self) -> None:
        folder = self.root / "packaging"
        folder.mkdir()
        path = folder / "install.sh"
        canonical = b"#!/bin/sh\necho verified\n"
        for physical, accepted in (
            (canonical, True),
            (canonical.replace(b"\n", b"\r\n"), True),
            (b"#!/bin/sh\necho changed\n", False),
            (b"#!/bin/sh\r\necho verified\n", False),
        ):
            path.write_bytes(physical)
            with self.subTest(physical=physical), patch.object(
                b.native, "PRIMARY", self.root
            ), patch.object(b.release, "git", return_value="a" * 40), patch.object(
                b.delivery, "canonical_blob", return_value=canonical
            ):
                if accepted:
                    content, identity = b.script_input("install.sh", COMMIT)
                    self.assertEqual(content, canonical)
                    self.assertEqual(identity["canonicalSHA256"], row(canonical)["sha256"])
                else:
                    with self.assertRaises(RuntimeError):
                        b.script_input("install.sh", COMMIT)


class ReceiptBoundaries(Scratch):
    def test_qualification_is_only_exact_pinned_root_not_reusable_decision(self) -> None:
        checks = {
            name: ("NOT_REUSABLE" if name in ("learning", "autoApply") else "PASS")
            for name in b.real.CHECKS
        }
        actual = {"outcome": "PARTIAL_NOT_REUSABLE", "checks": checks}
        path = self.root / "root-decision.json"
        value = {
            "schema": "strata.publication-qualification.v1",
            "sourceCommit": COMMIT,
            "realModelSHA256": "b" * 64,
            "decision": "ALLOW_LIMITED_PUBLICATION",
            "rationale": "Synthetic owner decision only",
        }
        path.write_text(json.dumps(value))
        with patch.object(b, "QUALIFICATION_ROOT", self.root):
            digest = b.release.sha(path)
            self.assertEqual(
                b.qualification(actual, "b" * 64, COMMIT, path, digest)["sha256"], digest
            )
            for field, changed in (
                ("sourceCommit", "c" * 40),
                ("realModelSHA256", "d" * 64),
                ("decision", "PASS_EVERYTHING"),
                ("rationale", ""),
                ("rationale", "x" * 1001),
            ):
                altered = {**value, field: changed}
                path.write_text(json.dumps(altered))
                with self.subTest(field=field), self.assertRaises(RuntimeError):
                    b.qualification(actual, "b" * 64, COMMIT, path, b.release.sha(path))
            path.write_text(json.dumps(value))
            for digest in (None, "0" * 64):
                with self.assertRaises(RuntimeError):
                    b.qualification(actual, "b" * 64, COMMIT, path, digest)
            for status in ("FAIL", "NOT_RUN", "PASS"):
                bad = deepcopy(actual)
                bad["checks"]["learning"] = status
                with self.subTest(status=status), self.assertRaises(RuntimeError):
                    b.qualification(bad, "b" * 64, COMMIT, path, b.release.sha(path))
            complete = {
                "outcome": "MECHANICS_PASSED",
                "checks": {name: "PASS" for name in b.real.CHECKS},
            }
            self.assertIsNone(b.qualification(complete, "b" * 64, COMMIT, None, None))
            with self.assertRaises(RuntimeError):
                b.qualification(complete, "b" * 64, COMMIT, path, b.release.sha(path))

    def test_stale_or_mismatched_pin_rejected_before_receipt_use(self) -> None:
        path = self.root / "proof.json"
        path.write_text('{"sourceCommit":"old"}')
        with self.assertRaises(RuntimeError):
            b.delivery.pinned(path, "0" * 64)
        expected = b.release.sha(path)
        path.write_text('{"sourceCommit":"new"}')
        with self.assertRaises(RuntimeError):
            b.delivery.pinned(path, expected)

    def test_typed_proof_cannot_differ_from_raw_drained_receipt(self) -> None:
        path = self.root / "original-engine.json"
        value = {
            "schema": o.SCHEMA,
            "receiptSHA256": "a" * 64,
            "ownedJobSHA256": "b" * 64,
            "workerLogSHA256": "c" * 64,
            "sourceCommit": COMMIT,
        }
        with patch.object(o.delivery, "pinned", return_value=value), patch.object(
            o.delivery, "receipt", return_value={"schema": o.SCHEMA, "sourceCommit": "b" * 40}
        ), self.assertRaisesRegex(RuntimeError, "differs from drained"):
            o.proof(path, "d" * 64, "original-engine", o.SCHEMA, COMMIT)

    def test_missing_current_ram_provenance_rejected(self) -> None:
        raw = {"schema": o.SCHEMA}
        value = {
            **raw,
            "receiptSHA256": "a" * 64,
            "ownedJobSHA256": "b" * 64,
            "workerLogSHA256": "c" * 64,
        }
        with patch.object(o.delivery, "pinned", return_value=value), patch.object(
            o.delivery, "receipt", return_value=raw
        ), self.assertRaisesRegex(RuntimeError, "operation provenance"):
            o.proof(
                self.root / "original-engine.json", "d" * 64, "original-engine", o.SCHEMA, COMMIT
            )

    def test_named_self_tests_reject_missing_duplicate_failed_and_truthy_status(self) -> None:
        good = {
            "schemaVersion": 1,
            "kind": "strata-desktop-self-test",
            "carrier": "tauri",
            "ok": True,
            "error": "",
            "steps": [{"name": n, "ok": True} for n in b.SELF_TESTS],
        }
        b.validate_self_test(good)
        for change in ("missing", "duplicate", "failed", "truthy", "error"):
            value = deepcopy(good)
            if change == "missing":
                value["steps"].pop()
            elif change == "duplicate":
                value["steps"][0] = value["steps"][1]
            elif change == "failed":
                value["steps"][0]["ok"] = False
            elif change == "truthy":
                value["ok"] = "yes"
            else:
                value["error"] = "retained failure"
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                b.validate_self_test(value)

    def test_feed_rejects_extra_file_and_changed_predecessor(self) -> None:
        folder = self.root / "feed-r1"
        assets = folder / "assets"
        assets.mkdir(parents=True)
        path = folder / "receipt.json"
        path.write_text("{}")
        for name in b.FEED_NAMES:
            (assets / name).write_bytes(name.encode())
        baseline = {
            "source": b.feed.CURRENT,
            "sequence": 8,
            "sha256": "a" * 64,
            "signatureSHA256": "b" * 64,
            "keyId": "c" * 16,
        }
        previous = self.root / "snapshot.json"
        previous.write_text(json.dumps({"predecessor": baseline}))
        snapshot = {
            "results": {"snapshot": {"path": str(previous)}, "previousFeed": {"path": "previous"}}
        }
        record = {
            "assets": b.release.inventory(assets),
            "signatureVerified": True,
            "published": False,
            "predecessor": baseline,
            "sequence": 9,
            "inputs": {"control": {}, "component": {}},
        }
        for change in ("extra", "predecessor"):
            value = deepcopy(record)
            if change == "extra":
                value["assets"]["private.json"] = "0" * 64
            else:
                value["predecessor"]["sequence"] = 7
            with self.subTest(change=change), patch.object(
                b.delivery, "receipt", return_value=value
            ), patch.object(b, "job_of", return_value={}), patch.object(
                b.acceptance, "successful_job"
            ), self.assertRaises(
                RuntimeError
            ):
                b.validate_feed(path, "a" * 64, COMMIT, snapshot, {}, {})


class EngineBoundaries(unittest.TestCase):
    def test_exact_import_job_success_and_replacement_error_cancel_refusal(self) -> None:
        good = {"id": "owned", "kind": "install-engine-file", "state": "done"}
        with patch.object(o.normal, "read_api", return_value={"jobs": [good]}):
            self.assertEqual(o.wait_job("http://127.0.0.1:1", "owned")["id"], "owned")
        for job in (
            {**good, "id": "other"},
            {**good, "kind": "update-engine"},
            {**good, "state": "error"},
            {**good, "state": "cancelled"},
        ):
            with self.subTest(job=job), patch.object(
                o.normal, "read_api", return_value={"jobs": [job]}
            ), self.assertRaises(RuntimeError):
                o.wait_job("http://127.0.0.1:1", "owned")
        with self.assertRaises(RuntimeError):
            o.wait_job("unused", "owned", timeout=0)

    def test_idle_requires_no_runtime_turn_operation_or_loaded_engine(self) -> None:
        good = {
            "/api/native": {"instances": [], "jobs": []},
            "/api/runtime": {"instances": []},
            "/api/chat/active": {"turns": []},
            "/api/health": {"engine": {"status": "stopped", "running": False}},
        }
        with patch.object(o.normal, "read_api", side_effect=lambda _base, route: good[route]):
            o.check_idle("unused")
        changes = (
            ("/api/native", {"instances": [{"id": "live"}]}),
            ("/api/native", {"instances": [], "jobs": [{"state": "running"}]}),
            ("/api/runtime", {"instances": [{"id": "live"}]}),
            ("/api/chat/active", {"turns": [{"id": "active"}]}),
            ("/api/health", {"engine": {"status": "ready", "running": True}}),
        )
        for route, value in changes:
            state = {**good, route: value}
            with self.subTest(route=route), patch.object(
                o.normal, "read_api", side_effect=lambda _base, route: state[route]
            ), self.assertRaises(RuntimeError):
                o.check_idle("unused")

    def test_original_profile_and_choices_checked_before_mutation(self) -> None:
        args = SimpleNamespace(port=10)
        baseline = {"profileIdentity": {"home": "original"}, "choicesSHA256": "a" * 64}
        for profile, digest in (({"home": "other"}, "a" * 64), ({"home": "original"}, "b" * 64)):
            with patch.object(o.normal, "owned_processes"), patch.object(
                o.normal, "read_api", return_value={"host": "0.6.0", "shell": "0.6.0"}
            ), patch.object(o.normal, "profile_identity", return_value=profile), patch.object(
                o.normal, "choices", return_value={}
            ), patch.object(
                o.normal, "choices_digest", return_value=digest
            ), self.assertRaises(
                RuntimeError
            ):
                o.stable(args, baseline)

    def test_engine_library_observation_is_not_signature_verification(self) -> None:
        with self.assertRaises(RuntimeError):
            o.entries({"available": True, "serving": True, "families": []})
        with patch.object(o.release, "inventory", return_value={}), self.assertRaises(RuntimeError):
            o.installed_component(
                Path("C:/foreign/engine"), Path("C:/signed"), {}, Path("C:/original")
            )

    def test_normal_owner_creation_order_and_choices_cross_binding(self) -> None:
        def command(script: str) -> list[str]:
            return [
                sys.executable,
                "-B",
                str(b.HERE / script),
                "--shell-pid",
                "10",
                "--shell-created",
                "100.0",
                "--host-pid",
                "11",
                "--host-created",
                "101.0",
                "--port",
                "90",
                "--choices-sha256",
                "a" * 64,
            ]

        engine_path, normal_path = Path("C:/proof/engine.json"), Path("C:/later/normal.json")
        engine = {
            "installerSHA256": "b" * 64,
            "choicesSHA256": "a" * 64,
            "profileIdentitySHA256": "c" * 64,
            "drainedAtFiletime": 150,
        }
        record = {
            **engine,
            "normalStartup": True,
            "originalChoicesAndSkillsUnchanged": True,
            "originalEffectiveProfileUnchanged": True,
            "modelUnloaded": True,
            "observationOnly": True,
            "settingsWritten": False,
            "persistentAppStartedByThisController": False,
            "activeTurns": 0,
            "shellPID": 10,
            "hostPID": 11,
        }

        def jobs(path: Path) -> dict:
            return {
                "root_creation_filetime": 200 if path == normal_path else 100,
                "operationExecution": {
                    "command": command(
                        "normal-start.py" if path == normal_path else "original-engine.py"
                    )
                },
            }

        with patch.object(b, "job_of", side_effect=jobs), patch.object(
            b.acceptance, "successful_job"
        ):
            b.normal_pair(record, normal_path, engine, engine_path, COMMIT)
            for change in ("order", "owner", "choices", "loaded"):
                bad = deepcopy(record)

                def altered(path: Path) -> dict:
                    value = jobs(path)
                    if change == "order" and path == normal_path:
                        value["root_creation_filetime"] = 99
                    if change == "owner" and path == normal_path:
                        value["operationExecution"]["command"][6] = "102.0"
                    return value

                if change == "choices":
                    bad["choicesSHA256"] = "d" * 64
                if change == "loaded":
                    bad["modelUnloaded"] = False
                with self.subTest(change=change), patch.object(
                    b, "job_of", side_effect=altered
                ), self.assertRaises(RuntimeError):
                    b.normal_pair(bad, normal_path, engine, engine_path, COMMIT)


class EnginePhaseBoundaries(Scratch):
    """Run the real phase on synthetic API/files and observe the mutation boundary."""

    def exercise(self, change: str | None = None, *, post_change: str | None = None) -> dict:
        from strata_control import engine_component, signing

        self.attempt = getattr(self, "attempt", 0) + 1
        root = self.root / str(self.attempt)
        components = root / "profile/components"
        installed_path = components / "engine" / o.release.NAME
        signed = root / "signed" / o.release.NAME
        assets, work = root / "assets", root / "work"
        for folder in (installed_path, signed, assets, work):
            folder.mkdir(parents=True)
        manifest = {
            "version": "0.6.0",
            "build_id": COMMIT + "-hakua-runtimes-v1",
            "signature": {"synthetic": True},
        }
        for folder in (installed_path, signed):
            (folder / "engine-component.json").write_text(json.dumps(manifest))
            (folder / "strata.exe").write_bytes(b"synthetic native binary")
            (folder / "strata-q36.exe").write_bytes(b"synthetic q36 binary")
        expected_files = o.release.inventory(signed)
        archive = assets / b.ENGINE
        archive.write_bytes(b"synthetic signed archive fixture")
        receipts = {name: root / (name + ".json") for name in o.INPUTS}
        receipts["component"].write_text(
            json.dumps({"signedComponent": str(signed), "files": expected_files})
        )
        for name in ("install", "original"):
            receipts[name].write_text("{}")
        args = SimpleNamespace(reviewed_commit=COMMIT, port=12345, choices_sha256="a" * 64)
        for name, path in receipts.items():
            setattr(args, name + "_receipt", path)
            setattr(args, name + "_receipt_sha256", o.release.sha(path))
        original = {
            "profileIdentity": {"effectiveHost": {"components": {"physical": str(components)}}},
            "profileIdentitySHA256": "b" * 64,
            "choicesSHA256": args.choices_sha256,
        }
        payload = {"strata-desktop.exe": {"bytes": 1, "sha256": "c" * 64}}
        installed = {
            "installed": True,
            "exactPayloadMatch": True,
            "payload": payload,
            "installerSHA256": "d" * 64,
        }
        imported = {
            "family": "qwen",
            "folder": o.release.NAME,
            "version": "0.6.0",
            "backend": "cuda",
            "variant": "cuda13",
            "platform": "windows-x86_64",
            "signed": True,
            "fits": True,
            "active": False,
            "build_id": COMMIT + "-hakua-runtimes-v1",
            "path": str(installed_path),
        }
        incorrect = {
            "family": "deepseek",
            "folder": "other",
            "version": "0.5.9",
            "backend": "hip",
            "variant": "other",
            "platform": "linux-x86_64",
            "signed": False,
            "fits": False,
            "build_id": "e" * 40 + "-hakua-runtimes-v1",
            "path": str(root / "foreign"),
        }
        if change in incorrect:
            imported[change] = incorrect[change]
        if change == "installed-inventory":
            (installed_path / "unexpected.dat").write_bytes(b"unexpected payload")
        if change == "signed-inventory":
            (signed / "unexpected.dat").write_bytes(b"changed pinned payload")
        if change == "native-binary":
            (installed_path / "strata.exe").write_bytes(b"wrong native binary")
        if change == "q36-binary":
            (installed_path / "strata-q36.exe").write_bytes(b"wrong q36 binary")
        self.events, self.use_count, self.used = [], 0, False
        reads = 0

        def read_api(_base: str, route: str) -> dict:
            nonlocal reads
            self.assertEqual(route, "/api/engines")
            reads += 1
            selected = deepcopy(imported)
            if self.used:
                selected["active"] = post_change != "inactive"
                if post_change == "identity":
                    selected["build_id"] = "f" * 40 + "-hakua-runtimes-v1"
            rows = [] if reads == 1 else [selected]
            if change == "duplicate" and reads == 2:
                rows.append(deepcopy(selected))
            return {
                "available": True,
                "serving": False,
                "families": [{"family": "qwen", "engines": rows}],
            }

        def call(_args: object, route: str, **kwargs: object) -> dict:
            if route == "/api/engine/from-file":
                self.events.append("import")
                self.assertEqual(kwargs["archive"], archive)
                return {
                    "id": "synthetic-owned-job",
                    "kind": "install-engine-file",
                    "state": "running",
                }
            self.assertEqual(route, "/api/engines/use")
            self.assertEqual(kwargs["body"], {"family": "qwen", "folder": o.release.NAME})
            self.events.append("use")
            self.use_count += 1
            self.used = True
            if post_change == "binary":
                (installed_path / "strata.exe").write_bytes(b"changed while selecting")
            return {"ok": True}

        def read_manifest(path: Path) -> object:
            self.assertEqual(path, installed_path)
            self.events.append("manifest:" + ("after" if self.used else "before"))
            if change == "manifest" and not self.used:
                raise ValueError("Synthetic malformed manifest")
            return SimpleNamespace(version="0.6.0", build_id=COMMIT + "-hakua-runtimes-v1")

        def verify_signature(*_args: object) -> None:
            self.events.append("signature:" + ("after" if self.used else "before"))
            if change == "signature" and not self.used:
                raise ValueError("Synthetic signature refusal")

        def verify_payload(_item: object, *, full: bool) -> None:
            self.assertIs(full, True)
            self.events.append("payload:" + ("after" if self.used else "before"))
            if change == "payload" and not self.used:
                raise ValueError("Synthetic full payload refusal")

        command = [
            str(o.HERE / "original-engine.py"),
            "--shell-pid",
            "10",
            "--shell-created",
            "100.0",
            "--host-pid",
            "11",
            "--host-created",
            "101.0",
            "--port",
            str(args.port),
        ]
        with ExitStack() as stack:
            for target, name, kwargs in (
                (o.normal, "original_baseline", {"return_value": original}),
                (o.delivery, "prior", {"return_value": installed}),
                (o.delivery, "payload", {"return_value": payload}),
                (
                    o.feed,
                    "receipt_assets",
                    {
                        "return_value": {
                            "assets": str(assets),
                            "entry": {"url": archive.name, "sha256": o.release.sha(archive)},
                        }
                    },
                ),
                (o, "stable", {}),
                (o, "call", {"side_effect": call}),
                (
                    o,
                    "wait_job",
                    {
                        "return_value": {
                            "id": "synthetic-owned-job",
                            "kind": "install-engine-file",
                            "state": "done",
                        }
                    },
                ),
                (o.normal, "read_api", {"side_effect": read_api}),
                (
                    o,
                    "helper_inputs",
                    {"return_value": {"fixture": o.artifact(receipts["original"])}},
                ),
                (engine_component, "read_manifest", {"side_effect": read_manifest}),
                (
                    engine_component,
                    "signing_payload",
                    {"return_value": b"synthetic manifest payload"},
                ),
                (engine_component, "verify", {"side_effect": verify_payload}),
                (
                    signing,
                    "load_trusted_keys",
                    {"return_value": {"fixture": b"synthetic public key"}},
                ),
                (signing, "verify_bytes", {"side_effect": verify_signature}),
                (sys, "argv", {"new": command}),
            ):
                stack.enter_context(patch.object(target, name, **kwargs))
            return o.phase(args, work)

    def test_each_incorrect_imported_identity_blocks_selection(self) -> None:
        for change in (
            "family",
            "folder",
            "duplicate",
            "version",
            "build_id",
            "backend",
            "variant",
            "platform",
            "signed",
            "fits",
            "path",
        ):
            with self.subTest(change=change), self.assertRaises((RuntimeError, ValueError)):
                self.exercise(change)
            self.assertEqual(self.use_count, 0, change)
            self.assertNotIn("use", self.events, change)

    def test_manifest_signature_inventory_and_both_binary_failures_block_selection(self) -> None:
        for change in (
            "manifest",
            "signature",
            "signed-inventory",
            "installed-inventory",
            "native-binary",
            "q36-binary",
            "payload",
        ):
            with self.subTest(change=change), self.assertRaises((RuntimeError, ValueError)):
                self.exercise(change)
            self.assertEqual(self.use_count, 0, change)
            self.assertNotIn("use", self.events, change)

    def test_success_verifies_all_bytes_and_signature_before_use_and_again_after(self) -> None:
        result = self.exercise()
        self.assertEqual(self.use_count, 1)
        self.assertEqual(
            self.events,
            [
                "import",
                "manifest:before",
                "signature:before",
                "payload:before",
                "use",
                "manifest:after",
                "signature:after",
                "payload:after",
            ],
        )
        self.assertEqual(result["schema"], o.SCHEMA)
        self.assertTrue(result["modelUnloaded"])

    def test_post_selection_still_requires_active_identity_and_unchanged_payload(self) -> None:
        for change in ("inactive", "identity", "binary"):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.exercise(post_change=change)
            self.assertEqual(self.use_count, 1, change)
            self.assertLess(self.events.index("signature:before"), self.events.index("use"))
            self.assertLess(self.events.index("payload:before"), self.events.index("use"))


class RemoteBoundaries(Scratch):
    def test_gh_metadata_is_captured_without_printing_credential_diagnostics(self) -> None:
        with patch.object(
            p.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout='{"id":1}\n')
        ) as child:
            self.assertEqual(p.api("releases/latest"), {"id": 1})
            self.assertEqual(
                child.call_args.args[0][1:], ["api", "repos/zapabob/Strata/releases/latest"]
            )
            self.assertIs(child.call_args.kwargs["stderr"], subprocess.DEVNULL)
        with patch.object(
            p.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")
        ), self.assertRaises(RuntimeError):
            p.gh("api", "user", "--jq", ".login")

    def test_resume_accepts_only_missing_assets_and_exact_existing_bytes(self) -> None:
        expected = {"a.exe": row(b"a"), "feed.json": row(b"feed")}
        existing = server({"a.exe": expected["a.exe"]})
        self.assertEqual(p.remote_assets(existing, expected, complete=False), {"feed.json"})
        for field, value in (("digest", "sha256:" + "a" * 64), ("size", 99), ("state", "new")):
            changed = deepcopy(existing)
            changed["assets"][0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                p.remote_assets(changed, expected, complete=False)
        for value in (
            server({"extra": row(b"x")}),
            server({}),
            {**existing, "assets": existing["assets"] * 2},
        ):
            with self.assertRaises(RuntimeError):
                p.remote_assets(value, expected)

    def test_existing_immutable_tag_refused_before_git_mutation(self) -> None:
        with patch.object(p, "local_ref", return_value="b" * 40), patch.object(
            p, "git_mutate", side_effect=AssertionError("mutation")
        ), self.assertRaises(RuntimeError):
            p.tag_phase(SimpleNamespace(reviewed_commit=COMMIT), self.root)

    def test_remote_tag_requires_annotation_and_exact_commit(self) -> None:
        obj = "b" * 40
        with patch.object(
            p, "tags", return_value={"refs/tags/" + p.TAG: {"type": "tag", "sha": obj}}
        ), patch.object(
            p, "api", return_value={"tag": p.TAG, "object": {"type": "commit", "sha": "c" * 40}}
        ), self.assertRaises(
            RuntimeError
        ):
            p.remote_tag(COMMIT)
        with patch.object(
            p, "tags", return_value={"refs/tags/" + p.TAG: {"type": "commit", "sha": COMMIT}}
        ), self.assertRaises(RuntimeError):
            p.remote_tag(COMMIT)

    def test_alias_lease_drift_refused_before_any_asset_deletion(self) -> None:
        path = self.root / "snapshot.json"
        path.write_text(
            json.dumps({"tags": {"refs/tags/latest": {"sha": "a" * 40}}, "localAlias": "a" * 40})
        )
        snapshot = {"results": {"snapshot": {"path": str(path)}}}
        with patch.object(p, "remote_tag", return_value={"object": "b" * 40}), patch.object(
            p, "tags", return_value={"refs/tags/latest": {"sha": "c" * 40}}
        ), patch.object(p, "gh", side_effect=AssertionError("mutation")), self.assertRaisesRegex(
            RuntimeError, "lease changed"
        ):
            p.alias_phase(SimpleNamespace(reviewed_commit=COMMIT), self.root, {}, snapshot, {})

    def test_historical_release_changes_refused(self) -> None:
        path = self.root / "snapshot.json"
        before = server({"old.exe": row(b"old")})
        before["tag_name"] = "v0.5.9"
        path.write_text(json.dumps({"releases": {"v0.5.9": p.identity(before)}, "tags": {}}))
        changed = deepcopy(before)
        changed["assets"][0]["digest"] = "sha256:" + "0" * 64
        with patch.object(p, "releases", return_value={"v0.5.9": changed}), self.assertRaisesRegex(
            RuntimeError, "Historical release"
        ):
            p.unchanged({"results": {"snapshot": {"path": str(path)}}})

    def test_anonymous_stream_rejects_wrong_content_and_overrun(self) -> None:
        class Response(io.BytesIO):
            status = 200
            headers: dict = {}

        args = SimpleNamespace(allow_low_ram=True)
        for content, accepted in ((b"good", True), (b"evil", False), (b"good-plus-extra", False)):
            target = self.root / ("result-" + str(len(content)) + ("yes" if accepted else "no"))
            with patch.object(p, "public_open", return_value=Response(content)), patch.object(
                p.release,
                "admit",
                return_value={"available_ram": 1, "commit_headroom": 2, "H_free": 3, "C_free": 4},
            ):
                if accepted:
                    self.assertTrue(
                        p.download(args, "unused", target, row(b"good"), io.StringIO())[
                            "unauthenticated"
                        ]
                    )
                else:
                    with self.assertRaises(RuntimeError):
                        p.download(args, "unused", target, row(b"good"), io.StringIO())

    def test_public_client_rejects_credentials_http_and_foreign_redirects(self) -> None:
        for url in (
            "http://github.com/x",
            "https://person:secret@github.com/x",
            "https://evil.example/x",
        ):
            with self.subTest(url=url), self.assertRaises(RuntimeError):
                p.public_url(url)
        p.public_url("https://release-assets.githubusercontent.com/path?signature=public-download")
        with patch.object(p.urllib.request, "build_opener") as builder:
            builder.return_value.open.return_value = "sentinel"
            self.assertEqual(
                p.public_open("https://github.com/zapabob/Strata/releases/download/v/x"), "sentinel"
            )
            request = builder.return_value.open.call_args.args[0]
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn("Cookie", request.headers)

    def test_wrong_source_and_incomplete_content_outcomes_rejected(self) -> None:
        expected = {"a.exe": row(b"a")}
        value = {
            "tag": {"commit": COMMIT, "object": "b" * 40},
            "releaseId": 10,
            "releaseIdentity": p.identity({**server(expected), "draft": False}),
            "complete": True,
            "anonymousBytes": {
                tag: {n: {**r, "unauthenticated": True} for n, r in expected.items()}
                for tag in (p.TAG, "latest")
            },
        }
        p.validate_outcome("readback", value, {"assets": expected}, {}, COMMIT)
        for change in ("source", "tag", "bytes", "alias", "auth"):
            bad = deepcopy(value)
            if change == "source":
                bad["tag"]["commit"] = "c" * 40
            elif change == "tag":
                bad["releaseIdentity"]["tag_name"] = "v0.5.9"
            elif change == "bytes":
                bad["anonymousBytes"][p.TAG]["a.exe"]["sha256"] = "d" * 64
            elif change == "alias":
                bad["anonymousBytes"].pop("latest")
            else:
                bad["anonymousBytes"]["latest"]["a.exe"]["unauthenticated"] = False
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                p.validate_outcome("readback", bad, {"assets": expected}, {}, COMMIT)


class DefaultInert(unittest.TestCase):
    def test_real_cli_defaults_are_readonly_for_every_phase(self) -> None:
        for script, argv in (
            ("bundle.py", []),
            ("original-engine.py", []),
            *(("publish.py", [action]) for action in p.ACTIONS),
        ):
            with self.subTest(script=script, argv=argv):
                result = subprocess.run(
                    [sys.executable, "-B", str(HERE / script), *argv],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('"execute": false', result.stderr + result.stdout)

    def test_default_path_never_calls_review_or_mkdir_or_network(self) -> None:
        with patch.object(
            b.release, "reviewed", side_effect=AssertionError("reviewed execution")
        ), patch.object(Path, "mkdir", side_effect=AssertionError("mutation")), patch.object(
            b.release, "command", side_effect=AssertionError("child")
        ):
            self.assertEqual(b.main([]), 0)
        with patch.object(
            p.release, "reviewed", side_effect=AssertionError("reviewed execution")
        ), patch.object(Path, "mkdir", side_effect=AssertionError("mutation")), patch.object(
            p, "gh", side_effect=AssertionError("network")
        ):
            for action in p.ACTIONS:
                self.assertEqual(p.main([action]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
