"""Direct adapter regressions; CPU only, no production keys or inference."""

import argparse
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

sys.dont_write_bytecode = True


def module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), Path(__file__).with_name(name + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class AdapterTests(unittest.TestCase):
    def test_cmake_real_blank_and_comment_layout(self) -> None:
        native = module("native-input")
        text = """# Cache header

//portable build
STRATA_PORTABLE:BOOL=ON

//CUDA operations
STRATA_ENABLE_CUDA:BOOL=ON

CMAKE_CUDA_ARCHITECTURES:STRING=120
//A comment contains :STRING=ignored
CMAKE_HOME_DIRECTORY:INTERNAL=C:/work
"""
        self.assertEqual(
            native.cache_values(text),
            {
                "STRATA_PORTABLE": "ON",
                "STRATA_ENABLE_CUDA": "ON",
                "CMAKE_CUDA_ARCHITECTURES": "120",
                "CMAKE_HOME_DIRECTORY": "C:/work",
            },
        )

    def test_dependency_receipt_cannot_replace_phase_source_identity(self) -> None:
        release = module("release")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("harness", "local-backends"):
                (root / name).mkdir()
                (root / name / "fixture").write_text("verified")
            args = argparse.Namespace(public_assets_receipt_sha256="a" * 64)
            with patch.object(release, "PRIMARY_RES", root), patch.object(
                release, "RES", root
            ), patch.object(release, "verify"):
                result = release.dependencies(args, root)
            identity = {"sourceRoot": "C:/actual-checkout", "sourceCommit": "b" * 40}
            merged = {**identity, **result}
            self.assertEqual(merged["sourceRoot"], identity["sourceRoot"])
            self.assertEqual(result["dependencySourceRoot"], str(root))

    def test_frozen_q38_staging_is_portable_and_pins_compiled_source(self) -> None:
        native = module("native-input")
        with tempfile.TemporaryDirectory() as temp:
            record = native.Q38.stage(native.release, native.PRIMARY, Path(temp))
            self.assertEqual(record["sourceCommit"], native.Q38.COMMIT)
            self.assertEqual(len(record["sources"]), 387)
            self.assertEqual(record["sha256"], native.Q38.EXECUTABLE_SHA256)
            output = native.release.command([Path(temp) / "strata-q38.exe", "--version"], timeout=20)
            self.assertEqual(output.strip(), "strata-q38 own-native 0.6.0 CUDA13 SM120")
            native.validate_version(output)
            self.assertEqual(
                record["requiredLibraries"],
                ["MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll"],
            )
            self.assertEqual(
                record["recipe"]["embeddedProfileSHA256"],
                record["sources"]["data/spin8/reference.profile"],
            )
            text = json.dumps(record)
            for private in ("C:/Users", "C:\\\\Users", "H:/", "H:\\\\", "third_party/strata-llama"):
                self.assertNotIn(private, text)
            self.assertEqual(
                {p.name for p in Path(temp).iterdir()},
                {
                    "strata-q38.exe",
                    "q38-build.json",
                    *native.Q38.NOTICES.values(),
                    *record["requiredLibraries"],
                },
            )
            # Run the real JS consumer against the actual frozen PE/CRT payload.
            manifest = {
                "version": "0.6.0",
                "files": [{"path": p.name} for p in Path(temp).iterdir()],
            }
            (Path(temp) / "engine-component.json").write_text(json.dumps(manifest))
            expression = "import{verifyQ38Build}from './apps/host/scripts/release-integrity.mjs';import{readFileSync}from'node:fs';const folder=process.argv[1];verifyQ38Build(folder,process.cwd(),JSON.parse(readFileSync(folder+'/engine-component.json')));"
            result = subprocess.run(
                [str(native.release.NODE), "--input-type=module", "-e", expression, temp],
                cwd=native.PRIMARY,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            sys.path.insert(0, str(native.PRIMARY / "packaging"))
            import binary_deps

            self.assertEqual(binary_deps.uncovered(Path(temp)), [])

    def test_real_signed_parent_resolves_q38_and_rejects_changed_sibling(self) -> None:
        from terminal.strata_control import engine_component as ec, signing
        from tests.terminal.test_engine_component import make_package, rewrite

        key = signing.generate_key()
        trust = {key["key_id"]: base64.b64decode(key["public"])}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = "strata-q38.exe" if sys.platform == "win32" else "strata-q38"
            package = make_package(
                root, extra={name: b"covered q38", "q38-build.json": b"covered provenance"}
            )
            manifest = json.loads((package / ec.MANIFEST).read_text())
            rewrite(package, signature=signing.sign_bytes(key, ec.signing_payload(manifest)))
            with patch.object(ec, "_verifier", ec._verifier):
                signing.install_component_verifier(trust)
                self.assertEqual(
                    Path(
                        ec.expand(
                            "component:strata-engine",
                            {"native_runner": "qwen38_dense", "backend": "cuda"},
                            root=root,
                            full=False,
                        )
                    ),
                    package / name,
                )
                (package / "q38-build.json").write_bytes(b"changed provenance")
                with self.assertRaises(ec.EngineMissing):
                    ec.expand(
                        "component:strata-engine",
                        {"native_runner": "qwen38_dense", "backend": "cuda"},
                        root=root,
                        full=False,
                    )

    def test_altered_manifest_fails_before_any_payload_copy(self) -> None:
        native = module("native-input")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "final-build-manifest.json").write_text("{}")
            with patch.object(native.Q38, "EVIDENCE", root), self.assertRaisesRegex(
                RuntimeError, "manifest changed"
            ):
                native.Q38.stage(native.release, native.PRIMARY, root)
            self.assertFalse((root / "strata-q38.exe").exists())


if __name__ == "__main__":
    unittest.main()
