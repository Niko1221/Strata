"""CPU-only file installation fixtures; never execute a packaged binary."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import install_preview_overlay as install


class OverlayTests(unittest.TestCase):
    def fixture(self):
        root = Path(tempfile.mkdtemp(prefix="preview-overlay-", dir=os.environ["TEMP"]))
        stage = root / "stage"
        stage.mkdir()
        payload = {"strata.exe": b"CPU fixture, not executable", "control/serve/server.py": b"new server",
                   "control/serve/perf131k_e2e.py": b"new observer",
                   "control/serve/memory_harness.py": b"new memory helper"}
        for name, data in payload.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            if name == 'strata.exe':
                path.chmod(0o755)  # inert bytes; POSIX verification still checks the file mode
        component = {"schema_version": 2, "id": "strata-engine", "version": "0.3.2-preview.1",
                     "platform": "windows-x86_64", "backend": "cuda", "entrypoint": "strata.exe",
                     "archs": ["sm_120"], "build_id": "CPU-fixture",
                     "files": [{"path": name, "sha256": install.sha(stage / name), "bytes": len(data)}
                               for name, data in payload.items()]}
        (stage / "engine-component.json").write_text(json.dumps(component))
        roots = [root / "bundle", root / "component"]
        for path in roots:
            path.mkdir()
            (path / "old-version.txt").write_text("preserve")
        control = root / "control"
        (control / "serve").mkdir(parents=True)
        (control / "serve/server.py").write_bytes(b"original server")
        (control / "VERSION").write_text("original version")
        expected = {"server.py": install.sha(control / "serve/server.py"), "perf131k_e2e.py": None,
                    "memory_harness.py": None}
        return root, stage, roots, control, expected

    def test_install_preserves_old_engines_and_whole_control_version(self):
        root, stage, roots, control, expected = self.fixture()
        result = install.apply(stage, roots, control, root / "backup", expected)
        self.assertEqual(result["status"], "INSTALLED_VERIFIED_FILES")
        for path in roots:
            self.assertEqual((path / "old-version.txt").read_text(), "preserve")
            self.assertEqual(install.sha(Path(result["components"][roots.index(path)]) / "strata.exe"),
                             install.sha(stage / "strata.exe"))
        self.assertEqual((control / "VERSION").read_text(), "original version")
        self.assertEqual((root / "backup/server.py").read_bytes(), b"original server")

    def test_control_drift_prevents_every_install_write(self):
        root, stage, roots, control, expected = self.fixture()
        (control / "serve/server.py").write_bytes(b"concurrent edit")
        with self.assertRaises(ValueError):
            install.apply(stage, roots, control, root / "backup", expected)
        self.assertEqual(list(roots[0].iterdir()), [roots[0] / "old-version.txt"])
        self.assertEqual((control / "serve/server.py").read_bytes(), b"concurrent edit")

    def test_server_is_replaced_only_after_both_imported_helpers(self):
        root, stage, roots, control, expected = self.fixture()
        copy = install.copy_checked
        observed = []
        def check_order(source, target, digest):
            if source == stage / "control/serve/server.py" and target.parent == control / "serve":
                self.assertEqual((control / "serve/memory_harness.py").read_bytes(), b"new memory helper")
                self.assertEqual((control / "serve/perf131k_e2e.py").read_bytes(), b"new observer")
                observed.append(True)
            return copy(source, target, digest)
        with patch.object(install, "copy_checked", side_effect=check_order):
            install.apply(stage, roots, control, root / "backup", expected)
        self.assertEqual(observed, [True])

    def test_missing_memory_payload_blocks_before_backup_or_install(self):
        root, stage, roots, control, expected = self.fixture()
        manifest = stage / "engine-component.json"
        value = json.loads(manifest.read_text())
        value["files"] = [row for row in value["files"] if row["path"] != "control/serve/memory_harness.py"]
        manifest.write_text(json.dumps(value))
        (stage / "control/serve/memory_harness.py").unlink()
        with self.assertRaises(ValueError):
            install.apply(stage, roots, control, root / "backup", expected)
        self.assertFalse((root / "backup").exists())
        self.assertEqual((control / "serve/server.py").read_bytes(), b"original server")

    def test_existing_memory_helper_is_restored_on_publication_failure(self):
        root, stage, roots, control, expected = self.fixture()
        helper = control / "serve/memory_harness.py"
        helper.write_bytes(b"original memory helper")
        expected["memory_harness.py"] = install.sha(helper)
        with patch.object(install, "publish", side_effect=OSError("injected publish failure")):
            with self.assertRaises(OSError):
                install.apply(stage, roots, control, root / "backup", expected)
        self.assertEqual(helper.read_bytes(), b"original memory helper")
        self.assertEqual((control / "serve/server.py").read_bytes(), b"original server")

    def test_publish_failure_restores_control_and_hides_new_components(self):
        root, stage, roots, control, expected = self.fixture()
        original = install.publish
        calls = []
        def fail_second(source, target):
            calls.append(target)
            if len(calls) == 2:
                raise OSError("injected second-root publish failure")
            original(source, target)
        with patch.object(install, "publish", side_effect=fail_second):
            with self.assertRaises(OSError):
                install.apply(stage, roots, control, root / "backup", expected)
        self.assertEqual((control / "serve/server.py").read_bytes(), b"original server")
        for path in roots:
            self.assertFalse(any(p.is_dir() and not p.name.startswith(".") for p in path.iterdir()))
        self.assertEqual(json.loads((root / "backup/result.json").read_text())["status"], "FAILED_ROLLED_BACK_EXISTING_FILES")

    def test_corrupt_stage_is_rejected(self):
        root, stage, roots, control, expected = self.fixture()
        (stage / "strata.exe").write_bytes(b"tampered")
        with self.assertRaises(Exception):
            install.apply(stage, roots, control, root / "backup", expected)
        self.assertFalse((root / "backup").exists())

    def test_existing_preview_is_not_overwritten(self):
        root, stage, roots, control, expected = self.fixture()
        old = roots[0] / "0.3.2-preview.1-windows-x86_64-cuda-cuda13"
        old.mkdir()
        (old / "strata.exe").write_bytes(b"existing preview")
        with self.assertRaises(ValueError):
            install.apply(stage, roots, control, root / "backup", expected)
        self.assertEqual((old / "strata.exe").read_bytes(), b"existing preview")
        self.assertFalse((root / "backup").exists())

    def test_required_signature_policy_is_not_bypassed(self):
        root, stage, roots, control, expected = self.fixture()
        with patch.dict(os.environ, STRATA_REQUIRE_SIGNED_COMPONENTS="1"):
            with self.assertRaises(Exception):
                install.apply(stage, roots, control, root / "backup", expected)
        self.assertFalse((root / "backup").exists())

    def test_manifest_drift_after_signature_check_never_reaches_control(self):
        root, stage, roots, control, expected = self.fixture()
        original = install.engine_component.verify_signature
        def change_after_verification(component):
            status = original(component)
            path = stage / "engine-component.json"
            metadata = json.loads(path.read_text())
            metadata["notes"] = "changed after initial verification"
            path.write_text(json.dumps(metadata))
            return status
        with patch.object(install.engine_component, "verify_signature", side_effect=change_after_verification):
            with self.assertRaises(ValueError):
                install.apply(stage, roots, control, root / "backup", expected)
        self.assertEqual((control / "serve/server.py").read_bytes(), b"original server")
        self.assertFalse((control / "serve/perf131k_e2e.py").exists())
        for path in roots:
            self.assertFalse(any(p.is_dir() and not p.name.startswith(".") for p in path.iterdir()))

    def test_metadata_drift_at_publish_rolls_back_original_control(self):
        root, stage, roots, control, expected = self.fixture()
        original = install.publish
        def change_at_publish(source, target):
            original(source, target)
            manifest = target / "engine-component.json"
            value = json.loads(manifest.read_text())
            value["notes"] = "changed at publication"
            manifest.write_text(json.dumps(value))
        with patch.object(install, "publish", side_effect=change_at_publish):
            with self.assertRaises(ValueError):
                install.apply(stage, roots, control, root / "backup", expected)
        self.assertEqual((control / "serve/server.py").read_bytes(), b"original server")
        for path in roots:
            self.assertFalse(any(p.is_dir() and not p.name.startswith(".") for p in path.iterdir()))


if __name__ == "__main__":
    unittest.main()
