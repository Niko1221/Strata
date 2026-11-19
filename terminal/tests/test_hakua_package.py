"""The pinned Hakua wheel must be selected locally even with an index enabled."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import package_terminal as pt


class HakuaPackage(unittest.TestCase):
    def test_locked_install_uses_the_exact_vendor_wheel_and_retains_hash_check(self):
        lock = pt.lock_for("windows-x86_64")
        expected = pt.lock_names(lock)
        invocations = []

        def run(command, **kwargs):
            invocations.append(command)
            if "install" in command:
                resolved = Path(command[command.index("-r") + 1]).read_text(encoding="utf-8")
                vendor = ROOT / "packaging/vendor/hakua-memory/hakua_memory-0.3.5-py3-none-any.whl"
                self.assertIn("hakua-memory @ " + vendor.as_uri(), resolved)
                self.assertIn("--hash=sha256:" + hashlib.sha256(vendor.read_bytes()).hexdigest(), resolved)
                self.assertIn("--require-hashes", command)
                self.assertIn("--no-deps", command)
            output = json.dumps([{"name": name, "version": version} for name, version in expected.items()])
            return types.SimpleNamespace(stdout=output)

        with tempfile.TemporaryDirectory() as directory, patch.object(pt.subprocess, "run", side_effect=run):
            installed = pt.install_locked(Path("bundled-python"), lock, Path(directory))
        self.assertEqual(installed["hakua-memory"], "0.3.5")
        self.assertEqual(len(invocations), 3)  # install, dependency check, exact inventory

    def test_public_bundle_drops_local_origin_path_and_record_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "python/Lib/site-packages/hakua_memory-0.3.5.dist-info"
            meta.mkdir(parents=True)
            (meta / "direct_url.json").write_text('{"url":"file:///private-build/wheel.whl"}')
            (meta / "RECORD").write_text("hakua_memory-0.3.5.dist-info/direct_url.json,sha256=old,12\nkeep.py,sha256=keep,5\n")
            removed = pt.drop_build_path_launchers(root, root, "windows-x86_64")
            self.assertFalse((meta / "direct_url.json").exists())
            self.assertEqual((meta / "RECORD").read_text().strip(), "keep.py,sha256=keep,5")
            self.assertTrue(any(name.endswith("direct_url.json") for name in removed))
