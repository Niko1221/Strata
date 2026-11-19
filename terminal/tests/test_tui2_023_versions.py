"""0.2.3 RC stress test, #3 (H01): the version bump is complete and consistent, and a 0.2.2 install sees the update."""

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import strata_cli
from strata_control import feed as feedmod, updater
from tests.terminal import test_updater

ROOT = Path(__file__).resolve().parents[2]


def first(path, pattern):
    return re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.MULTILINE).group(1)


class VersionBump(unittest.TestCase):
    def test_product_control_and_engine_versions_are_aligned(self):
        # The user requested a coordinated 0.5.8 SemVer release of all four.
        # Upstream reference versions remain independently recorded.
        product = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual(strata_cli.PRODUCT_VERSION, product)
        self.assertEqual(strata_cli.version(), (ROOT / "CONTROL_VERSION").read_text(encoding="utf-8").strip())
        self.assertEqual(strata_cli.version(), product)
        self.assertEqual(first("CMakeLists.txt", r"^project\(strata VERSION ([0-9.]+) "), product)
        self.assertIn(f'STRATA_VERSION="{product}"', (ROOT / "metal" / "CMakeLists.txt").read_text(encoding="utf-8"))

    def test_all_app_version_sources_agree(self):
        product = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual(first("apps/desktop/src-tauri/Cargo.toml", r'^version = "([^"]+)"'), product)
        lock = (ROOT / "apps/desktop/src-tauri/Cargo.lock").read_text(encoding="utf-8")
        self.assertRegex(lock, rf'name = "strata-desktop"\r?\nversion = "{re.escape(product)}"')
        self.assertEqual(json.loads((ROOT / "apps/desktop/package.json").read_text(encoding="utf-8"))["version"], product)
        host = json.loads((ROOT / "apps/host/package.json").read_text(encoding="utf-8"))
        self.assertEqual(host["version"], product)
        locked = json.loads((ROOT / "apps/host/package-lock.json").read_text(encoding="utf-8"))
        self.assertEqual((locked["version"], locked["packages"][""]["version"]), (product, product))
        desk_lock = json.loads((ROOT / "apps/desktop/package-lock.json").read_text(encoding="utf-8"))
        self.assertEqual((desk_lock["version"], desk_lock["packages"][""]["version"]), (product, product))

    def test_the_documents_that_state_the_current_version_agree(self):
        product = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn(f"Hakua product version {product}", agents)
        self.assertIn(f"engine {product}", agents)
        self.assertIn("0.2.13", agents)
        for name in ("README.md", "README.de.md", "README.es.md", "README.fr.md", "README.ja.md", "README.pl.md", "README.pt-BR.md", "README.zh-CN.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            # Every fork README links to this product, rather than an upstream installer.
            self.assertIn(f"Strata-{product}-windows-x64-setup.exe", text, name)
            self.assertNotIn("Strata-0.2.2-", text, name)

    def test_version_prints_the_product_and_the_build(self):
        product = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        done = subprocess.run([sys.executable, str(ROOT / "strata.py"), "--version"], capture_output=True, text=True, cwd=str(ROOT),
                              env={**os.environ, "STRATA_NO_HANDOVER": "1"})
        self.assertTrue(done.stdout.startswith(f"strata {product} (build {strata_cli.version()};"), done.stdout)


class UpdateFromTheLastRelease(test_updater.UpdaterCase):
    def feed_with_control(self, version):
        bundle = self.root / "control-bundle"
        (bundle / "strata_control").mkdir(parents=True)
        (bundle / "strata.py").write_text(f"print('strata 0.2.3 (build {version})')\n")
        (bundle / "strata_control" / "__init__.py").write_text("")
        from feed_tool import build_feed, pack_control
        (self.release / "control").mkdir()
        pack_control(bundle, self.release / "control", version=version, platform=test_updater.PLATFORM)
        build_feed(self.release, key=self.key, sequence=1, control_dir=self.release / "control")

    def test_a_0_2_2_install_is_offered_control_0_1_48_and_a_0_2_3_install_is_current(self):
        self.feed_with_control("0.1.48")
        (self.components / "control").mkdir(parents=True)
        (self.components / "control" / "active.json").write_text(json.dumps({"version": "0.1.47", "folder": "0.1.47"}))
        report = updater.check(self.ctx())
        self.assertEqual((report["control"]["status"], report["control"]["installed"], report["control"]["offered"]),
                         ("available", "0.1.47", "0.1.48"))
        out = updater.run_update(self.ctx(), engine=False, control=True)
        self.assertEqual((out["exit_code"], out["results"][0]["status"]), (0, "updated"), out)
        self.assertEqual(updater.active_control(self.components)["version"], "0.1.48")
        self.assertEqual(updater.check(self.ctx())["control"]["status"], "current")

    def test_the_bumped_numbers_sort_above_the_last_release(self):
        self.assertGreater(feedmod.version_key(strata_cli.version()), feedmod.version_key("0.1.47"))
        self.assertGreater(feedmod.version_key("0.1.48"), feedmod.version_key("0.1.46.3"))
        self.assertGreater(feedmod.version_key("0.1.48"), feedmod.version_key("0.1.45.3"))
