"""docker-entrypoint.sh keeps the install config on the /data volume as the one that counts (#1244): with REINSTALL=0
and a config on the volume, /opt/strata/strata-<tag>.json is a link to it, even when a regular file is already there.
Runs the real script in a temporary tree with a stub for .venv/bin/python (POSIX sh and symlinks needed: skipped on
Windows).

    python -m unittest tools.test_docker_entrypoint
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = shutil.which("sh")


@unittest.skipIf(os.name == "nt" or SH is None, "needs POSIX sh and symlinks")
class Entrypoint(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.opt = self.tmp / "opt"
        self.data = self.tmp / "data"
        (self.opt / ".venv" / "bin").mkdir(parents=True)
        (self.data / "config").mkdir(parents=True)
        # the stub "python": setup.py --setup writes the config the way setup does; a plain start prints what it would load
        stub = self.opt / ".venv" / "bin" / "python"
        stub.write_text('#!/bin/sh\nif [ "$2" = "--setup" ]; then echo \'{"args": ["from-setup"]}\' > strata-iq3_s.json\n'
                        'else echo "STARTED WITH: $(cat strata-iq3_s.json)"; fi\n', encoding="utf-8")
        stub.chmod(0o755)
        script = (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8").replace("/opt/strata", str(self.opt))
        self.script = self.tmp / "entry.sh"
        self.script.write_text(script, encoding="utf-8")

    def run_entry(self, reinstall, **extra):
        env = dict(os.environ, STRATA_DATA=str(self.data), MODEL="IQ3_S", REINSTALL=reinstall, **extra)
        r = subprocess.run([SH, str(self.script)], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_edited_volume_config_wins_over_a_regular_file_in_opt(self):
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["edited", "--kv-resident", "32768"]}\n')
        (self.opt / "strata-iq3_s.json").write_text('{"args": ["stale"]}\n')      # left by an earlier setup
        out = self.run_entry("0")
        self.assertIn("edited", out)
        self.assertNotIn("stale", out)
        self.assertTrue((self.opt / "strata-iq3_s.json").is_symlink())

    def test_link_is_made_when_there_is_no_file(self):
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["on-volume"]}\n')
        self.assertIn("on-volume", self.run_entry("0"))

    def test_first_setup_copies_the_config_to_the_volume_and_the_next_start_links_it(self):
        out = self.run_entry("0")                                    # no config on the volume: setup runs
        self.assertIn("from-setup", out)
        cfg = self.data / "config" / "strata-iq3_s.json"
        self.assertIn("from-setup", cfg.read_text())
        cfg.write_text('{"args": ["edited-later"]}\n')
        self.assertIn("edited-later", self.run_entry("0"))           # the regular file setup left is replaced by the link

    def test_existing_link_into_the_data_config_dir_is_kept(self):      # a pod command that picked a config by linking
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["plain"]}\n')
        other = self.data / "config" / "strata-iq3_s.1x4.json"
        other.write_text('{"args": ["batch8"]}\n')
        os.symlink(other, self.opt / "strata-iq3_s.json")
        out = self.run_entry("0")
        self.assertIn("batch8", out)
        self.assertNotIn("plain", out)
        self.assertIn(f"Config: {other}", out)

    def test_link_to_somewhere_else_is_replaced(self):
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["on-volume"]}\n')
        elsewhere = self.tmp / "elsewhere.json"
        elsewhere.write_text('{"args": ["elsewhere"]}\n')
        os.symlink(elsewhere, self.opt / "strata-iq3_s.json")
        self.assertIn("on-volume", self.run_entry("0"))

    def test_config_env_wins(self):
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["plain"]}\n')
        other = self.tmp / "mine.json"
        other.write_text('{"args": ["mine"]}\n')
        os.symlink(self.data / "config" / "strata-iq3_s.json", self.opt / "strata-iq3_s.json")
        out = self.run_entry("0", CONFIG=str(other))
        self.assertIn("mine", out)
        self.assertNotIn("plain", out)
        self.assertIn(f"Config: {other}", out)

    def test_model_names_the_config_and_it_is_printed(self):
        (self.data / "config" / "strata-iq3_s.json").write_text('{"args": ["by-model"]}\n')
        out = self.run_entry("0")
        self.assertIn("by-model", out)
        self.assertIn(f"Config: {self.data}/config/strata-iq3_s.json", out)


if __name__ == "__main__":
    unittest.main()
