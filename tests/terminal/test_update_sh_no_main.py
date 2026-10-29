"""A failed update on a fork without main is not an old-history recovery."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipIf(os.name == "nt" or not shutil.which("git") or not shutil.which("sh"), "requires git and POSIX sh")
class UpdateWithoutMainTests(unittest.TestCase):
    def test_diverged_fork_without_main_stops_before_setup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
            env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", LC_ALL="C",
                       GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
                       GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid")

            def git(cwd, *args):
                return subprocess.run(["git", *map(str, args)], cwd=cwd, env=env, check=True,
                                      capture_output=True, text=True, timeout=10).stdout

            origin = root / "origin.git"
            clone = root / "clone"
            upstream = root / "upstream"
            git(root, "init", "--bare", origin)
            git(origin, "symbolic-ref", "HEAD", "refs/heads/memory-core")
            git(root, "clone", origin, clone)
            shutil.copyfile(ROOT / "update.sh", clone / "update.sh")
            (clone / "setup.sh").write_text("#!/bin/sh\ntouch setup-ran\nexit 99\n", encoding="utf-8")
            git(clone, "add", ".")
            git(clone, "commit", "-m", "Initial updater")
            git(clone, "push", "-u", "origin", "memory-core")
            git(root, "clone", origin, upstream)
            (upstream / "remote.txt").write_text("remote\n", encoding="utf-8")
            git(upstream, "add", ".")
            git(upstream, "commit", "-m", "Remote change")
            git(upstream, "push")
            (clone / "local.txt").write_text("local\n", encoding="utf-8")
            git(clone, "add", ".")
            git(clone, "commit", "-m", "Local change")
            self.assertEqual(git(origin, "for-each-ref", "--format=%(refname)", "refs/heads").strip(),
                             "refs/heads/memory-core")

            r = subprocess.run(["sh", "update.sh"], cwd=clone, env=env,
                               capture_output=True, text=True, timeout=10)
            output = r.stdout + r.stderr
            self.assertEqual(r.returncode, 1, output)
            self.assertNotIn("old history", output)
            self.assertIn("git pull did not succeed", output)
            self.assertEqual(git(clone, "branch", "--list", "pre-cleanup-backup*"), "")
            self.assertFalse((clone / "setup-ran").exists())


if __name__ == "__main__":
    unittest.main()
