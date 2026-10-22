"""Tests for the Mac's double-click launchers (START-HERE.command, SETUP.command, UPDATE.command, BENCH.command: the
.bat files' counterparts, opened from Finder) and for setup.sh's Mac check: they run the right script from their own
folder with the options passed on, keep the Terminal window open on an error, refuse to run off a Mac, and setup.sh
stops an Intel Mac before it downloads anything. Runs on any PC: uname / sysctl / curl are stubs on PATH, the scripts
they run are stubs too. No GPU, no network.

    python -m unittest tools.test_mac_launchers
"""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ["START-HERE.command", "SETUP.command", "UPDATE.command", "BENCH.command"]


def stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


class Folder:
    """A copy of the launchers beside stub scripts, with a fake uname (and sysctl) first on PATH."""

    def __init__(self, system="Darwin", machine="arm64", arm64_chip="1"):
        self.dir = Path(tempfile.mkdtemp())
        for name in LAUNCHERS:
            (self.dir / name).write_bytes((ROOT / name).read_bytes())
            (self.dir / name).chmod(0o755)
        self.log = self.dir / "calls.log"
        for script in ("setup.sh", "update.sh", "bench.sh"):
            stub(self.dir / script, f'echo "{script} $*" >> "{self.log}"\nexit "${{STUB_RC:-0}}"\n')
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        stub(self.bin / "uname", f'case "$1" in -m) echo {machine} ;; *) echo {system} ;; esac\n')
        stub(self.bin / "sysctl", f'echo {arm64_chip}\n')
        stub(self.bin / "curl", 'echo "curl must not run" >&2\nexit 99\n')

    def run(self, launcher: str, *args: str, rc: int = 0, cwd: Path | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ, PATH=f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}", STUB_RC=str(rc))
        return subprocess.run(["sh", str(self.dir / launcher), *args], cwd=cwd or Path.home(), env=env,
                              capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []


class Files(unittest.TestCase):
    def test_each_launcher_is_an_executable_lf_sh_script(self):
        for name in LAUNCHERS:
            with self.subTest(name):
                p = ROOT / name
                raw = p.read_bytes()
                self.assertTrue(raw.startswith(b"#!/bin/sh\n"), name)
                self.assertNotIn(b"\r", raw, f"{name} must keep LF line endings (macOS sh reads it)")
                self.assertTrue(p.stat().st_mode & stat.S_IXUSR, f"{name} must be executable (Finder runs it)")
                self.assertEqual(subprocess.run(["sh", "-n", str(p)], capture_output=True).returncode, 0, name)

    def test_gitattributes_keeps_the_launchers_lf_on_every_checkout(self):
        self.assertIn("*.command text eol=lf", (ROOT / ".gitattributes").read_text(encoding="utf-8"))

    def test_the_docs_name_the_launcher(self):
        for doc in ("README.md", "README.zh-CN.md", "docs/MAC.md", "docs/INSTALL.md"):
            with self.subTest(doc):
                self.assertIn("START-HERE.command", (ROOT / doc).read_text(encoding="utf-8"))


class StartHere(unittest.TestCase):
    def test_runs_setup_sh_from_its_own_folder_with_the_options(self):
        f = Folder()
        r = f.run("START-HERE.command", "--model", "Q2_0", "--yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(f.calls(), ["setup.sh --model Q2_0 --yes"])
        self.assertNotIn("Press Enter", r.stdout)             # a clean exit closes the window as Terminal is set

    def test_an_error_keeps_the_window_open_and_the_exit_code(self):
        f = Folder()
        r = f.run("START-HERE.command", rc=3)
        self.assertEqual(r.returncode, 3)
        self.assertIn("Press Enter to close this window", r.stdout)

    def test_refuses_to_run_off_a_mac(self):
        f = Folder(system="Linux", machine="x86_64")
        r = f.run("START-HERE.command")
        self.assertEqual(r.returncode, 1)
        self.assertIn("./setup.sh", r.stdout)
        self.assertEqual(f.calls(), [])

    def test_setup_command_is_start_here_with_setup(self):
        f = Folder()
        r = f.run("SETUP.command", "--context", "32768")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(f.calls(), ["setup.sh --setup --context 32768"])


class UpdateAndBench(unittest.TestCase):
    def test_update_runs_update_sh_and_waits_before_closing(self):
        f = Folder()
        r = f.run("UPDATE.command", "--yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(f.calls(), ["update.sh --yes"])
        self.assertIn("START-HERE.command", r.stdout)         # says how to start the model afterwards
        r = f.run("UPDATE.command", rc=2)
        self.assertEqual(r.returncode, 2)
        self.assertIn("error", r.stdout)

    def test_bench_runs_bench_sh_and_keeps_the_report_on_screen(self):
        f = Folder()
        r = f.run("BENCH.command", "--long")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(f.calls(), ["bench.sh --long"])
        self.assertIn("Press Enter", r.stdout)


class SetupShOnAMac(unittest.TestCase):
    """setup.sh (the real one, in a folder of its own) on a Mac: an Intel Mac stops before the Python download; an Apple
    Silicon one, also in a Rosetta terminal (uname -m x86_64, the chip arm64), goes on."""

    def folder(self, machine, arm64_chip):
        f = Folder(machine=machine, arm64_chip=arm64_chip)
        (f.dir / "setup.sh").write_bytes((ROOT / "setup.sh").read_bytes())
        # a Mac with only its own Python 3.9: setup.sh then downloads the standalone Python (the curl stub refuses, so
        # that is where a run that passed the chip check ends)
        for py in ("python3", "python"):
            stub(f.bin / py, "exit 1\n")
        return f

    def test_an_intel_mac_is_told_before_anything_is_downloaded(self):
        f = self.folder("x86_64", "0")
        r = f.run("setup.sh", "--yes")
        self.assertEqual(r.returncode, 1)
        self.assertIn("Apple Silicon", r.stdout)
        self.assertNotIn("curl must not run", r.stderr)
        self.assertFalse((f.dir / ".venv").exists())

    def test_apple_silicon_goes_on_also_under_rosetta(self):
        for machine, chip in (("arm64", "1"), ("x86_64", "1")):
            with self.subTest(machine=machine):
                f = self.folder(machine, chip)
                r = f.run("setup.sh", "--check")
                self.assertNotIn("Intel processor", r.stdout)
                self.assertIn("curl must not run", r.stderr)      # it went on, to the Python download
                self.assertIn("could not download Python", r.stdout)
                self.assertEqual(r.returncode, 1)

if __name__ == "__main__":
    unittest.main()
