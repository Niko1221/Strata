"""serve/test_effort_position.py - #1781: "effort_position": "end" probes the engine for --tail-role-token by
reading `exe`; on the SYCL port `exe` is the strata-sycl.sh launcher, so the probe must look through the script at
the binary it runs (${STRATA_SYCL_BIN:-build-sycl-aot/strata}, relative to the repo two levels up).

    python -m unittest serve.test_effort_position
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve.server import engine_blob, effort_end_args


class Tok:
    def encode(self, text, parse_special=False, plain=()):
        return [7] if text == "system" else list(text.encode())


def fake_port(root: Path, flag: bool, env_bin: str | None = None) -> tuple[str, dict]:
    """a SYCL-style layout: <root>/sycl/serve/strata-sycl.sh plus the binary it execs; returns (exe, cfg)."""
    script = root / "sycl" / "serve" / "strata-sycl.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\nexec docker run strata-sycl-dev "
                      "${STRATA_SYCL_BIN:-build-sycl-aot/strata}\n")
    binary = root / (env_bin or "build-sycl-aot/strata")
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x7fELF engine" + (b" --tail-role-token" if flag else b""))
    return str(script), {"effort_position": "end", "env": ({"STRATA_SYCL_BIN": env_bin} if env_bin else {})}


class Probe(unittest.TestCase):
    def test_binary_exe_as_before(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "strata"
            exe.write_bytes(b"bin --tail-role-token x")
            self.assertEqual(effort_end_args({"effort_position": "end"}, str(exe), Tok()),
                             ["--tail-role-token", "7"])
            exe.write_bytes(b"old engine without the flag")
            self.assertIsNone(effort_end_args({"effort_position": "end"}, str(exe), Tok()))

    def test_sycl_launcher_default_binary(self):
        with tempfile.TemporaryDirectory() as td:
            exe, cfg = fake_port(Path(td), flag=True)
            self.assertEqual(effort_end_args(cfg, exe, Tok()), ["--tail-role-token", "7"])

    def test_sycl_launcher_env_binary(self):
        with tempfile.TemporaryDirectory() as td:
            exe, cfg = fake_port(Path(td), flag=True, env_bin="out/custom/strata")
            self.assertEqual(effort_end_args(cfg, exe, Tok()), ["--tail-role-token", "7"])
        with tempfile.TemporaryDirectory() as td:                   # process env works the same way
            exe, cfg = fake_port(Path(td), flag=True, env_bin="out/custom/strata")
            cfg["env"] = {}
            with mock.patch.dict(os.environ, {"STRATA_SYCL_BIN": "out/custom/strata"}):
                self.assertEqual(effort_end_args(cfg, exe, Tok()), ["--tail-role-token", "7"])

    def test_launcher_binary_without_the_flag(self):
        with tempfile.TemporaryDirectory() as td:
            exe, cfg = fake_port(Path(td), flag=False)
            self.assertIsNone(effort_end_args(cfg, exe, Tok()))

    def test_unreadable_binary_falls_back_to_the_script(self):
        with tempfile.TemporaryDirectory() as td:
            exe, cfg = fake_port(Path(td), flag=True)
            (Path(td) / "build-sycl-aot" / "strata").unlink()       # image not built/pulled locally
            self.assertIsNone(effort_end_args(cfg, exe, Tok()))
            self.assertIsNone(effort_end_args(cfg, str(Path(td) / "none"), Tok()))

    def test_value_validation(self):
        self.assertIsNone(effort_end_args({"effort_position": "start"}, "missing", Tok()))
        self.assertIsNone(effort_end_args({}, "missing", Tok()))
        with self.assertRaises(ValueError):
            effort_end_args({"effort_position": "middle"}, "missing", Tok())


if __name__ == "__main__":
    unittest.main()
