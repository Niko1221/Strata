"""Preparation boundary tests, not real-model acceptance or release evidence."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("real_model_boundary_release", HERE / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_production_gate_boundary() -> None:
    """Observe permission decisions with fake services; execute no proposed tools."""
    completed = subprocess.run(
        [
            str(release.NODE),
            "--test",
            "--test-reporter=tap",
            "--test-concurrency=1",
            str(HERE / "real-model-boundary-fixture.mjs"),
        ],
        cwd=release.ROOT,
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # Never print raw captured errors, which could contain inherited host details.
    passed = completed.returncode == 0 and all(
        counter in completed.stdout for counter in (b"# pass 3", b"# fail 0", b"# skipped 0")
    )
    assert passed, "Synthetic permission boundary fixture failed; raw output suppressed"


def test_actual_runtime_capability_admission() -> None:
    """Local scripted inference only; actual pinned registry and MCP transport."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATH",
            "PATHEXT",
            "TEMP",
            "TMP",
            "PROGRAMFILES",
            "PROGRAMFILES(X86)",
            "PROGRAMDATA",
            "SYSTEMDRIVE",
        }
    }
    env["STRATA_HARNESS_DIR"] = str(release.RES / "harness")
    completed = subprocess.run(
        [
            str(release.NODE),
            "--test",
            "--test-reporter=tap",
            "--test-concurrency=1",
            str(HERE / "real-model-runtime-fixture.mjs"),
        ],
        cwd=release.ROOT,
        env=env,
        capture_output=True,
        timeout=170,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # The environment deliberately excludes credentials; do not expose raw errors.
    (Path(env["TEMP"]) / "real-model-runtime-synthetic.tap").write_bytes(completed.stdout + completed.stderr)
    passed = completed.returncode == 0 and all(
        counter in completed.stdout for counter in (b"# pass 2", b"# fail 0", b"# skipped 0")
    )
    assert passed, "Synthetic runtime admission failed; captured diagnostics retained privately"
