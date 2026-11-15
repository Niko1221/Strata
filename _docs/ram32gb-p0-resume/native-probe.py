"""Observe executable metadata startup through a retained native handle; no model request."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from native_model_resource_guard import available_memory
from perf131k.windows_job import process_memory_reader

BASE = Path("H:/Strata/evidence/ram32gb-p0-resume-20261115")
ENGINE = Path("H:/Strata/evidence/ram32gb-pr00-20261115/candidate-r4/strata-q36.exe")
EXPECTED = "312ad0589cdc90c53e2461455e05bb5b03ced0dafc01fb0e87ef5bbd3b9b6c06"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.name):
        raise ValueError("fresh contained name required")
    module = importlib.util.spec_from_file_location(
        "probe_guard", ROOT / "_docs/local-053-20261109/native/common.py"
    )
    guard = importlib.util.module_from_spec(module)
    module.loader.exec_module(guard)
    guard.require_current_job(ROOT)
    with ENGINE.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != EXPECTED:
        raise ValueError("native identity changed")
    folder = BASE / args.name
    folder.mkdir()
    read_memory = process_memory_reader()
    rows = []
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open(
        "xb"
    ) as stderr, (folder / "native-memory.jsonl").open("x", encoding="utf-8") as trace:
        start = time.monotonic()
        proc = subprocess.Popen(
            [str(ENGINE), "--version"],
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        while True:
            row = dict(
                read_memory(int(proc._handle)),
                elapsed_s=time.monotonic() - start,
                engine_pid=proc.pid,
                system=available_memory(),
                scope="held_native_handle_sampled",
            )
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            code = proc.poll()
            if code is not None:
                break
            time.sleep(0.05)
    text = (folder / "stdout.log").read_text(encoding="utf-8")
    result = {
        "scope": "executable_metadata_startup_not_model_inference",
        "engine_sha256": actual,
        "native_exit_code": code,
        "elapsed_s": time.monotonic() - start,
        "samples": len(rows),
        "sampled_max_private_bytes": max(
            (r["private_bytes"] for r in rows if r["private_bytes"] is not None),
            default=None,
        ),
        "sampled_max_working_set_bytes": max(
            (
                r["working_set_bytes"]
                for r in rows
                if r["working_set_bytes"] is not None
            ),
            default=None,
        ),
        "version_matched": text.strip() == "strata-q36 0.5.9",
        "qualification": "NOT_RUN",
    }
    with (folder / "result.json").open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2)
    if code != 0 or not result["version_matched"]:
        raise RuntimeError("native_metadata_probe_failed")


if __name__ == "__main__":
    main()
