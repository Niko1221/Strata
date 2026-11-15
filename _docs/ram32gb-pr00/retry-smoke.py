"""Retain the failed run, reuse its frozen identities, and verify them again in a new run."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qwen35.q36_bounded_run import digest, read_json, run_bounded
from tools.qwen35.q36_collect_environment import collect_environment

OUT = Path("H:/Strata/evidence/ram32gb-pr00-20261115")


def save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def main() -> None:
    spec = read_json(OUT / "smoke-inputs/spec.json")
    engine = OUT / "candidate-r4/strata-q36.exe"
    build = OUT / "candidate-r4/build-receipt.json"
    spec.update(
        run_id="q36-short-observation-r2",
        output_dir=str(OUT / "q36-short-observation-r2"),
        engine_path=str(engine),
        engine_sha256=digest(engine),
        toolchain_sha256=digest(build),
    )
    environment = OUT / "smoke-inputs/environment-r2.json"
    save(
        environment,
        collect_environment(
            ROOT, binaries=[engine], dependencies=[Path("H:/Strata/deps/ggml-3cf03257")]
        ),
    )
    spec["observation"]["environment_manifest"] = {
        "path": str(environment),
        "sha256": digest(environment),
    }
    spec["observation"]["identity_files"]["build_receipt"] = {
        "path": str(build),
        "sha256": digest(build),
    }
    save(OUT / "smoke-inputs/spec-r2.json", spec)
    result = run_bounded(spec)
    manifest = read_json(OUT / "q36-short-observation-r2/run_manifest.json")
    save(
        OUT / "smoke-summary-r2.json",
        {
            "status": result["status"],
            "reason": result["reason"],
            "telemetry_errors": manifest["telemetry_errors"],
            "qualification": "NOT_RUN",
        },
    )
    if result["status"] != "PASS" or manifest["telemetry_errors"]:
        raise RuntimeError("short_observation_did_not_pass")


if __name__ == "__main__":
    main()
