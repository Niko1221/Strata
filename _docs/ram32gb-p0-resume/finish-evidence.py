"""Bind the completed P0 resume observations; never start a model or infer a pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from native_model_resource_guard import available_memory
from qwen35.q36_collect_environment import collect_environment

BASE = Path("H:/Strata/evidence/ram32gb-p0-resume-20261115")
PRIOR = Path("H:/Strata/evidence/ram32gb-pr00-20261115")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict:
    with path.open("rb") as stream:
        return {
            "path": str(path),
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
        }


def save(name: str, value: object) -> None:
    with (BASE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main() -> None:
    admission = read(BASE / "dense9b-admission-r1/run_manifest.json")
    save(
        "environment-final.json",
        collect_environment(ROOT, binaries=[Path(admission["runtime"]["path"])]),
    )
    receipts = {}
    for name in (
        "resume-metadata-probe-v1",
        "resume-baseline65536-r1",
        "resume-9b-admission-r1",
        "resume-final-pytests-r2",
        "resume-flake",
        "resume-black-final",
    ):
        path = PRIOR / "receipts" / (name + ".json")
        row = read(path)
        run = row.get("run", {})
        receipts[name] = dict(
            identity(path),
            passed=row["passed"],
            termination_confirmed=run.get("termination_confirmed"),
            job_active_after=run.get("job_active_after"),
        )
    save(
        "RESUME_RESULT.json",
        {
            "schema_version": 1,
            "status": "INCOMPLETE",
            "user_selected_track": "existing_9b_separate_p0",
            "q36": {
                "status": "NOT_RUN",
                "reason": "model_deleted_user_confirmed",
                "attempt": "baseline-capacity65536-r1",
                "stage": "artifact_preflight_before_engine_launch",
                "native_exit_code": None,
                "controller_receipt": receipts["resume-baseline65536-r1"],
            },
            "dense_9b": {
                "status": admission["status"],
                "admission_manifest": identity(
                    BASE / "dense9b-admission-r1/run_manifest.json"
                ),
                "generation": "NOT_RUN",
                "requires_more_than_ram_bytes": admission["admission"][
                    "requiredExclusiveBytes"
                ],
                "profile_unchanged_during_admission": admission["profileUnchanged"],
            },
            "p1_and_later": "NOT_STARTED_P0_INFERENCE_GATE_NOT_PASSED",
            "production_ram_guard": identity(
                ROOT / "apps/host/server/local-memory-guard.mjs"
            ),
            "final_resources": dict(
                available_memory(),
                H_free_bytes=shutil.disk_usage("H:/").free,
                C_free_bytes=shutil.disk_usage("C:/").free,
            ),
            "receipts": receipts,
            "performance_eligible": False,
            "releaseQualified": False,
        },
    )


if __name__ == "__main__":
    main()
