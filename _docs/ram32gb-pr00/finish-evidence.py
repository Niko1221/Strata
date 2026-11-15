"""Index all attempts; explicitly record a host-guard controller abort without inventing native results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qwen35.q36_phase_metrics import write_artifacts

OUT = Path("H:/Strata/evidence/ram32gb-pr00-20261115")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    attempt = sys.argv[1] if len(sys.argv) > 1 else "r4"
    if attempt not in {"r4", "r5"}:
        raise ValueError("only the recorded r4/r5 attempts can be reconciled")
    guard_file = (
        "dynamic-host-guard.json" if attempt == "r4" else "dynamic-host-guard-r5.json"
    )
    outer_file = f"receipts/model-smoke-v{attempt[1:]}.json"
    dynamic = read(OUT / guard_file)
    outer = read(OUT / outer_file)
    owned = outer["run"]
    if not (owned["termination_confirmed"] and owned["job_active_after"] == 0):
        raise RuntimeError("await_confirmed_owned_job_cleanup")
    spec = read(OUT / f"smoke-inputs/spec-{attempt}.json")
    folder = Path(spec["output_dir"])
    if not (folder / "result.json").exists():
        if not dynamic["guard"]["pressure"]:
            raise RuntimeError("no_observed_host_guard_abort")
        events = [
            json.loads(line)
            for line in (folder / "events.jsonl").read_text().splitlines()
        ]
        if any(event["event"] in ("request", "token", "done") for event in events):
            raise RuntimeError("partial_request_requires_separate_reconciliation")
        instrumentation = next(
            (e["value"] for e in events if e["event"] == "instrumentation"), None
        )
        # This is an explicit post-controller receipt. It never claims a native
        # exit code, READY, input consumption, completed verification or model output.
        result = {
            "schema_version": 1,
            "run_id": spec["run_id"],
            "status": "ABORTED",
            "reason": "dynamic_host_memory_pressure",
            "last_stage": "before_request_dispatch",
            "artifact_source": "post_controller_abort_reconciliation",
            "native_exit_code": None,
            "startup_generation": None,
            "requests_status": "NOT_RUN",
            "instrumentation": instrumentation,
            "cases": [],
            "qualification": "NOT_RUN",
            "ownership_cleanup": {
                "termination_confirmed": True,
                "job_active_after": 0,
                "outer_receipt_sha256": sha(OUT / outer_file),
            },
            "host_guard": dynamic,
        }
        save(folder / "result.json", result)
        write_artifacts(folder, result, spec)
    attempts = []
    suffixes = ["", "-r2", "-r3", "-r4"] + (["-r5"] if attempt == "r5" else [])
    for suffix in suffixes:
        folder = OUT / ("q36-short-observation" + suffix)
        result = read(folder / "result.json")
        spec_file = OUT / "smoke-inputs" / ("spec" + suffix + ".json")
        expected = read(spec_file)
        attempts.append(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "reason": result["reason"],
                "requests_status": result.get("requests_status", "NOT_RUN"),
                "result_sha256": sha(folder / "result.json"),
                "manifest_sha256": sha(folder / "run_manifest.json"),
                "declared_engine_sha256": expected["engine_sha256"],
                "declared_model_sha256": expected["model_sha256"],
                "declared_pack_sha256": expected["pack_sha256"],
                "qualification": "NOT_RUN",
            }
        )
    save(
        OUT
        / (
            "baseline-attempt-index.json"
            if attempt == "r4"
            else "baseline-attempt-index-r5.json"
        ),
        {
            "schema": "strata-pr00-baseline-attempt-index-v1",
            "attempts": attempts,
            "native_generation_status": "NOT_RUN",
            "baseline_status": "NOT_RUN",
            "scope": "isolated_Q36_4096_capacity_smoke_not_the_selected_dense_9B_model",
            "qualification": "NOT_RUN",
            "releaseQualified": False,
            "plan_sha256": sha(
                Path(
                    "C:/Users/downl/Downloads/Strata_RAM32GB_VRAM16GB_Implementation_Plan_20261115 (1).md"
                )
            ),
        },
    )


if __name__ == "__main__":
    main()
