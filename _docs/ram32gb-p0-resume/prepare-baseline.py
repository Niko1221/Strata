"""Freeze an actual 65536-capacity, FP32, MTP2, auto-prefill Q36 baseline spec."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qwen35.q36_collect_environment import collect_environment

BASE = Path("H:/Strata/evidence/ram32gb-p0-resume-20261115")
PRIOR = Path("H:/Strata/evidence/ram32gb-pr00-20261115")


def save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def identity(path: Path) -> dict:
    with path.open("rb") as stream:
        return {
            "path": str(path),
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
        }


def main() -> None:
    BASE.mkdir(exist_ok=True)
    spec = json.loads((PRIOR / "smoke-inputs/spec-r5.json").read_text())
    fixture = json.loads((PRIOR / "smoke-inputs/fixture.json").read_text())
    original = fixture["cases"][0]
    fixture["cases"] = [
        dict(original, prompt_id=f"public-capital-repeat-{i}", mtp_reservation=2)
        for i in range(3)
    ]
    fixture_path = BASE / "baseline-capacity65536-fixture.json"
    save(fixture_path, fixture)
    environment = BASE / "baseline-capacity65536-environment.json"
    environment_data = collect_environment(
        ROOT,
        binaries=[Path(spec["engine_path"])],
        dependencies=[Path("H:/Strata/deps/ggml-3cf03257")],
    )
    save(environment, environment_data)
    name = "baseline-capacity65536-r1"
    spec.update(
        run_id=name,
        output_root=str(BASE),
        output_dir=str(BASE / name),
        prompt_fixture_path=str(fixture_path),
        prompt_fixture_sha256=identity(fixture_path)["sha256"],
        ready_timeout_s=600,
        total_timeout_s=1200,
        min_available_ram_bytes=3 * 1024**3,
    )
    args = list(spec["argv"])
    args[args.index("--max-context") + 1] = "65536"
    args[args.index("--mtp") + 1] = "2"
    chunk = args.index("--prefill")
    del args[
        chunk : chunk + 2
    ]  # The existing engine chooses its chunk; do not assume 2048.
    args.remove("--no-warmup")
    args.remove("--no-graph")
    spec["argv"] = args
    spec["requested_settings"].update(context=65536, mtp=2, prefill="auto")
    spec["observation"]["source_sha"] = environment_data["source"]["sha"]["value"]
    spec["observation"]["environment_manifest"] = identity(environment)
    save(BASE / (name + ".spec.json"), spec)


if __name__ == "__main__":
    main()
