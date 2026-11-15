"""One isolated short Q36 observation, not the selected 9B model or a long-context benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from gguf_reader import GGUFFile
from native_model_resource_guard import available_memory
from strata_tokenizer import Tokenizer
from tools.qwen35.q36_bounded_run import digest, run_bounded
from tools.qwen35.q36_collect_environment import collect_environment

OUT = Path("H:/Strata/evidence/ram32gb-pr00-20261115")
MODEL = Path("H:/Strata/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
ENGINE = OUT / "candidate-r3/strata-q36.exe"
GIB = 1024**3


def check() -> None:
    memory = available_memory()
    # This isolated smoke uses a stricter fixed floor than the existing host's
    # maximum dynamic stop threshold. No host/native guard setting is changed.
    if (
        memory["available_physical_bytes"] < 8 * GIB
        or memory["available_commit_bytes"] < 4 * GIB
    ):
        raise RuntimeError("smoke_preflight_memory_pressure")


def save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def main() -> None:
    check()
    inputs = OUT / "smoke-inputs"
    inputs.mkdir()
    md = GGUFFile(MODEL).metadata
    tokenizer = Tokenizer.from_gguf(MODEL)
    prompt = "The capital of France is"
    ids = tokenizer.encode(prompt)
    if tokenizer.decode(ids) != prompt:
        raise RuntimeError("public_fixture_tokenizer_roundtrip")
    save(
        inputs / "fixture.json",
        {
            "cases": [
                dict(
                    prompt_id="public-capital-completion",
                    input_ids=ids,
                    max_tokens=8,
                    output_reservation=8,
                    mtp_reservation=0,
                )
            ]
        },
    )
    save(
        inputs / "tokenizer.json",
        {k: v for k, v in md.items() if k.startswith("tokenizer.ggml.")},
    )
    save(
        inputs / "model-metadata.json",
        {
            k: v
            for k, v in md.items()
            if k.startswith("qwen35moe.")
            or k in ("general.architecture", "general.name", "general.file_type")
        },
    )
    save(
        inputs / "sampler.json",
        {
            "temperature": 0,
            "top_p": 1,
            "top_k": 20,
            "min_p": 0,
            "repeat": 1,
            "freq": 0,
            "present": 0,
            "last_n": 0,
            "seed": 0,
        },
    )
    (inputs / "raw-completion-template.txt").write_text("{prompt}", encoding="utf-8")
    environment = collect_environment(
        ROOT, binaries=[ENGINE], dependencies=[Path("H:/Strata/deps/ggml-3cf03257")]
    )
    save(inputs / "environment.json", environment)

    def identity(path: Path) -> dict:
        return {"path": str(path), "sha256": digest(path, check)}

    spec = {
        "schema_version": 1,
        "run_id": "q36-short-observation",
        "engine_path": str(ENGINE),
        "engine_sha256": digest(ENGINE, check),
        "model_path": str(MODEL),
        "model_sha256": digest(MODEL, check),
        "pack_path": str(MODEL) + ".q36epk",
        "pack_sha256": digest(Path(str(MODEL) + ".q36epk"), check),
        "prompt_fixture_path": str(inputs / "fixture.json"),
        "prompt_fixture_sha256": digest(inputs / "fixture.json"),
        "toolchain_sha256": digest(OUT / "candidate-r3/build-receipt.json"),
        "argv": [
            "--serve",
            "--native",
            str(MODEL),
            "--q36-expert-store",
            "mapped-pack",
            "--q36-memory",
            "bounded-experts-v1",
            "--expert-slots",
            "5",
            "--max-context",
            "4096",
            "--prefill",
            "512",
            "--mtp",
            "0",
            "--no-warmup",
            "--no-graph",
            "--prompt-cache",
            "0",
            "--conversation-cache-mib",
            "0",
            "--private-conversation-cache",
            "--cache-type-k",
            "f32",
            "--cache-type-v",
            "f32",
        ],
        "requested_settings": dict(
            context=4096,
            prefill=512,
            parallel=1,
            kv_k="f32",
            kv_v="f32",
            mtp=0,
            router_top_k=8,
            expert_quantization="UD-Q4_K_XL",
        ),
        "experiment": {"dimensions": [], "max_relative_regression": 0.05},
        "ready_timeout_s": 180,
        "total_timeout_s": 600,
        "min_available_ram_bytes": 8 * GIB,
        "min_commit_headroom_bytes": 4 * GIB,
        "min_disk_free_bytes": 10 * GIB,
        "output_root": str(OUT),
        "output_dir": str(OUT / "q36-short-observation"),
        "observation": {
            "schema_version": 1,
            "family": "qwen36",
            "source_sha": environment["source"]["sha"]["value"],
            "binary_source_sha": None,
            "environment_manifest": identity(inputs / "environment.json"),
            "identity_files": {
                "tokenizer": identity(inputs / "tokenizer.json"),
                "model_metadata": identity(inputs / "model-metadata.json"),
                "chat_template": identity(inputs / "raw-completion-template.txt"),
                "sampler": identity(inputs / "sampler.json"),
                "build_receipt": identity(OUT / "candidate-r3/build-receipt.json"),
            },
        },
    }
    save(inputs / "spec.json", spec)
    result = run_bounded(spec)
    manifest = json.loads((OUT / "q36-short-observation/run_manifest.json").read_text())
    save(
        OUT / "smoke-summary.json",
        {
            "status": result["status"],
            "reason": result["reason"],
            "telemetry_errors": manifest["telemetry_errors"],
            "profile_write_operations": "none_in_this_driver",
            "qualification": "NOT_RUN",
        },
    )
    if result["status"] != "PASS" or manifest["telemetry_errors"]:
        raise RuntimeError("short_observation_did_not_pass")


if __name__ == "__main__":
    main()
