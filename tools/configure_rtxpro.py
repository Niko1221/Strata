#!/usr/bin/env python3
"""Resolve a fleet research preset to a private Strata engine configuration."""
import argparse
import json
from pathlib import Path


def configure(preset, root, home, context):
    if preset["weights"] not in ("UD-Q4_K_XL", "Q8_0"):
        raise ValueError("unsupported weights")
    if preset["kv"] not in ("int8", "fp16"):
        raise ValueError("unsupported KV precision")
    q4 = preset["weights"] == "UD-Q4_K_XL"
    name = preset["weights"]
    pack = home / ("fleet-downloads/q4-strata-0134-llm60/pack-ud-q4-k-xl" if q4 else
                   "fleet-downloads/q8-ple-prototype-2be5cf1/pack-q8")
    native = home / "models/qwen38-flash-next-unsloth-gguf" / name / (
        f"Qwen3.8-Flash-Next-{name}-00001-of-0000{4 if q4 else 6}.gguf")
    args = ["--pack", str(pack), "--native", str(native),
            "--expert-cache", "auto", "--resident-budget-gib", "40" if q4 else "56",
            "--expert-profile", str(root / "data/expert-profile.bin"),
            "--adapt-swaps", "96", "--adapt-decay", "0.7", "--pcie-frac", "-1",
            "--mtp", str(home / "Strata-data/mtp/rt"), "--spec", "8", "--mtp-max-t", "4",
            "--spec-min-p", "0.5", "--kv", preset["kv"], "--max-context", str(context),
            "--prompt-cache", "0", "--conversation-cache-mib", "0", "--prefill", "8192",
            "--no-prefill-borrow", "--vram-reserve-mib", "2048", "--ple-io", "ram",
            "--ple-row-cache", "1048576", "--stats"]
    if preset.get("load_projection", False):
        vector = root / "data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
        args += ["--control-vector-scaled", str(vector) + ":1.0",
                 "--control-vector-layer-range", "4", "44", "--cvec-mode", "project",
                 "--cvec-dir", "per-layer"]
    return {"exe": str(root / "build/strata"), "cwd": str(root), "args": args,
            "tokenizer": str(pack / "tokenizer"), "model_name": "flash-next-" + name,
            "env": {"STRATA_ADAPT_NOWAIT": "0"}, "research_preset": preset}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", type=Path, required=True)
    ap.add_argument("--engine-root", type=Path, required=True)
    ap.add_argument("--home", type=Path, default=Path.home())
    ap.add_argument("--context", type=int, default=73728)
    ap.add_argument("--output", type=Path, required=True)
    opt = ap.parse_args()
    cfg = configure(json.loads(opt.preset.read_text()), opt.engine_root, opt.home, opt.context)
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    # A resolved configuration is a new experiment input, never a live service edit.
    with opt.output.open("x") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
