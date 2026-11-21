#!/usr/bin/env python3
"""Prepare a Strata server config that runs a GGUF through llama.cpp Vulkan.

It does two things Strata's own setup would do for a CUDA/HIP engine:
  1. export the model's tokenizer into <data-dir>/tokenizer/ (server.py needs it),
  2. write <data-dir>/config.json pointing `exe` at the bridge binary.

It is NOT the Vulkan engine. It is the temporary bridge that lets `strata`
actually generate tokens from this GGUF on the B390 today.

Example:
  python engine/vulkan/bridge/setup_bridge.py \
      --gguf /home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf \
      --data-dir /home/g/.cache/strata-vulkan-bridge \
      --port 8099
  cd terminal && python -m serve.server --engine strata \
      --config /home/g/.cache/strata-vulkan-bridge/config.json
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOOLS = REPO / "terminal" / "tools"
LLAMA_LIB = pathlib.Path(os.environ.get("LLAMA_CPP_LIB", "/home/g/Code/llama.cpp/build/bin"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--exe", default=str(HERE / "strata-llama-vulkan-engine"))
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model-name", default="qwen3.8-35b-a3b-vulkan")
    a = ap.parse_args()

    gguf = pathlib.Path(a.gguf).resolve()
    data = pathlib.Path(a.data_dir).resolve()
    data.mkdir(parents=True, exist_ok=True)

    if not pathlib.Path(a.exe).exists():
        print(f"bridge binary missing: {a.exe} (run engine/vulkan/bridge/build.sh)", file=sys.stderr)
        return 1

    # Strata's reference tokenizer exporter reads vocab + merges + chat template
    # straight from the GGUF; no HF download and no BPE reimplementation here.
    subprocess.run([sys.executable, str(TOOLS / "strata_tokenizer.py"),
                    "--gguf", str(gguf), "--out", str(data)], check=True)

    cfg = {
        "exe": str(pathlib.Path(a.exe).resolve()),
        "args": ["--model", str(gguf), "--ctx", str(a.ctx),
                 "--threads", str(a.threads), "--backend-dir", str(LLAMA_LIB)],
        "lib_dirs": [str(LLAMA_LIB)],
        "cwd": str(data),
        "tokenizer": str(data / "tokenizer"),
        "model_name": a.model_name,
        # server.py takes --port from its CLI (default 8095); config "port" is ignored.
        "log": str(data / "strata-vulkan-bridge.log"),
        # The bridge does not implement Strata's VRAM planner or drafter.
        "drafting": {},
        "fit_max_tokens": False,
    }
    out = data / "config.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"run: python {HERE / 'serve.py'} --gguf {gguf} --data-dir {data} --port {a.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
