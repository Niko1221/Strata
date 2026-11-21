#!/usr/bin/env python3
"""Run a GGUF through Strata's server, with llama.cpp's Vulkan backend.

One command for the "bridge now" step:

  python engine/vulkan/bridge/serve.py \
      --gguf /path/to/model.gguf --port 8095

It builds the bridge if needed, exports the tokenizer and writes the config
(through setup_bridge.py), then starts `python -m serve.server --engine strata`
with that config. The server prints its own ready banner and keeps running.
"""
import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_DATA = pathlib.Path.home() / ".cache" / "strata-vulkan-bridge"
DEFAULT_GGUF = pathlib.Path("/home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", default=str(DEFAULT_GGUF))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    gguf = pathlib.Path(a.gguf).resolve()
    if not gguf.exists():
        print(f"model not found: {gguf}", file=sys.stderr)
        return 1
    binary = HERE / "strata-llama-vulkan-engine"
    if not binary.exists():
        print("building the bridge ...", flush=True)
        subprocess.run(["bash", str(HERE / "build.sh")], check=True)

    cfg = pathlib.Path(a.data_dir).resolve() / "config.json"
    subprocess.run([sys.executable, str(HERE / "setup_bridge.py"),
                    "--gguf", str(gguf), "--data-dir", str(cfg.parent),
                    "--port", str(a.port), "--ctx", str(a.ctx),
                    "--threads", str(a.threads)], check=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "terminal") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"starting Strata on http://127.0.0.1:{a.port}/v1 (Vulkan via llama.cpp bridge)", flush=True)
    return subprocess.call([sys.executable, "-m", "serve.server", "--engine", "strata",
                            "--config", str(cfg), "--port", str(a.port)],
                           cwd=str(REPO / "terminal"), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
