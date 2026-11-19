#!/usr/bin/env python3
"""Run the native Vulkan engine skeleton through Strata's server.

  python engine/vulkan/engine/serve.py --port 8095

It builds the engine, writes a Strata config with our shader paths, and starts
`python -m serve.server --engine strata`. The whole forward (dense weights,
streamed MoE experts, router, sampler) runs on our Vulkan kernels; llama.cpp is
not involved in inference.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
VULKAN = HERE.parent
REPO = VULKAN.parents[1]
SHADERS = VULKAN / "shaders"
SETUP = VULKAN / "bridge" / "setup_bridge.py"
DEFAULT_GGUF = pathlib.Path("/home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf")
DEFAULT_DATA = pathlib.Path.home() / ".cache" / "strata-vulkan-native"


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
    exe = HERE / "strata-vulkan-engine"
    if not exe.exists():
        print("building the native engine skeleton ...", flush=True)
        subprocess.run(["bash", str(HERE / "build.sh")], check=True)

    data = pathlib.Path(a.data_dir).resolve()
    data.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(SETUP), "--gguf", str(gguf),
                    "--data-dir", str(data), "--port", str(a.port),
                    "--ctx", str(a.ctx), "--threads", str(a.threads),
                    "--exe", str(exe), "--model-name", "qwen3.8-35b-a3b-vulkan-native"],
                   check=True)

    cfg_path = data / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["args"] += [
        "--native-shaders", str(VULKAN / "native" / "shaders"),
        "--expert-shaders", str(SHADERS),
        "--q4-shader", str(SHADERS / "expert_q4_k_gemv.comp.spv"),
        "--down-shader", str(SHADERS / "expert_q6_k_gemv.comp.spv"),
        "--q8-shader", str(SHADERS / "expert_q8_0_gemv.comp.spv"),
        "--silu-shader", str(SHADERS / "expert_silu_mul.comp.spv"),
        "--router-gemv-shader", str(SHADERS / "router_gemv.comp.spv"),
        "--router-topk-shader", str(SHADERS / "router_topk.comp.spv"),
    ]
    cfg["log"] = str(data / "strata-vulkan-native.log")
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "terminal") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"starting Strata on http://127.0.0.1:{a.port}/v1 (native Vulkan skeleton)", flush=True)
    return subprocess.call([sys.executable, "-m", "serve.server", "--engine", "strata",
                            "--config", str(cfg_path), "--port", str(a.port)],
                           cwd=str(REPO / "terminal"), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
