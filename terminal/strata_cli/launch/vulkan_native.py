"""``strata launch --backend vulkan`` — the native Vulkan engine.

This is the default Vulkan path. It starts Strata's own server with
``engine/vulkan/engine/strata-vulkan-engine`` as the engine exe. That engine runs
the whole forward itself (dense weights resident, MoE experts streamed from the
GGUF, router/combine/sampler on our Vulkan kernels) — llama.cpp is not involved
in inference. It serves the Qwen3.8-35B-A3B GGUF on an Intel Arc B390 today.

The llama.cpp bridge (``vulkan_bridge.py``) is still available with
``--vulkan-impl bridge``: it is much faster but does not stream experts. Both
stay out of cmd.py's normal CUDA/HIP/SYCL/Metal flow.
"""
import json
import pathlib
import subprocess
import sys

from strata_control.childenv import quiet_crash_env

REPO = pathlib.Path(__file__).resolve().parents[3]
NATIVE = REPO / "engine" / "vulkan" / "engine"
VULKAN = NATIVE.parent
SHADERS = VULKAN / "shaders"
SETUP = VULKAN / "bridge" / "setup_bridge.py"
TERMINAL = REPO / "terminal"
DEFAULT_DATA = pathlib.Path.home() / ".cache" / "strata-vulkan-native"


def plan(args) -> dict:
    gguf = pathlib.Path(str(args.model_path or "")).expanduser().resolve()
    data = pathlib.Path(str(args.data_dir or DEFAULT_DATA)).expanduser().resolve()
    port = int(getattr(args, "port", 8080) or 8080)
    return {
        "gguf": str(gguf),
        "data_dir": str(data),
        "port": port,
        "context": int(args.context) if getattr(args, "context", None) else 32768,
        "threads": int(getattr(args, "threads", 8) or 8),
        "slots": max(0, int(getattr(args, "vulkan_slots", 2048) or 0)),
        "url": f"http://127.0.0.1:{port}/v1",
        "exe": str(NATIVE / "strata-vulkan-engine"),
        "impl": "native",
    }


def _native_config(p: dict) -> None:
    """Run the shared tokenizer/config setup, then add our shader paths."""
    data = pathlib.Path(p["data_dir"])
    subprocess.run([sys.executable, str(SETUP),
                    "--gguf", p["gguf"], "--data-dir", str(data),
                    "--port", str(p["port"]), "--ctx", str(p["context"]),
                    "--threads", str(p["threads"]),
                    "--exe", p["exe"],
                    "--model-name", "qwen3.8-35b-a3b-vulkan-native"], check=True,
                    env=quiet_crash_env())

    cfg_path = data / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if p["slots"]:
        cfg["args"] += ["--slots", str(p["slots"])]
    cfg["args"] += [
        "--native-shaders", str(VULKAN / "native" / "shaders"),
        "--expert-shaders", str(SHADERS),
        "--q4-shader", str(SHADERS / "expert_q4_k_gemv.comp.spv"),
        "--down-shader", str(SHADERS / "expert_q6_k_gemv.comp.spv"),
        "--q8-shader", str(SHADERS / "expert_q8_0_gemv.comp.spv"),
        "--iq3-xxs-shader", str(SHADERS / "expert_iq3_xxs_gemv.comp.spv"),
        "--iq2-xs-shader", str(SHADERS / "expert_iq2_xs_gemv.comp.spv"),
        "--iq1-m-shader", str(SHADERS / "expert_iq1_m_gemv.comp.spv"),
        "--silu-shader", str(SHADERS / "expert_silu_mul.comp.spv"),
        "--router-gemv-shader", str(SHADERS / "router_gemv.comp.spv"),
        "--router-topk-shader", str(SHADERS / "router_topk.comp.spv"),
    ]
    cfg["log"] = str(data / "strata-vulkan-native.log")
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


def launch(args) -> int:
    """Prepare the native config and start it through Strata's runtime."""
    from strata_cli.launch import vulkan_common

    p = plan(args)
    gguf = pathlib.Path(p["gguf"])
    if not gguf.is_file() or gguf.suffix.lower() != ".gguf":
        print(f"strata: --backend vulkan needs one .gguf file (--model-path PATH.gguf), got {gguf}",
              file=sys.stderr)
        return 2

    data = pathlib.Path(p["data_dir"])
    data.mkdir(parents=True, exist_ok=True)
    if not pathlib.Path(p["exe"]).exists():
        print("strata: building the native Vulkan engine ...", flush=True)
        subprocess.run(["bash", str(NATIVE / "build.sh")], check=True,
                       env=quiet_crash_env())
    _native_config(p)

    return vulkan_common.start(args, "native", gguf, data / "config.json",
                               machine=getattr(args, "output", "human") == "json")
