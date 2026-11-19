"""The llama.cpp-Vulkan bridge, available as `--backend vulkan --vulkan-impl bridge`.

This is the fast research fallback, not the default: it starts Strata's own
server with `engine/vulkan/bridge/strata-llama-vulkan-engine` as the engine exe,
so a GGUF generates tokens on an Intel iGPU without oneAPI or Level Zero. It is
much faster than the native engine (~31 tok/s vs ~1 tok/s) but does not stream
experts. `--backend vulkan` now uses `vulkan_native.py` by default.

Kept out of cmd.py so the normal CUDA/HIP/SYCL/Metal launch flow is untouched.
"""
import pathlib
import subprocess
import sys

from strata_control.childenv import quiet_crash_env

REPO = pathlib.Path(__file__).resolve().parents[3]
BRIDGE = REPO / "engine" / "vulkan" / "bridge"
TERMINAL = REPO / "terminal"
DEFAULT_DATA = pathlib.Path.home() / ".cache" / "strata-vulkan-bridge"


def plan(args) -> dict:
    gguf = pathlib.Path(str(args.model_path or "")).expanduser().resolve()
    data = pathlib.Path(str(args.data_dir or DEFAULT_DATA)).expanduser().resolve()
    port = int(getattr(args, "port", 8080) or 8080)
    return {
        "gguf": str(gguf),
        "data_dir": str(data),
        "port": port,
        "context": int(args.context) if getattr(args, "context", None) else 32768,
        "url": f"http://127.0.0.1:{port}/v1",
        "exe": str(BRIDGE / "strata-llama-vulkan-engine"),
    }


def launch(args) -> int:
    """Prepare the bridge config and start it through Strata's runtime."""
    from strata_cli.launch import vulkan_common

    p = plan(args)
    gguf = pathlib.Path(p["gguf"])
    if not gguf.is_file() or gguf.suffix.lower() != ".gguf":
        print(f"strata: --backend vulkan needs one .gguf file (--model-path PATH.gguf), got {gguf}",
              file=sys.stderr)
        return 2

    data = pathlib.Path(p["data_dir"])
    data.mkdir(parents=True, exist_ok=True)
    if not (BRIDGE / "strata-llama-vulkan-engine").exists():
        print("strata: building the Vulkan bridge ...", flush=True)
        subprocess.run(["bash", str(BRIDGE / "build.sh")], check=True,
                       env=quiet_crash_env())
    subprocess.run([sys.executable, str(BRIDGE / "setup_bridge.py"),
                    "--gguf", p["gguf"], "--data-dir", str(data),
                    "--port", str(p["port"]), "--ctx", str(p["context"])], check=True,
                   env=quiet_crash_env())

    return vulkan_common.start(args, "bridge", gguf, data / "config.json",
                               machine=getattr(args, "output", "human") == "json")
