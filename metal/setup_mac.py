"""setup.py on a Mac (Apple Silicon): upstream's installer, steered onto the Metal engine from outside.

setup.py runs this by itself on macOS (metal_setup()); `python metal/setup_mac.py [setup.py's options]` does the same.

Like sycl/setup_intel.py this does not edit setup.py's steps: it imports it, replaces the few that are NVIDIA/AMD
specific, and runs setup's own main().  The model choice, the download, the tokenizer and the context and KV
questions are setup's.  What is replaced (docs/MACOS_PLAN.md, phase A):

  - the GPU check: the Mac's GPU, offered through setup's AMD path (the one that compiles locally and has no images);
    its memory is the share of the unified memory macOS lets the GPU wire (iogpu.wired_limit_mb, else ~75%);
  - the CPU check: Apple Silicon has no AVX; the Metal engine computes everything on the GPU;
  - the engine step: metal/ compiled here (CMake + the Xcode command-line tools), into engine-metal/;
  - the MTP draft layer: not downloaded (the Metal engine does not draft yet), nor the low-RAM mode's expert file;
  - the config: `strata-metal --gguf <shard 1> --max-context N --kv K`, backend "metal".  The server is unchanged.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402

ENGINE = ROOT / "engine-metal"                         # not engine/: update_installed_engine() manages that one
EXE = "strata-metal"
TRAINED_CTX = 262144                                   # the Metal engine has no rope scaling past it yet


def sysctl(name: str) -> str:
    return S.out(["sysctl", "-n", name]).strip()


def metal_working_set_gb() -> float:
    """Metal's recommendedMaxWorkingSetSize (what llama.cpp treats as the GPU's memory; it follows a user's
    iogpu.wired_limit_mb), asked through the Objective-C runtime.  0.0 when it cannot be asked."""
    import ctypes
    try:
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.dylib")
        metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        dev = metal.MTLCreateSystemDefaultDevice()
        send = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        return send(dev, objc.sel_registerName(b"recommendedMaxWorkingSetSize")) / 2**30 if dev else 0.0
    except (OSError, AttributeError):
        return 0.0


def apple_gpu(ram: float) -> dict:
    """The Mac's GPU as setup's AMD path lists a card.  vram_gb: what Metal lets the GPU use (about 75-90% of the RAM
    by default: 108 GiB of 128 measured on an M5 Max); 75% when Metal cannot be asked."""
    chip = sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    limit_mb = int(sysctl("iogpu.wired_limit_mb") or 0)
    return {"index": 0, "name": f"{chip} GPU", "vram_gb": metal_working_set_gb() or 0.75 * ram,
            "arch": "metal", "driver": "Metal", "wired_limit_set": limit_mb > 0}


def strata_version() -> str:
    m = re.search(r"project\(\s*\S+\s+VERSION\s+([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig"))
    return m.group(1) if m else "0"


def llama_commit() -> str:
    m = re.search(r'STRATA_LLAMA_COMMIT "([0-9a-f]{40})"', (ROOT / "metal" / "CMakeLists.txt").read_text())
    return m.group(1) if m else ""


def build_engine(*_a, **_k) -> Path:
    """metal/ compiled here (once per Strata version and llama.cpp commit) -> engine-metal/ with BUILD.json."""
    meta = {"version": strata_version(), "source": "local", "backend": "metal", "llama": llama_commit(), "lib_dirs": []}
    info = ENGINE / "BUILD.json"
    try:
        old = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    if (ENGINE / EXE).exists() and all(old.get(k) == meta[k] for k in ("version", "llama")):
        return ENGINE
    if not S.out(["xcrun", "--find", "clang++"]).strip():
        S.fail("the Xcode command-line tools are missing (the compiler)", "run: xcode-select --install, then this again")
    venv_bin = Path(sys.executable).parent             # setup's step 3 pip-installs cmake and ninja here
    cmake = shutil.which("cmake", path=f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    ninja = shutil.which("ninja", path=f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    if not cmake:
        S.fail("cmake is missing", "brew install cmake (or pip install cmake), then run this again")
    build = ROOT / "build-metal"
    S.say("  Compiling the Metal engine (llama.cpp's Metal backend + Strata's protocol; 3-6 minutes, once) ...")
    cfg = [cmake, "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_METAL=ON"]
    if ninja:
        cfg += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    if os.environ.get("STRATA_LLAMA_DIR"):             # an offline build: a llama.cpp checkout at the pinned commit
        cfg.append(f"-DSTRATA_LLAMA_DIR={os.environ['STRATA_LLAMA_DIR']}")
    S.run(cfg, quiet=True)
    S.run([cmake, "--build", str(build), "--target", EXE, "-j", str(os.cpu_count() or 8)], quiet=True)
    ENGINE.mkdir(exist_ok=True)
    shutil.copy2(build / "metal" / EXE, ENGINE / EXE)
    info.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return ENGINE


def flag(args, name):
    return args[args.index(name) + 1] if name in args[:-1] else None


def to_metal(cfg: dict) -> dict:
    """setup's config (written for its HIP path) -> the Metal engine's.  Every key setup does not own (a hand-set
    "sampling", "model_switcher", ...) stays as it is."""
    args = cfg["args"]
    gguf, ctx = flag(args, "--native"), int(flag(args, "--max-context") or 32768)
    if gguf is None:
        S.fail("setup's config has no --native GGUF any more: metal/setup_mac.py needs updating for this setup.py")
    if ctx > TRAINED_CTX:
        S.fail(f"a {ctx // 1024}K context needs rope scaling, which the Metal engine does not have yet",
               f"run setup again with --context {TRAINED_CTX} or less")
    out = {k: v for k, v in cfg.items() if k not in ("lib_dirs", "env", "vision", "gpu", "gpus_asked", "layer_split",
                                                     "draft_vocab", "cuda")}
    out.update({"backend": "metal", "exe": str(ENGINE / EXE),
                "args": ["--gguf", gguf, "--max-context", str(ctx), "--kv", flag(args, "--kv") or "int8"]})
    return out


def install(argv) -> None:
    if platform.machine() != "arm64":
        S.fail("this Mac has an Intel processor; Strata's Metal engine runs on Apple Silicon (M1 or newer) only")
    for name in ("gpus", "amd_gpus", "amd_problem", "hip_vision", "build_engine_hip", "hipblaslt_table", "cpu_info",
                 "run", "mtp_corrupt", "refresh_draft_vocab", "bench_tips", "parallel_note", "write_run_script", "say", "main"):
        if not callable(getattr(S, name, None)):
            S.fail(f"setup.py has no {name}() any more: metal/setup_mac.py needs updating for this setup.py")
    ram = S.ram_gb()
    gpu = apple_gpu(ram)
    chip = gpu["name"][:-len(" GPU")]
    say = S.say

    def say_mac(msg=""):
        """setup's words for its AMD and CPU paths, said for the Mac."""
        msg = str(msg)
        msg = msg.replace("Your AMD GPUs:", "Your GPU:").replace(" (AMD: docs/AMD_HIP.md)", " (Metal: docs/MACOS_PLAN.md)")
        msg = re.sub(r"([\d.]+) GB VRAM(, metal)?", r"\1 GB of the unified memory usable by the GPU", msg)
        msg = msg.replace("(AVX2)", "(Apple Silicon: the GPU computes every expert)")
        if "MTP draft layer (speculative" in msg or "only its ~5 GB of MTP tensors" in msg:
            return                                      # not fetched here (run_mac skips it)
        if "MTP draft layer: " in msg:
            msg = msg.split("MTP")[0] + "MTP draft layer: not used by the Metal engine yet (no speculative decoding)"
        say(msg)
    S.say = say_mac
    S.EXE = EXE
    S.gpus = lambda *a, **k: []
    S.amd_gpus = lambda *a, **k: [gpu]
    S.amd_problem = lambda g: None
    S.hip_vision = lambda asked: "none"                 # images: not in the Metal engine yet
    S.hipblaslt_table = lambda *a, **k: None
    S.build_engine_hip = build_engine
    S.cpu_info = lambda: (chip, True, False)            # no AVX: nothing of setup's CPU-kernel choices applies
    S.mtp_corrupt = lambda *a, **k: False
    S.refresh_draft_vocab = lambda *a, **k: None
    S.bench_tips = lambda *a, **k: []                   # the CUDA engine's flags (--prefill, --conversation-cache-mib)
    S.parallel_note = lambda *a, **k: []                # --parallel: no batch slots in the Metal engine yet
    run = S.run

    def run_mac(cmd, *a, **k):
        """setup's commands, less the MTP draft layer's (fetched and built for the CUDA engine's speculation)."""
        if len(cmd) > 1 and Path(str(cmd[1])).name in ("mtp_fetch.py", "mtp_pack.py", "mtp_rt.py"):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return run(cmd, *a, **k)
    S.run = run_mac

    write = S.write_run_script

    def write_run_script(model, cfg_path, port, open_browser=True):   # setup.write_run_script's signature
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8-sig"))
        if cfg.get("backend") != "metal":
            cfg = to_metal(cfg)
            Path(cfg_path).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        return write(model, cfg_path, port, open_browser)
    S.write_run_script = write_run_script

    if not gpu["wired_limit_set"]:
        S.say(f"  The GPU may use about {gpu['vram_gb']:.0f} GB of this Mac's {ram:.0f} GB (macOS' default share). "
              "More, until the next restart: sudo sysctl iogpu.wired_limit_mb=<MB> (setup changes no system setting)")
    sys.argv = [str(ROOT / "setup.py"), *argv]
    for opt, default in (("--backend", "hip"), ("--vision", "none"), ("--low-ram", "off")):
        if not any(x == opt or x.startswith(opt + "=") for x in argv):
            sys.argv += [opt, default]
    sys.exit(S.main())


if __name__ == "__main__":
    try:
        install(sys.argv[1:])
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
