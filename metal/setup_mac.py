"""setup.py on a Mac (Apple Silicon): upstream's installer, steered onto the Metal engine from outside.

setup.py runs this by itself on macOS (metal_setup()); `python metal/setup_mac.py [setup.py's options]` does the same.

Like sycl/setup_intel.py this does not edit setup.py's steps: it imports it, replaces the few that are NVIDIA/AMD
specific, and runs setup's own main().  The model choice, the download, the tokenizer and the context and KV
questions are setup's.  What is replaced (docs/MACOS.md):

  - the GPU check: the Mac's GPU, offered through setup's AMD path (the one that compiles locally and has no images);
    its memory is the share of the unified memory macOS lets the GPU wire (iogpu.wired_limit_mb, else ~75%);
  - the CPU check: Apple Silicon has no AVX; the Metal engine computes everything on the GPU;
  - the engine step: metal/ compiled here (CMake + the Xcode command-line tools), into engine-metal/;
  - the MTP draft layer: not downloaded (the Metal engine does not draft yet), nor the low-RAM mode's expert file;
  - the config: `strata-metal --gguf <shard 1> --max-context N --kv K`, backend "metal".  The server is unchanged.
"""
from __future__ import annotations

import hashlib
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
MTP_FILE = "mtp-qwen3.8-flash-next-q8_0.gguf"
CONVERT_NUMPY = "2.4.0"                                # see metal/mtp_gguf.py: Q8_0_DIGEST
ENGINE_SOURCES = ("metal/strata_metal.cpp", "metal/CMakeLists.txt", "tools/vision/strata_vision.cpp",
                  *sorted(f"metal/patches/{p.name}" for p in (ROOT / "metal" / "patches").glob("*.patch")))   # rebuilt on change
MTP = {"on": False, "dir": None}                       # --mtp on|off (this file's own option), tools/mtp_fetch.py's --out


def sysctl(name: str) -> str:
    return S.out(["sysctl", "-n", name]).strip()


def metal_working_set_gb() -> float:
    """Metal's recommendedMaxWorkingSetSize in GiB (serve/telemetry.py asks it); 0.0 when it cannot be asked."""
    from serve.telemetry import metal_working_set_bytes
    return metal_working_set_bytes() / 2**30


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
    src = hashlib.sha256(b"".join((ROOT / f).read_bytes() for f in ENGINE_SOURCES)).hexdigest()[:16]
    meta = {"version": strata_version(), "source": "local", "backend": "metal", "llama": llama_commit(), "src": src,
            "lib_dirs": []}
    info = ENGINE / "BUILD.json"
    try:
        old = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    if all((ENGINE / x).exists() for x in (EXE, S.VEXE)) and all(old.get(k) == meta[k] for k in ("version", "llama", "src")):
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
    S.run([cmake, "--build", str(build), "--target", EXE, S.VEXE, "-j", str(os.cpu_count() or 8)], quiet=True)
    ENGINE.mkdir(exist_ok=True)
    for exe in (EXE, S.VEXE):                          # the engine and the image encoder (tools/vision, on Metal)
        shutil.copy2(build / "metal" / exe, ENGINE / exe)
    info.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return ENGINE


def llama_source() -> Path:
    """The llama.cpp source the engine was built from (its converter makes the MTP file): CMake's fetched copy, or
    STRATA_LLAMA_DIR."""
    src = Path(os.environ.get("STRATA_LLAMA_DIR") or ROOT / "build-metal" / "_deps" / "llama_cpp-src")
    if not (src / "convert_hf_to_gguf.py").exists():
        S.fail(f"llama.cpp's converter is not in {src}", "run setup again: it compiles the engine and fetches llama.cpp")
    return src


def mtp_gguf(mtp_dir: Path) -> Path:
    """The MTP draft head for llama.cpp (metal/mtp_gguf.py), made once: tools/mtp_fetch.py's tensors (fetched in step
    6) plus the embeddings and LM head, converted by llama.cpp's own converter, which needs PyTorch: its pinned
    requirements go into .venv-mtp (~730 MB, once)."""
    out = mtp_dir / MTP_FILE
    if out.exists():
        return out
    llama = llama_source()
    py = ROOT / ".venv-mtp" / "bin" / "python"
    ready = "import torch, transformers, sentencepiece, numpy; assert numpy.__version__ == '" + CONVERT_NUMPY + "'"
    if not py.exists() or subprocess.run([str(py), "-c", ready], capture_output=True).returncode != 0:
        S.say("  Installing llama.cpp's converter (PyTorch etc., ~730 MB, once) into .venv-mtp ...")
        S.run([sys.executable, "-m", "venv", str(ROOT / ".venv-mtp")])
        S.run([str(py), "-m", "pip", "install", "-q", "-r", str(llama / "requirements" / "requirements-convert_hf_to_gguf.txt")])
        # llama.cpp pins numpy 2.2.6, which converts this head wrongly (metal/mtp_gguf.py checks the result)
        S.run([str(py), "-m", "pip", "install", "-q", f"numpy=={CONVERT_NUMPY}"])
    S.say("  Converting the MTP draft head for llama.cpp (one time; 2.5 GB more of the checkpoint is read) ...")
    S.run([sys.executable, str(ROOT / "metal" / "mtp_gguf.py"), "--mtp-dir", str(mtp_dir), "--llama", str(llama),
           "--out", str(out), "--python", str(py)])
    return out


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
    out = {k: v for k, v in cfg.items() if k not in ("lib_dirs", "env", "gpu", "gpus_asked", "layer_split",
                                                     "draft_vocab", "cuda")}
    kv = flag(args, "--kv") or "int8"
    extra = []
    if MTP["on"]:
        # drafts are checked in batches; with a quantized KV cache llama.cpp's batched attention rounds differently
        # from its one-token decode, and the answer moved after ~17 tokens (f16: after ~128 on one prompt of three)
        extra = ["--mtp", str(mtp_gguf(MTP["dir"])), "--spec", "3"]
        if kv != "f16":
            S.ok(f"KV cache: 16-bit with the MTP drafts (not {kv}: a quantized cache lets the drafts change the answer)")
        kv = "f16"
    out.update({"backend": "metal", "exe": str(ENGINE / EXE),
                "args": ["--gguf", gguf, "--max-context", str(ctx), "--kv", kv, *extra]})
    return out


_write_setup_config = S.write_setup_config


def write_setup_config(cfg_path, cfg, source=None):
    """setup.write_setup_config (#629) with the config converted first: it compares the earlier run config with this
    one, so a HIP-form one made every run again name the Metal engine's --gguf as dropped and replace the .bak."""
    return _write_setup_config(cfg_path, cfg if cfg.get("backend") == "metal" else to_metal(cfg), source)


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
        msg = msg.replace("Your AMD GPUs:", "Your GPU:").replace(" (AMD: docs/AMD_HIP.md)", " (Metal: docs/MACOS.md)")
        msg = re.sub(r"([\d.]+) GB VRAM(, metal)?", r"\1 GB of the unified memory usable by the GPU", msg)
        msg = msg.replace("(AVX2)", "(Apple Silicon: the GPU computes every expert)")
        msg = msg.replace("(speculative decoding, ~2x faster output)", "(speculative decoding, 1.2-1.5x faster output "
                          "on a Mac)")
        if not MTP["on"] and ("MTP draft layer (speculative" in msg or "only its ~5 GB of MTP tensors" in msg):
            return                                      # not fetched (run_mac skips it)
        if "MTP draft layer: " in msg:
            msg = msg.split("MTP")[0] + ("MTP draft layer: on (--mtp on; it is converted for llama.cpp at the end)"
                                         if MTP["on"] else "MTP draft layer: off (./setup.sh --setup --mtp on: 1.2-1.5x "
                                         "faster answers, which can differ from the plain ones after many tokens)")
        say(msg)
    S.say = say_mac
    S.EXE = EXE
    S.gpus = lambda *a, **k: []
    S.amd_gpus = lambda *a, **k: [gpu]
    S.amd_problem = lambda g: None
    S.hip_vision = lambda asked: {"yes": "gpu", "gpu": "gpu", "cpu": "cpu"}.get(asked, "none")   # strata-vision, Metal
    S.hipblaslt_table = lambda *a, **k: None
    S.build_engine_hip = build_engine
    S.cpu_info = lambda: (chip, True, False)            # no AVX: nothing of setup's CPU-kernel choices applies
    S.mtp_corrupt = lambda *a, **k: False
    S.refresh_draft_vocab = lambda *a, **k: None
    S.bench_tips = lambda *a, **k: []                   # the CUDA engine's flags (--prefill, --conversation-cache-mib)
    S.parallel_note = lambda *a, **k: []                # --parallel: no batch slots in the Metal engine yet
    run = S.run

    def run_mac(cmd, *a, **k):
        """setup's commands, less the CUDA engine's MTP packing; its fetch only when --mtp on (where it went is kept)."""
        name = Path(str(cmd[1])).name if len(cmd) > 1 else ""
        if name == "mtp_fetch.py" and "--out" in cmd:
            MTP["dir"] = Path(str(cmd[cmd.index("--out") + 1]))
        if name in ("mtp_pack.py", "mtp_rt.py") or (name == "mtp_fetch.py" and not MTP["on"]):
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
    S.write_setup_config = write_setup_config

    if not gpu["wired_limit_set"]:
        S.say(f"  The GPU may use about {gpu['vram_gb']:.0f} GB of this Mac's {ram:.0f} GB (macOS' default share). "
              "More, until the next restart: sudo sysctl iogpu.wired_limit_mb=<MB> (setup changes no system setting)")
    argv = list(argv)
    if "--mtp" in argv[:-1]:                           # this file's option: setup.py does not know it
        i = argv.index("--mtp")
        if argv[i + 1] not in ("on", "off"):
            S.fail("--mtp takes on or off")
        MTP["on"] = argv[i + 1] == "on"
        del argv[i:i + 2]
    else:                                              # a setup run again keeps an earlier choice
        def has_mtp(p):
            try:
                return "--mtp" in json.loads(p.read_text(encoding="utf-8-sig")).get("args", [])
            except (OSError, ValueError, AttributeError):
                return False
        MTP["on"] = any(has_mtp(p) for p in ROOT.glob("strata-*.json"))
    sys.argv = [str(ROOT / "setup.py"), *argv]
    for opt, default in (("--backend", "hip"), ("--low-ram", "off")):
        if not any(x == opt or x.startswith(opt + "=") for x in argv):
            sys.argv += [opt, default]
    sys.exit(S.main())


if __name__ == "__main__":
    try:
        install(sys.argv[1:])
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
