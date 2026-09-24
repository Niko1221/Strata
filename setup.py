#!/usr/bin/env python3
"""Strata one-click setup and start (Windows and Linux, NVIDIA GPUs).

    START-HERE.bat  (Windows)   /   ./setup.sh  (Linux)      - they install Python if needed and run this file

The first time it asks four questions - which model (the original Qwen3.8-Flash-Next or the Swift 1.5 fine-tune),
which size, how much context, and whether the model should also read images - then installs everything and starts the model on http://127.0.0.1:8080 (OpenAI- and Anthropic-compatible
API; a small page there shows that it runs). Every later start skips straight to running the model: nothing that
is already downloaded, installed or prepared is done again.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: NVIDIA GPU and driver, RAM, CPU, free disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv (numpy, jinja2, ..., and NVIDIA's CUDA libraries)
  4. gets the Strata engine: a ready-made build for RTX 30/40/50 cards (no compiler needed); if none fits your PC,
     it installs the build tools (asks first) and compiles the engine for your GPU
  5. downloads the model from Hugging Face (resumable), and the vision encoder if you want images
  6. prepares the model for Strata and fetches the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  7. writes run-<model>.bat / run-<model>.sh and starts the model

Options: --family qwen|swift, --model Q2_0|IQ2_XS|IQ3_XXS, --context 32768, --vision yes|no|gpu|cpu, --port 8080, --yes (recommended
answers, no questions), --setup (install another model / change settings instead of starting), --no-start,
--models-dir DIR, --gguf-dir DIR (use GGUF files you already have), --build (compile instead of the ready-made
engine), --check (only check this PC).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
HF = "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/"
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"

# The ready-made engine: <PREBUILT_URL><asset>, a zip with strata(.exe), strata-vision(.exe) and BUILD.json, built
# by tools/make_release.py.  Set this to the GitHub release download folder when publishing, e.g.
# "https://github.com/<you>/Strata/releases/latest/download/" (or pass --prebuilt / set STRATA_PREBUILT_URL).
PREBUILT_URL = "https://github.com/Niko1221/Strata/releases/latest/download/"
PREBUILT_ASSET = "strata-windows-x64.zip" if WIN else "strata-linux-x64.zip"
# the CUDA libraries the ready-made engine loads (the same CUDA 13.0 it is built with), from NVIDIA's pip packages
CUDA_WHEELS = ["nvidia-cublas==13.0.2.14", "nvidia-cuda-runtime==13.0.96"]
MIN_DRIVER = 580                       # CUDA 13.0
MIN_ENGINE = (0, 1, 1)                 # the ready-made engine that reads split models (Swift 1.5)
PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "cmake", "ninja"]

MODELS = {
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3-bit i-quant, the best quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9},
}
CONTEXTS = [8192, 32768, 65536, 131072, 262144]
# The model families: the same architecture, weights in the same three GSQ-RCO sizes, different files.
FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen; GSQ-RCO quants by ISTA-DASLab",
             "about": "the original model",
             "hf": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/{q}/",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "",
             "mmproj_hf": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/",
             "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next"},
    "swift": {"title": "Swift 1.5", "by": "UkisAI's fine-tune of Qwen3.8-Flash-Next",
              "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its authors' numbers)",
              "hf": "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/",
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "swift-",
              "mmproj_hf": "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/",
              "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", "name": "swift-1.5",
              "license": "Swift Open License 1.0: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"},
}
MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
# the image encoder on the GPU: ~0.9 GB of weights + ~0.3 GB of work buffers at 1024 image tokens, kept free of
# expert slots (the engine's default reserve is 700 MiB)
VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 2100},
          "cpu": {"max_tokens": 300, "reserve_mib": 700}}
EXE = "strata.exe" if WIN else "strata"
VEXE = "strata-vision.exe" if WIN else "strata-vision"


# ------------------------------------------------------------------------------------------------ output
def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say()
    say(f"=== Step {n}: {title} ===")


def ok(msg):
    say(f"  [ok] {msg}")


def warn(msg):
    say(f"  [!]  {msg}")


def fail(msg, hint=None):
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    say("\nSetup stopped. Fix the item above and run it again - everything already done is kept and skipped.")
    sys.exit(1)


def ask(question, choices, default, yes):
    if yes:
        return default
    while True:
        try:
            a = input(f"{question} [{default}]: ").strip()
        except EOFError:
            return default
        if not a:
            return default
        if a.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == a.lower())
        say(f"  please answer one of: {', '.join(choices)}")


def run(cmd, cwd=None, env=None, check=True, quiet=False):
    say("  > " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
                       text=True)
    if check and r.returncode != 0:
        if quiet and r.stdout:
            say(r.stdout[-4000:])
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
    return r


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def done(path: Path) -> bool:
    """A step's finish mark: <path>.done exists (written only after the step completed)."""
    return path.with_name(path.name + ".done").exists()


def mark(path: Path, text=""):
    path.with_name(path.name + ".done").write_text(text or time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the PC
def ram_gb():
    if WIN:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 2**30
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024 / 2**30
    return 0.0


def cpu_info():
    """(name, avx2, avx512): avx512 means everything Strata's fast AVX-512 kernels use (F, BW, VL, VNNI, VBMI),
    the same test the engine makes (cpu_avx512_ok), not just AVX-512F."""
    name, avx2, avx512 = platform.processor() or "unknown CPU", False, False
    if WIN:
        pf = ctypes.windll.kernel32.IsProcessorFeaturePresent
        avx2 = bool(pf(40))                       # PF_AVX2_INSTRUCTIONS_AVAILABLE
        n = out(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip()
        name = n or name
        avx512 = bool(pf(41)) and _cpuid_avx512_full()
    else:
        try:
            txt = open("/proc/cpuinfo").read()
            flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
            avx2 = "avx2" in flags
            avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"} <= flags
            m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
            name = m.group(1) if m else name
        except OSError:
            pass
    return name, avx2, avx512


def _cpuid_avx512_full() -> bool:
    """Windows has no feature bit for VNNI / VBMI: ask the CPU (CPUID leaf 7) through a tiny machine-code stub."""
    try:
        code = bytes([0x53, 0x49, 0x89, 0xC8, 0xB8, 0x07, 0x00, 0x00, 0x00, 0x31, 0xC9, 0x0F, 0xA2,   # push rbx; r8=rcx; cpuid(7,0)
                      0x41, 0x89, 0x18, 0x41, 0x89, 0x48, 0x04, 0x5B, 0xC3])                   # [r8]=ebx,[r8+4]=ecx; pop rbx
        k32 = ctypes.windll.kernel32
        k32.VirtualAlloc.restype = ctypes.c_void_p
        buf = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)
        if not buf:
            return False
        ctypes.memmove(buf, code, len(code))
        regs = (ctypes.c_uint32 * 2)()
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(buf)(ctypes.addressof(regs))
        ebx, ecx = regs[0], regs[1]
        need_ebx = (1 << 16) | (1 << 30) | (1 << 31)                   # F, BW, VL
        need_ecx = (1 << 1) | (1 << 11)                                # VBMI, VNNI
        return (ebx & need_ebx) == need_ebx and (ecx & need_ecx) == need_ecx
    except Exception:
        return False


def gpu_info():
    s = out(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits"])
    if not s.strip():
        return None
    name, mem, cc, drv = [x.strip() for x in s.strip().splitlines()[0].split(",")]
    return {"name": name, "vram_gb": float(mem) / 1024.0, "arch": cc.replace(".", ""), "driver": drv}


def find_nvcc():
    cands = [shutil.which("nvcc")]
    if os.environ.get("CUDA_PATH"):
        cands.append(str(Path(os.environ["CUDA_PATH"]) / "bin" / ("nvcc.exe" if WIN else "nvcc")))
    if WIN:
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.exists():
            cands += [str(p / "bin" / "nvcc.exe") for p in sorted(base.iterdir(), reverse=True)]
    else:
        cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/usr/local").glob("cuda*"), reverse=True)]
    best = (None, None)
    for c in dict.fromkeys(cands):                     # every toolkit found; the newest wins
        if c and Path(c).exists():
            v = re.search(r"release (\d+)\.(\d+)", out([c, "--version"]))
            if v and (best[1] is None or (int(v.group(1)), int(v.group(2))) > best[1]):
                best = (c, (int(v.group(1)), int(v.group(2))))
    return best


def find_vcvars():
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        return None
    p = out([str(vswhere), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]).strip()
    v = Path(p) / "VC/Auxiliary/Build/vcvars64.bat" if p else None
    return v if v and v.exists() else None


def find_tool(name):
    """A tool on PATH, or the one pip installed next to this Python (cmake, ninja)."""
    p = shutil.which(name)
    if p:
        return p
    for d in (Path(sys.executable).parent / "Scripts", Path(sys.executable).parent,
              Path.home() / ".local" / "bin"):
        c = d / (name + (".exe" if WIN else ""))
        if c.exists():
            return str(c)
    return None


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# ------------------------------------------------------------------------------------------------ downloads
def download(url, dst: Path, what=None):
    """Resumable HTTP(S) download with a progress line; `file://` and plain paths are copied (tests, mirrors).
    A finished file gets a <name>.done mark, so a later run skips it without asking the server."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what or dst.name} already downloaded")
        return
    if not url.startswith(("http://", "https://")):
        src = Path(url[7:] if url.startswith("file://") else url)
        if not src.exists():
            fail(f"not found: {src}")
        shutil.copyfile(src, dst)
        mark(dst)
        ok(f"{what or dst.name} copied")
        return
    part = dst.with_name(dst.name + ".part")
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"})
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", "check your internet connection and run it again")
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:    # finished by an older setup (no mark yet)
        mark(dst)
        ok(f"{what or dst.name} already downloaded")
        return
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                     # the server ignored the range: start over
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 2:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total \
                            else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what or dst.name}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            warn(f"download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}", "check your internet connection and run it again")
    part.replace(dst)
    mark(dst)
    ok(f"{what or dst.name} downloaded")


def get_llama_cpp():
    """llama.cpp at the pinned commit (ggml for the build, gguf-py for the tools, mtmd for images), as a zip: no git."""
    llama = ROOT / "third_party" / "llama.cpp"
    if (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir():
        return llama
    z = ROOT / "third_party" / f"llama.cpp-{LLAMA_CPP_COMMIT[:7]}.zip"
    download(LLAMA_CPP_ZIP, z, "llama.cpp source")
    tmp = ROOT / "third_party" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    top = next(tmp.iterdir())
    shutil.rmtree(llama, ignore_errors=True)
    top.replace(llama)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    return llama


def pip_install(packages, what):
    """pip install into .venv, skipped when the same list was installed before."""
    stamp = Path(sys.prefix) / ".strata-pip.json"
    have = json.loads(stamp.read_text()) if stamp.exists() else []
    need = [p for p in packages if p not in have]
    if not need:
        ok(f"{what} already installed")
        return
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *need])
    stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
    ok(f"{what} installed")


def cuda_lib_dirs():
    """Where pip put NVIDIA's CUDA libraries (nvidia/cu13/bin/x86_64 on Windows, nvidia/cu13/lib on Linux)."""
    pattern = "cublas64_13.dll" if WIN else "libcublas.so.13*"
    dirs = []
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        for hit in (sp / "nvidia").rglob(pattern) if (sp / "nvidia").is_dir() else []:
            if hit.parent not in dirs:
                dirs.append(hit.parent)
    return [str(d) for d in dirs]


# ------------------------------------------------------------------------------------------------ the engine
def driver_major(gpu):
    try:
        return int(gpu["driver"].split(".")[0])
    except (ValueError, KeyError):
        return 0


def get_prebuilt(url_base, gpu, vision) -> Path | None:
    """The ready-made engine in engine/ (kept between runs), or None when there is none for this PC."""
    eng = ROOT / "engine"
    info = eng / "BUILD.json"
    if info.exists() and (eng / EXE).exists():
        meta = json.loads(info.read_text())
        ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
        if meta.get("source") == "local" or ver >= MIN_ENGINE:
            ok("ready-made engine already installed")
            return eng
        say(f"  Updating the ready-made engine ({meta.get('version')} -> {'.'.join(map(str, MIN_ENGINE))} or newer) ...")
        info.unlink()
    if not url_base:
        return None
    z = ROOT / "engine" / PREBUILT_ASSET
    base = url_base if url_base.endswith(("/", "\\")) else url_base + "/"
    if base.startswith(("http://", "https://")):
        try:                                           # not published (yet), or no internet: compile instead
            req = urllib.request.Request(base + PREBUILT_ASSET, method="HEAD", headers={"User-Agent": "strata-setup"})
            urllib.request.urlopen(req, timeout=60).close()
        except OSError as e:
            warn(f"no ready-made engine at {base} ({e}): compiling instead")
            return None
    say("  Downloading the ready-made Strata engine ...")
    download(base + PREBUILT_ASSET, z, "Strata engine")
    tmp = ROOT / "engine" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    meta = json.loads((tmp / "BUILD.json").read_text())
    if tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit()) < MIN_ENGINE:
        warn(f"the ready-made engine at {base} is version {meta.get('version')}; this setup needs "
             f"{'.'.join(map(str, MIN_ENGINE))}: compiling instead")
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    archs = [int(a) for a in meta.get("archs", [])]
    arch = int(gpu["arch"])
    if arch not in archs and not (meta.get("ptx") and arch > max(archs)):
        warn(f"the ready-made engine is built for {', '.join(str(a) for a in archs)}; your GPU is {arch}: compiling instead")
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    for p in tmp.iterdir():
        dst = eng / p.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        p.replace(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    if not (eng / EXE).exists():
        fail("the ready-made engine archive has no " + EXE)
    if not WIN:
        for x in (EXE, VEXE):
            if (eng / x).exists():
                (eng / x).chmod(0o755)
    ok(f"ready-made engine {meta.get('version', '')} for {', '.join('sm_' + str(a) for a in archs)} (CUDA "
       f"{meta.get('cuda', '?')})")
    return eng


def install_build_tools(gpu, yes):
    """The compiler and the CUDA toolkit, installed for the user (asks once).  Returns (nvcc, vcvars)."""
    nvcc, cuda_v = find_nvcc()
    need_cuda = (12, 8) if int(gpu["arch"]) >= 120 else (12, 0)
    vcvars = find_vcvars() if WIN else None
    have_cc = vcvars is not None if WIN else shutil.which("g++") is not None
    missing = []
    if not have_cc:
        missing.append("Visual Studio 2022 Build Tools (C++)" if WIN else "the C++ compiler (build-essential)")
    if nvcc is None or cuda_v < need_cuda:
        missing.append("the NVIDIA CUDA Toolkit 13.0")
    if not missing:
        ok(f"build tools present (CUDA {cuda_v[0]}.{cuda_v[1]})")
        return nvcc, vcvars
    say("  The engine has to be compiled for your PC, which needs: " + " and ".join(missing) + ".")
    say("  They can be installed now (about 8-10 GB, 15-40 minutes" + (", Windows will ask for permission" if WIN else
                                                                        ", sudo will ask for your password") + ").")
    if ask("  Install them now?", ["y", "n"], "y", yes) != "y":
        fail("the build tools are needed", "install them yourself (see README.md) and run it again")
    if WIN:
        if shutil.which("winget") is None:
            fail("winget (Windows package manager) is not available",
                 "install 'App Installer' from the Microsoft Store, or install the tools by hand (README.md)")
        wg = ["winget", "install", "-e", "--source", "winget", "--accept-package-agreements",
              "--accept-source-agreements", "--disable-interactivity"]
        if not have_cc:
            run([*wg, "--id", "Microsoft.VisualStudio.2022.BuildTools", "--override",
                 "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"],
                check=False)
        if nvcc is None or cuda_v < need_cuda:
            run([*wg, "--id", "Nvidia.CUDA", "--version", "13.0"], check=False)
        vcvars = find_vcvars()
    else:
        apt = shutil.which("apt-get")
        if apt is None:
            fail("automatic install is only done on Ubuntu/Debian",
                 "install g++ and the CUDA Toolkit 13 (https://developer.nvidia.com/cuda-downloads), then run it again")
        if not have_cc:
            run(["sudo", "apt-get", "install", "-y", "build-essential"])
        if nvcc is None or cuda_v < need_cuda:
            osr = dict(line.split("=", 1) for line in open("/etc/os-release").read().splitlines() if "=" in line)
            ver = osr.get("VERSION_ID", "").strip('"').replace(".", "")
            if osr.get("ID") != "ubuntu" or ver not in ("2204", "2404"):
                fail("the CUDA Toolkit can be installed automatically on Ubuntu 22.04 / 24.04 only",
                     "install it from https://developer.nvidia.com/cuda-downloads and run it again")
            deb = Path("/tmp/cuda-keyring.deb")
            download(f"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{ver}/x86_64/cuda-keyring_1.1-1_all.deb",
                     deb, "CUDA repository key")
            run(["sudo", "dpkg", "-i", str(deb)])
            run(["sudo", "apt-get", "update"])
            run(["sudo", "apt-get", "install", "-y", "cuda-toolkit-13-0"])
    nvcc, cuda_v = find_nvcc()
    if (WIN and find_vcvars() is None) or (not WIN and shutil.which("g++") is None):
        fail("the C++ build tools did not install", "install them by hand (README.md) and run it again")
    if nvcc is None or cuda_v < need_cuda:
        fail("the CUDA Toolkit did not install", "install it from https://developer.nvidia.com/cuda-downloads, then run it again")
    ok(f"build tools installed (CUDA {cuda_v[0]}.{cuda_v[1]})")
    return nvcc, find_vcvars() if WIN else None


def cmake_build(src, bdir, target, defs, vcvars, bat_name):
    cmake, ninja = find_tool("cmake"), find_tool("ninja")
    if cmake is None or ninja is None:
        fail("cmake / ninja not found after installing them", "run: .venv python -m pip install cmake ninja")
    conf = [cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(src), "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Release", *defs]
    build = [cmake, "--build", str(bdir), "--target", target, "-j", str(max(2, (os.cpu_count() or 4) // 2))]
    if WIN:
        bat = ROOT / bat_name
        q = lambda c: " ".join(f'"{x}"' if " " in str(x) else str(x) for x in c)  # noqa: E731
        bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{q(conf)} || exit /b 1\r\n{q(build)} || exit /b 1\r\n',
                       encoding="utf-8")
        run(["cmd", "/c", str(bat)])
    else:
        run(conf)
        run(build)


def build_engine(gpu, vision, yes, llama) -> Path:
    """Compile the engine (and, for images, the encoder) for this GPU; the results go to engine/."""
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    want_vision = vision != "none"
    if meta.get("source") == "local" and (eng / EXE).exists() and (not want_vision or (eng / VEXE).exists()):
        ok("engine already built for this PC")
        return eng
    nvcc, vcvars = install_build_tools(gpu, yes)
    if not (eng / EXE).exists() or meta.get("source") != "local":
        say("  Compiling the Strata engine for your GPU (10-20 minutes, once) ...")
        cmake_build(ROOT, ROOT / "build", "strata",
                    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", f"-DCMAKE_CUDA_ARCHITECTURES={gpu['arch']}",
                     f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama}"], vcvars, "build-strata.bat")
        shutil.copy2(ROOT / "build" / EXE, eng / EXE)
    if want_vision and not (eng / VEXE).exists():
        say("  Compiling the image encoder" + (" with CUDA (10-20 minutes, once) ..." if vision == "gpu" else " ..."))
        defs = [f"-DLLAMA_DIR={llama}", f"-DSTRATA_VISION_CUDA={'ON' if vision == 'gpu' else 'OFF'}"]
        if vision == "gpu":
            defs += [f"-DCMAKE_CUDA_ARCHITECTURES={gpu['arch']}", f"-DCMAKE_CUDA_COMPILER={nvcc}"]
        cmake_build(ROOT / "tools" / "vision", ROOT / "build-vision", "strata-vision", defs, vcvars, "build-vision.bat")
        shutil.copy2(ROOT / "build-vision" / "bin" / VEXE, eng / VEXE)
    bindir = Path(nvcc).parent                            # the toolkit's own libraries (bin, bin/x64, lib64)
    dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
    stamp.write_text(json.dumps({"source": "local", "archs": [int(gpu["arch"])], "vision": vision,
                                 "cuda_dirs": dirs}, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


# ------------------------------------------------------------------------------------------------ start
def installed_configs():
    return sorted(ROOT.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def start(cfg_path: Path, port: int | None, open_browser=True) -> int:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    cfg_path.touch()                                     # the most recently used model
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if open_browser:
        cmd.append("--open")
    say()
    say(f"Starting {cfg.get('model_name', 'the model')} (loads 34-43 GB into RAM: 30-90 s). Close this window to stop it.")
    return subprocess.call(cmd)


def write_run_script(model, cfg_path, port):
    serve = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
             "--port", str(port), "--open"]
    if WIN:
        script = ROOT / f"run-{model.lower()}.bat"
        script.write_text("@echo off\r\ntitle Strata " + model + "\r\ncd /d \"" + str(ROOT) + "\"\r\n" +
                          " ".join(f'"{x}"' for x in serve) + "\r\npause\r\n", encoding="utf-8")
    else:
        script = ROOT / f"run-{model.lower()}.sh"
        script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n",
                          encoding="utf-8")
        script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), help="qwen = Qwen3.8-Flash-Next, swift = Swift 1.5")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--context", type=int)
    ap.add_argument("--vision", choices=["yes", "no", "none", "gpu", "cpu"],
                    help="let the model read images (yes = the encoder on the GPU)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--models-dir", default=str(ROOT / "models"), help="where the GGUF files go (~70 GB)")
    ap.add_argument("--gguf-dir", help="use GGUF files you already have (a folder with the two shards)")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--setup", action="store_true", help="install another model or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--build", action="store_true", help="compile the engine instead of using the ready-made one")
    ap.add_argument("--prebuilt", default=os.environ.get("STRATA_PREBUILT_URL", PREBUILT_URL),
                    help="where the ready-made engine is (a URL folder or a local folder)")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--skip-build", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    say("Strata - Qwen3.8-Flash-Next on a normal PC (NVIDIA GPU + system RAM + CPU)")

    # ---- 0. already installed: just start it
    have = installed_configs()
    if have and not (a.setup or a.model or a.family or a.check or a.no_start):
        if len(have) == 1:
            return start(have[0], None)
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {json.loads(c.read_text(encoding='utf-8')).get('model_name', c.stem)}")
        say(f"  {len(have) + 1}) install another model / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], None)

    # ---- 1. the PC
    step(1, "checking your PC")
    gpu = gpu_info()
    if gpu is None:
        fail("no NVIDIA GPU found (nvidia-smi did not answer)",
             "install the NVIDIA driver from https://www.nvidia.com/drivers and restart the PC")
    ok(f"GPU: {gpu['name']}, {gpu['vram_gb']:.1f} GB VRAM, compute capability {gpu['arch'][:-1]}.{gpu['arch'][-1]}, "
       f"driver {gpu['driver']}")
    if int(gpu["arch"]) < 80:
        fail("this GPU is older than the RTX 30 series (compute capability 8.0 is required)")
    if driver_major(gpu) < MIN_DRIVER:
        fail(f"the NVIDIA driver is too old ({gpu['driver']}; {MIN_DRIVER} or newer is needed)",
             "update it with the NVIDIA App or from https://www.nvidia.com/drivers, restart, and run this again")
    if gpu["vram_gb"] < 11:
        warn("less than 12 GB of VRAM: Strata will run, but most experts stay on the CPU and it will be slow")
    ram = ram_gb()
    cpu, avx2, avx512 = cpu_info()
    ok(f"RAM: {ram:.0f} GB")
    ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'})")
    if not avx2:
        fail("this CPU has no AVX2; Strata needs at least AVX2")
    if a.check:
        say()
        for m, d in MODELS.items():
            verdict = "fits" if ram >= d["ram_gb"] else "tight" if ram >= d["ram_gb"] - 8 else "does not fit"
            say(f"  {m:8s} needs ~{d['ram_gb']} GB RAM: {verdict}")
        say("\nThis PC can run Strata. Run it again without --check to install.")
        return 0

    # ---- 2. the questions
    step(2, "your choices")
    fams = list(FAMILIES)
    if a.family:
        family = a.family
    else:
        for i, f in enumerate(fams, 1):
            d = FAMILIES[f]
            say(f"  {i}) {d['title']:20s} {d['by']} - {d['about']}")
        family = fams[int(ask("Which model?", [str(i) for i in range(1, len(fams) + 1)], "1", a.yes)) - 1]
    fam = FAMILIES[family]
    ok(f"model: {fam['title']}")
    if fam.get("license"):
        say(f"  Its license: {fam['license']}")
    say()
    names = list(MODELS)
    for i, m in enumerate(names, 1):
        d = MODELS[m]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        say(f"  {i}) {m:8s} {d['about']}; download {d['download_gb']:.0f} GB, uses ~{d['arena_gb']:.0f} GB of RAM{fit}")
    rec = "3" if ram >= 60 else "1"
    model = a.model or names[int(ask("Which size?", ["1", "2", "3"], rec, a.yes)) - 1]
    if ram < MODELS[model]["ram_gb"] - 4:
        fail(f"{model} needs about {MODELS[model]['ram_gb']} GB of RAM; this PC has {ram:.0f} GB",
             "choose Q2_0 or IQ2_XS, or add RAM")
    ok(f"size: {model}")
    tag = fam["tag"] + model                           # names of the pack, config and start script
    rec_ctx = 32768 if gpu["vram_gb"] < 14 else 65536 if gpu["vram_gb"] < 20 else 131072
    if a.context:
        ctx = a.context
    else:
        say()
        say("  Context length = how much text the model can see at once (your chat, files, tool output).")
        say("  Longer needs more VRAM for it, so fewer experts fit on the GPU:")
        for i, c in enumerate(CONTEXTS, 1):
            say(f"  {i}) {c // 1024}K tokens" + ("   (recommended for your GPU)" if c == rec_ctx else ""))
        ctx = CONTEXTS[int(ask("Context?", [str(i) for i in range(1, 6)], str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    if model == "IQ3_XXS" and ram < 90 and ctx > 131072:
        warn("IQ3_XXS with a 262K context needs more than 64 GB of RAM (43 GB of experts + the context): using 128K")
        ctx = 131072
    ok(f"context: {ctx} tokens")
    if a.vision:
        vision = {"yes": "gpu", "no": "none"}.get(a.vision, a.vision)
    else:
        say()
        say("  Images: the model can also read pictures (screenshots, photos, scanned pages). This adds a 0.9 GB")
        say("  download and keeps ~1.4 GB of VRAM free for the image encoder, so text is a few % slower.")
        vision = "gpu" if ask("Do you want images?", ["y", "n"], "n", a.yes) == "y" else "none"
    ok("images: " + {"none": "off", "gpu": "on", "cpu": "on (encoder on the CPU)"}[vision])
    models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / tag
    shards = [models_dir / fam["file"].format(q=model, i=i) for i in (1, 2)]
    have_model = all(s.exists() and (done(s) or a.gguf_dir) for s in shards)
    need = (0 if a.gguf_dir or have_model else MODELS[model]["download_gb"]) + 8 + \
        (40 if model == "Q2_0" and avx512 and family == "qwen" else 0) + (1 if vision != "none" else 0)
    if free_gb(models_dir) < need:
        fail(f"not enough free disk space in {models_dir}: need ~{need:.0f} GB", "use --models-dir on a bigger drive")

    # ---- 3. python packages
    step(3, "Python packages")
    pip_install(PY_PACKAGES, "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja")

    # ---- 4. the engine
    step(4, "the Strata engine")
    llama = get_llama_cpp()
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py, ggml, mtmd)")
    eng = None if a.build else get_prebuilt(a.prebuilt, gpu, vision)
    if eng is not None and json.loads((eng / "BUILD.json").read_text()).get("source") != "local":
        pip_install(CUDA_WHEELS, "NVIDIA CUDA libraries (cuBLAS, CUDA runtime; ~0.4 GB)")
        if vision != "none" and not (eng / VEXE).exists():
            warn("the ready-made engine has no image encoder: compiling it")
            eng = None
    if eng is None:
        eng = build_engine(gpu, vision, a.yes, llama)
    meta = json.loads((eng / "BUILD.json").read_text())
    lib_dirs = meta.get("cuda_dirs") or cuda_lib_dirs()
    ok(f"engine: {eng / EXE}")

    # ---- 5. the model files
    step(5, f"downloading {fam['title']} {model}")
    if not a.gguf_dir:
        for s in shards:
            if s.exists() and done(s):
                ok(f"{s.name} already downloaded")
                continue
            # the original's shard 2 is the same file for all three sizes: reuse one that is already here
            other = [p for p in Path(a.models_dir).glob("*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf") if done(p)]
            if family == "qwen" and s.name.endswith("00002-of-00002.gguf") and other and not s.exists():
                try:
                    os.link(other[0], s)
                    mark(s)
                    ok(f"{s.name} shared with {other[0].parent.name} (identical file)")
                    continue
                except OSError:
                    pass
            download(fam["hf"].format(q=model) + s.name, s)
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
    ok("model files present")
    mmproj = Path(a.models_dir) / fam["mmproj"]
    if vision != "none":
        if not mmproj.exists() and a.gguf_dir and (Path(a.gguf_dir) / fam["mmproj"]).exists():
            mmproj = Path(a.gguf_dir) / fam["mmproj"]
        else:
            download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
        ok(f"vision encoder: {mmproj}")

    # ---- 6. the pack and the MTP draft layer
    step(6, "preparing the model for Strata")
    pack = ROOT / "packs" / tag.lower()
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    if model == "Q2_0" and avx512 and family == "qwen":
        # the Q2_0 experts repacked for the AVX-512 kernel (the measured speed): a one-time ~40 GB conversion
        if not (pack / "index.txt").exists() or not (pack / "experts.bin").exists():   # index.txt is written last
            say("  Converting the Q2_0 experts for the AVX-512 kernel (one time, ~40 GB written, 2-5 min) ...")
            run([sys.executable, str(ROOT / "tools" / "strata_pack.py"), "build", "--gguf", str(shards[0]),
                 "--out", str(pack), "--skip-hash"], env=env)
            run([sys.executable, str(ROOT / "tools" / "pack_index.py"), "--pack", str(pack)], env=env)
        if not (pack / "tokenizer" / "vocab.json").exists():
            run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]),
                 "--out", str(pack)], env=env)   # writes <pack>/tokenizer/
    elif not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        # every tensor as the GGUF stores it; the experts are read from the GGUF at start (seconds to build)
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack)], env=env)
    ok(f"model prepared: {pack}")
    mtp = ROOT / "mtp"
    rt = mtp / "rt"
    if not (rt / "experts.bin").exists():
        say("  The MTP draft layer (speculative decoding, ~2x faster output) comes from the original Qwen checkpoint:")
        say("  only its ~5 GB of MTP tensors are downloaded.")
        run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(rt)],
            env=env)
    if not (rt / "draft_vocab.bin").exists():
        shutil.copyfile(ROOT / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
    ok(f"MTP draft layer: {rt}")

    # ---- 7. the start script
    step(7, "writing the start script")
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile                   # the PLE table's shard: shard 2 (original) or 1 (Swift)
    ple = next((s for s in shards if any(t.name == "per_layer_token_embd.weight" for t in GGUFFile(s).tensors)), None)
    if ple is None:
        fail("the model has no per_layer_token_embd tensor (is this a Qwen3.8-Flash-Next GGUF?)")
    args = ["--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(ple),
            "--expert-profile", str(ROOT / "data" / "expert-profile.bin"), "--expert-cache", "auto",
            "--prefill", "2048", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt),
            "--max-context", str(ctx)]
    if ctx > 8192:
        args += ["--kv", "int8"]
    if vision != "none":
        args += ["--vision", "--vram-reserve-mib", str(VISION[vision]["reserve_mib"])]
    cfg = {"exe": str(eng / EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{model.lower()}", "log": str(ROOT / f"strata-{tag.lower()}.log"),
           "lib_dirs": lib_dirs, "port": a.port}
    if vision != "none":
        cfg["vision"] = {"exe": str(eng / VEXE), "mmproj": str(mmproj), "model": str(shards[0]),
                         "gpu": vision == "gpu", "max_tokens": VISION[vision]["max_tokens"]}
        if vision == "cpu":
            cfg["vision"]["threads"] = max(1, (os.cpu_count() or 8) // 2)
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    script = write_run_script(tag, cfg_path, a.port)
    ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{a.port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{a.port}/v1/messages")
    say(f"  Next time:        just run {'START-HERE.bat' if WIN else './setup.sh'} (or {script.name}) - it starts right away")
    if vision != "none":
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if a.no_start:
        return 0
    return start(cfg_path, a.port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
