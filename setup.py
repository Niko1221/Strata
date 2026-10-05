#!/usr/bin/env python3
"""Strata one-click setup and start (Windows and Linux, NVIDIA or AMD GPUs).

    START-HERE.bat  (Windows)   /   ./setup.sh  (Linux)      - they install Python if needed and run this file

The first time it asks four questions - which model (the original Qwen3.8-Flash-Next or the Swift 1.5 fine-tune),
which size, how much context, and whether the model should also read images - then installs everything and starts the model on http://127.0.0.1:8080 (OpenAI- and Anthropic-compatible
API; a small page there shows that it runs). Every later start skips straight to running the model: nothing that
is already downloaded, installed or prepared is done again.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: NVIDIA or AMD GPU and driver, RAM, CPU, free disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv (numpy, jinja2, ..., and NVIDIA's CUDA libraries)
  4. gets the Strata engine: a ready-made build for RTX 20/30/40/50 cards (no compiler needed); if none fits your PC,
     it installs the build tools (asks first) and compiles the engine for your GPU.  AMD (--backend hip, chosen by
     itself on a PC with no usable NVIDIA card): the ready-made HIP engine on Windows, compiled here on Linux
  5. downloads the model from Hugging Face (resumable), and the vision encoder if you want images
  6. prepares the model for Strata and fetches the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  7. writes run-<model>.bat / run-<model>.sh and starts the model

Options: --family qwen|swift, --model Q2_0|IQ2_XS|IQ3_XXS|IQ3_S, --context 32768, --rope-scaling none|linear|yarn
(--rope-scale F; past the trained 262144 the setup adds yarn and the factor is the final context over 262144,
at least 1 - an explicit --rope-scaling none is refused for such a context), --vision yes|no|gpu|cpu, --port
8080, --yes (recommended
answers, no questions), --setup (install another model / change settings instead of starting), --no-start,
--host 0.0.0.0 --api-key KEY (reach it from other devices on your network), --experimental-speed-projection on|off
(EXPERIMENTAL, off by default),
--models-dir DIR, --gguf-dir DIR (use GGUF files you already have), --build (compile instead of the ready-made
engine), --check (only check this PC), --resident-budget-gib N (UD-Q4_K_XL's or UD-IQ4_XS's experts in RAM),
--kv-streaming on|off|auto.

Setup recommends, it never forces: the recommended answers are the defaults (--yes, or Enter), and a bigger choice
than it recommends - a longer context, more GPUs, a bigger RAM budget, a size it thinks will not fit - is kept, with
what it risks.  With --yes, an explicit flag (--model, --gpus, ...) is the consent to a risk setup would otherwise
stop at; --yes alone is not.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
# #214: every Hugging Face file comes from a fixed commit of its repository (the `sha` of
# https://huggingface.co/api/models/<repo> when this was pinned), so a checkout installs the same files on any
# day.  A revision the repository no longer has falls back to its current files, with a message (download()).
HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",        # 2026-09-29
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "b22d729eae29b5796f76fb70f91aef549b9fc52c",   # 2026-09-24
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",  # 2026-09-29
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",                   # 2026-09-30
}


HF_DEFAULT = "https://huggingface.co"


def hf_endpoint() -> str:
    """#495: the Hugging Face host - HF_ENDPOINT as huggingface_hub reads it (a mirror, e.g. https://hf-mirror.com),
    else huggingface.co.  The pinned revisions and the SHA-256 checks are the same whichever host serves the files."""
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def hf(repo: str) -> str:
    """The download folder of a Hugging Face repository at its pinned revision."""
    return f"{hf_endpoint()}/{repo}/resolve/{HF_REVISIONS[repo]}/"


def hf_unpinned(url: str) -> str:
    """The same file at the repository's current revision (main)."""
    return re.sub(r"^(https?://[^/]+/.+?/resolve/)[0-9a-f]{40}/", r"\1main/", url, count=1)


HF = hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"

# The ready-made engine: <PREBUILT_URL><asset>, a zip with strata(.exe), strata-vision(.exe) and BUILD.json, built
# by tools/make_release.py.  Set this to the GitHub release download folder when publishing, e.g.
# "https://github.com/<you>/Strata/releases/latest/download/" (or pass --prebuilt / set STRATA_PREBUILT_URL).
# With the default, the release of this checkout's own version (PREBUILT_TAG_URL, CMakeLists.txt's version) is
# tried first and the latest release is the fallback (#214): an older checkout keeps the engine it shipped with.
PREBUILT_URL = "https://github.com/Niko1221/Strata/releases/latest/download/"
PREBUILT_TAG_URL = "https://github.com/Niko1221/Strata/releases/download/v{version}/"
PREBUILT_ASSET = "strata-windows-x64.zip" if WIN else "strata-linux-x64.zip"
# the CUDA libraries the ready-made engine loads (the same CUDA 13.0 it is built with), from NVIDIA's pip packages
CUDA_WHEELS = ["nvidia-cublas==13.0.2.14", "nvidia-cuda-runtime==13.0.96"]
MIN_DRIVER = 580                       # CUDA 13.0
# Older NVIDIA GPUs (experimental): CUDA 13 dropped Pascal (sm_60/61) and Volta (sm_70), so a model whose GPUs include
# one runs a second engine, built with CUDA 12.9 (-DSTRATA_EXPERIMENTAL_SM60=ON) and kept in its own folder: the
# ready-made one is CUDA12_ASSET (Windows; on Linux it is compiled here with a CUDA 12.x toolkit).  One engine runs per
# model, so the choice is per model config, by its oldest GPU; --cuda 12|13 overrides it (docs/OLDER_GPUS.md).
CUDA13_MIN_ARCH = 75                   # the oldest compute capability CUDA 13 compiles for (sm_75, RTX 20)
CUDA12_ASSET = "strata-windows-x64-cuda12.zip" if WIN else "strata-linux-x64-cuda12.zip"
CUDA12_WHEELS = ["nvidia-cublas-cu12==12.9.1.4", "nvidia-cuda-runtime-cu12==12.9.79"]
# CUDA 12.x minor-version compatibility (NVIDIA's table: Linux 525.60.13, Windows 527.41); the wheels match the 12.9.1
# toolkit the CUDA 12 zip is built with (cuBLAS 12.9.1.4, runtime 12.9.79).  Not tested on such an old driver here.
CUDA12_MIN_DRIVER = 528 if WIN else 525
ENGINE12_DIR = "engine-cuda12"
MIN_ENGINE = (0, 1, 39)                # v0.1.39: the #577 file-tier regression fixed, the OpenAI Responses API (#451, Codex), a reply stuck on one token ended (#606), the head before the arena (#620), effort_position (#458), --vram-reserve hot resize opt-in (#533), PR batch; v0.1.38: prompts faster (one gather per expert group #372, the first chunk's PLE rows beside layer 0 #374, DeltaNet three heads per thread #413), --kv q4_0 prompts on tensor cores (#452), Q5_0 experts on the GPU (#473), IQ4_XS on AVX-2 (#415), unbuffered expert loading on Windows (#357 #362), --peer-device (#531), a 6 GB card starts (#496), PR batch; v0.1.37: a silent engine is restarted (#481), Windows AMD counts the desktop's VRAM (#380 #377 #497), a steadier PCIe probe (#485), fixes #496 #495 #498 #505 #493; v0.1.36: a cancelled prompt logged as read so far (#471), the draft-head hint (#474), UPDATE.bat (#475), --expert-profile-save (#477); v0.1.35: Windows AMD uses its bundled HIP runtime (#468 #461), the low-RAM resident mode on Windows 32 GB (#467), fixes #460 #459 #446 #447 #457 #448 #444; v0.1.34: AMD on Windows (a ready-made HIP engine), an MCP server for AI assistants (tools/strata_mcp.py), a shorter README; v0.1.33: a portable image encoder again (#411 #412), setup recommends instead of forcing (#406 #403 #364 #384), fixes #352 #365 #369 #371 #375 #393 #408 #414; v0.1.32: split prompts faster (#340), AMD router +12%, Unsloth Q4 in setup, faster Q4 prompts, #326/#327/#342/#344 fixes, PR batch; v0.1.31: Unsloth UD-Q4_K_XL (experimental), GGUF-in-place low-RAM mode, Windows GGUF load 2x, server race + tokenizer fixes, AMD intrinsics; v0.1.30: short prompts faster (streaming from 1024 tokens), resident low-RAM variant, multi-GPU session carve, RDNA4; v0.1.29: sampled answers faster (split top-k), #154 correctness fixes; v0.1.28: the expert cache reserves the draft head, a cancelled request no longer fails the next; v0.1.27: RTX 20 (sm_75) in the ready-made engine, the HIP build without CUDA headers; v0.1.26: the draft layer's prompt pass in batches; v0.1.25: faster prompts (grouping off the copy engine, fused hyper-connection kernels), AMD HIP backend, --kv k8v4; v0.1.24: long prompts faster (QSA select on tensor cores); v0.1.23: image requests honor sampling, 8 GB cards start, batched verify window; v0.1.22: faster prompts (tensor-core attention), multi-GPU across images/steering/KV streaming; v0.1.21: multi-GPU layer split (--gpus); v0.1.20: system-prompt checkpoint, PCIe probe, hit rate; v0.1.19: penalties
PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "cmake", "ninja", "pillow", "psutil"]
REQUIREMENTS = ROOT / "requirements.txt"   # the same packages and their dependencies, pinned (#214)

MODELS = {
    # the original model only for now: Swift 1.5's Q2_0 files split one layer's experts across the two shards, which
    # the pack tool (tools/iq_pack.py) cannot prepare yet (#171)
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0, "families": ("qwen",)},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3-bit i-quant, better quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9},
    # the original model only (Swift 1.5 has no IQ3_S): matches the full BF16 model on the published benchmarks
    "IQ3_S": {"about": "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a 64 GB PC "
                       "with little else running", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3,
              "families": ("qwen",)},
    # the Coder release: 256 of the 512 experts kept (the ones code, tools and vision use), IQ2_S-IQ4_XS like IQ3_S
    "IQ1_M": {"about": "the Coder's only size: half the experts, stored like IQ3_S (3.5 bits)", "download_gb": 58.4,
              "ram_gb": 32, "arena_gb": 23.4, "families": ("coder",)},
    # EXPERIMENTAL (docs/UNSLOTH_Q4.md): Unsloth's 4-bit file; its 77 GB of experts do not fit a 64 GB PC, so the engine
    # keeps a RAM budget of them (--resident-budget-gib, chosen below) and reads the rest from the GGUF on the SSD
    "UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic), EXPERIMENTAL: the best quality, but most experts come from the "
                            "SSD on a 64 GB PC (7-8.5 tokens/s measured)", "download_gb": 111.3, "ram_gb": 48,
                   "arena_gb": 77.0, "families": ("unsloth",), "budget": True, "nvidia_only": True,
                   "experimental": True},
    # #621: Unsloth's UD-IQ4_XS - IQ3_S gate/up experts with IQ4_NL (43 layers) or Q8_0 (5) downs, the dense side as
    # UD-Q4_K_XL's; three shards.  A regular choice from 0.1.39 (no longer experimental).  Its 59.5 GB of experts: a
    # RAM budget of them, like UD-Q4_K_XL, but far fewer read from the SSD on a 64 GB PC and none from ~80 GB of RAM.
    # Images: the vision path has no restriction for this pack (the same base model and image encoder), so setup asks
    "UD-IQ4_XS": {"about": "~4-bit i-quant (Unsloth Dynamic), between IQ3_S and UD-Q4_K_XL in quality; on a PC with "
                           "less than ~80 GB of RAM part of its experts are read from the SSD",
                  "download_gb": 93.7, "ram_gb": 48, "arena_gb": 59.5, "families": ("unsloth",), "budget": True,
                  "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "engine": (0, 1, 38),
                  "vision": True},
}
# The experimental Unsloth file's four shards at the pinned revision: name -> (bytes, sha256), checked after the
# download (setup trusts no other model file by name and size alone either: check_shards reads their directories).
UNSLOTH_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf":
        (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf":
        (49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf":
        (49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf":
        (12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
}
# #621: UD-IQ4_XS's three shards at the same revision (sizes and SHA-256: the Hub's LFS pointers)
UNSLOTH_IQ4_XS_SHARDS = {
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf":
        (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf":
        (49835229856, "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf":
        (43836407744, "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
}
UNSLOTH_ENGINE = (0, 1, 32)     # the first engine setup configures for UD-Q4_K_XL (0.1.31 ran it by hand)
UNSLOTH_RAM_LEFT_GB = 24        # RAM beside the budget: the OS, the engine, and the file cache the rest is read through
# Contexts past 262144 (the model's trained length) extend it by rope scaling: for the context it will
# serve the setup resolves the method (yarn, or one question when interactive) and derives the factor
# from the final context (final / 262144, at least 1) itself (below), keeps an explicit
# --rope-scaling/--rope-scale, and refuses an explicit --rope-scaling none there - the stock angles past
# the trained range are out of spec.
CONTEXTS = [8192, 32768, 65536, 131072, 262144, 393216, 524288]
# The model families: the same architecture, weights in the same three GSQ-RCO sizes, different files.
FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen; GSQ-RCO quants by ISTA-DASLab",
             "about": "the original model",
             "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF") + "{q}/",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "",
             "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
             "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next"},
    "swift": {"title": "Swift 1.5", "by": "UkisAI's fine-tune of Qwen3.8-Flash-Next",
              "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its authors' numbers)",
              "hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "swift-",
              "mmproj_hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", "name": "swift-1.5",
              "license": "Swift Open License 1.0: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"},
    # ISTA-DASLab's expert-pruned release: half of each layer's experts removed, chosen for code, agentic tool use and
    # vision; its shard 2 (the n-gram table) and vision encoder are the original's files, shared with it
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "by": "ISTA-DASLab's coding version",
              "about": "half the experts (code, tools, images kept): needs ~32 GB of RAM, faster; weaker outside coding",
              "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF") + "{q}/",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "coder-",
              "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"),
              "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-coder",
              "profile": "expert-profile-coder.bin"},
    # Unsloth's UD-IQ4_XS (three shards, #621; a regular choice from 0.1.39) and the EXPERIMENTAL UD-Q4_K_XL (four)
    # of the original model (docs/UNSLOTH_Q4.md); "experimental" and "vision" are per model (MODELS)
    "unsloth": {"title": "Qwen3.8-Flash-Next (Unsloth)", "by": "Unsloth's ~4-bit quantizations",
                "about": "UD-IQ4_XS: a 94 GB download; with less than ~80 GB of RAM part of its experts are read from "
                         "the SSD (UD-Q4_K_XL, 111 GB: experimental)",
                "hf": hf("unsloth/Qwen3.8-Flash-Next-GGUF") + "{q}/",
                "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "shards": 4, "tag": "unsloth-",
                "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-unsloth",
                "vision": False, "pack_args": ["--compat-bf16"],
                "sha256": {**UNSLOTH_SHARDS, **UNSLOTH_IQ4_XS_SHARDS}},
}
MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
# EXPERIMENTAL, off by default (setup asks): a control vector shipped with the repository, see its README
ESP_VECTOR = ROOT / "data" / "experimental-speed-projection" / "Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
# the image encoder on the GPU (~1.2 GB at 1024 image tokens) warms up before the engine starts, so the engine
# sizes its expert slots around it and the default reserve (700 MiB) is enough; engines before 0.1.2 need more
VISION_GPU_SMALL_RESERVE_MIB = 1000    # the tip for images on a <= 12 GB card (the engine's LOW line asked ~1003)
VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 700},
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
            fail("input ended before a setup answer was received",
                 "run setup in a terminal, or pass --yes to accept the recommended answers")
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
def _memory_status():
    """Windows' GlobalMemoryStatusEx: RAM, and the commit limit (ullTotalPageFile = RAM + page file)."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


def ram_gb():
    if WIN:
        return _memory_status().ullTotalPhys / 2**30
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024 / 2**30
    return 0.0


def page_file_gb():
    """The page file's current size (GB) on Windows, None elsewhere.  The graphics card's memory needs room there
    too: under Windows' driver model every allocation on the card is also charged to the commit (RAM + page file),
    so with the page file off or tiny the engine cannot use the free VRAM (issue #60)."""
    if not WIN:
        return None
    m = _memory_status()
    return max(0.0, (m.ullTotalPageFile - m.ullTotalPhys) / 2**30)


def cpu_cores():
    """#642: (performance cores, efficiency cores) of a hybrid CPU (Intel 12th gen+, AMD Zen 5 + Zen 5c), counted
    as the engine's pool counts them (detect_cpu_topology: physical cores, by Windows' EfficiencyClass or Linux's
    cpu_capacity); None on a CPU whose cores are all alike, or when the OS does not say."""
    classes = []                                       # one entry per physical core: its efficiency/capacity class
    try:
        if WIN:
            k32 = ctypes.windll.kernel32
            n = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))   # RelationProcessorCore: the size
            if not n.value:
                return None
            buf = ctypes.create_string_buffer(n.value)
            if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
                return None
            raw, at = buf.raw[:n.value], 0
            while at + 10 <= len(raw):   # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship, Size, then
                rel, size = struct.unpack_from("<II", raw, at)               # PROCESSOR_RELATIONSHIP (Flags,
                if size <= 0:                                                # EfficiencyClass, ...)
                    break
                if rel == 0:
                    classes.append(raw[at + 9])
                at += size
        else:
            seen = {}
            for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
                cap = cpu / "cpu_capacity"
                pkg, core = cpu / "topology" / "physical_package_id", cpu / "topology" / "core_id"
                if not cap.exists():
                    return None
                key = (pkg.read_text().strip(), core.read_text().strip()) if pkg.exists() and core.exists() else cpu.name
                seen.setdefault(key, int(cap.read_text().strip()))
            classes = list(seen.values())
    except (OSError, ValueError, AttributeError):
        return None
    if not classes or max(classes) == min(classes):
        return None
    p = sum(1 for c in classes if c == max(classes))
    return p, len(classes) - p


def hybrid_pool_workers(cores) -> int | None:
    """#642 (Hardin22's measurement): on a hybrid CPU the expert pool runs best on the P-cores but the host loop's one
    plus HALF of the E-cores - an E-core runs the expert kernels ~2.2x slower and each layer waits for its slowest
    part (i9-14900KF, 8P + 16E: 15 workers decoded 165 / 116 tok/s against 106 / 84 with all 23).  Only on a CPU with
    more E-cores than P-cores: on an i7-13700KF (8P + 8E, docs/AMD_HIP.md's gfx1030 report) all 15 workers decoded
    38-42 tok/s against 36 with 8, so there the engine's own count stays.  None: the engine's own default (one worker
    per physical core but the host's) stays."""
    if not cores:
        return None
    p, e = cores
    if e <= p:
        return None
    return max(1, p - 1 + e // 2)


def recommend_pool_workers(args: list) -> list:
    """`args` with setup's recommended `--pool-workers` for a hybrid CPU, unless they set one already (a calibration's
    measured count, or the user's own).  A recommendation: the config line can be edited or removed."""
    n = hybrid_pool_workers(cpu_cores())
    if n is None or "--pool-workers" in args:
        return args
    p, e = cpu_cores()
    ok(f"hybrid CPU ({p} performance + {e} efficiency cores): {n} CPU expert workers - the performance cores and half "
       "of the efficiency cores (--pool-workers in the config; START-HERE --calibrate measures it on this PC)")
    return [*args, "--pool-workers", str(n)]


def cpu_info():
    """(name, avx2, avx512): avx512 means everything Strata's fast AVX-512 kernels use (F, BW, VL, VNNI, VBMI),
    the same test the engine makes (cpu_avx512_ok), not just AVX-512F."""
    name, avx2, avx512 = platform.processor() or "unknown CPU", False, False
    if WIN:
        pf = ctypes.windll.kernel32.IsProcessorFeaturePresent
        avx2 = bool(pf(40)) or _cpuid_avx2()      # PF_AVX2_INSTRUCTIONS_AVAILABLE, else the CPU itself (#159)
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


def _cpuid_floor() -> str:
    """Below AVX2 (Windows): "avx" when the CPU has AVX and the OS saves the YMM registers, "sse4.2" with SSE4.2 and
    POPCNT, else ""."""
    try:
        regs = (ctypes.c_uint32 * 4)()
        _run_stub(bytes([0x53, 0x49, 0x89, 0xC8, 0x89, 0xD0, 0x31, 0xC9, 0x0F, 0xA2,      # push rbx; r8=rcx; eax=edx; ecx=0; cpuid
                         0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04, 0x41, 0x89, 0x48, 0x08,  # [r8]=eax, [r8+4]=ebx, [r8+8]=ecx
                         0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3]),                             # [r8+12]=edx; pop rbx
                  ctypes.addressof(regs), 1)
        ecx1 = regs[2]
        if (ecx1 >> 27) & 1 and (ecx1 >> 28) & 1:                     # OSXSAVE, AVX
            xcr0 = (ctypes.c_uint32 * 2)()
            _run_stub(bytes([0x49, 0x89, 0xC8, 0x31, 0xC9, 0x0F, 0x01, 0xD0,                    # r8=rcx; ecx=0; xgetbv
                             0x41, 0x89, 0x00, 0x41, 0x89, 0x50, 0x04, 0xC3]), ctypes.addressof(xcr0))
            if xcr0[0] & 6 == 6:
                return "avx"
        return "sse4.2" if (ecx1 >> 20) & 1 and (ecx1 >> 23) & 1 else ""
    except Exception:
        return ""


def cpu_floor(avx2: bool) -> str:
    """The experimental older-CPU build this PC needs (#394 #595 #623): "" with AVX2 (the normal engine), "avx" (Sandy /
    Ivy Bridge, AMD Bulldozer), "none" (SSE4.2 + POPCNT: Nehalem, Westmere), or "unsupported".  STRATA_ISA_FLOOR=avx|
    none asks for that build on any PC (testing it on a newer one)."""
    forced = os.environ.get("STRATA_ISA_FLOOR", "").strip().lower()
    if forced in ("avx", "none"):
        return forced
    if avx2:
        return ""
    if WIN:
        f = _cpuid_floor()
    else:
        try:
            txt = open("/proc/cpuinfo").read()
            flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
        except (OSError, AttributeError):
            flags = set()
        f = "avx" if "avx" in flags else "sse4.2" if {"sse4_2", "popcnt"} <= flags else ""
    return {"avx": "avx", "sse4.2": "none"}.get(f, "unsupported")


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


def _run_stub(code: bytes, *args) -> None:
    """Runs a few bytes of x64 machine code (Windows calling convention: the arguments in rcx, rdx)."""
    k32 = ctypes.windll.kernel32
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    buf = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)
    if not buf:
        raise OSError("VirtualAlloc failed")
    try:
        ctypes.memmove(buf, code, len(code))
        ctypes.CFUNCTYPE(None, *[ctypes.c_void_p] * len(args))(buf)(*args)
    finally:
        k32.VirtualFree(buf, 0, 0x8000)


def _cpuid_avx2() -> bool:
    """AVX2 asked from the CPU (CPUID leaf 7 EBX bit 5), with the OS saving the YMM registers (OSXSAVE + XCR0):
    Windows' IsProcessorFeaturePresent(PF_AVX2) says no on some PCs whose CPU has it (a Ryzen 9 3950X, #159)."""
    try:
        def cpuid(leaf):
            regs = (ctypes.c_uint32 * 4)()
            _run_stub(bytes([0x53, 0x49, 0x89, 0xC8, 0x89, 0xD0, 0x31, 0xC9, 0x0F, 0xA2,      # push rbx; r8=rcx; eax=edx; ecx=0; cpuid
                             0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04, 0x41, 0x89, 0x48, 0x08,  # [r8]=eax, [r8+4]=ebx, [r8+8]=ecx
                             0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3]),                             # [r8+12]=edx; pop rbx
                      ctypes.addressof(regs), leaf)
            return list(regs)
        if cpuid(0)[0] < 7:
            return False
        ecx1 = cpuid(1)[2]
        if not (ecx1 >> 27) & 1 or not (ecx1 >> 28) & 1:             # OSXSAVE, AVX
            return False
        xcr0 = (ctypes.c_uint32 * 2)()
        _run_stub(bytes([0x49, 0x89, 0xC8, 0x31, 0xC9, 0x0F, 0x01, 0xD0,                        # r8=rcx; ecx=0; xgetbv
                         0x41, 0x89, 0x00, 0x41, 0x89, 0x50, 0x04, 0xC3]), ctypes.addressof(xcr0))
        if xcr0[0] & 6 != 6:                                           # the OS saves XMM and YMM
            return False
        return bool((cpuid(7)[1] >> 5) & 1)
    except Exception:
        return False


def gpus():
    """Every NVIDIA GPU, numbered as nvidia-smi numbers them (by PCI bus, the order the engine is told to use)."""
    s = out(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"])
    found = []
    for line in s.strip().splitlines():
        try:
            idx, name, mem, cc, drv = [x.strip() for x in line.split(",")]
            found.append({"index": int(idx), "name": name, "vram_gb": float(mem) / 1024.0, "arch": cc.replace(".", ""),
                          "driver": drv})
        except ValueError:
            continue
    return found


GPU_PICK = None                                         # --gpu N (issue #51); None: the card with the most VRAM
SPLIT_MIN_VRAM_GB = 8                                   # a card sharing a model holds the dense weights and its own
                                                        # prompt buffers too (docs/MULTI_GPU.md)
SPLIT_PROMPT_VRAM_GB = 12                               # #448: a split stage lends a prompt chunk's buffers from its
                                                        # own cache; below this one cannot fund a 4096-token chunk
                                                        # (a 10 GB RTX 3080 beside a 32 GB card: 512 tokens, prompts
                                                        # 6.2x slower), while the big card alone could


def cc(g) -> str:
    return f"{g['arch'][:-1]}.{g['arch'][-1]}"


OLD_GPUS = None       # why Pascal / Volta cards are admitted in this run (old_gpus_opt_in), None: they are not


def experimental_sm60() -> bool:
    """#295: STRATA_EXPERIMENTAL_SM60=1 admits Pascal (6.x) and Volta (7.0) cards, run by the experimental CUDA 12
    engine (-DSTRATA_EXPERIMENTAL_SM60=ON).  So does naming such a card (--gpu N / --gpus), --cuda 12, or a PC that
    has no newer card (old_gpus_opt_in)."""
    return os.environ.get("STRATA_EXPERIMENTAL_SM60", "").strip() == "1" or OLD_GPUS is not None


def sm60_card(arch) -> bool:
    return 60 <= int(arch) <= 70


def old_gpus_opt_in(found, named=(), cuda=None, other=False):
    """Why this run may use Pascal / Volta cards (the experimental CUDA 12 engine), or None.  The cards are an opt-in:
    the user named one (`named`: --gpu / --gpus), asked for --cuda 12, set STRATA_EXPERIMENTAL_SM60=1, or the PC has
    no card the ready-made engine runs on and no supported AMD card (`other`; it used to stop there).  A PC with a
    newer card keeps recommending it."""
    old = [g for g in found if sm60_card(g["arch"])]
    if not old:
        return None
    if os.environ.get("STRATA_EXPERIMENTAL_SM60", "").strip() == "1":
        return "STRATA_EXPERIMENTAL_SM60=1"
    if str(cuda) == "12":
        return "--cuda 12"
    picked = [g for g in old if g["index"] in set(named)]
    if picked:
        return "you chose " + ", ".join(f"GPU {g['index']} ({g['name']})" for g in picked)
    if not other and not any(int(g["arch"]) >= CUDA13_MIN_ARCH for g in found):
        return "it is the only kind of NVIDIA GPU in this PC"
    return None


def named_gpus(gpu, gpus) -> list:
    """The card numbers --gpu / --gpus name (an unreadable value: none; parse_gpus says what is wrong later)."""
    try:
        if gpus and str(gpus).strip().lower() != "all":
            return [int(x) for x in str(gpus).split(",") if x.strip()]
        return [int(gpu)] if gpu is not None else []
    except ValueError:
        return []


def cuda_choice(archs, cuda=None):
    """The CUDA toolkit of one model's engine: (12 or 13, why).  13 (the ready-made engine) unless a card is older
    than CUDA 13 supports (Pascal / Volta: CUDA 13 cannot compile for them) - one engine runs per model, so its oldest
    card decides.  `cuda` (--cuda 12|13) overrides it; setup recommends, it does not refuse (the caller warns)."""
    archs = sorted({int(x) for x in archs})
    old = [a for a in archs if a < CUDA13_MIN_ARCH]
    if str(cuda) == "13":
        return 13, ("--cuda 13 (as you chose)" + (f"; CUDA 13 has no code for sm_{old[0]}: the engine will not run "
                                                   "on that card" if old else ""))
    if str(cuda) == "12":
        return 12, "--cuda 12 (as you chose" + ("; RTX 50 (sm_120) engines built with CUDA 12.8 crashed on long "
                                                  "prompts, #220" if archs and archs[-1] >= 120 else "") + ")"
    if old:
        return 12, (f"sm_{old[0]} is older than CUDA 13 supports (it dropped Pascal and Volta): this model runs the "
                    "experimental CUDA 12 engine")
    return 13, None


def engine_dir(toolkit=13) -> Path:
    """The folder of the engine a model runs: engine/ (CUDA 13, or HIP), engine-cuda12/ (the experimental one)."""
    return ROOT / (ENGINE12_DIR if int(toolkit) == 12 else "engine")


def config_toolkit(cfg: dict) -> int:
    """12 when a model config runs the experimental CUDA 12 engine (its exe is in engine-cuda12/), else 13."""
    return 12 if cfg.get("cuda") == 12 or Path(str(cfg.get("exe", ""))).parent.name == ENGINE12_DIR else 13


def gpu_problem(g, together=False):
    """Why Strata cannot use this card, in plain words (None: it can)."""
    if int(g["arch"]) < 75 and not (sm60_card(g["arch"]) and experimental_sm60()):
        return (f"not supported - older than the RTX 20 series (compute capability {cc(g)}; Strata needs 7.5 or "
                "newer" + ("; experimental: choose it with --gpu " + str(g["index"]) + " (the CUDA 12 engine, "
                          "docs/OLDER_GPUS.md)" if sm60_card(g["arch"]) else "") + ")")
    if together and g["vram_gb"] < SPLIT_MIN_VRAM_GB - 0.5:
        return (f"not supported together with other GPUs - {g['vram_gb']:.0f} GB of VRAM (a card sharing the model "
                f"needs {SPLIT_MIN_VRAM_GB} GB or more)")
    return None


def gpu_rank(g):
    """The order cards share a model in: the newest generation first (it gets the first layers and most of the
    work), then the most VRAM."""
    return (-int(g["arch"]), -round(g["vram_gb"]), g["index"])


def gpu_name(g) -> str:
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB)"


def gpu_table(found) -> None:
    say("  Your NVIDIA GPUs:")
    for g in found:
        p = gpu_problem(g)
        say(f"    GPU {g['index']}: {g['name']}, {g['vram_gb']:.0f} GB VRAM - " + ("can be used" if p is None else p))


def together_ok(found) -> list:
    """The cards that can share one model, in the order they would (empty if fewer than two)."""
    ok_ = sorted([g for g in found if gpu_problem(g, together=True) is None], key=gpu_rank)
    return ok_ if len(ok_) >= 2 else []


def split_short(cards) -> list:
    """#448: the later cards of a split that would cap its prompt chunk below what the first card alone reads (a card
    under SPLIT_PROMPT_VRAM_GB beside one that has it).  Empty: the split is recommended as before."""
    if len(cards) < 2 or cards[0]["vram_gb"] < SPLIT_PROMPT_VRAM_GB - 0.5:
        return []
    return [g for g in cards[1:] if g["vram_gb"] < SPLIT_PROMPT_VRAM_GB - 0.5]


def split_short_note(g) -> str:
    return (f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB) is too small to lend a split its prompt buffers: "
            "it would cap prompt reading at 512-2048-token chunks, several times slower than the first card alone "
            "(#448). It can serve as a helper expert cache instead (docs/SECOND_GPU.md)")


def parse_gpus(text, found) -> list:
    """--gpus / --gpu with several: "0,2" or "all" (every card that can share the model)."""
    if str(text).strip().lower() == "all":
        sel = [g["index"] for g in together_ok(found)]
        if not sel:
            gpu_table(found)
            fail("--gpus all: this PC does not have two GPUs Strata can use together")
        return sel
    try:
        sel = [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,2 (or --gpus all), not {text!r}")
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail("--gpus takes two or more different GPUs, e.g. --gpus 0,2 (one GPU: --gpu 0)")
    return sel


def check_gpus(sel, found, what="", yes=False, named=False) -> None:
    """Stops with a plain message when a chosen card is missing or cannot be used, and says what can.  named: the user
    named these cards (--gpus 0,1, or a config that has them): a card that is only short of VRAM for sharing the model
    is then a risk to confirm, not a stop (the owner's rule; --yes with the named cards is the consent)."""
    together = len(sel) > 1
    for i in sel:
        g = next((x for x in found if x["index"] == i), None)
        p = "not found on this PC" if g is None else gpu_problem(g, together)
        if p is None:
            continue
        if named and g is not None and gpu_problem(g) is None:     # it runs Strata; only its VRAM is small
            confirm_risk(f"GPU {i} ({g['name']}) has {g['vram_gb']:.0f} GB of VRAM: a card sharing the model needs "
                         f"{SPLIT_MIN_VRAM_GB} GB or more (it holds the dense weights of its layers and its own prompt "
                         "buffers), so the model may not start, or run slower than without it", True, yes,
                         f"GPU {i} ({g['name']}) {what}is not used together with other GPUs: {p}",
                         "leave it out of --gpus, or answer y to use it anyway", "  Use it anyway?")
            warn(f"GPU {i} ({g['name']}) is used together with the others, as you chose")
            continue
        say()
        gpu_table(found)
        can = together_ok(found)
        single = [x for x in found if gpu_problem(x) is None]
        ones = " or ".join(f"--gpu {x['index']}" for x in single)
        both = "--gpus " + ",".join(str(x["index"]) for x in can) if can else ""
        hint = ((f"use these together: {both}" + (f" (or one card: {ones})" if not together else "")) if can else
                f"use one card: {ones}" if single else "Strata needs an NVIDIA RTX 20 series or newer card")
        fail(f"GPU {i}{'' if g is None else ' (' + g['name'] + ')'} {what}cannot be used: {p}", hint)


def engine_archs(toolkit=13):
    """The GPU generations the installed engine has code for: (archs, ptx), or None when there is none."""
    info = engine_dir(toolkit) / "BUILD.json"
    try:
        meta = json.loads(info.read_text())
    except (OSError, ValueError):
        return None
    return [int(x) for x in meta.get("archs", [])], bool(meta.get("ptx"))


def engine_archs_hip():
    """The AMD architectures the installed HIP engine was compiled for ("gfx1201", ...), or None."""
    try:
        meta = json.loads((ROOT / "engine" / "BUILD.json").read_text())
    except (OSError, ValueError):
        return None
    return [str(x) for x in meta.get("archs", [])] if meta.get("backend") == "hip" else None


def engine_runs_on(g, toolkit=13) -> bool:
    ea = engine_archs(toolkit)
    if ea is None or not ea[0]:
        return True
    archs, ptx = ea
    return int(g["arch"]) in archs or (ptx and int(g["arch"]) > max(archs))


def start_gpus(text):
    """--gpus when starting an installed model: NVIDIA cards as nvidia-smi numbers them, or on a PC whose AMD cards
    are the ones Strata can use, AMD cards as setup lists them ("all": every supported AMD card)."""
    if not text:
        return None
    if str(text).strip().lower() == "all" and not WIN and not together_ok(gpus()):
        amd = amd_gpus()
        if len([g for g in amd if amd_problem(g) is None]) >= 2:
            return [g["index"] for g in amd_parse_gpus("all", amd)]
    return parse_gpus(text, gpus())


def choose_gpus(a, found) -> list:
    """Which cards this install uses: --gpus / --gpu, or asked when two or more can share the model (the two best
    together recommended), else the supported card with the most VRAM.  Returns their numbers, the main one first."""
    if a.gpus:
        sel = parse_gpus(a.gpus, found)
        check_gpus(sel, found, yes=a.yes, named=str(a.gpus).strip().lower() != "all")
        return sel
    if a.gpu is not None:
        check_gpus([a.gpu], found)
        return [a.gpu]
    single = sorted([g for g in found if gpu_problem(g) is None], key=lambda x: (-round(x["vram_gb"]), x["index"]))
    if not single:
        gpu_table(found)
        fail("none of your GPUs can run Strata", "it needs an NVIDIA RTX 20 series or newer (compute capability 7.5+)")
    can = together_ok(found)
    if not can:
        return [single[0]["index"]]
    say()
    say(f"  Strata can run the model on one GPU, or share it across {'these' if len(can) > 2 else 'both'}: then each"
        " card holds the")
    say("  experts of its own layers, so together they hold about twice as many, and prompts are read about 20%")
    say("  faster (details: docs/MULTI_GPU.md). A much slower extra card can also make it slower.")
    opts = [can[:2]] + ([can] if len(can) > 2 else []) + [[g] for g in single]
    # #448: a pair whose second card cannot lend a 4096-token chunk recommends the first card alone (still offered)
    short = split_short(can[:2])
    rec = next(i for i, o in enumerate(opts, 1) if o == [can[0]]) if short else 1
    for i, o in enumerate(opts, 1):
        label = (" + ".join(gpu_name(g) for g in o) + " together") if len(o) > 1 else gpu_name(o[0]) + " only"
        say(f"  {i}) {label}" + ("   (recommended)" if i == rec else ""))
    for g in found:
        if gpu_problem(g, together=True) is not None:
            say(f"     (GPU {g['index']}, {g['name']}: {gpu_problem(g, together=True)})")
    for g in short:
        say(f"     ({split_short_note(g)})")
    pick = opts[int(ask("Which GPUs?", [str(i) for i in range(1, len(opts) + 1)], str(rec), a.yes or a.check)) - 1]
    return [g["index"] for g in pick]


def split_mmap(cfg: dict) -> bool:
    """#364 #384: the low-RAM mode's resident variant (--resident-experts) has no layer split yet.  A config with it
    that runs on several GPUs reads the experts the GPUs do not hold through the OS file cache instead
    (--mmap-experts: the placement those reports measured 1.3-1.6x faster than one GPU), said plainly - the engine
    used to refuse the pair.  True when the config changed."""
    a = cfg.get("args", [])
    if "--resident-experts" not in a:
        return False
    a[a.index("--resident-experts")] = "--mmap-experts"
    warn("the low-RAM mode's resident variant (--resident-experts) has no layer split yet: on several GPUs the experts "
         "the GPUs do not hold are read through the OS file cache (--mmap-experts) instead, and RAM can fill up to 0 "
         "free during long prompts. One GPU keeps them in RAM (steady RAM use): START-HERE --setup, or --gpu N for a "
         "start")
    return True


def model_file(fam: dict, model: str, i: int) -> str:
    """Shard i's file name: the family's pattern, or the model's own (#621: UD-IQ4_XS has three shards, not four)."""
    return MODELS.get(model, {}).get("file", fam["file"]).format(q=model, i=i)


def model_shards(fam: dict, model: str) -> int:
    return MODELS.get(model, {}).get("shards", fam.get("shards", 2))


def budget_model(cfg: dict) -> str:
    """The Unsloth model a config with a RAM budget runs, from its --native shard's name (UD-Q4_K_XL by default)."""
    a = cfg.get("args", [])
    native = Path(a[a.index("--native") + 1]).name.upper() if "--native" in a and a.index("--native") + 1 < len(a) \
        else ""
    return next((m for m, d in MODELS.items() if d.get("budget") and f"-{m}-" in native), "UD-Q4_K_XL")


def unsloth_split_need_gb(model="UD-Q4_K_XL") -> float:
    """#498: the RAM UD-Q4_K_XL needs on several GPUs, where it has no RAM budget (the engine refuses
    --resident-budget-gib with a layer split): its GGUF files and UNSLOTH_RAM_LEFT_GB more (~135 GB).  Measured safe
    at 165 GiB (2x RTX 3090: MemAvailable never under 68 GiB); the 0-free case of #384 was 47 GB with a 70 GB model."""
    return MODELS[model]["download_gb"] + UNSLOTH_RAM_LEFT_GB


def split_budget(cfg: dict) -> bool:
    """#498: a UD-Q4_K_XL config (its RAM budget, --resident-budget-gib) started on several GPUs.  The engine refuses
    the budget with a layer split (it exited with code 2), so the split runs without it - all the experts loaded into
    RAM at start - where the RAM holds the GGUFs and 24 GB more; else setup stops, before the config is saved.  True
    when the config changed."""
    a = cfg.get("args", [])
    if "--resident-budget-gib" not in a:
        return False
    model = budget_model(cfg)
    need, ram = unsloth_split_need_gb(model), ram_gb()
    if ram < need:
        fail(f"{model} cannot share its RAM budget across GPUs (the engine has no layer split with it), and without "
             f"the budget it needs ~{need:.0f} GB of RAM (its GGUF files and {UNSLOTH_RAM_LEFT_GB} GB more); this PC "
             f"has {ram:.0f} GB", "start it on one GPU: START-HERE.bat --gpu N (Linux: ./setup.sh --gpu N)")
    i = a.index("--resident-budget-gib")
    del a[i:i + 2]
    ok(f"{model} on several GPUs: no RAM budget (the engine has none with a layer split) - all its experts are "
       "loaded into RAM from the model files at start, and the files pass through the OS file cache (#498)")
    return True


REMOTE_EXPERT_OPT = "--remote-expert-opt"


def recommend_remote_expert_opt(cfg: dict, off: bool = False) -> None:
    """0.1.39b (#578): a config on two or more GPUs gets --remote-expert-opt - the helper expert caches
    (--expert-cache-device1..3) then stay complementary to the main GPU's, return their rows already weighted and skip
    the CPU's activation quantization where no expert is left to it (dual RTX 4090: +63% mixed, +132% code over the
    plain helper path).  The engine uses it only with a helper cache; a layer split runs as before.  A recommendation:
    `off` (setup's --no-remote-expert-opt) or "remote_expert_opt": false in the config keeps it out, and a single-GPU
    config is not touched."""
    if not isinstance(cfg.get("gpu"), list) or len(cfg["gpu"]) < 2:
        return
    args = cfg.setdefault("args", [])
    if off or cfg.get("remote_expert_opt") is False:
        if REMOTE_EXPERT_OPT in args:
            args.remove(REMOTE_EXPERT_OPT)
        return
    if REMOTE_EXPERT_OPT not in args:
        args.append(REMOTE_EXPERT_OPT)
        ok("multi-GPU: --remote-expert-opt (helper expert caches complementary to the main GPU's, #578; "
           "--no-remote-expert-opt leaves it out)")


def offer_together(cfg_path: Path, cfg: dict, yes: bool) -> dict:
    """Starting a model set up for one card on a PC with two or more that can share it: asked once (the answer is
    saved in its config)."""
    if isinstance(cfg.get("gpu"), list) or cfg.get("gpus_asked"):
        return cfg
    found = gpus()
    can = together_ok(found)
    if not can:
        return cfg
    # #498: UD-Q4_K_XL's RAM budget has no layer split; without it the RAM must hold the GGUFs and 24 GB more
    budget = "--resident-budget-gib" in cfg.get("args", [])
    if budget and ram_gb() < unsloth_split_need_gb(budget_model(cfg)):
        return cfg
    pair = can[:2]
    cfg["gpus_asked"] = True
    # #364 #384: the resident low-RAM variant stays on one card unless the user says otherwise (its RAM use is steady)
    resident = "--resident-experts" in cfg.get("args", [])
    say()
    say("  This PC has " + " and ".join(gpu_name(g) for g in pair) + ": Strata can share the model across both.")
    say("  Together they hold about twice the model's experts and read prompts about 20% faster (docs/MULTI_GPU.md).")
    if resident:
        say("  This model runs in the low-RAM mode with its experts kept in RAM, on one GPU (recommended: steady RAM")
        say("  use). On both, the experts the GPUs do not hold are read through the OS file cache instead: faster in")
        say("  two reports (#364, #384), but RAM can fill up to 0 free during long prompts.")
    if budget:
        say(f"  This model ({budget_model(cfg)}) runs on one GPU with a RAM budget of its experts (recommended: the tested")
        say("  setup). On both it has no budget: all its experts are loaded into RAM at start, which this PC's RAM")
        say("  holds - about twice as fast in #498 (2x RTX 3090: 31 -> 64-78 tokens/s).")
    short = split_short(pair)             # #448: one card recommended (asked "n" by default), as for --resident
    for g in short:
        say(f"  {split_short_note(g)}.")
    tk = config_toolkit(cfg)
    missing = [g for g in pair if not (engine_runs_on(g) if tk == 13 else engine_runs_on(g, tk))]
    if missing:
        say("  The installed engine has no code for " + ", ".join(g["name"] for g in missing) + ": to use them "
            "together, run START-HERE.bat --setup --gpus " + ",".join(str(g["index"]) for g in pair))
    elif ask("  Use both from now on? (you can change it later: START-HERE.bat --gpu N for one card)",
             ["y", "n"], "n" if resident or short or budget else "y", yes) == "y":
        cfg["gpu"] = [g["index"] for g in pair]
        cfg["layer_split"] = cfg.get("layer_split") or "auto"
        split_mmap(cfg)
        split_budget(cfg)
        recommend_remote_expert_opt(cfg)
        ok("from now on this model runs on " + " + ".join(gpu_name(g) for g in pair))
    else:
        ok("staying on one GPU (START-HERE.bat --gpus " + ",".join(str(g["index"]) for g in pair) + " switches)")
    write_config(cfg_path, cfg)
    return cfg


def gpu_info(pick=None):
    """The GPU Strata runs on: `pick` (nvidia-smi's number) if given, else the one with the most VRAM (ties: the
    lower number).  None when there is no NVIDIA GPU.  The dict also says how many there are ("count")."""
    found = gpus()
    if not found:
        return None
    pick = GPU_PICK if pick is None else pick
    if pick is not None:
        g = next((x for x in found if x["index"] == pick), None)
        if g is None:
            fail(f"there is no GPU {pick}: " + ", ".join(f"{x['index']} = {x['name']}" for x in found))
    else:
        g = max(found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
    return {**g, "count": len(found)}


def find_nvcc(below=None):
    """The newest CUDA toolkit's nvcc and its (major, minor); with `below`, the newest older than that version.
    #601: STRATA_NVCC=<path to nvcc> is the only one considered (a newer toolkit beside it that cannot build on this
    PC - CUDA 12.9 with glibc 2.43 - is not taken instead)."""
    pick = os.environ.get("STRATA_NVCC")
    if pick:
        if not Path(pick).exists():
            warn(f"STRATA_NVCC={pick}: no such file; looking for a CUDA toolkit as usual")
        else:
            v = re.search(r"release (\d+)\.(\d+)", out([pick, "--version"]))
            ver = (int(v.group(1)), int(v.group(2))) if v else None
            if ver and below is not None and ver >= below:
                warn(f"STRATA_NVCC={pick} is CUDA {ver[0]}.{ver[1]}; this build needs one older than "
                     f"{below[0]}.{below[1]}")
                return (None, None)
            return (pick, ver) if ver else (None, None)
    cands = [shutil.which("nvcc")]
    for var in ("CUDA_PATH", "CUDA_HOME"):           # CUDA_HOME: Linux's usual name (#601)
        if os.environ.get(var):
            cands.append(str(Path(os.environ[var]) / "bin" / ("nvcc.exe" if WIN else "nvcc")))
    if WIN:
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.exists():
            cands += [str(p / "bin" / "nvcc.exe") for p in sorted(base.iterdir(), reverse=True)]
    else:
        cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/usr/local").glob("cuda*"), reverse=True)]
        cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/opt").glob("cuda*"), reverse=True)]   # Arch (#46)
    best = (None, None)
    for c in dict.fromkeys(cands):                     # every toolkit found; the newest wins
        if c and Path(c).exists():
            v = re.search(r"release (\d+)\.(\d+)", out([c, "--version"]))
            ver = (int(v.group(1)), int(v.group(2))) if v else None
            if ver and (below is None or ver < below) and (best[1] is None or ver > best[1]):
                best = (c, ver)
    return best


def find_vcvars():
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        return None
    # CUDA 13 accepts Visual Studio 2019 and 2022 only: a newer one (2026 = version 18) installed next to them
    # must not be picked ("unsupported Microsoft Visual Studio version"); with only a newer one there is none
    p = out([str(vswhere), "-latest", "-products", "*", "-version", "[16.0,18.0)", "-requires",
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
def drop_archive(z: Path) -> None:
    """An unpacked or refused engine archive and its .done mark go: a refused one kept them, and every later run
    reused it ("already downloaded") instead of the published one (PR #324)."""
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)


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
        except urllib.error.HTTPError as e:
            if e.code == 404 and hf_unpinned(url) != url:  # #214: the pinned revision is gone from the repository
                warn(f"{what or dst.name}: not at the pinned revision any more; downloading the repository's "
                     "current file")
                url = hf_unpinned(url)
                continue
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", "check your internet connection and run it again")
            time.sleep(5)
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
        fail(f"could not finish downloading {dst.name}: {part.stat().st_size:,} bytes on disk, the server says {total:,}",
             "check your internet connection and run it again (the download resumes where it stopped)")
    part.replace(dst)
    mark(dst)
    ok(f"{what or dst.name} downloaded")


def whole_shard(s: Path) -> bool:
    """A shard as long as its own tensor directory says (check_shards' test, without stopping setup)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(s)
        return s.stat().st_size >= g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors),
                                                     default=0)
    except (OSError, ValueError, struct.error):
        return False


SHARD_NAME = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def gguf_dir_shards(folder: Path, fam: dict, model: str) -> list[Path]:
    """--gguf-dir's shards (#305): every -0000i-of-0000N file of the model, N read from the first shard's name as the
    engine and tools/iq_pack.py do.  The published name first; else the one first shard in the folder whose name has
    the size in it (an upload split or named differently: -00001-of-00003, Unsloth's ...-00001-of-00004.gguf).  A
    missing shard is check_shards' error later, as before."""
    first = folder / model_file(fam, model, 1)
    if not first.exists():
        found = sorted(p for p in folder.glob("*-00001-of-*.gguf") if SHARD_NAME.search(p.name))
        mine = [p for p in found if model.lower() in p.name.lower()]
        pick = mine if mine else found
        if len(pick) == 1:
            first = pick[0]
    m = SHARD_NAME.search(first.name)
    total = int(m.group(2)) if m else 1
    if not m or total < 1:
        return [first]
    stem = first.name[:m.start()]
    return [first.with_name("%s-%05d-of-%05d.gguf" % (stem, i, total)) for i in range(1, total + 1)]


# #444: the quantization in a GGUF's name (Unsloth's UD-IQ3_XXS, a K-quant's Q2_K_XL, a GSQ-RCO IQ3_S, ...)
GGUF_QUANT = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32))"
                        r"(?=-\d{5}-of-\d{5}\.gguf$|\.gguf$)", re.I)
SUPPORTED_GGUFS = ("Strata runs ISTA-DASLab's GSQ-RCO files (Qwen3.8-Flash-Next Q2_0, IQ2_XS, IQ3_XXS, IQ3_S; Swift "
                   "1.5's; the Coder's IQ1_M) and Unsloth's UD-Q4_K_XL and UD-IQ4_XS only: other GGUFs (Unsloth's "
                   "UD-IQ3_XXS or "
                   "UD-Q2_K_XL, K-quants) cannot be used")


def gguf_unsupported(name: str) -> str | None:
    """#444: the quantization a GGUF's name says, when it is one Strata cannot run (not a setup size); else None."""
    m = GGUF_QUANT.search(name)
    return m.group(1) if m and m.group(1).upper() not in MODELS and not name.lower().startswith("mmproj") else None


def gguf_choice(name: str) -> tuple | None:
    """#444: (--family, --model) whose published first shard this file is, or None (the Coder's IQ1_M is named like
    the original's sizes: the size tells them apart)."""
    for f, d in FAMILIES.items():
        for m in MODELS:
            if f in MODELS[m].get("families", ("qwen", "swift")) and name == model_file(d, m, 1):
                return f, m
    return None


def gguf_dir_problem(folder: Path, first: Path, fam: dict, model: str) -> tuple | None:
    """#444: (message, hint) when --gguf-dir has no file setup can use for this choice: the chosen shard is a GGUF
    Strata cannot run (Unsloth's UD-IQ3_XXS taken for IQ3_XXS by its name), or it is missing and the folder holds
    other GGUFs - then the hint names the --family/--model of the usable ones.  None otherwise (a missing shard in a
    folder without GGUFs stays check_shards' "missing")."""
    bad = gguf_unsupported(first.name) if first.exists() else None
    if bad:
        return f"{first.name} is {bad}, a GGUF Strata cannot run", SUPPORTED_GGUFS
    if first.exists():
        return None
    firsts = sorted(p.name for p in folder.glob("*.gguf")
                    if not p.name.lower().startswith("mmproj") and (not SHARD_NAME.search(p.name)
                                                                     or SHARD_NAME.search(p.name).group(1) == "00001"))
    usable = list(dict.fromkeys(c for c in map(gguf_choice, firsts) if c))
    unusable = [n for n in firsts if gguf_unsupported(n)]
    if not usable and not unusable:
        return None
    hint = SUPPORTED_GGUFS
    if unusable:
        hint += ".\n       Not usable here: " + ", ".join(unusable)
    if usable:
        hint += ".\n       Usable here: " + ", ".join(f"--family {f} --model {m}" for f, m in usable)
    return f"{folder} has no {fam['title']} {model} file", hint


def verify_sha256(s: Path, size: int, sha: str) -> None:
    """A shard's size and SHA-256 against the pinned values (the Unsloth file); the result is kept in its finish mark,
    so the ~5 minutes of hashing 111 GB happen once.  A wrong file is deleted, so the next run downloads it again."""
    m = s.with_name(s.name + ".done")
    if m.exists() and f"sha256 {sha}" in m.read_text(encoding="utf-8", errors="replace"):
        return
    have = s.stat().st_size if s.exists() else -1
    if have != size:
        fail(f"{s.name} is {have:,} bytes, not {size:,}", "delete it and run setup again (the download restarts)")
    say(f"  checking {s.name} (SHA-256, {size / 1e9:.1f} GB) ...")
    h = hashlib.sha256()
    with open(s, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    if h.hexdigest() != sha:
        s.unlink(missing_ok=True)
        m.unlink(missing_ok=True)
        fail(f"{s.name} has the wrong SHA-256 ({h.hexdigest()}, expected {sha}): deleted",
             "run setup again to download it again")
    mark(s, f"sha256 {sha}")


def resident_budget_gib(model, ram, kv_ram_gb=0.0) -> int:
    """UD-Q4_K_XL: the GiB of experts the engine keeps in RAM (--resident-budget-gib): the RAM (GiB, ram_gb()) less
    24 for the OS, the engine and the file cache the other experts are read through, less a KV cache streamed to
    RAM; at most all of them, at least 8.  64 GB: 40, the measured setting (docs/UNSLOTH_Q4.md)."""
    gib = round(ram) - UNSLOTH_RAM_LEFT_GB - math.ceil(kv_ram_gb)
    return max(8, min(gib, int(MODELS[model]["arena_gb"] / 1.073741824)))


def budget_choice(model, ram, asked) -> float:
    """S4: UD-Q4_K_XL's RAM budget: --resident-budget-gib N as given, else the recommendation (resident_budget_gib).
    More than the recommendation is kept, with what it risks (the owner's rule: setup recommends, it never forces)."""
    rec = resident_budget_gib(model, ram)
    if asked is None:
        return rec
    if asked > rec:
        warn(f"a {asked:g} GiB RAM budget is more than setup recommends for this PC ({rec} GiB: the RAM less "
             f"{UNSLOTH_RAM_LEFT_GB} GB for the OS, the engine and the file cache that reads the other experts). Kept "
             "as you chose: the engine clamps it to the RAM it finds free at start (less 4 GB), and the file cache "
             "gets less room - it may be slower, or run the PC out of RAM under load")
    return int(asked) if asked == int(asked) else asked


def check_shards(shards):
    """Every shard present and whole, or setup stops naming the file and the numbers.  Whole means as long as
    its own tensor directory says (the header is read, the data is not): a truncated copy (--gguf-dir, a .part
    renamed by hand, a download finished by an older setup) otherwise passes as a model file and the engine
    fails much later, at the first tensor that runs past the end."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
        try:
            g = GGUFFile(s)
        except (ValueError, struct.error) as e:
            fail(f"{s.name} is not a whole GGUF shard ({e})", "delete it and run setup again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        have = s.stat().st_size
        if have < need:
            fail(f"{s.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)",
                 "delete it and run setup again (or copy the whole file into --gguf-dir)")


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
        # llama.cpp's own web UI (tools/ui) is not used, and its deep paths passed Windows' 260-character limit in a
        # folder like Downloads\Strata-main\Strata-main (#206)
        f.extractall(tmp, [m for m in f.namelist() if "/tools/ui/" not in m])
    top = next(tmp.iterdir())
    shutil.rmtree(llama, ignore_errors=True)
    # PR #63: on Windows a rename can fail with PermissionError while an antivirus scanner still holds a file of the
    # fresh unpack; shutil.move falls back to copy-and-delete, and a few retries let the scanner finish.  The target
    # is `llama` itself - moving into its parent would keep the zip's `llama.cpp-<sha>` folder name.
    for attempt in range(5):
        try:
            shutil.move(str(top), str(llama))
            break
        except PermissionError:
            if attempt == 4:
                raise
            shutil.rmtree(llama, ignore_errors=True)   # a partial copy from the failed attempt
            time.sleep(2)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    return llama


def req_name(line: str) -> str:
    """The distribution name of a requirement line ("numpy==2.5.3; python_version >= '3.12'" -> "numpy")."""
    return re.split(r"[\s<>=!~;\[]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")


def requirement_lines(path: Path | None = None) -> list[str]:
    """requirements.txt's requirements, comments and blank lines left out."""
    lines = []
    for raw in (path or REQUIREMENTS).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _installed(name: str) -> bool:
    try:
        import importlib.metadata as md
        md.distribution(name)
        return True
    except Exception:
        return False


def pip_install(packages, what):
    """pip install into .venv, skipped when the same list was installed before.  An install from before the pinned
    requirements (#214) recorded bare names: those packages are kept as they are (nothing is reinstalled), and the
    pinned dependencies it already has count as installed."""
    stamp = Path(sys.prefix) / ".strata-pip.json"
    have = json.loads(stamp.read_text()) if stamp.exists() else []
    bare = {p.lower() for p in have if req_name(p) == p.lower()}
    need = [p for p in packages if p not in have and req_name(p) not in bare
            and not (bare and "==" in p and _installed(req_name(p)))]
    if not need:
        ok(f"{what} already installed")
        return
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *need])
    stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
    ok(f"{what} installed")


def cuda_lib_dirs(toolkit=13):
    """Where pip put NVIDIA's CUDA libraries (nvidia/cu13/bin/x86_64 on Windows, nvidia/cu13/lib on Linux).
    toolkit 12: the CUDA 12 wheels' cuBLAS and runtime, in two folders (nvidia/cublas/bin, nvidia/cuda_runtime/bin)."""
    if int(toolkit) == 12:
        patterns = ("cublas64_12.dll", "cudart64_12.dll") if WIN else ("libcublas.so.12*", "libcudart.so.12*")
    else:
        patterns = ("cublas64_13.dll",) if WIN else ("libcublas.so.13*",)
    dirs = []
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        for pattern in patterns:
            for hit in (sp / "nvidia").rglob(pattern) if (sp / "nvidia").is_dir() else []:
                if hit.parent not in dirs:
                    dirs.append(hit.parent)
    return [str(d) for d in dirs]


# ------------------------------------------------------------------------------------------------ AMD
# The RX 7900 XT / XTX (gfx1100) and the RX 9070 series / Radeon AI PRO R9700 (gfx1201) on Linux, through the HIP
# backend (docs/AMD_HIP.md); the RX 7800 XT / 7700 XT (gfx1101, #254) and the RX 9060 XT (gfx1200, #256) were run by
# their owners; the RX 6800 / 6900 series (gfx1030, #311) and the RX 6700 XT (gfx1031, #524) run but are unvalidated.  There is no ready-made AMD engine: ROCm comes from AMD's TheRock Python wheels into .venv (no sudo;
# a system ROCm 7 in /opt/rocm is used when it has hipcc and hipBLAS) and the engine is compiled here for the cards.
# No images yet.
ROCM_INDEXES = {"gfx1100": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",   # TheRock's wheels per GPU family
                "gfx1101": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",
                "gfx1200": "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
                "gfx1201": "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
                "gfx1030": "https://rocm.nightlies.amd.com/v2/gfx103X-all/",
                "gfx1031": "https://rocm.nightlies.amd.com/v2/gfx103X-all/"}
ROCM_VERSION = os.environ.get("STRATA_ROCM_VERSION", "7.10.0a20251120")   # what Strata's HIP build was tested with
ROCM_SYSTEM_MIN = (7, 0)       # an older system ROCm is passed over for the wheels (gfx1201 needs ROCm 6.4 or newer)
AMD_ARCHS = ("gfx1100", "gfx1101", "gfx1200", "gfx1201", "gfx1030", "gfx1031")
AMD_NAMES = {"gfx1100": "AMD Radeon RX 7900 series (gfx1100)",   # when sysfs has no product name
             "gfx1101": "AMD Radeon RX 7800 XT / 7700 XT (gfx1101)",
             "gfx1200": "AMD Radeon RX 9060 series (gfx1200)",
             "gfx1201": "AMD Radeon RX 9070 series / AI PRO R9700 (gfx1201)",
             "gfx1030": "AMD Radeon RX 6800 / 6900 series (gfx1030)",
             "gfx1031": "AMD Radeon RX 6700 XT series (gfx1031)"}
AMD_CARDS = ("the RX 7900 XT / XTX (gfx1100), RX 7800 XT / 7700 XT (gfx1101), RX 9060 XT (gfx1200) and "
             "RX 9070 / 9070 XT / Radeon AI PRO R9700 (gfx1201), and the RX 6800 / 6900 series (gfx1030) and RX 6700 XT "
             "(gfx1031, #524), both unvalidated")


def rocm_index(arch):
    return os.environ.get("STRATA_ROCM_INDEX") or ROCM_INDEXES[arch]


def amd_gpus(sysfs="/sys"):
    """AMD GPUs from the kernel's KFD topology (the amdgpu driver; no ROCm needed), numbered as HIP numbers them:
    the GPU nodes in order, the CPU nodes skipped.  Integrated GPUs are listed too (not supported).
    sysfs: the tree to read (tools/test_setup_amd.py passes a mocked one).  Windows: amd_gpus_win."""
    if WIN:
        return amd_gpus_win()
    base = Path(sysfs) / "class/kfd/kfd/topology/nodes"
    found = []
    if not base.is_dir():
        return found
    for node in sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        try:
            props = {}
            for line in (node / "properties").read_text().splitlines():
                k, _, v = line.partition(" ")
                props[k] = v.strip()
            ver = int(props.get("gfx_target_version") or 0)
            if ver == 0 or int(props.get("simd_count") or 0) == 0:
                continue
        except (OSError, ValueError):
            continue
        arch = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
        dev = Path(sysfs) / f"class/drm/renderD{props.get('drm_render_minor', '')}/device"
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            vram = 0.0
        try:
            name = (dev / "product_name").read_text().strip() or f"AMD Radeon ({arch})"
        except OSError:
            name = f"AMD Radeon ({arch})"
        if name == f"AMD Radeon ({arch})" and arch in AMD_NAMES:
            name = AMD_NAMES[arch]
        found.append({"index": len(found), "name": name, "vram_gb": vram, "arch": arch, "driver": "amdgpu",
                      "vendor": "amd"})
    return found


def amd_problem(g):
    if g["arch"] not in AMD_ARCHS:
        return f"not supported - Strata's AMD backend runs on {AMD_CARDS} only, this is {g['arch']}"
    if g.get("cannot_run"):                            # Windows: the installed engine's own check (--list-devices)
        return g["cannot_run"]
    return None


def amd_gpus_win() -> list[dict]:
    """Windows: the AMD GPUs as the HIP runtime numbers them once the HIP engine is installed (hip_devices), else in
    the display-adapter order (amd_gpus_windows)."""
    return hip_devices() or amd_gpus_windows()


def amd_parse_gpus(text, amd) -> list:
    """--gpus with AMD cards, numbered as HIP numbers them (setup's list): "1,0", or "all" (every supported card, the
    most VRAM first).  Every chosen card must be one Strata supports (AMD_ARCHS; they may be of different
    architectures: the engine is compiled for each).  Returns the cards, the main one first."""
    usable = [g for g in amd if amd_problem(g) is None]
    if str(text).strip().lower() == "all":
        sel = [g["index"] for g in sorted(usable, key=lambda x: (-round(x["vram_gb"]), x["index"]))]
    else:
        try:
            sel = [int(x) for x in str(text).split(",") if x.strip()]
        except ValueError:
            fail(f"--gpus takes AMD GPU numbers as setup lists them, e.g. --gpus 1,0 (or --gpus all), not {text!r}")
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail("--gpus takes two or more different GPUs, e.g. --gpus 1,0 (one GPU: --gpu 1)",
             "this PC has " + (f"{len(usable)} AMD card{'s' if len(usable) != 1 else ''} Strata can use"
                               + (": " + ", ".join(f"GPU {g['index']} ({g['name']})" for g in usable) if usable else "")))
    byid = {g["index"]: g for g in amd}
    for i in sel:
        g = byid.get(i)
        p = "not found on this PC" if g is None else amd_problem(g)
        if p is not None:
            fail(f"AMD GPU {i}{'' if g is None else ' (' + g['name'] + ')'} cannot be used: {p}",
                 ("use these together: --gpus " + ",".join(str(x["index"]) for x in usable)) if len(usable) >= 2 else
                 ("use one card: --gpu " + str(usable[0]["index"])) if usable else f"the AMD backend runs on {AMD_CARDS}")
    return [byid[i] for i in sel]


# ------------------------------------------------------------------------------------------------ AMD on Windows
# Windows has no KFD topology: the cards are found from the display adapters (Win32_VideoController: the ones present,
# with their PCI ids) and the display-class registry (each adapter's 64-bit VRAM size), before any AMD software is
# needed.  The engine is the ready-made HIP one (WIN_HIP_ASSET: strata.exe, strata-device.exe and the ROCm libraries it
# loads, built by tools/hip/build_windows.bat); it needs only the AMD driver.  Once it is installed, the cards are
# numbered as the HIP runtime numbers them (`strata-device --list-devices`): an integrated Radeon takes HIP's device 0
# and pushes the discrete card to 1, which the display-adapter order does not show (#325).
WIN_HIP_ASSET = "strata-windows-x64-hip.zip"
WIN_HIP_MIN_ENGINE = max(MIN_ENGINE, (0, 1, 33))         # the first release with a Windows HIP engine
WIN_AMD_DRIVER = "https://www.amd.com/en/support/download/drivers.html"
# PCI device ids (VEN_1002) of the cards the AMD backend knows; the names below cover a card whose id is not listed
_WIN_AMD_DID = {0x744C: "gfx1100", 0x7448: "gfx1100", 0x745E: "gfx1100",            # RX 7900 XTX/XT/GRE, W7900, W7800
                0x747E: "gfx1101",                                                  # RX 7800 XT / 7700 XT
                0x7480: "gfx1102",                                                  # RX 7600 / 7600 XT
                0x7590: "gfx1200",                                                  # RX 9060 XT
                0x7550: "gfx1201", 0x7551: "gfx1201",                               # RX 9070 / 9070 XT, AI PRO R9700
                0x73BF: "gfx1030", 0x73AF: "gfx1030", 0x73A5: "gfx1030"}            # RX 6800 / 6800 XT / 6900 XT / 6950 XT
_WIN_AMD_NAME = ((re.compile(r"\b9070\b|R9700", re.I), "gfx1201"),
                 (re.compile(r"\b9060\b", re.I), "gfx1200"),
                 (re.compile(r"RX\s*7900|W7900|W7800", re.I), "gfx1100"),
                 (re.compile(r"RX\s*7800|RX\s*7700(?!\s*S)|W7700", re.I), "gfx1101"),
                 (re.compile(r"RX\s*7600|W7600|W7500", re.I), "gfx1102"),
                 (re.compile(r"RX\s*6800(?!\s*[MS])|RX\s*6900|RX\s*6950|W6800", re.I), "gfx1030"))
_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def _win_display_adapters() -> list[dict]:
    """The display adapters present ({"name", "pnp"}), from Win32_VideoController."""
    ps = ("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.PNPDeviceID + '|' + "
          "$_.AdapterRAM }")
    text = out(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps])
    found = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and parts[1]:
            ram = parts[2] if len(parts) > 2 else ""
            found.append({"name": parts[0].strip(), "pnp": parts[1].strip(),
                          "ram": int(ram) if ram.strip().isdigit() else 0})
    return found


def _win_display_registry() -> list[dict]:
    """Each display driver instance's description, matching PCI id and VRAM size (the 64-bit value; the WMI one stops
    at 4 GB), from the display-class registry key - readable without admin."""
    found = []
    try:
        import winreg
        cls = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except (ImportError, OSError):
        return found
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(cls, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            key = winreg.OpenKey(cls, sub)
        except OSError:
            continue
        vals = {}
        for name in ("DriverDesc", "MatchingDeviceId", "HardwareInformation.qwMemorySize", "DriverVersion"):
            try:
                vals[name] = winreg.QueryValueEx(key, name)[0]
            except OSError:
                pass
        found.append(vals)
    return found


def _pci_device_id(text: str) -> int | None:
    m = re.search(r"VEN_1002&DEV_([0-9A-F]{4})", str(text or ""), re.I)
    return int(m.group(1), 16) if m else None


def win_amd_arch(device_id: int | None, name: str) -> str:
    """A Windows AMD adapter's architecture from its PCI device id, else its name; "" when it is none Strata knows
    (an integrated Radeon, an older card)."""
    if device_id in _WIN_AMD_DID:
        return _WIN_AMD_DID[device_id]
    return next((a for rx, a in _WIN_AMD_NAME if rx.search(name or "")), "")


def amd_gpus_windows(adapters=None, registry=None) -> list[dict]:
    """The AMD display adapters present, in display-adapter order (setup's numbering until the HIP engine is
    installed), with the arch Strata would run them as ("" = unknown: listed, not supported).  adapters / registry:
    tools/test_setup_amd.py passes mocked ones."""
    adapters = _win_display_adapters() if adapters is None else adapters
    registry = _win_display_registry() if registry is None else registry
    if not adapters:                                   # no WMI answer: the registry alone (it can list removed cards)
        adapters = [{"name": r.get("DriverDesc", ""), "pnp": r.get("MatchingDeviceId", ""), "ram": 0} for r in registry]
    used = set()
    found = []
    for ad in adapters:
        did = _pci_device_id(ad.get("pnp"))
        if did is None:
            continue                                   # not an AMD (VEN_1002) PCI device
        vram, driver = 0.0, ""
        for k, r in enumerate(registry):               # the same card's driver instance: its 64-bit VRAM size
            if k in used or _pci_device_id(r.get("MatchingDeviceId")) != did:
                continue
            if r.get("DriverDesc") and ad.get("name") and r["DriverDesc"].strip() != ad["name"].strip():
                continue
            used.add(k)
            mem = r.get("HardwareInformation.qwMemorySize")
            if isinstance(mem, bytes):
                mem = int.from_bytes(mem[:8], "little")
            vram = int(mem) / 2 ** 30 if isinstance(mem, int) and mem > 0 else 0.0
            driver = str(r.get("DriverVersion") or "")
            break
        if vram == 0.0 and ad.get("ram", 0) > 0:
            vram = ad["ram"] / 2 ** 30                 # WMI's 32-bit figure (at most 4 GB)
        arch = win_amd_arch(did, ad.get("name", ""))
        name = ad.get("name") or AMD_NAMES.get(arch, f"AMD Radeon (device {did:04X})")
        found.append({"index": len(found), "name": name, "vram_gb": vram, "arch": arch or f"unknown (PCI {did:04X})",
                      "driver": driver or "amd", "vendor": "amd"})
    return found


def hip_devices(probe: Path | None = None, text: str | None = None) -> list[dict] | None:
    """The GPUs the HIP runtime enumerates, numbered as HIP_VISIBLE_DEVICES numbers them, from the installed engine's
    `strata-device --list-devices`; None when there is no HIP engine here or it does not answer.  text: its output
    (tests)."""
    if text is None:
        probe = probe or ROOT / "engine" / ("strata-device.exe" if WIN else "strata-device")
        try:
            hip_engine = json.loads((probe.parent / "BUILD.json").read_text()).get("backend") == "hip"
        except (OSError, ValueError):
            hip_engine = False
        if not probe.exists() or not hip_engine:
            return None
        try:
            hip_runtime_beside_exe(probe.parent)       # #468 #461: not the driver's System32 copy
            env = dict(os.environ)                     # the ready-made engine's ROCm DLLs (rocm/bin beside it)
            env["PATH"] = os.pathsep.join([str(d) for d in hip_lib_dirs(probe.parent)] + [env.get("PATH", "")])
            r = subprocess.run([str(probe), "--list-devices"], capture_output=True, text=True, timeout=120,
                               cwd=str(probe.parent), env=env)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if r.returncode != 0:
            return None
        text = r.stdout
    found = []
    for line in text.splitlines():
        m = re.match(r"device\s+(\d+):\s*(.*)$", line.strip())
        if m:
            found.append({"index": int(m.group(1)), "name": m.group(2).strip(), "vram_gb": 0.0, "arch": "",
                          "driver": "hip", "vendor": "amd"})
            continue
        if not found:
            continue
        a = re.match(r"arch\s+(gfx[0-9a-f]+)\s*,\s*([\d.]+)\s*GiB", line.strip())
        if a and not found[-1]["arch"]:
            found[-1]["arch"], found[-1]["vram_gb"] = a.group(1), float(a.group(2))
        elif line.strip().startswith("cannot run:"):
            found[-1]["cannot_run"] = line.strip()[len("cannot run:"):].strip()
    for g in found:
        if not g["arch"]:
            g["arch"] = "unknown"
        if g["arch"] in AMD_NAMES and g["name"] in ("", "AMD Radeon Graphics"):
            g["name"] = AMD_NAMES[g["arch"]]
    return found


def hip_lib_dirs(eng: Path) -> list[Path]:
    """Where the ready-made Windows HIP engine's ROCm DLLs are (its BUILD.json "lib_dirs", relative to engine/)."""
    try:
        rel = json.loads((eng / "BUILD.json").read_text()).get("lib_dirs") or []
    except (OSError, ValueError):
        rel = []
    return [eng / d for d in rel if (eng / d).is_dir()]


# #468 #461: the HIP runtime the ready-made engine was built with, next to strata.exe.  Windows looks for an imported
# DLL in the exe's folder, then System32, and only then on PATH (where rocm/bin is): an AMD driver that installs its own
# amdhip64_7.dll in System32 won, and the bundled rocBLAS/hipBLAS ran on that runtime - an access violation (W7900) or
# hipErrorInvalidDeviceFunction (7900 XTX) on the first prompt.  Only the runtime and the compiler it loads by name:
# rocBLAS/hipBLASLt stay in rocm/bin, where they find their kernel libraries and ../.kpack.
HIP_RUNTIME_DLLS = ("amdhip64_*.dll", "amd_comgr*.dll")


def hip_runtime_beside_exe(eng: Path) -> None:
    """Copy the bundled HIP runtime DLLs from rocm/bin next to the engine's exes when missing or different (a 0.1.34
    install, whose zip had them in rocm/bin only, is fixed on its next start)."""
    for d in hip_lib_dirs(eng):
        for pat in HIP_RUNTIME_DLLS:
            for src in d.glob(pat):
                dst = eng / src.name
                try:
                    if dst.exists() and dst.stat().st_size == src.stat().st_size and \
                            dst.stat().st_mtime >= src.stat().st_mtime:
                        continue
                    shutil.copy2(src, dst)
                except OSError as e:                   # e.g. the engine is running and holds the old copy
                    warn(f"could not put {src.name} next to the AMD engine ({e}); if the engine stops on its first "
                         "request, close Strata and run START-HERE.bat again")


def hip_match(card: dict, listed: list[dict], hip: list[dict]) -> dict | None:
    """The HIP device that is setup's `card` (from `listed`, the display-adapter order): the k-th device of the same
    architecture, k = the card's rank among the listed cards of that architecture.  None when HIP has no such card."""
    same = [g["index"] for g in listed if g["arch"] == card["arch"]]
    k = same.index(card["index"]) if card["index"] in same else 0
    cand = [g for g in hip if g["arch"] == card["arch"]]
    return cand[k] if k < len(cand) else None


def hip_card(eng: Path, gpu: dict, listed: list[dict]) -> dict:
    """Windows: setup's chosen AMD card as the installed HIP engine numbers it (HIP_VISIBLE_DEVICES), checked by the
    engine itself before the model download: an integrated Radeon is HIP's device 0 (#325), and a PC without a
    working AMD driver stops here with what to install."""
    hip = hip_devices(eng / "strata-device.exe")
    hint = (f"install or update the AMD driver (AMD Software: Adrenalin Edition) from {WIN_AMD_DRIVER}, restart the "
            f"PC and run this again; {eng / 'strata-device.exe'} --list-devices shows what the HIP runtime sees")
    if not hip:
        fail("the AMD HIP runtime finds no GPU (the ready-made engine's device check)", hint)
    m = hip_match(gpu, listed, hip) if gpu.get("driver") != "hip" else \
        next((x for x in hip if x["index"] == gpu["index"]), None)
    if m is None:
        fail(f"the HIP runtime does not list your {gpu['name']} ({gpu['arch']})", hint)
    if amd_problem(m):
        fail(f"HIP device {m['index']} ({m['name']}) cannot be used: {amd_problem(m)}")
    if gpu.get("driver") != "hip" and m["index"] != gpu["index"]:
        ok(f"HIP numbers this card {m['index']} (an integrated GPU comes first): the engine is pointed at it")
    return {**gpu, "index": m["index"], "count": len(hip), "vram_gb": m["vram_gb"] or gpu["vram_gb"], "driver": "hip"}


def get_prebuilt_hip(url_base, gpu, updating=False) -> Path | None:
    """The ready-made Windows HIP engine (WIN_HIP_ASSET) in engine/, kept between runs; None when it cannot be had
    (not published for this version, no internet) or has no code for the card."""
    eng = ROOT / "engine"
    info = eng / "BUILD.json"
    if info.exists() and (eng / EXE).exists():
        try:
            meta = json.loads(info.read_text())
        except ValueError:
            meta = {}
        ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
        if meta.get("backend") == "hip" and meta.get("source") == "prebuilt" and ver >= WIN_HIP_MIN_ENGINE and \
                gpu["arch"] in meta.get("archs", []) and not updating:
            ok("ready-made AMD engine already installed")
            return eng
    if not url_base:
        return None
    eng.mkdir(exist_ok=True)
    z = eng / WIN_HIP_ASSET
    bases = prebuilt_bases(url_base)
    for i, base in enumerate(bases):
        if not base.startswith(("http://", "https://")):
            break
        try:
            req = urllib.request.Request(base + WIN_HIP_ASSET, method="HEAD", headers={"User-Agent": "strata-setup"})
            urllib.request.urlopen(req, timeout=60).close()
            break
        except OSError as e:
            if i + 1 < len(bases):
                say(f"  No ready-made AMD engine for v{source_version()} ({e}): the latest release instead")
                continue
            warn(f"no ready-made AMD engine at {base} ({e})")
            return None
    say("  Downloading the ready-made Strata engine for AMD GPUs (with the ROCm libraries it uses) ...")
    download(base + WIN_HIP_ASSET, z, "Strata AMD engine")
    tmp = eng / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    try:
        meta = json.loads((tmp / "BUILD.json").read_text())
    except (OSError, ValueError):
        meta = {}
    ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
    why = None
    if meta.get("backend") != "hip" or not (tmp / EXE).exists():
        why = "it is not a HIP engine"
    elif ver < WIN_HIP_MIN_ENGINE:
        why = f"it is version {meta.get('version')}; this setup needs {'.'.join(map(str, WIN_HIP_MIN_ENGINE))}"
    elif gpu["arch"] not in meta.get("archs", []):
        why = f"it is built for {', '.join(meta.get('archs', []))}; your GPU is {gpu['arch']}"
    if why:
        warn(f"the ready-made AMD engine at {base} cannot be used: {why}")
        shutil.rmtree(tmp, ignore_errors=True)
        drop_archive(z)
        return None
    for p in tmp.iterdir():
        dst = eng / p.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        p.replace(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    drop_archive(z)
    hip_runtime_beside_exe(eng)                        # #468 #461
    ok(f"ready-made AMD engine {meta.get('version', '')} for {', '.join(meta.get('archs', []))} "
       f"(ROCm {meta.get('rocm', '?')})")
    return eng


def rocm_version(root):
    """(major, minor) of a ROCm install, from rocm-core's header; None when it has none."""
    try:
        text = (Path(root) / "include" / "rocm-core" / "rocm_version.h").read_text()
        return tuple(int(re.search(rf"#define\s+ROCM_VERSION_{k}\s+(\d+)", text).group(1)) for k in ("MAJOR", "MINOR"))
    except (OSError, AttributeError, ValueError):
        return None


def rocm_dev_missing(sysroot: Path) -> list:
    """#446: the HIP development files the engine's build needs that a system ROCm lacks (a runtime-only install has
    hipcc and libhipblas but not these, and cmake's enable_language(HIP) then fails on the hip-lang package)."""
    lang = "cmake/hip-lang/hip-lang-config.cmake"     # where CMake's HIP support looks for it
    need = {"lib/" + lang: [sysroot / d / lang for d in ("lib", "lib64", "lib/x86_64-unknown-linux-gnu")],
            "include/hip/hip_runtime.h": [sysroot / "include" / "hip" / "hip_runtime.h"]}
    return [name for name, paths in need.items() if not any(p.is_file() for p in paths)]


def rocm_root(archs):
    """ROCm for compiling and running the HIP engine for `archs` (one arch or a list: the cards of a layer split):
    (root, library folders).  A system ROCm 7 with hipcc, hipBLAS and the HIP development files (#446), else AMD's
    TheRock wheels (ROCM_VERSION, from the card family's index) installed into .venv."""
    archs = [archs] if isinstance(archs, str) else list(archs)
    sysroot = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    if (sysroot / "bin" / "hipcc").exists() and list((sysroot / "lib").glob("libhipblas.so*")):
        ver = rocm_version(sysroot)
        missing = rocm_dev_missing(sysroot)
        if (ver is None or ver >= ROCM_SYSTEM_MIN) and not missing:
            return sysroot, [str(sysroot / "lib")]
        if ver is not None and ver < ROCM_SYSTEM_MIN:
            warn(f"the ROCm in {sysroot} is {ver[0]}.{ver[1]}; Strata needs {ROCM_SYSTEM_MIN[0]}.{ROCM_SYSTEM_MIN[1]} "
                 "or newer: using AMD's wheels in .venv instead")
        else:                                          # #446: a runtime-only ROCm (no -dev packages): cmake would fail
            warn(f"the ROCm in {sysroot} has no HIP development files ({', '.join(missing)}): using AMD's wheels in "
                 ".venv instead (or install them, e.g. AMD's amdrocm-core-dev package for your ROCm and card)")
    indexes = list(dict.fromkeys(rocm_index(a) for a in archs))
    if len(indexes) > 1:                               # TheRock's wheels hold one GPU family's libraries
        fail(f"cards of two GPU families ({', '.join(archs)}) need a system ROCm 7 (in /opt/rocm): AMD's Python "
             "wheels come per family", "install ROCm 7 system-wide, or use cards of one family (--gpu N for one card)")
    index = indexes[0]
    stamp = Path(sys.prefix) / ".strata-rocm.json"
    have = json.loads(stamp.read_text()) if stamp.exists() else {}
    if have.get("version") != ROCM_VERSION or have.get("index") != index:
        say(f"  Installing ROCm {ROCM_VERSION} for AMD GPUs into .venv (AMD's TheRock wheels, ~10 GB, no sudo) ...")
        pip = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--index-url", index]
        if have.get("version") == ROCM_VERSION:        # the same version for another GPU family: its own libraries
            run(pip + ["--force-reinstall", "--no-deps", f"rocm=={ROCM_VERSION}"])
        run(pip + [f"rocm[libraries,devel]=={ROCM_VERSION}"])
        stamp.write_text(json.dumps({"version": ROCM_VERSION, "index": index}))
    sdk = Path(sys.executable).parent / "rocm-sdk"
    root = Path(out([str(sdk), "path", "--root"]).strip())
    if not (root / "llvm" / "bin" / "clang++").exists():
        fail(f"ROCm was installed but its compiler is missing ({root})",
             f"remove {stamp} and run this again; or install ROCm 7 system-wide")
    # the card family's libraries only (gfx120X-all -> _rocm_sdk_libraries_gfx120X_all): another family's
    # libhipblaslt.so first on the path would have no kernels for this card
    family = "_rocm_sdk_libraries_" + index.rstrip("/").rsplit("/", 1)[-1].replace("-", "_")
    dirs = [str(root / "lib")]
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        libs = sorted(sp.glob("_rocm_sdk_libraries_*"))
        libs = [d for d in libs if d.name.lower() == family.lower()] or libs
        dirs += [str(d / "lib") for d in libs if (d / "lib").is_dir()]
    ok(f"ROCm: {root}")
    return root, list(dict.fromkeys(dirs))


def hipblaslt_version(lib_dirs):
    """The installed hipBLASLt's version as the engine reads it (hipblasLtGetVersion: 1.4.1 -> 100401), from the
    header of the ROCm whose libraries the engine loads; None when not found."""
    for d in lib_dirs:
        try:
            text = (Path(d).parent / "include" / "hipblaslt" / "hipblaslt-version.h").read_text()
            v = [int(re.search(rf"#define\s+HIPBLASLT_VERSION_{k}\s+(\d+)", text).group(1))
                 for k in ("MAJOR", "MINOR", "PATCH")]
        except (OSError, AttributeError, ValueError):
            continue
        return v[0] * 100000 + v[1] * 100 + v[2]
    return None


def hipblaslt_table(arch, lib_dirs, ver=None):
    """tools/hip/<arch>-hipblaslt-<version>.txt for this card AND the installed hipBLASLt, else None: its solution
    ids are valid only for that pair (the engine refuses any other table and uses plain hipBLAS).  ver: the
    hipBLASLt version the ready-made Windows engine ships (its BUILD.json), else read from the installed headers."""
    ver = ver or hipblaslt_version(lib_dirs)
    table = ROOT / "tools" / "hip" / f"{arch}-hipblaslt-{ver}.txt"
    if ver is not None and table.exists():
        head = table.read_text().split("\n", 2)[:2]
        if f"STRATA_HIPBLASLT_TUNING_V1 {arch} {ver}" in (h.strip() for h in head):
            ok(f"hipBLASLt tuning table: {table.name} (faster prompts)")
            return table
    have = sorted(p.name for p in (ROOT / "tools" / "hip").glob(f"{arch}-hipblaslt-*.txt"))
    warn(f"no hipBLASLt tuning table for {arch} with hipBLASLt {ver or '(version unknown)'}"
       + (f" (have: {', '.join(have)})" if have else "")
       + ": the prompt's dense matrix products use plain hipBLAS (tools/hip/tune_hipblaslt makes a table: "
         "docs/AMD_HIP.md, Tuning table)")
    return None


def build_engine_hip(gpu, llama, vision="none") -> Path:
    """Compile the HIP engine for this AMD GPU into engine/ (again only when its source changed: a `git pull`).
    gpu["archs"]: every architecture it needs code for (the cards of a layer split), else gpu["arch"].  vision "cpu"
    (#304): the image encoder too, for the CPU (there is no HIP encoder build yet)."""
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    src, vsrc = source_hash(ENGINE_SOURCES), source_hash(VISION_SOURCES)
    archs = sorted(set(gpu.get("archs") or [gpu["arch"]]))
    has_archs = set(archs) <= set(meta.get("archs", []))
    floor = cpu_floor(cpu_info()[1])                     # "" on an AVX2 CPU: the normal engine
    engine_ok = meta.get("backend") == "hip" and (eng / EXE).exists() and meta.get("src") == src and has_archs and \
        (meta.get("isa_floor") or "") == floor
    vision_ok = vision == "none" or ((eng / VEXE).exists() and meta.get("vision_src") == vsrc)
    if engine_ok and vision_ok:
        ok("engine already built for this PC")
        return eng
    if engine_ok:
        return build_vision_cpu(eng, stamp, meta, llama, vsrc)
    if not (shutil.which("c++") or shutil.which("g++")) or not shutil.which("git"):
        fail("a C++ compiler and git are needed to compile the AMD engine",
             "Ubuntu/Debian: sudo apt install build-essential git   Fedora: sudo dnf install gcc-c++ git")
    root, dirs = rocm_root(archs)
    libs = [str(Path(d).parent) for d in dirs[1:]]
    bitcode = next((p for p in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode") if p.is_dir()),
                   root / "amdgcn" / "bitcode")
    os.environ.update({"HIP_PLATFORM": "amd", "HIP_COMPILER": "clang", "HIP_RUNTIME": "rocclr", "ROCM_PATH": str(root),
                       "HIP_PATH": str(root)})
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm" / "bin"), os.environ.get("PATH", "")])
    say("  The engine's source changed: compiling it again (only what changed, a few minutes) ..."
        if meta.get("backend") == "hip" and (eng / EXE).exists() and has_archs
        else f"  Compiling the Strata engine for your AMD GPU{'s' if len(archs) > 1 else ''} ({', '.join(archs)}; "
             "10-20 minutes, once) ...")
    cmake_build(ROOT, ROOT / "build-hip", "strata",
                ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_BUILD_TESTS=OFF",
                 "-DSTRATA_PREFILL_MMQ=ON", "-DCMAKE_HIP_ARCHITECTURES=" + ";".join(archs),
                 f"-DCMAKE_HIP_COMPILER={root / 'llvm' / 'bin' / 'clang++'}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
                 "-DCMAKE_PREFIX_PATH=" + ";".join([str(root), *libs]),
                 f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}",
                 f"-DSTRATA_GGML_DIR={llama}", *isa_floor_defs(floor, ROOT / "build-hip", meta)], None, "")
    shutil.copy2(ROOT / "build-hip" / EXE, eng / EXE)
    meta = {"source": "local-hip", "backend": "hip", "version": source_version(), "archs": archs, "vision": "none",
            "lib_dirs": dirs, "src": src, **({"isa_floor": floor} if floor else {})}
    if vision != "none":
        return build_vision_cpu(eng, stamp, meta, llama, vsrc)
    stamp.write_text(json.dumps(meta, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


def hip_vision(asked) -> str:
    """The image encoder with the AMD backend (--vision): the CPU one when asked for (#304); a HIP (GPU) encoder build
    is a later step, so `yes`/`gpu` leave images off, as before, and say how to get them."""
    if asked in ("yes", "gpu"):
        warn("the AMD backend has no GPU image encoder yet: images off"
             + ("" if WIN else " (--vision cpu reads them on the CPU)"))
    if asked == "cpu" and WIN:
        warn("images on the CPU with an AMD card are Linux-only for now (the ready-made Windows AMD engine has no "
             "image encoder): images off")
        return "none"
    return "cpu" if asked == "cpu" else "none"


def build_vision_cpu(eng: Path, stamp: Path, meta: dict, llama, vsrc) -> Path:
    """#304: the CPU image encoder beside the HIP engine (tools/vision without CUDA), recorded in its BUILD.json."""
    if not ((eng / VEXE).exists() and meta.get("vision_src") == vsrc):
        say("  Compiling the image encoder (for the CPU) ...")
        cmake_build(ROOT / "tools" / "vision", ROOT / "build-vision", "strata-vision",
                    [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=OFF"], None, "")
        shutil.copy2(ROOT / "build-vision" / "bin" / VEXE, eng / VEXE)
    stamp.write_text(json.dumps({**meta, "vision": "cpu", "vision_src": vsrc}, indent=1))
    ok(f"engine: {eng / EXE}, image encoder (CPU): {eng / VEXE}")
    return eng


# ------------------------------------------------------------------------------------------------ the engine
def driver_major(gpu):
    try:
        return int(gpu["driver"].split(".")[0])
    except (ValueError, KeyError):
        return 0


def prebuilt_bases(url_base) -> list[str]:
    """Where to look for the ready-made engine, in order (each ending in a slash).  The default: the release of this
    checkout's version first, then the latest (#214); an explicit --prebuilt / STRATA_PREBUILT_URL: only that."""
    base = url_base if url_base.endswith(("/", "\\")) else url_base + "/"
    if base != PREBUILT_URL:
        return [base]
    return [PREBUILT_TAG_URL.format(version=source_version()), base]


def get_prebuilt(url_base, gpu, vision, updating=False, toolkit=13) -> Path | None:
    """The ready-made engine in engine/ (kept between runs), or None when there is none for this PC.
    updating: called to replace an installed engine, which starts instead when this fails (no compile).
    toolkit 12: the experimental CUDA 12 engine (CUDA12_ASSET) in engine-cuda12/."""
    eng = engine_dir(toolkit)
    asset = CUDA12_ASSET if int(toolkit) == 12 else PREBUILT_ASSET
    info = eng / "BUILD.json"
    if info.exists() and (eng / EXE).exists() and json.loads(info.read_text()).get("backend") != "hip":
        meta = json.loads(info.read_text())
        ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
        if meta.get("source") == "local":              # compiled here: build_engine checks its source and cards
            return None
        have = [int(a) for a in meta.get("archs", [])]
        miss = [int(x) for x in gpu.get("archs", [gpu["arch"]])
                if have and int(x) not in have and not (meta.get("ptx") and int(x) > max(have))]
        if miss:                                       # a card it has no code for (#128): compiled here instead
            warn(f"the installed engine is built for {', '.join(str(a) for a in have)}; your GPU is "
                 f"{', '.join(str(x) for x in miss)}: compiling instead")
            return None
        if ver >= MIN_ENGINE:
            ok("ready-made engine already installed")
            return eng
        say(f"  Updating the ready-made engine ({meta.get('version')} -> {'.'.join(map(str, MIN_ENGINE))} or newer) ...")
        info.unlink()
    if not url_base:
        return None
    eng.mkdir(exist_ok=True)
    z = eng / asset
    bases = prebuilt_bases(url_base)
    for i, base in enumerate(bases):
        if not base.startswith(("http://", "https://")):
            break
        try:                                           # not published (yet), or no internet: compile instead
            req = urllib.request.Request(base + asset, method="HEAD", headers={"User-Agent": "strata-setup"})
            urllib.request.urlopen(req, timeout=60).close()
            break
        except OSError as e:
            if i + 1 < len(bases):                     # #214: this checkout's release is not published (yet)
                say(f"  No ready-made engine for v{source_version()} ({e}): the latest release instead")
                continue
            warn(f"no ready-made engine at {base} ({e})" + ("" if updating else ": compiling instead"))
            return None
    say("  Downloading the ready-made Strata engine" + (" (CUDA 12, experimental)" if int(toolkit) == 12 else "") + " ...")
    download(base + asset, z, "Strata engine")
    tmp = eng / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    meta = json.loads((tmp / "BUILD.json").read_text())
    if tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit()) < MIN_ENGINE:
        need = ".".join(map(str, MIN_ENGINE))
        if updating:                                   # these files are newer than the published release (#58)
            warn(f"engine {need} is not published yet (the release may still be uploading): run this again "
                 f"in a few minutes to update it")
        else:
            warn(f"the ready-made engine at {base} is version {meta.get('version')}; this setup needs "
                 f"{need}: compiling instead")
        shutil.rmtree(tmp, ignore_errors=True)
        drop_archive(z)
        return None
    archs = [int(a) for a in meta.get("archs", [])]
    miss = [int(x) for x in gpu.get("archs", [gpu["arch"]])
            if int(x) not in archs and not (meta.get("ptx") and int(x) > max(archs))]
    if miss:
        warn(f"the ready-made engine is built for {', '.join(str(a) for a in archs)}; your GPU is "
             f"{', '.join(str(x) for x in miss)}" + ("" if updating else ": compiling instead"))
        shutil.rmtree(tmp, ignore_errors=True)
        drop_archive(z)
        return None
    for p in tmp.iterdir():
        dst = eng / p.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        p.replace(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    drop_archive(z)
    if not (eng / EXE).exists():
        fail("the ready-made engine archive has no " + EXE)
    if not WIN:
        for x in (EXE, VEXE):
            if (eng / x).exists():
                (eng / x).chmod(0o755)
    ok(f"ready-made engine {meta.get('version', '')} for {', '.join('sm_' + str(a) for a in archs)} (CUDA "
       f"{meta.get('cuda', '?')})")
    return eng


def update_installed_engine(url_base, toolkit=None) -> None:
    """An installed ready-made engine older than MIN_ENGINE is replaced before the model starts, so a plain
    START-HERE.bat on an existing install picks up a new release.  If that cannot happen (no internet, the model
    still running, no ready-made engine for this GPU) the installed engine is kept and starts as before.
    toolkit None: engine/, then the experimental CUDA 12 engine in engine-cuda12/ when one is installed."""
    if toolkit is None:
        update_installed_engine(url_base, 13)
        if (engine_dir(12) / "BUILD.json").exists():
            update_installed_engine(url_base, 12)
        return
    eng = engine_dir(toolkit)
    info = eng / "BUILD.json"
    if not info.exists() or not (eng / EXE).exists():
        return
    meta_text = info.read_text()
    meta = json.loads(meta_text)
    if meta.get("backend") == "hip" and WIN:           # AMD on Windows: the ready-made HIP engine, when older
        ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
        if meta.get("source") == "prebuilt" and ver < WIN_HIP_MIN_ENGINE:
            try:
                g = next((x for x in amd_gpus() if amd_problem(x) is None), None)
                if g is None:
                    raise RuntimeError("no supported AMD GPU found")
                say(f"  Updating the ready-made AMD engine ({meta.get('version')} -> "
                    f"{'.'.join(map(str, WIN_HIP_MIN_ENGINE))} or newer) ...")
                if get_prebuilt_hip(url_base, g, updating=True) is None:
                    raise RuntimeError("not published yet")
            except (Exception, SystemExit) as e:
                warn(f"could not update the AMD engine{'' if isinstance(e, SystemExit) else f' ({e})'}: "
                     "starting the installed one")
        return
    if meta.get("backend") == "hip":                   # AMD: compiled here, again when its source changed
        if meta.get("src") != source_hash(ENGINE_SOURCES):
            try:
                usable = [x for x in amd_gpus() if amd_problem(x) is None]
                g = next((x for x in usable if x["arch"] in meta.get("archs", [])), usable[0] if usable else None)
                if g is None:
                    raise RuntimeError("no supported AMD GPU found")
                # every architecture it was built for (a layer split across two families keeps both)
                build_engine_hip({**g, "archs": [x for x in meta.get("archs", []) if x in AMD_ARCHS] or [g["arch"]]},
                                 get_llama_cpp(), meta.get("vision") or "none")
            except (Exception, SystemExit) as e:
                warn(f"could not compile the updated engine{'' if isinstance(e, SystemExit) else f' ({e})'}: "
                     "starting the installed one")
        return
    ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
    local = meta.get("source") == "local"
    vision = meta.get("vision") or "none"
    if local:                                          # compiled here: is it older than the source (a git pull)?
        if meta.get("src") == source_hash(ENGINE_SOURCES) and \
                (vision == "none" or meta.get("vision_src") == source_hash(VISION_SOURCES)):
            return
    elif ver >= MIN_ENGINE:
        return
    try:                                               # a running engine cannot be replaced (Windows keeps it locked)
        for x in (EXE, VEXE):
            if (eng / x).exists():
                with open(eng / x, "r+b"):
                    pass
    except OSError:
        warn(f"engine {meta.get('version') or ''} is in use: close the model window and run this again to update it")
        return
    gpu = gpu_info()
    if local:
        try:                                           # a failed compile must not stop the model from starting
            if gpu is None:
                raise RuntimeError("no NVIDIA GPU found")
            gpu = {**gpu, "archs": sorted({int(gpu["arch"]), *(int(x) for x in meta.get("archs", []))})}
            if int(toolkit) == 12:                     # the cards it was compiled for (the main GPU may be newer)
                gpu["archs"] = sorted({int(x) for x in meta.get("archs", [])}) or gpu["archs"]
            build_engine(gpu, vision, False, get_llama_cpp(), toolkit=toolkit)
        except (Exception, SystemExit) as e:
            warn(f"could not compile the updated engine{'' if isinstance(e, SystemExit) else f' ({e})'}: starting the installed one")
        return
    new = None
    if gpu is not None:
        try:
            if int(toolkit) == 12:                     # the cards the CUDA 12 engine serves, not the newest one
                gpu = {**gpu, "archs": [x for x in (int(a) for a in meta.get("archs", [])) if x < CUDA13_MIN_ARCH]
                       or [int(gpu["arch"])]}
            new = get_prebuilt(url_base, gpu, "gpu", updating=True, toolkit=toolkit)
        except Exception as e:                         # a failed download must not stop the model from starting
            warn(f"updating the engine failed ({e})")
    if new is None:
        if not info.exists():
            info.write_text(meta_text)                 # get_prebuilt drops it before downloading: put it back
        warn(f"could not update the engine: starting the installed {meta.get('version')}")
        return
    pip_cuda_libs(toolkit)


def pip_cuda_libs(toolkit=13) -> None:
    """NVIDIA's cuBLAS and CUDA runtime for a ready-made engine, from pip: CUDA 13's, or the CUDA 12 engine's."""
    if int(toolkit) == 12:
        pip_install(CUDA12_WHEELS, "NVIDIA CUDA 12 libraries for the experimental engine (cuBLAS, CUDA runtime; ~0.7 GB)")
    else:
        pip_install(CUDA_WHEELS, "NVIDIA CUDA libraries (cuBLAS, CUDA runtime; ~0.4 GB)")


def install_build_tools(gpu, yes):
    """The compiler and the CUDA toolkit, installed for the user (asks once).  Returns (nvcc, vcvars)."""
    archs = [int(x) for x in gpu.get("archs", [gpu["arch"]])]
    # #295: Pascal / Volta need a CUDA 12.x toolkit - CUDA 13 cannot build sm_60/sm_70; gpu["toolkit"] = 12: the
    # experimental CUDA 12 engine for any cards (--cuda 12, docs/OLDER_GPUS.md)
    old = int(gpu.get("toolkit") or (12 if min(archs) < CUDA13_MIN_ARCH else 13)) == 12
    need12 = (12, 8) if max(archs) >= 120 else (12, 0)     # sm_120 needs CUDA 12.8 or newer
    if old and max(archs) >= 120:
        warn("an RTX 50 card (sm_120) in a CUDA 12 engine: engines built with CUDA 12.8 crashed on long prompts there "
             "(#220, #224); the RTX 50 card alone (--gpu N) runs the ready-made CUDA 13 engine")
    nvcc, cuda_v = find_nvcc(below=(13, 0)) if old else find_nvcc()
    if old and (nvcc is None or cuda_v < need12):
        fail("the experimental CUDA 12 engine (Pascal / Volta, or --cuda 12) is compiled here with the NVIDIA CUDA "
             f"Toolkit {need12[0]}.{need12[1]} or a newer 12.x (CUDA 13 cannot compile for these cards)" +
             (f"; found CUDA {cuda_v[0]}.{cuda_v[1]}" if nvcc else ""),
             "install CUDA 12.9 (it can sit next to a newer one) from https://developer.nvidia.com/cuda-toolkit-archive "
             "and run it again (STRATA_NVCC=<its nvcc> picks one toolkit)")
    # RTX 50 (sm_120): CUDA 13.0 - an engine built with 12.8 crashed in the prompt path on Linux (#220)
    need_cuda = need12 if old else (13, 0) if max(archs) >= 120 else (12, 0)
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
            fail("missing: " + " and ".join(missing) + " (the automatic install is only done on Ubuntu/Debian)",
                 "install them with your distribution's packages (Arch: pacman -S base-devel cuda; nvcc is found on "
                 "PATH, in /usr/local/cuda* and in /opt/cuda*), then run it again")
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
    nvcc, cuda_v = find_nvcc(below=(13, 0)) if old else find_nvcc()
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
    # A failed build is tried once more: CUDA 13.0's ptxas now and then fails to parse a PTX file it just wrote, and
    # the same command then gets past it (issue #45); a second attempt only compiles what is still missing.
    if WIN:
        bat = ROOT / bat_name
        q = lambda c: " ".join(f'"{x}"' if " " in str(x) else str(x) for x in c)  # noqa: E731
        bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{q(conf)} || exit /b 1\r\n{q(build)} && exit /b 0\r\n'
                       f'echo   (the build stopped - trying it once more)\r\n{q(build)} || exit /b 1\r\n',
                       encoding="utf-8")
        run(["cmd", "/c", str(bat)])
    else:
        run(conf)
        if run(build, check=False).returncode != 0:
            say("  (the build stopped - trying it once more)")
            run(build)


ENGINE_SOURCES = ("CMakeLists.txt", "src", "include", "third_party/ggml")
VISION_SOURCES = ("tools/vision",)


def source_hash(parts) -> str:
    """A fingerprint of the files a compiled engine is built from, kept in engine/BUILD.json: when a `git pull`
    changes them, the engine is compiled again (issue #31)."""
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def isa_floor_defs(floor: str, bdir: Path, meta: dict) -> list:
    """The experimental older-CPU build's CMake definition (STRATA_ISA_FLOOR, CMakeLists.txt), none for the normal
    build.  A build folder configured for another floor is configured afresh: ggml's CPU options are cached there."""
    if (meta.get("isa_floor") or "") != floor and (bdir / "CMakeCache.txt").exists():
        (bdir / "CMakeCache.txt").unlink()
    return [f"-DSTRATA_ISA_FLOOR={floor}"] if floor else []


def engine_defs(archs, toolkit=13) -> list:
    """Extra CMake definitions for the engine: the experimental Pascal/Volta build (#295) for cards below sm_75, and
    for every CUDA 12 engine (the same build as the ready-made CUDA 12 one: it admits the older cards)."""
    return ["-DSTRATA_EXPERIMENTAL_SM60=ON"] if min(int(x) for x in archs) < 75 or int(toolkit) == 12 else []


def prebuilt_vision(meta: dict, gpu: dict, vision: str) -> str:
    """The image encoder to use with a ready-made engine (`meta`: its BUILD.json).  The encoder can cover fewer cards
    than the engine (0.1.30/0.1.31: no RTX 20 code, #331): such a card gets the CPU encoder - the same program - instead
    of compiling one, which fails on most Windows PCs (no Visual Studio / CUDA toolkit); --build compiles it."""
    if vision != "gpu":
        return vision
    va = [int(x) for x in meta.get("vision_archs", meta.get("archs", []))]
    if va and int(gpu["arch"]) not in va and not (meta.get("ptx") and int(gpu["arch"]) > max(va)):
        warn(f"the ready-made image encoder has no code for your GPU (sm_{gpu['arch']}): it runs on the CPU instead "
             "(images take longer; setup --build compiles one for your GPU)")
        return "cpu"
    return vision


def build_engine(gpu, vision, yes, llama, toolkit=None) -> Path:
    """Compile the engine (and, for images, the encoder) for this GPU; the results go to engine/.  A compiled
    engine whose source files changed since (a `git pull`) is compiled again: only the changed files, a few minutes.
    toolkit 12 (default: 12 for a card older than CUDA 13 supports): the experimental CUDA 12 engine, in
    engine-cuda12/ with its own build folders."""
    if toolkit is None:
        toolkit = 12 if min(int(x) for x in gpu.get("archs", [gpu["arch"]])) < CUDA13_MIN_ARCH else 13
    t12 = int(toolkit) == 12
    eng = engine_dir(toolkit)
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    want_vision = vision != "none"
    local = meta.get("source") == "local"
    src, vsrc = source_hash(ENGINE_SOURCES), source_hash(VISION_SOURCES)
    archs = sorted({int(x) for x in gpu.get("archs", [gpu["arch"]])})    # every card the model runs on
    built = {int(x) for x in meta.get("archs", [])}
    # a card the engine has no code for (a GPU added with --gpus, #128) needs a compile even when the source is the
    # same; the compile keeps the generations it was built for
    new_arch = local and not set(archs) <= built
    floor = cpu_floor(cpu_info()[1])                     # "" on an AVX2 CPU: the normal engine
    engine_ok = local and (eng / EXE).exists() and meta.get("src") == src and not new_arch and \
        (meta.get("isa_floor") or "") == floor
    vision_ok = not want_vision or ((eng / VEXE).exists() and (not local or meta.get("vision_src") == vsrc))
    if engine_ok and vision_ok:
        ok("engine already built for this PC")
        return eng
    if local:
        archs = sorted(built | set(archs))
    nvcc, vcvars = install_build_tools({**gpu, "archs": archs, "toolkit": toolkit}, yes)
    cuda_archs = ";".join(str(x) for x in archs)
    bdir, vdir = (ROOT / "build-cuda12", ROOT / "build-vision-cuda12") if t12 else (ROOT / "build", ROOT / "build-vision")
    if not engine_ok:
        say("  Compiling the engine for " + ", ".join(f"sm_{x}" for x in archs) + " (a card it had no code for; "
            "10-20 minutes, once) ..." if new_arch else
            "  The engine's source changed: compiling it again (only what changed, a few minutes) ..."
            if local and (eng / EXE).exists() else "  Compiling the Strata engine for your GPU (10-20 minutes, once) ...")
        cmake_build(ROOT, bdir, "strata",
                    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
                     f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama}", *engine_defs(archs, toolkit),
                     *isa_floor_defs(floor, bdir, meta)],
                    vcvars, "build-strata-cuda12.bat" if t12 else "build-strata.bat")
        shutil.copy2(bdir / EXE, eng / EXE)
    if not vision_ok:
        say("  Compiling the image encoder" + (" with CUDA (10-20 minutes, once) ..." if vision == "gpu" else " ..."))
        defs = [f"-DLLAMA_DIR={llama}", f"-DSTRATA_VISION_CUDA={'ON' if vision == 'gpu' else 'OFF'}"]
        if vision == "gpu":
            defs += [f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}", f"-DCMAKE_CUDA_COMPILER={nvcc}"]
        cmake_build(ROOT / "tools" / "vision", vdir, "strata-vision", defs, vcvars,
                    "build-vision-cuda12.bat" if t12 else "build-vision.bat")
        shutil.copy2(vdir / "bin" / VEXE, eng / VEXE)
    bindir = Path(nvcc).parent                            # the toolkit's own libraries (bin, bin/x64, lib64)
    dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
    stamp.write_text(json.dumps({"source": "local", "version": source_version(), "archs": archs,
                                 "vision": vision, **({"toolkit": 12} if t12 else {}),
                                 "cuda_dirs": dirs, "src": src, "vision_src": vsrc if want_vision else None,
                                 **({"isa_floor": floor} if floor else {})}, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


# ------------------------------------------------------------------------------------------------ the data folder
# The model files - the GGUFs, the prepared packs and the MTP layer, 70-120 GB - live in a data folder NEXT TO the
# Strata folder (`Strata-data`), not inside it: updating Strata by unzipping a new copy used to give a new, empty
# folder and a full download again.  Where it is, and which Strata folders this user ran, is kept in a small
# per-user file, so every Strata folder on the PC finds the same files.
DATA_ITEMS = ("models", "packs", "mtp")


LOW_RAM_HEADROOM_GB = 10   # RAM beside the experts: the OS, the engine's other buffers, the server
RESIDENT_ENGINE = (0, 1, 30)   # the first engine with --resident-experts (the low-RAM mode's resident variant)


def low_ram_needed(model, ram) -> bool:
    """The model's experts do not fit this PC's RAM with room left for the rest: they are then mapped from the pack's
    experts.bin instead of copied into RAM (the low-RAM mode)."""
    return ram < MODELS[model]["arena_gb"] + LOW_RAM_HEADROOM_GB


def low_ram_gpu_gb(model, vram_gb, ctx=32768, kv="int8") -> float:
    """About how many GB of the model's experts the GPU's cache holds: its VRAM minus ~5 GB for the dense weights,
    buffers and a 32K context's KV cache, minus the KV cache of a longer context (in VRAM in the low-RAM mode: its RAM
    has no room for KV streaming)."""
    kv_tok = 13 * (576 if kv == "q4_0" else 1056)       # bytes per context token: 12 QSA layers + the draft layer
    longer = max(0, ctx - 32768) * kv_tok / 1e9
    return max(0.0, min(MODELS[model]["arena_gb"], vram_gb - 5 - longer))


def low_ram_gpu_share(model, vram_gb, ctx=32768, kv="int8") -> float:
    """About how much of the model's experts the GPU holds."""
    return low_ram_gpu_gb(model, vram_gb, ctx, kv) / MODELS[model]["arena_gb"]


def low_ram_resident(model, ram, vram_gb, ctx=32768, kv="int8") -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM with the usual room beside them, so they are
    copied into RAM once (the resident variant, `--resident-experts`) instead of being read through the OS file cache
    (plain `--mmap-experts`, which a PC this short of RAM keeps re-reading from the SSD)."""
    rest = MODELS[model]["arena_gb"] - low_ram_gpu_gb(model, vram_gb, ctx, kv)
    return ram >= rest + LOW_RAM_HEADROOM_GB


def low_ram_fits(model, ram, vram_gb) -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM left beside the rest (as file cache)."""
    arena = MODELS[model]["arena_gb"]
    return ram - 6 + max(0.0, vram_gb - 5) >= arena


def low_ram_one_gpu_why(model, ram, choice, sel=None) -> list[str]:
    """#250: why the low-RAM mode recommends one GPU, with the RAM math that turned it on; #364 #384: and how to use
    all of them (sel: the cards, for the --gpus example)."""
    arena, need = MODELS[model]["arena_gb"], MODELS[model]["arena_gb"] + LOW_RAM_HEADROOM_GB
    if choice == "auto":
        why = [f"Why: {model}'s experts are {arena:.0f} GB and must fit in RAM with ~{LOW_RAM_HEADROOM_GB} GB beside "
               f"them for the OS and the rest: {arena:.0f} + {LOW_RAM_HEADROOM_GB} = {need:.0f} GB, and this PC has "
               f"{ram:.0f} GB.",
               "So setup uses the low-RAM mode: the experts come from the model's file (copied into RAM as far as "
               "it fits), the GPU holds the most-used ones."]
    else:
        why = [f"Why: you chose the low-RAM mode (--low-ram {choice}); without it {model} needs {arena:.0f} + "
               f"{LOW_RAM_HEADROOM_GB} = {need:.0f} GB of RAM, this PC has {ram:.0f} GB."]
    return why + ["Its resident variant (the experts the GPU does not hold copied into RAM once: steady RAM use) runs "
                  "on one GPU: the engine has no layer split for it yet.",
                  f"To use all the GPUs: --gpus {','.join(str(i) for i in sel) if sel else '0,1'} - the experts the "
                  "GPUs do not hold are then read through the OS file cache: faster in two reports (1.3-1.6x, #364 "
                  f"#384), but RAM can fill up to 0 free during long prompts. Or {need:.0f} GB of RAM or more, or a "
                  "smaller size."]


def low_ram_together(a, model, ram, gpu, chosen) -> bool:
    """#364 #384: the low-RAM mode with several GPUs chosen.  One GPU is recommended: the resident variant keeps the
    experts the GPU does not hold in RAM (steady RAM use) and has no layer split.  All the GPUs together read those
    experts through the OS file cache instead (--mmap-experts) - 1.3-1.6x faster in those reports, but RAM can fill
    up to 0 free during long prompts.  An explicit --gpus (or an earlier install's cards) is kept; otherwise asked,
    one GPU by default (--yes: one GPU, as before).  True: all of them."""
    sel = [g["index"] for g in chosen]
    names = " + ".join(gpu_name(g) for g in chosen)
    if a.gpus:
        warn(f"the low-RAM mode on {names}, as you chose (--gpus): the experts the GPUs do not hold are read through "
             "the OS file cache (the resident variant has no layer split yet), and RAM can fill up to 0 free during "
             "long prompts")
        say(f"       One GPU keeps them in RAM (steady RAM use, recommended): --gpu {gpu['index']}")
        if a.low_ram == "resident":
            warn("--low-ram resident has no layer split yet: the experts are read through the OS file cache instead")
        return True
    if a.low_ram != "resident" and not a.yes:
        say()
        for line in low_ram_one_gpu_why(model, ram, a.low_ram, sel):
            say("  " + line)
        say(f"  1) {gpu_name(gpu)} only: the experts it does not hold kept in RAM where they fit   (recommended: "
            "steady RAM use)")
        say(f"  2) {names} together: the experts the GPUs do not hold read through the OS file cache - faster")
        say("     in two reports (1.3-1.6x, #364 #384), but RAM can fill up to 0 free during long prompts")
        if ask("Low-RAM mode: which GPUs?", ["1", "2"], "1", a.yes) == "2":
            ok(f"the low-RAM mode on {names}: the experts read through the OS file cache, as you chose")
            return True
        warn("the low-RAM mode: using " + gpu_name(gpu) + " only")
        return False
    warn("the low-RAM mode: using " + gpu_name(gpu) + " only (recommended)")
    for line in low_ram_one_gpu_why(model, ram, a.low_ram, sel):
        say("       " + line)
    return False


def unsloth_together(a, model, ram, gpu, chosen) -> bool:
    """#498: UD-Q4_K_XL with several GPUs chosen.  Its RAM budget (--resident-budget-gib) has no layer split, so a
    split runs without it: all its experts loaded into RAM from the GGUFs at start, as with the 2-3-bit models - only
    where the RAM holds the GGUF files and 24 GB more (unsloth_split_need_gb; 165 GiB, 2x RTX 3090: 31 -> 64-78
    tok/s).  An explicit --gpus is honoured there; otherwise asked, one GPU by default (--yes: one GPU, as before); an
    explicit --resident-budget-gib keeps one GPU.  True: all of them."""
    names = " + ".join(gpu_name(g) for g in chosen)
    need = unsloth_split_need_gb(model)
    if ram < need:
        warn(f"{model} runs on one GPU here: on several it has no RAM budget and needs ~{need:.0f} GB of RAM (its GGUF "
             f"files and {UNSLOTH_RAM_LEFT_GB} GB more), this PC has {ram:.0f} - using {gpu_name(gpu)} only")
        return False
    if a.resident_budget_gib is not None:
        warn(f"--resident-budget-gib has no layer split: {model} runs on one GPU with it - using {gpu_name(gpu)} only "
             f"(leave the budget out to use {names} together)")
        return False
    note = (f"no RAM budget - all of its experts (~{MODELS[model]['arena_gb']:.0f} GB) are loaded into RAM from the "
            f"model files at start, and the files pass through the OS file cache (needs ~{need:.0f} GB of RAM, this "
            f"PC has {ram:.0f})")
    if a.gpus:
        ok(f"{model} on {names}, as you chose (--gpus): {note}")
        return True
    if not a.yes:
        say()
        say(f"  {model} can run on one GPU with a RAM budget of its experts, or on {names} together without one:")
        say(f"  1) {gpu_name(gpu)} only: the most-used experts kept in RAM, the rest read from the SSD   (recommended: "
            "the tested setup)")
        say(f"  2) {names} together: {note};")
        say("     about twice as fast in #498 (2x RTX 3090: 31 -> 64-78 tokens/s)")
        if ask(f"{model}: which GPUs?", ["1", "2"], "1", a.yes) == "2":
            ok(f"{model} on {names}: {note}")
            return True
    warn(f"{model} runs on one GPU: using {gpu_name(gpu)} only (--gpus " + ",".join(str(g["index"]) for g in chosen) +
         f" shares it across {names} without the RAM budget: this PC's RAM holds it)")
    return False


def confirm_risk(msg, explicit, yes, stop, hint=None, question="  Go on anyway?", default="n") -> None:
    """The owner's rule: setup recommends, it never forces.  A choice setup expects to fail or run badly is said
    plainly (msg), then asked (default n; `default` keeps an older question's own default), or with --yes taken as
    consent when it was asked for explicitly (a flag such as --model or --gpus): --yes alone keeps the stop
    (stop, hint).  Returns when it goes on; the caller says what it does."""
    warn(msg)
    if yes and explicit:
        return
    if ask(question, ["y", "n"], default, yes) != "y":
        fail(stop, hint)


def confirm_paging(model, ram, choice, yes, explicit_model=False):
    """The model's experts do not fit this PC's RAM and the low-RAM mode is off.  #125: a warning and a question, not
    a stop - the user may accept paging.  Asked "no" by default, so an unattended --yes install stops here, unless
    the low-RAM mode was turned off explicitly (--low-ram off, #250) or the size was (--model): that is the choice
    already made."""
    need_gb, arena = MODELS[model]["ram_gb"], MODELS[model]["arena_gb"]
    off = choice == "off"
    confirm_risk(f"{model} needs about {need_gb} GB of RAM and this PC has {ram:.0f} GB: its experts alone are "
                 f"{arena:.0f} GB and must stay in RAM, so Windows/Linux will page part of them from disk. Expect it "
                 "to be much slower, and it may not start at all.\n       A smaller size (Q2_0 or IQ2_XS) fits; more "
                 "RAM fixes it.", off or explicit_model, yes,
                 f"{model} needs about {need_gb} GB of RAM; this PC has {ram:.0f} GB",
                 "choose Q2_0 or IQ2_XS, or add RAM" + ("" if off else f"; or --model {model} --yes (or --low-ram off "
                                                                        "--yes) to install it anyway"),
                 "  Install it anyway?", "y" if off else "n")
    warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose" + (" (--low-ram off)" if off else
                                                                          " (--model)" if explicit_model else ""))


def ctx_ram_need(model, ctx, low_ram=False):
    """#406: the RAM (GB) setup estimates for a long context with IQ3_XXS / IQ3_S: their experts + the context's
    8-bit KV cache + 24 GB of room for everything else (the 0.1.29 arithmetic, counted).  None where the context does
    not count against RAM by this rule: the other sizes, and the low-RAM mode (its KV cache stays in VRAM)."""
    if model not in ("IQ3_XXS", "IQ3_S") or low_ram:
        return None
    return MODELS[model]["arena_gb"] + ctx * 13 * 1056 / 1e9 + 24


def ram_ctx(model, ram, low_ram=False) -> int:
    """#406: the longest context the RAM rule recommends: 128K, or longer where the estimate fits this PC's RAM.  It
    is part of the recommended default (the smaller of it and the GPU's rule); a longer choice is kept, with a note."""
    return max(c for c in CONTEXTS if c <= 131072 or (ctx_ram_need(model, c, low_ram) or 0) <= ram)


def settings_path() -> Path:
    if WIN:
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Strata" / "settings.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "strata" / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(s, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {settings_path()} ({e})")


def has_data(folder: Path) -> bool:
    for d in DATA_ITEMS:
        try:
            if (folder / d).is_dir() and any((folder / d).iterdir()):
                return True
        except OSError:
            pass
    return False


def other_installs(settings: dict) -> list:
    """Strata folders besides this one that may hold model files: the ones this user ran before, and Strata* folders
    next to this one (a zip unpacked again lands in e.g. `Strata-main (1)\\Strata-main`)."""
    cands = [Path(p) for p in settings.get("installs", [])]
    for base in dict.fromkeys((ROOT.parent, ROOT.parent.parent)):
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.lower().startswith("strata"):
                    cands.append(d)
                    cands += [c for c in d.iterdir() if c.is_dir() and c.name.lower().startswith("strata")]
        except OSError:
            pass
    found = []
    for d in cands:
        try:
            d = d.resolve()
            if d != ROOT and d not in found and (d / "setup.py").is_file():
                found.append(d)
        except OSError:
            pass
    return found


def same_drive(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def move_into(src: Path, dst: Path) -> None:
    """A rename into the data folder (same drive: instant); a folder merges into one already there, keeping what the
    destination has.  Whatever cannot be moved (a file in use) stays where it is."""
    if not dst.exists():
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            return
        except OSError:
            if not src.is_dir():
                return
            dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and dst.is_dir():
        for c in list(src.iterdir()):
            move_into(c, dst / c.name)
        try:
            src.rmdir()
        except OSError:
            pass


def repoint_config(cfg_file: Path, old: Path, new: Path) -> None:
    """A config whose model files moved from `old` to `new` points at them there (each path only if its file is
    now there and no longer at the old place)."""
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, str):
            for d in DATA_ITEMS:
                o = str(old / d)
                nv, no = os.path.normcase(v), os.path.normcase(o)   # Windows: C:\ and c:\ are the same place
                if nv == no or nv.startswith(no + os.sep):
                    n = str(new / d) + v[len(o):]
                    if Path(n).exists() and not Path(v).exists():
                        return n
        return v

    new_cfg = fix(cfg)
    if new_cfg != cfg:
        write_config(cfg_file, new_cfg)


def data_folder(requested: str | None) -> tuple:
    """(the data folder, folders on other drives that still hold model files).  Moves the model files of this folder
    and of earlier Strata folders on the same drive into the data folder, and points their configs there."""
    settings = load_settings()
    dest = Path(requested).expanduser().resolve() if requested else \
        Path(settings["data_dir"]) if settings.get("data_dir") else ROOT.parent / "Strata-data"
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:                                # e.g. no write access next to the Strata folder
        warn(f"cannot use {dest} for the model files ({e}): keeping them in {ROOT}")
        dest = ROOT
    elsewhere = []
    # #198: the data folder remembered before (a --data-dir to a new place) is a source too, and so is a Strata-data
    # folder nested in any of them (an install that kept its models one level down)
    sources = [ROOT, *other_installs(settings)]
    if settings.get("data_dir") and Path(settings["data_dir"]) != dest:
        sources.append(Path(settings["data_dir"]))
    sources += [f / "Strata-data" for f in list(sources) if (f / "Strata-data") != dest]
    seen = set()
    for folder in sources:
        key = os.path.normcase(str(folder))
        if key in seen:
            continue
        seen.add(key)
        if folder == dest or not has_data(folder):
            continue
        if not same_drive(folder, dest):
            elsewhere.append(folder)                    # another drive: used where it is (no 70 GB copy)
            continue
        # the downloads merge file by file (the same file wherever it came from); a prepared pack or MTP layer moves
        # whole or not at all, so two copies are never mixed
        if (folder / "models").is_dir():
            move_into(folder / "models", dest / "models")
        for item in [*((folder / "packs").glob("*") if (folder / "packs").is_dir() else []), folder / "mtp"]:
            rel = item.relative_to(folder)
            if item.exists() and not (dest / rel).exists():
                move_into(item, dest / rel)
        for d in ("packs",):
            try:
                (folder / d).rmdir()                    # empty now
            except OSError:
                pass
        for c in folder.glob("strata-*.json"):
            repoint_config(c, folder, dest)
        if has_data(folder):
            elsewhere.append(folder)                    # in use, or a copy the data folder already has
            warn(f"some model files are still in {folder} (in use, or already in {dest})")
        else:
            ok(f"model files from {folder} moved to {dest} (a new copy of Strata finds them there)")
    installs = [str(ROOT)] + [p for p in settings.get("installs", []) if p != str(ROOT) and Path(p).is_dir()]
    save_settings({**settings, "data_dir": str(dest), "installs": installs[:20]})
    return dest, elsewhere


def write_config(path: Path, cfg: dict):
    """A run config, written whole or not at all (#459): to a temporary file first, then moved over the old one, so
    a setup stopped half-way (a closed window, a full disk) never leaves an empty strata-*.json behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# #629: the run config's keys setup writes itself (and rewrites on every setup run); any other key is the user's - a
# "sampling" or "mcp_servers" block, "allowed_hosts", "cors_origins", "open_browser" - and is kept when setup runs again
SETUP_KEYS = frozenset({"exe", "args", "cwd", "tokenizer", "model_name", "log", "lib_dirs", "port", "backend", "env",
                        "gpu", "gpus_asked", "layer_split", "host", "api_key", "draft_vocab", "vision"})
SETUP_ENV = frozenset({"STRATA_HIPBLASLT_TUNING", "STRATA_RESIDENT_PIN"})   # the "env" entries setup writes
SETUP_VISION = frozenset({"exe", "mmproj", "model", "gpu", "max_tokens", "threads"})


def carry_over(old: dict, cfg: dict) -> list[str]:
    """#629: setup run again for an installed model keeps what the user added to its run config: every key setup does
    not write (`SETUP_KEYS`), the "env" entries setup does not write, and in "vision" the keys setup does not write
    plus an mmproj of their own (a Q8_0 one, #625) that still exists.  `cfg` (the new config) is updated in place;
    the names of what was kept are returned.  Engine options added by hand to "args" are not merged (setup chooses
    those): `args_dropped` names them."""
    kept = []
    for k, v in old.items():
        if k not in SETUP_KEYS and k not in cfg:
            cfg[k] = v
            kept.append(k)
    env = {k: v for k, v in (old.get("env") or {}).items() if k not in SETUP_ENV and k not in (cfg.get("env") or {})} \
        if isinstance(old.get("env"), dict) else {}
    if env:
        cfg["env"] = {**(cfg.get("env") or {}), **env}
        kept += [f"env {k}" for k in env]
    ov, nv = old.get("vision"), cfg.get("vision")
    if isinstance(ov, dict) and isinstance(nv, dict):
        for k, v in ov.items():
            if k not in SETUP_VISION and k not in nv:
                nv[k] = v
                kept.append(f"vision {k}")
        mm = ov.get("mmproj")                          # a file of the user's own: not the one setup downloads
        if isinstance(mm, str) and Path(mm).name != Path(str(nv.get("mmproj"))).name and Path(mm).is_file():
            nv["mmproj"] = mm
            kept.append("vision mmproj")
    return kept


def args_dropped(old: dict, cfg: dict) -> list[str]:
    """#629: the engine options of the earlier run config that the new one has no more (by flag name): options added
    by hand, which a setup run does not carry over - the start of the line that names them."""
    def flags(c):
        a = c.get("args") if isinstance(c.get("args"), list) else []
        return [str(x) for x in a if str(x).startswith("--")]
    new = set(flags(cfg))
    return list(dict.fromkeys(f for f in flags(old) if f not in new))


def write_setup_config(cfg_path: Path, cfg: dict, source: Path | None = None) -> None:
    """#629: setup's run config, written over an earlier one for the same model without losing what the user added
    to it: the keys setup does not write are carried over (carry_over), and the earlier file is kept as
    strata-<model>.json.bak when it changes.  `source`: an earlier install's config to carry the keys over from when
    this folder has none yet (a copy set up like the last one).  A line says what was kept, one what was not."""
    old_path = cfg_path if cfg_path.is_file() else source
    old = None
    if old_path is not None and old_path.is_file():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass
        if not isinstance(old, dict):
            old = None
    kept = carry_over(old, cfg) if old is not None else []
    bak = None
    if cfg_path.is_file() and old != cfg:
        bak = cfg_path.with_name(cfg_path.name + ".bak")
        try:
            shutil.copyfile(cfg_path, bak)
        except OSError as e:
            warn(f"could not keep a copy of the earlier {cfg_path.name} ({e.strerror or e})")
            bak = None
    write_config(cfg_path, cfg)
    if kept:
        ok(f"kept from your earlier {old_path.name}: " + ", ".join(kept))
    if bak is not None:
        dropped = args_dropped(old, cfg) if old is not None else []
        say(f"  the earlier run config is kept as {bak.name}" + (
            f"; engine options it had that this one has not (setup chooses those): {' '.join(dropped)}"
            if dropped else ""))


def readable_config(path: Path) -> bool:
    """#459: a config that parses as a JSON object; any other gets a one-line warning naming it."""
    text = None
    try:
        text = path.read_text(encoding="utf-8-sig")
        if isinstance(json.loads(text), dict):
            return True
        why = "not a JSON object"
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:                                 # JSONDecodeError, or bytes that are not UTF-8
        why = "the file is empty" if text is not None and not text.strip() else "not valid JSON"
    warn(f"skipped the earlier config {path} ({why}): setting this copy up without it")
    return False


def previous_config(elsewhere_first: list, settings: dict):
    """The most recently used model config of another Strata folder on this PC, for a folder that has none yet.  One
    that does not parse (an empty or cut-off file, #459) is skipped with a warning: the newest readable one is used,
    and with none this copy is set up as a fresh install."""
    cands = []
    for folder in [*elsewhere_first, *other_installs(settings)]:
        cands += list(folder.glob("strata-*.json"))
    cands = [c for c in dict.fromkeys(cands) if c.is_file()]
    return next((c for c in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True) if readable_config(c)), None)


def choices_from_config(cfg_path: Path) -> dict:
    """The setup answers a config was written with (family, size, context, KV, images, projection, network)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    tag = cfg_path.stem[len("strata-"):]
    family = next((f for f, d in FAMILIES.items() if d["tag"] and tag.startswith(d["tag"])), "qwen")
    model = (tag[len(FAMILIES[family]["tag"]):] if tag.startswith(FAMILIES[family]["tag"]) else tag).upper()
    if model not in MODELS:                            # (sizes have no dash except UD-Q4_K_XL: the old rule)
        model = tag.split("-")[-1].upper()
    a = cfg.get("args", [])
    val = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None   # noqa: E731
    vis = cfg.get("vision")
    esp = val("--control-vector-scaled")
    esp_path = esp.rsplit(":", 1)[0] if esp else None
    return {"family": family, "model": model if model in MODELS else None,
            "context": int(val("--max-context")) if val("--max-context") else None,
            "kv": val("--kv") if val("--kv") in ("int8", "q4_0") else None,
            "vision": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "none",
            "esp": ("on" if Path(esp_path).name == ESP_VECTOR.name else esp_path) if esp_path else "off",
            "host": cfg.get("host"), "api_key": cfg.get("api_key"), "port": cfg.get("port"), "gpu": cfg.get("gpu"),
            "layer_split": cfg.get("layer_split"), "cuda": 12 if config_toolkit(cfg) == 12 else None,
            # #493: --vram-reserve-mib given at setup (images write the default 700 themselves)
            "vram_reserve_mib": int(val("--vram-reserve-mib")) if (val("--vram-reserve-mib") or "").isdigit() and (
                vis is None or int(val("--vram-reserve-mib")) != VISION["gpu"]["reserve_mib"]) else None}


def find_in(roots: list, rel: str):
    """The first of roots/rel that exists."""
    for r in roots:
        if (r / rel).exists():
            return r / rel
    return None


# ------------------------------------------------------------------------------------------------ start
def model_config(path: Path) -> bool:
    """#549: a model's run config (a JSON object with "exe" and "args"). Any other strata-*.json in the folder (a
    file of the user's own, a cut-off one) is skipped with a warning naming it instead of stopping setup."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            return True
        why = 'no "exe" or "args"'
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:
        why = "not valid JSON"
    warn(f"skipped {path.name} ({why}): it is not a Strata model config")
    return False


def installed_configs():
    return [p for p in sorted(ROOT.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if model_config(p)]


def source_version() -> str:
    """The engine version the source tree builds (CMakeLists.txt's project version)."""
    m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    return m.group(1) if m else "0"


def engine_version(exe: Path) -> tuple:
    """The version in the engine folder's BUILD.json, or else the one compiled into the binary.  A locally compiled
    engine is not necessarily the source's version: when compiling a `git pull` fails, the previous engine is kept
    (issue #49)."""
    try:
        meta = json.loads((Path(exe).parent / "BUILD.json").read_text())
    except (OSError, ValueError):
        meta = {}
    v = str(meta.get("version") or "")
    if not v:                                          # the version compiled into the binary: 0.1.13 and newer
        try:                                           # carry it, so a binary without it is older
            m = re.search(rb"engine=(\d+\.\d+\.\d+)\n", Path(exe).read_bytes())
            v = m.group(1).decode() if m else "0.1.12"
        except OSError:
            v = "0"
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())


def is_wsl() -> bool:
    return sys.platform.startswith("linux") and "microsoft" in platform.uname().release.lower()


def rotational_disk(path) -> str | None:
    """#605 (Linux): the disk's name when `path` is on a rotational disk (sysfs queue/rotational), else None."""
    if WIN:
        return None
    try:
        st = os.stat(path)
        p = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
        for q in (p, p.parent):                        # a partition has its disk's queue
            f = q / "queue" / "rotational"
            if f.exists():
                return q.name if f.read_text().strip() == "1" else None
    except (OSError, ValueError, AttributeError):
        pass
    return None


def hip_config_cards(sel) -> list[dict]:
    """#566: the AMD cards a HIP config runs on - its "gpu" (one index or a list) in HIP's numbering, as amd_gpus
    lists them; no "gpu": the supported card with the most VRAM, as setup picks it.  Each card's name carries its
    architecture (an RX 7900 XTX and an RX 7900 XT are both gfx1100; a card without a product name in sysfs is known
    by its arch only).  A card that is not found is {} (the key then says "?")."""
    amd = amd_gpus()
    if sel is None:
        usable = [g for g in amd if amd_problem(g) is None]
        sel = max(usable, key=lambda x: (round(x["vram_gb"]), -x["index"]))["index"] if usable else None
    byid = {g["index"]: g for g in amd}
    cards = []
    for i in (sel if isinstance(sel, list) else [sel]):
        g = byid.get(i)
        if g is None:
            cards.append({})
            continue
        arch = g.get("arch") or ""
        name = g.get("name") or "?"
        cards.append({**g, "name": name if not arch or arch in name else f"{name} ({arch})"})
    return cards


def hardware_key(cfg: dict) -> str:
    """What a calibration is valid for: this GPU, CPU and RAM, and the model with its context and images setting
    (the context's KV cache and the image encoder take VRAM from the expert cache).  #566: a HIP config's cards are
    AMD's (hip_config_cards) - nvidia-smi's list named them "?" (or another card with that number) before."""
    sel = cfg.get("gpu")
    if cfg.get("backend") == "hip":
        gl = hip_config_cards(sel)
    else:
        gl = [gpu_info(i) or {} for i in sel] if isinstance(sel, list) else [gpu_info(sel) or {}]
    g = {"name": " + ".join(x.get("name", "?") for x in gl), "vram_gb": sum(x.get("vram_gb", 0) for x in gl)}
    a = cfg.get("args", [])
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a else "?"
    return "|".join([g.get("name", "?"), f"{g.get('vram_gb', 0):.0f}GB", cpu_info()[0], f"{ram_gb():.0f}GB",
                     cfg.get("model_name", "?"), ctx, "images" if "--vision" in a else "text"])


def calibrate_config(cfg_path: Path) -> bool:
    """Measure the engine's hardware-dependent settings on this PC (tools/calibrate.py), write them into the run
    config and remember them per PC and model in the settings file, so an update or a reinstall keeps them."""
    sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    say()
    say("  Tuning Strata for this PC: the output speed is measured with a few engine settings (the PCIe share, the")
    say("  draft depth, the CPU threads). It takes about 5-10 minutes; the PC is busy meanwhile.")
    try:
        since = os.path.getsize(cfg["log"]) if cfg.get("log") and os.path.isfile(cfg["log"]) else 0
    except OSError:
        since = 0
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops an install: the defaults stay
        warn(f"the tuning did not finish ({e}): the default settings stay")
        why = CAL.engine_error(cfg.get("log"), since)  # #447: the engine's own reason, not only "see the log"
        if why:
            say(f"       the engine said: {why}")
        return False
    cfg["args"] = CAL.apply(cfg["args"], res["settings"])
    write_config(cfg_path, cfg)
    st = load_settings()
    st.setdefault("calibration", {})[hardware_key(cfg)] = {"settings": res["settings"], "tok_s": res["report"].get("tok_s"),
                                                           "date": time.strftime("%Y-%m-%d")}
    save_settings(st)
    if res["settings"]:
        ok("tuned for this PC: " + ", ".join(f"{k} {v}" for k, v in res["settings"].items())
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    else:
        ok("tuned for this PC: the default settings are already the fastest here"
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    return True


def saved_calibration(cfg: dict) -> dict | None:
    """The settings an earlier calibration found for this PC and model, if any."""
    return (load_settings().get("calibration") or {}).get(hardware_key(cfg))


def setup_calibration(cfg: dict, hip: bool) -> dict | None:
    """The calibration a (re-)install applies to its new config: the one saved for this PC and model.  #566: on Linux
    HIP too - hardware_key now names the AMD cards, so a `./setup.sh --calibrate` run is matched to its card and
    model.  Windows HIP keeps the defaults for now (not tried there).  Setup still offers the tuning itself on NVIDIA
    only: each control is verified on HIP first."""
    if hip and WIN:
        return None
    return saved_calibration(cfg)


def upgrade_config(cfg_path: Path, cfg: dict) -> dict:
    """Configs written before v0.1.13 read prompts in fixed 2048-token chunks; the engine now picks the chunk
    itself (`--prefill auto`: up to 8192, as the free VRAM allows - about 2x faster on long prompts).  Under WSL,
    KV streaming is dropped: its RAM copy must be pinned, and the driver pins only about 1 GB there."""
    a = cfg.get("args", [])
    changed = False
    ver = engine_version(cfg["exe"]) if "--prefill" in a else (0, 0, 0)
    if "--prefill" in a and a[a.index("--prefill") + 1] == "2048" and ver >= (0, 1, 13):
        a[a.index("--prefill") + 1] = "auto"
        changed = True
        ok("prompt reading: the engine now picks its chunk size (--prefill auto)")
    elif "--prefill" in a and a[a.index("--prefill") + 1] == "auto" and (0, 0, 0) < ver < (0, 1, 13):
        a[a.index("--prefill") + 1] = "2048"           # an older engine kept after a failed update (issue #49)
        changed = True
        warn(f"the installed engine is {'.'.join(map(str, ver))}: prompts are read in 2048-token chunks until it is updated")
    if is_wsl() and "--kv-resident" in a:
        i = a.index("--kv-resident")
        del a[i:i + 2]
        changed = True
        ok("WSL: KV streaming off (the driver pins only about 1 GB of RAM); the KV cache stays in VRAM")
    if changed:
        write_config(cfg_path, cfg)
    return cfg


def update_install(have: list, a) -> int:
    """#475: `setup.py --update` (UPDATE.bat / update.sh, after their git pull): what a plain START-HERE.bat does to
    an install before it starts the model, without starting it - the Python packages, the ready-made engine when this
    setup needs a newer one (MIN_ENGINE; a compiled engine when its source changed), each installed model's config
    upgrades and its draft subset.  No question is asked and the model files are not touched; a model still running
    keeps its engine (update_installed_engine says to close it and run this again)."""
    have = [p for p in have if model_config(p)]        # #549: a strata-*.json that is no model config is skipped
    if not have:
        say("  No model is installed in this Strata folder yet: run START-HERE.bat (Linux: ./setup.sh) to set it up -")
        say("  it finds an earlier install's model files next to it and reuses them.")
        return 0
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")
    if not a.build:
        update_installed_engine(a.prebuilt)
    for cfg_path in have:
        cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
        if "--mtp" in cfg["args"][:-1]:
            refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
        if cfg.get("backend") == "hip" and WIN:
            hip_runtime_beside_exe(Path(cfg["exe"]).parent)   # #468 #461
        ok(f"{cfg.get('model_name', cfg_path.stem)}: up to date")
    ver = engine_version(Path(json.loads(have[0].read_text(encoding="utf-8-sig"))["exe"]))
    say()
    ok("Strata is updated" + (f" (engine {'.'.join(map(str, ver))})" if any(ver) else "") +
       ". Start the model with " + ("START-HERE.bat" if WIN else "./setup.sh") + " when you want it.")
    return 0


def settings_summary(cfg: dict, port=None) -> str:
    """#564: the settings a start uses, in one line: the config's engine options (the model's file paths left out)
    and the server's own fields, so a change made by hand to strata-<model>.json can be checked without the log."""
    a, out, i = [str(x) for x in cfg.get("args") or []], [], 0
    while i < len(a):
        flag = a[i]
        val = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else None
        i += 1 if val is None else 2
        if not flag.startswith("--"):
            continue                                   # a positional: the model file
        if val is not None and ("/" in val or "\\" in val or val.lower().endswith((".gguf", ".bin"))):
            continue                                   # a path: --native, --mtp, --profile ...
        out.append(flag if val is None else f"{flag} {val}")
    srv = [f"{cfg.get('host', '127.0.0.1')}:{port or cfg.get('port', 8080)}"]
    if cfg.get("api_key"):
        srv.append("api key set")
    if cfg.get("open_browser") is False:               # #609
        srv.append("no browser")
    for k in ("gpu", "layer_split", "draft_vocab", "fit_max_tokens", "reasoning_budget_tokens", "anthropic_thinking"):
        if cfg.get(k) is not None:
            v = cfg[k]
            srv.append(f"{k} {','.join(map(str, v)) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v}")
    return " ".join(out) + ("; " if out else "") + "server " + ", ".join(srv)


def start(cfg_path: Path, port: int | None, gpu: int | list | None = None, open_browser=True, yes=False,
          layer_split=None, keep=None) -> int:
    """keep: settings given on this start that the model keeps from now on (--host, --api-key, --draft-vocab,
    --vram-reserve-mib)."""
    cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    keep = {k: v for k, v in (keep or {}).items() if v is not None}
    reserve = keep.pop("vram_reserve_mib", None)       # #493: an engine argument, kept in the config's args
    if reserve is not None:
        args = cfg["args"]
        if "--vram-reserve-mib" in args[:-1]:
            args[args.index("--vram-reserve-mib") + 1] = str(reserve)
        else:
            args += ["--vram-reserve-mib", str(reserve)]
        write_config(cfg_path, cfg)
        ok(f"saved for this model: {reserve} MiB of VRAM kept free for other programs (--vram-reserve-mib)")
    if keep and any(cfg.get(k) != v for k, v in keep.items()):   # #179: a --host/--api-key on a start was ignored
        cfg.update(keep)
        write_config(cfg_path, cfg)
        ok("saved for this model: " + ", ".join(
            "api key" if k == "api_key" else ("the browser opens" if v else "no browser") if k == "open_browser"
            else f"{k.replace('_', ' ')} {v}" for k, v in keep.items()))
    cfg_path.touch()                                     # the most recently used model
    if "--mtp" in cfg["args"][:-1]:
        refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if cfg.get("backend") == "hip":                    # AMD, numbered as HIP numbers them (setup's KFD order)
        if WIN:
            hip_runtime_beside_exe(Path(cfg["exe"]).parent)   # #468 #461: also fixes a 0.1.34 install
        amd = amd_gpus()
        if isinstance(gpu, list):                      # --gpus: saved, this model runs on these cards from now on
            cards = amd_parse_gpus(",".join(str(i) for i in gpu), amd)
            built = (engine_archs_hip() or [])
            miss = [x for x in cards if built and x["arch"] not in built]
            if miss:
                fail("the installed engine has no code for " + ", ".join(f"{x['name']} ({x['arch']})" for x in miss),
                     "set it up for these cards: ./setup.sh --setup --backend hip --gpus " + ",".join(map(str, gpu)))
            cfg["gpu"], cfg["gpus_asked"] = gpu, True
            cfg["layer_split"] = layer_split or cfg.get("layer_split") or "auto"
            split_budget(cfg)                          # #498: before it is saved (it stops when the RAM is short)
            write_config(cfg_path, cfg)
            gpu = None
        elif gpu is not None:
            cmd += ["--gpu", str(gpu)]
        found = []
        use = gpu if gpu is not None else cfg.get("gpu")
        if isinstance(use, list):
            byid = {x["index"]: x for x in amd}
            ok("GPUs: " + " + ".join(gpu_name(byid[i]) if i in byid else f"GPU {i} (not found)" for i in use)
               + f" together, AMD (layers split {cfg.get('layer_split') or 'auto'})")
        else:
            g = next((x for x in amd if x["index"] == (use if use is not None else x["index"])
                      and amd_problem(x) is None), None)
            if g is not None:
                ok(f"GPU: {g['name']} ({g['vram_gb']:.0f} GB, AMD)")
    else:
        found = gpus()
        global OLD_GPUS                                # Pascal / Volta: named on this start, or this model's CUDA 12 engine
        OLD_GPUS = OLD_GPUS or old_gpus_opt_in(found, gpu if isinstance(gpu, list) else [gpu] if gpu is not None else
                                               cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [cfg.get("gpu")],
                                               12 if config_toolkit(cfg) == 12 else None)
    if cfg.get("backend") == "hip":
        pass
    elif isinstance(gpu, list):                        # --gpus: saved, this model runs on these cards from now on
        check_gpus(gpu, found, yes=yes, named=True)
        cfg["gpu"], cfg["gpus_asked"] = gpu, True
        cfg["layer_split"] = layer_split or cfg.get("layer_split") or "auto"
        split_budget(cfg)                              # #498: before it is saved (it stops when the RAM is short)
        recommend_remote_expert_opt(cfg)
        write_config(cfg_path, cfg)
        gpu = None
    elif gpu is not None:                              # --gpu N: this start only, on that card
        check_gpus([gpu], found)
        cmd += ["--gpu", str(gpu)]
    else:
        cfg = offer_together(cfg_path, cfg, yes)
    use = gpu if gpu is not None else cfg.get("gpu")
    # #364 #384: a resident low-RAM config on several GPUs; #498: a UD-Q4_K_XL config with its RAM budget (by hand)
    if isinstance(use, list) and (split_mmap(cfg) | split_budget(cfg)):
        write_config(cfg_path, cfg)
    if cfg.get("backend") == "hip":
        pass
    elif isinstance(use, list):
        check_gpus(use, found, "(chosen for this model) ", yes=True, named=True)
        byid = {g["index"]: g for g in found}
        cfg = ensure_engine_for([byid[i] for i in use], cfg_path, cfg, yes)
        ok("GPUs: " + " + ".join(gpu_name(byid[i]) for i in use) + f" together (layers split {cfg.get('layer_split') or 'auto'})")
    elif found:
        g = next((x for x in found if x["index"] == use), None) if use is not None else max(
            found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
        if g is not None:
            cfg = ensure_engine_for([g], cfg_path, cfg, yes)
            ok("GPU: " + gpu_name(g))
    browser = open_browser and cfg.get("open_browser") is not False   # #609: "open_browser": false, --no-browser
    if browser:
        cmd.append("--open")
    gb = 0.0
    if "--native" in cfg["args"]:
        try:
            gb = Path(cfg["args"][cfg["args"].index("--native") + 1]).stat().st_size / 1e9
        except (OSError, IndexError):
            pass
    say()
    say("  " + "-" * 100)
    size = f'about {gb:.0f} GB' if gb >= 1 else '34-55 GB'
    a_ = cfg["args"]
    if "--mmap-experts" in a_ and "--resident-budget-gib" not in a_ and "--resident-experts" not in a_:
        # #505: the mapped low-RAM mode loads nothing into RAM up front (the server's narrator says the same)
        say(f"  Starting {cfg.get('model_name', 'the model')}: it maps {size} of experts from the model files (the OS "
            "file cache reads them).")
    else:
        say(f"  Starting {cfg.get('model_name', 'the model')}: it loads {size} into RAM and locks part of it for the "
            "GPU.")
    say("  While it does, YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES (longer the first time after a")
    say("  restart). That is normal: please wait and don't close this window - " + (
        "the browser opens when it is ready." if browser else "the server says when it is ready."))
    say("  Later, closing this window stops the model.")
    say("  " + "-" * 100)
    for n, line in enumerate(textwrap.wrap(f"Settings ({cfg_path.name}): {settings_summary(cfg, port)}", 100,
                                           break_on_hyphens=False)):   # #564: what this start uses
        say(("  " if n == 0 else "    ") + line)
    if not WIN and os.environ.get("STRATA_EXECV"):
        # Replace this process instead of spawning a child. The Docker image sets STRATA_EXECV=1,
        # so there the server is PID 1 and docker stop's SIGTERM reaches the process that can
        # answer the engine with QUIT. Normal Linux starts keep spawning the server as a child.
        os.execv(cmd[0], cmd)
    return subprocess.call(cmd)


# the draft subsets setup copied before (sha256): replaced by the current one, a subset made by hand is kept
OLD_DRAFT_VOCABS = {"369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25"}   # to 0.1.26: 27 Han tokens


DRAFT_VOCABS = {"cjk": "draft_vocab.bin", "en": "draft_vocab_en.bin", "cyrillic": "draft_vocab_cyrillic.bin",
                "fr": "draft_vocab_fr.bin"}


def saved_draft_vocab(cfg_path: Path) -> str | None:
    """The draft subset a model's config chose earlier (--draft-vocab), or None: a setup run again without the flag
    rewrites the config, and would otherwise put the default subset back."""
    try:
        v = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("draft_vocab")
    except (OSError, ValueError, AttributeError):
        return None
    return v if v in DRAFT_VOCABS else None


def vision_tokens(asked: int | None, vision: str, earlier: Path | None) -> int:
    """#625: the most image tokens a picture becomes (the config's vision.max_tokens): --vision-tokens N, else what an
    earlier config of this model chose for the same encoder device (a setup run again keeps it), else the device's
    default (VISION).  More is allowed with a note on the time it takes: setup recommends, it does not cap."""
    default = VISION[vision]["max_tokens"]
    if asked is None and earlier is not None:
        try:
            v = json.loads(earlier.read_text(encoding="utf-8-sig")).get("vision")
        except (OSError, ValueError, AttributeError):
            v = None
        mt = v.get("max_tokens") if isinstance(v, dict) and bool(v.get("gpu")) == (vision == "gpu") else None
        if isinstance(mt, int) and mt > 0 and mt != default:
            asked = mt
    if asked is None:
        return default
    note = ""
    if vision == "cpu" and asked > default:
        note = (" - on the CPU a picture takes longer to encode the more tokens it gets (several seconds more at "
                "1,024 than at 300)")
    elif asked > VISION["gpu"]["max_tokens"]:
        note = " - more than the encoder's default needs more VRAM and context per picture"
    ok(f"images: up to {asked} image tokens per picture (--vision-tokens; default {default}){note}")
    return asked


DRAFT_VOCAB_MIB = {"cjk": 348, "cyrillic": 193, "fr": 151, "en": 133}   # the draft head's VRAM per subset (IQ3_S: the largest)
SMALL_DRAFT_VRAM_GB = 14   # #474: below this the default subset's head can be what does not fit


def draft_vocab_note(vram_gb: float, chosen: str | None) -> list[str]:
    """#474: on a card under 14 GB, the default draft subset (cjk, ~348 MiB of VRAM) can be what does not fit at the
    start ("the draft head does not fit"), and the engine's expert cache gets what a smaller one leaves.  Setup
    RECOMMENDS a smaller one here and changes nothing (the owner's rule, #403 #406): a subset chosen with
    --draft-vocab, or kept from an earlier install, gets no note.  [] for every other case."""
    if chosen or not 0 < vram_gb < SMALL_DRAFT_VRAM_GB:
        return []
    start = "START-HERE.bat" if WIN else "./setup.sh"
    return [f"Tip for a {vram_gb:.0f} GB card: the draft layer's default token subset (with Chinese, Japanese and "
            f"Korean) needs up to ~{DRAFT_VOCAB_MIB['cjk']} MiB of VRAM.",
            f"  For English and code answers, {start} --draft-vocab en needs up to ~{DRAFT_VOCAB_MIB['en']} MiB "
            f"(cyrillic: ~{DRAFT_VOCAB_MIB['cyrillic']}) and leaves the rest to the expert cache - and it is the",
            "  fix when the start stops with \"the draft head does not fit\". The model keeps the choice."]


SMALL_CARD_GB = 7.5            # #496: a card under 8 GB gets a tip (an 8 GB card lists 7.99)


def small_card_note(ctx: int, draft_vocab: str | None) -> list[str]:
    """#496: what frees VRAM on a card under 8 GB when the start stops with "no VRAM is left for the expert cache"
    (the engine already lowers its own reserve on such a card) - a recommendation, setup changes none of it.  (The
    draft layer stays: the server needs it.)"""
    start = "START-HERE.bat --setup" if WIN else "./setup.sh"
    tips = []
    if ctx > 8192:
        tips.append("an 8K context (a smaller KV cache)")
    if draft_vocab != "en":
        tips.append(f"--draft-vocab en (a draft head of ~{DRAFT_VOCAB_MIB['en']} MiB instead of "
                    f"~{DRAFT_VOCAB_MIB[draft_vocab or 'cjk']})")
    lines = ["If the start stops with \"no VRAM is left for the expert cache\" (the engine's log says how much is "
             "short):"]
    if tips:
        lines.append(f"  run {start} again with " + " and ".join(tips) + ", or close other programs that use the GPU.")
    else:
        lines.append("  close other programs that use the GPU.")
    return lines


PARALLEL_MAX = 8               # #465: the engine's batch window holds at most 8 requests
PARALLEL_SHARE = 0.2           # #465: the slots' sessions may take this share of the VRAM the expert cache would hold
PARALLEL_HELD = 0.5            # #465: ... and only where the cache still holds this share of the experts beside them
PARALLEL_COST_NOTE = ("parallel N reduces waiting for several users but costs about 10-25% speed per request on this "
                      "card")


def parallel_slot_gb(ctx: int, kv: str, streaming: bool) -> float:
    """#465: the VRAM one batch slot's session takes: its KV cache (12 QSA layers; with KV streaming only the 32K
    positions the attention reads stay in VRAM) and the DeltaNet state (~0.17 GB).  Measured: 0.56 GiB at 32K int8."""
    kv_tok = 12 * (576 if kv == "q4_0" else 1056)
    return (min(ctx, 32768) if streaming else ctx) * kv_tok / 1e9 + 0.17


def parallel_recommend(vram_gbs, arena_gb: float, ctx: int, kv: str, streaming: bool) -> int:
    """#465: how many requests at once ("parallel") to recommend: 0 = none (one at a time).  Only where the experts
    mostly fit in VRAM - the expert cache (each card's VRAM less ~5 GB, every card of a layer split) still holds
    PARALLEL_HELD of the model's experts beside the slots' sessions, which take at most PARALLEL_SHARE of it, up to 4.
    Where the experts mostly run on the CPU a batch reads about as many experts as the requests one by one and every
    slot's VRAM is expert cache lost: measured on a 12 GB RTX 5070 (Q2_0, 32K), a request alone 11-24% slower with 2-4
    slots, 4 requests together 63 tok/s against 71 one after the other (docs/BATCHING.md)."""
    if isinstance(vram_gbs, (int, float)):
        vram_gbs = [vram_gbs]
    cache_gb = sum(max(0.0, v - 5) for v in vram_gbs)
    slot = parallel_slot_gb(ctx, kv, streaming)
    best = 0
    for n in (2, 3, 4):
        if n * slot <= PARALLEL_SHARE * cache_gb and (cache_gb - n * slot) >= PARALLEL_HELD * arena_gb:
            best = n
    return best


def parallel_note(asked: int | None, vram_gbs, arena_gb: float, ctx: int, kv: str, streaming: bool) -> list[str]:
    """#465: what setup says about "parallel": the recommendation (or, where it would cost speed, why it is left at
    one), or how the asked count compares with it (kept as asked: recommend, never force)."""
    rec = parallel_recommend(vram_gbs, arena_gb, ctx, kv, streaming)
    slot = parallel_slot_gb(ctx, kv, streaming)
    if asked is None or asked <= 1:
        if not rec:
            return [f"Several requests at once: left at one at a time - {PARALLEL_COST_NOTE} (docs/BATCHING.md)."]
        return [f"Several requests at once (opt-in): --parallel {rec} decodes up to {rec} together instead of one "
                f"after the other (each takes ~{slot:.1f} GB of VRAM from the expert cache; docs/BATCHING.md)."]
    lines = [f"parallel requests: {asked} at once (each takes ~{slot:.1f} GB of VRAM from the expert cache, "
             f"{asked * slot:.1f} GB in all)"]
    if asked > PARALLEL_MAX:
        lines.append(f"the engine runs at most {PARALLEL_MAX} at once; it will use {PARALLEL_MAX}")
    if not rec:
        lines.append(f"recommended for this card: one at a time - {PARALLEL_COST_NOTE}; kept as you chose")
    elif asked > rec:
        lines.append(f"recommended for this card: {rec} - more slots leave fewer experts in VRAM, which can make every "
                     "request slower; kept as you chose")
    return lines


DESKTOP_RESERVE_MIB = 3072     # #560 #516: what kept a KDE/Wayland desktop alive beside a full expert cache


def linux_desktop(env=None) -> bool:
    """A graphical session on Linux (Wayland or X)."""
    env = os.environ if env is None else env
    return sys.platform.startswith("linux") and bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY"))


def desktop_reserve_note() -> list[str]:
    """#560 #516: an AMD card that also drives a Linux desktop - with the default 700 MiB reserve the expert cache
    fills it, and when the desktop needs more VRAM amdgpu moves the cache to system RAM, where the OOM killer then ends
    the compositor.  A recommendation, setup changes nothing."""
    return [f"If this AMD card also drives your desktop and the desktop or apps crash once the model is loaded, keep "
            f"more VRAM free: ./setup.sh --vram-reserve-mib {DESKTOP_RESERVE_MIB}",
            "  (remembered for this model; the expert cache gets ~2.3 GB less, a few % of speed)"]


def mtp_corrupt(mtp: Path, env=None) -> bool:
    """#327: True when the MTP tensors an install fetched are not the pinned checkpoint's (tools/mtp_fetch.py verify,
    which hashes only files that changed since they last checked out).  A mirror that ignored range requests left the
    shards' starts there instead, and the draft layer built from them accepted nothing - with no error anywhere."""
    if not (mtp / "tensors").is_dir():
        return False
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "verify", "--out", str(mtp)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 3


def refresh_draft_vocab(rt: Path, choice: str = "cjk") -> None:
    """The draft layer's token subset in the MTP folder: `cjk` (data/draft_vocab.bin, since 0.1.27, #137), `en`
    (data/draft_vocab_en.bin, the English/code subset before it: ~110 MiB less VRAM, English answers 1-2% faster) or
    `cyrillic` (data/draft_vocab_cyrillic.bin: English/code and the whole Cyrillic script, for Ukrainian, Russian,
    Bulgarian, Serbian... answers) or `fr` (data/draft_vocab_fr.bin: English/code and the tokens of French text, #597).
    Copied when missing or when a shipped subset other than the chosen one is there; a subset made by hand is kept."""
    new, dst = ROOT / "data" / DRAFT_VOCABS.get(choice, "draft_vocab.bin"), rt / "draft_vocab.bin"
    if not new.exists() or not rt.is_dir():
        return
    if dst.exists():
        old = hashlib.sha256(dst.read_bytes()).hexdigest()
        shipped = OLD_DRAFT_VOCABS | {hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest()
                                      for f in DRAFT_VOCABS.values() if (ROOT / "data" / f).exists()}
        if old not in shipped or old == hashlib.sha256(new.read_bytes()).hexdigest():
            return
        ok("draft layer: the token subset " + {"cjk": "with Chinese, Japanese and Korean",
                                               "cyrillic": "with the Cyrillic script",
                                               "fr": "for French"}.get(choice,
                                                                                          "for English and code (less VRAM)"))
    shutil.copyfile(new, dst)


def ensure_engine_for(cards, cfg_path: Path, cfg: dict, yes: bool) -> dict:
    """The installed engine must have code for every card the model starts on: a card added later (--gpus with an
    older or newer generation, #128) or a new GPU in the PC otherwise stops the start with 'no kernel image'.  Such a
    card gets the engine compiled for all of them, before the start.  A Pascal / Volta card added to a model on the
    CUDA 13 engine moves the model to the experimental CUDA 12 engine (CUDA 13 has no code for it)."""
    tk = config_toolkit(cfg)
    if tk == 13 and any(int(g["arch"]) < CUDA13_MIN_ARCH for g in cards):
        return use_cuda12(cards, cfg_path, cfg, yes)
    missing = [g for g in cards if not (engine_runs_on(g) if tk == 13 else engine_runs_on(g, tk))]
    if not missing:
        return cfg
    eng = engine_dir(tk)
    info = eng / "BUILD.json"
    meta = json.loads(info.read_text())
    say()
    say("  The installed engine has no code for " + ", ".join(f"{g['name']} (sm_{g['arch']})" for g in missing) +
        ": it is compiled for " + ("these cards" if len(cards) > 1 else "it") + " now.")
    main = gpu_info(cards[0]["index"])
    archs = sorted({int(x) for x in meta.get("archs", [])} | {int(g["arch"]) for g in cards})
    vision = meta.get("vision") or ("gpu" if (eng / VEXE).exists() else "none")
    build_engine({**main, "archs": archs}, vision, yes, get_llama_cpp(), toolkit=tk)
    dirs = json.loads(info.read_text()).get("cuda_dirs") or []
    cfg["lib_dirs"] = dirs + [d for d in cfg.get("lib_dirs") or [] if d not in dirs]
    write_config(cfg_path, cfg)
    return cfg


def get_cuda12_engine(url_base, gpu, vision, yes, build=False) -> Path:
    """The experimental CUDA 12 engine for these cards (gpu["archs"]): the ready-made one (Windows) with NVIDIA's
    CUDA 12 libraries, or compiled here with a CUDA 12.x toolkit (Linux, --build, or no ready-made one)."""
    eng = None if build else get_prebuilt(url_base, gpu, vision, toolkit=12)
    if eng is not None and json.loads((eng / "BUILD.json").read_text()).get("source") != "local":
        pip_cuda_libs(12)
        if vision != "none" and not (eng / VEXE).exists():
            eng = None
    return eng if eng is not None else build_engine(gpu, vision, yes, get_llama_cpp(), toolkit=12)


def engine_lib_dirs(eng: Path, toolkit=13) -> list:
    """The library folders a CUDA engine loads from: its own (a compiled one: the toolkit's), else pip's wheels."""
    meta = json.loads((eng / "BUILD.json").read_text())
    return meta.get("lib_dirs") or meta.get("cuda_dirs") or cuda_lib_dirs(toolkit)


def use_cuda12(cards, cfg_path: Path, cfg: dict, yes: bool) -> dict:
    """A model on the CUDA 13 engine now has a Pascal / Volta card (a --gpus at start): the model's config moves to
    the experimental CUDA 12 engine (one engine per model; its other models keep theirs)."""
    old = min(int(g["arch"]) for g in cards)
    say()
    warn(f"sm_{old} is older than CUDA 13 supports (it dropped Pascal and Volta): this model moves to the experimental "
         "CUDA 12 engine (docs/OLDER_GPUS.md; START-HERE.bat --setup --cuda 13 and newer cards only moves it back)")
    main = gpu_info(cards[0]["index"]) or cards[0]
    vision = "gpu" if cfg.get("vision") else "none"
    eng = get_cuda12_engine(os.environ.get("STRATA_PREBUILT_URL", PREBUILT_URL),
                            {**main, "archs": sorted({int(g["arch"]) for g in cards})}, vision, yes)
    cfg["exe"] = str(eng / EXE)
    cfg["cuda"] = 12
    cfg["lib_dirs"] = engine_lib_dirs(eng, 12)
    if cfg.get("vision") and (eng / VEXE).exists():
        cfg["vision"]["exe"] = str(eng / VEXE)
    write_config(cfg_path, cfg)
    ok(f"engine: {eng / EXE} (CUDA 12, experimental)")
    return cfg


def write_run_script(model, cfg_path, port, open_browser=True):
    """run-<model>.bat / .sh: the server with this config; `open_browser` False (#609: --no-browser) leaves --open out."""
    serve = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
             "--port", str(port)] + (["--open"] if open_browser else [])
    if WIN:
        script = ROOT / f"run-{model.lower()}.bat"
        script.write_text("@echo off\r\ntitle Strata " + model + "\r\ncd /d \"" + str(ROOT) + "\"\r\n" +
                          " ".join(f'"{x}"' for x in serve) + "\r\nif errorlevel 1 pause\r\n", encoding="utf-8")
    else:
        script = ROOT / f"run-{model.lower()}.sh"
        script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n",
                          encoding="utf-8")
        script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ the rope config
def derived_factor(ctx: int, trained: int = 262144) -> float:
    """The automatic extension factor: the FINAL context over the trained one, at least 1.

    Factor 1 removes the automatic expansion - the trained angles stand as they are - but it is not a
    switch for rope as a whole: an explicitly chosen method's settings keep their defined behavior.
    """
    return max(1.0, float(ctx) / float(trained))


def resolve_rope(ctx: int, scaling, scale, trained: int = 262144):
    """The rope config for the context ACTUALLY SERVED: (scaling, scale); scaling None = no scaling flags.

    An explicit --rope-scaling/--rope-scale always wins - a user-supplied factor is kept verbatim even
    when a reduction changed the context.  Past the trained range an omitted method defaults to yarn -
    llama.cpp's extension method: the trained angles survive on the high-frequency pairs and the
    magnitude correction keeps the attention temperature - and an omitted factor is derived from the
    final context (final / trained, at least 1), as is an explicitly chosen method's missing factor
    inside the trained range: factor 1, the trained angles, no expansion.  An explicit none is refused
    past the trained range (the setup will not configure a run it knows is out of spec) rather than
    silently overridden.
    """
    if ctx <= trained:
        if scale is not None and scaling in (None, "none"):
            raise ValueError("--rope-scale needs --rope-scaling linear or yarn (the chosen context fits the "
                             "trained 262144, so there is nothing to scale)")
        if scaling in (None, "none"):
            return None, None          # the stock model, by choice or by default
        return scaling, scale if scale is not None else derived_factor(ctx, trained)
    if scaling == "none":
        raise ValueError(f"a {ctx // 1024}K context is past the model's trained 262144, and --rope-scaling none "
                         "keeps the stock angles there - the model has never seen those positions, so the setup "
                         "refuses the combination instead of quietly overriding it. Pick --rope-scaling yarn or "
                         "linear, or rerun with --context 262144 or lower")
    return scaling or "yarn", scale if scale is not None else derived_factor(ctx, trained)


# ------------------------------------------------------------------------------------------------ main
def sycl_setup(argv) -> int:
    """--backend sycl: the Intel Arc engine (the SYCL port in sycl/, PR #423), experimental. There is no ready-made
    Intel engine: it is compiled from source on the PC (docs/INTEL_ARC.md), then sycl/setup_intel.py runs this setup
    with the Intel steps swapped in. Nothing of the CUDA / HIP paths is used or changed."""
    say()
    warn("Intel Arc (--backend sycl) is EXPERIMENTAL: a community port of the engine, not tested by the Strata "
         "maintainers (no Intel card here). Expect rough edges; issues with your card and driver versions help.")
    if WIN:
        fail("the Intel Arc engine has no Windows setup yet (no ready-made Intel engine either)",
             "run it on Linux (Ubuntu 24.04 with Intel's GPU driver and oneAPI): docs/INTEL_ARC.md")
    say("  There is no ready-made Intel engine: it is built from source with Intel oneAPI (icpx + oneMKL),")
    say("  docs/INTEL_ARC.md. Setup continues with sycl/setup_intel.py.")
    rest, skip = [], False
    for x in argv:                                     # setup_intel.py drives this setup through its AMD path
        if skip:
            skip = False
        elif x == "--backend":
            skip = True
        elif not x.startswith("--backend="):
            rest.append(x)
    script = ROOT / "sycl" / "setup_intel.py"
    if not script.exists():
        fail(f"{script} is missing", "use a full Strata checkout (git clone) - docs/INTEL_ARC.md")
    return subprocess.call([sys.executable, str(script), *rest])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), help="qwen = Qwen3.8-Flash-Next, swift = Swift 1.5")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--context", type=int)
    ap.add_argument("--rope-scaling", choices=["none", "linear", "yarn"],
                    help="the RoPE extension for a context past the model's trained 262144: linear (position "
                         "interpolation) or yarn - llama.cpp's types. Omitted with a scaled context, the setup "
                         "picks yarn; 'none' is refused for such a context")
    ap.add_argument("--rope-scale", type=float,
                    help="the extension factor (default: the final context over the trained 262144, at least 1 - "
                         "1.5 for 384K, 2 for 512K, 1 inside the trained range)")
    ap.add_argument("--kv", choices=["int8", "q4_0", "k8v4"],
                    help="KV cache precision above 8K context: int8 (default), q4_0 (half the memory, a little less "
                         "precise) or k8v4 (hybrid: INT8 K + 4-bit V, 816 B/cell)")
    ap.add_argument("--vision", choices=["yes", "no", "none", "gpu", "cpu"],
                    help="let the model read images (yes = the encoder on the GPU)")
    ap.add_argument("--vision-tokens", type=int, metavar="N",
                    help="the most image tokens a picture becomes (default 1024 with the encoder on the GPU, 300 on "
                         "the CPU): more reads small text and charts better, and takes longer to encode; remembered "
                         "for this model")
    ap.add_argument("--experimental-speed-projection", metavar="on|off|GGUF",
                    help="EXPERIMENTAL, off by default: the control vector in data/experimental-speed-projection "
                         "(or another GGUF) as a projection on layers 4-44; see docs/DETAILS.md")
    ap.add_argument("--port", type=int, help="the server's port (default: the one the install was set up with, 8080 for a new one)")
    ap.add_argument("--gpu", help="one GPU, numbered as nvidia-smi numbers them (default: asked when several can be "
                                  "used; with --setup it is saved, when starting it is for that start only)")
    ap.add_argument("--gpus", help="several GPUs sharing one model, as nvidia-smi numbers them (AMD: as setup lists "
                                   "them): \"0,2\", or \"all\" (every card that can); the first is the main one. "
                                   "Saved, also when starting (see docs/MULTI_GPU.md)")
    ap.add_argument("--layer-split", help="with --gpus: where each later GPU's layers start (\"18\", \"16,32\"), one "
                                          "rising number per GPU after the first - not layers per card; default "
                                          "auto, placed from each GPU's free VRAM")
    ap.add_argument("--no-remote-expert-opt", action="store_true",
                    help="with two or more GPUs: leave out --remote-expert-opt, which setup adds there (#578)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = also other "
                                   "devices on your network (issue #26; set --api-key too)")
    ap.add_argument("--api-key", help="require this key from clients (recommended with --host 0.0.0.0)")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None,
                    help="do not open the chat page in the browser when the model is ready (for a harness or an app "
                         "that uses the API; remembered for this model, also in run-<model>.bat/.sh)")
    ap.add_argument("--browser", dest="browser", action="store_true",
                    help="open the chat page again when the model is ready (the default; undoes --no-browser)")
    ap.add_argument("--data-dir", help="where the model files go (~70-120 GB): default Strata-data next to this folder, "
                                       "remembered for every Strata folder on this PC")
    ap.add_argument("--models-dir", help="where the GGUF files go (default: <data folder>/models)")
    ap.add_argument("--gguf-dir", help="use GGUF files you already have (a folder with every shard: "
                                       "<name>-00001-of-0000N.gguf ... -0000N-of-0000N.gguf)")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--setup", action="store_true", help="install another model or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--update", action="store_true",
                    help="update the installed engine, Python packages and model settings as a start would, without "
                         "starting the model (UPDATE.bat / update.sh run it after a git pull)")
    ap.add_argument("--build", action="store_true", help="compile the engine instead of using the ready-made one")
    ap.add_argument("--cuda", choices=["12", "13", "auto"], default=os.environ.get("STRATA_CUDA") or None,
                    help="NVIDIA: the CUDA toolkit of this model's engine. auto (default): CUDA 13, the ready-made "
                         "engine; CUDA 12 (experimental) when a chosen card is older than CUDA 13 supports (Pascal, "
                         "Volta). 12 also runs with an older driver (Windows 528+, Linux 525+). docs/OLDER_GPUS.md")
    ap.add_argument("--prebuilt", default=os.environ.get("STRATA_PREBUILT_URL", PREBUILT_URL),
                    help="where the ready-made engine is (a URL folder or a local folder)")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--calibrate", action="store_true",
                    help="tune the engine's settings for this PC (about 5-10 minutes), then start the model")
    ap.add_argument("--draft-vocab", choices=list(DRAFT_VOCABS),
                    help="the draft layer's tokens: cjk = with Chinese, Japanese and Korean (default), en = English "
                         "and code only (~110 MiB less VRAM, English answers 1-2%% faster), cyrillic = English, code "
                         "and the Cyrillic script (Ukrainian, Russian... answers decode ~30%% faster), fr = English, "
                         "code and French (French answers: 18%% more drafts accepted)")
    ap.add_argument("--low-ram", choices=["auto", "on", "off", "resident", "mmap"], default="auto",
                    help="read the model's experts from one file in its folder instead of copying them all into RAM "
                         "(for a PC with a big GPU and little RAM); auto: when the experts would not fit the RAM. In "
                         "this mode the experts the GPU does not hold are copied into RAM once when they fit (resident), "
                         "else read through the OS file cache (mmap); resident / mmap force one of the two")
    ap.add_argument("--resident-budget-gib", type=float, metavar="N",
                    help="UD-Q4_K_XL, UD-IQ4_XS: the GiB of its experts kept in RAM (default: the RAM less 24 GB, 40 on 64 GB; "
                         "more is kept as you choose, with a note)")
    ap.add_argument("--vram-reserve-mib", type=int, metavar="N",
                    help="VRAM in MiB the engine leaves free for other programs (a game, another model; the engine's "
                         "default: 700); the expert cache takes that much less")
    ap.add_argument("--parallel", type=int, metavar="N",
                    help="up to N requests decode together (batch slots, opt-in; default: one at a time, the others "
                         "wait). Each slot takes VRAM from the expert cache; setup says what it recommends")
    ap.add_argument("--kv-streaming", choices=["auto", "on", "off"], default="auto",
                    help="from a 64K context: keep the KV cache in RAM and only the attention's window in VRAM (more "
                         "experts fit on the GPU); auto: when the RAM has room for it")
    ap.add_argument("--backend", choices=["cuda", "hip", "sycl"],
                    help="cuda = NVIDIA (default), hip = AMD RX 7900 / 7800 / 7700 XT, RX 9060 XT / 9070 / AI PRO R9700 on "
                         "Linux or Windows (chosen by itself when the PC has no NVIDIA card Strata can use), "
                         "sycl = Intel Arc, EXPERIMENTAL: Linux, built from source (docs/INTEL_ARC.md)")
    ap.add_argument("--skip-build", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.backend == "sycl":                            # Intel Arc: the SYCL port's own setup (sycl/setup_intel.py)
        return sycl_setup(sys.argv[1:])
    if a.resident_budget_gib is not None and not a.resident_budget_gib > 0:
        ap.error("--resident-budget-gib takes a number of GiB above 0, e.g. --resident-budget-gib 32")
    if a.vision_tokens is not None and a.vision_tokens < 1:
        ap.error("--vision-tokens takes a number of image tokens, 1 or more, e.g. --vision-tokens 768")
    if a.vram_reserve_mib is not None and a.vram_reserve_mib < 0:
        ap.error("--vram-reserve-mib takes a number of MiB, 0 or more, e.g. --vram-reserve-mib 2048")
    if a.gpu is not None:                             # --gpu 0,2 means --gpus 0,2 (a user tried it: issue report)
        if "," in a.gpu:
            a.gpus, a.gpu = a.gpus or a.gpu, None
        elif a.gpu.strip().isdigit():
            a.gpu = int(a.gpu)
        else:
            ap.error(f"--gpu takes a GPU number as nvidia-smi numbers them, e.g. --gpu 1 (or --gpus 0,2), not {a.gpu!r}")
    say("Strata - Qwen3.8-Flash-Next on a normal PC (a GPU + system RAM + CPU)")
    data, elsewhere = data_folder(a.data_dir)          # the model files: in the data folder, found from any copy
    roots = [data, *elsewhere]
    if a.models_dir is None:
        a.models_dir = str(data / "models")

    # ---- 0. already installed: just start it
    have = installed_configs()
    if a.update:                                       # #475: UPDATE.bat / update.sh - never starts the model
        return update_install(have, a)
    explicit = a.setup or a.model or a.family or a.check or a.no_start
    adopted = None                                     # #629: the earlier install this copy is set up like
    if not have and not explicit:                      # a new copy of Strata (an update unzipped elsewhere): set it
        prev = previous_config(elsewhere, load_settings())   # up like the last one, from the files already here
        if prev is not None:
            ch = choices_from_config(prev)
            if ch["model"]:
                say(f"  Found your earlier install in {prev.parent} ({prev.stem[len('strata-'):]}): setting up this "
                    "copy the same way - the model files are reused, nothing big is downloaded.")
                a.family, a.model, a.context = ch["family"], ch["model"], a.context or ch["context"]
                adopted = prev
                a.kv = a.kv or ch["kv"]
                a.vision = a.vision or ch["vision"]
                a.experimental_speed_projection = a.experimental_speed_projection or ch["esp"]
                a.host, a.api_key = a.host or ch["host"], a.api_key or ch["api_key"]
                a.port = a.port or ch["port"]
                if a.vram_reserve_mib is None:          # #493: an explicit reserve set up before
                    a.vram_reserve_mib = ch.get("vram_reserve_mib")
                if a.cuda is None and ch.get("cuda") == 12:   # the experimental CUDA 12 engine, as before
                    a.cuda = "12"
                if isinstance(ch.get("gpu"), list):     # a layer split: set up across the same cards again
                    a.gpus = a.gpus or ",".join(str(g) for g in ch["gpu"])
                    a.layer_split = a.layer_split or ch.get("layer_split")
                else:
                    a.gpu = a.gpu if a.gpu is not None else ch.get("gpu")
                a.yes = True
    global GPU_PICK, OLD_GPUS
    # starting an installed model: --gpus 0,2 (or all) saves those cards for it and starts on them (it used to start
    # on the first one alone unless given with --setup), --gpu N runs this start on one card; neither: the saved
    # choice, and asked once when the PC has cards that could share the model
    run_gpu = start_gpus(a.gpus) or a.gpu
    port = a.port or 8080                              # a new install's port (issue #32: --port for an existing one)
    if have and a.calibrate and not (a.setup or a.model or a.family or a.check):
        if not a.build:
            update_installed_engine(a.prebuilt)
        pick_cfg = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
            pick_cfg = have[int(ask("Tune which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        if not calibrate_config(pick_cfg):             # #447: said again where it is not lost above the start
            say()
            warn("this PC is NOT tuned: the tuning failed (the reason is above); the model "
                 + ("keeps" if a.no_start else "starts with") + " the default settings")
        return 0 if a.no_start else start(pick_cfg, a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})
    if have and not (a.setup or a.model or a.family or a.check or a.no_start):
        if not a.build:
            update_installed_engine(a.prebuilt)
        if len(have) == 1:
            return start(have[0], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
        say(f"  {len(have) + 1}) install another model / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})

    # ---- 1. the PC
    step(1, "checking your PC")
    found = gpus()
    amd = amd_gpus()
    amd_ok = [g for g in amd if amd_problem(g) is None]
    # older NVIDIA GPUs (Pascal / Volta): the experimental CUDA 12 engine, when chosen (docs/OLDER_GPUS.md)
    OLD_GPUS = OLD_GPUS or old_gpus_opt_in(found, named_gpus(a.gpu, a.gpus), a.cuda,
                                           other=bool(amd_ok) or a.backend == "hip")
    if OLD_GPUS and a.backend != "hip":
        warn(f"older NVIDIA GPUs (Pascal / Volta) can be used ({OLD_GPUS}): experimental, through a second engine "
             "built with CUDA 12 (docs/OLDER_GPUS.md)")
    nv_ok = any(gpu_problem(g) is None for g in found)
    hip = a.backend == "hip" or (a.backend is None and not nv_ok and bool(amd_ok))
    if a.backend is None and nv_ok and amd_ok:
        # both kinds of card: asked (a first run on such a PC used to take NVIDIA without mentioning the Radeon)
        say()
        say("  This PC has NVIDIA and AMD cards Strata can use:")
        say("  1) NVIDIA: " + ", ".join(f"{g['name']} ({g['vram_gb']:.0f} GB)" for g in found if gpu_problem(g) is None)
            + "   (recommended)")
        say("  2) AMD: " + ", ".join(f"{g['name']} ({g['vram_gb']:.0f} GB)" for g in amd_ok)
            + f"   ({'the ready-made AMD engine, no images' if WIN else 'compiled here, images on the CPU'}"
              " - docs/AMD_HIP.md)")
        hip = ask("Which cards?", ["1", "2"], "1", a.yes or a.check) == "2"
        if a.check and not hip:
            say(f"  (the AMD card: {'START-HERE.bat' if WIN else './setup.sh'} --backend hip)")
    cuda_tk = 13                                       # NVIDIA: the toolkit of this model's engine (cuda_choice)
    if hip:                                            # AMD: compiled here; Windows: ready-made
        if WIN and a.gpus:
            fail("several AMD cards sharing one model (--gpus) is Linux-only for now", "use one card: --gpu N")
        say("  Your AMD GPUs:" if amd else "  No AMD GPU found (" + ("Windows lists no AMD display adapter)." if WIN
                                                                   else "the amdgpu driver's KFD topology is empty)."))
        for g in amd:
            say(f"    GPU {g['index']}: {g['name']}, {g['vram_gb']:.0f} GB VRAM - " + (amd_problem(g) or "can be used"))
        usable = [g for g in amd if amd_problem(g) is None]
        if not usable:
            fail("no AMD GPU Strata can use", f"the AMD backend runs on {AMD_CARDS}")
        if a.gpus:                                     # a layer split across these cards, the first one the main
            chosen = amd_parse_gpus(a.gpus, amd)
            gpu = chosen[0]
        elif a.gpu is not None:
            gpu = next((g for g in usable if g["index"] == a.gpu), None)
            if gpu is None:
                fail(f"AMD GPU {a.gpu} cannot be used", "use one of: " + ", ".join(f"--gpu {g['index']}" for g in usable))
            chosen = [gpu]
        else:
            gpu = max(usable, key=lambda x: (round(x["vram_gb"]), -x["index"]))
            chosen = [gpu]
        # the engine is compiled for every chosen card's architecture
        gpu = {**gpu, "count": len(amd), "archs": sorted({g["arch"] for g in chosen})}
        chosen = [gpu] + chosen[1:]
        sel = [g["index"] for g in chosen]
        multi = sel if len(sel) > 1 else []
        a.gpu = gpu["index"] if len(amd) > 1 else a.gpu
        if multi:
            ok("GPUs: " + " + ".join(gpu_name(x) for x in chosen) + " together (the model's layers are split across them)")
        ok(f"GPU: {gpu['name']}, {gpu['vram_gb']:.1f} GB VRAM, {gpu['arch']} (AMD: docs/AMD_HIP.md)")
    else:
        if not found:
            fail("no NVIDIA GPU found (nvidia-smi did not answer)",
                 "install the NVIDIA driver from https://www.nvidia.com/drivers and restart the PC"
                 + (f"; AMD ({', '.join(AMD_ARCHS)}): --backend hip" if amd else ""))
        if len(found) > 1 or gpu_problem(found[0]) is not None:
            gpu_table(found)
        sel = choose_gpus(a, found)                    # asked when two or more cards can share the model
        multi = sel if len(sel) > 1 else []
        a.gpu = sel[0]                                 # the main GPU: the checks and the sizing below are its
        GPU_PICK = a.gpu
        gpu = gpu_info(a.gpu)
        chosen = [gpu_info(i) for i in sel]
        gpu["archs"] = sorted({x["arch"] for x in chosen})  # the engine needs code for every one of them
        if multi:
            ok("GPUs: " + " + ".join(gpu_name(x) for x in chosen) + " together (the model's layers are split across them)")
        ok(f"GPU: {gpu['name']}, {gpu['vram_gb']:.1f} GB VRAM, compute capability {cc(gpu)}, driver {gpu['driver']}")
        cuda_tk, why = cuda_choice(gpu["archs"], a.cuda)   # one engine per model: its oldest card decides
        if why:
            (warn if cuda_tk == 13 or str(a.cuda) == "12" else ok)(f"CUDA {cuda_tk}: {why}")
        min_driver = CUDA12_MIN_DRIVER if cuda_tk == 12 else MIN_DRIVER
        if driver_major(gpu) < min_driver:
            fail(f"the NVIDIA driver is too old ({gpu['driver']}; {min_driver} or newer is needed)",
                 "update it with the NVIDIA App or from https://www.nvidia.com/drivers, restart, and run this again" +
                 ("" if cuda_tk == 12 else f" (or --cuda 12: the experimental CUDA 12 engine runs with driver "
                                           f"{CUDA12_MIN_DRIVER} or newer, docs/OLDER_GPUS.md)"))
    if gpu["vram_gb"] < 11:
        warn("less than 12 GB of VRAM: Strata will run, but most experts stay on the CPU and it will be slow")
    ram = ram_gb()
    cpu, avx2, avx512 = cpu_info()
    need = min(d["ram_gb"] for d in MODELS.values())
    low_ok = low_ram_fits("IQ1_M", ram, gpu["vram_gb"]) and a.low_ram != "off"   # the smallest model, mapped
    if ram < need - 4 and not a.check and not low_ok:
        # every model keeps ALL its experts in RAM (23+ GB); VRAM only holds a copy of the most-used ones, so a
        # bigger GPU does not lower this.  The owner's rule: a stop by default, a risk the user can take (--model
        # with --yes, or y)
        confirm_risk(f"RAM: {ram:.0f} GB - the smallest model (the Coder) needs about {need} GB: Strata keeps all of "
                     "the model's experts in RAM (23-50 GB, whatever the GPU), so the OS will page them from disk. "
                     "Expect it to be very slow, and it may not start at all.", bool(a.model), a.yes,
                     f"RAM: {ram:.0f} GB - the smallest model (the Coder) needs about {need} GB",
                     "Strata keeps all of the model's experts in RAM (23-50 GB, whatever the GPU) and the GPU holds a "
                     "copy of the most-used ones: it needs 32 GB of RAM or more (48 GB for the full model); --model "
                     "NAME --yes installs one anyway")
        warn(f"going on with {ram:.0f} GB of RAM, as you chose")
    ok(f"RAM: {ram:.0f} GB" if ram >= need - 4 else f"RAM: {ram:.0f} GB (less than the {need} GB the smallest model needs)"
       + ("; the GPU's VRAM makes up for it (the low-RAM mode)" if ram < need - 4 and low_ok else ""))
    pf = page_file_gb()
    if pf is not None and pf < 4:
        warn(f"Windows' page file is {pf:.1f} GB: the graphics card's memory needs room there too (issue #60), so "
             "the model may not start or may use less VRAM. Set it to \"System managed\": System > About > "
             "Advanced system settings > Performance > Advanced > Virtual memory")
    ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'})")
    floor = cpu_floor(avx2)
    if floor == "unsupported":
        fail("this CPU has neither AVX2 nor SSE4.2; Strata needs at least SSE4.2 (Intel Nehalem, 2008, or newer)")
    if not avx2:
        # #394 #595 #623: the ready-made engine is AVX2; an older CPU gets one compiled here, whose CPU experts run on
        # ggml-cpu's kernels for this CPU.  Experimental: measured only on newer CPUs with the older path forced, and by
        # users on a few Xeons.  A warning, not a stop.
        warn(f"this CPU has no AVX2: Strata support for it is EXPERIMENTAL and slow. Setup compiles the engine on this "
             f"PC for {'AVX' if floor == 'avx' else 'SSE4.2'} (STRATA_ISA_FLOOR={floor}; 10-20 minutes, once), and the "
             "CPU's share of the experts runs on ggml-cpu's kernels, a few times slower than on an AVX2 CPU. "
             "See \"Older CPUs\" in docs/INSTALL.md")
        if hip and WIN:
            fail("the older-CPU engine is compiled from source, and setup compiles the AMD engine on Linux only",
                 "use Linux for an AMD card on this CPU, or an NVIDIA card")
        a.build = True
    if a.check:
        say()
        for m, d in MODELS.items():
            verdict = "fits" if ram >= d["ram_gb"] else "tight" if ram >= d["ram_gb"] - 8 else "does not fit"
            if d.get("budget"):
                verdict = (("EXPERIMENTAL, " if d.get("experimental") else "") +
                           f"fits with {resident_budget_gib(m, ram)} GiB of its experts in RAM, the rest "
                           "read from the SSD" if ram >= d["ram_gb"] else "does not fit")
                if hip:                                # #429: not run on AMD yet (its prompt kernels are CUDA-only)
                    verdict += " - NVIDIA only so far, untested on AMD"
            elif low_ram_needed(m, ram) and low_ram_fits(m, ram, gpu["vram_gb"]) and a.low_ram != "off":
                verdict = (f"fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, gpu['vram_gb']):.0f}% "
                           "of its experts, " + ("the rest stays in RAM)" if low_ram_resident(m, ram, gpu["vram_gb"])
                                                 else "the rest is read from the SSD as needed)"))
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
            say(f"  {i}) {d['title']:20s} {d['by']} - {d['about']}" + ("   [experimental]" if d.get("experimental") else ""))
        family = fams[int(ask("Which model?", [str(i) for i in range(1, len(fams) + 1)], "1", a.yes)) - 1]
    fam = FAMILIES[family]
    ok(f"model: {fam['title']}")
    if fam.get("license"):
        say(f"  Its license: {fam['license']}")
    say()
    names = [m for m in MODELS if family in MODELS[m].get("families", ("qwen", "swift"))]
    names.sort(key=lambda m: bool(MODELS[m].get("experimental")))   # an experimental size last, never the default
    if a.model and a.model not in names:
        # #444: say which family has that size, and (with --gguf-dir) which files Strata can run at all
        elsewhere_fams = [f for f in FAMILIES if f in MODELS[a.model].get("families", ("qwen", "swift"))]
        fail(f"{fam['title']} has no {a.model} model file", "choose one of: " + ", ".join(names)
             + (f" (or {a.model}: " + ", ".join(f"--family {f} --model {a.model}" for f in elsewhere_fams) + ")"
                if elsewhere_fams else "")
             + (f".\n       {SUPPORTED_GGUFS}" if a.gguf_dir else ""))
    for i, m in enumerate(names, 1):
        d = MODELS[m]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        if d.get("budget"):
            say(f"  {i}) {m} {d['about']}; download {d['download_gb']:.0f} GB, keeps ~"
                f"{resident_budget_gib(m, ram)} GB of its {d['arena_gb']:.0f} GB of experts in RAM{fit}")
            continue
        if low_ram_needed(m, ram) and low_ram_fits(m, ram, gpu["vram_gb"]) and a.low_ram != "off":
            fit = (f"   <- fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, gpu['vram_gb']):.0f}%, "
                   + ("the rest in RAM)" if low_ram_resident(m, ram, gpu["vram_gb"]) else "the rest from the SSD)"))
        say(f"  {i}) {m:8s} {d['about']}; download {d['download_gb']:.0f} GB, uses ~{d['arena_gb']:.0f} GB of RAM{fit}")
    rec = str(names.index("IQ3_XXS") + 1) if ram >= 60 and "IQ3_XXS" in names else "1"
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], rec, a.yes)) - 1]
    budget, q4_split = None, False
    if MODELS[model].get("budget"):
        # Unsloth's UD-Q4_K_XL: a RAM budget of experts, the rest from the GGUF on the SSD - not the low-RAM mode (no
        # experts.bin: it would be another 77 GB on the disk), and one GPU (the budget mode has no layer split) unless
        # the RAM holds the GGUFs and 24 GB more: then several, without the budget, if asked for (#498)
        if MODELS[model].get("experimental"):
            warn(f"{model} is EXPERIMENTAL (docs/UNSLOTH_Q4.md): most of its experts are read from the SSD while it "
                 "answers, so it is several times slower than the 2-3-bit models; quality checked against llama.cpp")
        if hip and MODELS[model].get("nvidia_only"):
            # #429 (jkuepker): checked before the 111 GB download.  The HIP engine has no prompt kernels for its
            # Q4_K / Q5_K experts (STRATA_MMQ_KQUANTS is CUDA-only) and it has not been run on AMD: asked, not refused
            confirm_risk(f"{model} has not been run on AMD cards yet: its prompt kernels are NVIDIA-only, so on "
                         f"{gpu_name(gpu)} long prompts read much more slowly, and it may not work at all",
                         bool(a.model), a.yes, f"{model} is NVIDIA-only so far", "choose one of the 2-3-bit models, "
                         f"or --model {model} --yes to try it on AMD anyway", "  Try it anyway?")
            warn(f"installing {model} on an AMD card, as you chose (please report how it runs)")
        if ram < MODELS[model]["ram_gb"]:
            confirm_risk(f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC has {ram:.0f} GB: "
                         f"its RAM budget would be {resident_budget_gib(model, ram)} GiB, so nearly every expert is "
                         "read from the SSD while it answers (very slow), and it may run out of RAM",
                         bool(a.model), a.yes, f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC "
                         f"has {ram:.0f} GB", f"choose one of the 2-3-bit models, or --model {model} --yes to "
                         "install it anyway", "  Install it anyway?")
            warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose")
        budget = budget_choice(model, ram, a.resident_budget_gib)
        if multi and not unsloth_together(a, model, ram, gpu, chosen):
            multi, sel, chosen = [], [gpu["index"]], [gpu]
        q4_split = bool(multi)                         # #498: on several GPUs without the RAM budget
        if not q4_split:
            ok(f"RAM budget: {budget:g} GiB of {model}'s experts in RAM, the rest read from the model files on the SSD")
        if a.low_ram not in ("auto", "off"):
            warn(f"--low-ram {a.low_ram} does not apply to {model}: it always reads part of its experts from the files")
    elif a.resident_budget_gib is not None:
        warn(f"--resident-budget-gib is for UD-Q4_K_XL and UD-IQ4_XS: {model} keeps all of its experts in RAM or in "
             "the low-RAM mode")
    low_ram = budget is None and (a.low_ram in ("on", "resident", "mmap") or
                                  (a.low_ram == "auto" and low_ram_needed(model, ram)))
    if low_ram and multi and not low_ram_together(a, model, ram, gpu, chosen):
        multi, sel, chosen = [], [gpu["index"]], [gpu]
    # (the low-RAM mode's variant is decided once the context is known, below; on several GPUs it is the mapped one)
    if not low_ram and budget is None and ram < MODELS[model]["ram_gb"] - 4:
        confirm_paging(model, ram, a.low_ram, a.yes, bool(a.model))
    ok(f"size: {model}")
    tag = fam["tag"] + model                           # names of the pack, config and start script
    small = min(x["vram_gb"] for x in chosen)         # each card keeps its layers' KV of the whole context
    rec_ctx = 32768 if small < 14 else 65536 if small < 20 else 131072
    if budget is not None:                             # UD-Q4_K_XL: every GB of KV is a GB fewer of cached experts
        rec_ctx = 8192 if small < 14 else 32768
    # #406: the RAM rule is part of the recommendation (the smaller of the two), no longer a cap over the user's choice
    rec_ctx = min(rec_ctx, ram_ctx(model, ram, low_ram))
    if a.context:
        ctx = a.context
    else:
        say()
        say("  Context length = how much text the model can see at once (your chat, files, tool output).")
        say("  Longer needs more VRAM for it, so fewer experts fit on the GPU:")
        for i, c in enumerate(CONTEXTS, 1):
            need_c = ctx_ram_need(model, c, low_ram)
            note = ("   (recommended for your GPU)" if c == rec_ctx else "") + \
                   ("   (experimental: setup adds rope scaling)" if c > 262144 else "") + \
                   (f"   (needs ~{need_c:.0f} GB RAM, this PC has {ram:.0f}: may run out of memory)"
                    if c > 131072 and need_c is not None and need_c > ram else "")
            say(f"  {i}) {c // 1024}K tokens{note}")
        ctx = CONTEXTS[int(ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)],
                               str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    # #406 #364: a context past the RAM rule (an explicit --context, a pick in the list, or the earlier install's) is
    # kept, with what it risks.  It used to become 128K: users ran 256K fine where setup's estimate said no.
    need_gb = ctx_ram_need(model, ctx, low_ram)
    if need_gb is not None and ram < need_gb and ctx > 131072:
        warn(f"{ctx // 1024}K with {model} needs ~{need_gb:.0f} GB of RAM by setup's estimate "
             f"({MODELS[model]['arena_gb']:.0f} GB of experts + the context + room for the rest); this PC has "
             f"{ram:.0f}. Kept as you chose: it may be slower or run out of RAM under load. {rec_ctx // 1024}K is the "
             "recommended size.")
    scaling = a.rope_scaling
    if ctx > 262144 and scaling is None and not a.yes:
        # the interactive path: one question, yarn preselected (llama.cpp's extension method, recall-tested
        # here at 512K).  With --yes nothing prints: resolve_rope takes yarn below and the ok() line says so.
        say()
        say(f"  A {ctx // 1024}K context runs the model past its trained 262,144 positions: the rotary angles")
        say("  get rescaled (llama.cpp's RoPE extension). yarn keeps the trained angles on the high-frequency")
        say("  pairs and corrects the magnitudes; linear shrinks every angle. Override any time with")
        say("  --rope-scaling.")
        scaling = ask("RoPE extension method?", ["yarn", "linear"], "yarn", a.yes)
    try:
        scaling, rope_scale = resolve_rope(ctx, scaling, a.rope_scale)
    except ValueError as e:
        fail(str(e))
    if scaling is not None:
        origin = ("final context / trained 262144; override with --rope-scale" if a.rope_scale is None
                  else "as requested")
        ok(f"rope scaling: {scaling}, factor {rope_scale:g} ({origin})")
    ok(f"context: {ctx} tokens")
    # the KV cache (the model's memory of the conversation): 8-bit, or 4-bit after a Hadamard rotation (PR #21)
    kv = "fp16" if ctx <= 8192 else (a.kv or "int8")
    if ctx > 8192 and not a.kv and not a.yes:
        say()
        say("  KV cache precision (the model's memory of the conversation):")
        say("  1) 8-bit   (recommended: what every published number was measured with)")
        say("  2) 4-bit   half the memory (about 4% faster at 128K), but measurably less precise on long")
        say("             documents; long-context lookups (needle tests) still pass")
        kv = ["int8", "q4_0"][int(ask("KV cache?", ["1", "2"], "1", a.yes)) - 1]
    if ctx > 8192:
        ok(f"KV cache: {'8-bit' if kv == 'int8' else '4-bit (Hadamard-rotated)'}")
    if MODELS[model].get("vision", fam.get("vision")) is False:     # UD-IQ4_XS: images, unlike UD-Q4_K_XL
        vision = "none"
        if a.vision not in (None, "no", "none"):
            warn(f"images are not available with {model} yet: off")
    elif hip:
        vision = hip_vision(a.vision)
    elif a.vision:
        vision = {"yes": "gpu", "no": "none"}.get(a.vision, a.vision)
    else:
        say()
        say("  Images: the model can also read pictures (screenshots, photos, scanned pages). This adds a 0.9 GB")
        say("  download and keeps ~1.4 GB of VRAM free for the image encoder, so text is a few % slower.")
        vision = "gpu" if ask("Do you want images?", ["y", "n"], "n", a.yes) == "y" else "none"
    ok("images: " + {"none": "off", "gpu": "on", "cpu": "on (encoder on the CPU)"}[vision])
    # The low-RAM mode's two variants.  resident: the experts the GPU's cache does not hold (and, as far as RAM allows,
    # the ones the prompt path borrows cache room from) are copied from the pack's experts.bin into RAM once, so
    # nothing is read from the SSD while it answers (engine 0.1.30, --resident-experts; the engine falls back to mmap
    # with a warning when they do not fit the RAM it finds free).  mmap: they are read through the OS file cache.
    # The GPU's share: its VRAM less the dense weights and buffers, this context's KV cache and the image encoder's room.
    resident = False
    if low_ram:
        arena = MODELS[model]["arena_gb"]
        vram = gpu["vram_gb"] - (VISION[vision]["reserve_mib"] / 1024 if vision != "none" else 0)
        share = low_ram_gpu_share(model, vram, ctx, kv)
        rest = arena - low_ram_gpu_gb(model, vram, ctx, kv)
        resident = a.low_ram == "resident" or (a.low_ram != "mmap" and low_ram_resident(model, ram, vram, ctx, kv))
        if multi:      # #364 #384: every chosen card's share (the image encoder on the main one), the mapped variant
            held = min(arena, low_ram_gpu_gb(model, vram, ctx, kv) +
                       sum(low_ram_gpu_gb(model, x["vram_gb"], ctx, kv) for x in chosen[1:]))
            share, resident = held / arena, False
            ok(f"low-RAM mode on {len(chosen)} GPUs: {model}'s experts ({arena:.0f} GB) are read from the model folder "
               f"through the OS file cache instead of a copy in RAM ({ram:.0f} GB); the GPUs hold ~{100 * share:.0f}% "
               "of them")
            if share < 0.6:
                warn("most of the experts are read from the SSD while it answers: expect it to be much slower than "
                     "with enough RAM (a faster SSD and a smaller size help)")
        elif resident:
            ok(f"low-RAM mode: the GPU holds ~{100 * share:.0f}% of {model}'s experts ({arena:.0f} GB) and the other "
               f"~{rest:.0f} GB stay in RAM ({ram:.0f} GB), read once from a copy in the model folder")
        else:
            ok(f"low-RAM mode: {model}'s experts ({arena:.0f} GB) are read from the model folder through the OS file "
               f"cache instead of a copy in RAM ({ram:.0f} GB); the GPU holds ~{100 * share:.0f}% of them")
            if share < 0.6:
                warn("most of the experts are read from the SSD while it answers: expect it to be much slower than "
                     "with enough RAM (a faster SSD and a smaller size help)")
    # EXPERIMENTAL: the experimental-speed-projection control vector (data/experimental-speed-projection), off unless
    # chosen here; with it loaded, the web app and the API switch it off per request
    esp = None
    esp_choice = (a.experimental_speed_projection or "").strip()
    if family in ("qwen", "coder"):                   # the Coder: the same model's residual stream
        if not esp_choice:
            say()
            say("  EXPERIMENTAL - speed projection: a small control vector applied while the model runs (layers 4-44).")
            say("  It changes how the model answers: its package describes it as a refusal-direction projection (the")
            say("  model declines far fewer requests). Off unless you choose it; when on, the web app can switch it off")
            say("  per chat. Details: data/experimental-speed-projection/README.md")
            esp_choice = "on" if ask("Turn on the experimental speed projection?", ["y", "n"], "n", a.yes) == "y" else "off"
        if esp_choice.lower() not in ("off", "no", "n", "0"):
            esp = ESP_VECTOR if esp_choice.lower() in ("on", "yes", "y", "1") else Path(esp_choice).expanduser().resolve()
            if not esp.is_file():
                fail(f"the experimental speed projection's vector is missing: {esp}")
        ok("experimental speed projection: " + ("ON (experimental)" if esp else "off"))
    elif esp_choice.lower() not in ("", "off", "no", "n", "0"):
        warn("the experimental speed projection is made for the original Qwen3.8-Flash-Next, not Swift 1.5: left off"
             if family == "swift" else f"the experimental speed projection is not tested with {model}: left off")
    models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / tag
    shards = gguf_dir_shards(models_dir, fam, model) if a.gguf_dir else \
        [models_dir / model_file(fam, model, i) for i in range(1, model_shards(fam, model) + 1)]
    problem = gguf_dir_problem(models_dir, shards[0], fam, model) if a.gguf_dir else None
    if problem:                                        # #444: files Strata cannot run, or another choice's files
        fail(*problem)
    if not a.gguf_dir and not all(sh.exists() and done(sh) for sh in shards):
        for r in elsewhere:                            # already downloaded in a Strata folder on another drive
            cand = [r / "models" / tag / sh.name for sh in shards]
            if all(c.exists() and done(c) for c in cand):
                models_dir, shards = cand[0].parent, cand
                ok(f"model files found in {models_dir}")
                break
    for s in shards:                                   # #173: a whole file copied in by hand has no finish mark
        if s.exists() and not done(s) and whole_shard(s):
            mark(s, "whole (checked against its own tensor directory)")
    have_model = all(s.exists() and (done(s) or a.gguf_dir) for s in shards)
    # #425 (jctaborda): a download that resumes needs room only for what is still missing - the finished shards and
    # the .part files already on the disk count
    on_disk = sum(f.stat().st_size for s in shards for f in (s, s.with_name(s.name + ".part")) if f.is_file()) / 1e9
    to_fetch = 0 if a.gguf_dir or have_model else max(MODELS[model]["download_gb"] - on_disk, 0)
    need = to_fetch + 8 + \
        (40 if model == "Q2_0" and avx512 and family == "qwen" else 0) + (1 if vision != "none" else 0) + \
        (MODELS[model]["arena_gb"] + 1 if low_ram and not (model == "Q2_0" and avx512 and family == "qwen") else 0)
    if free_gb(models_dir) < need:
        fail(f"not enough free disk space in {models_dir}: need ~{need:.0f} GB" +
             (f" ({on_disk:.0f} GB of the model is already there)" if on_disk >= 1 and not have_model else ""),
             "use --models-dir on a bigger drive")

    # ---- 3. python packages
    step(3, "Python packages")
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES,
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")

    # ---- 4. the engine
    step(4, "the Strata engine")
    llama = get_llama_cpp()
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py, ggml, mtmd)")
    if hip and WIN:                                    # AMD on Windows: the ready-made HIP engine (no compiler)
        eng = None if a.build else get_prebuilt_hip(a.prebuilt, gpu)
        if eng is None:
            fail("no ready-made AMD engine for this Strata version" + (" (--build)" if a.build else ""),
                 "compiling it on Windows: tools\\hip\\build_windows.bat makes strata-windows-x64-hip.zip, then run "
                 "START-HERE.bat --backend hip --prebuilt <its dist folder> (docs/AMD_HIP.md)")
        gpu = hip_card(eng, gpu, amd)
        a.gpu = gpu["index"] if gpu["count"] > 1 else a.gpu
    else:
        eng = None if a.build or hip else get_prebuilt(a.prebuilt, gpu, vision, **({"toolkit": 12} if cuda_tk == 12
                                                                                    else {}))
    if eng is not None and not hip and json.loads((eng / "BUILD.json").read_text()).get("source") != "local":
        pip_cuda_libs(cuda_tk)
        if vision != "none" and not (eng / VEXE).exists():
            warn("the ready-made engine has no image encoder: compiling it")
            eng = None
        else:
            vision = prebuilt_vision(json.loads((eng / "BUILD.json").read_text()), gpu, vision)
    if eng is None:
        eng = build_engine_hip(gpu, llama, vision) if hip else build_engine(gpu, vision, a.yes, llama, toolkit=cuda_tk)
    meta = json.loads((eng / "BUILD.json").read_text())
    if hip and WIN:                                    # the ready-made engine's rocm/bin, first on the engine's PATH
        lib_dirs = [str(d) for d in hip_lib_dirs(eng)]
    else:
        lib_dirs = meta.get("lib_dirs") or meta.get("cuda_dirs") or cuda_lib_dirs(cuda_tk)
    engine_ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
    need_engine = MODELS[model].get("engine", UNSLOTH_ENGINE)
    if budget is not None and engine_ver < need_engine:      # checked before the 94-111 GB download
        fail(f"{model} needs engine {'.'.join(map(str, need_engine))} or newer; this one is {meta.get('version')}",
             "update Strata (or compile the engine with --build) and run setup again")
    ok(f"engine: {eng / EXE}")

    # ---- 5. the model files
    step(5, f"downloading {fam['title']} {model}")
    if not a.gguf_dir:
        missing = [s.name for s in shards if not (s.exists() and done(s))]
        if missing:                                    # #495: files downloaded by hand go here, or --gguf-dir
            say(f"  The model files go in {models_dir}")
            say(f"  Files you already have: put them here with their original names ({', '.join(missing)}), or use "
                "--gguf-dir <their folder>.")
            if hf_endpoint() != HF_DEFAULT:
                say(f"  Downloading from {hf_endpoint()} (HF_ENDPOINT)")
        for s in shards:
            if s.exists() and done(s):
                ok(f"{s.name} already downloaded")
                continue
            # the original's shard 2 is the same file for all its sizes and the Coder: reuse one that is already here
            other = [p for p in Path(a.models_dir).glob("*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf") if done(p)]
            if family in ("qwen", "coder") and s.name.endswith("00002-of-00002.gguf") and other and not s.exists():
                try:
                    os.link(other[0], s)
                    mark(s)
                    ok(f"{s.name} shared with {other[0].parent.name} (identical file)")
                    continue
                except OSError:
                    pass
            download(fam["hf"].format(q=model) + s.name, s)
    check_shards(shards)
    for s in shards:                                   # the Unsloth files: pinned sizes and SHA-256
        if s.name in fam.get("sha256", {}):
            verify_sha256(s, *fam["sha256"][s.name])
    ok("model files present")
    mmproj = Path(a.models_dir) / fam["mmproj"]
    if not mmproj.exists():
        mmproj = find_in(roots, f"models/{fam['mmproj']}") or mmproj
    if vision != "none":
        if not mmproj.exists() and a.gguf_dir and (Path(a.gguf_dir) / fam["mmproj"]).exists():
            mmproj = Path(a.gguf_dir) / fam["mmproj"]
        else:
            download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
        ok(f"vision encoder: {mmproj}")

    # ---- 6. the pack and the MTP draft layer
    step(6, "preparing the model for Strata")
    pack = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
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
        # (UD-Q4_K_XL: --compat-bf16 - its Q8_0 hyper-connection projections become BF16, the form the engine reads)
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
             *fam.get("pack_args", [])], env=env)
    if low_ram and not (pack / "experts.bin").exists():
        say(f"  Writing the experts into one file for the low-RAM mode (one time, {MODELS[model]['arena_gb']:.0f} GB) ...")
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
             "--experts-bin"], env=env)
    ok(f"model prepared: {pack}")
    mtp = (find_in(roots, "mtp/rt/experts.bin") or data / "mtp/rt/experts.bin").parent.parent
    rt = mtp / "rt"
    corrupt = (rt / "experts.bin").exists() and mtp_corrupt(mtp, env)
    if corrupt:
        warn("some MTP tensors are not the checkpoint's (a download mirror that ignored range requests, #327): "
             "fetching them again and rebuilding the draft layer")
    if corrupt or not (rt / "experts.bin").exists():
        say("  The MTP draft layer (speculative decoding, ~2x faster output) comes from the original Qwen checkpoint:")
        say("  only its ~5 GB of MTP tensors are downloaded.")
        run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(rt)],
            env=env)
    # a setup run again without --draft-vocab keeps the subset this model's config chose before (cyrillic, fr, en)
    draft_vocab = a.draft_vocab or saved_draft_vocab(ROOT / f"strata-{tag.lower()}.json")
    refresh_draft_vocab(rt, draft_vocab or "cjk")
    ok(f"MTP draft layer: {rt}")
    for line in draft_vocab_note(gpu.get("vram_gb", 0.0), draft_vocab):   # #474: a recommendation, nothing changes
        say("  " + line)

    # ---- 7. the start script
    step(7, "writing the start script")
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile                   # the PLE table's shard: shard 2 (original) or 1 (Swift)
    ple = next((s for s in shards if any(t.name == "per_layer_token_embd.weight" for t in GGUFFile(s).tensors)), None)
    if ple is None:
        fail("the model has no per_layer_token_embd tensor (is this a Qwen3.8-Flash-Next GGUF?)")
    # (a 4-shard file: the engine finds the PLE table's shard itself from shard 1, the measured setup)
    args = ["--pack", str(pack), "--native", str(shards[0]), *(["--ple-gguf", str(ple)] if len(shards) <= 2 else []),
            "--expert-profile", str(ROOT / "data" / fam.get("profile", "expert-profile.bin")), "--expert-cache", "auto",
            "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt),
            "--max-context", str(ctx)]
    if scaling is not None:     # the resolved config: explicit flags as given, or the automatic yarn+factor
        args += ["--rope-scaling", scaling, "--rope-scale", f"{rope_scale:g}"]
    if ctx > 8192:
        args += ["--kv", kv]
    if resident and a.low_ram != "resident" and engine_ver < RESIDENT_ENGINE:
        resident = False                               # an engine from before --resident-experts would refuse it
        ok(f"low-RAM mode: engine {meta.get('version')} has no resident variant yet; the experts are read through "
           "the OS file cache (run setup again after the next engine update)")
    if low_ram:   # the experts from the pack's experts.bin: the ones the GPU does not hold copied into RAM, or mapped
        args += ["--resident-experts" if resident else "--mmap-experts"]
    disk = None if is_wsl() else rotational_disk(ple)  # #605 (WSL's virtual disk says rotational)
    if disk:
        tensor = next((t for t in GGUFFile(ple).tensors if t.name == "per_layer_token_embd.weight"), None)
        size = getattr(tensor, "expected_bytes", lambda: None)()
        table_gb = size / 1e9 if size else 28.8
        if ram >= MODELS[model]["ram_gb"] + table_gb + 4:
            args += ["--ple-io", "ram"]
            ok(f"the model is on a rotational disk ({disk}): its {table_gb:.0f} GB n-gram table is kept in RAM "
               "(--ple-io ram) - read from the disk at random, it can stall prompts for minutes (#605)")
        else:
            warn(f"the model is on a rotational disk ({disk}): its n-gram table is read from it at random, which can "
                 f"stall prompts for minutes (#605). An SSD is recommended; with ~{table_gb:.0f} GB more RAM, "
                 "--ple-io ram in the config's args keeps the table in RAM instead")
    # KV streaming: from 64K up the whole KV cache lives in RAM and only the part the attention reads (32K positions
    # per layer) stays in VRAM; the VRAM it frees holds more experts (+6% at 128K, +23% at 262K with Q2_0). It
    # costs ~13.7 KB of RAM per context token with 8-bit KV (1.7 GB at 128K), 7.5 KB with 4-bit, so only when it fits.
    kv_ram_gb = ctx * (13 * (576 if kv == "q4_0" else 1056)) / 1e9   # 12 QSA layers + the draft layer
    # Hybrid K8V4 never streams its KV (mode 0 only, layer.hpp), so it is excluded from the WHOLE streaming
    # decision rather than one threshold at a time - a future tier added to this chain cannot reintroduce the
    # combination the engine refuses (PR review).
    # --kv-streaming on|off overrides the RAM test (the owner's rule); k8v4 and WSL stay off - they cannot stream.
    stream_fits = ram >= MODELS[model]["ram_gb"] + kv_ram_gb + 1
    if kv == "k8v4":
        if ctx >= 65536:
            ok("KV streaming off: not supported with --kv k8v4; the KV cache stays in VRAM")
        if a.kv_streaming == "on":
            warn("--kv-streaming on: the engine has no KV streaming with --kv k8v4 (it refuses the pair): off")
    elif is_wsl() and ctx >= 65536:
        ok("WSL: KV streaming off (the driver pins only about 1 GB of RAM); the KV cache stays in VRAM")
        if a.kv_streaming == "on":
            warn("--kv-streaming on: WSL cannot stream the KV cache (its RAM copy must be pinned, and the driver pins "
                 "only about 1 GB there): off")
    elif ctx >= 65536 and a.kv_streaming == "off":
        ok("KV streaming off, as you chose (--kv-streaming off): the KV cache stays in VRAM")
    elif ctx >= 65536 and (stream_fits or a.kv_streaming == "on"):
        args += ["--kv-resident", "32768"]
        ok(f"KV streaming on: the context's KV cache lives in RAM ({kv_ram_gb:.1f} GB), more experts fit in VRAM")
        if not stream_fits:
            warn(f"KV streaming needs ~{kv_ram_gb:.1f} GB of RAM beside the ~{MODELS[model]['ram_gb']} GB {model} "
                 f"uses; this PC has {ram:.0f}. Kept as you chose (--kv-streaming on): it may page or run out of RAM "
                 "under load")
        if q4_split:                                   # #498: no budget to take it out of
            pass
        elif budget is not None and a.resident_budget_gib is None:   # its RAM comes out of the experts' budget
            budget = resident_budget_gib(model, ram, kv_ram_gb)
            ok(f"RAM budget: {budget} GiB (less the KV cache's RAM)")
        elif budget is not None and budget > resident_budget_gib(model, ram, kv_ram_gb):
            warn(f"the KV cache's {kv_ram_gb:.1f} GB of RAM come on top of your {budget:g} GiB RAM budget (setup "
                 f"would take them out of it: {resident_budget_gib(model, ram, kv_ram_gb)} GiB); kept as you chose")
    elif ctx >= 65536:   # #620: say why, so a regenerated config that lost --kv-resident is not a surprise
        ok(f"KV streaming off: it needs ~{kv_ram_gb:.1f} GB of RAM beside the ~{MODELS[model]['ram_gb']} GB {model} "
           f"uses, and this PC has {ram:.0f}; the KV cache stays in VRAM (fewer cached experts). --kv-streaming on "
           "turns it on anyway")
    elif a.kv_streaming == "on":
        warn("--kv-streaming on: a context under 64K is not streamed (the attention's window holds all of it): off")
    if budget is not None and not q4_split:   # UD-Q4_K_XL: the experts read from the GGUF in place, the most-used N
        args += ["--resident-budget-gib", f"{budget:g}"]   # GiB kept in RAM (#498: a layer split has no budget)
    if vision != "none":
        args += ["--vision", "--vram-reserve-mib", str(VISION[vision]["reserve_mib"])]
        if vision == "gpu" and a.vram_reserve_mib is None and 0 < gpu.get("vram_gb", 0.0) <= 12.5:
            # a tip only (recommend, never force): on a 12 GB card the encoder's 700 MiB can leave ~200 MiB free
            print(f"  tip: images on a {gpu['vram_gb']:.0f} GB card can leave little VRAM free; if a request stalls, "
                  f"run setup again with --vram-reserve-mib {VISION_GPU_SMALL_RESERVE_MIB}")
    if a.vram_reserve_mib is not None:                 # #493: VRAM left free for other programs (only when given)
        if "--vram-reserve-mib" in args:
            i = args.index("--vram-reserve-mib") + 1
            if vision == "gpu" and a.vram_reserve_mib < int(args[i]):
                warn(f"--vram-reserve-mib {a.vram_reserve_mib}: the image encoder on the GPU needs ~{args[i]} MiB of "
                     "it; kept as you chose (it may run out of VRAM when it reads a picture)")
            args[i] = str(a.vram_reserve_mib)
        else:
            args += ["--vram-reserve-mib", str(a.vram_reserve_mib)]
        ok(f"VRAM kept free for other programs: {a.vram_reserve_mib} MiB (--vram-reserve-mib; the expert cache takes "
           "that much less)")
    if not multi and 0 < gpu.get("vram_gb", 0.0) < SMALL_CARD_GB:
        # #496: on a 6 GB card the expert cache can get no room at all; the engine lowers its own reserve when that
        # is what it takes, and says what is short when even that is not enough.  Setup only says what helps.
        for line in small_card_note(ctx, draft_vocab):   # a recommendation: nothing changes
            say("  " + line)
    elif hip and a.vram_reserve_mib is None and linux_desktop():
        for line in desktop_reserve_note():              # #560 #516: a recommendation: nothing changes
            say("  " + line)
    if esp is not None:
        # the package's profile, with llama.cpp's flags (the engine takes the same ones)
        args += ["--control-vector-scaled", f"{esp}:1.0", "--control-vector-layer-range", "4", "44",
                 "--cvec-mode", "project", "--cvec-dir", "per-layer"]
    cfg = {"exe": str(eng / EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{model.lower()}", "log": str(ROOT / f"strata-{tag.lower()}.log"),
           "lib_dirs": lib_dirs, "port": port}
    if cuda_tk == 12:                                  # the experimental CUDA 12 engine (engine-cuda12/)
        cfg["cuda"] = 12
    if hip:
        cfg["backend"] = "hip"
        # the dense prompt GEMMs through hipBLASLt with kernels measured on this GPU generation (tools/hip; +40-60%
        # prompt speed on the 7900 XTX): only a table for this card's arch AND the installed hipBLASLt version (the
        # engine refuses any other one and falls back to plain hipBLAS)
        table = hipblaslt_table(gpu["arch"], lib_dirs, meta.get("hipblaslt_version"))
        if table:
            cfg["env"] = {"STRATA_HIPBLASLT_TUNING": str(table)}
        if resident:   # ROCm: large page-locked host allocations can fail or be slow for the CPU; keep the copy pageable
            cfg.setdefault("env", {})["STRATA_RESIDENT_PIN"] = "0"
    if gpu["count"] > 1 or a.gpu is not None:
        cfg["gpu"] = gpu["index"]                      # the engine is told this card (issue #51)
        cfg["gpus_asked"] = True                       # chosen at setup: not asked again at start
    if multi:                                          # a layer split across these cards (the server adds the flag)
        cfg["gpu"] = multi
        cfg["layer_split"] = a.layer_split or "auto"
        ok(f"layer split across GPUs {multi} ({cfg['layer_split']})")
        recommend_remote_expert_opt(cfg, off=a.no_remote_expert_opt)
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if draft_vocab:
        cfg["draft_vocab"] = draft_vocab
    if a.browser is not None:                          # #609: only when given (else an earlier choice is carried over)
        cfg["open_browser"] = a.browser
    # #465: requests at once - written only when given (else an earlier "parallel" is carried over); a recommendation
    streaming = "--kv-resident" in args
    if a.parallel is not None:
        if a.parallel >= 2:
            cfg["parallel"] = a.parallel
            for i, line in enumerate(parallel_note(a.parallel, [g.get("vram_gb", 0.0) for g in chosen],
                                                   MODELS[model]["arena_gb"], ctx, kv, streaming)):
                (ok if i == 0 else warn)(line)
        else:
            cfg["parallel"] = 1
            ok("parallel requests: one at a time (--parallel 1)")
    if vision != "none":
        old_cfg = ROOT / f"strata-{tag.lower()}.json"
        vt = vision_tokens(a.vision_tokens, vision, old_cfg if old_cfg.is_file() else adopted)
        cfg["vision"] = {"exe": str(eng / VEXE), "mmproj": str(mmproj), "model": str(shards[0]),
                         "gpu": vision == "gpu", "max_tokens": vt}
        if vision == "cpu":
            cfg["vision"]["threads"] = max(1, (os.cpu_count() or 8) // 2)
    elif a.vision_tokens is not None:
        warn("--vision-tokens: images are off for this model, so it is not used")
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    cal = setup_calibration(cfg, hip)                  # #566: Linux HIP too; the tuning is offered on NVIDIA only
    if cal is not None:
        sys.path.insert(0, str(ROOT / "tools"))
        import calibrate as CAL
        cfg["args"] = CAL.apply(cfg["args"], cal.get("settings") or {})
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else ""))
    else:                                              # #642: measured counts (a calibration) win over the rule
        cfg["args"] = recommend_pool_workers(cfg["args"])
    write_setup_config(cfg_path, cfg, adopted if adopted is not None and adopted.name == cfg_path.name else None)
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    # offered only when someone answers: --yes installs and adopted earlier installs are not held up by it
    if cal is None and not hip and not a.no_start and not a.yes and ask(
            "Tune Strata for this PC now? It measures a few engine settings (about 5-10 minutes; the PC is busy "
            "meanwhile; later: START-HERE --calibrate)", ["y", "n"], "y", a.yes) == "y":
        tuned = calibrate_config(cfg_path)
    else:
        tuned = None                                   # not asked for: nothing to repeat below
    ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    if a.host and a.host not in ("127.0.0.1", "localhost"):
        say(f"  Other devices:    the server window prints this PC's address (http://<IP>:{port}/)"
            + ("" if a.api_key else " - no API key set: anyone on your network can use it"))
    say(f"  Next time:        just run {'START-HERE.bat' if WIN else './setup.sh'} (or {script.name}) - it starts right away")
    if vision != "none":
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if a.parallel is None:                             # #465: the opt-in, said once (nothing changes)
        for line in parallel_note(None, [g.get("vram_gb", 0.0) for g in chosen], MODELS[model]["arena_gb"], ctx, kv,
                                  "--kv-resident" in cfg["args"]):
            say("  " + line)
    if tuned is False:                                 # #447: a failed tuning is repeated here, not only above
        say("  Tuning:           FAILED (the reason is above): the default settings stay - "
            f"{'START-HERE.bat' if WIN else './setup.sh'} --calibrate tries again")
    if a.no_start:
        return 0
    return start(cfg_path, port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
