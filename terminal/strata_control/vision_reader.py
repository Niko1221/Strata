"""Which picture reader an engine folder gets: the GPU one (`strata-vision-hip` on AMD, `strata-vision-sycl` on Intel Arc, `strata-vision --gpu`
on a Mac) or the CPU one (`strata-vision`).

An AMD engine package carries two programs for pictures.  `strata-vision` is the CPU reader: it needs nothing from the GPU
runtime, so it starts on every PC.  `strata-vision-hip` is the same reader on the card (llama.cpp's HIP backend), a
fraction of a second per picture instead of seconds.  Setup writes the run config with the GPU reader first and the CPU
reader as its fallback (`vision.cpu_exe`); the server (serve/server.py, Vision) starts the CPU reader by itself, with one
log line, when the GPU reader does not start: the HIP runtime does not load, the card is not one the build has code for
(the reader says so before it runs a kernel), or the card is busy.  Nothing is asked of the user.

A Mac has one program for both: `strata-vision` is built with Metal, `--gpu` reads on the Mac's GPU and without it on the CPU, so the
CPU reader is the same file and the fallback needs no second program.  Unified memory changes the 24 GB rule: nothing is taken from a
separate card, the reader's ~1.1 GB is the mmproj file itself on either side (measured on an M4, footprint 1,050 MB on the CPU, 1,179 MB
on the GPU), and the CPU reader takes ~50 s for a 260-token picture where the GPU one takes 0.45 s (ggml has no BF16 vector kernel on
ARM).  So a Mac reads on the GPU unless the person asks for the CPU.

An Intel Arc engine folder (engine-sycl/) carries the CPU reader; `strata-vision-sycl` (llama.cpp's SYCL backend, tools/vision with
STRATA_VISION_SYCL, built by `setup --backend sycl --vision gpu` on a PC that has the oneAPI compiler) is the GPU reader there.

`choose` is the whole decision; setup.py calls it, and tools/test_vision_reader.py checks every branch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CPU_READER = "strata-vision"
GPU_READER = "strata-vision-hip"
SYCL_READER = "strata-vision-sycl"
GPU_READERS = (GPU_READER, SYCL_READER)       # the GPU reader of an engine folder: the first one it carries

#: Owner decision 2026-10-18: pictures are ON by default, and the reader runs on the CPU for a card under 24 GB (the GPU reader keeps about
#: 1-1.4 GB of VRAM that the experts would use: a few % of decode, more on some PCs, #2406) and on the GPU for 24 GB and above - on every make
#: of card.  A person can switch it per model (--vision gpu / --vision cpu, the Pictures step); what was asked for is kept.
#: Owner rule 2026-11-01: the reader lives in RAM, not VRAM.  The GPU reader of an NVIDIA card with one GPU is therefore started by the first picture and
#: stopped when idle (serve/server.py apply_vision_defaults), with the CPU reader (the same program without --gpu) when the card has no room.
GPU_MIN_VRAM_GB = 23.5

#: The one exception to that rule.  An Intel Arc of 24 GB or more keeps the CPU reader by default: the SYCL GPU reader works (tested on an A750,
#: forced with --vision gpu: 0.16 s a picture against 1.7 s on the CPU, embeddings cosine >= 0.995 to the CPU reader's) but it needs the oneAPI
#: runtime on the PC and has not run on a 24 GB+ card (an Arc Pro B70 is allowed since 2026-10-22; the CPU reader is what it gets).  --vision gpu asks for it.  Set True when a 24 GB+ Arc has run it.
INTEL_GPU_READER_BY_DEFAULT = False


def default_mode(vram_gb: float | None, make: str = "", *, unified: bool = False) -> str:
    """"cpu" or "gpu": the reader an unspecific "pictures on" gets on a card with this much memory (unknown memory: "gpu").  `make` is "intel"
    for an Arc, which stays on the CPU reader until INTEL_GPU_READER_BY_DEFAULT.  unified: a Mac, where the chip's memory is the system's -
    always "gpu" (see the module doc)."""
    if unified:
        return "gpu"
    vram = float(vram_gb or 0.0)
    if make == "intel" and not INTEL_GPU_READER_BY_DEFAULT:
        return "cpu"
    return "cpu" if 0 < vram < GPU_MIN_VRAM_GB else "gpu"


def wanted(asked: str | None, vram_gb: float | None, make: str = "", *, unified: bool = False) -> str:
    """What is wanted for `asked` (None, "yes", "on", "gpu", "cpu", "no"/"none"/"off"): "gpu", "cpu" or "none".  Nothing asked and "yes" mean
    the default for this card; "gpu" and "cpu" are kept as asked."""
    a = (asked or "").lower()
    if a in ("no", "none", "off"):
        return "none"
    if a in ("gpu", "cpu"):
        return a
    return default_mode(vram_gb, make, unified=unified)

#: Inkling-2-Mini (owner 2026-11-26, "Should Inkling follow the Qwen rule for pictures everywhere?" - "yes"): the same rule, but its reader is built into
#: strata-inkling (no separate helper program), so the engine itself applies it on the card's real memory (engine/src/inkling2/ink_vision_policy.h: the same
#: 23.5 GiB, the same Intel exception).  Control resolves the default the same way where it knows the card and writes the result as the engine's
#: `--vision-reader gpu|cpu`, so the server (picture size cap, silence allowance) and the engine agree; where it cannot tell the card it leaves the flag out and the
#: engine's own "auto" decides.  tools/test_vision_reader.py reads ink_vision_policy.h and keeps the numbers the same.
INKLING_READER_FLAG = "--vision-reader"
INKLING_IDLE_FLAG = "--vision-idle-s"
#: Seconds without a picture before the NVIDIA (one GPU) reader gives its memory back; the same number as the Qwen reader's "elastic_idle_s" default and
#: the engine's kVisionIdleSecondsDefault.
ELASTIC_IDLE_S_DEFAULT = 120
#: What the Inkling GPU reader holds while it works (its weights, 0.64 GB, and the buffers of a 1,024-token picture): the engine keeps this much out of the experts' budget.
INKLING_READER_GB = 1.1

GFX = re.compile(r"gfx[0-9a-f]+", re.IGNORECASE)


def reader_name(base: str, windows: bool) -> str:
    return base + (".exe" if windows else "")


def find(folder: Path | str, windows: bool, *, mac: bool = False) -> dict:
    """The readers in an engine folder: {"cpu": Path | None, "gpu": Path | None}.  On a Mac (mac=True) `strata-vision` is both: it has
    Metal built in and `--gpu` picks the GPU."""
    folder = Path(folder)
    found = {}
    for key, bases in (("cpu", (CPU_READER,)), ("gpu", GPU_READERS)):
        paths = [folder / reader_name(base, windows) for base in bases]
        found[key] = next((p for p in paths if p.is_file()), None)
    if mac:
        found["gpu"] = found["cpu"]
    return found


def gfx_of(text: object) -> str:
    """"gfx1201" out of a gcnArchName ("gfx1201:sramecc-:xnack-"), a device line or a plain arch; "" when there is none."""
    m = GFX.search(str(text or ""))
    return m.group(0).lower() if m else ""


@dataclass
class Reader:
    mode: str                     # "gpu" | "cpu" | "none"
    exe: Path | None = None       # the program the run config starts
    cpu_exe: Path | None = None   # the CPU reader to fall back on when a GPU reader does not start (mode "gpu" only)
    note: str = ""                # one plain sentence for the setup output when it is not what was asked for


MAC_CPU_NOTE = ("the picture reader on the CPU takes about 50 seconds for one picture on a Mac (the GPU reader: under a second): kept as "
                "you chose; --vision gpu reads on the Mac's GPU")


def choose(folder: Path | str, asked: str, *, windows: bool, gpu_arch: str = "", archs: list | None = None, mac: bool = False) -> Reader:
    """The reader for `asked` ("yes"/"on"/"gpu" = the best the folder has, "cpu", "no"/"none"/"off").

    gpu_arch: the card's architecture when setup knows it ("gfx1201"); archs: what the GPU reader was built for
    (BUILD.json `vision_archs`, else the package's `archs`).  A card the build does not list gets the CPU reader here;
    when the card is unknown the GPU reader is tried and falls back by itself (see the module doc).
    mac: the Mac's one program, `strata-vision`: "gpu" runs it with --gpu and keeps the same file as the CPU reader to fall back on."""
    asked = (asked or "").lower()
    if asked in ("no", "none", "off", ""):
        return Reader("none")
    have = find(folder, windows, mac=mac)
    if mac:
        if have["cpu"] is None:
            return Reader("none", note="this engine has no picture reader: pictures off")
        if asked == "cpu":
            return Reader("cpu", have["cpu"], note=MAC_CPU_NOTE)
        return Reader("gpu", have["gpu"], have["cpu"])
    if asked == "cpu":
        if have["cpu"]:
            return Reader("cpu", have["cpu"])
        return Reader("none", note="this engine has no picture reader: pictures off")
    # yes / on / gpu
    gfx = gfx_of(gpu_arch)
    built = [gfx_of(a) for a in (archs or []) if gfx_of(a)]
    if have["gpu"] is None:
        if have["cpu"]:
            return Reader("cpu", have["cpu"], note="this engine has no GPU picture reader: pictures are read on the CPU "
                                                    "(a few seconds each)" if asked == "gpu" else "")
        return Reader("none", note="this engine has no picture reader: pictures off")
    if gfx and built and gfx not in built:
        if have["cpu"]:
            return Reader("cpu", have["cpu"], note=f"the GPU picture reader has no code for your card ({gfx}): pictures "
                                                    "are read on the CPU (a few seconds each)")
        return Reader("none", note=f"the GPU picture reader has no code for your card ({gfx}): pictures off")
    return Reader("gpu", have["gpu"], have["cpu"])


def vision_section(reader: Reader, base: dict) -> dict:
    """The run config's "vision" section for `reader` on top of `base` (mmproj, model, max_tokens ...): exe, gpu, and for the
    GPU reader the CPU reader to fall back on."""
    out = dict(base)
    out["exe"] = str(reader.exe)
    out["gpu"] = reader.mode == "gpu"
    if reader.mode == "gpu" and reader.cpu_exe:
        out["cpu_exe"] = str(reader.cpu_exe)
    else:
        out.pop("cpu_exe", None)
    return out


def shared_memory_nvidia_host() -> bool:
    """Linux on Arm: the DGX Spark class (GB10: the GPU shares the system RAM, 121 GB), where the NVIDIA picture reader stays loaded instead of starting with the first
    picture (owner 2026-11-11).  Measured on a Spark (IQ3_S, two starts per arm): the first picture after a start takes 0.84-0.89 s resident against 3.0-7.1 s on demand
    (the reader's start, and its context retry when the kernel has no free 2 MiB blocks), and again 2.2 s after every idle stop; the resident reader costs about 5 s of start
    time and about 1 GiB of the shared RAM, and no expert slot (every expert is resident there).  The same switch as the AMD reader and the Mac's: the reader starts with the
    server, before the engine; an explicit "elastic" key in the vision section wins either way."""
    import platform
    import sys
    return sys.platform.startswith("linux") and platform.machine().lower() in ("aarch64", "arm64")


#: A GPU reader's start that failed while it made its CUDA context on a Linux Arm machine with shared CPU / GPU memory (DGX Spark, GB10) is tried again.  Such a context needs
#: physically contiguous memory (2 MiB blocks); when the kernel has almost none free, ggml's cudaSetDevice answers "CUDA error: out of memory ... in function
#: ggml_cuda_init" and the reader aborts, although 20+ GiB are free.  The same start 1 to 5 s later works (the kernel compacts memory after the failed try; measured
#: on a Spark: three to five failed spawns in a row, then success: first failure to first success 9.6 to 14.6 s; one failed spawn takes 1.4 s, the pause is 1 s).  Without the
#: retry the picture reader falls back to the CPU for good (90 s for a large picture, 9.7 s against 2.0 s at 1,024 tokens on the card).
#: Bounded by time and by count: a new spawn may start only while less than CONTEXT_RETRY_BUDGET_S seconds have passed since the first spawn started (about 8 spawns at
#: 2.4 s a cycle; the five measured failures end at 12 s), and at most CONTEXT_RETRIES retries (fast attempts).  The worst wait before the CPU reader takes over is the
#: budget plus the last failed spawn, about 19.5 to 20 s.
CONTEXT_RETRIES = 12
CONTEXT_RETRY_WAIT_S = 1.0
CONTEXT_RETRY_BUDGET_S = 18.0


def retry_gpu_start(*, linux: bool, arm: bool, hip: bool, text: str, retries_done: int, next_start_s: float = 0.0) -> bool:
    """Should a failed GPU reader start be tried again?  Only for the context failure ("out of memory" in ggml_cuda_init / cudaSetDevice), only for the NVIDIA reader on
    Linux on Arm, at most CONTEXT_RETRIES times, and only if the next spawn would start less than CONTEXT_RETRY_BUDGET_S seconds after the first one (`next_start_s`: that
    time, the pause included).  A card another program filled says "cudaMalloc failed: out of memory" (a retry cannot help: reader_room reports it), every
    other error (no device, no kernel image, busy) is not this case.  `text`: what the failed start said (its message and the tail of the reader's log)."""
    if not linux or not arm or hip or retries_done >= CONTEXT_RETRIES or next_start_s >= CONTEXT_RETRY_BUDGET_S:
        return False
    t = text.lower()
    if "out of memory" not in t or "cudamalloc failed" in t or "failed to allocate" in t:
        return False
    return "ggml_cuda_init" in t or "cudasetdevice" in t or "cannot use gpu" in t


def cpu_fallback(vision_cfg: dict, same_program: bool = False) -> dict | None:
    """The same section for the CPU reader, when the GPU reader has one to fall back on; None otherwise.  AMD and Intel name it (`cpu_exe`);
    the NVIDIA reader (`strata-vision`) is the CPU reader too, without --gpu - `same_program` takes that one (the on-demand reader's
    start, which must read the picture whatever the card has left)."""
    if not vision_cfg.get("gpu"):
        return None
    cpu = vision_cfg.get("cpu_exe")
    if not cpu and same_program:
        # the NVIDIA reader and the Mac's are one program: without --gpu it reads on the CPU (same picture tokens, so the same answers)
        exe = str(vision_cfg.get("exe") or "")
        if Path(exe).name.lower() not in (CPU_READER, CPU_READER + ".exe"):
            return None
        cpu = exe
    if not cpu:
        return None
    out = {k: v for k, v in vision_cfg.items() if k not in ("cpu_exe", "elastic", "elastic_mib", "elastic_idle_s")}
    out["exe"], out["gpu"] = cpu, False
    return out


# ------------------------------------------------------------------------------------------------------------------- Inkling-2-Mini

def inkling_reader(asked: str | None, vram_gb: float | None, make: str = "", *, mac: bool = False) -> str | None:
    """The picture reader of an Inkling-2-Mini install for `asked` (None, "yes", "on", "gpu", "cpu", "no"/"none"/"off"): "gpu", "cpu", "none", or None when
    the card cannot be told (the flag is then left out and the engine's own rule decides).  The same rule as the Qwen readers (default_mode):
      * "no" is "none" (the Vision file is left out); "gpu" and "cpu" are kept as asked;
      * a Mac (mac=True): the GPU, whatever the memory;
      * an Intel Arc (make "intel"): the CPU reader at any memory size (INTEL_GPU_READER_BY_DEFAULT), the GPU one when asked;
      * NVIDIA and AMD: the CPU under 23.5 GiB (GPU_MIN_VRAM_GB), the GPU from there; the memory is what the card reports, and on a machine whose GPU shares
        the RAM (Strix Halo, DGX Spark) the shared pool it reports, so those read on the GPU, as the Qwen path does (setup passes the same gpu["vram_gb"])."""
    a = (asked or "").lower()
    if a in ("no", "none", "off"):
        return "none"
    if a in ("gpu", "cpu"):
        return a
    if mac:
        return "gpu"
    if make != "intel" and not float(vram_gb or 0.0) > 0.0:
        return None
    return default_mode(vram_gb, make)


def inkling_lines(asked: str | None, mode: str | None, vram_gb: float | None, make: str = "", *, mac: bool = False, backend: str = "") -> list[tuple[bool, str]]:
    """The plain-words screen lines for an Inkling picture reader: [(is_warning, text)].  Same wording style as the Qwen lines of setup.py
    (own_path_vision, flash_vision_encoder): which reader runs and why, what it costs, how to switch.  `mode` is what inkling_reader gave."""
    a = (asked or "").lower()
    vram = float(vram_gb or 0.0)
    small = 0 < vram < GPU_MIN_VRAM_GB
    if mode == "none":
        return [(False, "images: off (no picture reader is installed)")]
    if mode is None:
        return [(False, "images: on; the engine picks the picture reader by the card's memory (the CPU under 24 GB, the GPU from 24 GB; --vision gpu or --vision cpu sets it)")]
    cost = (f"about {INKLING_READER_GB:.1f} GB of VRAM while it reads"
            + (f", put on the card with the first picture and taken off after {ELASTIC_IDLE_S_DEFAULT // 60} minutes without one" if backend == "cuda" else ""))
    if mac:
        lines = [(False, "images: Inkling-2-Mini reads pictures on the Mac's GPU (the memory is shared, a fraction of a second per picture)")]
        if mode == "cpu":
            lines = [(True, MAC_CPU_NOTE)]
        return lines
    if mode == "gpu":
        if a == "gpu" and small:
            return [(True, f"the GPU picture reader keeps ~{INKLING_READER_GB:.1f} GB of VRAM that a {vram:.0f} GB card would use for experts (a few % of decode): "
                           "kept as you chose; --vision cpu reads pictures on the CPU instead (a few seconds each)")]
        if a == "gpu" and make == "intel":
            return [(False, "images: Inkling-2-Mini reads pictures on the Intel Arc card, as you asked")]
        return [(False, f"images: Inkling-2-Mini reads pictures on the GPU ({cost}; a fraction of a second per picture)")]
    # cpu
    if a == "cpu":
        return [(False, "images: Inkling-2-Mini reads pictures on the CPU, as you asked (a few seconds each); the card keeps all its memory for the model")]
    if make == "intel":
        return [(False, "images: Inkling-2-Mini reads pictures on the CPU (a few seconds each) on an Intel Arc card; --vision gpu reads them on the card")]
    return [(False, f"images: Inkling-2-Mini reads pictures on the CPU (a few seconds each), because the GPU reader would take ~{INKLING_READER_GB:.1f} GB of VRAM "
                    f"from the experts on this {vram:.0f} GB card (--vision gpu to use it anyway)")]


def reader_arg(args) -> str | None:
    """The value of `--vision-reader` in an engine argument list ("gpu", "cpu" or "auto"); None when it is not there."""
    items = [str(a) for a in (args or ())]
    for i, a in enumerate(items[:-1]):
        if a == INKLING_READER_FLAG:
            return items[i + 1].strip().lower()
    return None


def set_reader_arg(args: list, mode: str | None) -> list:
    """`args` with `--vision-reader <mode>` set (replacing an earlier one) or, for mode None / "none", taken out.  A new list."""
    out: list = []
    items = list(args or ())
    i = 0
    while i < len(items):
        if str(items[i]) == INKLING_READER_FLAG:
            i += 2 if i + 1 < len(items) else 1
            continue
        out.append(items[i])
        i += 1
    if mode in ("gpu", "cpu"):
        out += [INKLING_READER_FLAG, mode]
    return out
