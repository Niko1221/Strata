"""setup.py for an Intel Arc: upstream's installer, steered onto the SYCL port from outside.

    ./setup.sh --backend sycl [setup.py's options]       (Linux: this file, run through setup.sh's virtual environment)
    START-HERE.bat --backend sycl [setup.py's options]   (Windows: the same, through START-HERE.bat's .venv)

The SYCL port keeps out of the shared files (upstream merges stay clean), so this does not edit setup.py: it imports
it and replaces the few steps that are NVIDIA/AMD-specific, then runs setup's own main().  Everything else - the
model choice, the download, the pack, the tokenizer, the MTP draft layer, the context and KV questions - is
setup.py's, unchanged.  What is replaced:

  - the GPU check: the Intel Arc is offered through setup's AMD path (the one that compiles locally and has no
    images), named and sized from sysfs (Linux) or from Windows' display adapters and, once the engine is built,
    from `engine\\strata-device.exe --list-devices`, which numbers the cards as SYCL does;
  - the engine step: the SYCL build instead of a CUDA/HIP build.  Linux: build-sycl-aot/strata, run in the
    strata-sycl-dev image by sycl/serve/strata-sycl.sh (docs/INTEL.md).  Windows: engine\\strata.exe with the oneAPI
    runtime DLLs beside it, built by sycl\\tools\\build_windows.bat (docs/INTEL_WINDOWS.md); setup never downloads an
    engine and never runs Docker there;
  - the RAM rule: the CUDA engine keeps every expert in RAM, the SYCL port streams them from the GGUF into VRAM
    (--stream-experts), so RAM only bounds the KV streaming;
  - the config: the SYCL flags, backend "sycl", the container's paths (Linux) or native paths and the card's SYCL
    index (Windows); the run script starts sycl/serve/server_intel.py (serve/server.py plus the Intel Monitor
    readings, the engine's environment on Windows and the model menu).

When setup.py changes the steps this relies on, this stops with a message rather than writing a wrong config.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402

SYCL_WRAPPER = ROOT / "sycl" / "serve" / "strata-sycl.sh"
SYCL_IMAGE = "strata-sycl-dev"
SERVER = ROOT / "sycl" / "serve" / "server_intel.py"
MOUNT = Path(os.environ.get("STRATA_SYCL_ROOT") or ROOT.parent)   # what strata-sycl.sh mounts at /work
# Windows: the native build (sycl\tools\build_windows.bat) - strata.exe, strata-device.exe, the probes and the oneAPI
# runtime DLLs beside them, with a BUILD.json whose backend is "sycl"
ENGINE_DIR = ROOT / "engine"
ENGINE_EXE = ENGINE_DIR / "strata.exe"
PROBE_EXE = ENGINE_DIR / "strata-device.exe"
BUILD_BAT = ROOT / "sycl" / "tools" / "build_windows.bat"
DOC = "docs/INTEL_WINDOWS.md" if S.WIN else "docs/INTEL.md"
DRIVER_HINT = ("install the Intel Arc graphics driver 32.0.101.9034 or newer (it carries the Level Zero loader 1.32 "
               "the engine needs) from https://www.intel.com/content/www/us/en/download-center/home.html, restart "
               f"the PC and run this again; {PROBE_EXE} --list-devices shows what the SYCL runtime sees")
# the setup.py entry points the Windows path replaces or calls beyond the Linux list (install() checks them)
WIN_NEEDED = ("amd_gpus_win", "hip_devices", "hip_lib_dirs", "hip_runtime_beside_exe", "hip_card", "get_prebuilt_hip",
              "update_installed_engine", "_win_display_adapters", "_win_display_registry", "source_hash",
              "upgrade_config", "write_config", "refresh_draft_vocab", "settings_summary", "ok", "warn")

# Battlemage / Alchemist PCI device ids -> (name, VRAM GB). lspci's database lags new cards (an Arc Pro B70 reads
# "Intel Corporation Device [8086:e223]"), so the sysfs id is the reliable signal and the name table is ours.
INTEL_ARC = {"e223": ("Arc Pro B70", 32.0), "e221": ("Arc Pro B60", 24.0), "e211": ("Arc Pro B60", 24.0),
             "e20b": ("Arc B580", 12.0),
             "e20c": ("Arc B570", 10.0), "e212": ("Arc B50", 16.0),
             "56a0": ("Arc A770", 16.0), "56a1": ("Arc A750", 8.0), "56a2": ("Arc A580", 8.0),
             "56a5": ("Arc A380", 6.0), "56a6": ("Arc A310", 4.0), "5690": ("Arc A770M", 16.0)}

# Windows: PCI device ids (VEN_8086) of the discrete Arc cards -> (name, VRAM GB or 0 = the registry's figure, the SYCL
# architecture as strata-device --list-devices prints it).  BMG-G31 is E220-E223 (the Arc Pro B70 is E223, the Pro B65
# E222), BMG-G21 the other Battlemage ids, ACM the Alchemist A-series.  Any other Intel display adapter is the
# processor's graphics: listed, not supported.
_WIN_INTEL_DID = {0xE223: ("Arc Pro B70", 32.0, "intel_gpu_bmg_g31"), 0xE222: ("Arc Pro B65", 0.0, "intel_gpu_bmg_g31"),
                  0xE220: ("Arc Pro (BMG-G31)", 0.0, "intel_gpu_bmg_g31"), 0xE221: ("Arc Pro (BMG-G31)", 0.0, "intel_gpu_bmg_g31"),
                  0xE20B: ("Arc B580", 12.0, "intel_gpu_bmg_g21"), 0xE20C: ("Arc B570", 10.0, "intel_gpu_bmg_g21"),
                  0xE211: ("Arc Pro B60", 24.0, "intel_gpu_bmg_g21"), 0xE212: ("Arc Pro B50", 16.0, "intel_gpu_bmg_g21"),
                  0xE202: ("Arc (BMG-G21)", 0.0, "intel_gpu_bmg_g21"), 0xE209: ("Arc (BMG-G21)", 0.0, "intel_gpu_bmg_g21"),
                  0xE20D: ("Arc (BMG-G21)", 0.0, "intel_gpu_bmg_g21"), 0xE210: ("Arc (BMG-G21)", 0.0, "intel_gpu_bmg_g21"),
                  0xE216: ("Arc (BMG-G21)", 0.0, "intel_gpu_bmg_g21"),
                  0x56A0: ("Arc A770", 16.0, "intel_gpu_acm_g10"), 0x56A1: ("Arc A750", 8.0, "intel_gpu_acm_g10"),
                  0x56A2: ("Arc A580", 8.0, "intel_gpu_acm_g10"), 0x5690: ("Arc A770M", 16.0, "intel_gpu_acm_g10"),
                  0x56A5: ("Arc A380", 6.0, "intel_gpu_acm_g11"), 0x56A6: ("Arc A310", 4.0, "intel_gpu_acm_g11")}
# the SYCL architectures of the processors' own GPUs (sycl_ext_oneapi_device_architecture's intel_gpu_* names): a
# device the runtime lists under one of these is integrated graphics, which the port does not run on
_WIN_INTEGRATED_ARCH = ("intel_gpu_mtl", "intel_gpu_arl", "intel_gpu_lnl", "intel_gpu_ptl", "intel_gpu_tgl",
                        "intel_gpu_adl", "intel_gpu_rpl", "intel_gpu_rkl", "intel_gpu_ehl", "intel_gpu_icl",
                        "intel_gpu_skl", "intel_gpu_kbl", "intel_gpu_cfl", "intel_gpu_whl", "intel_gpu_aml",
                        "intel_gpu_cml", "intel_gpu_bdw", "intel_gpu_apl", "intel_gpu_glk")


def intel_gpus():
    """Intel discrete GPUs from sysfs: vendor 0x8086 under the xe or i915 driver, named by PCI device id. VRAM comes
    from the id table, else from the size of the card's VRAM BAR."""
    found = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:                            # connectors (card0-DP-1) share the card's device
            continue
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
            devid = (dev / "device").read_text().strip().lower().replace("0x", "")
            driver = os.path.basename(os.path.realpath(dev / "driver")) if (dev / "driver").exists() else ""
        except OSError:
            continue
        if vendor != "0x8086" or driver not in ("xe", "i915"):
            continue
        name, vram = INTEL_ARC.get(devid, (None, 0.0))
        if name is None:
            if driver != "xe":                          # i915 without a known Arc id is an integrated GPU
                continue
            name = f"Intel GPU {devid} (xe)"
        if not vram:                                    # resource line 2 (BAR 2) is the VRAM aperture on xe cards
            try:
                start, end, _ = (dev / "resource").read_text().splitlines()[2].split()
                vram = (int(end, 16) - int(start, 16) + 1) / 2**30
            except (OSError, IndexError, ValueError):
                vram = 0.0
        found.append({"index": len(found), "name": f"Intel {name}", "vram_gb": vram, "arch": driver, "driver": driver})
    return found


# ------------------------------------------------------------------------------------------------ Windows
def _intel_pci_device_id(text) -> int | None:
    """The PCI device id of an Intel (VEN_8086) adapter from its PNP id, else None (setup's _pci_device_id is AMD's)."""
    m = re.search(r"VEN_8086&DEV_([0-9A-F]{4})", str(text or ""), re.I)
    return int(m.group(1), 16) if m else None


def intel_gpus_windows(adapters=None, registry=None) -> list[dict]:
    """Windows: the Intel display adapters present, in display-adapter order (setup's numbering until the SYCL engine
    answers), from Win32_VideoController and the display-class registry - setup.py's own readers; the registry has
    the 64-bit VRAM size, WMI's stops at 4 GB.  Named from the PCI id table (the WMI name when it has one); an id the
    table does not know is the processor's graphics: listed, not supported.  adapters / registry:
    tools/test_setup_sycl_windows.py passes mocked ones."""
    adapters = S._win_display_adapters() if adapters is None else adapters
    registry = S._win_display_registry() if registry is None else registry
    if not adapters:                                   # no WMI answer: the registry alone (it can list removed cards)
        adapters = [{"name": r.get("DriverDesc", ""), "pnp": r.get("MatchingDeviceId", ""), "ram": 0} for r in registry]
    used = set()
    found = []
    for ad in adapters:
        did = _intel_pci_device_id(ad.get("pnp"))
        if did is None:
            continue                                   # not an Intel (VEN_8086) PCI device
        vram, driver = 0.0, ""
        for k, r in enumerate(registry):               # the same card's driver instance: its 64-bit VRAM size
            if k in used or _intel_pci_device_id(r.get("MatchingDeviceId")) != did:
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
        table, size, arch = _WIN_INTEL_DID.get(did, (None, 0.0, ""))
        if vram == 0.0:
            vram = size                                # the card's known size, before WMI's 32-bit figure
        if vram == 0.0 and ad.get("ram", 0) > 0:
            vram = ad["ram"] / 2 ** 30                 # WMI's 32-bit figure (at most 4 GB)
        name = ad.get("name") or (f"Intel {table}" if table else f"Intel GPU (device {did:04X})")
        found.append({"index": len(found), "name": name, "vram_gb": vram, "arch": arch or f"unknown (PCI {did:04X})",
                      "driver": driver or "intel", "vendor": "intel", "integrated": table is None})
    return found


def sycl_devices(probe: Path | None = None, text: str | None = None) -> list[dict] | None:
    """The Level Zero devices the SYCL runtime enumerates, numbered as SYCL numbers them (the config's "gpu",
    ONEAPI_DEVICE_SELECTOR=level_zero:N), from the installed engine's `strata-device --list-devices`: per device
    "device N: <name>", then "  arch <architecture>, <x.y> GiB", and "  cannot run: <reason>" for one the engine's own
    checks refuse.  None when there is no SYCL engine here or the probe does not answer.  text: its output (tests)."""
    if text is None:
        probe = probe or PROBE_EXE
        try:
            meta = json.loads((probe.parent / "BUILD.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            meta = {}
        if not probe.exists() or meta.get("backend") != "sycl":
            return None
        env = dict(os.environ)
        env.pop("ONEAPI_DEVICE_SELECTOR", None)         # every Level Zero device, in SYCL's own order
        env["SYCL_UR_USE_LEVEL_ZERO_V2"] = "1"           # the adapter the engine runs on (docs/INTEL_WINDOWS.md)
        env["SYCL_CACHE_PERSISTENT"] = "0"
        env["PATH"] = os.pathsep.join([str(probe.parent), env.get("PATH", "")])   # the oneAPI DLLs beside it
        try:
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
                          "driver": "sycl", "vendor": "intel"})
            continue
        if not found:
            continue
        a = re.match(r"arch\s+([A-Za-z0-9_]+)\s*,\s*([\d.]+)\s*GiB", line.strip())
        if a and not found[-1]["arch"]:
            found[-1]["arch"], found[-1]["vram_gb"] = a.group(1), float(a.group(2))
        elif line.strip().startswith("cannot run:"):
            found[-1]["cannot_run"] = line.strip()[len("cannot run:"):].strip()
    for g in found:
        if not g["arch"]:
            g["arch"] = "unknown"
        g["integrated"] = g["arch"].startswith(_WIN_INTEGRATED_ARCH)
    return found


def intel_problem(g: dict) -> str | None:
    """Why setup cannot use a card, or None: the engine's own refusal (the probe's "cannot run": no 32-wide sub-groups,
    no fp16, another die than the one the code was compiled for, ...), else the processor's graphics or a device the
    runtime does not name.  Any discrete Arc the engine's checks pass is offered; the Arc Pro B70 is the tested one."""
    if g.get("cannot_run"):
        return g["cannot_run"]
    arch = str(g.get("arch") or "")
    if g.get("integrated") or not arch.startswith("intel_gpu_"):
        what = "the processor's graphics" if g.get("integrated") else f"a GPU the SYCL runtime does not name ({arch or 'unknown'})"
        return f"not supported: {what}; the SYCL engine runs on a discrete Arc (tested: the Arc Pro B70, {DOC})"
    return None


def intel_card(gpu: dict, listed: list[dict]) -> dict:
    """Windows: setup's chosen Intel card, checked before the model download (setup's hip_card step): one the engine
    runs, numbered as SYCL numbers it when the probe answered."""
    p = intel_problem(gpu)
    if p:
        S.fail(f"GPU {gpu['index']} ({gpu['name']}) cannot be used: {p}")
    if gpu.get("driver") != "sycl":
        S.warn(f"the card's number ({gpu['index']}) is Windows' display-adapter order, not confirmed by "
               f"{PROBE_EXE.name} --list-devices: if the engine starts on the wrong card, set \"gpu\" in the config")
    return {**gpu, "count": len(listed)}


def card_driver(g: dict) -> str:
    """The driver class the config rules below go by: "xe" or "i915" on Linux (the kernel driver, from sysfs).  On
    Windows the equivalent from the SYCL architecture: an Alchemist (acm_*) behaves as an i915 card does on Linux
    (no big pinned host allocation, FP64 emulated), every other Arc as an xe card."""
    d = str(g.get("driver") or "")
    if d in ("xe", "i915"):
        return d
    arch = str(g.get("arch") or "")
    return "i915" if arch.startswith(("intel_gpu_acm", "intel_gpu_dg1")) else "xe"


# ------------------------------------------------------------------------------------------------ the engine
def sycl_engine(strict=True):
    """The SYCL build: (binary, why-not).  Windows: engine\\strata.exe from sycl\\tools\\build_windows.bat, whose
    BUILD.json says backend "sycl" and names the sources it was built from ("src": setup's source_hash over the engine
    sources plus sycl/, sycl_sources(), as upstream records its local builds): a git pull or an edit under sycl/
    makes it stale.
    strict: a stale engine is refused (a setup); else it is used with a warning (a start of an installed model, as
    upstream starts the installed engine when it cannot rebuild)."""
    if S.WIN:
        bat = BUILD_BAT.relative_to(ROOT)
        if not ENGINE_EXE.exists():
            return None, f"it is not built ({bat}; {DOC})"
        try:
            meta = json.loads((ENGINE_DIR / "BUILD.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            meta = {}
        if meta.get("backend") != "sycl":
            return None, (f"{ENGINE_DIR} holds a {meta.get('backend') or 'different'} engine, not the SYCL one "
                          f"({bat} builds it; {DOC})")
        if meta.get("src") != S.source_hash(sycl_sources()):
            why = (f"the engine in {ENGINE_DIR} was built from other sources than this tree (a git pull, or an edit "
                   f"under sycl/): run {bat} again")
            if strict:
                return None, why
            S.warn(why + "; starting the installed one")
        return ENGINE_EXE, None
    exe = next((b for b in (ROOT / "build-sycl-aot" / "strata", ROOT / "build-sycl" / "strata") if b.exists()), None)
    if exe is None:
        return None, "it is not built (docs/INTEL.md, \"How to build it\": Docker, then sycl/tools/build.sh in the image)"
    return exe, None


def sycl_path(path) -> str:
    """A host path as the engine sees it.  Windows: the engine runs natively, so the path is its own.  Linux: the
    folder above the Strata checkout (or STRATA_SYCL_ROOT) is mounted at /work in the engine's container."""
    if S.WIN:
        return str(path)
    p, root = Path(path).resolve(), MOUNT.resolve()
    try:
        return "/work/" + p.relative_to(root).as_posix()
    except ValueError:
        S.fail(f"{p} is outside {root}, which the SYCL engine's container mounts",
               f"keep the models and the data folder under {root} (--models-dir / --data-dir)")


def sycl_version() -> str:
    """The version the SYCL build reports: sycl/CMakeLists.txt's project VERSION, which since 0.1.40.2 is the
    top-level project's (a variable there, so the number is read from ../CMakeLists.txt)."""
    for f in (ROOT / "sycl" / "CMakeLists.txt", ROOT / "CMakeLists.txt"):
        m = re.search(r"project\(\s*\S+\s+VERSION\s+([\d.]+)", f.read_text(encoding="utf-8-sig"))
        if m:
            return m.group(1)
    return "0"


def sycl_sources() -> tuple:
    """The parts setup.source_hash fingerprints for BUILD.json's "src": upstream's ENGINE_SOURCES plus every file under
    sycl/, each as a part of its own, without Python's __pycache__ folders.  The server imports
    sycl/serve/xe_telemetry.py and the tests import the modules here, and the .pyc files Python then writes under
    sycl/ would change a hash taken over the whole folder between the build and the next setup - refusing an engine
    built from exactly this tree.  sycl/tools/package_windows.py writes the hash with this same list."""
    files = sorted(p for p in (ROOT / "sycl").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return tuple(S.ENGINE_SOURCES) + tuple(p.relative_to(ROOT).as_posix() for p in files)


def flag(args, name):
    return args[args.index(name) + 1] if name in args else None


def drop(args, name, value=False):
    while name in args:
        i = args.index(name)
        del args[i:i + 1 + int(value)]


SMALL_CARD_GB = 12.0       # under this the engine's own advice is --vram-reserve-mib 300 (a 1024 MiB reserve leaves an
                           # 8 GB card no room for even one expert slot) and the English draft vocabulary
SMALL_RESERVE_MIB = 300


def small_card(vram_gb: float, driver: str) -> bool:
    return 0 < vram_gb < SMALL_CARD_GB or driver == "i915"


def to_sycl(cfg: dict, exe: Path, ram: float, keep: dict, vram_gb: float = 0.0, driver: str = "xe", gpu=None) -> dict:
    """setup's config (written for its HIP path) -> the SYCL port's: the container's paths, experts streamed from the
    GGUF into VRAM (--stream-experts: the engine reads them from the GGUF and keeps what does not fit in a pinned RAM
    mirror; the VRAM reserve is the smallest that leaves the KV and the prompt buffers room - docs/INTEL.md), KV
    streaming from 64K up when the RAM holds the KV (the B70 at 256K decodes at 40+ tok/s with it, 4-9 without).
    A reserve the user asked for (--vram-reserve-mib) is kept.  An i915 card (Alchemist, the A-series) cannot do that:
    a single pinned host allocation above a few GB fails there, so the mirror cannot hold what the card does not, and
    the config loads the experts into a RAM arena instead (no --stream-experts, --ple-io ram, and the device-built
    verify plan's NO_HOST switch off - docs/INTEL.md, "Arc A750").
    Windows: native paths, engine\\strata.exe with its runtime DLLs beside it ("lib_dirs": ["engine"] goes first on
    the engine's PATH), and the card's SYCL index as "gpu" (`gpu`: the one card setup found, when setup wrote none
    because it was the only one)."""
    args = list(cfg["args"])
    for f in ("--resident-experts", "--mmap-experts"):  # setup's low-RAM mode is the CUDA engine's
        drop(args, f)
    drop(args, "--kv-resident", True)                   # decided below on the real RAM
    for f in ("--pack", "--native", "--ple-gguf", "--expert-profile", "--mtp", "--control-vector-scaled"):
        if f in args:
            v = args[args.index(f) + 1]
            path, _, scale = v.rpartition(":") if f == "--control-vector-scaled" else (v, "", "")
            args[args.index(f) + 1] = sycl_path(path) + (f":{scale}" if scale else "")
    ctx = int(flag(args, "--max-context") or 32768)
    kv = flag(args, "--kv") or "int8"
    kv_ram = ctx * (13 * (576 if kv == "q4_0" else 1056)) / 1e9
    if kv != "k8v4" and ctx >= 65536 and ram >= kv_ram + 6:
        args += ["--kv-resident", "32768"]
    asked_reserve = flag(args, "--vram-reserve-mib")    # setup writes it only when the user gave it
    drop(args, "--vram-reserve-mib", True)
    reserve = asked_reserve or (str(SMALL_RESERVE_MIB) if small_card(vram_gb, driver) else
                                "1024" if ctx <= 32768 else "2048")
    host_arena = driver == "i915"
    if host_arena:
        if "--ple-io" not in args:
            args += ["--ple-io", "ram"]
        args += ["--vram-reserve-mib", reserve]
    else:
        args += ["--stream-experts", "--vram-reserve-mib", reserve]
    if ctx > 32768 or vram_gb >= 24:                    # long contexts: 4096-token chunks keep the prompt buffers small; a
        drop(args, "--prefill", True)                   # 24 GB+ card with part of the experts in the RAM mirror streams those
        args += ["--prefill", "4096"]                   # over PCIe once per chunk, so fewer, bigger chunks read the prompt
                                                        # faster (Arc Pro B70, IQ3_S, a 4,095-token prompt: 2,048-token
                                                        # chunks 618 tok/s, 4,096: 1,002; docs/INTEL.md); it costs ~700 cache slots
    out = {k: v for k, v in cfg.items() if k not in ("lib_dirs", "env", "vision", "gpus")}
    env = {}
    if host_arena:
        env["STRATA_VERIFY_NO_HOST"] = "0"              # strata-sycl.sh / server_intel.py set 1 unless told otherwise (xe)
        # an Alchemist has no FP64 hardware: a kernel that declares double (one is on the sampled path) is refused at its
        # first launch - "'double' is not supported in ... device", the engine dies on the first request with a
        # temperature - unless the driver emulates it (docs/INTEL.md)
        env.update({"IGC_EnableDPEmulation": "1", "OverrideDefaultFP64Settings": "1", "NEOReadDebugKeys": "1"})
    if S.WIN:
        out.update({"backend": "sycl", "exe": str(ENGINE_EXE), "args": args, "lib_dirs": ["engine"]})
        if out.get("gpu") is None and gpu is not None:
            out["gpu"] = gpu                            # the server: ONEAPI_DEVICE_SELECTOR=level_zero:<gpu>
    else:
        out.update({"backend": "sycl", "exe": str(SYCL_WRAPPER), "args": args, "sycl_root": str(MOUNT)})
        if exe != ROOT / "build-sycl-aot" / "strata":
            env["STRATA_SYCL_BIN"] = exe.relative_to(ROOT).as_posix()
        if MOUNT.resolve() != ROOT.parent.resolve():
            env["STRATA_SYCL_ROOT"] = str(MOUNT)
    if env:
        out["env"] = env
    out.update(keep)
    return out


def start_windows(cfg_path: Path, cfg: dict, port, gpu, open_browser: bool, keep) -> int:
    """Windows: start an installed model - setup's start() for a "sycl" config, without its CUDA/HIP steps: the
    saved settings a start may change (--host, --api-key, --draft-vocab, --vram-reserve-mib), the draft subset,
    then sycl/serve/server_intel.py (which sets the engine's environment: ONEAPI_DEVICE_SELECTOR for the card, the
    Level Zero v2 adapter, the heartbeat) with --open, as upstream's run script does."""
    cfg = S.upgrade_config(cfg_path, cfg)             # --prefill rules, read with the SYCL engine's version
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        S.fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    keep = {k: v for k, v in (keep or {}).items() if v is not None}
    reserve = keep.pop("vram_reserve_mib", None)       # #493: an engine argument, kept in the config's args
    if reserve is not None:
        args = cfg["args"]
        if "--vram-reserve-mib" in args[:-1]:
            args[args.index("--vram-reserve-mib") + 1] = str(reserve)
        else:
            args += ["--vram-reserve-mib", str(reserve)]
        S.write_config(cfg_path, cfg)
        S.ok(f"saved for this model: {reserve} MiB of VRAM kept free for other programs (--vram-reserve-mib)")
    if keep and any(cfg.get(k) != v for k, v in keep.items()):
        cfg.update(keep)
        S.write_config(cfg_path, cfg)
        S.ok("saved for this model: " + ", ".join("api key" if k == "api_key" else f"{k.replace('_', ' ')} {v}"
                                                  for k, v in keep.items()))
    cfg_path.touch()                                   # the most recently used model
    if "--mtp" in cfg["args"][:-1]:
        S.refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
    if isinstance(gpu, list):
        S.fail("several Intel cards sharing one model (--gpus) is not supported by the SYCL port on Windows yet",
               "use one card: --gpu N")
    cmd = [sys.executable, str(SERVER), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if gpu is not None:                                # --gpu N: this start only, on that card (its SYCL index)
        cmd += ["--gpu", str(gpu)]
    if open_browser:
        cmd.append("--open")
    streamed = "--stream-experts" in cfg["args"]
    S.say()
    S.say("  " + "-" * 100)
    S.say(f"  Starting {cfg.get('model_name', 'the model')} on the Intel Arc" +
          (": its experts are streamed from the model file into VRAM" if streamed else ""))
    S.say("  (a minute or two; the first start of a model reads its weights from the disk). The browser opens when it is")
    S.say("  ready. Later, closing this window stops the model.")
    S.say("  " + "-" * 100)
    for n, line in enumerate(textwrap.wrap(f"Settings ({cfg_path.name}): {S.settings_summary(cfg, port)}", 100,
                                           break_on_hyphens=False)):
        S.say(("  " if n == 0 else "    ") + line)
    return subprocess.call(cmd, cwd=str(ROOT))


def install(argv) -> None:
    explicit = any(f in argv for f in ("--setup", "--model", "--family", "--check", "--no-start", "--update"))
    starting = bool(list(ROOT.glob("strata-*.json"))) and not explicit   # a plain start of an installed model
    listed = None                                       # the cards as SYCL numbers them (Windows, engine built)
    if S.WIN:
        if "--build" in argv:
            S.fail(f"setup does not compile the SYCL engine: {BUILD_BAT.relative_to(ROOT)} does ({DOC})",
                   "run it, then run this again without --build")
        exe, why = sycl_engine(strict=not starting)
        if exe is None and "--check" in argv:           # the PC check works before the engine is built
            S.warn(f"Strata's SYCL engine cannot be used yet: {why}")
        elif exe is None:
            S.fail(f"Strata's SYCL engine cannot be used: {why}", f"{DOC}: build it, then run this again")
        if exe is not None:
            listed = sycl_devices()
            if listed is None and PROBE_EXE.exists():
                S.fail(f"the SYCL runtime finds no GPU ({PROBE_EXE} --list-devices did not answer)", DRIVER_HINT)
            if listed == []:
                S.fail("the SYCL runtime lists no Level Zero device", DRIVER_HINT)
            if listed is None:
                S.warn(f"{PROBE_EXE.name} is missing: the cards are numbered in Windows' display-adapter order, "
                       f"which may not be SYCL's; {BUILD_BAT.relative_to(ROOT)} builds it with the engine")
        intel = listed or intel_gpus_windows()
        if not intel:
            S.fail("no Intel GPU found (Windows lists no Intel display adapter)",
                   "on an NVIDIA or AMD card, run START-HERE.bat")
    else:
        intel = intel_gpus()
        if not intel:
            S.fail("no Intel Arc found (an xe or i915 card in /sys/class/drm)", "on an NVIDIA or AMD card, run ./setup.sh")
        exe, why = sycl_engine()
        if exe is None:
            S.fail(f"Strata's SYCL engine cannot be used: {why}", "docs/INTEL.md: build it, then run this again")
    for name in ("gpus", "amd_gpus", "amd_problem", "hip_vision", "build_engine_hip", "hipblaslt_table", "ram_gb",
                 "write_run_script", "start", "say", "main", "bench_tips"):
        if not callable(getattr(S, name, None)):
            S.fail(f"setup.py has no {name}() any more: sycl/setup_intel.py needs updating for this setup.py")
    if S.WIN:
        for name in WIN_NEEDED:
            if not callable(getattr(S, name, None)):
                S.fail(f"setup.py has no {name}() any more: sycl/setup_intel.py needs updating for this setup.py")

    problem = intel_problem if S.WIN else (lambda g: None)
    usable = [g for g in intel if problem(g) is None]
    card = max(usable, key=lambda g: g["vram_gb"]) if usable else intel[0]   # the card setup will pick
    card_drv = card_driver(card)
    default_gpu = usable[0]["index"] if len(usable) == 1 else None    # setup writes no "gpu" for a lone card
    real_ram = S.ram_gb()
    keep = {}                                           # hand-set keys setup does not write: kept across a rerun
    for p in ROOT.glob("strata-*.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8-sig"))
            keep[p.name] = {k: c[k] for k in ("model_switcher", "sampling") if k in c}
        except (OSError, ValueError):
            pass
    stub = Path(tempfile.mkdtemp(prefix="strata-sycl-"))  # setup reads the engine's BUILD.json; the Linux SYCL build
    atexit.register(shutil.rmtree, stub, True)            # has none, and on Windows setup's hip path wants a "hip" one
    (stub / "BUILD.json").write_text(json.dumps({"version": sycl_version(), "source": "local", "lib_dirs": []}))

    say = S.say
    fake_ram = max(real_ram, 1024.0)
    entry = "START-HERE.bat --backend sycl" if S.WIN else "./setup.sh --backend sycl"

    def say_intel(msg=""):
        """setup's words for its AMD path and its RAM rule, said for the Intel card."""
        msg = str(msg).replace("(AMD, experimental: docs/AMD_HIP.md)", f"(Intel Arc: the SYCL port, {DOC})")
        msg = msg.replace("Your AMD GPUs:", "Your Intel GPUs:").replace("just run ./setup.sh", f"just run {entry}")
        msg = msg.replace("just run START-HERE.bat", f"just run {entry}")
        msg = re.sub(r"\b(xe|i915) \(AMD: docs/AMD_HIP\.md\)", r"\1 driver (Intel Arc: docs/INTEL.md)", msg)
        where = ("the experts are loaded into RAM, the card computes the ones it holds" if card_drv == "i915"
                 else "the experts are streamed into VRAM")
        msg = msg.replace(f"RAM: {fake_ram:.0f} GB", f"RAM: {real_ram:.0f} GB ({where})")
        if S.WIN:                                       # the long phrases first: the short ones are inside them
            for a, b in (("several AMD cards sharing one model (--gpus) is Linux-only for now",
                          "several Intel cards sharing one model (--gpus) is not supported by the SYCL port on Windows yet"),
                         (f"the AMD backend runs on {getattr(S, 'AMD_CARDS', '')}",
                          f"the SYCL port runs on a discrete Intel Arc (tested: the Arc Pro B70; {DOC})"),
                         ("(AMD: docs/AMD_HIP.md)", f"(Intel: {DOC})"),
                         ("AMD display adapter", "Intel display adapter"), ("an AMD card", "an Intel card"),
                         ("AMD cards", "Intel cards"), ("AMD GPU", "Intel GPU"), ("on AMD", "on Intel"),
                         (str(stub / S.EXE), str(exe or ENGINE_EXE))):
                msg = msg.replace(a, b)
        if re.match(r"\s+\S+\s+needs ~\d+ GB RAM:", msg):   # --check's CUDA verdicts: replaced by the Intel one
            m = msg.split()[0]
            d = S.MODELS.get(m, {})
            shard1 = d.get("download_gb", 0) - 28.8          # all but the per-layer lookup table (read from disk)
            room = card["vram_gb"] - 3 + max(0.0, real_ram - 10)   # VRAM, plus a pinned host mirror
            msg = (f"  {m:8s} ~{shard1:.0f} GB of weights: " +
                   ("fits in VRAM" if shard1 <= card["vram_gb"] - 1.5 else
                    "fits with part of its experts mirrored in RAM (slower)" if shard1 <= room else "does not fit"))
        say(msg)
    S.say = say_intel
    S.gpus = lambda *a, **k: []                         # never nvidia-smi: on Windows an NVIDIA card may drive the desktop
    S.amd_gpus = lambda *a, **k: intel
    S.amd_problem = problem
    S.hip_vision = lambda asked: "none"                 # images are not wired on the SYCL port yet
    S.build_engine_hip = lambda *a, **k: stub
    S.hipblaslt_table = lambda *a, **k: None
    S.ram_gb = lambda: fake_ram                         # the experts are in VRAM: setup's RAM rule does not apply
    if S.WIN:
        # the Windows AMD steps main() takes on its hip path (the ready-made zip, the HIP probe, the ROCm DLLs):
        # our list, our engine, no downloads
        S.amd_gpus_win = lambda: intel
        S.hip_devices = lambda probe=None, text=None: listed
        S.hip_lib_dirs = lambda eng: []
        S.hip_runtime_beside_exe = lambda eng: None
        S.get_prebuilt_hip = lambda url_base, gpu, updating=False: stub
        S.hip_card = lambda eng, gpu, listed: intel_card(gpu, listed)
        S.update_installed_engine = lambda url_base, toolkit=None: None   # sycl_engine() above is the engine check

    bench_tips = S.bench_tips

    def bench_tips_intel(args, env, ram, *a, **k):
        """The tips as written for the CUDA engine, said with the real RAM (the 1024 GB above is only for the RAM rule);
        the --prefill auto:32768 one was measured on the CUDA engine and is left out."""
        return [t for t in bench_tips(args, env, real_ram, *a, **k) if "auto:32768" not in t]
    S.bench_tips = bench_tips_intel

    write = S.write_run_script

    def write_run_script(model, cfg_path, port, open_browser=True):   # setup.write_run_script's signature (#870)
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        cfg = to_sycl(cfg, exe, real_ram, keep.get(Path(cfg_path).name, {}), card["vram_gb"], card_drv, default_gpu)
        need = S.MODELS.get(model, {}).get("ram_gb", 0)
        if card_drv == "i915" and need and real_ram < need:
            S.warn(f"{model} on this Arc loads its experts into RAM (about {need} GB; this PC has {real_ram:.0f} GB): "
                   "the start will be slow or fail - a smaller model (--model) fits better (docs/INTEL.md)")
        Path(cfg_path).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        script = write(model, cfg_path, port, open_browser)
        script.write_text(script.read_text(encoding="utf-8").replace(str(ROOT / "serve" / "server.py"), str(SERVER)),
                          encoding="utf-8")
        return script
    S.write_run_script = write_run_script

    start = S.start

    def start_sycl(cfg_path, port=None, gpu=None, open_browser=True, yes=False, layer_split=None, keep=None):
        cfg_path = Path(cfg_path)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        if cfg.get("backend") != "sycl":
            return start(cfg_path, port, gpu, open_browser, yes, layer_split, keep)
        if S.WIN:
            return start_windows(cfg_path, cfg, port, gpu, open_browser, keep)
        script = ROOT / f"run-{cfg_path.stem[len('strata-'):]}.sh"
        S.say(f"\nstarting {script.name} ...")
        os.execv("/bin/sh", ["/bin/sh", str(script)])
    S.start = start_sycl

    argv = list(argv)
    if small_card(card["vram_gb"], card_drv):           # the engine's own advice for a card this size
        if "--draft-vocab" not in argv:
            argv += ["--draft-vocab", "en"]             # ~110 MiB less VRAM than the default CJK subset
        if "--vram-reserve-mib" not in argv:
            argv += ["--vram-reserve-mib", str(SMALL_RESERVE_MIB)]
    sys.argv = [str(ROOT / "setup.py"), *argv]
    if "--backend" not in argv:
        sys.argv += ["--backend", "hip"]
    if "--vision" not in argv:
        sys.argv += ["--vision", "none"]
    if "--low-ram" not in argv:
        sys.argv += ["--low-ram", "off"]
    sys.exit(S.main())


if __name__ == "__main__":
    try:
        install(sys.argv[1:])
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
