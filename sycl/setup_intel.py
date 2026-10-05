"""setup.py for an Intel Arc: upstream's installer, steered onto the SYCL port from outside.

    python sycl/setup_intel.py [setup.py's options]

The SYCL port keeps out of the shared files (upstream merges stay clean), so this does not edit setup.py: it imports
it and replaces the few steps that are NVIDIA/AMD-specific, then runs setup's own main().  Everything else - the
model choice, the download, the pack, the tokenizer, the MTP draft layer, the context and KV questions - is
setup.py's, unchanged.  What is replaced:

  - the GPU check: the Intel Arc is offered through setup's AMD path (the one that compiles locally and has no
    images), named and sized from sysfs;
  - the engine step: the SYCL build (build-sycl-aot/strata, run in the strata-sycl-dev image by
    sycl/serve/strata-sycl.sh; docs/INTEL.md) instead of a CUDA/HIP build;
  - the RAM rule: the CUDA engine keeps every expert in RAM, the SYCL port streams them from the GGUF into VRAM
    (--stream-experts), so RAM only bounds the KV streaming;
  - the config: the container's paths, the SYCL flags, backend "sycl"; the run script starts
    sycl/serve/server_intel.py (serve/server.py plus the Intel Monitor readings and the model menu).

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402

SYCL_WRAPPER = ROOT / "sycl" / "serve" / "strata-sycl.sh"
SYCL_IMAGE = "strata-sycl-dev"
SERVER = ROOT / "sycl" / "serve" / "server_intel.py"
MOUNT = Path(os.environ.get("STRATA_SYCL_ROOT") or ROOT.parent)   # what strata-sycl.sh mounts at /work

# Battlemage / Alchemist PCI device ids -> (name, VRAM GB). lspci's database lags new cards (an Arc Pro B70 reads
# "Intel Corporation Device [8086:e223]"), so the sysfs id is the reliable signal and the name table is ours.
INTEL_ARC = {"e223": ("Arc Pro B70", 32.0), "e221": ("Arc Pro B60", 24.0), "e20b": ("Arc B580", 12.0),
             "e20c": ("Arc B570", 10.0), "e212": ("Arc B50", 16.0),
             "56a0": ("Arc A770", 16.0), "56a1": ("Arc A750", 8.0), "56a2": ("Arc A580", 8.0),
             "56a5": ("Arc A380", 6.0), "56a6": ("Arc A310", 4.0), "5690": ("Arc A770M", 16.0)}


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
        found.append({"index": len(found), "name": f"Intel {name}", "vram_gb": vram, "arch": "xe", "driver": driver})
    return found


def sycl_engine():
    """The SYCL build: (binary, why-not)."""
    if S.WIN:
        return None, "the SYCL port runs on Linux only"
    exe = next((b for b in (ROOT / "build-sycl-aot" / "strata", ROOT / "build-sycl" / "strata") if b.exists()), None)
    if exe is None:
        return None, "it is not built (sycl/tools/build.sh; docs/INTEL.md)"
    if not shutil.which("docker"):
        return None, "docker is not installed (the engine runs in the oneAPI image)"
    if subprocess.run(["docker", "image", "inspect", SYCL_IMAGE], capture_output=True).returncode != 0:
        return None, f"the runtime image {SYCL_IMAGE} is missing (sycl/tools/Dockerfile)"
    return exe, None


def sycl_path(path) -> str:
    """A host path as the engine's container sees it: the folder above the Strata checkout (or STRATA_SYCL_ROOT) is
    mounted at /work."""
    p, root = Path(path).resolve(), MOUNT.resolve()
    try:
        return "/work/" + p.relative_to(root).as_posix()
    except ValueError:
        S.fail(f"{p} is outside {root}, which the SYCL engine's container mounts",
               f"keep the models and the data folder under {root} (--models-dir / --data-dir)")


def sycl_version() -> str:
    m = re.search(r"project\(\s*\S+\s+VERSION\s+([\d.]+)", (ROOT / "sycl" / "CMakeLists.txt").read_text())
    return m.group(1) if m else "0"


def flag(args, name):
    return args[args.index(name) + 1] if name in args else None


def drop(args, name, value=False):
    while name in args:
        i = args.index(name)
        del args[i:i + 1 + int(value)]


def to_sycl(cfg: dict, exe: Path, ram: float, keep: dict) -> dict:
    """setup's config (written for its HIP path) -> the SYCL port's: the container's paths, experts streamed from the
    GGUF into VRAM (every expert must fit, so the VRAM reserve is the smallest that leaves the KV and the prompt
    buffers room - docs/INTEL.md), KV streaming from 64K up when the RAM holds the KV (the B70 at 256K decodes at
    40+ tok/s with it, 4-9 without)."""
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
        S.ok(f"KV streaming on: the context's KV cache lives in RAM ({kv_ram:.1f} GB), every expert stays in VRAM")
    drop(args, "--vram-reserve-mib", True)
    args += ["--stream-experts", "--vram-reserve-mib", "1024" if ctx <= 32768 else "2048"]
    if ctx > 32768:                                     # long contexts: 4096-token chunks keep the prompt buffers small
        drop(args, "--prefill", True)
        args += ["--prefill", "4096"]
    out = {k: v for k, v in cfg.items() if k not in ("lib_dirs", "env", "vision", "gpus")}
    out.update({"backend": "sycl", "exe": str(SYCL_WRAPPER), "args": args, "sycl_root": str(MOUNT)})
    env = {}
    if exe != ROOT / "build-sycl-aot" / "strata":
        env["STRATA_SYCL_BIN"] = str(exe.relative_to(ROOT))
    if MOUNT.resolve() != ROOT.parent.resolve():
        env["STRATA_SYCL_ROOT"] = str(MOUNT)
    if env:
        out["env"] = env
    out.update(keep)
    return out


def install(argv) -> None:
    intel = [] if S.WIN else intel_gpus()
    if not intel:
        S.fail("no Intel Arc found (an xe or i915 card in /sys/class/drm)", "on an NVIDIA or AMD card, run ./setup.sh")
    exe, why = sycl_engine()
    if exe is None:
        S.fail(f"Strata's SYCL engine cannot be used: {why}", "docs/INTEL.md: build it, then run this again")
    for name in ("gpus", "amd_gpus", "amd_problem", "hip_vision", "build_engine_hip", "hipblaslt_table", "ram_gb",
                 "write_run_script", "start", "say", "main"):
        if not callable(getattr(S, name, None)):
            S.fail(f"setup.py has no {name}() any more: sycl/setup_intel.py needs updating for this setup.py")

    real_ram = S.ram_gb()
    keep = {}                                           # hand-set keys setup does not write: kept across a rerun
    for p in ROOT.glob("strata-*.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8-sig"))
            keep[p.name] = {k: c[k] for k in ("model_switcher", "sampling") if k in c}
        except (OSError, ValueError):
            pass
    stub = Path(tempfile.mkdtemp(prefix="strata-sycl-"))  # setup reads the engine's BUILD.json; the SYCL build has none
    atexit.register(shutil.rmtree, stub, True)
    (stub / "BUILD.json").write_text(json.dumps({"version": sycl_version(), "source": "local", "lib_dirs": []}))

    say = S.say
    fake_ram = max(real_ram, 1024.0)

    def say_intel(msg=""):
        """setup's words for its AMD path and its RAM rule, said for the Intel card."""
        msg = str(msg).replace("(AMD, experimental: docs/AMD_HIP.md)", "(Intel Arc: the SYCL port, docs/INTEL.md)")
        msg = msg.replace("Your AMD GPUs:", "Your Intel GPUs:").replace("just run ./setup.sh", "just run sycl/setup_intel.py")
        msg = msg.replace(f"RAM: {fake_ram:.0f} GB", f"RAM: {real_ram:.0f} GB (the experts are streamed into VRAM)")
        if re.match(r"\s+\S+\s+needs ~\d+ GB RAM:", msg):   # --check's CUDA verdicts: replaced by the Intel one
            m = msg.split()[0]
            d = S.MODELS.get(m, {})
            shard1 = d.get("download_gb", 0) - 28.8          # all but the per-layer lookup table (read from disk)
            room = intel[0]["vram_gb"] - 3 + max(0.0, real_ram - 10)   # VRAM, plus a pinned host mirror
            msg = (f"  {m:8s} ~{shard1:.0f} GB of weights: " +
                   ("fits in VRAM" if shard1 <= intel[0]["vram_gb"] - 1.5 else
                    "fits with part of its experts mirrored in RAM (slower)" if shard1 <= room else "does not fit"))
        say(msg)
    S.say = say_intel
    S.gpus = lambda *a, **k: []
    S.amd_gpus = lambda *a, **k: intel
    S.amd_problem = lambda g: None
    S.hip_vision = lambda asked: "none"                 # images are not wired on the SYCL port yet
    S.build_engine_hip = lambda *a, **k: stub
    S.hipblaslt_table = lambda *a, **k: None
    S.ram_gb = lambda: fake_ram                         # the experts are in VRAM: setup's RAM rule does not apply

    write = S.write_run_script

    def write_run_script(model, cfg_path, port):
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        cfg = to_sycl(cfg, exe, real_ram, keep.get(Path(cfg_path).name, {}))
        Path(cfg_path).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        script = write(model, cfg_path, port)
        script.write_text(script.read_text().replace(str(ROOT / "serve" / "server.py"), str(SERVER)))
        return script
    S.write_run_script = write_run_script

    start = S.start

    def start_sycl(cfg_path, *a, **k):
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8-sig"))
        if cfg.get("backend") != "sycl":
            return start(cfg_path, *a, **k)
        script = ROOT / f"run-{Path(cfg_path).stem[len('strata-'):]}.sh"
        S.say(f"\nstarting {script.name} ...")
        os.execv("/bin/sh", ["/bin/sh", str(script)])
    S.start = start_sycl

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
