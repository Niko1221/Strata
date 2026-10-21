"""Write a run config from model files that are already prepared, for graphics cards the installer (setup.py) does not set up.

setup.py's Linux paths know NVIDIA (CUDA) and AMD (HIP); an Intel Arc card goes to a different, container-based setup (`--backend sycl`, docs/INTEL_ARC.md)
that this Terminal does not run. When setup says it cannot set this card up, and the model's files AND its prepared pack are already on disk (a box that was
prepared before, "use model files I already have"), the run config is simple enough to write here: the same arguments setup writes for the GSQ-RCO models,
with the engine named as the component (`component:strata-engine`). Nothing is downloaded, prepared or compiled; if a file is missing the person is told which.

Only the GSQ-RCO families (qwen, swift, coder) are adopted this way; images stay off.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SHARD = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")
FAMILIES = ("qwen", "swift", "coder")
UNSUPPORTED = ("no nvidia gpu found", "no amd gpu", "nvidia gpu", "not supported", "cannot be used")


def setup_cannot_handle_this_card(message: str) -> bool:
    low = (message or "").lower()
    return ("no nvidia gpu found" in low or "no amd gpu strata can use" in low or "nvidia-smi did not answer" in low
            or "no mac model installer" in low)


def intel_driver() -> str:
    """The kernel driver of the first Intel graphics card: "i915" (an A-series Arc) or "xe" (a B-series Arc), "" when it cannot be read."""
    import glob
    import os
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]/device")):
        try:
            if Path(card, "vendor").read_text().strip() != "0x8086":
                continue
            return os.path.basename(os.path.realpath(os.path.join(card, "driver")))
        except OSError:
            continue
    return ""


def find_shards(folder: Path) -> list[Path]:
    """The shards of one GGUF set in `folder` (-00001-of-0000N ... -0000N-of-0000N), sorted; [] if the set is incomplete."""
    found: dict[tuple[str, int], dict[int, Path]] = {}
    for p in sorted(Path(folder).glob("*.gguf")):
        m = SHARD.match(p.name)
        if m:
            found.setdefault((m.group("stem"), int(m.group("n"))), {})[int(m.group("i"))] = p
    for (_stem, n), items in found.items():
        if set(items) == set(range(1, n + 1)):
            return [items[i] for i in range(1, n + 1)]
    return []


def plan(*, family: str, model: str, model_dir: Path, data_dir: Path, src: Path, context: int, kv: str | None, model_name: str, family_def: dict,
         log: Path, backend: str | None) -> tuple[dict | None, list[str]]:
    """(config, problems): the config to write, or None with plain-words problems (what is missing)."""
    problems: list[str] = []
    if family not in FAMILIES:
        return None, [f"Strata can write the settings itself only for the Qwen3.8-Flash-Next, Swift and Coder files, not for {family}."]
    shards = find_shards(model_dir)
    if not shards:
        problems.append(f"the model files are not all in {model_dir} (every -0000N-of-0000N.gguf shard is needed)")
    pack = Path(data_dir) / "packs" / model.lower()
    if not (pack / "experts.bin").exists() and not (pack / "index.txt").exists():
        problems.append(f"the prepared pack {pack} is missing (this installer cannot prepare it for this graphics card)")
    tokenizer = pack / "tokenizer"
    if not tokenizer.exists():
        problems.append(f"the tokenizer {tokenizer} is missing")
    profile = src / "data" / (family_def.get("profile") or "expert-profile.bin")
    if not profile.exists():
        problems.append(f"the expert profile {profile} is missing")
    if problems:
        return None, problems
    mtp = Path(data_dir) / "mtp" / "rt"
    args = ["--pack", str(pack), "--native", str(shards[0])]
    if len(shards) > 1:
        args += ["--ple-gguf", str(shards[-1])]
    if backend == "metal":
        # a Mac: the Metal defaults metal/setup_mac.py writes for a model that streams from the SSD (one token per step, no draft layer, the GGUF mapped in place)
        args += ["--expert-profile", str(profile), "--expert-cache", "64", "--prefill", "auto", "--spec", "2", "--suffix-draft", "0",
                 "--max-context", str(context), "--mmap-experts"]
    else:
        args += ["--expert-profile", str(profile), "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5"]
        if mtp.exists():
            args += ["--mtp", str(mtp)]
        args += ["--max-context", str(context), "--kv", kv or ("int8" if context > 8192 else "fp16")]
        if backend == "sycl":
            # an Intel Arc, as sycl/setup_intel.py writes it: an A-series card on i915 loads the experts into a RAM arena and the CPU computes the misses
            # (--ple-io ram); every other Arc (xe: the B70) streams the experts from the GGUF into VRAM (--stream-experts); both keep a VRAM reserve
            reserve = "1024" if context <= 32768 else "2048"
            if intel_driver() == "i915":
                args += ["--ple-io", "ram", "--vram-reserve-mib", reserve]
            else:
                args += ["--stream-experts", "--vram-reserve-mib", reserve]
    cfg = {"exe": "component:strata-engine", "args": args, "cwd": str(src), "tokenizer": str(tokenizer), "model_name": model_name,
           "log": str(log), "lib_dirs": [], "port": 8080}
    if backend == "metal":
        cfg["env"] = {"STRATA_METAL_STREAM": "1"}
    if backend:
        cfg["backend"] = backend
    return cfg, []


def write(path: Path, cfg: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
