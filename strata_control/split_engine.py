"""strata-split: when the all-cards split engine may run, and the run config for it. OFF by default, opt-in.

`strata-split-N` (src/split, from sFNX, MIT, Copyright (c) 2026 John Fink) gives every card a share of EVERY layer instead of some
layers, and the cards exchange the layer sums through page-locked host memory (no peer access anywhere). On two RTX A6000 with
UD-Q4_K_XL it decodes 40-95% faster than the normal layer split and reads prompts 1.7-2.8x slower (phaseA-tests/sfnx/RESULTS.md,
section 8), so it is an option, never the default, until the owner has chosen.

This module is pure (no I/O besides reading GGUF directories): the launcher passes it facts and gets a Decision. The rule, from what
has actually been run:

  - the person turned it on (config key "split_engine": true, or `strata launch --split-engine on`); never otherwise;
  - Linux, 2 to 4 cards of ONE make (NVIDIA, or AMD RDNA4 gfx1201: the R9700 and RX 9070), all the same model of card and memory size;
  - a card architecture and card count that have been run (TESTED, per make): NVIDIA sm_86 on 2 cards and AMD gfx1201 on 4 cards (see
    TESTED below). Others build but stay off unless the caller says allow_untested (testers); the Decision says which part is untested
    in plain words;
  - the GGUF is UD-Q4_K_XL type: routed experts Q4_K/Q5_K/Q5_1/Q8_0, token embeddings and the shared expert Q8_0 (checked from the
    file's tensor types, not its name), the Qwen3.8-Flash-Next shapes (48 layers, 512 experts);
  - ALL routed experts fit in the cards' memory (the expert cache is off: with it the engine keeps a 74 GiB pinned copy of the
    experts in RAM) and the RAM that stays (the 27.5 GiB locked embedding table, a few GiB of buffers and the conversation cache)
    is available with room to spare;
  - context at most TESTED_CONTEXT (128K).
Anything else: use_split False and one plain line saying why. The server also falls back to the normal layer split if the split
engine refuses or fails to start (serve/server.py, "split_fallback").
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from typing import Any

# What has been run with the host exchange, per make: {make: {architecture: card counts}}.  NVIDIA: 2x RTX A6000 (boston, sm_86).
# AMD: 4x Radeon AI PRO R9700 (r9700, gfx1201; phaseA-tests/sfnx/RESULTS-HIP.md); two R9700 cannot hold UD-Q4_K_XL's experts.
TESTED = {"nvidia": {"sm_86": (2,)}, "amd": {"gfx1201": ()}}
BUILT_ARCHS = {"nvidia": ("sm_86",), "amd": ("gfx1201",)}   # the engine package's kernels: any other card is refused at start
TESTED_CARD_COUNTS = (2,)          # NVIDIA (kept for callers that name it)
TESTED_ARCHS = (86,)               # NVIDIA compute capability
TESTED_CONTEXT = 131072
MAX_CARDS = 4
LAYERS, EXPERTS = 48, 512          # Qwen3.8-Flash-Next
EXPERT_TYPES = {"Q4_K", "Q5_K", "Q5_1", "Q8_0", "Q4_0"}   # the routed-expert kernels
Q8_ROLES = ("token_embd.weight",)
# measured on 2x RTX A6000 48 GB at 128K context, 6 lanes (phaseA-tests/sfnx/RESULTS.md section 8): a card held half the experts plus
# 8.4 GiB of dense weights, buffers and the K/V pool, with 3.6 GiB left free. 9 GiB per card is that plus a little.
CARD_OVERHEAD_GIB = 9.0
RAM_SPARE_GIB = 8.0                # left over beside the engine's own RAM
RAM_PINNED_BUFFERS_GIB = 3.0       # the exchange buffers and heap (smaps: 1.7 GiB pinned + 0.9 heap)
CONVERSATION_CACHE_MIB = 4096      # parked conversations (host RAM); the engine's own default is 32-128 GiB
DEFAULT_LANES = 6


@dataclasses.dataclass
class Card:
    index: int
    name: str
    vram_mib: int
    cc: int                         # NVIDIA: compute capability as a number, 86 for 8.6; AMD: 0
    vendor: str = "nvidia"          # "nvidia" or "amd"
    arch: str = ""                  # "sm_86", "gfx1201"; empty: derived from cc (NVIDIA)

    @property
    def arch_name(self) -> str:
        return self.arch or (f"sm_{self.cc}" if self.vendor == "nvidia" else "?")


@dataclasses.dataclass
class ModelFacts:
    routed_types: dict[str, int]    # type name -> tensors, for the routed experts (gate/up/down, all layers)
    routed_bytes: int
    embd_type: str | None           # token_embd.weight
    shexp_types: dict[str, int]     # shared-expert tensors
    embd_ram_bytes: int             # token_embd + per_layer_token_embd: locked in RAM
    layers: int
    experts: int


@dataclasses.dataclass
class Decision:
    use_split: bool
    reasons: list[str]              # why not (plain words), empty when use_split
    cards: int = 0
    untested: list[str] = dataclasses.field(default_factory=list)   # what about this setup nobody has run
    need_vram_gib: float = 0.0
    need_ram_gib: float = 0.0

    @property
    def line(self) -> str:
        if self.use_split:
            note = f" ({'; '.join(self.untested)}: not tested)" if self.untested else ""
            return f"Using the all-cards split engine on {self.cards} cards{note}."
        return "Using the normal layer split: " + "; ".join(self.reasons) + "."


GIB = 1 << 30


def scan_model(first_shard: str | pathlib.Path) -> ModelFacts:
    """The facts the rule needs from the GGUF set whose first shard is named (all shards of the -0000N-of-0000M set are read)."""
    from tools.gguf_reader import GGUFFile

    first = pathlib.Path(first_shard)
    shards = [first]
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if m:
        total = int(m.group(3))
        shards = [first.with_name(f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf") for i in range(1, total + 1)]
    routed_types: dict[str, int] = {}
    routed_bytes = 0
    shexp: dict[str, int] = {}
    embd_type = None
    embd_ram = 0
    layers = 0
    experts = 0
    for shard in shards:
        g = GGUFFile(shard)
        for t in g.tensors:
            n = t.name
            b = t.expected_bytes() or 0
            if "_exps" in n and n.endswith(".weight") and "shexp" not in n:
                routed_types[t.type_name] = routed_types.get(t.type_name, 0) + 1
                routed_bytes += b
                if len(t.shape) >= 3:
                    experts = max(experts, t.shape[-1])
                mm = re.search(r"blk\.(\d+)\.", n)
                if mm:
                    layers = max(layers, int(mm.group(1)) + 1)
            elif "shexp" in n and n.endswith(".weight"):
                shexp[t.type_name] = shexp.get(t.type_name, 0) + 1
            elif n == "token_embd.weight":
                embd_type = t.type_name
                embd_ram += b
            elif n == "per_layer_token_embd.weight":
                embd_ram += b
    return ModelFacts(routed_types, routed_bytes, embd_type, shexp, embd_ram, layers, experts)


def decide(*, enabled: bool, system: str, cards: list[Card], ram_total_gib: float, ram_available_gib: float, model: ModelFacts,
           context: int = TESTED_CONTEXT, allow_untested: bool = False, cache_mib: int = CONVERSATION_CACHE_MIB) -> Decision:
    """Whether the all-cards split engine runs this launch, with plain reasons when not."""
    if not enabled:
        return Decision(False, ["the all-cards split engine is off (it is opt-in)"])
    why: list[str] = []
    untested: list[str] = []
    if system != "linux":
        why.append("the all-cards split engine runs on Linux only")
    n = len(cards)
    if n < 2:
        why.append("it needs two or more cards")
    elif n > MAX_CARDS:
        why.append(f"it handles at most {MAX_CARDS} cards, this PC offers {n}")
    make = cards[0].vendor if cards else "nvidia"
    if cards:
        if len({c.vendor for c in cards}) != 1:
            why.append("the cards are of different makes")
        if len({(c.name, c.vram_mib) for c in cards}) != 1:
            why.append("the cards are not all the same model and size")
        if make == "nvidia" and not all(c.cc >= 70 for c in cards):
            why.append("a card is too old for it")
        if make == "amd" and not all(c.arch_name in BUILT_ARCHS["amd"] for c in cards):
            why.append("the AMD engine is built for RDNA4 cards (gfx1201: Radeon AI PRO R9700, RX 9070), not " + cards[0].arch_name)
    tested = TESTED.get(make, {})
    archs = sorted({c.arch_name for c in cards})
    counts = sorted({k for a in archs for k in tested.get(a, ())})
    if 2 <= n <= MAX_CARDS and make in TESTED and n not in counts and not (len(archs) == 1 and archs[0] not in tested):
        if allow_untested:
            untested.append(f"{n} cards")
        else:
            ran = " and ".join(str(k) for k in counts) if counts else "no number of"
            why.append(f"it has only been run on {ran} cards, not {n}")
    if cards and not all(c.arch_name in tested for c in cards):
        if allow_untested:
            untested.append(f"card architecture {cards[0].arch_name}")
        else:
            ran = "/".join(tested) if tested else "no"
            why.append(f"it has only been run on {ran} cards, not {cards[0].arch_name}")
    if context > TESTED_CONTEXT:
        if allow_untested:
            untested.append(f"context {context}")
        else:
            why.append(f"it has only been run up to a context of {TESTED_CONTEXT}")
    bad = sorted(t for t in model.routed_types if t not in EXPERT_TYPES)
    if not model.routed_types:
        why.append("this is not a model it knows (no routed experts)")
    elif bad:
        why.append("the model's routed experts use " + "/".join(bad) + ", which it does not read")
    if model.embd_type != "Q8_0":
        why.append(f"the token embeddings are {model.embd_type}, it needs Q8_0")
    if model.shexp_types and set(model.shexp_types) - {"F32"} != {"Q8_0"}:   # the router gate vector beside them is F32
        why.append("the shared expert is not Q8_0")
    if (model.layers, model.experts) != (LAYERS, EXPERTS):
        why.append("it only knows Qwen3.8-Flash-Next (48 layers, 512 experts)")
    need_vram = model.routed_bytes / GIB / max(n, 1) + CARD_OVERHEAD_GIB
    need_ram = model.embd_ram_bytes / GIB + RAM_PINNED_BUFFERS_GIB + cache_mib / 1024
    if cards and n >= 2 and min(c.vram_mib for c in cards) / 1024 < need_vram:
        why.append(f"all the experts do not fit in the cards' memory (each card needs about {need_vram:.0f} GiB, has "
                   f"{min(c.vram_mib for c in cards) / 1024:.0f} GiB)")
    if ram_available_gib < need_ram + RAM_SPARE_GIB:
        why.append(f"there is not enough free RAM (about {need_ram + RAM_SPARE_GIB:.0f} GiB needed, {ram_available_gib:.0f} GiB free)")
    if why:
        return Decision(False, why, n, untested, need_vram, need_ram)
    return Decision(True, [], n, untested, need_vram, need_ram)


def split_config(base: dict, decision: Decision, *, exe: str, native: str, mtp_dir: str, draft_vocab: str, context: int,
                 lanes: int = DEFAULT_LANES, vision: bool = False, cache_mib: int = CONVERSATION_CACHE_MIB) -> dict:
    """The run config for strata-split: `base` is the normal run config (its tokenizer, port, log, model name, GPUs and everything else
    stay), and it is kept whole under "split_fallback" so the server can start the normal layer split if this engine refuses."""
    if not decision.use_split:
        raise ValueError("split_config: the decision is not to use the split engine")
    cfg = {k: v for k, v in base.items() if k not in ("exe", "args", "env", "layer_split", "replicas", "user_args", "drafting")}
    cfg["exe"] = exe
    cfg["args"] = ["--native", native, "--mtp", mtp_dir, "--draft-vocab", draft_vocab, "--max-context", str(context),
                   "--spec", "6", "--lookup", "15", "--conversation-cache-mib", str(cache_mib), "--batch", str(max(lanes, 1))]
    if vision:
        cfg["args"].append("--vision")
    # every expert in the cards' memory, no expert cache, no arena: the engine reads these two at start
    cfg["env"] = {**(base.get("env") or {}), "FNX_ARENA": "0", "FNX_EXPERT_RAM_GIB": "off"}
    cfg["split_fallback"] = {k: base[k] for k in ("exe", "args", "env", "layer_split", "replicas", "user_args", "drafting", "cwd") if k in base}
    cfg["split_engine_note"] = decision.line
    return cfg


# ---- the launcher side: facts from this PC, then the config ----------------------------------------------------------------------------
def probe_cards(indexes: list[int] | None = None) -> list[Card]:
    """The cards (all, or the named ones): NVIDIA from nvidia-smi, else AMD from the kernel's KFD topology and sysfs; [] when neither says."""
    cards = probe_nvidia_cards(indexes)
    return cards or probe_amd_cards(indexes)


def probe_amd_cards(indexes: list[int] | None = None, root: str = "/sys") -> list[Card]:
    """AMD GPUs the compute stack counts, in HIP's order: the KFD topology nodes with SIMDs (gfx_target_version 120001 = gfx1201), their
    memory from the node's VRAM bank (properties "size_in_bytes" of its memory bank 0) or the drm card's mem_info_vram_total. No ROCm
    tools needed. [] when there is no /sys/class/kfd."""
    import pathlib

    nodes = pathlib.Path(root) / "class/kfd/kfd/topology/nodes"
    cards: list[Card] = []
    try:
        dirs = sorted((d for d in nodes.iterdir() if d.name.isdigit()), key=lambda d: int(d.name))
    except OSError:
        return []
    idx = 0
    for d in dirs:
        try:
            props = dict(line.split(None, 1) for line in (d / "properties").read_text().splitlines() if line.strip())
            if int(props.get("simd_count", "0")) <= 0:
                continue                                     # the CPU node
            gv = int(props.get("gfx_target_version", "0"))
            arch = f"gfx{gv // 10000}{(gv // 100) % 100:x}{gv % 100:x}" if gv else "?"
            name = props.get("marketing_name", "").strip() or "AMD GPU"
            mem = 0
            bank = d / "mem_banks/0/properties"
            if bank.is_file():
                bp = dict(line.split(None, 1) for line in bank.read_text().splitlines() if line.strip())
                mem = int(bp.get("size_in_bytes", "0")) >> 20
        except (OSError, ValueError):
            continue
        if indexes is None or idx in indexes:
            cards.append(Card(idx, name, mem, 0, "amd", arch))
        idx += 1
    return cards


def probe_nvidia_cards(indexes: list[int] | None = None) -> list[Card]:
    """The NVIDIA cards (all, or the named ones) with their name, memory and compute capability, from nvidia-smi; [] when it cannot say."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx, mem = int(parts[0]), int(float(parts[2]))
            cc = int(round(float(parts[3]) * 10))
        except ValueError:
            continue
        if indexes is None or idx in indexes:
            cards.append(Card(idx, parts[1], mem, cc))
    return cards


def ram_gib() -> tuple[float, float]:
    """(total, available) GiB of this PC's RAM from /proc/meminfo; (0, 0) where there is none."""
    try:
        vals = {}
        for line in open("/proc/meminfo", encoding="ascii"):
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                vals[k] = int(v.split()[0]) / (1 << 20)
        return vals.get("MemTotal", 0.0), vals.get("MemAvailable", 0.0)
    except OSError:
        return 0.0, 0.0


def _flag(args: list, name: str) -> str | None:
    return str(args[args.index(name) + 1]) if name in args and args.index(name) + 1 < len(args) else None


def wanted(cfg: dict, environ: dict | None = None) -> bool:
    """The person turned the option on: "split_engine": true in the run config, or STRATA_SPLIT_ENGINE=1 in the environment (testers)."""
    import os

    return cfg.get("split_engine") is True or (environ if environ is not None else os.environ).get("STRATA_SPLIT_ENGINE") == "1"


def apply_to_config(cfg: dict, *, exe_for_cards, system: str | None = None, allow_untested: bool = False, probe=None, ram=None,
                    scan=None, data_dir: str | None = None) -> Decision:
    """Turn a normal run config into the split engine's config IN PLACE when the rule says yes, and say what was decided.
    exe_for_cards(n) -> the path of strata-split-n or None (the engine package has none); probe(indexes) -> [Card]; ram() -> (total,
    available) GiB; scan(first_shard) -> ModelFacts: the facts of this PC and model, replaceable for tests."""
    import os
    import pathlib
    import sys

    system = system or ("linux" if sys.platform.startswith("linux") else sys.platform)
    args = [str(a) for a in cfg.get("args") or []]
    native = _flag(args, "--native")
    if not native:
        return Decision(False, ["this model is not a single GGUF file set"])
    gpus = cfg.get("gpu")
    indexes = [int(g) for g in gpus] if isinstance(gpus, list) else ([int(gpus)] if isinstance(gpus, int) else None)
    cards = (probe or probe_cards)(indexes) if system == "linux" else []
    total, avail = (ram or ram_gib)()
    try:
        model = (scan or scan_model)(native)
    except Exception as e:                                   # noqa: BLE001 - a file it cannot read is a reason, not a failed start
        return Decision(False, [f"the model file could not be read ({type(e).__name__})"])
    context = int(_flag(args, "--max-context") or 8192)
    d = decide(enabled=True, system=system, cards=cards, ram_total_gib=total, ram_available_gib=avail, model=model, context=context,
               allow_untested=allow_untested)
    if not d.use_split:
        return d
    exe = exe_for_cards(d.cards)
    if not exe:
        return Decision(False, [f"this engine package has no strata-split-{d.cards}"], d.cards)
    mtp = _flag(args, "--mtp")
    if not mtp:
        return Decision(False, ["the model has no draft layer files (--mtp)"], d.cards)
    vocab = None
    for folder in (pathlib.Path(exe).parent, pathlib.Path(data_dir) if data_dir else None, pathlib.Path(cfg.get("cwd") or ".") / "data"):
        if folder and (folder / "draft_vocab_en.bin").is_file():
            vocab = str(folder / "draft_vocab_en.bin")
            break
    if not vocab:
        return Decision(False, ["draft_vocab_en.bin is missing"], d.cards)
    lanes = int(cfg.get("parallel") or _flag(args, "--batch") or DEFAULT_LANES)
    new = split_config(cfg, d, exe=exe, native=native, mtp_dir=mtp, draft_vocab=vocab, context=context, lanes=lanes,
                       vision=bool(cfg.get("vision")) or "--vision" in args)
    cfg.clear()
    cfg.update(new)
    return d
