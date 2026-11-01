"""strata-split: when the all-cards split engine may run, and the run config for it. OFF by default, opt-in.

`strata-split-N` (src/split, from sFNX, MIT, Copyright (c) 2026 John Fink) gives every card a share of EVERY layer instead of some
layers, and the cards exchange the layer sums through page-locked host memory (no peer access anywhere). On two RTX A6000 with
UD-Q4_K_XL it decodes 40-95% faster than the normal layer split and reads prompts 1.7-2.8x slower (phaseA-tests/sfnx/RESULTS.md,
section 8), so it is an option, never the default, until the owner has chosen.

This module is pure (no I/O besides reading GGUF directories): the launcher passes it facts and gets a Decision. The rule, from what
has actually been run:

  - the person turned it on (config key "split_engine": true, or `strata launch --split-engine on`); never otherwise;
  - Linux, NVIDIA cards, 2 to 4 of them, all the same model of card and memory size;
  - a card architecture and card count that have been run (TESTED): sm_86 on 2 cards today. Others build but stay off unless the
    caller says allow_untested (testers); the Decision says which part is untested in plain words;
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

TESTED_CARD_COUNTS = (2,)          # run with the host exchange: 2 cards (boston, 2x RTX A6000)
TESTED_ARCHS = (86,)               # compute capability: sm_86 only
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
    cc: int                         # compute capability as a number, 86 for 8.6


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
    if cards:
        if len({(c.name, c.vram_mib) for c in cards}) != 1:
            why.append("the cards are not all the same model and size")
        if not all(c.cc >= 70 for c in cards):
            why.append("a card is too old for it")
    if 2 <= n <= MAX_CARDS and n not in TESTED_CARD_COUNTS:
        if allow_untested:
            untested.append(f"{n} cards")
        else:
            why.append(f"it has only been run on {TESTED_CARD_COUNTS[0]} cards, not {n}")
    if cards and not all(c.cc in TESTED_ARCHS for c in cards):
        if allow_untested:
            untested.append(f"card architecture sm_{cards[0].cc}")
        else:
            why.append(f"it has only been run on sm_{TESTED_ARCHS[0]} cards, not sm_{cards[0].cc}")
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
    if model.shexp_types and set(model.shexp_types) != {"Q8_0"}:
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
