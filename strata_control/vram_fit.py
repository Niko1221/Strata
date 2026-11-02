"""Will the default start fit the card? If it is just short, lower the context, the reserve and the prompt chunk until it does.

Owner rule: fixes are automatic defaults, users never set flags. A model that is a few hundred MiB short of one expert-cache slot
(Gyro-M on an 8 GB Arc A750: "no VRAM is left for the expert cache, about 744 MiB more than this card has free") must start on its own
with a smaller context, never be refused. Pure functions, no I/O: sycl/setup_intel.py (setup), strata_cli/launch (strata launch and the
app's recommended context) import it.

The arithmetic (MiB), measured on the Arc A750 (8 GB, i915) with the SYCL engine of 0.2.2, Gyro-M and Gyro-S, 12 starts:

    slack = card - gap - base - session(ctx) - head - reserve - prompt buffers(chunk)      # what is left for the expert cache
    fits  = slack >= MIN_SLOTS * slot + MARGIN

* gap: the part of the card the driver does not give (Arc: 471 MiB, the engine sees 7.54 GiB of 8; CUDA / HIP: the runtime's context, 250).
* base: dense weights, the draft layer, the n-gram head's buffers and the engine's own allocations that do not depend on the context.
  Gyro-M 5945 (measured "holds" 6.02 GiB at 8K minus the 234 MiB session), Gyro-S 4720; any other model: 2448 (1.4 GiB dense + 0.8 GiB
  draft layer + slack), a figure that only a card under about 5 GB can get past, so a default that fits today is never changed.
* session: 121 MiB + the KV cache. int8 KV measured 234 / 347 / 573 MiB at 8192 / 16384 / 32768 tokens (13.8 KiB per token).
* head: the draft head (English subset) 110 MiB.
* prompt buffers: 170 + 0.66 * chunk MiB, measured at chunks 256 .. 2048 (338 / 508 / 850 / 1525 MiB), independent of the context.
  `--prefill auto` resolved to 2048 on that card.
* slot: 1.243 MiB (0.50 GiB / 412 slots).
* request slots ("Concurrent requests" 2 or more, the engine's --batch N): every slot holds a session of its own, the size of the main one, on every backend
  (the engine's "--batch 2: 2 slot sessions (0.23 GiB each)" is session(8192) = 234 MiB), taken out of the expert cache's room.  Only these starts also use the
  measured Qwen3.8-Flash-Next figures below (BASE_SLOTTED_MIB, SLOT_SLOTTED_MIB); the single-request estimate above is left as it is.

Why these numbers and not the engine's own: the engine only finds out after it has loaded 30 GB of weights; setup has to choose before.
It is an estimate; the engine still says what is short if it is off (and a user who asked for a context or a reserve keeps it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

CTX_STEPS = (32768, 16384, 8192)        # the ladder, biggest first: a context above 32K is never touched by this module
CHUNK_STEPS = (1024,)                   # a smaller prompt chunk, tried after the default one (1024 is the value measured end to end)
RESERVE_FLOOR_MIB = 100                 # the lowest reserve the engine accepts without complaint (docs/INTEL.md); never raised
AUTO_CHUNK = 2048                       # what --prefill auto came to on the 8 GB card
MIN_SLOTS = 128                         # the engine wants 128 cache slots to lend the prompt path; below that it is slow
SLOT_MIB = 1.243
MARGIN_MIB = 64                         # the estimate's error (the measurements agree to +-25 MiB)
HEAD_MIB = 110
SESSION_FIXED_MIB = 121
KV_BYTES = {"fp16": 2112, "int8": 1056, "q4_0": 576, "k8v4": 816}   # per token per attention layer (13) = setup.UD_FIT_KV_BYTES
KV_MEASURED_FACTOR = 1.054              # the engine's session is 5.4% over the bare cache (scales)
PROMPT_BASE_MIB = 170
PROMPT_PER_TOKEN_MIB = 0.66
GAP_MIB = {"intel": 471}
GAP_OTHER_MIB = 250
MIN_CARD_GB = 5.0                       # under this nothing here is claimed (the engine lowers its own reserve there, #496)

BASE_MIB = {"gyro-m": 5945, "gyro-s": 4720}
BASE_GENERIC_MIB = 2448

# Starts with request slots.  The generic base above is deliberately small (a default that fits today is never changed), so it cannot tell whether two more
# sessions leave the expert cache room; for Qwen3.8-Flash-Next the engine's own figures can.  Arc A750, IQ3_XXS with the MTP layer, 2 slots, 8192 tokens, the
# engine log: 7721 MiB usable (8 GB less the 471 MiB gap), 2519 MiB free after everything but the expert cache, which held the weights (1466 + 1781 MiB), the
# MTP layer (802), the Q5_K head (417), the main session (234) and the two slot sessions (468):  base = 7721 - 2519 - 234 - 468 = 4500 MiB.  The cache's
# largest expert blob is 2.33 MB (2.22 MiB).  Another quant or card holds somewhat different weights; MARGIN_MIB and the engine's own refusal cover the rest.
BASE_SLOTTED_MIB = {"flash-next": 4500}
SLOT_SLOTTED_MIB = 2.22
CTX_STEPS_SLOTTED = CTX_STEPS + (4096, 2048)     # below 8K only for a start with request slots: each slot's session is one more KV cache to pay for
SLOTS_MAX = 8                                    # the engine's batch window (serve/server.py PARALLEL_MAX)


def base_mib(model: str) -> int:
    low = str(model or "").lower().replace("_", "-")
    for key, mib in BASE_MIB.items():
        if key in low:
            return mib
    return BASE_GENERIC_MIB


def slotted_base_mib(model: str) -> int:
    """The base for a start with request slots: the measured Flash-Next figure, else the single-request one."""
    low = str(model or "").lower().replace("_", "-")
    for key, mib in BASE_SLOTTED_MIB.items():
        if key in low:
            return mib
    return base_mib(model)


def known_model(model: str) -> bool:
    low = str(model or "").lower().replace("_", "-")
    return any(k in low for k in BASE_MIB)


def session_mib(ctx: int, kv: str = "int8") -> float:
    return SESSION_FIXED_MIB + ctx * 13 * KV_BYTES.get(kv, 1056) * KV_MEASURED_FACTOR / 2 ** 20


def chunk_of(prefill) -> int:
    """The prompt chunk in tokens: a number as given, `auto` (or nothing, or auto:N) is what auto came to on the 8 GB card."""
    try:
        return max(1, int(str(prefill)))
    except (TypeError, ValueError):
        return AUTO_CHUNK


def slack_mib(model: str, vram_gb: float, ctx: int, kv: str = "int8", reserve_mib: int = 300, prefill="auto",
              vendor: str = "intel", slots: int = 0) -> float:
    """MiB the estimate leaves for the expert cache (negative: the engine would stop with "no VRAM is left for the expert cache").
    `slots` >= 2: a start with that many request slots, each one a session of the main one's size."""
    gap = GAP_MIB.get(str(vendor or "").lower(), GAP_OTHER_MIB)
    prompt = PROMPT_BASE_MIB + PROMPT_PER_TOKEN_MIB * chunk_of(prefill)
    base = base_mib(model)
    extra = 0.0
    if slots >= 2:
        base = slotted_base_mib(model)
        extra = min(int(slots), SLOTS_MAX) * session_mib(ctx, kv)
    return (vram_gb * 1024 - gap - base - session_mib(ctx, kv) - extra - HEAD_MIB - max(int(reserve_mib), 0) - prompt)


def needed_mib(model: str = "", slots: int = 0) -> float:
    if slots >= 2 and slotted_base_mib(model) != base_mib(model):
        return MIN_SLOTS * SLOT_SLOTTED_MIB + MARGIN_MIB
    return MIN_SLOTS * SLOT_MIB + MARGIN_MIB


def fits(model: str, vram_gb: float, ctx: int, kv: str = "int8", reserve_mib: int = 300, prefill="auto", vendor: str = "intel", slots: int = 0) -> bool:
    return slack_mib(model, vram_gb, ctx, kv, reserve_mib, prefill, vendor, slots) >= needed_mib(model, slots)


def ctx_label(ctx: int) -> str:
    return f"{ctx // 1024}K" if ctx % 1024 == 0 else f"{ctx} tokens"


@dataclass
class Fit:
    ctx: int
    reserve_mib: int
    prefill: str | None = None          # a new chunk size to write, or None to leave --prefill as it was
    changed: bool = False
    fits: bool = True
    slack_mib: float = 0.0
    line: str = ""                      # one plain sentence for the user, "" when nothing changed
    changes: list = field(default_factory=list)
    slots: int = 0                      # the request slots this start keeps (plan(slots=N)): N, or 1 when not even the smallest context leaves the cache room


def plan(model: str, vram_gb: float, ctx: int, kv: str = "int8", reserve_mib: int = 300, prefill="auto", vendor: str = "intel",
         title: str = "", ctx_fixed: bool = False, reserve_fixed: bool = False, prefill_fixed: bool = False, slots: int = 0) -> Fit:
    """The start that fits, as close to the one asked for as possible; the SAME values back when that one already fits (byte-identical).

    Order of harm, mildest first: the context down the ladder 32K -> 16K -> 8K (the outer loop), at each step the prompt chunk (1024)
    and then the reserve (down to RESERVE_FLOOR_MIB); a value the user chose (`*_fixed`) is not touched. Never refuses: when nothing on the
    ladder fits, the smallest start is returned with fits=False and a line that says so."""
    ctx = int(ctx)
    reserve_mib = int(reserve_mib)
    slots = int(slots) if slots and int(slots) >= 2 else 0
    keep = Fit(ctx, reserve_mib, None, False, True, 0.0, slots=slots)
    if not vram_gb or vram_gb < MIN_CARD_GB:
        return keep
    need = needed_mib(model, slots)
    keep.slack_mib = slack_mib(model, vram_gb, ctx, kv, reserve_mib, prefill, vendor, slots)
    if keep.slack_mib >= need:
        return keep
    steps = CTX_STEPS_SLOTTED if slots else CTX_STEPS
    ctxs = [ctx] if ctx_fixed else [ctx] + [c for c in steps if c < ctx]
    chunks = [prefill] if prefill_fixed else [prefill] + [c for c in CHUNK_STEPS if c < chunk_of(prefill)]
    reserves = [reserve_mib] if reserve_fixed or reserve_mib <= RESERVE_FLOOR_MIB else [reserve_mib, RESERVE_FLOOR_MIB]
    best = None
    for c in ctxs:
        for p in chunks:
            for r in reserves:
                s = slack_mib(model, vram_gb, c, kv, r, p, vendor, slots)
                best = (c, p, r, s)
                if s >= need:
                    out = _result(model, vram_gb, title, ctx, reserve_mib, prefill, c, p, r, s, True)
                    out.slots = slots
                    if slots and out.changed:
                        out.line = out.line.replace(" so it fits", f" so {slots} requests at once fit")
                    return out
    c, p, r, s = best
    out = _result(model, vram_gb, title, ctx, reserve_mib, prefill, c, p, r, s, False)
    out.slots = slots
    if slots and not ctx_fixed:      # not even the smallest context leaves the expert cache room for the request slots: one at a time, said plainly
        name = title or "This model"
        out.slots = 1
        out.changed = True
        out.line = (f"{name} on this {vram_gb:.0f} GB card: {slots} requests at once need more memory than is free here, even with the context at "
                    f"{ctx_label(c)}, so this start answers one request at a time")
    return out


def _result(model, vram_gb, title, ctx0, reserve0, prefill0, ctx, chunk, reserve, slack, ok) -> Fit:
    changes = []
    if ctx != ctx0:
        changes.append(f"context set to {ctx_label(ctx)}")
    if str(chunk) != str(prefill0):
        changes.append(f"prompts read in chunks of {chunk}")
    if reserve != reserve0:
        changes.append(f"memory kept free for other programs {reserve} MiB")
    name = title or "This model"
    card = f"{vram_gb:.0f} GB card"
    if ok:
        line = f"{name} on this {card}: " + ", ".join(changes) + " so it fits"
    else:
        line = (f"{name} on this {card}: " + (", ".join(changes) + ", but " if changes else "")
                + "it is still tight (the expert cache will get few slots); a smaller model or a card with more memory runs it better")
    return Fit(ctx, reserve, str(chunk) if str(chunk) != str(prefill0) else None, bool(changes), ok, slack, line, changes)


def fit_start(cfg: dict, args: list, vram_mib: float | None, backend: str | None) -> tuple[list, int | None, str]:
    """The start the config asks for, when it has request slots ("parallel" 2 or more, the app's "Concurrent requests"): (args, parallel, note).
    The slot sessions are taken out of the expert cache's room (plan(slots=N)); when that leaves too little, the context for this start is lowered to the
    largest that fits (the same ladder, then 4K and 2K), and when not even that fits the start runs one request at a time.  The person's own context
    (``--max-context`` in user_args) is kept: the engine's plain refusal answers it.  `parallel` is the effective count (None: the config's, untouched).
    Nothing changes without a single card's memory (several cards, a shared-memory chip: None) or for a layer split / DeepSeek, which size their slots themselves."""
    n = cfg.get("parallel")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2 or not vram_mib or backend in (None, "", "metal"):
        return args, None, ""
    if (cfg.get("arch") or "") == "deepseek41" or cfg.get("layer_split") or "--layer-split" in args or "--batch" in args or "--slots" in args:
        return args, None, ""

    def flag(name):
        return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None

    ctx_s = flag("--max-context")
    ctx = int(ctx_s) if ctx_s and str(ctx_s).isdigit() else 131072
    reserve_s = flag("--vram-reserve-mib")
    reserve = int(reserve_s) if reserve_s and str(reserve_s).isdigit() else 700      # the engine's default reserve
    user = [str(a) for a in (cfg.get("user_args") or [])]
    model = str(cfg.get("model_name") or flag("--native") or "")
    fr = plan(model, float(vram_mib) / 1024.0, ctx, flag("--kv") or "int8", reserve, flag("--prefill") or "auto", "intel" if backend == "sycl" else "other",
              title=str(cfg.get("model_name") or "This model"), ctx_fixed="--max-context" in user, reserve_fixed=True, prefill_fixed=True, slots=n)
    if not fr.changed:
        return args, None, ""
    out = list(args)
    if fr.ctx != ctx:
        if "--max-context" in out:
            out[out.index("--max-context") + 1] = str(fr.ctx)
        else:
            out += ["--max-context", str(fr.ctx)]
    return out, (fr.slots if fr.slots != n else None), fr.line


def fit_context(model: str, vram_gb: float, ctx: int, kv: str = "int8", reserve_mib: int = 300, vendor: str = "intel") -> int:
    """The context to recommend: `ctx` when it fits (unchanged), else the biggest step of the ladder that does (the reserve and
    chunk lowered the way plan() lowers them), else the smallest step."""
    return plan(model, vram_gb, ctx, kv, reserve_mib, "auto", vendor).ctx
