"""DeepSeek-V4.1-Flash `--parallel N`: several conversations answered at the same time.  What it does, what was measured, and what has not been run.

Off by default: no flag means requests take turns, exactly as before.  The engine side is `strata-dsv41 --parallel N` (src/dsv41, from Project Maya's
#332): N conversations each with their own attention memory, decoded together in one pass.  Strata's server maps the config key "parallel" to it
(serve/server.py parallel_args); `strata launch --parallel N`, a saved preset, the app's Expert setting "Concurrent requests" and `strata serve
--parallel N` all end there.

Pure functions, no I/O.  Recommend, never force: nothing here refuses a number or a computer; the engine itself runs fewer conversations when the
graphics memory holds fewer (ds4_model.cu alloc_seqs, said in its log), and 8 is the most it takes.  Numbers only, no verdict words: they are
measured figures and the person decides.
"""

from __future__ import annotations


MAX_CONVERSATIONS = 8      # the engine's limit (ds4_main.cpp: --parallel takes 1 to 8)

# Graphics memory one extra conversation keeps (its own window rings, compressor state and KV index), in GB: a + b * context tokens.
# Measured from the engine's own line "ds4: --parallel N: ... (X GB more VRAM for the other N-1 ...)" on 2 RTX A6000, UD-IQ1_M pack:
# 2 conversations at 8K 0.01 GB, 32K 0.05, 128K 0.17, 512K 0.68; 4 at 128K 0.52 for the other 3; 8 at 128K 1.20 for the other 7, at 1M 9.47 for the other 7.
EXTRA_GB_BASE = 0.0
EXTRA_GB_PER_TOKEN = 1.3e-6

# Where it was run (phaseA-tests/maya-port/INT/RESULTS.md, PAR/RESULTS.md).  Anything else runs the same code and was not tried.
MEASURED_ON = ("4 Radeon AI PRO R9700 (32 GB), 1 R9700 and 2 RTX A6000 (48 GB) and a DGX Spark, all on Linux")
TESTED_BACKENDS = ("CUDA sm_86", "CUDA sm_121", "HIP gfx1201")

SPEED_LINES = (
    "Several long answers at once: 4 together wrote 73 tokens a second in all against 41 for one answer alone, each of the four at 21 "
    "(4 R9700); 2 together 53 against 37, each at 30 (4 R9700), and 52 against 38 on 2 RTX A6000 (each at 31). The median wait for the "
    "first word was 0.9 s instead of 8.0 s with 4 sessions (4 R9700).",
    "An agent load of 4 sessions reading files and answering 40 tokens: the whole load took 13% less time on 4 R9700 (55-56 s against 64-65 s) "
    "and 11% less on 2 RTX A6000 (91 s against 102 s); a DGX Spark took 17-20% less (205-216 s against 256-261 s). One conversation alone ran as fast with the flag as without (a five-turn chat took "
    "23.3-24.9 s against 23.4-24.4 s on 4 R9700, 34.9-35.9 s against 35.2-35.6 s on 2 RTX A6000).",
)


def clamp(n: int | None) -> int | None:
    """The number of conversations the engine will run for a request of `n` (None when not asked or 1 or less: one at a time)."""
    if n is None or int(n) <= 1:
        return None
    return min(int(n), MAX_CONVERSATIONS)


def extra_gb(context: int | None) -> float | None:
    """Graphics memory one extra conversation keeps at this context size, in GB (None when the model has no figure yet)."""
    if not context or (EXTRA_GB_BASE == 0.0 and EXTRA_GB_PER_TOKEN == 0.0):
        return None
    return EXTRA_GB_BASE + EXTRA_GB_PER_TOKEN * float(context)


def _label(tokens: int | None) -> str:
    if not tokens:
        return "the default context"
    if tokens >= 1048576 and tokens % 1048576 == 0:
        return f"{tokens // 1048576}M"
    return f"{tokens // 1024}K" if tokens % 1024 == 0 else f"{tokens}"


def explain(n: int, context: int | None) -> list[str]:
    """What `--parallel n` does, in plain words with the measured numbers.  Empty when it is off."""
    k = clamp(n)
    if k is None:
        return []
    gb = extra_gb(context)
    cost = (f" Each extra conversation keeps its own attention memory in graphics memory: about {gb:.2f} GB more at {_label(context)}, "
            f"{(k - 1) * gb:.2f} GB for the other {k - 1}, taken from the room the experts would use.") if gb is not None else (
        " Each extra conversation keeps its own attention memory in graphics memory, taken from the room the experts would use; the engine "
        "prints the amount at start.")
    return [f"{k} conversations are answered at the same time (several chats, agents or sub-agents); without --parallel requests take turns."
            f" One answer is slower while the others run.{cost}",
            *SPEED_LINES,
            f"Measured on {MEASURED_ON}."]


def warnings(n: int | None, *, cards: list[dict] | None, context: int | None = None, os_name: str | None = None, machine: str | None = None,
             vendor: str | None = None, unified: bool = False) -> list[tuple[str, str]]:
    """(what, what to do) for a pick that goes beyond what was run: a warning, never a refusal.  `cards` are the usable cards of the run;
    `os_name` is sys.platform, `machine` sysinfo.machine().  Nothing for an unset or one-at-a-time pick."""
    out: list[tuple[str, str]] = []
    if n is None or int(n) <= 1:
        return out
    n = int(n)
    if n > MAX_CONVERSATIONS:
        out.append((f"--parallel {n} is more than the engine runs at once; it will use {MAX_CONVERSATIONS}.",
                    f"Pick a number from 2 to {MAX_CONVERSATIONS}."))
    os_name = (os_name or "").lower()
    machine = (machine or "").lower()
    mac = os_name == "darwin" or vendor == "apple" or any(str(c.get("vendor")) == "apple" for c in cards or [])
    windows = os_name.startswith("win")
    uma_gpu = any(c.get("uma") or c.get("integrated") for c in cards or [])
    spark = os_name.startswith("linux") and machine in ("aarch64", "arm64")
    if mac:
        out.append(("--parallel was not run on a Mac (Metal). The code adds no Metal kernel of its own, but nothing here has started it there.",
                    "If the engine stops at the start, launch without --parallel."))
    elif windows:
        out.append(("--parallel was run on Linux only. The Windows engine builds with it; a Windows run has not been done.",
                    "If the engine stops at the start or an answer looks wrong, launch without --parallel."))
    elif uma_gpu and not spark:
        out.append(("--parallel was not run on a Ryzen AI Max (shared memory) PC. Every extra conversation takes memory the experts and the system also use.",
                    "If the engine stops or slows down, launch without --parallel."))
    else:
        archs = {str(c.get("arch") or "") for c in cards or [] if c.get("arch")}
        odd = sorted(a for a in archs if a and not a.startswith("gfx12"))
        if str(vendor or "") == "amd" and odd:
            out.append((f"--parallel was run on Radeon AI PRO R9700 cards (gfx1201); this PC has {', '.join(odd)}, which was not tried.",
                        "If the engine stops at the start, launch without --parallel."))
    small = [c for c in cards or [] if 0 < float(c.get("vram_gb") or 0) < 24 and not (c.get("uma") or c.get("integrated"))]
    if small and not mac:
        c = small[0]
        gb = extra_gb(context)
        share = f" (about {gb:.2f} GB each at {_label(context)})" if gb is not None else ""
        out.append((f"{c.get('name') or 'A card'} has {float(c.get('vram_gb') or 0):.0f} GB; --parallel was run on cards with 32 GB and 48 GB. "
                    f"Each extra conversation takes graphics memory{share} that the experts would use, so fewer experts stay on the card and answers slow down; "
                    "if the buffers do not fit, the engine runs fewer conversations and says so in its log.",
                    "A smaller --context leaves more room."))
    return out
