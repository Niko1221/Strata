"""The conversation cache budget on a PC with a lot of RAM (#7044).

The engine sizes the RAM that holds parked conversations itself when no `--conversation-cache-mib` is given: 40 % of the RAM that is
free once the model is loaded, at most 8192 MiB (engine/include/strata/core/conversation_memory.hpp, #3493).  One conversation above
about 490,000 tokens (about 16.7 KB a token on Qwen) never fits in 8 GiB, whatever the RAM, so an agent session that long reads its whole
prompt again on every turn.

This module lifts that cap on a PC that has the RAM, without an engine change: the same rule with the cap replaced by a share of the PC's
RAM, passed as the engine's own `--conversation-cache-mib` at the start (serve/server.py engine_args).  It only ever RAISES the budget:
whatever it cannot show to be above the engine's 8 GiB, it leaves to the engine (no flag), so a PC of 32 GiB or less, a model that fills the RAM,
an unreadable RAM figure and every explicit setting of the person behave as before.

  budget = min( 40 % of (free RAM now - the model's size - slack - the RAM floor),  25 % of the RAM of the PC,  128 GiB )

The free RAM is read before the model is loaded, so the model's file size (the most it can take) is taken off it; the engine's own gate
(`--conversation-cache-min-free-mib`, 2560 MiB) still refuses a park that would leave less than the floor free, whatever the budget.  Inside a
container the limit of its memory counts as the RAM (strata_control/cgroup_memory.py).
"""
from __future__ import annotations

import os
from pathlib import Path

ENGINE_AUTO_CAP_MIB = 8192        # the engine's own cap: nothing at or below it is worth a flag
SHARE_OF_FREE = 0.40              # the engine's share of the RAM left above its floor
SHARE_OF_TOTAL = 0.25             # never more than this share of the PC's RAM
CEILING_MIB = 131072              # 128 GiB
MODEL_SLACK_MIB = 4096            # K/V, scratch and the OS' share, beside the model file, when the engine's RAM use is estimated
DEFAULT_FLOOR_MIB = 2560          # the engine's --conversation-cache-min-free-mib default
STEP_MIB = 256


def budget_mib(total_mib: float, free_mib: float, model_mib: float, floor_mib: float = DEFAULT_FLOOR_MIB) -> int | None:
    """The budget in MiB when it is above what the engine picks on its own, else None (leave it to the engine)."""
    if total_mib <= 0 or free_mib <= 0:
        return None
    left = free_mib - max(0.0, model_mib) - MODEL_SLACK_MIB - max(0.0, floor_mib)
    if left <= 0:
        return None
    mib = int(min(left * SHARE_OF_FREE, total_mib * SHARE_OF_TOTAL, CEILING_MIB))
    mib -= mib % STEP_MIB
    return mib if mib > ENGINE_AUTO_CAP_MIB else None


def _value(args: list, name: str) -> str | None:
    for i in range(len(args) - 2, -1, -1):
        if args[i] == name:
            return str(args[i + 1])
    return None


def model_mib(args: list) -> float:
    """The size of the model file named by `--native`, in MiB (the most RAM the model can take); 0 when it cannot be read."""
    path = _value(list(args), "--native")
    if not path:
        return 0.0
    try:
        return Path(path).stat().st_size / (1 << 20)
    except OSError:
        return 0.0


def wanted(args: list) -> bool:
    """False when the person set the cache themselves or turned conversation parking off (`--conversation-cache-mib`, `--prompt-cache 0`,
    `--conversation-cache-slots 0`): such a start is not touched."""
    if "--conversation-cache-mib" in args:
        return False
    for flag in ("--prompt-cache", "--conversation-cache-slots"):
        value = _value(args, flag)
        if value is not None and value.strip() in ("0", ""):
            return False
    return True


def host_ram_mib() -> tuple[float, float]:
    """(total, available) MiB of this PC's RAM, in a container its memory limit; (0, 0) when unknown."""
    try:
        from strata_control import split_engine
        total_gib, available_gib = split_engine.ram_gib()
        return total_gib * 1024.0, available_gib * 1024.0
    except Exception:  # noqa: BLE001 - an unreadable figure leaves the engine's own sizing
        return 0.0, 0.0


def args_with_budget(args: list, *, ram: tuple[float, float] | None = None, say=None) -> list:
    """`args` with `--conversation-cache-mib N` added when N is above the engine's 8 GiB and the start did not set it; else `args` as they are.
    `ram` is (total, available) MiB (tests; default: this PC).  `say` gets the one line that tells the person what was set."""
    if os.environ.get("STRATA_NO_AUTO_CONVERSATION_CACHE") == "1" or not wanted(list(args)):
        return args
    total, free = ram if ram is not None else host_ram_mib()
    floor = _value(list(args), "--conversation-cache-min-free-mib")
    try:
        floor_mib = float(floor) if floor is not None else DEFAULT_FLOOR_MIB
    except ValueError:
        floor_mib = DEFAULT_FLOOR_MIB
    mib = budget_mib(total, free, model_mib(args), floor_mib)
    if mib is None:
        return args
    if say:
        say(f"conversation cache: {mib} MiB of this PC's {total / 1024:.0f} GiB of RAM (the engine alone stops at {ENGINE_AUTO_CAP_MIB} MiB); "
            "the engine still keeps its RAM floor free. `--conversation-cache-mib N` in the run config's args sets it, 0 turns it off")
    return [*args, "--conversation-cache-mib", str(mib)]
