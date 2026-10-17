"""serve/arena_budget.py - a cap on the expert arena's RAM, as a share of this PC's memory ("arena_budget").

The engine sizes the expert arena from what the machine has free at start (generate.cpp).  On a 64 GB PC with a 68 GB
model it asks for ~33 GiB, and when other programs hold the rest the log says so and keeps going:

    WARNING: the expert arena needs 33.02 GiB of RAM but 26.21 GiB is available (6.81 GiB short):
             the load may swap or be stopped by the OS

What a user needs there is a cap - but a fixed number is the wrong shape: 28 GiB is generous on a 32 GB PC, wasteful
on a 128 GB one, and wrong again the moment the PC is used for something else beside the model.  So the key is a
**share** of physical RAM and the GiB are computed here, at every start, from the RAM this PC actually has:

    "arena_budget": 0.45                                        # the shorthand: a share of physical RAM
    "arena_budget": {"share": 0.45, "min_gib": 6, "headroom_gib": 8}
    "arena_budget": "auto"                                      # no cap, the engine decides (the default)

Defaults and range:  share 0.45 (allowed 0.10-0.85), min_gib 6, headroom_gib 8.
The cap is  clamp(share x total RAM, min_gib, total RAM - headroom_gib),  so it follows the hardware up and down, and
the two ends are what a scenario moves: `headroom_gib` is the RAM the rest of the PC keeps (a game, a modelling app, a
browser), `min_gib` is the floor below which the arena stops paying for itself.

A cap is passed to the engine as its own `--resident-budget-gib N`, which implies `--mmap-experts` (generate.cpp): the
N GiB of experts the profile ranks hottest are copied into RAM at start, the rest are read from the files.  That is a
different expert path from the default pinned arena, so this key is **opt-in** and an absent key leaves the engine's
arguments exactly as they were.  A `--resident-budget-gib` already in the config's "args" is never overridden: the
number written by hand wins, and the cap stays out of the way.

When it takes effect: the server reads the config at start and builds the engine's arguments from it, so a change is
used from the next start of the model - no other key is touched and nothing is written back to the file.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

SHARE_DEFAULT = 0.45      # a share of physical RAM, when the key is on and gives no number
SHARE_MIN = 0.10          # below this the arena is too small to hold a useful part of the experts
SHARE_MAX = 0.85          # above this nothing is left for the rest of the PC, which is what the cap exists to avoid
MIN_GIB_DEFAULT = 6.0     # the floor, unless the config sets "min_gib"
HEADROOM_GIB_DEFAULT = 8.0   # the RAM the rest of the PC keeps, unless the config sets "headroom_gib"
MIN_GIB_MIN = 1.0
HEADROOM_GIB_MIN = 0.0

# the engine's option (generate.cpp): N GiB of experts in RAM, the rest read from the files.  Parsing it also sets
# mmap_experts / resident_cpu_experts / resident_pin (generate.cpp:2223), so --mmap-experts is never added here: it
# would be redundant at best and is refused beside some pack layouts at worst.
BUDGET_FLAG = "--resident-budget-gib"


@dataclass(frozen=True)
class Plan:
    """One start's cap: the settings it came from, the RAM it was measured against, and the whole GiB it became."""
    share: float
    min_gib: float
    headroom_gib: float
    total_gib: float
    gib: int
    clamped: str          # "", "share" (the share decided it), "floor" (min_gib), "ceiling" (total - headroom)

    @property
    def args(self) -> list[str]:
        """The engine's arguments for this cap.  --resident-budget-gib alone: the engine sets mmap_experts itself."""
        return [BUDGET_FLAG, str(self.gib)]

    def describe(self) -> str:
        """One line for the server window: the share, the RAM it was measured on, the GiB, and what moved it."""
        why = {"floor": f"raised to \"min_gib\" {self.min_gib:g}",
               "ceiling": f"held to total RAM less \"headroom_gib\" {self.headroom_gib:g}",
               }.get(self.clamped, "")
        return (f"arena_budget: {self.share:.2f} x {self.total_gib:.1f} GiB of RAM = {self.gib} GiB for the expert "
                f"arena" + (f" ({why})" if why else "") +
                f" -> {BUDGET_FLAG} {self.gib}; the rest of the PC keeps "
                f"{max(0.0, self.total_gib - self.gib):.1f} GiB")


def total_ram_gib() -> float | None:
    """This PC's physical RAM in GiB, or None when it cannot be read (the cap is then skipped, never guessed)."""
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except Exception:  # noqa: BLE001 - psutil is optional here, as in warn_budget_over_ram
        pass
    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MemoryStatusEx()
            st.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None
            return st.ullTotalPhys / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:  # noqa: BLE001
        return None


def _number(key: str, v, lo: float, hi: float | None = None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"\"{key}\" must be a number, not {v!r}")
    if v < lo or (hi is not None and v > hi):
        rng = f"{lo:g}" if hi is None else f"{lo:g}-{hi:g}"
        raise ValueError(f"\"{key}\" must be {rng}, not {v:g}")
    return float(v)


def settings(raw) -> tuple[float, float, float] | None:
    """The (share, min_gib, headroom_gib) the config asks for, or None for "no cap".  ValueError on a bad value."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.strip().lower() in ("", "auto", "off", "none", "engine"):
            return None
        raise ValueError(f"expected a number, \"auto\" or an object, not {raw!r}")
    if raw is False:
        return None
    if raw is True:                                  # "arena_budget": true - the default share
        return SHARE_DEFAULT, MIN_GIB_DEFAULT, HEADROOM_GIB_DEFAULT
    if isinstance(raw, (int, float)):
        return _number("arena_budget", raw, SHARE_MIN, SHARE_MAX), MIN_GIB_DEFAULT, HEADROOM_GIB_DEFAULT
    if not isinstance(raw, dict):
        raise ValueError(f"expected a number, \"auto\" or an object, not {raw!r}")
    known = {"share", "min_gib", "headroom_gib"}
    unknown = [k for k in raw if k not in known]
    if unknown:
        raise ValueError(f"unknown key(s) {', '.join(sorted(unknown))}; expected {', '.join(sorted(known))}")
    share = _number("share", raw.get("share", SHARE_DEFAULT), SHARE_MIN, SHARE_MAX)
    min_gib = _number("min_gib", raw.get("min_gib", MIN_GIB_DEFAULT), MIN_GIB_MIN)
    headroom = _number("headroom_gib", raw.get("headroom_gib", HEADROOM_GIB_DEFAULT), HEADROOM_GIB_MIN)
    return share, min_gib, headroom


def plan(cfg: dict, total_gib: float | None = None) -> Plan | None:
    """The cap for this start, or None when the config asks for none (or the RAM cannot be read).
    ValueError when the config's value is not one this module accepts."""
    got = settings(cfg.get("arena_budget"))
    if got is None:
        return None
    share, min_gib, headroom = got
    total = total_ram_gib() if total_gib is None else total_gib
    if not total or total <= 0:
        return None
    ceiling = total - headroom
    if ceiling < MIN_GIB_MIN:
        return None                                  # a PC this small has no room for a cap: let the engine choose
    gib = share * total
    clamped = "share"
    if gib < min_gib and min_gib <= ceiling:
        gib, clamped = min_gib, "floor"
    if gib > ceiling:
        gib, clamped = ceiling, "ceiling"
    whole = int(math.floor(gib))                     # never above the cap, whatever the rounding
    if whole < 1:
        return None
    return Plan(share=share, min_gib=min_gib, headroom_gib=headroom, total_gib=total, gib=whole, clamped=clamped)


def args_for(cfg: dict, args: list[str], total_gib: float | None = None, out=None) -> list[str]:
    """`args` with the cap added, or `args` unchanged: no key, "auto", a bad value (said once, then ignored), RAM that
    cannot be read, or a --resident-budget-gib already written by hand."""
    if BUDGET_FLAG in args:
        return args                                  # the number in the config's "args" wins: never overridden
    stream = sys.stdout if out is None else out
    try:
        p = plan(cfg, total_gib)
    except ValueError as e:
        said = f"bad: {e}"
        if said not in _SAID:                        # engine_args runs twice at a start: say it once
            _SAID.add(said)
            print(f"[strata] arena_budget: {e} - ignored, the engine sizes the arena itself", file=stream, flush=True)
        return args
    if p is None:
        return args
    line = p.describe()
    if line not in _SAID:
        _SAID.add(line)
        print("[strata] " + line, file=stream, flush=True)
    return list(args) + p.args


_SAID: set[str] = set()


def reset() -> None:
    """Forget the lines already printed (tests; a restart in the same process says the cap again)."""
    _SAID.clear()
