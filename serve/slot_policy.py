"""The placement policy interface for STRATA_SLOT_CACHE_AFFINITY=1: what serve/server.py (pick_slot, _claim_park,
slots_view) and serve/cache_trace.py call.  No policy is here - which idle slot a request runs in, which
history is given up, how long one is protected and where the solo path's conversation is parked are a policy's
choice.  Without one, STRATA_SLOT_CACHE_AFFINITY=1 and STRATA_CACHE_TRACE=1 raise NotImplementedError; with both
off nothing here runs (configure() and settings() aside)."""
from __future__ import annotations

from dataclasses import dataclass

PARK_MIN_TOKENS = 8192       # a shorter main-session conversation is not parked


def configure(env) -> dict:
    """Called by StrataEngine with the engine's environment: read the policy's tunables, return settings()."""
    return settings()


def settings() -> dict:
    """The tunables in force (/metrics live.slot_policy)."""
    return {"park_min_tokens": PARK_MIN_TOKENS}


def cache_state(tokens: int, last_used: float, now: float, valid: bool = True) -> dict:
    """An idle history of `tokens`, last used at `last_used`: a dict with tokens, ttl_s, age_s, expired, valid,
    retention, overwrite_score (/metrics live.slots and serve/cache_trace.py show it)."""
    raise NotImplementedError("serve/slot_policy.py has no placement policy")


@dataclass(frozen=True)
class Placement:
    slot: int | None
    source_hint: int | None = None
    reused_hint: int = 0
    near_tokens: int = 0
    reason: str = "busy"


def reliable_flags(busy: list[bool], valid: list[bool] | None, group_size: int, padding_safe: bool) -> list[bool]:
    """Whether each slot's held tokens are still a usable cache.  Without the native active-row engine, an idle row
    below a busy row of its group is a pad row: its state is being overwritten."""
    gs = max(1, group_size)
    validity = valid if valid is not None else [True] * len(busy)
    return [bool(validity[b]) and (padding_safe or not any(busy[x] for x in range(b + 1, b - b % gs + gs)))
            for b in range(len(busy))]


def superseded_slots(held: list[list[int]], busy: list[bool], reliable: list[bool]) -> set[int]:
    """Idle histories that are the start of another idle slot's history: the conversation moved on in that other
    slot, so this copy saves nothing it does not.  They count as empty (no protection)."""
    out = set()
    for b, h in enumerate(held):
        if h and not busy[b]:
            for y, other in enumerate(held):
                if (y != b and not busy[y] and reliable[y] and
                        (len(other) > len(h) or (len(other) == len(h) and y < b)) and other[:len(h)] == h):
                    out.add(b)
                    break
    return out


def park_destination(main: list[int], busy: list[bool], held: list[list[int]], used: list[float],
                     order: list[int], group_size: int, now: float, valid: list[bool] | None = None,
                     padding_safe: bool = False, exclude=()) -> int | None:
    """The idle slot to keep the main session's conversation `main` in before another request replaces it (the
    engine's park=B), never one in `exclude`; None: do not park."""
    raise NotImplementedError("serve/slot_policy.py has no placement policy")


def place(prompt: list[int], busy: list[bool], held: list[list[int]], used: list[float],
          order: list[int], group_size: int, now: float, valid: list[bool] | None = None,
          prefill_rate: float | None = None, padding_safe: bool = False) -> Placement:
    """The idle slot `prompt` runs in (the caller holds slot_cv).  busy/held/used/valid: each slot's state (held: the
    tokens it holds, used: when it was last used, valid: the server's knowledge that its state is intact); order: the
    slots in the order that spreads requests over the pipeline groups; group_size: slots per group; prefill_rate: the
    observed prompt rate (tok/s); padding_safe: the engine runs a group's active slots only (INFO
    slot_padding_safe=1).  A busy slot is never returned; Placement(None) only when every slot is busy."""
    raise NotImplementedError("serve/slot_policy.py has no placement policy")
