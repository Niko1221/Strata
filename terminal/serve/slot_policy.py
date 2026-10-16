"""Cache retention and pipelined placement policy for STRATA_SLOT_CACHE_AFFINITY=1; no engine or I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass

MAX_CACHE_TOKENS = 262144
# Tunables (environment, read by configure() when the engine starts; the menu sets them per profile under
# config_overrides.env).  Defaults are the values the policy was specified with.
EXPIRED_KEEP_SCORE = 0.01    # STRATA_SLOT_EXPIRED_KEEP_SCORE: an expired history scoring at least this (retention at
                             # expiry, then decayed; 0.01 is about 2.5K tokens at expiry) goes last
PARK_VALUE_RATIO = 2.0       # STRATA_SLOT_PARK_VALUE_RATIO: a park takes an expired history worth keeping only when the
                             # main session's conversation is worth at least this many times more; 0: never
IDLE_VALUE_ZERO_S = 1800.0  # STRATA_SLOT_IDLE_VALUE_ZERO_S: idle age when expired value reaches zero; 0 disables decay
PARK_KEPT_SHARE = 0.98       # STRATA_SLOT_PARK_KEPT_SHARE: a slot holding at least this share of the main session's
                             # conversation already keeps it (no park); the rest is read again when it comes back
PARK_MIN_VALUE = 0.01        # STRATA_SLOT_PARK_MIN_VALUE: the main session's conversation is parked only while its
                             # value (stale_retention at its idle age) is at least this; 0.01 is EXPIRED_KEEP_SCORE's
                             # default
TTL_MAX_S = 120.0            # STRATA_SLOT_TTL_MAX_S: protection at 262144 tokens; the whole curve scales with it
_TTL_SHAPE = ((0, 0.0), (500, 0.1), (2000, 2.0), (8000, 10.0),
              (32000, 45.0), (50000, 65.0), (100000, 90.0), (MAX_CACHE_TOKENS, 120.0))
TTL_POINTS = _TTL_SHAPE


def configure(env) -> dict:
    """Read the tunables from `env` (the engine's environment); bad or missing values keep the defaults."""
    global TTL_MAX_S, TTL_POINTS, EXPIRED_KEEP_SCORE, IDLE_VALUE_ZERO_S, \
        PARK_VALUE_RATIO, PARK_MIN_VALUE, PARK_KEPT_SHARE

    def value(name, default):
        try:
            v = float((env or {}).get(name, default))
            return v if math.isfinite(v) and v >= 0 else default
        except (TypeError, ValueError):
            return default
    TTL_MAX_S = value("STRATA_SLOT_TTL_MAX_S", 120.0)
    EXPIRED_KEEP_SCORE = value("STRATA_SLOT_EXPIRED_KEEP_SCORE", 0.01)
    IDLE_VALUE_ZERO_S = value("STRATA_SLOT_IDLE_VALUE_ZERO_S", 1800.0)
    TTL_POINTS = tuple((n, s * TTL_MAX_S / 120.0) for n, s in _TTL_SHAPE)
    PARK_VALUE_RATIO = value("STRATA_SLOT_PARK_VALUE_RATIO", 2.0)
    PARK_MIN_VALUE = value("STRATA_SLOT_PARK_MIN_VALUE", 0.01)
    PARK_KEPT_SHARE = value("STRATA_SLOT_PARK_KEPT_SHARE", 0.98)
    return settings()


def settings() -> dict:
    return {"ttl_max_s": TTL_MAX_S, "expired_keep_score": EXPIRED_KEEP_SCORE,
            "idle_value_zero_s": IDLE_VALUE_ZERO_S, "park_value_ratio": PARK_VALUE_RATIO,
            "park_min_value": PARK_MIN_VALUE, "park_kept_share": PARK_KEPT_SHARE}


def cache_ttl(tokens: int) -> float:
    n = min(MAX_CACHE_TOKENS, max(0, tokens))
    for (lo, a), (hi, b) in zip(TTL_POINTS, TTL_POINTS[1:]):
        if n <= hi:
            return a + (b - a) * (n - lo) / (hi - lo)
    return TTL_POINTS[-1][1]


def stale_retention(tokens: int, age_s: float = 0.0) -> float:
    """Value at expiry times (1 - x*x), x = (age - TTL) / (IDLE_VALUE_ZERO_S - TTL), clamped to [0, 1].

    Age is time since last actual use, including protection. Loss per second increases until value reaches zero.
    Before expiry this is the projected value at expiry; zero disables decay. Protection takes precedence if a
    configured zero age is shorter than the TTL. Replacement priority never changes whether a history can match.
    """
    n = min(MAX_CACHE_TOKENS, max(0, tokens))
    base = (n / 50000.0) ** 0.7 * (n / (n + 2000.0)) ** 2 * 0.25
    ttl = cache_ttl(tokens)
    if IDLE_VALUE_ZERO_S == 0 or age_s < ttl:
        return base
    if age_s >= IDLE_VALUE_ZERO_S:
        return 0.0
    x = max(0.0, (age_s - ttl) / (IDLE_VALUE_ZERO_S - ttl))
    return base * (1.0 - x * x)


def cache_state(tokens: int, last_used: float, now: float, valid: bool = True) -> dict:
    age = max(0.0, now - last_used)
    ttl = cache_ttl(tokens)
    expired = tokens == 0 or age >= ttl or not valid
    # A 50K history has substantial value. The additional small-history factor makes
    # tiny probes much cheaper to replace without flattening all medium histories.
    n = min(MAX_CACHE_TOKENS, max(0, tokens))
    freshness = max(0.0, 1.0 - age / ttl) if ttl else 0.0
    retention = 0.0 if expired else (n / 50000.0) ** 0.7 * (n / (n + 2000.0)) ** 2 * (0.25 + 0.75 * freshness)
    return {"tokens": tokens, "ttl_s": ttl, "age_s": age, "expired": expired,
            "valid": valid, "retention": retention, "overwrite_score": 1.0 / (1.0 + retention)}


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
    slot, so this copy saves nothing it does not.  They count as empty (no protection, no wait)."""
    out = set()
    for b, h in enumerate(held):
        if h and not busy[b]:
            for y, other in enumerate(held):
                if (y != b and not busy[y] and reliable[y] and
                        (len(other) > len(h) or (len(other) == len(h) and y < b)) and other[:len(h)] == h):
                    out.add(b)
                    break
    return out


def _blank(held, busy, used, now, reliable):
    superseded = superseded_slots(held, busy, reliable)
    states = [cache_state(len(h), used[b], now, reliable[b] and b not in superseded) for b, h in enumerate(held)]
    # Nothing worth keeping: no tokens, known invalid, or a superseded copy.  An expired history counts as empty for
    # the protection rules, but it is still a cache (it can match until overwritten), so it is overwritten only after
    # these - the one with the lowest score first.
    blank = {b for b in range(len(held)) if not held[b] or not reliable[b] or b in superseded}
    return states, blank


def _slot_values(held, states, blank) -> dict[int, float]:
    """Each slot's current value: 0 for a blank slot, else its value decayed with its idle age."""
    return {b: 0.0 if b in blank else stale_retention(len(h), states[b]["age_s"]) for b, h in enumerate(held)}


def _worth_keeping(value: float) -> bool:
    return value >= EXPIRED_KEEP_SCORE and value > 0


def reclaim_slot(candidates, values, blank, rank, positions) -> int | None:
    """The reclaim order place() and park_destination() share, over idle expired slots (blank ones included): the
    slots not worth keeping (blank, or a value below EXPIRED_KEEP_SCORE) before histories worth keeping; then `rank`
    (the group's load); then the lowest value, a blank slot first at an equal value; then the slot order."""
    if not candidates:
        return None
    return min(candidates, key=lambda b: (_worth_keeping(values[b]), *rank(b), values[b], b not in blank,
                                          positions[b]))


def park_destination(main: list[int], busy: list[bool], held: list[list[int]], used: list[float],
                     order: list[int], group_size: int, now: float, valid: list[bool] | None = None,
                     padding_safe: bool = False, exclude=(), main_value: float = 0.0) -> int | None:
    """An idle slot to keep the main session's conversation in before another request replaces it (engine park=):
    its own older copy, else an expired slot in place()'s reclaim order (reclaim_slot), where a history worth keeping
    counts only when `main_value` (the main session's current value) is at least PARK_VALUE_RATIO times its own.
    Never a protected slot.  None: nowhere to keep it (or it is already kept)."""
    if not padding_safe or not main:
        return None
    gs = max(1, group_size)
    reliable = reliable_flags(busy, valid, gs, padding_safe)
    for b, h in enumerate(held):   # already kept (all but a short tail) - in an idle slot or one decoding it
        if (h and reliable[b] and len(h) >= PARK_KEPT_SHARE * len(main) and
                (main[:len(h)] == h or h[:len(main)] == main)):
            return None
    states, blank = _blank(held, busy, used, now, reliable)
    positions = {b: i for i, b in enumerate(order)}
    loads = {g: sum(busy[g:g + gs]) for g in range(0, len(busy), gs)}
    own = [b for b in order if not busy[b] and b not in exclude and held[b] and reliable[b] and
           len(held[b]) < len(main) and main[:len(held[b])] == held[b]]
    if own:
        return max(own, key=lambda b: (len(held[b]), -positions[b]))
    values = _slot_values(held, states, blank)
    candidates = [b for b in order if not busy[b] and b not in exclude and states[b]["expired"] and
                  (not _worth_keeping(values[b]) or
                   (PARK_VALUE_RATIO > 0 and main_value >= PARK_VALUE_RATIO * values[b]))]
    return reclaim_slot(candidates, values, blank, lambda b: (loads[b - b % gs],), positions)


def place(prompt: list[int], busy: list[bool], held: list[list[int]], used: list[float],
          order: list[int], group_size: int, now: float, valid: list[bool] | None = None,
          prefill_rate: float | None = None, padding_safe: bool = False) -> Placement:
    free = [b for b in order if not busy[b]]
    if not free:
        return Placement(None)
    gs = max(1, group_size)
    validity = valid if valid is not None else [True] * len(busy)
    # Conservative current-window check plus remembered native-line invalidations.
    reliable = reliable_flags(busy, validity, gs, padding_safe)
    states, blank = _blank(held, busy, used, now, reliable)
    matches = {b: len(held[b]) for b in free if reliable[b] and held[b] and
               len(held[b]) < len(prompt) and prompt[:len(held[b])] == held[b]}
    best = max(matches.values(), default=0)
    # Near = at most 2% of the best prefix, 1 second of observed prefill, and 4096 tokens.
    rate = prefill_rate if prefill_rate and math.isfinite(prefill_rate) and prefill_rate > 0 else 1000.0
    near = int(min(best * 0.02, rate, 4096))
    sources = [b for b, n in matches.items() if best - n <= near]
    loads = {g: sum(busy[g:g + gs]) for g in range(0, len(busy), gs)}
    positions = {b: i for i, b in enumerate(order)}

    def collateral(b):
        if padding_safe:
            return 0.0
        # Starting an upper row can pad over a lower idle protected history. A matching
        # ancestor is the same conversation, so its displacement does not lose another chat.
        return sum(states[x]["retention"] for x in range(b - b % gs, b)
                   if not busy[x] and x not in matches)

    def loss(b):
        return (0.0 if b in matches else states[b]["retention"]) + collateral(b)

    def pipeline(b):
        g0 = b - b % gs
        high = max([x - g0 for x in range(g0, g0 + gs) if busy[x]] + [-1])
        extra_rows = 1 if padding_safe else max(0, b - g0 - high)
        return loads[g0], extra_rows, positions[b]

    if sources:
        # Cache source and execution destination are separate: the native engine restores the longest valid
        # prefix from any idle slot, then copies the conversation into the destination (it copies even when they
        # are the same slot, so moving costs nothing extra).  Move only into a blank slot, and only for a less
        # loaded group; never over another conversation's history, expired or not (2026-10-14: two chats moved
        # into each other's expired slot on every turn and re-read 60-100K tokens each time).
        candidates = sources + [b for b in free if b not in sources and b in blank and collateral(b) == 0]
        slot = min(candidates, key=lambda b: (loss(b), *pipeline(b)[:2], b not in sources, positions[b]))
        source = max(sources, key=lambda b: (matches[b], -positions[b]))
        return Placement(slot, source, best, near, "reuse")

    safe = [b for b in free if states[b]["expired"] and collateral(b) == 0]

    values = _slot_values(held, states, blank)

    if safe:
        # Expired = empty for protection, but an expired history can still match until it is overwritten: one that
        # still scores EXPIRED_KEEP_SCORE goes only when no blank or worthless slot is free (else a new chat in the
        # idle group overwrote the other chat every turn - GPU run 2026-10-15).  Then group load; then the lowest score.
        slot = reclaim_slot(safe, values, blank, lambda b: pipeline(b)[:2], positions)
        return Placement(slot, reason="reclaim")
    # Every idle history is protected: no wait (it cost a free slot 5 s per request when requests were queued, and
    # ended in the same overwrite) - the history that loses least goes now.
    cheapest = min(loss(b) for b in free)
    # Loss scores within 10% are close enough to let group load/padding break the tie.
    candidates = [b for b in free if loss(b) <= cheapest * 1.10 + 1e-12]
    return Placement(min(candidates, key=lambda b: (*pipeline(b), loss(b), used[b])), reason="evict")
