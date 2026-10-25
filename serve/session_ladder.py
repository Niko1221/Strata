"""serve/session_ladder.py - the serve session ladder's decision core: pure functions, no I/O,
no imports from server.py.  It answers the ladder's three questions from bookkeeping alone:
which session an engine `conversation cache:` event belongs to (join_event), whether a session
should be saved to disk on switch-away (should_save), whether it should be restored on switch-back
(should_restore + prefix_fidelity), and which client header names the session (extract_session_id).

Event vocabulary as the engine writes it (ConvCacheLog's lines, parsed upstream into ConvEvent):
"conversation cache: parked N tokens in T ms; parked=P bytes=B evictions=E ...",
"conversation cache: skip parking (...)", "conversation cache: dropped ...", "conversation cache: restored ...".
`evictions` is a running global counter carried by every event; each join records the figure the event
carried into that session's evictions_at_last_switch (the engine's cache-wide counter at the last switch
touching the session), which is what the evictions-pressure signal compares against.
"""
from __future__ import annotations

from dataclasses import dataclass

# Pinned constants (measured / engine's own): park costs 40.4 KiB per token; the engine refuses to park
# below 8192 tokens ("skip parking (70 tokens, below the 8192 minimum; parked=4)").
PARK_COST_KIB_PER_TOKEN = 40.4
MIN_PARK_TOKENS = 8192

# Session identity headers, pinned first-wins precedence (OpenCode's native family; serve stamps them).
SESSION_HEADERS = ("x-session-id", "x-opencode-session-id", "x-session-affinity", "x-opencode-session")


@dataclass
class ConvEvent:
    """One parsed `conversation cache:` event line: kind is parked|skipped|restored|dropped,
    byte_count is the line's bytes=B figure, evictions is the engine's running eviction counter."""
    kind: str
    tokens: int
    byte_count: int
    evictions: int


@dataclass
class TrackedSession:
    """Ladder bookkeeping for one client session.  parked is the engine-side state as of the last
    joined event (cleared upstream at the session's next switch-away, so should_save is evaluated
    while it still reflects park).  byte_count is this session's half of the (tokens, byte_count)
    join key that join_event matches ConvEvent against - recorded by the intake layer from DONE
    figures / joined events; it trails the spec's eight fields so every field order stays positional."""
    session_id: str
    tokens: int
    parked: bool
    save_eligible: bool
    file_path: str | None
    saved_prefix: str | None
    last_active: float
    evictions_at_last_switch: int
    byte_count: int = 0


@dataclass
class LadderConfig:
    enabled: bool
    slot_save_path: str
    park_budget_mib: int
    idle_save_s: float = 8.0
    retention_max_files: int = 16


def extract_session_id(headers: dict) -> str | None:
    """The session id from OpenCode's session headers, first non-empty in SESSION_HEADERS order.
    Header names are matched case-insensitively (BaseHTTPRequestHandler's headers are too)."""
    lowered = {k.lower(): v for k, v in headers.items()}
    for name in SESSION_HEADERS:
        value = lowered.get(name)
        if value:
            return value
    return None


def join_event(ev: ConvEvent, tracked: dict) -> TrackedSession | None:
    """The tracked session an engine event belongs to, joined by (tokens, byte_count).
    None when nothing matches and - the honest case - when the key is ambiguous across sessions:
    the ladder refuses to guess, and the engine's fingerprint validation stays the backstop."""
    matches = [s for s in tracked.values() if s.tokens == ev.tokens and s.byte_count == ev.byte_count]
    return matches[0] if len(matches) == 1 else None


def should_save(s: TrackedSession, tracked: dict, cfg: LadderConfig) -> bool:
    """Save s to disk on switch-away when the engine lost - or could not hold - this conversation.
    False when the ladder is off, s is not save-eligible, or s is below the engine's park minimum
    (tiny aux calls must never churn).  Then any pressure signal carries it:
    - beyond budget: s can never park (tokens x 40.4 KiB > park_budget_mib - the `skipped` signature);
    - park lost: s.parked is False (a `dropped`, or park demonstrably not holding it);
    - evictions pressure: any joined event now records a higher engine evictions counter than s's
      baseline at its last switch - park is over-subscribed, so even cleanly-parked mids earn a save."""
    if not cfg.enabled or not s.save_eligible or s.tokens < MIN_PARK_TOKENS:
        return False
    if s.tokens * PARK_COST_KIB_PER_TOKEN / 1024.0 > cfg.park_budget_mib:
        return True
    if not s.parked:
        return True
    return any(v.evictions_at_last_switch > s.evictions_at_last_switch for v in tracked.values())


def should_restore(s: TrackedSession, cfg: LadderConfig) -> bool:
    """Restore s on switch-back only when a ladder file exists for it AND park demonstrably missed it
    (parked is False - park hits beat any restore).  Ambiguous joins never restore: a missed mid costs
    one cold step, a needless restore costs seconds and churns park."""
    return cfg.enabled and s.file_path is not None and not s.parked


def prefix_fidelity(saved_prefix: str | None, incoming_prompt: str) -> bool:
    """Is the checkpoint's saved prefix still a byte-prefix of the incoming prompt?  OpenCode rewrites
    context (compaction, edits); a stale checkpoint restores as waste, so serve byte-compares before
    issuing RESTORE.  No captured prefix -> not fidelity-checkable -> False (re-anchor instead)."""
    return bool(saved_prefix) and incoming_prompt.startswith(saved_prefix)
