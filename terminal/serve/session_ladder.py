"""serve/session_ladder.py - the serve session ladder: the decision core (pure functions answering the
ladder's three questions from bookkeeping alone: which session an engine `conversation cache:` event belongs
to (join_event), whether a session should be saved to disk on switch-away (should_save), whether it should be
restored on switch-back (should_restore + prefix_fidelity), which client header names the session
(extract_session_id)) plus the bookkeeping half (class SessionLadder: per-session TrackedSession state fed by
the intake - ConvCacheLog.drain_events() events and the request path's DONE figures - and its LRU file
retention; no imports from server.py, and the only I/O anywhere here is note_file_saved's removal of an
over-capacity file).

Event vocabulary as the engine writes it (ConvCacheLog's lines, parsed upstream into ConvEvent):
"conversation cache: parked N tokens in T ms; parked=P bytes=B evictions=E ...",
"conversation cache: skip parking (...)", "conversation cache: dropped ...", "conversation cache: restored ...".
`evictions` is a running global counter carried by every event; each join records the figure the event
carried into that session's evictions_at_last_switch (the engine's cache-wide counter at the last switch
touching the session), which is what the evictions-pressure signal compares against.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass

# Pinned constants (measured / engine's own): park costs 40.4 KiB per token; the engine refuses to park
# below 8192 tokens ("skip parking (70 tokens, below the 8192 minimum; parked=4)").
PARK_COST_KIB_PER_TOKEN = 40.4
MIN_PARK_TOKENS = 8192

# Session identity headers, pinned first-wins precedence (OpenCode's native family; serve stamps them).
SESSION_HEADERS = ("x-session-id", "x-opencode-session-id", "x-session-affinity", "x-opencode-session")


@dataclass
class ConvEvent:
    """One parsed `conversation cache:` event line (ConvCacheLog.drain_events): kind is
    parked|skipped_below_min|skipped_over_budget|restored|dropped, byte_count is the line's per-conversation
    bytes (its snapshot_bytes, else the 40.4 KiB/token estimate, 0 when the line names none), evictions is the
    engine's running eviction counter, and snapshot_mib is set only on a skipped_over_budget line that names a
    snapshot size without a conversation (the join rule: SessionLadder.on_conv_event)."""
    kind: str
    tokens: int
    byte_count: int
    evictions: int
    snapshot_mib: int = 0


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


# Engine.session_file rejects a path with control characters (ValueError "invalid session command"), so a
# session id keeps only the filename-safe family, each other character become an underscore: two ids never
# collapse onto one file unless they differ only in unsafe characters (which are not session ids).
_SAFE_SID = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_sid(session_id: str) -> str:
    return _SAFE_SID.sub("_", session_id)


def _refused_class():
    """The engine's SessionRefused contract.  server.py owns the class the Engine raises; session_ladder
    imports nothing from server.py at module level (its header contract), so the pairing resolves lazily -
    the same class object when the ladder runs inside the server, this module's own stand-in when it is
    driven without server.py loaded (the same constructor the engine's SERR parse uses)."""
    try:
        from serve.server import SessionRefused
        return SessionRefused
    except Exception:
        class SessionRefused(ValueError):
            def __init__(self, kind, message, published=False):
                super().__init__(message)
                self.kind = kind
                self.published = published
        return SessionRefused


class _locked:
    """A method that runs holding the ladder's lock.  Task 7 runs the ladder from two worlds: request
    threads (on_arrival/on_turn_done, serialized at the engine by the service FIFO) and the metrics()
    thread draining conv_log events (GET /metrics, any number of monitors) - the compound updates
    (join -> park flag; on_turn_done create + anchor) are not atomic on their own, so every public hook
    takes the ladder lock.  A public hook therefore calls the plain inner method (`_x`), never the
    locked attribute - the lock is not reentrant."""

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, owner):
        fn = self.fn

        def locked(*a, **kw):
            with obj._lock:
                return fn(obj, *a, **kw)

        locked.__name__ = fn.__name__
        locked.__doc__ = fn.__doc__
        return locked


class SessionLadder:
    """The ladder's bookkeeping half: per-session state (the `tracked` dict should_save/join_event read), fed
    by the intake - ConvCacheLog.drain_events() events (on_conv_event) and the request path's DONE figures
    (on_turn_done).  Per-session byte_count normalization, pinned here and in drain_events: the engine's
    bytes=B is the cache-wide cumulative total, so a `parked`/`skipped` line's per-conversation figure is its
    own snapshot_bytes - exactly the cumulative delta the park implies (P1: bytes=1046081816 then 2092276296 ->
    1046194480 = that line's snapshot_bytes) - and with no byte figure on hand (a DONE intake) a session's
    bytes are the 40.4 KiB/token estimate.  File saving is Task 7's; this half keeps the ledger, its LRU
    retention, and the one file removal that over-capacity implies."""

    # Pinned join tolerance for a `skip parking (snapshot N MiB exceeds available budget)` line - the only
    # event that names a size without a conversation: it joins the tracked session whose tokens x 40.4 KiB
    # estimate is nearest to N, and only within SNAPSHOT_JOIN_TOLERANCE of N (a match just inside joins;
    # just outside joins nothing).
    SNAPSHOT_JOIN_TOLERANCE = 0.15

    def __init__(self, cfg: LadderConfig):
        self.cfg = cfg
        self.tracked: dict = {}
        self.saved_files: list = []                     # file paths kept for their sessions, least-recent first
        self.last_prompts: dict = {}                    # each session's last rendered prompt the ladder saw,
        # kept only while it can matter for a save (see on_arrival/record_saved_prefix: bounded to
        # save-eligible sessions, pruned with the LRU file eviction) - the text a save records as the prefix
        self._lock = threading.Lock()
        self.last_active: str | None = None             # the session the engine is on right now (Task 7's on_arrival)

    def track(self, session_id: str) -> TrackedSession | None:
        """The session's tracked state, or None when the ladder never saw it (an event never invents one)."""
        return self.tracked.get(session_id)

    def on_arrival(self, service, session_id: str | None, prompt: str) -> None:
        """Task 7's request-path hook, called while the service FIFO is held, immediately before the prompt
        goes to the engine.  A switch (the arriving session differs from the last one the ladder saw) is
        where the engine's single conversation slot changes hands: first the outgoing session is saved when
        should_save still sees its park state (hence note_switch_away only after), recording as its prefix
        the last rendered prompt it completed with (the save-time value of this ladder's last_prompts;
        re-anchored turn by turn by record_saved_prefix); then the incoming session is restored before its
        prompt is forwarded when should_restore sees a file and a missed park AND prefix_fidelity still
        holds for the checkpoint against this request's rendered prompt - OpenCode rewrites context, and a
        stale checkpoint restores as waste, so the byte-compare is the gate (and the backstop for a near-tie
        event join).  A SessionRefused (engine memory/storage/io) degrades to a plain prefill: the request
        is not failed, the refusal is logged, and no phantom file state is recorded.  An EngineDied is
        re-raised unchanged: the engine is gone and the request path handles it as it always has.
        `service` duck-types only: .slot_save_path (the save/restore directory) and .engine.session_file."""
        if session_id is None or not self.cfg.enabled:
            return                                   # no session identity: the ladder has nothing to switch
        SessionRefused = _refused_class()
        incoming = self.tracked.get(session_id)
        outgoing = self.tracked.get(self.last_active) if self.last_active not in (None, session_id) else None
        self.last_prompts[session_id] = prompt            # what a save of this session will record next
        if outgoing is not None and should_save(outgoing, self.tracked, self.cfg):
            path = os.path.join(self.cfg.slot_save_path, f"llm-session-{_safe_sid(outgoing.session_id)}.bin")
            try:
                service.engine.session_file("save", path)
            except SessionRefused as e:
                print(f"[strata] session ladder: the save of {outgoing.session_id} was refused "
                      f"({e.kind}); this conversation continues as a cold prefill", flush=True)
            else:
                outgoing.saved_prefix = self.last_prompts.get(outgoing.session_id) or outgoing.saved_prefix
                self._note_file_saved(outgoing.session_id, path)
        if incoming is not None and should_restore(incoming, self.cfg):
            path = os.path.join(self.cfg.slot_save_path, f"llm-session-{_safe_sid(session_id)}.bin")
            if prefix_fidelity(incoming.saved_prefix, prompt):
                try:
                    service.engine.session_file("restore", path)
                except SessionRefused as e:
                    print(f"[strata] session ladder: the restore of {session_id} was refused "
                          f"({e.kind}); this conversation continues as a cold prefill", flush=True)
        if self.last_active is not None and self.last_active != session_id:
            self._note_switch_away(self.last_active)      # after should_save saw its park state
        self.last_active = session_id

    def record_saved_prefix(self, session_id: str, prompt: str) -> None:
        """Task 7 re-anchors a checkpoint's prefix to the conversation as it now renders (the prompt text
        travels from the request path: it is the only place the rendered string exists).  Only a session
        with a ladder file has a prefix worth anchoring - the bounded set (map §6), kept fresh so the next
        switch's prefix_fidelity compares against the conversation the file actually holds."""
        s = self.tracked.get(session_id)
        if s is not None and s.file_path is not None:
            s.saved_prefix = prompt

    def _on_conv_event(self, ev: ConvEvent) -> None:
        """One drained engine event.  Joined by join_event's (tokens, byte_count) - except a
        skipped_over_budget line that names only a snapshot MiB, which joins the nearest estimated session
        (SNAPSHOT_JOIN_TOLERANCE above).  Every join records the engine's running evictions counter into that
        session's baseline, and the join sets the session's own park state: a `parked` event parks it.  An
        unjoinable (or ambiguous) event records nothing: the engine's fingerprint validation is the backstop."""
        s = self._join(ev)
        if s is None:
            return
        s.evictions_at_last_switch = ev.evictions
        if ev.kind == "parked":
            s.parked = True

    def _on_turn_done(self, session_id: str, tokens: int, byte_count: int, reused: int) -> None:
        """A completed turn's DONE figures (Engine.last): re-anchors the session's token and byte counts
        (byte_count 0 when the front has none - then the pinned 40.4 KiB/token estimate) and marks
        save_eligible: True when the conversation is beyond the engine's park budget (tokens x 40.4 KiB >
        park_budget_mib - the `skipped`/never-parkable signature) or at least the engine's park minimum size
        (a mid that park could lose); False for an aux-sized turn (below MIN_PARK_TOKENS and within budget -
        tiny calls must never churn saves).  A session is created here: the ladder only tracks conversations
        it has actually seen completed."""
        s = self.tracked.get(session_id)
        if s is None:
            s = TrackedSession(session_id=session_id, tokens=0, parked=False, save_eligible=False,
                               file_path=None, saved_prefix=None, last_active=0.0, evictions_at_last_switch=0)
            self.tracked[session_id] = s
        s.tokens = tokens
        s.byte_count = byte_count if byte_count > 0 else int(tokens * PARK_COST_KIB_PER_TOKEN * 1024)
        s.last_active = time.time()
        over_budget = tokens * PARK_COST_KIB_PER_TOKEN / 1024.0 > self.cfg.park_budget_mib
        s.save_eligible = over_budget or tokens >= MIN_PARK_TOKENS
        if not s.save_eligible:
            self.last_prompts.pop(session_id, None)  # an aux-sized conversation earns no saves: no retained prefix

    def _note_switch_away(self, session_id: str) -> None:
        """The session's switch-away (Task 7 calls it after should_save was evaluated, which is why should_save
        reads `parked` while it still reflects park): from this point the conversation is the engine's to keep
        or lose, so the park flag clears - the next switch's parked reflects only the events since here."""
        s = self.tracked.get(session_id)
        if s is not None:
            s.parked = False

    def _note_file_saved(self, session_id: str, path: str) -> None:
        """Task 7 records a ladder file it saved for this session: the file joins the LRU ledger (a re-save of
        the same path moves it to the recent end - saves overwrite in place), the session's file_path follows
        it, and the ledger caps at cfg.retention_max_files: an over-cap save retires the least-recent file -
        removed from disk (one canonical file per session; the cap is what bounds the big ones, the tmpfiles
        age rule owns the rest) together with that session's tracked state."""
        if path in self.saved_files:
            self.saved_files.remove(path)
        self.saved_files.append(path)
        s = self.tracked.get(session_id)
        if s is not None:
            s.file_path = path
        while len(self.saved_files) > self.cfg.retention_max_files:
            old = self.saved_files.pop(0)
            for sid in [k for k, v in self.tracked.items() if v.file_path == old]:
                del self.tracked[sid]
                self.last_prompts.pop(sid, None)               # the prefix bound to a retired file retires with it
            try:
                os.remove(old)
            except OSError:
                pass                                       # already gone (the age rule got there first): the ledger is what matters

    def _join(self, ev: ConvEvent) -> TrackedSession | None:
        if ev.kind == "skipped_over_budget" and ev.tokens == 0:
            best, best_gap = None, None
            for s in self.tracked.values():
                gap = abs(s.byte_count / 1048576.0 - ev.snapshot_mib)
                if best_gap is None or gap < best_gap:
                    best, best_gap = s, gap
            return best if best is not None and best_gap <= self.SNAPSHOT_JOIN_TOLERANCE * ev.snapshot_mib else None
        return join_event(ev, self.tracked)

    # Every public hook runs holding the ladder lock (see _locked): on_conv_event/on_turn_done arrive from the
    # metrics() drain, on_arrival/record_saved_prefix from a FIFO-serialized request thread; track stays unlocked
    # (a single dict read) so a caller holding nothing else can peek without serializing behind a save.
    on_conv_event = _locked(_on_conv_event)
    on_turn_done = _locked(_on_turn_done)
    note_switch_away = _locked(_note_switch_away)
    note_file_saved = _locked(_note_file_saved)
    record_saved_prefix = _locked(record_saved_prefix)
    on_arrival = _locked(on_arrival)
