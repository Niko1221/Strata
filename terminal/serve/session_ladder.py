"""serve/session_ladder.py - the serve session ladder: the decision core (pure functions answering the
ladder's three questions from bookkeeping alone: which session an engine `conversation cache:` event belongs
to (join_event, with join_event_tolerant's pinned fallback for lines whose figures carry the generated turn /
estimate gap - F1), whether a session should be saved to disk on switch-away (should_save), whether it should be
restored on switch-back (should_restore + prefix_fidelity), which client header names the session
(extract_session_id)) plus the bookkeeping half (class SessionLadder: per-session TrackedSession state fed by
the intake - ConvCacheLog.drain_events() events and the request path's DONE figures - and its LRU file
retention; no imports from server.py; its own filesystem I/O is note_file_saved's removal of an
over-capacity file plus the R4 ledger sidecar the ladder's saves write and a restarted ladder reads
(see _write_index/_load_index) - engine save/restore calls go through the caller's
service.engine, in the F4 lock windows of on_arrival/on_idle/on_unload).

Event vocabulary as the engine writes it (ConvCacheLog's lines, parsed upstream into ConvEvent):
"conversation cache: parked N tokens in T ms; parked=P bytes=B evictions=E ...",
"conversation cache: skip parking (...)", "conversation cache: dropped ...", "conversation cache: restored ...".
`evictions` is a running global counter carried by every event; each join records the figure the event
carried into that session's evictions_at_last_switch (the engine's cache-wide counter at the last switch
touching the session), which is what the evictions-pressure signal compares against.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass

# Pinned constants (measured / engine's own): park costs 40.4 KiB per token; the engine refuses to park
# below 8192 tokens ("skip parking (70 tokens, below the 8192 minimum; parked=4)").  MIN_PARK_TOKENS is
# the engine's DEFAULT --conversation-cache-min-tokens, the value momo's render runs with; it is pinned
# here rather than read from the engine args (scope ruling: the ladder's budget comes from
# --conversation-cache-mib only, and a mid-size knob the config never changes is pinned with it).
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
    dirty: bool = True                # Task 8: unsaved since this session's last turn.  on_turn_done
    # dirties; a save the ladder itself routed (note_file_saved) cleans.  A save made outside the ladder
    # (an image job's, a /v1/slot save) is not visible here: the session stays dirty and the next ladder
    # save rewrites the file - a harmless duplicate, never a missed one (the pinned boundary).
    resident: bool = False            # momo 2026-10-27 restore-churn finding: the engine's slot still
    # holds THIS session's live conversation - set by a deep-reused DONE continuation (on_turn_done) or a
    # landed ladder restore, cleared by the session's switch-away.  A RESTORE over a resident session would
    # replace the deeper live state with the older file and re-read the difference every turn; resident
    # beats any restore (evidence: momo-restore-churn-diagnosis-20261027T0900.md, re-reads grew
    # 2045 -> 6869 -> 12471 tokens turn after turn).


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


# Pinned join tolerances (F1, final review).  The exact key is the FIRST and preferred join, but it can only
# match event-to-event: the request path anchors a session from DONE figures (byte_count 0 -> the 40.4 KiB/token
# estimate), while a park line carries the engine's real snapshot_bytes and the generated turn on top of the
# DONE prompt (the committed evidence pairs, momo-fleet-20261025T0336 / momo-restore-extend-20261025T0450:
# DONE "prompt 24504 tokens" parked as "parked 24511 tokens ... snapshot_bytes=1046081816" - a 7-token delta,
# a 3% byte delta).  So after exact match the ladder joins the nearest session within SNAPSHOT_JOIN_TOLERANCE
# (15% on bytes - the machinery skipped_over_budget's MiB join already pins) of the event's byte_count AND a
# token band max(1% of s.tokens, 256): 1% absorbs generation variance on big mids, the 256 floor covers small
# mids and long generated turns.  Pinned here and in test_join_tolerance_*; the integration proof is
# test_join_tolerance_fires_on_live_figures.
#
# The near-tie consequence, stated exactly: a mis-join between two candidates inside the band only misrecords
# the evictions baseline / park flag - at worst it SKIPS a save or DEFERS a restore, and both degrade to
# today's cold prefill; prefix_fidelity still gates every restore, so a mis-join can never restore a
# wrong file.
#
# Why there is deliberately NO "resident proxy" (DONE reused >= ~95% of prompt tokens -> parked=True without
# any event join): the reused figure describes the conversation resident in the engine's slot AT COMPLETION -
# which is precisely the conversation the next switch-away is about to lose.  Setting parked from it would
# suppress the save for the resident conversation itself, inverting the policy the save exists to implement.
# Only joined engine park lines set parked.
SNAPSHOT_JOIN_TOLERANCE = 0.15
JOIN_TOKEN_TOLERANCE = 0.01
JOIN_TOKEN_TOLERANCE_FLOOR = 256
# momo 2026-10-27 restore-churn finding: a DONE whose reused is at least this share of its prompt tokens
# is the engine resuming THIS conversation from its own slot - the session is resident, and a RESTORE
# of the older file over it would truncate the live state (churn rows 98.6-99.8% vs shallow resumes
# like 56.8% (x-2 park-checkpoint) and 29.7% (post-cancel rebuild)).  The line sits above both.
RESIDENT_REUSED_RATIO = 0.9
# R4 (momo 2026-10-27 battery): the name of the ledger sidecar the ladder's saves write into the sessions
# dir, so a restarted server re-adopts its checkpoints instead of meeting every session cold.  Distinct
# from the engine's `llm-session-*` name pattern, and its disappearance is only ever a lost warm resume.
INDEX_FILE_NAME = "ladder-index.json"
TOLERANT_JOIN_KINDS = ("parked", "skipped_below_min", "restored")   # the lines whose tokens track a conversation
# (a `dropped` line names no figures; a tokened `skipped 252953 ...` budget refusal keeps exact-only semantics,
# as a wrong blame there would record pressure on the wrong mid).


def join_event_tolerant(ev: ConvEvent, tracked: dict) -> TrackedSession | None:
    """The fallback join after join_event's exact key matched nothing.  The TOKEN BAND is the primary
    gate - every tracked session whose tokens sit within max(JOIN_TOKEN_TOLERANCE x s.tokens,
    JOIN_TOKEN_TOLERANCE_FLOOR) of the event's is a candidate - and byte proximity is used only to
    DISAMBIGUATE when the band holds more than one (SNAPSHOT_JOIN_TOLERANCE of the event's byte_count;
    a tie, or a sizeless byte_count-0 line facing several candidates, joins nothing - the refusal to
    guess survives tolerance).  Only the kinds listed in TOLERANT_JOIN_KINDS.
    Why bytes demoted (momo 2026-10-27 battery, probe-A): the DONE side's byte anchor is the pinned
    40.4 KiB/token estimate while park lines carry the engine's own measured snapshot_bytes - real
    sizes run 33.6 KiB/token for big sessions and 55-68 KiB/token for small ones, so the two sides
    miss each other's 15% band on any small mid (probe-A: ~363 MB estimate vs 603430952 measured:
    a 66% gap) and every join starved, leaving the session parked=False and restore-owed.  Measured
    vs measured varies more than 15% too (the same probe-A conversation parked at 603430952 and
    485492400).  A lone in-band candidate mis-joining is benign in this file's own words - it skips a
    save or defers a restore, and prefix_fidelity still gates every restore."""
    if ev.kind not in TOLERANT_JOIN_KINDS:
        return None
    candidates = [s for s in tracked.values()
                  if abs(ev.tokens - s.tokens) <= max(JOIN_TOKEN_TOLERANCE * s.tokens,
                                                      JOIN_TOKEN_TOLERANCE_FLOOR)]
    if len(candidates) == 1:
        return candidates[0]
    best, best_gap, ties = None, None, 0
    for s in candidates:
        gap = abs(ev.byte_count - s.byte_count)
        if gap > SNAPSHOT_JOIN_TOLERANCE * ev.byte_count:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap, ties = s, gap, 1
        elif gap == best_gap:
            ties += 1
    return best if best is not None and ties == 1 else None


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
    AND the engine does not still hold it live (parked is False and resident is False - park hits, and a
    resident slot, beat any restore).  Ambiguous joins never restore: a missed mid costs
    one cold step, a needless restore costs seconds and churns park."""
    return cfg.enabled and s.file_path is not None and not s.parked and not s.resident


def prefix_fidelity(saved_prefix: str | None, incoming_prompt: str) -> bool:
    """Is the checkpoint's saved prefix still the head of the incoming prompt?  OpenCode rewrites
    context (compaction, edits); a stale checkpoint restores as waste, so serve byte-compares before
    issuing RESTORE.  The strict compare is exact for turn-array prompts (the saved render ends at the
    generation opener, which is where the next render's completed assistant turn begins), but not for a
    client that re-sends ONE growing user message (the momo dormancy of the 2026-10-26 recheck): the
    incoming render re-renders the saved turn's closing marker after the growth, so the bytes diverge
    at exactly that marker while the engine's own prefix join still holds (bridge restores reused full
    checkpoint files on this shape; a DISC-row checkpoint reused 32768 of 35750 tokens).  So the saved
    prefix is also matched cut at its last turn-end marker - a rewrite diverging before that point is
    still rejected, only growth at the conversation's tail passes.  No captured prefix -> not
    fidelity-checkable -> False (re-anchor instead)."""
    if not saved_prefix:
        return False
    if incoming_prompt.startswith(saved_prefix):
        return True
    cut = saved_prefix.rfind(TURN_END)
    return cut >= 0 and incoming_prompt.startswith(saved_prefix[:cut])


# The chat template's message-closing marker (the tokenizer's literal every rendered turn ends with;
# encode_prompt renders prompts with these markers, so prefix_fidelity can cut a saved prefix at one).
TURN_END = "<|im_end|>"


# Engine.session_file rejects a path with control characters (ValueError "invalid session command"), so a
# session id keeps only the filename-safe family, each other character become an underscore: two ids never
# collapse onto one file unless they differ only in unsafe characters (which are not session ids).
_SAFE_SID = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_sid(session_id: str) -> str:
    return _SAFE_SID.sub("_", session_id)


class _StandInRefused(ValueError):
    """The fallback SessionRefused, used when the ladder is driven without server.py loaded (its own unit
    tests): the same constructor the engine's SERR parse uses."""

    def __init__(self, kind, message, published=False):
        super().__init__(message)
        self.kind = kind
        self.published = published


# The exact SessionRefused class object the running server module's Engine raises from session_file.
# server.py registers it at its own import time (register_refused_class), in module mode AND in the
# run-<model>.bat __main__ mode; session_ladder still imports nothing from server.py - registration is
# push-based, server -> ladder.  The earlier lazy `from serve.server import SessionRefused` inside
# on_arrival was removed: when server.py runs as a script, serve.server is not in sys.modules, so that
# import pulled in a second copy of the server module mid-request (while the FIFO is held) whose
# SessionRefused was a different class than the one __main__'s Engine raises - the except missed, the
# refusal escaped as an uncaught ValueError, and the refuse->prefill posture broke into a 500.
_refused_cls = _StandInRefused


def register_refused_class(cls):
    """server.py (module or __main__ copy) registers the class its Engine raises; the ladder's excepts
    pair with exactly that class object."""
    global _refused_cls
    _refused_cls = cls


def _refused_class():
    return _refused_cls


class _locked:
    """A method that runs holding the ladder's lock.  Task 7 runs the ladder from two worlds: request
    threads (on_arrival/on_turn_done, serialized at the engine by the service FIFO) and the metrics()
    thread draining conv_log events (GET /metrics, any number of monitors) - the compound updates
    (join -> park flag; on_turn_done create + anchor) are not atomic on their own, so every wrapped hook
    takes the ladder lock.  A wrapped body therefore calls the plain inner (`_x`), never the locked
    attribute - the lock is not reentrant - and never touches the engine (F4: engine I/O lives in the
    unlocked windows of on_arrival/on_idle/on_unload, the only public hooks that are not wrapped)."""

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
    # just outside joins nothing).  The constant is module-level (shared with join_event_tolerant's byte
    # band, F1); the same 0.15 machinery serves both joins.

    def __init__(self, cfg: LadderConfig):
        self.cfg = cfg
        self.tracked: dict = {}
        self.saved_files: list = []                     # file paths kept for their sessions, least-recent first
        self.last_prompts: dict = {}                    # each session's last rendered prompt the ladder saw,
        # kept only while it can matter for a save (see on_arrival/record_saved_prefix) - the text a save
        # records as the prefix; LRU-capped to cfg.retention_max_files entries (an arrival past the cap
        # evicts the oldest, the ConvCacheLog._pending ring idiom) and pruned with the LRU file eviction,
        # so the retained text can never outrun the 16-file bound (map §6) however many ids churn.
        self._lock = threading.Lock()
        self.last_active: str | None = None             # the session the engine is on right now (Task 7's on_arrival)
        if self.cfg.enabled and self.cfg.slot_save_path:
            self._load_index()                          # R4: adopt the ledger a previous server process left

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
        event join).  An arrival whose session_id is None under an enabled ladder is an IDENTITY-UNKNOWN
        switch (F2): the conversation taking the slot cannot be named, so the ladder notes the switch-away,
        sets last_active None, and saves/restores nothing.  A SessionRefused (engine memory/storage/io)
        degrades to a plain prefill: the request is not failed, the refusal is logged, and no phantom file
        state is recorded.  An EngineDied is re-raised unchanged: the engine is gone and the request path
        handles it as it always has.
        `service` duck-types only: .engine.session_file; the save/restore directory is the ladder's own
        cfg.slot_save_path.  F4 (final review): the hook runs in windows on the ladder lock - the DECISION
        window (all ladder bookkeeping, no engine touched), then the ENGINE I/O window (session_file
        save/restore, lock NOT held: the service FIFO already serializes engine-facing actors, and engine
        file I/O can run for minutes - /metrics's on_conv_event must not queue behind it), with the save's
        LEDGER commit and the switch-away note each re-taking the lock for its short bookkeeping burst."""
        if not self.cfg.enabled:
            return                                   # a disabled ladder writes nothing at all
        if session_id is None:
            # F2 (final review): an enabled-ladder arrival with no session id IS a switch - the engine's
            # slot now holds a conversation the ladder cannot name.  Leaving last_active stale would make
            # the NEXT switch-away save that foreign resident's engine state into
            # llm-session-<last_active>.bin, destroying the real checkpoint's value.  The ladder notes the
            # switch-away (park state of the departed conversation is now unknown) and forgets the identity;
            # no save, no restore - a header brings the identity back.
            with self._lock:
                if self.last_active is not None:
                    self._note_switch_away(self.last_active)
                self.last_active = None
            return
        with self._lock:                             # decision window: everything the ladder decides
            incoming = self.tracked.get(session_id)
            outgoing = (self.tracked.get(self.last_active)
                        if self.last_active not in (None, session_id) else None)
            self.last_prompts.pop(session_id, None)  # re-inserting moves a known id to the recent end (LRU)
            self.last_prompts[session_id] = prompt   # what a save of this session will record next
            save_plan = None
            if outgoing is not None and should_save(outgoing, self.tracked, self.cfg) \
                    and not (outgoing.file_path is not None and not outgoing.dirty):
                # F3 (final review): the dirty marker gates the switch-away save.  file_path set + dirty
                # False means the engine state is provably identical to the checkpoint the ladder itself
                # last wrote (idle save, earlier switch save) - re-saving is a 46-65 s identical rewrite.
                # A session whose file was CONSUMED by a warm restore (file_path retired by the on_start
                # handoff) re-saves even while clean: engine park does not hold over-budget mains and the
                # file is gone - that re-save is the needed protection (pinned by
                # test_a_consumed_checkpoint_re_saves_on_the_next_switch).
                save_plan = self._plan_save(service, outgoing)
            restore_path = None
            if incoming is not None and should_restore(incoming, self.cfg):
                path = os.path.join(self.cfg.slot_save_path, f"llm-session-{_safe_sid(session_id)}.bin")
                if prefix_fidelity(incoming.saved_prefix, prompt):
                    restore_path = path
            switched_from = self.last_active if self.last_active not in (None, session_id) else None
            while len(self.last_prompts) > self.cfg.retention_max_files:
                del self.last_prompts[next(iter(self.last_prompts))]   # the LRU cap on the text ring (map §6's
            # bound, review fix): retained prompt text can never outrun the retention_max_files file bound no
            # matter how many ids churn in and out - the least-recently-arrived prompt goes first, and the cap
            # runs after the decision window captured this request's outgoing save prefix in its plan.
            self.last_active = session_id
        saved_ok = save_plan is not None and self._engine_save(service, save_plan)   # engine I/O: no lock
        if saved_ok:
            with self._lock:                         # ledger window: the save landed, record it
                self._commit_save(save_plan)
        if restore_path is not None:
            landed = self._engine_restore(service, session_id, restore_path)          # engine I/O: no lock
            if landed:
                with self._lock:                      # ledger window: the restore put the file's
                    s = self.tracked.get(session_id)  # conversation into the slot - resident from now
                    if s is not None:
                        s.resident = True
        if switched_from is not None:
            with self._lock:
                self._note_switch_away(switched_from)   # after should_save saw its park state (Task 7's
            # ordering, kept through the F4 windows: the decision evaluated should_save, the note closes out
            # the switch once the engine I/O is done)

    def _plan_save(self, service, s: TrackedSession):
        """DECISION half of a ladder save (called holding the LADDER LOCK; never touches the engine - F4):
        the pinned `llm-session-<sid>.bin` name under cfg.slot_save_path plus the prefix the save will
        record (this ladder's last_prompts at decision time), as a (session_id, path, prefix) plan handed
        to _engine_save and _commit_save.  None when the save is moot: parallel/batch mode refuses
        save/restore at the engine (slot_action's precedent, map §2) - the ladder skips the save there too."""
        if getattr(service.engine, "batch", 0):
            return None                              # batch mode: the engine refuses these commands
        path = os.path.join(self.cfg.slot_save_path, f"llm-session-{_safe_sid(s.session_id)}.bin")
        return (s.session_id, path, self.last_prompts.get(s.session_id) or s.saved_prefix)

    def _engine_save(self, service, plan) -> bool:
        """ENGINE I/O half of a ladder save (runs OUTSIDE the ladder lock - F4; the FIFO already serializes
        engine-facing actors): engine.session_file("save", path) for the planned file.  Returns True when
        the save landed; a SessionRefused is swallowed with the ladder's log line and the ledger records
        nothing (no phantom file state); an EngineDied propagates (the caller's engine is gone - the request
        path / unload handles it as it always has)."""
        try:
            service.engine.session_file("save", plan[1])
            return True
        except _refused_class() as e:
            print(f"[strata] session ladder: the save of {plan[0]} was refused ({e.kind}); "
                  "this conversation continues as a cold prefill", flush=True)
            return False

    def _commit_save(self, plan) -> None:
        """LEDGER half of a ladder save (called holding the LADDER LOCK): the landed file becomes the
        session's checkpoint - saved_prefix anchored to the prefix the plan captured, file_path follows it,
        dirty flips clean - and the file joins the LRU ledger via the plain _note_file_saved (retention
        included).  Split out of the old _save_session by F4 so the engine call sits between lock windows."""
        sid, path, prefix = plan
        s = self.tracked.get(sid)
        if s is not None:
            s.saved_prefix = prefix or s.saved_prefix
        self._note_file_saved(sid, path)

    def _engine_restore(self, service, session_id: str, path: str) -> bool:
        """ENGINE I/O of a ladder restore (runs OUTSIDE the ladder lock - F4, same reasoning as
        _engine_save): engine.session_file("restore", path).  Returns True when it landed; a
        SessionRefused degrades to a plain prefill with the ladder's log line and answers False, and the
        file stays in the ledger; an EngineDied propagates."""
        try:
            service.engine.session_file("restore", path)
            return True                             # landed: the engine's slot now holds this file's
            # conversation - the caller records residency so the next arrival continues instead of
            # restoring the same file again (momo 2026-10-27 churn).
        except _refused_class() as e:
            print(f"[strata] session ladder: the restore of {session_id} was refused "
                  f"({e.kind}); this conversation continues as a cold prefill", flush=True)
            return False

    def on_idle(self, service, idle_seconds: float) -> None:
        """Task 8's idle hook, called while the service FIFO is held (the unload path's hold), with the
        same idle measurement the lazy-unload decision uses (time since last_request_at/started_at).
        idle_seconds >= cfg.idle_save_s is the human-latency head start: before the engine's park is
        possibly dropped by an idle unload, the last-active session is saved when it is save-eligible
        and still dirty - unsaved since its last turn (a clean checkpoint already on disk is not
        rewritten).  Nothing else is saved here: the switch path and on_unload own the other sessions.
        F4's windows: decide under the ladder lock, engine save outside it, commit the ledger back under it."""
        if not self.cfg.enabled or idle_seconds < self.cfg.idle_save_s:
            return
        with self._lock:
            s = self.tracked.get(self.last_active) if self.last_active is not None else None
            if s is None or not s.save_eligible or not s.dirty:
                return
            plan = self._plan_save(service, s)
        if plan is not None and self._engine_save(service, plan):
            with self._lock:
                self._commit_save(plan)

    def on_unload(self, service) -> None:
        """Task 8's unload hook, called while the FIFO is held, immediately before engine.unload() drops
        the engine - and with it every parked conversation.  Saves ALL tracked save-eligible sessions
        (should_save's switch signals are moot: the whole cache is going away), each already-saved-and-
        unchanged one (dirty False) skipped; an aux-sized session (not eligible) never churns.  A refused
        save is swallowed and the loop continues: the unload proceeds either way.  F4's windows: all save
        plans are decided in one lock window, each engine save runs outside the lock, each landed save
        re-takes the lock for its ledger commit."""
        if not self.cfg.enabled:
            return
        with self._lock:
            plans = [p for p in (self._plan_save(service, s) for s in list(self.tracked.values())
                                 if s.save_eligible and s.dirty) if p is not None]
        for plan in plans:
            if self._engine_save(service, plan):
                with self._lock:
                    self._commit_save(plan)

    def _on_start(self, service) -> None:
        """Task 8's engine-start hook, called from ensure_loaded (the FIFO held, the engine just reached
        READY).  A fresh engine parks nothing, so every `parked` flag resets - while the tracked entries
        and their file_paths survive, which is what makes the next switch-back restore-eligible
        (should_restore sees parked False).  Then the warm restore: when NO restore is already pending
        (svc.restore_after_load is the image job's mechanism and already owns this start - it is never
        overwritten or doubled) and the single last-active session has a checkpoint file that still exists
        AND a saved_prefix a prefix_fidelity check could one day pass, that file is handed to
        service.restore_after_load for the load path right below (images_api.restore_session issues the
        session_file("restore") and removes the file).  Because the load path CONSUMES the checkpoint, the
        ladder retires its ledger entry with the handoff (no restore attempt against a file that is gone);
        no pending-restore check, no warm.  A last-active without a file (or prefix) earns nothing."""
        if not self.cfg.enabled:
            return
        for s in self.tracked.values():
            s.parked = False
            s.resident = False                      # a fresh engine holds nothing either: residency
            # claims from the previous engine die with it (the warm handoff below re-earns it, from
            # its own restore, for exactly the one session it hands over)
        if getattr(service, "restore_after_load", None):
            return                                    # a pending restore owns this start: yield to it
        s = self.tracked.get(self.last_active) if self.last_active is not None else None
        if s is None or s.file_path is None or not s.saved_prefix or not os.path.exists(s.file_path):
            return
        path, s.file_path = s.file_path, None
        if path in self.saved_files:
            self.saved_files.remove(path)             # consumed by the restore below (it unlinks it)
        service.restore_after_load = path
        self._write_index()                           # R4: the consumed checkpoint leaves the sidecar too -
        # nothing re-adopts a file the load path is about to unlink

    def record_saved_prefix(self, session_id: str, prompt: str) -> None:
        """Task 7 re-anchors a checkpoint's prefix to the conversation as it now renders (the prompt text
        travels from the request path: it is the only place the rendered string exists).  Only a session
        with a ladder file has a prefix worth anchoring - the bounded set (map §6), kept fresh so the next
        switch's prefix_fidelity compares against the conversation the file actually holds.
        Stated conservatism (F8, final review): when the NEXT save of this session is REFUSED, saved_prefix
        is now ahead of what the file on disk holds - the newer prefix can fail a later prefix_fidelity even
        though the stale file could still have served the restore.  That over-restriction is kept on
        purpose: a fidelity miss degrades to the cold prefill (safe), while trusting a prefix that no
        longer describes the file is the wrong-file restore the gate exists to make impossible."""
        s = self.tracked.get(session_id)
        if s is not None and s.file_path is not None:
            s.saved_prefix = prompt

    def _on_conv_event(self, ev: ConvEvent) -> None:
        """One drained engine event.  Joined by join_event's (tokens, byte_count) - except a
        skipped_over_budget line that names only a snapshot MiB, which joins the nearest estimated session
        (SNAPSHOT_JOIN_TOLERANCE above), and except that when the exact key matches nothing a
        parked/skipped_below_min/restored line joins on the tolerant join's token band, bytes disambiguating
        only a crowded band (join_event_tolerant, F1 + the 2026-10-27 battery: a DONE-estimate and a park
        line's measured snapshot_bytes miss each other's byte band on any small mid - 33.6 to 68 KiB/token
        in the fleet - while their tokens stay within 1%).  Every join records the engine's running
        evictions counter
        into that session's baseline, re-anchors the session's token and byte figures to the event's
        (the engine's own - measured whenever the line names snapshot_bytes), and sets the session's own
        park state: a `parked` event parks it.  An unjoinable (or ambiguous) event records nothing: the
        engine's fingerprint validation is the backstop."""
        s = self._join(ev)
        if s is None:
            return
        s.evictions_at_last_switch = ev.evictions
        if ev.tokens > 0:
            s.tokens = ev.tokens                            # the join re-anchors the session to the
        if ev.byte_count > 0:
            s.byte_count = ev.byte_count                    # engine's own figures (measured whenever
        # the line names snapshot_bytes): from the first landed join on, joins compare measured to
        # measured instead of the pinned estimate against the engine's real sizes (probe-A battery).
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
        s.dirty = True                                # Task 8: a completed turn is unsaved until the
        # ladder's next save of it (on_idle / on_unload / a switch-away save)
        s.resident = s.resident or reused >= RESIDENT_REUSED_RATIO * s.tokens   # the slot holds THIS
        # conversation: a deep-reused DONE is the engine continuing from its own live/park state (momo
        # residency line above).  One-way grant: only the session's switch-away (_note_switch_away) and
        # an engine restart (on_start) can make residency false again - a shallow DONE cannot mean the
        # slot lost the conversation, because those two events are exactly what loses it.
        if reused == 0 and s.parked and s.file_path is not None:
            s.parked = False                          # park self-heal (the momo dormancy, recheck 2026-10-26):
        # DONE's reused is the engine's only session-attributed cache accounting - a park copy that held this
        # conversation reused its prefix (every observed park hit reused whole checkpoint blocks), while the
        # eviction lines name a count, never the victim, so a budget-evicted mid would otherwise keep its
        # joined parked=True forever and every switch-back silently pays the cold prefill.  The design floor's
        # one cold step is already paid by this turn; unparking lets the next switch-back reach should_restore's
        # file+fidelity gates (an aux without a file cannot restore anyway and is left untouched).
        if not s.save_eligible:
            self.last_prompts.pop(session_id, None)  # an aux-sized conversation earns no saves: no retained prefix

    def _note_switch_away(self, session_id: str) -> None:
        """The session's switch-away (Task 7 calls it after should_save was evaluated, which is why should_save
        reads `parked` while it still reflects park): from this point the conversation is the engine's to keep
        or lose, so the park flag clears - the next switch's parked reflects only the events since here."""
        s = self.tracked.get(session_id)
        if s is not None:
            s.parked = False
            s.resident = False                      # residency dies with the switch: from here the
            # engine keeps or loses the conversation, and a later switch-back may owe it a restore.

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
            s.dirty = False                           # Task 8: the ladder's own save makes it clean
        while len(self.saved_files) > self.cfg.retention_max_files:
            old = self.saved_files.pop(0)
            for sid in [k for k, v in self.tracked.items() if v.file_path == old]:
                del self.tracked[sid]
                self.last_prompts.pop(sid, None)               # the prefix bound to a retired file retires with it
            try:
                os.remove(old)
            except OSError:
                pass                                       # already gone (the age rule got there first): the ledger is what matters
        self._write_index()       # R4: this save's file facts (entry, move, retirement) into the sidecar

    def _index_path(self) -> str:
        return os.path.join(self.cfg.slot_save_path, INDEX_FILE_NAME)

    def _write_index(self) -> None:
        """R4 (momo 2026-10-27 battery): the ledger - the file list in LRU order, and each file's session
        with the save-time prefix and token anchor - survives the server process through a sidecar JSON in
        the sessions dir.  Written on the ledger's FILE moments only: a save (with or without a retirement),
        and the warm handoff's consumption - never per turn: record_saved_prefix's advance stays in memory,
        so a persisted prefix always pairs with the file it was saved against (the same conservatism F8
        states for the re-anchor; a restart between saves re-anchors from the file's own conversation).
        saved_files order IS the LRU record (last item = last saved); entries sit on top of the files -
        a ledger path whose tracked session is gone rides on with no entry, exactly as in memory.
        Atomic (tmp + os.replace); a sidecar that cannot be written costs only the next restart's warm
        resume - the in-memory ledger and the next save are untouched."""
        entries = {}
        for path in self.saved_files:
            for sid, s in self.tracked.items():
                if s.file_path == path:
                    entries[sid] = {"file": path, "tokens": s.tokens,
                                    "saved_prefix": s.saved_prefix or ""}
        try:
            tmp = self._index_path() + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"version": 1, "files": list(self.saved_files), "entries": entries}, f)
            os.replace(tmp, self._index_path())
        except OSError:
            pass                                        # an unwritable dir degrades to no-sidecar behavior

    def _load_index(self) -> None:
        """R4's adoption at ladder start: rebuild the ledger from the sidecar.  A file the tmpfiles age rule
        already removed is not adopted (and leaves the ledger - it is gone); a missing or corrupt sidecar is
        no ledger at all - today's cold behavior, never a crash.  An adopted session IS its saved state:
        clean (no instant re-save), unparked and non-resident (a fresh engine parks and holds nothing),
        save_eligible from its tokens (the same shape on_turn_done computes), its prefix as the save paired
        them.  last_active adopts the owner of the LAST saved file - exactly _on_start's warm-handoff
        candidate.  File-system reads run in the constructor, before the ladder is shared."""
        try:
            with open(self._index_path()) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        files = [p for p in data.get("files", []) if isinstance(p, str) and os.path.exists(p)]
        self.saved_files = files
        entries = data.get("entries")
        if not isinstance(entries, dict):
            entries = {}
        for sid, e in entries.items():
            if not isinstance(sid, str) or not isinstance(e, dict):
                continue
            path, tokens = e.get("file"), e.get("tokens")
            if path not in files or isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                continue
            prefix = e.get("saved_prefix")
            self.tracked[sid] = TrackedSession(
                session_id=sid, tokens=tokens, parked=False,
                save_eligible=(tokens * PARK_COST_KIB_PER_TOKEN / 1024.0 > self.cfg.park_budget_mib
                               or tokens >= MIN_PARK_TOKENS),
                file_path=path, saved_prefix=prefix if isinstance(prefix, str) and prefix else None,
                last_active=0.0, evictions_at_last_switch=0, dirty=False,
                byte_count=int(tokens * PARK_COST_KIB_PER_TOKEN * 1024))   # the same normalization an
            # on_turn_done intake gives: the estimate stands in until a joined event re-anchors to measured
        for path in reversed(files):
            owner = next((sid for sid, s in self.tracked.items() if s.file_path == path), None)
            if owner is not None:
                self.last_active = owner
                break

    def _join(self, ev: ConvEvent) -> TrackedSession | None:
        if ev.kind == "skipped_over_budget" and ev.tokens == 0:
            best, best_gap = None, None
            for s in self.tracked.values():
                gap = abs(s.byte_count / 1048576.0 - ev.snapshot_mib)
                if best_gap is None or gap < best_gap:
                    best, best_gap = s, gap
            return best if best is not None and best_gap <= SNAPSHOT_JOIN_TOLERANCE * ev.snapshot_mib else None
        exact = [s for s in self.tracked.values() if s.tokens == ev.tokens and s.byte_count == ev.byte_count]
        if len(exact) == 1:
            return exact[0]                     # exact match keeps precedence (Task 5's join rule, untouched)
        if exact:
            return None                         # ambiguous exact key: refuse to guess, as join_event does
        return join_event_tolerant(ev, self.tracked)   # F1's fallback: DONE-estimate vs engine-snapshot figures

    # Every public hook that only touches ladder state runs holding the ladder lock via _locked (see the
    # wrapper): on_conv_event/on_turn_done arrive from the metrics() drain, record_saved_prefix from a
    # FIFO-serialized request thread; track stays unlocked (a single dict read) so a caller holding nothing
    # else can peek without serializing behind a save.  on_arrival/on_idle/on_unload are NOT wrapped (F4,
    # final review): they take the lock themselves in short windows around their decisions and ledger
    # commits and run the engine.session_file calls between the windows, lock-free - the single-lock-point
    # pattern survives as "a wrapped body never takes the lock and never touches the engine": these three
    # are plain methods managing their own windows, and every state mutation inside a window goes through
    # the plain inners (_note_switch_away, _note_file_saved, _plan_save, _commit_save), never a _locked
    # attribute, because the lock is not reentrant.  on_start stays wrapped: it only queues
    # restore_after_load - it issues no engine I/O of its own.
    on_conv_event = _locked(_on_conv_event)
    on_turn_done = _locked(_on_turn_done)
    note_switch_away = _locked(_note_switch_away)
    note_file_saved = _locked(_note_file_saved)
    record_saved_prefix = _locked(record_saved_prefix)
    on_start = _locked(_on_start)
