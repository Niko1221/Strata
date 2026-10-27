"""serve/test_session_ladder.py - the ladder's decision core (event join, save/restore policy, prefix
fidelity) and its bookkeeping half: ConvCacheLog.drain_events intake plus SessionLadder's per-session
tracking.  The decision-core tests import no I/O; the intake tests drive serve.server.ConvCacheLog over
temp log files, the ladder half stays pure (no I/O beyond tempfile fixtures).

    python -m pytest serve/test_session_ladder.py -v

Fixtures are verbatim `conversation cache:` event lines from the P1 fleet probe
(nixos-config docs/superpowers/spikes/evidence/momo-fleet-20261025T0336.md) and the P2 restore probe
(momo-restore-extend-20261025T0450.md).
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from serve.session_ladder import (
    JOIN_TOKEN_TOLERANCE,
    JOIN_TOKEN_TOLERANCE_FLOOR,
    PARK_COST_KIB_PER_TOKEN,
    SNAPSHOT_JOIN_TOLERANCE,
    ConvEvent,
    LadderConfig,
    TrackedSession,
    SessionLadder,
    extract_session_id,
    join_event,
    join_event_tolerant,
    prefix_fidelity,
    should_restore,
    should_save,
)

# verbatim evidence knobs (Global Constraints): park cost 40.4 KiB/token, engine park minimum 8192.
CFG = LadderConfig(enabled=True, slot_save_path="/var/lib/strata/sessions", park_budget_mib=4096)


def session(session_id, tokens, byte_count, parked=True, save_eligible=True, evictions_at_last_switch=0, **kw):
    return TrackedSession(session_id=session_id, tokens=tokens, parked=parked, save_eligible=save_eligible,
                          file_path=kw.get("file_path"), saved_prefix=kw.get("saved_prefix"),
                          last_active=kw.get("last_active", 0.0), byte_count=byte_count,
                          evictions_at_last_switch=evictions_at_last_switch)


class SessionLadderDecisions(unittest.TestCase):
    def test_extract_session_id_precedence(self):
        # pinned order: x-session-id > x-opencode-session-id > x-session-affinity > x-opencode-session
        headers = {"x-session-id": "sid-a", "x-opencode-session-id": "sid-b",
                   "x-session-affinity": "sid-c", "x-opencode-session": "sid-d"}
        self.assertEqual(extract_session_id(headers), "sid-a")
        self.assertEqual(extract_session_id({"x-session-id": "", "x-opencode-session-id": "sid-b",
                                             "x-session-affinity": "sid-c", "x-opencode-session": "sid-d"}), "sid-b")
        self.assertEqual(extract_session_id({"x-session-affinity": "sid-c", "x-opencode-session": "sid-d"}), "sid-c")
        self.assertEqual(extract_session_id({"x-opencode-session": "sid-d"}), "sid-d")
        self.assertEqual(extract_session_id({"X-Session-Id": "sid-case"}), "sid-case")  # header names are case-insensitive
        self.assertIsNone(extract_session_id({}))
        self.assertIsNone(extract_session_id({"x-opencode-chat": "not-a-session-header"}))

    def test_join_event_unique(self):
        # verbatim P1 lines:
        #   conversation cache: parked 24511 tokens in 130.1 ms; parked=1 bytes=1046081816 evictions=39 ...
        #   conversation cache: parked 120379 tokens in 812.3 ms; parked=1 bytes=4223875128 evictions=38 ...
        tracked = {"mid-1": session("mid-1", 24511, 1046081816, evictions_at_last_switch=39),
                   "main": session("main", 120379, 4223875128, evictions_at_last_switch=38)}
        ev = ConvEvent(kind="parked", tokens=24511, byte_count=1046081816, evictions=39)
        joined = join_event(ev, tracked)
        self.assertIsNotNone(joined)
        self.assertEqual(joined.session_id, "mid-1")
        self.assertIsNone(join_event(ConvEvent(kind="parked", tokens=99999, byte_count=1, evictions=39), tracked))

    def test_join_event_ambiguous_returns_none(self):
        # two tracked sessions share the join key (tokens, byte_count): the engine logged
        #   conversation cache: parked 24590 tokens ...; parked=3 bytes=3226896568 evictions=41 ...
        # with identical figures for two conversations - the join refuses rather than guesses.
        tracked = {"mid-a": session("mid-a", 24590, 3226896568), "mid-b": session("mid-b", 24590, 3226896568)}
        ev = ConvEvent(kind="parked", tokens=24590, byte_count=3226896568, evictions=41)
        self.assertIsNone(join_event(ev, tracked))

    def test_should_save_skipped_beyond_budget(self):
        # 120379 tokens x 40.4 KiB ~ 4750 MiB > 4096 MiB park budget: the engine cannot hold the big main
        # (P1 saw it skipped: "skip parking (snapshot 5249 MiB exceeds available budget)") -> save it to disk.
        main = session("main", 120379, 4223875128, parked=False)
        self.assertTrue(should_save(main, {"main": main}, CFG))
        # a mid that fits the budget is not flagged by the budget signal (its line: 24511 x 40.4 KiB ~ 967 MiB)
        mid = session("mid", 24511, 1046081816, parked=True)
        self.assertFalse(should_save(mid, {"mid": mid}, CFG))

    def test_should_save_dropped_or_evictions_delta(self):
        # dropped: the engine lost the conversation's parked state -> parked flag is False at the switch.
        mid = session("mid-1", 24511, 1046081816, parked=False, evictions_at_last_switch=39)
        self.assertTrue(should_save(mid, {"mid-1": mid}, CFG))
        # evictions climbed since that session's last switch: the engine's evictions counter is a running
        # global figure carried by every joined event; a join for any session now records a higher figure
        # than this session's baseline (39) - P1 saw it climb to 40 on
        #   conversation cache: parked 24514 tokens ...; parked=3 bytes=3138583432 evictions=40 ...
        # - so park is over-subscribed and even a cleanly-parked mid earns a disk save.
        parked_mid = session("mid-1", 24511, 1046081816, parked=True, evictions_at_last_switch=39)
        newcomer = session("mid-2", 24514, 3138583432, evictions_at_last_switch=40)
        tracked = {"mid-1": parked_mid, "mid-2": newcomer}
        self.assertTrue(should_save(parked_mid, tracked, CFG))

    def test_should_save_false_for_parked_without_pressure(self):
        # P1's cleanly-parked mids (parked=True, evictions unchanged since their switch, within budget)
        # are never saved: park hits beat any save+restore cycle.
        mid = session("mid-1", 24511, 1046081816, parked=True, evictions_at_last_switch=39)
        mid2 = session("mid-2", 24514, 3138470776, parked=True, evictions_at_last_switch=39)
        tracked = {"mid-1": mid, "mid-2": mid2}
        self.assertFalse(should_save(mid, tracked, CFG))
        self.assertFalse(should_save(mid2, tracked, CFG))
        # and the whole ladder sleeps when disabled
        self.assertFalse(should_save(mid, tracked, LadderConfig(False, CFG.slot_save_path, CFG.park_budget_mib)))

    def test_should_save_false_for_aux_below_min_tokens(self):
        # tiny auxiliary calls (title/topic helpers) must never churn saves - engine's own park minimum:
        #   conversation cache: skip parking (70 tokens, below the 8192 minimum; parked=4)
        aux = session("aux-1", 70, 0, parked=False, save_eligible=False)
        self.assertFalse(should_save(aux, {"aux-1": aux}, CFG))
        aux_misflagged = session("aux-2", 70, 0, parked=False, save_eligible=True)
        self.assertFalse(should_save(aux_misflagged, {"aux-2": aux_misflagged}, CFG))

    def test_should_restore_only_with_valid_file_and_missed_park(self):
        # restore only when a ladder file exists AND park demonstrably missed this session.
        # File shapes from P2: session saved 25861 tokens ... to /var/lib/strata/sessions/probe-p2-20261025T0450
        missed = session("mid-1", 24511, 1046081816, parked=False,
                         file_path="/var/lib/strata/sessions/llm-session-mid-1.bin")
        self.assertTrue(should_restore(missed, CFG))
        parked_hit = session("mid-2", 24514, 3138470776, parked=True,
                             file_path="/var/lib/strata/sessions/llm-session-mid-2.bin")
        self.assertFalse(should_restore(parked_hit, CFG))
        no_file = session("mid-3", 24514, 3138583432, parked=False, file_path=None)
        self.assertFalse(should_restore(no_file, CFG))
        self.assertFalse(should_restore(missed, LadderConfig(False, CFG.slot_save_path, CFG.park_budget_mib)))

    def test_prefix_fidelity_saved_is_prefix(self):
        saved = "system prompt\nuser: hello\n"
        self.assertTrue(prefix_fidelity(saved, saved + "assistant: hi there\n"))
        self.assertTrue(prefix_fidelity(saved, saved))  # unchanged turn
        self.assertFalse(prefix_fidelity(None, saved))  # no captured baseline -> not fidelity-checkable

    def test_prefix_fidelity_surgery_skips(self):
        # OpenCode compaction/rewrite: the saved prefix is no longer a byte-prefix of the incoming prompt,
        # so the checkpoint is stale -> skip restore, plain prefill (P3 measured 19 such surgeries in 209 chats).
        saved = "system prompt\nuser: hello\nassistant: hi there\n"
        rewritten = "system prompt\nuser: hello (summarized)\nassistant: hi there\n"
        self.assertFalse(prefix_fidelity(saved, rewritten))
        self.assertFalse(prefix_fidelity(saved, "completely different prompt"))

    # The momo dormancy shape (recheck 2026-10-26): a client re-sends ONE growing user message (the probe
    # harness, single-message chats).  The saved render closes that turn (<|im_end|>) and opens the answer;
    # the next render grows the content INSIDE that closed turn, so the strict byte-compare breaks exactly
    # at the saved turn's closing marker (byte 2773 of 2814 on the real template) even though the engine's
    # own prefix join still holds - the bridge restores of the 21:1x battery reused full checkpoint files on
    # exactly this shape, and the DISC row's engine-side checkpoint restore reused 32768 of 35750 tokens.
    # Fidelity therefore also accepts the saved prefix cut at its last turn-end marker.
    SAVED_GROW = ("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>")

    def test_prefix_fidelity_grown_single_message(self):
        grown = ("<|im_start|>user\nhello and then some more text<|im_end|>\n"
                 "<|im_start|>assistant\n<think>")
        self.assertTrue(prefix_fidelity(self.SAVED_GROW, grown))
        self.assertTrue(prefix_fidelity(self.SAVED_GROW, self.SAVED_GROW))   # unchanged turn still exact

    def test_prefix_fidelity_grown_still_rejects_surgery(self):
        # The relaxation cuts at the saved prefix's LAST turn-end marker: a rewrite diverging before that
        # point (the compaction case) is still rejected - only growth at the conversation's tail passes.
        saved = ("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>\n"
                 "<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n<think>")
        self.assertFalse(prefix_fidelity(saved, saved.replace("hello", "hxllo")))
        self.assertFalse(prefix_fidelity(saved, saved.replace("again", "hxllo")))
        self.assertFalse(prefix_fidelity(saved, "completely different prompt"))
        self.assertTrue(prefix_fidelity(saved, saved.replace(
            "again", "again and some more tail text")))      # growth at the tail: still the same conversation


class JoinTolerance(unittest.TestCase):
    """F1 (final review): the tolerant join.  join_event's exact (tokens, byte_count) key never matches the
    request-path figures - on_turn_done records the 40.4 KiB/token estimate while a park line carries the
    engine's own snapshot_bytes and the generated turn's token delta (7 tokens in the 24.5k evidence pair) -
    so after exact match the ladder joins the NEAREST session within SNAPSHOT_JOIN_TOLERANCE of the event's
    bytes and a token band max(JOIN_TOKEN_TOLERANCE x s.tokens, JOIN_TOKEN_TOLERANCE_FLOOR) = max(1%, 256),
    pinned here.  The near-tie consequence is benign: a mis-join only skips a save or defers a restore - both
    degrade to today's cold prefill; prefix_fidelity still gates every restore, so a mis-join never restores
    a wrong file."""

    # the verbatim evidence pair: DONE "prompt 24504 tokens = 0 reused + 24504 read ..." -> the estimate
    # int(24504 x 40.4 KiB) = 1013722009-ish; the park line 24511 tokens / snapshot_bytes=1046081816.
    MID_ESTIMATE = int(24504 * PARK_COST_KIB_PER_TOKEN * 1024)

    def test_fires_on_the_evidence_pair(self):
        tracked = {"mid": session("mid", 24504, self.MID_ESTIMATE)}
        ev = ConvEvent(kind="parked", tokens=24511, byte_count=1046081816, evictions=39)
        joined = join_event_tolerant(ev, tracked)
        self.assertIsNotNone(joined)                      # exact fails (24511 != 24504); tolerance lands it
        self.assertEqual(joined.session_id, "mid")

    def test_token_band_pinned_at_max_1_percent_256(self):
        # the floor: a 24511-token session's band is 256 (1% would be 245): a 256-token delta joins, 257 does not
        near = session("near", 24511, 1046081816)
        self.assertIsNotNone(join_event_tolerant(ConvEvent(kind="parked", tokens=24511 + 256,
                                                            byte_count=1046081816, evictions=1), {"near": near}))
        self.assertIsNone(join_event_tolerant(ConvEvent(kind="parked", tokens=24511 + 257,
                                                        byte_count=1046081816, evictions=1), {"near": near}))
        # 1% takes over above 25600 tokens: a 30000-token session's band is 300
        big = session("big", 30000, 1241088000)
        self.assertIsNotNone(join_event_tolerant(ConvEvent(kind="parked", tokens=30300,
                                                           byte_count=1241088000, evictions=1), {"big": big}))
        self.assertIsNone(join_event_tolerant(ConvEvent(kind="parked", tokens=30301,
                                                        byte_count=1241088000, evictions=1), {"big": big}))

    def test_byte_band_is_the_snapshot_tolerance(self):
        # 15% (SNAPSHOT_JOIN_TOLERANCE, the machinery skipped_over_budget's MiB join already pins) of the
        # event's byte_count: a gap just inside joins, just outside joins nothing.
        near = session("near", 24511, 1_000_000_000)
        ev_in = ConvEvent(kind="parked", tokens=24600, byte_count=880_000_000, evictions=1)   # gap 120M <= 132M
        self.assertIsNotNone(join_event_tolerant(ev_in, {"near": near}))
        ev_out = ConvEvent(kind="parked", tokens=24600, byte_count=840_000_000, evictions=1)  # gap 160M > 126M
        self.assertIsNone(join_event_tolerant(ev_out, {"near": near}))

    def test_exact_match_keeps_precedence_and_ambiguity_still_refuses(self):
        # Task 5's rules stand: a unique exact match wins over any nearer tolerant candidate; an ambiguous
        # exact key joins nothing (the tolerant join's own tie refuses too - two candidates at equal gap).
        mid = session("mid", 24511, 1046081816)
        near = session("near", 24511, 1046081900)         # byte gap 84: inside tolerance
        ev = ConvEvent(kind="parked", tokens=24511, byte_count=1046081816, evictions=9)
        joined = join_event_tolerant(ev, {"mid": mid, "near": near})
        self.assertIsNotNone(joined)                      # the exact-key session has gap 0: it wins outright
        self.assertEqual(joined.session_id, "mid")
        twin = session("twin", 24511, 1046081816)         # the P1 parked=3 fingerprint shared by two conversations
        self.assertIsNone(join_event_tolerant(ev, {"mid": mid, "twin": twin}))   # tied gap 0: refuses to guess

    def test_only_parked_skipped_and_restored_kinds_join_tolerantly(self):
        near = session("near", 24511, 1046081816)
        for kind in ("parked", "skipped_below_min", "restored"):
            ev = ConvEvent(kind=kind, tokens=24518, byte_count=1046081900, evictions=1)
            self.assertIsNotNone(join_event_tolerant(ev, {"near": near}))       # the token-delta kinds
        for kind in ("dropped", "skipped_over_budget"):   # the no-figures line and the tokened budget refusal
            ev = ConvEvent(kind=kind, tokens=24518, byte_count=1046081900, evictions=1)
            self.assertIsNone(join_event_tolerant(ev, {"near": near}))          # keep exact-only semantics


class SessionLadderIntake(unittest.TestCase):
    """Task 6's intake: ConvCacheLog.drain_events (the engine log's `conversation cache:` lines, one parse,
    consumed exactly once, the monitor card's poll() untouched) and SessionLadder's per-session bookkeeping."""

    # verbatim engine lines (P1 fleet 20261025T0336, P2 restore-extend 20261025T0450), plus SKIP252: NOT a
    # verbatim evidence line - the committed evidence only carries `skip parking (snapshot N MiB ...)` shapes -
    # this line is sourced from the pre-existing ConvCacheLog EVENT regex shape (F9 label), with the brief's
    # pinned big-put-failure figures (`skipped 252953 tokens ... evictions=7`; engine: put() refused, prints "skipped").
    PARK1 = ("strata serve: conversation cache: parked 24511 tokens in 130.1 ms; parked=1 bytes=1046081816 "
             "evictions=39 snapshot_bytes=1046081816 reused_kv_bytes=0\n")
    PARK2 = ("strata serve: conversation cache: parked 24514 tokens in 130.1 ms; parked=2 bytes=2092276296 "
             "evictions=39 snapshot_bytes=1046194480 reused_kv_bytes=0\n")
    SKIP_MIN = ("strata serve: conversation cache: skip parking (70 tokens, below the 8192 minimum; "
                "parked=4)\n")
    SKIP_BUDGET = "strata serve: conversation cache: skip parking (snapshot 5249 MiB exceeds available budget)\n"
    SKIP252 = ("strata serve: conversation cache: skipped 252953 tokens in 1200.0 ms; parked=4 bytes=6400000000 "
               "evictions=7 snapshot_bytes=10451128370 reused_kv_bytes=0\n")
    RESTORE = ("strata serve: conversation cache: restored 16384 tokens (checkpoint) in 71.4 ms; "
               "parked=3 bytes=3828845760\n")
    DROP = ("strata serve: conversation cache: dropped 1 superseded copy of this conversation; parked=1\n")

    def ladder(self):
        return SessionLadder(CFG)

    def drained(self, *lines):
        """drain_events over a temp engine log whose run starts right after the header line."""
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            header = "strata serve: an earlier run\n"
            log.write_text(header + "".join(lines), encoding="utf-8")
            c = ConvCacheLog()
            c.poll(str(log), len(header))  # this run starts after the header: the earlier run is not read
            return c, c.drain_events()

    def test_drain_events_parses_parked_skipped_restored_dropped(self):
        c, evs = self.drained(self.PARK1, self.PARK2, self.SKIP_MIN, self.SKIP_BUDGET, self.SKIP252,
                              self.RESTORE, self.DROP)
        self.assertEqual([e.kind for e in evs],
                         ["parked", "parked", "skipped_below_min", "skipped_over_budget", "skipped_over_budget",
                          "restored", "dropped"])
        p1, p2, smin, sbud, s252, res, drop = evs
        self.assertEqual((p1.kind, p1.tokens, p1.byte_count, p1.evictions), ("parked", 24511, 1046081816, 39))
        # normalization pinned: a parked line's byte_count is the per-conversation bytes - the engine's own
        # snapshot_bytes on the line, which is exactly the cumulative bytes=B delta the park implies
        # (2092276296 - 1046081816 = 1046194480):
        self.assertEqual((p2.kind, p2.tokens, p2.byte_count, p2.evictions), ("parked", 24514, 1046194480, 39))
        self.assertEqual((smin.kind, smin.tokens, smin.byte_count, smin.evictions), ("skipped_below_min", 70, 0, 39))
        self.assertEqual((sbud.kind, sbud.tokens, sbud.byte_count, sbud.snapshot_mib, sbud.evictions),
                         ("skipped_over_budget", 0, 0, 5249, 39))
        self.assertEqual((s252.kind, s252.tokens, s252.byte_count, s252.snapshot_mib, s252.evictions),
                         ("skipped_over_budget", 252953, 10451128370, 0, 7))
        self.assertEqual((res.kind, res.tokens, res.byte_count, res.evictions), ("restored", 16384, 677799526, 7))
        self.assertEqual((drop.kind, drop.tokens, drop.byte_count, drop.evictions), ("dropped", 0, 0, 7))
        self.assertEqual(c.drain_events(), [])  # consumed exactly once: a second drain sees nothing new

    def test_drain_events_leaves_the_monitor_card_unchanged(self):
        # poll()'s card state is exactly what it was before drain_events existed: drain-only skip lines move
        # nothing; park/restore lines move it the same way whether drain_events() reads them or not.
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("strata serve: an earlier run\n", encoding="utf-8")
            start = log.stat().st_size
            c = ConvCacheLog()
            base = c.poll(str(log), start)
            self.assertEqual(base["parks"], 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.SKIP_MIN + self.SKIP_BUDGET)
            st = c.poll(str(log), start)
            self.assertEqual(st, base)  # the card: parks/restores/last_event/last_tokens/bytes all unmoved
            self.assertEqual([e.kind for e in c.drain_events()], ["skipped_below_min", "skipped_over_budget"])
            self.assertEqual(c.drain_events(), [])  # consumed exactly once
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.PARK1 + self.RESTORE)
            st = c.poll(str(log), start)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"],
                              st["last_event"], st["last_tokens"]),
                             (3, 3828845760, 1, 1, "restored", 16384))  # the monitor card, as in test_monitor
            self.assertEqual([e.kind for e in c.drain_events()], ["parked", "restored"])
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.PARK1 + self.DROP)
            st = c.poll(str(log), start)
            self.assertEqual((st["parked"], st["parks"]), (1, 2))
            self.assertEqual([e.kind for e in c.drain_events()], ["parked", "dropped"])

    def test_on_conv_event_joins_park_to_session(self):
        lad = self.ladder()
        lad.on_turn_done("probe-p2-20261025T0450", 25861, 1084175728, 0)  # the P2 mid, with its own park snapshot_bytes
        lad.on_conv_event(ConvEvent(kind="parked", tokens=25861, byte_count=1084175728, evictions=54))
        s = lad.track("probe-p2-20261025T0450")
        self.assertTrue(s.parked)  # a parked event with the mid's token/byte count parks that session
        self.assertEqual((s.tokens, s.byte_count, s.evictions_at_last_switch), (25861, 1084175728, 54))
        # every join records the engine's running evictions counter - also for a skip the session can explain
        lad.on_conv_event(ConvEvent(kind="skipped_below_min", tokens=25861, byte_count=1084175728, evictions=55))
        self.assertEqual(lad.track("probe-p2-20261025T0450").evictions_at_last_switch, 55)
        self.assertTrue(lad.track("probe-p2-20261025T0450").parked)  # a skip line does not unpark
        lad.on_conv_event(ConvEvent(kind="dropped", tokens=0, byte_count=0, evictions=56))  # unjoinable: state intact
        self.assertEqual(lad.track("probe-p2-20261025T0450").evictions_at_last_switch, 55)
        self.assertIsNone(lad.track("untracked"))  # an event never invents a session

    def test_on_conv_event_joins_skipped_over_budget_by_snapshot_size(self):
        lad = self.ladder()
        lad.on_turn_done("main", 120379, 0, 0)      # ~4750 MiB at 40.4 KiB/token
        lad.on_turn_done("mid-1", 24511, 0, 0)      # ~967 MiB
        lad.on_conv_event(ConvEvent(kind="skipped_over_budget", tokens=0, byte_count=0, evictions=38,
                                    snapshot_mib=5249))  # the P1 refusal for the big main (120379 x 40.4 KiB ~ 4750 MiB)
        self.assertEqual(lad.track("main").evictions_at_last_switch, 38)  # joined: the counter landed on main
        self.assertEqual(lad.track("mid-1").evictions_at_last_switch, 0)  # the mid is not blamed for it

    def test_skipped_over_budget_join_tolerance(self):
        # pinned tolerance (SessionLadder.SNAPSHOT_JOIN_TOLERANCE = 0.15, of the snapshot the engine printed):
        # a snapshot within 15% of a tracked session's tokens x 40.4 KiB joins; just outside it does not.
        lad = self.ladder()
        lad.on_turn_done("mid-1", 24511, 0, 0)  # ~967 MiB estimated (tokens x 40.4 KiB)
        lad.on_conv_event(ConvEvent(kind="skipped_over_budget", tokens=0, byte_count=0, evictions=5,
                                    snapshot_mib=850))  # 24511 x 40.4 KiB ~ 967 MiB: |967 - 850| / 850 ~ 13.8% -> inside
        self.assertEqual(lad.track("mid-1").evictions_at_last_switch, 5)
        lad.on_conv_event(ConvEvent(kind="skipped_over_budget", tokens=0, byte_count=0, evictions=6,
                                    snapshot_mib=800))  # ~20.9% -> outside: no join, the counter is not recorded
        self.assertEqual(lad.track("mid-1").evictions_at_last_switch, 5)

    def test_join_tolerance_fires_on_live_figures(self):
        """F1's integration proof, driven with VERBATIM evidence figures: the request path feeds the ladder
        the DONE figures (momo-fleet c1-1 / momo-restore-extend e2: "prompt 24504 tokens = 0 reused + 24504
        read ..." -> byte_count 0 -> the 40.4 KiB/token estimate) and the park line that names the same mid
        carries the engine's real snapshot (self.PARK1, "parked 24511 tokens ... snapshot_bytes=1046081816",
        a 7-token delta) - in production the exact key NEVER matches, so the tolerant join must land this
        pair, park the mid, and hold should_save back at the switch-away while an over-budget session still
        saves."""
        lad = self.ladder()
        lad.on_turn_done("probe-c1-20261025T0341", 24504, 0, 0)      # the DONE figures as on_turn_done sees them
        mid = lad.track("probe-c1-20261025T0341")
        self.assertEqual(mid.byte_count, int(24504 * PARK_COST_KIB_PER_TOKEN * 1024))  # ~1,013,722,000 vs the line's 1,046,081,816
        lad.on_turn_done("main-252953", 252953, 0, 0)                # ~9,980 MiB: beyond the 4096 MiB budget
        for ev in self.drained(self.PARK1)[1]:
            lad.on_conv_event(ev)                                     # the verbatim park line
        self.assertTrue(mid.parked)                                   # the join landed: parked is set
        self.assertEqual(mid.evictions_at_last_switch, 39)            # ...with the engine's counter recorded on the mid
        self.assertEqual(lad.track("main-252953").evictions_at_last_switch, 0)   # the big main is not blamed for it
        self.assertFalse(should_save(mid, lad.tracked, CFG))          # park holds the mid: no save at switch-away
        self.assertTrue(should_save(lad.track("main-252953"), lad.tracked, CFG))  # the over-budget one still saves

    def test_note_switch_away_clears_parked_after_should_save(self):
        # Task 7's ordering (pinned by should_save's docstring): evaluate should_save while parked still
        # reflects park; note_switch_away is what clears it at the switch-away.
        lad = self.ladder()
        lad.on_turn_done("mid-1", 24511, 1046081816, 0)
        lad.on_conv_event(ConvEvent(kind="parked", tokens=24511, byte_count=1046081816, evictions=39))
        self.assertFalse(should_save(lad.track("mid-1"), lad.tracked, CFG))  # cleanly parked, no pressure: no save
        lad.note_switch_away("mid-1")
        self.assertFalse(lad.track("mid-1").parked)
        self.assertTrue(should_save(lad.track("mid-1"), lad.tracked, CFG))  # unparked after the clear: the dropped signature

    def test_on_turn_done_marks_eligibility_by_budget(self):
        lad = self.ladder()
        big_bytes = int(252953 * PARK_COST_KIB_PER_TOKEN * 1024)  # 252953 x 40.4 KiB ~ 9,980 MiB > 4096
        lad.on_turn_done("main", 252953, big_bytes, 0)  # the P1 big save: beyond the park budget
        self.assertTrue(lad.track("main").save_eligible)  # the engine can never hold it
        lad.on_turn_done("main-zero-bytes", 120379, 0, 0)  # no byte figure given: estimated at tokens x 40.4 KiB
        self.assertTrue(lad.track("main-zero-bytes").save_eligible)  # ~4,750 MiB > 4096
        lad.on_turn_done("aux-1", 70, int(70 * PARK_COST_KIB_PER_TOKEN * 1024), 70)
        self.assertFalse(lad.track("aux-1").save_eligible)  # aux-sized: below the engine's own park minimum
        lad.on_turn_done("mid-1", 24514, 1046194480, 0)  # P1 mid-2's own snapshot bytes (~998 MiB), inside the budget
        self.assertTrue(lad.track("mid-1").save_eligible)  # a mid is save-worthy under park pressure (Task 5)
        self.assertEqual(lad.track("mid-1").byte_count, 1046194480)  # recorded as given, not replaced by the estimate
        self.assertEqual(lad.track("main").tokens, 252953)
        lad.on_turn_done("main", 253100, int(253100 * PARK_COST_KIB_PER_TOKEN * 1024), 4096)  # a later grown turn re-anchors the session's figures
        self.assertEqual(lad.track("main").tokens, 253100)

    def test_eviction_delta_flags_pressure(self):
        # built entirely from the live intake: m-1 parks at the engine's evictions=39, cleanly and within budget:
        # no pressure, no save.  Then P1's verbatim line -
        #   conversation cache: parked 24514 tokens in 124.9 ms; parked=3 bytes=3138583432 evictions=40 snapshot_bytes=1046194480 ...
        # - joins m-2 (normalized to its snapshot_bytes) and records the climb; m-1's baseline (39) now trails
        # the engine's counter, so park is over-subscribed and even m-1's clean park earns a disk save.
        lad = self.ladder()
        lad.on_turn_done("m-1", 24511, 1046081816, 0)
        lad.on_conv_event(ConvEvent(kind="parked", tokens=24511, byte_count=1046081816, evictions=39))
        self.assertTrue(lad.track("m-1").parked)
        self.assertFalse(should_save(lad.track("m-1"), lad.tracked, CFG))  # baseline level with the engine: no pressure
        lad.on_turn_done("m-2", 24514, 1046194480, 0)
        lad.on_conv_event(ConvEvent(kind="parked", tokens=24514, byte_count=1046194480, evictions=40))
        self.assertEqual(lad.track("m-2").evictions_at_last_switch, 40)
        self.assertTrue(should_save(lad.track("m-1"), lad.tracked, CFG))  # cleanly parked, yet the climb flags it

    def test_retention_lru_caps_files(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = LadderConfig(enabled=True, slot_save_path=d, park_budget_mib=4096)  # retention_max_files: 16
            lad = SessionLadder(cfg)
            for i in range(1, 17):
                lad.on_turn_done(f"sid-{i}", 10000, int(10000 * PARK_COST_KIB_PER_TOKEN * 1024), 0)
                path = os.path.join(d, f"llm-session-sid-{i}.bin")
                Path(path).write_bytes(b"checkpoint")
                lad.note_file_saved(f"sid-{i}", path)
            self.assertEqual(len(lad.saved_files), 16)
            self.assertTrue(lad.track("sid-1"))  # exactly at the cap: nothing evicted yet
            lad.on_turn_done("sid-17", 10000, int(10000 * PARK_COST_KIB_PER_TOKEN * 1024), 0)
            path17 = os.path.join(d, "llm-session-sid-17.bin")
            Path(path17).write_bytes(b"checkpoint")
            lad.note_file_saved("sid-17", path17)  # the 17th tracked file
            self.assertEqual(len(lad.saved_files), cfg.retention_max_files)  # the cap holds
            self.assertIsNone(lad.track("sid-1"))  # the least-recent file's session state is gone...
            self.assertNotIn(os.path.join(d, "llm-session-sid-1.bin"), lad.saved_files)  # ...the file leaves the ledger...
            self.assertFalse(os.path.exists(os.path.join(d, "llm-session-sid-1.bin")))  # ...and is removed: the cap is real
            self.assertTrue(lad.track("sid-2"))  # everything else survives, newest last
            self.assertEqual(lad.saved_files[-1], path17)
            # the order is LRU, not insertion: re-saving a session (overwrite in place) moves its file to the recent end
            lad.note_file_saved("sid-2", os.path.join(d, "llm-session-sid-2.bin"))
            lad.on_turn_done("sid-18", 10000, int(10000 * PARK_COST_KIB_PER_TOKEN * 1024), 0)
            path18 = os.path.join(d, "llm-session-sid-18.bin")
            Path(path18).write_bytes(b"checkpoint")
            lad.note_file_saved("sid-18", path18)
            self.assertTrue(lad.track("sid-2"))  # re-saved: kept...
            self.assertIsNone(lad.track("sid-3"))  # ...sid-3 is the least recent now, so it goes
            self.assertEqual(len(lad.saved_files), cfg.retention_max_files)


class _FakeEngine:
    """An engine as the ladder's config reader sees it (Task 9): the spawn args, else the INFO line it
    reported - the two sources conversation_slots() reads --conversation-cache-slots from."""

    def __init__(self, args, info=None):
        self.spawn = ("exe", list(args), ".", None, {}, False, None)
        self.info = info or {}


class LadderConfigFromJson(unittest.TestCase):
    """Task 9: strata.json's `session_ladder` block becomes a LadderConfig through serve.server.ladder_config_of,
    with the engine's own --conversation-cache-mib as its park budget."""

    def test_ladder_config_from_json(self):
        from serve.server import ladder_config_of
        engine = _FakeEngine(["--conversation-cache-mib", "8192"])
        cfg = ladder_config_of({"session_ladder": {"idle_save_s": 8.0, "retention_max_files": 16}},
                               engine, "/var/lib/strata/sessions")
        self.assertTrue(cfg.enabled)
        self.assertEqual((cfg.idle_save_s, cfg.retention_max_files), (8.0, 16))  # the dict's values
        self.assertEqual(cfg.park_budget_mib, 8192)  # from the fake engine args, like conversation_slots()
        self.assertEqual(cfg.slot_save_path, "/var/lib/strata/sessions")  # the service's slot save dir
        # absent or False: disabled, exactly what Service.__init__ builds today
        for off_cfg in ({}, {"session_ladder": False}, {"session_ladder": True}):
            cfg = ladder_config_of(off_cfg, engine, "/var/lib/strata/sessions")
            self.assertFalse(cfg.enabled)
            self.assertEqual(cfg.park_budget_mib, 0)  # the disabled ladder's budget today
        # no --conversation-cache-mib argument: the INFO line names it instead (lazy-start order).
        # Neither source: park_budget_mib stays None - faithful to conversation_slots() - and an enabled
        # ladder whose budget resolves to None refuses to start (no silent 0: over_budget would be true
        # for EVERY turn, so idle-save/unload would save aux checkpoints - the churn the design forbids).
        cfg = ladder_config_of({"session_ladder": {}}, _FakeEngine([], {"conversation_cache_mib": 4096}), "/d")
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.park_budget_mib, 4096)
        self.assertIsNone(ladder_config_of({"session_ladder": {}}, _FakeEngine([]), "/d").park_budget_mib)
        from serve.server import start_ladder_of
        with self.assertRaises(SystemExit) as cm:  # main()'s guard: refuse, like the slot_save_path one
            start_ladder_of({"session_ladder": {"idle_save_s": 8.0, "retention_max_files": 16}},
                            _FakeEngine([]), "/var/lib/strata/sessions")
        self.assertIn("--conversation-cache-mib", str(cm.exception))  # the message names the flag
        cfg = start_ladder_of({"session_ladder": {"idle_save_s": 8.0, "retention_max_files": 16}},
                              engine, "/var/lib/strata/sessions")  # a resolvable budget passes
        self.assertEqual(cfg.park_budget_mib, 8192)
        # a dict absent key keeps LadderConfig's pinned defaults
        cfg = ladder_config_of({"session_ladder": {}}, engine, "/d")
        self.assertEqual((cfg.idle_save_s, cfg.retention_max_files), (8.0, 16))
        # pre_save_idle_s (unload-works Task 3): an absent key or an explicit null keeps the adaptive
        # default None (quiet >= this save's expected length); a number pins the quiet gate.
        self.assertIsNone(cfg.pre_save_idle_s)
        cfg = ladder_config_of({"session_ladder": {"pre_save_idle_s": None}}, engine, "/d")
        self.assertIsNone(cfg.pre_save_idle_s)
        cfg = ladder_config_of({"session_ladder": {"pre_save_idle_s": 30}}, engine, "/d")
        self.assertEqual(cfg.pre_save_idle_s, 30.0)

    def test_start_ladder_of_refuses_an_explicit_zero_budget(self):
        # F5 (final review): `park_budget_mib is None` let an explicit --conversation-cache-mib 0 start an
        # enabled budget-0 ladder: over_budget is then true for EVERY turn, so the aux churn the design
        # forbids is exactly what it would do.  Same refusal as the None budget, same message naming the flag.
        from serve.server import ladder_config_of, start_ladder_of
        zero = _FakeEngine(["--conversation-cache-mib", "0"])
        with self.assertRaises(SystemExit) as cm:
            start_ladder_of({"session_ladder": {"idle_save_s": 8.0, "retention_max_files": 16}}, zero,
                            "/var/lib/strata/sessions")
        self.assertIn("--conversation-cache-mib", str(cm.exception))
        # a DISABLED ladder still carries budget 0 (what Service.__init__ builds); the guard is for enabled ones
        self.assertEqual(ladder_config_of({"session_ladder": False}, zero, "/d").park_budget_mib, 0)
        self.assertFalse(ladder_config_of({"session_ladder": False}, zero, "/d").enabled)

    def test_ladder_config_of_names_a_malformed_dict_value(self):
        # F6 (final review): a malformed strata.json value ("idle_save_s": "eight") raised a bare ValueError
        # at startup.  The conversions are wrapped: SystemExit names the bad key AND the bad value.
        from serve.server import ladder_config_of
        engine = _FakeEngine(["--conversation-cache-mib", "8192"])
        for key, bad in (("idle_save_s", "eight"), ("retention_max_files", "many"),
                         ("pre_save_idle_s", "soon")):
            with self.assertRaises(SystemExit) as cm:
                ladder_config_of({"session_ladder": {key: bad}}, engine, "/d")
            self.assertIn(key, str(cm.exception))            # names the bad key...
            self.assertIn(bad, str(cm.exception))            # ...and the value that failed to convert


if __name__ == "__main__":
    unittest.main()
