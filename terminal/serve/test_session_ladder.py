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
    PARK_COST_KIB_PER_TOKEN,
    ConvEvent,
    LadderConfig,
    TrackedSession,
    SessionLadder,
    extract_session_id,
    join_event,
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


class SessionLadderIntake(unittest.TestCase):
    """Task 6's intake: ConvCacheLog.drain_events (the engine log's `conversation cache:` lines, one parse,
    consumed exactly once, the monitor card's poll() untouched) and SessionLadder's per-session bookkeeping."""

    # verbatim engine lines (P1 fleet 20261025T0336, P2 restore-extend 20261025T0450), plus the brief's
    # pinned big-put-failure line (`skipped 252953 tokens ... evictions=7`; engine: put() refused, prints "skipped").
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


if __name__ == "__main__":
    unittest.main()
