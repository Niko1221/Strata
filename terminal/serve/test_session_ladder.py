"""serve/test_session_ladder.py - the ladder's pure decision core: event join, save/restore policy,
prefix fidelity.  No I/O, no engine, no server.py imports.

    python -m pytest serve/test_session_ladder.py -v

Fixtures are verbatim `conversation cache:` event lines from the P1 fleet probe
(nixos-config docs/superpowers/spikes/evidence/momo-fleet-20261025T0336.md) and the P2 restore probe
(momo-restore-extend-20261025T0450.md); ConvEvent mirrors what Task 6's ConvCacheLog.drain_events
will hand over.
"""
from __future__ import annotations

import unittest

from serve.session_ladder import (
    ConvEvent,
    LadderConfig,
    TrackedSession,
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


if __name__ == "__main__":
    unittest.main()
