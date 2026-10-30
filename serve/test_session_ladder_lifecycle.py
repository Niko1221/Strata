"""serve/test_session_ladder_lifecycle.py - Task 8: the ladder's lifecycle hooks.

on_idle / on_unload / on_start over the recording-mock engine idiom of
serve/test_session_ladder_wiring.py (a SimpleNamespace service whose engine.session_file is a
mock.Mock; a `save` also materialises the file the engine would have written, so the start hook's
"has a valid file" check sees a real path).  The idle window, the save-all unload, and the
warm-restore handoff through svc.restore_after_load are checked without a server; the disabled
config must keep all three hooks inert.

    python -m pytest serve/test_session_ladder_lifecycle.py -v
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service
from serve.session_ladder import ConvEvent, LadderConfig, SessionLadder, extract_session_id, should_restore
from serve.test_session_ladder_wiring import LadderHarness, ladder_service

MID = 20000                     # a mid: above the engine's 8192-token park minimum
MID_BYTES = 800_000_000         # its joined byte figure (the engine's snapshot scale)
P1 = "HEAD one"
P2 = "HEAD two"

# The OpenCode 2.1.4 subagent (child session) request shape, observed in the shipped binary: the affinity
# family (X-Session-Id / x-session-affinity / x-opencode-session) carries the PARENT id - the child's own
# id sits only in x-opencode-session-id.  See test_session_ladder.py::test_extract_session_id_opencode_child.
MAIN = "ses_main00000000000000000000"
CHILD = "ses_child0000000000000000000"
CHILD_HEADERS = {"x-opencode-session-id": CHILD, "x-opencode-parent-session-id": MAIN,
                 "x-session-affinity": MAIN, "X-Session-Id": MAIN, "x-parent-session-id": MAIN,
                 "x-opencode-project": "proj", "x-opencode-session": MAIN, "x-opencode-client": "cli"}


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        cfg = LadderConfig(enabled=True, slot_save_path=self.dir.name, park_budget_mib=4096)
        self.calls = []

        def session_file(action, path):
            self.calls.append((action, path))
            if action == "save":
                Path(path).write_bytes(b"checkpoint")   # what the engine's SAVE would have written
            return {"tokens": MID}

        self.engine = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None), ended=False, batch=0,
                                      session_file=mock.Mock(side_effect=session_file))
        self.service = SimpleNamespace(engine=self.engine, slot_save_path=self.dir.name,
                                       restore_after_load=None)
        self.ladder = SessionLadder(cfg)

    def seen(self, session_id, tokens=MID, byte_count=MID_BYTES):
        """A completed mid turn: tracked state created (save_eligible, dirty)."""
        self.ladder.on_turn_done(session_id, tokens, 0, 0)
        s = self.ladder.tracked[session_id]
        s.byte_count = byte_count

    def actions(self, wanted):
        return [a for a, _ in self.calls if a == wanted]

    def path(self, session_id):
        return str(Path(self.dir.name) / f"llm-session-{session_id}.bin")

    def test_idle_saves_eligible_after_window(self):
        self.seen("ses-a")                              # the active mid, dirty since its last turn
        self.ladder.on_arrival(self.service, "ses-a", P1)
        self.ladder.on_idle(self.service, 3.0)          # below the 8 s idle_save_s window: nothing
        self.assertEqual(self.calls, [])
        self.ladder.on_idle(self.service, 8.0)          # the window: one save of the last-active session
        self.assertEqual(self.calls, [("save", self.path("ses-a"))])
        s = self.ladder.tracked["ses-a"]
        self.assertEqual(s.file_path, self.path("ses-a"))
        self.assertTrue(s.saved_prefix.startswith("HEAD one"))   # recorded as the checkpoint's prefix
        self.ladder.on_idle(self.service, 30.0)         # saved and unchanged since: skipped
        self.assertEqual(len(self.actions("save")), 1)
        self.ladder.on_turn_done("ses-a", MID, 0, 0)    # a new turn dirties it again
        self.ladder.on_idle(self.service, 8.0)          # ...and earns the next save
        self.assertEqual(len(self.actions("save")), 2)

    def test_unload_saves_all_eligible(self):
        self.seen("ses-a")
        self.seen("ses-b")
        self.ladder.on_turn_done("aux", 100, 0, 0)      # aux-sized: not save-eligible, never churns
        self.ladder.on_arrival(self.service, "ses-a", P1)
        self.ladder.on_arrival(self.service, "ses-b", P2)     # the switch already saved ses-a clean
        self.assertEqual(self.actions("save"), ["save"])
        self.ladder.on_unload(self.service)             # the engine is about to go: save the rest
        self.assertEqual(self.actions("save"), ["save", "save"])
        self.assertEqual(self.calls[-1], ("save", self.path("ses-b")))
        self.assertTrue(Path(self.path("ses-a")).exists())     # ses-a's clean checkpoint is skipped, not
        self.assertEqual(len(self.actions("save")), 2)         # rewritten - and nothing saved for aux
        self.ladder.on_unload(self.service)             # all clean now: no churn
        self.assertEqual(len(self.actions("save")), 2)

    def test_subagent_child_is_tracked_as_its_own_session(self):
        # OpenCode 2.1.4 subagents ride the parent's affinity headers (CHILD_HEADERS above): the ladder must
        # resolve the CHILD's own id, or the subagent is invisible - never saved, and its turns land on the
        # main session's ledger (live momo 2026-10-28/30: only main session ids ever reached the ladder).
        child = extract_session_id(CHILD_HEADERS)         # the id the request path would pass on_arrival
        self.assertEqual(child, CHILD)                    # pinned first: identity, then the switch choreography
        self.seen(MAIN)                                   # main completed a turn: dirty, park unknown
        self.ladder.on_arrival(self.service, MAIN, P1)    # main takes the slot (no outgoing)
        self.ladder.on_arrival(self.service, child, P2)   # the subagent arrives: main switches away...
        self.assertEqual(self.calls, [("save", self.path(MAIN))])
        self.ladder.on_turn_done(child, MID, MID_BYTES, 0)          # ...the subagent completes a turn (mid-sized)
        self.assertTrue(self.ladder.tracked[CHILD].dirty)           # dirty on ITS OWN ledger, not main's
        self.assertFalse(self.ladder.tracked[MAIN].dirty)           # main stayed clean since its switch save
        self.ladder.on_arrival(self.service, MAIN, P1)              # main returns: the child switches away...
        # on_arrival's engine I/O window runs the switch save first, then the incoming restore.
        self.assertEqual(self.calls, [("save", self.path(MAIN)), ("save", self.path(CHILD)),
                                      ("restore", self.path(MAIN))])  # child earns its own checkpoint file;
        self.assertEqual(self.actions("restore"), ["restore"])      # main comes back from its file (park lost)
        self.assertTrue(Path(self.path(CHILD)).exists())

    def test_child_files_retire_before_older_main_files(self):
        # A (retirement order): an over-cap save retires a CHILD file, never an older MAIN one.  A
        # subagent fleet may consume ledger slots - what it may not do is evict the expensive file: a
        # child re-prefills a subagent conversation, a main its whole multi-GB context.
        cfg = LadderConfig(enabled=True, slot_save_path=self.dir.name, park_budget_mib=4096,
                           retention_max_files=3)
        ladder = SessionLadder(cfg)
        other = "ses_other00000000000000000000"
        c1, c2 = "ses_c100000000000000000000", "ses_c20000000000000000000"
        self.calls.clear()
        ladder.on_arrival(self.service, MAIN, P1)
        ladder.on_turn_done(MAIN, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, other, P2)                 # MAIN away: save MAIN   [ledger 1]
        ladder.on_turn_done(other, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, c1, P1, parent_id=MAIN)    # other away: save other [ledger 2]
        ladder.on_turn_done(c1, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, c2, P1, parent_id=MAIN)    # c1 away: save c1       [ledger 3 = cap]
        ladder.on_turn_done(c2, MID, MID_BYTES, 0)
        self.assertEqual(ladder.saved_files, [self.path(MAIN), self.path(other), self.path(c1)])
        ladder.on_arrival(self.service, "ses_third0000000000000000000", P2)   # c2 away: save c2 -> over cap
        self.assertEqual(ladder.saved_files, [self.path(MAIN), self.path(other), self.path(c2)])  # c1 retired, not MAIN
        self.assertIsNone(ladder.track(c1))                         # the retired file's state retires with it
        self.assertFalse(Path(self.path(c1)).exists())              # ...and it leaves the disk
        self.assertTrue(Path(self.path(MAIN)).exists())             # the OLDER main file stands untouched

    def test_child_cap_bounds_children_below_the_file_cap(self):
        # A (separate cap): retention_max_child_files caps CHILD files on top of the global cap - a
        # fleet cannot occupy the whole ledger while retention_max_files still has room.
        cfg = LadderConfig(enabled=True, slot_save_path=self.dir.name, park_budget_mib=4096,
                           retention_max_child_files=2)
        ladder = SessionLadder(cfg)
        c1, c2, c3 = ("ses_c100000000000000000000", "ses_c2000000000000000000",
                      "ses_c30000000000000000000")
        self.calls.clear()
        ladder.on_arrival(self.service, MAIN, P1)
        ladder.on_turn_done(MAIN, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, c1, P1, parent_id=MAIN)    # MAIN away: save MAIN (main file)
        ladder.on_turn_done(c1, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, c2, P1, parent_id=MAIN)    # c1 away: child 1
        ladder.on_turn_done(c2, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, c3, P1, parent_id=MAIN)    # c2 away: child 2 = the cap
        ladder.on_turn_done(c3, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, "ses_main2000000000000000000", P2)   # c3 away: child 3 > cap 2
        self.assertEqual(ladder.saved_files, [self.path(MAIN), self.path(c2), self.path(c3)])  # c1 (oldest child) retired
        self.assertFalse(Path(self.path(c1)).exists())
        self.assertTrue(Path(self.path(MAIN)).exists())

    def test_child_ttl_sweep_retires_dead_children_keeps_everything_else(self):
        # B (the TTL): a finished OpenCode subagent never speaks to the engine again - silence is the
        # only completion signal there is.  A clean CHILD file untouched past child_ttl_s is dead
        # weight and the sweep retires it: file, ledger entry, tracked state, index entry.  It never
        # touches a MAIN file, a dirty child (its save is still owed), or the current session.
        cfg = LadderConfig(enabled=True, slot_save_path=self.dir.name, park_budget_mib=4096,
                           child_ttl_s=1000.0)
        ladder = SessionLadder(cfg)
        dead = "ses_dead00000000000000000000"
        dirt = "ses_dirt00000000000000000000"
        self.calls.clear()
        ladder.on_arrival(self.service, MAIN, P1)
        ladder.on_turn_done(MAIN, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, dead, P1, parent_id=MAIN)  # MAIN away: save MAIN
        ladder.on_turn_done(dead, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, dirt, P1, parent_id=MAIN)  # dead away: save dead (clean child file)
        ladder.on_turn_done(dirt, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, MAIN, P2)                  # dirt away: save dirt (clean file)
        ladder.on_turn_done(dirt, MID, MID_BYTES, 0)               # dirt dirties again: save owed
        for sid in (MAIN, dead, dirt):                             # age everything past the TTL
            ladder.tracked[sid].last_active -= 3600.0
            ladder.tracked[sid].last_save -= 3600.0
        retired = ladder.sweep_child_ttl()
        self.assertEqual(retired, 1)                               # only `dead` qualified
        self.assertFalse(Path(self.path(dead)).exists())           # file off disk...
        self.assertNotIn(self.path(dead), ladder.saved_files)      # ...ledger entry gone...
        self.assertIsNone(ladder.track(dead))                      # ...tracked state gone...
        data = json.loads((Path(self.dir.name) / "ladder-index.json").read_text())
        self.assertNotIn(self.path(dead), data["files"])           # ...and the sidecar agrees
        self.assertTrue(Path(self.path(MAIN)).exists())            # the main survives (no parent mark)
        self.assertTrue(Path(self.path(dirt)).exists())            # the dirty child survives (save owed)

    def test_child_ttl_sweep_leaves_the_current_session_alone(self):
        cfg = LadderConfig(enabled=True, slot_save_path=self.dir.name, park_budget_mib=4096,
                           child_ttl_s=1000.0)
        ladder = SessionLadder(cfg)
        dead = "ses_dead00000000000000000000"
        self.calls.clear()
        ladder.on_arrival(self.service, MAIN, P1)
        ladder.on_turn_done(MAIN, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, dead, P1, parent_id=MAIN)  # MAIN away: save MAIN
        ladder.on_turn_done(dead, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, MAIN, P2)                  # dead away: save dead (clean child file)
        ladder.tracked[MAIN].last_active -= 3600.0                 # age everything past the TTL...
        ladder.tracked[MAIN].last_save -= 3600.0
        ladder.tracked[dead].last_active -= 3600.0
        ladder.tracked[dead].last_save -= 3600.0
        ladder.on_arrival(self.service, dead, "P9 unrelated text")  # dead becomes the CURRENT session
        # (its saved prefix no longer matches this arrival, so prefix_fidelity refuses the restore -
        # the file stays and the slot is what the sweep must not strip of its safety net)
        self.assertEqual(ladder.sweep_child_ttl(), 0)              # current: swept around
        self.assertTrue(Path(self.path(dead)).exists())
        self.assertIn(self.path(dead), ladder.saved_files)

    def test_child_ttl_zero_disables_the_sweep(self):
        cfg = LadderConfig(enabled=True, slot_save_path=self.dir.name, park_budget_mib=4096,
                           child_ttl_s=0.0)
        ladder = SessionLadder(cfg)
        dead = "ses_dead00000000000000000000"
        self.calls.clear()
        ladder.on_arrival(self.service, MAIN, P1)
        ladder.on_turn_done(MAIN, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, dead, P1, parent_id=MAIN)
        ladder.on_turn_done(dead, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, MAIN, P2)                  # save dead's checkpoint
        for sid in (MAIN, dead):
            ladder.tracked[sid].last_active -= 3600.0
            ladder.tracked[sid].last_save -= 3600.0
        self.assertEqual(ladder.sweep_child_ttl(), 0)              # 0 = off: nothing retires
        self.assertTrue(Path(self.path(dead)).exists())

    def test_parent_identity_survives_the_index_round_trip(self):
        # The child mark rides the sidecar: a restarted ladder still retires the ADOPTED child file
        # before the older main file (without the adopted parent, the plain-LRU victim would be MAIN).
        cfg = LadderConfig(enabled=True, slot_save_path=self.dir.name, park_budget_mib=4096,
                           retention_max_files=2)
        ladder = SessionLadder(cfg)
        other = "ses_other00000000000000000000"
        self.calls.clear()
        ladder.on_arrival(self.service, MAIN, P1)
        ladder.on_turn_done(MAIN, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, CHILD, P1, parent_id=MAIN)  # MAIN away: save MAIN    [ledger 1]
        ladder.on_turn_done(CHILD, MID, MID_BYTES, 0)
        ladder.on_arrival(self.service, other, P2)                  # CHILD away: save CHILD  [ledger 2 = cap]
        self.assertEqual(ladder.saved_files, [self.path(MAIN), self.path(CHILD)])
        fresh = SessionLadder(cfg)                                   # restart: adopt from the sidecar
        self.assertEqual(fresh.parent_of.get(CHILD), MAIN)           # the child mark survived
        self.assertEqual(fresh.last_active, CHILD)                   # (the last-saved file's owner)
        main2 = "ses_main20000000000000000000"
        fresh.on_turn_done(CHILD, MID, MID_BYTES, 0)                 # dirties the adopted child
        fresh.on_arrival(self.service, main2, P1)                    # CHILD away: re-save CHILD (moves to recent end)
        self.assertEqual(fresh.saved_files, [self.path(MAIN), self.path(CHILD)])
        fresh.on_turn_done(main2, MID, MID_BYTES, 0)
        fresh.on_arrival(self.service, "ses_third0000000000000000000", P2)   # main2 away: save -> over cap 2
        self.assertEqual(fresh.saved_files, [self.path(MAIN), self.path(main2)])  # CHILD retired, not the older MAIN
        self.assertFalse(Path(self.path(CHILD)).exists())
        self.assertTrue(Path(self.path(MAIN)).exists())

    def test_idle_save_then_switch_away_does_not_re_save(self):
        # F3 (final review): the switch-away save honors the dirty marker - an idle-save already wrote this
        # session's checkpoint and nothing dirtied it since, so the switch must not rewrite the identical file.
        self.seen("ses-a")
        self.ladder.on_arrival(self.service, "ses-a", P1)
        self.ladder.on_idle(self.service, 8.0)                # idle save: checkpoint written, session clean
        self.assertEqual(self.actions("save"), ["save"])
        self.ladder.on_arrival(self.service, "ses-b", P2)     # switch-away: clean + filed -> skipped
        self.assertEqual(self.actions("save"), ["save"])

    def test_a_consumed_checkpoint_re_saves_on_the_next_switch(self):
        # F3's documented half: a session whose file was CONSUMED by a warm restore (the start hook's
        # handoff retires file_path) re-saves at its next switch-away even while dirty is False - engine
        # park does not hold over-budget mains and the file is gone, so the next save is the protection
        # that keeps a checkpoint available.  Pinned as intended behavior.
        self.seen("ses-a")
        self.ladder.on_arrival(self.service, "ses-a", P1)     # ses-a active
        self.ladder.on_arrival(self.service, "ses-b", P2)     # switch-away: checkpointed, clean
        self.assertEqual(self.actions("save"), ["save"])
        self.ladder.on_arrival(self.service, "ses-a", P1)     # back: ses-a is last active again, prefix intact
        self.ladder.on_start(self.service)                    # the warm handoff consumes the file...
        self.assertIsNone(self.ladder.tracked["ses-a"].file_path)
        self.assertFalse(self.ladder.tracked["ses-a"].dirty)  # ...still clean from its own save
        self.ladder.on_arrival(self.service, "ses-b", P2)     # switch-away: file gone -> re-save anyway
        self.assertEqual(self.actions("save"), ["save", "save"])

    def test_lifecycle_engine_io_runs_outside_the_ladder_lock(self):
        # F4 (final review): the idle save and the unload save-all also run their engine.session_file calls
        # outside the ladder lock (decision + ledger windows take it, the engine I/O between them does not).
        probe = []

        def session_file(action, path):
            got = self.ladder._lock.acquire(blocking=False)
            if got:
                self.ladder._lock.release()
            probe.append((action, got))
            self.calls.append((action, path))
            if action == "save":
                Path(path).write_bytes(b"checkpoint")
            return {"tokens": MID}

        self.engine.session_file = mock.Mock(side_effect=session_file)
        self.seen("ses-a")
        self.seen("ses-b")
        self.ladder.on_arrival(self.service, "ses-a", P1)      # ses-a active (last_active)
        self.ladder.on_idle(self.service, 8.0)                 # idle save of ses-a...
        self.ladder.on_unload(self.service)                    # ...then the unload save-all (ses-b, dirty)
        self.assertEqual([a for a, _ in probe], ["save", "save"])
        self.assertTrue(all(got for _, got in probe))          # lock free at both engine calls

    def _two_checkpoints(self):
        """ses-a and ses-b both checkpointed by real switches; ses-b is the last-active session with a
        valid file and a fidelity-able prefix - the state a warm restore is queued for."""
        self.seen("ses-a")
        self.seen("ses-b")
        self.ladder.on_arrival(self.service, "ses-a", P1)
        self.ladder.on_arrival(self.service, "ses-b", P2)      # ses-a saved (prefix "HEAD one")
        self.ladder.on_arrival(self.service, "ses-a", P1)      # ses-b saved (prefix "HEAD two")
        self.ladder.on_arrival(self.service, "ses-b", P2)      # back to ses-b: it is the last active
        return self.ladder.tracked["ses-a"], self.ladder.tracked["ses-b"]

    def test_start_warms_last_active_only(self):
        a, b = self._two_checkpoints()
        self.assertEqual(self.ladder.last_active, "ses-b")
        before = list(self.calls)                        # the setup's switches already issued their own
        # save/restore calls; what matters is what on_start itself adds
        self.ladder.on_start(self.service)
        self.assertEqual(self.service.restore_after_load, self.path("ses-b"))   # the single warm restore,
        # queued through the existing restore_after_load -> ensure_loaded mechanism (which consumes and
        # removes the file): the ladder hands its ledger entry over with it.
        self.assertIsNone(b.file_path)
        self.assertEqual(self.ladder.saved_files, [self.path("ses-a")])
        self.assertEqual(a.file_path, self.path("ses-a"))      # the others keep their checkpoints
        self.assertEqual(self.calls, before)                   # on_start queues, it does not restore:
        # the warm session_file("restore") is the load path's (the same images_api.restore_session call
        # an image job's pending restore already uses)

    def test_bookkeeping_reset_after_start(self):
        self.seen("ses-a")
        self.ladder.on_arrival(self.service, "ses-a", P1)
        self.ladder.on_arrival(self.service, "ses-b", P2)      # switch-away: ses-a checkpointed
        self.ladder.on_conv_event(ConvEvent(kind="parked", tokens=MID, byte_count=MID_BYTES, evictions=3))
        self.assertTrue(self.ladder.tracked["ses-a"].parked)   # the pre-start park state the event joined
        self.ladder.on_start(self.service)
        self.assertFalse(self.ladder.tracked["ses-a"].parked)  # a fresh engine parks nothing: every flag
        # resets, while the tracked entry and its file_path survive - so the next switch treats park as
        # empty and the session is restore-eligible:
        self.assertTrue(should_restore(self.ladder.tracked["ses-a"], self.ladder.cfg))
        self.assertIsNone(self.service.restore_after_load)     # ses-b (last active) has no file: no warm
        self.ladder.on_arrival(self.service, "ses-a", P1 + " tail")   # the next switch back
        self.assertEqual(self.actions("restore"), ["restore"])
        self.assertEqual(self.calls[-1], ("restore", self.path("ses-a")))

    def test_parked_flag_self_heals_on_zero_reuse_done(self):
        # momo dormancy (recheck 2026-10-26): the engine's park evicts copies under budget pressure but its
        # eviction lines name a count, never the victim, so an evicted mid keeps its joined parked=True
        # forever - every switch-back then silently pays a cold prefill (the x-2/x-4 rows; sim scenario B).
        # DONE's reused is the engine's only session-attributed cache accounting: a park that held this
        # conversation reused its prefix (every observed park hit reused whole checkpoint blocks), so a
        # reused==0 DONE on a PARKED session whose checkpoint is on disk proves the park did not serve it:
        # unpark. The design floor's one cold step is already paid by that turn; the next switch-back
        # reaches should_restore's file+fidelity gates.
        self.ladder.on_arrival(self.service, "ses-a", P1)
        self.ladder.on_turn_done("ses-a", MID, 0, MID)              # a completed turn with reuse
        self.ladder.on_arrival(self.service, "ses-b", P2)           # switch away: ses-a checkpointed
        self.ladder.on_conv_event(ConvEvent(kind="parked", tokens=MID, byte_count=MID_BYTES, evictions=3))
        self.assertTrue(self.ladder.tracked["ses-a"].parked)        # the joined park claim
        self.ladder.on_arrival(self.service, "ses-a", P1 + " tail")  # switch back while parked:
        self.assertEqual(self.actions("restore"), [])               # park hits beat any restore (pinned)
        self.ladder.on_turn_done("ses-a", MID + 8, 0, 0)            # the engine's own books: this prompt
        # was served with 0 reused - the park demonstrably did not hold it
        self.assertFalse(self.ladder.tracked["ses-a"].parked)       # the stale claim self-heals
        self.ladder.on_arrival(self.service, "ses-b", P2)           # switch away again (dirty: re-save)
        self.ladder.on_arrival(self.service, "ses-a", P1 + " tail more")   # switch back: the gate is open
        self.assertEqual(self.calls[-1], ("restore", self.path("ses-a")))  # and the file is restored

    def test_ladder_yields_to_pending_restore(self):
        a, b = self._two_checkpoints()
        pending = str(Path(self.dir.name) / "llm-session-pending.bin")
        self.service.restore_after_load = pending              # an image job already queued its restore
        self.ladder.on_start(self.service)
        self.assertEqual(self.service.restore_after_load, pending)  # not overwritten, not doubled...
        self.assertEqual(b.file_path, self.path("ses-b"))      # ...so ses-b's checkpoint stays in the
        self.assertEqual(sorted(self.ladder.saved_files), sorted([self.path("ses-a"), self.path("ses-b")]))
        self.assertEqual(a.file_path, self.path("ses-a"))      # ledger untouched
        self.assertFalse(any(s.parked for s in self.ladder.tracked.values()))   # the reset still applies

    def test_disabled_hooks_stay_inert(self):
        ladder = SessionLadder(LadderConfig(enabled=False, slot_save_path=self.dir.name,
                                            park_budget_mib=4096))
        ladder.on_turn_done("ses-a", MID, 0, 0)
        ladder.on_arrival(self.service, "ses-a", P1)
        ladder.on_idle(self.service, 60.0)
        ladder.on_unload(self.service)
        ladder.on_start(self.service)
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.service.restore_after_load)
        self.assertIsNone(ladder.tracked["ses-a"].file_path)


class LifecycleEngine(LadderHarness):
    """Task 8's engine half: the recording LadderHarness plus the unload/reload surface the lifecycle
    hooks wire around (the ReloadableEngine idiom of serve/test_monitor.py), with the transitions
    recorded into the same calls list - so hook-vs-transition order is assertable - and a `save`
    materialising the file the real SAVE would write (the start hook's "valid file" check sees it)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded = True, False

        def session_file(action, path):
            self.calls.append((action, path))
            if action == "save":
                Path(path).write_bytes(b"checkpoint")
            return {"tokens": 1}

        self.session_file = mock.Mock(side_effect=session_file)

    def alive(self):
        return self.running

    def unload(self):
        self.calls.append(("engine-unload", ""))
        self.running, self.unloaded = False, True

    def restart(self):
        self.calls.append(("engine-restart", ""))
        self.running, self.unloaded = True, False


class WiringLifecycle(unittest.TestCase):
    """The real Service regions: on_idle/on_unload inside Service.unload's FIFO hold (immediately before
    the lazy-unload decision and the engine.unload transition), on_start in ensure_loaded after the
    engine reaches READY, the warm restore consumed by the existing restore_after_load line."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.engine = LifecycleEngine(ByteTokenizer(), "Answer.", max_context=32768)
        self.svc = ladder_service(self.engine, self.dir.name)

    def path(self, session_id):
        return str(Path(self.dir.name) / f"llm-session-{session_id}.bin")

    def feed(self, session_id, prompt="the active conversation"):
        self.svc.session_ladder.on_turn_done(session_id, MID, 0, 0)
        self.svc.session_ladder.on_arrival(self.svc, session_id, prompt)

    def test_wiring_idle_and_unload_save_before_the_engine_goes(self):
        self.feed("ses-a")                                                   # active, dirty
        self.svc.session_ladder.on_turn_done("ses-b", MID, 0, 0)            # tracked and dirty, never active
        self.svc.last_request_at = time.time() - 9        # 9 s idle: past the ladder's 8 s idle_save_s
        self.assertEqual(self.svc.unload(idle_for=8), "unloaded")
        self.assertEqual(self.engine.calls,
                         [("save", self.path("ses-a")),     # on_idle, immediately before the lazy-unload
                          # decision: the last active session, dirty and eligible...
                          ("save", self.path("ses-b")),     # ...then on_unload, immediately before the
                          # engine.unload transition: save-all reaches ses-b...
                          ("engine-unload", "")])           # ...and only then the engine itself goes
        self.assertFalse(any(s.parked for s in self.svc.session_ladder.tracked.values()))

    def test_wiring_idle_window_respects_recent_activity(self):
        self.feed("ses-a")
        self.svc.last_request_at = time.time()             # the decision's own measurement: fresh
        self.assertEqual(self.svc.unload(idle_for=8), "busy")
        self.assertEqual(self.engine.calls, [])            # no save, no unload

    def test_wiring_start_warms_through_ensure_loaded(self):
        self.feed("ses-a")
        self.assertEqual(self.svc.unload(), "unloaded")    # manual unload: on_unload checkpoints ses-a
        self.svc.fifo.acquire()                            # ensure_loaded's contract: the caller holds it
        try:
            self.svc.ensure_loaded()                       # restart -> on_start -> the warm restore
        finally:
            self.svc.fifo.release()
        self.assertEqual(self.engine.calls,
                         [("save", self.path("ses-a")), ("engine-unload", ""), ("engine-restart", ""),
                          ("restore", self.path("ses-a"))])   # the warm session_file("restore") is the
        # existing restore_after_load -> images_api.restore_session line of ensure_loaded, in the same hold
        self.assertIsNone(self.svc.restore_after_load)     # consumed, not left dangling...
        self.assertFalse(Path(self.path("ses-a")).exists())  # ...and the consumed file is gone, with the
        self.assertIsNone(self.svc.session_ladder.tracked["ses-a"].file_path)   # ladder's ledger moved over
        self.assertFalse(self.svc.session_ladder.tracked["ses-a"].parked)

    def test_wiring_disabled_service_stays_inert(self):
        engine = LifecycleEngine(ByteTokenizer(), "Answer.", max_context=32768)
        tok = ByteTokenizer()
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.assertFalse(svc.session_ladder.cfg.enabled)   # the Service default: disabled
        svc.session_ladder.on_turn_done("ses-a", MID, 0, 0)
        svc.last_request_at = time.time() - 9              # past every window a hook would check
        self.assertEqual(svc.unload(idle_for=8), "unloaded")
        self.assertEqual(engine.calls, [("engine-unload", "")])   # no ladder save touched the engine
        svc.fifo.acquire()
        try:
            svc.ensure_loaded()
        finally:
            svc.fifo.release()
        self.assertEqual(engine.calls, [("engine-unload", ""), ("engine-restart", "")])
        self.assertIsNone(svc.restore_after_load)          # no warm restore queued either


if __name__ == "__main__":
    unittest.main()
