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

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service
from serve.session_ladder import ConvEvent, LadderConfig, SessionLadder, should_restore
from serve.test_session_ladder_wiring import LadderHarness, ladder_service

MID = 20000                     # a mid: above the engine's 8192-token park minimum
MID_BYTES = 800_000_000         # its joined byte figure (the engine's snapshot scale)
P1 = "HEAD one"
P2 = "HEAD two"


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
