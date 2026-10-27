"""serve/test_session_ladder_wiring.py - Task 7: the ladder on the request path.

Two halves.  The ladder half drives SessionLadder.on_arrival over the recording-mock engine idiom of
serve/test_session_save_retry.py (a SimpleNamespace service whose engine.session_file is a mock.Mock), so
save-on-switch / restore-before-forward / refuse-degrades-to-prefill are checked without a server.  The
wiring half drives the real Service over live HTTP (the serve(svc, port=0) fixture of serve/test_monitor.py)
with a session_file-recording MockEngine: session headers reach on_arrival through the request path, DONE
figures reach on_turn_done, conv_log events reach on_conv_event through metrics(), and the disabled default
plus the parallel/batch refusal keep the engine untouched.

    python -m pytest serve/test_session_ladder_wiring.py -v
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve import session_ladder
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, SessionRefused, serve
from serve.session_ladder import ConvEvent, LadderConfig, SessionLadder, prefix_fidelity

MID = 20000                     # a mid: above the engine's 8192-token park minimum
MID_BYTES = 800_000_000         # its joined byte figure (the engine's snapshot scale)
P1 = "HEAD one"
P2 = "HEAD two"
REFUSED_MEMORY = SessionRefused("memory", "not enough RAM to save the session")


def parked_event(tokens, byte_count, evictions=3):
    return ConvEvent(kind="parked", tokens=tokens, byte_count=byte_count, evictions=evictions)


class LadderArrival(unittest.TestCase):
    """on_arrival over a recording-mock engine: the ladder's save/restore decisions at the FIFO-held point."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        cfg = LadderConfig(enabled=True, slot_save_path=self.dir.name, park_budget_mib=4096)
        self.calls = []

        def session_file(action, path):
            self.calls.append((action, path))
            return {"tokens": MID}

        self.engine = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None), ended=False,
                                      session_file=mock.Mock(side_effect=session_file))
        self.service = SimpleNamespace(engine=self.engine, slot_save_path=self.dir.name)
        self.ladder = SessionLadder(cfg)

    def seen(self, session_id, tokens=MID, byte_count=MID_BYTES, parked=False):
        """A completed mid turn: tracked state created; parked=False by default - the engine lost (or
        never held) the mid, the state a switch-away save is for; parked=True passes the park event."""
        self.ladder.on_turn_done(session_id, tokens, 0, 0)
        s = self.ladder.tracked[session_id]
        s.byte_count = byte_count
        if parked:
            self.ladder.on_conv_event(parked_event(tokens, byte_count))

    def actions(self, wanted):
        return [a for a, _ in self.calls if a == wanted]

    def test_wiring_switch_saves_outgoing_when_lost(self):
        self.seen("ses-b")                      # parked=False: the engine lost (or never held) the mid
        self.ladder.on_arrival(self.service, "ses-b", P2)   # ses-b's turn as the engine saw it: its last
        # rendered prompt arrives with the request that completed it, and makes ses-b the active session
        self.ladder.on_arrival(self.service, "ses-a", P1)
        self.assertEqual(self.actions("save"), ["save"])
        path = self.calls[0][1]
        self.assertEqual(path, str(Path(self.dir.name) / "llm-session-ses-b.bin"))
        self.assertEqual(self.ladder.tracked["ses-b"].file_path, path)
        self.assertFalse(self.ladder.tracked["ses-b"].parked)      # note_switch_away applied after should_save
        self.assertTrue(self.ladder.tracked["ses-b"].saved_prefix.startswith("HEAD two"))

    def test_wiring_no_save_when_parked(self):
        self.seen("ses-b", parked=True)         # parked=True, no pressure: park holds it, nothing to save
        self.ladder.on_arrival(self.service, "ses-b", P2)   # ses-b active (the state park holds it in)
        self.ladder.on_arrival(self.service, "ses-a", P1)
        self.assertEqual(self.calls, [])

    def test_wiring_restores_on_missed_park(self):
        self.seen("ses-a")
        self.ladder.on_arrival(self.service, "ses-a", P1)   # ses-a's turn arrives rendered: its saved prefix
        self.ladder.on_arrival(self.service, "ses-b", P1)          # switch away: ses-a saved (park lost)
        self.seen("ses-b")                                          # ses-b completes and is then lost too
        self.ladder.on_arrival(self.service, "ses-a", P1 + " tail")  # switch back: park missed ses-a
        self.assertEqual(self.actions("restore"), ["restore"])
        self.assertEqual(self.calls[-1], ("restore", str(Path(self.dir.name) / "llm-session-ses-a.bin")))
        self.assertEqual(self.actions("save"), ["save", "save"])

    def test_wiring_skips_restore_after_surgery(self):
        self.seen("ses-a")
        self.ladder.on_arrival(self.service, "ses-a", P1)   # ses-a's rendered turn makes "HEAD one" its prefix
        self.ladder.on_arrival(self.service, "ses-b", P1)          # ses-a checkpointed with prefix "HEAD one"
        self.seen("ses-b")
        self.ladder.on_arrival(self.service, "ses-a", "SURGERY new head")   # context rewritten: not a prefix
        self.assertEqual(self.actions("restore"), [])
        self.assertEqual(self.actions("save"), ["save", "save"])   # the switch itself proceeds: plain prefill

    def test_wiring_refused_memory_degrades_to_prefill(self):
        self.seen("ses-a")
        self.seen("ses-b")
        self.ladder.on_arrival(self.service, "ses-b", P1)          # ses-b active: the switch-away is its save
        self.engine.session_file = mock.Mock(side_effect=[REFUSED_MEMORY, {"tokens": MID}])
        self.ladder.on_arrival(self.service, "ses-a", P1)          # save refused...
        self.assertEqual(self.engine.session_file.call_count, 1)   # ...no retry of the save...
        self.assertIsNone(self.ladder.tracked["ses-a"].file_path)  # ...and no phantom file state to restore from
        self.assertEqual(self.actions("restore"), [])

    def test_wiring_aux_does_not_trigger_saves(self):
        self.seen("ses-big", parked=True)                    # park holds the big mid: switching to aux saves nothing
        self.ladder.on_arrival(self.service, "ses-big", P1)  # the big mid is the active session
        self.ladder.on_turn_done("aux", 100, 0, 0)                # an aux-sized completed turn
        self.ladder.on_conv_event(parked_event(100, 100 * 41370))  # an aux park line: it matches no tracked
        # fingerprint exactly (the estimate is 100 x 40.4 KiB = 4136960) and joins aux through the tolerant
        # join (F1: a 40-byte gap, identical tokens) - either way the point stands: nothing churns
        self.ladder.on_arrival(self.service, "aux", "tiny question")  # switch to it: nothing churns
        self.assertEqual(self.calls, [])
        self.assertFalse(self.ladder.tracked["aux"].save_eligible)

    def test_wiring_no_session_id_switches_identity_without_touching_the_engine(self):
        # F2 (final review): an enabled-ladder arrival with session_id None IS a switch - the engine's slot
        # now holds a conversation the ladder cannot name.  Saving the previous last_active at the next
        # switch-away would write that FOREIGN resident's engine state into llm-session-<last_active>.bin,
        # destroying the real checkpoint's value.  The ladder notes the switch-away and forgets the identity:
        # no save, no restore, last_active None until a header brings one back.
        self.seen("ses-a")
        self.seen("ses-b", parked=True)
        self.ladder.on_arrival(self.service, "ses-b", P2)              # ses-b active (park holds it)
        self.ladder.on_arrival(self.service, None, P1)                 # headerless request between two sessioned ones
        self.assertEqual(self.calls, [])                                # ...no save, no restore...
        self.assertIsNone(self.ladder.last_active)                      # ...identity forgotten...
        self.assertFalse(self.ladder.tracked["ses-b"].parked)           # ...park state now unknown
        self.seen("ses-a")
        self.ladder.on_arrival(self.service, "ses-a", P1)              # next sessioned arrival: no outgoing known
        self.assertEqual(self.calls, [])                                # nothing saved for the headerless resident

    def test_wiring_refused_restore_keeps_the_file(self):
        self.seen("ses-a")
        self.ladder.on_arrival(self.service, "ses-a", P1)         # ses-a's rendered turn: prefix "HEAD one"
        self.ladder.on_arrival(self.service, "ses-b", P1)         # checkpoint with fidelity-able prefix
        self.seen("ses-b", parked=True, tokens=20500, byte_count=810_000_000)  # park holds ses-b (a distinct
        # fingerprint: an event matching two sessions joins neither - join_event refuses to guess)
        self.engine.session_file = mock.Mock(side_effect=[REFUSED_MEMORY])
        self.ladder.on_arrival(self.service, "ses-a", P1 + " tail")
        self.assertEqual(self.engine.session_file.call_count, 1)  # one refused restore, no retry
        self.assertIsNotNone(self.ladder.tracked["ses-a"].file_path)  # the file stays in the ledger

    def test_wiring_sanitizes_the_session_filename(self):
        self.seen("ses/../evil")
        self.ladder.on_arrival(self.service, "ses/../evil", P1)   # the evil id is the active session...
        self.ladder.on_arrival(self.service, "ses-a", P1)         # ...so its save-away builds the filename
        self.assertEqual(self.calls, [("save", str(Path(self.dir.name) / "llm-session-ses_.._evil.bin"))])
        self.assertEqual(list(Path(self.dir.name).glob("*")), [])  # nothing landed outside the save dir

    def test_wiring_refused_class_is_the_server_class_object(self):
        """Eager registration: the ladder excepts the exact class object serve.server exports (module mode
        proof; the __main__ script mode registers its own copy the same way)."""
        self.assertIs(session_ladder._refused_class(), SessionRefused)

    def test_wiring_registered_stand_in_class_pairs_with_refusals(self):
        class StandIn(ValueError):
            def __init__(self, kind, message, published=False):
                super().__init__(message)
                self.kind = kind
                self.published = published

        session_ladder.register_refused_class(StandIn)
        self.addCleanup(session_ladder.register_refused_class, SessionRefused)
        self.seen("ses-a")
        self.seen("ses-b")
        self.ladder.on_arrival(self.service, "ses-b", P1)          # ses-b active: the switch-away saves it
        self.engine.session_file = mock.Mock(side_effect=StandIn("memory", "nope"))
        self.ladder.on_arrival(self.service, "ses-a", P1)          # refused IN THE REGISTERED CLASS: ...
        self.assertEqual(self.engine.session_file.call_count, 1)   # ...swallowed, no retry (not a missed
        # except escaping as an uncaught ValueError)...
        self.assertIsNone(self.ladder.tracked["ses-b"].file_path)  # ...and no phantom file state

    def test_wiring_last_prompts_is_lru_bounded(self):
        cfg = LadderConfig(enabled=True, slot_save_path=self.dir.name, park_budget_mib=4096,
                           retention_max_files=3)                # a tiny cap to watch the ring turn over
        ladder = SessionLadder(cfg)
        for i in range(6):                                       # six arrivals past a three-slot ring
            ladder.on_arrival(self.service, f"ses-{i}", f"HEAD {i}")
        self.assertEqual(list(ladder.last_prompts), ["ses-3", "ses-4", "ses-5"])  # the oldest prompts are
        # gone: retained text can never outrun the retention_max_files bound however many ids churn


    def test_wiring_switch_save_skips_an_unchanged_checkpoint(self):
        # F3 (final review): the switch-away save honors the dirty marker.  A session whose file the ladder
        # itself saved and that completed no turn since (file_path set, dirty False) has engine state
        # provably identical to that file - re-saving it is a 46-65 s identical rewrite.
        self.seen("ses-a")
        self.ladder.on_arrival(self.service, "ses-a", P1)         # ses-a active
        self.ladder.on_arrival(self.service, "ses-b", P2)         # switch-away: ses-a saved (dirty -> clean)
        self.assertEqual(self.actions("save"), ["save"])
        self.ladder.on_arrival(self.service, "ses-a", P1)         # switch-back: restore, nothing dirties ses-a
        self.assertEqual(self.actions("restore"), ["restore"])
        self.ladder.on_arrival(self.service, "ses-b", P2)         # switch-away again: clean + filed -> skipped
        self.assertEqual(self.actions("save"), ["save"])          # ...no identical re-save...
        self.ladder.on_arrival(self.service, "ses-a", P1)         # back to ses-a (still clean)...
        self.ladder.on_turn_done("ses-a", MID, 0, 0)              # ...a new turn dirties it...
        self.ladder.on_arrival(self.service, "ses-b", P2)         # ...and earns the next switch-away save
        self.assertEqual(self.actions("save"), ["save", "save"])


    def test_wiring_engine_io_runs_outside_the_ladder_lock(self):
        # F4 (final review): the ladder lock is held only for the decision and the ledger windows; the
        # engine.session_file call (engine file I/O that can run for minutes) happens OUTSIDE it, so
        # /metrics's on_conv_event never queues behind a save.  Probe: inside the engine call a
        # non-blocking ladder._lock acquire must succeed.
        probe = []

        def session_file(action, path):
            got = self.ladder._lock.acquire(blocking=False)
            if got:
                self.ladder._lock.release()
            probe.append((action, got))
            self.calls.append((action, path))
            return {"tokens": MID}

        self.engine.session_file = mock.Mock(side_effect=session_file)
        self.seen("ses-a")
        self.seen("ses-b")
        self.ladder.on_arrival(self.service, "ses-a", P1)
        self.ladder.on_arrival(self.service, "ses-b", P2)      # switch-away: save ses-a
        self.assertEqual([a for a, _ in probe], ["save"])      # the probe fired at the engine call...
        self.assertTrue(all(got for _, got in probe))          # ...and the ladder lock was free there

    def test_refused_save_leaves_the_ledger_consistent(self):
        # F4's regression guard after the lock restructure: a refused switch-away save records nothing
        # anywhere - no phantom file on the session or in the LRU ledger, the session stays dirty (so a
        # later save can still land), and no restore is ever issued against the file that was never written.
        self.seen("ses-a")
        self.seen("ses-b")
        self.ladder.on_arrival(self.service, "ses-b", P1)      # ses-b active
        self.engine.session_file = mock.Mock(side_effect=REFUSED_MEMORY)
        self.ladder.on_arrival(self.service, "ses-a", P1)      # switch-away: ses-b's save refused...
        s_b = self.ladder.tracked["ses-b"]
        self.assertIsNone(s_b.file_path)                        # ...no phantom file state...
        self.assertTrue(s_b.dirty)                              # ...still dirty...
        self.assertEqual(self.ladder.saved_files, [])           # ...ledger empty...
        self.assertEqual(self.actions("restore"), [])           # ...no restore issued for ses-b
        self.engine.session_file = mock.Mock(side_effect=lambda a, p: (self.calls.append((a, p)),
                                                                       {"tokens": MID})[1])
        self.ladder.on_unload(self.service)                     # a later save can still land normally
        self.assertEqual(s_b.file_path, str(Path(self.dir.name) / "llm-session-ses-b.bin"))
        self.assertIn(s_b.file_path, self.ladder.saved_files)


class LadderHarness(MockEngine):
    """MockEngine + what the ladder hooks need: a recording session_file, DONE figures, a batch knob."""

    def __init__(self, tokenizer, script, **kw):
        super().__init__(tokenizer, script, **kw)
        self.calls = []
        self.last = None
        self.batch = 0
        self.session_file = mock.Mock(side_effect=lambda action, path: (self.calls.append((action, path)),
                                                                        {"tokens": 1})[1])


def ladder_service(engine, directory):
    tok = ByteTokenizer()
    svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
    svc.slot_save_path = directory
    svc.session_ladder = SessionLadder(LadderConfig(enabled=True, slot_save_path=directory, park_budget_mib=4096))
    return svc


def big_request(content):
    return {"messages": [{"role": "user", "content": content}], "stream": False}


LONG = "long text " * 1500          # ~12000 bytes: over the 8192-token park minimum on the byte tokenizer


class WiringHTTP(unittest.TestCase):
    """The real request path: headers -> on_arrival, DONE -> on_turn_done, metrics() -> on_conv_event."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.engine = LadderHarness(ByteTokenizer(), "Answer.", max_context=32768)
        self.svc = ladder_service(self.engine, self.dir.name)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)

    def post(self, session_id, content):
        headers = {"Content-Type": "application/json"}
        if session_id is not None:
            headers["x-session-id"] = session_id
        req = urllib.request.Request(self.base + "/v1/chat/completions", headers=headers,
                                     data=json.dumps(big_request(content)).encode())
        with urllib.request.urlopen(req, timeout=15) as r:
            self.assertEqual(r.status, 200)
            return json.loads(r.read())

    def actions(self, wanted):
        return [a for a, _ in self.engine.calls if a == wanted]

    def test_wiring_switch_saves_outgoing_when_lost(self):
        self.post("ses-a", LONG + " first")
        self.post("ses-b", LONG + " second")     # switch: ses-a is a tracked eligible mid whose park was never held
        self.assertEqual(self.actions("save"), ["save"])
        self.assertEqual(self.engine.calls[0], ("save", str(Path(self.dir.name) / "llm-session-ses-a.bin")))
        s = self.svc.session_ladder.tracked["ses-a"]
        self.assertTrue(s.save_eligible)         # on_turn_done (DONE figures) anchored it
        self.assertGreaterEqual(s.tokens, 8192)

    def test_wiring_restores_on_missed_park(self):
        self.post("ses-a", LONG + " first")
        self.post("ses-b", LONG + " first")
        self.post("ses-a", LONG + " first")      # back to ses-a's own conversation: prefix intact
        self.assertEqual(self.actions("restore"), ["restore"])
        self.assertEqual(self.engine.calls[-1], ("restore", str(Path(self.dir.name) / "llm-session-ses-a.bin")))

    def test_wiring_restores_a_grown_single_message_session(self):
        # The momo dormancy shape end-to-end (recheck 2026-10-26): a growing single user message re-renders
        # with the saved turn's closing marker now in the middle of the new text, so the strict byte-prefix
        # was never true for such a session and its switch-backs stayed permanently dormant (the x-2..x-5
        # probe rows: file=y, prefix anchored, restore=n at every gate-open arrival).  The engine serves
        # this shape fine - checkpoint blocks align under the growth - so the switch-back must restore.
        self.post("ses-a", LONG + " first")
        self.post("ses-b", LONG + " second")                 # switch away: ses-a checkpointed
        self.post("ses-a", LONG + " first and more text")    # switch back with the message grown
        self.assertEqual(self.actions("restore"), ["restore"])
        self.assertEqual(self.engine.calls[-1], ("restore", str(Path(self.dir.name) / "llm-session-ses-a.bin")))

    def test_wiring_disabled_leaves_the_request_path_inert(self):
        svc = Service(LadderHarness(ByteTokenizer(), "Answer.", max_context=32768), ByteTokenizer(),
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.assertFalse(svc.session_ladder.cfg.enabled)             # the Service default: disabled
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for sid in ("ses-a", "ses-b"):                           # a real switch through a disabled ladder
                req = urllib.request.Request(base + "/v1/chat/completions",
                                             headers={"Content-Type": "application/json", "x-session-id": sid},
                                             data=json.dumps(big_request(LONG)).encode())
                with urllib.request.urlopen(req, timeout=15) as r:
                    self.assertEqual(r.status, 200)
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(svc.engine.calls, [])
        self.assertEqual(svc.session_ladder.tracked, {})             # the ladder never saw a session

    def test_wiring_no_session_id_skips_ladder_over_http(self):
        self.post("ses-a", LONG + " first")                             # sessioned: the ladder knows ses-a
        self.post(None, LONG + " second")                               # F2: a headerless request is an
        # identity-unknown switch - the engine now holds a nameless conversation, and saving ses-a would
        # write THAT conversation into ses-a's file: the ladder saves nothing and forgets the identity.
        self.assertIsNone(self.svc.session_ladder.last_active)
        self.post("ses-b", LONG + " third")                             # the next sessioned request: no known
        # outgoing at its switch point either - a foreign checkpoint is never minted
        self.assertEqual(self.engine.calls, [])                         # no save, no restore
        self.assertTrue(self.svc.session_ladder.tracked["ses-a"])      # tracked state survives...
        self.assertEqual(self.svc.session_ladder.last_active, "ses-b")  # ...and identity resumes with the header

    def test_wiring_batch_refuses_the_ladder(self):
        self.engine.batch = 2                    # parallel mode: the FIFO is not held, save/restore refused
        self.post("ses-a", LONG + " first")
        self.post("ses-b", LONG + " second")     # a real switch would have saved... nothing is called
        self.assertEqual(self.engine.calls, [])

    def test_pre_save_thread_parks_after_a_turn(self):
        # Task 4's timer thread end-to-end: cfg.pre_save_idle_s = 0.1, start_session_pre_save() running,
        # one big completed turn, then quiet - the pass parks the session while the engine stays resident
        # (no unload: the save lands through the ladder's pre-save route, nothing closes the engine).
        self.svc.session_ladder.cfg.pre_save_idle_s = 0.1
        self.svc.start_session_pre_save()
        self.post("ses-a", LONG + " first")
        path = str(Path(self.dir.name) / "llm-session-ses-a.bin")
        deadline = time.monotonic() + 12                    # the thread's 2.0 s cadence; the plan's poll cap
        while time.monotonic() < deadline and ("save", path) not in self.engine.calls:
            time.sleep(0.25)
        self.assertIn(("save", path), self.engine.calls)    # the pre-staged checkpoint landed...
        self.assertEqual(self.actions("save"), ["save"])    # ...exactly one save...
        self.assertEqual(self.actions("restore"), [])       # ...no restore churn...
        self.assertTrue(self.svc.loaded())                  # ...the engine was never unloaded...
        self.assertIn("choices", self.post("ses-a", "ping"))  # ...and still serves the next request

    def test_wiring_conv_events_flow_through_metrics(self):
        log = Path(self.dir.name) / "engine.log"
        log.write_text("conversation cache: parked 1234 tokens in 10.0 ms; parked=1 bytes=50000000 evictions=7\n")
        self.engine.log_path, self.engine.log_start = str(log), 0
        self.svc.session_ladder.on_turn_done("ses-m", 1234, 0, 0)
        before = self.svc.session_ladder.tracked["ses-m"].parked
        view = self.svc.metrics()                # the monitor's poll (GET /metrics does the same)
        self.assertFalse(before)
        self.assertTrue(self.svc.session_ladder.tracked["ses-m"].parked)
        self.assertIn("conversation_cache", view)


class PreSavePass(unittest.TestCase):
    """Task 4: Service.pre_save_pass - one opportunistic non-blocking pass: grab the FIFO (never wait), skip
    a busy queue and an engine not in memory, survive engine death mid-save, and park the dirty current
    session when the queue is free and quiet (cfg.pre_save_idle_s = 0 here: the quiet gate is Task 3's, the
    pass's own gates - FIFO grab, engine state, engine trouble - are what these tests drive)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.engine = LadderHarness(ByteTokenizer(), "Answer.", max_context=32768)
        self.svc = ladder_service(self.engine, self.dir.name)
        self.svc.session_ladder.cfg.pre_save_idle_s = 0

    def seen_turn(self, session_id):
        self.svc.session_ladder.on_arrival(self.svc, session_id, P1)      # the request path's arrival...
        self.svc.session_ladder.on_turn_done(session_id, MID, 0, 0)       # ...and its DONE figures: a
        # completed mid turn, tracked, dirty, last_active - the state a real turn leaves the queue in.

    def test_pre_save_pass_skips_a_busy_queue(self):
        self.seen_turn("ses-a")
        self.svc.fifo.acquire()                            # a live request (or unload) owns the engine
        self.addCleanup(self.svc.fifo.release)
        self.assertFalse(self.svc.pre_save_pass())         # the grab is non-blocking: skip, never wait
        self.assertEqual(self.engine.calls, [])

    def test_pre_save_pass_skips_an_unstarted_engine(self):
        self.seen_turn("ses-a")
        self.engine.unloaded = True                        # the engine flag (server.py:1108/1396): nothing
        self.assertFalse(self.svc.pre_save_pass())         # in memory to pre-stage
        self.assertEqual(self.engine.calls, [])

    def test_pre_save_pass_survives_engine_death(self):
        self.seen_turn("ses-a")
        self.engine.session_file = mock.Mock(side_effect=EngineDied("engine died mid-save"))
        self.assertFalse(self.svc.pre_save_pass())         # EngineDied is caught: no exception escapes...
        self.assertEqual(self.engine.calls, [])            # ...nothing was recorded...
        self.assertTrue(self.svc.fifo.acquire(blocking=False))  # ...and the FIFO came back clean
        self.svc.fifo.release()

    def test_pre_save_pass_parks_the_current_session(self):
        self.seen_turn("ses-a")                            # completed turn, FIFO free, quiet gate open
        self.assertTrue(self.svc.pre_save_pass())
        self.assertEqual(self.engine.calls,
                         [("save", str(Path(self.dir.name) / "llm-session-ses-a.bin"))])


class NearTie(unittest.TestCase):
    """Near-tie joins decide by argmin; prefix_fidelity is the backstop that keeps a wrong join harmless."""

    def test_argmin_winner_and_fidelity_backstop(self):
        cfg = LadderConfig(enabled=True, slot_save_path="/unused", park_budget_mib=4096)
        ladder = SessionLadder(cfg)
        ladder.on_turn_done("x", 20000, 0, 0)                    # estimate ~789 MiB (tokens x 40.4 KiB)
        ladder.on_turn_done("y", 20100, 0, 0)                    # estimate ~793 MiB
        ladder.on_conv_event(ConvEvent(kind="skipped_over_budget", tokens=0, byte_count=0,
                                       evictions=4, snapshot_mib=791))  # the engine prints `estimate >> 20` MiB
        self.assertEqual(ladder.tracked["x"].evictions_at_last_switch, 4)   # the nearer estimate won the join
        self.assertEqual(ladder.tracked["y"].evictions_at_last_switch, 0)   # the join records the evictions
        # baseline; only a `parked` line sets the park flag (a skipped_over_budget line says park refused)
        ladder.tracked["y"].file_path = "/unused/llm-session-y.bin"
        ladder.tracked["y"].saved_prefix = "the other conversation"
        self.assertTrue(ladder.tracked["y"].saved_prefix)        # a checkpoint may exist for the loser too...
        self.assertFalse(prefix_fidelity(ladder.tracked["y"].saved_prefix, "HEAD one tail"))
        self.assertTrue(prefix_fidelity("HEAD one", "HEAD one tail"))


if __name__ == "__main__":
    unittest.main()
