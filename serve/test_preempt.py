"""serve/test_preempt.py - the prefill-preemption scheduler state machine, against a protocol-level fake engine
(no GPU, no pack).  The fake speaks the engine's stdin/stdout protocol - GEN/SUSPENDED/RESUME/CANCEL with id
suffixes, T/DONE/ERR - so the REAL Service.run_preemptable, EngineRequest demux and FIFO ownership periods are
what's under test (docs/PREFILL-PREEMPT.md, test group A):

  A1  no contention: a request never parks, finishes normally
  A2  a queued request causes the park: A -> SUSPENDED, B runs to DONE, A resumes to DONE
  A3  FIFO order: B completes before C; A resumes after both
  A4  the queued request cancels before the yield: A is never parked for it
  A5  the parked request cancels while B runs: snapshot released, A never restored, B unaffected
  A6  the interim request fails: A is still resumed and completes
  A7  the engine dies while A is parked: A fails cleanly, the server stays healthy
  A8  shutdown with one parked and two queued: everything terminates, no deadlock

    python -m unittest serve.test_preempt -v
"""
from __future__ import annotations

import collections
import json
import queue
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, EngineDied, EngineRequest, Parked, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeEngine:
    """The engine's pipe, in-process.  One request runs at a time; a request whose prompt carries `park_at`
    parks there when another GEN line is already queued (the engine's own rule), emitting SUSPENDED and waiting
    for RESUME id=N.  Every line an id-carrying request produces carries ` id=N`.  The journal records the event
    order, which is what the FIFO tests assert on."""

    max_context = 100000
    can_preempt = True
    can_stop = True

    def __init__(self, scripts: dict[str, list[int]] | None = None, park_grace: float = 0.0):
        self.lines: queue.Queue[str | None] = queue.Queue()    # engine -> the server's pump (StrataEngine's name)
        self.inq: collections.deque[str] = collections.deque() # server -> the engine, in arrival order
        self.mu = threading.Lock()
        self.scripts = scripts or {}
        self.park_grace = park_grace   # how long a request waits AT its park point for a queued request to land
        self.journal: list[str] = []
        self.records: dict[int, dict] = {}                     # rid -> the parked request's remaining tokens
        self.restarted = 0
        self.running: int | None = None
        self.stop_req = False
        self.yield_req = False          # the server's YIELD flag (a park offer), like the engine's
        self.token_s = 0.01             # per-token pacing: the scheduler's heartbeats need wall time to act
        self.closed = False
        self.progress = None
        self.prefill_tok_s_mean = None
        self.info = {"preempt": 1}
        self.last: dict | None = None
        threading.Thread(target=self._run, daemon=True).start()

    # ---- the server's view (EngineRequest / Service call these)
    def alive(self):
        return not self.closed

    def exit_code(self):
        return None

    def restart(self):
        self.restarted += 1
        with self.mu:
            self.records.clear()

    def death_note(self):
        return ""

    def write_line(self, line: str):
        if line == "YIELD":
            self.yield_req = True       # a flag like STOP: the running request acts on it at its park point
            return
        with self.mu:
            self.inq.append(line)

    def sampling_keys(self, sampling):
        return ""                       # the fake is always greedy

    def _parse_done(self, line):
        f = line.split()
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}

    def open_request(self, ids, max_new, sampling, rid: int, embeddings=None) -> EngineRequest:
        self.progress = None
        self.prefill_tok_s_mean = None
        return EngineRequest(self, ids, max_new, sampling, rid, embeddings)

    # ---- the engine side
    def _emit(self, line: str):
        self.journal.append(line.split(" id=")[0])
        self.lines.put(line)

    def _parse(self, line: str) -> tuple[str, list[str], int | None]:
        # OUTBOUND lines end with " id=N"; an inbound GEN carries id=N as a key between max_new and the ids
        f = line.split()
        if len(f) > 1 and f[-1].startswith("id=") and f[-1][3:].isdigit():
            return f[0], f[:-1], int(f[-1][3:])
        return f[0], f, None

    def _run(self):
        while not self.closed:
            with self.mu:
                line = self.inq.popleft() if self.inq else None
            if line is None:
                time.sleep(0.002)
                continue
            cmd, f, rid = self._parse(line)
            if cmd == "QUIT":
                return
            if cmd == "STOP":
                self.stop_req = True
                continue
            if cmd == "GEN":
                max_new = int(f[1])
                ids: list[int] = []
                for tok in f[2:]:
                    if "=" in tok:
                        k, _, v = tok.partition("=")
                        if k == "id" and not rid:
                            rid = int(v)
                    elif tok:
                        ids = [int(x) for x in tok.split(",")]
                try:
                    self._serve(rid, ids, max_new)
                except EngineDied:
                    return
            elif cmd == "RESUME":
                with self.mu:
                    rec = self.records.pop(rid, None) if rid is not None else (
                        next(iter(self.records.values())) if self.records else None)
                if rec is None:
                    self._emit(f"ERR no parked request id={rid}")
                    continue
                self._finish(rec)
            elif cmd == "CANCEL":
                with self.mu:
                    rec = self.records.pop(rid, None)
                if rec is not None:
                    self.last = {"generated": 0, "prompt_tokens": len(rec["ids"]), "prompt_ms": 0.0,
                                 "decode_ms": 0.0, "finish": "cancel"}
                    self._emit(f"DONE 0 {len(rec['ids'])} 0.0 0.0 cancel 0 0 0 0 0 id={rec['rid']}")
                else:
                    self._emit(f"ERR no parked request id={rid}")

    def _serve(self, rid, ids, max_new):
        """The running request, in the protocol's own shape: the prompt is read as PP lines (a park offer acts at
        one of them), then REUSED, then the completion as T lines."""
        script = self.scripts.get("main", list(range(100, 300)))
        if rid is not None and ids and ids[0] == -9:      # a prompt whose first token asks for an engine error
            self._emit(f"ERR the fake engine refuses id={rid}")
            return
        park_at = ids[0] if ids and ids[0] > 0 else None   # the prompt's first token names its park position
        tokens = script[:max_new]
        suf = f" id={rid}" if rid is not None else ""
        self.running = rid
        self._emit(f"RESUME 0{suf}")
        # ---- the prompt read: one PP line per 10 prompt tokens
        pos = 0
        while pos < len(ids):
            pos = min(len(ids), pos + 10)
            if self.stop_req:
                self.stop_req = False
                self.yield_req = False
                self.last = {"generated": 0, "prompt_tokens": pos, "prompt_ms": 1.0, "decode_ms": 0.0,
                             "finish": "cancel"}
                self._emit(f"DONE 0 {len(ids)} 1.0 0.0 cancel 0 0 0 0 0{suf}")
                self.running = None
                return
            if park_at is not None and pos >= park_at and rid is not None and self.yield_req:
                # the boundary: the server offered a yield and the prompt asked for a park near here
                self.yield_req = False
                with self.mu:
                    self.records[rid] = {"rid": rid, "ids": ids, "tokens": tokens, "at": pos}
                self._emit(f"SUSPENDED {pos} {len(ids)}{suf}")
                return                    # the prompt is parked; the queue's next line is another request
            self._emit(f"PP {pos} {len(ids)} 1.0 1.0{suf}")
            time.sleep(self.token_s)
        self._emit(f"REUSED 0{suf}")
        # ---- the completion
        emitted = 0
        while emitted < len(tokens):
            if self.stop_req:
                self.stop_req = False
                self.last = {"generated": emitted, "prompt_tokens": len(ids), "prompt_ms": 1.0,
                             "decode_ms": 1.0, "finish": "cancel"}
                self._emit(f"DONE {emitted} {len(ids)} 1.0 1.0 cancel 0 0 0 0 0{suf}")
                self.running = None
                return
            self._emit(f"T {tokens[emitted]}{suf}")
            emitted += 1
            time.sleep(self.token_s)
        self.yield_req = False           # a park offer dies with the request that carried it
        self.last = {"generated": emitted, "prompt_tokens": len(ids), "prompt_ms": 1.0, "decode_ms": 1.0,
                     "finish": "stop" if len(tokens) < max_new else "length"}
        fin = self.last["finish"]
        self._emit(f"DONE {emitted} {len(ids)} 1.0 1.0 {fin} 0 0 0 0 0{suf}")
        self.running = None


    def _queued_gen(self) -> bool:
        with self.mu:
            return any(l.startswith("GEN") for l in self.inq)

    def _finish(self, rec: dict):
        """The resumed leg: the whole completion (the park happened during the READ phase, so the decode has not
        emitted anything yet - `rec['at']` is the prompt position, not a completion index)."""
        tokens = rec["tokens"]
        suf = f" id={rec['rid']}"
        self.running = rec["rid"]
        self._emit(f"RESUME {rec['at']}{suf}")
        emitted = 0
        while emitted < len(tokens):
            if self.stop_req:
                self.stop_req = False
                self._emit(f"DONE {emitted} {len(rec['ids'])} 1.0 1.0 cancel 0 0 0 0 0{suf}")
                self.running = None
                return
            self._emit(f"T {tokens[emitted]}{suf}")
            emitted += 1
            time.sleep(self.token_s)
        self.last = {"generated": emitted, "prompt_tokens": len(rec["ids"]), "prompt_ms": 1.0,
                     "decode_ms": 1.0, "finish": "length"}
        self._emit(f"DONE {emitted} {len(rec['ids'])} 1.0 1.0 length 0 0 0 0 0{suf}")
        self.running = None



def make_service(preempt=True, park_grace=0.0):
    tok = ByteTokenizer()
    engine = FakeEngine(park_grace=park_grace)
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), preempt=preempt)
    return svc, engine


def drive(svc, ids, cancel=None):
    """svc.run consumed to its ("done", ...) item on the calling thread; -> (events, done)."""
    cancel = cancel or threading.Event()
    evs = []
    for kind, x in svc.run(ids, True, None, 64, {}, cancel):
        evs.append((kind, x))
        if kind == "done":
            return evs, x
    return evs, None


class Base(unittest.TestCase):
    park_grace = 0.0

    def setUp(self):
        self.svc, self.engine = make_service(park_grace=self.park_grace)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    PARK = 120                      # park late enough that the server's YIELD (once B is queued) lands first
    PROMPT_FILL = 199               # the prompt's length: a read phase of 20 PP lines

    def prompt(self, park_at=None, tag=0):
        """A prompt the fake understands: the first token names the park position (0 = never park); True = the
        class default; -1 asks for an engine error; the tag keeps prompts distinct."""
        head = -1 if park_at == "error" else (-9 if park_at == "crash" else (self.PARK if park_at is True else (park_at or 0)))
        return [head] + [(1000 + tag * 7 + i) % 30000 for i in range(self.PROMPT_FILL)]


class TestScheduler(Base):
    park_grace = 5.0

    def test_a1_no_contention_never_parks(self):
        evs, done = drive(self.svc, self.prompt(0, tag=1))
        self.assertEqual(done["finish"], "length")
        self.assertEqual(self.svc.totals["prefill_preemptions"], 0)
        self.assertFalse(self.svc.status.get("parked"))
        self.assertNotIn("SUSPENDED", " ".join(self.engine.journal))

    def test_a2_queued_request_causes_preemption(self):
        out_a, out_b = {}, {}

        def run_a():
            out_a["done"] = drive(self.svc, self.prompt(True, tag=1))[1]

        def run_b():
            time.sleep(0.05)                       # A owns the engine and is reading when B queues
            out_b["done"] = drive(self.svc, self.prompt(0, tag=2))[1]

        ta, tb = threading.Thread(target=run_a), threading.Thread(target=run_b)
        ta.start(), tb.start()
        ta.join(30), tb.join(30)
        self.assertFalse(ta.is_alive() and tb.is_alive())
        self.assertEqual(out_a["done"]["finish"], "length")
        self.assertEqual(out_b["done"]["finish"], "length")
        self.assertEqual(self.svc.totals["prefill_preemptions"], 1)
        self.assertEqual(self.svc.totals["prefill_resumes"], 1)
        j = self.engine.journal
        susp = next(l for l in j if l.startswith("SUSPENDED 120 "))
        self.assertTrue(susp.startswith("SUSPENDED 120 200"), susp)   # the boundary the prompt asked for
        susp_a = j.index(susp)
        resume_a = next(k for k, l in enumerate(j) if k > susp_a and l.startswith("RESUME 120"))
        dones = [k for k, l in enumerate(j) if l.startswith("DONE") and "cancel" not in l]
        self.assertEqual(len(dones), 2)
        self.assertGreater(dones[0], susp_a)      # B's DONE: after A parked
        self.assertLess(dones[0], resume_a)       # and before A resumed
        self.assertGreater(dones[1], resume_a)    # A's DONE is last

    def test_a3_fifo_order(self):
        order = []

        def run(tag, park_at):
            evs, done = drive(self.svc, self.prompt(park_at, tag=tag))
            order.append((tag, done["finish"]))

        threads = [threading.Thread(target=run, args=(1, True)),
                   threading.Thread(target=run, args=(2, 0)),
                   threading.Thread(target=run, args=(3, 0))]
        threads[0].start()
        time.sleep(0.05)
        threads[1].start()
        time.sleep(0.05)
        threads[2].start()
        for t in threads:
            t.join(30)
        for t in threads:
            self.assertFalse(t.is_alive())
        j = self.engine.journal
        b_done = next(k for k, l in enumerate(j) if l.startswith("DONE") and "cancel" not in l)
        c_done = next(k for k, l in enumerate(j) if l.startswith("DONE") and "cancel" not in l and k > b_done)
        self.assertLess(b_done, c_done)            # strict FIFO: B completes before C starts, let alone ends
        dones = [k for k, l in enumerate(j) if l.startswith("DONE") and "cancel" not in l]
        self.assertEqual(len(dones), 3)            # everyone finished

    def test_a4_queued_request_cancels_before_its_turn(self):
        """B queues, A offers its boundary (YIELD), B's client hangs up before B ever owns the engine: the park
        is withdrawn at once (nobody wants the engine) and A runs on without releasing the lock for nobody."""
        out_a, out_b = {}, {}

        def run_a():
            out_a["done"] = drive(self.svc, self.prompt(True, tag=1))[1]

        def run_b(cancel):
            try:
                out_b["done"] = drive(self.svc, self.prompt(0, tag=2), cancel)[1]
            except Exception as e:
                out_b["error"] = e

        ta = threading.Thread(target=run_a)
        cancel_b = threading.Event()
        tb = threading.Thread(target=run_b, args=(cancel_b,))
        ta.start()
        time.sleep(0.05)
        tb.start()
        deadline = time.time() + 10
        while time.time() < deadline and self.svc.totals["prefill_preemptions"] < 1:
            time.sleep(0.01)
        self.assertEqual(self.svc.totals["prefill_preemptions"], 1, "A never offered its boundary")
        cancel_b.set()                              # B's client hangs up while it is still queued
        ta.join(30)
        tb.join(30)
        self.assertFalse(ta.is_alive() or tb.is_alive())
        self.assertEqual(out_a["done"]["finish"], "length")
        self.assertFalse(self.engine.records, "the withdrawn park left a snapshot behind")
        _, healthy = drive(self.svc, self.prompt(0, tag=3))
        self.assertEqual(healthy["finish"], "length")

    def test_a5_parked_request_cancelled_while_b_runs(self):
        out_a, out_b = {}, {}

        def run_a(cancel):
            try:
                out_a["done"] = drive(self.svc, self.prompt(True, tag=1), cancel)[1]
            except Exception as e:                  # a GeneratorExit mid-park unwinds without a done item
                out_a["error"] = e

        def run_b():
            time.sleep(0.05)
            out_b["done"] = drive(self.svc, self.prompt(0, tag=2))[1]

        cancel_a = threading.Event()
        ta = threading.Thread(target=run_a, args=(cancel_a,))
        tb = threading.Thread(target=run_b)
        ta.start(), tb.start()
        deadline = time.time() + 10
        while time.time() < deadline and not (self.engine.records or self.svc.status.get("parked")):
            time.sleep(0.01)
        self.assertTrue(self.engine.records or self.svc.status.get("parked"), "A never parked")
        self.assertTrue(self.engine.records, "the engine holds no snapshot")
        cancel_a.set()                              # A's client goes away while it is parked
        ta.join(30)
        tb.join(30)
        self.assertFalse(ta.is_alive())
        self.assertEqual(out_b["done"]["finish"], "length")   # B unaffected
        self.assertFalse(self.engine.records, "the parked snapshot was not released")
        self.assertEqual(self.svc.totals["prefill_preemptions"], 1)
        # A and B can finish within one bookkeeping step of each other: what matters is that A's own entry
        # (the cancelled one) is in the history at all
        self.assertIn("cancel", [h["finish"] for h in self.svc.history])
        # the engine is healthy for the next request
        _, done = drive(self.svc, self.prompt(0, tag=3))
        self.assertEqual(done["finish"], "length")

    def test_a6_interim_request_fails_active_still_resumes(self):
        out_a, out_b = {}, {}

        def run_a():
            out_a["done"] = drive(self.svc, self.prompt(True, tag=1))[1]

        def run_b():
            time.sleep(0.05)
            try:
                out_b["done"] = drive(self.svc, self.prompt("crash", tag=2))[1]
            except ValueError as e:
                out_b["error"] = str(e)

        ta, tb = threading.Thread(target=run_a), threading.Thread(target=run_b)
        ta.start(), tb.start()
        ta.join(30), tb.join(30)
        self.assertFalse(ta.is_alive())
        self.assertIn("error", out_b)
        self.assertEqual(out_a["done"]["finish"], "length")   # A resumed and finished despite B's failure

    def test_a7_engine_death_while_parked(self):
        out_a, out_b = {}, {}

        def run_a():
            try:
                out_a["done"] = drive(self.svc, self.prompt(True, tag=1))[1]
            except ValueError as e:
                out_a["error"] = str(e)

        def run_b():
            time.sleep(0.05)
            out_b["done"] = drive(self.svc, self.prompt(0, tag=2))[1]

        ta = threading.Thread(target=run_a)
        tb = threading.Thread(target=run_b)
        ta.start(), tb.start()
        deadline = time.time() + 10
        while time.time() < deadline and not self.engine.records:
            time.sleep(0.01)
        self.assertTrue(self.engine.records, "A never parked")
        self.engine.restart()                       # the process died and came back: no records survive
        ta.join(30)
        tb.join(30)
        self.assertFalse(ta.is_alive() or tb.is_alive())
        self.assertIn("error", out_a)               # A failed cleanly, it did not resume into foreign state
        self.assertIn("parked", out_a["error"])
        self.assertEqual(out_b["done"]["finish"], "length")   # B was served by the surviving engine
        _, done = drive(self.svc, self.prompt(0, tag=3))
        self.assertEqual(done["finish"], "length")  # the server is healthy

    def test_a8_shutdown_with_parked_and_queued(self):
        cancels = {i: threading.Event() for i in (1, 2, 3)}
        outs = {}

        def run(tag, park_at):
            try:
                outs[tag] = drive(self.svc, self.prompt(park_at, tag=tag), cancels[tag])[1]
            except Exception as e:
                outs[tag] = e

        threads = [threading.Thread(target=run, args=(1, True)),
                   threading.Thread(target=run, args=(2, 0)),
                   threading.Thread(target=run, args=(3, 0))]
        threads[0].start()
        time.sleep(0.05)
        threads[1].start()
        threads[2].start()
        deadline = time.time() + 10
        while time.time() < deadline and not self.svc.status.get("parked"):
            time.sleep(0.01)
        for e in cancels.values():
            e.set()                                 # everyone's client hangs up at once
        for t in threads:
            t.join(30)
        for t in threads:
            self.assertFalse(t.is_alive(), "a request thread is stuck: shutdown would deadlock")
        self.assertFalse(self.engine.records)
        self.httpd.shutdown()                       # and the server itself stops cleanly


class TestMetrics(Base):
    def test_metrics_expose_preemption_counters(self):
        import urllib.request as rq
        out = {}

        def run_a():
            drive(self.svc, self.prompt(True, tag=1))

        def run_b():
            time.sleep(0.05)
            drive(self.svc, self.prompt(0, tag=2))

        ta, tb = threading.Thread(target=run_a), threading.Thread(target=run_b)
        ta.start(), tb.start()
        ta.join(30), tb.join(30)
        with rq.urlopen(self.base + "/metrics", timeout=10) as r:
            m = json.loads(r.read())
        self.assertEqual(m["totals"]["prefill_preemptions"], 1)
        self.assertEqual(m["totals"]["prefill_resumes"], 1)
        self.assertGreaterEqual(m["totals"]["max_queue_wait_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
