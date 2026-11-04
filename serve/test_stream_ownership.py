"""P0 (round 0213): streamed output must be delivered only to the request that owns it.

#2892 (unbatched replicas cross DONE ownership) and #4451 (--batch: one client's stream reaches another connection).

The fake engine is the one of test_parallel.py with a canary reply: the reply to a prompt is the prompt's own canary
(two capital letters and 4 digits) repeated, so a token that reaches the wrong request is visible as a foreign canary.  Every
test forces a LEGAL schedule with events or a hooked lock (the way serve/test_replica_lifetime.py of PR #2886 does),
so a failure does not depend on winning a scheduler race.
"""
import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.replicas import ReplicaEngine
from serve.server import StrataEngine
from serve.test_parallel import FAKE_BATCH

CANARY_REPEAT = 60
CANARY_RE = re.compile(rb"[A-Z]{2}\d{4}")

# the fake's `continued` (the whole reply, how much of it the prompt ends with) with a canary reply: the last canary
# of the prompt, repeated.  STEP is shortened so the many repetitions stay fast.
CANARY_PATCH = '''
import re as _re
_orig_continued = continued
def continued(ids):
    pb = bytes(int(x) & 255 for x in ids)
    found = _re.findall(rb"[A-Z]{2}\\d{4}", pb)
    if found:
        R = list((found[-1] + b" ") * %d) + [257]
        ints = [int(x) for x in ids]
        return R, max(k for k in range(len(R)) if k == 0 or ints[-k:] == R[:k])
    return _orig_continued(ids)
def reply(ids):
    R, k = continued(ids)
    return R[k:]
''' % CANARY_REPEAT


def canary_script(step=0.004):
    text = FAKE_BATCH.replace("STEP = 0.02", f"STEP = {step}")
    # SLOWPREFILL in a prompt: its read takes a while (a cancel can land inside it).  DEAFSTOP: the engine reads the
    # whole prompt whatever STOP says and then goes on as if it had not come (a drain that does not get its line in time).
    old = "        given = None\n        for q in range(CH, len(ids) - 1, CH):\n            time.sleep(STEP / 2)\n"
    assert old in text
    text = text.replace(old, "        given = None\n        pbm = bytes(int(x) & 255 for x in ids)\n"
                        "        slow, deaf = b\"SLOWPREFILL\" in pbm, b\"DEAFSTOP\" in pbm\n"
                        "        for q in range(CH, len(ids) - 1, CH):\n"
                        "            time.sleep(STEP / 2 + (0.02 if slow or deaf else 0))\n", 1)
    old = "            if stop.is_set():\n                break\n            if active and (q // CH) % 4 == 0:"
    assert old in text
    text = text.replace(old, "            if stop.is_set() and not deaf:\n                break\n"
                        "            if active and (q // CH) % 4 == 0:", 1)
    old = "        if given:\n            if log:\n                log.write(f\"YIELD"
    assert old in text
    text = text.replace(old, "        if deaf:\n            stop.clear()\n" + old, 1)
    text = text.replace("while True:\n    try:\n        line = lines.get(timeout=STEP if active else None)",
                        CANARY_PATCH + "while True:\n    try:\n        line = lines.get(timeout=STEP if active else None)", 1)
    assert CANARY_PATCH in text
    return text


def prompt_ids(canary: str, filler: int = 0) -> list[int]:
    return list(f"say {canary} ".encode()) + [ord("x")] * filler + list(f" {canary} please".encode())


def expected_reply(canary: str) -> list[int]:
    return list((canary.encode() + b" ") * CANARY_REPEAT) + [257]


def text_of(tokens) -> bytes:
    return bytes(t for t in tokens if 0 <= t < 256)


def foreign(tokens, own: str) -> list[bytes]:
    """The canaries in a token stream that are not `own`."""
    return [c for c in CANARY_RE.findall(text_of(tokens)) if c != own.encode()]


class FakeEngines:
    """Real StrataEngine objects (the reader/pump, the generate paths) over the canary fake process."""

    def make_engines(self, n, batch=0, step=0.004):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(canary_script(step), encoding="utf-8")
        real = server.subprocess.Popen
        # kept for the whole test: restart() starts the fake again
        self.popen_patch = mock.patch.object(server.subprocess, "Popen",
                                             lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        self.popen_patch.start()
        engines = []
        for i in range(n):
            extra = (["--batch", str(batch)] if batch else [])
            engines.append(StrataEngine("strata", extra))
        for e in engines:
            e.silence_s = 20
        return engines

    def cleanup_engines(self):
        for e in getattr(self, "engines", []):
            try:
                e.unload()
            except Exception:                                    # noqa: BLE001
                pass
        if getattr(self, "popen_patch", None):
            self.popen_patch.stop()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()


class HookLock:
    """A run lock that calls a hook right after it is released: the successor is then free to take the replica at the
    exact moment the real race needs (before the previous request's child generator has been finalized)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.after_release = None

    def acquire(self, *a, **k):
        return self._lock.acquire(*a, **k)

    def release(self):
        self._lock.release()
        hook, self.after_release = self.after_release, None
        if hook is not None:
            hook()

    def locked(self):
        return self._lock.locked()


def collect(gen, into: list) -> None:
    for t in gen:
        if t is not None:
            into.append(t)


class ReplicaOwnership(FakeEngines, unittest.TestCase):
    """#2892: an unbatched replica takes one request at a time under its run lock.  The lock must cover the previous
    request's whole native exchange (STOP, DONE), and the DONE statistics must belong to the request that read them."""

    def setUp(self):
        self.engines = self.make_engines(2)
        self.rep = ReplicaEngine(self.engines)
        self.rep.heal = lambda: None
        self.engines[1].ended = True          # only replica 0 is alive: every request goes to it
        self.lock = HookLock()
        self.rep.run_locks[0] = self.lock

    def tearDown(self):
        self.cleanup_engines()

    def wait_reading(self, engine, timeout=5.0):
        """True once some thread waits on the engine's line queue (the successor sent its GEN and reads)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if engine.lines.not_empty._waiters:
                return True
            time.sleep(0.001)
        return False

    def run_handoff(self, a_can, b_can, hold_s):
        """A reads two tokens and its consumer closes it; B takes the replica the moment the lock is free and reads
        for `hold_s` seconds before A's child is finalized (so A's tokens are already in the line queue)."""
        a_got, b_got = [], []
        threads = []

        def successor():
            def run():
                collect(self.rep.generate(prompt_ids(b_can), 500, {}, threading.Event()), b_got)
            t = threading.Thread(target=run, daemon=True)
            threads.append(t)
            t.start()
            self.wait_reading(self.engines[0])
            time.sleep(hold_s)
        self.lock.after_release = successor
        gen_a = self.rep.generate(prompt_ids(a_can), 500, {}, threading.Event())
        for t in gen_a:
            if t is not None:
                a_got.append(t)
            if len(a_got) >= 2:
                break
        gen_a.close()                           # what Service does on a stop token, a stop string or a disconnect
        del gen_a
        threads[0].join(30)
        self.assertFalse(threads[0].is_alive())
        return a_got, b_got

    def test_a_successor_never_reads_the_predecessors_done(self):
        """#2892: B is the next request on the replica and A's DONE is still unread."""
        a_can, b_can = "QA0001", "QB0002"
        a_got, b_got = self.run_handoff(a_can, b_can, 0)
        self.assertEqual(foreign(a_got, a_can), [])
        self.assertEqual(foreign(b_got, b_can), [], "B received A's tokens")
        self.assertEqual(b_got, expected_reply(b_can), "B's own reply is complete and exactly its own")

    def test_a_successor_never_reads_the_predecessors_tokens(self):
        """#2892: A's remaining tokens are already queued when B reads."""
        a_can, b_can = "QA0011", "QB0012"
        a_got, b_got = self.run_handoff(a_can, b_can, 0.08)
        self.assertEqual(foreign(b_got, b_can), [], "B received A's tokens")
        self.assertEqual(b_got, expected_reply(b_can), "B's own reply is complete and exactly its own")

    def test_many_forced_handoffs_never_cross(self):
        """The forced schedule above, 30 times with different hold times and different cut-off points."""
        import random
        rnd = random.Random(2892)
        bad = []
        for i in range(30):
            a_can, b_can = f"QA{i:04d}", f"QB{i:04d}"
            self.lock.after_release = None
            cut = rnd.choice((1, 2, 3, 10, 40))
            hold = rnd.choice((0, 0.002, 0.01, 0.05))
            a_got, b_got = [], []
            threads = []

            def successor(b_can=b_can, b_got=b_got, hold=hold, threads=threads):
                t = threading.Thread(target=lambda: collect(
                    self.rep.generate(prompt_ids(b_can), 500, {}, threading.Event()), b_got), daemon=True)
                threads.append(t)
                t.start()
                self.wait_reading(self.engines[0])
                time.sleep(hold)
            self.lock.after_release = successor
            gen_a = self.rep.generate(prompt_ids(a_can), 500, {}, threading.Event())
            for t in gen_a:
                if t is not None:
                    a_got.append(t)
                if len(a_got) >= cut:
                    break
            gen_a.close()
            del gen_a
            threads[0].join(30)
            if foreign(b_got, b_can) or b_got != expected_reply(b_can):
                bad.append((i, cut, hold, len(b_got), foreign(b_got, b_can)[:2]))
        self.assertEqual(bad, [], f"{len(bad)} of 30 handoffs crossed")

    def test_unforced_back_to_back_requests_never_cross(self):
        """No hook: B waits for the replica's lock while A is cut off at a random token; 60 repetitions."""
        import random
        rnd = random.Random(4451)
        self.rep.run_locks[0] = threading.Lock()
        bad = []
        for i in range(60):
            a_can, b_can = f"QA{i:04d}", f"QB{i:04d}"
            cut = rnd.choice((1, 2, 3, 5, 20, 100))
            a_got, b_got = [], []
            go = threading.Event()
            tb = threading.Thread(target=lambda: (go.wait(), collect(
                self.rep.generate(prompt_ids(b_can), 500, {}, threading.Event()), b_got)), daemon=True)
            tb.start()
            gen_a = self.rep.generate(prompt_ids(a_can), 500, {}, threading.Event())
            for t in gen_a:
                if t is not None:
                    a_got.append(t)
                if len(a_got) >= cut:
                    break
            go.set()
            gen_a.close()
            del gen_a
            tb.join(30)
            if foreign(b_got, b_can) or b_got != expected_reply(b_can):
                bad.append((i, cut, len(b_got), foreign(b_got, b_can)[:2]))
        self.assertEqual(bad, [], f"{len(bad)} of 60 crossed")

    def test_the_figures_of_a_request_are_its_own(self):
        """A runs to its max_tokens; before A's caller reads its DONE figures, B runs to completion on the replica."""
        a_can, b_can = "QA0003", "QB0004"
        done = []

        def successor():
            def run():
                collect(self.rep.generate(prompt_ids(b_can, 40), 25, {}, threading.Event()), [])
            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(30)
            done.append(t)
        self.lock.after_release = successor
        got = []
        collect(self.rep.generate(prompt_ids(a_can), 10, {}, threading.Event()), got)
        self.assertTrue(done, "the successor ran")
        last = self.rep.last
        self.assertEqual(last.get("generated"), 10, last)
        self.assertEqual(last.get("prompt_tokens"), len(prompt_ids(a_can)), last)

    def test_an_undrainable_replica_is_not_handed_to_the_next_request(self):
        """The child cannot be drained (the engine never answers STOP): the replica is ended, not reused."""
        from serve.server import EngineSilent
        a_can = "QA0005"
        engine = self.engines[0]
        engine.silence_s = 1.0
        gen_a = self.rep.generate(prompt_ids(a_can), 500, {}, threading.Event())
        got = []
        for t in gen_a:
            if t is not None:
                got.append(t)
            if len(got) >= 2:
                break
        engine.can_stop = True
        # the engine stops answering: its process is frozen
        import os
        import signal
        if hasattr(signal, "SIGSTOP"):
            os.kill(engine.proc.pid, signal.SIGSTOP)
        else:
            self.skipTest("needs SIGSTOP to freeze the fake engine")
        try:
            try:
                gen_a.close()
            except EngineSilent:
                pass
        finally:
            if engine.proc.poll() is None:
                os.kill(engine.proc.pid, signal.SIGCONT)
        self.assertFalse(engine.alive(), "the replica that could not be drained is out of the routing")
        self.assertFalse(self.lock.locked(), "and its run lock is free")
        self.assertEqual(self.rep.route(prompt_ids("QA0006")), -1)


if __name__ == "__main__":
    unittest.main()
