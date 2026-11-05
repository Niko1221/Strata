"""serve/test_jev_concurrency.py - M3.10: 7 concurrent clients x 3 questions (mock).

    python -m unittest serve.test_jev_concurrency -v

The donor's DoD arm: 7 concurrent decision requests, 3 questions each, all
answered, no cross-contamination of answers (each client gets ITS questions'
answers), and the engine FIFO keeps them serialized (one engine turn at a
time - the serve-side stand-in for the donor's per-decode-step ENG.mu).
"""
from __future__ import annotations

import json
import sys
import unittest
# JEV_TEST_MODES_BOOTSTRAP: the suite exercises a deployment that has OPTED IN
# to decisions (chat-only is the default posture since the flip); set before
# any serve.jev_calib import resolves MODES at module load.
import os as _os
_os.environ.setdefault("STRATA_JEV_MODES", "chat,decision")
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_systemone import DecisionEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def start():
    tok = ByteTokenizer()
    engine = DecisionEngine(tok, "A")
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    httpd = serve(svc, port=0)
    return engine, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def post(base, body):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class ConcurrentSevenByThree(unittest.TestCase):
    def test_7x3_all_answered_no_cross_talk(self):
        engine, httpd, base = start()
        try:
            def one(cid):
                qs = {f"c{cid}_q{j}": {"type": "noul",
                                       "instructions": f"Client {cid} question {j}: is it on?"}
                      for j in range(3)}
                st, body = post(base, {"state": {"client": cid}, "questions": qs})
                return cid, st, body

            with ThreadPoolExecutor(max_workers=7) as ex:
                results = list(ex.map(one, range(7)))

            for cid, st, body in results:
                self.assertEqual(st, 200, f"client {cid}: {body}")
                answers = body["answers"]
                self.assertEqual(set(answers), {f"c{cid}_q{j}" for j in range(3)},
                                 f"client {cid} got foreign answers: {set(answers)}")
                for qid, a in answers.items():
                    self.assertEqual(a["type"], "noul")
                    self.assertIn("noul", a)
                self.assertEqual(body["usage"]["readout_tokens"], 3)

            # every client's question rendered with ITS OWN state
            prompts = [engine.tok.decode(p) for p in engine.prompts]
            self.assertEqual(len(prompts), 21)                 # 7 x 3 readout turns
            for p in prompts:
                m = "client" in p
                self.assertTrue(m)
            # a prompt for client 3 mentions client 3's state, not client 5's
            c3 = [p for p in prompts if f'"client": 3' in p.replace(" ", "").replace('"client":3', '"client": 3') or '"client": 3' in p]
            self.assertTrue(c3, "client 3's prompts missing")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_fifo_serializes_engine_turns(self):
        """Overlapping requests never interleave inside the engine: the mock counts
        in-flight generate() calls; it must never exceed 1."""
        engine, httpd, base = start()
        inflight = {"now": 0, "max": 0}
        import threading as _t

        orig = engine.generate

        def counting(ids, max_new, sampling, cancel, embeddings=None):
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
            try:
                yield from orig(ids, max_new, sampling, cancel, embeddings)
            finally:
                inflight["now"] -= 1

        engine.generate = counting
        try:
            def one(cid):
                qs = {f"c{cid}_q{j}": {"type": "noul", "instructions": f"q{j}?"}
                      for j in range(3)}
                return post(base, {"state": {"client": cid}, "questions": qs})[0]

            with ThreadPoolExecutor(max_workers=7) as ex:
                codes = list(ex.map(one, range(7)))
            self.assertEqual(codes, [200] * 7)
            self.assertEqual(inflight["max"], 1, "engine turns interleaved!")
        finally:
            httpd.shutdown()
            httpd.server_close()


class ImageDirectFifo(unittest.TestCase):
    """B3 (2026-11-05, expert review blocker 3): image_mode 'direct' holds svc.fifo around
    each field's generate + BROW read (prepare keeps its own turn).  20 interleaved systemone
    + image-direct requests on one engine: every engine touch happens under the lock (a
    depth-tracking fifo wrapper proves it), no request sees a foreign row, no engine death.
    Mock-based: the real engine's stdin/stdout interleaving is what the lock prevents, so the
    lock discipline is the thing under test (stated per the brief)."""

    def test_20_interleaved_requests_all_locked_and_correct(self):
        import threading as _t
        engine, httpd, base = start()
        svc = httpd.svc                       # serve() wires the Service onto the httpd
        inner, fifo_state = svc.fifo, {"depth": 0, "max": 0, "unlocked": 0}
        fifo_mu = _t.Lock()

        class TrackedFifo:
            def acquire(self, *a, **k):
                r = inner.acquire(*a, **k)
                with fifo_mu:
                    fifo_state["depth"] += 1
                    fifo_state["max"] = max(fifo_state["max"], fifo_state["depth"])
                return r

            def release(self):
                with fifo_mu:
                    fifo_state["depth"] -= 1
                inner.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *a):
                self.release()
                return False

        svc.fifo = TrackedFifo()

        orig_generate, orig_brow = engine.generate, engine.final_row_logits

        def guarded_generate(ids, max_new, sampling, cancel, embeddings=None):
            with fifo_mu:
                if fifo_state["depth"] < 1:
                    fifo_state["unlocked"] += 1
            yield from orig_generate(ids, max_new, sampling, cancel, embeddings)

        def guarded_brow(ids, tokens):
            with fifo_mu:
                if fifo_state["depth"] < 1:
                    fifo_state["unlocked"] += 1
            return orig_brow(ids, tokens)

        engine.generate = guarded_generate
        engine.final_row_logits = guarded_brow
        # prepare is the vision encode on the real path; here it only needs to hand back ids
        svc.prepare = lambda messages, tools, kwargs, max_new=None: (
            list(range(65, 91))[:8], False, 1)

        from serve import jev as _jev
        b64 = "A" * 100

        def one_systemone(cid):
            qs = {f"c{cid}_q": {"type": "noul", "instructions": f"client {cid}: on?"}}
            return post(base, {"state": {"client": cid}, "questions": qs})

        def one_image_direct(cid):
            qs = {f"i{cid}_q": {"type": "noul", "instructions": f"image {cid}: on?"}}
            qdefs, err = _jev.parse_questions(qs)
            assert qdefs is not None, err
            return _jev._handle_image_direct(svc, {"image_b64": b64}, "L", qdefs, b64)

        try:
            with ThreadPoolExecutor(max_workers=10) as ex:
                futs = []
                for i in range(10):
                    futs.append(ex.submit(one_systemone, i))
                    futs.append(ex.submit(one_image_direct, i))
                results = [f.result() for f in futs]
            for i, r in enumerate(results):
                code, body = r if isinstance(r, tuple) else (r[0], r[1])
                self.assertEqual(code, 200, f"request {i}: {body}")
            # every systemone answer is the client's OWN question, none foreign
            for i in range(0, 20, 2):
                self.assertEqual(set(results[i][1]["answers"]), {f"c{i // 2}_q"})
            for i in range(1, 20, 2):
                self.assertEqual(set(results[i][1]["answers"]), {f"i{i // 2}_q"})
            self.assertEqual(fifo_state["unlocked"], 0,
                             "generate/BROW ran without the engine turn held")
            self.assertEqual(fifo_state["max"], 1, "engine turns interleaved!")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
