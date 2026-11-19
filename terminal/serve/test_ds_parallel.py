"""DeepSeek --parallel (Project Maya #332, mw00): several conversations in one engine, the server's side.

A fake engine (a Python script speaking maya-ds4's multi protocol: READY ... multi=N, GEN ... rid=<id>, then
`T <rid> <id>`, `PP <rid> ...`, `DONE <rid> ...`, STOP <rid>) runs behind the real StrataEngine and Service, so the
server's half is tested without a GPU: requests up to N run at once on their own queues, a request that leaves
is STOPped alone, one at a time (no multi=) is the old path, and "parallel": N in the config becomes --parallel N for
DeepSeek only.

    python -m unittest serve.test_ds_parallel -v
"""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, parallel_args, serve

# One token every STEP seconds per active request (a multi step serves every active request at once).  A request's
# reply is the byte of its first prompt id six times, then 257 (<|im_end|> in the ByteTokenizer), or "ok, done." then
# 257 for a longer prompt (a chat's).
FAKE_DS = r'''import os, queue, sys, threading, time
args = sys.argv[1:]
n = int(args[args.index("--parallel") + 1]) if "--parallel" in args else 1
n = min(n, int(os.environ.get("FAKE_CAP", "99")))   # (the engine runs fewer when the cards hold fewer)
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
STEP = 0.03
lines = queue.Queue()
stopped, stop_all = set(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if log: log.write(l[:60] + "\n"); log.flush()
        if l == "STOP":
            stop_all.set()
        elif l.startswith("STOP "):
            stopped.add(int(l.split()[1]))
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.1.50", flush=True)
print("READY 4096 stop" + (f" multi={n}" if n > 1 else ""), flush=True)
active = {}    # rid -> [tokens left, produced, max_new, ids, t0]
def admit(l):
    f = l.split(" ")
    rid = next(int(x[4:]) for x in f if x.startswith("rid="))
    ids = [int(x) for x in f[-1].split(",")]
    max_new = int(f[1])
    rep = list(b"ok, done.") + [257] if len(ids) > 10 else [ids[0]] * 6 + [257]
    print(f"PP {rid} {len(ids)} {len(ids)} 1.0 1000.0 0", flush=True)
    active[rid] = [rep, 0, max_new, ids, time.time()]
def step():
    for rid in sorted(active):
        left, produced, max_new, ids, t0 = active[rid]
        if rid in stopped or stop_all.is_set():
            print(f"DONE {rid} {produced} {len(ids)} 1.0 1.0 cancel 0 0 0 0 0", flush=True)
            stopped.discard(rid)
            del active[rid]
            continue
        t = left.pop(0)
        produced += 1
        print(f"T {rid} {t}", flush=True)
        if t == 257 or produced >= max_new:
            print(f"DONE {rid} {produced} {len(ids)} 1.0 1.0 {'stop' if t == 257 else 'length'} 0 0 0 0 0", flush=True)
            del active[rid]
        else:
            active[rid][1] = produced
    if len(active) > 0 and sum(1 for _ in active) >= 0:
        print(f"STAT tok_s=1.00 ms_tok=1.00 active={len(active)}", flush=True)
while True:
    try:
        l = lines.get(timeout=STEP if active else 0.5)
    except queue.Empty:
        l = ""
    if l is None or l == "QUIT":
        break
    if l.startswith("GEN "):
        admit(l)
    if active:
        time.sleep(STEP)
        step()
'''


class Args(unittest.TestCase):
    def test_parallel_key_is_deepseek_parallel(self):
        ds = {"arch": "deepseek41"}
        self.assertEqual(parallel_args({**ds, "parallel": 4}, []), ["--parallel", "4"])
        self.assertEqual(parallel_args({**ds, "parallel": 1}, []), [])
        self.assertEqual(parallel_args({**ds}, []), [])                                       # (absent: as before)
        self.assertEqual(parallel_args({**ds, "parallel": 3}, ["--parallel", "2"]), [])       # the args' own value wins
        self.assertEqual(parallel_args({**ds, "parallel": 3}, ["--batch", "2"]), [])
        self.assertEqual(parallel_args({**ds, "parallel": "4"}, []), [])                      # said and ignored
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(parallel_args({**ds, "parallel": 12}, []), ["--parallel", "8"])
        self.assertIn("at most 8", buf.getvalue())

    def test_qwen_keeps_batch(self):
        self.assertEqual(parallel_args({"parallel": 4}, []), ["--batch", "4"])
        self.assertEqual(parallel_args({"arch": "qwen38", "parallel": 4}, []), ["--batch", "4"])


class Engine(unittest.TestCase):
    def start(self, n):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_ds.py"
        script.write_text(FAKE_DS, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        extra = ["--parallel", str(n), "--log", str(self.log)]
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", extra)

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def run_one(self, ids, out, cancel=None, stop_after=None):
        cancel = cancel or threading.Event()
        got = []
        g = self.engine.generate(ids, 64, {}, cancel)
        for t in g:
            if t is None:
                continue
            got.append(t)
            if stop_after is not None and len(got) >= stop_after:
                g.close()
                break
        out.append((got, dict(self.engine.last)))

    def test_an_engine_that_runs_fewer_than_asked_is_said(self):
        import contextlib, io
        with mock.patch.dict("os.environ", {"FAKE_CAP": "2"}), contextlib.redirect_stdout(io.StringIO()) as out:
            self.start(4)
        self.assertEqual(self.engine.multi, 2)
        self.assertEqual(self.engine.batch, 2)
        self.assertIn("parallel requests: 4 asked, the engine runs 2", out.getvalue())

    def test_ready_multi_makes_the_engine_parallel(self):
        self.start(3)
        self.assertEqual(self.engine.multi, 3)
        self.assertEqual(self.engine.batch, 3)
        self.engine.unload()
        self.start(1)
        self.assertEqual(self.engine.multi, 0)
        self.assertEqual(self.engine.batch, 0)

    def test_requests_at_once_each_their_own_tokens(self):
        self.start(3)
        outs = [[], [], []]
        ths = [threading.Thread(target=self.run_one, args=([65 + i, 66, 67], outs[i])) for i in range(3)]
        t0 = time.monotonic()
        for t in ths:
            t.start()
        for t in ths:
            t.join(30)
        took = time.monotonic() - t0
        for i in range(3):
            got, last = outs[i][0]
            self.assertEqual(got, [65 + i] * 6 + [257])
            self.assertEqual(last["generated"], 7)         # (each its own DONE: thread-local figures)
            self.assertEqual(last["finish"], "stop")
        self.assertLess(took, 7 * 0.03 * 3 * 0.9 + 1.0)      # (together, not one after the other)
        self.assertEqual(self.engine.rid_q, {})             # every request's queue taken down

    def test_a_request_that_leaves_is_stopped_alone(self):
        self.start(2)
        outs = [[], []]
        a = threading.Thread(target=self.run_one, args=([70, 1, 2], outs[0]), kwargs={"stop_after": 2})
        b = threading.Thread(target=self.run_one, args=([71, 1, 2], outs[1]))
        a.start()
        b.start()
        a.join(30)
        b.join(30)
        self.assertEqual(outs[1][0][0], [71] * 6 + [257])   # (the other one ran to its end)
        self.assertEqual(outs[0][0][0], [70, 70])
        text = self.log.read_text()
        self.assertRegex(text, r"STOP \d+")
        self.assertNotIn("\nSTOP\n", text)                  # (not a bare STOP: that ends every request)

    def test_the_gen_line_carries_rid(self):
        self.start(2)
        out = []
        self.run_one([80, 1], out)
        first = self.log.read_text().splitlines()[0]
        self.assertTrue(first.startswith("GEN 64 rid=1 80,1"), first)

    def test_chat_requests_run_at_once(self):
        self.start(2)
        tok = ByteTokenizer()
        svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

        def chat(text, res, k):
            body = {"messages": [{"role": "user", "content": text}], "max_tokens": 32, "reasoning_effort": "none"}
            req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                res[k] = json.loads(r.read().decode())

        def timed(texts):
            res = {}
            ths = [threading.Thread(target=chat, args=(t, res, i)) for i, t in enumerate(texts)]
            t0 = time.monotonic()
            for t in ths:
                t.start()
            for t in ths:
                t.join(60)
            return time.monotonic() - t0, res

        one, r1 = timed(["u1"])
        two, r2 = timed(["u1", "u2"])
        self.assertEqual(len(r2), 2)
        for r in r2.values():
            self.assertIn("ok, done.", r["choices"][0]["message"]["content"])
        self.assertLess(two, one * 1.7 + 0.3)                  # two at once cost about one


if __name__ == "__main__":
    unittest.main()
