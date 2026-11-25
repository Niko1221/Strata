"""MemT-1.0 serving writes, over fake engines."""
import contextlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import ChatTemplate
from serve import server
from serve.server import ByteTokenizer, Service, StrataEngine, serve

FAKE = r'''import queue, sys, threading, time
STEP = 0.02
lines, stop = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
args = sys.argv[1:]
LOG = open(args[args.index("--log") + 1], "a") if "--log" in args else None
FAIL = "--commit-fails" in args
SILENT = "--commit-silent" in args
CALL = "--call" in args
REPLY = ([ord(c) for c in '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>']
         if CALL else list(b"ok " * 150) if "--long" in args else list(b"ok, done."))
if "--trailing-newline" in args:
    REPLY.append(10)
STOP_ID = 258 if "--end-of-text" in args else 257
REPLY.append(STOP_ID)
print("INFO engine=0.1.39", flush=True)
if "--no-memt" not in args and "--no-identity" not in args:
    print("INFO memt_identity=%s" % ("1234" if "--digit-identity" in args else "0a" * 32), flush=True)
print("READY 4096 stop" + ("" if "--no-memt" in args else " memt"), flush=True)
def note(kind, ids):
    if LOG:
        LOG.write(kind + " " + ",".join(str(t) for t in ids) + "\n")
        LOG.flush()
def mark(kind):
    if LOG:
        LOG.write(kind + "\n")
        LOG.flush()
while True:
    try:
        line = lines.get(timeout=STEP if stop.is_set() else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("GEN "):
        f = line.split()
        max_new = int(f[1])
        ids = [int(x) for x in f[-1].split(",")]
        note("GEN", ids)
        out = 0
        for t in REPLY[:max_new]:
            if stop.is_set():
                break
            print("T %d" % t, flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and REPLY[out - 1] == STOP_ID else ("cancel" if stop.is_set() else "length")
        print("DONE %d %d 5.0 %.1f %s 0 0 0" % (out, len(ids), out * STEP * 1000, fin), flush=True)
        stop.clear()
        continue
    if line.startswith("MEMCOMMIT "):
        ids = [int(x) for x in line.split(" ", 1)[1].split(",")]
        note("COMMIT", ids)
        if SILENT:
            continue
        if FAIL:
            print("ERR MemT commit refused: memory full", flush=True)
            continue
        if "--bad-memdone" in args:
            print("MEMDONE 2 %d 1.0" % len(ids), flush=True)
            continue
        if "--bad-chunk" in args:
            print("MEMCHUNK %d %d" % (len(ids), len(ids)), flush=True)
            print("MEMCHUNK 1 %d" % len(ids), flush=True)
            continue
        if "--commit-chunks" in args:
            for k in range(1, 5):
                time.sleep(0.12)
                print("MEMCHUNK %d %d" % (len(ids) * k // 4, len(ids)), flush=True)
        print("MEMDONE %d %d 1.0" % (len(ids), len(ids)), flush=True)
        continue
    if line.startswith("PING"):
        print("PONG", flush=True)
        continue
    if line.startswith(("MEMSNAP ", "MEMLOAD ", "MEMRESET")):
        f = line.split(" ")
        mark(f[0])
        if "--snap-exists" in args:
            print("ERR snapshot already exists: /tmp/x.safetensors", flush=True)
            continue
        if "--snap-cannot" in args:
            print("ERR cannot write /tmp/x.tmp", flush=True)
            continue
        if "--snap-fails" in args:
            print("ERR MemT snapshot refused: no such memory", flush=True)
            continue
        if f[0] == "MEMSNAP":
            open(f[1], "w").write("snapshot " + f[1])
        print(f[0] + " OK", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
'''

REPLY_IDS = list(b"ok, done.") + [257]


class FakeCase(unittest.TestCase):
    """The real StrataEngine and Service, the fake engine behind them."""

    def setUp(self):
        self.tmps, self.engines = [], []
        self.memt_dir = None

    def snapshots(self) -> str:
        if self.memt_dir is None or not self.memt_dir.is_dir():
            tmp = tempfile.TemporaryDirectory()
            self.tmps.append(tmp)
            self.memt_dir = Path(tmp.name) / "memt"
            self.memt_dir.mkdir()
        return str(self.memt_dir)

    def spawn(self, extra=()):
        tmp = tempfile.TemporaryDirectory()
        self.tmps.append(tmp)
        script = Path(tmp.name) / "fake_memt.py"
        script.write_text(FAKE, encoding="utf-8")
        log = Path(tmp.name) / "memt.log"
        log.write_text("")
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            engine = StrataEngine("strata", ["--log", str(log), *extra])
        self.engines.append(engine)
        return engine, log

    def start(self, extra=(), silence=None):
        self.stop()
        self.engine, self.log = self.spawn(extra)
        if silence is not None:
            self.engine.silence_s = silence
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.memt_save_path = self.snapshots()
        return self.engine, self.svc

    def chat(self, messages, max_tokens=60):
        body = {"messages": messages, "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def serve_http(self):
        self.httpd = serve(self.svc, port=0)
        self.base = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        return self.base

    def stop(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        for engine in self.engines:
            engine.unload()
        self.engines.clear()
        for tmp in self.tmps:
            tmp.cleanup()
        self.tmps.clear()

    def tearDown(self):
        self.stop()

    def rows(self, log=None):
        """(kind, ids) per line of a fake engine's log."""
        log = self.log if log is None else log
        out = []
        for line in log.read_text().splitlines():
            kind, _, rest = line.partition(" ")
            if kind:
                out.append((kind, [int(x) for x in rest.split(",")] if rest.strip() else []))
        return out

    def commits(self, log=None):
        return [ids for kind, ids in self.rows(log) if kind == "COMMIT"]


class ReadyLine(FakeCase):
    def test_a_memt_model_says_so_and_a_plain_one_does_not(self):
        engine, _ = self.start()
        self.assertTrue(engine.memt)
        engine.unload()
        self.start(("--no-memt",))
        self.assertFalse(self.engine.memt)


class CommitLine(FakeCase):
    """`MEMCOMMIT` between requests: one exchange, strict."""

    def test_a_commit_is_one_line_and_an_answer(self):
        engine, _ = self.start()
        answer = engine.memt_commit([1, 2, 3])
        self.assertEqual(answer, {"tokens": 3, "ms": 1.0})
        self.assertEqual(self.commits(), [[1, 2, 3]])
        self.assertTrue(engine.alive())

    def test_a_refusal_is_the_engines_reason(self):
        engine, _ = self.start(("--commit-fails",))
        with self.assertRaises(ValueError) as caught:
            engine.memt_commit([1])
        self.assertIn("memory full", str(caught.exception))
        self.assertTrue(engine.alive())

    def test_a_stray_line_ends_the_engine(self):
        engine, _ = self.start()
        engine.proc.stdin.write("PING\n")
        engine.proc.stdin.flush()
        with self.assertRaises(server.EngineDied):
            engine._memt_commit([1])
        self.assertTrue(engine.ended)

    def test_silence_ends_the_engine(self):
        engine, _ = self.start(("--commit-silent",), silence=0.3)
        with self.assertRaises(server.EngineSilent):
            engine.memt_commit([1])
        self.assertTrue(engine.ended)

    def test_a_long_commit_reports_its_chunks(self):
        engine, _ = self.start(("--commit-chunks",), silence=0.3)
        answer = engine.memt_commit([1, 2, 3])
        self.assertEqual(answer["tokens"], 3)
        self.assertTrue(engine.alive())

    def test_a_malformed_answer_ends_the_engine(self):
        engine, _ = self.start(("--bad-memdone",))
        with self.assertRaises(server.EngineDied):
            engine.memt_commit([1, 2, 3])
        self.assertTrue(engine.ended)

    def test_a_chunk_that_goes_backwards_ends_the_engine(self):
        engine, _ = self.start(("--bad-chunk",))
        with self.assertRaises(server.EngineDied):
            engine.memt_commit([1, 2, 3])
        self.assertTrue(engine.ended)

    def test_no_ids_is_refused_here(self):
        engine, _ = self.start()
        with self.assertRaises(ValueError):
            engine.memt_commit([])
        self.assertEqual(self.rows(), [])


class Turns(FakeCase):
    """The service's rule: one write per completed turn, of the text the memory does not hold yet."""

    def write(self, ids, out=REPLY_IDS):
        self.svc._memt_write_turn(list(ids), list(out))

    def held(self, engine=None):
        return self.svc.memt_history[engine or self.engine].committed_tokens

    def test_the_first_turn_goes_in_whole(self):
        _, svc = self.start()
        self.write([1, 2, 3])
        self.assertEqual(self.commits(), [[1, 2, 3, *REPLY_IDS]])
        self.assertEqual(self.held(), [1, 2, 3, *REPLY_IDS])

    def test_a_continuing_turn_writes_only_its_tail(self):
        _, svc = self.start()
        self.write([1, 2, 3])
        first = [1, 2, 3, *REPLY_IDS]
        self.write(first + [4, 5])
        self.assertEqual(self.commits()[-1], [4, 5, *REPLY_IDS])
        self.assertEqual(self.held(), first + [4, 5, *REPLY_IDS])

    def test_continuation_with_hidden_thinking_writes_only_the_new_turn(self):
        class ThinkingTokenizer(ByteTokenizer):
            SPECIALS = [*ByteTokenizer.SPECIALS, "<think>", "</think>"]

        _, svc = self.start()
        svc.tok = ThinkingTokenizer()

        def encode(text):
            return svc.tok.encode(text, parse_special=True)

        first = "<|im_start|>user\nRemember the name.<|im_end|>\n<|im_start|>assistant\n"
        reply = "Silver Maple<|im_end|>"
        tail = "\n<|im_start|>user\nWhat day?<|im_end|>\n<|im_start|>assistant\n"
        self.write(encode(first + "<think>\nprivate thoughts\n</think>\n\n"), encode(reply))
        self.write(encode(first + "Edited answer<|im_end|>" + tail))
        self.assertEqual(len(self.commits()), 1)
        self.write(encode(first + reply + tail))
        self.assertEqual(self.commits()[-1], encode(tail) + REPLY_IDS)

    def test_a_prompt_that_is_not_a_continuation_is_not_written(self):
        _, svc = self.start()
        self.write([1, 2, 3])
        self.write([9, 9, 9, 9])
        self.assertEqual(len(self.commits()), 1)
        self.assertEqual(self.held(), [1, 2, 3, *REPLY_IDS])

    def test_writes_can_be_off(self):
        _, svc = self.start()
        svc.memt_writes = False
        self.write([1, 2, 3])
        self.assertEqual(self.commits(), [])

    def test_a_plain_engine_has_no_memory_to_write(self):
        _, svc = self.start(("--no-memt",))
        self.write([1, 2, 3])
        self.assertEqual(self.commits(), [])

    def test_a_refused_turn_is_tried_again(self):
        _, svc = self.start(("--commit-fails",))
        self.write([1, 2, 3])
        self.assertEqual(self.held(), [])
        self.assertEqual(len(self.commits()), 1)
        self.write([1, 2, 3])
        self.assertEqual(len(self.commits()), 2)
        self.assertEqual(self.commits()[-1], [1, 2, 3, *REPLY_IDS])

    def test_a_new_engine_process_writes_its_whole_prompt(self):
        _, svc = self.start()
        self.write([1, 2, 3])
        self.engine.gen = (self.engine.gen or 0) + 1
        self.write([1, 2, 3, 4])
        self.assertEqual(self.commits()[-1], [1, 2, 3, 4, *REPLY_IDS])

    def test_a_split_pair_writes_nothing(self):
        from types import SimpleNamespace
        self.start()
        self.svc.engine = SimpleNamespace(prefill=object(), memt=True, batch=0, gen=None,
                                          memt_commit=self.engine.memt_commit)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.write([1, 2, 3])
            self.write([1, 2, 3])
        self.assertEqual(self.commits(), [])
        self.assertEqual(out.getvalue().count("split pair"), 1)

    def test_several_engine_slots_write_nothing(self):
        engine, svc = self.start()
        engine.batch = 2
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.write([1, 2, 3])
            self.write([1, 2, 3])
        self.assertEqual(self.commits(), [])
        self.assertEqual(out.getvalue().count("are not written"), 1)


class Replicas(FakeCase):
    """A pool of replicas: each engine process has a bank of its own, and a turn is written to the one that served it."""

    def pool(self):
        from serve import replicas
        a, la = self.spawn()
        b, lb = self.spawn()
        pool = replicas.ReplicaEngine([a, b])
        self.svc = Service(pool, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        return pool, a, la, b, lb

    def test_each_replica_has_its_own_bank(self):
        pool, a, la, b, lb = self.pool()
        pool.tl.cur = a
        self.svc._memt_write_turn([1, 2, 3], REPLY_IDS)
        self.assertEqual(self.commits(la), [[1, 2, 3, *REPLY_IDS]])
        self.assertEqual(self.commits(lb), [])
        pool.tl.cur = b
        turn2 = [1, 2, 3, *REPLY_IDS, 4]
        self.svc._memt_write_turn(turn2, REPLY_IDS)
        self.assertEqual(self.commits(lb), [[*turn2, *REPLY_IDS]])
        pool.tl.cur = a
        self.svc._memt_write_turn([*turn2, 9], REPLY_IDS)
        self.assertEqual(self.commits(la)[-1], [4, 9, *REPLY_IDS])
        self.assertEqual(len(self.commits(la)), 2)

    def test_a_pool_of_slot_engines_writes_nothing(self):
        pool, a, la, _, lb = self.pool()
        a.batch = 2
        pool.tl.cur = a
        self.svc._memt_write_turn([1, 2, 3], REPLY_IDS)
        self.assertEqual(self.commits(la), [])
        self.assertEqual(self.commits(lb), [])

    def test_a_restarted_replica_writes_its_whole_prompt_again(self):
        pool, a, la, _, _ = self.pool()
        pool.tl.cur = a
        self.svc._memt_write_turn([1, 2, 3], REPLY_IDS)
        a.gen = (a.gen or 0) + 1
        self.svc._memt_write_turn([1, 2, 3, 4], REPLY_IDS)
        self.assertEqual(self.commits(la)[-1], [1, 2, 3, 4, *REPLY_IDS])

    def test_a_chat_through_a_pool_writes_the_replica_that_served_it(self):
        pool, a, la, b, lb = self.pool()
        self.serve_http()
        self.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(len(self.commits(la)) + len(self.commits(lb)), 1)
        log = la if self.commits(la) else lb
        prompt = [ids for kind, ids in self.rows(log) if kind == "GEN"][0]
        self.assertEqual(self.commits(log), [[*prompt, *REPLY_IDS]])

    def test_a_dead_replica_does_not_forget_the_other_banks(self):
        pool, a, la, b, lb = self.pool()
        pool.tl.cur = a
        self.svc._memt_write_turn([1, 2, 3], REPLY_IDS)
        pool.tl.cur = b
        self.svc._memt_write_turn([1, 2, 3], REPLY_IDS)
        a.proc.kill()
        a.proc.wait(timeout=5)
        next_turn = [1, 2, 3, *REPLY_IDS, 4]
        pool.tl.cur = a
        self.svc._memt_write_turn(next_turn, REPLY_IDS)
        self.assertNotIn(a, self.svc.memt_history)
        self.assertIn(b, self.svc.memt_history)
        pool.tl.cur = b
        self.svc._memt_write_turn(next_turn, REPLY_IDS)
        self.assertEqual(self.commits(lb)[-1], [4, *REPLY_IDS])

    def test_a_pool_has_no_single_memory_to_snapshot(self):
        self.pool()
        for action in ("status", "reset", "save"):
            code, body = self.svc.memt_control(action, "x.safetensors")
            self.assertEqual(code, 501)

    def test_a_replica_busy_with_another_request_is_not_written(self):
        pool, a, la, _, _ = self.pool()
        pool.tl.cur = a
        lock = pool.run_locks[0]
        lock.acquire()
        try:
            self.svc._memt_write_turn([1, 2, 3], REPLY_IDS)
        finally:
            lock.release()
        self.assertEqual(self.commits(la), [])
        self.assertEqual(self.svc.memt_history[a].committed_tokens, [])
        self.svc._memt_write_turn([1, 2, 3], REPLY_IDS)
        self.assertEqual(self.commits(la), [[1, 2, 3, *REPLY_IDS]])

    def test_a_slow_commit_does_not_block_another_replica(self):
        pool, a, la, b, lb = self.pool()
        started, release = threading.Event(), threading.Event()
        commit = a.memt_commit

        def slow_commit(ids):
            started.set()
            if not release.wait(10):
                raise TimeoutError("the first commit is still waiting")
            return commit(ids)

        def write(engine, token):
            pool.tl.cur = engine
            self.svc._memt_write_turn([token], REPLY_IDS)

        with mock.patch.object(a, "memt_commit", slow_commit), ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(write, a, 1)
            try:
                self.assertTrue(started.wait(5))
                workers.submit(write, b, 2).result(timeout=5)
                self.assertEqual(self.commits(lb), [[2, *REPLY_IDS]])
                self.assertEqual(self.commits(la), [])
            finally:
                release.set()
            first.result(timeout=5)
        self.assertEqual(self.commits(la), [[1, *REPLY_IDS]])


class MemoryState(FakeCase):
    """`POST /v1/memt`: the memory state itself, and its snapshot file."""

    def setUp(self):
        FakeCase.setUp(self)
        self.start()
        self.svc.memt_save_path = str(self.memt_dir)

    def write(self, ids, out=REPLY_IDS):
        self.svc._memt_write_turn(list(ids), list(out))

    def held(self):
        return self.svc.memt_history[self.engine].committed_tokens

    def control(self, action, file=None, overwrite=False):
        return self.svc.memt_control(action, file, overwrite)

    def marks(self):
        return [kind for kind, _ in self.rows()]

    def test_status_names_the_memory_and_its_identity(self):
        self.write([1, 2, 3])
        code, body = self.control("status")
        self.assertEqual(code, 200)
        self.assertTrue(body["memt"])
        self.assertEqual(body["identity"], "0a" * 32)
        self.assertEqual(body["held_tokens"], 3 + len(REPLY_IDS))
        self.assertEqual(body["dir"], str(self.memt_dir))

    def test_reset_zeroes_the_memory_and_the_bank(self):
        self.write([1, 2, 3])
        code, body = self.control("reset")
        self.assertEqual(code, 200)
        self.assertIn("MEMRESET", self.marks())
        self.assertEqual(self.held(), [])
        self.write([7, 8])
        self.assertEqual(self.commits()[-1], [7, 8, *REPLY_IDS])

    def test_save_writes_the_snapshot_and_the_tokens_it_holds(self):
        self.write([1, 2, 3])
        code, body = self.control("save", "chat.safetensors")
        self.assertEqual(code, 200)
        self.assertEqual(body, {"status": "saved", "file": "chat.safetensors",
                                "held_tokens": 3 + len(REPLY_IDS), "prefix_saved": True})
        self.assertIn("MEMSNAP", self.marks())
        self.assertTrue((self.memt_dir / "chat.safetensors").is_file())
        side = json.loads((self.memt_dir / "chat.safetensors.json").read_text())
        self.assertEqual(side["held"], [1, 2, 3, *REPLY_IDS])
        self.assertEqual(side["identity"], "0a" * 32)

    def test_save_does_not_replace_a_file_without_overwrite(self):
        self.control("save", "chat.safetensors")
        code, body = self.control("save", "chat.safetensors")
        self.assertEqual(code, 409)
        code, body = self.control("save", "chat.safetensors", overwrite=True)
        self.assertEqual(code, 200)
        self.assertEqual(self.marks().count("MEMSNAP"), 2)

    def test_load_brings_back_the_tokens_the_snapshot_holds(self):
        self.write([1, 2, 3])
        self.control("save", "chat.safetensors")
        self.control("reset")
        self.assertEqual(self.held(), [])
        code, body = self.control("load", "chat.safetensors")
        self.assertEqual(code, 200)
        self.assertIn("MEMLOAD", self.marks())
        self.assertEqual(self.held(), [1, 2, 3, *REPLY_IDS])
        self.assertEqual(body["held_tokens"], 3 + len(REPLY_IDS))
        self.write([*self.held(), 4])
        self.assertEqual(self.commits()[-1], [4, *REPLY_IDS])

    def test_a_snapshot_without_its_sidecar_is_read_only(self):
        self.control("save", "chat.safetensors")
        (self.memt_dir / "chat.safetensors.json").unlink()
        self.control("reset")
        code, body = self.control("load", "chat.safetensors")
        self.assertEqual(code, 200)
        self.assertEqual(self.held(), [])
        committed = len(self.commits())
        self.write([1, 2, 3, 4])
        self.assertEqual(len(self.commits()), committed)
        code, body = self.control("status")
        self.assertTrue(body["sealed"])

    def test_a_sealed_memory_stays_sealed_through_save_and_load(self):
        self.control("reset")
        self.control("save", "plain.safetensors")
        (self.memt_dir / "plain.safetensors.json").unlink()
        self.control("load", "plain.safetensors")
        self.control("save", "sealed.safetensors")
        self.assertTrue(json.loads((self.memt_dir / "sealed.safetensors.json").read_text())["sealed"])
        self.control("reset")
        code, body = self.control("load", "sealed.safetensors")
        self.assertTrue(self.control("status")[1]["sealed"])
        committed = len(self.commits())
        self.write([5, 6])
        self.assertEqual(len(self.commits()), committed)
        self.control("reset")
        self.assertFalse(self.control("status")[1]["sealed"])

    def test_a_name_outside_the_directory_is_refused(self):
        for bad in ("../x.safetensors", "/etc/passwd.safetensors", "a/b.safetensors",
                    ".hidden.safetensors", "chat.bin", "C:x.safetensors"):
            code, body = self.control("save", bad)
            self.assertEqual(code, 400, bad)
        self.assertNotIn("MEMSNAP", self.marks())

    def test_a_sidecar_from_another_snapshot_is_not_used(self):
        self.write([1, 2, 3])
        self.control("save", "first.safetensors")
        self.write([4, 5])
        self.control("save", "second.safetensors")
        (self.memt_dir / "second.safetensors.json").write_bytes(
            (self.memt_dir / "first.safetensors.json").read_bytes())
        self.control("reset")
        code, body = self.control("load", "second.safetensors")
        self.assertEqual(code, 200)
        self.assertEqual(self.held(), [])
        self.assertTrue(self.control("status")[1]["sealed"])

    def test_a_sidecar_for_another_identity_is_not_used(self):
        self.write([1, 2, 3])
        self.control("save", "chat.safetensors")
        side = json.loads((self.memt_dir / "chat.safetensors.json").read_text())
        side["identity"] = "f" * 64
        (self.memt_dir / "chat.safetensors.json").write_text(json.dumps(side))
        self.control("reset")
        code, body = self.control("load", "chat.safetensors")
        self.assertEqual(self.held(), [])
        self.assertTrue(self.control("status")[1]["sealed"])

    def test_a_failed_sidecar_write_leaves_no_stale_prefix(self):
        self.control("save", "x.safetensors")
        self.write([4, 5])
        (self.memt_dir / "x.safetensors.json").unlink()
        (self.memt_dir / "x.safetensors.json").mkdir()
        code, body = self.control("save", "x.safetensors", overwrite=True)
        self.assertEqual(code, 200)
        self.assertFalse(body["prefix_saved"])
        code, body = self.control("load", "x.safetensors")
        self.assertEqual(self.held(), [])
        self.assertTrue(self.control("status")[1]["sealed"])

    def test_an_engine_that_says_it_exists_is_a_conflict(self):
        self.start(("--snap-exists",))
        self.svc.memt_save_path = str(self.memt_dir)
        code, body = self.control("save", "new.safetensors")
        self.assertEqual(code, 409)
        self.assertIn("already exists", body["error"]["message"])

    def test_an_engine_that_cannot_write_is_a_server_error(self):
        self.start(("--snap-cannot",))
        self.svc.memt_save_path = str(self.memt_dir)
        code, body = self.control("save", "new.safetensors")
        self.assertEqual(code, 500)

    def test_an_unloaded_engine_is_not_a_model_without_memory(self):
        self.engine.unload()
        code, body = self.control("status")
        self.assertEqual((code, body["loaded"], body["memt"]), (200, False, None))
        code, body = self.control("save", "x.safetensors")
        self.assertEqual(code, 503)

    def test_a_split_pair_has_no_single_memory_to_snapshot(self):
        from types import SimpleNamespace
        self.svc.engine = SimpleNamespace(prefill=object(), decode=self.engine)
        for action in ("status", "reset", "save"):
            code, body = self.svc.memt_control(action, "x.safetensors")
            self.assertEqual(code, 501)

    def test_a_missing_file_is_not_loaded(self):
        code, body = self.control("load", "none.safetensors")
        self.assertEqual(code, 404)

    def test_a_refusing_engine_is_the_reason(self):
        self.start(("--snap-fails",))
        self.svc.memt_save_path = str(self.memt_dir)
        code, body = self.control("save", "chat.safetensors")
        self.assertEqual(code, 400)
        self.assertIn("no such memory", body["error"]["message"])

    def test_a_file_cannot_be_a_symlink(self):
        target = self.memt_dir / "real.safetensors"
        target.write_text("x")
        link = self.memt_dir / "link.safetensors"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("no symlinks here")
        code, body = self.control("load", "link.safetensors")
        self.assertEqual(code, 404)

    def test_a_model_without_an_identity_says_so(self):
        self.start(("--no-identity",))
        self.svc.memt_save_path = str(self.memt_dir)
        code, body = self.control("save", "chat.safetensors")
        self.assertEqual(code, 400)
        self.assertIn("memt.identity", body["error"]["message"])
        code, body = self.control("status")
        self.assertEqual((code, body["identity"]), (200, None))

    def test_an_identity_that_looks_like_a_number_stays_a_string(self):
        self.start(("--digit-identity",))
        self.svc.memt_save_path = str(self.memt_dir)
        self.assertEqual(self.engine.info["memt_identity"], "1234")
        code, body = self.control("save", "num.safetensors")
        self.assertEqual(code, 200)

    def test_a_restarted_engine_has_no_stale_prefix(self):
        self.write([1, 2, 3])
        self.engine.gen = (self.engine.gen or 0) + 1
        code, body = self.control("status")
        self.assertEqual(body["held_tokens"], 0)
        code, body = self.control("save", "fresh.safetensors")
        self.assertEqual(body["held_tokens"], 0)
        self.assertEqual(json.loads((self.memt_dir / "fresh.safetensors.json").read_text())["held"], [])

    def test_a_plain_model_has_nothing_to_snapshot(self):
        self.start(("--no-memt",))
        self.svc.memt_save_path = str(self.memt_dir)
        for action in ("reset", "save"):
            code, body = self.control(action, "chat.safetensors")
            self.assertEqual(code, 501)
        self.assertEqual(self.marks(), [])

    def test_a_batch_engine_is_refused(self):
        self.engine.batch = 2
        code, body = self.control("save", "chat.safetensors")
        self.assertEqual(code, 501)

    def test_snapshots_are_off_without_a_directory(self):
        self.svc.memt_save_path = None
        code, body = self.control("save", "chat.safetensors")
        self.assertEqual(code, 501)


class MemoryRoute(FakeCase):
    """The HTTP route end to end."""

    def setUp(self):
        FakeCase.setUp(self)
        self.start()
        self.svc.memt_save_path = str(self.memt_dir)
        self.serve_http()

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/memt", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())

    def test_the_route_reports_the_memory_and_saves_it(self):
        self.chat([{"role": "user", "content": "hello"}])
        code, status = self.post({"action": "status"})
        self.assertEqual(code, 200)
        self.assertTrue(status["memt"])
        self.assertGreater(status["held_tokens"], 0)
        code, saved = self.post({"action": "save", "file": "turn.safetensors"})
        self.assertEqual(saved["status"], "saved")
        self.assertTrue((self.memt_dir / "turn.safetensors").is_file())

    def test_a_bad_action_is_bad_request(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({"action": "dance"})
        self.assertEqual(caught.exception.code, 400)


class ToolTurns(unittest.TestCase):
    """The gate's rule, without an engine: a call that may still be finished after the FIFO is a tool turn."""

    @staticmethod
    def parser(**kw):
        from types import SimpleNamespace
        return SimpleNamespace(**{"pending": [], "scall": None, "state": "content", **kw})

    def test_a_plain_reply_is_not_a_tool_turn(self):
        from types import SimpleNamespace
        self.assertFalse(Service._memt_tool_turn(SimpleNamespace(calls=False), self.parser()))

    def test_a_call_already_out_is_a_tool_turn(self):
        from types import SimpleNamespace
        self.assertTrue(Service._memt_tool_turn(SimpleNamespace(calls=True), self.parser()))

    def test_a_call_the_parser_may_still_settle_is_a_tool_turn(self):
        from types import SimpleNamespace
        hold = SimpleNamespace(calls=False)
        for kw in ({"pending": [["x", None]]}, {"scall": object()},
                   {"state": "call"}, {"state": "invoke"}, {"state": "rcall"}):
            self.assertTrue(Service._memt_tool_turn(hold, self.parser(**kw)), kw)

    def test_a_json_state_buffer_is_not_a_tool_turn(self):
        from types import SimpleNamespace
        self.assertFalse(Service._memt_tool_turn(SimpleNamespace(calls=False), self.parser(state="json")))


class Serving(FakeCase):
    """The HTTP path: two chat turns, the memory holds the conversation as it grows."""

    def setUp(self):
        FakeCase.setUp(self)
        self.start()
        self.serve_http()

    def prompts(self):
        return [ids for kind, ids in self.rows() if kind == "GEN"]

    def held(self, engine=None):
        return self.svc.memt_history[engine or self.engine].committed_tokens

    def test_two_turns_write_the_conversation_into_the_memory(self):
        first = self.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")
        self.chat([{"role": "user", "content": "hello"},
                   {"role": "assistant", "content": "ok, done."},
                   {"role": "user", "content": "again"}])
        prompts = self.prompts()
        commits = self.commits()
        self.assertEqual(len(prompts), 2)
        self.assertEqual(len(commits), 2)
        self.assertEqual(commits[0], [*prompts[0], *REPLY_IDS])
        self.assertEqual([*commits[0], *commits[1]], [*prompts[1], *REPLY_IDS])
        self.assertEqual(self.held(), [*prompts[1], *REPLY_IDS])

    def test_a_rendered_reply_continues_after_restoring_its_raw_prefix(self):
        for option, raw_reply in (
            ("--trailing-newline", [*REPLY_IDS[:-1], 10, 257]),
            ("--end-of-text", [*REPLY_IDS[:-1], 258]),
        ):
            with self.subTest(option=option):
                self.start((option,))
                self.serve_http()
                user = {"role": "user", "content": "hello"}
                reply = self.chat([user])["choices"][0]["message"]["content"]
                self.assertEqual(self.commits(), [[*self.prompts()[0], *raw_reply]])
                self.assertEqual(self.svc.memt_control("save", "chat.safetensors")[0], 200)
                self.assertEqual(self.svc.memt_control("reset")[0], 200)
                self.assertEqual(self.svc.memt_control("load", "chat.safetensors")[0], 200)

                self.chat([user, {"role": "assistant", "content": reply},
                           {"role": "user", "content": "again"}])
                first_prompt, next_prompt = self.prompts()
                canonical = [*first_prompt, *REPLY_IDS]
                self.assertEqual(next_prompt[:len(canonical)], canonical)
                self.assertEqual(self.commits(), [[*first_prompt, *raw_reply],
                                                  [*next_prompt[len(canonical):], *raw_reply]])

    def test_a_reply_cut_by_max_tokens_is_not_written(self):
        self.chat([{"role": "user", "content": "hello"}], max_tokens=3)
        self.assertEqual(self.commits(), [])

    def test_a_turn_the_client_abandons_is_not_written(self):
        self.start(("--long",))
        self.serve_http()
        body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 300,
                "reasoning_effort": "none", "stream": True}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.assertIsNotNone(r.read(400))
        time.sleep(0.5)
        self.assertEqual(self.commits(), [])
        self.assertEqual(self.svc.history[-1]["finish"], "disconnect")

    def test_a_turn_that_ends_in_a_tool_call_is_not_written(self):
        self.start(("--call",))
        self.serve_http()
        body = {"messages": [{"role": "user", "content": "weather in Paris?"}], "max_tokens": 200,
                "reasoning_effort": "none",
                "tools": [{"type": "function", "function": {"name": "get_weather", "description": "",
                                                             "parameters": {"type": "object", "properties": {
                                                                 "city": {"type": "string"}}}}}]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            answer = json.loads(r.read().decode())
        message = answer["choices"][0]["message"]
        self.assertTrue(message.get("tool_calls"), message)
        self.assertEqual(self.commits(), [])


if __name__ == "__main__":
    unittest.main()
