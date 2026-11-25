"""tools/test_memt_engine.py - the MemT control-line client (no GPU, no model).

    python tools/test_memt_engine.py [-v]

A scripted in-process engine pins the lines the client sends and the answers it
parses; a spawned stub pins READY and the process handshake; the identity helper
is checked against memt_snapshot.snapshot_identity on a synthetic tokenizer.
"""
from __future__ import annotations

import json
import sys
import time
import tempfile
import textwrap
import unittest
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memt_engine as E  # noqa: E402
import memt_snapshot as S  # noqa: E402


class Fake:
    """A line protocol peer: records writes, answers from a script."""

    def __init__(self, answers=()):
        self.sent: list[str] = []
        self.answers = deque(answers)

    def write(self, line: str) -> None:
        self.sent.append(line)

    def readline(self) -> str:
        return self.answers.popleft() if self.answers else ""

    def engine(self) -> E.MemtEngine:
        return E.MemtEngine(self.write, self.readline)


class Transport(unittest.TestCase):
    def test_query(self):
        fake = Fake(["PP 3 3 0.0 0.0\n", "T 10\n", "T 11\n", "DONE 2 3 1.0 2.0 stop 0 0 3 0 0 0 0 0.0 3 0\n"])
        tokens = fake.engine().query([1, 2, 3], 2, temperature=0.7)
        self.assertEqual(tokens, [10, 11])
        self.assertEqual(fake.sent, ["MEMQUERY 2 temperature=0.7 1,2,3\n"])

    def test_query_sampling_key_order(self):
        fake = Fake(["DONE 0 1 0.0 0.0 length 0 0 1 0 0 0 0 0.0 1 0\n"])
        fake.engine().query([7], 4, top_k=5, temperature=0.25)
        self.assertEqual(fake.sent, ["MEMQUERY 4 temperature=0.25 top_k=5 7\n"])

    def test_commit(self):
        fake = Fake(["MEMDONE 6 6 12.5\n"])
        done = fake.engine().commit([4, 5, 6, 7, 8, 9])
        self.assertEqual(fake.sent, ["MEMCOMMIT 4,5,6,7,8,9\n"])
        self.assertEqual(done, {"read": 6, "prompt": 6, "ms": 12.5})

    def test_commit_tolerates_chunk_progress(self):
        fake = Fake(["MEMCHUNK 2 6\n", "MEMCHUNK 5 6\n", "MEMDONE 6 6 12.5\n"])
        done = fake.engine().commit([4, 5, 6, 7, 8, 9])
        self.assertEqual(done, {"read": 6, "prompt": 6, "ms": 12.5})

    def test_select_session(self):
        fake = Fake(["MEMSELECT 2 OK\n", "MEMDONE 2 2 0.4\n"])
        engine = fake.engine()
        engine.select_session(2)
        engine.commit([1, 2])
        self.assertEqual(fake.sent, ["MEMSELECT 2\n", "MEMCOMMIT 1,2\n"])
        with self.assertRaises(E.MemtEngineError):
            engine.select_session(-1)

    def test_snapshot_and_load(self):
        fake = Fake(["MEMSNAP OK\n", "MEMSNAP OK\n", "MEMLOAD OK\n", "MEMRESET OK\n", "MEMFORGET OK\n"])
        engine = fake.engine()
        engine.snapshot("/tmp/a.safetensors", "abc")
        engine.snapshot("/tmp/a.safetensors", "abc", overwrite=True)
        engine.load("/tmp/a.safetensors", "abc")
        engine.reset()
        engine.forget()
        self.assertEqual(fake.sent, ["MEMSNAP /tmp/a.safetensors abc\n",
                                     "MEMSNAP /tmp/a.safetensors abc overwrite\n",
                                     "MEMLOAD /tmp/a.safetensors abc\n", "MEMRESET\n", "MEMFORGET\n"])

    def test_query_keeps_the_old_signature(self):
        fake = Fake(["T 10\n", "DONE 1 3 5.0 2.0 stop 0 0 3 0 0 0 0 0.0 3 0\n"])
        self.assertEqual(fake.engine().query([1, 2, 3], 1), [10])

    def test_query_timed_reports_the_done_timings(self):
        fake = Fake(["PP 3 3 0.0 0.0\n", "T 10\n", "T 11\n",
                     "DONE 2 3 5.5 7.5 stop 0 0 3 0 0 0 0 0.0 3 0\n"])
        timed = fake.engine().query_timed([1, 2, 3], 2)
        self.assertEqual(timed["ids"], [10, 11])
        self.assertEqual(timed["prompt_ms"], 5.5)
        self.assertEqual(timed["decode_ms"], 7.5)
        self.assertEqual(timed["finish"], "stop")
        self.assertEqual(timed["prompt_tokens"], 3)
        self.assertIsNotNone(timed["first_ms"])
        self.assertGreaterEqual(timed["end_ms"], timed["first_ms"])

    def test_query_timed_without_tokens(self):
        fake = Fake(["DONE 0 1 2.0 0.0 length 0 0 1 0 0 0 0 0.0 1 0\n"])
        timed = fake.engine().query_timed([7], 1)
        self.assertEqual(timed["ids"], [])
        self.assertIsNone(timed["first_ms"])

    def test_query_timed_reads_the_cancel_line(self):
        # the cancelled form still carries prompt/decode ms and the finish word
        fake = Fake(["T 10\n", "DONE 1 3 4.0 0.0 cancel 0 0 3 0 0 0 0 0.0 3 0\n"])
        timed = fake.engine().query_timed([1, 2, 3], 5)
        self.assertEqual(timed["finish"], "cancel")
        self.assertEqual(timed["prompt_ms"], 4.0)
        self.assertEqual(timed["decode_ms"], 0.0)

    def test_query_timed_refuses_a_short_done(self):
        fake = Fake(["T 10\n", "DONE 1\n"])
        with self.assertRaises(E.MemtEngineError) as caught:
            fake.engine().query_timed([1], 1)
        self.assertIn("DONE line with timings", str(caught.exception))

    def test_error_raises(self):
        fake = Fake(["ERR the snapshot has 9 memory layers, this model 10\n"])
        with self.assertRaises(E.MemtEngineError) as caught:
            fake.engine().load("/tmp/a.safetensors", "abc")
        self.assertIn("9 memory layers", str(caught.exception))

    def test_stream_end_raises(self):
        with self.assertRaises(E.MemtEngineError):
            Fake([]).engine().commit([1])

    def test_newline_injection_refused(self):
        engine = Fake([]).engine()
        with self.assertRaises(E.MemtEngineError):
            engine.send("MEMRESET\nMEMRESET")
        with self.assertRaises(E.MemtEngineError):
            engine.snapshot("/tmp/a b.safetensors", "abc")

    def test_bad_arguments(self):
        engine = Fake([]).engine()
        with self.assertRaises(E.MemtEngineError):
            engine.query([], 1)
        with self.assertRaises(E.MemtEngineError):
            engine.query([1], 0)
        with self.assertRaises(E.MemtEngineError):
            engine.commit([])


STUB = textwrap.dedent('''
    import sys
    print("READY 32768 stop", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.startswith("MEMCOMMIT"):
            print("MEMDONE 2 2 0.5", flush=True)
        elif line.startswith("MEMRESET"):
            print("MEMRESET OK", flush=True)
        elif line.startswith("MEMQUERY"):
            print("T 42", flush=True)
            print("DONE 1 1 0.0 0.1 length 0 0 1 0 0 0 0 0.0 1 0", flush=True)
        else:
            print("ERR unknown command", flush=True)
''')


class Spawned(unittest.TestCase):
    def test_ready_and_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_engine.py"
            script.write_text(STUB, encoding="utf-8")
            engine = E.MemtEngine.spawn([sys.executable, "-u", str(script)])
            try:
                self.assertEqual(engine.context, 32768)
                engine.reset()
                self.assertEqual(engine.commit([1, 2])["read"], 2)
                self.assertEqual(engine.query([9], 1), [42])
            finally:
                engine.close()
            self.assertIsNotNone(engine.process)
            self.assertIsNotNone(engine.process.returncode)

    def test_spawn_error_before_ready(self):
        script = Path(tempfile.mkdtemp()) / "bad_engine.py"
        script.write_text("print('ERR no memt', flush=True)\n", encoding="utf-8")
        with self.assertRaises(E.MemtEngineError):
            E.MemtEngine.spawn([sys.executable, "-u", str(script)])

    def test_ready_timeout_kills_a_silent_engine(self):
        script = Path(tempfile.mkdtemp()) / "silent.py"
        script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
        started = time.monotonic()
        with self.assertRaises(E.MemtEngineError) as caught:
            E.MemtEngine.spawn([sys.executable, "-u", str(script)], ready_timeout=0.4)
        self.assertLess(time.monotonic() - started, 10.0)
        self.assertIn("READY", str(caught.exception))

    def test_ready_timeout_does_not_break_a_fast_engine(self):
        script = Path(tempfile.mkdtemp()) / "fake_engine.py"
        script.write_text(STUB, encoding="utf-8")
        engine = E.MemtEngine.spawn([sys.executable, "-u", str(script)], ready_timeout=30.0)
        try:
            self.assertEqual(engine.context, 32768)
        finally:
            engine.close()


class Identity(unittest.TestCase):
    def test_synthetic_checkpoint_matches_the_codec(self):
        config = {
            "model_type": "memt_qwen3_5_moe",
            "backbone_configs": {"hidden_size": 16, "num_hidden_layers": 2},
            "memt": {"memory_layers": [1], "head_count": 2},
        }
        template = "{{ messages }}"
        tokenizer = {
            "model": {"vocab": {"a": 0, "b": 1}},
            "added_tokens": [{"id": 2, "content": "<eos>"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "chat_template.jinja").write_text(template, encoding="utf-8")
            (root / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
            settings = {"added_tokens_decoder": {"3": {"content": "<audio>"}}}
            (root / "tokenizer_config.json").write_text(json.dumps(settings), encoding="utf-8")
            got = E.identity_from_model_dir(root)
        want = S.snapshot_identity(config, {
            "chat_template": template,
            "vocab": {"a": 0, "b": 1, "<eos>": 2, "<audio>": 3},
        })
        self.assertEqual(got, want)
        self.assertEqual(len(got), 64)

    def test_missing_files_are_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(E.MemtEngineError):
                E.identity_from_model_dir(tmp)


if __name__ == "__main__":
    unittest.main()
