"""Data-parallel replicas (serve/replicas.py): the config, the router, and the real Service over HTTP with two fake
engines (the fake of test_parallel.py) behind a ReplicaEngine."""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.replicas import (BLOCK, SESSION_MAP, SESSION_MAP_MAX, ReplicaConfigError, ReplicaEngine, Router,
                            conversation_id, conversation_key, cpu_sets, gpu_groups, replica_cfg,
                            request_conversation)
from serve.server import ByteTokenizer, Service, StrataEngine, serve
from serve.test_parallel import FAKE_BATCH


class Config(unittest.TestCase):
    def test_a_number_cuts_the_gpu_list(self):
        self.assertEqual(gpu_groups({"gpu": [0, 1, 2, 3], "replicas": 2}), [[0, 1], [2, 3]])
        self.assertEqual(gpu_groups({"gpu": "0,1,2,3", "replicas": 4}), [[0], [1], [2], [3]])
        self.assertEqual(gpu_groups({"gpu": [0, 1], "replicas": 1}), [])
        self.assertEqual(gpu_groups({"gpu": [0, 1]}), [])

    def test_named_groups(self):
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}, {"gpus": [2]}]}), [[0, 1], [2]])
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}]}), [])

    def test_mistakes_are_said(self):
        for cfg in ({"gpu": [0, 1, 2], "replicas": 2}, {"replicas": 2}, {"replicas": "2"},
                    {"replicas": [{"gpus": [0, 1]}, {"gpus": [1, 2]}]}, {"replicas": [{"gpus": []}, {"gpus": [1]}]}):
            with self.assertRaises(ReplicaConfigError, msg=cfg):
                gpu_groups(cfg)

    def test_replica_cfg(self):
        cfg = {"gpu": [0, 1, 2, 3], "args": ["--x"], "log": "/l/strata.log", "hip_ordinal": 1, "env": {"A": "1"},
               "expert_profile_save": "/p/prof.bin",
               "replicas": [{"gpus": [0, 1], "env": {"B": 2}}, {"gpus": [2, 3], "args": ["--y"]}]}
        a, b = replica_cfg(cfg, 0, [0, 1]), replica_cfg(cfg, 1, [2, 3])
        self.assertEqual((a["gpu"], b["gpu"]), ([0, 1], [2, 3]))
        self.assertEqual((a["log"], b["log"]), ("/l/strata.r0.log", "/l/strata.r1.log"))
        self.assertEqual(a["env"], {"A": "1", "B": "2"})
        self.assertEqual(b["args"], ["--x", "--y"])
        self.assertNotIn("replicas", a)
        self.assertNotIn("hip_ordinal", a)
        self.assertEqual(a["expert_profile_save"], "/p/prof.r0.bin")
        self.assertEqual(cfg["args"], ["--x"])                       # the original is untouched
        self.assertEqual(replica_cfg({"gpu": [0, 1], "args": []}, 0, [3])["gpu"], 3)

    def test_cpu_sets_off(self):
        self.assertIsNone(cpu_sets(2, enabled=False))
        self.assertIsNone(cpu_sets(1))


class Routing(unittest.TestCase):
    def test_matched_prefix_per_replica(self):
        r = Router(3)
        a = list(range(BLOCK * 4))
        r.note(0, a)
        r.note(1, a[:BLOCK * 2])
        self.assertEqual(r.matched(a + [9] * 70), [BLOCK * 4, BLOCK * 2, 0])
        self.assertEqual(r.matched([7] * 200), [0, 0, 0])
        r.forget(0)
        self.assertEqual(r.matched(a), [0, BLOCK * 2, 0])

    def test_the_index_is_bounded(self):
        r = Router(2, cap=10)
        for i in range(50):
            r.note(i % 2, [i] * (BLOCK * 2))
        self.assertLessEqual(len(r.blocks), 10)


class ConversationId(unittest.TestCase):
    """The standard field only: no header, no user field (that one is the end user, not the conversation)."""

    def test_the_standard_key_names_it(self):
        self.assertEqual(conversation_id({"prompt_cache_key": "cache"}), "cache")
        self.assertEqual(conversation_id({"prompt_cache_key": "  cache  "}), "cache")   # spaces are not the id
        self.assertEqual(conversation_id({"prompt_cache_key": "x" * 500}), "x" * 128)   # bounded

    def test_other_fields_are_not_a_conversation(self):
        self.assertIsNone(conversation_id({"user": "session-1"}))
        self.assertIsNone(conversation_id({}))
        self.assertIsNone(conversation_id({"prompt_cache_key": "   "}))
        self.assertIsNone(conversation_id({"prompt_cache_key": 7}))
        self.assertIsNone(conversation_id(None))


class ConversationKey(unittest.TestCase):
    """No client change: the request's own first user message names the conversation."""

    def test_the_first_user_message_names_it(self):
        grown = [{"role": "user", "content": "build a thing"}, {"role": "assistant", "content": "ok"},
                 {"role": "user", "content": "next"}]
        self.assertEqual(conversation_key([{"role": "user", "content": "build a thing"}]), conversation_key(grown))
        self.assertNotEqual(conversation_key([{"role": "user", "content": "build another thing"}]),
                            conversation_key(grown))

    def test_text_parts_are_read(self):
        self.assertEqual(conversation_key([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
                         conversation_key([{"role": "user", "content": "hi"}]))

    def test_a_system_prompt_is_not_the_conversation(self):
        a = [{"role": "system", "content": "same system prompt"}, {"role": "user", "content": "one"}]
        b = [{"role": "system", "content": "same system prompt"}, {"role": "user", "content": "two"}]
        self.assertNotEqual(conversation_key(a), conversation_key(b))

    def test_nothing_to_name(self):
        self.assertIsNone(conversation_key([]))
        self.assertIsNone(conversation_key([{"role": "assistant", "content": "hi"}]))
        self.assertIsNone(conversation_key([{"role": "user", "content": "   "}]))
        self.assertIsNone(conversation_key([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]))
        self.assertIsNone(conversation_key(None))

    def test_the_standard_key_wins(self):
        msgs = [{"role": "user", "content": "same task"}]
        self.assertEqual(request_conversation({"prompt_cache_key": "named"}, msgs), "named")
        self.assertEqual(request_conversation({}, msgs), conversation_key(msgs))
        self.assertIsNone(request_conversation({}, []))


class FakeEngine:
    def __init__(self, n=4):
        self.batch, self.up = n, True
        self.max_context = 4096
        self.info = {}
        self.last = {}

    def alive(self):
        return self.up

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        yield 7


class RouteChoice(unittest.TestCase):
    def test_least_loaded_then_affinity(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        picks = [e.route([i] * 100) for i in range(4)]             # nothing cached: spread
        self.assertEqual(sorted(picks), [0, 0, 1, 1])
        ids = list(range(BLOCK * 8))
        e.inflight = [0, 0]
        k = e.route(ids)
        e.router.note(k, ids)
        e.inflight = [0, 0]
        self.assertEqual(e.route(ids + [5] * 10), k)                # the conversation goes back
        e.inflight = [0, 0]
        e.inflight[k] = 1                                           # 512 tokens held < 3000 per running request
        self.assertNotEqual(e.route(ids + [5] * 10), k)

    def test_a_long_prefix_outweighs_load(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        ids = list(range(BLOCK * 200))                              # 12,800 tokens held on replica 0
        e.router.note(0, ids)
        e.inflight = [2, 0]
        self.assertEqual(e.route(ids + [1]), 0)
        e.inflight = [5, 0]
        self.assertEqual(e.route(ids + [1]), 1)

    def test_a_new_talk_takes_a_free_card_before_a_warm_one(self):
        """A warm card that keeps taking new conversations leaves the other card idle: load comes first for a new one."""
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        warm = list(range(BLOCK * 200))                             # 12,800 tokens held on replica 0
        e.router.note(0, warm)
        ids = warm[:BLOCK * 100] + [42]                             # 6,400 of them match, so 0 is the cheap card
        e.inflight = [1, 0]
        self.assertEqual(e.route(ids, "first:abc"), 1)               # a named new conversation: the free card
        e.inflight = [1, 0]
        self.assertEqual(e.route(ids), 1)                            # an unnamed new talk: the same

    def test_sessions_split_across_cards_even_while_idle(self):
        """Two sessions arriving between bursts must not both pin to the warm card: placement counts conversations."""
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        warm = list(range(BLOCK * 200))                             # both cards idle, one holds the shared prefix
        e.router.note(0, warm)
        first = e.route(warm[:BLOCK * 50], "first:a")
        second = e.route(warm[:BLOCK * 50], "first:b")              # no request running: nothing says "busy"
        self.assertNotEqual(first, second)
        third = e.route(warm[:BLOCK * 50], "first:c")               # 1 vs 1: either card, both work
        self.assertIn(third, [0, 1])
        self.assertEqual(sorted(e.router.conv_balance), [1, 2])     # spread, not piled on the warm card
        for _ in range(10):                                         # turns never move a bound session
            e.inflight, e.conv_inflight = [0, 0], {}                # the previous turn of each finished
            self.assertEqual(e.route(warm[:BLOCK * 50] + [1], "first:a"), first)
            self.assertEqual(e.route(warm[:BLOCK * 50] + [2], "first:b"), second)

    def test_a_dead_replica_is_skipped(self):
        a, b = FakeEngine(), FakeEngine()
        e = ReplicaEngine([a, b])
        e.heal = lambda: None
        ids = list(range(BLOCK * 3))
        e.router.note(0, ids)
        a.up = False
        self.assertEqual(e.route(ids), 1)
        b.up = False
        self.assertEqual(e.route(ids), -1)

    def test_nonfinite_forgets_the_replica(self):
        """#879: after DONE ... nonfinite the engine holds no conversation: the router forgets that replica."""
        from serve.server import EngineNonFinite

        class NonFinite(FakeEngine):
            def generate(self, ids, max_new, sampling, cancel):
                yield 7
                raise EngineNonFinite("the engine's logits became non-finite (#879)")

        e = ReplicaEngine([NonFinite(), NonFinite()])
        other, ids = [9] * (BLOCK * 4), list(range(BLOCK * 3))
        e.router.note(0, other)
        e.router.note(1, other)
        k = 0
        e.inflight[k] += 1                                          # as route() counts it
        with mock.patch.object(e, "route", return_value=k):
            with self.assertRaises(EngineNonFinite):
                list(e.generate(ids, 10, {}, threading.Event()))
        self.assertEqual(e.router.matched(ids)[k], 0)
        self.assertEqual(e.router.matched(other)[k], 0)             # its older conversations too
        self.assertEqual(e.router.matched(other)[1 - k], len(other))
        self.assertEqual(e.inflight[k], 0)

    def test_batch_is_the_sum(self):
        self.assertEqual(ReplicaEngine([FakeEngine(4), FakeEngine(4)]).batch, 8)
        self.assertEqual(ReplicaEngine([FakeEngine(0), FakeEngine(0)]).batch, 2)

    def test_selected_replica_is_reported_for_solo_requests_too(self):
        engine = ReplicaEngine([FakeEngine(), FakeEngine()])
        engine.inflight[0] = 1
        cancel = threading.Event()
        selected = []
        cancel._strata_replica_selected = selected.append
        self.assertEqual(list(engine.generate([1, 2], 1, {}, cancel)), [7])
        self.assertEqual(selected, [1])


class Conversations(unittest.TestCase):
    """Lever 3: a client that names its conversation gets strict stickiness; a new conversation picks by load."""

    def engine(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()])
        e.heal = lambda: None
        return e

    def test_a_named_conversation_keeps_its_replica(self):
        e = self.engine()
        ids = [1] * 100
        k = e.route(ids, "c1")
        e.inflight, e.conv_inflight = [0, 0], {}              # the first turn finished
        e.inflight[k] = 3                                     # the other card is idle: a warm talk still stays
        self.assertEqual(e.route(ids, "c1"), k)

    def test_a_new_conversation_takes_the_least_busy_card(self):
        e = self.engine()
        e.routed, e.inflight = [5, 0], [2, 0]
        self.assertEqual(e.route([1] * 100, "c2"), 1)
        self.assertEqual(e.router.held("c2"), 1)
        e.inflight, e.conv_inflight = [0, 0], {}              # the first request finished
        self.assertEqual(e.route([1] * 100, "c2"), 1)         # and stays there

    def test_a_replica_that_restarted_forgets_its_conversations(self):
        e = self.engine()
        k = e.route([1] * 100, "c3")
        e.router.forget(k)
        self.assertIsNone(e.router.held("c3"))
        self.assertEqual(e.route([1] * 100, "c3"), 1 - k)     # the card with nothing routed wins the tie
        self.assertEqual(e.router.held("c3"), 1 - k)

    def test_a_dead_replica_hands_its_conversations_back(self):
        a, b = FakeEngine(), FakeEngine()
        e = ReplicaEngine([a, b])
        e.heal = lambda: None
        self.assertEqual(e.route([1] * 100, "c5"), 0)
        a.up = False
        self.assertEqual(e.route([1] * 100, "c5"), 1)         # a conversation does not go to a corpse
        self.assertEqual(e.router.held("c5"), 1)
        self.assertEqual(e.route([1] * 100, "c5"), 1)         # and the binding survives a later busy card

    def test_identical_openings_do_not_pile_on_one_card(self):
        """Content-derived names collide: sessions with byte-equal first messages are separate sessions, not one
        conversation asking twice — five A/B arms with identical prompts ran on one card (2026-10-20)."""
        e = self.engine()
        ids = [1] * 100
        arms = [e.route(ids, "first:same") for _ in range(4)]       # all four arrive while each other is live
        self.assertEqual(arms, [0, 1, 0, 1])                        # strictly alternating, one session per card
        self.assertEqual(e.routed, [2, 2])                          # both cards worked
        e.inflight, e.conv_inflight = [0, 0], {}                    # all four finished
        self.assertIn(e.route(ids, "first:same"), {0, 1})           # a quiet shared name: either card, by load

    def test_the_conversation_map_is_bounded(self):
        r = Router(2, conv_cap=4)
        for i in range(10):
            r.bind(f"c{i}", i % 2)
        self.assertEqual(len(r.convs), 4)
        self.assertIsNone(r.held("c0"))
        self.assertEqual(r.held("c9"), 1)

    def test_unnamed_conversations_keep_the_prefix_rule(self):
        e = self.engine()
        ids = list(range(BLOCK * 8))
        k = e.route(ids)
        e.router.note(k, ids)
        e.inflight = [0, 0]
        self.assertEqual(e.route(ids, None), k)               # the next turn goes back by the hashes

    def test_generate_reads_the_conversation_from_the_cancel(self):
        e = self.engine()
        e.inflight[1] = 2                                     # the load rule would pick replica 0
        cancel = threading.Event()
        cancel._strata_conversation = "c4"
        self.assertEqual(list(e.generate([1] * 100, 1, {}, cancel)), [7])
        self.assertEqual(e.router.held("c4"), 0)
        self.assertEqual(e.inflight, [0, 2])


class SessionEngine(FakeEngine):
    """A fake engine that answers SAVE / RESTORE like the real one, so the fan-out is testable."""

    def __init__(self, n=0, refuse=None):
        super().__init__()
        self.batch = 0                      # one request at a time, like every replica of the deployment
        self.k, self.calls, self.refuse = n, [], refuse

    def session_file(self, action, path):
        self.calls.append((action, path))
        if self.refuse:
            raise self.refuse
        Path(path).write_bytes(b"%d" % self.k)
        return {"tokens": 10 + self.k, "bytes": 2, "ms": 1.0 + self.k}


class SessionFiles(unittest.TestCase):
    """llama-server's single-slot SAVE / RESTORE, once per replica (serve/replicas.py)."""

    def test_a_save_writes_one_file_per_replica_and_the_binding_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "auto.bin")
            engines = [SessionEngine(0), SessionEngine(1)]
            engine = ReplicaEngine(engines)
            engine.router.bind("conv-a", 1)
            engine.router.bind("conv-b", 0)
            out = engine.session_file("save", path)
            self.assertEqual(out, {"tokens": 21, "bytes": 4, "ms": 2.0})
            self.assertEqual(engines[0].calls, [("save", path)])
            self.assertEqual(engines[1].calls, [("save", str(Path(tmp) / "auto.r1.bin"))])
            self.assertTrue((Path(tmp) / "auto.bin").exists())
            self.assertTrue((Path(tmp) / "auto.r1.bin").exists())
            self.assertEqual(json.loads(Path(str(path) + SESSION_MAP).read_text())["bindings"],
                             {"conv-a": 1, "conv-b": 0})

    def test_a_restore_reads_the_bindings_and_skips_a_missing_replica_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "auto.bin")
            (Path(tmp) / "auto.bin").write_bytes(b"0")
            (Path(tmp) / "auto.r1.bin").write_bytes(b"1")
            (Path(str(path) + SESSION_MAP)).write_text(json.dumps({"bindings": {"conv-a": 1, "conv-b": 0}}))
            engines = [SessionEngine(0), SessionEngine(1)]
            engine = ReplicaEngine(engines)
            out = engine.session_file("restore", path)
            self.assertEqual(out["tokens"], 21)
            self.assertEqual([c[0] for c in engines[0].calls], ["restore"])
            self.assertEqual([c[0] for c in engines[1].calls], ["restore"])
            self.assertEqual(engine.router.held("conv-a"), 1)
            self.assertEqual(engine.router.held("conv-b"), 0)

    def test_a_replica_without_a_saved_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "auto.bin")
            (Path(tmp) / "auto.bin").write_bytes(b"0")
            engines = [SessionEngine(0), SessionEngine(1)]
            engine = ReplicaEngine(engines)
            engine.session_file("restore", path)
            self.assertEqual([c[0] for c in engines[0].calls], ["restore"])
            self.assertEqual(engines[1].calls, [])

    def test_one_replica_refusing_does_not_stop_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "auto.bin")
            engines = [SessionEngine(0), SessionEngine(1, refuse=RuntimeError("no complete session to save"))]
            out = ReplicaEngine(engines).session_file("save", path)
            self.assertEqual(out["tokens"], 10)
            self.assertTrue((Path(tmp) / "auto.bin").exists())
            self.assertFalse((Path(tmp) / "auto.r1.bin").exists())

    def test_all_replicas_refusing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "auto.bin")
            engines = [SessionEngine(0, refuse=RuntimeError("a")), SessionEngine(1, refuse=RuntimeError("b"))]
            with self.assertRaises(RuntimeError):
                ReplicaEngine(engines).session_file("save", path)

    def test_the_map_keeps_the_newest_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "auto.bin")
            engine = ReplicaEngine([SessionEngine(0), SessionEngine(1)])
            for i in range(SESSION_MAP_MAX + 5):
                engine.router.bind("conv-%d" % i, i % 2)
            engine.session_file("save", path)
            saved = json.loads((Path(str(path) + SESSION_MAP)).read_text())["bindings"]
            self.assertEqual(len(saved), SESSION_MAP_MAX)
            self.assertNotIn("conv-0", saved)
            self.assertIn("conv-%d" % (SESSION_MAP_MAX + 4), saved)

    def test_parallel_slots_refuse_sessions(self):
        engines = [SessionEngine(0), SessionEngine(1)]
        for engine in engines:
            engine.batch = 4
        replica_engine = ReplicaEngine(engines)
        self.assertFalse(replica_engine.session_serial)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                replica_engine.session_file("save", str(Path(tmp) / "auto.bin"))

    def test_a_save_waits_for_the_request_running_on_that_replica(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "auto.bin")
            engines = [SessionEngine(0), SessionEngine(1)]
            engine = ReplicaEngine(engines)
            done = threading.Event()
            busy = engine.run_locks[1]
            self.assertTrue(busy.acquire(timeout=0))          # a request is running on replica 1
            thread = threading.Thread(target=lambda: (engine.session_file("save", path), done.set()))
            thread.start()
            self.assertFalse(done.wait(0.5))                   # the save waits for it
            busy.release()
            self.assertTrue(done.wait(10))
            thread.join()
            self.assertTrue((Path(tmp) / "auto.r1.bin").exists())


class ReplicaService(unittest.TestCase):
    def start(self, slots=4, n=2):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        real = server.subprocess.Popen
        self.logs = [Path(self.tmp.name) / f"r{i}.log" for i in range(n)]
        engines = []
        for i in range(n):
            extra = (["--batch", str(slots)] if slots else []) + ["--log", str(self.logs[i])]
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engines.append(StrataEngine("strata", extra))
        self.engine = ReplicaEngine(engines)
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def post(self, messages, max_tokens=64, **extra):
        body = {"messages": messages, "max_tokens": max_tokens, "reasoning_effort": "none", **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def chat(self, text, max_tokens=64, **extra):
        return self.post([{"role": "user", "content": text}], max_tokens, **extra)

    def served(self):
        return [sum(1 for l in p.read_text().splitlines() if l.split()[0] in ("GEN", "BGEN")) if p.exists() else 0
                for p in self.logs]

    def burst(self, prompts):
        out = [None] * len(prompts)

        def one(i):
            out[i] = self.chat(prompts[i])
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return out

    def test_requests_spread_over_both_and_answers_are_the_solo_answer(self):
        self.start()
        solo = self.chat("hello there")
        before = self.served()
        out = self.burst([f"request number {i} " + "x" * 300 for i in range(6)])
        got = [b - a for a, b in zip(before, self.served())]
        self.assertTrue(all(g >= 2 for g in got), got)
        self.assertEqual(set(out), {solo})

    def test_metrics_and_status_see_all_slots(self):
        self.start()
        self.assertEqual(self.svc.v1_status()["concurrency"]["serving"], 8)
        m = self.svc.metrics()
        self.assertEqual(len(m["live"]["slots"]), 8)
        self.assertEqual([r["replica"] for r in m["live"]["replicas"]], [0, 1])
        self.assertEqual(m["engine"]["replicas"], 2)

    def test_terminal_telemetry_correlates_second_replica_without_global_sample_claim(self):
        self.start()
        self.svc.replica_gpu_groups = [[0, 1], [2, 3]]
        second = self.engine.engines[1]
        request_id = "abcdef012345"
        second.slot_live[1] = {"request_id": request_id, "state": "reading", "prompt_tokens": 100,
                               "prompt_read": 12, "generated": 0, "started": time.time(), "first_token": None}
        first = self.engine.engines[0]
        first.terminal_batch_sample = {"windows": 2, "processed_rows": 8, "active_rows": 4,
                                       "elapsed_ms": 1000, "observed_monotonic": time.monotonic()}
        second.terminal_batch_sample = {"windows": 2, "processed_rows": 6, "active_rows": 4,
                                        "elapsed_ms": 1000, "observed_monotonic": time.monotonic()}
        first.terminal_stage_sample = {"elapsed_us": 1_000_000, "in_flight_us": [800_000, 600_000],
                                       "observed_monotonic": time.monotonic()}
        second.terminal_stage_sample = {"elapsed_us": 1_000_000, "in_flight_us": [700_000, 500_000],
                                        "observed_monotonic": time.monotonic()}
        self.svc.backend = "cuda"
        for child, cards in ((first, [0, 1]), (second, [2, 3])):
            child.spawn = (*child.spawn[:4], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                              "CUDA_VISIBLE_DEVICES": ",".join(map(str, cards))},
                           *child.spawn[5:])
            child.terminal_stages = [
                {"stage": 0, "layer_start": 0, "layer_end": 19, "engine_ordinal": 0},
                {"stage": 1, "layer_start": 20, "layer_end": 39, "engine_ordinal": 1}]
        with self.svc.status_lock:
            self.svc.terminal_requests[request_id] = {"id": request_id, "state": "waiting",
                                                       "queued_at": time.time(), "admitted_at": time.time(),
                                                       "prompt_tokens": 100}
        with urllib.request.urlopen(self.base + "/v1/telemetry", timeout=10) as response:
            snapshot = json.load(response)
        slot = snapshot["work"]["slots"][5]
        self.assertEqual((slot["slot"], slot["replica"], slot["group"], slot["request_id"]),
                         (5, 1, 1, request_id))
        request = snapshot["work"]["requests"][0]
        self.assertEqual((request["slot"], request["replica"], request["group"], request["state"]),
                         (5, 1, 1, "reading"))
        self.assertEqual(snapshot["work"]["groups"], 2)
        self.assertIsNone(snapshot["work"]["effective_rows_window"])
        self.assertEqual([(row["replica"], row["batch_sample_rows"], row["batch_sample_active_rows"])
                          for row in snapshot["work"]["replica_batches"]], [(0, 8, 4), (1, 6, 4)])
        self.assertEqual([(row["replica"], row["stage_in_flight_us"])
                          for row in snapshot["work"]["replica_stage_samples"]],
                         [(0, [800_000, 600_000]), (1, [700_000, 500_000])])
        self.assertNotIn("stage_in_flight_us", snapshot["work"])
        self.assertEqual(snapshot["topology"]["stages"], [])
        self.assertEqual(snapshot["topology"]["replica_stages"][1]["stages"][1]["device_index"], 3)
        self.assertEqual(snapshot["topology"]["replica_devices"],
                         [{"replica": 0, "devices": [0, 1]}, {"replica": 1, "devices": [2, 3]}])
        second.spawn = (*second.spawn[:4], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                            "CUDA_VISIBLE_DEVICES": "9,3"}, *second.spawn[5:])
        with urllib.request.urlopen(self.base + "/v1/telemetry", timeout=10) as response:
            mismatch = json.load(response)
        self.assertIsNone(mismatch["topology"]["replica_stages"][1]["stages"][0]["device_index"])
        self.assertEqual(mismatch["topology"]["replica_stages"][1]["stages"][1]["device_index"], 3)

    def test_engines_without_slots_run_one_each_at_once(self):
        self.start(slots=0)
        out = self.burst(["a" * 40, "b" * 40])
        self.assertEqual(self.served(), [1, 1])
        self.assertEqual(len(set(out)), 1)

    def test_a_conversation_goes_back_to_its_replica(self):
        self.start()
        first = "begin " + "lorem ipsum " * 40
        reply = self.chat(first)
        k = self.served().index(1)
        self.post([{"role": "user", "content": first}, {"role": "assistant", "content": reply},
                   {"role": "user", "content": "and then?"}], 16)
        self.assertEqual(self.served()[k], 2)

    def test_a_named_conversation_keeps_its_replica_across_turns(self):
        self.start()
        first = "begin " + "lorem ipsum " * 40
        reply = self.chat(first, user="named-conversation")
        k = self.served().index(1)
        for i in range(3):
            self.post([{"role": "user", "content": first}, {"role": "assistant", "content": reply},
                       {"role": "user", "content": f"and then? {i}"}], 16, user="named-conversation")
            self.assertEqual(self.served()[k], 2 + i)

    def test_a_replica_that_died_is_skipped(self):
        self.start()
        self.engine.heal = lambda: None
        self.engine.engines[0].proc.kill()
        self.engine.engines[0].proc.wait()
        self.assertTrue(self.engine.alive())
        for i in range(3):
            self.assertIn("ok, done.", self.chat(f"after the loss {i}"))
        self.assertEqual(self.served()[1], 3)


if __name__ == "__main__":
    unittest.main()
