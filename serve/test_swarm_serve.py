"""serve/test_swarm_serve.py - swarm serving (opt-in): the admission order and the give-way rule by UNCACHED prompt length
(STRATA_ADMIT_UNCACHED) and the cache-aware choice of the slot a new conversation overwrites (STRATA_SLOT_AFFINITY).
No GPU: the unit tests use the slot tables alone, the service tests the fake engine of serve/test_parallel.py.

    python -m unittest serve.test_swarm_serve -v
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from array import array
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import StrataEngine  # noqa: E402
from serve import test_admit_order as tao  # noqa: E402
from serve import test_parallel as tp  # noqa: E402

T = StrataEngine.TURN_ID


def engine(n=4, groups=1, pipelined=False, args=()):
    e = StrataEngine("missing-executable", list(args), lazy=True)
    gs = n // groups
    e.batch = n
    e.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
    e.slot_busy = [False] * n
    e.slot_held = [[] for _ in range(n)]
    e.slot_used = [0.0] * n
    e.slot_live = [None] * n
    e.slot_group = [b // gs for b in range(n)]
    e.slot_groups = groups
    e.slot_gs = gs if pipelined and groups > 1 else 0
    e.slot_cv = threading.Condition()
    e._mirror, (e._mirror_cap, e._mirror_min) = [], StrataEngine._mirror_limits(["--batch", str(n)] + list(args))
    e.swarm_stats = {"estimates": 0, "est_hit_tokens": 0, "reused_tokens": 0, "overestimates": 0,
                     "mirror_dropped": 0}
    return e


def conv(seed, n):
    """A conversation's stream (n tokens): a system prompt shared by every agent, its own history, the assistant
    header (<|im_start|>: the engine's turn checkpoint) and the 50 tokens of the reply it decoded."""
    system = list(range(1000, 1300))
    own = [seed * 100_000 + i for i in range(n - len(system) - 52)]
    return system + [T] + own + [T] + [seed * 100_000 + 90_000 + i for i in range(50)]


class Defaults(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(server.ADMIT_UNCACHED)
        self.assertFalse(server.SLOT_AFFINITY)
        e = engine()
        e.slot_held[1] = conv(1, 5000)
        self.assertEqual(e._admit_len(conv(1, 5000) + [5, 6]), (5002, 0))     # the whole length, as before
        e._mirror_note(conv(2, 5000))
        self.assertEqual(e._mirror, [])                                         # nothing noted
        self.assertIsNone(e.swarm_view())

    def test_mirror_size_from_the_engine_args(self):
        self.assertEqual(StrataEngine._mirror_limits(["--batch", "4"]), (4 + 4 + 1, 256))
        self.assertEqual(StrataEngine._mirror_limits(["--batch", "4", "--conversation-cache-slots", "12",
                                                      "--conversation-cache-min-tokens", "4096"]), (17, 4096))
        self.assertEqual(StrataEngine._mirror_limits(["--batch", "8", "--conversation-cache-mib", "0"]), (9, 256))
        with mock.patch.object(server, "ADMIT_MIRROR", 3):
            self.assertEqual(StrataEngine._mirror_limits(["--batch", "8"])[0], 3)


class FromTheRunConfig(unittest.TestCase):
    def test_the_config_env_turns_them_on(self):
        """The run config's "env" block (the engine's environment) turns the options on for a swarm profile; it also
        gives STRATA_ADMIT_SHORT_FIRST / STRATA_SLOT_KEEP_MIN then, and only then (an older config is unchanged)."""
        e = engine()
        e._swarm_config({"STRATA_ADMIT_SHORT_FIRST": "4096", "STRATA_SLOT_KEEP_MIN": "100"})
        self.assertEqual((e._uncached_on(), e._affinity_on(), e._short_first(), e._keep_min()), (False, False, 0, 4096))
        e._swarm_config({"STRATA_ADMIT_UNCACHED": "1", "STRATA_ADMIT_SHORT_FIRST": "4096",
                         "STRATA_SLOT_AFFINITY": "1", "STRATA_SLOT_KEEP_MIN": "100"})
        self.assertEqual((e._uncached_on(), e._affinity_on(), e._short_first(), e._keep_min()), (True, True, 4096, 100))
        self.assertEqual(e._admit_len(conv(1, 3000))[0], 3000)
        e._mirror_note(conv(1, 3000))
        self.assertEqual(e._admit_len(conv(1, 3000) + [1, 2])[0], 2)

    def test_images_use_the_full_admission_length(self):
        e = engine()
        e.alive = lambda: True
        e.gen = 1
        e._swarm_config({"STRATA_ADMIT_UNCACHED": "1"})
        prompt = conv(1, 3000)
        e._mirror_note(prompt)
        lengths = []
        def decline(cancel, plen, **kwargs):
            lengths.append(plen)
            yield from ()
            return False
        e._take_control = decline
        self.assertEqual(list(e.generate_batched(prompt + [1, 2], 1, {}, threading.Event(), embeddings=[object()])), [])
        self.assertEqual(lengths, [3002])
        self.assertEqual(e.swarm_stats["estimates"], 0)
        self.assertTrue(e.swarm_view()["admit_uncached"])
        e._swarm_config({"STRATA_ADMIT_UNCACHED": "0", "STRATA_SLOT_AFFINITY": ""})
        self.assertIsNone(e.swarm_view())


class ServedPrefix(unittest.TestCase):
    def test_lists_and_arrays_agree(self):
        e = engine()
        h = [1, 2, T, 3, 4]
        for f in (list, lambda x: array("i", x)):
            self.assertEqual(e._served_prefix(f(h), f(h + [9])), 5)          # the whole stream continued
            self.assertEqual(e._served_prefix(f(h), f([1, 2, T, 7, 7, 7])), 2)   # up to its turn checkpoint
            self.assertEqual(e._served_prefix(f(h), f([1, 9, T, 3])), 0)
            self.assertEqual(e._served_prefix(f(h), f(h)), 2)                 # never the whole prompt: its turn
            self.assertEqual(e._served_prefix(f([]), f(h)), 0)


@mock.patch.object(server, "ADMIT_UNCACHED", True)
class UncachedLength(unittest.TestCase):
    def test_an_idle_slot_and_the_mirror(self):
        e = engine()
        a, b = conv(1, 20_000), conv(2, 30_000)
        e.slot_held[2] = a
        self.assertEqual(e._admit_len(a + [7] * 600), (600, 20_000))           # back to its idle slot: 600 to read
        e.slot_busy[2] = True                                                   # (a busy slot holds nothing for it)
        self.assertEqual(e._admit_len(a + [7] * 600), (20_600, 0))
        e._mirror_note(b)                                                       # parked / main session
        self.assertEqual(e._admit_len(b + [8] * 900), (900, 30_000))
        # a next turn that renders the history again (not the reply as decoded): up to the turn checkpoint
        b2 = b + [T, 55, 56]
        e._mirror_note(b2)
        self.assertEqual(len(e._mirror), 1)                                     # it replaced the older turn
        self.assertEqual(e._admit_len(b2[:-2] + [9] * 100), (101, len(b2) - 3))
        # the usual next turn: the history up to the reply's header, the reply rendered again, a tool result
        c = conv(4, 50_000)
        e._mirror_note(c)
        self.assertEqual(e._admit_len(c[:-50] + [7] * 60 + [T] + [8] * 700), (1 + 60 + 1 + 700, 50_000 - 51))
        self.assertEqual(e._admit_len(conv(3, 10_000))[0], 10_000)              # a cold agent: all of it
        self.assertEqual(e.swarm_stats["estimates"], 6)

    def test_pipelined_slot_below_a_busy_one_holds_nothing(self):
        e = engine(4, groups=2, pipelined=True)
        a = conv(1, 8000)
        e.slot_held[0] = a
        self.assertEqual(e._admit_len(a + [1] * 10)[0], 10)
        e.slot_busy[1] = True                                                   # its group's window pads slot 0
        self.assertEqual(e._admit_len(a + [1] * 10)[0], 8010)

    def test_mirror_size_and_short_streams(self):
        e = engine(args=["--conversation-cache-slots", "1", "--conversation-cache-min-tokens", "1000"])
        self.assertEqual(e._mirror_cap, 4 + 1 + 1)
        e._mirror_note(conv(9, 900))                                            # shorter than the cache keeps
        self.assertEqual(e._mirror, [])
        for k in range(10):
            e._mirror_note(conv(k + 1, 2000))
        self.assertEqual(len(e._mirror), 6)
        self.assertEqual(e._admit_len(conv(1, 2000) + [5])[1], 0)               # the oldest went
        self.assertEqual(e._admit_len(conv(10, 2000) + [5])[1], 2000)

    def test_feedback_drops_what_the_engine_no_longer_holds(self):
        e = engine()
        a, b = conv(1, 40_000), conv(2, 40_000)
        e._mirror_note(a)
        e._mirror_note(b)
        p = a + [3] * 500
        n, hit = e._admit_len(p)
        self.assertEqual(hit, 40_000)
        e._mirror_feedback(p, hit, 40_000 - 100)                                # met (within the slack): kept
        self.assertEqual(len(e._mirror), 2)
        e._mirror_feedback(p, hit, 301)                                         # evicted: only the system prompt
        self.assertEqual(len(e._mirror), 1)
        self.assertEqual(e._admit_len(p)[1], 0)
        self.assertEqual(e._admit_len(b + [1])[1], 40_000)                      # the other conversation stays
        st = e.swarm_view()
        self.assertEqual((st["overestimates"], st["mirror_dropped"], st["mirror_streams"]), (1, 1, 1))
        e._mirror_feedback(p, 0, None)                                          # no reused field: nothing
        self.assertEqual(st["overestimates"], 1)

    def test_admission_order_by_what_is_read(self):
        """E8 B2: a returning agent (100K cached, 600 new) goes before a cold 30K prompt that came first, with
        STRATA_ADMIT_SHORT_FIRST; the cold one still goes after STRATA_ADMIT_MAX_WAIT_S (unchanged)."""
        e = engine()
        back = conv(1, 100_000)
        e.slot_held[0] = back
        cold_len, _ = e._admit_len(conv(2, 30_000))
        back_len, _ = e._admit_len(back + [4] * 600)
        self.assertEqual((cold_len, back_len), (30_000, 600))
        case = tao.AdmitOrderTest()                         # its queue(): requests that wait behind a read
        case.setUp()
        try:
            with mock.patch.object(server, "ADMIT_SHORT_FIRST", 4096):
                self.assertEqual(case.queue(("cold", cold_len), ("back", back_len)), ["back", "cold"])
        finally:
            case.tearDown()

    def test_give_way_compares_uncached_lengths(self):
        e = tao.engine_without_process(4)
        e.wait_lens.append([600, time.monotonic(), True])        # a returning agent: 600 of its 100K to read
        self.assertTrue(e._shorter_waiting(30_000))              # a cold 30K read gives way to it
        self.assertFalse(e._shorter_waiting(1000))


@mock.patch.object(server, "SLOT_AFFINITY", True)
class Affinity(unittest.TestCase):
    def test_a_short_side_request_slot_goes_before_an_agent_slot(self):
        e = engine(3)
        e.slot_held = [conv(1, 50_000), conv(2, 900), conv(3, 60_000)]
        e.slot_used = [1.0, 9.0, 5.0]                     # the agent in slot 0 was used longest ago
        with mock.patch.object(server, "SLOT_AFFINITY", False):
            self.assertEqual(e.pick_slot(conv(4, 10_000)), 0)   # before: least recent, the agent's 50K go
        self.assertEqual(e.pick_slot(conv(4, 10_000)), 1)       # the 900-token title request's slot instead

    def test_a_conversation_held_again_elsewhere_is_worth_nothing(self):
        e = engine(3)
        a = conv(1, 20_000)
        e.slot_held = [a, a + [T, 5, 6], conv(2, 30_000)]
        e.slot_used = [5.0, 6.0, 1.0]
        self.assertEqual(e.pick_slot(conv(3, 10_000)), 0)       # slot 1 continues slot 0's conversation

    def test_the_slot_of_a_waiting_agent_comes_last(self):
        e = engine(3)
        a, b, c = conv(1, 20_000), conv(2, 30_000), conv(3, 40_000)
        e.slot_held = [a, b, c]
        e.slot_used = [1.0, 2.0, 3.0]
        me, other = object(), object()
        e.slot_wanted = {other: a + [9] * 500, me: conv(4, 5000)}   # agent A's next turn waits for a slot
        self.assertEqual(e.pick_slot(conv(4, 5000), me), 1)      # not A's slot 0 although it is least recent
        self.assertEqual(e.pick_slot(a + [9] * 500, other), 0)   # A itself goes back to its own slot

    def test_a_held_start_still_comes_first(self):
        e = engine(3)
        a = conv(1, 20_000)
        e.slot_held = [[], a, []]
        self.assertEqual(e.pick_slot(a + [1, 2]), 1)

    def test_pipelined_groups_balance_first_then_what_is_lost(self):
        e = engine(4, groups=2, pipelined=True)             # groups [0 1] [2 3], order 0 2 1 3
        e.slot_held = [conv(1, 50_000), [], [], []]
        with mock.patch.object(server, "SLOT_AFFINITY", False):
            self.assertEqual(e.pick_slot(conv(5, 1000)), 0)   # before: the spreading order writes over slot 0
        self.assertEqual(e.pick_slot(conv(5, 1000)), 2)       # an empty group's slot instead
        e.slot_busy[2] = True
        self.assertEqual(e.pick_slot(conv(5, 1000)), 0)       # group balance comes first: [_ _] before [B _]
        e.slot_busy[2], e.slot_busy[1] = False, True          # slot 0 is a hole below busy slot 1: it costs nothing
        self.assertEqual(e.pick_slot(conv(5, 1000)), 0)
        e.slot_busy[1] = False
        self.assertEqual(e.pick_slot(conv(1, 50_000) + [3]), 0)   # its own conversation: back to slot 0


class DirectRequests(unittest.TestCase):
    """Real engine protocol over subprocess pipes; no HTTP listener or GPU."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "fake_strata.py"
        script.write_text(tp.FAKE_BATCH)
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", ["--batch", "2"])
        self.addCleanup(self.engine.close)
        self.engine._swarm_config({"STRATA_ADMIT_UNCACHED": "1"})

    def test_completed_request_enters_the_mirror(self):
        result = list(self.engine.generate_batched([123] * 300, 4, {}, threading.Event()))
        self.assertEqual(len([x for x in result if x is not None]), 4)
        self.assertEqual(len(self.engine._mirror), 1)

    def test_early_close_does_not_guess_the_final_cache_state(self):
        request = self.engine.generate_batched([123] * 300, 4, {}, threading.Event())
        for token in request:
            if token is not None:
                break
        request.close()
        self.assertEqual(self.engine._mirror, [])

    def test_close_during_terminal_token_flush_does_not_note_unvalidated_state(self):
        def terminal(cancel, on_token, **kwargs):
            on_token(111)
            self.engine.last = {"finish": "nonfinite"}
            yield from ()
        with mock.patch.object(self.engine, "_control", terminal):
            request = self.engine.generate_batched([123] * 300, 4, {}, threading.Event())
            self.assertEqual(next(request), 111)
            request.close()
        self.assertEqual(self.engine._mirror, [])


class SwarmService(tp.FakeEngineCase):
    """The real StrataEngine and Service with both options on, the fake engine behind them."""

    def setUp(self):
        p = [mock.patch.object(server, "ADMIT_UNCACHED", True), mock.patch.object(server, "SLOT_AFFINITY", True),
             mock.patch.object(server, "ADMIT_SHORT_FIRST", 64)]
        for x in p:
            x.start()
            self.addCleanup(x.stop)

    def test_requests_and_next_turns_are_answered_and_noted(self):
        self.start(2, reuse=True)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        out = self._two_at_once("hello " * 100)
        self.assertEqual(out["a"][0], "ok", out)
        self.assertEqual(out["b"][0], "ok", out)
        self.assertEqual(self.chat("long " * 600)["choices"][0]["message"]["content"], "ok, done.")
        sw = self.get("/metrics")["live"]["swarm"]
        self.assertTrue(sw["admit_uncached"] and sw["slot_affinity"])
        self.assertGreaterEqual(sw["estimates"], 4)
        self.assertGreaterEqual(sw["mirror_streams"], 1)        # the 3,000-token conversation was noted
        self.assertFalse(any(self.engine.slot_busy))
        self.assertEqual(self.engine.__dict__.get("slot_wanted", {}), {})   # nobody left waiting for a slot

    def test_a_long_cold_prompt_still_gives_way_to_a_short_one(self):
        self.start(2)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        self.assertTrue(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))


if __name__ == "__main__":
    unittest.main()
