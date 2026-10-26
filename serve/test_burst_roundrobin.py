"""Round-robin over the burst cursors, against a fake engine: three requests interleave, each advances."""
import unittest

from serve.burst import BurstCursor


class FakeEngine:
    """One token per generate() call, recording who asked."""

    def __init__(self):
        self.calls = []

    def generate(self, ids, max_new, who):
        self.calls.append(who)
        return [100 + len(ids) % 7] * max_new


def run_round_robin(cursors, engine, names, rounds):
    """What the server would do: one burst per request per round."""
    out = {n: [] for n in names}
    for _ in range(rounds):
        for c, name in zip(cursors, names):
            k = c.next_chunk()
            if k == 0:
                continue
            tokens = engine.generate(c.continuation(), k, name)
            out[name].extend(tokens)
            c.accept(tokens)
    return out


class RoundRobin(unittest.TestCase):
    def test_three_requests_interleave(self):
        engine = FakeEngine()
        cs = [BurstCursor(ids=[1 + i], max_new=6, burst=2) for i in range(3)]
        got = run_round_robin(cs, engine, ["a", "b", "c"], rounds=3)
        # each round every live request takes the engine once: strict round robin
        self.assertEqual(engine.calls, ["a", "b", "c"] * 3)
        self.assertEqual(len(got["a"]), 6)
        self.assertEqual(len(got["b"]), 6)
        self.assertEqual(len(got["c"]), 6)
        self.assertTrue(all(c.finished for c in cs))

    def test_a_short_request_leaves_the_rotation(self):
        engine = FakeEngine()
        a = BurstCursor(ids=[1], max_new=2, burst=2)      # one burst and done
        b = BurstCursor(ids=[2], max_new=4, burst=2)
        got = run_round_robin([a, b], engine, ["a", "b"], rounds=3)
        self.assertEqual(engine.calls, ["a", "b", "b"])   # a leaves after its only chunk
        self.assertEqual(len(got["a"]), 2)
        self.assertEqual(len(got["b"]), 4)

    def test_continuation_carries_the_prompt_and_the_tokens(self):
        engine = FakeEngine()
        c = BurstCursor(ids=[7, 8], max_new=4, burst=2)
        seen = []
        for _ in range(2):
            ids = c.continuation()
            seen.append(len(ids))
            c.accept(engine.generate(ids, c.next_chunk(), "x"))
        self.assertEqual(seen, [2, 4])
