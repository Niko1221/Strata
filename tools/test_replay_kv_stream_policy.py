"""Tests for tools/replay_kv_stream_policy.py's policy models (no GPU, no model pack, no network).

    python -m unittest tools.test_replay_kv_stream_policy

The hand-worked traces pin the engine's CLOCK second chance (kv_stream.cu's resolve_kernel) against LRFU and
Belady; the invariants hold for any trace; the synthetic patterns are deterministic per seed.
"""
from __future__ import annotations

import struct
import unittest

from tools.replay_kv_stream_policy import ClockPool, LrfuPool, LruPool, load_trace, oracle_misses, run, synthetic


def misses(pool_cls, calls, slots, **kw):
    pool = pool_cls(slots, **kw) if kw else pool_cls(slots)
    return sum(pool.call(ids, [1.0] * len(ids)) for ids in calls)


class HandWorked(unittest.TestCase):
    """Two traces small enough to follow by hand, with the misses every policy must produce."""

    def test_clock_loses_the_hot_block_a_sweep_cleared(self):
        # 0 is read every other call, 1..4 once each and never again: the sweep clears 0's reference on the
        # pass that takes a one-off, so the next call re-reads 0.  LRFU's keep score knows 0 is the hot one.
        calls = [[0], [0], [1], [0], [2], [0], [3], [0], [4], [0]]
        self.assertEqual(misses(ClockPool, calls, 2), 7)
        self.assertEqual(misses(LruPool, calls, 2), 5)
        self.assertEqual(misses(LrfuPool, calls, 2), 5)          # lambda 0.5, no decay tick in 10 calls
        self.assertEqual(oracle_misses(calls, 2), 5)

    def test_lrfu_keeps_the_historically_hot_block_lru_ages_out(self):
        # 0 is read three times and then rests one call while a one-off passes through the second slot: LRU
        # evicts 0 for being the older resident, LRFU keeps it for its keep score.
        calls = [[0], [0], [0], [1], [2], [0]]
        self.assertEqual(misses(LrfuPool, calls, 2), 3)
        self.assertEqual(misses(LruPool, calls, 2), 4)
        self.assertEqual(misses(ClockPool, calls, 2), 4)
        self.assertEqual(oracle_misses(calls, 2), 3)

    def test_only_the_top_weight_mass_accumulates_score(self):
        # top-frac 0.9 of a call whose weight mass is all in one block: the zeros never gain score, so the
        # winner of the eviction is decided by the block that did.
        p = LrfuPool(1, lam=0.5, gain=1.0, top_frac=0.9, decay_every=64)
        p.call([0, 1], [1.0, 0.0])
        self.assertEqual((p.ks.get(0), p.ks.get(1, 0.0)), (1.0, 0.0))
        p.call([2], [1.0])                                        # 1 is the only resident to evict: KS 0 vs 0 tie
        self.assertIn(2, p.resident)

    def test_a_call_that_fills_the_pool_overflows_instead_of_hanging(self):
        # every slot holds a block this call names: the kernel stops its sweep at 3*n and leaves the rest out
        # ("OVERFLOW (too few resident cells)"), and the model must terminate the same way
        for pool in (ClockPool(4), LruPool(4), LrfuPool(4)):
            self.assertEqual(pool.call([0, 1, 2, 3], [1.0] * 4), 4)          # cold: fills the pool
            self.assertEqual(pool.call([4, 5, 6, 7, 8], [1.0] * 5), 5)       # all five miss; four slots can move


class Invariants(unittest.TestCase):
    """What must hold of every trace and every seed."""

    def test_no_policy_beats_the_compulsory_load(self):
        tr = synthetic("agent", layers=2, calls=200, universe=5000, width=64, tail=64, task=32, oneshot=8,
                       every=1, seed=1)
        t = run(tr, layers=2, slots=256, block_bytes=4224, lam=0.5, gain=1.0, top_frac=0.9, decay_every=64)
        comp = t["policies"]["compulsory"]["misses"]
        for name, p in t["policies"].items():
            self.assertGreaterEqual(p["misses"], comp, name)
        self.assertLessEqual(t["policies"]["oracle"]["misses"], t["policies"]["lrfu"]["misses"])
        self.assertLessEqual(t["policies"]["oracle"]["misses"], t["policies"]["lru"]["misses"])
        self.assertLessEqual(t["policies"]["oracle"]["misses"], t["policies"]["clock"]["misses"])

    def test_a_scan_reads_every_block_once_whatever_the_policy(self):
        tr = synthetic("scan", layers=1, calls=40, universe=1000, width=25, tail=0, task=0, oneshot=0,
                       every=1, seed=2)
        t = run(tr, layers=1, slots=64, block_bytes=4224, lam=0.5, gain=1.0, top_frac=0.9, decay_every=64)
        self.assertEqual(t["unique"], 1000)
        for name, p in t["policies"].items():
            self.assertEqual(p["misses"], 1000, name)             # a scan is all compulsory, no policy helps

    def test_synthetic_is_deterministic_per_seed(self):
        # the generators' miss counts are structural (any set of N distinct blocks misses the same), so what a
        # seed decides is the trace itself - same seed, same blocks; other seed, other blocks
        def tr(seed):
            return synthetic("intermittent", layers=1, calls=50, universe=300, width=8, tail=0, task=0, oneshot=4,
                             every=5, seed=seed)
        self.assertEqual(tr(7), tr(7))
        self.assertNotEqual(tr(7), tr(8))
        def snap(seed):
            return run(tr(seed), layers=1, slots=32, block_bytes=4224, lam=0.5, gain=1.0, top_frac=0.9,
                       decay_every=8)
        self.assertEqual(snap(7), snap(7))


class TraceFile(unittest.TestCase):
    def test_round_trip(self):
        import tempfile
        from pathlib import Path
        # weights that are exact in binary32, so the file reads back what was written; k varies per record, as the
        # engine's --dump-kv-select writes it
        recs = [(0, [1, 2, 3], [0.5, 0.25, 0.25]), (1, [4, 5], [1.0, 0.5]), (0, [7, 8, 9], [0.25, 0.25, 0.5])]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.bin"
            with open(path, "wb") as f:
                for layer, ids, ws in recs:
                    f.write(struct.pack("<ii", layer, len(ids)))
                    f.write(struct.pack(f"<{len(ids)}i", *ids))
                    f.write(struct.pack(f"<{len(ids)}f", *ws))
            got = load_trace(str(path))
        self.assertEqual(got, recs)

    def test_a_bad_file_is_refused(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.bin"
            path.write_bytes(b"\x01\x00\x00\x00\x03\x00\x00\x00")   # one record of k=3 with nothing behind it
            with self.assertRaises(SystemExit):
                load_trace(str(path))


if __name__ == "__main__":
    unittest.main()
