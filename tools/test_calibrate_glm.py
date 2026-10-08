"""Tests for tools/calibrate_glm.py without a GPU: a stand-in engine whose decode speed is a function of the CPU lane
settings a request names (`strata_tune`: pcie_frac, cpu_threads).

    python -m unittest tools.test_calibrate_glm
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_glm as CAL  # noqa: E402


class FakeEngine:
    """Decode tok/s = speed(share, threads); the engine started with share `own_share` and `threads` threads."""

    def __init__(self, speed, threads=40, own_share=0.0):
        self.speed = speed
        self.own = (own_share, threads)
        self.info = {"cpu_threads": threads, "pcie_share": f"{own_share:.2f}"}
        self.last = {}
        self.requests = []
        self.closed = False

    def generate(self, ids, max_new, sampling, cancel):
        tune = sampling.get("strata_tune") or {}
        share = tune.get("pcie_frac", self.own[0])
        threads = tune.get("cpu_threads", self.own[1])
        self.requests.append((share, threads))
        self.last = {"decode_ms": max_new / self.speed(share, threads) * 1000.0}
        for i in range(max_new):
            yield i


def run_measure(engine):
    started = []

    def start(cfg):
        started.append(cfg)
        return engine
    res = CAL.measure({"env": {}}, [[1, 2, 3]] * 3, start, say=lambda *a: None, extra_threads=())
    return res, started


class Measure(unittest.TestCase):
    def test_faster_share_and_threads_are_kept(self):
        # more PCIe share and fewer threads are faster here: 0.25 and 20 threads win by far more than MIN_GAIN
        eng = FakeEngine(lambda s, t: 15.0 + (5.0 if abs(s - 0.25) < 1e-9 else 0.0) + (3.0 if t == 20 else 0.0))
        res, started = run_measure(eng)
        self.assertEqual(len(started), 1)                       # one engine start: no restarts
        self.assertEqual(res["settings"], {CAL.SHARE_ENV: "0.25", CAL.THREADS_ENV: "20"})
        self.assertEqual(res["report"]["tok_s"], 23.0)

    def test_small_gain_keeps_the_engines_own(self):
        # 2% better is within the noise: nothing changes
        eng = FakeEngine(lambda s, t: 20.0 * (1.02 if t == 30 else 1.0))
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {})
        self.assertEqual(res["report"]["tok_s"], 20.0)

    def test_own_split_is_measured_without_a_share_key(self):
        eng = FakeEngine(lambda s, t: 18.0)
        run_measure(eng)
        self.assertIn((0.0, 40), eng.requests)                  # the engine's own: no keys at all
        shares = {r[0] for r in eng.requests}
        self.assertTrue(set(CAL.PCIE_SHARES) <= shares)

    def test_sweeps_stop_once_far_behind(self):
        # the PCIe share costs a lot from 0.1 on, fewer threads from 20 down: neither sweep goes past its first loser
        eng = FakeEngine(lambda s, t: 20.0 * (0.4 if s >= 0.1 else 1.0) * (0.5 if t <= 20 else 1.0))
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {})
        self.assertNotIn(0.25, {r[0] for r in eng.requests})
        self.assertNotIn(10, {r[1] for r in eng.requests})

    def test_threads_only(self):
        eng = FakeEngine(lambda s, t: 10.0 + (2.0 if t == 10 else 0.0))
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {CAL.THREADS_ENV: "10"})

    def test_no_cpu_lane(self):
        eng = FakeEngine(lambda s, t: 30.0, threads=0, own_share=1.0)
        res, _ = run_measure(eng)
        self.assertEqual(res["settings"], {})
        self.assertEqual(eng.requests, [])


class Helpers(unittest.TestCase):
    def test_apply_replaces_and_clears(self):
        env = {"STRATA_GLM_RAM_HEADROOM_GB": "4", CAL.SHARE_ENV: "0.50", CAL.THREADS_ENV: "12"}
        self.assertEqual(CAL.apply(env, {CAL.THREADS_ENV: "20"}),
                         {"STRATA_GLM_RAM_HEADROOM_GB": "4", CAL.THREADS_ENV: "20"})
        self.assertEqual(CAL.apply(env, {}), {"STRATA_GLM_RAM_HEADROOM_GB": "4"})

    def test_thread_candidates(self):
        self.assertEqual(CAL.thread_candidates(40, (43, 21)), [40, 30, 27, 20, 10, 21])
        self.assertEqual(CAL.thread_candidates(3), [3, 2])

    def test_pick(self):
        self.assertEqual(CAL.pick({"a": [10.0], "b": [10.2]}, "a"), "a")
        self.assertEqual(CAL.pick({"a": [10.0], "b": [10.5]}, "a"), "b")

    def test_cpu_list(self):
        self.assertEqual(CAL.cpu_list("0-3,8,10-11\n"), [0, 1, 2, 3, 8, 10, 11])

    def test_host_thread_extras(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "cpu"
            (root).mkdir()
            (root / "online").write_text("0-7\n")
            for c in range(8):
                t = root / f"cpu{c}" / "topology"
                t.mkdir(parents=True)
                t.joinpath("physical_package_id").write_text("0\n" if c < 4 else "1\n")
            hybrid = Path(d) / "cpu_core_cpus"
            self.assertEqual(CAL.host_thread_extras(root, hybrid), [3])        # two sockets of 4: 4 - 1
            hybrid.write_text("0-5\n")
            self.assertEqual(CAL.host_thread_extras(root, hybrid), [3, 5])     # + a hybrid CPU's 6 P-core CPUs - 1


if __name__ == "__main__":
    unittest.main()
