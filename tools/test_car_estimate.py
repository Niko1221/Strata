"""Tests for tools/car_estimate.py - the offline cache-aware routing estimate - with no GPU and no trace from a
real model.

The tool's own rules are self-tested inside it (`python tools/car_estimate.py --selftest`), including a
cross-check against a golden case emitted by the engine's `car_substitute_test`.  What this file adds is the
part a shell user hits: the reader refuses malformed traces with a clear message, the residency helpers do
what the engine's fill does, the report carries the numbers it promises, and `--json` is machine-readable.

    python -m unittest tools.test_car_estimate
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import car_estimate as CAR  # noqa: E402


def trace_with(n_layers: int, n_expert: int, records) -> CAR.Trace:
    t = CAR.Trace(n_layers, n_expert)
    for layer, n_tok, k, ids, logits in records:
        t.add(layer, n_tok, k, ids, logits)
    return t


class TraceReader(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.bin")
            CAR.write_trace(path, 2, 4, [(0, 1, 2, [3, 1], [-1.0, -2.0, -3.0, 7.0]),
                                        (1, 2, 2, [0, 1, 2, 3], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2])])
            t = CAR.read_trace(path)
            self.assertEqual((t.n_layers, t.n_expert), (2, 4))
            self.assertEqual(t.n_tokens(), 3)
            self.assertEqual(t.records[0][3], [3, 1])
            self.assertAlmostEqual(t.records[1][4][0], 0.5)

    def test_refusals_name_the_problem(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "nope.bin")
            with self.assertRaises(CAR.TraceError) as ctx:
                CAR.read_trace(missing)
            self.assertIn("no such file", str(ctx.exception))

            bad_magic = os.path.join(d, "magic.bin")
            Path(bad_magic).write_bytes(b"not-a-trace-at-all")
            with self.assertRaises(CAR.TraceError) as ctx:
                CAR.read_trace(bad_magic)
            self.assertIn("not a STRCS1 trace", str(ctx.exception))

            truncated = os.path.join(d, "trunc.bin")
            Path(truncated).write_bytes(CAR.MAGIC + struct.pack("<IIII", 1, 2, 4, 0) + struct.pack("<III", 0, 2, 2) + b"\x00" * 4)
            with self.assertRaises(CAR.TraceError) as ctx:
                CAR.read_trace(truncated)
            self.assertIn("truncated", str(ctx.exception))
            self.assertIn("needs 48 bytes and 4 are left", str(ctx.exception))

            # ... but a trace whose LAST record is mid-flush is a live probe file: the complete records are used
            # and the dropped tail is named, unless --strict (strict=True) is asked for.
            live = os.path.join(d, "live.bin")
            CAR.write_trace(live, 1, 4, [(0, 1, 2, [0, 1], [-1.0] * 4)])
            with open(live, "ab") as f:
                f.write(struct.pack("<III", 0, 1, 2) + b"\x00" * 4)
            import io, contextlib
            notes = io.StringIO()
            with contextlib.redirect_stderr(notes):
                t_live = CAR.read_trace(live)
            self.assertEqual(t_live.n_tokens(), 1)
            self.assertIn("ends mid-write", notes.getvalue())
            with self.assertRaises(CAR.TraceError):
                CAR.read_trace(live, strict=True)

            bad_version = os.path.join(d, "ver.bin")
            Path(bad_version).write_bytes(CAR.MAGIC + struct.pack("<IIII", 9, 2, 4, 0))
            with self.assertRaises(CAR.TraceError) as ctx:
                CAR.read_trace(bad_version)
            self.assertIn("version 9", str(ctx.exception))

    def test_record_shapes_are_checked_when_writing_a_trace_in_memory(self):
        t = CAR.Trace(1, 4)
        with self.assertRaises(CAR.TraceError):
            t.add(0, 1, 2, [1, 2, 3], [-1.0] * 4)          # three ids for k = 2
        with self.assertRaises(CAR.TraceError):
            t.add(0, 1, 2, [1, 2], [-1.0] * 3)             # three logits for four experts
        with self.assertRaises(CAR.TraceError):
            t.add(7, 1, 2, [1, 2], [-1.0] * 4)             # a layer outside the trace
        t.add(0, 1, 2, [1, 2], [-1.0] * 4)                 # and the valid one is accepted


class Residency(unittest.TestCase):
    def test_profile_fill_takes_the_ranked_prefix_per_layer(self):
        pairs = [(1, 5), (0, 9), (1, 6), (0, 3), (1, 7)]
        res = CAR.residency_from_profile(pairs, 2, 16, 2)
        self.assertEqual(res[0], [9, 3])
        self.assertEqual(res[1], [5, 6])

    def test_profile_fill_ignores_pairs_outside_the_geometry(self):
        res = CAR.residency_from_profile([(0, 99), (0, 2)], 1, 4, 1)
        self.assertEqual(res[0], [2])

    def test_frequency_is_an_oracle_and_ties_go_to_the_lower_id(self):
        t = trace_with(1, 4, [(0, 3, 1, [3, 3, 1], [-50.0] * 12)])
        self.assertEqual(CAR.residency_by_frequency(t, 1), [[3]])
        # expert 3 is routed twice, 1 once, and 0 and 2 never: the oracle keeps the routed ones, hottest first
        self.assertEqual(CAR.residency_by_frequency(t, 4), [[3, 1]])


class Estimate(unittest.TestCase):
    def test_counts_and_ratio(self):
        # one layer, k = 2, resident = {3}: miss 1 (9.0) -> ratio exp(1); miss 0 (9.5) loses the rescan.
        logits = [-50.0] * 4
        logits[3], logits[1], logits[0] = 10.0, 9.0, 9.5
        t = trace_with(1, 4, [(0, 1, 2, [1, 0], logits)])
        est = CAR.estimate(t, [[3]], tau=0.35)
        self.assertEqual(est["misses"], 2)
        self.assertEqual(est["substitutions"], 1)
        self.assertAlmostEqual(est["mean_ratio"], math.exp(1.0), places=6)
        self.assertAlmostEqual(est["misses_after_per_token"], 1.0)
        self.assertEqual(est["rank_histogram"], {"0": 1})

    def test_threshold_one_is_off(self):
        logits = [-50.0] * 4
        logits[3], logits[1] = 10.0, -10.0           # the resident expert is scored well ABOVE the miss
        t = trace_with(1, 4, [(0, 1, 2, [1, 0], logits)])
        est = CAR.estimate(t, [[3]], tau=1.0)
        self.assertEqual(est["substitutions"], 0)
        self.assertEqual(est["misses"], 2)

    def test_work_not_moved_is_reported_in_the_trace_s_units(self):
        logits = [-50.0] * 8
        logits[0], logits[4] = 10.0, 5.0
        t = trace_with(1, 8, [(0, 4, 2, [4, 5, 6, 7, 4, 5, 6, 7], logits * 4)])
        est = CAR.estimate(t, [[0]], tau=0.0, blob_bytes=1000000, pcie_frac=0.5)
        self.assertEqual(est["substitutions_per_token"], 1.0)
        self.assertAlmostEqual(est["expert_mb_not_moved_per_token"], 1000000 / (1024 * 1024))
        self.assertAlmostEqual(est["pcie_mb_not_moved_per_token"], 500000 / (1024 * 1024))

    def test_k_mismatch_is_refused_when_it_is_asked_for(self):
        t = trace_with(1, 8, [(0, 1, 2, [0, 1], [-1.0] * 8)])
        with self.assertRaises(CAR.TraceError):
            CAR.estimate(t, [[]], tau=0.35, k_expected=10)


class CommandLine(unittest.TestCase):
    def test_selftest_passes(self):
        proc = subprocess.run([sys.executable, str(ROOT / "tools" / "car_estimate.py"), "--selftest"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("all cases passed", proc.stdout)

    def test_report_and_json_from_a_trace(self):
        with tempfile.TemporaryDirectory() as d:
            trace = os.path.join(d, "scores.bin")
            logits = [-50.0] * 8
            logits[0], logits[4], logits[5] = 10.0, 5.0, 5.0
            CAR.write_trace(trace, 1, 8, [(0, 2, 2, [4, 5, 6, 7], logits * 2)])
            out = os.path.join(d, "est.json")
            proc = subprocess.run([sys.executable, str(ROOT / "tools" / "car_estimate.py"),
                                   "--trace", trace, "--per-layer", "1", "--tau", "0.35,0.8", "--json", out],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("substitutions", proc.stdout)
            doc = json.loads(Path(out).read_text())
            self.assertEqual(doc["tokens"], 2)
            self.assertEqual(len(doc["estimates"]), 2)
            for est in doc["estimates"]:
                # the oracle's one resident expert is 4 (the only routed one), so token 1's pick 4 is a hit and
                # its pick 5 is the single miss; token 2 has no resident pick at all and misses both, one of
                # which the rescued expert takes over.
                self.assertEqual(est["misses"], 3)
                self.assertGreaterEqual(est["substitutions"], 1)

    def test_a_malformed_trace_fails_with_a_message_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            trace = os.path.join(d, "junk.bin")
            Path(trace).write_bytes(b"junk")
            proc = subprocess.run([sys.executable, str(ROOT / "tools" / "car_estimate.py"), "--trace", trace],
                                  capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("not a STRCS1 trace", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main()
