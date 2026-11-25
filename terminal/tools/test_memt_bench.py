"""tools/test_memt_bench.py - the P7 measurement, without a GPU.

    python tools/test_memt_bench.py [-v]

A scripted engine peer and a stand-in reference pin what the harness measures:
warmup is discarded, every engine run starts with `MEMFORGET`, TPOT follows
(end - TTFT) / (n - 1), the summary is a median, and a side that did not run is
an error rather than a zero.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memt_bench as B  # noqa: E402
import memt_engine as ME  # noqa: E402


class Peer:
    """Answers MEMFORGET and MEMQUERY from a script; records what it was sent."""

    def __init__(self, tokens=(7, 8, 9)):
        self.tokens = list(tokens)
        self.answers: deque[str] = deque()
        self.sent: list[str] = []

    def write(self, line: str) -> None:
        self.sent.append(line)
        command = line.split(" ", 1)[0].strip()
        if command == "MEMFORGET":
            self.answers.append("MEMFORGET OK\n")
        elif command == "MEMQUERY":
            for token in self.tokens:
                self.answers.append("T %d\n" % token)
            self.answers.append("DONE %d 4 1.0 2.0 stop 0 0 4 0 0 0 0 0.0 4 0\n" % len(self.tokens))
        elif command in ("MEMRESET", "MEMSNAP", "MEMLOAD", "MEMCOMMIT"):
            self.answers.append("%s OK\n" % command)
        else:
            self.answers.append("ERR unknown\n")

    def readline(self) -> str:
        return self.answers.popleft() if self.answers else ""

    def engine(self) -> ME.MemtEngine:
        return ME.MemtEngine(self.write, self.readline)


class FakeRef:
    def __init__(self, tokens=(7, 8, 9), first_ms=5.0, e2e_ms=25.0):
        self.tokens = list(tokens)
        self.first_ms = first_ms
        self.e2e_ms = e2e_ms
        self.all_terminal = False
        self.calls = []
        self.sessions = []

    def create(self):
        session = "s%d" % len(self.sessions)
        self.sessions.append(session)
        return session

    def acquire(self, session):
        return "lease-" + session

    def close(self, session, lease):
        self.calls.append(("close", session, lease))
        return {}

    def query(self, session, ids, *, lease, max_new, sampling=None, stream=False):
        self.calls.append(("query", list(ids), max_new, dict(sampling or {}), stream))
        return {"ids": list(self.tokens), "text": "", "finish": "stop",
                "first_ms": self.first_ms, "end_ms": self.e2e_ms,
                "frames": 1 if self.all_terminal else len(self.tokens) - 1,
                "first_from_terminal": self.all_terminal, "prompt_tokens": len(ids)}


class Math(unittest.TestCase):
    def test_rate(self):
        self.assertEqual(B.rate(10, 0.5), 20.0)
        self.assertIsNone(B.rate(10, 0.0))
        self.assertIsNone(B.rate(0, 1.0))
        self.assertIsNone(B.rate(10, None))

    def test_run_record_tpot_and_tps(self):
        record = B._run_record([1, 2, 3], 10.0, 50.0)
        self.assertEqual(record["output_tokens"], 3)
        self.assertAlmostEqual(record["tpot_ms"], 20.0)
        self.assertAlmostEqual(record["tps"], 60.0)
        self.assertAlmostEqual(record["decode_tps"], 50.0)

    def test_single_token_has_no_tpot(self):
        record = B._run_record([1], 10.0, 20.0)
        self.assertIsNone(record["tpot_ms"])
        self.assertIsNone(record["decode_tps"])
        self.assertEqual(record["tps"], 50.0)

    def test_missing_first_token_keeps_e2e(self):
        record = B._run_record([], None, 20.0)
        self.assertIsNone(record["ttft_ms"])
        self.assertIsNone(record["tps"])

    def test_summary_is_a_median(self):
        runs = [{"ttft_ms": 10.0, "e2e_ms": 30.0, "tpot_ms": 1.0, "tps": 2.0, "decode_tps": 3.0,
                 "prompt_ms": 1.0, "decode_ms": 1.0, "output_tokens": 2, "ids_digest": "a"},
                {"ttft_ms": 20.0, "e2e_ms": 50.0, "tpot_ms": 3.0, "tps": 4.0, "decode_tps": 5.0,
                 "prompt_ms": 3.0, "decode_ms": 3.0, "output_tokens": 2, "ids_digest": "a"},
                {"ttft_ms": 30.0, "e2e_ms": 40.0, "tpot_ms": 2.0, "tps": 3.0, "decode_tps": 4.0,
                 "prompt_ms": 2.0, "decode_ms": 2.0, "output_tokens": 2, "ids_digest": "a"}]
        summary = B.summarize_side(runs)
        self.assertEqual(summary["ttft_ms"], 20.0)
        self.assertEqual(summary["e2e_ms"], 40.0)
        self.assertEqual(summary["tpot_ms"], 2.0)
        self.assertEqual(summary["output_tokens"], 2)
        self.assertTrue(summary["identical_outputs"])

    def test_compare_sides(self):
        ref = {"summary": {"ttft_ms": 20.0, "tpot_ms": 4.0, "e2e_ms": 100.0, "tps": 10.0, "output_tokens": 9}}
        eng = {"summary": {"ttft_ms": 10.0, "tpot_ms": 2.0, "e2e_ms": 50.0, "tps": 20.0, "output_tokens": 9}}
        ratio = B.compare_sides(ref, eng)
        self.assertAlmostEqual(ratio["ttft_ms"], 2.0)
        self.assertAlmostEqual(ratio["tps"], 2.0)
        self.assertTrue(ratio["same_output_tokens"])
        eng["summary"]["output_tokens"] = 8
        self.assertFalse(B.compare_sides(ref, eng)["same_output_tokens"])


class Measurement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_engine_runs_discard_warmup_and_forget(self):
        peer = Peer()
        measured = B.engine_measure(peer.engine(), [1, 2, 3, 4], max_new=3, runs=2, warmup=1)
        self.assertEqual(len(measured["runs"]), 2)
        forgets = [line for line in peer.sent if line.startswith("MEMFORGET")]
        self.assertEqual(len(forgets), 3)
        self.assertEqual(measured["summary"]["output_tokens"], 3)
        self.assertIsNotNone(measured["runs"][0]["ttft_ms"])
        self.assertAlmostEqual(measured["runs"][0]["prompt_ms"], 1.0)

    def test_reference_runs_close_the_session(self):
        ref = FakeRef()
        measured = B.reference_measure(ref, [1, 2], max_new=3, runs=2, warmup=1)
        self.assertEqual(len(measured["runs"]), 2)
        queries = [call for call in ref.calls if call[0] == "query"]
        self.assertEqual(len(queries), 3)
        self.assertTrue(all(call[4] for call in queries))          # streamed: TTFT is measurable
        self.assertEqual(queries[0][3]["temperature"], 0.0)        # greedy
        self.assertEqual([call[0] for call in ref.calls][-1], "close")
        self.assertAlmostEqual(measured["summary"]["ttft_ms"], 5.0)

    def test_run_bench_both_sides(self):
        report = B.run_bench(prompts=[("q1", [1, 2])], max_new=3, runs=2, warmup=0,
                             ref=FakeRef(), engine=Peer().engine(), outdir=self.root)
        self.assertTrue(report["ok"], B.format_report(report))
        row = report["prompts"][0]
        self.assertIn("reference", row["sides"])
        self.assertIn("engine", row["sides"])
        self.assertAlmostEqual(row["engine_vs_reference"]["ttft_ms"], 5.0 / row["sides"]["engine"]["summary"]["ttft_ms"], places=3)
        self.assertTrue((self.root / "bench.json").exists())
        text = B.format_report(report)
        self.assertIn("engine vs reference", text)
        self.assertIn("result: MEASURED", text)

    def test_different_output_counts_are_not_a_comparison(self):
        report = B.run_bench(prompts=[("q1", [1, 2])], max_new=3, runs=2, warmup=0,
                             ref=FakeRef(tokens=(7, 8, 9)), engine=Peer(tokens=(7, 8)).engine(),
                             outdir=self.root)
        self.assertFalse(report["ok"])
        self.assertFalse(report["prompts"][0]["ok"])
        self.assertFalse(report["prompts"][0]["engine_vs_reference"]["comparable"])
        text = B.format_report(report)
        self.assertIn("NOT COMPARABLE", text)
        self.assertNotIn("tok/s x", text)
        self.assertIn("result: INCOMPLETE", text)

    def test_a_terminal_only_stream_is_noted(self):
        ref = FakeRef()
        ref.all_terminal = True
        report = B.run_bench(prompts=[("q1", [1])], max_new=3, runs=1, warmup=0,
                             ref=ref, side="ref", outdir=self.root)
        self.assertTrue(any("terminal frame" in note for note in report["notes"]),
                        report["notes"])

    def test_run_bench_one_side(self):
        report = B.run_bench(prompts=[("q1", [1])], max_new=1, runs=1, warmup=0,
                             engine=Peer().engine(), side="engine", outdir=self.root)
        self.assertTrue(report["ok"])
        self.assertNotIn("engine_vs_reference", report["prompts"][0])

    def test_both_side_needs_both(self):
        with self.assertRaises(ValueError):
            B.run_bench(prompts=[("q1", [1])], max_new=1, runs=1, warmup=0,
                        engine=Peer().engine(), side="both", outdir=self.root)

    def test_one_side_run_is_measured_but_has_no_ratio(self):
        report = B.run_bench(prompts=[("q1", [1])], max_new=1, runs=1, warmup=0,
                             ref=FakeRef(), side="ref", outdir=self.root)
        self.assertTrue(report["ok"])
        self.assertNotIn("engine", report["prompts"][0]["sides"])
        self.assertIn("result: MEASURED", B.format_report(report))

    def test_cli_rejects_missing_arguments(self):
        self.assertEqual(B.main(["--side", "both", "--prompt", "x.ids"]), 2)
        self.assertEqual(B.main(["--side", "engine", "--prompt", "x.ids"]), 2)


if __name__ == "__main__":
    unittest.main()
