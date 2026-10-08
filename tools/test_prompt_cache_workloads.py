"""Offline tests for tools/prompt_cache_workloads.py (no server, no GPU).

    python -m unittest tools.test_prompt_cache_workloads
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prompt_cache_workloads as W  # noqa: E402
import analyze_prompt_cache_log as ANALYZER  # noqa: E402


class Scenarios(unittest.TestCase):
    def test_all_scenarios_present(self):
        self.assertEqual(set(W.SCENARIOS), {"long_document_many_questions", "edited_last_message",
                                            "branch_switching", "periodic_checkpoint_gap", "pin_siblings",
                                            "branch_pressure"})

    def test_long_document_grows_a_shared_prefix(self):
        turns = W.SCENARIOS["long_document_many_questions"]()
        self.assertEqual(len(turns), 5)
        shared = [W._shared_prefix_tokens(turns[i - 1], turns[i]) for i in range(1, len(turns))]
        self.assertTrue(all(s > 0 for s in shared))
        self.assertEqual(shared, sorted(shared))   # the prefix only grows

    def test_edited_last_message_reuses_a_partial_prefix(self):
        turns = W.SCENARIOS["edited_last_message"]()
        self.assertEqual(len(turns), 3)
        shared = W._shared_prefix_tokens(turns[1], turns[2])
        self.assertGreater(shared, 0)
        self.assertLess(shared, W._prompt_tokens(turns[1]))   # reuse stops at the edit

    def test_branch_switching_shares_a_prefix_but_diverges(self):
        turns = W.SCENARIOS["branch_switching"]()
        self.assertEqual(len(turns), 6)
        shared = W._shared_prefix_tokens(turns[0], turns[1])
        self.assertGreater(shared, 0)
        self.assertLess(shared, W._prompt_tokens(turns[1]))

    def test_pin_siblings_carry_a_prefix_hint(self):
        turns = W.SCENARIOS["pin_siblings"]()
        self.assertEqual(len(turns), 4)
        self.assertTrue(all(t["strata_prefix"] == {"messages": 1} for t in turns))
        self.assertTrue(all(len(t["messages"]) == 2 for t in turns))

    def test_periodic_gap_and_pressure_turn_counts(self):
        gap = W.SCENARIOS["periodic_checkpoint_gap"]()
        self.assertEqual(len(gap), 2)
        self.assertGreater(W._shared_prefix_tokens(gap[0], gap[1]), 0)
        self.assertEqual(len(W.SCENARIOS["branch_pressure"]()), 9)

    def test_reply_placeholders_use_actual_previous_responses(self):
        turns = W.SCENARIOS["long_document_many_questions"]()
        resolved = W._resolve_turn(turns[2], ["first actual reply", "second actual reply"])
        replies = [m["content"] for m in resolved["messages"] if m["role"] == "assistant"]
        self.assertEqual(replies, ["first actual reply", "second actual reply"])


class SyntheticTrace(unittest.TestCase):
    def test_every_scenario_trace_parses_with_the_analyzer(self):
        for name, build in W.SCENARIOS.items():
            with self.subTest(scenario=name):
                turns = build()
                records = [ANALYZER.parse_line(line, name, i)
                           for i, line in enumerate(W.synthetic_trace(name, turns), 1) if not line.startswith("#")]
                self.assertTrue(all(r is not None for r in records))
                summary = ANALYZER.analyze([r for r in records if r])["summary"]
                self.assertEqual(summary["event_counts"]["decision"], len(turns))
                self.assertIsNotNone(summary["reuse_rate"])
                self.assertGreaterEqual(summary["reuse_rate"], 0)
                self.assertLessEqual(summary["reuse_rate"], 1)

    def test_reuse_scenarios_report_reuse(self):
        turns = W.SCENARIOS["long_document_many_questions"]()
        records = [ANALYZER.parse_line(l) for l in W.synthetic_trace("ldq", turns) if not l.startswith("#")]
        self.assertGreater(ANALYZER.analyze(records)["summary"]["reused"], 0)


class Cli(unittest.TestCase):
    def test_list(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "prompt_cache_workloads.py"), "--list"],
                                check=True, capture_output=True, text=True)
        for name in W.SCENARIOS:
            self.assertIn(name, result.stdout)

    def test_emit_trace_then_analyze(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trace.log"
            subprocess.run([sys.executable, str(ROOT / "tools" / "prompt_cache_workloads.py"),
                            "--emit-trace", str(log)], check=True, capture_output=True, text=True)
            records = ANALYZER.read_logs([log])
            report = ANALYZER.analyze(records, [str(log)])
            self.assertEqual(report["summary"]["event_counts"]["decision"],
                             sum(len(b()) for b in W.SCENARIOS.values()))
            self.assertGreater(report["summary"]["events"], 0)


if __name__ == "__main__":
    unittest.main()
