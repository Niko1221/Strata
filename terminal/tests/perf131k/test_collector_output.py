"""Check artifacts from separately resource-gated CPU fixtures; never launch an engine."""

import importlib.util
import json
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("capture_trace", ROOT / "tools/perf131k/capture_trace.py")
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)
FIXTURES = ROOT / "_scratch/perf131k"
SERIES = os.environ.get("P01_FIXTURE_SERIES", "02")


class CollectorOutputTest(unittest.TestCase):
    def read(self, mode: str) -> tuple[dict, list[dict]]:
        path = FIXTURES / f"cpu-{mode}-{SERIES}.jsonl"
        return TRACE.analyze(path), [json.loads(line) for line in path.read_text().splitlines()]

    def test_off_has_no_output(self) -> None:
        self.assertFalse((FIXTURES / f"cpu-off-{SERIES}.jsonl").exists())

    def test_graph_lifetime(self) -> None:
        path = FIXTURES / f"cpu-lifecycle-{SERIES}.jsonl"
        spans = [json.loads(line) for line in path.read_text().splitlines()][1:-1]
        self.assertIsNotNone(spans[0]["graph_instance"])
        self.assertNotEqual(spans[0]["graph_instance"], spans[1]["graph_instance"])
        self.assertEqual(spans[0]["graph_handle_id"], spans[1]["graph_handle_id"])

    def test_real_collector_concurrency_and_context(self) -> None:
        result, rows = self.read("normal")
        spans = rows[1:-1]
        self.assertEqual(result["record_count"], 164)
        self.assertEqual(len({s["thread"] for s in spans}), 5)
        self.assertEqual(len({s["id"] for s in spans}), 164)
        self.assertEqual(spans[2]["parent"], 2)
        self.assertEqual(spans[2]["generation"], 1)
        self.assertEqual(spans[2]["token_start"], 4095)
        self.assertEqual(spans[2]["graph_replay"], 3)
        self.assertGreater(spans[2]["graph_instance"], 0)
        self.assertEqual(result["trace_status"], "COMPLETE_HOST_TRACE")

    def test_exact_buffer_limit_and_drops(self) -> None:
        result, _ = self.read("overflow")
        self.assertEqual(result["record_count"], 4096)
        self.assertEqual(result["counts"]["dropped"], 3)
        self.assertEqual(result["trace_status"], "INCOMPLETE")

    def test_exception_unwinds_scope(self) -> None:
        result, rows = self.read("unwind")
        self.assertEqual(result["counts"]["unfinished"], 0)
        self.assertEqual(rows[1]["end_ns"] - rows[1]["start_ns"], 10)

    def test_identity_overflow_remains_parseable_incomplete(self) -> None:
        result, _ = self.read("identities")
        self.assertGreater(result["counts"]["identity_overflow"], 0)
        self.assertEqual(result["trace_status"], "INCOMPLETE")

    def test_shutdown_with_live_scope_is_incomplete(self) -> None:
        result, _ = self.read("unfinished")
        self.assertEqual(result["counts"]["unfinished"], 1)
        self.assertEqual(result["trace_status"], "INCOMPLETE")

    def test_runtime_output_and_exclusive_rejection(self) -> None:
        result, _ = self.read("runtime")
        self.assertEqual(result["record_count"], 2)
        raw = (ROOT / f"_docs/perf131k/tests/cpu-runtime-repeat-{SERIES}.txt").read_text()
        self.assertIn("host trace unavailable", raw)


if __name__ == "__main__":
    unittest.main()
