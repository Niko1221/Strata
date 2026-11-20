"""Deterministic accounting fixtures, not GPU or model performance evidence."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "capture_trace", os.environ.get("P01_ANALYZER_UNDER_TEST", ROOT / "tools/perf131k/capture_trace.py")
)
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


class AccountingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="p01-accounting-"))

    def span(self, identity: int = 1, **changes: object) -> dict:
        row = dict(
            type="span",
            id=identity,
            parent=0,
            generation=0,
            thread=1,
            stage="verify_host",
            start_ns=10,
            end_ns=100,
            token_start=0,
            token_end=1,
            layer=-1,
            stream=0,
            graph_handle_id=0,
            graph_instance=None,
            graph_replay=0,
            logical_read_requested_bytes=-1,
            copy_requested_bytes=-1,
            gpu_elapsed_ns=None,
            physical_read_bytes=None,
            event_generation=None,
            allocation_id=None,
        )
        row.update(changes)
        return row

    def trace(self, spans: list[dict], **footer_changes: int) -> Path:
        footer = dict(
            type="footer",
            records=len(spans),
            dropped=0,
            unfinished=0,
            identity_overflow=0,
            diagnostic_failures=0,
            unregistered_graphs=0,
        )
        footer.update(footer_changes)
        rows = [dict(type="header", schema=1, source="host_only", clock="steady_ns", capacity=4096), *spans, footer]
        path = self.directory / "trace.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_parent_child_and_parallel_wait_not_added(self) -> None:
        spans = [
            {"start_ns": 0, "end_ns": 100},
            {"start_ns": 10, "end_ns": 70},
            {"start_ns": 20, "end_ns": 90},
        ]
        self.assertEqual(TRACE.summarize({"spans": spans})["host_covered_ns"], 100)

    def test_unknown_gpu_and_physical_not_zero_or_bandwidth(self) -> None:
        result = TRACE.analyze(self.trace([self.span(logical_read_requested_bytes=4096)]))
        self.assertIsNone(result["physical_read_bytes"])
        self.assertIsNone(result["gpu_elapsed_ns"])
        self.assertIsNone(result["nvme_bandwidth_bytes_per_second"])
        self.assertEqual(result["performance_status"], "NOT_MEASURED")

    def test_cache_hit_cannot_forge_physical_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot establish physical"):
            TRACE.analyze(self.trace([self.span(physical_read_bytes=0)]))

    def test_logical_bytes_cannot_be_relabelled_physical(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot establish physical"):
            TRACE.analyze(self.trace([self.span(logical_read_requested_bytes=4096, physical_read_bytes=4096)]))

    def test_host_wait_cannot_be_relabelled_gpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot establish gpu"):
            TRACE.analyze(self.trace([self.span(gpu_elapsed_ns=90)]))

    def test_dropped_is_incomplete(self) -> None:
        self.assertEqual(TRACE.analyze(self.trace([self.span()], dropped=2))["trace_status"], "INCOMPLETE")

    def test_request_owns_its_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "generation"):
            TRACE.analyze(self.trace([self.span(stage="request", generation=0)]))

    def test_generation_thread_and_interval(self) -> None:
        request = self.span(stage="request", generation=1)
        for changes in ({"thread": 2}, {"end_ns": 101}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "generation"):
                TRACE.analyze(self.trace([request, self.span(2, generation=1, **changes)]))

    def test_malformed_stage_is_validation_error(self) -> None:
        with self.assertRaises(ValueError):
            TRACE.analyze(self.trace([self.span(stage=[])]))

    def test_diagnostic_failure_is_incomplete(self) -> None:
        self.assertEqual(TRACE.analyze(self.trace([self.span()], diagnostic_failures=1))["trace_status"], "INCOMPLETE")

    def test_enqueue_and_wait_remain_host_only(self) -> None:
        rows = [
            self.span(1, stage="graph_enqueue_host", stream=1, graph_handle_id=2, graph_instance=1, graph_replay=1),
            self.span(2, stage="host_wait"),
        ]
        result = TRACE.analyze(self.trace(rows))
        self.assertEqual(result["host_covered_ns"], 90)
        self.assertIsNone(result["gpu_elapsed_ns"])

    def test_unfinished_parent_remains_incomplete(self) -> None:
        self.assertEqual(
            TRACE.analyze(self.trace([self.span(2, parent=1)], unfinished=1))["trace_status"], "INCOMPLETE"
        )

    def test_parent_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "parent interval"):
            TRACE.analyze(self.trace([self.span(), self.span(2, parent=1, end_ns=101)]))

    def test_disjoint_union_keeps_gaps(self) -> None:
        self.assertEqual(
            TRACE.analyze(self.trace([self.span(end_ns=20), self.span(2, start_ns=30, end_ns=40)]))["host_covered_ns"],
            20,
        )

    def test_graph_replay_reuse_rejected(self) -> None:
        rows = [
            self.span(i, stage="graph_launch_issue", graph_handle_id=1, graph_instance=1, graph_replay=1, stream=2)
            for i in (1, 2)
        ]
        with self.assertRaisesRegex(ValueError, "reused"):
            TRACE.analyze(self.trace(rows))

    def test_unique_replays_allowed(self) -> None:
        rows = [
            self.span(i, stage="graph_launch_issue", graph_handle_id=1, graph_instance=1, graph_replay=i, stream=2)
            for i in (1, 2)
        ]
        self.assertEqual(TRACE.analyze(self.trace(rows))["trace_status"], "COMPLETE_HOST_TRACE")

    def test_footer_required(self) -> None:
        path = self.trace([self.span()])
        path.write_text("\n".join(path.read_text().splitlines()[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "footer"):
            TRACE.analyze(path)

    def test_private_or_unrecognised_fields_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fields"):
            TRACE.analyze(self.trace([self.span(prompt="must not be accepted")]))

    def test_negative_or_reversed_time_rejected(self) -> None:
        for changes in ({"start_ns": -1}, {"end_ns": 9}, {"start_ns": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TRACE.analyze(self.trace([self.span(**changes)]))

    def test_duplicate_json_field_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            json.loads('{"id": 1, "id": 2}', object_pairs_hook=TRACE.unique_object)

    def test_oversized_line_rejected(self) -> None:
        path = self.directory / "large.jsonl"
        path.write_bytes(b" " * 2049)
        with self.assertRaisesRegex(ValueError, "limit"):
            TRACE.analyze(path)


if __name__ == "__main__":
    unittest.main()
