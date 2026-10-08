"""Offline tests for tools/analyze_prompt_cache_log.py.

    python -m unittest tools.test_analyze_prompt_cache_log
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_prompt_cache_log as ANALYZER  # noqa: E402


PREFIX = "strata serve: prompt cache:"


class Parsing(unittest.TestCase):
    def test_parses_arbitrary_field_order_and_absent_fields(self):
        row = ANALYZER.parse_line(
            f"2026-10-07 info {PREFIX} decision scan_ms=2.5 source=slot req=8 prompt=100 resume=75 ckpt=1",
            "a.log", 12)
        self.assertEqual(row, {"event": "decision", "file": "a.log", "line": 12, "scan_ms": 2.5,
                               "source": "slot", "req": 8, "prompt": 100, "resume": 75, "ckpt": 1})

    def test_parses_every_event_format(self):
        lines = [
            ("checkpoint req=1 kind=periodic tokens=200 save_ms=3.25 sync_ms=.25 copy_ms=2.5 "
             "stage_sync_ms=.1 stage_copy_ms=.2 alloc_ms=.4 gdn_ms=1.2 ple_ms=.3 index_ms=.6 "
             "gdn_bytes=128 ple_bytes=32 index_bytes=64 kept=2 evicted_kind=old evicted_tokens=90",
             "checkpoint"),
            ("park req=2 tokens=180 estimate_bytes=1000 fresh_estimate_bytes=900 retained_bytes=700 "
             "additional_bytes=200 held_bytes=800 save_ms=4.5 estimate_ms=.2 capture_ms=3 "
             "stage_sync_ms=.1 stage_capture_ms=.5 put_ms=.05 stored=1", "park"),
            ("restore req=3 tokens=170 source=parked restore_ms=6.75 bytes=4096", "restore"),
        ]
        for body, event in lines:
            with self.subTest(event=event):
                self.assertEqual(ANALYZER.parse_line(f"{PREFIX} {body}")["event"], event)

    def test_ignores_unrelated_unknown_and_malformed_lines(self):
        malformed = [
            "ordinary server output",
            f"{PREFIX} unknown req=1",
            f"{PREFIX} decision req=bad source=none",
            f"{PREFIX} park stored=2",
            f"{PREFIX} restore req=1 broken",
            f"{PREFIX} checkpoint future=value",
        ]
        self.assertEqual([ANALYZER.parse_line(line) for line in malformed], [None] * len(malformed))


class Analysis(unittest.TestCase):
    def records(self):
        bodies = [
            "decision req=1 prompt=100 resume=90 source=slot scan_ms=1.5",
            "decision read_from=20 prompt=100 req=2 scan_ms=8 source=none",
            "decision req=3 prompt=50 source=checkpoint scan_ms=3",
            "checkpoint tokens=80 req=2 save_ms=7 sync_ms=1 copy_ms=5 stage_sync_ms=.2 stage_copy_ms=.3 kind=full",
            "checkpoint save_ms=2 req=1",
            "park req=2 save_ms=9 estimate_ms=1 capture_ms=6 stage_sync_ms=.2 stage_capture_ms=1 put_ms=.3 stored=1",
            "restore restore_ms=11 req=3 source=park bytes=400 tokens=20",
        ]
        return [ANALYZER.parse_line(f"{PREFIX} {body}", "trace.log", n)
                for n, body in enumerate(bodies, 1)]

    def test_aggregates_reuse_and_ranks_hot_spots(self):
        report = ANALYZER.analyze(self.records(), ["trace.log"])
        self.assertEqual(report["summary"]["event_counts"],
                         {"decision": 3, "checkpoint": 2, "park": 1, "restore": 1})
        self.assertEqual(report["summary"]["reused"], 2)
        self.assertEqual(report["summary"]["misses"], 1)
        self.assertAlmostEqual(report["summary"]["reuse_rate"], 2 / 3)
        rereads = report["hot_spots"]["largest_rereads"]
        self.assertEqual([(row["req"], row["reread_tokens"], row["reread_basis"]) for row in rereads],
                         [(1, 10, "resume")])
        self.assertEqual(report["hot_spots"]["cold_reads"][0]["req"], 2)
        self.assertEqual(report["hot_spots"]["slow_scans"][0]["req"], 2)
        self.assertEqual(report["hot_spots"]["checkpoint_saves"][0]["req"], 2)
        self.assertEqual(report["hot_spots"]["park_saves"][0]["save_ms"], 9)
        self.assertEqual(report["hot_spots"]["restores"][0]["restore_ms"], 11)
        self.assertEqual(report["hot_spots"]["misses"][0]["req"], 2)
        phases = report["summary"]["phase_totals_ms"]
        self.assertEqual(phases["checkpoint_sync"], 1)
        self.assertEqual(phases["checkpoint_copy"], 5)
        self.assertEqual(phases["checkpoint_allocation"], 0)
        self.assertEqual(phases["park_capture"], 6)
        self.assertEqual(phases["park_put"], .3)

    def test_missing_source_is_unclassified_and_empty_rate_is_null(self):
        row = ANALYZER.parse_line(f"{PREFIX} decision req=9 prompt=10", "x", 1)
        summary = ANALYZER.analyze([row])["summary"]
        self.assertEqual(summary["unclassified"], 1)
        self.assertEqual(summary["reuse_rate"], 0)
        self.assertIsNone(ANALYZER.analyze([])["summary"]["reuse_rate"])

    def test_checkpoint_copy_detail_is_aggregated(self):
        row = ANALYZER.parse_line(f"{PREFIX} checkpoint req=1 alloc_ms=2 gdn_ms=3 ple_ms=4 "
                                  "index_ms=5 gdn_bytes=128 ple_bytes=32 index_bytes=16")
        phases = ANALYZER.analyze([row])["summary"]["phase_totals_ms"]
        self.assertEqual((phases["checkpoint_allocation"], phases["checkpoint_gdn"],
                          phases["checkpoint_ple"], phases["checkpoint_index"]), (2, 3, 4, 5))
        self.assertEqual((row["gdn_bytes"], row["ple_bytes"], row["index_bytes"]), (128, 32, 16))

    def test_text_names_each_hot_spot_section(self):
        text = ANALYZER.format_text(ANALYZER.analyze(self.records()))
        for heading in ("Largest partial rereads", "Largest cold reads", "Slow scans", "Checkpoint saves",
                        "Park saves", "Restores", "Checkpoint phases total", "Park phases total",
                        "Misses / source none"):
            self.assertIn(heading, text)


class Cli(unittest.TestCase):
    def test_json_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                f"noise\n{PREFIX} restore bytes=512 source=park tokens=64 req=4 restore_ms=2.25\n",
                encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "analyze_prompt_cache_log.py"), "--json", str(log)],
                check=True, capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["summary"]["event_counts"]["restore"], 1)
            self.assertEqual(report["hot_spots"]["restores"][0]["bytes"], 512)
            self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
