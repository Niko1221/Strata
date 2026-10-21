"""Tests for tools/window_trace.py: reads the engine's STRATA_WINDOW_TRACE layout.  No binary, no GPU.

    python -m unittest tools.test_window_trace
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_trace as W  # noqa: E402


def record(req, win, total):
    ints = [req, win, 3, 2, 1, 0]
    doubles = [win * 50.0, total, 0.0, 1.0, 0.5, 0.0, total - 4.0, 0.03, 2.5, 0.0, 18.0, 30.0, 11.0]
    counts = [690, 360, 0, 0, 270, 18, 22_000_000, 6, -1, 23]
    return struct.pack(W.FMT, *ints, 1234 + win, *doubles, *counts)


class Layout(unittest.TestCase):
    def test_record_size_matches_the_engine(self):
        self.assertEqual(struct.calcsize(W.FMT), 216)   # static_assert in src/program/generate.cpp
        self.assertEqual(len(W.FIELDS), 30)

    def test_reads_records_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 't.wtr'
            p.write_bytes(W.MAGIC + struct.pack('<I', 216) + record(0, 0, 50.0) + record(0, 1, 60.0) + record(1, 0, 70.0))
            rows = W.read(p)
        self.assertEqual([(r['req'], r['win'], r['total']) for r in rows], [(0, 0, 50.0), (0, 1, 60.0), (1, 0, 70.0)])
        self.assertEqual(rows[1]['pos'], 1235)
        self.assertEqual(rows[0]['file_reads'], 18)
        self.assertEqual(rows[0]['ram_staged'], -1)

    def test_a_partial_last_record_is_left_out(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 't.wtr'
            p.write_bytes(W.MAGIC + struct.pack('<I', 216) + record(0, 0, 50.0) + record(0, 1, 60.0)[:100])
            self.assertEqual(len(W.read(p)), 1)

    def test_refuses_another_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 't.wtr'
            p.write_bytes(b'not a trace file at all')
            with self.assertRaises(SystemExit):
                W.read(p)


if __name__ == '__main__':
    unittest.main()
