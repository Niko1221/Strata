"""serve/test_usage.py - the lifetime token statistics (serve/usage.py, the Monitor's Usage panel).

Off by default: a Usage(None) records nothing and costs nothing.  On: one small JSON file beside
the run config holds the counters, so the page can show lifetime and 1 day / 7 day / 30 day views.
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from serve import usage


class UsageTest(unittest.TestCase):
    def test_disabled_is_a_noop(self):
        u = usage.Usage(None)
        self.assertFalse(u.enabled)
        u.record(prompt_tokens=100, output_tokens=50, now=1000.0)        # must not raise or count
        self.assertEqual(u.lifetime["requests"], 0)
        v = u.view(now=1000.0)
        self.assertEqual(v, {"enabled": False, "note": v["note"]})       # the page shows the note

    def test_counts_a_request(self):
        u = usage.Usage(None)
        u.path = Path(tempfile.mkdtemp()) / "u.json"
        u.record(prompt_tokens=1000, reused=800, output_tokens=250, prompt_ms=500.0, decode_ms=2500.0,
                 drafts_offered=300, drafts_accepted=210, now=10_000.0)
        v = u.view(now=10_000.0)
        self.assertTrue(v["enabled"])
        life = v["lifetime"]
        self.assertEqual((life["requests"], life["prompt_tokens"], life["reused"]), (1, 1000, 800))
        self.assertEqual((life["output_tokens"], life["drafts_accepted"]), (250, 210))
        # the 1 day view includes it (bucketed to the hour)
        self.assertEqual(v["views"]["1d"]["total"]["output_tokens"], 250)

    def test_views_roll_old_data_out(self):
        u = usage.Usage(None)
        u.path = Path(tempfile.mkdtemp()) / "u.json"
        now = 1_700_000_000.0
        u.record(output_tokens=100, now=now - 40 * usage.DAY)     # older than the 30 day view keeps
        u.record(output_tokens=10, now=now - 3 * usage.DAY)       # inside 7 day
        u.record(output_tokens=1, now=now - 2 * usage.HOUR)       # inside 1 day
        v = u.view(now=now)
        self.assertEqual(v["views"]["1d"]["total"]["output_tokens"], 1)
        self.assertEqual(v["views"]["7d"]["total"]["output_tokens"], 11)
        self.assertEqual(v["views"]["30d"]["total"]["output_tokens"], 11)   # the 40-day-old one fell out
        self.assertEqual(v["lifetime"]["output_tokens"], 111)               # lifetime keeps everything

    def test_survives_a_restart(self):
        path = Path(tempfile.mkdtemp()) / "u.json"
        u = usage.Usage(path)
        u.record(prompt_tokens=500, output_tokens=200, decode_ms=2000.0, now=5_000.0)
        u.flush()
        again = usage.Usage(path)                                 # a fresh server, same file
        self.assertEqual(again.lifetime["output_tokens"], 200)
        self.assertEqual(again.lifetime["requests"], 1)
        again.record(output_tokens=5, now=5_100.0)
        self.assertEqual(again.view(now=5_100.0)["lifetime"]["output_tokens"], 205)

    def test_a_corrupt_file_starts_over(self):
        path = Path(tempfile.mkdtemp()) / "u.json"
        path.write_text("{not json", encoding="utf-8")
        u = usage.Usage(path)                                     # must not raise (the boot continues)
        self.assertTrue(u.enabled)
        self.assertEqual(u.lifetime["requests"], 0)
        u.record(output_tokens=7, now=100.0)
        self.assertEqual(u.view(now=100.0)["lifetime"]["output_tokens"], 7)

    def test_buckets_stay_bounded(self):
        u = usage.Usage(None)
        u.path = Path(tempfile.mkdtemp()) / "u.json"
        now = 1_700_000_000.0
        for i in range(200):                                      # 200 hours of requests
            u.record(output_tokens=1, now=now - i * usage.HOUR)
        u.flush()
        data = json.loads(u.path.read_text(encoding="utf-8"))
        self.assertLessEqual(len(data["hour"]), usage.KEEP["hour"])
        self.assertLessEqual(len(data["day"]), usage.KEEP["day"])

    def test_write_is_throttled(self):
        path = Path(tempfile.mkdtemp()) / "u.json"
        u = usage.Usage(path)
        u.record(output_tokens=1, now=1000.0)                     # first write
        self.assertTrue(path.exists())
        mtime = path.stat().st_mtime_ns
        u.record(output_tokens=1, now=1000.0 + usage.WRITE_EVERY_S / 2)   # too soon: no write
        self.assertEqual(path.stat().st_mtime_ns, mtime)
        u.record(output_tokens=1, now=1000.0 + usage.WRITE_EVERY_S + 1)   # a minute on: written
        self.assertNotEqual(path.stat().st_mtime_ns, mtime)

    def test_custom_range(self):
        u = usage.Usage(None)
        u.path = Path(tempfile.mkdtemp()) / "u.json"
        now = 1_700_000_000.0
        u.since = now - 45 * usage.DAY                           # the store started 45 days before `now`
        u.record(output_tokens=100, now=now - 40 * usage.DAY)     # older than anything kept
        u.record(output_tokens=10, now=now - 2 * usage.DAY)       # a past day
        u.record(output_tokens=1, now=now - 2 * usage.HOUR)       # this hour
        v = u.view(now=now)
        self.assertNotIn("custom", v["views"])                    # not asked for: not computed
        v = u.view(now=now, custom=(now - 3 * usage.DAY, now))   # a date range, hour resolution
        self.assertEqual(v["views"]["custom"]["total"]["output_tokens"], 11)
        v = u.view(now=now, custom=(now - 3 * usage.HOUR, now))  # a short window sees hour buckets
        self.assertEqual(v["views"]["custom"]["total"]["output_tokens"], 1)
        v = u.view(now=now, custom=(now - 90 * usage.DAY, now))  # clamped to what the buckets keep
        self.assertGreaterEqual(v["views"]["custom"]["from"],
                                now - usage.KEEP["day"] * usage.DAY - usage.HOUR)  # floor aligns down
        self.assertLess(v["views"]["custom"]["from"], now - 30 * usage.DAY)        # but not by much
        self.assertEqual(v["views"]["custom"]["total"]["output_tokens"], 11)
        v = u.view(now=now, custom=(now - usage.HOUR, now - usage.HOUR))  # empty window: zero, not missing
        self.assertEqual(v["views"]["custom"]["total"]["output_tokens"], 0)
        v = u.view(now=now, custom=(now - 90 * usage.DAY, now - 60 * usage.DAY))  # entirely ancient
        self.assertNotIn("custom", v["views"])

    def test_speeds(self):
        row = usage.totals_speed({"prompt_tokens": 1000, "reused": 800, "output_tokens": 250,
                                  "prompt_ms": 500.0, "decode_ms": 2500.0})
        self.assertEqual(row["read"], 200)
        self.assertEqual(row["decode_tok_s"], 100.0)              # 250 / 2.5 s
        self.assertEqual(row["prompt_tok_s"], 400.0)              # 200 read / 0.5 s
        empty = usage.totals_speed({})
        self.assertIsNone(empty["decode_tok_s"])
        self.assertIsNone(empty["prompt_tok_s"])


if __name__ == "__main__":
    unittest.main()
