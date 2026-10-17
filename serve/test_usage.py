"""serve/test_usage.py - the lifetime token statistics (serve/usage.py, the Monitor's Usage panel).

Off by default: a Usage(None) records nothing and costs nothing.  On: one small JSON file beside
the run config holds the counters, so the page can show lifetime and 1 day / 7 day / 30 day views.
"""
from __future__ import annotations

import json
import re
import tempfile
import time
import unittest
from pathlib import Path

from serve import usage


class UsageTest(unittest.TestCase):
    def make_usage(self, *args, **kwargs):
        u = usage.Usage(*args, **kwargs)
        self.addCleanup(u.close)
        return u

    def test_disabled_is_a_noop(self):
        u = self.make_usage(None)
        self.assertFalse(u.enabled)
        u.record(prompt_tokens=100, output_tokens=50, now=1000.0)        # must not raise or count
        self.assertEqual(u.lifetime["requests"], 0)
        v = u.view(now=1000.0)
        self.assertEqual(v, {"enabled": False, "note": v["note"]})       # the page shows the note

    def test_counts_a_request(self):
        u = self.make_usage(None)
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
        u = self.make_usage(None)
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
        u = self.make_usage(path)
        u.record(prompt_tokens=500, output_tokens=200, decode_ms=2000.0, now=5_000.0)
        u.close()
        again = self.make_usage(path)                                 # a fresh server, same file
        self.assertEqual(again.lifetime["output_tokens"], 200)
        self.assertEqual(again.lifetime["requests"], 1)
        again.record(output_tokens=5, now=5_100.0)
        self.assertEqual(again.view(now=5_100.0)["lifetime"]["output_tokens"], 205)

    def test_a_corrupt_file_starts_over(self):
        path = Path(tempfile.mkdtemp()) / "u.json"
        path.write_text("{not json", encoding="utf-8")
        u = self.make_usage(path)                                     # must not raise (the boot continues)
        self.assertTrue(u.enabled)
        self.assertEqual(u.lifetime["requests"], 0)
        u.record(output_tokens=7, now=100.0)
        self.assertEqual(u.view(now=100.0)["lifetime"]["output_tokens"], 7)

    def test_buckets_stay_bounded(self):
        u = self.make_usage(None)
        u.path = Path(tempfile.mkdtemp()) / "u.json"
        now = 1_700_000_000.0
        for i in range(200):                                      # 200 hours of requests
            u.record(output_tokens=1, now=now - i * usage.HOUR)
        u.flush()
        data = json.loads(u.path.read_text(encoding="utf-8"))
        self.assertLessEqual(len(data["hour"]), usage.KEEP["hour"])
        self.assertLessEqual(len(data["day"]), usage.KEEP["day"])

    def test_every_settled_request_is_saved_even_before_a_quiet_spell(self):
        path = Path(tempfile.mkdtemp()) / "u.json"
        u = self.make_usage(path)
        u.record(output_tokens=7, now=1000.0)
        u.record(output_tokens=9, now=1000.1)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["lifetime"]["output_tokens"], 16)
        self.assertEqual(data["lifetime"]["requests"], 2)

    def test_custom_range(self):
        u = self.make_usage(None)
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

    def test_the_panel_opens_on_seven_days_and_remembers_the_choice(self):
        """The Usage panel opens on the 7-day view; a pick is kept in the browser (localStorage)."""
        web = Path(__file__).parent / "web"
        js, html = (web / "app.js").read_text(encoding="utf-8"), (web / "index.html").read_text(encoding="utf-8")
        self.assertIn('store.get("usage.view", "")', js)                                  # read what the browser remembers
        self.assertIn('USAGE_VIEWS.includes(savedUsageView) ? savedUsageView : "7d"', js)  # nothing remembered: 7 days
        handler = js[js.index('".usage-range")) b.addEventListener("click"'):][:400]
        self.assertIn('store.set("usage.view", usageRange)', handler)                     # a click is remembered

        # a remembered value is only honoured if it is still a button on the page
        views = re.findall(r'"([a-z0-9]+)"', js.split("USAGE_VIEWS = [")[1].split("]")[0])
        buttons = re.findall(r'usage-range" data-range="([a-z0-9]+)"', html)
        self.assertEqual(sorted(views), sorted(buttons))


    def test_the_date_picker_uses_the_browsers_own_day(self):
        """A date input gets the browser's day, not UTC's, and an empty window says which kind it is."""
        js = (Path(__file__).parent / "web" / "app.js").read_text(encoding="utf-8")
        prefill = re.search(r"function prefillCustomRange\(\).*?\n\}", js, re.S)
        self.assertIsNotNone(prefill, "the custom range needs a starting window")
        body = prefill.group(0) if prefill else ""
        self.assertIn("localDateStr(", body)
        self.assertNotIn("toISOString", body)         # UTC's day is not the user's day
        self.assertIn("function localDateStr(", js)
        # a window the user did set, but that holds nothing, must not read as an unset one
        self.assertIn("Nothing in that window.", js)
        self.assertIn("Pick a start and end date above.", js)


    def test_a_wrongly_typed_counter_does_not_break_the_boot(self):
        """A hand-edited or half-written statistics file must not take the server down."""
        for label, body in [("a string", '{"lifetime": {"requests": "abc"}}'),
                            ("an infinity", '{"lifetime": {"requests": Infinity}}'),
                            ("a list", '{"lifetime": {"requests": [1, 2]}}'),
                            ("a bucket counter", '{"hour": [{"t": 1, "requests": "x"}]}'),
                            ("a bad since", '{"since": "yesterday"}'),
                            ("a negative count", '{"lifetime": {"output_tokens": -5}}')]:
            path = Path(tempfile.mkdtemp()) / "u.json"
            path.write_text(body, encoding="utf-8")
            u = self.make_usage(path, now=1_000.0)                    # must not raise
            self.assertTrue(u.enabled, label)
            self.assertGreaterEqual(u.lifetime["output_tokens"], 0, label)
            u.record(output_tokens=3, now=1_000.0)                # and it still counts from here
            self.assertEqual(u.view(now=1_000.0)["views"]["1d"]["total"]["output_tokens"], 3, label)

    def test_a_clock_that_steps_backwards_keeps_every_bucket(self):
        """An NTP step, a resume, or a manual change must not lose counts from the 1 day view."""
        u = self.make_usage(None)
        u.path = Path(tempfile.mkdtemp()) / "u.json"
        now = 1_700_000_000.0
        u.since = now - 10 * usage.DAY
        u.record(output_tokens=10, now=now)                       # this hour
        u.record(output_tokens=7, now=now - 2 * usage.HOUR)       # the clock steps back
        u.record(output_tokens=3, now=now)                        # and steps forward again
        v = u.view(now=now)
        self.assertEqual(v["views"]["1d"]["total"]["output_tokens"], 20)
        self.assertEqual(v["lifetime"]["output_tokens"], 20)
        r = u.view(now=now, custom=(now - 3 * usage.HOUR, now))   # the same window as a date range
        self.assertEqual(r["views"]["custom"]["total"]["output_tokens"], 20)
        u.flush()
        starts = [b["t"] for b in json.loads(u.path.read_text(encoding="utf-8"))["hour"]]
        self.assertEqual(starts, sorted(set(starts)))             # one row per bucket, in time order

    def test_two_rows_for_one_bucket_are_merged_when_read(self):
        """A file written by an older version may hold two rows for one bucket: they sum."""
        path = Path(tempfile.mkdtemp()) / "u.json"
        now = 1_700_000_000.0
        hour = usage._align(now, "hour")
        path.write_text(json.dumps(
            {"since": now - 10 * usage.DAY, "lifetime": {"output_tokens": 17},
             "hour": [{"t": hour, "output_tokens": 10}, {"t": hour, "output_tokens": 7}]}),
            encoding="utf-8")
        u = self.make_usage(path, now=now)
        self.assertEqual(u.lifetime["output_tokens"], 17)
        self.assertEqual(u.view(now=now)["views"]["1d"]["total"]["output_tokens"], 17)


if __name__ == "__main__":
    unittest.main()
