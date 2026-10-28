"""The monitor's Overview (view 1): numbers from /metrics, the drawing, glyph fallbacks, and the monitor showing it."""

from __future__ import annotations

import asyncio
import io
import time
import unittest

from rich.console import Console

from strata_cli import overview
from strata_cli.monitor import MonitorApp

GIB = 2 ** 30


def metrics(state="reading", **over):
    now = time.time()
    hist = lambda base, amp: [base + amp * ((i * 7) % 5) / 4 for i in range(60)]     # noqa: E731
    m = {"time": now,
         "engine": {"model": "IQ3_S", "max_context": 131072, "expert_slots": 1644, "experts": 5120, "expert_cache_mib": 3280},
         "live": {"state": state, "queued": 0, "prompt_read": 35610 if state == "reading" else None, "prompt_total": 36742 if state == "reading" else None,
                  "prompt_tokens": 36742, "generated": 120 if state == "generating" else 0, "max_tokens": 4096,
                  "elapsed_s": 46.0 if state in ("reading", "generating") else None,
                  "tok_s": 50.4 if state == "generating" else None, "prefill_tok_s_mean": 820.0},
         "hardware": {"gpu_util": 88, "gpu_mem_used": 11.2 * GIB, "gpu_mem_total": 12 * GIB, "gpu_temp": 54, "gpu_power": 114, "gpu_power_limit": 250,
                      "gpu_pcie_rx_mb": 18237, "gpu_pcie_gen": 4, "gpu_pcie_width": 16, "cpu": 14, "ram_used": 58.5 * GIB, "ram_total": 63 * GIB,
                      "disk_read_mb": 0.1, "disk_write_mb": 0.0},
         "hardware_static": {"gpu_name": "RTX 5070", "gpu_count": 1, "cores": 6, "threads": 12},
         "history": {"tok_s": hist(40, 10), "gpu_util": hist(80, 10), "gpu_mem_used": [11.2 * GIB] * 60, "gpu_temp": [54] * 60,
                     "gpu_power": hist(110, 10), "gpu_pcie_rx_mb": hist(10000, 9000), "cpu": hist(10, 8), "disk_read_mb": hist(0, 3),
                     "ram_used": [58.5 * GIB + i * (GIB >> 6) for i in range(60)]},      # telemetry keeps a minute of ram_used too
         "requests": [{"id": "a", "time": now - 5, "duration_s": 429.8, "finish_reason": "stop", "prompt_tokens": 17183, "reused": 0,
                       "prompt_read": 17183, "prompt_ms": 21478.7, "output_tokens": 19511, "decode_ms": 386000},
                      {"id": "b", "time": now - 9, "duration_s": 4.0, "finish_reason": "stop", "prompt_tokens": 626, "reused": 0,
                       "prompt_read": 626, "prompt_ms": 760.0, "output_tokens": 11, "decode_ms": 410}]}
    m.update(over)
    return m


def draw(model, width=110, height=30, **kw) -> str:
    c = Console(width=width + 2, record=True, file=io.StringIO(), force_terminal=False, legacy_windows=False)
    c.print(overview.render(model, width=width, height=height, connection="http://127.0.0.1:8080", **kw))
    return c.export_text()


class Build(unittest.TestCase):
    def test_reading_state_and_progress(self):
        m = overview.build(metrics("reading"))
        self.assertEqual(m["state"], "Reading")
        self.assertEqual(m["progress"], (35610.0, 36742.0))
        self.assertEqual(m["context"], (36742.0, 131072.0))
        self.assertEqual(m["speed"][:2], (820.0, "reading the prompt"))

    def test_generating_uses_the_live_rate_and_prompt_plus_generated_for_context(self):
        m = overview.build(metrics("generating"))
        self.assertEqual(m["state"], "Generating")
        self.assertEqual(m["speed"][:2], (50.4, "now"))
        self.assertEqual(m["context"][0], 36742 + 120)
        self.assertEqual(m["progress"], (120.0, 4096.0))

    def test_idle_shows_the_last_request(self):
        m = overview.build(metrics("idle"))
        self.assertEqual(m["state"], "Idle")
        self.assertEqual(m["speed"][1], "last request")
        self.assertAlmostEqual(m["speed"][0], 50.5, places=1)               # 19511 tokens in 386 s
        self.assertEqual(m["context"][0], 17183 + 19511)
        self.assertEqual(m["progress"], (None, None))

    def test_queued_and_error(self):
        self.assertEqual(overview.build(metrics("idle", live={"state": "idle", "queued": 3}))["state"], "Queued")
        self.assertEqual(overview.build(metrics("idle", engine={"model": "x", "error": "boom"}))["state"], "Error")

    def test_hardware_in_gb(self):
        m = overview.build(metrics())
        self.assertAlmostEqual(m["gpu"]["vram_used"], 11.2)
        self.assertEqual(m["gpu"]["vram_total"], 12.0)
        self.assertAlmostEqual(m["ram"][0], 58.5)
        self.assertEqual(m["experts"], {"slots": 1644.0, "total": 5120.0, "gb": 3280 / 1024.0})

    def test_missing_numbers_stay_missing(self):
        m = overview.build({"live": {"state": "idle"}, "engine": {}, "hardware": {}, "requests": []})
        self.assertIsNone(m["gpu"]["util"])
        self.assertIsNone(m["speed"][0])
        text = draw(m)
        self.assertIn("-- tok/s", text)
        self.assertIn("No requests yet", text)
        self.assertNotIn("0 %", text.replace("-- %", ""))                    # no invented zero

    def test_prefill_speed_is_the_read_tokens_over_the_read_time(self):
        self.assertAlmostEqual(overview.build(metrics())["recent"][0]["prefill"], 800.0, places=1)     # 17,183 read in 21.5 s
        # an older server that sends no prompt timing at all: no invented number
        self.assertIsNone(overview.build(metrics(requests=[{"id": "c", "time": time.time(), "prompt_tokens": 626,
                                                            "output_tokens": 11, "decode_ms": 410}]))["recent"][0]["prefill"])
        # a prompt the server reused is not read again: only the read part counts against the time
        self.assertAlmostEqual(overview.build(metrics(requests=[{"id": "d", "time": time.time(), "prompt_tokens": 1000,
                                                                 "reused": 400, "prompt_ms": 1000.0}]))["recent"][0]["prefill"], 600.0, places=1)

    def test_the_request_running_now_is_an_active_row(self):
        """The request the engine is working on is a row of its own: Read or Gen, its own tokens, the live rate as its speed, and no
        invented `reused` (the server only reports that when the prompt has been read)."""
        reading, generating = overview.build(metrics("reading"))["active"], overview.build(metrics("generating"))["active"]
        self.assertEqual([r["state"] for r in reading], ["Read"])
        self.assertEqual((reading[0]["prompt"], reading[0]["output"], reading[0]["reused"]), (36742.0, None, None))
        self.assertAlmostEqual(reading[0]["prefill"], 820.0, places=1)                     # the server's own prompt reading mean
        self.assertIsNone(reading[0]["tps"])
        self.assertEqual([r["state"] for r in generating], ["Gen"])
        self.assertEqual((generating[0]["output"], generating[0]["tps"], generating[0]["duration"]), (120.0, 50.4, 46.0))
        self.assertEqual(overview.build(metrics("idle"))["active"], [])                    # idle: nothing to show on top
        # an idle engine with a request waiting: a count on the state line, not a row
        self.assertEqual(overview.build(metrics("idle", live={"state": "idle", "queued": 2}))["active"], [])

    def test_parallel_slots_give_one_active_row_each(self):
        """With --parallel the server lists every occupied slot, so each running one gets its own row; an idle slot gets none, and a
        slot's own reading speed beats the engine-wide mean."""
        slots = [{"slot": 0, "state": "reading", "prompt_tokens": 4000, "prompt_read": 2000, "generated": 0, "elapsed_s": 4.0, "tok_s": None},
                 {"slot": 1, "state": "decoding", "prompt_tokens": 900, "prompt_read": None, "generated": 55, "elapsed_s": 3.0, "tok_s": 41.0},
                 {"slot": 2, "state": "idle", "held_tokens": 1200}]
        rows = overview.build(metrics("reading", live={"state": "reading", "queued": 0, "slots": slots}))["active"]
        self.assertEqual([r["state"] for r in rows], ["Read", "Gen"])
        self.assertAlmostEqual(rows[0]["prefill"], 500.0, places=1)                        # 2,000 read in 4 s, not the 820 mean
        self.assertEqual((rows[1]["tps"], rows[1]["prefill"]), (41.0, None))

    def test_a_request_that_ended_badly_is_the_error_row(self):
        """The server's `finish` says how a request ended.  "error" and "timeout" are the ones whose reply never came, and they are the
        rows drawn in red; a client that walked away ("cancel", "disconnect") is the amber Stopped."""
        states = {f: overview.build(metrics(requests=[{"id": "x", "time": time.time(), "finish": f}]))["recent"][0]["state"]
                  for f in ("stop", "length", "tool_calls", "error", "timeout", "cancel", "disconnect")}
        self.assertEqual(states, {"stop": "Done", "length": "Done", "tool_calls": "Done", "error": "Error", "timeout": "Error",
                                  "cancel": "Stopped", "disconnect": "Stopped"})

    def test_a_server_without_metrics_gives_none(self):
        self.assertIsNone(overview.build(None))
        self.assertIsNone(overview.build({"engine": {}}))


class Draw(unittest.TestCase):
    def test_the_approved_layout(self):
        text = draw(overview.build(metrics("reading")), connect=True)
        for part in ("STRATA  Overview", "IQ3_S 128K", "Reading prompt · 97%", "[LIVE]", "[1 Overview]", "7 Connect",
                     "Model state", "[Reading]", "35,610 / 36,742 tokens · 97%",
                     "Speed", "GPU load", "Unified Memory", "GPU temp", "Power", "PCIe", "CPU", "Disk read",
                     "88 %", "RTX 5070", "58.5 / 63 GB", "1,644 - 3.2 GB", "54 °C", "114 W", "of 250 W limit", "Gen4 x16",
                     "to GPU 18,237 MB/s", "6 cores · 12 threads", "0.1 MB/s", "write 0.0 MB/s",
                     "Context fill", "36K / 128K", "KV Cache Hit", "0 / 17K", "System RAM", "58.5 / 63 GB",
                     "Recent requests", "TIME", "STATUS", "PROMPT", "REUSED", "OUTPUT", "READ t/s", "GEN t/s", "DURATION",
                     "17,183", "19,511", "800.0", "823.7", "50.5", "429.8 s",
                     "1-7 views", "b open in browser", "q quit"):
            self.assertIn(part, text, part)
        self.assertRegex(text, r"\d\d:\d\d:\d\d")            # the request times to the second

    def test_the_error_chip_appears_only_with_a_real_error(self):
        """The state line lists the four states a healthy model passes through; the red [Error] only shows when the model has one."""
        healthy = draw(overview.build(metrics("reading")))
        self.assertNotIn("Error", healthy, healthy)
        broken = draw(overview.build(metrics("idle", engine={"model": "x", "error": "the engine died"})))
        self.assertIn("[Error]", broken)
        self.assertIn("Model unavailable", broken)

    def test_the_request_running_now_is_the_top_row(self):
        """One active row on top (its status a Read or a Gen that pulses), the finished ones under it newest first, times to the second."""
        m = overview.build(metrics("generating"))
        text = draw(m, height=40)
        active_stamp = time.strftime("%H:%M:%S", time.localtime(m["active"][0]["time"]))
        finished_stamp = time.strftime("%H:%M:%S", time.localtime(m["recent"][0]["time"]))
        self.assertIn(active_stamp, text, active_stamp)
        self.assertLess(text.index(active_stamp), text.index(finished_stamp))     # active above the finished
        self.assertLess(text.index(finished_stamp), text.index(time.strftime("%H:%M:%S", time.localtime(m["recent"][1]["time"]))))
        self.assertIn("820.0", text)                                              # the running request's own reading mean

    def test_the_active_status_pulses(self):
        """A running request's status alternates bright and dim second by second, so it reads as blinking in any terminal."""
        self.assertTrue(overview.blink(10.0))
        self.assertFalse(overview.blink(11.0))
        self.assertTrue(overview.blink(12.0))
        for t in (10.0, 11.0):
            self.assertIn("Gen", draw(overview.build(metrics("generating")), height=40, now=t))

    def test_at_most_ten_requests_are_listed(self):
        """Ten rows at most, and the newest ones: the server sends them newest first."""
        many = metrics("idle")
        many["requests"] = [{"id": str(i), "time": time.time() - i, "duration_s": 5.0 + i, "finish": "stop", "prompt_tokens": 100 + i,
                             "reused": 0, "prompt_read": 100 + i, "prompt_ms": 200.0, "output_tokens": 20 + i, "decode_ms": 500.0}
                            for i in range(14)]
        m = overview.build(many)
        self.assertEqual(len(m["recent"]), 10)
        text = draw(m, height=40)
        self.assertEqual(text.count("Done"), 10, text)

    def test_without_connect_the_hint_says_six_views(self):
        text = draw(overview.build(metrics()), connect=False)
        self.assertIn("1-6 views", text)
        self.assertNotIn("7 Connect", text)

    def test_ascii_mode_has_no_glyphs_the_classic_windows_console_font_lacks(self):
        text = draw(overview.build(metrics()), glyphs="ascii", color=False)
        for bad in "▁▂▃▄▅▆▇█░╭╮╰╯✓●":
            self.assertNotIn(bad, text, bad)
        self.assertIn("#", text)
        self.assertIn("54 C", text)

    def test_fits_the_width_it_is_given(self):
        for w in (80, 100, 120):
            text = draw(overview.build(metrics()), width=w - 2)
            self.assertTrue(all(len(line) <= w for line in text.splitlines()), w)

    def test_a_narrow_terminal_keeps_both_speeds(self):
        """Below 100 columns the request table keeps both speeds and the full numbers, and gives up the duration."""
        text = draw(overview.build(metrics()), width=80)
        for part in ("READ t/s", "GEN t/s", "17,183", "800.0", "50.5"):
            self.assertIn(part, text, part)
        for cut in ("READ…", "GEN…", "17,1…", "429.8 s"):
            self.assertNotIn(cut, text, cut)

    def test_a_hundred_columns_keep_every_column_of_the_table(self):
        """The monitor draws the Overview about 100 columns wide: TIME to the second, both speeds and the duration keep their column, the
        bars give the room back, and their captions go short rather than be cut."""
        text = draw(overview.build(metrics()), width=100)
        for part in ("READ t/s", "GEN t/s", "DURATION", "429.8 s", "KV hit 0 / 17K", "RAM 58.5 / 63 GB"):
            self.assertIn(part, text, part)
        for cut in ("READ…", "GEN…", "DURATI…", "429.8…", "KV Cache Hit", "36K /…"):
            self.assertNotIn(cut, text, cut)

    def test_kv_cache_hit_is_the_newest_request_that_reports_reuse(self):
        """The KV Cache Hit bar carries the newest request the server gave a reuse count for: the prompt tokens the cache
        already held over that request's context.  A request that reports no `reused` is passed over, and none at all is
        "-- / --" - never a made-up 0%."""
        text = draw(overview.build(metrics(requests=[
            {"id": "a", "time": time.time(), "prompt_tokens": 1000, "reused": 600, "prompt_read": 400, "prompt_ms": 1000.0,
             "output_tokens": 5, "decode_ms": 100},
            {"id": "b", "time": time.time() - 5, "prompt_tokens": 500, "reused": 500, "prompt_read": 0, "prompt_ms": 100.0,
             "output_tokens": 5, "decode_ms": 100}])))
        self.assertIn("KV Cache Hit", text, text)
        self.assertIn("600 / 1000  60%", text, text)               # newest first: a's 600 of 1,000, not b's 500 of 500
        bare = draw(overview.build(metrics(requests=[{"id": "c", "time": time.time(), "prompt_tokens": 900,
                                                      "output_tokens": 5, "decode_ms": 100}])))
        self.assertIn("KV Cache Hit    -- / --", bare, bare)       # no reuse reported: no invented rate

    def test_spark(self):
        self.assertEqual(len(overview.spark([1, 2, 3], 8, "unicode")), 8)
        self.assertEqual(overview.spark([None, None], 4, "unicode"), "    ")
        self.assertEqual(overview.spark([0, 100], 2, "unicode", lo=0, hi=100), "▁█")
        self.assertEqual(overview.spark([0, 100], 2, "ascii", lo=0, hi=100), ".@")


class Source:
    root = "http://127.0.0.1:8095"

    def __init__(self, raw):
        self.raw = raw
        self.base_url = self.root

    def get(self, path):
        if path == "/metrics":
            return self.raw
        raise AssertionError(path)

    def snapshot(self):
        return {"source": "fixture", "sampled_at": time.time(), "schema_version": 1, "instance": {"id": "x"},
                "model": {"id": "IQ3_S", "loaded": True}, "work": {}, "rates": {}, "devices": [], "recent": [], "totals": {},
                "topology": {}, "quality": {}}


class InTheMonitor(unittest.TestCase):
    def test_view_one_is_the_new_overview_and_the_others_stay(self):
        async def go():
            app = MonitorApp(Source(metrics("generating")), None, interval=30)
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.5)
                shown = "\n".join(s.text for s in app.screen._compositor.render_strips())
                self.assertIn("Model state", shown)
                self.assertIn("Generating", shown)
                await pilot.press("2")
                await pilot.pause(0.2)
                shown = "\n".join(s.text for s in app.screen._compositor.render_strips())
                self.assertNotIn("Model state", shown)
                self.assertIn("Serving", shown)
        asyncio.run(go())

    def test_an_older_server_without_metrics_keeps_the_earlier_overview(self):
        class Old(Source):
            def get(self, path):
                from strata_cli.client import ClientError
                raise ClientError("not found", http_status=404)

        async def go():
            app = MonitorApp(Old(None), None, interval=30)
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.5)
                shown = "\n".join(s.text for s in app.screen._compositor.render_strips())
                self.assertNotIn("Model state", shown)
                self.assertIn("MONITOR", shown)
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
