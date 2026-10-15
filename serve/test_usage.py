"""Persistent usage units, measured coverage and cache/decode accounting without hardware."""
import json
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service
from serve.test_request_accounting import ClockedThinkingEngine
from serve.usage import UsageLedger


class UsageAccounting(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "usage.json"
        self.at = 0.0
        self.wall = datetime(2024, 1, 1, 12).timestamp()
        self.energy = 0
        self.power = 120.0
        self.counter = True
        self.ledger = UsageLedger(self.path, "test", self.sample, lambda: self.at, lambda: self.wall)

    def sample(self):
        return {"power_w": self.power, "energy_mj": self.energy if self.counter else None}

    def tick(self, seconds, watts=120):
        self.at += seconds
        self.wall += seconds
        self.energy += watts * seconds * 1000

    def test_counter_units_and_electricity_cost(self):
        self.ledger.configure({"electricity_per_kwh": 0.2})
        self.ledger.begin("request")
        self.tick(3600)
        self.ledger.end("request")
        row = self.ledger.snapshot()
        self.assertAlmostEqual(row["energy_kwh"], 0.12)
        self.assertAlmostEqual(row["electricity_cost"], 0.024)
        self.assertEqual(row["measured_s"], 3600)
        self.assertEqual(row["energy_method"], "energy_counter")

    def test_idle_gaps_and_overlapping_requests_count_only_the_active_union(self):
        self.tick(3600, watts=20)
        self.ledger.begin("first")
        self.tick(10)
        self.ledger.begin("second")
        self.tick(10)
        self.ledger.end("first")
        self.tick(10)
        self.ledger.end("second")
        self.tick(7200, watts=20)
        row = self.ledger.snapshot()
        self.assertEqual(row["active_s"], 30)
        self.assertAlmostEqual(row["energy_kwh"], 0.001)
        self.assertFalse(row["active"])
        self.assertIsNone(row["power_w"])

    def test_counter_reset_uses_power_fallback(self):
        self.energy = 9_000_000
        self.ledger.begin("request")
        self.tick(30)
        self.energy = 500
        self.ledger.end("request")
        self.assertAlmostEqual(self.ledger.snapshot()["energy_kwh"], 0.001)
        self.assertEqual(self.ledger.snapshot()["energy_method"], "sampled_power")

    def test_missing_readings_are_unmeasured_not_zero_energy(self):
        self.counter = False
        self.power = None
        self.ledger.begin("request")
        self.tick(30)
        self.ledger.end("request")
        row = self.ledger.snapshot()
        self.assertIsNone(row["energy_kwh"])
        self.assertEqual(row["unmeasured_s"], 30)
        self.assertTrue(row["profit_partial"])

    def test_power_fallback_uses_both_endpoints(self):
        self.counter = False
        self.power = 100
        self.ledger.begin("request")
        self.tick(3600)
        self.power = 140
        self.ledger.end("request")
        self.assertAlmostEqual(self.ledger.snapshot()["energy_kwh"], 0.12)

    def test_midnight_splits_active_work_and_energy(self):
        self.wall = datetime(2024, 1, 1, 23, 59, 50).timestamp()
        self.ledger.begin("request")
        self.tick(20)
        self.ledger.end("request")
        rows = self.ledger.snapshot()["daily"]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["active_s"], 10)
            self.assertAlmostEqual(row["energy_kwh"], 0.0003333333333333333)

    def test_cache_hits_use_prompt_tokens_and_do_not_increase_total_tokens(self):
        self.ledger.record(100, 60, 40)
        row = self.ledger.snapshot()
        self.assertEqual(row["total_tokens"], 140)
        self.assertEqual(row["cache_hit_pct"], 60)

    def test_decode_average_is_time_weighted_and_ignores_untimed_work(self):
        self.ledger.record(100, 0, 100, decode_ms=2000)
        self.ledger.record(100, 0, 50, decode_ms=8000)
        self.ledger.record(100, 0, 1000)
        row = self.ledger.snapshot()
        self.assertEqual(row["decode_tok_s"], 15)
        self.assertEqual(row["decode_tokens"], 150)

    def test_decode_speed_excludes_injected_api_output_tokens(self):
        self.ledger.record(100, 0, 140, decode_ms=1000, decoded_tokens=120)
        row = self.ledger.snapshot()
        self.assertEqual(row["decode_tok_s"], 120)
        self.assertEqual(row["output_tokens"], 140)

    def test_invalid_timing_cannot_fabricate_speed(self):
        for milliseconds in (None, 0, -1, float("nan"), float("inf"), True):
            self.ledger.record(100, 0, 10, decode_ms=milliseconds)
        self.assertIsNone(self.ledger.snapshot()["decode_tok_s"])

    def test_restart_preserves_counts_without_adding_offline_energy(self):
        self.ledger.record(100, 50, 20, decode_ms=1000)
        self.ledger.begin("request")
        self.tick(30)
        self.ledger.end("request")
        self.tick(7200, watts=20)
        restored = UsageLedger(self.path, "test").snapshot()
        self.assertEqual(restored["cached_tokens"], 50)
        self.assertEqual(restored["decode_tok_s"], 20)
        self.assertAlmostEqual(restored["energy_kwh"], 0.001)

    def test_old_ledger_without_decode_counters_remains_readable(self):
        self.ledger.record(100, 50, 20)
        old = json.loads(self.path.read_text())
        for row in (old, old["undated"], *old["daily"].values()):
            row.pop("decode_s", None)
            row.pop("decode_tokens", None)
        self.path.write_text(json.dumps(old))
        self.assertIsNone(UsageLedger(self.path, "test").snapshot()["decode_tok_s"])

    def test_service_records_native_decode_count_across_budget_continuations(self):
        tok = ByteTokenizer()
        engine = ClockedThinkingEngine(tok)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        svc.usage = self.ledger
        ids, thinking, maximum = svc.prepare([{"role": "user", "content": "2+2?"}], None,
                                             {"reasoning_effort": "low"}, 400)
        list(svc.run(ids, thinking, None, maximum, {"reasoning_budget_tokens": 20}, threading.Event()))
        generated = sum(row["generated"] for row in engine.done)
        row = self.ledger.snapshot()
        self.assertGreater(row["output_tokens"], generated)
        self.assertEqual(row["decode_tokens"], generated)
        self.assertAlmostEqual(row["decode_tok_s"], generated / 0.28)
