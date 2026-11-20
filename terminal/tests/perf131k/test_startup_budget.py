"""CPU-only checks of phase overlap; no model, Job or large allocations."""

import unittest
from pathlib import Path

from tools.perf131k.startup_budget import (
    DIAGNOSTIC_JOB_LIMIT, PHASES, direct_scratch_bound, evaluate, stage_payload_bound,
)


def proposed(events=None, reserve=1024, limit=DIAGNOSTIC_JOB_LIMIT):
    return {"job_limit_bytes": limit, "untracked_reserve_bytes": reserve,
            "untracked_reserve_basis": "synthetic test bound, not a runtime measurement",
            "events": events if events is not None else
            [{"action": "enter", "phase": phase} for phase in PHASES]}


class StartupBudgetTests(unittest.TestCase):
    def test_scratch_formula_is_bound_to_current_native_constants(self):
        source = (Path(__file__).resolve().parents[2] / "src/core/expert_source.cpp").read_text(encoding="utf-8")
        direct = source.split("bool FileExpertSource::read_direct(", 1)[1].split("#elif defined(__linux__)", 1)[0]
        self.assertIn("kSector = 4096, kGap = 1ull << 20, kMerge = 32ull << 20", direct)
        self.assertIn("x.a0 <= end + kGap", direct)
        self.assertIn("sc.cap = (size_t) ((total + (1u << 20) - 1) >> 20 << 20)", direct)

    def test_conditional_payload_bounds_keep_thread_and_slot_units_distinct(self):
        blob = 1382400
        self.assertEqual(stage_payload_bound(blob, 65), 443750400)
        for fills, roles, mib in ((16, 1, 37), (16, 3, 69), (32, 1, 74), (32, 3, 138)):
            self.assertEqual(direct_scratch_bound(blob, fills, roles), mib * 1024**2)
        # A one-byte unaligned read still needs the capacity rounding unit.
        self.assertEqual(direct_scratch_bound(1, 1, 1), 1024**2)
        for invalid in (None, True, -1):
            with self.assertRaises(ValueError):
                stage_payload_bound(blob, invalid)
        with self.assertRaises(ValueError):
            direct_scratch_bound(blob, 16, 2)

    def test_replacement_overlap_counts_old_and_new_until_actual_uncommit(self):
        events = [
            {"action": "enter", "phase": "source_open"},
            {"action": "acquire", "id": "old", "category": "scratch", "bytes": 3000,
             "basis": "synthetic retained scratch"},
            {"action": "enter", "phase": "cache_attempt"},
            {"action": "enter", "phase": "profile_fill"},
            {"action": "acquire", "id": "new", "category": "scratch", "bytes": 5000,
             "basis": "synthetic replacement scratch"},
            {"action": "release", "id": "old", "kind": "actual_uncommit",
             "outcome": "confirmed_success", "basis": "synthetic successful uncommit receipt"},
            {"action": "enter", "phase": "resident_complement"},
            {"action": "enter", "phase": "ready"},
        ]
        result = evaluate(proposed(events))
        self.assertEqual(result["estimated_peak_host_commit_bytes"], 9024)
        self.assertEqual(result["live_at_ready_bytes"], 5000)
        self.assertFalse(result["runtime_proof"] or result["launch_authorized"])

    def test_pool_return_does_not_uncharge_commit(self):
        events = proposed()["events"]
        events.insert(1, {"action": "acquire", "id": "stage", "category": "stage",
                          "bytes": 4096, "basis": "synthetic retained stage"})
        events.insert(3, {"action": "release", "id": "stage", "kind": "pool_return"})
        with self.assertRaisesRegex(ValueError, "logical release"):
            evaluate(proposed(events))

    def test_unknown_or_unsubstantiated_uncommit_cannot_reduce_peak(self):
        for release in (
            {"kind": "actual_uncommit"},
            {"kind": "actual_uncommit", "outcome": "unknown", "basis": "release API failed"},
            {"kind": "actual_uncommit", "outcome": "confirmed_success", "basis": " "},
        ):
            with self.subTest(release=release):
                events = proposed()["events"]
                events[1:1] = [
                    {"action": "acquire", "id": "old", "category": "scratch", "bytes": 3000,
                     "basis": "synthetic allocation"},
                    {"action": "release", "id": "old", **release},
                ]
                with self.assertRaisesRegex(ValueError, "confirmed success"):
                    evaluate(proposed(events))

    def test_unknown_charge_and_missing_phase_block(self):
        with self.assertRaisesRegex(ValueError, "known nonnegative"):
            evaluate(proposed(reserve=None))
        incomplete = proposed()
        incomplete["events"].pop()
        with self.assertRaisesRegex(ValueError, "all startup phases"):
            evaluate(incomplete)

    def test_over_cap_blocks_without_allocating_large_buffer(self):
        events = proposed()["events"]
        events.insert(1, {"action": "acquire", "id": "fixed", "category": "fixed",
                          "bytes": DIAGNOSTIC_JOB_LIMIT, "basis": "synthetic logical byte count"})
        result = evaluate(proposed(events))
        self.assertEqual(result["diagnostic_cap_comparison"], "ESTIMATE_OVER_CAP")
        self.assertLess(result["estimated_cap_margin_bytes"], 0)
        self.assertFalse(result["product_fit_claim"])

    def test_same_demand_is_independent_of_diagnostic_cap(self):
        low = evaluate(proposed(limit=8 * 1024**3))
        high = evaluate(proposed(limit=DIAGNOSTIC_JOB_LIMIT))
        without_cap = evaluate(proposed(limit=None))
        self.assertEqual(low["estimated_peak_host_commit_bytes"],
                         high["estimated_peak_host_commit_bytes"])
        self.assertEqual(low["estimated_peak_host_commit_bytes"],
                         without_cap["estimated_peak_host_commit_bytes"])
        self.assertEqual(without_cap["diagnostic_cap_comparison"], "CAP_UNSPECIFIED")
        self.assertFalse(without_cap["launch_authorized"])


if __name__ == "__main__":
    unittest.main()
