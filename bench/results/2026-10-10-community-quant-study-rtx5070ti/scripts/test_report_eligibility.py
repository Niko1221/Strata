"""Regression tests for scored-axis eligibility and composite-speed gating."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_eligibility import decode_axis_status, gyro_s_scoring_gate

BASE = Path(__file__).resolve().parent.parent


class ReportEligibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads((BASE / "data/ALL-METRICS.json").read_text())

    def test_valid_scored_axes_survive_invalid_auxiliary_and_controller_exit(self):
        campaign = self.metrics["gyro_s_campaign"]
        original = copy.deepcopy(campaign)
        result = gyro_s_scoring_gate(campaign)
        self.assertTrue(result["eligible"], result["reasons"])
        self.assertEqual(campaign["outcome"], "aborted_or_invalid")
        self.assertFalse(campaign["controller_closeout_audit"]["controller_success"])
        self.assertIn("plugin-mmlu", campaign["scoring_slice_audit"]["invalid_non_scoring_slices"])
        self.assertEqual(campaign, original, "eligibility checks must not mutate raw campaign status")

    def test_missing_required_scored_slice_is_rejected(self):
        campaign = copy.deepcopy(self.metrics["gyro_s_campaign"])
        del campaign["scoring_slice_audit"]["required_slices"]["teb-xhigh"]
        result = gyro_s_scoring_gate(campaign)
        self.assertFalse(result["eligible"])
        self.assertTrue(any("teb-xhigh" in reason for reason in result["reasons"]))

    def test_invalid_required_scored_slice_is_rejected(self):
        campaign = copy.deepcopy(self.metrics["gyro_s_campaign"])
        campaign["scoring_slice_audit"]["required_slices"]["teb-low"]["valid"] = False
        result = gyro_s_scoring_gate(campaign)
        self.assertFalse(result["eligible"])

    def test_failed_restore_check_is_rejected(self):
        campaign = copy.deepcopy(self.metrics["gyro_s_campaign"])
        campaign["controller_closeout_audit"]["checks"]["stock_digests_unchanged"] = False
        result = gyro_s_scoring_gate(campaign)
        self.assertFalse(result["eligible"])
        self.assertTrue(any("stock_digests_unchanged" in reason for reason in result["reasons"]))

    def test_gyro_m_zero_background_decode_is_unrankable(self):
        models = self.metrics["models"]
        for effort in ("low", "xhigh"):
            result = decode_axis_status(models["gyro-m"]["perf"][effort])
            self.assertFalse(result["eligible"], (effort, result))
            self.assertIsNone(result["value"])
            self.assertTrue(result["reason"])

    def test_gyro_m_speed_route_is_unverified_at_every_depth(self):
        models = self.metrics["models"]
        for effort in ("low", "xhigh"):
            cells = models["gyro-m"]["perf"][effort]
            self.assertEqual(set(cells), {"0", "32768", "122880"})
            self.assertTrue(all(cell["comparison_eligible"] is False for cell in cells.values()))
            self.assertTrue(all(cell["speed_attribution_status"] == "unverified_endpoint_attribution" for cell in cells.values()))
            self.assertIn("8090", cells["0"]["comparison_ineligibility_reason"])
            self.assertIn("8080", cells["0"]["comparison_ineligibility_reason"])

    def test_speed_route_audit_is_hash_pinned(self):
        campaign = self.metrics["gyro_m_campaign"]["performance_attribution_audit"]
        audit_path = BASE / campaign["path"]
        import hashlib
        self.assertTrue(audit_path.is_file())
        self.assertEqual(hashlib.sha256(audit_path.read_bytes()).hexdigest(), campaign["sha256"])
        audit = json.loads(audit_path.read_text())
        self.assertEqual(audit["status"], "unverified_endpoint_attribution")
        self.assertFalse(audit["evidence"]["route_resolution"]["8090_to_8080_mapping_proven"])

    def test_audited_gyro_s_engine_decode_is_eligible(self):
        models = self.metrics["models"]
        for effort, expected in (("low", 99.033), ("xhigh", 89.3)):
            result = decode_axis_status(models["gyro-s"]["perf"][effort])
            self.assertTrue(result["eligible"], (effort, result))
            self.assertAlmostEqual(result["value"], expected, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
