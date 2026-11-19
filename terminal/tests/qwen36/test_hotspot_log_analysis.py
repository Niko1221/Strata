"""RED/GREEN for Q36_HOTSPOT schema (T07 profile capture). Not a kernel speed claim."""

from __future__ import annotations

import unittest

from hotspot_log_analysis import extract_hotspot_lines, parse_hotspot_line


_SAMPLE = (
    "Q36_HOTSPOT schema=1 phase=warmup layers_gdn=48 layers_attn=16 "
    "gdn_host_ms=1200.500 attn_host_ms=800.250 moe_gpu_host_ms=400.125 moe_cpu_ms=90.000 "
    "stream_copy_wait_ms=1500.000 stream_mmq_ms=2200.000 stream_h2d_bytes=123456789 "
    "stream_d2d_bytes=98765 hy_cpu=100 hy_gpu=900 hy_swaps=8 rebalance_h2d_bytes=1111 "
    "hardware_counters=null note=diagnostic_sync"
)


class HotspotLogAnalysisTest(unittest.TestCase):
    def test_parses_schema1_summary(self) -> None:
        line = parse_hotspot_line(_SAMPLE)
        self.assertIsNotNone(line)
        assert line is not None
        self.assertEqual(line.schema, 1)
        self.assertEqual(line.phase, "warmup")
        self.assertEqual(line.layers_gdn, 48)
        self.assertEqual(line.layers_attn, 16)
        self.assertEqual(line.hardware_counters, "null")
        self.assertEqual(line.note, "diagnostic_sync")
        self.assertEqual(line.dominant_host_bucket(), "moe_gpu")

    def test_rejects_garbage(self) -> None:
        self.assertIsNone(parse_hotspot_line("Q36_STARTUP schema=1 stage=buffers state=end"))
        self.assertIsNone(parse_hotspot_line("Q36_HOTSPOT schema=1 phase=warmup incomplete"))

    def test_extract_from_mixed_stderr(self) -> None:
        blob = (
            "strata-q36: hybrid experts: 41 of 256\n"
            + _SAMPLE
            + "\n"
            + "Q36_STARTUP schema=1 stage=warmup_prefill state=end elapsed_ms=1 stage_elapsed_ms=1 error_code=null\n"
        )
        lines = extract_hotspot_lines(blob)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].phase, "warmup")

    def test_env_off_means_no_lines_is_valid_default(self) -> None:
        # Default path must not invent hotspot lines; empty extract is the env-off contract.
        self.assertEqual(extract_hotspot_lines("strata-q36: warmed up in 1.0 s\n"), [])


if __name__ == "__main__":
    unittest.main()
