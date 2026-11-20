"""Startup-log rules for the Qwen3.6 32GB/16GB plan. Not a speed result."""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from startup_log_analysis import analyze_startup_logs
from serve.native_family import NativeAdmission, native_admission_from_env, q36_memory_mode, validate_q36_memory_args


def _placement(seconds: str, generation: str) -> str:
    return (
        f"strata-q36: startup generation {generation}\n"
        "strata-q36: context 65536 cells, prompt chunk 2048, 4.91 GiB of VRAM free\n"
        "strata-q36: hybrid experts: 41 of 256 per layer in VRAM (3.00 GiB), all 18.75 GiB "
        f"in pageable host memory, the rest of a batch of <= 128 rows on 6 CPU threads, "
        f"bigger prompt chunks streamed (1.10 GiB staging); {seconds} s\n"
    )


class StartupLogAnalysisTest(unittest.TestCase):
    def test_native_legacy_record_source_is_byte_exact_with_trace_disabled(self) -> None:
        # A source-format fixture while native compilation is explicitly pending.
        # This checks the real fprintf literal and its independence from tracing;
        # it does not claim execution of the native binary.
        root = Path(__file__).resolve().parents[2]
        header = (root / "include/strata/qwen35/startup_trace.hpp").read_text(encoding="utf-8")
        body = header[header.index("inline void q36_startup("):]
        body = body[:body.index("\n}") + 2]
        format_literal = re.search(r'"(Q36_STARTUP schema=1[^"\n]*)"', body).group(1)
        native_format = bytes(format_literal, "ascii").decode("unicode_escape")
        expected = "Q36_STARTUP schema=1 stage=expert_repack state=end elapsed_ms=27 stage_elapsed_ms=3 error_code=null\n"
        self.assertEqual(native_format.replace("%lld", "%d") % ("expert_repack", "end", 27, 3, "null"), expected)
        self.assertNotIn("generation", body)
        self.assertNotIn("getenv", body)
        self.assertNotIn("q36_startup_detail_enabled", body)

    def test_native_upload_source_counts_only_confirmed_transfer_bytes(self) -> None:
        # Source/call-order evidence only; GPU transfer completion is NOT_RUN.
        root = Path(__file__).resolve().parents[2]
        native = (root / "src/qwen35/q36_engine.cu").read_text(encoding="utf-8")
        transfer = native[native.index("    void h2d_from_host("):native.index("    void stream_begin(")]
        self.assertRegex(transfer, r"\*completed \+= bytes; startup->progress\(\*completed, layer\)")
        self.assertRegex(transfer, r"CK\(cudaEventSynchronize\(pin_transfers.done\[oldest\]\)\);\s+complete\(pin_transfers.startup_bytes\[oldest\]\);\s+pin_transfers.startup_bytes\[oldest\] = 0;")
        self.assertRegex(transfer, r"CK\(cudaEventRecord\(pin_transfers.done\[slot\], stream\)\);\s+pin_transfers.startup_bytes\[slot\] = startup \? chunk : 0;")
        self.assertRegex(transfer, r"CK\(cudaStreamSynchronize\(stream\)\);\s+complete\(chunk\);")
        # Scheduling a fallback async copy alone must never count its bytes as complete.
        fallback = transfer[transfer.index("        if (!pool || n == 0)"):transfer.index("        static const bool pipeline")]
        self.assertIn("cudaMemcpyAsync", fallback)
        self.assertNotIn("complete(", fallback)
        placement = native[native.index("    uint64_t upload_bytes = 0, uploaded = 0;", native.index("bool place_experts")):]
        main = placement[:placement.index("        if (mapped)")]
        self.assertRegex(main, r"CK\(cudaMemcpy\([^;]+;\s+uploaded \+= L.L.bytes;\s+upload.progress\(uploaded, \(int\) li\);")
        self.assertRegex(placement, r"CK\(cudaMemcpy\(L.blobs, L.hblobs,[^;]+;\s+uploaded \+= L.stride \* \(size_t\) S;")
        self.assertNotIn("upload.progress(0,", placement)
        self.assertRegex(placement, r"CK\(cudaDeviceSynchronize\(\)\);\s+upload.end\(\);")

    def test_real_child_spawn_prefix_owns_a_fresh_generation_on_each_attempt(self) -> None:
        # Execute the real constructor up to its spawn receipt with a mocked Popen.
        # No engine, model, server listener, narrator thread or CUDA call is started.
        path = Path(__file__).resolve().parents[2] / "serve" / "server.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q36_startup_child_env")
        checks = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                  n.name in ("q36_verify_memory_args", "q36_verify_kv_args")]
        engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StrataEngine")
        init = copy.deepcopy(next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"))
        stop = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name) and n.value.func.id == "startup_child_receipt"
                    and n.value.args[0].value == "spawned")
        init.body = init.body[:stop + 1]
        fixture_class = ast.ClassDef(name="SpawnPrefix", bases=[], keywords=[], body=[init], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[helper, *checks, fixture_class], type_ignores=[]))
        fake_popen = mock.Mock(return_value=SimpleNamespace(pid=8765))
        namespace = {"os": os, "re": re, "uuid": uuid, "time": time, "Path": Path, "json": json,
                     "native_admission_from_env": native_admission_from_env,
                     "NativeAdmission": NativeAdmission, "q36_memory_mode": q36_memory_mode,
                     "validate_q36_memory_args": validate_q36_memory_args,
                     "threading": SimpleNamespace(Event=threading.Event, Thread=mock.Mock(), RLock=threading.RLock), "subprocess": subprocess,
                     "engine_nice": lambda env: 0, "end_leftover_engines": mock.Mock(), "popen": fake_popen,
                     "narrate_start": mock.Mock(), "_echoing": set()}
        exec(compile(module, str(path), "exec"), namespace)
        supplied = {"STRATA_REQUIRE_Q36_KV": "1", "STRATA_Q36_LAUNCH_GENERATION": "owned-launch", "STRATA_Q36_STARTUP_GENERATION": "stale", "KEEP": "unchanged"}
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {}, clear=True):
            log = str(Path(folder) / "engine.log")
            generations = []
            for _ in range(2):
                obj = namespace["SpawnPrefix"]("fake-engine", ["--max-context", "4096"], log=log, env=supplied)
                actual = fake_popen.call_args.kwargs["env"]
                generations.append(actual["STRATA_Q36_STARTUP_GENERATION"])
                self.assertEqual(actual["STRATA_Q36_LAUNCH_GENERATION"], "owned-launch")
                self.assertEqual(actual["KEEP"], "unchanged")
                obj.log.close()
            self.assertNotEqual(generations[0], generations[1])
            self.assertEqual(supplied["STRATA_Q36_STARTUP_GENERATION"], "stale")
            text = Path(log).read_text(encoding="utf-8")
            for generation in generations:
                self.assertIn(f"generation={generation} state=spawned process_id=8765", text)
            self.assertNotIn("generation=stale", text)
        child, generation = namespace["q36_startup_child_env"]({"STRATA_Q36_LAUNCH_GENERATION": "bad token", "STRATA_Q36_STARTUP_GENERATION": "stale"})
        self.assertIsNone(generation)
        self.assertNotIn("STRATA_Q36_STARTUP_GENERATION", child)

    def test_duplicate_upload_is_counted_once(self) -> None:
        one = (
            "strata-q36: context 65536 cells, prompt chunk 2048, 4.91 GiB of VRAM free\n"
            "strata-q36: no host memory for the experts (19195 MiB)\n"
        ).encode()
        single = analyze_startup_logs([one])
        doubled = analyze_startup_logs([one, one])
        self.assertEqual(doubled["counts"], single["counts"])
        self.assertEqual(doubled["uploads_counted"], 1)
        self.assertEqual(doubled["uploads_received"], 2)
        self.assertEqual(single["counts"]["host_allocation_failure"], 1)

    def test_no_followup_stays_unknown(self) -> None:
        blob = b"strata-q36: context 65536 cells, prompt chunk 2048, 4.91 GiB of VRAM free\n"
        result = analyze_startup_logs([blob])
        self.assertEqual(result["attempts"][0]["outcome"], "unknown")
        self.assertEqual(result["counts"]["unknown"], 1)
        self.assertNotIn("failed", result["attempts"][0]["outcome"])

    def test_zero_of_zero_has_no_hit_rate(self) -> None:
        blob = (
            "strata-q36: context 65536 cells, prompt chunk 2048, 4.91 GiB of VRAM free\n"
            "strata-q36: hybrid experts: 0 of 0 batch entries on the GPU (0.0%), 0 on the "
            "CPU pool in 0 ms, 0 slot replacements\n"
        ).encode()
        attempt = analyze_startup_logs([blob])["attempts"][0]
        self.assertIsNone(attempt["route_hit_rate"])
        self.assertEqual(attempt["gpu_route_entries"], 0)
        self.assertEqual(attempt["total_route_entries"], 0)

    def test_different_generations_are_not_added(self) -> None:
        blob = (_placement("10.0", "g1") + _placement("20.0", "g2")).encode()
        result = analyze_startup_logs([blob])
        self.assertIsNone(result["cross_generation_placement_seconds"])
        self.assertEqual(
            [a["placement_seconds"] for a in result["attempts"]],
            [10.0, 20.0],
        )

    def test_same_generation_can_report_its_own_sum(self) -> None:
        blob = (_placement("10.0", "g1") + _placement("5.5", "g1")).encode()
        result = analyze_startup_logs([blob])
        self.assertEqual(result["cross_generation_placement_seconds"], 15.5)

    def test_synthetic_mix_matches_its_own_counts(self) -> None:
        lines = ["strata-q36: context 4096 cells, prompt chunk 512, 1.00 GiB of VRAM free\n"] * 4
        lines += ["strata-q36: context 65536 cells, prompt chunk 2048, 4.91 GiB of VRAM free\n"
                  "strata-q36: no host memory for the experts (19195 MiB)\n"] * 14
        lines += ["strata-q36: context 65536 cells, prompt chunk 2048, 1.00 GiB of VRAM free\n"
                  "strata-q36: not enough VRAM left for the hybrid experts' buffers and slots (900 MiB needed, 100 MiB free after the weights, the context and the sessions)\n"] * 2
        lines.append(_placement("266.0", "done"))
        lines.append(_placement("266.0", "done"))
        lines.append(
            "strata-q36: startup generation warm1\n"
            "strata-q36: context 65536 cells, prompt chunk 2048, 4.91 GiB of VRAM free\n"
            "strata-q36: hybrid experts: 41 of 256 per layer in VRAM (3.00 GiB), all 18.75 GiB "
            "in pageable host memory, the rest of a batch of <= 128 rows on 6 CPU threads, "
            "bigger prompt chunks streamed (1.10 GiB staging); 266.0 s\n"
            "strata-q36: warmed up in 306.6 s (one prompt chunk, a decode step, a speculative step)\n"
        )
        result = analyze_startup_logs(["".join(lines).encode()])
        counts = result["counts"]
        self.assertEqual(counts["host_allocation_failure"], 14)
        self.assertEqual(counts["vram_failure"], 2)
        self.assertGreaterEqual(counts["expert_placement_complete"], 2)
        self.assertEqual(counts["warmup_complete"], 1)
        self.assertEqual(counts["unknown"], 4)

    def test_packaged_evidence_summary_is_historical_not_a_new_run(self) -> None:
        path = Path(__file__).resolve().parents[2] / "_docs" / "q36-32gb16gb" / "evidence-summary.json"
        if not path.is_file():
            self.skipTest("evidence summary is not in this tree")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["kind"], "user_supplied_historical_evidence_not_new_benchmark")
        self.assertEqual(summary["source_date"], "2026-11-02")
        self.assertEqual(summary["counts"]["engine_start"], 22)
        self.assertEqual(summary["counts"]["host_allocation_failure"], 14)
        # Later progress labels do not turn the unchanged historical input into
        # a new timing run. Keep the accepted labels explicit and its unknown
        # throughput/wall time unknown, including the preview-6 checkpoint.
        self.assertIn(
            summary["plan_execution_status"],
            {
                "NOT_IMPLEMENTED_NOT_GPU_TESTED",
                "PARTIAL_IMPL_GPU_SPEED_NOT_RUN",
                "PARTIAL_IMPL_MAPPED_LIVE_READY_T03_PASS_T04_KEEP_DEFAULT_PREVIEW_6_SHIPPED_"
                "T06_BLOCKED_T07_PROFILE_PASS_KERNEL_NOT_RUN_T08_REFUSE_PASS_MEASURED_NOT_RUN_T09_PARTIAL",
            },
        )
        self.assertNotIn("GPU_SPEED_PASS", summary["plan_execution_status"])
        self.assertIsNone(summary["installed_binary_matches_reviewed_source"])
        self.assertIsNone(summary["completed_timing_observation"]["decode_tps"])
        self.assertIsNone(summary["completed_timing_observation"]["total_wall_seconds"])


if __name__ == "__main__":
    unittest.main()
