"""Tests for ab_engine's versions mode and its CLI.  Everything from ab_versions and ab_report is patched: no network, no GPU, no subprocess."""
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_engine  # noqa: E402
import ab_report  # noqa: E402
import ab_versions  # noqa: E402
from ab_versions import LaunchResult, Packages  # noqa: E402


def a(name):
    return {"name": name, "browser_download_url": f"https://example.invalid/dl/{name}"}


RELEASES = [
    {"tag_name": "v0.2.10", "assets": [a("strata-engine-0.1.56-windows-x86_64-cuda-cuda13.tar.gz"),
                                       a("strata-engine-0.1.56-windows-x86_64-hip-rdna2.tar.gz"),
                                       a("strata-engine-0.1.56-windows-x86_64-hip-rdna3.tar.gz"),
                                       a("strata-control-0.1.56-windows-x86_64.zip"), a("SHA256SUMS.txt")]},
    {"tag_name": "v0.2.9", "assets": [a("strata-engine-0.1.55-windows-x86_64-cuda-cuda12.tar.gz"),
                                      a("strata-engine-0.1.55-windows-x86_64-cuda-cuda13.tar.gz"),
                                      a("strata-control-0.1.55-windows-x86_64.zip"), a("SHA256SUMS.txt")]},
    {"tag_name": "v0.2.8", "assets": [a("strata-engine-0.1.54-windows-x86_64-cuda-cuda12.tar.gz"),
                                      a("strata-control-0.1.54-windows-x86_64.zip"), a("SHA256SUMS.txt")]},
]
ANSWER = {"timings": {"predicted_per_second": 50.0, "prompt_per_second": 900.0, "prompt_n": 40}, "usage": {"completion_tokens": 256},
          "choices": [{"message": {"content": "hi"}}]}
ROW = {"round": 0, "arm": "x", "req": "chat0", "decode": 50.0, "prompt": 900.0, "prompt_n": 40, "n": 256, "text": "hi"}


class VersionsMode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bench = Path(self.tmp.name) / "bench"
        self.model = Path(self.tmp.name) / "model.gguf"
        self.model.write_bytes(b"gguf")

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, extra=(), **over):
        """Run versions mode with every outside effect patched; returns (exit code, results.json, mocks)."""
        def packages(pair, cache, opener=None):
            return Packages(pair.version, cache / f"engine-{pair.version}", cache / f"control-{pair.version}")

        m = {"list_releases": mock.Mock(return_value=RELEASES), "ensure_packages": mock.Mock(side_effect=packages),
             "provision_home": mock.Mock(side_effect=lambda home, p: home / "control"), "assert_idle": mock.Mock(),
             "snapshot": mock.Mock(return_value={}), "changed": mock.Mock(return_value=[]), "live_roots": mock.Mock(return_value=[]),
             "launch": mock.Mock(return_value=LaunchResult(True, 12.0)), "stop": mock.Mock(),
             "collect_configs": mock.Mock(return_value=[]), "version_problem": mock.Mock(return_value=None)}
        e = {"run_requests": mock.Mock(side_effect=lambda port, reqs, max_new, rnd, arm, out, rows: rows.append(dict(ROW, round=rnd, arm=arm))),
             "post": mock.Mock(), "gpu_name": mock.Mock(return_value="RTX 3090, 24576 MiB"),
             "detect": mock.Mock(return_value=("windows-x86_64", "cuda"))}
        r = {"write_report": mock.Mock(return_value=(self.bench / "report.html", None))}
        for key, value in over.items():
            for d in (m, e, r):
                if key in d:
                    d[key] = value
        argv = ["--model-path", str(self.model), "--engines", "0.1.55,0.1.56", "--backend", "cuda-cuda13",
                "--bench-dir", str(self.bench), "--rounds", "2", *extra]
        with ExitStack() as st:
            for name, value in m.items():
                st.enter_context(mock.patch.object(ab_versions, name, value))
            for name, value in e.items():
                st.enter_context(mock.patch.object(ab_engine, name, value))
            st.enter_context(mock.patch.object(ab_report, "write_report", r["write_report"]))
            st.enter_context(mock.patch("sys.stdout"))
            st.enter_context(mock.patch("sys.stderr"))
            code = ab_engine.main(argv)
        runs = sorted((self.bench / "runs").glob("*/results.json"))
        results = json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None
        return code, results, {**m, **e, **r}

    def test_old_mode_argv_unchanged(self):
        a = ab_engine.parse_args(["cfg.json", "A:exe=x", "--rounds", "2"])
        self.assertEqual((a.config, a.arms, a.rounds, a.port, a.max_new, a.long_words, a.log),
                         ("cfg.json", ["A:exe=x"], 2, 8090, 256, 6000, "ab_engine.jsonl"))
        a = ab_engine.parse_args(["cfg.json", "--rounds", "2", "A:exe=x", "B:exe=y"])
        self.assertEqual(a.arms, ["A:exe=x", "B:exe=y"])

    def test_modes_are_exclusive(self):
        with mock.patch("sys.stderr"):
            for argv in (["cfg.json", "A:exe=x", "--engines", "0.1.55"], ["cfg.json"], ["--engines", "0.1.55"], [],
                         ["--model-path", "m.gguf", "--engines", "0.1.55", "A:exe=x"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    ab_engine.parse_args(argv)

    def test_empty_engine_list_is_refused_before_anything_starts(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self.run_tool(["--engines", ","])

    def test_skipped_arm_is_reported_not_dropped(self):
        problem = mock.Mock(side_effect=lambda home, p: "served engine x, expected y" if p.version == "0.1.55" else None)
        code, results, m = self.run_tool(version_problem=problem)
        self.assertEqual(code, 0)
        arm = results["arms"][0]
        self.assertEqual((arm["version"], arm["status"], arm["reason"], arm["rows"]), ("0.1.55", "skipped", "served engine x, expected y", []))
        self.assertEqual(results["arms"][1]["status"], "ran")
        self.assertIn("served engine x, expected y", ab_report.build_html(results))
        self.assertEqual([c.args[0] for c in m["launch"].call_args_list].count(self.bench / "homes" / "0.1.55" / "control"), 1)

    def test_tune_only_in_first_round(self):
        code, results, m = self.run_tool(["--tune", "quick"])
        tunes = [(c.args[0].parent.name, c.args[5]) for c in m["launch"].call_args_list]
        self.assertEqual(tunes, [("0.1.55", "quick"), ("0.1.56", "quick"), ("0.1.56", None), ("0.1.55", None)])
        self.assertEqual(results["tune"], "quick")
        self.assertEqual((results["arms"][0]["setup_s"], results["arms"][0]["start_s"]), (12.0, [12.0]))

    def test_interrupt_stops_server(self):
        def answer(port, reqs, max_new, rnd, arm, out, rows):
            if arm != "v0.1.55":
                raise KeyboardInterrupt
            rows.append(dict(ROW, round=rnd, arm=arm))

        requests = mock.Mock(side_effect=answer)
        code, results, m = self.run_tool(run_requests=requests)
        self.assertEqual(m["stop"].call_count, 2)
        self.assertEqual(m["snapshot"].call_count, 2)
        first, second = results["arms"]
        self.assertEqual((first["version"], first["status"], len(first["rows"])), ("0.1.55", "ran", 1))
        self.assertEqual((second["version"], second["status"]), ("0.1.56", "skipped"))
        self.assertNotEqual(code, 0)
        self.assertIn('data-version="0.1.55" data-req="chat0" data-decode="50.0"', ab_report.build_html(results))

    def test_failed_launch_still_stops(self):
        code, results, m = self.run_tool(launch=mock.Mock(return_value=LaunchResult(False, 3.0, "timed out after 7200 s")))
        self.assertEqual(m["stop"].call_count, 2)
        self.assertEqual({x["status"] for x in results["arms"]}, {"skipped"})
        self.assertEqual(results["arms"][0]["reason"], "timed out after 7200 s")
        m["run_requests"].assert_not_called()

    def test_changed_install_fails_run(self):
        code, results, m = self.run_tool(changed=mock.Mock(return_value=["~x"]))
        self.assertEqual(code, 1)
        self.assertEqual((results["untouched"], results["changed"]), (False, ["~x"]))

    def test_configs_collected_per_version_even_on_failed_launch(self):
        launch = mock.Mock(side_effect=lambda control, *a, **k: LaunchResult(control.parent.name == "0.1.56", 5.0, "boom"))
        code, results, m = self.run_tool(launch=launch)
        dests = [c.args[1] for c in m["collect_configs"].call_args_list]
        self.assertEqual([d.parent.name for d in dests], ["0.1.55", "0.1.56", "0.1.56"])
        self.assertTrue(all(d.parent.parent == self.bench / "configs" for d in dests))
        self.assertEqual(len({d.name for d in dests}), 1)
        self.assertTrue((self.bench / "runs" / dests[0].name / "results.json").is_file())

    def test_nothing_deleted_in_bench_dir(self):
        planted = self.bench / "homes" / "0.1.55" / "keep.txt"
        planted.parent.mkdir(parents=True)
        planted.write_text("x")
        self.run_tool()
        self.assertTrue(planted.is_file())

    def test_results_schema(self):
        code, results, m = self.run_tool()
        self.assertEqual(set(results), {"created", "model", "gpu", "tune", "rounds", "requests", "arms", "untouched", "changed"})
        self.assertEqual((results["model"], results["gpu"], results["tune"], results["rounds"]), ("model.gguf", "RTX 3090, 24576 MiB", None, 2))
        self.assertEqual(results["requests"], ["chat0", "chat1", "chat2", "long"])
        self.assertEqual(set(results["arms"][0]["rows"][0]), {"round", "req", "decode", "prompt", "prompt_n", "n"})
        m["assert_idle"].assert_called_once_with([8090])
        self.assertEqual(m["live_roots"].call_args.args[1], self.model)
        m["write_report"].assert_called_once()

    def engines_used(self, m):
        return [c.args[0].engine_name for c in m["ensure_packages"].call_args_list]

    def test_bare_cuda_prefers_cuda13_and_falls_back_per_version(self):
        code, results, m = self.run_tool(["--backend", "cuda", "--engines", "0.1.54..0.1.56"])
        self.assertEqual(self.engines_used(m), ["strata-engine-0.1.54-windows-x86_64-cuda-cuda12.tar.gz",
                                                "strata-engine-0.1.55-windows-x86_64-cuda-cuda13.tar.gz",
                                                "strata-engine-0.1.56-windows-x86_64-cuda-cuda13.tar.gz"])

    def test_detected_family_needs_no_backend(self):
        argv = ["--model-path", str(self.model), "--engines", "0.1.55,0.1.56", "--bench-dir", str(self.bench)]
        with mock.patch.object(ab_engine, "run_versions", mock.Mock(return_value=0)) as rv:
            ab_engine.main(argv)
        self.assertIsNone(rv.call_args.args[0].backend)
        code, results, m = self.run_tool(["--backend", ""])          # "" -> the detected family, "cuda"
        self.assertEqual(self.engines_used(m)[0], "strata-engine-0.1.55-windows-x86_64-cuda-cuda13.tar.gz")

    def test_exact_backend_overrides(self):
        code, results, m = self.run_tool(["--backend", "cuda-cuda12", "--engines", "0.1.55"])
        self.assertEqual(self.engines_used(m), ["strata-engine-0.1.55-windows-x86_64-cuda-cuda12.tar.gz"])

    def test_several_hip_builds_ask(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self.run_tool(["--backend", "hip", "--engines", "0.1.56"])

    def test_interrupt_mid_round_keeps_answered_rows(self):
        post = mock.Mock(side_effect=[ANSWER, ANSWER, KeyboardInterrupt])     # warm-up, chat0, then Ctrl-C on chat1
        code, results, m = self.run_tool(run_requests=ab_engine.run_requests, post=post)
        first = results["arms"][0]
        self.assertEqual((first["status"], [r["req"] for r in first["rows"]]), ("ran", ["chat0"]))
        self.assertEqual(m["stop"].call_count, 1)
        self.assertEqual(code, 130)

    def test_changed_files_are_named_even_when_the_run_crashes(self):
        with mock.patch("builtins.print") as pr, self.assertRaises(RuntimeError):
            self.run_tool(changed=mock.Mock(return_value=["~x"]), launch=mock.Mock(side_effect=RuntimeError("boom")))
        self.assertTrue(any("~x" in str(c) for c in pr.call_args_list))

    def test_old_mode_refuses_versions_flags(self):
        with mock.patch("sys.stderr"):
            for flag in (["--tune", "quick"], ["--backend", "cuda"], ["--bench-dir", "x"]):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    ab_engine.parse_args(["cfg.json", "A:exe=x", *flag])


if __name__ == "__main__":
    unittest.main()
