"""Tests for ab_engine's versions mode and its CLI.  Everything from ab_versions and ab_report is patched: no network, no GPU, no subprocess."""
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
import urllib.error
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_engine  # noqa: E402
import ab_report  # noqa: E402
import ab_suites  # noqa: E402
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
SUITE_ROW = {"round": 0, "req": "ladder.8k", "suite": "ladder", "decode": 40.0, "prompt": 800.0, "prompt_n": 8000, "n": 30,
             "predicted_n": 30, "wall_s": 12.5, "ok": True, "check": "4/4 Codes"}


def fake_reqs(suite, n=1):
    return [ab_suites.Req(suite, f"{suite}.{i}", {}, lambda r: (True, "")) for i in range(n)]


def answer_base(port, reqs, max_new, rnd, arm, out, rows, check=None):
    rows.append(dict(ROW, round=rnd, arm=arm, **({"suite": "base", "ok": True, "check": ""} if check else {})))


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
             "served_context": mock.Mock(return_value=262144),
             "collect_configs": mock.Mock(return_value=[]), "version_problem": mock.Mock(return_value=None)}
        e = {"run_requests": mock.Mock(side_effect=answer_base),
             "run_suite": mock.Mock(side_effect=lambda port, reqs, rnd, arm, out, rows: rows.append(dict(SUITE_ROW, round=rnd))),
             "post": mock.Mock(), "gpu_name": mock.Mock(return_value="RTX 3090, 24576 MiB"),
             "detect": mock.Mock(return_value=("windows-x86_64", "cuda"))}
        r = {"write_report": mock.Mock(return_value=(self.bench / "report.html", None))}
        s = {"need_corpus": mock.Mock(), "model_context": mock.Mock(return_value=262144),
             "ladder_reqs": mock.Mock(return_value=fake_reqs("ladder", 2)), "think_reqs": mock.Mock(return_value=fake_reqs("think", 6)),
             "agent_reqs": mock.Mock(return_value=(fake_reqs("agent"), ["agent.x.9"]))}
        for key, value in over.items():
            for d in (m, e, r, s):
                if key in d:
                    d[key] = value
        argv = ["--model-path", str(self.model), "--engines", "0.1.55,0.1.56", "--backend", "cuda-cuda13",
                "--bench-dir", str(self.bench), "--rounds", "2", *extra]
        with ExitStack() as st:
            for name, value in m.items():
                st.enter_context(mock.patch.object(ab_versions, name, value))
            for name, value in e.items():
                st.enter_context(mock.patch.object(ab_engine, name, value))
            for name, value in s.items():
                st.enter_context(mock.patch.object(ab_suites, name, value))
            st.enter_context(mock.patch.object(ab_report, "write_report", r["write_report"]))
            st.enter_context(mock.patch("sys.stdout"))
            st.enter_context(mock.patch("sys.stderr"))
            code = ab_engine.main(argv)
        runs = sorted((self.bench / "runs").glob("*/results.json"))
        results = json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None
        return code, results, {**m, **e, **r, **s}

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
        self.assertEqual([c.args[0].parent.parent.name for c in m["launch"].call_args_list].count("0.1.55"), 1)

    def test_tune_only_in_first_round(self):
        code, results, m = self.run_tool(["--tune", "quick"])
        tunes = [(c.args[0].parent.parent.name, c.args[5]) for c in m["launch"].call_args_list]
        self.assertEqual(tunes, [("0.1.55", "quick"), ("0.1.56", "quick"), ("0.1.56", None), ("0.1.55", None)])
        self.assertEqual(results["tune"], "quick")
        self.assertEqual((results["arms"][0]["setup_s"], results["arms"][0]["start_s"]), (12.0, [12.0]))

    def test_interrupt_stops_server(self):
        def answer(port, reqs, max_new, rnd, arm, out, rows, check=None):
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
        launch = mock.Mock(side_effect=lambda control, *a, **k: LaunchResult(control.parent.parent.name == "0.1.56", 5.0, "boom"))
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
        self.assertEqual(set(results), {"created", "model", "gpu", "tune", "rounds", "requests", "arms", "untouched", "changed",
                                        "suites", "context", "corpus", "run_code", "params", "done", "sessions", "homes"})
        self.assertEqual((results["suites"], results["context"], results["corpus"], results["run_code"]), ([], None, None, False))
        self.assertEqual((results["model"], results["gpu"], results["tune"], results["rounds"]), ("model.gguf", "RTX 3090, 24576 MiB", None, 2))
        self.assertEqual(results["requests"], ["chat0", "chat1", "chat2", "long"])
        self.assertEqual(set(results["arms"][0]["rows"][0]), {"round", "req", "decode", "prompt", "prompt_n", "n", "suite", "ok", "check"})
        self.assertIs(m["run_requests"].call_args.kwargs["check"], ab_suites.base_check)
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
            self.run_tool(changed=mock.Mock(return_value=["~x"]), write_report=mock.Mock(side_effect=RuntimeError("boom")))
        self.assertTrue(any("~x" in str(c) for c in pr.call_args_list))

    def test_each_run_gets_a_fresh_home(self):
        homes = []
        for stamp in ("20261116-120000", "20261116-120001"):
            fmt = mock.Mock(side_effect=lambda f, *x, s=stamp: s if f == "%Y%m%d-%H%M%S" else "2026-11-16T12:00:00")
            with mock.patch("time.strftime", fmt):
                code, results, m = self.run_tool()
            homes += [c.args[0] for c in m["provision_home"].call_args_list]
        self.assertEqual(homes, [self.bench / "homes" / v / s for s in ("20261116-120000", "20261116-120001")
                                 for v in ("0.1.55", "0.1.56")])

    def test_crash_in_one_version_skips_it_and_the_rest_run(self):
        def launch(control, *a, **k):
            if control.parent.parent.name == "0.1.55":
                raise OSError("no such file: strata.cmd")
            return LaunchResult(True, 12.0)

        code, results, m = self.run_tool(launch=mock.Mock(side_effect=launch))
        first, second = results["arms"]
        self.assertEqual((first["status"], first["reason"]), ("skipped", "OSError: no such file: strata.cmd"))
        self.assertEqual((second["status"], len(second["rows"])), ("ran", 2))
        self.assertEqual(m["stop"].call_count, 3)                  # 0.1.55 once, 0.1.56 in both rounds
        self.assertEqual(code, 0)

    def test_crash_after_a_row_keeps_the_row_but_not_ran(self):
        def answer(port, reqs, max_new, rnd, arm, out, rows, check=None):
            rows.append(dict(ROW, round=rnd, arm=arm))
            if arm == "v0.1.55":
                raise TimeoutError("timed out")

        code, results, m = self.run_tool(run_requests=mock.Mock(side_effect=answer))
        first = results["arms"][0]
        self.assertEqual((first["status"], first["reason"], len(first["rows"])), ("skipped", "TimeoutError: timed out", 1))
        self.assertEqual(results["arms"][1]["status"], "ran")

    def test_bench_dir_inside_the_install_is_refused(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as cm:
            self.run_tool(live_roots=mock.Mock(return_value=[self.bench.parent]))
        self.assertIn("--bench-dir", str(cm.exception))
        self.assertFalse(self.bench.exists())

    def test_old_mode_refuses_versions_flags(self):
        with mock.patch("sys.stderr"):
            for flag in (["--tune", "quick"], ["--backend", "cuda"], ["--bench-dir", "x"]):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    ab_engine.parse_args(["cfg.json", "A:exe=x", *flag])

    def test_without_suite_one_launch_per_round(self):
        code, results, m = self.run_tool()
        self.assertEqual(m["launch"].call_count, 4)
        self.assertTrue(all("context" not in c.kwargs for c in m["launch"].call_args_list))
        m["need_corpus"].assert_not_called()
        m["run_suite"].assert_not_called()

    def test_suite_adds_a_second_launch_with_context(self):
        code, results, m = self.run_tool(["--suite", "ladder", "--tune", "quick"])
        calls = m["launch"].call_args_list
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(c.kwargs.get("context") == 262144 and c.args[5] is None for c in calls[1::2]))
        self.assertTrue(all("context" not in c.kwargs for c in calls[0::2]))
        self.assertEqual(m["stop"].call_count, 8)
        m["model_context"].assert_called_once_with(self.model)
        m["ladder_reqs"].assert_called_once_with(262144)              # built once per n_ctx
        m["think_reqs"].assert_not_called()
        m["agent_reqs"].assert_not_called()
        self.assertEqual(m["run_suite"].call_count, 4)
        self.assertEqual(code, 0)

    def test_explicit_context_wins(self):
        code, results, m = self.run_tool(["--suite", "ladder,think,agent", "--context", "65536"],
                                         served_context=mock.Mock(return_value=None))
        self.assertTrue(all(c.kwargs["context"] == 65536 for c in m["launch"].call_args_list[1::2]))
        m["model_context"].assert_not_called()
        m["ladder_reqs"].assert_called_once_with(65536)              # /props gave nothing: the asked context
        m["think_reqs"].assert_called_with(8192, False)
        self.assertEqual(len(m["run_suite"].call_args_list[0].args[1]), 2 + 6 + 1)
        self.assertEqual(results["context"], 65536)

    def test_suite_launch_failure_keeps_base_rows(self):
        launch = mock.Mock(side_effect=lambda *a, **k: LaunchResult(False, 5, "out of memory") if "context" in k else LaunchResult(True, 12.0))
        code, results, m = self.run_tool(["--suite", "ladder"], launch=launch)
        for arm in results["arms"]:
            self.assertEqual((arm["status"], arm["suites_status"]), ("ran", "skipped"))
            self.assertIn("out of memory", arm["suites_reason"])
            self.assertEqual([r["suite"] for r in arm["rows"]], ["base", "base"])
        m["run_suite"].assert_not_called()
        self.assertEqual(m["stop"].call_count, m["launch"].call_count)
        self.assertEqual(code, 0)

    def test_missing_corpus_stops_before_download(self):
        ensure, corpus = mock.Mock(), mock.Mock(side_effect=SystemExit("x"))
        with self.assertRaises(SystemExit):
            self.run_tool(["--suite", "think"], ensure_packages=ensure, need_corpus=corpus)
        corpus.assert_called_once()
        ensure.assert_not_called()

    def test_interrupt_during_suites_stops_and_keeps_rows(self):
        def suite(port, reqs, rnd, arm, out, rows):
            rows.append(dict(SUITE_ROW, round=rnd))
            raise KeyboardInterrupt

        code, results, m = self.run_tool(["--suite", "ladder"], run_suite=mock.Mock(side_effect=suite))
        self.assertEqual(code, 130)
        self.assertEqual(m["stop"].call_count, 2)                    # base launch, then the suite launch
        first = results["arms"][0]
        self.assertEqual([r["suite"] for r in first["rows"]], ["base", "ladder"])
        self.assertEqual(first["rows"][1], SUITE_ROW)
        self.assertEqual(first["suites_status"], "ran")

    def test_suite_crash_skips_suites_not_the_version(self):
        code, results, m = self.run_tool(["--suite", "ladder"], run_suite=mock.Mock(side_effect=TimeoutError("timed out")))
        for arm in results["arms"]:
            self.assertEqual((arm["status"], arm["suites_status"], arm["suites_reason"]), ("ran", "skipped", "TimeoutError: timed out"))
            self.assertEqual(len(arm["rows"]), 2)
        self.assertEqual(code, 0)

    def test_suite_flags_refused_in_old_mode(self):
        with mock.patch("sys.stderr"):
            for flag in (["--suite", "ladder"], ["--context", "65536"], ["--think-tokens", "4096"], ["--run-code"]):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    ab_engine.parse_args(["cfg.json", "A:", *flag])

    def refused_without_suite(self, *flag):
        argv = ["--model-path", "m.gguf", "--engines", "0.1.55", *flag]
        self.assertEqual(ab_engine.parse_args([*argv, "--suite", "think"]).suite, ["think"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            ab_engine.parse_args(argv)
        self.assertIn("need --suite", err.getvalue())

    def test_context_needs_suite(self):
        self.refused_without_suite("--context", "65536")

    def test_think_tokens_needs_suite(self):
        self.refused_without_suite("--think-tokens", "4096")

    def test_run_code_needs_suite(self):
        self.refused_without_suite("--run-code")

    def test_senseless_numbers_are_refused(self):
        base = ["--model-path", "m.gguf", "--engines", "0.1.55"]
        for flags in (["--suite", "think", "--context", "0"], ["--suite", "think", "--context", "-5"],
                      ["--suite", "think", "--think-tokens", "0"], ["--suite", "think", "--think-tokens", "-1"],
                      ["--suite", "ladder", "--context", "9215"], ["--suite", "think,ladder", "--context", "4096"],
                      ["--suite", "ladder,agent", "--run-code"], ["--suite", "agent", "--run-code"]):
            err = io.StringIO()
            with self.subTest(flags=flags), contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                ab_engine.parse_args([*base, *flags])
            self.assertTrue(err.getvalue())
        self.assertEqual(ab_engine.parse_args([*base, "--suite", "ladder", "--context", "9216"]).context, 9216)
        self.assertEqual(ab_engine.parse_args([*base, "--suite", "think", "--context", "4096"]).context, 4096)  # no ladder: fine
        self.assertTrue(ab_engine.parse_args([*base, "--suite", "agent,think", "--run-code"]).run_code)

    def test_suites_run_think_agent_ladder_in_that_order(self):
        code, results, m = self.run_tool(["--suite", "ladder,agent,think"])
        tags = [r.tag for r in m["run_suite"].call_args_list[0].args[1]]
        self.assertEqual([t.split(".")[0] for t in tags], ["think"] * 6 + ["agent"] + ["ladder"] * 2)

    def test_a_bigger_served_context_does_not_change_the_requests(self):
        code, results, m = self.run_tool(["--suite", "ladder,agent,think"], served_context=mock.Mock(return_value=524288))
        m["ladder_reqs"].assert_called_once_with(262144)
        m["agent_reqs"].assert_called_once_with(262144)
        self.assertEqual([a["n_ctx"] for a in results["arms"]], [524288, 524288])

    def test_unknown_suite_is_refused(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            ab_engine.parse_args(["--model-path", "m.gguf", "--engines", "0.1.55", "--suite", "ladder,speed"])

    def test_run_code_warns_before_and_in_help(self):
        with mock.patch("builtins.print") as pr:
            self.run_tool(["--suite", "think", "--run-code"])
        self.assertTrue(any("not isolated" in str(c) for c in pr.call_args_list))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            ab_engine.parse_args(["--help"])
        self.assertIn("not isolated", " ".join(out.getvalue().split()))

    def test_results_schema_with_suites(self):
        code, results, m = self.run_tool(["--suite", "ladder,think,agent", "--context", "131072", "--run-code"],
                                         served_context=mock.Mock(return_value=131072))
        self.assertEqual((results["suites"], results["context"], results["corpus"], results["run_code"]),
                         (["ladder", "think", "agent"], 131072, "v0.2.12", True))
        arm = results["arms"][0]
        self.assertEqual((arm["n_ctx"], arm["suites_status"], arm["suites_reason"], arm["left_out"]), (131072, "ran", "", ["agent.x.9"]))
        self.assertEqual(set(arm["rows"][1]), set(SUITE_ROW))
        self.assertEqual(arm["start_s"], [12.0])                      # base launch only


class Resume(unittest.TestCase):
    """A stopped run continues where it stopped: only the missing launches run, in the same run folder and homes."""
    setUp, tearDown, run_tool = VersionsMode.setUp, VersionsMode.tearDown, VersionsMode.run_tool

    def killed(self, n, extra=()):
        """Launch number n dies hard: the tool's own finally never gets to write (its install check fails too), only checkpoints stay."""
        calls = []

        def launch(*args, **kw):
            calls.append(args)
            if len(calls) == n:
                raise SystemExit("killed")
            return LaunchResult(True, 12.0)
        with self.assertRaises(SystemExit):
            self.run_tool(list(extra), launch=mock.Mock(side_effect=launch), snapshot=mock.Mock(side_effect=[{}, SystemExit("killed")]))

    def blocks(self, m):
        return [(c.args[0].parent.parent.name, "suites" if c.kwargs.get("context") else "base") for c in m["launch"].call_args_list]

    def test_checkpoint_after_every_block(self):
        self.killed(3, ["--suite", "think"])
        r = json.loads(next((self.bench / "runs").glob("*/results.json")).read_text(encoding="utf-8"))
        self.assertEqual(r["done"], ["0:0.1.55:base", "0:0.1.55:suites"])
        self.assertEqual(r["sessions"][0]["checked"], False)

    def test_only_the_missing_blocks_run(self):
        self.killed(4, ["--suite", "think"])                         # dies in the suites start of 0.1.56, round 0
        code, results, m = self.run_tool(["--suite", "think"])
        self.assertEqual(code, 0)
        self.assertEqual(self.blocks(m), [("0.1.56", "suites"), ("0.1.56", "base"), ("0.1.56", "suites"),
                                          ("0.1.55", "base"), ("0.1.55", "suites")])
        self.assertEqual(len(list((self.bench / "runs").iterdir())), 1)
        for arm in results["arms"]:                                  # one base and one suite row per round, no duplicate
            self.assertEqual(sorted((r["round"], r["suite"]) for r in arm["rows"]), [(0, "base"), (0, "ladder"), (1, "base"), (1, "ladder")])
        self.assertEqual([x["checked"] for x in results["sessions"]], [False, True])

    def test_an_unfinished_block_is_redone_without_its_old_rows(self):
        def answer(port, reqs, max_new, rnd, arm, out, rows, check=None):
            rows.append(dict(ROW, round=rnd, arm=arm, suite="base", ok=True, check=""))
            if arm == "v0.1.56":
                raise KeyboardInterrupt                              # Ctrl-C: rows of this block are in the file, block not done
        self.run_tool(run_requests=mock.Mock(side_effect=answer))
        code, results, m = self.run_tool()
        self.assertEqual([len(a["rows"]) for a in results["arms"]], [2, 2])

    def test_resume_reuses_homes_and_stops_leftover_servers_first(self):
        self.killed(2)
        first = next((self.bench / "runs").iterdir()).name
        code, results, m = self.run_tool()
        homes = [c.args[0] for c in m["provision_home"].call_args_list][-2:]   # before them: the cleanup of 0.1.55
        self.assertEqual(homes[0], self.bench / "homes" / "0.1.55" / first)
        self.assertEqual(homes[1], self.bench / "homes" / "0.1.56" / f"{first}-1")  # no block done: a fresh home, a clean setup time
        self.assertEqual(m["stop"].call_count, 4)                   # 3 launches, and first the leftover server of 0.1.55
        self.assertEqual(results["arms"][0]["setup_s"], 12.0)
        self.assertEqual(results["arms"][0]["start_s"], [12.0])

    def test_fresh_starts_over(self):
        self.killed(2)
        with mock.patch("time.strftime", mock.Mock(side_effect=lambda f, *x: "20991231-235959" if f == "%Y%m%d-%H%M%S" else "x")):
            code, results, m = self.run_tool(["--fresh"])
        self.assertEqual(len(m["launch"].call_args_list), 4)
        self.assertEqual(len(list((self.bench / "runs").iterdir())), 2)

    def test_other_parameters_start_a_new_run(self):
        self.killed(2)
        with mock.patch("time.strftime", mock.Mock(side_effect=lambda f, *x: "20991231-235959" if f == "%Y%m%d-%H%M%S" else "x")):
            code, results, m = self.run_tool(["--max-new", "128"])
        self.assertEqual(len(m["launch"].call_args_list), 4)

    def test_more_rounds_and_a_new_version_join_a_finished_run(self):
        self.run_tool(["--engines", "0.1.55"])
        code, results, m = self.run_tool(["--rounds", "3"])
        self.assertEqual(self.blocks(m), [("0.1.56", "base"), ("0.1.56", "base"), ("0.1.55", "base"), ("0.1.56", "base")])
        self.assertEqual(len(list((self.bench / "runs").iterdir())), 1)
        self.assertEqual(results["rounds"], 3)

    def test_a_finished_run_is_not_continued(self):
        self.run_tool()
        with mock.patch("time.strftime", mock.Mock(side_effect=lambda f, *x: "20991231-235959" if f == "%Y%m%d-%H%M%S" else "x")):
            code, results, m = self.run_tool()
        self.assertEqual(len(m["launch"].call_args_list), 4)
        self.assertEqual(len(list((self.bench / "runs").iterdir())), 2)

    def test_fresh_belongs_to_versions_mode(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            ab_engine.parse_args(["cfg.json", "A:", "--fresh"])


GOLDEN_JSONL = ('{"round": 0, "arm": "A", "req": "chat0", "decode": 50.0, "prompt": 900.0, "prompt_n": 40, "n": 256, "text": "hi there"}\n'
                '{"round": 0, "arm": "A", "req": "long", "decode": 50.0, "prompt": 900.0, "prompt_n": 40, "n": 256, "text": "hi there"}\n')
GOLDEN_STDOUT = ("  chat0  decode 50.0 tok/s  prompt 900.0 tok/s (40)  n 256\n"
                 "  long   decode 50.0 tok/s  prompt 900.0 tok/s (40)  n 256\n")
FULL = {"timings": {"predicted_per_second": 50.0, "prompt_per_second": 900.0, "prompt_n": 40, "predicted_n": 12},
        "usage": {"completion_tokens": 256}, "choices": [{"message": {"content": "hi there"}}]}
SUITE_KEYS = ["round", "req", "suite", "decode", "prompt", "prompt_n", "n", "predicted_n", "wall_s", "ok", "check"]


def capture(fn):
    out, so = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so):
        rows = fn(out)
    return rows, out.getvalue(), so.getvalue()


class SendingRequests(unittest.TestCase):
    def test_run_requests_old_call_unchanged(self):
        with mock.patch.object(ab_engine, "post", lambda port, p, m: FULL):
            _, jsonl, stdout = capture(lambda out: ab_engine.run_requests(8080, [("chat0", "a"), ("long", "b")], 256, 0, "A", out))
        self.assertEqual(jsonl, GOLDEN_JSONL)
        self.assertEqual(stdout, GOLDEN_STDOUT)

    def test_run_requests_with_check_adds_fields(self):
        with mock.patch.object(ab_engine, "post", lambda port, p, m: FULL):
            rows, jsonl, _ = capture(lambda out: ab_engine.run_requests(
                8080, [("chat0", "a")], 256, 0, "A", out, check=lambda r: (False, "x")))
        self.assertEqual((rows[0]["suite"], rows[0]["ok"], rows[0]["check"]), ("base", False, "x"))
        self.assertIn('"suite": "base"', jsonl)

    def test_run_suite_row_fields(self):
        req = types.SimpleNamespace(suite="ladder", tag="ladder.4k", body={"a": 1}, check=lambda r: (True, ""))
        sent = []
        with mock.patch.object(ab_engine, "send", lambda port, body: sent.append(body) or FULL):
            rows, jsonl, stdout = capture(lambda out: ab_engine.run_suite(8080, [req], 1, "v1", out, []))
        self.assertEqual(sent, [{"a": 1}])
        self.assertEqual(list(rows[0]), SUITE_KEYS)
        self.assertEqual((rows[0]["predicted_n"], rows[0]["suite"], rows[0]["req"], rows[0]["ok"]), (12, "ladder", "ladder.4k", True))
        line = json.loads(jsonl)
        self.assertEqual((line["arm"], line["text"]), ("v1", "hi there"))
        self.assertIn("ladder.4k", stdout)

    def test_send_turns_http_error_into_error_dict(self):
        err = urllib.error.HTTPError("u", 400, "Bad", {}, io.BytesIO(b"prompt too long"))
        with mock.patch.object(ab_engine.urllib.request, "urlopen", side_effect=err):
            self.assertEqual(ab_engine.send(8080, {}), {"error": "HTTP 400: prompt too long"})

    def test_run_suite_http_400_is_a_wrong_row_and_continues(self):
        reqs = [types.SimpleNamespace(suite="ladder", tag=t, body={}, check=lambda r: (True, "")) for t in ("a", "b")]
        answers = iter([{"error": "HTTP 400: prompt too long"}, FULL])
        with mock.patch.object(ab_engine, "send", lambda port, body: next(answers)):
            rows, _, _ = capture(lambda out: ab_engine.run_suite(8080, reqs, 0, "v1", out, []))
        self.assertEqual([r["ok"] for r in rows], [False, True])
        self.assertEqual(rows[0]["check"], "HTTP 400: prompt too long")
        self.assertIsNone(rows[0]["decode"])
        self.assertIsNone(rows[0]["predicted_n"])


if __name__ == "__main__":
    unittest.main()
