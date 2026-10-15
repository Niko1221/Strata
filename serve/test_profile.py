"""serve/test_profile.py - the Profile tab (serve/profile_page.py): off by default, the engine's switches, the last
request's breakdown and the last 50 from the engine log, the settings and findings, and a run of the tool through
HTTP against the mock engine.  No GPU.

    python -m unittest serve.test_profile
"""
from __future__ import annotations

import http.client
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import profile_page as P
from serve import runconfig
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, child_env, serve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import test_strata_profile as T  # noqa: E402  the real engine lines

REQUEST_1 = [T.PREFILL_2K, T.PREFILL_HOST, T.DECODE_2K,
             "strata serve: prompt 1889 tokens = 0 reused + 1889 read in 1980 ms (954.0 tok/s), 160 generated in 2424 ms "
             "(66.0 tok/s), drafts accepted 90 of 140, 1 checkpoints", T.HIT_PLAIN, T.KV]
REQUEST_2 = ["strata verify: capturing the 6-token window (284 MiB of VRAM free)", T.PREFILL_118K, T.PREFILL_HOST, T.DECODE_118K,
             T.SUMMARY, T.HIT, "strata serve: route tail skip: 18700 missed experts skipped (20063 entries) since the start",
             "strata serve: KV streaming: 94.89% of 1386456 block reads hit VRAM, 285.2 MiB read from RAM"]


class Env(unittest.TestCase):
    def test_timing_env_adds_the_switches_and_keeps_the_users(self):
        self.assertEqual(P.timing_env({"A": "1"}), {"A": "1", "STRATA_PREFILL_TIMING": "1", "STRATA_DECODE_TIMING": "1"})
        self.assertEqual(P.timing_env({"STRATA_DECODE_TIMING": "0"})["STRATA_DECODE_TIMING"], "0")

    def test_child_env_sets_them_only_with_the_key(self):
        cfg = {"exe": "x", "args": [], "env": {"STRATA_X": "1"}}
        self.assertNotIn("STRATA_DECODE_TIMING", child_env(cfg))
        on = child_env({**cfg, "profile": True})
        self.assertEqual((on["STRATA_PREFILL_TIMING"], on["STRATA_DECODE_TIMING"], on["STRATA_X"]), ("1", "1", "1"))
        self.assertNotIn("STRATA_DECODE_TIMING", child_env({**cfg, "profile": "yes"}))   # true only


class LastRequestLines(unittest.TestCase):
    def test_the_last_request_s_lines_and_its_own_kv_share(self):
        r = P.last_request(REQUEST_1 + REQUEST_2)
        self.assertEqual(r["summary"]["generated"], 160)
        self.assertEqual((r["prompt"]["tokens"], r["prompt"]["reused"], r["prompt"]["read"], r["prompt"]["tok_s"]), (117895, 0, 117895, 2840.0))
        self.assertEqual(r["prompt"]["groups"]["tokens"], 117888)                   # this request's prefill line, not the first's
        self.assertEqual(r["decode"]["groups"]["ms_per_window"], 21.61)
        self.assertEqual(r["decode"]["groups"]["cpu_experts"], 2.15)
        self.assertEqual((r["decode"]["drafts_accepted"], r["decode"]["drafts_offered"]), (82, 125))
        self.assertEqual(r["hit"]["hit_rate"], 0.88)                                 # the line after the summary
        self.assertEqual(r["kv"]["block_reads"], 1386456 - 1294032)                   # the difference from the first request's line
        self.assertEqual(r["prompt"]["host"]["ple_ms"], 284.0)
        self.assertIsInstance(r["prompt"]["groups"]["top"][0], list)                 # JSON-ready

    def test_the_first_request_has_no_kv_share_and_a_reused_prompt_has_no_phases(self):
        r = P.last_request(REQUEST_1)
        self.assertIsNone(r["kv"])
        self.assertEqual(r["hit"]["hit_rate"], 0.726)
        resumed = [T.DECODE_118K, T.SUMMARY_RESUMED.replace("625 read", "0 read"), T.HIT_PLAIN]
        r = P.last_request(REQUEST_1 + resumed)
        self.assertIsNone(r["prompt"]["groups"])
        self.assertEqual(r["decode"]["groups"]["ms_per_window"], 21.61)
        self.assertIsNone(P.last_request(["strata serve: ready", T.PREFILL_2K]))   # no finished request yet
        self.assertIsNone(P.last_request([]))

    def test_the_log_s_tail_is_read_once_per_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text("older engine run\n" + "\n".join(REQUEST_1) + "\n", encoding="utf-8")
            start = os.path.getsize(log)                                          # this engine's run begins here
            lr = P.LastRequest()
            self.assertIsNone(lr.read(str(log), start))                             # nothing of this run yet
            with open(log, "a", encoding="utf-8") as f:
                f.write("\n".join(REQUEST_2) + "\n")
            r = lr.read(str(log), start)
            self.assertEqual(r["summary"]["generated"], 160)
            self.assertIsNone(r["kv"])                                              # the earlier run's KV line is before `start`
            self.assertIs(lr.read(str(log), start), r)                              # unchanged log: the same answer, not parsed again
            self.assertIsNone(lr.read(str(log), None))                              # a lazy start: no run of this engine yet
            with open(log, "a", encoding="utf-8") as f:
                f.write(T.DECODE_2K + "\n")                                         # a half-finished next request
            self.assertEqual(lr.read(str(log), start)["summary"]["generated"], 160)
            self.assertIsNone(lr.read(None, 0))
            self.assertIsNone(lr.read(str(Path(tmp) / "missing.log"), 0))


FAKE_TOOL = textwrap.dedent('''
    import json, os, sys, time
    out = sys.argv[sys.argv.index("--out") + 1]
    d = os.path.join(out, "2026-10-15-profile-fake")
    os.makedirs(d, exist_ok=True)
    print("warming up (not measured) ...", flush=True)
    time.sleep(0.2)
    print("chat: 3 short questions", flush=True)
    results = {"results": {"prompts": {"2K": {"prompt_tokens": 2000, "prompt_tps": 1000.0, "groups": None}}, "decode": {}, "agent": {},
                           "findings": ["one finding"]}, "machine": {}}
    open(os.path.join(d, "results.json"), "w").write(json.dumps(results))
    open(os.path.join(d, "README.md"), "w").write("# fake report\\n")
    print("key " + ("set" if os.environ.get("STRATA_KEY") == "sesame" else "unset"), flush=True)
    print("args " + " ".join(sys.argv[1:]), flush=True)
    print("Report: " + os.path.join(d, "README.md") + " (home folders, user and host names and IP addresses are replaced)", flush=True)
    sys.exit(int(os.environ.get("FAKE_RC", "0")))
''')


class Runs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.script = Path(self.tmp.name) / "fake_tool.py"
        self.script.write_text(FAKE_TOOL, encoding="utf-8")

    def run_of(self, **kw):
        import subprocess
        script = self.script

        def popen(cmd, **k):                                 # the tool's command line, run through the fake
            return subprocess.Popen([sys.executable, str(script)] + cmd[2:], **k)
        return P.ProfileRun(base="http://127.0.0.1:1", api_key="sesame", config_path="c.json", out_dir=Path(self.tmp.name), popen=popen, **kw)

    @staticmethod
    def wait(run, timeout=10):
        t0 = time.time()
        while run.status()["state"] == "running" and time.time() - t0 < timeout:
            time.sleep(0.05)
        return run.status()

    def test_a_run_reports_its_lines_and_the_report(self):
        run = self.run_of()
        self.assertEqual(run.status()["state"], "idle")
        self.assertEqual(run.start("/tmp/engine.log", long=0), "started")
        self.assertEqual(run.start("/tmp/engine.log"), "running")              # one at a time
        st = self.wait(run)
        self.assertEqual(st["state"], "done")
        self.assertEqual(st["returncode"], 0)
        self.assertIn("key set", st["lines"])                                    # the API key went through the environment
        args = next(x for x in st["lines"] if x.startswith("args "))
        self.assertIn("--url http://127.0.0.1:1 --config c.json --engine-log /tmp/engine.log --long 0 --out", args)
        self.assertNotIn("sesame", args)
        self.assertEqual(st["report"]["results"]["results"]["findings"], ["one finding"])
        self.assertEqual(st["report"]["readme"], "# fake report\n")
        self.assertTrue(st["report"]["path"].endswith("2026-10-15-profile-fake"))
        self.assertEqual(run.start(None), "started")                             # again, without an engine log
        st = self.wait(run)
        self.assertNotIn("--engine-log", next(x for x in st["lines"] if x.startswith("args ")))

    def test_a_failed_run_says_so(self):
        run = self.run_of()
        os.environ["FAKE_RC"] = "3"
        try:
            run.start("e.log")
            st = self.wait(run)
        finally:
            del os.environ["FAKE_RC"]
        self.assertEqual((st["state"], st["returncode"]), ("failed", 3))
        self.assertIsNotNone(st["report"])                                       # what was written is still shown

    def test_only_a_readme_in_the_results_folder_is_read_back(self):
        run = P.ProfileRun(base="x", out_dir=Path(self.tmp.name))
        d = Path(self.tmp.name) / "2026-10-15-profile-x"
        d.mkdir()
        (d / "README.md").write_text("# r\n", encoding="utf-8")
        (d / "results.json").write_text("{}", encoding="utf-8")
        secret = Path(self.tmp.name) / "secret.txt"
        secret.write_text("key", encoding="utf-8")
        self.assertEqual(run.find_report([f"Report: {d / 'README.md'} (home folders, user and host names are replaced)"])["readme"], "# r\n")
        self.assertIsNone(run.find_report([f"Report: {secret} (home folders ...)"]))              # not a README.md
        self.assertIsNone(run.find_report([f"Report: {d / '..' / 'secret.txt'}"]))
        self.assertIsNone(run.find_report(["Report: /etc/README.md"]))                            # outside the results folder
        self.assertIsNone(run.find_report(["chat: 3 short questions"]))
        self.assertEqual(run.find_report([f"Report: {d / 'README.md'}", "later line"])["path"], str(d))

    def test_a_tool_that_cannot_start(self):
        def popen(cmd, **k):
            raise OSError("no python")
        run = P.ProfileRun(base="http://127.0.0.1:1", popen=popen)
        self.assertEqual(run.start(None), "failed")
        self.assertEqual(run.status()["state"], "failed")
        self.assertIn("no python", run.status()["lines"][0])

    def test_command(self):
        run = P.ProfileRun(base="http://127.0.0.1:8080")
        self.assertEqual(run.command(None, None), [sys.executable, str(P.TOOL), "--url", "http://127.0.0.1:8080"])


class Tune(unittest.TestCase):
    def test_command_and_summary(self):
        run = P.TuneRun(config_path="/x/strata-q2_0.json")
        self.assertEqual(run.command(None, None), [sys.executable, "-m", "serve.profile_page", "--calibrate", "/x/strata-q2_0.json"])
        run.returncode = 0
        self.assertEqual(run.find_report(["  Loading ...", "  [ok] tuned for this PC: --pcie-frac 0.35, --spec-min-p 0.70 (61.2 tok/s)"]),
                         {"summary": "--pcie-frac 0.35, --spec-min-p 0.70 (61.2 tok/s)", "tok_s": 61.2, "changes": None})
        self.assertIsNone(run.find_report(["the tuning did not finish"]))
        self.assertIsNone(run.find_report(["[ok] tuned for this PC: the default settings are already the fastest here"])["tok_s"])
        run.returncode = 1
        self.assertIsNone(run.find_report(["  [ok] tuned for this PC: --pcie-frac 0.35 (61.2 tok/s)"]))   # a failed run has no report

    def test_the_tuned_flags_go_into_the_engine_s_start_command(self):
        class Engine:
            spawn = ("strata", ["--pack", "p", "--pcie-frac", "0.90", "--max-context", "8192"], None, "e.log", {"PATH": "/bin"}, False, None)
        cfg = {"exe": "strata", "args": ["--pack", "p", "--max-context", "8192", "--pcie-frac", "0.35", "--spec-min-p", "0.70",
                                          "--pool-workers", "6"], "env": {"STRATA_PREFILL_CPU_SHARE_MAX": "3072"}}
        e = Engine()
        applied = P.apply_to_spawn(e, cfg)
        self.assertEqual(applied, {"--pcie-frac": "0.35", "--spec-min-p": "0.70", "--pool-workers": "6"})
        exe, args, cwd, log, env, lazy, cpus = e.spawn
        self.assertEqual(args[:4], ["--pack", "p", "--max-context", "8192"])        # the rest of the command as it was
        self.assertEqual(args[4:], ["--pcie-frac", "0.35", "--spec-min-p", "0.70", "--pool-workers", "6"])
        self.assertEqual(env, {"PATH": "/bin"})                                     # a variable the calibration does not manage is not touched
        self.assertEqual((exe, log, lazy), ("strata", "e.log", False))
        # a calibration that kept the defaults takes an older value out again
        P.apply_to_spawn(e, {"args": ["--pack", "p", "--max-context", "8192", "--spec-min-p", "0.5"]})
        self.assertEqual(e.spawn[1], ["--pack", "p", "--max-context", "8192", "--spec-min-p", "0.5"])
        self.assertEqual(P.apply_to_spawn(object(), cfg), {})                       # an engine without a start command

    def test_on_done_runs_after_a_good_run(self):
        import subprocess
        done = []
        script = Path(tempfile.mkdtemp()) / "t.py"
        script.write_text("print('  [ok] tuned for this PC: --pcie-frac 0.35')\n", encoding="utf-8")
        ended = []
        run = P.TuneRun(config_path="c.json", popen=lambda cmd, **k: subprocess.Popen([sys.executable, str(script)], **k),
                        on_done=lambda: (done.append(1), "applied")[1], on_end=lambda: ended.append(1))
        run.start(None)
        st = Runs.wait(run)
        self.assertEqual((st["state"], st["report"]["summary"], st["note"], done, ended), ("done", "--pcie-frac 0.35", "applied", [1], [1]))
        run.on_done = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        run.start(None)
        st = Runs.wait(run)
        self.assertEqual((st["state"], ended), ("done", [1, 1]))
        self.assertIn("could not be applied: boom", st["note"])

    def test_managed_variables_and_replicas(self):
        import calibrate as CAL
        class Engine:
            spawn = ("strata", ["--pack", "p"], None, None, {"STRATA_PREFILL_CPU_SHARE_MAX": "3072", "OTHER": "1"}, False, None)
        with mock.patch.object(CAL, "ENV_DEFAULTS", {"STRATA_PREFILL_CPU_SHARE_MAX": None}, create=True):
            e = Engine()
            applied = P.apply_to_spawn(e, {"args": ["--pack", "p"], "env": {"OTHER": "2"}})   # the calibration took the variable away
            self.assertEqual(applied, {"STRATA_PREFILL_CPU_SHARE_MAX": None})
            self.assertEqual(e.spawn[4], {"OTHER": "1"})                                       # a variable it does not manage is left alone
            applied = P.apply_to_spawn(e, {"args": ["--pack", "p"], "env": {"STRATA_PREFILL_CPU_SHARE_MAX": "3072"}})
            self.assertEqual((applied, e.spawn[4]["STRATA_PREFILL_CPU_SHARE_MAX"]), ({"STRATA_PREFILL_CPU_SHARE_MAX": "3072"}, "3072"))
        class Replicas:
            spawn = Engine.spawn
            engines = [Engine()]
        self.assertEqual(P.apply_to_spawn(Replicas(), {"args": ["--pack", "p", "--pcie-frac", "0.3"]}), {})


# the calibration child, as far as the server sees it: it writes its winners into the run config and says so
TUNE_SCRIPT = textwrap.dedent("""
    import json, sys
    path = {config!r}
    cfg = json.load(open(path))
    args = cfg["args"]
    for flag, value in (("--pcie-frac", "0.35"), ("--spec-min-p", "0.70"), ("--adapt-every", "1"),
                        ("--adapt-swaps", "160"), ("--adapt-decay", "0.97")):
        if flag in args:
            del args[args.index(flag):args.index(flag) + 2]
        args += [flag, value]
    json.dump(cfg, open(path, "w"))
    print("  [ok] tuned for this PC: --pcie-frac 0.35, --spec-min-p 0.70, --adapt-every 1, --adapt-swaps 160, "
          "--adapt-decay 0.97 (96.6 tok/s)", flush=True)
    sys.exit({rc})
""")


def request_lines(i: int) -> list[str]:
    """A synthetic request #i: its two timing lines, its summary and its hit rate line."""
    summary = (f"strata serve: prompt {1000 + i} tokens = 0 reused + {1000 + i} read in 1000 ms (1000.0 tok/s), 10 generated in "
               f"100 ms (100.0 tok/s), drafts accepted 1 of 2, 0 checkpoints")
    return [T.PREFILL_2K, T.DECODE_2K, summary, T.HIT_PLAIN]


class History(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "engine.log"
        self.log.write_text("", encoding="utf-8")
        self.h = P.ProfileHistory()

    def write(self, lines, mode="a"):
        with open(self.log, mode, encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def read(self, start=0):
        return self.h.read(str(self.log), start)["requests"]

    def test_two_requests_in_order_the_newest_open(self):
        self.write(REQUEST_1 + REQUEST_2)
        rows = self.read()
        self.assertEqual([r["n"] for r in rows], [1, 2])
        self.assertEqual([r["decode_tok_s"] for r in rows], [66.0, 93.7])
        self.assertEqual([r["open"] for r in rows], [False, True])
        share = P.last_request(REQUEST_1 + REQUEST_2)["prompt"]["groups"]["share"]
        self.assertEqual(rows[1]["prompt_top"], max(share, key=share.get))
        self.assertEqual((rows[1]["prompt_tokens"], rows[1]["hit_rate"], rows[1]["pcie_share"]), (117895, 0.88, 0.033))
        self.assertEqual(rows[1]["ms"]["CPU experts"], 7.4)
        self.assertEqual(rows[0]["hit_rate"], 0.726)                                   # the hit line after its own summary

    def test_a_new_summary_closes_the_open_entry(self):
        self.write(REQUEST_1 + REQUEST_2)
        first = self.read()
        self.assertTrue(first[1]["open"])
        self.assertEqual(self.read(), first)                                              # nothing new: not parsed again, not added
        self.write([T.PREFILL_2K, T.DECODE_2K, T.SUMMARY_RESUMED, T.HIT_PLAIN])
        rows = self.read()
        self.assertEqual([r["n"] for r in rows], [1, 2, 3])
        self.assertEqual([r["open"] for r in rows], [False, False, True])
        self.assertEqual(rows[1]["hit_rate"], 0.88)                                       # from the line after its own summary
        self.assertEqual(rows[1]["seen_at"], first[1]["seen_at"])                          # the same entry, closed, not a new one
        self.assertEqual(rows[2]["hit_rate"], 0.726)

    def test_the_same_summary_is_replaced_when_its_lines_grow(self):
        self.write(REQUEST_1[:4])                                                         # up to its summary: no hit line yet
        rows = self.read()
        self.assertEqual((len(rows), rows[0]["hit_rate"], rows[0]["open"]), (1, None, True))
        self.write(REQUEST_1[4:])
        rows = self.read()
        self.assertEqual((len(rows), rows[0]["n"], rows[0]["hit_rate"], rows[0]["open"]), (1, 1, 0.726, True))
        self.assertEqual(self.h.n, 1)

    def test_the_ring_keeps_the_last_50(self):
        self.write([line for i in range(30) for line in request_lines(i)])
        self.assertEqual(len(self.read()), 30)
        self.write([line for i in range(30, 60) for line in request_lines(i)])
        rows = self.read()
        self.assertEqual(len(rows), 50)
        self.assertEqual((rows[0]["n"], rows[-1]["n"], rows[-1]["prompt_tokens"]), (11, 60, 1059))
        self.assertEqual(sum(r["open"] for r in rows), 1)
        # all 60 in one read: the same ring, counted from 1
        other = P.ProfileHistory()
        rows = other.read(str(self.log), 0)["requests"]
        self.assertEqual((len(rows), rows[0]["n"], rows[-1]["n"]), (50, 11, 60))

    def test_a_new_engine_start_counts_a_run(self):
        self.write(REQUEST_1)
        self.assertEqual(self.h.read(str(self.log), 0)["engine_run"], 1)
        start = os.path.getsize(self.log)                                                 # the engine started again here
        self.write(request_lines(7))
        out = self.h.read(str(self.log), start)
        self.assertEqual(out["engine_run"], 2)
        rows = out["requests"]
        self.assertEqual([(r["n"], r["engine_run"], r["open"]) for r in rows], [(1, 1, False), (2, 2, True)])
        self.assertIsNone(self.h.read(None, None)["requests"][-1]["kv_hit"])               # a lazy start reads nothing, adds no run
        self.assertEqual(self.h.engine_run, 2)

    def test_reads_whole_lines_and_at_most_1_MiB(self):
        text = "\n".join(REQUEST_1) + "\n"
        cut = text.index("(66.0 tok/s)") + 4
        self.log.write_text(text[:cut], encoding="utf-8")                                 # the summary is half written
        self.assertEqual(self.read(), [])
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(text[cut:])
        rows = self.read()
        self.assertEqual((len(rows), rows[0]["decode_tok_s"], rows[0]["hit_rate"]), (1, 66.0, 0.726))
        with open(self.log, "a", encoding="utf-8") as f:                                   # 3 MiB of other lines between two reads
            f.write(("x" * 99 + "\n") * 31000)
            f.write("\n".join(REQUEST_2) + "\n")
        rows = self.read()
        self.assertLessEqual(self.h.read_bytes, 1 << 20)
        self.assertEqual([r["n"] for r in rows], [1, 2])
        self.assertEqual([r["open"] for r in rows], [False, True])
        self.assertEqual(rows[1]["decode_tok_s"], 93.7)

    def test_a_log_that_was_emptied_starts_again(self):
        self.write(REQUEST_1)
        self.assertEqual(len(self.read()), 1)
        self.log.write_text("", encoding="utf-8")
        self.assertEqual(len(self.read()), 1)
        self.write(request_lines(3))
        self.assertEqual([r["n"] for r in self.read()], [1, 2])


    def test_a_batch_that_cannot_be_parsed_changes_nothing(self):
        self.write(request_lines(1) + request_lines(2))
        before = self.read()
        self.assertEqual([r["prompt_tokens"] for r in before], [1001, 1002])
        calls = []
        real = P.last_request

        def flaky(lines):                                                                   # raises once, on the batch's second request
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("unparsable")
            return real(lines)
        self.write(request_lines(3) + request_lines(4) + request_lines(5))
        with mock.patch.object(P, "last_request", flaky), mock.patch("builtins.print"):
            self.assertEqual(self.read(), before)                                           # nothing half-done
        self.assertTrue(self.h.failed)
        self.write(request_lines(6))
        rows = self.read()
        self.assertEqual([(r["n"], r["prompt_tokens"], r["open"]) for r in rows], [(1, 1001, False), (2, 1002, False), (3, 1006, True)])

    def test_a_replaced_log_is_read_from_its_start(self):
        self.write(request_lines(1) + request_lines(2))
        self.assertEqual(len(self.read()), 2)
        other = Path(self.tmp.name) / "other.log"
        other.write_text("\n".join(line for i in range(3, 9) for line in request_lines(i)) + "\n", encoding="utf-8")
        self.assertGreater(os.path.getsize(other), os.path.getsize(self.log))                # larger than the old read position
        os.replace(other, self.log)
        rows = self.read()
        self.assertEqual([r["prompt_tokens"] for r in rows], [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008])
        self.assertEqual(rows[-1]["n"], 8)

    def test_a_long_tail_without_a_newline_is_not_read_again(self):
        self.write(REQUEST_1)
        self.assertEqual(len(self.read()), 1)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write("x" * (3 << 20))                                                          # no line end yet
        self.read()
        self.assertLessEqual(self.h.read_bytes, 1 << 20)
        self.read()
        self.assertEqual(self.h.read_bytes, 0)                                                # not 1 MiB again
        self.write(["", "\n".join(request_lines(5))])
        self.assertEqual([r["n"] for r in self.read()], [1, 2])

    def test_skip_to_end_closes_the_open_entry_and_drops_what_was_written(self):
        self.write(REQUEST_1)
        self.assertTrue(self.read()[0]["open"])
        self.write(request_lines(2))
        self.h.skip_to_end(str(self.log))
        rows = self.h.snapshot()["requests"]
        self.assertEqual([(r["n"], r["open"]) for r in rows], [(1, False)])
        self.write(request_lines(3))
        self.assertEqual([(r["n"], r["prompt_tokens"]) for r in self.read()], [(1, 1889), (2, 1003)])


class Findings(unittest.TestCase):
    # the two advice sentences of bench/results/2026-10-15-profile-rtx-5070-ti-swift-1-5-iq3-s/results.json
    SHARE = ('A short read that waits for expert copies is on the path the CPU prefill share works on (STRATA_PREFILL_CPU_SHARE, '
             'its range STRATA_PREFILL_CPU_SHARE_MAX; docs/DETAILS.md, "Short prompts: the CPU shares the experts"). Its range '
             'stops at 1,024 tokens by default and a wider one is not a sure win (2,048-token chunks measured 2% slower on one '
             'PC, docs/DETAILS.md): measure it here before setting it (tools/ab_engine.py A/Bs an environment variable).')
    CACHE = ('A decode whose larger share is the experts computed on the CPU follows the expert cache\'s hit rate: the settings '
             'on that path are the cache\'s slots (VRAM: --max-context, --kv-resident, --vram-reserve-mib; docs/DETAILS.md, '
             '"The expert cache size is a budget"), the PCIe share and the adaptive tier (setup --calibrate measures both on '
             'this PC) and a saved expert profile (expert_profile_save).')

    def test_the_example_findings_name_their_settings(self):
        self.assertEqual(P.finding_settings(self.SHARE), ["prefill_cpu_share", "prefill_cpu_share_max"])
        self.assertEqual(P.finding_settings(self.CACHE), ["max_context", "kv_resident", "vram_reserve_mib", "expert_cache",
                                                          "pcie_frac", "adapt_every", "adapt_swaps", "adapt_decay",
                                                          "expert_profile_save"])
        self.assertEqual(P.finding_settings("The 2K read (2,023 tokens, 1,118 tok/s) waits for the experts streamed over PCIe: "
                                            "43% of its GPU time waiting for their copies and 15% dequantizing them."), [])
        self.assertEqual(P.finding_settings("chat: the draft takes 20% of the window (2.00 tokens accepted per window): --spec and "
                                            "the draft floor (--spec-min-p, setup --calibrate) are its settings."),
                         ["spec_min_p", "spec"])
        self.assertEqual(P.finding_settings("x: 60% of the KV block reads hit VRAM (12 MiB read from RAM): --kv-resident sets how "
                                            "much of the KV cache stays on the card."), ["kv_resident"])

    def test_every_setting_the_tool_names_is_mapped(self):
        source = inspect.getsource(P.parsers().findings)
        named = set(re.findall(r"--[a-z-]+|STRATA_[A-Z_]+|expert_profile_save", source)) - {"--calibrate", "--verify-profile"}
        mapped = {words.rstrip(", ") for words, _ in P.FINDING_SETTINGS}
        self.assertTrue(named)
        self.assertEqual(named - mapped, set(), "the tool's findings name a setting the tab does not map")
        ids = {row[0] for row in P.SPEED_SETTINGS}
        self.assertEqual({i for _, found in P.FINDING_SETTINGS for i in found} - ids, set())

    def test_status_report_carries_them(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "fake_tool.py"
        script.write_text(FAKE_TOOL.replace('["one finding"]', 'json.loads(os.environ["FAKE_FINDINGS"])'), encoding="utf-8")
        run = P.ProfileRun(base="http://127.0.0.1:1", config_path="c.json", out_dir=Path(tmp.name),
                           popen=lambda cmd, **k: subprocess.Popen([sys.executable, str(script)] + cmd[2:], **k))
        os.environ["FAKE_FINDINGS"] = json.dumps(["The 2K read waits for something.", self.SHARE])
        try:
            run.start(None)
            st = Runs.wait(run)
        finally:
            del os.environ["FAKE_FINDINGS"]
        self.assertEqual(st["state"], "done")
        self.assertEqual(st["report"]["findings"], [{"text": "The 2K read waits for something.", "settings": []},
                                                    {"text": self.SHARE, "settings": ["prefill_cpu_share", "prefill_cpu_share_max"]}])


class Engine:
    spawn = ("strata", ["--pack", "p", "--pcie-frac", "0.35", "--max-context", "8192", "--expert-profile-save", "/home/x/learned.bin"],
             None, "e.log", {"PATH": "/bin", "STRATA_PREFILL_CPU_SHARE": "0.5", "OPENAI_API_KEY": "sk-secret"}, False, None)
    info = {"pcie_frac": "0.55", "kv_resident": 32768, "expert_slots": 3880, "expert_cache_mib": 7586, "context": 262144}


class Settings(unittest.TestCase):
    def test_rows_from_spawn_config_and_engine(self):
        cfg = {"exe": "strata", "args": ["--pack", "p", "--pcie-frac", "0.2", "--vram-reserve-mib", "900"],
               "env": {"STRATA_PREFILL_CPU_SHARE_MAX": "2048"}, "expert_profile_save": "/home/x/learned.bin"}
        out = P.speed_settings(Engine(), True, cfg, "/x/strata-swift.json")
        self.assertEqual((out["file"], out["loaded"]), ("strata-swift.json", True))
        self.assertEqual([r["id"] for r in out["settings"]], [r[0] for r in P.SPEED_SETTINGS])
        rows = {r["id"]: r for r in out["settings"]}
        self.assertEqual((rows["pcie_frac"]["loaded_now"], rows["pcie_frac"]["next_load"], rows["pcie_frac"]["run_config"]),
                         ("0.55", "0.35", "0.2"))
        self.assertEqual(rows["expert_cache"]["loaded_now"], "3880 experts (7586 MiB)")
        self.assertEqual((rows["kv_resident"]["loaded_now"], rows["max_context"]["loaded_now"]), ("32768", "262144"))
        self.assertEqual(rows["max_context"]["next_load"], "8192")
        self.assertEqual(rows["vram_reserve_mib"]["edit"], "about")
        self.assertEqual((rows["vram_reserve_mib"]["next_load"], rows["vram_reserve_mib"]["run_config"]), (None, "900"))
        self.assertEqual((rows["prefill_cpu_share"]["next_load"], rows["prefill_cpu_share_max"]["run_config"]), ("0.5", "2048"))
        self.assertTrue(rows["pcie_frac"]["calibrated"])
        self.assertIsNone(rows["adapt_every"]["loaded_now"])                              # the engine does not report it
        self.assertEqual([r["id"] for r in out["settings"] if r["calibrated"]],
                         ["pcie_frac", "spec_min_p", "pool_workers", "adapt_every", "adapt_swaps", "adapt_decay"])
        # nothing but the speed settings: no path, no other variable
        self.assertEqual(rows["expert_profile_save"]["next_load"], "set")
        text = json.dumps(out)
        for secret in ("sk-secret", "learned.bin", "OPENAI_API_KEY"):
            self.assertNotIn(secret, text)

    def test_a_flag_the_server_adds_is_in_next_load_not_in_the_file(self):
        class VisionEngine:
            spawn = ("strata", ["--pack", "p", "--vision", "--vram-reserve-mib", "700"], None, "e.log", {}, False, None)
            info = {}
        cfg = {"exe": "strata", "args": ["--pack", "p"], "vision": {"gpu": True}}          # engine_args() adds the two flags
        row = {r["id"]: r for r in P.speed_settings(VisionEngine(), True, cfg, "/x/c.json")["settings"]}["vram_reserve_mib"]
        self.assertEqual((row["next_load"], row["run_config"]), ("700", None))              # so the tab words the file's value as a fact

    def test_unloaded_no_spawn_no_config(self):
        class Bare:
            info = Engine.info
        out = P.speed_settings(Bare(), False, None, None)
        self.assertEqual((out["file"], out["loaded"]), (None, False))
        for r in out["settings"]:
            self.assertIsNone(r["loaded_now"], r["id"])                                   # not loaded: nothing is reported
            self.assertIsNone(r["run_config"], r["id"])                                   # no config
            self.assertEqual(r["next_load"], r["run_config"], r["id"])
        out = P.speed_settings(Bare(), False, {"args": ["--pcie-frac", "0.2"], "expert_profile_save": "/p/x.bin"}, "/x/c.json")
        rows = {r["id"]: r for r in out["settings"]}
        self.assertEqual((rows["pcie_frac"]["next_load"], rows["pcie_frac"]["run_config"]), ("0.2", "0.2"))
        self.assertEqual((rows["expert_profile_save"]["next_load"], rows["expert_profile_save"]["run_config"]), ("set", "set"))
        class Replicas(Engine):
            engines = [Engine()]
        out = P.speed_settings(Replicas(), True, {"args": ["--pcie-frac", "0.2"]}, "/x/c.json")   # no start command of its own
        self.assertEqual({r["id"]: r for r in out["settings"]}["pcie_frac"]["next_load"], "0.2")


class TuneReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Path(self.tmp.name) / "strata-x.json"
        self.cfg.write_text(json.dumps({"exe": "x", "args": ["--pack", "p", "--spec-min-p", "0.5"]}), encoding="utf-8")

    def tune(self, rc):
        script = Path(self.tmp.name) / f"calibrate{rc}.py"
        script.write_text(TUNE_SCRIPT.format(config=str(self.cfg), rc=rc), encoding="utf-8")
        run = P.TuneRun(config_path=str(self.cfg), popen=lambda cmd, **k: subprocess.Popen([sys.executable, str(script)], **k))
        run.before = P.tune_snapshot(json.loads(self.cfg.read_text()), {"pcie_frac": "0.55", "spec_min_p": "0.50", "pool_workers": 7})
        run.start(None)
        return Runs.wait(run)

    def test_before_after_and_tok_s(self):
        self.assertEqual(self.cfg.exists(), True)
        before = P.tune_snapshot(json.loads(self.cfg.read_text()), {"pcie_frac": "0.55", "pool_workers": 7})
        self.assertEqual(before["args"]["--spec-min-p"], "0.5")
        self.assertIsNone(before["args"]["--pcie-frac"])
        st = self.tune(0)
        self.assertEqual(st["state"], "done")
        rep = st["report"]
        self.assertEqual(rep["tok_s"], 96.6)
        self.assertTrue(rep["summary"].endswith("(96.6 tok/s)"))
        rows = {c["flag"]: c for c in rep["changes"]}
        self.assertEqual(list(rows), ["--pcie-frac", "--spec-min-p", "--pool-workers", "--adapt-every", "--adapt-swaps", "--adapt-decay"])
        self.assertEqual(rows["--pcie-frac"], {"flag": "--pcie-frac", "before": None, "before_loaded": "0.55", "after": "0.35", "changed": True})
        self.assertEqual(rows["--spec-min-p"], {"flag": "--spec-min-p", "before": "0.5", "before_loaded": "0.50", "after": "0.70", "changed": True})
        self.assertEqual(rows["--pool-workers"], {"flag": "--pool-workers", "before": None, "before_loaded": "7", "after": None, "changed": False})
        self.assertEqual(sorted(f for f, c in rows.items() if c["changed"]),
                         ["--adapt-decay", "--adapt-every", "--adapt-swaps", "--pcie-frac", "--spec-min-p"])
        self.assertIsNone(rows["--adapt-every"]["before_loaded"])                            # the engine does not report it

    def test_numbers_are_compared_as_numbers(self):
        self.cfg.write_text(json.dumps({"exe": "x", "args": ["--pack", "p", "--spec-min-p", "0.50"]}), encoding="utf-8")
        script = Path(self.tmp.name) / "same.py"
        script.write_text(TUNE_SCRIPT.format(config=str(self.cfg), rc=0).replace('("--spec-min-p", "0.70")', '("--spec-min-p", "0.5")'),
                          encoding="utf-8")
        run = P.TuneRun(config_path=str(self.cfg), popen=lambda cmd, **k: subprocess.Popen([sys.executable, str(script)], **k))
        run.before = P.tune_snapshot(json.loads(self.cfg.read_text()), {})
        run.start(None)
        rows = {c["flag"]: c for c in Runs.wait(run)["report"]["changes"]}
        self.assertEqual((rows["--spec-min-p"]["before"], rows["--spec-min-p"]["after"], rows["--spec-min-p"]["changed"]),
                         ("0.50", "0.5", False))

    def test_the_table_does_not_need_the_summary_line(self):
        script = Path(self.tmp.name) / "quiet.py"
        script.write_text(TUNE_SCRIPT.format(config=str(self.cfg), rc=0).replace("tuned for this PC: ", "finished: "), encoding="utf-8")
        run = P.TuneRun(config_path=str(self.cfg), popen=lambda cmd, **k: subprocess.Popen([sys.executable, str(script)], **k))
        run.before = P.tune_snapshot(json.loads(self.cfg.read_text()), {})
        run.start(None)
        st = Runs.wait(run)
        self.assertEqual(st["state"], "done")
        self.assertEqual((st["report"]["summary"], st["report"]["tok_s"]), (None, None))
        self.assertTrue(any(c["changed"] for c in st["report"]["changes"]))

    def test_a_failed_run_has_no_report(self):
        st = self.tune(1)
        self.assertEqual((st["state"], st["returncode"], st["report"]), ("failed", 1, None))


class Page(unittest.TestCase):
    WEB = ROOT / "serve" / "web"
    IDS = """pf-off pf-notice pf-updated
        pf-last pf-last-empty pf-inflight pf-decode-tps pf-hit pf-pcie pf-prompt-tps pf-window pf-prompt-bar pf-prompt-legend
        pf-decode-bar pf-decode-legend pf-facts pf-kv
        pf-hist pf-hist-empty pf-spark-tps pf-spark-hit pf-spark-ms pf-hist-body
        pf-run pf-run-state pf-run-step pf-run-log pf-report pf-t-prompts pf-t-decode pf-t-agent pf-findings pf-report-path
        pf-tune pf-tune-state pf-tune-step pf-tune-log pf-tune-warn pf-tune-result pf-t-tune pf-tune-note
        pf-settings-body pf-off-link pf-profile-link""".split()

    def test_the_tab_is_on_the_page(self):
        html = (self.WEB / "index.html").read_text(encoding="utf-8")
        js = (self.WEB / "app.js").read_text(encoding="utf-8")
        button = re.search(r'<button[^>]*id="tab-btn-profile"[^>]*>', html)
        self.assertTrue(button)
        self.assertRegex(button.group(0), r"\shidden[\s>]")                                  # hidden until /health says "profile": true
        self.assertIn('id="view-profile"', html)
        for i in self.IDS:
            self.assertIn(f'id="{i}"', html, i)
            self.assertIn(f'"{i}"', js, i)
        for text in ("health.profile", '"profile/history"', '"profile/settings"', '"profile/last"', '"profile/status"'):
            self.assertIn(text, js)
        self.assertIn('$("tab-btn-profile").hidden = !health.profile;', js)                  # the only thing that shows it
        self.assertEqual(len(re.findall(r'\$\("tab-btn-profile"\)\.hidden\s*=', js)), 1)
        start = js.index("// ------------------------------------------------------------------ Profile (GET /profile/*)")
        end = js.index("\n// ----", start + 10)
        self.assertNotIn("innerHTML", js[start:end])
        self.assertNotIn("insertAdjacentHTML", js[start:end])
        self.assertIn("the run config file has", js)                                          # a fact, not a promise about the next start
        self.assertNotIn("used from the next start", js)
        self.assertIn("Largest part of the prompt read", html)
        for gone in ("profile.html", "profile.js"):
            self.assertFalse((self.WEB / gone).exists(), gone)
        self.assertNotIn("profile.js", html)

    def test_the_about_inputs_carry_their_key(self):
        js = (self.WEB / "app.js").read_text(encoding="utf-8")
        self.assertEqual(js.count('data-key="${esc(k.key)}"'), 2)                            # the select and the input


class ReloadableEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = True

    def alive(self):
        return self.running

    def unload(self):
        self.running = False

    def restart(self):
        self.running = True


class Http(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, "Hello.", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def request(self, path, body=None, headers=None, json_body=True):
        data = (json.dumps(body).encode() if json_body else body) if body is not None else None
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={**({"Content-Type": "application/json"} if json_body and body is not None else {}), **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def raw(self, path, headers=None):
        """One GET that is not redirected: (status, headers, body)."""
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        try:
            conn.request("GET", path, headers=headers or {})
            r = conn.getresponse()
            return r.status, r, r.read()
        finally:
            conn.close()

    def test_off_by_default(self):
        for path in ("/profile", "/profile/", "/profile/last", "/profile/status", "/profile/history", "/profile/settings"):
            self.assertEqual(self.request(path)[0], 404, path)
        for path in ("/profile/run", "/profile/tune"):
            self.assertEqual(self.request(path, {})[0], 404, path)
        self.assertNotIn("STRATA_DECODE_TIMING", child_env({"exe": "x", "args": []}))
        # the default /health keeps its keys: no "profile" while the tab is off
        status, health = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(set(health), {"status", "max_context", "model", "images", "api_key", "image_generation", "loaded", "service"})
        # and /metrics never reads the history
        for name in ("read", "poll"):
            with mock.patch.object(P.ProfileHistory, name, side_effect=AssertionError("the history was read while profile is off")):
                self.assertEqual(self.request("/metrics")[0], 200)

    def test_the_page_and_the_last_request(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        status, response, body = self.raw("/profile")                              # the page became the tab: a redirect
        self.assertEqual((status, response.getheader("Location"), response.getheader("Content-Length"), body),
                         (302, "./#profile", "0", b""))
        status, response, _ = self.raw("/profile/?x=1")                             # a trailing slash must not loop
        self.assertEqual((status, response.getheader("Location")), (302, "../#profile"))
        status, last = self.request("/profile/last")
        self.assertEqual((status, last["request"], last["log"]), (200, None, False))   # the mock engine has no log
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text("\n".join(REQUEST_1 + REQUEST_2) + "\n", encoding="utf-8")
            self.engine.log_path, self.engine.log_start = str(log), 0
            status, last = self.request("/profile/last")
            self.assertEqual(status, 200)
            self.assertEqual(last["request"]["decode"]["tok_s"], 93.7)
            self.assertEqual(last["request"]["prompt"]["groups"]["share"]["expert GEMM"] > 0.2, True)
            self.assertTrue(last["log"])
        status, st = self.request("/profile/status")
        self.assertEqual((status, st["state"]), (200, "idle"))

    def test_a_run_only_from_the_own_page(self):
        started = []
        run = P.ProfileRun(base=self.base)
        run.start = lambda log, long=None: (started.append((log, long)), "started")[1]
        self.svc.profile = run
        self.assertEqual(self.request("/profile/run", {}, {"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.request("/profile/run", b"x=1", {"Content-Type": "application/x-www-form-urlencoded"}, json_body=False)[0], 415)
        self.assertEqual(started, [])
        status, body = self.request("/profile/run?long=4096", {})
        self.assertEqual((status, body), (200, {"status": "started"}))
        self.assertEqual(started, [(None, 4096)])
        self.assertEqual(self.request("/profile/run?long=abc", {})[0], 400)
        self.assertEqual(self.request("/profile/run?long=-5", {})[0], 200)
        self.assertEqual(started[-1], (None, 0))                                   # clamped
        run.start = lambda log, long=None: "running"
        status, body = self.request("/profile/run", {})
        self.assertEqual((status, body["error"]["message"]), (409, "a profile run is still going"))

    def test_tune_unloads_the_model_and_runs_the_calibration(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        self.assertEqual(self.request("/profile/tune", {})[0], 404)                 # no run config known (tune is None)
        started = []
        self.svc.tune = P.TuneRun(config_path="c.json")
        self.svc.tune.start = lambda log, long=None: (started.append(1), "started")[1]
        self.assertEqual(self.request("/profile/tune", {}, {"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.request("/profile/tune", {}), (200, {"status": "started"}))
        self.assertEqual((started, self.engine.running, self.svc.tuning), ([1], False, True))   # the model was unloaded for it
        # while it runs: no engine load (503 + Retry-After), no profile run, no settings save, no second tuning
        status, body = self.request("/load", {})
        self.assertEqual(status, 503)
        self.assertIn("tuning run", body["error"]["message"])
        self.assertEqual(self.request("/profile/run", {})[0], 409)
        self.assertEqual(self.request("/profile/tune", {})[0], 409)
        self.svc.config_path = "x.json"
        self.assertEqual(self.request("/config", {"set": {"lazy_load": True}})[0], 409)
        self.svc.tuning_ended()
        self.assertFalse(self.svc.tuning)
        self.assertEqual(self.request("/load", {})[0], 200)                        # loads again once it ended
        self.assertTrue(self.engine.running)
        self.svc.unload = lambda: "busy"
        status, body = self.request("/profile/tune", {})
        self.assertEqual((status, body["status"], self.svc.tuning), (409, "busy", False))
        self.assertEqual(started, [1])                                               # not started with a request running
        status, st = self.request("/profile/status")
        self.assertEqual((status, st["tune"]["state"], st["tune_unavailable"]), (200, "idle", None))
        self.svc.tune, self.svc.tune_unavailable = None, "replicas are not tuned from the page: run setup --calibrate"
        status, body = self.request("/profile/tune", {})
        self.assertEqual((status, body["error"]["message"]), (404, "replicas are not tuned from the page: run setup --calibrate"))
        self.assertEqual(self.request("/profile/status")[1]["tune_unavailable"], "replicas are not tuned from the page: run setup --calibrate")

    def test_the_api_key_guards_the_endpoints(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        self.svc.api_key = "sesame"
        self.assertEqual(self.request("/profile/last")[0], 401)
        self.assertEqual(self.request("/profile/status", headers={"Authorization": "Bearer sesame"})[0], 200)
        self.assertEqual(self.raw("/profile")[0], 302)                           # the redirect needs no key, as the page did
        for path in ("/profile/history", "/profile/settings"):
            self.assertEqual(self.request(path)[0], 401, path)
            self.assertEqual(self.request(path, headers={"Authorization": "Bearer sesame"})[0], 200, path)

    # ---- the history and settings routes
    def test_history_route_shape(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text("\n".join(REQUEST_1 + REQUEST_2) + "\n", encoding="utf-8")
            self.engine.log_path, self.engine.log_start = str(log), 0
            status, body = self.request("/profile/history")
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"max", "engine_run", "requests"})
        self.assertEqual((body["max"], body["engine_run"], len(body["requests"])), (50, 1, 2))
        self.assertEqual(set(body["requests"][0]), {
            "n", "engine_run", "seen_at", "open", "prompt_tokens", "reused", "read", "prompt_tok_s", "prompt_ms", "prompt_gpu_ms",
            "prompt_top", "generated", "decode_tok_s", "decode_ms", "ms_per_window", "tokens_per_window", "ms", "hit_rate",
            "pcie_share", "kv_hit"})
        self.assertEqual(set(body["requests"][1]["ms"]), {"GPU", "CPU experts", "draft", "adapt wait", "other"})
        self.assertEqual(self.request("/profile/history")[1]["requests"], body["requests"])   # no log of its own: same answer

    def test_the_standalone_page_files_are_gone(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        for path in ("/web/profile.js", "/web/profile.html"):
            self.assertEqual(self.request(path)[0], 404, path)

    def test_lines_written_during_a_tune_never_become_entries(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text("\n".join(REQUEST_1) + "\n", encoding="utf-8")
            self.engine.log_path, self.engine.log_start = str(log), 0
            self.assertEqual(len(self.request("/profile/history")[1]["requests"]), 1)
            self.svc.tuning = True                                                           # the calibration's engines write to the log
            with open(log, "a", encoding="utf-8") as f:
                f.write("\n".join(request_lines(5) + request_lines(6)) + "\n")
            self.assertEqual(self.request("/metrics")[0], 200)                               # the 1 s poll
            self.assertEqual(len(self.request("/profile/history")[1]["requests"]), 1)
            self.svc.tuning_ended()                                                          # a poll after the end, before the next load
            rows = self.request("/profile/history")[1]["requests"]
            self.assertEqual([(r["n"], r["open"]) for r in rows], [(1, False)])
            with open(log, "a", encoding="utf-8") as f:
                f.write("\n".join(request_lines(7)) + "\n")                                  # the user's next request
            rows = self.request("/profile/history")[1]["requests"]
            self.assertEqual([(r["n"], r["prompt_tokens"]) for r in rows], [(1, 1889), (2, 1007)])

    def test_health_says_profile_when_on(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        status, health = self.request("/health")
        self.assertEqual((status, health["profile"]), (200, True))
        self.assertEqual(list(health)[-1], "profile")

    def test_metrics_feeds_the_history_when_on(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        with mock.patch.object(P.ProfileHistory, "poll", return_value=None) as poll:
            self.assertEqual(self.request("/metrics")[0], 200)
        self.assertEqual(poll.call_count, 1)
        self.assertEqual(poll.call_args.args, (None, None))                          # the mock engine has no log

    def test_the_settings_route(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        status, body = self.request("/profile/settings")
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"file", "loaded", "settings"})
        self.assertEqual([r["id"] for r in body["settings"]], [r[0] for r in P.SPEED_SETTINGS])
        self.assertEqual(set(body["settings"][0]), {"id", "name", "kind", "loaded_now", "next_load", "run_config", "calibrated", "edit", "about"})
        self.assertIsNone(body["file"])                                              # started without a run config

    # ---- the tuning's before and after
    def test_the_snapshot_is_taken_before_the_unload(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "strata-x.json"
            cfg.write_text(json.dumps({"exe": "x", "args": ["--spec-min-p", "0.5"]}), encoding="utf-8")
            script = Path(tmp) / "calibrate.py"
            script.write_text(TUNE_SCRIPT.format(config=str(cfg), rc=0), encoding="utf-8")
            self.svc.profile = P.ProfileRun(base=self.base)
            self.svc.config_path = str(cfg)
            self.svc.tune = P.TuneRun(config_path=str(cfg), on_end=self.svc.tuning_ended,
                                      popen=lambda cmd, **k: subprocess.Popen([sys.executable, str(script)], **k))
            self.engine.info = {"pcie_frac": "0.55", "pool_workers": 7}
            unload = self.engine.unload

            def unload_and_forget():                                                   # the engine's report goes with the engine
                self.engine.info = {}
                unload()
            self.engine.unload = unload_and_forget
            self.assertEqual(self.request("/profile/tune", {}), (200, {"status": "started"}))
            self.assertEqual(self.engine.info, {})
            st = Runs.wait(self.svc.tune)
            self.assertEqual(st["state"], "done")
            rows = {c["flag"]: c for c in st["report"]["changes"]}
            self.assertEqual((rows["--pcie-frac"]["before_loaded"], rows["--pool-workers"]["before_loaded"]), ("0.55", "7"))
            status, body = self.request("/profile/status")
            self.assertEqual(body["tune"]["report"]["tok_s"], 96.6)


class ConfigKey(unittest.TestCase):
    def test_profile_is_an_editable_key(self):
        new, changed = runconfig.apply({"exe": "x", "args": []}, {"profile": True})
        self.assertEqual((changed, new["profile"]), (["profile"], True))
        new, changed = runconfig.apply(new, {"profile": None})
        self.assertNotIn("profile", new)
        help_text = dict((k, h) for k, _, h in runconfig.EDITABLE)["profile"]
        self.assertEqual(help_text, "Start the engine with its timing switches and show the Profile tab (where the time goes)")


if __name__ == "__main__":
    unittest.main()
