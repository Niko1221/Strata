"""serve/test_profile.py - the Profile page (serve/profile.py): off by default, the engine's switches, the last
request's breakdown from the engine log, and a run of the tool through HTTP against the mock engine.  No GPU.

    python -m unittest serve.test_profile
"""
from __future__ import annotations

import json
import os
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
        self.assertEqual(run.find_report(["  Loading ...", "  [ok] tuned for this PC: --pcie-frac 0.35, --spec-min-p 0.70 (61.2 tok/s)"]),
                         {"summary": "--pcie-frac 0.35, --spec-min-p 0.70 (61.2 tok/s)"})
        self.assertIsNone(run.find_report(["the tuning did not finish"]))

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

    def test_off_by_default(self):
        for path in ("/profile", "/profile/last", "/profile/status"):
            self.assertEqual(self.request(path)[0], 404, path)
        self.assertEqual(self.request("/profile/run", {})[0], 404)
        self.assertNotIn("STRATA_DECODE_TIMING", child_env({"exe": "x", "args": []}))

    def test_the_page_and_the_last_request(self):
        self.svc.profile = P.ProfileRun(base=self.base)
        status, page = self.request("/profile")
        self.assertEqual(status, 200)
        self.assertIn("<title>Strata Profile</title>", page)
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
        self.assertEqual(self.request("/profile")[0], 200)                       # the page itself, like /api-monitor


class ConfigKey(unittest.TestCase):
    def test_profile_is_an_editable_key(self):
        new, changed = runconfig.apply({"exe": "x", "args": []}, {"profile": True})
        self.assertEqual((changed, new["profile"]), (["profile"], True))
        new, changed = runconfig.apply(new, {"profile": None})
        self.assertNotIn("profile", new)


if __name__ == "__main__":
    unittest.main()
