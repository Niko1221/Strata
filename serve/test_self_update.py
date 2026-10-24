"""The web page's Updates card (serve/self_update.py): check, install, the restart in place, over the real HTTP API."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import self_update
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

CHECK = {"schema_version": 1, "ok": True,
         "engine": {"installed": {"version": "0.1.47"}, "offered": {"version": "0.1.48"}, "status": "available",
                    "download_bytes": 198405760},
         "control": {"installed": "0.1.47", "offered": "0.1.47", "status": "current"}, "serving": []}


def fake_strata(folder: str, final: dict | None, code: int = 0, check: dict = CHECK) -> str:
    """A `strata` that answers --check with `check` and an update with two events, then `final` (none: an error)."""
    path = Path(folder) / "strata"
    events = [{"event": "phase", "phase": "test", "message": "Testing the new engine ..."},
              {"event": "progress", "label": "engine 0.1.48", "percent": 50, "done": 1, "total": 2}]
    lines = [json.dumps(e) for e in events] + ([json.dumps({"schema_version": 1, **final})] if final else [])
    path.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, sys
        open({str(Path(folder) / 'args')!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")
        if "--check" in sys.argv:
            print(json.dumps({check!r}))
            sys.exit(0)
        for line in {lines!r}:
            print(line, flush=True)
        if {final is None!r}:
            print(json.dumps({{"schema_version": 1, "ok": False, "error": {{"code": 6, "message": "the feed is gone"}}}}))
            print("strata: the feed is gone", file=sys.stderr)
        sys.exit({code})
        """))
    path.chmod(0o755)
    return str(path)


def updated(component="engine"):
    return {"exit_code": 0, "results": [{"component": component, "status": "updated",
                                         "installed_before": {"version": "0.1.47"}, "installed_after": {"version": "0.1.48"}}]}


@unittest.skipUnless(os.name == "posix", "the update and restart in place are POSIX only")
class SelfUpdateHttp(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hello."), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def use(self, final, code=0, reason=None, config=None):
        self.svc.updater = self_update.SelfUpdate(self.svc, strata=fake_strata(self.dir, final, code), reason=reason,
                                                  config=config)

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def settle(self):
        for _ in range(200):
            view = self.request("/update")[1]
            if view["state"] not in ("checking", "updating"):
                return view
            time.sleep(0.05)
        self.fail(f"still {view['state']}")

    def chat(self):
        return self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8,
                                                     "reasoning_effort": "none"})

    def test_check_reports_what_would_change_and_changes_nothing(self):
        self.use(None)
        self.assertEqual(self.request("/update", {"action": "check"})[0], 202)
        view = self.settle()
        self.assertEqual(view["state"], "idle")
        self.assertTrue(view["check"]["available"])
        self.assertEqual((view["check"]["engine"]["installed"], view["check"]["engine"]["offered"]), ("0.1.47", "0.1.48"))
        self.assertEqual(json.loads((Path(self.dir) / "args").read_text()), ["update", "--check", "--output", "json"])
        self.assertEqual(self.chat()[0], 200)

    def test_an_update_drains_runs_with_the_run_config_and_restarts(self):
        self.use(updated(), config="strata-x.json")
        self.assertEqual(self.request("/update", {"action": "install"})[0], 202)
        view = self.settle()
        self.assertEqual(view["state"], "restarting")
        self.assertTrue(self.svc.updater.restart_wanted)
        self.assertEqual(view["results"], [{"component": "engine", "status": "updated", "error": None,
                                            "before": "0.1.47", "after": "0.1.48"}])
        self.assertIn("Downloading engine 0.1.48: 50%", view["log"])
        args = json.loads((Path(self.dir) / "args").read_text())
        self.assertEqual(args, ["update", "--output", "jsonl", "--model-config", str(Path("strata-x.json").resolve())])
        code, body = self.chat()                    # stopping: no new work until the new start
        self.assertEqual((code, body["error"]["code"]), (503, "server_draining"))

    def test_nothing_to_update_serves_again_without_a_restart(self):
        self.use({"exit_code": 0, "results": [{"component": "engine", "status": "current"}]})
        self.request("/update", {"action": "install"})
        view = self.settle()
        self.assertEqual(view["state"], "idle")
        self.assertFalse(self.svc.updater.restart_wanted)
        self.assertEqual(self.chat()[0], 200)

    def test_a_failed_update_says_why_and_serves_again(self):
        self.use(None, code=6)
        self.request("/update", {"action": "install"})
        view = self.settle()
        self.assertEqual(view["state"], "failed")
        self.assertIn("the feed is gone", view["message"])
        self.assertFalse(self.svc.updater.restart_wanted)
        self.assertEqual(self.chat()[0], 200)

    def test_a_failed_component_is_named(self):
        self.use({"exit_code": 6, "results": [{"component": "engine", "status": "failed", "error": "test start failed"}]},
                 code=6)
        self.request("/update", {"action": "install"})
        view = self.settle()
        self.assertEqual(view["state"], "failed")
        self.assertIn("engine: test start failed", view["message"])
        self.assertEqual(self.chat()[0], 200)

    def test_an_unsupported_server_refuses_and_only_json_from_its_own_page_starts_anything(self):
        self.use(updated(), reason="This server was started by Strata's app")
        view = self.request("/update")[1]
        self.assertEqual((view["supported"], view["reason"]), (False, "This server was started by Strata's app"))
        self.assertEqual(self.request("/update", {"action": "install"})[0], 409)
        self.use(updated())
        self.assertEqual(self.request("/update", {"action": "install"}, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("/update", {"action": "install"}, {"Origin": "http://evil.example.com"})[0], 403)
        self.assertEqual(self.request("/update", {"action": "reboot"})[0], 400)
        self.assertFalse((Path(self.dir) / "args").exists())


class Reasons(unittest.TestCase):
    def test_reasons(self):
        ok = dict(supervised=False, engine_kind="strata", exe="component:strata-engine", strata="/bin/strata", environ={})
        if os.name != "posix":
            self.assertIn("Windows", self_update.unsupported_reason(**ok))
            return
        self.assertIsNone(self_update.unsupported_reason(**ok))
        self.assertIsNone(self_update.unsupported_reason(**{**ok, "engine_kind": "mock", "exe": None}))
        self.assertIn("app", self_update.unsupported_reason(**{**ok, "supervised": True}))
        self.assertIn("engine/strata", self_update.unsupported_reason(**{**ok, "exe": "engine/strata"}))
        self.assertIn("not installed", self_update.unsupported_reason(**{**ok, "strata": None}))
        self.assertIn("STRATA_NO_ACQUIRE", self_update.unsupported_reason(**{**ok, "environ": {"STRATA_NO_ACQUIRE": "1"}}))


@unittest.skipUnless(os.name == "posix", "exec in place is POSIX only")
class Restart(unittest.TestCase):
    def test_the_same_process_starts_again_with_its_arguments_environment_and_folder(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "again.py"
            script.write_text(textwrap.dedent(f"""\
                import json, os, sys
                sys.path.insert(0, {str(root)!r})
                from serve import self_update
                if os.environ.get("ROUND") == "2":
                    print(json.dumps({{"pid": os.getpid(), "argv": sys.argv[1:], "cwd": os.getcwd(), "x": os.environ["X"]}}))
                    sys.exit(0)
                print(os.getpid(), flush=True)
                env = dict(os.environ, ROUND="2")
                cwd = os.getcwd()
                os.chdir("/")
                os.environ["X"] = "changed after the start"
                self_update.restart({{**env, "X": "as started"}}, cwd)
                """))
            done = subprocess.run([sys.executable, str(script), "--port", "8876"], cwd=tmp, capture_output=True, text=True,
                                  timeout=60, env={**os.environ, "X": "as started"})
            first, second = done.stdout.strip().splitlines()
            again = json.loads(second)
            self.assertEqual(again["pid"], int(first))
            self.assertEqual(again["argv"], ["--port", "8876"])
            self.assertEqual(os.path.realpath(again["cwd"]), os.path.realpath(tmp))
            self.assertEqual(again["x"], "as started")


if __name__ == "__main__":
    unittest.main()
