"""Several models behind one port (serve/model_host.py, server.py --models): the models file, routing by model name,
switching (drain, unload, load, rollback) and the HTTP answers, with fake engines (no GPU)."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import model_host as mh
from serve.frontend import ChatTemplate
from serve.server import (BadBody, ByteTokenizer, EngineDied, MockEngine, Server, Service, StrataEngine,
                          host_names_for, make_handler)

TEMPLATE = Path(__file__).parent / "chat_template.jinja"


class FakeEngine(StrataEngine):
    """An engine that starts and stops on request and answers with a fixed text (slowly, when asked)."""

    def __init__(self, tok, text, delay_s=0.0):
        self.reply = MockEngine(tok, text, max_context=4096, delay_s=delay_s)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.running, self.starts, self.closes = False, 0, 0
        self.unloaded = True
        self.progress = None
        self.fail_start = None

    def alive(self):
        return self.running

    def exit_code(self):
        return None

    def restart(self):
        if self.fail_start is not None:
            raise self.fail_start
        self.running = True
        self.starts += 1
        self.unloaded = False

    def close(self):
        self.running = False
        self.closes += 1

    def unload(self):
        self.close()
        self.unloaded = True

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


def member(name, text, delay_s=0.0, aliases=None, key="", activate=None):
    tok = ByteTokenizer()
    eng = FakeEngine(tok, text, delay_s)
    svc = Service(eng, tok, ChatTemplate(TEMPLATE), model_name=name)
    if aliases:
        svc.set_aliases(aliases)
    svc.api_key = key
    svc.host_names = host_names_for("127.0.0.1", svc.allowed_hosts, svc.trusted_origins)
    return mh.Member(svc, make_handler(svc), activate=activate)


class Base(unittest.TestCase):
    def start(self, members, **kw):
        self.host = mh.ModelHost(members, **kw)
        self.httpd = Server(("127.0.0.1", 0), mh.make_router_handler(self.host, bad_body=BadBody))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def request(self, path, body=None, headers=None, timeout=20):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            ctype = response.headers.get("Content-Type", "")
            return response.status, (data if "event-stream" in ctype else json.loads(data)), response.headers

    def chat(self, model, stream=False, **kw):
        body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64, "reasoning_effort": "none",
                "stream": stream}
        if model is not None:
            body["model"] = model
        return self.request("/v1/chat/completions", body, **kw)


class ModelsFile(unittest.TestCase):
    def write(self, obj):
        d = tempfile.mkdtemp()
        p = Path(d) / "models.json"
        p.write_text(json.dumps(obj), encoding="utf-8")
        return p

    def test_paths_defaults_and_refusals(self):
        p = self.write({"models": [{"config": "a.json"}, {"config": "/abs/b.json", "args": ["--parallel", "2"]}]})
        spec = mh.load_models_file(p)
        self.assertEqual(spec["models"][0]["config"], str(p.parent / "a.json"))
        self.assertEqual(spec["models"][1], {"config": "/abs/b.json", "args": ["--parallel", "2"]})
        self.assertIs(spec["default"], ...)
        self.assertEqual((spec["auto_swap"], spec["swap_wait_s"], spec["settle_s"]), (False, 120.0, 0.0))
        self.assertIsNone(mh.load_models_file(self.write({"models": [{"config": "a"}], "default": None}))["default"])
        for bad in ({"models": []}, {"models": [{"config": ""}]}, {"models": [{"config": "a"}], "auto_swap": "yes"},
                    {"models": [{"config": "a", "args": ["--port", "1"]}]}, {"models": [{"config": "a"}], "x": 1},
                    {"models": [{"config": "a", "gpu": 0}]}, {"models": [{"config": "a"}], "swap_wait_s": -1}):
            with self.subTest(bad=bad), self.assertRaises(mh.ModelsFileError):
                mh.load_models_file(self.write(bad))

    def test_member_argv_drops_process_only_arguments(self):
        argv = ["--engine", "strata", "--models", "m.json", "--host", "0.0.0.0", "--port", "8081", "--offline",
                "--default-model=none", "--lazy"]
        self.assertEqual(mh.member_argv(argv, {"config": "/c/a.json", "args": ["--parallel", "2"]}),
                         ["--engine", "strata", "--host", "0.0.0.0", "--port", "8081", "--offline",
                          "--config", "/c/a.json", "--lazy", "--parallel", "2"])


class Registry(unittest.TestCase):
    def test_names_aliases_default_and_keys(self):
        a, b = member("qwen-a", "A."), member("ink-b", "B.", aliases=["inkling"])
        host = mh.ModelHost([a, b])
        self.assertEqual(host.default, "qwen-a")
        self.assertIs(host.find("INKLING"), b)
        self.assertIs(host.route("gpt-4"), a)           # a name no model has: the current (here: default) one
        with self.assertRaises(mh.ModelsFileError):
            mh.ModelHost([member("x", "1"), member("X", "2")])
        with self.assertRaises(mh.ModelsFileError):
            mh.ModelHost([member("x", "1"), member("y", "2")], default="z")
        with self.assertRaises(mh.ModelsFileError):
            mh.ModelHost([member("x", "1", key="k1"), member("y", "2", key="k2")])
        self.assertIsNone(mh.ModelHost([member("x", "1")], default=None).default)


class Switching(Base):
    def setUp(self):
        self.a, self.b = member("qwen-a", "Answer from A."), member("ink-b", "Answer from B.", aliases=["inkling"])

    def test_requests_go_to_their_model_and_load_switches(self):
        self.start([self.a, self.b])
        code, body, _ = self.chat("qwen-a")
        self.assertEqual((code, body["choices"][0]["message"]["content"]), (200, "Answer from A."))
        self.assertEqual(self.host.active, "qwen-a")
        code, body, _ = self.chat("inkling")             # configured, not loaded, auto_swap off: a clear refusal
        self.assertEqual((code, body["error"]["code"]), (409, "model_not_loaded"))
        self.assertEqual(self.b.svc.engine.starts, 0)
        code, body, _ = self.request("/v1/load", {"model": "inkling"})
        self.assertEqual((code, body["model"], body["from"]), (200, "ink-b", "qwen-a"))
        self.assertEqual((self.a.svc.engine.closes, self.b.svc.engine.starts), (1, 1))
        self.assertEqual(self.chat("ink-b")[1]["choices"][0]["message"]["content"], "Answer from B.")
        self.assertEqual(self.chat("gpt-4")[1]["choices"][0]["message"]["content"], "Answer from B.")
        listed = {m["id"]: m["status"]["value"] for m in self.request("/v1/models")[1]["data"]}
        self.assertEqual(listed, {"qwen-a": "unloaded", "ink-b": "loaded", "inkling": "loaded"})
        self.assertEqual(self.request("/health?model=qwen-a")[1]["model"], "qwen-a")
        self.assertEqual(self.request("/health")[1]["model"], "ink-b")
        code, body, _ = self.request("/v1/unload", {"model": "ink-b"})
        self.assertEqual((code, body["status"]), (200, "unloaded"))
        self.assertIsNone(self.host.active)
        self.assertEqual(self.chat("qwen-a")[0], 200)    # nothing holds the GPU: a request loads its model

    def test_auto_swap_switches_for_the_request(self):
        self.start([self.a, self.b], auto_swap=True)
        self.assertEqual(self.chat("qwen-a")[0], 200)
        code, body, _ = self.chat("ink-b")
        self.assertEqual((code, body["choices"][0]["message"]["content"]), (200, "Answer from B."))
        self.assertEqual((self.a.svc.engine.running, self.b.svc.engine.running), (False, True))

    def test_a_failed_load_brings_the_previous_model_back(self):
        self.start([self.a, self.b])
        self.assertEqual(self.chat("qwen-a")[0], 200)
        self.b.svc.engine.fail_start = EngineDied("no memory")
        code, body, _ = self.request("/v1/load", {"model": "ink-b"})
        self.assertEqual((code, body["error"]["code"]), (503, "model_load_failed"))
        self.assertIn("qwen-a is loaded again", body["error"]["message"])
        self.assertEqual((self.host.active, self.a.svc.engine.running, self.a.svc.engine.starts), ("qwen-a", True, 2))
        self.assertEqual(self.chat("qwen-a")[0], 200)

    def test_a_switch_waits_for_running_requests_and_refuses_new_ones(self):
        slow = member("qwen-a", "x" * 40, delay_s=0.02)   # ~0.8 s per answer
        self.start([slow, self.b], swap_wait_s=30)
        self.assertEqual(self.chat("qwen-a")[0], 200)
        results, done = [], []

        def stream():
            results.append(self.chat("qwen-a", stream=True))

        def switch():
            done.append(self.request("/v1/load", {"model": "ink-b"}, timeout=30))
        streams = [threading.Thread(target=stream) for _ in range(2)]   # one runs, one queues: both counted
        for t in streams:
            t.start()
        deadline = time.time() + 5
        while slow.inflight < 2 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(slow.inflight, 2)
        sw = threading.Thread(target=switch)
        sw.start()
        while self.host.switching is None and time.time() < deadline:
            time.sleep(0.01)
        code, body, headers = self.chat("qwen-a")       # the old model during the switch: 503 with Retry-After
        self.assertEqual((code, body["error"]["code"], headers.get("Retry-After")), (503, "model_switching", "5"))
        for t in streams:
            t.join(20)
        sw.join(30)
        self.assertEqual([r[0] for r in results], [200, 200])
        self.assertTrue(all("data: [DONE]" in r[1] for r in results))   # both streams ran to their end
        self.assertEqual(done[0][0], 200)
        self.assertEqual((slow.svc.engine.running, self.b.svc.engine.running, slow.inflight), (False, True, 0))

    def test_a_switch_that_cannot_drain_in_time_changes_nothing(self):
        slow = member("qwen-a", "y" * 100, delay_s=0.02)  # ~2 s
        self.start([slow, self.b], swap_wait_s=0.3)
        self.assertEqual(self.chat("qwen-a")[0], 200)
        t = threading.Thread(target=lambda: self.chat("qwen-a"))
        t.start()
        deadline = time.time() + 5
        while slow.inflight < 1 and time.time() < deadline:
            time.sleep(0.01)
        code, body, _ = self.request("/v1/load", {"model": "ink-b"})
        self.assertEqual((code, body["error"]["code"]), (409, "model_busy"))
        t.join(10)
        self.assertEqual((self.host.active, self.host.switching, self.b.svc.engine.starts), ("qwen-a", None, 0))

    def test_load_without_a_model_and_keys(self):
        a, b = member("qwen-a", "A.", key="k"), member("ink-b", "B.", key="k")
        self.start([a, b])
        self.assertEqual(self.request("/v1/models")[0], 401)
        self.assertEqual(self.request("/v1/load", {"model": "ink-b"})[0], 401)
        auth = {"Authorization": "Bearer k"}
        self.assertEqual(self.request("/v1/load", {}, auth)[0], 200)      # no model: the current one, as before
        self.assertTrue(a.svc.engine.running)
        self.assertEqual(self.request("/v1/load", {"model": "nope"}, auth)[0], 404)
        self.assertEqual(self.request("/v1/load", {"model": "ink-b"}, {**auth, "Content-Type": "text/plain"})[0], 415)


class ProcessSettings(Base):
    """A run config also sets serve settings that live in modules, not in its Service (the model family's effort words,
    env_to_user, engine_ready_s): the model that holds the GPU must be the one whose settings are in force."""

    def test_the_model_that_takes_the_gpu_brings_its_settings(self):
        seen = []
        a = member("qwen-a", "Answer from A.", activate=lambda: seen.append("qwen-a"))
        b = member("ink-b", "Answer from B.", activate=lambda: seen.append("ink-b"))
        self.start([a, b])
        self.assertEqual(seen, [])
        self.assertEqual(self.chat("qwen-a")[0], 200)    # nothing held the GPU: the request's model takes it
        self.assertEqual(self.chat("qwen-a")[0], 200)    # already its own: nothing to apply again
        self.assertEqual(seen, ["qwen-a"])
        self.assertEqual(self.request("/v1/load", {"model": "ink-b"})[0], 200)
        self.assertEqual(seen, ["qwen-a", "ink-b"])
        a.svc.engine.fail_start = EngineDied("no memory")
        self.assertEqual(self.request("/v1/load", {"model": "qwen-a"})[0], 503)
        self.assertEqual(seen, ["qwen-a", "ink-b", "qwen-a", "ink-b"])   # the failed load, then ink-b back
        self.assertEqual(self.chat("ink-b")[1]["choices"][0]["message"]["content"], "Answer from B.")

    def test_each_model_is_built_from_a_fresh_servers_settings(self):
        from serve import frontend, server
        d = Path(tempfile.mkdtemp())
        (d / "alpha.json").write_text(json.dumps({"model_name": "alpha", "env_to_user": True}), encoding="utf-8")
        (d / "beta.json").write_text(json.dumps({"model_name": "beta"}), encoding="utf-8")
        base = server._process_state()
        argv = ["--engine", "mock", "--port", "0", "--tokenizer", str(d / "none")]
        built = []
        try:
            for name in ("alpha", "beta"):
                built.append(server._build_member(argv, {"config": str(d / f"{name}.json"), "args": []}, base))
            (_, _, alpha), (_, _, beta) = built
            self.assertEqual(alpha, {**base, "env_to_user": True})
            self.assertEqual(beta, base)                 # not alpha's env_to_user
            server._apply_process_state({**alpha, "engine_ready_s": 50.0})   # (the mock engine reads no engine_ready_s)
            self.assertEqual((frontend.ENV_TO_USER, server.ENGINE_READY_S), (True, 50.0))
            server._apply_process_state({**base, "family": "inkling2"})
            self.assertEqual(frontend.effort_kwargs(False), {"reasoning_effort": "none"})
            server._apply_process_state(beta)
            self.assertEqual(frontend.effort_kwargs(False), {"enable_thinking": False})
            self.assertEqual((frontend.ENV_TO_USER, server.ENGINE_READY_S), (base["env_to_user"], base["engine_ready_s"]))
        finally:
            server._apply_process_state(base)
            for _, closers, _ in built:
                for close in closers:
                    close()


class CommandLine(unittest.TestCase):
    """server.py --models with the mock engine: the models file, members built by _main, one port."""

    def test_models_server_end_to_end(self):
        d = Path(tempfile.mkdtemp())
        for name in ("alpha", "beta"):
            (d / f"{name}.json").write_text(json.dumps({"model_name": name}), encoding="utf-8")
        (d / "models.json").write_text(json.dumps({"models": [{"config": "alpha.json"}, {"config": "beta.json"}],
                                                   "default": None}), encoding="utf-8")
        root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "STRATA_BLOCKING_CONSOLE": "1", "PYTHONPATH": str(root)}
        proc = subprocess.Popen([sys.executable, str(root / "serve" / "server.py"), "--engine", "mock", "--models",
                                 str(d / "models.json"), "--port", "0", "--tokenizer", str(d / "none")],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(d))
        try:
            port = None
            deadline = time.time() + 60
            lines = []
            while time.time() < deadline:
                line = proc.stdout.readline()
                if not line:
                    break
                lines.append(line)
                if line.startswith("ready: http://"):
                    port = int(line.split("/v1", 1)[0].rsplit(":", 1)[1])
                    break
            self.assertIsNotNone(port, "".join(lines))
            self.assertIn("models: alpha, beta; loaded: none", lines[-1])

            def req(path, body=None):
                r = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                           data=json.dumps(body).encode() if body is not None else None,
                                           headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(r, timeout=20) as resp:
                        return resp.status, json.loads(resp.read())
                except urllib.error.HTTPError as e:
                    return e.code, json.loads(e.read())
            self.assertEqual([m["id"] for m in req("/v1/models")[1]["data"]], ["alpha", "beta"])
            chat = {"model": "beta", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 32}
            code, body = req("/v1/chat/completions", chat)
            self.assertEqual((code, body["model"]), (200, "beta"))
            self.assertEqual(req("/v1/chat/completions", {**chat, "model": "alpha"})[1]["error"]["code"],
                             "model_not_loaded")
        finally:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
