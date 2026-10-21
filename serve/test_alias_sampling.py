"""Defaults per model alias (`alias_sampling`): request > shared defaults (POST /settings) > alias > config sampling."""
import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.test_parallel import FAKE_BATCH  # noqa: E402


def service(alias_sampling=None):
    from serve.server import ByteTokenizer, ChatTemplate, Service
    svc = Service(None, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
    svc.model = "m"
    svc.alias_sampling = alias_sampling or {}
    return svc


class Parse(unittest.TestCase):
    def test_parsed(self):
        from serve.server import parse_alias_sampling
        self.assertEqual(parse_alias_sampling({"m-uncensored": {"experimental_speed_projection": True}}, "m"),
                         {"m-uncensored": {"experimental_speed_projection": True}})
        self.assertEqual(parse_alias_sampling({}, "m"), {})

    def test_refused(self):
        from serve.server import parse_alias_sampling
        for bad in ([], {"m": {"top_k": 20}}, {"": {"top_k": 20}}, {"x": {"temperature": "hot"}},
                    {"x": {"unknown": 1}}, {"x": []}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_alias_sampling(bad, "m")


class WithShared(unittest.TestCase):
    def test_alias_fills_a_missing_field(self):
        svc = service({"m-u": {"experimental_speed_projection": True}})
        self.assertIs(svc.with_shared({"model": "m-u"}, "openai")["experimental_speed_projection"], True)
        self.assertNotIn("experimental_speed_projection", svc.with_shared({"model": "m"}, "openai"))

    def test_request_wins(self):
        svc = service({"m-u": {"experimental_speed_projection": True, "temperature": 1.0}})
        req = svc.with_shared({"model": "m-u", "experimental_speed_projection": False, "temperature": 0}, "openai")
        self.assertIs(req["experimental_speed_projection"], False)
        self.assertEqual(req["temperature"], 0)

    def test_shared_defaults_win_over_alias(self):
        svc = service({"m-u": {"temperature": 1.0, "top_k": 20, "reasoning_effort": "low"}})
        svc.shared = {"temperature": 0.3, "reasoning_effort": "high"}
        req = svc.with_shared({"model": "m-u"}, "openai")
        self.assertNotIn("temperature", req)                 # the shared value is merged later, in run()
        self.assertEqual(req["top_k"], 20)
        self.assertEqual(req["reasoning_effort"], "high")

    def test_alias_beats_config(self):
        svc = service({"m-u": {"experimental_speed_projection": True, "reasoning_effort": "none"}})
        svc.sampling_defaults = {"experimental_speed_projection": False}
        svc.config_effort = "high"
        req = svc.with_shared({"model": "m-u"}, "openai")
        self.assertIs(req["experimental_speed_projection"], True)
        self.assertEqual(req["reasoning_effort"], "none")

    def test_equivalent_fields(self):
        svc = service({"m-u": {"max_tokens": 100, "reasoning_effort": "low"}})
        self.assertNotIn("max_tokens", svc.with_shared({"model": "m-u", "max_completion_tokens": 9}, "openai"))
        for extra in ({"reasoning": {"effort": "high"}}, {"chat_template_kwargs": {"enable_thinking": False}},
                      {"chat_template_kwargs": {"reasoning_effort": "high"}}):
            with self.subTest(extra=extra):
                self.assertNotIn("reasoning_effort", svc.with_shared({"model": "m-u", **extra}, "openai"))
        for extra in ({"thinking": {"type": "enabled", "budget_tokens": 2000}}, {"output_config": {"effort": "high"}}):
            with self.subTest(extra=extra):
                req = svc.with_shared({"model": "m-u", **extra}, "anthropic")
                self.assertEqual({k: req.get(k) for k in extra}, extra)
                self.assertNotIn("output_config" if "thinking" in extra else "thinking", req)

    def test_alias_effort_on_anthropic(self):
        svc = service({"m-none": {"reasoning_effort": "none"}, "m-low": {"reasoning_effort": "low"}})
        self.assertEqual(svc.with_shared({"model": "m-none"}, "anthropic")["thinking"], {"type": "disabled"})
        self.assertEqual(svc.with_shared({"model": "m-low"}, "anthropic")["output_config"], {"effort": "low"})
        self.assertNotIn("reasoning_effort", svc.with_shared({"model": "m-low"}, "anthropic"))

    def test_request_budget_wins(self):
        svc = service({"m-u": {"reasoning_budget_tokens": 512}})
        req = svc.with_shared({"model": "m-u", "thinking": {"type": "enabled", "budget_tokens": 16000}}, "anthropic")
        self.assertNotIn("reasoning_budget_tokens", req)
        self.assertNotIn("reasoning_budget_tokens", svc.with_shared({"model": "m-u", "thinking_budget_tokens": 9},
                                                                    "openai"))
        self.assertEqual(svc.with_shared({"model": "m-u"}, "anthropic")["reasoning_budget_tokens"], 512)
        svc.anthropic_think_unasked = False               # "anthropic_thinking": "on_request": no thinking unasked
        self.assertNotIn("reasoning_budget_tokens", svc.with_shared({"model": "m-u"}, "anthropic"))

    def test_max_tokens_zero_as_shared(self):
        svc = service({"m-u": {"max_tokens": 512}})
        self.assertEqual(svc.with_shared({"model": "m-u", "max_tokens": 0}, "openai")["max_tokens"], 512)

    def test_odd_model_field(self):
        svc = service({"m-u": {"top_k": 20}})
        for model in (["m-u"], {"a": 1}, None, 3):
            with self.subTest(model=model):
                self.assertNotIn("top_k", svc.with_shared({"model": model}, "openai"))

    def test_no_alias_unchanged(self):
        svc = service({"m-u": {"top_k": 20}})
        req = {"model": "m", "messages": []}
        self.assertIs(svc.with_shared(req, "openai"), req)


class Http(unittest.TestCase):
    def setUp(self):
        import serve.server as server
        self.server = server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = server.StrataEngine("strata", ["--batch", "2"])
        self.svc = server.Service(self.engine, server.ByteTokenizer(),
                                  server.ChatTemplate(Path(__file__).parent / "chat_template.jinja"),
                                  sampling_defaults={"experimental_speed_projection": False})
        self.svc.alias_sampling = server.parse_alias_sampling({"m-u": {"experimental_speed_projection": True}},
                                                              self.svc.model)
        self.svc.set_aliases(list(self.svc.alias_sampling))
        self.httpd = server.serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.engine.unload()
        self.tmp.cleanup()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return json.loads(r.read().decode())

    def chat(self, model):
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()

    def test_alias_listed(self):
        self.assertIn("m-u", [m["id"] for m in self.get("/v1/models")["data"]])

    def test_projection_key(self):
        sent = []
        real = self.server.StrataEngine._send

        def record(engine, text):
            sent.append(text)
            return real(engine, text)
        with mock.patch.object(self.server.StrataEngine, "_send", record):
            self.chat("m-u")
            self.chat(self.svc.model)
        gens = [t for t in sent if t.split(" ", 1)[0] in ("GEN", "BGEN")]
        self.assertEqual(len(gens), 2, sent)
        self.assertIn(" cvec=1", gens[0])
        self.assertIn(" cvec=0", gens[1])


if __name__ == "__main__":
    unittest.main()
