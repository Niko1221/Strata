"""serve/test_usage.py - token use kept across starts and priced (serve/usage.py): the per-day counts and their
cost, the pricing rules, GET /usage and POST /pricing against the mock engine, and the files next to the run
config (strata-<model>.usage.json / .pricing.json).

    python -m unittest serve.test_usage
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import usage
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]


def post(url, body, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, method="POST", headers=h,
                                 data=json.dumps(body).encode() if not isinstance(body, bytes) else body)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        with e:
            return e.code, json.loads(e.read())


class Pricing(unittest.TestCase):
    def test_defaults_and_costs(self):
        p = usage.clean_pricing(None)
        self.assertEqual(p, usage.DEFAULT_PRICING)
        # 1M fresh input at 2, 1M cached at 0.4, 1M output at 8 -> 2 + 0.4 + 8
        c = usage.cost_of({"prompt_tokens": 2_000_000, "reused": 1_000_000, "output_tokens": 1_000_000}, p)
        self.assertEqual((c["currency"], c["input_cost"], c["cached_cost"], c["output_cost"], c["total_cost"]),
                         ("CNY", 2.0, 0.4, 8.0, 10.4))
        self.assertEqual(usage.cost_of({}, p)["total_cost"], 0.0)
        self.assertEqual(usage.cost_of({"prompt_tokens": 5, "reused": 9, "output_tokens": 3}, p)["input_cost"], 0.0)

    def test_own_prices_and_validation(self):
        p = usage.clean_pricing({"currency": "USD", "input_per_mtok": 5, "output_per_mtok": ""},
                                base=usage.clean_pricing({"cached_input_per_mtok": 0.1}))
        self.assertEqual(p, {"currency": "USD", "input_per_mtok": 5.0, "cached_input_per_mtok": 0.1,
                             "output_per_mtok": 8.0})   # "" -> the default again
        for bad in ({"currency": "BTC"}, {"input_per_mtok": -1}, {"output_per_mtok": "free"},
                    {"input_per_mtok": True}, {"what": 1}, [1], {"input_per_mtok": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                usage.clean_pricing(bad)


class Store(unittest.TestCase):
    def test_counts_survive_a_reload(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-m.usage.json"
            s = usage.UsageStore(path)
            s.record(100, 40, 20)
            s.record(50, 0, 10)
            v = s.view()
            self.assertEqual((v["today"]["requests"], v["today"]["prompt_tokens"], v["today"]["reused"],
                              v["today"]["output_tokens"]), (2, 150, 40, 30))
            self.assertEqual(v["total"]["requests"], 2)
            again = usage.UsageStore(path)               # a new server, the same file
            self.assertEqual(again.view()["total"]["prompt_tokens"], 150)
            self.assertEqual(again.view()["total"]["requests"], 2)

    def test_a_broken_file_starts_over(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-m.usage.json"
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(usage.UsageStore(path).view()["total"]["requests"], 0)


class Http(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        base = Path(self.dir.name) / "strata-m"
        self.svc.usage = usage.UsageStore(str(base) + ".usage.json")
        self.svc.usage_path = str(base) + ".usage.json"
        self.svc.pricing_path = str(base) + ".pricing.json"
        self.base = base
        self.httpd = serve(self.svc, port=0)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def get(self, path):
        with urllib.request.urlopen(f"http://{self.host}{path}", timeout=10) as r:
            return r.status, json.loads(r.read())

    def chat(self, content="hi"):
        body = json.dumps({"model": "m", "max_tokens": 5,
                           "messages": [{"role": "user", "content": content}]}).encode()
        req = urllib.request.Request(f"http://{self.host}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(r.status, 200)

    def test_requests_are_recorded_and_priced(self):
        self.chat()
        self.chat()
        status, u = self.get("/usage")
        self.assertEqual(status, 200)
        self.assertEqual(u["usage"]["today"]["requests"], 2)
        self.assertEqual(u["usage"]["total"]["requests"], 2)
        self.assertGreater(u["usage"]["total"]["prompt_tokens"], 0)
        self.assertGreater(u["usage"]["total"]["output_tokens"], 0)
        self.assertEqual(u["pricing"], usage.DEFAULT_PRICING)
        self.assertGreater(u["usage"]["total"]["cost"]["total_cost"], 0)
        # the same numbers ride along on /metrics, so the Monitor needs no extra call
        _, m = self.get("/metrics")
        self.assertEqual(m["usage"]["total"]["requests"], 2)
        self.assertEqual(m["pricing"], usage.DEFAULT_PRICING)
        # kept on disk next to the run config
        self.assertTrue((self.base.parent / "strata-m.usage.json").is_file())

    def test_pricing_changes_the_costs_at_once(self):
        self.chat()
        with contextlib.redirect_stdout(io.StringIO()):
            status, b = post(f"http://{self.host}/pricing",
                             {"pricing": {"currency": "USD", "input_per_mtok": 10, "cached_input_per_mtok": None,
                                          "output_per_mtok": 20}},
                             {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, b)
        self.assertEqual(b["pricing"], {"currency": "USD", "input_per_mtok": 10.0,
                                        "cached_input_per_mtok": 0.4, "output_per_mtok": 20.0})
        self.assertEqual(b["usage"]["total"]["cost"]["currency"], "USD")
        self.assertEqual(self.get("/usage")[1]["pricing"]["currency"], "USD")
        self.assertEqual(json.loads((self.base.parent / "strata-m.pricing.json").read_text(encoding="utf-8"))["currency"],
                         "USD")

    def test_pricing_is_guarded(self):
        status, b = post(f"http://{self.host}/pricing", {"pricing": {"input_per_mtok": -1}})
        self.assertEqual((status, "message" in b.get("error", {})), (400, True))
        status, _ = post(f"http://{self.host}/pricing", {"pricing": {}}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = post(f"http://{self.host}/pricing", b'{"pricing": {}}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        self.svc.api_key = "secret"
        self.assertEqual(post(f"http://{self.host}/pricing", {"pricing": {}})[0], 401)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(post(f"http://{self.host}/pricing", {"pricing": {}},
                                  {"Authorization": "Bearer secret"})[0], 200)

    def test_without_files_the_numbers_stay_in_memory(self):
        self.svc.usage = usage.UsageStore()
        self.svc.pricing_path = None
        self.chat()
        status, u = self.get("/usage")
        self.assertEqual(status, 200)
        self.assertEqual(u["usage"]["today"]["requests"], 1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(post(f"http://{self.host}/pricing", {"pricing": {"currency": "USD"}})[1]["pricing"]
                             ["currency"], "USD")

    def test_the_page_has_the_cards(self):
        web = ROOT / "serve" / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("usage-card", "usage-facts", "usage-days-body", "price-card", "price-save", "price-input"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn('fetch("pricing", {method: "POST", headers: headers(true)', js)


if __name__ == "__main__":
    unittest.main()
