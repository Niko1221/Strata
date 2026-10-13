"""serve/test_usage.py - token use kept across starts and priced (serve/usage.py): the per-day counts, the
watt-hours the power sampler adds and their cost, the pricing rules, GET /usage and POST /pricing against the mock
engine, and the files next to the run config (strata-<model>.usage.json / .pricing.json).

    python -m unittest serve.test_usage
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
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
        self.assertEqual(c["token_cost"], 10.4)
        self.assertEqual((c["gpu_energy_wh"], c["energy_kwh"], c["energy_cost"]), (0.0, 0.0, 0.0))
        self.assertEqual(usage.cost_of({}, p)["total_cost"], 0.0)
        self.assertEqual(usage.cost_of({"prompt_tokens": 5, "reused": 9, "output_tokens": 3}, p)["input_cost"], 0.0)

    def test_electricity_is_the_card_plus_the_rest_of_the_box(self):
        p = usage.clean_pricing(None)                    # 0.55 per kWh, 100 W for the rest of the box
        # the card drew 300 Wh over the hour; the box adds 100 W x 1 h = 100 Wh -> 0.4 kWh at 0.55
        c = usage.cost_of({"energy_wh": 300, "seconds": 3600}, p)
        self.assertEqual((c["gpu_energy_wh"], c["energy_kwh"], c["energy_cost"]), (300.0, 0.4, 0.22))
        self.assertEqual(c["total_cost"], 0.22)
        # an hour of the box alone (no reading from the card): the fixed part still bills
        alone = usage.cost_of({"seconds": 3600}, p)
        self.assertEqual((alone["gpu_energy_wh"], alone["energy_kwh"], alone["energy_cost"]), (0.0, 0.1, 0.055))
        # watt-hours without seconds bill the card alone, at the default 0.55
        self.assertEqual(usage.cost_of({"energy_wh": 1000}, p)["energy_cost"], 0.55)
        # own prices, and no card in the bill at all
        own = usage.cost_of({"energy_wh": 1000, "seconds": 1}, usage.clean_pricing({"price_per_kwh": 2.0,
                                                                                    "other_watts": 0}))
        self.assertEqual(own["energy_cost"], 2.0)
        # the token bill and the power bill are summed into the one total
        both = usage.cost_of({"prompt_tokens": 1_000_000, "energy_wh": 1000},
                             usage.clean_pricing({"price_per_kwh": 1.0, "other_watts": 0}))
        self.assertEqual((both["token_cost"], both["energy_cost"], both["total_cost"]), (2.0, 1.0, 3.0))

    def test_own_prices_and_validation(self):
        p = usage.clean_pricing({"currency": "USD", "input_per_mtok": 5, "output_per_mtok": ""},
                                base=usage.clean_pricing({"cached_input_per_mtok": 0.1}))
        self.assertEqual(p, {"currency": "USD", "input_per_mtok": 5.0, "cached_input_per_mtok": 0.1,
                             "output_per_mtok": 8.0, "price_per_kwh": 0.55, "other_watts": 100.0})  # "" -> default
        # the power prices come down the same road, and 0 is a price one may well mean
        self.assertEqual(usage.clean_pricing({"price_per_kwh": 0.8, "other_watts": 0}),
                         {**usage.DEFAULT_PRICING, "price_per_kwh": 0.8, "other_watts": 0.0})
        self.assertEqual(usage.clean_pricing({"price_per_kwh": None})["price_per_kwh"], 0.55)
        for bad in ({"currency": "BTC"}, {"input_per_mtok": -1}, {"output_per_mtok": "free"},
                    {"price_per_kwh": -0.1}, {"other_watts": "lots"}, {"other_watts": True},
                    {"input_per_mtok": True}, {"what": 1}, [1], {"input_per_mtok": float("nan")},
                    {"price_per_kwh": float("nan")}):
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

    def test_energy_survives_a_reload_and_splits_the_busy_share(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-m.usage.json"
            s = usage.UsageStore(path)
            s.record_energy(300.0, 6.0)                  # 300 W for 6 s -> 0.5 Wh
            s.record_energy(300.0, 6.0, busy=True)       # the same, with the GPU working
            s.record_energy(None, 10.0)                  # a card whose power cannot be read: the time still counts
            s.flush()
            v = s.view()
            self.assertEqual((v["today"]["energy_wh"], v["today"]["busy_wh"], v["today"]["seconds"]),
                             (1.0, 0.5, 22.0))
            again = usage.UsageStore(path)
            self.assertEqual((again.view()["total"]["energy_wh"], again.view()["total"]["seconds"]), (1.0, 22.0))
            # a nonsense reading is no reading, not a negative bill
            for w in (-5, float("nan"), "watts"):
                with self.subTest(w=w):
                    again.record_energy(w, 1.0)
            self.assertEqual(again.view()["total"]["seconds"], 25.0)
            self.assertEqual(again.view()["total"]["energy_wh"], 1.0)

    def test_a_file_one_edited_by_hand_still_reads(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-m.usage.json"
            path.write_text(json.dumps({"days": {"2026-01-01": {"requests": 2, "energy_wh": "x", "seconds": -3,
                                                                "output_tokens": 7, "nonsense": 1}}}), encoding="utf-8")
            v = usage.UsageStore(path).view()
            self.assertEqual((v["total"]["requests"], v["total"]["output_tokens"]), (2, 7))
            self.assertEqual((v["total"]["energy_wh"], v["total"]["seconds"]), (0.0, 0.0))

    def test_a_broken_file_starts_over(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-m.usage.json"
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(usage.UsageStore(path).view()["total"]["requests"], 0)


class Folders(unittest.TestCase):
    """The models sharing the running model's folder: what they are called, how they add up, and whose counts
    win when the model is the one the server is running."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)

    def put(self, stem, model=None, days=None):
        """One model in the folder: its run config (when it has one) and its usage file."""
        if model is not None:
            (self.dir / f"{stem}.json").write_text(json.dumps({"model_name": model}), encoding="utf-8")
        path = self.dir / f"{stem}.usage.json"
        path.write_text(json.dumps({"days": days or {}}), encoding="utf-8")
        return path

    def test_a_model_is_named_after_its_run_config(self):
        p = self.put("strata-a", "model-a")
        self.assertEqual(usage.model_name(p), "model-a")
        # a config one edited by hand, a config that is gone, a file someone copied here: the file names itself
        (self.dir / "strata-a.json").write_text("not json", encoding="utf-8")
        self.assertEqual(usage.model_name(p), "strata-a")
        self.assertEqual(usage.model_name(self.put("strata-c")), "strata-c")
        self.assertEqual(usage.model_name(self.put("names", "  spaced  ")), "spaced")

    def test_the_folder_adds_up_and_one_model_stays_one_model(self):
        live = self.put("strata-live", "model-live",
                        {"2026-01-01": {"requests": 5, "prompt_tokens": 500, "energy_wh": 1.0, "seconds": 60.0}})
        self.put("strata-a", "model-a", {"2026-01-01": {"requests": 2, "prompt_tokens": 200}})
        # the same model at another context or quantisation: two run configs, one model, one line
        self.put("strata-b1", "model-b", {"2026-01-01": {"requests": 3}})
        self.put("strata-b2", "model-b", {"2026-01-01": {"requests": 4}})
        store = usage.ModelsUsage(live)

        models = dict(store.models({"2026-01-01": {"requests": 6, "prompt_tokens": 600,
                                                   "energy_wh": 2.0, "seconds": 120.0}}, "model-live"))
        self.assertEqual(sorted(models), ["model-a", "model-b", "model-live"])
        self.assertEqual(models["model-b"]["2026-01-01"]["requests"], 7)
        # the running model's counts come from memory - its file may be up to SAVE_EVERY_S behind, and the two
        # must not be added to one another: the memory copy replaces what the file holds
        self.assertEqual(models["model-live"]["2026-01-01"]["requests"], 6)
        self.assertEqual(models["model-live"]["2026-01-01"]["prompt_tokens"], 600)
        self.assertEqual(models["model-live"]["2026-01-01"]["energy_wh"], 2.0)

        merged = usage.summed(models.values())
        self.assertEqual(merged["2026-01-01"]["requests"], 6 + 2 + 7)
        self.assertEqual(merged["2026-01-01"]["prompt_tokens"], 600 + 200)
        v = usage.days_view(merged)
        self.assertEqual((v["total"]["requests"], v["total"]["prompt_tokens"]), (15, 800))
        self.assertEqual(v["total"]["energy_wh"], 2.0)
        self.assertEqual(v["today"]["requests"], 0)                  # that day is not today
        self.assertEqual(v["total"]["cost"]["currency"], "CNY")

        # a file another server rewrote is read again, not remembered
        self.put("strata-a", "model-a", {"2026-01-01": {"requests": 9}})
        again = dict(store.models({}, "model-live"))
        self.assertEqual(again["model-a"]["2026-01-01"]["requests"], 9)

    def test_a_model_with_nothing_to_show_is_still_listed(self):
        live = self.put("strata-live", "model-live")
        self.assertEqual(usage.ModelsUsage(live).models({}, "model-live"), [("model-live", {})])
        # no folder at all (a server started without a run config): the running model is the folder
        self.assertEqual(usage.ModelsUsage().models({"x": {"requests": 1}}, "m"), [("m", {"x": {"requests": 1}})])


class Http(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        base = Path(self.dir.name) / "strata-m"
        self.svc.usage = usage.UsageStore(str(base) + ".usage.json")
        self.svc.usage_path = str(base) + ".usage.json"
        self.svc.models_usage = usage.ModelsUsage(self.svc.usage_path)
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
        self.assertEqual(b["pricing"], {**usage.DEFAULT_PRICING, "currency": "USD", "input_per_mtok": 10.0,
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

    def test_the_power_prices_go_through_the_same_door(self):
        self.svc.usage.record_energy(1000.0, 3600.0)      # 1000 W for an hour -> 1 kWh, so the cost is easy to see
        with contextlib.redirect_stdout(io.StringIO()):
            status, b = post(f"http://{self.host}/pricing",
                             {"pricing": {"price_per_kwh": 1.0, "other_watts": 0}},
                             {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, b)
        self.assertEqual((b["pricing"]["price_per_kwh"], b["pricing"]["other_watts"]), (1.0, 0.0))
        self.assertEqual(b["usage"]["total"]["cost"]["energy_cost"], 1.0)
        # back to the defaults leaves the watt-hours alone and only re-prices them - and brings the rest of the
        # box (100 W over the hour, so 100 Wh more) back into the bill: the card's 1000 + 100 = 1.1 kWh
        status, b = post(f"http://{self.host}/pricing", {"pricing": {"price_per_kwh": None, "other_watts": None}},
                         {"Origin": f"http://{self.host}"})
        self.assertEqual((status, b["pricing"]["price_per_kwh"], b["usage"]["total"]["cost"]["energy_kwh"]),
                         (200, 0.55, 1.1))
        self.assertEqual(json.loads((self.base.parent / "strata-m.pricing.json").read_text(encoding="utf-8"))
                         ["price_per_kwh"], 0.55)

    def test_the_sampler_feeds_the_usage_store(self):
        # the wiring, not the hardware: what the sampler hands over becomes watt-hours (Service._sample_power)
        self.assertEqual(self.svc.telemetry.on_sample.__func__, self.svc._sample_power.__func__)
        self.svc.telemetry.close()                        # the real thread would add readings of its own
        time.sleep(1.2)
        self.svc.usage = usage.UsageStore()               # in memory; the file is not what this checks
        self.svc._sample_power({"gpu_power": 300.0, "gpu_util": 80}, 6.0)
        self.svc._sample_power({"gpu_util": 0}, 6.0)      # no reading from the card: the seconds still count
        v = self.svc.usage.view()["total"]
        self.assertEqual((v["energy_wh"], v["busy_wh"], v["seconds"]), (0.5, 0.5, 12.0))
        self.assertGreater(v["cost"]["energy_cost"], 0)

    def test_the_card_adds_the_models_up_and_can_show_them_apart(self):
        self.chat()
        # another model that ran out of this folder: its own usage file, named by its own run config
        here = self.base.parent
        (here / "strata-other.json").write_text(json.dumps({"model_name": "other-model"}), encoding="utf-8")
        (here / "strata-other.usage.json").write_text(json.dumps({"days": {
            usage.UsageStore._today(): {"requests": 4, "prompt_tokens": 4_000_000, "reused": 1_000_000,
                                        "output_tokens": 1000, "energy_wh": 10.0, "seconds": 3600.0}}}),
            encoding="utf-8")
        status, u = self.get("/usage")
        self.assertEqual(status, 200)
        usage_v, models = u["usage"], u["usage"]["models"]
        self.assertEqual(usage_v["today"]["requests"], 5)                    # 1 here + 4 in the other file
        self.assertEqual(usage_v["total"]["prompt_tokens"],
                         sum(m["total"]["prompt_tokens"] for m in models))
        self.assertEqual([m["name"] for m in models], ["other-model", self.svc.model])   # dearest first
        other, mine = models
        self.assertEqual(other["total"]["requests"], 4)
        self.assertEqual(other["today"]["requests"], 4)
        self.assertEqual(other["days"][0]["date"], usage.UsageStore._today())
        self.assertEqual(mine["total"]["requests"], 1)                       # the running model, from memory
        self.assertEqual(mine["total"]["cost"]["currency"], "CNY")
        self.assertGreater(other["total"]["cost"]["total_cost"], mine["total"]["cost"]["total_cost"])
        # /metrics carries the same card, so the Monitor needs no second call
        self.assertEqual(self.get("/metrics")[1]["usage"]["models"], models)
        # the prices the page sets re-price every model, not only this one
        with contextlib.redirect_stdout(io.StringIO()):
            _, b = post(f"http://{self.host}/pricing", {"pricing": {"currency": "USD"}},
                        {"Origin": f"http://{self.host}"})
        self.assertEqual([m["total"]["cost"]["currency"] for m in b["usage"]["models"]], ["USD", "USD"])

    def test_the_page_has_the_cards(self):
        web = ROOT / "serve" / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("usage-card", "usage-facts", "usage-days-body", "usage-model-th", "usage-scope", "usage-scope-card",
                   "price-card", "price-save", "price-input", "price-kwh", "price-other-w"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn('fetch("pricing", {method: "POST", headers: headers(true)', js)
        # the choice, on the card and in the Settings card: merged, or one line per model
        self.assertEqual(html.count('data-v="merged"'), 2)
        self.assertEqual(html.count('data-v="models"'), 2)


if __name__ == "__main__":
    unittest.main()
