"""serve/test_prometheus.py - GET /metrics in the Prometheus text format (vLLM's names), against the mock engine.

    python -m unittest serve.test_prometheus -v
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.prometheus import BUCKETS, TOKEN_BUCKETS, Latencies, render, wants_prometheus  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse(text):
    """{(name, labels): value} for every sample line; checks each family has HELP and TYPE first."""
    out, typed = {}, set()
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            assert line.split()[2] not in typed, f"two # TYPE lines for {line.split()[2]}"
            typed.add(line.split()[2])
            continue
        if line.startswith("#") or not line:
            continue
        m = re.fullmatch(r'([a-z0-9_:]+)\{([^}]*)\} (\S+)', line)
        assert m, f"not a sample line: {line!r}"
        name, labels, value = m.groups()
        family = re.sub(r"_(bucket|sum|count)$", "", name)
        assert name in typed or family in typed, f"{name} has no # TYPE"
        out[(name, labels)] = float(value)
    return out


class Units(unittest.TestCase):
    def test_who_gets_the_text(self):
        self.assertTrue(wants_prometheus("text/plain;version=0.0.4", ""))
        self.assertTrue(wants_prometheus("application/openmetrics-text;version=1.0.0", ""))
        self.assertTrue(wants_prometheus("", "format=prometheus"))
        self.assertFalse(wants_prometheus("application/json", ""))
        self.assertFalse(wants_prometheus("*/*", "requests=all"))

    def test_histograms_are_cumulative(self):
        lat = Latencies()
        lat.observe(0.3, 11, 1.0, 1.5)       # 10 gaps of 0.1 s
        lat.observe(None, 1, 0.0, 0.2)       # no first token, one token: no gap
        s = lat.snapshot()
        buckets, counts, total, n = s["itl"]
        self.assertEqual(n, 10)
        self.assertAlmostEqual(total, 1.0)
        self.assertEqual(counts[BUCKETS.index(0.1)], 10)
        self.assertEqual(counts[BUCKETS.index(0.08)], 0)
        self.assertEqual(s["ttft"][3], 1)
        self.assertEqual(s["e2e"][3], 2)
        self.assertEqual(counts, sorted(counts))

    def test_token_histograms_use_the_capped_125_buckets(self):
        lat = Latencies()
        lat.observe(0.1, 3, 0.2, 0.4, prompt_tokens=150, computed_tokens=90, prefill_s=0.05, finish="stop")
        s = lat.snapshot()
        self.assertEqual(s["prompt_tokens"][0], TOKEN_BUCKETS)
        self.assertEqual(TOKEN_BUCKETS[-1], 262144)
        self.assertEqual(s["prompt_tokens"][3], 1)            # one request observed
        self.assertEqual(s["prompt_tokens"][2], 150)          # _sum = prompt tokens
        self.assertEqual(s["prompt_tokens"][1][TOKEN_BUCKETS.index(200)], 1)   # 150 <= 200
        self.assertEqual(s["prompt_tokens"][1][TOKEN_BUCKETS.index(100)], 0)
        self.assertEqual(s["generation_tokens"][2], 3)
        self.assertEqual(s["prefill_kv_computed"][2], 90)
        self.assertAlmostEqual(s["prefill"][2], 0.05)
        self.assertAlmostEqual(s["decode"][2], 0.2)

    def test_time_per_output_token(self):
        lat = Latencies()
        lat.observe(0.2, 3, 0.8, 1.0)        # e2e 1.0, ttft 0.2, 3 tokens -> (1.0 - 0.2) / 2 = 0.4
        lat.observe(0.2, 1, 0.0, 0.5)        # one token: no gap -> 0
        s = lat.snapshot()
        self.assertEqual(s["tpot"][3], 2)
        self.assertAlmostEqual(s["tpot"][2], 0.4)
        self.assertEqual(s["tpot"][1][0], 1)                  # the 0.0 sample lands in the smallest bucket

    def test_finish_reason_labels(self):
        lat = Latencies()
        for finish in ("stop", "cancel", "disconnect", "length", "tool_calls", "error"):
            lat.observe(0.1, 2, 0.2, 0.4, finish=finish)
        s = lat.snapshot()
        self.assertEqual(s["finish"], {"stop": 1, "cancel": 1, "disconnect": 1, "length": 1,
                                       "tool_calls": 1, "error": 1})
        got = parse(render({"engine": {"model": "m", "max_context": 1000}, "live": {}, "totals": {}}, s))
        lab = 'model_name="m"'
        self.assertEqual(got[("vllm:request_success_total", f'finished_reason="stop",{lab}')], 2)    # stop + tool_calls
        self.assertEqual(got[("vllm:request_success_total", f'finished_reason="abort",{lab}')], 2)   # cancel + disconnect
        self.assertEqual(got[("vllm:request_success_total", f'finished_reason="length",{lab}')], 1)
        self.assertEqual(got[("vllm:request_success_total", f'finished_reason="error",{lab}')], 1)

    def test_conversation_cache_only_when_enabled(self):
        m = {"engine": {"model": "m", "max_context": 1000}, "live": {}, "totals": {},
             "conversation_cache": {"enabled": True, "budget_mib": 4096, "bytes": 2 << 30, "slots": 16,
                                    "parked": 3, "evictions": 1, "parks": 5, "restores": 4, "reused_tokens": 999}}
        got = parse(render(m, Latencies().snapshot()))
        lab = 'model_name="m"'
        self.assertEqual(got[("strata:conversation_cache_budget_bytes", lab)], 4096 * (1 << 20))
        self.assertEqual(got[("strata:conversation_cache_used_bytes", lab)], 2 << 30)
        self.assertEqual(got[("strata:conversation_cache_slots", lab)], 16)
        self.assertEqual(got[("strata:conversation_cache_parked", lab)], 3)
        self.assertEqual(got[("strata:conversation_cache_evictions_total", lab)], 1)
        self.assertEqual(got[("strata:conversation_cache_parks_total", lab)], 5)
        self.assertEqual(got[("strata:conversation_cache_restores_total", lab)], 4)
        self.assertEqual(got[("strata:conversation_cache_reused_tokens_total", lab)], 999)
        off = parse(render({**m, "conversation_cache": {"enabled": False, "budget_mib": 4096, "bytes": 0}},
                           Latencies().snapshot()))
        self.assertNotIn(("strata:conversation_cache_budget_bytes", lab), off)
        self.assertNotIn(("strata:conversation_cache_used_bytes", lab), off)

    def test_render_reads_the_server_record(self):
        m = {"engine": {"model": 'a "b"', "max_context": 1000},
             "live": {"state": "generating", "queued": 2, "prompt_tokens": 400, "generated": 100},
             "totals": {"requests": 3, "prompt_tokens": 900, "reused": 600, "output_tokens": 50,
                        "drafts_offered": 40, "drafts_accepted": 30}}
        got = parse(render(m, Latencies().snapshot()))
        lab = 'model_name="a \\"b\\""'
        self.assertEqual(got[("vllm:num_requests_running", lab)], 1)
        self.assertEqual(got[("vllm:num_requests_waiting", lab)], 2)
        self.assertEqual(got[("vllm:kv_cache_usage_perc", lab)], 0.5)
        self.assertEqual(got[("vllm:prefix_cache_hits_total", lab)], 600)
        self.assertEqual(got[("vllm:spec_decode_num_accepted_tokens_total", lab)], 30)
        self.assertEqual(got[("vllm:time_to_first_token_seconds_count", lab)], 0)

    def test_strata_names_follow_the_json(self):
        m = {"engine": {"model": "m", "max_context": 1000},
             "live": {"state": "reading", "queued": 0, "prompt_read": 300, "prefill_tok_s_mean": 2500.0},
             "totals": {"requests": 1, "prompt_ms": 1500.0, "decode_ms": 2000.0},
             "requests": [{"hit_rate": 0.97, "decode_tok_s": 101.5}],
             "hardware": {"gpu_util": 40.0, "gpus": [{"index": 0, "util": 30.0, "mem_used": 1 << 30, "temp": 60},
                                                     {"index": 1, "util": 50.0, "mem_used": 2 << 30, "temp": 65}],
                          "cpu": 12.5, "ram_used": 8 << 30, "ram_total": 64 << 30}}
        got = parse(render(m, Latencies().snapshot()))
        lab = 'model_name="m"'
        self.assertEqual(got[("strata:live_state", lab + ',state="reading"')], 1)
        self.assertEqual(got[("strata:live_state", lab + ',state="generating"')], 0)
        self.assertEqual(got[("strata:live_prompt_read", lab)], 300)
        self.assertEqual(got[("strata:totals_decode_seconds_total", lab)], 2.0)
        self.assertEqual(got[("strata:last_hit_rate", lab)], 0.97)
        self.assertEqual(got[("strata:gpu_util", lab + ',gpu="1"')], 50)
        self.assertEqual(got[("strata:gpu_mem_used_bytes", lab + ',gpu="0"')], 1 << 30)
        self.assertNotIn(("strata:gpu_power_watts", lab + ',gpu="0"'), got)   # not measured: no sample
        self.assertEqual(got[("strata:ram_total_bytes", lab)], 64 << 30)


class Endpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "</think>\n\nhello there", max_context=4096), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path, accept):
        req = urllib.request.Request(self.base + path, headers={"Accept": accept})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.headers.get("Content-Type"), r.read().decode()

    def test_a_request_shows_in_both_formats(self):
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            usage = json.loads(r.read())["usage"]
        ctype, text = self.get("/metrics", "text/plain")
        self.assertTrue(ctype.startswith("text/plain"))
        got = parse(text)
        by_name = {}
        for (name, _), v in got.items():
            by_name[name] = by_name.get(name, 0) + v
        success = {l: v for (name, l), v in got.items() if name == "vllm:request_success_total"}
        self.assertGreaterEqual(sum(success.values()), 1)
        self.assertTrue(any('finished_reason="stop"' in l for l in success))
        self.assertGreaterEqual(by_name["vllm:generation_tokens_total"], usage["completion_tokens"])
        self.assertGreaterEqual(by_name["vllm:e2e_request_latency_seconds_count"], 1)
        self.assertGreaterEqual(by_name["vllm:request_prompt_tokens_count"], 1)
        self.assertGreaterEqual(by_name["vllm:request_generation_tokens_count"], 1)
        ctype, text = self.get("/metrics", "application/json")
        self.assertTrue(ctype.startswith("application/json"))
        self.assertGreaterEqual(json.loads(text)["totals"]["requests"], 1)
        ctype, _ = self.get("/metrics?format=prometheus", "*/*")
        self.assertTrue(ctype.startswith("text/plain"))

    def test_telemetry_key_opens_metrics_not_inference(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nhi", max_context=4096), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key, svc.telemetry_key = "secret", "tele-secret"
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for path, accept in (("/metrics", "text/plain"), ("/metrics/prometheus", "")):
                req = urllib.request.Request(base + path,
                                             headers={"Accept": accept, "Authorization": "Bearer tele-secret"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    self.assertEqual(r.status, 200)
            # the read-only key must not run inference
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer tele-secret"})
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req, timeout=30)
            self.assertEqual(e.exception.code, 401)
        finally:
            httpd.shutdown()
            httpd.server_close()


class BatchEngine(MockEngine):
    """--batch as the server sees it: several requests generate at once, none waits for the control lines."""
    batch, waiting = 4, 0


class Batched(unittest.TestCase):
    def test_every_concurrent_request_is_counted(self):
        tok = ByteTokenizer()
        svc = Service(BatchEngine(tok, "</think>\n\n" + "y" * 40, max_context=4096, delay_s=0.02), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 64}).encode()

            def ask():
                req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    r.read()

            threads = [threading.Thread(target=ask) for _ in range(3)]
            [t.start() for t in threads]
            time.sleep(0.4)                              # the three are generating
            req = urllib.request.Request(base + "/metrics", headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=30) as r:
                during = {k[0]: v for k, v in parse(r.read().decode()).items()}
            [t.join() for t in threads]
            with urllib.request.urlopen(req, timeout=30) as r:
                after = parse(r.read().decode())
        finally:
            httpd.shutdown()
            httpd.server_close()

        def total(name):
            return sum(v for (n, _), v in after.items() if n == name)

        self.assertEqual(during["vllm:num_requests_running"], 3)
        self.assertEqual(total("vllm:num_requests_running"), 0)
        self.assertEqual(total("vllm:request_success_total"), 3)      # each one, not only the last to finish
        self.assertEqual(total("vllm:e2e_request_latency_seconds_count"), 3)
        self.assertEqual(total("vllm:time_to_first_token_seconds_count"), 3)
        self.assertEqual(total("vllm:request_prompt_tokens_count"), 3)
        self.assertEqual(total("vllm:request_generation_tokens_count"), 3)
        self.assertEqual(total("vllm:request_prefill_time_seconds_count"), 3)
        self.assertEqual(total("vllm:request_decode_time_seconds_count"), 3)
        lab = 'model_name="qwen3.8-flash-next"'
        self.assertEqual(after[("vllm:request_success_total", f'finished_reason="stop",{lab}')], 3)
        self.assertEqual(svc.metrics()["totals"]["requests"], 3)          # the JSON agrees


if __name__ == "__main__":
    unittest.main()
