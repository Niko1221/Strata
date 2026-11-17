"""SMI2000: the Monitor's extras on GET /metrics - POST /metrics/reset clears the requests and the totals of the /metrics
view only, the default sampling is reported, the pack's expert totals and the VRAM parts the engine wrote at start."""
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, Service, kv_estimate_mib, pack_expert_totals, serve,
                          vram_parts_from_log)
from serve.test_lifecycle import ResidentEngine

START_LOG = """strata generate: 216 MiB of weights loaded from C:\\packs\\q6 in 0.1 s
strata: STRATA_HC_Q8=1: 0.63 GiB of Q8_0 hyper-connection projections for the verify read
strata generate: 499 native projection matrices, 3619.14 MiB of weights, in 3.2 s
strata mtp: draft layer loaded, 876 MiB of VRAM (experts 675, dense 111), files read in 0.36 s (2197 MiB/s)
strata verify: window up to 6 tokens, 113.8 MiB of device buffers
strata mtp: draft head over 106299 tokens (275.7 MiB)
"""


class Helpers(unittest.TestCase):
    def test_vram_parts(self):
        self.assertEqual(vram_parts_from_log(START_LOG), {"weights": 4480, "draft": 1152, "buffers": 114})
        self.assertEqual(vram_parts_from_log("nothing here"), {})

    def test_kv_estimate(self):
        self.assertEqual(kv_estimate_mib("qwen4exp", "k8v4", 65536), 820)
        self.assertIsNone(kv_estimate_mib("deepseek41", "fp4", 65536))
        self.assertIsNone(kv_estimate_mib("qwen4exp", "k8v4", None))

    def test_pack_totals(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "native_experts.txt").write_text(
                "# strata native experts v4: layer ... (n_expert 512, total 1073741824; absolute offsets)\n0 1\n1 2\n",
                encoding="utf-8")
            self.assertEqual(pack_expert_totals(["--pack", d]), {"experts": 1024, "expert_total_mib": 1024})
        self.assertEqual(pack_expert_totals(["--pack", "/no/such/pack"]), {})
        self.assertEqual(pack_expert_totals([]), {})


class Route(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.engine.loaded = True
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.sampling_defaults = {"temperature": 0.2, "top_k": 1}
        self.svc.model_arch = "qwen4exp"
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def test_sampling_reported(self):
        body = self.request("/metrics")[1]
        self.assertEqual(body["sampling"]["config"], {"temperature": 0.2, "top_k": 1})
        self.assertEqual(body["sampling"]["thinking"]["temperature"], 0.2)
        self.assertEqual(body["sampling"]["instruct"]["top_k"], 1)

    def test_reset_clears_the_metrics_view_only(self):
        with self.svc.status_lock:
            self.svc.history.append({"time": 1.0, "prompt_tokens": 10, "output_tokens": 5})
            self.svc.totals.update(requests=1, prompt_tokens=10, output_tokens=5)
        self.assertEqual(len(self.request("/metrics")[1]["requests"]), 1)
        code, body = self.request("/metrics/reset", {})
        self.assertEqual((code, body), (200, {"status": "cleared"}))
        after = self.request("/metrics")[1]
        self.assertEqual(after["requests"], [])
        self.assertEqual((after["totals"]["requests"], after["totals"]["output_tokens"]), (0, 0))
        self.assertEqual(self.svc.totals["requests"], 1)               # /v1/status and Prometheus keep counting
        with self.svc.status_lock:
            self.svc.totals.update(requests=2, output_tokens=12)
        self.assertEqual(self.request("/metrics")[1]["totals"]["output_tokens"], 7)

    def test_reset_wants_json_from_the_own_page(self):
        req = urllib.request.Request(self.base + "/metrics/reset", data=b"x=1",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(caught.exception.code, 415)


if __name__ == "__main__":
    unittest.main()
