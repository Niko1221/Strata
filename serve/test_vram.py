"""#533: POST /v1/vram (opt-in hot VRAM resize) - the engine command, the route, the config key, a reserve kept for
an unloaded engine and applied when it loads again, and the per-GPU form of it (#533 on a layer split): one cache per
GPU, `gpu` naming which one moves."""
import io
import json
import queue
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, Service, StrataEngine, engine_args, serve
from serve.test_lifecycle import ResidentEngine
from serve import runconfig


class VramEngine(ResidentEngine):
    """The resident test engine with an elastic cache on two cards: `VRAM n [gpu]` frees what the reserve asks for on
    that card (no gpu: every card of the run), None takes the start's reserve (700 MiB).  The caches start at 6000
    and 5000 MiB; a reserve above the start's comes out of the cache it names."""

    def __init__(self, tok, elastic=True, gpus=(0, 1)):
        super().__init__(tok)
        self.elastic, self.calls = elastic, []
        self.gpus = list(gpus)
        self.full = {g: 6000 - 1000 * i for i, g in enumerate(self.gpus)}
        self.cache, self.free = dict(self.full), {g: 700 for g in self.gpus}

    def vram(self, reserve_mib, gpu=None, timeout=120.0):
        self.calls.append((reserve_mib, gpu))
        if not self.elastic:
            raise ValueError("VRAM needs an engine started with --vram-elastic (NVIDIA, --serve)")
        r = 700 if reserve_mib is None else reserve_mib
        moved = self.gpus if gpu is None else [gpu]
        parts = []
        for g in self.gpus:
            if g in moved:
                self.free[g] = r
                self.cache[g] = max(500, self.full[g] - max(0, r - 700))
            parts.append({"reserve_mib": r, "gpu": g, "expert_slots": self.cache[g] // 4,
                          "expert_slots_full": self.full[g] // 4, "expert_cache_mib": self.cache[g],
                          "expert_cache_full_mib": self.full[g], "vram_free_mib": self.free[g],
                          "prompt_chunk": 4096})
        out = dict(parts[0])
        out["gpus"] = parts
        out["expert_slots"] = sum(p["expert_slots"] for p in parts)
        out["expert_cache_mib"] = sum(p["expert_cache_mib"] for p in parts)
        out["expert_cache_full_mib"] = sum(p["expert_cache_full_mib"] for p in parts)
        out["vram_free_mib"] = min(p["vram_free_mib"] for p in parts)
        self.info["vram"] = out
        return out


class EngineCommand(unittest.TestCase):
    def engine(self, replies):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.proc.stdin = io.StringIO()
        e.ended = False
        e.lines = queue.Queue()
        for r in replies:
            e.lines.put(r)
        return e

    def test_the_line_and_the_answer(self):
        e = self.engine(["strata serve: something else\n",
                         "VRAM reserve_mib=6000 gpu=0 expert_slots=1200 expert_slots_full=3700 expert_cache_mib=1536 "
                         "expert_cache_full_mib=4900 vram_free_mib=6100 prompt_chunk=2048\n",
                         "VRAM-DONE\n"])
        out = e.vram(6000)
        self.assertEqual(e.proc.stdin.getvalue(), "VRAM 6000\n")
        self.assertEqual(out["expert_slots"], 1200)
        self.assertEqual(out["prompt_chunk"], 2048)
        self.assertEqual(len(out["gpus"]), 1)
        self.assertEqual(out["gpus"][0]["gpu"], 0)
        self.assertEqual(e.info["expert_slots"], 1200)
        e = self.engine(["VRAM reserve_mib=700 gpu=0 expert_slots=3700\n", "VRAM-DONE\n"])
        e.vram(None)
        self.assertEqual(e.proc.stdin.getvalue(), "VRAM\n")      # no device: no dash, so an older engine parses it

    def test_one_device_in_the_line(self):
        e = self.engine(["VRAM reserve_mib=16000 gpu=1 expert_slots=512 expert_slots_full=5632 "
                         "expert_cache_mib=768 expert_cache_full_mib=8448 vram_free_mib=16100 prompt_chunk=512\n",
                         "VRAM-DONE\n"])
        out = e.vram(16000, 1)
        self.assertEqual(e.proc.stdin.getvalue(), "VRAM 16000 1\n")
        self.assertEqual(out["gpu"], 1)
        self.assertEqual(out["vram_free_mib"], 16100)

    def test_the_start_reserve_with_a_device(self):
        """`reserve_mib: None` with a `gpu`: the start's reserve on one card.  The reserve slot cannot be left
        empty - a number there IS the reserve - so the line names the start's reserve with a dash.  Without it
        `VRAM 1` asked for a 1 MiB reserve on every card of the run, not for device 1."""
        e = self.engine(["VRAM reserve_mib=700 gpu=1 expert_slots=3700 expert_slots_full=5632 "
                         "expert_cache_mib=4900 expert_cache_full_mib=8448 vram_free_mib=700 prompt_chunk=2048\n",
                         "VRAM-DONE\n"])
        out = e.vram(None, 1)
        self.assertEqual(e.proc.stdin.getvalue(), "VRAM - 1\n")
        self.assertEqual(out["gpu"], 1)
        self.assertEqual(out["reserve_mib"], 700)

    def test_every_gpu_of_a_split(self):
        e = self.engine(["VRAM reserve_mib=16000 gpu=0 expert_slots=1200 expert_cache_mib=4800 vram_free_mib=100\n",
                         "VRAM reserve_mib=16000 gpu=1 expert_slots=512 expert_cache_mib=2048 vram_free_mib=16100\n",
                         "VRAM-DONE\n"])
        out = e.vram(16000)
        self.assertEqual(e.proc.stdin.getvalue(), "VRAM 16000\n")
        self.assertEqual([p["gpu"] for p in out["gpus"]], [0, 1])
        self.assertEqual(out["expert_cache_mib"], 6848)          # summed over both cards
        self.assertEqual(out["vram_free_mib"], 100)              # the tightest card

    def test_an_engine_without_the_done_line(self):
        """An engine older than the per-GPU protocol answers with one line and no VRAM-DONE."""
        e = self.engine(["VRAM reserve_mib=6000 expert_slots=1200 expert_cache_mib=1536 vram_free_mib=6100\n"])
        out = e.vram(6000)
        self.assertEqual(out["expert_slots"], 1200)
        self.assertEqual(len(out["gpus"]), 1)

    def test_refused_and_died(self):
        e = self.engine(["ERR VRAM needs an engine started with --vram-elastic\n"])
        with self.assertRaisesRegex(ValueError, "--vram-elastic"):
            e.vram(100)
        e = self.engine(["ERR VRAM: this run has no GPU 3\n"])
        with self.assertRaisesRegex(ValueError, "no GPU 3"):
            e.vram(100, 3)
        e = self.engine([None])
        with self.assertRaises(EngineDied):
            e.vram(100)

    def test_config_key(self):
        cfg = {"args": ["--max-context", "8192"]}
        self.assertEqual(engine_args(cfg), ["--max-context", "8192"])           # off unless asked
        self.assertEqual(engine_args({**cfg, "vram_elastic": True, "vram_segment_mib": 256}),
                         ["--max-context", "8192", "--vram-elastic", "--vram-segment-mib", "256"])
        self.assertEqual(engine_args({**cfg, "vram_elastic": "yes"}), ["--max-context", "8192"])

    def test_the_elastic_keys_are_editable(self):
        """POST /config can turn elasticity on: the handoff to an image model on the second card is a run-config
        decision, and hand-editing the JSON is the kind of step that gets missed."""
        cfg = {"args": ["--max-context", "8192"]}
        self.assertEqual(runconfig.check("vram_elastic", True, cfg), True)
        self.assertEqual(runconfig.check("vram_segment_mib", 256, cfg), 256)
        with self.assertRaises(ValueError):
            runconfig.check("vram_elastic", "yes", cfg)
        new, changed = runconfig.apply(cfg, {"vram_elastic": True, "vram_segment_mib": 256})
        self.assertEqual(changed, ["vram_elastic", "vram_segment_mib"])
        self.assertEqual(engine_args(new), ["--max-context", "8192", "--vram-elastic", "--vram-segment-mib", "256"])
        self.assertEqual(runconfig.view(new, "strata-x.json")["keys"][-2]["key"], "vram_elastic")


class Route(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = VramEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.gpu_indices = [0, 1]                       # a layer split: two cards, engine numbers 0 and 1
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

    def test_resize_one_gpu(self):
        self.engine.loaded = True
        code, body = self.request("/v1/vram", {"reserve_mib": 4000, "gpu": 1})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["status"], "ok")
        per = {p["gpu"]: p["expert_cache_mib"] for p in body["gpus"]}
        self.assertEqual(per, {0: 6000, 1: 1700})           # only the named card gave VRAM back
        self.assertEqual(self.engine.calls, [(4000, 1)])
        self.assertEqual(self.request("/v1/status")[1]["vram"]["expert_cache_mib"], 7700)

    def test_resize_every_gpu_and_the_start_reserve(self):
        self.engine.loaded = True
        code, body = self.request("/v1/vram", {"reserve_mib": 4000})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["expert_cache_mib"], 4400)    # 2700 + 1700
        code, body = self.request("/v1/vram", {"reserve_mib": None})   # back to the start's reserve
        self.assertEqual(code, 200)
        self.assertEqual(self.engine.calls, [(4000, None), (None, None)])

    def test_the_start_reserve_on_one_card(self):
        """null + gpu: the named card takes its VRAM back, the other card is left alone."""
        self.engine.loaded = True
        self.assertEqual(self.request("/v1/vram", {"reserve_mib": 4000, "gpu": 1})[0], 200)
        code, body = self.request("/v1/vram", {"reserve_mib": None, "gpu": 1})
        self.assertEqual(code, 200, body)
        per = {p["gpu"]: p["expert_cache_mib"] for p in body["gpus"]}
        self.assertEqual(per, {0: 6000, 1: 5000})           # only the named card grew back
        self.assertEqual(self.engine.calls, [(4000, 1), (None, 1)])

    def test_bad_input_busy_and_foreign_pages(self):
        self.engine.loaded = True
        for bad in ({"reserve_mib": -1}, {"reserve_mib": "8 GiB"}, {"reserve_mib": True}, {"reserve_mib": 1.5},
                    {"reserve_mib": 1, "gpu": 2}, {"reserve_mib": 1, "gpu": -1}, {"reserve_mib": 1, "gpu": True},
                    {"reserve_mib": 1, "gpu": "1"}):
            self.assertEqual(self.request("/v1/vram", bad)[0], 400, bad)
        self.assertEqual(self.request("/v1/vram", {"reserve_mib": 1}, {"Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.request("/v1/vram", {"reserve_mib": 1}, {"Content-Type": "text/plain"})[0], 415)
        self.svc.vram_wait_s = 0.2
        with self.svc.fifo:                                   # a request is running
            self.assertEqual(self.request("/v1/vram", {"reserve_mib": 1})[0], 409)
        self.assertEqual(self.engine.calls, [])

    def test_an_engine_without_the_flag_says_so(self):
        self.engine.loaded, self.engine.elastic = True, False
        code, body = self.request("/v1/vram", {"reserve_mib": 3000})
        self.assertEqual(code, 400)
        self.assertIn("--vram-elastic", body["error"]["message"])

    def test_unloaded_keeps_the_reserve_for_the_next_load(self):
        code, body = self.request("/v1/vram", {"reserve_mib": 5000, "gpu": 1})
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "not loaded")
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.request("/v1/load", {})[0], 200)
        self.assertEqual(self.engine.calls, [(5000, 1)])


if __name__ == "__main__":
    unittest.main()
