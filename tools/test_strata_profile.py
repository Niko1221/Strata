"""Tests for tools/strata_profile.py: the engine-line parsers on real lines, the grouping, the findings, and the run
against a fake server that writes an engine log.  Light: no GPU, no Strata server, a few seconds."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_profile as P  # noqa: E402
from test_strata_bench import Handler, text_of  # noqa: E402

# lines an RTX 5070 Ti + Ryzen 7 5800X3D wrote (Swift 1.5 IQ3_S, 2026-10-15)
PREFILL_2K = ("strata prefill timing: 1882 tokens, GPU timeline 1761 ms, wall 1766 ms, host staging 87 ms: embed+steps 1 (0.0%) "
              "hc read 152 (8.6%) gdn 191 (10.8%) qsa proj 56 (3.2%) qsa indexer 0 (0.0%) qsa select 2 (0.1%) qsa attn 23 (1.3%) "
              "router+shared 23 (1.3%) host grouping 10 (0.5%) gather 6 (0.3%) wait copy 674 (38.3%) dequant 250 (14.2%) "
              "gemm gate/up 144 (8.2%) gemm down 87 (4.9%) combine 44 (2.5%) ple 3 (0.2%) kv stage 0 (0.0%) gdn conv+gates 11 (0.6%) "
              "gdn recurrence 35 (2.0%) gdn out proj 49 (2.8%)")
PREFILL_118K = ("strata prefill timing: 117888 tokens, GPU timeline 41220 ms, wall 41226 ms, host staging 878 ms: embed+steps 859 (2.1%) "
                "hc read 3616 (8.8%) gdn 4340 (10.5%) qsa proj 2499 (6.1%) qsa indexer 2 (0.0%) qsa select 2384 (5.8%) qsa attn 3030 (7.4%) "
                "router+shared 1035 (2.5%) host grouping 238 (0.6%) gather 371 (0.9%) wait copy 1453 (3.5%) dequant 3467 (8.4%) "
                "gemm gate/up 5266 (12.8%) gemm down 3233 (7.8%) combine 2269 (5.5%) ple 174 (0.4%) kv stage 1876 (4.6%) "
                "gdn conv+gates 675 (1.6%) gdn recurrence 2212 (5.4%) gdn out proj 2220 (5.4%)")
PREFILL_HOST = ("strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) 10 ms, waiting for each chunk 244 ms, "
                "after each chunk (the draft layer, progress) 446 ms, PLE 284 ms")
DECODE_2K = ("strata decode timing: 80 windows, avg T 2.70, 2.00 tokens/window, 30.30 ms/window = verify 27.22 (GPU-reach wait 10.87 "
             "+ per-layer host 14.94 [plan 0.09 actq 0.11 jobs 0.04 CPU 14.69] + stage 0.01) + commit/emit 0.03 + draft 1.62; "
             "per layer-window: CPU experts 3.63 (5.58 entries), VRAM hits 14.74, PCIe 2.71; adapt wait 1.41 join 0.02, 24.0 swaps/window")
DECODE_118K = ("strata decode timing: 79 windows, avg T 2.58, 2.03 tokens/window, 21.61 ms/window = verify 18.28 (GPU-reach wait 10.10 "
               "+ per-layer host 7.43 [plan 0.07 actq 0.08 jobs 0.02 CPU 7.23] + stage 0.01) + commit/emit 0.02 + draft 2.03; "
               "per layer-window: CPU experts 2.15 (2.79 entries), VRAM hits 20.47, PCIe 0.70; adapt wait 1.28 join 0.00, 23.1 swaps/window")
SUMMARY = ("strata serve: prompt 117895 tokens = 0 reused + 117895 read in 41513 ms (2840.0 tok/s), 160 generated in 1707 ms "
           "(93.7 tok/s), drafts accepted 82 of 125, 6 checkpoints")
SUMMARY_RESUMED = ("strata serve: prompt 118513 tokens = 117888 reused + 625 read in 935 ms (668.4 tok/s), 3 generated in 132 ms "
                   "(22.7 tok/s), drafts accepted 3 of 6, 6 checkpoints")
HIT = ("strata serve: decode expert cache hit rate: 88.0% (77632 hits / 88209 lookups); 2997 more read by the GPU over PCIe or "
       "from another GPU (3.3% of all 91206 routed)")
HIT_PLAIN = "strata serve: decode expert cache hit rate: 72.6% (1111 hits / 1530 lookups)"
KV = "strata serve: KV streaming: 94.82% of 1294032 block reads hit VRAM, 270.2 MiB read from RAM"
STAGES = ("strata decode GPU stages (ms/window): GDN layers: hc-read0 0.12 q8+qkv/q-idx gemv 0.30 conv 0.05 rec 0.41 out-proj 0.20 "
          "router+plan+bell 0.33 VRAM hits 3.10 waitCPU 2.20 copy+combine 0.40 | QSA layers: hc-read0 0.03 k/v+norm-rope 0.08 "
          "attention 0.90 VRAM hits 1.05 (gap) 0.11 head 0.44 | total 9.72 ms/window over 80 windows")


class ParserTests(unittest.TestCase):
    def test_prefill(self):
        d = P.parse_prefill(PREFILL_2K)
        self.assertEqual((d["tokens"], d["gpu_ms"], d["wall_ms"], d["host_staging_ms"]), (1882, 1761.0, 1766.0, 87.0))
        self.assertEqual(len(d["phases"]), 20)
        self.assertEqual(d["phases"]["wait copy"], 674.0)
        self.assertEqual(d["phases"]["gemm gate/up"], 144.0)
        self.assertEqual(d["phases"]["gdn conv+gates"], 11.0)
        self.assertEqual(d["phases"]["embed+steps"], 1.0)
        self.assertIsNone(P.parse_prefill(PREFILL_HOST))
        self.assertIsNone(P.parse_prefill(DECODE_2K))

    def test_prefill_host(self):
        d = P.parse_prefill_host(PREFILL_HOST)
        self.assertEqual(d, {"chunk_setup_ms": 10.0, "wait_chunk_ms": 244.0, "after_chunk_ms": 446.0, "ple_ms": 284.0})

    def test_decode(self):
        d = P.parse_decode(DECODE_2K)
        self.assertEqual(d["windows"], 80)
        self.assertEqual((d["avg_T"], d["tokens_per_window"], d["ms_per_window"]), (2.70, 2.00, 30.30))
        self.assertEqual((d["verify"], d["gpu"], d["host"], d["host_cpu"], d["stage"]), (27.22, 10.87, 14.94, 14.69, 0.01))
        self.assertEqual((d["commit"], d["draft"]), (0.03, 1.62))
        self.assertEqual((d["cpu_experts"], d["cpu_entries"], d["vram_hits"], d["pcie"]), (3.63, 5.58, 14.74, 2.71))
        self.assertEqual((d["adapt_wait"], d["adapt_join"], d["swaps"]), (1.41, 0.02, 24.0))
        self.assertIsNone(P.parse_decode(SUMMARY))

    def test_gpu_stages(self):
        d = P.parse_gpu_stages(STAGES)
        self.assertEqual(d["gdn"]["q8+qkv/q-idx gemv"], 0.30)
        self.assertEqual(d["gdn"]["VRAM hits"], 3.10)
        self.assertEqual(d["gdn"]["router+plan+bell"], 0.33)
        self.assertEqual(d["qsa"]["k/v+norm-rope"], 0.08)
        self.assertEqual(d["qsa"]["(gap)"], 0.11)
        self.assertEqual((d["total"], d["windows"]), (9.72, 80))
        self.assertEqual(len(d["gdn"]), 9)
        self.assertEqual(len(d["qsa"]), 6)

    def test_summary_hit_kv(self):
        s = P.parse_summary(SUMMARY)
        self.assertEqual((s["prompt_tokens"], s["reused"], s["read"], s["prompt_ms"], s["prompt_tok_s"]), (117895, 0, 117895, 41513.0, 2840.0))
        self.assertEqual((s["generated"], s["decode_ms"], s["decode_tok_s"], s["drafts_accepted"], s["drafts_offered"]), (160, 1707.0, 93.7, 82, 125))
        r = P.parse_summary(SUMMARY_RESUMED)
        self.assertEqual((r["reused"], r["read"]), (117888, 625))
        c = P.parse_summary(SUMMARY.replace("117895 read", "500 of 117895 read"))   # cancelled mid-read
        self.assertEqual(c["read"], 500)
        h = P.parse_hit_rate(HIT)
        self.assertEqual((h["hit_rate"], h["hits"], h["lookups"], h["offloaded"], h["pcie_share"], h["routed"]), (0.88, 77632, 88209, 2997, 0.033, 91206))
        h2 = P.parse_hit_rate(HIT_PLAIN)
        self.assertEqual((h2["hit_rate"], h2["pcie_share"], h2["offloaded"]), (0.726, 0.0, 0))
        k = P.parse_kv(KV)
        self.assertEqual((k["kv_hit"], k["block_reads"], k["ram_mib"], k["misses"]), (0.9482, 1294032, 270.2, 67075))

    def test_kv_request_is_the_difference_of_two_cumulative_lines(self):
        a = P.parse_kv(KV)
        b = P.parse_kv("strata serve: KV streaming: 94.89% of 1386456 block reads hit VRAM, 285.2 MiB read from RAM")
        d = P.kv_request(b, a)
        self.assertEqual(d["block_reads"], 1386456 - 1294032)
        self.assertEqual(d["ram_mib"], 15.0)
        self.assertAlmostEqual(d["kv_hit"], 1 - (70799 - 67075) / (1386456 - 1294032), places=4)
        self.assertIsNone(P.kv_request(b, None))
        self.assertIsNone(P.kv_request(a, b))                      # counts went backwards: a new engine

    def test_engine_facts_takes_the_last_of_each_kind(self):
        f = P.engine_facts([PREFILL_2K, DECODE_2K, "strata verify: something else", PREFILL_118K, PREFILL_HOST, DECODE_118K,
                            SUMMARY, HIT, KV, STAGES])
        self.assertEqual(f["prefill"]["tokens"], 117888)
        self.assertEqual(f["decode"]["windows"], 79)
        self.assertEqual(f["summary"]["generated"], 160)
        self.assertEqual(f["hit"]["hit_rate"], 0.88)
        self.assertEqual(f["kv"]["kv_hit"], 0.9482)
        self.assertEqual(f["gpu_stages"]["total"], 9.72)
        self.assertEqual(f["prefill_host"]["ple_ms"], 284.0)
        self.assertEqual(P.engine_facts(["nothing here"]), {})


class GroupingTests(unittest.TestCase):
    def test_prompt_groups(self):
        g = P.prompt_groups(P.parse_prefill(PREFILL_2K))
        self.assertAlmostEqual(g["ms"]["experts over PCIe"], 674 + 6 + 10)
        self.assertAlmostEqual(g["ms"]["dequant"], 250)
        self.assertAlmostEqual(g["ms"]["expert GEMM"], 144 + 87 + 44)
        self.assertAlmostEqual(g["ms"]["attention"], 56 + 0 + 2 + 23 + 0)
        self.assertAlmostEqual(g["ms"]["GDN"], 191 + 11 + 35 + 49)
        self.assertAlmostEqual(g["ms"]["other"], 1 + 152 + 23 + 3)
        self.assertAlmostEqual(sum(g["share"].values()), sum(g["ms"].values()) / 1761)
        self.assertEqual(g["top"][0], ("wait copy", 674.0))
        self.assertIsNone(P.prompt_groups(None))

    def test_a_phase_the_engine_adds_later_lands_in_other(self):
        pf = P.parse_prefill(PREFILL_2K)
        pf["phases"]["new thing"] = 100.0
        g = P.prompt_groups(pf)
        self.assertAlmostEqual(g["ms"]["other"], 1 + 152 + 23 + 3 + 100)

    def test_decode_groups(self):
        g = P.decode_groups(P.parse_decode(DECODE_2K))
        self.assertEqual(g["ms"]["GPU"], 10.87)
        self.assertEqual(g["ms"]["CPU experts"], 14.94)
        self.assertEqual(g["ms"]["draft"], 1.62)
        self.assertEqual(g["ms"]["adapt wait"], 1.41)
        self.assertAlmostEqual(g["ms"]["other"], 30.30 - 10.87 - 14.94 - 1.62 - 1.41)
        self.assertAlmostEqual(sum(g["share"].values()), 1.0)
        self.assertEqual((g["cpu_experts"], g["cpu_entries"]), (3.63, 5.58))
        self.assertIsNone(P.decode_groups(None))

    def test_merge_decode_weights_by_windows(self):
        a, b = P.parse_decode(DECODE_2K), P.parse_decode(DECODE_118K)
        m = P.merge_decode([a, b, None])
        self.assertEqual(m["windows"], 159)
        self.assertAlmostEqual(m["ms_per_window"], (30.30 * 80 + 21.61 * 79) / 159)
        self.assertAlmostEqual(m["gpu"], (10.87 * 80 + 10.10 * 79) / 159)
        self.assertIsNone(P.merge_decode([None]))


def decode_result(line, context, tps, hit=None, kv=None):
    d = P.parse_decode(line)
    return {"context": context, "decode_tps": {"median": tps}, "groups": P.decode_groups(d), "decode": d,
            "hit": {"hit_rate": hit} if hit is not None else None, "summary": None, "kv": kv}


class FindingsTests(unittest.TestCase):
    def test_pcie_bound_read_and_compute_bound_read(self):
        res = {"prompts": {"2K": {"groups": P.prompt_groups(P.parse_prefill(PREFILL_2K)), "prompt_tps": 1066.0, "summary": None},
                           "118K": {"groups": P.prompt_groups(P.parse_prefill(PREFILL_118K)), "prompt_tps": 2840.0, "summary": None}}}
        f = P.findings(res)
        self.assertEqual(len(f), 3)
        self.assertIn("waits for the experts streamed over PCIe: 39% of its GPU time", f[0])
        self.assertIn("1,882 tokens, 1,066 tok/s", f[0])
        self.assertIn("compute-bound", f[1])
        self.assertIn("expert GEMM 26%", f[1])
        self.assertNotIn("decides", f[1])
        self.assertIn("STRATA_PREFILL_CPU_SHARE_MAX", f[2])                     # the advice once, last
        self.assertIn("measure it here before setting it", f[2])
        self.assertIn("not a sure win", f[2])

    def test_cpu_bound_and_gpu_bound_decode(self):
        res = {"decode": {"chat": decode_result(DECODE_2K, 1889, 66.0, hit=0.726),
                          "long context": decode_result(DECODE_118K, 117895, 93.7, hit=0.88, kv=P.parse_kv(KV))}}
        f = P.findings(res)
        self.assertIn("the experts computed on the CPU 14.9 ms (3.6 experts, 5.6 entries per layer-window)", f[0])
        self.assertIn("misses of the VRAM expert cache take the larger share", f[0])
        self.assertIn("expert cache hit rate 73%", f[0])
        self.assertIn("the card takes the larger share", f[1])
        self.assertEqual(len(f), 3)                         # the draft is under 15%, the KV reads over 90%: nothing to say
        self.assertIn("expert_profile_save", f[2])          # the advice once, last

    def test_draft_share_and_kv_streaming(self):
        d = P.parse_decode(DECODE_118K)
        d["draft"] = 6.0                                     # 28% of the window
        kv = P.parse_kv(KV.replace("94.82%", "71.00%"))
        res = {"decode": {"long context": {"context": 117895, "decode_tps": {"median": 50.0}, "groups": P.decode_groups(d), "decode": d,
                                           "hit": None, "summary": None, "kv": kv}}}
        f = P.findings(res)
        self.assertEqual(len(f), 3)                         # GPU-bound: no cache advice
        self.assertIn("the draft takes 28% of the window", f[1])
        self.assertIn("71% of the KV block reads hit VRAM (270 MiB read from RAM): --kv-resident", f[2])

    def test_agent_turns(self):
        turns = [{"turn": t, "hit": {"hit_rate": h}, "summary": {"read": r}}
                 for t, h, r in ((0, 0.80, 300), (1, 0.49, 2100), (2, 0.76, 2050), (3, 0.85, 2080))]
        f = P.findings({"agent": {"turns": turns}})
        self.assertIn("runs from 49% (turn 1) to 85% (turn 3)", f[0])
        self.assertIn("read 2,050-2,100 tokens", f[1])

    def test_no_lines_no_findings(self):
        self.assertEqual(P.findings({"prompts": {"2K": {"groups": None}}, "decode": {"chat": {"groups": None}}, "agent": {"turns": []}}), [])


class LoggingHandler(Handler):
    """The bench's fake Strata, and after every answer the engine lines a real engine would have written to its log."""

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        req = json.loads(raw)
        self.rfile = _Replay(raw)                        # the parent reads the body again
        super().do_POST()
        s = self.server
        if not s.log_path:
            return
        text = text_of(req["messages"])
        prompt = max(1, len(text) // 4)
        gen = min(req.get("max_tokens", 16), 12)
        with s.lock:
            s.engine_requests += 1
            s.kv_reads += prompt * 10
            s.kv_miss_mib += round(prompt * 10 * 0.05 * 4224 / 1048576, 1)
            reads, mib = s.kv_reads, s.kv_miss_mib
        before = [PREFILL_2K.replace("1882 tokens", f"{prompt} tokens"), PREFILL_HOST, DECODE_2K]
        if s.gpu_stages:
            before.append(STAGES)
        # a real engine writes the summary lines after its DONE, i.e. after the answer went out: so does this one
        after = [f"strata serve: prompt {prompt} tokens = 0 reused + {prompt} read in 100 ms (1000.0 tok/s), {gen} generated in 200 ms "
                 f"(60.0 tok/s), drafts accepted 5 of 9, 2 checkpoints", HIT_PLAIN,
                 f"strata serve: KV streaming: 95.00% of {reads} block reads hit VRAM, {mib} MiB read from RAM"]
        with open(s.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(before) + "\n")
        threading.Timer(0.15, lambda: open(s.log_path, "a", encoding="utf-8").write("\n".join(after) + "\n")).start()


class _Replay:
    def __init__(self, data):
        self.data = data

    def read(self, n=-1):
        d, self.data = self.data, b""
        return d

    def close(self):
        pass


class FakeStrata(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, ctx=65536, log_path=None, gpu_stages=False):
        super().__init__(("127.0.0.1", 0), LoggingHandler)
        self.ctx, self.prev, self.requests, self.fail_after = ctx, "", 0, None
        self.engine_requests, self.log_path, self.gpu_stages = 0, log_path, gpu_stages
        self.kv_reads, self.kv_miss_mib = 0, 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def stop(self):
        self.shutdown()
        self.server_close()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "engine.log"
        self.log.write_text("strata serve: ready\n", encoding="utf-8")

    def profile(self, srv, log=True, **kw):
        return P.Profile(srv.base, "", srv.ctx, P.EngineLog(self.log if log else None), say=lambda *a: None, **kw)

    def test_the_run_reads_every_request_s_lines(self):
        srv = FakeStrata(log_path=self.log)
        self.addCleanup(srv.stop)
        out = self.profile(srv).run()
        self.assertEqual(out["failures"], [])
        self.assertEqual(set(out["prompts"]), {"2K", "16K", "32K (long)"})
        for label, r in out["prompts"].items():
            self.assertIsNotNone(r["groups"], label)
            self.assertEqual(r["summary"]["prompt_tok_s"], 1000.0)
            self.assertEqual(r["engine"]["prefill"]["tokens"], r["prompt_tokens"])   # this request's own line, not an older one
        self.assertEqual(set(out["decode"]), {"chat", "long context", "agent turns"})
        self.assertEqual(out["decode"]["chat"]["decode"]["windows"], 240)            # three questions merged
        self.assertEqual(out["decode"]["chat"]["hit"]["hit_rate"], 0.726)
        kv = out["decode"]["long context"]["kv"]                                  # this request's reads, not the cumulative line
        self.assertEqual(kv["block_reads"], out["prompts"]["32K (long)"]["prompt_tokens"] * 10)
        self.assertAlmostEqual(kv["kv_hit"], 0.95, places=2)
        self.assertEqual(len(out["agent"]["turns"]), 6)
        self.assertGreater(out["agent"]["turns"][2]["reused_tokens"], 0)
        self.assertTrue(any("waits for the experts streamed over PCIe" in f for f in out["findings"]))
        self.assertTrue(any("misses of the VRAM expert cache" in f for f in out["findings"]))
        self.assertIsNone(out["decode"]["chat"]["gpu_stages"])
        # every request got its own summary, written after the answer: the hit rate is the engine's, not /metrics
        for r in out["chat"]["requests"]:
            self.assertEqual(r["engine"]["summary"]["generated"], 12)
            self.assertNotIn("source", r["engine"]["hit"])
        for t in out["agent"]["turns"]:
            self.assertEqual(t["summary"]["prompt_tokens"], t["prompt_tokens"])

    def test_without_a_log_the_api_figures_remain(self):
        srv = FakeStrata()
        self.addCleanup(srv.stop)
        out = self.profile(srv, log=False).run()
        self.assertEqual(out["failures"], [])
        self.assertIsNone(out["prompts"]["2K"]["groups"])
        self.assertEqual(out["prompts"]["2K"]["prompt_tps"], 1000.0 + 8)            # the fake counts its requests (4 warm-up, 3 chat)
        self.assertIsNone(out["decode"]["chat"]["groups"])
        self.assertEqual(out["decode"]["chat"]["decode_tps"]["median"], 60.0)
        self.assertEqual(out["findings"], [])
        self.assertIn("## Decode windows", P.render({"machine": {}, "results": out, "date": "2026-10-15", "params": {"engine_log": False}}))

    def test_a_small_context_skips_what_does_not_fit(self):
        srv = FakeStrata(ctx=8192, log_path=self.log)
        self.addCleanup(srv.stop)
        out = self.profile(srv).run()
        self.assertEqual(set(out["prompts"]), {"2K"})
        self.assertEqual(set(out["skipped"]), {"16K", "long", "agent"})
        self.assertEqual(set(out["decode"]), {"chat"})

    def test_long_can_be_sized_or_off(self):
        srv = FakeStrata(log_path=self.log)
        self.addCleanup(srv.stop)
        out = self.profile(srv, long_tokens=0).run()
        self.assertNotIn("long context", out["decode"])
        out = self.profile(srv, long_tokens=16384).run()
        self.assertEqual(set(out["prompts"]), {"2K", "16K", "16K (long)"})   # a --long of a suite size overwrites nothing

    def test_gpu_stages_are_reported_when_the_engine_prints_them(self):
        srv = FakeStrata(log_path=self.log, gpu_stages=True)
        self.addCleanup(srv.stop)
        out = self.profile(srv, long_tokens=0).run()
        self.assertEqual(out["decode"]["chat"]["gpu_stages"]["total"], 9.72)
        text = P.render({"machine": {"gpu": ["X"]}, "results": out, "date": "2026-10-15", "params": {}})
        self.assertIn("GDN VRAM hits 3.10", text)

    def test_the_log_is_consumed_whole_lines_at_a_time(self):
        log = P.EngineLog(self.log)
        with open(self.log, "a") as f:
            f.write(DECODE_2K + "\n" + SUMMARY + "\n" + HIT_PLAIN[:20])             # the hit line is still being written
        f1 = log.facts(timeout=0.2, settle=0.05)
        self.assertEqual(set(f1), {"decode", "summary"})
        with open(self.log, "a") as f:
            f.write(HIT_PLAIN[20:] + "\n")
        f2 = log.facts(wait_for=("hit",), timeout=0.5, settle=0.05)
        self.assertEqual(set(f2), {"hit"})                                        # the half line was not lost, nor counted twice
        self.assertEqual(log.facts(wait_for=(), timeout=0.1, settle=0.0), {})

    def test_a_failed_request_is_reported_not_hidden(self):
        srv = FakeStrata(log_path=self.log)
        srv.fail_after = 3
        self.addCleanup(srv.stop)
        out = self.profile(srv).run()
        self.assertTrue(out["failures"])
        self.assertTrue(all("boom" in f["error"] for f in out["failures"]))


class ConfigAndReportTests(unittest.TestCase):
    def test_profile_config_adds_the_switches_and_keeps_the_users_file(self):
        cfg = {"exe": "x", "args": ["--expert-profile-save", "p.bin", "--max-context", "8192"], "env": {"A": "1"}, "log": "old.log", "port": 8080}
        before = json.dumps(cfg)
        out = P.profile_config(cfg, Path("/tmp/e.log"), verify_profile=False)
        self.assertEqual(out["env"], {"A": "1", "STRATA_PREFILL_TIMING": "1", "STRATA_DECODE_TIMING": "1"})
        self.assertEqual(out["args"], ["--max-context", "8192"])
        self.assertEqual(out["log"], "/tmp/e.log")
        self.assertEqual(json.dumps(cfg), before)
        self.assertIn("STRATA_VERIFY_PROFILE", P.profile_config(cfg, Path("e.log"), verify_profile=True)["env"])
        self.assertNotIn("STRATA_VERIFY_PROFILE", out["env"])

    def test_render_and_write(self):
        res = {"schema": 1, "date": "2026-10-15", "params": {"engine_log": True, "long": 32768},
               "machine": {"gpu": ["NVIDIA GeForce RTX 5070 Ti"], "gpu_vram_mib": 16303, "cpu": "AMD Ryzen 7 5800X3D", "cpu_cores": 8,
                           "ram_gib": 76.0, "os": "Linux 7.0", "model": "swift-1.5-iq3_s", "context": 262144, "launch": "/home/someone/strata --x"},
               "results": {"prompts": {"2K": {"prompt_tokens": 1882, "prompt_tps": 1066.0, "summary": None,
                                              "groups": P.prompt_groups(P.parse_prefill(PREFILL_2K))}},
                           "decode": {"chat": decode_result(DECODE_2K, 1889, 66.0, hit=0.726)},
                           "agent": {"turns": [{"turn": 0, "prompt_tokens": 400, "reused_tokens": 0, "read_tokens": 400, "prompt_tps": 900.0,
                                                "decode_tps": 70.0, "hit": {"hit_rate": 0.8}, "groups": P.decode_groups(P.parse_decode(DECODE_2K))}]},
                           "skipped": {"16K": "the context is 8192 tokens"}, "failures": [], "findings": ["one", "two"]}}
        text = P.render(res)
        self.assertIn("| 2K fresh | 1,882 | 1,066 | 1,761 | 39% | 14% | 16% | 5% | 16% | 10% | wait copy 38%, dequant 14%, gdn 11% |", text)
        self.assertIn("| chat | 1,889 | 66.0 | 73% | - | 30.3 | 10.9 | 14.9 | 1.6 | 1.4 | 1.5 | 2.00 | 3.6 | 5.6 |", text)
        self.assertIn("| 0 | 400 | 0 | 400 | 900 | 70 | 80% | 30.3 | 10.9 | 14.9 |", text)
        self.assertIn("- one\n- two", text)
        self.assertIn("Skipped: 16K (the context is 8192 tokens)", text)
        with tempfile.TemporaryDirectory() as tmp:
            red = lambda s: s.replace("/home/someone", "<home>") if isinstance(s, str) else s   # noqa: E731
            d = P.write_results(res, Path(tmp), red)
            self.assertEqual(d.name, "2026-10-15-profile-rtx-5070-ti-swift-1-5-iq3-s")
            readme = (d / "README.md").read_text(encoding="utf-8")
            self.assertIn("<home>/strata --x", readme)
            self.assertNotIn("/home/someone", readme)
            saved = json.loads((d / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["machine"]["launch"], "<home>/strata --x")

    def test_the_temporary_config_is_private_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            server_log = Path(tmp) / "s.log"
            with mock.patch("subprocess.Popen") as popen:
                P.start_server({"exe": "x", "args": [], "api_key": "secret"}, 8099, None, server_log, Path(tmp) / "e.log", False)
            self.assertTrue(popen.called)
            tmp_cfg = server_log.with_suffix(".config.json")
            self.assertEqual(tmp_cfg.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(tmp_cfg.read_text())["env"]["STRATA_DECODE_TIMING"], "1")
            P.remove_config(server_log)
            self.assertFalse(tmp_cfg.exists())
            P.remove_config(server_log)                                            # twice is fine

    def test_main_against_a_running_server_without_its_log(self):
        srv = FakeStrata()
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as tmp:
            rc = P.main(["--url", srv.base, "--out", tmp, "--quiet", "--long", "0"])
            self.assertEqual(rc, 0)
            dirs = list(Path(tmp).iterdir())
            self.assertEqual(len(dirs), 1)
            readme = (dirs[0] / "README.md").read_text(encoding="utf-8")
            self.assertIn("No engine log was read", readme)
            self.assertIn("| chat |", readme)

    def test_no_server_is_an_error_not_a_crash(self):
        self.assertEqual(P.main(["--url", "http://127.0.0.1:9", "--quiet"]), 2)


if __name__ == "__main__":
    unittest.main()
