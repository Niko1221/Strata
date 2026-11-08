"""EngineHttp：MeZO 的**在线** transport（驱动驻留推理引擎），对桩 HTTP 服务验证接口语义。

与 test_engine_api.py 的分工：那一个守的是 `/v1/training/*`（离线，卸载推理引擎）；这里守的是
`/v1/engine/score` + `/v1/engine/lora`（在线，不卸载）。两者的 score/lora 必须对 MeZO 循环同构，唯一的硬差别
是 `gen` 不可用（服务端没有按 ids 生成的端点）。

最容易静默错的一条是长度契约：引擎的 `SL` 只发 n-1 行，而 `/v1/engine/score` 为了让 `logprobs` 与
`tokens` 对齐补了一个末位 null。客户端若照抄 n 个，answer 段的切片会整体错一位 —— 损失照样能算
出来、曲线照样往下走，只是学的东西不是要学的。这里把它钉死。
"""

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import http.server
import json
import threading
import unittest

import mezo


class StubServeApi(http.server.BaseHTTPRequestHandler):
    """仿真 serve/server.py 的 /v1/engine/score 与 /v1/engine/lora。calls 记录请求；fail_* 注入错误路径。"""

    calls: list = []
    score_fail = 0           # >0: /v1/engine/score 返回这个状态码
    lora_fail = 0            # >0: /v1/engine/lora 返回这个状态码

    def log_message(self, *a):
        pass

    def _reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        StubServeApi.calls.append((self.path, body))
        if self.path == "/v1/engine/score":
            if StubServeApi.score_fail:
                self._reply(StubServeApi.score_fail,
                            {"error": {"message": "the sequence is 9 tokens; this server holds 8"}})
                return
            ids = body.get("tokens") or []
            # 服务端的真实口径：n 个 figure，末位永远是 null（对齐 tokens），见 service.score()
            lps = [-0.5] * max(len(ids) - 1, 0) + [None]
            self._reply(200, {"object": "score", "model": "stub", "tokens": ids, "n": len(ids),
                              "scored": len(ids) - 1, "logprobs": lps,
                              "sum_logprob": -0.5 * max(len(ids) - 1, 0), "mean_logprob": -0.5,
                              "nll": 0.5 * max(len(ids) - 1, 0), "perplexity": 1.65})
        elif self.path == "/v1/engine/lora":
            if StubServeApi.lora_fail:
                self._reply(StubServeApi.lora_fail,
                            {"error": {"message": "this engine cannot swap an adapter; "
                                                  "start it with --lora <the initial one>"}})
                return
            self._reply(200, {"status": "ok", "path": body.get("path"), "digest": 7,
                              "summary": "1 layer", "note": "every cached K/V was dropped"})
        else:
            self._reply(404, {"error": {"message": "not found"}})


class EngineHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubServeApi)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        StubServeApi.calls = []
        StubServeApi.score_fail = 0
        StubServeApi.lora_fail = 0

    # ---- score -------------------------------------------------------------------------------

    def test_score_drops_the_alignment_slot(self):
        eng = mezo.EngineHttp(self.base)
        ids = [10, 11, 12, 13, 14, 15, 16, 17]
        lps, _ms = eng.score(ids)
        # 桩回 8 个（末位 null），客户端必须只留 7 个 —— 与引擎的 `t <= n-2` 逐位同构
        self.assertEqual(len(lps), len(ids) - 1)
        self.assertTrue(all(v == -0.5 for v in lps))

    def test_score_posts_tokens_not_ids(self):
        eng = mezo.EngineHttp(self.base)
        eng.score([1, 2, 3])
        path, body = StubServeApi.calls[-1]
        self.assertEqual(path, "/v1/engine/score")
        self.assertEqual(body, {"tokens": [1, 2, 3]})

    def test_score_error_carries_the_server_reason(self):
        StubServeApi.score_fail = 400
        eng = mezo.EngineHttp(self.base)
        with self.assertRaises(RuntimeError) as cm:
            eng.score([1, 2, 3])
        self.assertIn("9 tokens", str(cm.exception))
        self.assertIn("HTTP 400", str(cm.exception))

    # ---- lora --------------------------------------------------------------------------------

    def test_lora_posts_the_path_unchanged(self):
        eng = mezo.EngineHttp(self.base)
        eng.lora("/tmp/plus.gguf")
        path, body = StubServeApi.calls[-1]
        self.assertEqual(path, "/v1/engine/lora")
        self.assertEqual(body, {"path": "/tmp/plus.gguf"})

    def test_lora_error_carries_the_server_reason(self):
        StubServeApi.lora_fail = 400
        eng = mezo.EngineHttp(self.base)
        with self.assertRaises(RuntimeError) as cm:
            eng.lora("/tmp/plus.gguf")
        self.assertIn("cannot swap an adapter", str(cm.exception))

    # ---- the two things that make this transport "online" -------------------------------------

    def test_gen_is_unavailable_and_says_so(self):
        eng = mezo.EngineHttp(self.base)
        self.assertFalse(eng.supports_gen)
        with self.assertRaises(RuntimeError) as cm:
            eng.gen([1, 2, 3], 8)
        self.assertIn("no generate-from-ids endpoint", str(cm.exception))

    def test_close_never_touches_the_engine(self):
        """close() 不能卸载推理引擎 —— 那正是「无感」的定义（TrainBridge 才会 unload）。"""
        eng = mezo.EngineHttp(self.base)
        eng.close()
        self.assertEqual(StubServeApi.calls, [])

    def test_base_url_trailing_slash_is_tolerated(self):
        eng = mezo.EngineHttp(self.base + "/")
        eng.lora("/tmp/x.gguf")
        self.assertEqual(StubServeApi.calls[-1][0], "/v1/engine/lora")


class EngineHttpWiringTest(unittest.TestCase):
    """main() 的选路：两条 HTTP transport 互斥；选中在线那条时**绝不能**自己 spawn 引擎。

    后者是「无感」的硬不变量：一旦 `launch_engine` 被调到，`mem_guard.require_exclusive_gpu`
    就会要求独占，驻留的推理引擎被判定为冲突 —— 整个在线形态就没了。
    """

    def test_both_transports_is_an_error(self):
        from unittest import mock
        argv = ["mezo.py", "--pack", "p", "--gguf", "g",
                "--engine-api", "http://a", "--serve-url", "http://b"]
        with mock.patch.object(_s, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                mezo.main()
        self.assertIn("只能选一个", str(cm.exception))

    def test_serve_url_never_spawns_an_engine(self):
        import tempfile
        from unittest import mock
        work = tempfile.mkdtemp(prefix="mezo-online-")
        argv = ["mezo.py", "--pack", "p", "--gguf", "g", "--work", work,
                "--serve-url", "http://127.0.0.1:1", "--steps", "0"]
        geom = {"block_count": 1, "embedding_length": 8, "full_attention_interval": 1}
        # main() 现在把 system / reasoning_effort 一起传给 load_template（渲染契约），
        # mock 必须收 **kw，否则测的是旧签名而不是线上模式。
        with mock.patch.object(_s, "argv", argv), \
                mock.patch.object(mezo, "load_tokenizer", lambda p: object()), \
                mock.patch.object(mezo, "load_template",
                                  lambda p, **kw: (lambda q: q)), \
                mock.patch.object(mezo, "read_geometry", lambda g: geom), \
                mock.patch.object(mezo, "adapter_layout", lambda g: {0: ("ssm_out", 4, 4)}), \
                mock.patch.object(mezo, "launch_engine",
                                  side_effect=AssertionError("online mode must not spawn an engine")), \
                mock.patch.object(mezo, "EngineHttp",
                                  side_effect=SystemExit("online transport chosen")):
            with self.assertRaises(SystemExit) as cm:
                mezo.main()
        self.assertIn("online transport chosen", str(cm.exception))


class TestPrefillChunk(unittest.TestCase):
    """`--prefill` 是 prompt 的**分块大小**，不是长度上限；own buffer 下它是整个会话的预留。

    钉死的是两件事：① 默认值由数据里最长样本定（对齐 256），不是常数 8192；② 用户显式给了
    就照用（长上下文数据只有用户知道）。8192 在 300-600 token 的数学数据上就是白占专家缓存。
    """

    def test_no_data_falls_back_to_the_engines_own_choice(self):
        self.assertEqual(mezo.resolve_prefill_chunk(0), "auto")

    def test_the_longest_sample_decides_and_is_aligned_to_the_256_grid(self):
        # 336 -> 512：request_chunk 也按 256 取整，所以「一条样本一块读完」等价于对齐后的值
        self.assertEqual(mezo.resolve_prefill_chunk(336), "512")
        self.assertEqual(mezo.resolve_prefill_chunk(512), "512")
        self.assertEqual(mezo.resolve_prefill_chunk(513), "768")

    def test_never_below_one_grid_step(self):
        self.assertEqual(mezo.resolve_prefill_chunk(1), "256")

    def test_capped_at_the_engine_auto_ceiling(self):
        # 128K 数据：8192 一块，按块数读完 —— 不是截断；要一块读完得显式给 --prefill-chunk
        self.assertEqual(mezo.resolve_prefill_chunk(131072), "8192")

    def test_an_explicit_choice_wins(self):
        self.assertEqual(mezo.resolve_prefill_chunk(336, "8192"), "8192")
        self.assertEqual(mezo.resolve_prefill_chunk(131072, "auto"), "auto")


class TestEngineArgvCarriesTheChunk(unittest.TestCase):
    def _argv(self, chunk):
        from types import SimpleNamespace
        args = SimpleNamespace(engine="strata.exe", pack="p", gguf="g", expert_profile="e",
                               max_context=4096, prefill_chunk=chunk)
        return mezo.engine_argv(args, "init.gguf")

    def test_the_chunk_reaches_the_engine(self):
        a = self._argv("512")
        self.assertEqual(a[a.index("--prefill") + 1], "512")

    def test_an_empty_choice_is_left_to_the_engine(self):
        a = self._argv("")
        self.assertEqual(a[a.index("--prefill") + 1], "auto")

    def test_borrow_stays_off(self):
        # ── 这条与 chunk 无关，但同样只能由启动参数保证（BRANCH-NOTES §4.3）
        self.assertIn("--no-prefill-borrow", self._argv("512"))


if __name__ == "__main__":
    unittest.main()
