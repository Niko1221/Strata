"""EngineApi：MeZO 训练的 HTTP transport，对桩 HTTP 服务（仿真 /v1/training/*）验证接口语义。

真实引擎的端到端（真 TRAINING 引擎 + 真 server.py 路由）在 serve/test_training_api.py；这里守的是
客户端自己的契约：score/lora/gen 的解析与 Engine（pipe）逐位同构、ERR→RuntimeError、桥死→会话关、
close 幂等。"""

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


class StubTrainingApi(http.server.BaseHTTPRequestHandler):
    """仿真 server.py 的三个训练端点。calls 记录请求；fail 参数注入错误路径。"""

    calls: list = []
    fail_open = 0            # >0: session open 返回这个状态码
    fail_mode = ""           # "err"（引擎说 ERR）/ "died"（桥死，服务端关会话）/ "lost"（409 会话没了）

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
        StubTrainingApi.calls.append((self.path, body))
        if self.path == "/v1/training/session":
            if body.get("op") == "open":
                if StubTrainingApi.fail_open:
                    self._reply(StubTrainingApi.fail_open,
                                {"error": {"message": "a training session is already open"}})
                    return
                self._reply(200, {"status": "open", "pid": 4242, "log": "stub"})
            else:
                self._reply(200, {"status": "closed", "was_active": True})
        elif self.path == "/v1/training/command":
            line = body.get("line", "")
            mode = StubTrainingApi.fail_mode
            if mode == "err":
                self._reply(502, {"error": {"message": "the engine said: ERR score: bad ids"}})
            elif mode == "died":
                self._reply(502, {"error": {"message": "the training engine died mid-command "
                                                        "(see the bridge's log)"}})
            elif mode == "lost":
                self._reply(409, {"error": {"message": "a training session owns the engine"}})
            elif line.startswith("LORA "):
                self._reply(200, {"lines": ["LORAOK 7 deadbeef"]})
            elif line.startswith("SCORE "):
                n_ids = len([x for x in line.split(" ", 1)[1].split(",") if x])
                lines = [f"SL {t} -0.5" for t in range(n_ids - 1)]
                lines.append(f"SSUM {-0.5 * (n_ids - 1):.6f} {n_ids - 1} 12.0 3.0 64")
                self._reply(200, {"lines": lines})
            elif line.startswith("GEN "):
                self._reply(200, {"lines": ["T 42", "DONE"]})
            else:
                self._reply(502, {"error": {"message": "the engine said: ERR unknown command"}})
        else:
            self._reply(404, {"error": {"message": "not found"}})


class EngineApiAgainstStub(unittest.TestCase):
    def setUp(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubTrainingApi)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        t.start()
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)
        StubTrainingApi.calls = []
        StubTrainingApi.fail_open = 0
        StubTrainingApi.fail_mode = ""

    def eng(self):
        return mezo.EngineApi(self.base, ["strata-train", "--lora", "init.gguf"])

    def test_open_score_lora_gen_match_pipe_semantics(self):
        eng = self.eng()
        self.addCleanup(eng.close)
        self.assertEqual(StubTrainingApi.calls[0], ("/v1/training/session", {"op": "open", "argv": eng.argv}))

        slp, ms = eng.score([1, 2, 3])
        self.assertEqual(slp, [-0.5, -0.5])               # n ids -> n-1 个逐位 logp，与 pipe 的 SL 解析同构
        self.assertGreater(ms, 0.0)

        eng.lora("plus.gguf")                             # LORAOK 即成功；ERR 会抛
        out, ms = eng.gen([9], 4)
        self.assertEqual(out, [42])

        # close 落盘为 op=close
        eng.close()
        self.assertEqual(StubTrainingApi.calls[-1], ("/v1/training/session", {"op": "close"}))
        eng.close()                                       # 幂等：不再发请求
        n = len(StubTrainingApi.calls)
        eng.close()
        self.assertEqual(len(StubTrainingApi.calls), n)

    def test_engine_err_raises_and_session_stays(self):
        eng = self.eng()
        StubTrainingApi.fail_mode = "err"
        with self.assertRaisesRegex(RuntimeError, "ERR score"):
            eng.score([1])
        # 引擎说 ERR 不是桥死：会话保持，后续命令还能走（换回正常模式验证）
        StubTrainingApi.fail_mode = ""
        slp, _ = eng.score([1, 2])
        self.assertEqual(slp, [-0.5])
        eng.close()

    def test_bridge_death_marks_session_lost(self):
        eng = self.eng()
        StubTrainingApi.fail_mode = "died"
        with self.assertRaisesRegex(RuntimeError, "died"):
            eng.score([1])
        with self.assertRaises(SystemExit):
            eng.score([1])                                # _open 已 False：不再打服务端

    def test_open_refusal_is_loud(self):
        StubTrainingApi.fail_open = 409
        with self.assertRaisesRegex(SystemExit, "409"):
            self.eng()

    def test_session_lost_409_fails_loud(self):
        eng = self.eng()
        StubTrainingApi.fail_mode = "lost"
        with self.assertRaises(SystemExit):
            eng.score([1])


if __name__ == "__main__":
    unittest.main()
