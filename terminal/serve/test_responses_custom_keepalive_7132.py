"""#7132: `/v1/responses` was silent for the whole time a custom tool call (Codex's `apply_patch`) was written.

The Qwen-form parser streams a custom tool's string parameter as `tool_args` events, so the token loop counted them as output and never sent its keep-alive
ping; the Responses assembler then dropped them (a custom call's input goes out once, at its end).  Zero bytes between `response.output_item.added` and the end
of the call, so a proxy or client that cuts an idle stream (125 s on the reporter's Cloudflare path) ended a healthy request.  `events()` of `_responses` now
sends a keep-alive when nothing has gone out for KEEPALIVE_S.

Scripted engine (a token every 10 ms, no model), keep-alive period 0.5 s, a call that takes several seconds: the longest gap between two lines on the wire must
stay under 2 s.  The same call on the unpatched route leaves a gap as long as the call.

    python -m pytest serve/test_responses_custom_keepalive_7132.py
"""
from __future__ import annotations

import http.client
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import serve.server as S  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402

PERIOD_S = 0.5
DELAY_S = 0.01
BODY = "*** Begin Patch\n*** Add File: x.py\n" + "".join(f"+line {i} of the file\n" for i in range(15)) + "*** End Patch"
CUSTOM = "</think>\n\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n" + BODY + "\n</parameter>\n</function>\n</tool_call>"
FUNCTION = "</think>\n\n<tool_call>\n<function=write_file>\n<parameter=content>\n" + BODY + "\n</parameter>\n</function>\n</tool_call>"


class SilentCustomCall(unittest.TestCase):
    def setUp(self):
        old = S.KEEPALIVE_S
        S.KEEPALIVE_S = PERIOD_S
        self.addCleanup(setattr, S, "KEEPALIVE_S", old)

    def stream(self, req, script):
        tok = S.ByteTokenizer()
        eng = S.MockEngine(tok, script, max_context=65536, delay_s=DELAY_S)
        svc = S.Service(eng, tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"), model_name="m")
        httpd = S.serve(svc, port=0)
        try:                                                  # serve() already runs the loop in its own thread
            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=60)
            conn.request("POST", "/v1/responses", json.dumps(req), {"Content-Type": "application/json"})
            r = conn.getresponse()
            self.assertEqual(r.status, 200)
            stamps, events, keepalives, t0 = [], [], 0, time.perf_counter()
            while True:
                line = r.fp.readline()
                if not line:
                    break
                stamps.append(time.perf_counter() - t0)
                s = line.decode("utf-8", "replace").strip()
                if s.startswith("event:"):
                    events.append(s[6:].strip())
                elif s.startswith(": keep-alive"):
                    keepalives += 1
        finally:
            httpd.shutdown()
            httpd.server_close()
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        return {"total": stamps[-1], "max_gap": max(gaps), "events": events, "keepalives": keepalives}

    def test_custom_tool_call_keeps_the_stream_alive(self):
        res = self.stream({"model": "m", "input": "go", "stream": True,
                           "tools": [{"type": "custom", "name": "apply_patch", "description": "Edit files."}]}, CUSTOM)
        # the call has to take much longer than the limit, else the test proves nothing
        self.assertGreater(res["total"], 3.0, res)
        self.assertLess(res["max_gap"], 2.0, res)
        self.assertGreaterEqual(res["keepalives"], 4, res)
        # the call itself is whole and unchanged: its input once, at its end
        self.assertIn("response.output_item.added", res["events"])
        self.assertEqual(res["events"].count("response.custom_tool_call_input.delta"), 1)
        self.assertIn("response.completed", res["events"])

    def test_function_tool_is_untouched(self):
        res = self.stream({"model": "m", "input": "go", "stream": True,
                           "tools": [{"type": "function", "name": "write_file", "description": "w",
                                      "parameters": {"type": "object", "properties": {"content": {"type": "string"}}}}]},
                          FUNCTION)
        self.assertLess(res["max_gap"], 2.0, res)
        self.assertIn("response.completed", res["events"])

    def test_a_short_answer_sends_no_keepalive(self):
        res = self.stream({"model": "m", "input": "go", "stream": True}, "ok")
        self.assertEqual(res["keepalives"], 0, res)
        self.assertIn("response.completed", res["events"])


if __name__ == "__main__":
    unittest.main()
