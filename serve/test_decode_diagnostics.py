"""Engine-reported loop diagnostics, without a model or GPU.

python -m unittest serve.test_decode_diagnostics -v
"""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import urllib.request

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, request_stats, request_timings, serve
from serve.test_request_accounting import ClockedThinkingEngine


DONE = "DONE 5 12000 900.0 80.0 stop 3 4 0 7 9 0 0 0.0 12000 2 chunk=1536 experts_streamed=100"


def suffix(mode="serial", windows=2, reason="repetition_penalties"):
    return f" decode_mode={mode} pipeline_windows={windows} decode_reason={reason}"


def parsed(extra):
    engine = SimpleNamespace()
    StrataEngine._parse_done(engine, DONE + extra)
    return engine.last


class DecodeDiagnostics(unittest.TestCase):
    def test_modes_come_from_the_engine_not_the_number_of_generated_tokens(self):
        for mode, windows, reason in [("pipeline", 2, "none"), ("none", 2, "no_decode_loop"),
                                      ("serial", 0, "not_requested"), ("serial", 1, "prefill_only"),
                                      ("serial", 2, "repetition_penalties"),
                                      ("serial", 2, "coupled_draft_sampling"),
                                      ("serial", 2, "configuration"), ("serial", 2, "snapshot_unavailable")]:
            with self.subTest(mode=mode, reason=reason):
                last = parsed(suffix(mode, windows, reason))
                self.assertEqual(last["decode_segments"],
                                 [{"mode": mode, "pipeline_windows": windows, "reason": reason}])
                # Existing clocks, cache accounting and positional lookup-chain stats remain untouched.
                self.assertEqual({k: v for k, v in last.items() if k != "decode_segments"}, parsed(""))

    def test_legacy_done_clears_a_previous_requests_diagnostics(self):
        engine = SimpleNamespace()
        for extra in (suffix(), suffix("pipeline", 2, "none"), "", suffix()):
            StrataEngine._parse_done(engine, DONE + extra)
            self.assertEqual("decode_segments" in engine.last, bool(extra))
        self.assertNotIn("decode_segments", parsed(""))
        self.assertNotIn("decode_segments", request_timings(12000, 5, parsed("")))

    def test_malformed_or_incomplete_optional_metadata_is_ignored(self):
        for extra in (" decode_mode=serial", suffix(windows="x"), suffix(windows=-1),
                      suffix(mode="future"), suffix(reason="future"), suffix("pipeline", 0, "none"),
                      suffix("serial", 2, "none"), suffix("none", 2, "repetition_penalties")):
            with self.subTest(extra=extra):
                self.assertEqual(parsed(extra), parsed(""))

    def test_appended_lookup_chain_fields_do_not_move_metadata(self):
        last = parsed(" 1 2 3 4 5" + suffix())
        self.assertEqual(last["decode_segments"], parsed(suffix())["decode_segments"])

    def test_recovery_preserves_both_modes_and_the_existing_totals(self):
        first, second = parsed(suffix("pipeline", 2, "none")), parsed(suffix())
        total = request_stats([first, second])
        self.assertEqual(total["decode_segments"], first["decode_segments"] + second["decode_segments"])
        self.assertEqual(total["generated"], 10)
        self.assertEqual(total["decode_ms"], 160.0)
        self.assertEqual(request_timings(12000, 10, total)["decode_segments"], total["decode_segments"])
        self.assertEqual(len(first["decode_segments"]), 1)

    def test_unreported_continuations_are_unknown_in_either_position(self):
        known = parsed(suffix())
        for segments in ([parsed(""), known], [known, parsed("")]):
            reports = request_stats(segments)["decode_segments"]
            self.assertEqual(reports, [s.get("decode_segments", [{"mode": "unknown"}])[0] for s in segments])
        self.assertNotIn("decode_segments", request_stats([parsed(""), parsed("")]))
        self.assertEqual(request_stats([]), {})


class ReportedEngine(MockEngine):
    extra = ""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            StrataEngine._parse_done(self, DONE + self.extra)


class DecodeDiagnosticsHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = ReportedEngine(tok, "ok")
        cls.svc = Service(cls.engine, tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        cls.httpd = serve(cls.svc, port=0)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}/v1/chat/completions"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_complete_and_streamed_responses_follow_each_requests_report(self):
        for stream in (False, True):
            for extra in (suffix("pipeline", 2, "none"), suffix(), ""):
                with self.subTest(stream=stream, extra=extra):
                    self.engine.extra = extra
                    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 30, "reasoning_effort": "none", "stream": stream}
                    request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                                     headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=10) as response:
                        raw = response.read().decode()
                    if stream:
                        messages = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                        result = messages[-1]
                    else:
                        result = json.loads(raw)
                    expected = parsed(extra).get("decode_segments")
                    self.assertEqual(result["timings"].get("decode_segments"), expected)


class DiagnosticThinkingEngine(ClockedThinkingEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            extra = suffix("pipeline", 2, "none") if self.attempts == 1 else suffix()
            self.last["decode_segments"] = parsed(extra)["decode_segments"]


class DecodeRecoveryHTTP(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = DiagnosticThinkingEngine(tok)
        svc = Service(self.engine, tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        self.httpd = serve(svc, port=0)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def check_recovery(self, stream=False, cancel=False):
        self.engine.cancel_continuation = cancel
        body = {"model": "m", "messages": [{"role": "user", "content": "2+2?"}],
                "max_tokens": 400, "reasoning_budget_tokens": 20, "stream": stream}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = response.read().decode()
        result = ([json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")][-1]
                  if stream else json.loads(raw))
        self.assertEqual(self.engine.attempts, 2)
        self.assertEqual(result["timings"]["decode_segments"],
                         parsed(suffix("pipeline", 2, "none"))["decode_segments"] + parsed(suffix())["decode_segments"])
        self.assertEqual(result["timings"]["predicted_ms"], 280.0)

    def test_complete_thinking_budget_continuation_preserves_both_modes(self):
        self.check_recovery()

    def test_streamed_thinking_budget_continuation_preserves_both_modes(self):
        self.check_recovery(stream=True)

    def test_cancelled_continuation_preserves_completed_segment_reports(self):
        self.check_recovery(cancel=True)


if __name__ == "__main__":
    unittest.main()
