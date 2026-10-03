"""Context compaction: token bounds, cancellation, and preserving the input."""
import copy
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.compaction import compact_history, summary_messages, transcript_of
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


class Compaction(unittest.TestCase):
    def test_history_over_context_is_split_without_losing_text(self):
        history = [{"role": "user", "content": "予定は11月15日。" * 1200},
                   {"role": "assistant", "content": "確認しました。", "tool_calls": [{"id": "x"}]}]
        original = copy.deepcopy(history)
        fragments = []
        count = lambda messages: len(json.dumps(messages, ensure_ascii=False))

        def summarize(messages, budget):
            self.assertLessEqual(count(messages) + budget + 8, 2048)
            data = json.loads(messages[-1]["content"])
            if fragments:
                self.assertEqual(data["previous_memory"], "予定は11月15日。")
            fragments.append(data["conversation_fragment"])
            return "予定は11月15日。"

        result = compact_history(history, "", count_prompt=count, summarize=summarize, max_context=2048)
        self.assertGreater(result["chunks"], 1)
        self.assertEqual("".join(fragments), transcript_of(history))
        self.assertEqual(history, original)

    def test_previous_memory_is_incorporated(self):
        seen = []
        result = compact_history([{"role": "user", "content": "追加の決定"}], "前回の決定",
            count_prompt=lambda _: 200, summarize=lambda msgs, _: seen.append(msgs) or "統合した決定", max_context=4096)
        self.assertEqual(json.loads(seen[0][-1]["content"])["previous_memory"], "前回の決定")
        self.assertEqual(result["summary"], "統合した決定")

    def test_empty_summary_does_not_replace_history(self):
        history = [{"role": "user", "content": "重要な情報"}]
        original = copy.deepcopy(history)
        with self.assertRaisesRegex(ValueError, "empty summary"):
            compact_history(history, "前回", count_prompt=lambda _: 200,
                            summarize=lambda *_: " ", max_context=4096)
        self.assertEqual(history, original)

    def test_cancelled_after_generation_discards_result(self):
        cancelled = [False]

        def summarize(*_):
            cancelled[0] = True
            return "中断した要約"

        with self.assertRaisesRegex(ValueError, "cancelled"):
            compact_history([{"role": "user", "content": "元の情報"}], "", count_prompt=lambda _: 200,
                            summarize=summarize, max_context=4096, cancelled=lambda: cancelled[0])

    def test_images_are_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "images"):
            transcript_of([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}])

    def test_prior_memory_too_large_does_not_loop_or_generate(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            compact_history([{"role": "user", "content": "x"}], "huge", count_prompt=lambda _: 5000,
                            summarize=lambda *_: self.fail("must not generate"), max_context=4096)


class CompactionHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "予定は11月15日。次は会場を決める。", max_context=4096),
                          tok, ChatTemplate(root / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body, headers=None):
        request = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                        headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_token_count_matches_generation_including_tools_and_thinking(self):
        for effort in ("none", "high"):
            body = {"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": effort,
                    "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
                    "max_tokens": 8}
            status, count = self.post("/v1/chat/completions/count_tokens", body)
            self.assertEqual(status, 200, count)
            status, reply = self.post("/v1/chat/completions", body)
            self.assertEqual(status, 200, reply)
            self.assertEqual(count["input_tokens"], reply["usage"]["prompt_tokens"])
            self.assertEqual(count["max_context"], 4096)

    def test_full_history_compacts_without_context_error(self):
        history = [{"role": "user", "content": "予定は11月15日。" + "計画の背景です。" * 1000},
                   {"role": "assistant", "content": "次は会場を決めます。"}]
        status, result = self.post("/v1/chat/compact", {"messages": history})
        self.assertEqual(status, 200, result)
        self.assertGreater(result["chunks"], 1)
        self.assertIn("11月15日", result["summary"])

    def test_empty_input_is_rejected(self):
        status, _ = self.post("/v1/chat/compact", {"messages": []})
        self.assertEqual(status, 400)

    def test_other_websites_cannot_initiate_compaction(self):
        status, _ = self.post("/v1/chat/compact", {"messages": [{"role": "user", "content": "hi"}]},
                              {"Origin": "https://unrelated.example"})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
