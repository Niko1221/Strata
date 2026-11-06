import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from serve import format_logprobs, render_chat


class Tokenizer:
    def token_bytes(self, token):
        return {1: b" Paris", 2: b",", 3: b"."}[token]

    def decode(self, tokens):
        return b"".join(self.token_bytes(token) for token in tokens).decode()


class TextProtocolTests(unittest.TestCase):
    def setUp(self):
        self.result = {"tokens": [1, 2], "token_logprobs": [-0.1, -0.3],
                       "top5_logprobs": [[(1, -0.1), (3, -2.0)], [(2, -0.3), (3, -1.2)]]}

    def test_chat_logprobs_keep_sampled_probability_and_raw_bytes(self):
        output = format_logprobs(self.result, Tokenizer(), True, 1)
        self.assertEqual(output["content"][0], {
            "token": " Paris", "bytes": [32, 80, 97, 114, 105, 115], "logprob": -0.1,
            "top_logprobs": [{"token": " Paris", "bytes": [32, 80, 97, 114, 105, 115],
                              "logprob": -0.1}],
        })

    def test_legacy_offsets_follow_decoded_text(self):
        output = format_logprobs(self.result, Tokenizer(), False, 2)
        self.assertEqual(output["tokens"], [" Paris", ","])
        self.assertEqual(output["text_offset"], [0, 6])
        self.assertEqual(output["top_logprobs"][1], {",": -0.3, ".": -1.2})

    def test_empty_top_logprobs_still_returns_sampled_probability(self):
        output = format_logprobs(self.result, Tokenizer(), True, 0)
        self.assertEqual(output["content"][1]["logprob"], -0.3)
        self.assertEqual(output["content"][1]["top_logprobs"], [])

    def test_template_receives_thinking_switch_and_generation_marker(self):
        metadata = {"tokenizer.chat_template":
                    "{{messages[0].content}}|{{enable_thinking}}|{{add_generation_prompt}}"}
        self.assertEqual(render_chat(metadata, [{"role": "user", "content": "hello"}]),
                         "hello|False|True")

    def test_rejects_nontext_and_tool_messages(self):
        for message in ({"role": "tool", "content": "output"},
                        {"role": "user", "content": [{"type": "image"}]}):
            with self.assertRaises(ValueError):
                render_chat({}, [message])


if __name__ == "__main__":
    unittest.main()
