"""tool_choice naming a tool on Inkling-2-Mini: the call opens the reply and the next turn's history chains onto this prompt.

Before: the opening was written by the thinking loop after the model's first token as Qwen text (`\\n\\n<|message_model|>read_file<|content_invoke_tool_json|>...`),
so the client got that text as content, not a tool call, and the prompt did not carry the forced call.  (That the next turn chains onto it is checked with
the real template in test_inkling_prefix_chain.py.)  Now the opening ends the prompt (after the empty thinking turn every tool request ends with) and the parser reads it as the model's own.

    python -m pytest serve/test_inkling_forced_call.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from serve.test_inkling_serve_e2e import _MockServerHarness, _post_json  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "run", "description": "Run.", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}]
# what the model writes after the forced opening `<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":`
REMAINDER = '{"path":"a.py"}}<|end_message|><|content_model_end_sampling|>'
FORCE = '<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":'
EMPTY_THINK = "<|message_model|><|content_thinking|><|end_message|>"


class ForcedCall(unittest.TestCase):
    def check(self, level, before):
        with tempfile.TemporaryDirectory() as td:
            with _MockServerHarness(Path(td), [REMAINDER], parallel=1) as h:
                base = f"http://127.0.0.1:{h.port}"
                r = _post_json(f"{base}/v1/chat/completions", {
                    "model": "inkling-2-mini-sq-20", "messages": [{"role": "user", "content": "open a.py"}], "tools": TOOLS,
                    "tool_choice": {"type": "function", "function": {"name": "read_file"}}, "max_tokens": 64, "temperature": 0,
                    "reasoning_effort": level})
                msg = r["choices"][0]["message"]
                self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
                self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["read_file"])
                self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]), {"path": "a.py"})
                self.assertNotIn("<|message_model|>", msg.get("content") or "")
                sent = Path(h.record_path).read_text(encoding="utf-8").splitlines()[-1].split()[-1].split(",")
                ids = [int(x) for x in sent]
                tail = h.tok.encode(before + FORCE, parse_special=True)
                self.assertEqual(ids[-len(tail):], tail, f"level {level}: the prompt ends with {before!r} and the forced opening")

    def test_level_zero_ends_the_prompt_with_the_empty_thinking_turn_and_the_opening(self):
        self.check("none", EMPTY_THINK)

    def test_a_thinking_level_puts_the_opening_right_after_the_user_turn(self):
        self.check("low", "")


if __name__ == "__main__":
    unittest.main()
