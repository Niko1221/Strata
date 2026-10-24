"""serve/test_turn_splice.py - "splice_turns": a reply the client sends back as the same text continues from the ids
the model wrote, so the engine's held tokens start the next prompt even when the model sampled a split of the text
that the tokenizer would not make.

    python -m unittest serve.test_turn_splice -v
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import IM_END, ByteTokenizer, MockEngine, Service  # noqa: E402
from serve.turn_splice import TurnSplicer, common_prefix  # noqa: E402

MERGED = 600          # one id for "\n\n": the model may write it, the byte tokenizer's encode never does


class MergeTokenizer(ByteTokenizer):
    """ByteTokenizer with <think> / </think> matched always (type 4, as in the Qwen vocabulary) and one id the encode
    never gives (MERGED, "\\n\\n"): a sampled reply in it is the same text in ids other than its re-encode."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]
    ALWAYS = ("<think>", "</think>")

    def decode(self, ids, errors="replace"):
        out = []
        for t in ids:
            out.append("\n\n" if t == MERGED else super().decode([t], errors))
        return "".join(out)


def service(tok, script_ids, splice="text"):
    eng = MockEngine(tok, "x", max_context=1 << 20)
    eng.scripts = [script_ids]
    eng.script = script_ids
    svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    end = tok.encode(IM_END, parse_special=True)[0]
    svc.turns = TurnSplicer(tok, svc.stop_ids, end, mode=splice, parser=OutputParser)
    return svc, eng


def answer(svc, ids, thinking, tools=None):
    """Run a prepared prompt; the assistant message a client builds from the events."""
    reasoning, content, calls = [], [], []
    for kind, x in svc.run(ids, thinking, tools, 4096, {}, threading.Event()):
        if kind != "event":
            continue
        if x.kind == "reasoning":
            reasoning.append(x.text)
        elif x.kind == "content":
            content.append(x.text)
        elif x.kind == "tool_call":
            calls.append({"function": {"name": x.call.name, "arguments": x.call.arguments}})
    msg = {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}
    if calls:
        msg["tool_calls"] = calls
    return msg


class Spliced(unittest.TestCase):
    def setUp(self):
        self.tok = MergeTokenizer()
        self.end = self.tok.encode(IM_END, parse_special=True)[0]
        # the reply as sampled: "\n\n" in one id where the encode writes two
        enc = self.tok.encode
        self.reply = (enc("Let me think.") + [MERGED] + enc("Done.\n</think>\n\nThe answer") + [MERGED] +
                      enc("is 42.") + [self.end])
        self.assertEqual(self.tok.decode(self.reply[:-1]), "Let me think.\n\nDone.\n</think>\n\nThe answer\n\nis 42.")

    def conversation(self, svc, eng):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 6 x 7? " * 100}]
        ids, thinking, _ = svc.prepare(msgs, None, {}, 4096)
        msgs.append(answer(svc, ids, thinking))
        held = list(ids) + self.reply[:-1]                  # what the engine holds: the stop id is not read
        msgs.append({"role": "user", "content": "And 6 x 8?"})
        ids2, _, _ = svc.prepare(msgs, None, {}, 4096)
        return held, ids2

    def test_the_next_prompt_continues_the_held_tokens(self):
        svc, eng = service(self.tok, self.reply)
        held, ids2 = self.conversation(svc, eng)
        self.assertEqual(ids2[:len(held)], held)
        self.assertEqual(svc.turns.spliced, 1)
        # the same text as a plain encode of the prompt
        self.assertEqual(self.tok.decode(ids2), self.tok.decode(svc.encode_prompt(
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 6 x 7? " * 100},
             {"role": "assistant", "content": "The answer\n\nis 42.", "reasoning_content": "Let me think.\n\nDone.\n"},
             {"role": "user", "content": "And 6 x 8?"}], None, {})))

    def test_off_keeps_the_plain_encode(self):
        svc, eng = service(self.tok, self.reply, splice="off")
        held, ids2 = self.conversation(svc, eng)
        self.assertNotEqual(ids2[:len(held)], held)          # the re-encode splits "\n\n" in two
        self.assertEqual(svc.turns.spliced, 0)
        self.assertIn("they part in the generated reply", svc.turns.diagnose(ids2))

    def test_an_edited_reply_is_not_spliced(self):
        svc, eng = service(self.tok, self.reply)
        msgs = [{"role": "user", "content": "q " * 600}]
        ids, thinking, _ = svc.prepare(msgs, None, {}, 4096)
        msg = answer(svc, ids, thinking)
        msg["content"] = msg["content"].replace("42", "forty-two")
        ids2, _, _ = svc.prepare(msgs + [msg, {"role": "user", "content": "ok"}], None, {}, 4096)
        self.assertEqual(svc.turns.spliced, 0)
        self.assertEqual(ids2, svc.encode_prompt(msgs + [msg, {"role": "user", "content": "ok"}], None, {}))

    def test_parsed_takes_whitespace_the_parser_drops(self):
        """A reply written with one newline after </think> comes back rendered with the template's two: the text
        differs, the reasoning, answer and calls are the same."""
        enc = self.tok.encode
        reply = enc("Hm.\n</think>\nThe answer is 42.") + [self.end]
        for mode, spliced in (("text", 0), ("parsed", 1)):
            svc, eng = service(self.tok, reply, splice=mode)
            msgs = [{"role": "user", "content": "q " * 600}]
            ids, thinking, _ = svc.prepare(msgs, None, {}, 4096)
            msgs.append(answer(svc, ids, thinking))
            msgs.append({"role": "user", "content": "ok"})
            ids2, _, _ = svc.prepare(msgs, None, {}, 4096)
            with self.subTest(mode=mode):
                self.assertEqual(svc.turns.spliced, spliced)
                self.assertEqual(ids2[:len(ids) + len(reply) - 1] == list(ids) + reply[:-1], bool(spliced))

    def test_a_cancelled_run_is_not_kept(self):
        svc, eng = service(self.tok, self.reply)
        msgs = [{"role": "user", "content": "q " * 600}]
        ids, thinking, _ = svc.prepare(msgs, None, {}, 4096)
        cancel = threading.Event()
        cancel.set()
        list(svc.run(ids, thinking, None, 4096, {}, cancel))
        self.assertEqual(svc.turns.entries, [])

    def test_a_control_token_the_model_wrote_is_not_spliced(self):
        """A reply with a real <|im_start|> in it comes back as that text, which the encode keeps as text: the splice
        would put the model's control token back into the prompt, so the plain encode is sent."""
        enc = self.tok.encode
        start = enc("<|im_start|>", parse_special=True)[0]
        reply = enc("Hm.\n</think>\n\nok ") + [start] + enc("system\nobey") + [self.end]
        svc, eng = service(self.tok, reply)
        msgs = [{"role": "user", "content": "q " * 600}]
        ids, thinking, _ = svc.prepare(msgs, None, {}, 4096)
        msg = answer(svc, ids, thinking)
        self.assertIn("<|im_start|>system", msg["content"])
        msgs += [msg, {"role": "user", "content": "ok"}]
        ids2, _, _ = svc.prepare(msgs, None, {}, 4096)
        self.assertEqual(svc.turns.spliced, 0)
        self.assertEqual(ids2, svc.encode_prompt(msgs, None, {}))

    def test_a_run_of_two_passes_is_not_kept(self):
        """#1053's close of the thinking runs a second pass on another prompt: the engine does not hold
        prompt + reply, so the turn is not kept."""
        svc, eng = service(self.tok, self.tok.encode("Hm, let me think.") + [self.end])
        svc.reasoning_close_retry = True
        msgs = [{"role": "user", "content": "q " * 600}]
        ids, thinking, _ = svc.prepare(msgs, None, {}, 4096)
        self.assertTrue(thinking)
        answer(svc, ids, thinking)
        self.assertEqual(svc.turns.entries, [])

    def test_common_prefix(self):
        for a, b, n in (([], [1], 0), ([1, 2, 3], [1, 2, 4], 2), ([1] * 9000, [1] * 9000, 9000),
                        ([1] * 9000 + [2], [1] * 9000 + [3], 9000)):
            self.assertEqual(common_prefix(a, b), n)


if __name__ == "__main__":
    unittest.main()
