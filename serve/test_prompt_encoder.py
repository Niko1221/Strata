"""serve/test_prompt_encoder.py - #567's incremental prompt encoding with #537's plain spans: every prompt's ids
are those of a full encode, and a turn reuses the previous one's.

    python -m unittest serve.test_prompt_encoder -v
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402
from strata_tokenizer import PromptEncoder, plain_agree_until  # noqa: E402


class ThinkTokenizer(ByteTokenizer):
    """ByteTokenizer whose <think> / </think> are specials matched even without parse_special (type 4), as in the
    Qwen vocabulary: the case #537's plain spans exist for."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]
    ALWAYS = ("<think>", "</think>")
    max_special_len = max(len(s) for s in SPECIALS)


PIECES = ["<|im_start|>user\n", "<|im_end|>\n", "<think>", "</think>", "hello ", "é你 ", "<|im_sta", "x" * 7,
          "<|im_start|>assistant\n", "</thi", "\n\n"]


def spans_of(text, rng):
    """Random plain spans: some </think> and <think> occurrences, as unmark_think_literals marks quoted ones."""
    out, i = [], 0
    while True:
        j = min((k for k in (text.find("<think>", i), text.find("</think>", i)) if k >= 0), default=-1)
        if j < 0:
            return tuple(out)
        n = len("</think>") if text.startswith("</think>", j) else len("<think>")
        if rng.random() < 0.5:
            out.append((j, j + n))
        i = j + n


class Units(unittest.TestCase):
    def test_plain_agree_until(self):
        self.assertEqual(plain_agree_until((), (), 50), 50)
        self.assertEqual(plain_agree_until(((5, 13),), ((5, 13),), 50), 50)
        self.assertEqual(plain_agree_until(((5, 13),), (), 50), 5)
        self.assertEqual(plain_agree_until(((5, 13),), ((5, 9),), 50), 9)
        self.assertEqual(plain_agree_until(((60, 70),), (), 50), 50)          # beyond the shared text: no matter
        self.assertEqual(plain_agree_until(((5, 13), (20, 28)), ((5, 13),), 50), 20)

    def test_same_ids_as_a_full_encode(self):
        tok = ThinkTokenizer()
        rng = random.Random(7)
        for conv in range(60):
            enc = PromptEncoder(tok)
            text, plain = "", ()
            for turn in range(8):
                text += "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 12)))
                if rng.random() < 0.6:
                    plain = spans_of(text, rng)                     # the spans may change in the shared part too
                with self.subTest(conv=conv, turn=turn):
                    self.assertEqual(enc.encode(text, plain), tok.encode(text, True, plain))
                if rng.random() < 0.2 and len(text) > 10:              # an edit: a client re-rendering its history
                    cut = rng.randrange(len(text))
                    text = text[:cut]
                    plain = tuple((a, b) for a, b in plain if b <= cut)

    def test_a_turn_reuses_the_previous_one(self):
        tok = ThinkTokenizer()
        enc = PromptEncoder(tok)
        # as a chat turn: the earlier prompt goes on past its last special (the answer being written), and the
        # next one shares it up to there - the resume point needs max_special_len characters after it
        base = "<|im_start|>user\n" + "quoted </think> here " * 200 + "<|im_end|>\n<|im_start|>assistant\nan answer"
        plain = tuple((i, i + 8) for i in range(len(base)) if base.startswith("</think>", i))
        enc.encode(base, plain)
        more = base + " that goes on<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n"
        self.assertEqual(enc.encode(more, plain), tok.encode(more, True, plain))
        self.assertGreater(enc.last_reused, len(base) // 2)
        # the same text with the quoted </think> no longer plain: nothing before it may be reused past its start
        self.assertEqual(enc.encode(more, ()), tok.encode(more, True, ()))
        self.assertLessEqual(enc.last_reused, plain[0][0])


class Served(unittest.TestCase):
    def test_quoted_think_across_turns(self):
        tok = ThinkTokenizer()
        template = ChatTemplate(ROOT / "serve/chat_template.jinja")
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=1 << 20), tok, template)
        full = Service(MockEngine(tok, "</think>\n\nok", max_context=1 << 20), tok, template)
        full.prompts = None                                        # the reference: a full encode every time
        self.assertIsNotNone(svc.prompts)
        msgs = [{"role": "system", "content": "Explain tags."}]
        for turn in range(5):
            msgs.append({"role": "user", "content": f"turn {turn}: what does </think> mean? <think> too " * 3})
            got, want = svc.encode_prompt(msgs, None, {}), full.encode_prompt(msgs, None, {})
            self.assertEqual(got, want)
            if turn:
                self.assertGreater(svc.prompts.last_reused, 0)
            msgs.append({"role": "assistant", "content": "It closes the reasoning: </think> is a marker."})


if __name__ == "__main__":
    unittest.main()
