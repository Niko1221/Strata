"""serve/test_nemotron_serve.py - the server's Nemotron-H pieces without the model: the stop ids keep only one-token
markers (Nemotron-H's vocabulary has no <|endoftext|>), nemotron_vision finds <img> / <image> / </img> only in a
vocabulary without <|vision_start|>, and the tekken pre-tokenizer's ignore_merges.

    python -m unittest serve.test_nemotron_serve
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from serve import server  # noqa: E402
import strata_tokenizer as ST  # noqa: E402


class Fake:
    def __init__(self, specials):
        self.special_tokens = specials


class NemotronVision(unittest.TestCase):
    def test_found(self):
        self.assertEqual(server.nemotron_vision(Fake({"<img>": 19, "<image>": 18, "</img>": 20})), (19, 18, 20))

    def test_qwen_vocab(self):
        self.assertIsNone(server.nemotron_vision(Fake({"<|vision_start|>": 1, "<img>": 2, "<image>": 3, "</img>": 4})))
        self.assertIsNone(server.nemotron_vision(Fake({"<|im_end|>": 5})))


class Tekken(unittest.TestCase):
    def tok(self, pre):
        # a toy byte-level vocabulary: single bytes, "ab", "abc" as a whole entry but no merge that builds it
        tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["ab", "abc", "<|im_end|>"]
        return ST.Tokenizer(tokens, ["a b"], [1] * 258 + [3], pre)

    def test_ignore_merges(self):
        t = self.tok("tekken")
        self.assertEqual(t.encode("abc"), [257])            # the whole piece is a vocabulary entry
        self.assertEqual(t.decode(t.encode("abc ab")), "abc ab")

    def test_qwen_merges(self):
        t = self.tok("qwen35")
        self.assertEqual(t.encode("abc"), [256, ord("c")])  # BPE: "ab" + "c"

    def test_stop_ids_one_token_only(self):
        t = self.tok("tekken")
        self.assertEqual(t.encode("<|endoftext|>", parse_special=True)[:1], [ord("<")])   # pieces, not one token
        stop = set()
        for s in ("<|im_end|>", "<|endoftext|>"):
            got = t.encode(s, parse_special=True)
            if len(got) == 1:
                stop.update(got)
        self.assertEqual(stop, {258})


if __name__ == "__main__":
    unittest.main()
