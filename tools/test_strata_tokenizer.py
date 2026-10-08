"""The tokenizer's piece cache (0.1.41): the ids are the ones the merge loop gives, cache cold, warm or switched off.

    python -m unittest tools.test_strata_tokenizer
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402


def toy() -> ST.Tokenizer:
    base = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    merges = ["t h", "th e", "i n", "a n", "an d", "Ġ t", "Ġt h", "Ġth e", "e r", "o n"]
    merges = [m.replace("Ġ", ST.BYTE_TO_UNICODE[32]) for m in merges]
    tokens = list(base)
    for m in merges:
        a, b = m.split(" ")
        tokens.append(a + b)
    return ST.Tokenizer(tokens, merges)


class PieceCache(unittest.TestCase):
    def test_same_ids_cold_warm_and_off(self):
        rng = random.Random(7)
        words = ["the", "and", "in", "on", "there", "other", "x" * 70, "你好", "def", "return", "\n\n", "  "]
        text = "".join(rng.choice(words) + rng.choice([" ", "", "\n", "\t"]) for _ in range(3000))
        a = toy()
        a.PIECE_CACHE_MAX = 0
        want = a.encode(text)
        b = toy()
        cold = b.encode(text)
        warm = b.encode(text)
        self.assertEqual(cold, want)
        self.assertEqual(warm, want)
        self.assertGreater(len(b._piece_ids), 0)
        self.assertEqual(b.decode(warm), text)

    def test_the_cache_is_bounded(self):
        t = toy()
        t.PIECE_CACHE_MAX = 5
        t.encode(" ".join("w%d" % i for i in range(100)))
        self.assertLessEqual(len(t._piece_ids), 5)


class ByteCache(unittest.TestCase):
    def test_decode_before_encode_cold_and_warm(self):
        t = toy()
        text = "café 日本語 🙂\r\n\x00"
        ids = list(text.encode("utf-8"))  # toy's first 256 token IDs are the raw bytes
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode([]), "")
        for i in range(256):
            self.assertEqual(t.token_bytes(i), bytes([i]))

    def test_invalid_ids_before_and_after_cache_use(self):
        t = toy()
        for _ in range(2):
            for i in (-1, len(t.tokens), len(t.tokens) + 100):
                with self.assertRaises(IndexError):
                    t.token_bytes(i)
            self.assertEqual(t.token_bytes(65), b"A")

    def test_instances_with_different_vocabularies_stay_independent(self):
        tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
        a, b = ST.Tokenizer(tokens, []), ST.Tokenizer(tokens[::-1], [])
        text = "the café 日本語 🙂\n"
        for t in (a, b, a, b):
            expected = [t.ids[ST.BYTE_TO_UNICODE[byte]] for byte in text.encode("utf-8")]
            self.assertEqual(t.encode(text), expected)
            self.assertEqual(t.decode(expected, errors="strict"), text)
        self.assertEqual(a.token_bytes(0), b"\x00")
        self.assertEqual(b.token_bytes(0), b"\xff")

    def test_shared_tokenizer_concurrent_first_decode(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        t = toy()
        texts = ["café", "日本語", "🙂\r\n", "\x00plain"]
        barrier = threading.Barrier(len(texts))
        def decode(text):
            ids = list(text.encode("utf-8"))
            barrier.wait(timeout=5)
            for _ in range(100):
                if t.decode(ids, errors="strict") != text:
                    raise AssertionError("concurrent decode changed the text")
            return text
        with ThreadPoolExecutor(len(texts)) as pool:
            self.assertEqual(list(pool.map(decode, texts)), texts)


if __name__ == "__main__":
    unittest.main()


class GptOssPreTokenizer(unittest.TestCase):
    """`gpt-4o` (GPT-OSS's o200k): the split pattern and ignore_merges, against openai/gpt-oss-120b's tokenizer.json.
    The expected pieces are that tokenizer's own pre_tokenize_str output (combining marks, contractions, digits)."""

    REFERENCE = {
        "Hello world's 12345 HELLO'S": ["Hello", " world's", " ", "123", "45", " HELLO'S"],
        "été café!!\n\n": ["été", " café", "!!\n\n"],
        "a/b/c\n/ x": ["a", "/b", "/c", "\n", "/", " x"],
        "  \t\tdef f():\n    return 1": ["  \t", "\tdef", " f", "():\n", "   ", " return", " ", "1"],
    }

    def tok(self, pre):
        base = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
        return ST.Tokenizer(base + ["th", "thx"], ["t h"], pre=pre)     # "thx": a token no merge reaches

    def test_the_split_matches_the_reference(self):
        tk = self.tok("gpt-4o")
        for text, pieces in self.REFERENCE.items():
            self.assertEqual(tk._re.findall(text), pieces, text)

    def test_a_piece_that_is_a_token_is_that_token(self):
        whole = len(self.tok("gpt-4o").tokens) - 1                      # ignore_merges, as tiktoken
        self.assertEqual(self.tok("gpt-4o").encode("thx"), [whole])
        self.assertNotEqual(self.tok("qwen35").encode("thx"), [whole])  # qwen35 merges: "th" + "x"

    def test_an_unknown_pre_tokenizer_is_refused(self):
        with self.assertRaises(ValueError):
            self.tok("llama3")
