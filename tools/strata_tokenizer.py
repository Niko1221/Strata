"""tools/strata_tokenizer.py - the tokenizer, as the reference implementation of record.

WHY THIS EXISTS AND WHERE IT SITS.  The GGUF carries the whole tokenizer in metadata: `tokenizer.ggml.model`
= `gpt2`, 248,320 `tokens`, 247,587 `merges`, `token_type`, `pre` = `qwen35`, and a Jinja `chat_template`.
None of that needs the 35 GB of weights, so it is extracted into the pack's `tokenizer/` directory
(docs/pack-format.md §6) and this module is what reads it back.  A C++ port follows and is tested against
this one.

THE DECISIVE PROPERTY IS THE ROUND TRIP.  Byte-level BPE maps each BYTE to a printable unicode character so
that any UTF-8 input is representable with no UNK token.  Every failure mode of that mapping - the wrong
offset for a byte, a merge applied in the wrong order, a pre-tokenizer split that drops a character - still
produces plausible token ids.  `decode(encode(s)) == s` is the check that sees them, and it is exact because
the byte layer is lossless.  It is asserted over a corpus chosen to hit the boundaries: multi-byte UTF-8,
emoji (4-byte), combining marks, whitespace runs, and C0 control bytes.
"""
from __future__ import annotations

import json
import pathlib
import sys

import regex

# ------------------------------------------------------------------ byte <-> unicode (GPT-2 byte-level BPE)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte encoder: 256 bytes -> 256 printable characters, reversibly.

    Bytes 33..126, 161..172 and 174..255 map to themselves; the remaining 68 (space, newline, the C0
    controls and the high range that would be invisible) are shifted into 256+n so that no byte is
    unprintable.  The shift is the whole trick and getting its ORDER wrong is invisible in the output ids.
    """
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# The `qwen35` pre-tokenizer, transcribed from the ORACLE rather than from the family resemblance:
# `.ref/llama.cpp/src/llama-vocab.cpp` L396, `case LLAMA_VOCAB_PRE_TYPE_QWEN35`.  The commented-out line
# above it is the `tokenizer.json` original, and llama.cpp's active version differs from it - it spells the
# contraction classes out instead of using `(?i:...)`, which is behaviourally the same.
#
# THE `\p{M}` IS THE WHOLE POINT OF THIS BEING TRANSCRIBED.  The QWEN3 pattern two cases earlier (L389) is
# the same shape with `\p{L}` where this has `[\p{L}\p{M}]` and without `\p{M}` in the punctuation negation.
# Writing the QWEN3 pattern for `qwen35` looks right, round-trips perfectly, and is wrong: a combining mark
# is `\p{M}`, so without it the mark gets swallowed into the following punctuation run and a token like `_j`
# never forms.  It cost 3 strings out of 1875 and only `tokenize_oracle_check` could see it.
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class Tokenizer:
    def __init__(self, tokens: list[str], merges: list[str], token_types: list[int] | None = None,
                 pre: str = "qwen35", special_ids: dict[str, int] | None = None):
        self.tokens = tokens
        self.pre = pre
        self.token_types = token_types
        self.special_ids = special_ids or {}
        self.ids = {t: i for i, t in enumerate(tokens)}
        if len(self.ids) != len(tokens):
            raise ValueError("vocabulary has duplicate tokens: %d entries, %d unique"
                             % (len(tokens), len(self.ids)))
        # Merge rules as (left, right) -> rank.  A merge list that is not a total order over its pairs, or one
        # naming a token that is not in the vocabulary, would make BPE silently produce different ids than the
        # model was trained with, so both are checked here rather than discovered as a bad answer later.
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError("merge %d is not a pair: %r" % (i, m))
            if parts[0] not in self.ids or parts[1] not in self.ids:
                raise ValueError("merge %d names a token outside the vocabulary: %r" % (i, m))
            self.ranks[(parts[0], parts[1])] = i
        self._re = regex.compile(QWEN35_PATTERN)

        # The literals matched directly instead of being run through BPE.  GGUF token types: 3 = CONTROL,
        # 4 = USER_DEFINED.  The two classes behave DIFFERENTLY and llama.cpp's own tokenizer settled which:
        #
        #   * type 4 (USER_DEFINED: `<think>`, `<tool_call>`, `<tool_response>`, ...) is matched ALWAYS,
        #     with or without parse_special.
        #   * type 3 (CONTROL: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, ...) is matched ONLY when
        #     parse_special is set.
        #
        # Measured, not assumed: with parse_special=False the oracle still emitted `<tool_response>` as one
        # token and mine emitted four, and the three disagreements were exactly the type-4 cases while every
        # type-3 case agreed.  Treating both classes alike costs 3 strings in 1229 and silently changes a chat
        # prompt, because `<|im_end|>` decomposed into ordinary pieces is not the token the model expects.
        self.special_tokens: dict[str, int] = {}
        if token_types:
            for i, ty in enumerate(token_types):
                if ty in (3, 4):
                    self.special_tokens[tokens[i]] = i
        always = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 4]
        # Longest literal first, or `<|im_end|>` could match a shorter prefix of itself.  `regex.escape` so a
        # token containing regex metacharacters (several do: `<|`, `[`, `(`) is matched literally.
        self._always_re = self._alt(always)
        self._special_re = self._alt(list(self.special_tokens))

    @staticmethod
    def _alt(literals: list[str]):
        if not literals:
            return None
        return regex.compile("|".join(regex.escape(s) for s in sorted(literals, key=len, reverse=True)))

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_gguf(cls, path) -> "Tokenizer":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(path)).metadata
        need = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
        missing = [k for k in need if k not in md]
        if missing:
            raise ValueError("GGUF is missing %s; it does not carry a tokenizer" % ", ".join(missing))
        model = md.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError("expected a byte-level BPE tokenizer (model 'gpt2'), got %r" % model)
        special = {k: int(md[k]) for k in md
                   if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        return cls(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                   list(md.get("tokenizer.ggml.token_type") or []) or None,
                   md.get("tokenizer.ggml.pre", "qwen35"), special)

    # -------------------------------------------------------------- the algorithm
    def _bpe(self, word: str) -> list[str]:
        """Merge `word` (already byte-mapped) by LOWEST RANK first, repeatedly - not left to right.

        Applying merges in list order rather than rank order is the classic BPE bug: it produces a different
        segmentation and a plausible token count.
        """
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in self._re.findall(text):
            mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                i = self.ids.get(tok)
                if i is None:
                    raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                out.append(i)
        return out

    def _encode_matching(self, text: str, pat) -> list[int]:
        """Encode `text`, emitting any literal `pat` matches as single tokens and BPE-ing the rest.

        The split happens on the RAW text, before the byte mapping, because a special token's string is a
        literal to match rather than bytes to decompose.  Everything between the matches is tokenized
        normally - which is why a near-miss like `<|im_star` still costs ordinary tokens.
        """
        if pat is None:
            return self._encode_plain(text)
        out: list[int] = []
        pos = 0
        for m in pat.finditer(text):
            if m.start() > pos:
                out.extend(self._encode_plain(text[pos:m.start()]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._encode_plain(text[pos:]))
        return out

    def encode(self, text: str, parse_special: bool = False) -> list[int]:
        """Tokenize `text`.

        `parse_special` controls only the type-3 CONTROL literals such as `<|im_end|>`; the type-4
        USER_DEFINED ones such as `<think>` are matched either way.  See the note in `__init__`.
        """
        return self._encode_matching(text, self._special_re if parse_special else self._always_re)

    def token_bytes(self, i: int) -> bytes:
        """The raw bytes of one token (a multi-byte character can be split across tokens)."""
        cache = self.__dict__.setdefault("_bytes_cache", {})
        b = cache.get(i)
        if b is None:
            if i < 0 or i >= len(self.tokens):
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, len(self.tokens)))
            raw = bytearray()
            for ch in self.tokens[i]:
                v = UNICODE_TO_BYTE.get(ch)
                if v is None:
                    raise KeyError("token %d contains a character outside the byte alphabet: %r" % (i, ch))
                raw.append(v)
            b = cache[i] = bytes(raw)
        return b

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)


def dry_sequence_breaker_tokens(tokenizer, breakers: list[str], max_char_len: int = 40,
                               max_tail_len: int = 20) -> list[tuple[int, tuple[int, ...]]]:
    """Map DRY breaker strings to the overlapping token sequences used by llama.cpp.

    A breaker can start or end inside a vocabulary token, so encoding the whole string alone is not enough.
    Match llama.cpp's `get_overlapping_token_sequences`: walk every token's decoded piece, record a direct
    match as an empty tail, and tokenize the part of a breaker that continues after a matching token prefix.
    The result is `(head_token, tail_tokens)`; tails are stored in normal text order.
    """
    vocab = getattr(tokenizer, "tokens", None)
    if vocab is None:
        vocab_size = 256 + len(getattr(tokenizer, "SPECIALS", ()))
        special_ids = set()
    else:
        vocab_size = len(vocab)
        special_ids = set(getattr(tokenizer, "special_tokens", {}).values())

    processed: dict[int, set[tuple[int, ...]]] = {}
    for raw_breaker in breakers:
        if not isinstance(raw_breaker, str) or not raw_breaker:
            continue
        # llama.cpp bounds the UTF-8 string to 40 bytes before tokenizing it.
        breaker = raw_breaker.encode("utf-8")[:max_char_len].decode("utf-8", "ignore")
        if not breaker:
            continue
        for token_id in range(vocab_size):
            # A GGUF special token is stored as its printable spelling, not a byte-alphabet BPE piece. llama.cpp's
            # vocab.detokenize(..., true) renders that spelling directly, so mirror it instead of passing it through
            # Tokenizer.decode(), which correctly rejects characters outside the byte alphabet. Fall back for other
            # non-byte vocabulary entries too (for example an UNK or UNUSED token).
            if token_id in special_ids:
                word = vocab[token_id]
            else:
                try:
                    word = tokenizer.decode([token_id])
                except KeyError:
                    word = vocab[token_id]
            if breaker in word:
                processed.setdefault(token_id, set()).add(())
                continue
            pos = -1
            while True:
                pos = word.find(breaker[0], pos + 1)
                if pos < 0:
                    break
                match = True
                i = 1
                while i < len(breaker) and i + pos < len(word):
                    if word[pos + i] != breaker[i]:
                        match = False
                        break
                    i += 1
                if match:
                    tail = tuple(tokenizer.encode(breaker[i:], parse_special=False)[:max_tail_len])
                    processed.setdefault(token_id, set()).add(tail)

    return [(head, tail) for head in sorted(processed) for tail in sorted(processed[head])]


# ------------------------------------------------------------------ the pack's tokenizer/ directory
def extract(gguf_path, out_dir) -> dict:
    """Write the tokenizer into `<out_dir>/tokenizer/` so the engine never opens the weight shards for it."""
    out = pathlib.Path(out_dir) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer.from_gguf(gguf_path)
    cfg = {
        "model": "gpt2",
        "pre": tk.pre,
        "vocab_size": len(tk.tokens),
        "n_merges": len(tk.ranks),
        "special_ids": tk.special_ids,
        "add_bos_token": False,
        # The pattern is SHIPPED, not recomputed by the reader: it is transcribed from llama.cpp for the
        # declared `pre` type, and a C++ port that re-derived it would be free to get `\p{M}` wrong again.
        "pre_pattern": QWEN35_PATTERN,
        "pre_pattern_source": ".ref/llama.cpp src/llama-vocab.cpp L396 (LLAMA_VOCAB_PRE_TYPE_QWEN35)",
    }
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join("%s %s" % k for k, _ in
                                              sorted(tk.ranks.items(), key=lambda kv: kv[1])), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(tk.token_types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    # the model's own chat template: fine-tunes change it (Swift 1.5 differs from Qwen3.8-Flash-Next's)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    tpl = GGUFFile(pathlib.Path(gguf_path)).metadata.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(tpl, encoding="utf-8", newline="\n")
    return cfg


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="round-trip a corpus and report")
    args = ap.parse_args()
    cfg = extract(args.gguf, args.out)
    print("tokenizer/: vocab %d, merges %d, pre %s, specials %s"
          % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"], cfg["special_ids"]))
    if args.check:
        tk = Tokenizer.from_gguf(args.gguf)
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
                  "def f(x):\n\treturn x  # comment\n", "\u4f60\u597d\uff0c\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "e\u0301\u0301 combining", "\x00\x01\x7f control",
                  "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
                  "x" * 5000]
        bad = 0
        for s in corpus:
            ids = tk.encode(s)
            back = tk.decode(ids)
            if back != s:
                print("  *** ROUND TRIP FAILED *** %r -> %d ids -> %r" % (s[:40], len(ids), back[:40]))
                bad += 1
        print("round trip: %d strings, %d failed" % (len(corpus), bad))
        return 0 if bad == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
