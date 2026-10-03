"""The draft head's token subset (rt/draft_vocab.bin): the vocabulary rows the MTP draft layer may propose.

The engine reads rt/draft_vocab.bin (int32 token ids) when the draft layer binds and computes the draft logits over
those rows only: a smaller head is faster and takes less VRAM, but a token outside the subset can never be drafted.
The shipped subset was built from English and code and had 27 of the vocabulary's 55,328 Han tokens, so an answer in
Chinese drafted almost nothing (#137).  This adds whole scripts to a subset:

    python tools/draft_vocab.py --gguf <model>-00001-of-0000N.gguf --base data/draft_vocab.bin --add cjk \
        --out data/draft_vocab.bin

--add takes han, kana, hangul, cjk_punct, cjk (those four) or cyrillic.  The base ids keep their order; the added ones follow in
id order.  --stats prints what a subset holds.

A language that shares its script with English (French, Spanish, German...) has no range to add: its words are
made of Latin tokens the English/code subset never needed.  For those, --cover-text takes a text in the language
and adds its most frequent tokens: the ones that make --cover of the text's token occurrences (0.99 by default),
less those the subset already has:

    python tools/draft_vocab.py --gguf <model>-00001-of-0000N.gguf --base data/draft_vocab_en.bin \
        --cover-text corpus_fr.txt --out data/draft_vocab_fr.bin

The tokens are taken most frequent first (the first seen first, among equals) and added in id order, after the
ones --add brought: the same text gives the same subset.
"""
from __future__ import annotations

import argparse
import sys
from array import array
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

SCRIPTS = {
    "han": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)],
    "kana": [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)],
    "hangul": [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    "cjk_punct": [(0x3000, 0x303F), (0xFF00, 0xFF65), (0xFFA0, 0xFFEF)],
    # Ukrainian, Russian, Bulgarian, Serbian...: the shipped subset held 142 of the vocabulary's 18,580 Cyrillic tokens
    "cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
}
GROUPS = {"cjk": ["han", "kana", "hangul", "cjk_punct"]}


def scripts_of(text: str) -> set[str]:
    found = set()
    for ch in text:
        c = ord(ch)
        for name, ranges in SCRIPTS.items():
            if any(a <= c <= b for a, b in ranges):
                found.add(name)
    return found


def token_text(tok, i: int) -> str | None:
    """A token's text, or None when its bytes are not whole UTF-8 characters (a piece of one) or it is special."""
    try:
        return tok.token_bytes(i).decode("utf-8")
    except (KeyError, UnicodeDecodeError, IndexError):
        return None


def covering(freq: Counter, cover: float) -> list[int]:
    """The most frequent tokens of a text, until they make `cover` of its occurrences (the first seen first, among
    equals: Counter keeps the order of first occurrence)."""
    total, got, out = sum(freq.values()), 0, []
    for i, c in freq.most_common():
        if got >= cover * total:
            break
        out.append(i)
        got += c
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the model's (first) GGUF file: its vocabulary")
    ap.add_argument("--base", help="the subset to extend (int32 ids)")
    ap.add_argument("--add", default="", help="comma list: han, kana, hangul, cjk_punct, cjk, cyrillic")
    ap.add_argument("--cover-text", help="a UTF-8 text in the language to cover: its most frequent tokens are added")
    ap.add_argument("--cover", type=float, default=0.99,
                    help="the share of --cover-text's token occurrences those tokens make (default 0.99)")
    ap.add_argument("--out", help="where to write the new subset")
    ap.add_argument("--stats", action="store_true", help="print what the base (and the new) subset holds")
    a = ap.parse_args()
    if not 0 < a.cover <= 1:
        sys.exit(f"--cover {a.cover}: expected a share above 0, up to 1")

    tok = ST.Tokenizer.from_gguf(a.gguf)
    n = len(tok.tokens)
    kinds = [scripts_of(t) if (t := token_text(tok, i)) is not None else set() for i in range(n)]
    base = list(array("i", Path(a.base).read_bytes())) if a.base else []
    if any(i < 0 or i >= n for i in base):
        sys.exit(f"{a.base} has ids outside the vocabulary ({n})")

    want = set()
    for x in filter(None, (s.strip() for s in a.add.split(","))):
        names = GROUPS.get(x, [x])
        if any(nm not in SCRIPTS for nm in names):
            sys.exit(f"--add {x}: expected one of {', '.join([*SCRIPTS, *GROUPS])}")
        want.update(names)
    have = set(base)
    added = [i for i in range(n) if i not in have and kinds[i] & want]
    freq = Counter(tok.encode(Path(a.cover_text).read_text(encoding="utf-8"))) if a.cover_text else Counter()
    added += sorted(set(covering(freq, a.cover)) - have - set(added))
    ids = base + added

    def stats(label, sel):
        counts = {nm: sum(1 for i in sel if nm in kinds[i]) for nm in SCRIPTS}
        full = {nm: sum(1 for k in kinds if nm in k) for nm in SCRIPTS}
        print(f"{label}: {len(sel)} ids; " + ", ".join(f"{nm} {counts[nm]}/{full[nm]}" for nm in SCRIPTS))

    def outside(sel) -> str:
        inside = set(sel)
        return f"{sum(c for i, c in freq.items() if i not in inside) / sum(freq.values()):.2%}"

    if a.stats or not a.out:
        if base:
            stats("base", base)
        if added:
            stats("new", ids)
    if freq:
        print(f"{a.cover_text}: {sum(freq.values())} tokens, {len(freq)} distinct; outside the base "
              f"{outside(base)}, outside the new subset {outside(ids)}")
    if a.out:
        Path(a.out).write_bytes(array("i", ids).tobytes())
        print(f"wrote {a.out}: {len(ids)} ids ({len(added)} added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
