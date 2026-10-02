#!/usr/bin/env python3
"""Tokenize a fixed prompt with the pack's own tokenizer, for the PR #500 A/B benchmark.

    tools/tokens_for_bench.py --tokenizer DIR --text "..."     -> comma-separated token ids
    tools/tokens_for_bench.py --tokenizer DIR --file PATH
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST


def load(tdir: Path) -> ST.Tokenizer:
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens: list = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tdir / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text")
    ap.add_argument("--file")
    a = ap.parse_args()
    text = Path(a.file).read_text(encoding="utf-8") if a.file else a.text
    if not text:
        ap.error("--text or --file is required")
    tok = load(Path(a.tokenizer))
    ids = tok.encode(text, parse_special=True)
    print(",".join(str(int(i)) for i in ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
