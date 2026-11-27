#!/usr/bin/env python3
"""Compare token IDs with llama.cpp using its llama-tokenize executable."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "terminal" / "tools"))

from strata_tokenizer import Tokenizer            # noqa: E402

AWKWARD = [
    "你好",
    "The capital of France is Paris.",
    "a\\nb",                      # a LITERAL backslash-n: the case that caught llama-tokenize
    "um\\x64;C:\\Program Files",  # a Windows path, which is where the corpus diverged
    "20 21 2021 1.5 -3",          # digit runs: the hy_v4 / qwen35 difference
    "  spaces   and\ttabs\n\nand newlines",
    "emoji 😀 and a combining é",
    "混元 is 腾讯's model; 中文 and ASCII mixed",
    "<think>not a marker</think>",
    "".join(chr(c) for c in range(1, 32)),
]


def reference_ids(executable: str, model: str, text: str) -> list[int]:
    result = subprocess.run(
        [executable, "-m", model, "--stdin", "--no-escape", "--no-bos", "--ids"],
        input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise SystemExit(result.stderr.decode("utf-8", errors="replace"))
    return json.loads(result.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Target GGUF file or first shard")
    ap.add_argument("--llama-tokenize", default="llama-tokenize", help="Path to the llama.cpp reference tokenizer")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()

    tok = Tokenizer.from_gguf(a.model)
    print(f"strata_tokenizer: pre {tok.pre!r}, {len(tok.tokens)} tokens")

    cases: list[tuple[str, str]] = [(f"awkward[{i}]", s) for i, s in enumerate(AWKWARD)]
    for f in a.files:
        cases.append((f, Path(f).read_text(encoding="utf-8")))

    bad = 0
    for name, text in cases:
        want = reference_ids(a.llama_tokenize, a.model, text)
        got = tok.encode(text, parse_special=True)
        if got == want:
            print(f"  ok   {name}  ({len(want)} ids)")
            continue
        bad += 1
        print(f"  FAIL {name}  llama {len(want)} ids, strata {len(got)}")
        for i, (x, y) in enumerate(zip(want, got)):
            if x != y:
                print(f"       first difference at {i}: llama {x} {tok.tokens[x]!r}, "
                      f"strata {y} {tok.tokens[y]!r}")
                break
        # the round trip is the other half: a tokenizer can agree and still not be lossless
        if tok.decode(got) != text:
            print("       and strata's own round trip does not hold here")

    print(f"\n{len(cases) - bad} agree, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
