#!/usr/bin/env python3
"""Rebuild reproduce/dlss.ids.json (or verify it): tokenize the ten corpus files with the
pack tokenizer and sample 3 x 1536 evenly spaced slices per file (the corpus of this report).

    python reproduce/make_ids.py --docs <docs-dir> --tokenizer <pack>/tokenizer --out reproduce/dlss.ids.json

Nine of the ten are public NVIDIA documents converted to text - the Ada / Blackwell architecture
whitepapers (consumer and PRO editions), the DLSS 4 announcement article, the DLSS 5 report and the
DLSS SR / FG / RR guides; the tenth, dlss-prompt3.txt, is a page-marked concatenation of six of them
(the DLSS guides and Blackwell whitepapers therefore carry double weight). reproduce/corpus.json
carries each file's name, byte size and sha256 prefix so the same texts can be fetched and checked;
dlss-prompt3.txt is locally derived, so a full rebuild is best-effort (PDF-to-text conversion varies)
and the committed ids file stays the authority. The ids file is
committed, so reproducing the numbers needs no tokenizer and no fetch: the corpus hash (f553e2c3, the
sha256 prefix of the ids bytes) is stamped into every arm's identity by ab_kld.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402  (the pack tokenizer, as quant_quality.py cmd_ids drives it)

SLICES, SLICE_TOKENS = 3, 1536


def load_tokenizer(tp: pathlib.Path) -> ST.Tokenizer:
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tp / "token_type.json").read_text()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True, help="directory holding the ten .txt documents")
    ap.add_argument("--tokenizer", required=True, help="the pack's tokenizer directory")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "dlss.ids.json"))
    a = ap.parse_args()
    tok = load_tokenizer(pathlib.Path(a.tokenizer))
    src = pathlib.Path(a.docs)
    segs, manifest = [], []
    for f in sorted(src.glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="replace")
        ids = tok.encode(text, parse_special=False)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", f.stem)[:48]
        n = min(SLICES, max(1, len(ids) // SLICE_TOKENS))
        step = max(SLICE_TOKENS, len(ids) // n)
        for i in range(n):
            start = min(i * step, max(0, len(ids) - SLICE_TOKENS))
            chunk = ids[start:start + SLICE_TOKENS]
            if len(chunk) < 256:
                continue
            segs.append({"name": f"{stem}#{i}@{start}", "ids": chunk})
        manifest.append({"file": f.name, "bytes": f.stat().st_size,
                         "sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16], "tokens": len(ids),
                         "slices": n, "scored": n * SLICE_TOKENS})
        print(f"{f.name}: {len(ids)} tokens, sampled {n}x{SLICE_TOKENS}", flush=True)
    out = json.dumps({"version": "dlss-v2-sampled", "seg_tokens": SLICE_TOKENS, "segments": segs})
    pathlib.Path(a.out).write_text(out)
    print(f"wrote {a.out}: {len(segs)} segments; corpus hash "
          f"{hashlib.sha256(out.encode()).hexdigest()[:8]} (expected f553e2c3 for the same texts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
