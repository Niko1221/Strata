#!/usr/bin/env python3
"""Write the server's tokenizer files from an Ornith/Qwen35MoE GGUF.

`serve/server.py` loads a tokenizer directory with `vocab.json`, `merges.txt` and `token_type.json`.  Ornith
ships its tokenizer only inside the GGUF (`tokenizer.ggml.tokens` / `.merges` / `.token_type`), so this
extracts them.  The GGUF header is parsed with tools/gguf_reader.py; no tensor data is read.

    python tools/ornith_tokenizer.py --model Ornith.gguf --out /work/tokenizer/ornith
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g = GGUFFile(pathlib.Path(a.model))
    md = g.metadata
    tokens = md.get("tokenizer.ggml.tokens")
    merges = md.get("tokenizer.ggml.merges")
    types = md.get("tokenizer.ggml.token_type")
    if tokens is None or merges is None or types is None:
        sys.exit("ornith_tokenizer: the GGUF has no gpt2 tokenizer metadata "
                 "(tokenizer.ggml.tokens/.merges/.token_type)")
    if not (len(tokens) == len(types)):
        sys.exit(f"ornith_tokenizer: {len(tokens)} tokens but {len(types)} token types")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # The server indexes by id, so the map is token -> id (duplicates keep the first, as the model does).
    vocab: dict[str, int] = {}
    for i, t in enumerate(tokens):
        vocab.setdefault(t, i)
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    tpl = md.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(tpl, encoding="utf-8")
    print(f"ornith_tokenizer: wrote {out} ({len(tokens)} tokens, {len(merges)} merges, "
          f"template={'yes' if tpl else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
