"""把本地仓库文本打包成等长 token 块（预训练形状），供 sft.py --pretrain-data 使用。

动机：身份 SFT 每条只有 21~33 token，训练每步的固定 I/O（每步至少触碰全部
48×512 专家一遍）摊不出去，测不出真实吞吐。预训练形状 = seq 2048 全 token
loss，专家去重后每步流量饱和（64GB），才是"训练版 Strata"的真实形状。

只读本地文件，零下载。文档间用 eos 分隔；块间无重叠、无 padding。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SKIP_DIRS = {".git", ".workbuddy", "node_modules", "__pycache__", ".venv",
             "build", "dist", "third_party"}
TEXT_EXT = (".md", ".py", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".sh", ".rst")


def iter_text(roots, max_file_mb: float):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in sorted(filenames):
                if not fn.lower().endswith(TEXT_EXT):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > max_file_mb * 2**20:
                        continue
                    with open(p, encoding="utf-8", errors="ignore") as f:
                        t = f.read()
                except OSError:
                    continue
                if len(t.strip()) >= 64:
                    yield p, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=40_000_000,
                    help="采够这么多 token 就停（uint32 文件 ≈ 4 bytes/token）")
    ap.add_argument("--max-file-mb", type=float, default=8.0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []

    buf = []          #攒批再过 tokenizer，快
    batch_chars, total_tokens = [], 0
    n_files = 0

    def flush():
        nonlocal total_tokens, n_files
        texts = [b for b in batch_chars if b]
        batch_chars.clear()
        if not texts:
            return
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            total_tokens += len(ids) + 1
            buf.extend(ids)
            buf.extend(eos)
        n_files += len(texts)

    for p, t in iter_text(args.dirs, args.max_file_mb):
        batch_chars.append(t)
        if len(batch_chars) >= 256 or sum(len(b) for b in batch_chars) > 8 * 2**20:
            flush()
            if total_tokens >= args.max_tokens:
                break
    flush()

    n_blocks = total_tokens // args.seqlen
    arr = np.asarray(buf[: n_blocks * args.seqlen], dtype=np.uint32).reshape(n_blocks, args.seqlen)
    np.save(args.out, arr)
    mb = os.path.getsize(args.out) / 2**20
    print(f"files {n_files}  tokens {total_tokens:,}  blocks {n_blocks} × {args.seqlen}  "
          f"-> {args.out} ({mb:.0f} MiB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
