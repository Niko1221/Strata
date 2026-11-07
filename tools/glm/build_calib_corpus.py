#!/usr/bin/env python3
"""Build the imatrix calibration corpus for the GLM-5.3-Flash-UNCENSORED UD-Q2 rebuild.

Composition (deterministic, local, no network): the project's English docs, the Chinese
GLM-port notes, and representative source files (python + C++), concatenated with section
markers and capped at --max-bytes.  The same bytes every run: the sha256 is recorded.

    python tools/glm/build_calib_corpus.py --out <dir>/calibration.txt
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib

ROOT = pathlib.Path("/data/AI/Strata")

SOURCES = [
    "README.md",
    "AGENTS.md",
    "docs/DETAILS.md",
    "docs/AI_REFERENCE.md",
    "docs/TERMINAL.md",
    "docs/TERMINAL_LAUNCH.md",
    "docs/MULTI_GPU.md",
    "GLM-PORT-STATUS.md",
    "GLM-PORT-LESSONS.md",
    "serve/server.py",
    "src/program/glm_main.cpp",
    "tools/iq_pack.py",
    "tools/nvfp4_convert.py",
    "tools/glm/start.sh",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bytes", type=int, default=900_000,
                    help="corpus size cap (bytes); ~230k tokens of mixed zh/en/code")
    a = ap.parse_args()

    parts, used = [], 0
    for rel in SOURCES:
        p = ROOT / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        block = f"\n\n===== FILE: {rel} =====\n\n" + text
        if used + len(block.encode()) > a.max_bytes:
            take = max(0, a.max_bytes - used)
            block = block[:take]
            if not block.strip():
                break
        parts.append(block)
        used += len(block.encode())
        if used >= a.max_bytes:
            break

    corpus = "".join(parts)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(corpus, encoding="utf-8")
    sha = hashlib.sha256(corpus.encode()).hexdigest()
    out.with_suffix(".sha256").write_text(sha + "\n", encoding="utf-8")
    print(f"calibration corpus: {out} ({used} bytes, ~{used // 4} tokens), sha256 {sha[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
