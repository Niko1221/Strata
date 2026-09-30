#!/usr/bin/env python3
"""Build a canonical Strata pack from a single Qwen3.8-27B GGUF shard.

This is deliberately model-specific. It reuses Strata's already-validated
canonicalisation table and emits the same manifest/index contract consumed by
WeightTable, but does not assume Flash-Next's two-shard expert layout.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ref"))

import gguf_reader as G  # noqa: E402
from strata_pack import tensor_entry  # noqa: E402


ALIGN = 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = pathlib.Path(args.gguf).resolve()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    g = G.GGUFFile(src)
    if not g.tensors:
        raise SystemExit("GGUF has no tensors")

    dense = out / "dense.bin"
    manifest = {
        "format": "strata-qwen35-pack-v1",
        "align": ALIGN,
        "source": {"gguf": str(src)},
        "tensors": {},
    }

    def align_up(n: int) -> int:
        return (n + ALIGN - 1) // ALIGN * ALIGN

    with src.open("rb") as sf, dense.open("wb") as df:
        data_start = g.data_start
        cursor = 0
        for i, t in enumerate(g.tensors):
            raw_bytes = t.expected_bytes()
            if raw_bytes is None:
                raise SystemExit(
                    f"{t.name}: unsupported GGUF type {t.type_name} or invalid shape"
                )
            sf.seek(data_start + t.offset)
            raw = sf.read(raw_bytes)
            if len(raw) != raw_bytes:
                raise SystemExit(f"{t.name}: short read ({len(raw)} != {raw_bytes})")

            cursor = align_up(cursor)
            df.seek(cursor)
            body, ent = tensor_entry(
                t.name,
                t,
                t.type_name,
                raw,
                cursor,
            )
            df.write(body)
            ent["file"] = "dense.bin"
            manifest["tensors"][t.name] = ent
            cursor += len(body)

            if (i + 1) % 128 == 0 or i + 1 == len(g.tensors):
                print(f"  packed {i + 1}/{len(g.tensors)} tensors", flush=True)

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )

    subprocess.run([sys.executable, str(HERE / "pack_index.py"), "--pack", str(out)], check=True)

    print(f"pack: {out}")
    print(f"dense.bin: {dense.stat().st_size / 2**30:.3f} GiB")
    print(f"tensors: {len(manifest['tensors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
