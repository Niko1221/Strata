#!/usr/bin/env python3
"""Prepare and run Qwen3.8-27B-GGUF through Strata's native Qwen3.5 runtime.

The script deliberately does not invoke llama-server, Ollama, Transformers or
PyTorch. It only downloads the GGUF (when requested), canonicalises it into
Strata's pack format, builds Strata, writes the normal serve/server.py config,
and optionally starts the existing Strata web/API frontend.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/"
    "Qwen3.8-27B-Q4_0.gguf?download=true"
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=pathlib.Path)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--models-dir", type=pathlib.Path,
                     default=ROOT / "models" / "qwen3.8-27b")
    ap.add_argument("--pack-dir", type=pathlib.Path,
                     default=ROOT / "data" / "packs" / "qwen3.8-27b-q4_0")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--resident-layers", default="auto")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build-qwen35")
    ap.add_argument("--cuda-arch", default="120")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    if args.gguf is None:
        if not args.download:
            ap.error("--gguf is required unless --download is specified")
        args.models_dir.mkdir(parents=True, exist_ok=True)
        args.gguf = args.models_dir / "Qwen3.8-27B-Q4_0.gguf"
        if not args.gguf.exists():
            print(f"downloading {args.url}")
            with urllib.request.urlopen(args.url) as src, args.gguf.open("wb") as dst:
                while True:
                    chunk = src.read(8 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
    args.gguf = args.gguf.resolve()
    if not args.gguf.is_file():
        raise SystemExit(f"GGUF not found: {args.gguf}")

    args.pack_dir = args.pack_dir.resolve()
    run([sys.executable, str(ROOT / "tools" / "qwen35_pack.py"),
         "--gguf", str(args.gguf), "--out", str(args.pack_dir)])
    run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"),
         "--gguf", str(args.gguf), "--out", str(args.pack_dir)])

    if args.build:
        run(["cmake", "-S", str(ROOT), "-B", str(args.build_dir),
             "-DSTRATA_ENABLE_CUDA=ON",
             f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}"])
        run(["cmake", "--build", str(args.build_dir), "--config", "Release",
             "--target", "strata", "-j"])

    exe = (args.build_dir / "strata")
    if os.name == "nt":
        exe = args.build_dir / "Release" / "strata.exe"
    elif not exe.exists():
        exe = args.build_dir / "strata"
    if not exe.exists():
        exe = ROOT / "strata"
    if os.name == "nt" and exe.suffix.lower() != ".exe":
        exe = exe.with_suffix(".exe")

    cfg = {
        "exe": str(exe.resolve()),
        "args": [
            "--family", "qwen35",
            "--pack", str(args.pack_dir),
            "--max-context", str(args.context),
            "--resident-layers", str(args.resident_layers),
        ],
        "cwd": str(ROOT),
        "tokenizer": str(args.pack_dir / "tokenizer"),
        "model_name": "qwen3.8-27b",
        "log": str(ROOT / "strata-qwen35.log"),
        "port": args.port,
    }
    cfg_path = ROOT / "qwen35.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"wrote {cfg_path}")

    if args.start:
        run([sys.executable, str(ROOT / "serve" / "server.py"),
             "--engine", "strata", "--config", str(cfg_path),
             "--port", str(args.port)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
