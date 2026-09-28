#!/usr/bin/env python3
"""
quantize_mmproj.py - Quantize a multimodal projector (mmproj) GGUF file for Strata.

Reduces mmproj memory footprint (e.g. BF16 907 MB -> Q4_0 ~461 MB), allowing
significantly larger MoE expert cache residency on memory-constrained GPUs.

Wraps the official llama-quantize binary or provides an automated workflow
to quantize mmproj files to Q4_0, Q8_0, etc.

Usage:
    python tools/quantize_mmproj.py input_mmproj.gguf [output_quantized.gguf] [--type Q4_0]
"""

import sys
import shutil
import argparse
import subprocess
from pathlib import Path


def find_llama_quantize() -> Path | None:
    # 1. Check PATH
    cmd = shutil.which("llama-quantize")
    if cmd:
        return Path(cmd)

    # 2. Check standard common Windows locations
    candidates = [
        Path(r"C:\llama-cpp\llama-quantize.exe"),
        Path.home() / "AppData/Local/Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe/llama-quantize.exe",
        Path(__file__).resolve().parent.parent / "build/bin/llama-quantize.exe",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def quantize_mmproj(src_path: Path, dst_path: Path, quant_type: str = "Q4_0"):
    quantize_exe = find_llama_quantize()
    if not quantize_exe:
        sys.exit(
            "Error: 'llama-quantize' executable not found on PATH or standard locations.\n"
            "Please install llama.cpp (e.g. via winget or build) so llama-quantize is available."
        )

    print(f"Using quantizer: {quantize_exe}")
    print(f"Source mmproj:   {src_path} ({src_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Target:          {dst_path}")
    print(f"Target format:   {quant_type}")

    # Run llama-quantize src dst type
    cmd = [
        str(quantize_exe),
        str(src_path),
        str(dst_path),
        quant_type
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"Quantization failed with return code {proc.returncode}")

    if dst_path.exists():
        src_sz = src_path.stat().st_size / 1024 / 1024
        dst_sz = dst_path.stat().st_size / 1024 / 1024
        saved = src_sz - dst_sz
        reduction = (saved / src_sz) * 100.0 if src_sz > 0 else 0
        print(f"\nSuccess!")
        print(f"  Input size:   {src_sz:.1f} MB")
        print(f"  Output size:  {dst_sz:.1f} MB")
        print(f"  Saved VRAM:   {saved:.1f} MB ({reduction:.1f}% reduction)")
    else:
        sys.exit(f"Output file {dst_path} was not created.")


def main():
    parser = argparse.ArgumentParser(description="Quantize mmproj GGUF projector for Strata")
    parser.add_argument("src", type=Path, help="Path to input mmproj GGUF (e.g. BF16 / F16)")
    parser.add_argument("dst", type=Path, nargs="?", help="Path to output quantized GGUF")
    parser.add_argument("--type", default="Q4_0", help="Target quantization type (Q4_0, Q8_0, etc., default: Q4_0)")
    args = parser.parse_args()

    if not args.src.exists():
        sys.exit(f"Input file not found: {args.src}")

    if args.dst is None:
        name = args.src.name
        for old in ["BF16", "bf16", "F16", "f16", "F32", "f32"]:
            if old in name:
                name = name.replace(old, args.type)
                break
        else:
            name = f"{args.src.stem}-{args.type}.gguf"
        args.dst = args.src.with_name(name)

    quantize_mmproj(args.src, args.dst, args.type)


if __name__ == "__main__":
    main()
