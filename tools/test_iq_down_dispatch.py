#!/usr/bin/env python3
"""CPU-only dispatch regression: real native_expert TU, instrumented backends.

This checks control flow/fallbacks, NOT numerical parity. Run test_iq_down_exact.py
separately for real IQ/GGML arithmetic. Uses only an existing GGML checkout.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    ggml = args.llama_dir.resolve() / 'ggml'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cxx = os.environ.get('CXX', 'g++')
    flags = ['-std=c++20', '-O2', '-ffunction-sections', '-fdata-sections',
             '-I' + str(ROOT / 'include'), '-I' + str(ggml / 'include')]
    commands = [
        [cxx, *flags, '-c', str(ROOT / 'src/kernels/cpu/native_expert.cpp'), '-o', str(out / 'native.o')],
        [cxx, *flags, str(ROOT / 'tools/iq_down_dispatch_test.cpp'), str(out / 'native.o'),
         '-Wl,--gc-sections', '-o', str(out / 'dispatch')],
        *[[str(out / 'dispatch'), mode] for mode in ['on', 'off', 'zero', 'invalid', 'no-iq', 'mt9']],
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
    receipts = []
    for command in commands:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        receipts.append(dict(argv=command, exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        (out / 'receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
        print(result.stdout, end='')
        print(result.stderr, end='')
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
