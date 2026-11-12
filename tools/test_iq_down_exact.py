#!/usr/bin/env python3
"""Build and run real public exact/native down rows against existing GGML AVX2 dots.

Linux GCC runner. No downloads, GPU, model, or service operations. The test supplies
ISA probes and unused backend stand-ins, but links the actual IQ and native TUs.
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
    gg = args.llama_dir.resolve() / 'ggml'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    flags = ['-O3', '-fno-fast-math', '-ffp-contract=off', '-mavx2', '-mfma',
             '-mf16c', '-ffunction-sections', '-fdata-sections']
    includes = [v for p in [ROOT / 'include', gg / 'include', gg / 'src', gg / 'src/ggml-cpu']
                for v in ['-I', str(p)]]
    cc, cxx = os.environ.get('CC', 'gcc'), os.environ.get('CXX', 'g++')
    commands = []
    for source, obj in [(gg / 'src/ggml-cpu/arch/x86/quants.c', 'ref.o'),
                        (gg / 'src/ggml-quants.c', 'quant.o')]:
        commands.append([cc, *flags, *includes, '-c', str(source), '-o', str(out / obj)])
    # The native dispatcher is deliberately compiled for the baseline ISA.
    commands.append([cxx, '-std=c++20', '-O2', '-ffunction-sections', '-fdata-sections', *includes,
                     '-c', str(ROOT / 'src/kernels/cpu/native_expert.cpp'), '-o', str(out / 'native.o')])
    commands.append([cxx, '-std=c++20', *flags, *includes,
                     str(ROOT / 'tools/iq_down_exact_test.cpp'), str(ROOT / 'src/kernels/cpu/iq_avx2.cpp'),
                     str(out / 'native.o'), str(out / 'ref.o'), str(out / 'quant.o'),
                     '-Wl,--gc-sections', '-o', str(out / 'exact')])
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
    receipts = []
    jobs = [(command, env) for command in commands]
    jobs += [([str(out / 'exact'), entry], dict(env, STRATA_IQ_DOWN_EXACT=enabled))
             for entry, enabled in [('public', '0'), ('native', '0'), ('native', '1')]]
    for command, job_env in jobs:
        result = subprocess.run(command, env=job_env, capture_output=True, text=True, timeout=180)
        receipts.append(dict(argv=command, exact=job_env.get('STRATA_IQ_DOWN_EXACT'),
                             exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
        (out / 'receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
        print(result.stdout, end='')
        print(result.stderr, end='')
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
