# AMD HIP backend on gfx1200 (RDNA4, RX 9060 XT)

The HIP backend ships validated for gfx1100 (RDNA3, RX 7900 XTX); see [AMD_HIP.md](AMD_HIP.md) and
[AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md). This note records what it takes to run the same
backend on **gfx1200** (RDNA4, RX 9060 XT 16 GB, ROCm 7.2, hipBLASLt 100202), which this repository's
author has not tested. Everything else on this page was measured on that card; numbers below are from
that setup (Ryzen 9 7950X, 64 GiB DDR5, PLE on SSD), not the gfx1100 hosts.

gfx1200 keeps the two properties the backend relies on: **wave32** (verified at run time) and the
**64 KiB workgroup LDS** limit. It has no `sudot4` (the shim's dp4a takes the portable emulation) and
its driver reports `multiProcessorCount` as the WGP count (16, not 32 CUs); nothing in the engine or
the ggml MMQ shapes turned out to depend on that (A/B-tested, see the bench README).

## Three deltas from gfx1100

1. `cmake/hip_backend.cmake`: accept gfx11xx **and gfx12xx** in the arch gate.
2. `src/core/device.cu`: the run-time device check accepts `gfx1200` beside `gfx1100`.
3. `include/math_constants.h` (new): some ROCm installs (this 7.2 one) do not ship
   `<math_constants.h>`, which `qsa_prompt_attn.cu` includes. The file defines `CUDART_INF_F`,
   `CUDART_NAN_F` and `CUDART_PI_F`; harmless where ROCm does provide the header, because the
   hip_compat include dir is searched AFTER the toolchain's.

Plus one data file: `tools/hip/gfx1200-hipblaslt-100202.txt`, a hipBLASLt solution table **calibrated
on a gfx1200** (the shipped gfx1100 tables do not dispatch on this card: the runtime guards arch and
version and falls back to plain hipBLAS). See "Recalibrating" below.

## Build

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1200 \
  -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_GGML_DIR=/path/to/pinned/llama.cpp   # optional: reuse a checkout of the pinned commit
cmake --build build-hip --target strata -j
```

Tests (the suite of AMD_HIP.md, plus the Lt GEMM test once the table exists):

```sh
cmake --build build-hip -j
STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1200-hipblaslt-100202.txt" \
  ctest --test-dir build-hip --output-on-failure --timeout 120 \
  -E '^(ple_parity|platform_memory_test)$'
```

30/30 on the measurement card (the two exclusions are the same as gfx1100: an external fixture and a
locked-memory limit).

## Recalibrating the hipBLASLt table

The tuned prefill path only dispatches when the table's rows match the card AND the shape. Calibrate
on the card, at the shapes the engine actually launches (the three dense projections and the
bf16/f16 split at the chunk sizes you will run; 4096 and 512 buckets cover the rest):

```sh
cmake --build build-hip --target tune_hipblaslt -j
./build-hip/tune_hipblaslt --tuning-out tools/hip/gfx1200-hipblaslt-<version>.txt \
  --case f16,2560,10240,2560,10240 --case f16,2560,320,10240,320 --case f16,2560,2560,320,2560 \
  --case bf16,2560,10240,2560,10240 --case bf16,2560,320,10240,320 --case bf16,2560,2560,320,2560 \
  --case f16,512,10240,2560,10240  --case f16,512,320,10240,320  --case f16,512,2560,320,2560 \
  --case bf16,512,10240,2560,10240 --case bf16,512,320,10240,320 --case bf16,512,2560,320,2560
```

(The committed table adds the long-context chunk buckets and the two rows
`hip_prefill_hipblaslt_gemm` requires: `bf16 48 2560 96 4096` and `f16 512 2560 512 4096`.)
Per-shape speedups over plain hipBLAS on this card were 7-15x; end-to-end prefill +43% at 8K.
A `nsm`-doubling experiment (gfx1200 reports 16 WGPs, not 32 CUs, in `multiProcessorCount`) measured
726.6 vs 731.1 tok/s at 64K - no effect; the upstream code is kept unchanged.

## Running

Pack the model with the experts file if you want the decode configuration
(`--mmap-experts` needs the pack's `experts.bin`; for a native (IQ) pack:
`python tools/iq_pack.py --gguf <shard 1> --out <pack> --experts-bin`), then:

```sh
export STRATA_NATIVE_GGUF=/path/to/model-00001-of-00002.gguf
export STRATA_PLE_GGUF=/path/to/model-00002-of-00002.gguf
export STRATA_MTP_RT=/path/to/mtp/rt
./gfx1200-run.sh --tokens 9707,11,420,374,30,311,2581,1490,13 --max-new 40 --stats
```

The launcher sets `STRATA_HIPBLASLT_TUNING`, `STRATA_PREFILL_MMQ=1` and the measured engine
configuration (`--mmap-experts --resident-cpu-experts --kv int8 --pcie-frac 0 --expert-cache auto
--adapt-every 0 --prefill 2560 --spec 4 --spec-min-p 0.5 --vram-reserve-mib 1792`; if you applied the
changes as a patch, make it executable first: `chmod +x gfx1200-run.sh`). The 1792 MiB
reserve leaves ~1.4 GiB of VRAM free at steady state on a 16 GB card with a desktop compositor
running; `1024` left only ~360 MiB.

Long context (64K-128K): append

```
--max-context 262144 --kv int8 --kv-resident 65536 --prefill 16384
```

`--prefill 32768` does not fit a 16 GB card ("device buffers for a chunk of 32768 tokens do not
fit"). `--kv k8v4` runs but its full-VRAM KV shrinks the expert tier and decode drops (~19 vs 26
tok/s at 64K); int8 KV with `--kv-resident` streaming wins on 16 GB.

## Measured (IQ1_M Coder, greedy, MTP)

| Case | decode tok/s | prefill tok/s |
| --- | ---: | ---: |
| 2,374-token prompt, 128 new | 31.0 | 541.0 |
| 65,045-token prompt, 512 new | 26.3 | 760.8 |
| 130,091-token prompt, 32 new | 22.7 | 637.3 |

Full arm-by-arm tables, rejected experiments and stability notes:
[bench results](../bench/results/2026-09-30-gfx1200/README.md). Determinism: greedy output
byte-identical across 3 runs. Correctness: coherent long-form code answers, verified against the
pack's tokenizer.

## Local changes to the portable attention path (kept)

The CUDA sm80 tensor-core prompt-attention kernel is compiled out on HIP, so prefill runs the decode
kernel 32 queries at a time ("qsa attn" is 33-37% of long-context prefill GPU time). One
bitwise-neutral tweak was measured and kept: `#pragma unroll 4` on the value-accumulation loop.
Doubling the prefill query batch (32 -> 64, for L2 reuse of overlapping selections) measured within
noise on 0.1.26 and BROKE the prompt path's buffer fitting under 0.1.27's lend math ("device buffers
for a chunk of 256 tokens do not fit") - reverted; the batch stays at the upstream 32. An RDNA4 MFMA
rewrite of the prompt kernel remains the main unexplored lever.

## Not validated here

Other RDNA4 cards, Windows HIP, multi-GPU, vision, answer-quality benchmarks, and full 262144-token
contexts (tested to 131K). The gfx1100 caveats in AMD_HIP.md apply unchanged.
