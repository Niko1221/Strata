# gfx1200 (RX 9060 XT) — HIP backend, engine 0.1.26

Source: OriginalStrata tip `4c68013` (AMD HIP backend PR #121 rebased onto 0.1.24, engines 0.1.25/0.1.26),
with exactly three gfx1200 deltas: the hip_backend arch-gate regex (gfx1[12]xx), the device.cu arch check
(gfx1100|gfx1200), and a minimal `math_constants.h` (this ROCm install does not ship one).
Build: `STRATA_ENABLE_HIP=ON STRATA_PREFILL_MMQ=ON`, `CMAKE_HIP_ARCHITECTURES=gfx1200`,
ggml from the pinned llama.cpp `3cf03257f`. Setup and deltas: [docs/AMD_HIP_GFX1200.md](../../../docs/AMD_HIP_GFX1200.md);
launcher `gfx1200-run.sh` at the repository root.

Hardware: RX 9060 XT 16 GB (gfx1200, wave32), Ryzen 9 7950X, 64 GiB DDR5, ROCm 7.2 (/opt/rocm,
hipBLASLt 100202). Model: GSQ-RCO Coder **IQ1_M** (2 shards), PLE on SSD, expert profile
`expert-profile-coder.bin` (12288 pairs). Sampling greedy; MTP `--spec 4 --spec-min-p 0.5`.
Every run: `STRATA_HIPBLASLT_TUNING=tools/hip/gfx1200-hipblaslt-100202.txt` (67 rows calibrated
on this card with `tools/hip/tune_hipblaslt`; 7-15x over plain hipBLAS per shape), `STRATA_PREFILL_MMQ=1`.

Baseline for comparison: llama.cpp HIP on the same card measured 20 tok/s decode / 450 tok/s prefill.

## Short context (2374-token code prompt, 128 generated)

| Arm | decode tok/s | prefill tok/s |
| --- | ---: | ---: |
| hand-rolled gfx1200 port (before) | 14-19 (garbage output) | 13-20 |
| original backend, defaults | 27.2 | 304.7 |
| + MMQ + prefill 8192 | 27.3 | 394.9 |
| + Lt table (T=8192 rows only) | 28.1 | 393.2 |
| + Lt chunk table (59 rows) + prefill 2560 | 27.8 | 565.9 |
| **final: + mmap/resident-cpu-experts, kv int8, pcie-frac 0, reserve 1792 MiB** | **31.0** | **541.0** |

Rejected arms: 31 pool workers 26.7 (DDR5 contention; 15 workers is right), `--spec 6` 30.4,
`--prefill 8192` 514.6, Lt table with T=8192 rows at chunk 2560 (no dispatch match).

Steady state: **1.39 GiB VRAM free** (rocm-smi, `--vram-reserve-mib 1792`; 1024 left only ~360 MiB
with a desktop running). 30/30 ctest (`hip_prefill_hipblaslt_gemm` included, needs the table).
Greedy output byte-identical over 3 runs; 512-token run clean, no watchdog lines.

## Long context (`--kv int8 --kv-resident 65536 --prefill 16384`, 32-512 generated)

| Prompt | prefill tok/s | decode tok/s (steady) | notes |
| --- | ---: | ---: | --- |
| 65,045 tokens | 754-761 | 26.3 (512-tok run) | KV: 65,536/131,072 cells in VRAM, 1.55 GiB pinned |
| 130,091 tokens | 630-637 | 22.7 | 3.09 GiB pinned KV; end-to-end clean |

Long-context arms: `--kv-resident 32768` 662 (64K); `--prefill auto` (8192) 731; `--prefill 4096` 703;
`--kv-resident 98304` 25.6 decode (worse); `--kv k8v4` prefill 680 but decode 19.0 (its full-VRAM KV
shrinks the expert tier more than streaming costs); `--prefill 32768` refuses (buffers > 16 GB).
A/B-tested the gfx1200 `multiProcessorCount`=16-WGP quirk by doubling ggml's `nsm`: 726.6 vs 731.1 —
no effect (only ggml stream-k reads it, and the MMQ shapes do not partition by it here); reverted.

## Local `qsa_decode_attn.cu` tweak (kept, small)

`#pragma unroll 4` on the V accumulation loop: 64K prefill 754 -> 760 (within noise, not harmful);
correctness unchanged (greedy outputs identical). A prefill query batch of 64 (instead of 32) also
measured within noise on 0.1.26 but was REVERTED: under 0.1.27 it overflows the prompt path's
borrowed buffer region ("device buffers for a chunk of 256 tokens do not fit"). The 32.6-36.5%
"qsa attn" prefill share is structural: the CUDA sm80
tensor-core kernel is compiled out on HIP, so prefill runs the decode kernel 32 queries at a time.
A real fix is an RDNA4 MFMA rewrite of `qsa_prompt_attn.cu` (future work).

## Reproduce

```sh
export STRATA_NATIVE_GGUF=/path/to/model-00001-of-00002.gguf
export STRATA_PLE_GGUF=/path/to/model-00002-of-00002.gguf
export STRATA_MTP_RT=/path/to/mtp/rt
./gfx1200-run.sh --tokens-file <prompt tokens, comma/space separated> --max-new 512 --stats \
  --max-context 131072 --kv int8 --kv-resident 65536 --prefill 16384
```
