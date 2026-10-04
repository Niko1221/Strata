# RX 6900 XT (gfx1030): the prompt path's 16-bit GEMMs in FP16 in and out (2026-10-04)

On gfx103x (RDNA2) the prompt path ran its 16-bit GEMMs as rocBLAS FP16-in / FP32-out (`Gemm::f16`, `Gemm::native`) and
BF16-in / FP32-out (`Gemm::bf16`). rocBLAS on gfx1030 has tuned kernels for FP16-in / **FP16-out** only; the other two
fall back to generic kernels about 6.6x slower. This change runs those GEMMs FP16 in and out on gfx103x, writing the FP16
result into the start of each FP32 row of Y and widening it there (no extra buffer), and has the kernels that produce the
BF16 GEMMs' activation images (`gr_*`, `to_bf16`) write FP16 instead. `STRATA_HIP_PROMPT_F16=0` is the old path, so every
number below is the same binary with and without that switch.

## Rig

- 2x AMD Radeon RX 6900 XT 16 GB (gfx1030), each PCIe 4.0 x8, no P2P used. AMD Ryzen 5 5600X, 128 GB DDR4-3200.
- Ubuntu 26.04, kernel 7.0.0-38-generic, ROCm 10.0.0 (HIP 7.15.26333, rocBLAS 5.6).
- Engine: `origin/main` at `6f32ec0` (0.1.39) with this change, `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON`.
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S (2 shards, `--native`), MTP draft layer on, `--kv int8 --kv-resident 32768
  --max-context 131072`, `--expert-cache auto`. Not one of setup's models (see the limits).

## Why: rocBLAS on gfx1030

A 7.3K-token prompt on one card (rocprofv3, 0.1.38): rocBLAS `Cijk_..._HSS_..._MT64x32x8` (FP16 in, FP32 out, a fallback
tile) was 54% of the prompt's 15.2 s of GPU time, the BF16 products (`BSS`) another ~15%. Microbenchmark, N = 10240,
T = 7313, K = 2560:

| rocBLAS GEMM on gfx1030 | TFLOPS |
| --- | ---: |
| FP16 in, FP32 out (HSS) | 5.6 |
| BF16 in, any out | ~5.3 |
| **FP16 in, FP16 out (HH, fp32 accumulate)** | **37.7** |

## Prompt speed

One card, cold server per run, the first message of each chat below read by the batched prompt path; the
server's own line `strata serve: prompt N tokens = 0 reused + N read in T ms`. Three runs per path (two of them the
distribution check's), all within 0.3%:

| Prompt tokens | old path (`STRATA_HIP_PROMPT_F16=0`) | FP16 | gain | read time, old -> FP16 |
| ---: | ---: | ---: | ---: | ---: |
| 9,427 | 439 tok/s | 744 tok/s | +69% | 21.5 s -> 12.7 s |
| 34,659 | 466 tok/s | 915 tok/s | +96% | 74.4 s -> 37.9 s |
| 105,811 | 461 tok/s | 926 tok/s | +101% | 229.5 s -> 114.3 s |

Decode is unchanged within the noise (44-48 tok/s on one card): it does not use these GEMMs.

Both cards, cold server per run, one 8,275- and one 33,586-token prompt of repository text (128 tokens answered):

| | old path | FP16 | gain |
| --- | ---: | ---: | ---: |
| `--layer-split auto` (25 / 23 layers), 8.3K | 444 tok/s | 833 tok/s | +88% |
| `--layer-split auto`, 33.6K | 707 tok/s | 1,356 tok/s | +92% |
| one card + the second as expert helper (`--expert-cache-device1 auto --remote-expert-opt`), 8.3K | 432 tok/s | 798 tok/s | +85% |
| the same, 33.6K | 454 tok/s | 887 tok/s | +95% |

Decode is the same within the noise in both setups (these short answers: 51-58 tok/s split, 36-42 helper).

## Range check

FP16 ends at 65504 where BF16 reaches 3.4e38, so the precondition is that nothing the prompt path feeds these GEMMs or gets
out of them leaves FP16's range. Measured on 0.1.38 with a diagnostic counter (not part of this change) over 33.7K tokens of
English docs, 43.3K of Chinese logs and 32.4K of C++ on one card: every BF16 GEMM's X and W and every FP16 output - **0**
values beyond 65504, **0** infinities. Non-zero |x| < 6.1e-5 (FP16's subnormal band): 0.008-1.8% of X, 0.1-0.9% of W.
Out-of-range values would still be finite: the activation writers saturate to +-65504 (`hf_sat`, as the FP16 images
already did), a NaN stays a NaN.

## Distribution check (teacher-forced)

The method of `bench/results/2026-10-03-v100-prompt-attn` (`docs/UNSLOTH_Q4.md`): `STRATA_LOGPOS=<file>
STRATA_LOGPOS_TOPK=256` on a serve engine (one card, `--short-read 640 --adapt-every 100000`, greedy). Each chat is a long
first message of this repository's docs and source (9,427 / 34,659 / 105,811 tokens, disjoint, read by the batched prompt
path - the code under test), a fixed assistant reply, and a last message of ~580 tokens of other repository text that is
read through the verify windows, where every position is scored: 578 / 519 / 481 positions. KL is P || Q over P's top 256
plus one bucket for the rest.

| comparison | KL mean (9K / 35K / 106K) | KL median | argmax same | top-10 overlap |
| --- | --- | --- | --- | --- |
| old path vs old path (a second run) | 0 / 0 / 0 | 0 | 100% | 100% |
| FP16 vs FP16 (a second run) | 0 / 0 / 0 | 0 | 100% | 100% |
| old path vs FP16 | 0.032 / 0.021 / 0.019 | 0.0056 / 0.0041 / 0.0038 | 94.5 / 95.0 / 96.5% | 91.4 / 92.8 / 91.5% |
| **reference vs old path** | 0.061 / 0.033 / 0.023 | 0.0092 / 0.0096 / 0.0055 | 93.1 / 93.3 / 93.8% | 90.6 / 92.8 / 91.9% |
| **reference vs FP16** | 0.020 / 0.015 / 0.028 | 0.0034 / 0.0044 / 0.0038 | 95.5 / 96.5 / 95.8% | 92.7 / 93.2 / 91.6% |

The reference is the old path with `STRATA_PREFILL_BF16X2=1` (every BF16 activation image carries its low part too, close
to fp32 activations; 393-411 tok/s). The FP16 path is closer to it than the old path in all three chats by median KL and
argmax agreement, and by mean KL in two of three (in the 106K chat the FP16 path's p99 is lower, 0.20 vs 0.24, and its mean is pulled up
by a few positions beyond that). That is the expected direction: an FP16 activation keeps 10 mantissa bits where BF16 keeps 7, which outweighs
rounding the GEMM's output to FP16 before it is widened.

A rounding-order control as in the V100 report did not work here: `--prefill 2048` instead of auto (8192; the smaller
chunk did take effect, 1,059 lent slots instead of the 8192-token ring) gave results **bit-identical** to the old path, so on
this path the chunking does not change the arithmetic and cannot serve as a control.

## Output checks

- **Needle recall** (`tools/needle_bench.py --lengths 8k,32k,128k --depths 10,30,50,70,90`, greedy, thinking off, a fresh
  server per path, one card; prompts of 7.9K, 32.2K and 125.4K tokens): **15 of 15 found with the old path, 15 of 15 with
  FP16.** A 125K-token read takes 275 s on the old path, 139 s with FP16 (the later depths are shorter because the server
  reuses the checkpoint of the text before the needle).
- **Repeatability:** two runs of each path give bit-identical log-probabilities (table above), and no run of this report
  (17 server starts) stalled.

## What changed

- `Gemm::set_f16_io(bool)`: off by default, so every caller other than the prompt path (e.g. `gemm_bf16_parity`) runs what
  it ran. `Prefill::init` turns it on together with `set_act_f16` when `prompt_f16()` is true for its device.
- `prompt_f16()`: gfx103x, cached per device (a layer split can mix cards); `STRATA_HIP_PROMPT_F16=0/1` overrides.
- With it on: `Gemm::f16` / `Gemm::native` run HH into Y's own rows (`ldc = 2 ldy` halves) and `widen_rows_f16` widens each
  row from its end; `Gemm::bf16` converts W row slices to FP16 through the existing dequantization scratch (as
  `native()` does) and takes X as the FP16 image. `bf16x2_mode()` is 0 there (the BF16 low part has no FP16 meaning).
- No new device memory, no host sync, no fallback branch; CUDA builds are unchanged (`#if defined(__HIPCC__)`).

## Limits

- One machine, one model (IQ3_S, packed from the GSQ-RCO GGUF, not one of setup's), one corpus for the distribution check.
- gfx1031/1032 (the same rocBLAS family) are enabled by the same `gfx103` prefix but were not run.
