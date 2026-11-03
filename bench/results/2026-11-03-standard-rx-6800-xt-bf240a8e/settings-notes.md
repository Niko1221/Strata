# Settings behind these numbers (RX 6800 XT, gfx1030, ROCm 7.2.4)

`strata bench run` records the engine arguments and the sampling it sent in `config.json`, but not the
`env` block of the run config. That block is where most of this card's prompt speed comes from, so it is
listed here with what it was measured to be worth.

Everything in this file was measured **outside** the standard protocol, on 2026-10-23, with the custom
arms in `bench-prefill/` (`bench_arm.py`, restart per arm, fresh prompts, A-B-A control arms, engine
0.1.41 and 0.1.48.1). It explains the absolute numbers in `README.md`; it is not a second result and the
two runs in this folder do not depend on it.

## The env block

```
STRATA_HIP_PROMPT_F16=1
STRATA_GDN_HEAD=1
STRATA_GDN_PP=2
STRATA_GDN_CONVL2=1
STRATA_GDN_NOY=1
STRATA_RESIDENT_PIN=0
STRATA_ROCBLAS_TUNING=<repo>/tools/hip
```

| Variable | What it does | Measured on this card (Swift IQ3_XXS) |
| --- | --- | --- |
| `STRATA_HIP_PROMPT_F16=1` | Runs the prompt path's 16-bit dense GEMMs (GDN and QSA projections, the hidden-cache read/down, PLE) in FP16. Acts on activations, never on expert weights, so it is quantization-independent. | **468.6 → 834.3 tok/s at 4,011 tokens (+78%)**, 490.1 → 1,044.3 at 64,857 (+113%). IQ2_XS: +79% / +114%. Four greedy prompts byte-identical with and without it; `STRATA_DBG_NAN=1` found no non-finite values (max magnitudes ~3 orders below the FP16 ceiling). |
| `STRATA_GDN_*, 4 switches` | Gated delta-net kernel selection for this architecture. | +0 to +0.5% alone, +1 to +2.7% on top of F16. Bit-exact on gfx1030 (`hip_gdn_rec_head`: `FAILURES: 0`, same as without them). |
| `STRATA_ROCBLAS_TUNING=<dir>` | Directory holding a per-architecture rocBLAS solution table for the prompt GEMMs. The engine picks `<arch>-rocblas-<version>.txt` and ignores a table built for another rocBLAS. | On **ROCm 7.2.4 / rocBLAS 5.2.0** the default pick is poor at every bucket (unlike ROCm 10's rocBLAS 5.6, where the win is confined to short prompts): **403 → 463 tok/s at 817 tokens (+14.9%)**, 886 → 1,015 at 4,011 (+14.6%), 1,076 → 1,103 at 64,857 (+2.5%), F16+GDN in both arms, engine 0.1.48.1. A repeat of the table-off arm agreed within 0.7%. |
| `STRATA_RESIDENT_PIN=0` | Leaves the expert cache evictable. | Not measured separately; kept at the value the rest of the series used. |

The table itself is `tools/hip/gfx1030-rocblas-5.2.0.dabb6df2b9.txt` (140 rows, 124 with a solution).
It was written by the in-tree tuner:

```sh
cmake --build build-hip --target tune_rocblas
./build-hip/tune_rocblas --tokens 256,512,768,1024,1152,1536,2048,4096,8192,16384 \
  --tuning-out bench-prefill/rocblas-table-7.2.4.txt    # ~11 min, 140 cases
```

Together, the same 12,050-token request on this card read **488 tok/s before this work and 1,157 tok/s
after** it (engine 0.1.48.1, F16 route + table; both runs in this folder land in the same band).

Both runs' `engine.log` shows the two switches engaged:

```
prefill gemm: rocBLAS tuning enabled (140 rows, 124 with a solution, gfx1030, rocBLAS 5.2.0.dabb6df2b9)
strata prefill: HIP device 0 gfx1030 - STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out
```

## Engine arguments that matter on this card

| Argument | Why |
| --- | --- |
| `--prefill auto:16384` | 16K prompt chunk. Plain `auto` (8,192) read 6% lower at 12K and 4.6% lower at 64K, measured on Swift and ISTA IQ3_XXS. |
| `--vram-reserve-mib 768` | 768 leaves 908-912 MiB free with everything loaded and no low-VRAM warning in either run. At 256 this card ends under the engine's 512 MiB line: the log prints the #2065 warning and one arm read 2x slower — which looks exactly like a regression and is not one. |
| `--kv int8 --kv-resident 32768 --max-context 262144` | The model's native window, 8-bit KV, 32K tokens of KV kept in VRAM. |
| `--spec 4 --spec-min-p 0.5 --mtp <draft vocab>` | Speculative decode with the model's own MTP head; both runs accepted roughly three quarters of the drafts, so the decode figures include it. |
| `--expert-profile data/expert-profile.bin --expert-cache auto` | Ranks the experts so the cache pre-fills with the ones this model actually routes to. |
| `--vision` | Pictures on, with the encoder on the CPU. No image was sent by the protocol. |

## Traps for anyone repeating this on gfx1030

- **The first arm of a sweep is cold.** The same 817-token prompt took 2,983 ms as a first arm and
  1,862 ms later in the same sweep, a 40% swing from GPU clock ramp. Discard a throwaway arm.
- **Under-reserving VRAM looks like a slowdown, not an error.** Check `grep 'WARNING: only'` in the
  engine log before believing a slow number.
- **The rocBLAS table is version-guarded.** It must be named for the rocBLAS the engine actually
  loaded (`gfx1030-rocblas-5.2.0.dabb6df2b9.txt` here), and `STRATA_ROCBLAS_TUNING` takes the
  directory, not the file.
