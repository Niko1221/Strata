# DFlash2 code optimization, 2026-10-25

Local branch: `dflash2`, starting at `11f5d512`. All new paths are opt-in.
These results do not establish an increased whole-model advantage over MTP.

Hardware: RTX 4070 Ti SUPER 16 GiB (SM89), driver 610.57.04,
Ryzen 9 5900X, about 125 GiB RAM, Linux, GCC 14.2, CUDA 13.4.92.

## Implemented paths

- `STRATA_DFLASH2_PROMPT_BATCH=1`: group prompt-context projections by matrix,
  retaining the original groups of at most eight rows and each row's arithmetic.
  Separate scratch uses about 31 MiB at 512 rows. Batches are bounded by distinct
  physical KV cells. CUDA graphs are cached only for 64/128/256/512 rows.
- `STRATA_DFLASH2_DIRECT_QUANT=1`: quantize BF16 directly to Q8_1, removing an
  intermediate FP32 conversion. The existing float quantizer retains identical
  machine encoding (176 instruction words compared).
- `STRATA_DFLASH2_CONTEXT_GRAPH=1`: cache context projection graphs for 1–8 rows.
- `STRATA_DFLASH2_BF16_TILE=1`: select row tiles for the measured 2560-input,
  256/640-output BF16 shapes with at least seven rows. Values 2/4 force a tile
  for those shapes. The final automatic selection has parity coverage but has
  not had a separate whole-model timing run.

## Component measurements

Real official DFlash2 Q8_0 weights, synthetic 512-row target features, alternating
24 paired samples after warmup, input upload excluded:

| Context projection | Original median | Batched median |
| --- | ---: | ---: |
| Graphs enabled | 13.509 ms | 8.903 ms |
| Graphs disabled | 13.518 ms | 9.347 ms |

This is 34.1% and 30.9% lower component latency, respectively. It is not a
measurement of full prompt processing or generation. Samples are in
`prompt-component-summary.json`.

Microbenchmarks also found approximately 44% lower direct-quantization latency
(30 projections of 8×2560: 0.0534 → 0.0301 ms) and 37–52% lower latency for
selected small BF16 projections. These gains did not translate into a confirmed
whole-model gain in the runs below.

## Whole-model measurements

Qwen3.8-Flash-Next NVFP4 target, official Q8_0 drafter, 8192 logical/resident
context, prefill 512, short-read 64, expert cache 2900 MiB, reserve 480 MiB,
prompt cache off, 384 generated tokens, five measured repeats per task.
DFlash2: block 7, confidence floor .7, spec 8. MTP: max-T 4, floor .5.
Both use matching head-IL settings. Commands (local executable paths replaced with `${STRATA_DATA}`), executable hashes, environment,
token IDs, counters and raw samples are preserved in the adjacent JSON files.
The study runner is `../2026-10-21-dflash2-long-goal/run.py`.

| Configuration | code_short tok/s | code_long tok/s | cache_long tok/s |
| --- | ---: | ---: | ---: |
| DFlash2 baseline A | 44.765 | 43.024 | 45.114 |
| MTP A | 39.484 | 39.121 | 41.835 |
| Direct quantization | 44.689 | 43.018 | 45.208 |
| Context graphs | 44.967 | 43.007 | 45.174 |
| Both | 44.817 | 42.975 | 45.162 |
| MTP B | 39.507 | 39.217 | 41.775 |
| DFlash2 baseline B | 44.949 | 42.917 | 45.187 |
| Projection baseline | 44.650 | 42.969 | 45.107 |
| BF16 tile 2 | 44.965 | 43.063 | 45.155 |
| BF16 tile 4 | 44.940 | 43.047 | 45.189 |
| Tile 4 + quant + context graph | 44.981 | 43.037 | 45.103 |

Small differences are comparable to baseline variation. The pre-existing
DFlash2 advantage is about 8–13% on these tasks; an increase is unconfirmed.
All study arms passed their same-process target-only and common target-only
checks. The queued full-request prompt batching study was cancelled before its
first arm when the original five-hour deadline had elapsed. No result from it
is claimed. A separate four-prompt, 32-output-token smoke run passed controls,
including a 7227-token input; that is correctness evidence only.

## Correctness and limits

Passed:

- CUDA builds with the DFlash2 extension enabled and disabled.
- Final focused CTest: seven passed, one skipped because no model argument is
  supplied by CTest. The model-dependent fixture was run explicitly and passed
  in both configurations below. An additional broad CTest run was interrupted
  after 70 completed passing tests and one skipped test; the full suite was not
  completed.
- Policy tests and extension kernel parity tests, including direct quantization
  through 512 rows and BF16 projection tile/warp variants.
- Final Q8_0 context tests: all five physical FP16 KV layers are bit-identical,
  843,776,000 bytes compared each for prompt batching and small context graphs,
  across three ring wraps. Capture counts are checked (four and three graphs).
- Earlier Q4_0 and BF16 prompt batching checks: 843,776,000 bytes each, exact.
- Graph-disabled Q8_0 parity and synthetic distant-position checks at 262141
  and 263679 (928,153,600 bytes). This is not full 256K prompt generation.
- Compute Sanitizer checks of direct quantization, projections and prompt
  fusion: zero errors on the tested revision before the final graph diagnostic
  and per-instance context-flag changes.

An initial parity fixture omitted synchronization after uploading pageable
input to a stream shared with nonblocking drafter streams. The fixture was
fixed with an explicit input-completion barrier. Those failed preliminary logs
are not evidence of an engine defect.

Not tested: HIP, SYCL (compilers unavailable), AMD/Metal, full long-context
inference, final automatic BF16 selection throughput, full-request prompt batch
speedup. Defaults remain off; no default performance claim is made.

Rejected experiments: shared attention merge scales, staged selector codebook,
broad packed/interleaved Q8 projection layouts, two AVX2 NVFP4 accumulator
schedules and a causal confidence-budget replay. They were removed after mixed
or slower timings; replay is not runtime throughput evidence. Full experimental
patches and logs remain in the local sibling directory
`Strata-data/dflash2-opt-20261025`.
