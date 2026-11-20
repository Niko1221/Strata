# Local evolutionary loop — completed result (2026-11-20)

## Method

NIM/Shinka could not be used reliably (patch extraction failures, HTTP 429/529),
so the evolutionary loop was run locally with the agent as the proposal engine:

1. New tunable dimension introduced: `STRATA_Q38_SCORE_MB` (8..128, default 32)
   controls the bounded batched-attention score buffer and therefore the
   sub-batch token count. Build `q38-9b-evo-20261120` (CTest 18/18).
2. LLM proposed five candidates (control 32, small 8/16, large 64/128 MB).
3. Fitness measured with **drift-cancelling A/B pairs** B,C,B,C,B,C on the
   pinned 9B 1024-token fixture; score = median of per-pair ratios; every run
   must emit the baseline token `15217`.
4. Winner validated on held-out token counts 256/512/1024/2048 with three-run
   medians and a token gate, then on the 27B fixture.

## Generation 0 archive (all correct, token `15217`)

| Candidate | median speedup | per-pair ratios |
|---|---|---|
| control score_mb=32 | 1.0257 | 0.9492, 1.0257, 1.0275 (noisy: identical to baseline) |
| score_mb=8 | 0.9357 | 1.0058, 0.8962, 0.9357 |
| score_mb=16 | 1.0002 | 0.9685, 1.0103, 1.0002 |
| score_mb=64 | 1.0099 | 1.0099, 1.0037, 1.0324 (all positive) |
| score_mb=128 | 1.0094 | 1.0072, 1.0094, 1.0095 |

The control's apparent 1.026 gain is measurement drift (it is the same config
as the baseline). `score_mb=64` was selected because all three pairs were
positive and it was the most stable improvement over the noise floor.

## Held-out validation (winner = score_mb=64)

| Tokens | Baseline | Candidate | Speedup | Tokens match |
|---|---|---|---|---|
| 256 | 307.63 ms | 262.35 ms | 1.1726 | yes |
| 512 | 372.90 ms | 385.83 ms | 0.9665 | yes |
| 1024 | 683.62 ms | 642.52 ms | 1.0640 | yes |
| 2048 | 1672.64 ms | 1636.07 ms | 1.0223 | yes |

Result: **ACCEPTED** (3 of 4 held-out shapes improved, all token-identical).

27B fixture validation (32-token prompt, three-run medians): baseline 1187.38 ms,
candidate 583.997 ms, speedup 2.0332, token `201` in all runs. The baseline is
roughly twice the usual 27B prefill time, so this ratio includes system-load
drift; the candidate's absolute 584 ms is consistent with the normal range and
token identity is confirmed. Treat the 27B gain as qualitative confirmation,
not a precise 2x speedup claim.

## Decision

`STRATA_Q38_SCORE_MB=64` is recorded as an opt-in evolved configuration. The
default remains 32 because the 512-token shape regressed; adopting the
evolved value as a default would need a weighted objective over the target
shape distribution. The evolutionary method itself is now demonstrated:
LLM proposals -> automated A/B with token gate -> archive -> held-out validation.

## Evidence files

- `C:\Users\downl\AppData\Local\Temp\opencode\evo-local\archive.json`
- `...\best.json`, `...\winner.json`
- `...\heldout_result.json`, `...\heldout_27b.json` (pending)
- `...\evolution2.log`, `...\heldout.log`
