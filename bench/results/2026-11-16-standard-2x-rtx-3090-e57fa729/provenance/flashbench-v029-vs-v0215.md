# Extra context: the same box on v0.2.9 and v0.2.15 (flashbench, not the standard protocol)

Same PC, same Gyro-M file and pack, same layer split (24/24), int8 KV at 262,144, every expert resident, MTP drafter
(`--spec 4 --spec-min-p 0.7`), `STRATA_GDN_TC=0 STRATA_GDN_CONVL2=0 STRATA_ROUTE_TAIL_SKIP=0 STRATA_SPLIT_P2P=1 STRATA_HC_Q8=1`.
Tool: our own `flashbench.py` (OpenAI streaming API, a random marker at the start of every prompt so no prompt is reused,
512 output tokens, thinking off, median of 3 runs, 2 at 128K). v0.2.9 = engine 0.1.55, run 2026-11-07;
v0.2.15 = engine 0.1.61, run 2026-11-16. Both builds carry the same three patches (#1937, #1099 decode hand-off,
table-only PLE shard).

Prompt reading, tok/s (greedy / sampled):

| Build | 1K | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| v0.2.9 | 1,321 / 1,316 | 2,491 / 2,481 | 4,135 / 4,101 | 4,780 / 4,764 | 5,088 / 5,027 |
| v0.2.15 | 1,534 / 1,532 | 2,736 / 2,735 | 4,539 / 4,396 | 5,091 / 5,079 | 5,013 / 5,321 |

Writing, tok/s (greedy / sampled):

| Build | 1K | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| v0.2.9 | 162.8 / 169.3 | 165.9 / 174.9 | 156.2 / 160.1 | 145.0 / 156.8 | 151.8 / 145.1 |
| v0.2.15 | 169.0 / 158.6 | 172.1 / 176.2 | 161.8 / 167.1 | 160.3 / 147.4 | 145.9 / 143.4 |

Same-day check with the standard runner (`strata bench run --url`, 2026-11-16, required stage only): v0.2.9 was
already serving and had answered other requests (not a fresh load, model files not hashed, so not a conforming run
and not submitted): prompt-4k 2,331 tok/s, prompt-16k 3,472 tok/s, decode 175 / 152 tok/s, sanity 12/12. This
folder's fresh v0.2.15 run: prompt-4k 2,692, prompt-16k 3,883, decode 163 / 145, sanity 12/12.

So on this box 0.2.14's own prompt kernels read Gyro-M prompts 6-16 % faster up to 64K (128K about equal); writing is
within run-to-run noise. `STRATA_PF_FUSED_TQ=1` changed nothing measurable here (32K: 4,649 vs 4,649 tok/s, one quick run each).
