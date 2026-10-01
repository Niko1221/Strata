# Context x quant: decode tok/s on an RX 9070 XT (gfx1201, 16 GB)

Full model, Qwen3.8-Flash-Next, ROCm 10.2.0a20260930, engine 0.1.30 + PR #325.
Windows 11, Ryzen 7 7800X3D, 63 GB RAM.

## Generation speed (the headline) — tok/s

Decode only: a short prompt, 160 generated tokens, `reasoning_effort: none`. Every trial
uses a **fresh salted prompt** and is rejected if the server reports any cached prompt
tokens — the server keeps a conversation cache, and repeating a prompt verbatim measures
that cache rather than generation. (The first version of the harness did repeat it, and
reported an impossible 160 tok/s.)

| quant | 32K | 131K | 262K |
|---|---:|---:|---:|
| **IQ2_XS** | **27.5** | **24.0** | **23.2** |
| **IQ3_XXS** | **25.3** | **21.8** | **22.0** |

Expert cache slots at each setting — the mechanism behind the numbers:

| quant | 32K | 131K | 262K |
|---|---:|---:|---:|
| IQ2_XS | 7,546 | 6,396 | 5,043 |
| IQ3_XXS | 6,009 | 5,050 | 3,915 |

## Why the columns are close together

Context length costs VRAM, and on this card the VRAM the KV cache takes comes straight out
of the expert cache — which is what actually makes generation fast. Measured KV
(`strata-plan`, int8): 0.43 GB at 32K, 1.71 GB at 131K, 3.42 GB at 262K.

So 262K costs ~1,100 cached experts (IQ2_XS) or ~2,100 (IQ3_XXS) and buys back only
**~15%** of decode speed. Going 32K → 262K is not the 2x the window size suggests, because
the model is expert-bound, not context-bound: only ~10 of 24,576 experts are used per token,
and the cache hit rate is what sets the pace.

## Why IQ2_XS is faster than IQ3_XXS

The counterintuitive one. IQ3_XXS stores each expert more precisely, so each blob is
bigger — which means **fewer experts fit in the same VRAM** (6,009 vs 7,546 slots at 32K).
The dequant work per token is more CPU, and the hit rate is lower. IQ2_XS wins on speed
because it fits more of the model on the GPU, at some cost in answer quality. That trade is
the whole ballgame on a 16 GB card.

## Measurement noise — read this before quoting a single figure

Per-trial spread is wide, and the first two requests after any reload are much slower
while the path warms:

| setting | min | median | max | n |
|---|---:|---:|---:|---:|
| IQ3_XXS @ 131K | 19.3 | 21.8 | 23.6 | 9 |
| IQ3_XXS @ 262K | **8.3** | 22.0 | 26.4 | 9 |

The 262K row is the reason the table above is not a straight line: with only 3 trials its
median landed at 24.5, above 131K's, which is backwards — 131K has 1,135 more slots and
must be faster. Nine trials show why: two of the nine came in at 8.3 and 8.8 tok/s during
warm-up, dragging the median down. **The 3-trial numbers in the headline table should be
read as ±2 tok/s, and the 131K/262K ordering is within that noise.** The 32K column is
solid; the 131K vs 262K column is not, and the honest statement is that they are the same.

## Method

- `bench/cell_bench.py <quant> <context> <kv>` — reloads at the setting, waits for
  `/health`, 3 generations, prints `RESULT`.
- `bench/repeat_bench.py <label> <n>` — n trials, all printed, for the noise check.
- `bench/context_bench.py <port> <label> <ctx>` — decode plus cold-prompt prefill.
- Each cell is a full model reload (~1-2 min); the six-cell matrix is ~15 min.
