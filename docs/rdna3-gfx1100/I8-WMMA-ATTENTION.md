# [Benchmark] int8-WMMA prompt attention on RDNA3 / gfx1100 (RX 7900 XTX): recipe, validation, results — and the RDNA4 (gfx12) relevance

Companion to the gfx1100 measurement report. This one documents the int8-WMMA prompt-attention
kernel (int8 Q·K and P·V through `V_WMMA_I32_16X16X16_I8` with exact i32 accumulation, fp32 online
softmax) implemented for gfx1100, the fragment-layout recipe the hardware probe established, the
numerics gates it passed, and where it does and does not pay off. All numbers are from one machine
(RX 7900 XTX, 24 GiB, the pinned nightly SDK), the Qwen3.8-Flash-Next IQ3_S artifact, `--kv int8`,
paired in-session instruments (the same binary both arms unless stated).

## The WMMA i8 recipe (probe-proven on gfx1100)

- `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(NEG_A, a, NEG_B, b, c, NEG_C)` with natural packing:
  register `t`, byte `b` = k `4t+b`; each lane holds its row of A (16 contiguous k values) / its
  column of B; lanes 16–31 repeat lanes 0–15.
- C/D fragment: lane `l`, index `i` → `D[2i + l/16][l%16]` (the same row mapping the f16 WMMA uses
  on gfx11: `PA_ROW(i, half)`).
- `(NEG_A, NEG_B) = (true, true)` is the signed emulation on the u8 byte containers: exact s8×s8
  at full magnitude. It is also exact for u8(0..127)×s8 (P's quantization range), because bytes
  < 128 are signed-identical.
- The accumulator (`c`) operand adds as-is even under the neg flags, and passes through at A=B=0
  (probe: `c=0` 256/256, `c=C1` → 2×CPU 256/256, `c=C1` at A=B=0 → c 256/256). Chaining k-steps
  through `c` is safe.
- A note for anyone re-deriving the packing: a probe that permutes BOTH operands' k-order
  identically (as natural-vs-W11 sweeps do) cannot see the intra-operand k-order — a shared
  k-permutation cancels in the dot. Both packings read 256/256; the layout, flags and signed
  semantics are what the probe actually establishes. Pack A and B with the SAME convention and the
  product is exact.

## The kernel (prompt attention, both products int8)

- Q·K: Q normalized by its max to the 16384 range (`qup = 2^(14−qe)`), carried as a hi/lo int8
  pair (`hi = rint(v/256)`, `lo = rint(v − 256·hi)`, two chained i32 WMMA dots — the same
  two-image staging the W11 MoE kernel uses); K codes feed the WMMA directly as raw bytes (the
  int8→fp16 conversion disappears). Epilogue `(256·shi + slo) × k_scale` — the codes already carry
  the `qup` factor the fp16 path's accumulator carries; the softmax's `qdown` cancels it. (Getting
  this wrong doubles the division and flattens the softmax: every score shrinks by `2^(14−qe)`,
  which reads as a plausible-but-wrong 65% relative RMS. The scale bookkeeping, not the
  quantization, is where this kernel bites.)
- P·V: P quantized u8 on the V-normalized range (`rint(pv · 127/16384)`, values 0..127), V codes
  raw s8, one i32 WMMA per 16-cell step, epilogue `× (16384/127) × v_down × alpha`.
- int8 products stay below 2³¹ everywhere (|Q codes| ≤ 64/128, |K| ≤ 128, 16 k per step), so the
  i32 accumulator is exact and no fp32 magic-constant accumulation is needed.

## Validation on the real replay (one recorded 94K-depth selection: T=28,280, cap=2,051, 24/2 heads × 256)

- vs the deployed f16-WMMA kernel (`STRATA_HIP_WMMA=1`, the hi+lo FP16-images class), paired
  interleaved sessions: **−32.1% kernel time** (69.0 vs 101.5 ms median). The f16 kernel here does
  two WMMAs per k-step (hi+lo); the i8 kernel also does two for Q·K but one for P·V instead of two,
  and skips the int8→fp16 conversions.
- Output agreement vs the f16 kernel on the same replay: relative RMS 1.9e-2 (Q·K alone: 6.5e-5 —
  the rounding lives in P's 1/127), worst |diff| 1.5e-2, output rms matches to 7 digits.
- First-token KL vs the deployed f16-WMMA path, four real 31.5K-token prompts served twice
  (`max_tokens 1`, top-20 logprobs, KL over the union support, natural log): mean 0.0036, with the
  FP32-kernel swap floor measured at 0.0005 on the same prompts and the same instrument; argmax
  agreed on every prompt.
- A predeclared 2.00%-on-the-median component bar vs the precision-class baseline passes 16×.

## Where it does not pay off

The deployed fast configuration on this card runs the prompt attention with a SINGLE fp16 image
per k-step (one WMMA per step, FP16-rounding accuracy). Against that arm the i8 kernel is **3.8%
slower** on the same replay (69.3 vs 66.8 ms): its Q·K needs two WMMAs for the hi/lo pair while
the fast fp16 path needs one, and the P·V saving nets out. End-to-end (94K-token prefill, the full
engine, five interleaved pairs, fresh launch per launch): median prompt throughput ratio 0.995,
bootstrap 95% CI [0.994, 1.004] — no end-to-end win, small regression. The i8 kernel is the right
tool where the two-image precision class is wanted (cards/configs that did not adopt the
single-image fast path): there it is a straight 32% kernel win at FP32-level Q·K accuracy.

## Why this matters for RDNA4 (gfx12)

RDNA4 has the same `V_WMMA_I32_16X16X16_IU8` instruction family, and upstream's prompt attention
already defaults to the f16 WMMA there — so an int8 tier meets its audience on gfx12 by default,
where on gfx11 the whole WMMA path is opt-in. On gfx11 the int8 and fp16 WMMAs issue at the same
rate, and the win measured here is structural (the int8 path drops the second fp16 image pair and
the int8→fp16 conversions), not instruction-rate — the same structural analysis carries over.

What a gfx12 port needs: the fragment layout differs (PA_KL=8 there, and the f16 WMMA uses the
`_gfx12` builtin variant), so the i8 builtin's gfx12 operand/fragment layout must be re-proven on
the card before the kernel's gfx11-only assumptions are touched — the kernel is `static_assert`
-gated to gfx11 so nothing changes for gfx12 until then. The layout probe is ~130 lines and ships
with the code PR on scope agreement; on an R9700 it re-establishes the gfx12 recipe in one run,
and the same two-step follow-up (replay accuracy, then paired timing) is minutes per step. Until
then this report makes no gfx12 performance claim — the gfx11 recipe and its validation method are
the portable part.

## Artifacts

- `i8-wmma-probe-20261108.txt` — the capability+layout probe output (packings × flags × layouts,
  the P·V check, the accumulator checks).
- `i8-wmma-paired-20261108.txt` — the paired interleaved sessions (raw rows, both classes).
- `i8-wmma-kl-20261108.json` — the per-prompt first-token KL rows (floor and candidate).
- `i8-wmma-window-20261108.json` — the end-to-end paired launch rows and the bootstrap CI.
