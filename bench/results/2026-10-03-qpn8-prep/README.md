# 2026-10-03: the QPN8 entry point cashes the win - the per-window A fragment + mma chains

Device: Tesla V100-SXM2-16GB, CUDA 12.8, driver 570.  Same harness as
`bench/results/2026-10-02-skinny-expert/`: `s2_qpn8_parity --bench`, full pipeline (gate/up -> swiglu ->
quantize -> down) over 48 blobs cycled past the 6 MB L2, CUDA events, 20 reps.  Run-to-run spread ~3%.

## The problem after the first commit

The integrated path was 0.81x at 10 experts x 8 entries even though the m8n8k4 inner loop alone was 1.77x
(43.0 vs 76.1 us).  An ablation of `gu_qpn8_kernel` (`STRATA_QPN8_ABL` bits, since removed) split its 74 us:

| piece turned off | gu us |
|---|---|
| none (baseline) | 74.0 |
| A int8 -> f16 conversion | 72.0 (A convert is only 2 us) |
| B record decode | 64.6 |
| per-chunk readout | 64.8 |
| A LOAD + convert (both) | 45.8 |

So the activation load - not the conversion - was 28 us.  `memcpy` of the q8_0 body (2-byte aligned, 34-byte
blocks) scalarised into byte loads, and every one of the 40 row-tiles re-read AND re-converted the same 16
bytes per (entry, group).  Two fixes:

1. `load_a16`: five aligned 32-bit `__ldg` + funnel shifts instead of the byte-load `memcpy`.
2. `prep_a_kernel`: once per window, one thread per (entry, k16 group) converts the int8 to the exact 32-byte
   A fragment the mma wants; gu/down then read two `uint4` per group.  The 40-tile redo is gone.
   A chunk's two k16 groups also issue into two mma accumulator chains (the eight dependent HMMA of a chunk
   were the pipeline's critical path), worth ~4 us on gu.

## The shipped entry points (`s2_qpn8_parity --bench`)

| shape (all entries shared) | unpack DP4A | repack m8n8k4 | ratio |
|---|---|---|---|
| 10 experts x 1 entry | 50.3 | 76.4 | 0.66x |
| 10 experts x 2 entries | 56.3 | 77.5 | 0.73x |
| 10 experts x 4 entries | 74.5 | 82.2 | 0.91x |
| 10 experts x 8 entries | 109.3 | 90.9 | **1.20x** |
| window 16e: 16 x 1 | 64.3 | 106.4 | 0.60x |
| window 16e: 8 x 2 | 52.6 | 70.6 | 0.75x |
| window 16e: 4 x 4 | 46.0 | 57.3 | 0.80x |
| window 16e: 2 x 8 | 52.7 | 52.4 | 1.01x |

The target shape (10 experts x 8 entries) is now 1.19-1.22x over three runs - past the 15% bar.  The crossover
is FOUR OR MORE entries per expert; at 1-2 entries the m8 tile's unused rows lose, so the knob stays OFF by
default (`STRATA_QPN8=1`, cc 7.0) and the DP4A kernels keep every default.  M=1 is unchanged by design (the
single-token hit path is per-expert).

## Numerics

`s2_qpn8_parity --selftest`: PASS on both activation contracts.  The pipelines' quantized intermediates are
BYTE-identical; worst absolute 9.5e-6, L1 relative 1.6e-7 - the same float-order bound as before (the prep
pass and the chunk-sum reorder change no per-chunk term).  `s2_expert_grouped_parity --selftest`: PASS.

Reproduce: `ninja -C build strata s2_qpn8_parity`, then `STRATA_QPN8=1 build/s2_qpn8_parity --selftest` and
`--bench` (`STRATA_QPN8_BENCH_SHAPE="10 experts x 8"` narrows the sweep).
