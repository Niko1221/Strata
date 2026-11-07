# RDNA3 / gfx1100 (RX 7900 XTX): deep-prefill and expert-kernel measurements on Qwen3.8-Flash-Next IQ3_S

Results-only report (no engine changes). Machine: Ryzen 9 7900X, RX 7900 XTX 24 GB, 96 GB RAM,
Fedora, HIP SDK 7.17.26392 (pinned nightly), engine 0.1.55 (v0.2.9 + 10 locally replayed fork
commits, base c62d5e3a1, measured tree b706bc808), binary sha256 125759e9…. Model: the user's
GSQ/RCO IQ3_S two-shard artifact (experts IQ3_S gate/up, IQ4_NL down, 48 layers, H 2560, FF 640).

Method: every timing comparison is either in-process interleaved (bench medians over 12 runs) or
paired alternating launches with a bootstrap CI (seed 20261107); correctness gates ran before
timing (bitwise where the mechanism claims bitwise). A cross-process sequential sweep was
discarded after the paired instrument reversed its sign - the raw sweep is preserved alongside the
paired data for exactly that reason.

## Findings

1. Deep-prefill stack (deployment-level, bit-changing where noted): `STRATA_PA_FAST=1` +5.43%
   (CI95 [1.0475, 1.0606]); `STRATA_PF_FUSED=1`+`STRATA_PREFILL_RING=512` +9.08% (CI95
   [1.0874, 1.0958]); combined +13.82% (CI95 [1.1028, 1.1460]) over `--prefill auto:32768` at 94K
   depth. Deployed on this card as user-level configuration (the table below stays
   bit-identical-only).
2. Defaults-table candidate: `STRATA_TSUM=1` (decode mmvq transposed butterfly sums) -
   bitwise-equal, paired decode_ms -0.91% (CI95 [0.9847, 0.9928], 15 request-pairs, excludes
   1.00). Proposed as a gfx1100 row in `arch_default_env` (issue filed separately per the
   defaults scope rule).
3. Negative results (all bitwise-clean, all slower on gfx1100): expert layout modes 2/4 (3.0x
   slower), mode 8 (2.1x slower all / down 53% slower); S26 `STRATA_EXPERT_V2=1` 1.31x slower
   (gu 2.1x, down 3.7x) and +TSUM 1.11x; `STRATA_EXPDEC=1` 1.09x; the mmvq upstream multi-column
   layout 17-22% slower than the exact layout at ncols 8 (the reverse of NVIDIA - per-arch
   dispatch is earning its keep). The gfx1151 mode-8 default does not transfer to gfx1100.
4. Bandwidth context: the dense draft-head GEMV (2560x248320) reaches 890.7 GB/s (~93% of peak) in
   this kernel family; the mid-shape multi-column regime plateaus at 195-360 GB/s, which the knob
   sweep could not improve - the shape class, not a knob, bounds it.

Raw data: the four txt files in this directory (ceiling table, the discarded sweep + the paired
reversal, the expert mode/env arms). Sanitized configuration: `CONFIG.md`.
