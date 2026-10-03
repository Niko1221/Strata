# 2026-10-02: the V100 skinny-expert path - load-time QPN8 repack + m8n8k4 vs the in-kernel unpack

Device: Tesla V100-SXM2-16GB, CUDA 12.8, driver 570.  48 blobs (13.8 MB per call) cycled past the 6 MB L2 so
the codes come from DRAM as the engine's do.  CUDA events, 200 (ablation) or 20 (entry points) reps.  Numbers
move ~10% run to run (the two sweeps below disagree at that level); the SHAPE of every table is stable.

Question: `src/kernels/cuda/s2_expert_grouped.cu` unpacks the Q2_0 codes inside its M=2..8 GEMV
(`expand_codes`, 2-bit fields -> one byte per element).  Can a load-time rearrangement of the weights
(keep the byte count) remove that unpack, and what is it worth?

## 1. Where the time goes (`bench/micro/s2_unpack_bench.cu`, inner loop only, 10 experts, all M entries shared)

Each variant replaces one piece of the shipped inner loop with zeros (or a different form) and times
gate/up + down for one window call.  `M` is the entries per expert of the window.

| variant | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|
| 0 base (shipped unpack) | 30.7 | 33.1 | 46.8 | 76.1 |
| 1 no-expand (unpack removed) | 15.1 | 16.6 | 25.0 | 41.0 |
| 2 no-xload | 29.7 | 31.8 | 44.3 | 72.0 |
| 3 no-dp4a | 13.4 | 14.9 | 22.3 | 36.4 |
| 4 no-sum | 9.5 | 10.3 | 14.7 | 23.1 |
| 5 clean (codes pre-expanded, 4x bytes) | 54.2 | 53.9 | 67.5 | 96.8 |
| 6 chains2 | 30.7 | 33.0 | 46.4 | 75.6 |
| 7 chains4 | 30.8 | 33.2 | 47.5 | 77.0 |
| 8 hmma s16/8 (QPN8 repack) | 50.2 | 43.1 | 43.0 | 43.0 |
| 9 hmma gather (no repack) | 164.0 | 152.3 | 153.4 | 156.8 |
| 10 hmma s8/4 | 46.7 | 46.3 | 46.2 | 46.7 |
| 11 hmma s4/4 | 45.0 | 44.5 | 44.4 | 44.8 |
| 12 hmma s8/4 n2 | 43.1 | 43.0 | 42.9 | 45.1 |
| 13 s2t+lut dp4a (bit-transposed codes + LUT) | 47.9 | 53.5 | 68.4 | 95.9 |

What this says:

* The unpack is real money once entries share an expert: 35 us of the 76.1 us call at M=8 (base minus
  no-expand).  At M=1 it is 15.6 of 30.7 - smaller, and the shipped M=1 hit path keeps its kernels.
* An unpack-free DP4A form at CONSTANT BYTES does not exist - a dp4a operand is one byte per element, so
  the 2-bit codes must expand 4x somewhere.  The same-bytes attempts all lose to the shipped unpack:
  pre-expanded codes pay the 4x DRAM (96.8 vs 76.1 at M=8), the bit-transposed + LUT form pays shared
  loads (95.9), wider dp4a chains are neutral (75.6 / 77.0).
* The QPN8 repack (permutation, same bytes) + m8n8k4 with the 1024/1025 bias trick is the one same-bytes
  form that removes the unpack: 43.0 us at M=8 (1.77x over base), flat 43-50 us across M.  The crossover
  is M >= 4; at M <= 2 the m8 tile's unused rows are pure loss (50.2 vs 30.7 at M=1).
* Variant 9 gathers the same records from the canonical layout: 152-164 us.  The repack IS the coalescing.

## 2. The shipped entry points (`s2_qpn8_parity --bench`, full pipeline: gate/up -> swiglu -> quantize -> down)

| shape (all entries shared) | unpack DP4A | repack m8n8k4 | ratio |
|---|---|---|---|
| 10 experts x 1 entry | 50.6 | 100.2 | 0.51x |
| 10 experts x 2 entries | 56.6 | 102.9 | 0.55x |
| 10 experts x 4 entries | 74.0 | 109.1 | 0.68x |
| 10 experts x 8 entries | 109.5 | 135.0 | 0.81x |
| window 16e: 16 x 1 | 65.1 | 148.6 | 0.44x |
| window 16e: 8 x 2 | 53.1 | 96.1 | 0.55x |
| window 16e: 4 x 4 | 46.5 | 69.8 | 0.67x |
| window 16e: 2 x 8 | 52.9 | 66.3 | 0.80x |

**The integrated path does NOT yet cash the inner-loop win.**  The gap to the 43 us of table 1 is the
per-group instruction cost the inner-loop benchmark does not pay: the int8 activation -> A-fragment
conversion (the reference loads f16), the per-chunk `dw * dx * s` readout that keeps the DP4A kernels'
float expression bit-exact (the reference accumulates plain f32 and scales in-fragment), and the q8_0
block IO.  So the tensor path stays OFF (`STRATA_QPN8=1` opts in on cc 7.0) and the DP4A kernels keep
every default; the repack, the cache-admission hook and the kernels are the durable part of the work.

## 3. Numerics

Per chunk the QPN8 terms are bit-identical to the DP4A kernels' (the 1025 bias trick gives the exact
integer `s - hx`, every product and chunk sum is an integer below 2^24, exact in f32; the readout is the
same `dw * dx * (float)(s - hx)`).  Only the order the chunk terms are ADDED differs (lane-strided
shuffle tree vs k order through the split-K reduce).  Measured (`s2_qpn8_parity --selftest`, 45 entries
in groups of 1..8, both activation contracts): L1 relative 7e-8..1.6e-7, worst absolute 9.5e-6, and the
two pipelines' quantized intermediates are BYTE-identical.  Float-order level - not the 4.761e-04 of an
f16 scale rounding.

## 4. Bugs this cost (kept in the test code as comments)

* `__byte_perm` selects by NIBBLE (result byte i = source byte `(sel >> 4*i) & 7`); the A-fragment
  converter put its second selector at `<< 16` and every half2 came out `(pair, first byte)`.
* The down kernel's code pointer missed the `O_D_CODES` plane offset and read the gate/up records.
* The parity's `ent_dst` mapped 45 entries onto 32 destinations: both kernels ASSIGN `out[dst]`, so the
  comparison measured the scheduler (nondeterministic counts across runs).  The engines' plans give every
  entry its own row; the test now shuffles a permutation.

Reproduce: `ninja -C build strata s2_qpn8_parity s2_expert_grouped_parity`, then
`STRATA_QPN8=1 build/s2_qpn8_parity --selftest` and `--bench`; the ablation builds standalone:
`nvcc -arch=sm_70 -O3 -o s2_unpack_bench bench/micro/s2_unpack_bench.cu`.
