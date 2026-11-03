# Prompt experts: the same weights are dequantized once per 64-row block

A measurement report and a proposal. **No code change in this PR**: it is here so the number and
the design are on the record, and so the maintainers can decide whether the CUDA path should take it.

## What the prompt set does today

`src/dsv41/ds4_iq.cu:1595` (gate/up) and `:1614` (down) build the grid as

```cpp
const dim3 grid((unsigned) ((a.max_rows + BM - 1) / BM),   // 64-row blocks
                (unsigned) (a.n_ff / BN), (unsigned) a.n_exp);
```

The work unit is (row-block, column-block, expert), and the fused kernel dequantizes the weights of
its tile from the trellis every time. So an expert with `R` rows has its weights dequantized
`ceil(R/64)` times. At 900 rows/expert that is **15 passes over the same bytes**.

The redundancy is deliberate in one sense - it buys a small working set (no f16 scratch) and it is
exactly right for decode, where `R == 1` and the redundancy is 1 - but the prompt path is the case
where it costs the most.

## What it costs, and what removing it is worth

Measured on an Intel Arc Pro B70 32 GB, DeepSeek-V4.1-Flash IQ2_XXS experts (2.06 bit/weight),
6 experts/word, `n_embd 5120`, `n_ff 2304`, 900 rows/expert, gate/up, 6 experts:

| structure | time |
|---|---:|
| fused (today) | 58,334 us |
| dequantize once into f16, then one GEMM per matrix (library BLAS) | 4,011 us |

Repeating only the dequantize `N` times inside the second path separates the two effects, because
the cost per pass is exactly linear (1,704 us at these shapes):

| N | time |
|---:|---:|
| 1 | 4,011 us |
| 2 | 5,715 us |
| 15 | 27,870 us |

So of the 14.5x, **6.95x is the redundancy going away** (4,011 -> 27,870) and **2.09x is the inner
product moving from a fused F32 dot product to a library f16 GEMM on the tensor cores**
(27,870 -> 58,334). Down is 8.3x.

End to end, a 62,290-token prompt through the service: prefill **375.8 -> 944.1 tok/s** (2.51x),
decode unchanged (`R == 1` keeps the fused kernel), output identical. Numerics improve slightly
(3.34e-04 vs 7.02e-03 relative to a double-precision reference on gate/up, 1.42e-04 vs 4.19e-04 on
down): f16 weights with an f32 accumulator round less than per-value f32.

The 96-row case is the control: `ceil(96/64) = 2`, and the measured speedup drops to 3.7x, which is
the same story with a smaller multiplier.

## Proposal

For row-heavy sets only, dequantize each expert's gate/up/down **once** into an f16 scratch and run
one GEMM per matrix:

```cpp
if (two_phase_on() && a.max_rows >= kTwoPhaseMinRows)   // 96 by measurement, not by taste
    if (set_gate_up_2ph(a, s)) return true;             // falls through on any failure
```

Three things a CUDA implementation must not get wrong (all of them cost me a debug cycle):

1. **The row threshold is not optional.** At `R <= 64` the redundancy is 1-2x and the scratch
   traffic makes it slower; decode must stay fused. 96 is the measured crossover at these shapes.
2. **`SetArgs::bounds` is device memory** (`d_bounds`). A two-phase path is host-side control flow,
   so it has to copy the bounds back to the host first.
3. **`stream == 0` means the default stream** in this codebase. Normalize before using the handle.

On CUDA the plumbing is a little larger than on the port I measured on, because `ds4_iq.cu` has no
cuBLAS handle in `SetArgs` (`ds4_model.cu` keeps one per stream for the dense path); the patch would
thread that handle through `run_set` into the two set entry points. I did not write CUDA for this PR
because I have no CUDA hardware here to run it on, and shipping you untested code seemed worse than
shipping you the measurement.

## Reproduce the measurement

The port I measured on is a SYCL translation of this tree; the numbers above come from
`--bench-tiles` on the same shapes and the same IQ2_XXS blocks, with `DS4_EXP_2PHASE=1` selecting
the two-phase path and `DS4_EXP_2PHASE_DUP=N` repeating the dequantize for the attribution table.
Happy to publish the harness if that is useful.

## What I would like from the maintainers

Either "yes, thread the handle and send the CUDA patch" or "not interested, the fused kernel's
working set is the constraint we care about" - both are useful. If it is the second, the same
redundancy argument still applies to `kTok` in the vendor paths and might be worth a look there.
