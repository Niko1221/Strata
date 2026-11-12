# Opt-in reference-order CPU mixed-IQ down rows (experimental)

The CPU native-down dispatcher can reuse the existing reference-order
`iq256x_rows` API for IQ2_S gate/up with IQ2_S or IQ3_S down. Set
`STRATA_IQ_DOWN_EXACT=1` **before process startup** to enable it. Unset, `0`, or
any other value keeps the existing CPU path. `STRATA_NO_IQ256` disables it and
`STRATA_IQ_MT_MIN` still sets the lower group-size threshold. This adds dispatch,
not new dot-product arithmetic. Upstream's legacy multi-token rounding is
intentional; this is not an arithmetic bugfix.

## Scope and guards

The opt-in path is deliberately narrow: 2048 hidden values, 512 intermediate
values, 2–8 tokens, Q8_K activations, the original contiguous blob geometry, no
scale tail and no Hadamard rotations. It requires both runtime AVX2 without
AVX-512 and a GGML reference compiled for AVX2 without AVX-512. A runtime ISA cap
cannot replace the compiled-reference check. Other layouts keep the existing
fallback. The default CPU path is unchanged.

This contribution is **CPU-only**. It does not register new CUDA/HIP down formats,
change device arithmetic or device fixtures, or add SYCL/Metal support,
auxiliary-weight loading, or a prefill algorithm. No model-support or speedup
claim is made. Full-engine/model decode, prefill, rotated-model fallback and
performance are not qualified by these synthetic tests. The opt-in switch is
experimental; enabling it is not a promise of end-to-end token identity.

## CPU regression checks

On Linux with GCC/G++, an existing llama.cpp checkout, and AVX2/FMA/F16C:

```sh
python3 tools/test_iq_down_dispatch.py --llama-dir /path/to/llama.cpp --output /fresh/dispatch
python3 tools/test_iq_down_exact.py --llama-dir /path/to/llama.cpp --output /fresh/exact
```

Use a fresh output directory for each invocation. The runners do not download
dependencies or access a GPU. They write compiler commands and results to
`receipt.json`. `CC` and `CXX` can select compatible GCC drivers.

- Dispatch: the real baseline-ISA `native_expert.cpp` with instrumented row
  backends. Checks selection, argument forwarding, switches, compiled/runtime ISA
  mismatch, geometry, rotation and existing fallbacks. This is not arithmetic.
- Exact: separately compiles the real IQ and native translation units and GGML's
  AVX2 dots. Compares public `iq256x_rows` and `native_down_rows` bitwise against
  the reference for both formats, scalar/gather decode, widths 1–8, split and empty
  row ranges, padding and unchanged inputs/inactive outputs. CPU/reference probes
  and unrelated backends are stand-ins; this is not full-engine integration.

The CPU run used GCC/G++ 15.2.0 and llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`: 10,368 public exact-row comparisons and
1,179,648 native comparisons each with the switch off and on, all bitwise equal,
with no input/output-preservation failures. The routing check passed 64 cases in
each of six fresh-process modes. These are synthetic CPU results, not model
accuracy or performance measurements. Project-only ASan+UBSan checks also passed;
instrumenting the external GGML reference additionally reported signed-shift UB
in unchanged `arch/x86/quants.c:3452,3457`, so the wider run is not sanitizer-clean.

## Why the device registration was removed

An earlier draft also registered IQ3_S/IQ2_S down on CUDA/HIP. Its focused CUDA
Q6_K OFF build linked, but an independent down-only numerical fixture failed its
frozen `2e-5 + 2e-5 * abs(reference)` criterion. The first IQ3_S counterexample was:

```text
seed=1, grouped-v1=true, token=2, row=963, hidden/intermediate=2048/512
observed device = -1.79437255859375
GGML oracle     = -1.794281005859375
absolute error  = 9.1552734375e-05
allowed error   = 5.588562011718751e-05
```

A CPU reconstruction of those retained packed bytes reproduces the device value
exactly using the existing upstream `Split<21>` lane-product rounding and
`warp_sum` XOR reduction (offsets 16, 8, 4, 2, 1). The sum of rounded lane values
is `-1.7942733764648438`; the final FP32 tree produces the observed device value.
The relevant arithmetic functions were byte-identical to the upstream base.
This explains this counterexample without a packing/layout defect or a new
arithmetic regression. It does **not** qualify every device path. The original
failed gate remains failed; no tolerance was relaxed or device success invented.

Rather than change established device arithmetic or introduce an unreviewed
replacement numerical contract, this PR removes all new device reachability and
restores the device source and grouped fixture byte-for-byte to its upstream
base. CUDA Q6_K ON, remaining independent device cases, grouped execution and
HIP remain unqualified and are outside this narrowed contribution. A future
device-support PR needs its own reviewed numerical contract and actual backend
qualification. No new CUDA/model run was used to justify this scope reduction.

## Origin

Developed as part of [MoEmentum low-VRAM inference experiments](https://github.com/CrashxZ/MoEmentum).
The original down-row work is in
[`bc7a84d1`](https://github.com/CrashxZ/MoEmentum/commit/bc7a84d14a663d8fe63a56fd2695038f8018b36c);
the later compiled-reference/rotation/geometry guards are in
[`67bb9c51`](https://github.com/CrashxZ/MoEmentum/commit/67bb9c5157b53b8b1b9ff8075bcf939223ae9f1e).
This is a focused port of that dispatch work to the current upstream API,
not a claim to have introduced upstream's existing exact arithmetic. Historical
benchmark and full-engine qualification claims are not transferred to this port.
