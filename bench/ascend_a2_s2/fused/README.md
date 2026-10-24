# Ascend A2 fused S2 projection experiment

This standalone kernel computes one S2 matrix projection with already quantized
activations. It does not implement a complete expert or an inference backend.
It is built separately from Strata. Measurements and rejected attempts are in
[RESULTS.md](RESULTS.md).

The [A3 validation report](../a3/RESULTS.md) records independent builds and
execution of these unchanged sources on `Ascend910_9362`, including comparison
with the A2 output hashes. Use its explicit SoC override for that target.

## Contract

Inputs are resident device buffers:

- Weight codes: `M * (K / 64) * 16` bytes, row-major canonical S2, LSB-first.
- Weight scales: `M * (K / 64)` FP32 values, widened from the Q2_0 FP16 scales.
- Activations: `T * K` INT8 codes and `T * (K / 32)` FP32 scales.
- Output: `T * round_up(M, 8)` FP32 values; tail rows are zero padding.

`K` must be a positive multiple of 64, at most 2560. `T` is 1 through 8.
The supplied fixtures use `M` from zero through 2560. Pointers must have the
stated storage and alignment; this is a benchmark entry point, not a general
checked tensor interface. The wrapper rejects invalid `K` or `T` without
submitting a kernel and treats zero rows as a no-op.

For every 64-weight block, form two independent integer dots:

```text
dot[h] = sum((code[j] - 1) * q[j]), j in half h of the block
block  = activation_scale[2*b] * dot[0]
       + activation_scale[2*b+1] * dot[1]
result += weight_scale[b] * block
```

The fixture transcribes the scalar legacy quantizer in
`src/kernels/cpu/expert.cpp`: 32-element chunks, FP32 scales, half-away rounding
and clamping to [-127, 127]. Known ties and zero chunks are checked separately.
This does not execute the x86 AVX implementation on aarch64. It also does not
select Strata's experimental FP16-scale `oracle_q8_0` activation mode.

The block reduction order differs from the default AVX eight-lane reduction.
A passing result is numerical agreement for this projection contract, not
bitwise agreement with the default engine or model output.

## Kernel and oracle

Each core copies the INT8 activations into UB, then widens them through half
to FP32. Both conversions are exact for INT8 values. A row's unscaled S2 codes
are unpacked once and reused across tokens. Vector multiplication and
`WholeReduceSum` compute each 32-element dot. These products and sums remain
exact FP32 integers: their magnitude is at most 8128 for the supplied codes.
Two activation scales and the weight scale are applied afterward in FP32.

The kernel never writes a full decoded FP32 matrix to device global memory.
The largest matrix in the experiment would occupy 6,553,600 bytes in that
form. This is an allocation calculation, not a measured HBM traffic saving.
The vector kernel requests 164,192 bytes in its fixed TBuf allocations per
active core, including activation conversions, row scratch and output scratch.
It still uses scalar packed-code loads and unpacking; it is not fully tiled or
double buffered.

The independent oracle calls the existing C++ Q2_0 decoder from the parent
experiment, then multiplies decoded weights and reconstructed quantized
activations in FP64 on the host. No new scalar weight decoder is used as the
oracle. Accuracy uses a cancellation-aware forward bound:
`gamma(4*K/64 + 4) * sum(abs(weight * activation))`, with FP32 unit roundoff
`u=2^-24` and `gamma(n)=n*u/(1-n*u)`. Fixtures avoid FP32 overflow and underflow.
The report includes absolute error and the fraction of that bound actually used.

## Reproduce

Build the parent decoder/reference first as described in its README. Use the
same A2 SDK environment and fresh directories for new build configurations.

```sh
cmake -S "$POC_ROOT/bench/ascend_a2_s2/fused" \
  -B "$POC_ROOT/bench/ascend_a2_s2/fused/build-release" \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build "$POC_ROOT/bench/ascend_a2_s2/fused/build-release" -j2
OPENBLAS_NUM_THREADS=1 python3 "$POC_ROOT/bench/ascend_a2_s2/fused/probe.py" \
  --build "$POC_ROOT/bench/ascend_a2_s2/fused/build-release" \
  --decode-build "$POC_ROOT/bench/ascend_a2_s2/build-release" \
  --report "$POC_ROOT/new-projection-result.json" --benchmark
```

The split baseline decodes the same packed matrix on the NPU, synchronizes at
the raw ACL/framework boundary, then runs FP32 `torch.mm` with HF32 disabled.
Its activations are reconstructed and resident before timing. The fused path
receives resident INT8 codes and scales. Quantization, input preparation, H2D,
allocation, and CPU oracle execution are excluded from both compute timings.
The fused time includes its on-device activation copy and casts. Split timing
includes an extra interop synchronization; it is not an optimized native
two-kernel pipeline. See the official [HF32 setting documentation](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu-matmul-allow_hf32.md).

Two warm-up rounds precede five measured rounds, three calls per round. Order
alternates. Host wall times include dispatch and final synchronization; they
are not device-event timings. Reports retain individual rounds, source and
binary hashes, guard checks, unchanged-input checks, and output hashes. Existing
report files are never overwritten.

## Limits

The separate [real-weight MoE probe](../moe/README.md) tests composition,
intermediate quantization, routing and all 512 experts on A3. It found a
rounding-boundary failure in the original accumulation. For that experiment,
`-DSTRATA_S2_COMPENSATED=ON` compensates product and sum rounding across the
32-element chunk contributions. It also disables fast-math reassociation and FP contraction for this
target. CANN's `ascendc_compile_*` functions pass those settings to the device
compiler; ordinary `target_compile_*` settings on the final library did not.
The option defaults to OFF. The original measurements in `RESULTS.md` describe OFF;
the [MoE report](../moe/RESULTS.md) records the separate A3 comparison. The ON
path has no A2 execution or performance measurements.

The ON path expects FP16-derived weight scales, finite normal products and
no overflow in the `4097*x` splitting steps. Multiplying a weight scale by
the exact integer chunk dot is exact in FP32 (at most 11 + 13 significant
bits). Dekker splitting retains the next product's low part; TwoSum retains
the addition's low part. The accumulated correction is applied at the store.
See [Ogita, Rump and Oishi, 2005](https://doi.org/10.1137/030601818) for the
error-free transformations. This is not a correctly-rounded dot-product
guarantee for arbitrary inputs. `compensated_regression.py` checks two known
cancellation results, including one lost by product rounding alone.

Synthetic weights and host activation quantization only. SwiGLU, intermediate
requantization, down-after-gate/up composition, expert routing, cache transfers,
and full-model generation are untested. NaN and infinity scales are untested.
The canonical FP32 scale planes are used instead of directly consuming the
legacy interleaved expert blob. A3 and other CANN versions need their own
build and execution. No CUDA, HIP, SYCL or Metal file is changed.
