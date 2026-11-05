# `cann/` - the Ascend port of the engine

This directory is to the CANN backend what `sycl/` is to the Intel backend: a separate project
that grows next to the CUDA engine until it can take the engine's sources.

| phase | scope | state |
|---|---|---|
| P1 | device, stream, HBM buffers, first aclnn ops, CPU oracle | done, parity on hardware |
| P2 | Q4_K / Q6_K decoders, dequantize then matmul on the NPU | done, bit-exact vs ggml |
| P3 | attention: GDN (done), RoPE / KV cache / softmax | in progress |
| P4 | MoE routing and expert matmuls | not started |
| P5 | `serve/` wiring, 18789, benchmarks | not started |

## Files

| file | what |
|---|---|
| `include/strata/cann/device.hpp` | `Device`, `Stream`, `Buffer`, `DeviceInfo`, `Error`; one place that turns `aclError` into an exception |
| `include/strata/cann/ops.hpp` | `matmul`, `rms_norm`, `silu`, `add`, `softmax` on fp16 device tensors, each with a CPU oracle |
| `include/strata/cann/quants.hpp` | `block_q4_K` (144 B) and `block_q6_K` (210 B) plus `dequantize()` |
| `include/strata/cann/gdn.hpp` | gated delta net: float reference and the device path |
| `src/device.cpp` | `aclInit` / `aclrtSetDevice` / `aclrtCreateStream` / `aclrtMalloc` / `aclrtMemcpy` |
| `src/ops.cpp` | the aclnn two-phase call (`*GetWorkspaceSize` then the op) behind one helper |
| `src/quants.cpp` | the two decoders, written against ggml's `dequantize_row_q4_K` / `_q6_K` |
| `tests/parity.cpp` | random fp16 inputs, device result against the CPU reference |
| `tests/quant_parity.cpp` | decoders against ggml's own `to_float`, then a dequantized matmul on the NPU |
| `tests/run_parity.sh` | build and run `cann_parity` |
| `tests/run_quant.sh` | build and run `quant_parity` (needs the llama.cpp checkout) |
| `tests/gdn_parity.cpp` | gated delta net against the float reference, both decay modes |
| `tests/run_gdn.sh` | build and run `gdn_parity` |

## Build and run

```bash
# P1 only
cmake -S cann -B build-cann && cmake --build build-cann -j
./build-cann/cann_parity

# P1 + P2, with ggml's decoders as the oracle
cmake -S cann -B build-cann \
  -DSTRATA_CANN_GGML_DIR=/path/to/llama.cpp \
  -DSTRATA_CANN_GGML_LIB_DIR=/path/to/llama.cpp/build/bin
cmake --build build-cann -j
./build-cann/quant_parity
```

The toolkit comes from `ASCEND_HOME_PATH` (default `/usr/local/Ascend/cann-9.0.0`). The ggml
oracle links `libggml-base.so` only for the test; nothing in the port depends on ggml.

## Tolerances

The parity test rounds inputs through fp16 before the CPU oracle runs, because that is what the
device receives.

| op | shape | tolerance |
|---|---|---|
| matmul | 256 x 512 x 384 | 0.5 (512-term fp16 dot) |
| rms_norm | 512 x 2048 | 0.02 |
| silu | 4096 | 0.02 |
| add | 4096 | 0.01 |
| softmax | 64 x 1024 | 0.01 |
| Q4_K dequant | any | bit-exact against ggml |
| Q6_K dequant | any | bit-exact against ggml |
| GDN, kda S=64 T=8 | one head | attn 0.05, state 0.05 |
| GDN, scalar S=64 T=8 | one head | attn 0.05, state 0.05 |

## Two API facts worth writing down

- aclnn tensor descriptors are **row-major**: `shape = {rows, cols}`, `stride = {cols, 1}`.
  Handing it a ggml-style column-major descriptor (`{cols, rows}` / `{1, cols}`) fails with
  `161002` on the first matmul.
- `aclnnRmsNorm` takes an extra `rstdOut` tensor (the reciprocal std per row) that the header
  comment lists only briefly, and `gamma` must be 1-D. A 2-D `{1, C}` gamma fails with `561002`.

## What is not here yet

- The engine's own kernels: `src/` still compiles for CUDA only. P3 is the first place a kernel
  has to run behind the engine's interface instead of behind this test.
- `ggml-cann` still has no `GGML_OP_GATED_DELTA_NET`, so the llama.cpp server keeps those 3 of 4
  layers on the CPU. This directory now has a working device path for it, but wiring it into
  `ggml-cann` is a separate change in that backend.
- Fused dequant+matmul. P2 proves the numbers; it dequantizes on the host and multiplies fp16 on
  the device, which is correct but not the shape a fast engine needs.

## Verified on the machine

`Ascend910B2C`, CANN 9.0.0, driver 26.0.rc1, 2026-11-05:

```
device: Ascend910B2C, HBM 61.0 GiB total / 58.0 GiB free

op                          max|diff|   mean|diff|  tolerance  result
matmul 256x512x384           0.002529     0.000266     0.5000  PASS
rms_norm 512x2048            0.000977     0.000152     0.0200  PASS
silu 4096                    0.001947     0.000319     0.0200  PASS
add 4096                     0.000977     0.000209     0.0100  PASS
softmax 64x1024              0.000004     0.000000     0.0100  PASS
```

```
Q4_K   vs ggml: max|diff| = 0  BIT-EXACT
Q6_K   vs ggml: max|diff| = 0  BIT-EXACT
Q4_K dequant -> NPU matmul 64x512x256: max|diff| vs CPU = 0.013428  PASS
```

```
case                 attn|diff|  state|diff|  result
kda    S=64  T=8       0.000360     0.003886  PASS
scalar S=64  T=8       0.000360     0.001168  PASS
kda    S=128 T=4       0.000175     0.001516  PASS
```

Reproduce with `bash cann/tests/run_parity.sh`, `bash cann/tests/run_quant.sh` and
`bash cann/tests/run_gdn.sh`.
