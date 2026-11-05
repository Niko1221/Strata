# `cann/` - the Ascend port of the engine

This directory is to the CANN backend what `sycl/` is to the Intel backend: a separate project
that grows next to the CUDA engine until it can take the engine's sources.

| phase | scope | state |
|---|---|---|
| P1 | device, stream, HBM buffers, first aclnn ops, CPU oracle | done, parity on hardware |
| P2 | Q4_K / Q5_K / Q6_K decoders, dequantize then matmul on the NPU | done, bit-exact vs ggml |
| P2b | GGUF reader, real model file, real-weight GEMV on the NPU | done |
| P3 | GDN done; RoPE / KV cache / prefill shapes | in progress |
| P4 | MoE routing and expert matmuls | not started |
| P5 | `serve/` wiring, 18789, benchmarks | not started |

## Files

| file | what |
|---|---|
| `include/strata/cann/device.hpp` | `Device`, `Stream`, `Buffer`, `DeviceInfo`, `Error`; one place that turns `aclError` into an exception |
| `include/strata/cann/ops.hpp` | `matmul`, `rms_norm`, `silu`, `add`, `sub`, `mul`, `muls`, `exp`, `softmax` on fp16 device tensors, each with a CPU oracle |
| `include/strata/cann/quants.hpp` | `block_q4_K` (144 B), `block_q5_K` (176 B), `block_q6_K` (210 B) plus `dequantize()` |
| `include/strata/cann/gdn.hpp` | gated delta net: float reference and the device path |
| `include/strata/cann/gguf.hpp` | GGUF v3 reader: header, metadata, tensor index, on-demand tensor bytes |
| `src/device.cpp` | `aclInit` / `aclrtSetDevice` / `aclrtCreateStream` / `aclrtMalloc` / `aclrtMemcpy` |
| `src/ops.cpp` | the aclnn two-phase call (`*GetWorkspaceSize` then the op) behind one helper |
| `src/quants.cpp` | the three K-quant decoders, written against ggml's `dequantize_row_*` |
| `src/gdn.cpp` | the recurrence, one aclnn step per token |
| `src/gguf.cpp` | the reader; weights stay on disk, `tensor_bytes` reads a range |
| `tests/parity.cpp` | random fp16 inputs, device result against the CPU reference |
| `tests/quant_parity.cpp` | decoders against ggml's `to_float`, then a dequantized matmul on the NPU |
| `tests/gdn_parity.cpp` | GDN against the float reference, both decay modes |
| `tests/gguf_parity.cpp` | the real 22.85 GB model: header, weight mix, real GEMV on the NPU, real-tensor decoders |
| `tests/run_parity.sh` / `run_quant.sh` / `run_gdn.sh` / `run_gguf.sh` | build and run the four tests |

## Build and run

```bash
# P1 only
cmake -S cann -B build-cann && cmake --build build-cann -j
./build-cann/cann_parity

# with ggml's decoders as the oracle (needed by quant_parity and gguf_parity)
cmake -S cann -B build-cann \
  -DSTRATA_CANN_GGML_DIR=/path/to/llama.cpp \
  -DSTRATA_CANN_GGML_LIB_DIR=/path/to/llama.cpp/build/bin
cmake --build build-cann -j
./build-cann/quant_parity
./build-cann/gguf_parity /path/to/model.gguf
```

`ASCEND_HOME_PATH` defaults to `/usr/local/Ascend/cann-9.0.0`. The ggml oracle links
`libggml-base.so` for the tests only; nothing in the port depends on ggml.

## Tolerances

Inputs are rounded through fp16 before the CPU oracle runs, because that is what the device
receives.

| check | tolerance |
|---|---|
| matmul 256x512x384 | 0.5 (512-term fp16 dot) |
| rms_norm 512x2048 / silu / add / softmax | 0.02 / 0.02 / 0.01 / 0.01 |
| Q4_K / Q5_K / Q6_K decode | bit-exact against ggml |
| GDN, one head | attn 0.05, state 0.05 |
| real-weight GEMV 512x2048 on the NPU | 0.5 |

## Two API facts worth writing down

- aclnn tensor descriptors are **row-major**: `shape = {rows, cols}`, `stride = {cols, 1}`.
  Handing it a ggml-style column-major descriptor (`{cols, rows}` / `{1, cols}`) fails with
  `161002` on the first matmul.
- `aclnnRmsNorm` takes an extra `rstdOut` tensor (the reciprocal std per row), and `gamma` must
  be 1-D. A 2-D `{1, C}` gamma fails with `561002`.

## What the model file is made of

`gguf_parity` prints the real mix, and it is not "Q4_K and Q6_K":

```
weight mix :
  f32    368 tensors      0.10 GB
  q8_0   260 tensors      2.61 GB
  q4_K    80 tensors     12.08 GB
  q5_K    40 tensors      7.38 GB
  q6_K     3 tensors      0.66 GB
  bf16     2 tensors      0.00 GB
  sum                    22.84 GB
```

So 32% of the file is Q5_K, which is why its decoder is here too. Q8_0 (2.61 GB, the attention
and embedding tensors) and the F32 norms are still open.

## What is not here yet

- A decoder for Q8_0 and the F32/BF16 tensors; the three K-quants are the bulk but not all of it.
- The engine's own kernels: `src/` still compiles for CUDA only.
- `ggml-cann` still has no `GGML_OP_GATED_DELTA_NET`, so the llama.cpp server keeps 3 of 4 layers
  on the CPU. This directory has a working device path, but wiring it into that backend is a
  separate change.
- Fused dequant+matmul. The decoder works on the host today; a fast engine wants it fused on the
  device.
- Prefill-shaped GDN (the current path is one token at a time).

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
muls 8x8 by 0.5              0.000000     0.000000     0.0100  PASS
matmul 8x8x1                 0.000205     0.000088     0.0100  PASS
mul 8x1 * 1x8                0.000237     0.000062     0.0100  PASS
```

```
Q4_K   vs ggml: max|diff| = 0  BIT-EXACT
Q5_K   vs ggml: max|diff| = 0  BIT-EXACT
Q6_K   vs ggml: max|diff| = 0  BIT-EXACT
Q4_K dequant -> NPU matmul 64x512x256: max|diff| vs CPU = 0.013428  PASS
```

```
case                 attn|diff|  state|diff|  result
kda    S=64  T=8       0.000360     0.003886  PASS
scalar S=64  T=8       0.000360     0.001168  PASS
kda    S=128 T=4       0.000175     0.001516  PASS
```

```
gguf       : v3, 753 tensors, data at 10991392
arch       : qwen35moe, 41 layers, 256 experts (top 8), embedding 2048

real GEMV  blk.0.ffn_gate_exps.weight [512 x 2048] on the NPU: max|diff| = 0.000116  PASS

blk.0.ffn_gate_exps.weight         Q4_K  ... vs ggml: BIT-EXACT
blk.0.ffn_down_exps.weight         Q5_K  ... vs ggml: BIT-EXACT
blk.34.ffn_down_exps.weight        Q6_K  ... vs ggml: BIT-EXACT
```

Reproduce with `bash cann/tests/run_parity.sh`, `run_quant.sh`, `run_gdn.sh`, `run_gguf.sh`.
