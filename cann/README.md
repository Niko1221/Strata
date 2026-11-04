# `cann/` - the Ascend port of the engine (P1)

This directory is to the CANN backend what `sycl/` is to the Intel backend: a separate project
that grows next to the CUDA engine until it can take the engine's sources.

P1 (this commit) is only the layer everything else sits on:

| file | what |
|---|---|
| `include/strata/cann/device.hpp` | `Device`, `Stream`, `Buffer`, `DeviceInfo`, `Error`; one place that turns `aclError` into an exception |
| `include/strata/cann/ops.hpp` | `matmul`, `rms_norm`, `silu`, `add`, `softmax` on fp16 device tensors, each with a CPU oracle |
| `src/device.cpp` | `aclInit` / `aclrtSetDevice` / `aclrtCreateStream` / `aclrtMalloc` / `aclrtMemcpy` |
| `src/ops.cpp` | the aclnn two-phase call (`*GetWorkspaceSize` then the op) behind one helper |
| `tests/parity.cpp` | random fp16 inputs, device result against the CPU reference |

## Build and run

```bash
cmake -S cann -B build-cann
cmake --build build-cann -j
./build-cann/cann_parity
```

The toolkit comes from `ASCEND_HOME_PATH` (default `/usr/local/Ascend/cann-9.0.0`).

## Reference noise floor

The parity test rounds inputs through fp16 before the CPU oracle runs, because that is what the
device receives. It then compares against these tolerances:

| op | shape | tolerance |
|---|---|---|
| matmul | 256 x 512 x 384 | 0.5 (512-term fp16 dot) |
| rms_norm | 512 x 2048 | 0.02 |
| silu | 4096 | 0.02 |
| add | 4096 | 0.01 |
| softmax | 64 x 1024 | 0.01 |

## What is not here yet

- The engine's own kernels. `src/` still compiles for CUDA only; P2 starts with the Q4_K/Q6_K
  dequant + matmul that the model's weights need.
- `GGML_OP_GATED_DELTA_NET`. `qwen35moe` (Qwen3.6-35B-A3B) runs 3 of every 4 layers through it;
  `ggml-cann` does not implement it, so those layers fall back to the CPU. The same op is what
  P3 needs, so it is the first shared piece.
- Any link to `include/strata/...` engine headers. That happens when a kernel can run behind the
  engine's own interface instead of behind this test.

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

Reproduce with `bash cann/tests/run_parity.sh` or the three commands above.
