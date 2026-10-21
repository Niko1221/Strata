# Ascend 910B3: S2 decode correctness and timing diagnostic

Measured on 2026-10-21. Upstream source:
`9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`.

The standalone Ascend C decoder agrees with Strata's existing C++ Q2_0
reference on the tested synthetic blocks. The engine is not connected to this
decoder. No model was loaded or generated text, and no model speedup is claimed.

## Environment

- One Ascend 910B3. `npu-smi` reported 65,536 MiB HBM and driver 25.5.0.
- aarch64, Ubuntu 20.04.5, Python 3.11.4, CANN 9.0.0.
- PyTorch 2.7.1+cpu, torch_npu 2.7.1.post4.
- Host compiler GCC 9.4.0. Ascend compiler reported clang 15.0.5.
- 32 logical CPUs and two NUMA nodes visible. The cgroup CPU quota was 29 CPU
  equivalents; its memory limit was 200 GiB. `lscpu` did not expose a CPU model.
- CPU affinity, NUMA placement and clocks were not controlled. Host memory
  bandwidth was not measured. The timing input is a constant synthetic matrix,
  not a representative model workload.

## Correctness

The reference reads raw 18-byte Q2_0 blocks through the unchanged
`strata::dequantize_q2_0`. The NPU reads separate code and FP32-scale planes.
The 12 block counts are 0, 1, 2, 3, 7, 8, 31, 32, 33, 256, 257 and 4,096.
Tests include every possible packed byte, negative scales, signed zero, finite
FP16 extremes and FP16 subnormal scales widened to FP32. Outputs are compared
by FP32 bit pattern. Guards and unchanged input buffers are checked separately.

Three independent processes passed all 12 cases after the task-queue fix below.
The final harness, which adds the optional timing diagnostic and source hashes,
then passed those 12 cases and the larger 640 x 2560 matrix check. All reported
bit-mismatch counts are zero, and all guard/input checks passed.

The [final report](evidence/benchmark-final.json) records the kernel, reference
library and final Python harness SHA-256 values. The
[source manifest](evidence/SOURCE_MANIFEST.final.json) records the transferred
source hashes. The three earlier process results are separate files in
`evidence/`; they predate the optional benchmark code.

## Timing diagnostic

One 640 x 2560 matrix contains 25,600 S2 blocks. Its canonical inputs occupy
512,000 bytes; its decoded FP32 output occupies 6,553,600 bytes. Buffers were
allocated before timing. Two warm-up rounds precede three measured rounds.
Each compute round contains five calls. Warm-up samples and individual measured
samples are preserved in the JSON.

| Operation | Median | Measured range |
| --- | ---: | ---: |
| Pageable host inputs copied to existing NPU buffers, then synchronized | 0.388 ms | 0.370-0.403 ms |
| NPU launch calls plus final synchronization, divided by five | 0.868 ms | 0.866-0.870 ms |
| Existing C++ scalar reference, five calls | 4.356 ms/call | 4.172-4.699 ms/call |

These are host wall times. They include dispatch and synchronization costs;
they are not isolated device-kernel timings. Copies and decode were measured
separately, so the table does not establish overlap. The CPU reference is not
Strata's optimized expert implementation. Do not turn this table into a claim
about model acceleration or all CPU implementations.

## Failures retained

- The first build omitted `CMAKE_BUILD_TYPE`; the CANN merge script received an
  empty build-type argument and failed. The standalone CMake now defaults to
  Release.
- Reusing that directory with Release caused duplicate host objects from the
  old configuration. A new `build-release` directory built successfully. The
  failed and successful build logs are retained separately.
- The first runtime harness submitted a direct CANN launch before draining
  torch_npu's host task queue. The 32-block case retained all 2,048 sentinel
  values, while adjacent sizes passed. Adding an explicit synchronization
  before the ctypes boundary was followed by three passing fresh processes.
  The [failed report](evidence/parity-initial.json) is retained. This was a
  prototype integration failure, not an observed bug in Strata's engine.

## Reproduce

Use a fresh build directory and a new report path. Set `POC_ROOT` to this source
checkout and use the existing SDK environment:

```sh
cmake -S "$POC_ROOT/bench/ascend_a2_s2" \
  -B "$POC_ROOT/bench/ascend_a2_s2/build-release" \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build "$POC_ROOT/bench/ascend_a2_s2/build-release" -j2
python3 -u "$POC_ROOT/bench/ascend_a2_s2/probe.py" \
  --build "$POC_ROOT/bench/ascend_a2_s2/build-release" \
  --benchmark --report "$POC_ROOT/new-result.json"
```

Passed: clean Release build; the reported parity runs; final timing diagnostic;
source-transfer hash checks. Only files under `bench/ascend_a2_s2` were added;
the root CMake file, engine and server sources have no diff from the stated
upstream revision. Existing backend binaries were not rebuilt or compared.

Not run: real model blocks, optimized expert math and activation quantization,
full-model generation, GDN/QSA, MTP, vision, cache restoration, A3, multi-NPU,
long soak, CUDA, HIP, SYCL and Metal. This is a correctness prototype and timing
diagnostic, not the repository's standard model benchmark protocol.
