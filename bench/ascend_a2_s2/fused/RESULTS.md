# Ascend 910B3 S2 projection results

Measured on 2026-10-21 against Strata source revision
`9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`. This standalone experiment fuses
S2 unpacking with a quantized-activation matrix projection. It avoids writing a
full FP32 weight matrix to global memory. It does not run Strata inference.

The vector implementation passed 21 fixtures in each of three fresh processes.
Its host-call times were lower than this experiment's split decoder plus
`torch.mm` baseline in all eight measured shapes. Earlier scalar implementations
were slower than that baseline and are retained below.

## Environment and scope

One Ascend 910B3 (A2), exposed as logical `npu:0`; aarch64, Ubuntu 20.04.5,
Python 3.11.4, CANN 9.0.0, driver/npu-smi 25.5.0, torch 2.7.1+cpu and
torch_npu 2.7.1.post4. Host g++ 9.4.0, CMake 3.20.5, Ascend compiler based on
clang 15.0.5. The same supplied container as the [decode experiment](../RESULTS.md)
was used. It has a 200 GiB memory limit and a 29-CPU-equivalent quota. The CPU
model, storage bandwidth, PCIe bandwidth and NPU clocks were not measured.
Clock locking, NUMA binding and background-load control were not performed.

The [README](README.md) defines the exact layout, arithmetic and commands.
Inputs are deterministic synthetic Q2_0 blocks, not model weights. Two
independent 32-element activation scales are retained for every 64 weights.
The block reduction order differs from the default CPU AVX implementation.
Legacy activation quantization is a host fixture transcribed from its scalar
source, not execution of the x86 implementation. The experiment does not
establish bitwise agreement with Strata's default expert implementation.

## Correctness

All 63 case evaluations passed. Output SHA-256 values for corresponding cases
matched across the three processes. Each process checked:

- Zero rows and odd row counts, eight-row store boundaries, and 640 x 2560 and
  2560 x 640 matrices. Token counts 1 through 8 are covered; performance uses
  T=1, 2, 4 and 8.
- All 256 packed byte patterns in each byte position, negative weight scales,
  signed zero, FP16 subnormal scales and maximum finite FP16 scales.
- Different adjacent activation scales, zero activation chunks, negative INT8
  codes and known half-away quantizer ties.
- FP64 oracle comparison, output guards, zero output padding, unchanged inputs,
  and host rejection of five invalid dimension combinations.

The oracle uses Strata's unchanged CPU Q2_0 decoder and FP64 matrix products.
The largest error used 0.0772 of the declared FP32 forward bound (7.72% of
the bound, not a 7.72% relative output error). The ordinary-scale benchmark
fixtures had maximum absolute error `6.034e-6`. The extreme-scale stress set
had maximum absolute error `0.528201`; absolute errors must be read with their
input magnitude and bound. See [SUMMARY.json](evidence/SUMMARY.json) and the
[first](evidence/fused-vector-run-1.json),
[second](evidence/fused-vector-run-2.json) and
[third](evidence/fused-vector-run-3.json) reports for every case.

## Same-day compute diagnostic

Both paths use preallocated buffers and resident inputs. Activation
quantization, H2D copies, allocation and host oracle work are excluded. Fused
timing includes its on-device INT8 copy and exact INT8 -> half -> FP32 casts.
The split path expands the same S2 matrix with the parent experiment's scalar
decoder, synchronizes at the raw ACL/framework boundary, then calls FP32
`torch.mm` with HF32 explicitly disabled. Reconstruction of its FP32 activation
input precedes timing. All split results also passed their stated FP32 bound.

The split path has an extra interop synchronization and an unoptimized decoder.
This comparison does not establish an advantage over an optimized native
two-kernel implementation, other Ascend libraries or Strata's existing backends.

Each fresh process retained two warm-up rounds and five measured rounds, three
calls per round with a final synchronization. Order alternated. Entries below
are medians and ranges across the 15 measured round averages from all three
processes, in milliseconds per call. They are host wall times, not device-event
measurements or model throughput.

| Matrix M x K | T | Fused median (range), ms | Split median (range), ms |
| --- | ---: | ---: | ---: |
| 640 x 2560 | 1 | 0.259 (0.256-0.266) | 0.974 (0.956-0.984) |
| 640 x 2560 | 2 | 0.292 (0.289-0.295) | 0.971 (0.957-0.983) |
| 640 x 2560 | 4 | 0.361 (0.359-0.363) | 0.970 (0.956-0.984) |
| 640 x 2560 | 8 | 0.503 (0.500-0.505) | 0.972 (0.959-0.982) |
| 2560 x 640 | 1 | 0.223 (0.221-0.225) | 0.974 (0.956-0.977) |
| 2560 x 640 | 2 | 0.256 (0.253-0.259) | 0.971 (0.962-0.979) |
| 2560 x 640 | 4 | 0.321 (0.318-0.325) | 0.973 (0.958-0.979) |
| 2560 x 640 | 8 | 0.449 (0.447-0.454) | 0.974 (0.955-0.984) |

Both matrices contain 1,638,400 weights. The split path allocates an extra
6,553,600-byte (6.25 MiB) decoded weight matrix. The fused kernel uses row scratch
and activation scratch in UB instead. This allocation difference is known from
the code; HBM traffic and whole-process peak memory were not profiled.

## Rejected attempts retained

All three implementations built and passed their own correctness fixtures.
The initial scalar implementation computed INT32 dots using scalar GM reads.
Copying activations to UB but continuing to read them one scalar at a time made
the measured time worse. The final implementation uses vector multiplication
and grouped reductions over exact FP32 integer products.

| Matrix | T | Scalar GM, ms | Cached scalar, ms | Final vector, ms |
| --- | ---: | ---: | ---: | ---: |
| 640 x 2560 | 1 | 1.351 | 1.454 | 0.259 |
| 640 x 2560 | 8 | 6.149 | 8.673 | 0.503 |
| 2560 x 640 | 1 | 1.127 | 1.219 | 0.223 |
| 2560 x 640 | 8 | 4.610 | 7.241 | 0.449 |

Each earlier version had one fresh process, 17 fixtures and five measured rounds
per shape. They are screening results, not three-process confirmations.
The first 17 fixtures and eight benchmark inputs are unchanged in the final
harness; four additional fixtures were appended. Sources are retained in
[scalar-source](evidence/scalar-source/) and
[cached-scalar-source](evidence/cached-scalar-source/), with their original
[scalar](evidence/fused-run-1.json) and
[cached scalar](evidence/fused-cache-run-1.json) reports and build/run logs.
Their source hashes differ from the final source by design.

## Reproduction and provenance

Run the exact commands in [README.md](README.md) three times with different
report names. The submitted computational sources match the transferred files
in [the manifest](evidence/fused-SOURCE-MANIFEST.json). All three final reports
record identical source and binary hashes. The final kernel library SHA-256 is
`16fdf59a91974cc86cc0a130174bd427fb4c8a584b885256070a23bfdf53a5c2`.
The decoder and CPU reference binaries have the same hashes as the parent
experiment. Evidence exports were checked with SHA-256 before import. Public
logs replace workspace, SDK and Python installation paths with placeholders.

Passed: three Ascend C Release builds (scalar, cached scalar, vector); the two
screening runs; three final process runs; source-transfer and output-hash checks;
split-baseline numerical checks. Final Python syntax compilation also passed.

Failed: no build or numerical check failed in this projection stage. Both
scalar approaches failed the performance objective and were replaced. The
parent decoder experiment's earlier failures remain in its separate report.

Not run: real model weights, complete gate/up-SwiGLU-down expert composition,
intermediate requantization on NPU, default AVX expert parity, routing, CPU/NPU
cache transfers, overlap, full-model generation, A3, multi-NPU or long soak.
CUDA, HIP, SYCL and Metal files are untouched; their binaries were not rebuilt
or compared. The root build and engine/server/default-path sources have no diff
from the stated revision. This is a bounded benchmark proposal; a complete
Ascend backend needs separate maintainer scope agreement.
