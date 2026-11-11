# Ascend A3 S2 decode and projection validation

Measured on 2026-10-21 on the user's A3 environment, reported by the runtime as
`Ascend910_9362`. The unchanged decoder, projection kernel and harness from
commit `e1e538174af1ec65ec5f9c124d2fee35f6f2b79f` were rebuilt with
`-DSOC_VERSION=Ascend910_9362`. No computational source change was required.
The upstream reference remains `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`.

Three fresh decoder processes passed 12 boundary cases each. Three fresh
projection processes passed 21 cases each: **36 decode checks and 63 projection
checks passed**. The 21 projection output hashes were stable across A3 processes
and all matched the recorded A2 hashes. This validates the tested synthetic
operators on one A3 chip, not a Strata inference backend.

## Environment

| Item | Observed value |
| --- | --- |
| Platform label / runtime device name | A3 / Ascend910_9362 |
| Selected runtime device | Logical `npu:0` |
| Physical mapping observed during smoke test | NPU 1, chip 0, physical ID 2 |
| Visible hardware | Two chips, each showing 65,536 MiB in npu-smi; only one used |
| Runtime accessible total memory on device 0 | 65,787,658,240 bytes |
| Host | aarch64, Ubuntu 20.04.5, 40 visible logical CPUs, one NUMA node |
| Container limits | 210 GiB RAM; 36 CPU equivalents |
| Driver / npu-smi | 25.5.5 |
| CANN | 9.0.0, V100R001C10SPC001B250 |
| Compiler | Ascend clang 15.0.5; host g++ 9.4.0; CMake 3.20.5 |
| Python / torch / torch_npu | 3.11.4 / 2.7.1+cpu / 2.7.1.post4 |
| NumPy | 1.26.4 (A2 used 2.4.6) |

The environment variables `ASCEND_VISIBLE_DEVICES` and `NPU_VISIBLE_DEVICES`
were both `3`, while `ASCEND_RT_VISIBLE_DEVICES` was unset. The runtime exposed
two logical devices. The harness uses logical device zero; its physical mapping
above was verified by matching the smoke-test process in npu-smi. Do not infer
the mapping from the environment-variable value alone. No second-chip or HCCL
test was performed.

The [host facts](evidence/host-facts.a3.json),
[smoke log](evidence/smoke.log),
[before](evidence/npu-smi-before.txt) and
[after](evidence/npu-smi-after.txt) snapshots retain the observations. The FP16
16 x 16 identity-matmul smoke test had maximum absolute error zero. No other NPU
process was listed after the tests. Clocks, background load and NUMA placement
were not controlled; CPU model, storage bandwidth and PCIe link bandwidth were
not established.

## Correctness and provenance

The input layout, FP32 forward bound and quantization contract are unchanged
from the [projection README](../fused/README.md). The scalar legacy activation
quantizer is still a host fixture; no AVX expert or NPU activation quantizer is
executed. The existing C++ Q2_0 decoder is the weight oracle, followed by FP64
host products for projection comparison.

- Decoder: all 36 boundary checks were bit-exact against the CPU decoder,
  including signed zero and FP16 subnormal/extreme scales. Each process's larger
  640 x 2560 decode diagnostic also had zero bit mismatches.
- Projection: all 63 checks passed the FP32 bound, guard regions, tail padding,
  unchanged inputs and invalid-dimension checks. All eight split-baseline shapes
  passed their numerical checks in every process.
- The maximum error used `0.0772` of the declared bound. Ordinary-scale projection
  fixtures had maximum absolute error `6.034e-6`; the extreme-scale set reached
  `0.528201`, to be read with its much larger input magnitude.
- All three A3 projection reports have identical source and binary hashes.
  Corresponding output hashes match across all three A3 processes and the A2
  reference report for **21/21** cases. The generator, seed and shapes are the
  same. Input-byte hashes were not recorded in the original A2 harness, so this
  does not assert independently verified byte parity of every input buffer.

The [source manifest](evidence/SOURCE_MANIFEST.a3.json) records 13 transferred
files, including the license. All match the A2 files. The historical directory
name and `ascend-a2` report kind were retained because the harness was run
unchanged; the reports' `device` field and separate host facts identify A3.
The [build settings](evidence/build-settings.a3.json) record the A3 SoC override.

The A3 projection library SHA-256 is
`e49fd7ae0bf22b8cc80ffc1b76acbc42aa1a6f2eabd7220e897326fc3dcf86b9`.
The A3 decoder library is
`a1782aeaba9c0f1bdafcfcf39009073a1dad3526e6e58868c92cdb31ce1f6ec7`.
The CPU reference library hash matches A2. Sources and raw data were checked
with SHA-256 during export. Public logs/facts replace workspace, SDK and Python
installation paths with placeholders; original exports are retained privately.

## Same-day projection diagnostic

Two warm-ups and five measured rounds were retained per process. Each round
averages three calls; order alternates. A3 entries below aggregate 15 measured
round averages from three fresh processes. Buffers are preallocated, inputs
resident, HF32 disabled, and timing uses the host clock with synchronization.

The split path is the same unoptimized S2 decoder plus FP32 `torch.mm`, including
an extra ACL/framework synchronization. It is not an optimized native baseline.
Quantization, input preparation, H2D and allocation are excluded; the fused time
includes the kernel's activation copy and casts. A2 medians come from its
[published summary](../fused/evidence/SUMMARY.json).

| M x K | T | A2 fused, ms | A3 fused median (range), ms | A3 split median (range), ms |
| --- | ---: | ---: | ---: | ---: |
| 640 x 2560 | 1 | 0.259 | 0.266 (0.263-0.267) | 0.952 (0.946-0.956) |
| 640 x 2560 | 2 | 0.292 | 0.302 (0.301-0.307) | 0.948 (0.946-0.954) |
| 640 x 2560 | 4 | 0.361 | 0.378 (0.376-0.380) | 0.950 (0.947-0.952) |
| 640 x 2560 | 8 | 0.503 | 0.530 (0.528-0.532) | 0.950 (0.946-0.955) |
| 2560 x 640 | 1 | 0.223 | 0.228 (0.227-0.230) | 0.951 (0.948-0.958) |
| 2560 x 640 | 2 | 0.256 | 0.263 (0.262-0.276) | 0.951 (0.946-0.957) |
| 2560 x 640 | 4 | 0.321 | 0.334 (0.332-0.335) | 0.951 (0.947-0.953) |
| 2560 x 640 | 8 | 0.449 | 0.474 (0.472-0.475) | 0.950 (0.948-0.955) |

A3's fused host-call medians were **2.3-5.5% higher** than the earlier A2
medians in these eight workloads. This observation is retained. The hosts,
driver, NumPy version, CPU quota and NUMA topology differ; no clock control or
device-event timing isolates a chip-level cause. These results do not establish
an A3 hardware regression, a general A2/A3 ranking or any model speedup.

Both shapes contain 1,638,400 weights. The fused path avoids a 6,553,600-byte
full FP32 weight buffer. This is an allocation fact, not measured HBM traffic.

## Decode diagnostic

Each decoder process retained two warm-ups and three measured rounds, five
compute calls per round. Entries aggregate nine measured rounds. Copy timing
uses pageable host inputs and existing device buffers.

| Operation | A3 median (range), ms |
| --- | ---: |
| H2D copy plus synchronization | 0.137 (0.127-0.148) |
| Native decode launch plus final synchronization, per call | 0.909 (0.907-0.912) |
| Existing scalar CPU decoder, per call | 1.520 (1.510-1.530) |

The CPU value is not an optimized expert implementation or a controlled
cross-host CPU comparison. The [summary](evidence/SUMMARY.json) aggregates measured
values; raw process reports retain every warm-up and measured value.

## Reproduce

Use the existing CANN SDK environment, a fresh build directory for each target,
and new report names. `POC_ROOT` is the checkout or transferred source root.
These are single-device tests; the unchanged harness selects logical device 0.

```sh
cmake -S "$POC_ROOT/bench/ascend_a2_s2" \
  -B "$POC_ROOT/bench/ascend_a2_s2/build-a3-release" \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910_9362
cmake --build "$POC_ROOT/bench/ascend_a2_s2/build-a3-release" -j2
cmake -S "$POC_ROOT/bench/ascend_a2_s2/fused" \
  -B "$POC_ROOT/bench/ascend_a2_s2/fused/build-a3-release" \
  -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910_9362
cmake --build "$POC_ROOT/bench/ascend_a2_s2/fused/build-a3-release" -j2
for run in 1 2 3; do
  OPENBLAS_NUM_THREADS=1 python3 "$POC_ROOT/bench/ascend_a2_s2/probe.py" \
    --build "$POC_ROOT/bench/ascend_a2_s2/build-a3-release" \
    --report "$POC_ROOT/decode-run-$run.json" --benchmark
done
for run in 1 2 3; do
  OPENBLAS_NUM_THREADS=1 python3 "$POC_ROOT/bench/ascend_a2_s2/fused/probe.py" \
    --build "$POC_ROOT/bench/ascend_a2_s2/fused/build-a3-release" \
    --decode-build "$POC_ROOT/bench/ascend_a2_s2/build-a3-release" \
    --report "$POC_ROOT/projection-run-$run.json" --benchmark
done
```

Passed: clean A3 Release builds, Python syntax compilation, identity smoke,
all six process runs, the same-day split baselines, output-hash comparisons and
source/evidence checks. Logs for both builds and all runs are in `evidence/`.

Failed: no build or numerical check failed in this A3 stage. No A3 fused timing
improvement over the earlier A2 medians was observed. The earlier A2 failures
and slower scalar attempts remain in their own reports.

Not run: real model weights, complete gate/up-SwiGLU-down composition, NPU
activation quantization, default AVX expert parity, routing, cache transfers,
overlap, full-model output, the second visible chip, HCCL, multi-NPU or long soak.
No engine/default-path source changed; existing backend binaries were not
rebuilt or compared. A complete Ascend backend still needs separate scope
agreement.
