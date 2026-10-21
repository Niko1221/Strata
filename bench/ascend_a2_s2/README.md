# Ascend A2 S2 correctness experiment

This standalone experiment implements Strata's canonical S2 weight decode on an
Ascend NPU. It is not an inference backend. It does not change the engine build,
installation, defaults, or model selection. Do not describe a passing result as
Strata model support on Ascend.

Measured results, failures, reproduction commands and limits are in
[RESULTS.md](RESULTS.md).

The unchanged sources were also rebuilt and executed on an A3
(`Ascend910_9362`): [A3 results and reproduction](a3/RESULTS.md).

The separate [fused projection experiment](fused/README.md) extends this decode
check to S2 matrix projections with quantized activations. Its measurements and
limits are separate from the decode results below.

The input contract is `include/strata/kernels/dequant_s2.hpp`: each block has 16
code bytes and one FP32 scale. Four codes are packed per byte, least significant
first. Each output is `float(code - 1) * scale`, with the subtraction performed
in the integer domain. The raw Q2_0 reference has an FP16 scale followed by the
same 16 code bytes. The probe calls the existing
`strata::dequantize_q2_0` from `include/strata/artifact/dequant.hpp`; it does not
implement another scalar decoder as its oracle.

## Build and run

Use an existing CANN SDK and a compatible `torch_npu` environment. The first
target is the supplied Ascend 910B3, CANN 9.0.0, aarch64 machine. Other targets
need their own build and execution evidence.

```sh
cmake -S bench/ascend_a2_s2 -B bench/ascend_a2_s2/build \
  -DSOC_VERSION=Ascend910B3 -DCMAKE_BUILD_TYPE=Release
cmake --build bench/ascend_a2_s2/build -j2
python3 bench/ascend_a2_s2/probe.py --report /absolute/new-result.json
```

Set `ASCEND_HOME_PATH` to the installed SDK. If its CMake files use a different
layout, pass `-DASCENDC_CMAKE_DIR=/path/to/ascendc_kernel_cmake`. The source root
for the reference can be set with `-DSTRATA_SOURCE_ROOT=/path/to/Strata`.

The probe checks 12 block counts, including zero, odd counts, core-count
boundaries and a larger case. It covers all 256 packed byte patterns, normal
FP16 scales, signed zero, FP16 subnormals and maximum finite FP16 values.
It compares FP32 bit patterns, checks output guard regions and verifies that
inputs were not modified. Existing report files are never overwritten.

Add `--benchmark` to measure one synthetic 640 x 2560 weight matrix. The JSON
preserves two warm-up rounds and three measured rounds' settings and timings.
Copies use pageable host memory and preallocated NPU buffers. The launch time
includes five calls and a final synchronization, divided by five; it is not an
isolated device-kernel measurement. CPU reference time is reported separately.
These diagnostics do not measure expert computation, overlap or model speed.
The ctypes boundary explicitly drains torch_npu's host task queue before a
direct ACL-stream launch; using the same stream handle alone is insufficient
to order tensor initialization queued on the host.

## Limits

Inputs are deterministic synthetic blocks, not weights extracted from a model.
The kernel is a serialized scalar-unpack baseline. It is not optimized, and no
performance improvement is claimed. The experiment does not validate expert
matrix multiplication, activation quantization, routing, GDN/QSA, MTP, vision,
conversation caching or full-model output. NaN and infinity scales are not
covered. CUDA, HIP, SYCL and Metal code is untouched.
