# SM75 split head Top64 selector

This report measures one opt-in CUDA selector, based on upstream `c234774038d1c9249cb33637d27b895d47b39786`
(Strata 0.2.22, engine 0.1.69). Communication, expert placement, weights and the target sampler keep their existing
implementations. The inputs are synthetic logits and penalty lists from
[`top64_sm75_test.cu`](../../../engine/src/split/tools/top64_sm75_test.cu); no model or private prompt is required.

## Machine and method

- Ubuntu 26.04.1 LTS, kernel 7.0.0-43-generic; Intel Core i7-7700, 32 GiB RAM.
- Two RTX 2080 Ti, 22 GiB each; driver 595.99.02, CUDA 13.0, 280W limits.
- SM75 Release build, `--use_fast_math`, Ninja `-j4`, NVCC `--split-compile=2`; compilation concurrency limited to two jobs.
- Each reported timing is a CUDA-event measurement of 64 selector invocations: 16 calls per captured graph, four
  launches after two warmups. Three reference/candidate pairs alternate order; the median-ratio pair is reported.
- Correctness uses random values, tied values and signed zeros, with and without penalties. The CPU selection oracle
  sorts the actual reference-penalized values by ordered float bits, then token ID.

## Measured selector time

Random inputs, 124,160 logits per row. Every case has exact candidate values, IDs and penalized logits.

| GPU | Rows | Reference, µs | Tiled, µs | Ratio |
|---|---:|---:|---:|---:|
| 0 | 1 | 174.9435 | 51.7120 | 3.38304 |
| 0 | 3 | 188.3915 | 54.2420 | 3.47317 |
| 0 | 4 | 179.7120 | 53.9840 | 3.32899 |
| 0 | 7 | 180.1290 | 61.9200 | 2.90906 |
| 1 | 1 | 169.7265 | 50.1505 | 3.38434 |
| 1 | 3 | 170.9760 | 52.6720 | 3.24605 |
| 1 | 4 | 174.4715 | 52.3140 | 3.33508 |
| 1 | 7 | 174.8160 | 60.1345 | 2.90708 |

The complete [final timing log](microbenchmark.log) retains all 48 cases, including smaller vocabulary slices, ties
and signed zeros. This is a kernel/selector comparison, not an end-to-end model benchmark.

### Rejected smaller-slice dispatch

The first candidate used the tiled path from 2,048 logits. A [crossover sweep](crossover-exploration.log) on both cards
and 1/2/3/4/7/8/16 rows found these ranges of reference time / tiled time on random inputs:

| Logits per row | Minimum ratio | Maximum ratio |
|---:|---:|---:|
| 2,048 | 0.40089 | 0.62466 |
| 4,096 | 0.41769 | 0.68968 |
| 8,192 | 0.49134 | 0.75852 |
| 16,384 | 0.59700 | 1.05297 |
| 32,768 | 0.93076 | 1.54147 |
| 53,149 | 1.12836 | 2.00925 |
| 65,536 | 1.28035 | 2.37645 |
| 124,160 | 2.57722 | 3.48798 |

The final dispatch conservatively keeps the reference below 65,536 logits. The exploratory data belongs to the
earlier threshold; final parity tests include 65,535/65,536/65,537 to check the new boundary.

## Local validation

See [check results](local-check-results.json) for source digests, build results and binary comparison. The relevant
split CTest suite contains 15 tests: eleven CPU policy tests, the quantized-dot golden and three Top64 paths
(direct, runtime disabled and runtime enabled). All pass; each Top64 path also passed three consecutive repetitions.
The Python split suite passed 72 tests.

The repository's standard `tools/check --only cmake-cpu --strict` also passed: 173 tests, zero failures and four
conditional skips (`io_priority_test`, `expert_multi_test`, `expert_parity`, `pool_test`). These skips are retained in
[the CPU result](cpu-check.json) and are not counted as passes.

Each Top64 run checks 462 cases per card: eleven vocabulary sizes, seven row counts, three input types and two
global token offsets, plus two concurrent-stream cases. It compares candidate values and full penalized rows
bitwise, checks IDs, guards both output arrays, verifies NaN/±Inf error bit 16, replays captured graphs with changed
inputs, and exercises graph update/reinstantiation. With the runtime flag off, captured selection has the original
single kernel node; the enabled large-slice path has allocation, two kernels and free nodes.

The reference kernel object's `.nv_fatbin` is bit-identical between enabled and disabled builds:
`91d976ff50f8d79d29df6f7b3e080adb5c6e95b77f70556389df80da54224f01`.
Both full `strata-split-2` builds pass the existing no-peer undefined-symbol packaging check. The CMake option is off
by default, and no shared host, shared kernel or non-CUDA backend sources are changed.

Compute Sanitizer memcheck, racecheck and synccheck passed on both cards with zero errors or race warnings; memcheck
reported zero leaked bytes. [Sanitizer settings and results](sanitizer-results.json) record the changed-kernel filter
and stream-ordered race tracking. API-error reporting was disabled because the fixture intentionally accepts
`cudaGraphExecUpdate` rejection and follows the engine's reinstantiation path; unexpected API failures are checked by
the fixture itself. [Validation summaries](validation-summary.log) retain the actual tool output, with repetitive
per-case banners omitted from the sanitizer portions.

Source digests in `local-check-results.json` include the tested checkout bytes and the committed LF-normalized Git
blobs. Only line endings differ between those forms.

### Test development failures

The initial concurrent test mixed legacy-stream transfers with independent capture streams and crashed in the CUDA
driver. Switching only to nonblocking streams exposed missing transfer ordering. The final fixture orders and
synchronizes its own stream's transfers; eight preliminary repetitions and the final repeated suite passed.
The smaller-slice performance regressions above were retained and addressed with the fallback threshold.

## Reproduce

Configure a Linux CUDA SM75 build with:

```sh
cmake -S engine -B build-top64 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_SPLIT=ON -DSTRATA_SPLIT_ARCH=75 \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DSTRATA_SPLIT_SM75_TOP64=ON
NVCC_APPEND_FLAGS=--split-compile=2 cmake --build build-top64 \
  --target strata-split-2 split_top64_sm75_test -j4
ctest --test-dir build-top64 -R '^split_top64_sm75' --output-on-failure
ctest --test-dir build-top64 -R '^split_top64_sm75' --repeat until-fail:3 --output-on-failure
build-top64/src/split/split_top64_sm75_test --benchmark
```

Reconfiguring with `-DSTRATA_SPLIT_SM75_TOP64=OFF` builds the original entry point and omits the optional CUDA
translation unit. HIP, SYCL, Metal and non-SM75 sources are outside this change and were not built for this selector
report. The source build and tests were run on the machine above before opening the PR; they are not CI claims.
