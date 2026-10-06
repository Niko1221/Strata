# RX 7700 XT: calibrating the container's prompt GEMMs

This calibration is for the `./run.sh` Swift 1.5 IQ3_XXS line on an RX 7700 XT
(gfx1101, 12,272 MiB), Ryzen 9 7900 (12 physical cores), Linux, and the
`rocm/dev-ubuntu-24.04:7.2.1-complete` container. The model, quantization,
131,072-token context, 2,048-token prefill chunk, 48-slot ring, automatic
expert cache and workers, MTP, and cache-aware routing remain unchanged.
Strata's VRAM ceiling remains 10,240 MiB, including library allocations.

## References and choice

[rocm-infer at commit 033f3c7](https://github.com/kevinroy75/rocm-infer/tree/033f3c7d9ba9d737be5b2377b8b9ee0b947d1190)
uses PyTorch operations for attention, normalization and rotary embeddings;
it did not supply a HIP kernel or a matching calibration to transplant.

[AMD's hipBLASLt tuning guide](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-hipblaslt-tuning-utility.html)
searches the library's kernel pool for a given shape. Its example uses repeated
measurements and also describes rotating buffers when tuning cache behavior.
[AMD QuickTune](https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt_offline_tuning/README.html)
collects actual GEMM shapes, tests candidate solutions, and loads the chosen
solution ids at runtime. These are the applicable patterns here.

[AMD's RDNA3 WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/)
explains the wave32 register layout and FP16/BF16 matrix instructions.
[Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) has RDNA3 WMMA
kernels for other quantized formats; its formats do not directly match Strata's
GGUF IQ expert layout. [qingming-gfx1100-gemv](https://github.com/uulong950/qingming-gfx1100-gemv)
is an FP32 GEMV reference for a different card and workload. No matching public
Swift/gfx1101/hipBLASLt 1.2.2 calibration table was found in the searches.

Strata already has guarded hipBLASLt dispatch. The selected change calibrates
that path rather than adding a matrix kernel. No external kernel code was
copied. The engine source and its model arithmetic options are unchanged.

## Calibration and validation

`tools/hip/gfx1101-hipblaslt-100202.txt` contains 58 rows: 16 dense geometries
at T=128, 512 and 2048, and 10 additional FP16 expert cases at T=16..512.
The live verbose trace exposed the BF16 N=10240/K=2560 geometry missing from
the initial geometry list; it is included. The current expert path uses MMQ,
so the additional FP16 expert rows are numerical coverage of the fallback path,
not evidence that those products run in this configuration.

The tuner now accepts `--warmups`, `--repetitions`, `--iterations` and
`--max-algos`. This run requested up to 64 candidates, used three warmups and
five measurements of five or ten calls each, and kept the existing 32 MiB
workspace. The library returned fewer candidates for these shapes. Every
chosen solution required zero workspace. Tuner input was synthetic, and
repeated buffers were warm; application measurements below determine whether
the choices help the actual workload.

Every candidate was checked against hipBLASEx before timing: relative L2
error <=1e-4, maximum absolute error <=1e-2, finite outputs, and unchanged
padding. The extended `hip_prefill_hipblaslt_gemm --all` check exercised every
row with beta=0 and with beta=1 plus an offset output, using a maximum absolute
error limit of 5e-3. All 116 cases passed, with zero hipBLAS fallbacks in the
verbose summaries. The standard smoke also covers a non-tile-multiple tail.
Different GEMM reduction orders can change generated tokens; these checks do
not establish bitwise model equivalence or a broad quality benchmark.

The default tail/cache smoke passed its seven cases (1,007 Lt launches,
zero fallbacks), in addition to the 116 exact-row cases above. The runtime
contract suite passed 6/6 checks. Seven selection checks covered the default,
other model, quantization and architecture, explicit disable/override, and
a different library digest. The runtime image rebuilt successfully without
recompiling the engine.

The broader Docker suite passed 28 checks and had six errors from absent
historical benchmark fixtures. `tools/test_setup_amd.py` passed 27/28 checks;
its existing `test_prebuilt_hip_zip` failed in the unmodified ZIP installer
path. These suites did not fully pass in this checkout.

The table is scoped to gfx1101 and hipBLASLt version 100202. The container
selects it automatically only for Swift IQ3_XXS and the measured library:

```text
libhipblaslt.so SHA-256
c40df6fe45de5ae3ee60eccf5885536b21486cfff7d361ccdf4955a1db971c1f
```

An explicit `STRATA_HIPBLASLT_TUNING` wins; an empty value disables tuning.
The engine separately checks architecture, version, actual dimensions,
strides, beta and workspace, and retains its supported fallback behavior.

## Application measurements

Measured on 2026-10-05 (EDT); raw results are in
`bench/results/2026-10-05-gfx1101-gemm/`. Controls and candidates use the same
engine binary (SHA-256
`f338acfdeb31b9e4f216f6f8e0504d3a50bf31a59a255beaa0461c2731a33cc2`).
The A/B/A/B arms restart through `./run.sh`; each runs four fresh coding
prompts alternating 4,210 and 8,830 tokens, plus cached follow-ups.
`tools/hip/bench_prefill.py` rejects reuse on fresh prompts and records actual
generated tokens. Sampling is greedy, reasoning is disabled, and the output
cap is 128 tokens. CPU/GPU profiling is disabled for these throughput arms.
First-use and later requests are reported separately. File pages can be warm.

The 50 ms VRAM audit counts AMD DRM allocation totals for the engine process,
deduplicated by client. It includes library allocations; desktop VRAM is
recorded separately as the raw-card total. Zero tolerance is used.

The two control and two candidate starts gave the following means (four
fresh samples per length per arm; two first-use and two later 4K requests):

| Fresh prompt | Plain hipBLAS | Calibrated table | Prompt-speed gain | Request wall time |
|---|---:|---:|---:|---:|
| 4,210 tokens | 227.1 tok/s | 313.5 tok/s | +38.1% | 22.03 -> 16.84 s |
| 8,830 tokens | 250.1 tok/s | 365.2 tok/s | +46.0% | 38.93 -> 27.74 s |

The raw 4K control speeds were 212.4/258.4/212.4/225.0 tok/s, candidate
290.6/354.1/280.9/328.4. The 8K controls were 257.3/269.2/237.4/236.5,
candidate 370.4/378.9/365.1/346.4. The mean decode rates were 38.2 -> 39.0
at 4K and 36.7 -> 36.4 at 8K; no decode gain is claimed. Generated text
and draft acceptance can differ. Cached follow-ups all reused state, but
are not used as comparable fresh-prompt throughput evidence.

The four short-arm VRAM audits passed: controls peaked at 8,689.5 and
8,689.2 MiB of Strata allocation, candidates 8,734.3 and 8,734.0 MiB.
The largest raw-card total in the short candidate arms was 10,101.0 MiB,
including the desktop. No budget, runtime reserve, or slack was raised.

The rebuilt container selected the table without an explicit tuning setting.
Its generated `merge_intervals` implementation passed 205 input cases. A
separate token-counted streaming probe completed the following fresh prompts
with no reused state, then continued each conversation with cached state:

| Actual fresh tokens | Prompt speed | Time to first token | Cached follow-up time to first token |
|---|---:|---:|---:|
| 4,096 | 347.9 tok/s | 11.83 s | 0.50 s |
| 32,768 | 401.4 tok/s | 81.78 s | 0.59 s |
| 65,536 | 394.3 tok/s | 166.46 s | 0.71 s |
| 130,944 | 365.9 tok/s | 358.34 s | 0.92 s |

These are single validation samples, not comparisons against the control.
Every response was `READY` with a normal stop; the longest follow-up had
130,963 prompt tokens. The server still advertised a 131,072-token context.

Two recall probes found their hidden code words at 50% depth (4,102 and
32,078 actual prompt tokens). The combined coding, long-context and recall
VRAM audit passed over 730 seconds / 14,329 samples: Strata allocation peaked
at 8,735.0 MiB (8.53 GiB), and raw-card usage including the desktop peaked at
10,101.7 MiB (9.87 GiB). There were no sampled budget breaches. As with any
50 ms sampler, this does not rule out shorter unsampled spikes.

## Reproduce

Use the same library build and the actual discrete render node; this machine
also has an integrated gfx1036 device. Build `tune_hipblaslt` and
`hip_prefill_hipblaslt_gemm` inside `strata-hip-builder:gfx1101`. Calibration
commands, shape JSON, raw candidate measurements, and the arm runner are in
the results directory. The table header records its library digest.

Compare the packaged defaults with plain hipBLAS by restarting between arms:

```sh
./run.sh --offline --detach
# Benchmark the idle server and audit its VRAM, then stop it before the next arm.
./run.sh --offline --detach -e STRATA_HIPBLASLT_TUNING=
```

For another ROCm build, recalibrate instead of reusing solution ids. A template
invocation inside the matching builder is:

```sh
/src/build-hip/tune_hipblaslt --shapes-file /src/shapes.json --tokens 128,512,2048 \
  --warmups 3 --repetitions 5 --iterations 10 --max-algos 64 \
  --workspace-mib 32 --tuning-out /src/table.txt
STRATA_HIPBLASLT_TUNING=/src/table.txt STRATA_HIPBLASLT_VERBOSE=1 \
  /src/build-hip/hip_prefill_hipblaslt_gemm --all
```

Read the validation summary as well as its exit code: `fallbacks=0` establishes
that the calibrated path ran. Repeat end-to-end measurements and the VRAM audit
before retaining another table.
