# Cross-GPU validation, 2026-10-09

Follow-up to the initial P4 measurements, with three repetitions per arm at
4,096, 8,192 and 12,288 fresh prompt tokens. Opt-in active-layer caching is
compared with an unmodified `fb58e0db` reference on each host. Candidate native
implementation matches PR #1704 at `76ce987a`; no engine changes were needed during this
validation. The P4 reused the previously validated binary built on #1667, whose
native implementation matches the standalone PR.

## Prompt throughput

Values are median tok/s, with the full three-run range in parentheses. Each host
is its own comparison; CPU, host memory and I/O settings differ between hosts.

| GPU | Prompt tokens | Off median (range) | On median (range) | Change |
| --- | ---: | --- | --- | ---: |
| Tesla P4 | 4096 | 108.541 (108.201-108.587) | 116.162 (115.366-116.243) | +7.021% |
| Tesla P4 | 8192 | 115.240 (115.120-115.477) | 122.832 (122.790-123.057) | +6.589% |
| Tesla P4 | 12288 | 115.253 (115.243-115.306) | 131.061 (131.055-131.426) | +13.716% |
| RX 5500 XT | 4096 | 59.008 (57.528-62.069) | 45.128 (41.283-45.203) | -23.523% |
| RX 5500 XT | 8192 | 52.191 (49.571-53.070) | 42.692 (42.025-43.470) | -18.201% |
| RX 5500 XT | 12288 | 57.667 (56.964-59.839) | 54.762 (53.948-56.967) | -5.036% |
| RTX 3070 | 4096 | 521.850 (499.092-524.765) | 485.233 (469.246-485.492) | -7.017% |
| RTX 3070 | 8192 | 527.129 (524.140-529.606) | 617.928 (614.314-618.442) | +17.225% |
| RTX 3070 | 12288 | 529.678 (529.575-529.886) | 584.455 (583.914-584.650) | +10.342% |

The RTX 3070 4K regression is included. This remains opt-in; these measurements
do not justify enabling it for every request size. Exact per-run durations,
prompt IDs, output IDs and logs are in [P4](r730.json), [RX 5500 XT](a5500.json)
and [RTX 3070](g3070.json). [summary.json](summary.json) contains computed
medians/ranges. No failed or cancelled request is included in throughput medians.

## Expert-transfer diagnostic at 4K

The off counter comes from the separate candidate-with-cache-disabled correctness
check; the unmodified reference has no counter. The on counter is the first
primary cache-on 4K sample. Both produce identical outputs on that GPU.

| GPU | Off expert H2D (GB) | On expert H2D (GB) | Reduction |
| --- | ---: | ---: | ---: |
| Tesla P4 | 105.829222 | 54.287283 | 48.703% |
| RX 5500 XT | 64.628275 | 45.392256 | 29.764% |
| RTX 3070 | 85.218406 | 46.951552 | 44.904% |

GB is decimal. These count expert-weight copies, including admission, and exclude
residual copies and other PCIe traffic. Reduced expert traffic alone does not
establish a wall-time gain: the RTX 4K timing regressed despite fewer expert bytes.

## Conditions

- Model: Qwen3.8-Flash-Next-GSQ-RCO IQ3_XXS, the two GGUF shards named in the
  attached configurations. Native mmap experts, int8 KV, 16K context, MTP spec 2,
  temperature 0, 64 generated tokens, no prefix reuse, static expert placement,
  CPU expert share 0. Same configuration within each off/on pair apart from
  executable and the cache flag. The existing expert-ranking profile is fixed.
- Tesla P4 7680 MiB, dual Xeon E5-2697 v3, 251.773 GiB RAM, CUDA 12/GCC 12,
  sm_61. 27 CPU workers plus host; NUMA interleave; PLE locked in RAM; 64 GiB
  memlock allowance. Both arms selected the warm-page-cache RAM staging profile.
- RX 5500 XT 8 GB, Ryzen 5 3600, about 31.3 GiB RAM, HIP 5.7/Clang 17,
  gfx1012. Five CPU workers plus host; PLE direct I/O. HIP MMQ is enabled at
  build and runtime. Both arms used the default staging profile; the full model
  does not fit in host RAM. Timings include the resulting storage activity.
- RTX 3070 8 GB, Ryzen 7 5800X, about 62.7 GiB RAM, CUDA 13.2, sm_86.
  Seven CPU workers plus host; PLE direct I/O. Reported primary measurements
  are the second full sweep, with both arms selecting RAM staging. Its first
  sweep is preserved separately: the reference selected SSD/default staging
  before the files warmed, while the candidate selected RAM staging. That
  unmatched sweep is excluded from speedup claims, including its initially
  observed approximately 66% 8K gain.
- Automatic chunk sizing and default layer-cache budgets. The P4 selected
  1792-token chunks and a 7168-token window; RTX selected 2048-token chunks and
  an 8192-token window. AMD selected 3072-token chunks, rounding the default
  8192-token target down to a 6144-token window (two chunks); this run did not
  tune that window. Startup and model loading are excluded from engine
  prompt durations. Each engine first processed 512 tokens and generated 64
  tokens to allocate MTP graphs. Requests used synthetic maintenance notes.

## Checks and builds

All 54 primary measured requests (18 per GPU) produced the same 64 output tokens
as the other arm and repeats on that GPU. This is equality within each backend,
not a claim of cross-backend equality. All measured cache-on requests activated
the scheduler. The initial RTX sweep also matched, and its outputs match the
warmed sweep. Each GPU passed cancellation/recovery and a separate 4K check that
the candidate with the flag off matches its unmodified reference.

CUDA (sm_61 and sm_86), HIP (gfx1012 with MMQ) and SYCL engine builds passed.
SYCL used Intel oneAPI DPC++ 2026.1.1 plus oneMKL 2026.1, compiling the `strata`
target; [build identity](sycl-build.json). No Intel GPU was available, so no
SYCL inference claim is made. The SYCL backend retains its existing scheduler;
this change's shared header was checked by compiling that backend.

The unmodified HIP reference first failed to link because the installed HIP 5.7
BF16 header defines six host helpers without `inline`. A local header overlay
added `inline` to those same six functions for both reference and candidate;
the installed headers and PR engine code were not changed. The overlay's
original/patched SHA-256 values and function names are recorded in `a5500.json`.
Its include directory was passed through `CMAKE_HIP_FLAGS=-I/path/to/overlay`.
Both HIP builds used the repository-pinned llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`.

Reproduce the primary sweep with `tools/bench_prefill_layer_cache.py --config
CONFIG --output NEW_DIRECTORY --tokens 4096 8192 12288 --chunk auto --repeats 3
--generate 64 --cancel-test --reference-exe UNMODIFIED_BINARY`. For a RAM-staging
comparison, warm the model files before starting either engine and confirm the
same staging profile in both startup logs. The portable harness's small MTP
warmup alone does not guarantee a warmed filesystem profile.
The harness now rejects differing automatic staging profiles. That guard was
checked against these actual logs: the matched P4/RTX sweeps pass, and the
initial unmatched RTX sweep is rejected. This changes result validation only;
the engine source used for the GPU runs is unchanged.

User and asset-directory paths are normalized in published evidence. No private
persona content, service configuration changes or new default settings are
included. Multi-GPU scheduling, image prompts, larger contexts and other model
quants are outside this validation.
