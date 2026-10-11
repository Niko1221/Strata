# Active-layer cache: P4 validation

The [cross-GPU follow-up](cross-gpu/README.md) adds three-repeat 4K/8K/12K
measurements, default-off regression checks and CUDA/HIP/SYCL builds. It completes
the backend builds listed as pending in the initial PR description. This page
preserves the initial experiment and its original sample counts.

Measured 2026-10-09. This is validation evidence for the opt-in native change,
not a general performance claim. It was developed on public #1667 at `378f3f58`
and committed as `2d2096c1`, then extracted onto main (`fb58e0db`) without server
changes. The native sources are identical. The unmodified reference is `fb58e0db`.

## Setup

- One Tesla P4 (GPU 0, PCIe x16, 7680 MiB); the other cards were idle.
- Dual Xeon E5-2697 v3; 251.773 GiB RAM; Ubuntu 24.04; driver 580.178.04;
  CUDA 12.0 and GCC 12. Disk model and enforced GPU power limit were not recorded.
- Model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` and
  `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf`. Repository revision
  was not recorded. Native mmap experts, warm OS page cache, PLE in locked RAM.
- 16K context, int8 KV, 27 CPU workers plus host, NUMA interleave, 64 GiB memlock,
  MTP spec 2, temperature 0, 64 generated tokens, static expert placement,
  CPU expert share 0, no prompt or conversation-prefix reuse. Same existing
  expert-ranking profile in both arms; this PR adds no profile or profile policy.
- Automatic prefill selected 1792-token chunks. Cache budget settings use defaults:
  8192-token target window, up to 1536 MiB, 256 MiB free-VRAM margin. Actual
  capacity shrinks with free VRAM; the warmed 4K runs had about 434.6 MiB available.

The [off](off-config.json) and [on](on-config.json) configurations and complete
engine logs ([off](off.log), [on](on.log)) record startup settings and request
timings. User and asset-directory paths are normalized; measurement values and
token IDs are unchanged. No private persona or prompt content is included.

## Measurements

Steady-state 4K used the unmodified reference for the off arm. Each arm first
processed 512 tokens and generated 64 tokens to capture MTP graphs. Two measured
repetitions per arm, existing automatic chunk sizing in both:

| Arm | Prompt durations (ms) | Prompt tok/s | Median tok/s |
| --- | --- | --- | ---: |
| Off | 37967.5, 37791.1 | 107.881741, 108.385308 | 108.133524 |
| On | 35479.4, 35254.2 | 115.447274, 116.184738 | 115.816006 |

Observed gain: **7.105%**. [steady-state.json](steady-state.json) includes exact
prompt IDs, output IDs, per-request log excerpts and cancellation results.
Model loading is excluded from engine prompt timing.

Initial automatic-chunk comparisons used a 512-token warmup with only one output
token, followed by 4K and 8K requests. One measured request per length and arm:

| Tokens | Off / on prompt ms | Off / on tok/s | Gain | Off / on expert H2D bytes |
| ---: | --- | --- | ---: | --- |
| 4096 | 37912.8 / 35709.1 | 108.037391 / 114.704655 | 6.171% | 105829222400 / 53896192000 |
| 8192 | 71185.7 / 66813.1 | 115.079293 / 122.610686 | 6.545% | 193275468800 / 98652032000 |

[initial.json](initial.json) preserves all initial runs, including a fixed-512
8K cached-only sample that is excluded from paired speed claims. The initial 4K
request precedes MTP graph capture; later requests have less free VRAM. Do not
pool initial and steady-state samples. Expert H2D counts instrumented prefill
weight copies, including cache admission, not disk reads or total PCIe traffic.
Approximately 49% fewer transferred bytes produced a smaller wall-time gain.

## Correctness and integration

- All paired 4K/8K automatic-chunk outputs match all 64 generated token IDs.
  All four steady-state outputs match the initial 4K outputs and the unmodified
  main reference. This establishes equality for these cases, not every prompt.
- Fixed-512 smoke: one-chunk fallback and active 4K outputs match all 64 tokens.
  All 64 sampled 4K final-residual rows are bit-identical (2,621,952 bytes including
  positions; SHA-256 `f02eab4f83839a7b86d1431e260f008bdc2a6ba94bed1ddd37439a237c8a1afa`).
  At 8K automatic chunks, all 16 sampled rows common to the baseline dump and
  the final cached window are bit-identical. The request spans multiple windows.
  Large residual dumps are retained locally, not included here. The first smoke
  build predates only the diagnostic transfer counter and defensive error guard.
- Cancellation returned in 2.406 seconds with no output tokens; the same engine
  then returned the expected first token `27775` on a new 512-token request.
- Integration with #1667: all eight requests passed, including a restore with
  4089 reused plus 1601 fresh tokens, branching without modifying parent history,
  protected-only restart retention, and replay after deleting isolated test cache
  blocks. This used 512-token chunks, 2048-token cache windows, a 512-token
  checkpoint interval, 8K context and MTP spec 4. Synthetic three-code recall was
  exact. Checkpoint timings in [summary.json](summary.json) have different input
  lengths and are not an off/on speed comparison.
- Companion #1667 Python suite: 98 run, 96 passed, 2 skipped for absent optional
  OpenAI SDK/jsonschema dependencies. These server tests are not added by this PR.

## Reproduction and limits

CUDA Release build used `CMAKE_CUDA_ARCHITECTURES=61`, GCC/G++ 12,
`STRATA_ENABLE_CUDA=ON`, `STRATA_EXPERIMENTAL_SM60=ON`, `STRATA_BUILD_TESTS=OFF`.
Tested engine SHA-256:
`54f42260a7f93f9f5212fe9fff3d0d2f53bf9f084ca300cad9e447200dfafd57`.
Tested prefill.cpp SHA-256:
`2ed5d4bf406af39251b232dfdb1a72408b74de1b7d4293d78f9e359531aa253e`.

Update asset/executable paths in the attached configuration for your machine,
then run from the repository using its Python environment:

```bash
python tools/bench_prefill_layer_cache.py --config on-config.json \
  --output new-results --tokens 4096 --chunk auto --repeats 2 --generate 64 \
  --cancel-test --reference-exe /path/to/unmodified-fb58e0db/strata
```

The portable harness defaults to three repetitions at 4K and 8K when those
options are omitted. It generates the synthetic maintenance-notes prompt and
records its exact token IDs. Checkpoint integration additionally requires #1667;
native prefill and its benchmark do not.

This initial experiment built and exercised CUDA only; the follow-up above
covers the later HIP/SYCL builds and additional GPUs. Larger contexts, other
quants and other models were not validated here. Multi-GPU, images and unsupported layouts decline this experimental
path. Performance depends on free VRAM, chunk size and expert reuse. No default
is changed.
