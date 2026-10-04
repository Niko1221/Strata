# Strata-V100

**A community-maintained V100-focused fork of [Strata](https://github.com/Niko1221/Strata).** This fork keeps NVIDIA Volta (`sm_70`) support working, develops and measures performance changes on Tesla V100 hardware, and periodically merges upstream improvements for features and fixes beyond V100.

Strata is an open-source local runtime for [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). It combines GPU, system memory, and storage to run this large model on a personal computer. This fork retains the broader Strata project: other supported NVIDIA and AMD hardware, the web interface, OpenAI-compatible API, tools, image input, and upstream features remain important too.

> **Huge credit to Niko ([@Niko1221](https://github.com/Niko1221)) and the original Strata contributors.** This project builds on their work: Strata's model runtime, architecture, features, and the work that made running this model locally possible all come from the upstream project. The V100 fork would not exist without that foundation. Thank you for making it possible, and for continuing to develop Strata.

[Upstream Strata](https://github.com/Niko1221/Strata) · [Issues](https://github.com/jmnargi/Strata-V100/issues) · [Pull requests](https://github.com/jmnargi/Strata-V100/pulls) · [Discussions](https://github.com/jmnargi/Strata-V100/discussions)

## What this fork focuses on

- **Tesla V100 / Volta support (`sm_70`)**: preserve normal V100 builds and runtime support. V100 builds use a CUDA 12.x toolkit; CUDA 13 does not compile `sm_70` code.
- **Test machine:** two Tesla V100-PCIE-16GB cards; one uses PCIe Gen3 x4 and the other Gen3 x16. The benchmark below uses only the x16 card. A system with both cards on full Gen3 x16 links could perform better, especially for multi-GPU workloads, but that configuration has not been measured here.
- **Upstream integration**: this is not a separate replacement for Strata. Upstream changes are merged periodically, with V100 compatibility checked and corrected where needed. New upstream features may have hardware requirements above `sm_70`; they do not automatically accelerate V100.
- **The wider Strata community**: improvements, bug reports, testing, documentation, and contributions for other supported platforms are welcome.

Support for V100 does not imply that every upstream feature or every other GPU configuration has been tested on V100. Check the relevant code, release notes, and measurement reports before relying on a hardware-specific feature.

## Upstream v0.1.39 integration

This update integrates [Strata v0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39), including the intervening releases. It keeps the fork's V100 tensor-core experts, attention changes, asymmetric GPU split, and disk conversation cache.

- **Parallel requests:** set `"parallel": 2` in the model configuration, or use `./setup.sh --parallel 2`. Setup writes this default for a new Volta configuration and preserves an existing setting. The engine can decode separate conversations together. Extra requests wait for a free slot. Slots use additional GPU memory and can reduce single-request speed; if two slot sessions do not fit, the engine says so and serves one request at a time, with the configuration unchanged. See [batch slots](docs/BATCHING.md) for exactness settings, memory costs, and limits.
- **APIs and interface:** the update adds the OpenAI Responses API, model settings, conversation-cache monitoring, and per-slot metrics. See [API details](docs/DETAILS.md#using-it).
- **Performance and memory:** the update includes verify-pass launch reductions, byte-sized prompt rings, Linux huge pages, RAM-budget corrections, and multi-GPU options. An explicit `--layer-split` now loads only each stage's own layers' dense weights automatically (the upstream `--trim-stage-weights` flag and `STRATA_STAGE_TRIM` are gone); the freed VRAM goes to the expert cache. An optimization for newer GPUs does not imply a V100 speed improvement. Optional paths that change rounding must be measured before use.
- **Other platforms:** the upstream Intel Arc, older CPU, and additional AMD paths remain available. This machine does not provide hardware validation for those platforms.

V100 requires CUDA 12.x and a build that includes `sm_70`. Do not install the CUDA 13 engine on a V100. Explicit parallel configuration is recommended for existing installations; an update does not overwrite the model configuration.

**Measured on the two V100 cards (4 October 2026, final build).** A controlled before/after
run of the integrated build against the previously installed runtime, one request at a time
(no batch slots), cold, Qwen3.8-Flash-Next Q2_0 with int8 KV at the 524,288-token context,
the 20/28 layer split and MTP (`--spec 8`) unchanged; two fresh requests per size, medians:

| Target prompt | Before (tok/s) | Integrated (tok/s) | Change |
| ---: | ---: | ---: | ---: |
| 2K | 701.41 | 718.5 | +2.44 % |
| 8K | 1,402.82 | 1,458.0 | +3.93 % |
| 32K | 2,262.4 | 2,283.7 | +0.94 % |

Each value is the median of two fresh requests (`reused=0`). The gains are small; two repeats
per size are not a statistical claim. All six paired first tokens matched the baseline's.
The full replies are not claimed to be identical: the adaptive expert tier keeps a different
mix of experts on the GPU and the CPU for the integrated build, and the two paths round
differently. No decode or quality change is claimed. These are the final build's measured
values (engine SHA-256 `e9ad337a…`), reported in the
[full report](benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md).

Parallel slots at the 524,288-token context depend on the split and the memory left on the
last stage; the measured fallback (requested 2, served one request at a time, configuration
untouched) is described in [docs/BATCHING.md](docs/BATCHING.md).

## V100 benchmark

The table below records a single-card run on one Tesla V100-PCIE-16GB using PCIe Gen3 x16. The test machine also has a second V100 on PCIe Gen3 x4; it was not used for this run. Results from a two-card setup or from a machine with two full-width links may differ and could be better. It is a reproducible measurement, not a promise of speed on other systems.

| Target prompt | Median prompt tokens | Prefill (tokens/s) | Decode (tokens/s) | Maximum GPU temperature |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,036 | 398.6 | 52.0 | 56 °C |
| 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
| 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
| 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
| 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
| 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
| 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
| 256K | 256,080 | 617.8 † | 36.8 | 84 °C |

**Test setup:** Qwen3.8-Flash-Next Q2_0; one V100 only (GPU1, PCIe Gen3 x16); Ryzen 5 3600; 48 GB DDR4-3200; CUDA 12.8; 262,144-token context; int8 KV cache; automatic prefill; MTP with `--spec 8` and draft floor 0.70; paired expert variant; 700 MiB vision reserve. Each value is the median of three fresh, uncached requests (`reused=0`) with exactly 256 output tokens. Timings are from the engine's `/metrics` endpoint.

Before each request, the passively cooled card idled for at least 120 seconds and reached 55 °C or below without software thermal slowdown for 15 seconds. The starting temperatures were 51–55 °C. All requests are included. The 128K and 256K prompts reached 84 °C and experienced software thermal slowdown during prompt processing, despite the cooled start; long runs can throttle. † indicates slowdown during the prompt.

Measured 2026-10-02. Engine: fork version 0.1.31, build commit `78417ea` (SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`). The benchmark was recorded before the later upstream v0.1.36 integration and V100 attention PRs; it is not a measurement of the current `main` build. The historical table is kept with its original provenance. Do not interpret it as a controlled comparison against earlier tables or current builds.

Raw data: [`summary.json`](bench/results/2026-10-02-v100-single-cooled/summary.json), [`protocol.json`](bench/results/2026-10-02-v100-single-cooled/protocol.json), [`matrix.json`](bench/results/2026-10-02-v100-single-cooled/matrix.json), [`completed-requests.json`](bench/results/2026-10-02-v100-single-cooled/completed-requests.json), and [`GPU telemetry`](bench/results/2026-10-02-v100-single-cooled/gpu.csv). See the [full method and history](docs/DETAILS.md#tesla-v100-fork-benchmark), [benchmark instructions](benchmarks/README.md), and [V100 performance reports](benchmarks/).

Recent V100 pull requests include measured kernel-level changes and explicitly report when end-to-end gains are not established. For example, the prompt-attention port from upstream PR #600 reported 27.9% less int8 attention kernel time at its measured shape, while model prefill changes ranged from +1.3% to +3.0% in the reported runs; it did not establish a reliable decode gain. Read the [full report](benchmarks/v100-q2_0-pr600-2026-10-03.md) before comparing results.

## V100 performance work

The V100 work spans the prompt path, decode path, memory use, storage reads, and multi-GPU execution. The changes below are in the fork's merged history; each link includes its own test method, hardware details, and limitations. Kernel timing improvements do not necessarily produce the same percentage gain in full-model throughput.

### Prompt processing

- **BF16-to-FP16 tensor-core GEMMs on Volta:** Volta does not have native BF16 GEMM support. The fork converts BF16 weights exactly to FP16 and uses V100 tensor cores for prompt GEMMs instead of the scalar fallback. The associated FP16 prompt activation images use the right bit representation.
- **Faster PLE table reads:** switch POSIX PLE reads to queued `io_uring` O_DIRECT operations, with a synchronous fallback. One measured fresh 8K PLE gather dropped from about 6.9 seconds to 7 milliseconds.
- **Q2_0 MMQ prefill:** enable the existing llama.cpp Q2_0 matrix-matrix quantized path for Strata's internal expert type. A measured 14,486-token prompt improved 6.5% in prefill throughput (926.1 to 986.2–986.9 tokens/s).
- **Score only live QSA blocks:** avoid scoring blocks beyond the current prompt's live range. On a measured 15,423-token prompt, QSA selection time fell 91.1%; end-to-end prefill throughput rose from 961.0 to 1,111.3–1,112.9 tokens/s in the reported tests.
- **Tensor-core prompt attention for Volta:** use `mma.m8n8k4` on the V100 instead of the FP32 fallback. An earlier PR measured roughly 1.6–1.9× speed in the attention kernel and higher prompt throughput in its tests. The later upstream PR #600 port reduced int8 attention kernel time 27.9% at one measured shape; reported end-to-end prefill deltas were smaller (+1.3% to +3.0% in its 4K–32K tests).

### Decode and multi-GPU execution

- **Batched verification window:** run multiple speculative decode positions together, reducing per-token launch overhead. Paired historical tests reported faster output generation, but results vary by prompt and runtime conditions.
- **Tiled QSA block scoring:** reuse selected key data across query tiles instead of repeatedly reading it. The measured 256-query / 65K-block kernel fell from 3.5 ms to 1.3 ms; its per-score arithmetic remained bit-identical.
- **Fused Volta GR and batched KV append:** combine work across verification positions and append KV data in fewer launches. On the tested dual-V100 setup, decode increased 5.6–17.3% across 4K–32K prompts, while prefill was approximately unchanged.
- **Asymmetric dual-V100 tuning:** improve host-row handling, device-plan dispatch, and memory ownership for a two-card configuration. One earlier installed configuration showed prefill gains of 13.3–18.7% at 4K–32K, but decode results were mixed and both GPUs thermally slowed during that run. These values do not isolate each code change.
- **Reduce weight and dispatch overhead:** load only the layers owned by each card when using an explicit split, avoid unnecessary CPU activation quantization and empty dispatch barriers, and optionally run paired-row expert kernels. GPU0's measured native dense allocation fell from 1,376.20 MiB to 541.27 MiB. The paired-row mode is opt-in; the reported whole-model results did not establish a decode improvement.

### Attention kernels and memory traffic

- **Decode attention (upstream PR #540 port):** reduce score shuffles and shared-memory traffic while preserving accumulation order. The measured kernel time fell from 52.90 to 31.13 microseconds on GPU0 for M=1, and from 180.42 to 111.85 microseconds for M=8. The reported whole-model decode changes were small; prefill was effectively unchanged.
- **Prompt attention (upstream PR #600 port):** use the Volta tensor-core implementation with four independent MMA computations. The PR reports correctness checks and kernel-level / model-level measurements; the 27.9% kernel reduction should not be read as a 27.9% increase in full-model speed.
- **KV gather and GDN recurrence (upstream PR #627 ports):** use wider aligned int8 KV loads with a safe original-width fallback, and use multiple accumulators for the V100 prompt recurrence. Measured kernel times improved about 10% for a 32K KV gather and about 7% for a 4,096-token recurrence. Paired model prefill differed by less than 0.2%; no reliable end-to-end decode gain was established.

### Build support and evidence

The fork has also repaired CUDA configuration and linking for `sm_70`, maintained V100 device admission through upstream merges, and adapted upstream changes when their architecture requirements exclude Volta. These compatibility fixes keep the optimized paths buildable; they are not themselves performance claims.

For the full evidence and implementation scope, see [PR #1](https://github.com/jmnargi/Strata-V100/pull/1), [#2](https://github.com/jmnargi/Strata-V100/pull/2), [#5](https://github.com/jmnargi/Strata-V100/pull/5), [#6](https://github.com/jmnargi/Strata-V100/pull/6), [#11](https://github.com/jmnargi/Strata-V100/pull/11), [#12](https://github.com/jmnargi/Strata-V100/pull/12), [#15](https://github.com/jmnargi/Strata-V100/pull/15), [#16](https://github.com/jmnargi/Strata-V100/pull/16), and [#17](https://github.com/jmnargi/Strata-V100/pull/17). The results use different dates, builds, prompts, GPU placements, and protocols. They are historical measurements, not a single controlled before/after comparison or a guarantee of gains on every V100 system.

## Get started

This fork tracks the upstream installation experience. For complete and current platform requirements, model choices, and options, see the [installation guide](docs/INSTALL.md), [model guide](docs/MODELS.md), and [upstream setup guide](docs/AI_SETUP.md).

- **Windows:** download or clone this repository and run `START-HERE.bat`.
- **Linux:** clone this repository and run `./setup.sh`.
- Follow the prompts to select a model and context size. Setup downloads model data and starts the local service.
- Open the address printed by setup (normally `http://127.0.0.1:8080`).

For a Tesla V100, use the CUDA 12 engine with `sm_70` support. Setup selects the older-GPU engine for Volta; Linux builds it locally. The upstream Windows CUDA 12 archive includes Volta code, but it was not tested on this Linux machine. See the [older-GPU guide](docs/OLDER_GPUS.md) for toolkit and driver requirements. For Docker, follow the [Docker instructions](docs/INSTALL.md) and build for the target GPU architecture.

## Use Strata

- **Web app:** use the local URL printed by setup for chat and the live monitor.
- **OpenAI-compatible API:** set your app's base URL to `http://127.0.0.1:8080/v1`.
- **Anthropic-compatible API:** use `http://127.0.0.1:8080/v1/messages`.
- **OpenAI Responses API:** use `http://127.0.0.1:8080/v1/responses`. This stateless endpoint supports Codex-style tool calls; it does not store `previous_response_id`.
- **Images, MCP, multi-GPU, and configuration:** see [details](docs/DETAILS.md), [MCP server](docs/MCP_SERVER.md), and [multi-GPU guide](docs/MULTI_GPU.md).

The server normally listens on localhost. If you expose it to other machines, configure an API key and use a trusted network. Do not publish secrets or private configuration in issues, benchmark results, or pull requests.

### Conversation cache disk (opt-in)

The conversation state has three tiers. Tier 1 (L1) is the live session in GPU memory; it always exists. Tier 2 (L2) is the optional host-RAM cache (`--conversation-cache-mib`). Tier 3 (L3) is an optional disk store that keeps parked conversations in files across restarts.

To use the disk store, add `--conversation-cache-disk strata-conversations --conversation-cache-disk-gib 25` to the engine arguments. The path and a positive GiB budget are required together. The store is off by default, it needs `--serve`, it works with `--conversation-cache-mib 0`, and it supports `--layer-split` on multiple GPUs. Optional: `--conversation-cache-disk-slots N` caps the record count, and `--conversation-cache-disk-min-free-mib N` keeps free space on the filesystem.

A record is reused only when the prompt starts with exactly its tokens and images and its control-vector mode matches. The engine picks the longest valid prefix across the live session, the RAM cache, the disk records and the idle batch slots, so a shorter disk record never displaces a longer resident state. The files are checksummed and versioned, the engine writes each one through a temporary file, and it removes invalid files at startup. The store evicts the least recently used record when it reaches its byte budget, its record cap, or the free-space floor. A corrupt or incompatible record is removed, and the request falls back to normal prompt processing.

The disk tier accelerates conversation alternation and server restarts. It does not itself add concurrent execution. Use `"parallel": 2` for parallel requests; see [batch slots](docs/BATCHING.md) for memory costs and limits. GPU capture must complete before the active session is overwritten, but the file write runs asynchronously while the next request uses the GPUs. Restore is synchronous. Save and restore stage the record in host RAM, and the engine releases that memory after the file operation. Each park writes a full record to the storage device, so frequent switching consumes flash write endurance; stop the server and delete the directory to clear the cache. Size the GiB budget for the conversations you want to keep, not for one record. See [details](docs/DETAILS.md#using-it) for the full behavior, limits, and sizing guidance.

#### Measured V100 result

On this fork's two-V100, layer-split configuration, resuming a 33,725-token conversation from a 25 GiB NVMe cache took 1,614.6 ms to read and 315.8 ms to restore. The complete resumed prompt phase, including 22 new tokens, took 3,015.1 ms. A cold read of the same 33,725-token prefix took 20,332.7 ms. This is a **6.74x speed-up** and an **85.2% prompt-latency reduction** for the resumed request. The 943.6 MiB record restored byte-exact main-model state and identical output across both GPUs. Restart recovery passed byte-exact parity, and corrupt-record fallback produced matching output. See [the benchmark report](benchmarks/v100-l3-conversation-cache-2026-10-03.md).

## Contributing

Contributions are welcome. You do not need a V100 to help: documentation, tests, setup, server behavior, API compatibility, and improvements for other supported devices are useful. V100-specific code and performance results benefit from validation on real Volta hardware.

1. Check [open issues](https://github.com/jmnargi/Strata-V100/issues) and [pull requests](https://github.com/jmnargi/Strata-V100/pulls) to avoid duplicate work.
2. For broad upstream features, check whether the change belongs in [Niko's upstream repository](https://github.com/Niko1221/Strata). This fork periodically integrates upstream changes; focused V100 fixes and measurements can be proposed here.
3. Make a focused change, describe the hardware and exact steps used to test it, and include relevant tests.
4. For performance claims, report the baseline and candidate, full runtime settings, workload, number of repetitions, and limits. Include raw data or a reproducible command when possible. Separate kernel timing from whole-model throughput and do not claim a gain that the measurements do not show.
5. Open a pull request against this repository's `main` branch. Explain whether the change is V100-specific, an upstream integration, or a general Strata improvement. Keep credentials, personal configuration, and unrelated local files out of the change.

See [`benchmarks/README.md`](benchmarks/README.md) and [`docs/DETAILS.md`](docs/DETAILS.md) for the current measurement approach. Contributions remain subject to the project license and the licenses of included components and model files.

## Project history

This fork develops V100 support and V100-specific work while retaining Strata's upstream development. Recent integration and performance pull requests illustrate that process:

- [#14 — Merge upstream v0.1.36 while preserving V100 support](https://github.com/jmnargi/Strata-V100/pull/14)
- [#15 — Port measured V100 KV gather and GDN recurrence changes from upstream PR #627](https://github.com/jmnargi/Strata-V100/pull/15)
- [#16 — Reduce V100 decode-attention shuffles and shared-memory traffic](https://github.com/jmnargi/Strata-V100/pull/16)
- [#17 — Port upstream PR #600's Volta prompt-attention kernel](https://github.com/jmnargi/Strata-V100/pull/17)

These reports describe specific tested changes; they are not a blanket claim of faster performance across models or GPUs. For more detail, browse the [complete pull request history](https://github.com/jmnargi/Strata-V100/pulls?q=is%3Apr+is%3Amerged) and [commit history](https://github.com/jmnargi/Strata-V100/commits/main).

## Credits and license

This repository is a fork of [Strata by Niko1221](https://github.com/Niko1221/Strata). **We are deeply grateful to Niko and all upstream contributors.** Their original work is the reason this fork, its V100 support, and this local model runtime are possible. We aim to credit and follow upstream work as we periodically bring in its changes.

Strata incorporates [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) and other open-source components. The project is licensed under the [MIT License](LICENSE); components and model files may have separate terms. See the [credits and license notes](docs/HOW_IT_WORKS.md#credits).
