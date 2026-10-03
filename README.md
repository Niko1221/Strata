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

## Get started

This fork tracks the upstream installation experience. For complete and current platform requirements, model choices, and options, see the [installation guide](docs/INSTALL.md), [model guide](docs/MODELS.md), and [upstream setup guide](docs/AI_SETUP.md).

- **Windows:** download or clone this repository and run `START-HERE.bat`.
- **Linux:** clone this repository and run `./setup.sh`.
- Follow the prompts to select a model and context size. Setup downloads model data and starts the local service.
- Open the address printed by setup (normally `http://127.0.0.1:8080`).

For a Tesla V100, build with a CUDA 12.x toolkit and retain `sm_70` support. See the install guide for build details and known platform limits. If using Docker, follow the repository's [Docker instructions](docs/INSTALL.md) and build for the target GPU architecture; the default prebuilt engine architecture list does not include `sm_70`.

## Use Strata

- **Web app:** use the local URL printed by setup for chat and the live monitor.
- **OpenAI-compatible API:** set your app's base URL to `http://127.0.0.1:8080/v1`.
- **Anthropic-compatible API:** use `http://127.0.0.1:8080/v1/messages`.
- **Images, MCP, multi-GPU, and configuration:** see [details](docs/DETAILS.md), [MCP server](docs/MCP_SERVER.md), and [multi-GPU guide](docs/MULTI_GPU.md).

The server normally listens on localhost. If you expose it to other machines, configure an API key and use a trusted network. Do not publish secrets or private configuration in issues, benchmark results, or pull requests.

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
