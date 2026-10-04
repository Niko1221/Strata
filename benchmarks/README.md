# Community benchmarks

This directory is the contributor-facing index for Strata benchmark results across different hardware. The reproducible benchmark tooling and raw machine-readable output remain under [`bench/`](../bench/).

## Published results

| Accelerator | Model | Quantization | Context | Results |
| --- | --- | --- | ---: | --- |
| NVIDIA Tesla V100-PCIE-16GB | Qwen3.8-Flash-Next | Q2_0 | 262,144 | [V100, 2026-09-28](v100-pcie-16gb-q2_0-2026-09-28.md) |
| 2 × NVIDIA Tesla V100-PCIE-16GB | Qwen3.8-Flash-Next | Q2_0 | 4K / 32K / 128K | [PR 600 Volta prompt attention, 2026-10-03](v100-q2_0-pr600-2026-10-03.md) |
| 2 × NVIDIA Tesla V100-PCIE-16GB, Gen3 x2 + x16 | Qwen3.8-Flash-Next | Q2_0 | 262,144 | [Asymmetric PCIe, 2026-09-30](v100-asymmetric-pcie-q2_0-2026-09-30.md) |
| 2 × NVIDIA Tesla V100-PCIE-16GB | Qwen3.8-Flash-Next | Q2_0 | 4K / 32K | [Selected PR 627 kernels, 2026-10-03](v100-q2_0-pr627-2026-10-03.md) |
| 2 × NVIDIA Tesla V100-PCIE-16GB | Qwen3.8-Flash-Next | Q2_0 | 4K / 32K | [PR 540 decode attention, 2026-10-03](v100-q2_0-pr540-2026-10-03.md) |
| 2 × NVIDIA Tesla V100-PCIE-16GB | Qwen3.8-Flash-Next | Q2_0 | 2K / 4K / 8K / 16K / 32K | [INT8-KV prefill with FP16 tensor-core experts, 2026-10-03](v100-q2_0-prefill-2026-10-03.md) |

## Contributing results
1. Run the existing [`bench/run_v100_bench.py`](../bench/run_v100_bench.py) runner for the published workload, or add the closest equivalent runner under `bench/`.
2. Keep the workload comparable: report the exact model and quantization, prompt token counts, generated-token count, cache state, and relevant runtime settings.
3. Add a Markdown summary in this directory named `<accelerator>-<quant>-<date>.md` and put raw output in `bench/results/<date>-<short-hardware-name>/`.
4. Include enough system information to explain performance: GPU model and count, VRAM, CPU, RAM, storage, operating system, CUDA/driver versions, Strata commit, and thermal or power limits.
5. Link the new result from the table above.

Do not include API keys, usernames, home-directory paths, hostnames, private IP addresses, or other machine-specific identifiers. Results should distinguish measured values from estimates and mention throttling, warm caches, or other conditions that can materially affect comparisons.
