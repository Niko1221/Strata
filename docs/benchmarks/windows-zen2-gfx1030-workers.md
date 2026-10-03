# Windows Threadripper 3990X and RX 6900 XT worker measurement

On 2026-10-03, Strata v0.1.38 was measured on Windows 11 Pro with a Ryzen Threadripper 3990X (64 physical cores,
128 logical processors), Radeon RX 6900 XT (16 GiB, gfx1030), and 128 GiB RAM. The official Windows HIP engine used
the bundled ROCm 10.2.0a20260930 and hipBLASLt 100500. It was a normal interactive desktop; unrelated applications
were not stopped.

The workload was Qwen3.8-Flash-Next Coder IQ1_M, context 65,536, INT8 KV with 32,768 resident cells, automatic
prefill and expert cache, MTP depth 4, and no images. Each case reports the median of three requests, each generating
256 tokens from a fresh prompt without prompt-prefix reuse. The 63-worker run used the stock setup default; 31 and
15 were explicitly selected with `--pool-workers`.

| Explicit CPU workers | Input tokens | Prompt tok/s | Decode tok/s |
| ---: | ---: | ---: | ---: |
| 63 (default) | 94 | 41.6 | 12.5 |
| 63 (default) | 4,022 | 313.2 | 11.6 |
| 63 (default) | 32,685 | 349.8 | 6.1 |
| 31 | 94 | 46.8 | 48.0 |
| 31 | 4,022 | 312.5 | 45.3 |
| 31 | 32,685 | 352.1 | 46.0 |
| 15 | 94 | 45.6 | 44.4 |

The values in the full report include per-run ranges and latency. 31 was fastest among the three tested worker counts
for the short case, and was measured at the 4K and 32K cases; worker counts were not exhaustively searched. Prompt
throughput changed little in these runs. This comparison does not establish the cause of the default's slowdown, an
overall performance optimum, coding quality, full 64K-context correctness, or results on another CPU, GPU, model,
operating system, or workload. In particular, other Zen2 Threadripper models were not measured. The resulting
automatic rule is scoped to Windows HIP on a single gfx1030 GPU, a Threadripper 3990X, and no remote expert cache;
the policy's automatic selection has not itself been benchmarked. An explicit positive `--pool-workers` count takes
precedence.

The original evidence and reproduction instructions are preserved at the
[immutable measurement commit](https://github.com/Yasei-no-otoko/Strata/tree/4cdef220820b8876c3e357e38e4eea9d088b25a8/bench/results/2026-10-03-rx6900xt-coder), including the full methodology, hardware snapshots, raw request bodies, outputs, engine logs, and scripts. See its [README](https://github.com/Yasei-no-otoko/Strata/blob/4cdef220820b8876c3e357e38e4eea9d088b25a8/bench/results/2026-10-03-rx6900xt-coder/README.md) for the exact commands and details.
