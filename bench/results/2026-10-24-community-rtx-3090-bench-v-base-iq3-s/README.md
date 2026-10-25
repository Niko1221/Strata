# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-10-24 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz (10 cores, 20 threads); installed RAM: 63.9 GiB; storage: not measured
- OS: Windows 11 (10.0.22631); GPU driver: 616.64; CUDA: not read
- Strata: project 0.1.48.1, commit 7ddc17a3; engine 0.1.48
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: bench-v-base-IQ3_S
- Context: 524288 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.70 --mtp rt --max-context 524288 --kv k8v4 --vision --vram-reserve-mib 700 --rope-scaling yarn --rope-scale 2 --kv-grow
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 200 (197-205) | 0 | 100 (100-100) | 9 | not measured | 86.9 (70.9-104.7) | 0.73 (0.70-0.76) |
| 4K prompt, fresh | 4024 (4024-4024) | 0 (0-0) | 24 cap | 3 | 1102.0 (1094.3-1108.4) | not measured | 3.67 (3.65-3.70) |
| 16K prompt, fresh | 16056 (16056-16056) | 0 (0-0) | 24 cap | 3 | 1675.1 (1672.8-1676.1) | not measured | 9.62 (9.61-9.65) |
| agent session, follow-up turns | 14784 (14761-14797) at the last turn | see below | up to 64 | 3 | see below | 94.4 (78.7-120.4) | 3.50 (3.22-3.71) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.2 % (70.1-70.3); earlier-turn tokens read again per session: 4 (3-73) (earlier turns that the cache did not hold; the 13729 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 1.84 s (1.81-1.88).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 811 | 0 | 811 | 454.3 | 1.809 | 58.3 |
| 1 | 4291 | 822 | 3469 | 946.2 | 3.693 | 82.2 |
| 2 | 5824 | 4355 | 1469 | 453.1 | 3.268 | 87.1 |
| 3 | 9257 | 5817 | 3440 | 938.6 | 3.713 | 94.4 |
| 4 | 12024 | 9316 | 2708 | 777.2 | 3.51 | 120.4 |
| 5 | 14797 | 12080 | 2717 | 786.2 | 3.505 | 93.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 133 s.

## Appendix: seven-configuration A/B on engine 0.1.44.1 (2026-10-16)

The same machine and the same suite, run the night of 2026-10-16 on engine 0.1.44.1
(commit 296b59d2): seven configurations, one suite each, sequential, with a full engine
restart and fresh model load between arms. Each arm differs from the base only in the
settings listed. Per-run data and the full template README of every arm are in the linked
folders of this pull request.

| Arm | Changed settings vs base | Chat decode tok/s | Chat TTFT s | 4K prompt tok/s | 16K prompt tok/s | Agent decode tok/s | Agent cache hit % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [base](../2026-10-16-community-rtx-3090-bench-v-base-iq3-s/README.md) | install defaults | 78.3 | 0.42 | 1085.5 | 1636.8 | 88.7 | 68.6 |
| [w4](../2026-10-16-community-rtx-3090-bench-v-w4-iq3-s/README.md) | `--pool-workers 4` | 69.5 | 0.51 | 1084.5 | 1634.2 | 84.9 | 68.6 |
| [w6](../2026-10-16-community-rtx-3090-bench-v-w6-iq3-s/README.md) | `--pool-workers 6` | 73.2 | 0.44 | 1083.4 | 1631.8 | 84.2 | 68.7 |
| [w4p2](../2026-10-16-community-rtx-3090-bench-v-w4p2-iq3-s/README.md) | `--pool-workers 4`, `--pcie-frac 0.20` | 65.8 | 0.55 | 1083.6 | 1631.1 | 87.8 | 68.7 |
| [fused](../2026-10-16-community-rtx-3090-bench-v-fused-iq3-s/README.md) | `--pcie-frac 0.00`, `STRATA_PF_FUSED=1` | 74.9 | 0.43 | 1103.2 | 1695.9 | 88.4 | 68.7 |
| [la0](../2026-10-16-community-rtx-3090-bench-v-la0-iq3-s/README.md) | `--pcie-frac 0.00`, `STRATA_LOOKAHEAD=0` | 75.5 | 0.42 | 1085.6 | 1634.9 | 88.9 | 68.6 |
| [combo](../2026-10-16-community-rtx-3090-bench-v-combo-iq3-s/README.md) | `--pcie-frac 0.00`, `STRATA_PF_FUSED=1` + `STRATA_LOOKAHEAD=0` | 74.1 | 0.43 | 1105.1 | 1693.4 | 90.4 | 68.6 |

Reading of the A/B: chat decode varies by roughly +/-10% between arms of the same engine
(run-to-run noise on this desktop); the only change that moves 16K prompt throughput beyond
that band is `STRATA_PF_FUSED=1` (+3.6%, 1636.8 -> 1695.9 tok/s, with chat TTFT and decode
unchanged), and the gain carries into the combo arm. Reducing `--pool-workers` below the
default of 9 costs chat and agent decode on this 10-core/20-thread CPU. `STRATA_LOOKAHEAD=0`
was neutral. One suite per arm, so the ranges come from the 3 runs inside a single suite.

The engine-0.1.48 table above was measured a week later and is not directly comparable to
these 0.1.44.1 arms (newer engine, and the short-chat prompts of the suite changed between
the two dates), so the two tables should be read separately.

