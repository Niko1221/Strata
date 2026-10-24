# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 121.5 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02; CUDA: 13.2
- Strata: project 0.1.44, commit ed4773eb; engine 0.1.44
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-nvfp4-nvfp4-BF16
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack pack --native qwen-nvfp4-gptq-dense.gguf --native-dense-gguf qwen-nvfp4-gptq-dense.gguf --ple-gguf ple-fp8.gguf --embd-gguf token-embd-bf16.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 64.3 (57.8-75.2) | 0.56 (0.51-0.63) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1851.6 (1848.3-1895.8) | not measured | 2.19 (2.14-2.19) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3611.6 (3484.4-3622.1) | not measured | 4.47 (4.46-4.63) |
| agent session, follow-up turns | 14589 (14481-14591) at the last turn | see below | up to 64 | 3 | see below | 90.8 (41.4-158.3) | 2.17 (2.07-2.32) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 4 (4-74) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 1.62 s (1.59-1.63).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 387.5 | 1.592 | 44.0 |
| 1 | 3741 | 620 | 3121 | 1360.0 | 2.319 | 62.0 |
| 2 | 5274 | 3804 | 1470 | 706.2 | 2.102 | 78.9 |
| 3 | 8718 | 5337 | 3381 | 1489.2 | 2.292 | 89.8 |
| 4 | 11644 | 8781 | 2863 | 1282.0 | 2.257 | 121.0 |
| 5 | 14589 | 11690 | 2899 | 1321.6 | 2.216 | 72.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 92 s.

## Comparison across saved runs

Submitted by [Radi-m](https://github.com/Radi-m). These are community speed measurements, not results from the draft standard benchmark. The CPU identified in all four saved reports is a Ryzen 9 5950X.

| Run | Reported version / commit | Chat decode tok/s | Chat first token (s) | 4K prompt tok/s | 16K prompt tok/s | Agent decode tok/s | Agent turn first token (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [October 14 baseline](../2026-10-14-community-rtx-5090-qwen3-8-flash-next-nvfp4-nvfp4-bf16/README.md) | 0.1.44 / ed4773eb | 65.9 | 0.55 | 1913.3 | 3627.1 | 95.9 | 2.16 |
| [October 23 baseline](./results.json) | 0.1.44 / ed4773eb | 64.3 | 0.56 | 1851.6 | 3611.6 | 90.8 | 2.17 |
| [October 23 updated, first run](../2026-10-23-community-rtx-5090-qwen3-8-flash-next-nvfp4-nvfp4-bf16-2/README.md) | 0.1.48.1 / 7ddc17a3 | 64.3 | 0.71 | 1727.9 | 3399.0 | 101.9 | 2.20 |
| [October 23 updated, repeat](../2026-10-23-community-rtx-5090-qwen3-8-flash-next-nvfp4-nvfp4-bf16-3/README.md) | 0.1.48.1 / 7ddc17a3 | 65.7 | 0.70 | 1732.9 | 3422.1 | 94.7 | 2.21 |

Values are medians; each linked report keeps its ranges and per-request data. Compared with the same-day baseline, the updated runs read the 4K prompt 6.4-6.7% slower and the 16K prompt 5.2-5.9% slower. Chat decode is similar. The agent decode medians are higher, but their ranges overlap; this does not establish a reliable speedup.

The saved launch arguments are identical, including GPU vision, but automatic settings are not fixed:

| Run | GPU expert slots | GPU expert cache (MiB) | Reported conversation cache (MiB) |
| --- | ---: | ---: | ---: |
| October 14 baseline | 7911 | 20861 | 0 |
| October 23 baseline | 7843 | 20681 | 0 |
| October 23 updated, first run | 7615 | 20080 | 8192 |
| October 23 updated, repeat | 7627 | 20112 | 8192 |

All four report 15 CPU workers, int8 KV with 32768 resident tokens, `spec: 6`, `mtp_max: 4` and `lookup: 3`. The launch command asks for `--spec 4`; the effective server values above are preserved as reported, not inferred from that command. Only the updated reports expose a prefill cap (8192), so an equal effective prompt chunk across versions is not established.

Version labels in this comparison are the values captured by the benchmark and server. The updated source checkout reports 0.1.48.1; this is not a claim that a released engine package has that version. Release packages for Strata 0.2.3/0.2.3.1 are engine 0.1.48. Build provenance, model/profile hashes, clock and power measurements, and controlled alternating old/new runs are not included.

All four JSON files record empty failure and skipped-case collections. Correctness, DFlash2, concurrency and long-context recall were not tested. No new benchmark was run while preparing this submission. The observed slowdown is not attributed to a particular change.
