# Community benchmark on NVIDIA RTX 2000 Ada Generation

Measured on 2026-10-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX 2000 Ada Generation, 16 GiB VRAM
- CPU: 12th Gen Intel(R) Core(TM) i7-12700 (12 cores, 20 threads); installed RAM: 63.6 GiB; storage: NVMe on both volumes; D: J.ZAO KP Series 2TB SSD holds the model shards, C: Samsung EVO Plus 2TB
- OS: Windows 11 (10.0.26300); GPU driver: 596.51; CUDA: 13.2
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: nothing else using the GPU while the suite ran (one engine loaded at a time); GPU board power limit 70 W, this card's default and maximum

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: this machine's own daily configuration, tuned earlier in a design-of-experiments study (PCIe share 0.29, KV resident 20480 MiB, prefill 4096, short-read 64, vision with a 700 MiB reserve); thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 4096 --spec 5 --spec-min-p 0.7 --mtp rt --max-context 262144 --kv q4_0 --kv-resident 20480 --pcie-frac 0.29 --pool-affinity p-cores --vision --vram-reserve-mib 700 --short-read 64
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 41.2 (36.6-49.9) | 0.76 (0.67-0.86) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 879.7 (879.4-880.0) | not measured | 4.62 (4.61-4.62) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 905.0 (899.7-906.7) | not measured | 17.79 (17.76-17.90) |
| agent session, follow-up turns | 14513 (14475-14521) at the last turn | see below | up to 64 | 3 | see below | 42.0 (21.8-50.7) | 4.37 (3.65-4.92) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.7-68.8); earlier-turn tokens read again per session: 73 (4-74) (earlier turns that the cache did not hold; the 13635 (13635-13636) new tokens of the files are read once); first turn (nothing cached) first token: 2.94 s (2.94-2.95).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 211.0 | 2.953 | 29.7 |
| 1 | 3706 | 623 | 3083 | 712.8 | 4.432 | 38.8 |
| 2 | 5239 | 3769 | 1470 | 407.0 | 3.674 | 40.6 |
| 3 | 8683 | 5303 | 3380 | 721.6 | 4.741 | 37.9 |
| 4 | 11605 | 8745 | 2860 | 664.9 | 4.356 | 42.0 |
| 5 | 14475 | 11629 | 2846 | 656.3 | 4.388 | 21.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 203 s. A same-machine A/B across 0.1.42 and 0.1.43 is in the engine-version report beside this folder; the pull request that adds them describes how the file-tier fix behaves differently on prose and on generated word lists.
