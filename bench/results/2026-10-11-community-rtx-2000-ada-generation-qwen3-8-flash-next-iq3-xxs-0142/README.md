# Community benchmark on NVIDIA RTX 2000 Ada Generation

Measured on 2026-10-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX 2000 Ada Generation, 16 GiB VRAM
- CPU: 12th Gen Intel(R) Core(TM) i7-12700 (12 cores, 20 threads); installed RAM: 63.6 GiB; storage: NVMe on both volumes; D: J.ZAO KP Series 2TB SSD holds the model shards, C: Samsung EVO Plus 2TB
- OS: Windows 11 (10.0.26300); GPU driver: 596.51; CUDA: 13.2
- Strata: project 0.1.43, commit 7695859a; engine 0.1.42
- Background workloads and power limits: nothing else using the GPU while the suite ran (one engine loaded at a time); GPU board power limit 70 W, this card's default and maximum

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: this machine's own daily configuration, tuned earlier in a design-of-experiments study (PCIe share 0.29, KV resident 20480 MiB, prefill 4096, short-read 64, vision with a 700 MiB reserve); thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 4096 --spec 5 --spec-min-p 0.7 --mtp rt --max-context 262144 --kv q4_0 --kv-resident 20480 --pcie-frac 0.29 --pool-affinity p-cores --vision --vram-reserve-mib 700 --short-read 64
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 42.8 (36.8-51.8) | 0.70 (0.65-0.80) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 861.3 (856.6-862.8) | not measured | 4.71 (4.70-4.74) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 895.5 (893.2-899.3) | not measured | 17.98 (17.90-18.02) |
| agent session, follow-up turns | 14504 (14466-14511) at the last turn | see below | up to 64 | 3 | see below | 41.5 (25.5-48.8) | 4.40 (3.61-4.77) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.5 % (68.5-68.6); earlier-turn tokens read again per session: 144 (144-145) (earlier turns that the cache did not hold; the 13635 (13632-13636) new tokens of the files are read once); first turn (nothing cached) first token: 2.94 s (2.91-3.02).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 204.1 | 3.025 | 22.1 |
| 1 | 3702 | 621 | 3081 | 703.3 | 4.437 | 41.5 |
| 2 | 5235 | 3765 | 1470 | 409.9 | 3.626 | 39.6 |
| 3 | 8679 | 5228 | 3451 | 732.7 | 4.755 | 40.3 |
| 4 | 11603 | 8672 | 2931 | 672.9 | 4.403 | 48.8 |
| 5 | 14466 | 11620 | 2846 | 662.2 | 4.344 | 25.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 199 s.
