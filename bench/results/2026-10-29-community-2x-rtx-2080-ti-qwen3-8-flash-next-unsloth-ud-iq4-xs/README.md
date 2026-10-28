# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-10-29 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 44 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz (44 cores, 88 threads); installed RAM: 94.3 GiB; storage: sda     WDC WD30EFRX-68EUZN0 sata; nvme0n1 WD Green SN350 1TB   nvme
- OS: Linux 7.0.0-38-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: project 0.1.52, commit 5e8a4cca; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=180.00, power_draw_w=156.14, pcie=gen3 x16, pstate=P2, temp_c=60
- GPU detail (after the run, near idle): power_limit_w=180.00, power_draw_w=184.88, pcie=gen3 x16, pstate=P2, temp_c=70
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --resident-budget-gib 55 --vision --vram-reserve-mib 700 --prompt-cache-every 4096
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 49.7 (37.9-61.4) | 0.75 (0.65-0.83) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 707.2 (596.7-709.5) | not measured | 5.71 (5.70-6.77) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1104.0 (1063.9-1113.6) | not measured | 14.60 (14.48-15.15) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1269.0 (1265.3-1274.7) | not measured | 25.33 (25.22-25.41) |
| agent session, follow-up turns | 12824 (12806-12834) at the last turn | see below | up to 64 | 3 | see below | 60.4 (44.7-78.6) | 4.62 (3.79-6.37) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.3); earlier-turn tokens read again per session: 73 (73-74) (earlier turns that the cache did not hold; the 11951 (11951-11952) new tokens of the files are read once); first turn (nothing cached) first token: 3.74 s (3.74-3.77).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 164.3 | 3.744 | 31.5 |
| 1 | 3303 | 626 | 2677 | 424.2 | 6.366 | 44.7 |
| 2 | 4836 | 3367 | 1469 | 360.8 | 4.111 | 47.0 |
| 3 | 8080 | 4899 | 3181 | 595.2 | 5.393 | 48.6 |
| 4 | 10107 | 8143 | 1964 | 449.2 | 4.487 | 60.4 |
| 5 | 12834 | 10100 | 2734 | 489.5 | 5.631 | 53.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 275 s.
