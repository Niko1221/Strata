# Community benchmark on NVIDIA GeForce RTX 2080 Ti

Measured on 2026-10-29 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti, 22 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz (44 cores, 88 threads); installed RAM: 94.3 GiB; storage: sda     WDC WD30EFRX-68EUZN0 sata; nvme0n1 WD Green SN350 1TB   nvme
- OS: Linux 7.0.0-38-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: project 0.1.52, commit 5e8a4cca; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=180.00, power_draw_w=174.11, pcie=gen3 x16, pstate=P2, temp_c=65
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 36.9 (34.8-48.5) | 0.78 (0.68-1.02) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 575.4 (574.0-587.9) | not measured | 7.03 (6.88-7.04) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 774.8 (770.0-784.5) | not measured | 20.78 (20.53-20.92) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 768.3 (765.3-782.1) | not measured | 41.80 (41.06-41.95) |
| agent session, follow-up turns | 12837 (12815-12840) at the last turn | see below | up to 64 | 3 | see below | 40.8 (28.1-50.2) | 5.90 (5.21-7.14) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.0-69.5); earlier-turn tokens read again per session: 73 (4-213) (earlier turns that the cache did not hold; the 11951 (11950-11951) new tokens of the files are read once); first turn (nothing cached) first token: 3.41 s (3.20-3.50).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 175.9 | 3.5 | 27.5 |
| 1 | 3308 | 633 | 2675 | 383.6 | 7.029 | 37.4 |
| 2 | 4841 | 3372 | 1469 | 276.2 | 5.364 | 37.0 |
| 3 | 8085 | 4834 | 3251 | 463.0 | 7.075 | 43.3 |
| 4 | 10113 | 8078 | 2035 | 320.3 | 6.396 | 28.1 |
| 5 | 12840 | 10106 | 2734 | 385.2 | 7.144 | 50.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 377 s.
