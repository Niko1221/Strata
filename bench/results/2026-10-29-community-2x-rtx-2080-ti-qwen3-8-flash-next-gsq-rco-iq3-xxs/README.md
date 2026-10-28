# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-10-29 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 44 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz (44 cores, 88 threads); installed RAM: 94.3 GiB; storage: sda     WDC WD30EFRX-68EUZN0 sata; nvme0n1 WD Green SN350 1TB   nvme
- OS: Linux 7.0.0-38-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: project 0.1.52, commit 5e8a4cca; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=180.00, power_draw_w=147.37, pcie=gen3 x16, pstate=P2, temp_c=62
- GPU detail (after the run, near idle): power_limit_w=180.00, power_draw_w=182.47, pcie=gen3 x16, pstate=P2, temp_c=70
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-gsq-rco-iq3_xxs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack gsq-rco-iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --resident-budget-gib 55 --vision --vram-reserve-mib 700 --prompt-cache-every 4096
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 68.3 (58.5-90.0) | 0.71 (0.65-0.86) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 820.7 (807.8-822.1) | not measured | 4.93 (4.92-5.00) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1190.8 (1187.1-1211.3) | not measured | 13.54 (13.31-13.58) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1390.7 (1389.6-1393.9) | not measured | 23.12 (23.07-23.14) |
| agent session, follow-up turns | 12822 (12808-12843) at the last turn | see below | up to 64 | 3 | see below | 69.7 (58.9-74.0) | 3.73 (2.54-4.33) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.5); earlier-turn tokens read again per session: 72 (4-73) (earlier turns that the cache did not hold; the 11951 (11950-11951) new tokens of the files are read once); first turn (nothing cached) first token: 2.47 s (2.12-2.57).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 291.1 | 2.123 | 66.5 |
| 1 | 3327 | 649 | 2678 | 749.0 | 3.632 | 58.9 |
| 2 | 4860 | 3391 | 1469 | 586.3 | 2.544 | 66.4 |
| 3 | 8104 | 4923 | 3181 | 769.4 | 4.179 | 69.7 |
| 4 | 10127 | 8163 | 1964 | 649.5 | 3.072 | 67.0 |
| 5 | 12843 | 10180 | 2663 | 717.8 | 3.818 | 62.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 228 s.
