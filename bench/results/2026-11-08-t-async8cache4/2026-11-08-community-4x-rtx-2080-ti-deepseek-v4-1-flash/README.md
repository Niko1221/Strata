# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 88 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz (14 cores, 28 threads); installed RAM: 121.4 GiB; storage: nvme0n1 Lexar SSD NM620 2TB nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.55, commit c62d5e3a; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=93.44, pcie=gen3 x16, pstate=P2, temp_c=64
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=134.32, pcie=gen3 x16, pstate=P2, temp_c=78
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=143.42, pcie=gen3 x16, pstate=P2, temp_c=71
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=145.54, pcie=gen3 x16, pstate=P2, temp_c=64
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-dsv41 --pack deepseek-v4.1-flash-ist-das-2.00bpw --max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.5 --tp 4 --experts 5 --threads on
```

## Method

Run by `python tools/strata_bench.py --long --runs 5` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 5 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 5 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 5 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 15 | not measured | 39.2 (28.0-55.4) | 0.40 (0.34-0.51) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 5 | 552.6 (508.9-566.7) | not measured | 7.48 (7.30-8.12) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 5 | 918.0 (574.8-927.3) | not measured | 18.09 (17.90-28.86) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 5 | 1042.6 (998.2-1056.2) | not measured | 31.87 (31.46-33.28) |
| agent session, follow-up turns | 12659 (12639-12681) at the last turn | see below | up to 64 | 5 | see below | 31.3 (26.7-39.5) | 11.82 (8.26-14.92) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.5-69.7); earlier-turn tokens read again per session: 1 (0-2) (earlier turns that the cache did not hold; the 11797 (11796-11797) new tokens of the files are read once); first turn (nothing cached) first token: 5.55 s (5.48-6.74).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 89.4 | 6.743 | 22.9 |
| 1 | 3336 | 637 | 2699 | 191.6 | 14.118 | 26.7 |
| 2 | 4845 | 3392 | 1453 | 146.7 | 9.94 | 27.8 |
| 3 | 7952 | 4905 | 3047 | 204.8 | 14.917 | 31.3 |
| 4 | 10037 | 8015 | 2022 | 185.3 | 10.952 | 33.1 |
| 5 | 12659 | 10082 | 2577 | 196.6 | 13.17 | 35.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 736 s.
