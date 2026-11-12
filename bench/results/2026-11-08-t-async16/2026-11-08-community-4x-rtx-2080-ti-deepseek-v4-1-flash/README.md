# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 88 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz (14 cores, 28 threads); installed RAM: 121.4 GiB; storage: nvme0n1 Lexar SSD NM620 2TB nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.55, commit c62d5e3a; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=119.76, pcie=gen3 x16, pstate=P2, temp_c=66
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=133.69, pcie=gen3 x16, pstate=P2, temp_c=78
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=144.84, pcie=gen3 x16, pstate=P2, temp_c=71
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=132.40, pcie=gen3 x16, pstate=P2, temp_c=64
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
| short chat | 32 (29-38) | 0 | 100 (100-100) | 15 | not measured | 40.3 (29.9-57.5) | 0.42 (0.35-0.54) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 5 | 538.6 (467.5-558.4) | not measured | 7.68 (7.41-8.84) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 5 | 909.8 (872.1-919.7) | not measured | 18.25 (18.05-19.03) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 5 | 1038.4 (988.4-1053.9) | not measured | 32.00 (31.53-33.61) |
| agent session, follow-up turns | 12673 (12655-12679) at the last turn | see below | up to 64 | 5 | see below | 31.5 (26.4-37.9) | 11.93 (8.27-15.13) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.6-69.6); earlier-turn tokens read again per session: 0 (0-2) (earlier turns that the cache did not hold; the 11796 (11796-11798) new tokens of the files are read once); first turn (nothing cached) first token: 5.57 s (5.50-6.91).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 87.2 | 6.913 | 22.7 |
| 1 | 3336 | 637 | 2699 | 190.5 | 14.202 | 27.9 |
| 2 | 4852 | 3399 | 1453 | 150.4 | 9.694 | 26.4 |
| 3 | 7956 | 4909 | 3047 | 201.9 | 15.129 | 29.1 |
| 4 | 10038 | 8018 | 2020 | 177.2 | 11.44 | 28.3 |
| 5 | 12673 | 10096 | 2577 | 189.5 | 13.647 | 34.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 735 s.
