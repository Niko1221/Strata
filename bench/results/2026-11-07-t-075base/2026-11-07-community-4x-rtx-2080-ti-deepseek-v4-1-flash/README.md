# Community benchmark on NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-11-07 with `tools/strata_bench.py` (suite 1, 5 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti + NVIDIA GeForce RTX 2080 Ti, 88 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz (14 cores, 28 threads); installed RAM: 121.4 GiB; storage: nvme0n1 Lexar SSD NM620 2TB nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.55, commit c62d5e3a; engine ds4-0.1
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=89.68, pcie=gen3 x16, pstate=P2, temp_c=63
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=99.07, pcie=gen3 x16, pstate=P2, temp_c=75
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=87.49, pcie=gen3 x16, pstate=P2, temp_c=68
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=129.96, pcie=gen3 x16, pstate=P2, temp_c=64
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: deepseek-v4.1-flash
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-dsv41 --pack deepseek-v4.1-flash-ist-das-2.00bpw --max-context 131072 --eos 1 --vision on --dspark auto --dspark-conf 0.5 --tp 4 --experts 5
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
| short chat | 32 (29-38) | 0 | 100 (100-100) | 15 | not measured | 31.0 (23.5-41.1) | 0.47 (0.40-0.58) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 5 | 426.0 (387.2-432.2) | not measured | 9.70 (9.56-10.67) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 5 | 673.2 (610.6-676.4) | not measured | 24.64 (24.53-27.19) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 5 | 764.8 (743.1-770.9) | not measured | 43.41 (43.07-44.66) |
| agent session, follow-up turns | 12680 (12663-12683) at the last turn | see below | up to 64 | 5 | see below | 28.3 (19.4-37.9) | 13.11 (8.66-25.06) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.7 % (69.6-69.7); earlier-turn tokens read again per session: 2 (1-2) (earlier turns that the cache did not hold; the 11798 (11797-11798) new tokens of the files are read once); first turn (nothing cached) first token: 5.85 s (5.80-9.81).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 61.5 | 9.812 | 20.9 |
| 1 | 3336 | 637 | 2699 | 107.8 | 25.061 | 25.6 |
| 2 | 4851 | 3398 | 1453 | 137.0 | 10.64 | 25.7 |
| 3 | 7963 | 4914 | 3049 | 183.6 | 16.645 | 25.7 |
| 4 | 10041 | 8021 | 2020 | 163.8 | 12.376 | 19.4 |
| 5 | 12663 | 10086 | 2577 | 107.8 | 23.961 | 31.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 907 s.
