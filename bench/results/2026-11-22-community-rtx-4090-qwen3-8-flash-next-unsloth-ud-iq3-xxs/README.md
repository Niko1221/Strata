# Community benchmark on NVIDIA GeForce RTX 4090

Measured on 2026-11-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 48 GiB VRAM
- CPU: Intel(R) Core(TM) i7-14700KF (20 cores, 28 threads); installed RAM: 95.8 GiB; storage: Samsung SSD 990 PRO 2TB SSD/NVMe
- OS: Windows 11 (10.0.26320); GPU driver: 621.24; CUDA (driver supports up to): 13.5
- Strata: project 0.1.61, commit not a git checkout; engine 0.1.59.1
- GPU detail (after the run, near idle): power_limit_w=450.00, power_draw_w=328.36, pcie=gen4 x16, pstate=P0, temp_c=49.56
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-iq3_xxs --native Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 65536 --vision --vram-elastic --vram-reserve-mib 1536 --pcie-frac 0.75 --spec-min-p 0.30 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 174.5 (139.3-332.1) | 0.23 (0.22-0.28) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3520.5 (3488.1-3545.7) | not measured | 1.17 (1.15-1.19) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 5012.3 (4983.4-5030.1) | not measured | 3.25 (3.24-3.27) |
| agent session, follow-up turns | 13194 (13159-13201) at the last turn | see below | up to 64 | 3 | see below | 176.7 (144.8-210.0) | 1.00 (0.67-1.13) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.2-69.2); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12325 (12324-12326) new tokens of the files are read once); first turn (nothing cached) first token: 0.76 s (0.76-0.80).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 876.9 | 0.764 | 86.3 |
| 1 | 3423 | 618 | 2805 | 2655.0 | 1.078 | 180.5 |
| 2 | 4956 | 3486 | 1470 | 2046.8 | 0.736 | 153.8 |
| 3 | 8184 | 5019 | 3165 | 2948.0 | 1.094 | 144.8 |
| 4 | 10279 | 8247 | 2032 | 2399.3 | 0.88 | 154.0 |
| 5 | 13201 | 10342 | 2859 | 2585.0 | 1.126 | 199.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 48 s.
