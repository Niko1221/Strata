# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080

Measured on 2026-10-29 with `tools/strata_bench.py` (suite 1, 1 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080, 32 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 95.7 GiB; storage: HP SSD FX900 Pro 4TB SSD/NVMe; HP SSD FX900 Pro 4TB SSD/NVMe
- OS: Windows 11 (10.0.26200); GPU driver: 621.01; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- Strata: project 0.1.52, commit 5e8a4cca; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=150.00, power_draw_w=27.29, pcie=gen5 x8, pstate=P3, temp_c=40.50
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=75.11, pcie=gen5 x8, pstate=P1, temp_c=52.53
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q6_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-q6_k_xl --native Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:4096 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --resident-budget-gib 72 --vision --vram-reserve-mib 1024 --vram-reserve-later-mib 768
```

Environment of the config: `STRATA_MMQ_KQ=1`, `STRATA_HC_Q8=1`

## Method

Run by `python tools/strata_bench.py --runs 1` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 1 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 1 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 1 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 3 | not measured | 24.8 (23.9-26.6) | 1.34 (1.17-1.48) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 1 | 447.4 (447.4-447.4) | not measured | 9.04 (9.04-9.04) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 1 | 668.4 (668.4-668.4) | not measured | 24.07 (24.07-24.07) |
| agent session, follow-up turns | 12830 (12830-12830) at the last turn | see below | up to 64 | 1 | see below | 27.4 (26.9-32.2) | 8.20 (7.59-8.61) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.9 % (68.9-68.9); earlier-turn tokens read again per session: 214 (214-214) (earlier turns that the cache did not hold; the 11952 (11952-11952) new tokens of the files are read once); first turn (nothing cached) first token: 6.17 s (6.17-6.17).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 99.5 | 6.174 | 17.2 |
| 1 | 3306 | 629 | 2677 | 327.2 | 8.241 | 26.9 |
| 2 | 4839 | 3299 | 1540 | 204.0 | 7.59 | 27.4 |
| 3 | 8083 | 4832 | 3251 | 380.1 | 8.612 | 31.5 |
| 4 | 10111 | 8076 | 2035 | 255.3 | 8.041 | 32.2 |
| 5 | 12830 | 10167 | 2663 | 327.3 | 8.204 | 27.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 127 s.
