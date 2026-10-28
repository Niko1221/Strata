# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080, 32 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 95.7 GiB; storage: HP SSD FX900 Pro 4TB SSD/NVMe; HP SSD FX900 Pro 4TB SSD/NVMe
- OS: Windows 11 (10.0.26200); GPU driver: 621.01; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- Strata: project 0.1.52, commit 5e8a4cca; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=150.00, power_draw_w=24.24, pcie=gen5 x8, pstate=P1, temp_c=41.24
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=85.05, pcie=gen5 x8, pstate=P1, temp_c=46.44
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:4096 --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 2048 --vram-reserve-later-mib 1500 --pcie-frac 1.00 --spec-min-p 0.70
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 43.3 (38.8-55.2) | 0.75 (0.65-0.87) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1000.1 (738.8-1010.5) | not measured | 4.06 (4.01-5.48) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1715.8 (1514.2-1716.7) | not measured | 9.39 (9.39-10.64) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1793.5 (1748.9-1794.7) | not measured | 17.93 (17.91-18.40) |
| agent session, follow-up turns | 12815 (12786-12828) at the last turn | see below | up to 64 | 3 | see below | 51.4 (44.2-59.2) | 3.99 (3.61-6.23) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.2-69.4); earlier-turn tokens read again per session: 73 (4-74) (earlier turns that the cache did not hold; the 11950 (11949-11952) new tokens of the files are read once); first turn (nothing cached) first token: 2.98 s (2.96-3.42).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 180.4 | 3.42 | 32.9 |
| 1 | 3296 | 620 | 2676 | 452.8 | 5.947 | 46.7 |
| 2 | 4829 | 3289 | 1540 | 317.5 | 4.901 | 44.2 |
| 3 | 8073 | 4892 | 3181 | 514.9 | 6.23 | 50.1 |
| 4 | 10101 | 8137 | 1964 | 379.8 | 5.215 | 51.4 |
| 5 | 12828 | 10164 | 2664 | 498.6 | 5.401 | 51.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 233 s.
