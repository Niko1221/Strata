# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080

Measured on 2026-10-28 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5080, 32 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 95.7 GiB; storage: HP SSD FX900 Pro 4TB SSD/NVMe; HP SSD FX900 Pro 4TB SSD/NVMe
- OS: Windows 11 (10.0.26200); GPU driver: 621.01; CUDA (driver supports up to): 13.5; engine built for CUDA 13.0
- Strata: project 0.1.52, commit 5e8a4cca; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=150.00, power_draw_w=25.44, pcie=gen5 x8, pstate=P1, temp_c=49.37
- GPU detail (after the run, near idle): power_limit_w=250.00, power_draw_w=102.53, pcie=gen5 x8, pstate=P1, temp_c=46.52
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:4096 --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 2048 --vram-reserve-later-mib 1500 --pcie-frac 0.55 --spec-min-p 0.70
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 52.0 (48.0-61.8) | 0.70 (0.59-0.81) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1186.5 (854.7-1194.6) | not measured | 3.44 (3.40-4.75) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1890.1 (1679.8-1892.5) | not measured | 8.54 (8.53-9.60) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1967.4 (1962.4-1970.3) | not measured | 16.35 (16.31-16.39) |
| agent session, follow-up turns | 12817 (12797-12824) at the last turn | see below | up to 64 | 3 | see below | 64.1 (53.8-78.6) | 3.11 (2.79-5.38) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.5); earlier-turn tokens read again per session: 74 (2-144) (earlier turns that the cache did not hold; the 11950 (11950-11952) new tokens of the files are read once); first turn (nothing cached) first token: 2.39 s (2.33-2.76).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 222.7 | 2.759 | 42.5 |
| 1 | 3308 | 630 | 2678 | 530.8 | 5.092 | 53.8 |
| 2 | 4841 | 3372 | 1469 | 366.2 | 4.057 | 56.6 |
| 3 | 8085 | 4905 | 3180 | 595.5 | 5.381 | 68.2 |
| 4 | 10113 | 8149 | 1964 | 456.4 | 4.35 | 78.6 |
| 5 | 12817 | 10154 | 2663 | 581.5 | 4.624 | 60.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 198 s.
