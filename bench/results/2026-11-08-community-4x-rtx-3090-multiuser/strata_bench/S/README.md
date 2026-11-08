# Community benchmark on NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090, 96 GiB VRAM
- CPU: AMD Ryzen Threadripper 3990X 64-Core Processor (64 cores, 128 threads); installed RAM: 251.5 GiB; storage: sda     Samsung SSD 860 EVO 500GB sata; nvme1n1 CT1000P310SSD8            nvme; nvme2n1 CT1000P310SSD8            nvme; nvme0n1 CT1000P310SSD8            nvme
- OS: Linux 7.0.0-38-generic; GPU driver: 580.196.07; CUDA (driver supports up to): 13.0
- Strata: project 0.1.58, commit 146df3c4; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=131.62, pcie=gen3 x16, pstate=P2, temp_c=61
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=138.50, pcie=gen3 x16, pstate=P2, temp_c=61
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=139.41, pcie=gen4 x8, pstate=P2, temp_c=54
- GPU detail (after the run, near idle): power_limit_w=320.00, power_draw_w=211.99, pcie=gen4 x8, pstate=P2, temp_c=64
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 131072 tokens; parallel slots: 3
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-212-stocktab-f83fd81691a5b053 --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --ple-io ram --kv-resident 32768
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py --clients 16 --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 134.5 (113.6-179.0) | 0.23 (0.20-0.30) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2081.4 (2016.0-2085.0) | not measured | 1.95 (1.95-2.01) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3212.6 (3203.2-3253.8) | not measured | 5.02 (4.96-5.04) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 4577.0 (4551.6-4611.0) | not measured | 7.04 (6.99-7.08) |
| agent session, follow-up turns | 13603 (13559-13614) at the last turn | see below | up to 64 | 3 | see below | 153.3 (125.8-167.8) | 1.62 (1.04-1.93) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.5-69.9); earlier-turn tokens read again per session: 143 (4-144) (earlier turns that the cache did not hold; the 12725 (12723-12726) new tokens of the files are read once); first turn (nothing cached) first token: 0.80 s (0.75-0.94).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 827.1 | 0.751 | 140.7 |
| 1 | 3841 | 638 | 3203 | 1849.5 | 1.755 | 156.6 |
| 2 | 5374 | 3834 | 1540 | 1345.1 | 1.164 | 146.6 |
| 3 | 8618 | 5438 | 3180 | 1822.7 | 1.768 | 155.8 |
| 4 | 10840 | 8676 | 2164 | 1575.7 | 1.395 | 150.7 |
| 5 | 13614 | 10833 | 2781 | 1458.8 | 1.928 | 167.8 |

### Concurrency: 16 clients

Total 160.7 tok/s (158.1-161.5); first token p50 5.465 s, p95 11.224 s; request latency p50 7.187 s, p95 12.464 s; errors 0. Server parallel slots: 3 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 125 s.
