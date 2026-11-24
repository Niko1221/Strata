# Community benchmark on NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 3090

Measured on 2026-11-07 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 3090, 40 GiB VRAM
- CPU: AMD Ryzen Threadripper PRO 3945WX 12-Cores (12 cores, 24 threads); installed RAM: 123.6 GiB; storage: sda     MSI S270 480GB          sata; sdb     Portable SSD            usb; nvme1n1 INTEL SSDPELKX010T8     nvme; nvme0n1 Samsung SSD 990 PRO 1TB nvme
- OS: Linux 7.0.0-42-generic; GPU driver: 595.99.02; CUDA (driver supports up to): 13.2
- Strata: project 0.1.58, commit 146df3c4; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=165.00, power_draw_w=32.60, pcie=gen4 x8, pstate=P2, temp_c=47
- GPU detail (after the run, near idle): power_limit_w=350.00, power_draw_w=209.07, pcie=gen4 x16, pstate=P2, temp_c=62
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q5_k_xl
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-q5_k_xl --native Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.55 --spec-min-p 0.70 --pool-workers 7 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

Environment of the config: `STRATA_MMQ_KQ=1`, `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 14.7 (10.8-17.0) | 2.49 (1.50-3.13) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 401.7 (359.2-424.5) | not measured | 10.06 (9.50-11.26) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 820.4 (813.0-824.5) | not measured | 19.61 (19.52-19.79) |
| agent session, follow-up turns | 13542 (13533-13598) at the last turn | see below | up to 64 | 3 | see below | 21.4 (17.2-31.6) | 8.83 (8.24-9.52) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.6 % (69.5-69.8); earlier-turn tokens read again per session: 73 (3-144) (earlier turns that the cache did not hold; the 12724 (12723-12725) new tokens of the files are read once); first turn (nothing cached) first token: 7.34 s (7.26-7.36).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 85.3 | 7.364 | 15.6 |
| 1 | 3837 | 633 | 3204 | 348.6 | 9.52 | 18.9 |
| 2 | 5370 | 3900 | 1470 | 175.7 | 8.547 | 20.7 |
| 3 | 8614 | 5434 | 3180 | 341.4 | 9.413 | 20.2 |
| 4 | 10783 | 8619 | 2164 | 249.1 | 8.759 | 23.1 |
| 5 | 13542 | 10832 | 2710 | 301.9 | 9.033 | 21.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 406 s.
