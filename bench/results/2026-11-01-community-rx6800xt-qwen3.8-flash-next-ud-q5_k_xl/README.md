# Community benchmark on AMD Radeon

Measured on 2026-11-01 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz (10 cores, 20 threads); installed RAM: 125.5 GiB; storage: sda     SAMSUNG MZ7KM960HAHP-000D6 sata; sdb     SD/MMC CRW                 usb; sr0     PLDS DVD+/-RW DU-8A5LH     sata; zram0; nvme0n1 TEAM TM8FP6001T            nvme
- OS: Linux 7.2.11-200.fc44.x86_64; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.55, commit c62d5e3a; engine 0.1.55
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q5_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-q5_k_xl --native Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vision
```

Environment of the config: `STRATA_MMQ_KQ=1`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_ROCBLAS_TUNING=hip`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 21.5 (19.1-23.8) | 1.65 (1.44-1.77) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 424.0 (423.0-433.7) | not measured | 9.54 (9.32-9.57) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 474.1 (473.6-479.0) | not measured | 33.93 (33.59-33.96) |
| agent session, follow-up turns | 12939 (12931-12941) at the last turn | see below | up to 64 | 3 | see below | 23.8 (19.2-27.6) | 8.06 (7.91-8.68) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.4); earlier-turn tokens read again per session: 72 (4-144) (earlier turns that the cache did not hold; the 12063 (12062-12064) new tokens of the files are read once); first turn (nothing cached) first token: 7.39 s (7.19-7.50).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 85.4 | 7.187 | 13.7 |
| 1 | 3348 | 621 | 2727 | 336.6 | 8.157 | 19.2 |
| 2 | 4881 | 3341 | 1540 | 191.4 | 8.099 | 20.7 |
| 3 | 8125 | 4874 | 3251 | 377.1 | 8.684 | 22.2 |
| 4 | 10192 | 8187 | 2005 | 252.4 | 8.012 | 22.5 |
| 5 | 12939 | 10255 | 2684 | 333.9 | 8.109 | 23.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 395 s.
