# Community benchmark on AMD Radeon

Measured on 2026-11-01 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 16 GiB VRAM
- CPU: Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz (10 cores, 20 threads); installed RAM: 125.5 GiB; storage: sda     SAMSUNG MZ7KM960HAHP-000D6 sata; sdb     SD/MMC CRW                 usb; sr0     PLDS DVD+/-RW DU-8A5LH     sata; zram0; nvme0n1 TEAM TM8FP6001T            nvme
- OS: Linux 7.2.11-200.fc44.x86_64; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.55, commit c62d5e3a; engine 0.1.55
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --ple-io ram --kv-resident 32768 --vision
```

Environment of the config: `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_ROCBLAS_TUNING=hip`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 31.3 (27.1-34.3) | 1.16 (1.02-1.22) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 453.9 (451.1-458.0) | not measured | 8.90 (8.82-8.96) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 483.2 (482.6-487.6) | not measured | 33.28 (32.99-33.33) |
| agent session, follow-up turns | 12935 (12917-12951) at the last turn | see below | up to 64 | 3 | see below | 35.9 (28.8-46.4) | 6.38 (4.83-7.71) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.5); earlier-turn tokens read again per session: 73 (3-73) (earlier turns that the cache did not hold; the 12064 (12062-12064) new tokens of the files are read once); first turn (nothing cached) first token: 4.96 s (4.81-5.07).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 127.8 | 4.81 | 31.0 |
| 1 | 3366 | 638 | 2728 | 431.4 | 6.371 | 30.7 |
| 2 | 4899 | 3430 | 1469 | 296.6 | 4.998 | 28.8 |
| 3 | 8143 | 4892 | 3251 | 424.4 | 7.707 | 37.9 |
| 4 | 10212 | 8206 | 2006 | 387.2 | 5.228 | 35.6 |
| 5 | 12951 | 10268 | 2683 | 422.8 | 6.395 | 34.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 315 s.
