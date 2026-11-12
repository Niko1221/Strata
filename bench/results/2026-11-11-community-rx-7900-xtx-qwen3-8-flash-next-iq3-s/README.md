# Community benchmark on AMD Radeon RX 7900 XTX

Measured on 2026-11-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i7-14700KF (20 cores, 28 threads); installed RAM: 159.8 GiB; storage: WD Game Drive HDD/USB; Generic Physical Disk Unspecified/Unknown; Samsung SSD 970 EVO 1TB SSD/NVMe; Samsung SSD 870 EVO 4TB SSD/SATA
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.32015.7029; CUDA (driver supports up to): not read
- Strata: project 0.1.59.1, commit 6c642df7; engine 0.1.59
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --pool-workers 13
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100500.txt`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 32.7 (29.4-42.4) | 1.08 (0.89-1.27) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 954.5 (947.5-955.5) | not measured | 4.26 (4.25-4.27) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1101.0 (1099.9-1102.7) | not measured | 14.63 (14.62-14.67) |
| agent session, follow-up turns | 13819 (13814-13852) at the last turn | see below | up to 64 | 3 | see below | 53.5 (40.5-71.9) | 4.27 (3.76-4.55) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.1 % (70.1-70.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12990 (12990-12991) new tokens of the files are read once); first turn (nothing cached) first token: 3.53 s (3.44-3.54).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 178.5 | 3.439 | 38.2 |
| 1 | 4084 | 640 | 3444 | 844.1 | 4.121 | 44.9 |
| 2 | 5617 | 4147 | 1470 | 390.4 | 3.808 | 44.0 |
| 3 | 8863 | 5680 | 3183 | 718.7 | 4.478 | 48.4 |
| 4 | 11090 | 8901 | 2189 | 516.6 | 4.283 | 57.3 |
| 5 | 13814 | 11105 | 2709 | 599.6 | 4.555 | 40.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 201 s.
