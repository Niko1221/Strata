# Community benchmark on AMD Radeon RX 7900 XTX + AMD Radeon RX 7900 XTX

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX + AMD Radeon RX 7900 XTX, 48 GiB VRAM
- CPU: AMD Ryzen 9 PRO 6950H with Radeon Graphics (8 cores, 16 threads); installed RAM: 61.6 GiB; storage: sda     BUP Slim BK           usb; nvme0n1 TWSC TSC3AN1T0-F6Q10S nvme
- OS: Linux 7.0.0-42-generic; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.56, commit ffa80d73; engine 0.1.55
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --vision --vram-reserve-mib 700
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100401.txt`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 38.2 (34.0-40.9) | 0.74 (0.68-0.82) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 427.6 (416.5-428.2) | not measured | 9.43 (9.42-9.68) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 836.6 (827.7-836.8) | not measured | 19.23 (19.23-19.44) |
| agent session, follow-up turns | 13142 (13132-13159) at the last turn | see below | up to 64 | 3 | see below | 38.3 (34.9-55.3) | 7.85 (6.58-9.16) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.3); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12279 (12278-12279) new tokens of the files are read once); first turn (nothing cached) first token: 3.81 s (3.78-3.82).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 167.6 | 3.82 | 38.9 |
| 1 | 3392 | 634 | 2758 | 330.8 | 8.706 | 34.9 |
| 2 | 4925 | 3455 | 1470 | 212.4 | 7.202 | 37.2 |
| 3 | 8169 | 4988 | 3181 | 358.0 | 9.156 | 37.7 |
| 4 | 10397 | 8232 | 2165 | 279.4 | 7.931 | 36.6 |
| 5 | 13159 | 10449 | 2710 | 331.4 | 8.231 | 52.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 285 s.
