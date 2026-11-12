# Community benchmark on AMD Radeon VII

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon VII, VRAM not read
- CPU: AMD Ryzen 9 9900X 12-Core Processor (12 cores, 24 threads); installed RAM: 123.4 GiB; storage: nvme0n1 T-FORCE TM8FPZ004T                       nvme; nvme1n1 CT2000E100SSD8                           nvme
- OS: Linux 6.12.113-0612113-generic; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.54, commit not a git checkout; engine 0.1.54
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s-vii
- Context: 61440 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --mmap-experts --prefill auto --prompt-cache-tail --spec 4 --mtp rt --max-context 61440 --kv int8 --vram-reserve-mib 700 --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.00 --spec-min-p 0.70
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 34.1 (16.7-44.0) | 0.70 (0.65-0.81) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 224.6 (224.6-225.0) | not measured | 17.92 (17.89-17.92) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 320.3 (319.3-320.4) | not measured | 50.17 (50.16-50.32) |
| agent session, follow-up turns | 12948 (12906-12949) at the last turn | see below | up to 64 | 3 | see below | 39.1 (32.2-46.4) | 15.34 (13.90-16.84) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.5); earlier-turn tokens read again per session: 73 (1-143) (earlier turns that the cache did not hold; the 12065 (12065-12066) new tokens of the files are read once); first turn (nothing cached) first token: 11.87 s (11.54-12.48).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 52.9 | 11.542 | 29.9 |
| 1 | 3355 | 626 | 2729 | 178.2 | 15.342 | 39.1 |
| 2 | 4888 | 3348 | 1540 | 109.4 | 14.103 | 35.1 |
| 3 | 8132 | 4881 | 3251 | 195.1 | 16.695 | 38.1 |
| 4 | 10201 | 8196 | 2005 | 138.4 | 14.514 | 32.2 |
| 5 | 12906 | 10223 | 2683 | 173.6 | 15.488 | 40.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 556 s.
