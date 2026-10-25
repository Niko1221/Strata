# Community benchmark on NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 2080 Ti

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 2080 Ti, 38 GiB VRAM
- CPU: 11th Gen Intel(R) Core(TM) i5-11400F @ 2.60GHz (6 cores, 12 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 6.8.0-138-generic; GPU driver: 595.71.05; CUDA: 13.2
- Strata: project 0.1.48.1, commit 7ddc17a3; engine 0.1.48.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s-calib
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv q4_0 --vision --vram-reserve-mib 700 --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --remote-expert-opt --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97 --pcie-frac 0.35
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 57.4 (52.7-79.0) | 0.56 (0.50-0.61) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 814.2 (785.4-814.9) | not measured | 4.96 (4.96-5.14) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1346.2 (1339.1-1351.2) | not measured | 11.96 (11.91-12.02) |
| agent session, follow-up turns | 14579 (14539-14582) at the last turn | see below | up to 64 | 3 | see below | 69.2 (58.7-86.5) | 4.56 (3.75-5.19) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.6); earlier-turn tokens read again per session: 73 (4-74) (earlier turns that the cache did not hold; the 13730 (13729-13731) new tokens of the files are read once); first turn (nothing cached) first token: 2.75 s (2.71-2.80).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 226.5 | 2.712 | 48.7 |
| 1 | 4100 | 631 | 3469 | 671.6 | 5.193 | 65.0 |
| 2 | 5633 | 4093 | 1540 | 376.5 | 4.115 | 67.0 |
| 3 | 9066 | 5696 | 3370 | 659.2 | 5.139 | 65.5 |
| 4 | 11820 | 9112 | 2708 | 585.6 | 4.651 | 86.5 |
| 5 | 14579 | 11862 | 2717 | 599.0 | 4.563 | 69.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 168 s.
