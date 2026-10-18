# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores, 32 threads); installed RAM: 93.3 GiB; storage: SSD (read speed not measured)
- OS: Windows 10 (10.0.19045); GPU driver: 591.86; CUDA: 13.1
- Strata: project 0.1.45, commit 477ac13c; engine 0.1.45
- Power limit: 600 W; background workloads: idle (only Strata and Claude/ChatGPT were open)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0
- Experimental speed projection: on (layers 4-44, project/per-layer); vision: on

```text
strata.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 4 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --vram-reserve-mib 1920 --conversation-cache-mib 16384 --conversation-cache-slots 4 --vision --control-vector-scaled Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --adapt-async 1
```

Environment of the config: `STRATA_GDN_CHUNKED=1`, `STRATA_PIN_BY_NEED=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 172.6 (166.4-207.1) | 0.21 (0.18-0.23) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3745.6 (3608.7-3786.9) | not measured | 1.09 (1.09-1.15) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 6003.7 (5948.3-6075.2) | not measured | 2.71 (2.66-2.73) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 6403.8 (6376.9-6430.6) | not measured | 5.03 (5.01-5.08) |
| agent session, follow-up turns | 14584 (14546-14587) at the last turn | see below | up to 64 | 3 | see below | 182.1 (160.1-244.4) | 0.92 (0.73-1.14) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 5 (4-74) (earlier turns that the cache did not hold; the 13719 (13717-13719) new tokens of the files are read once); first turn (nothing cached) first token: 0.70 s (0.65-0.71).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 958.8 | 0.651 | 160.8 |
| 1 | 3751 | 629 | 3122 | 3051.5 | 1.061 | 188.5 |
| 2 | 5284 | 3814 | 1470 | 1860.1 | 0.827 | 168.5 |
| 3 | 8717 | 5277 | 3440 | 3110.0 | 1.138 | 178.3 |
| 4 | 11636 | 8774 | 2862 | 2965.8 | 0.997 | 172.9 |
| 5 | 14584 | 11685 | 2899 | 3118.9 | 0.971 | 160.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; the power limit and background load are listed above. Total time of the suite: 57 s.
