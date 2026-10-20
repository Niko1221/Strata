# Community benchmark on NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 5060 Ti, 64 GiB VRAM
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 125.7 GiB; storage: not measured
- OS: Linux 7.2.11-1-cachyos; GPU driver: 615.78.08; CUDA: not read
- Strata: project 0.1.45.3, commit 3d1fb480; engine 0.1.45.3
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: UD-IQ4_XS
- Context: 200192 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack ud-iq4_xs-up --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 32768 --spec 4 --spec-min-p 0.5 --mtp rt-es --max-context 200192 --kv int8 --kv-resident 32768 --ple-io ram --adapt-every 1 --adapt-swaps 192 --pool-workers 10 --layer-split auto --split-device 1 --expert-cache-device1 auto --remote-expert-opt --vram-reserve-mib 832 --vram-reserve-later-mib 1148
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 122.5 (109.4-153.6) | 0.24 (0.21-0.27) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 855.0 (854.5-858.0) | not measured | 4.72 (4.70-4.72) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1830.4 (1828.4-1831.0) | not measured | 8.80 (8.79-8.81) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1917.4 (1913.4-1923.6) | not measured | 16.75 (16.69-16.78) |
| agent session, follow-up turns | 14636 (14604-14639) at the last turn | see below | up to 64 | 3 | see below | 118.0 (96.4-143.2) | 4.67 (4.29-4.87) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.3); earlier-turn tokens read again per session: 74 (73-74) (earlier turns that the cache did not hold; the 13746 (13744-13748) new tokens of the files are read once); first turn (nothing cached) first token: 2.85 s (2.84-2.85).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 215.1 | 2.849 | 105.5 |
| 1 | 4057 | 638 | 3419 | 712.2 | 4.823 | 118.0 |
| 2 | 5590 | 4121 | 1469 | 343.9 | 4.29 | 112.6 |
| 3 | 9023 | 5653 | 3370 | 700.9 | 4.829 | 96.4 |
| 4 | 11770 | 9016 | 2754 | 592.5 | 4.669 | 133.9 |
| 5 | 14636 | 11829 | 2807 | 603.1 | 4.675 | 123.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 194 s.
