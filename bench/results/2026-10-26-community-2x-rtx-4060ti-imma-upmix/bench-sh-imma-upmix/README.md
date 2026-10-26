# Community benchmark on NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 4060 Ti

Measured on 2026-10-26 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4060 Ti + NVIDIA GeForce RTX 4060 Ti, 32 GiB VRAM
- CPU: AMD Ryzen Threadripper PRO 3975WX 32-Cores (32 cores, 64 threads); installed RAM: 125.6 GiB; storage: not measured
- OS: Linux 7.1.5-76070105-generic; GPU driver: 610.57.04; CUDA: not read
- Strata: project 0.1.51, commit 4b9e05a2; engine 0.1.51
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt-iten --max-context 262144 --kv int8 --remote-expert-opt --pcie-frac 0.00 --spec-min-p 0.50 --kv-resident 32768 --vision --vram-reserve-mib 700 --conversation-cache-mib 8192 --conversation-cache-slots 4
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 52.0 (47.6-75.7) | 0.59 (0.56-0.67) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 879.2 (872.0-882.7) | not measured | 4.61 (4.59-4.64) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1758.6 (1758.0-1761.8) | not measured | 9.18 (9.16-9.18) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2043.3 (2028.9-2054.2) | not measured | 15.74 (15.66-15.86) |
| agent session, follow-up turns | 12743 (12743-12773) at the last turn | see below | up to 64 | 3 | see below | 59.8 (53.7-70.4) | 4.44 (4.12-4.90) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.1-69.3); earlier-turn tokens read again per session: 73 (1-73) (earlier turns that the cache did not hold; the 11896 (11896-11896) new tokens of the files are read once); first turn (nothing cached) first token: 3.46 s (3.26-3.54).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 188.7 | 3.263 | 62.2 |
| 1 | 3217 | 638 | 2579 | 565.0 | 4.604 | 58.1 |
| 2 | 4750 | 3210 | 1540 | 357.2 | 4.348 | 59.5 |
| 3 | 7994 | 4814 | 3180 | 654.9 | 4.896 | 59.1 |
| 4 | 10011 | 8047 | 1964 | 446.3 | 4.44 | 53.7 |
| 5 | 12743 | 10037 | 2706 | 595.7 | 4.582 | 67.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 213 s.
