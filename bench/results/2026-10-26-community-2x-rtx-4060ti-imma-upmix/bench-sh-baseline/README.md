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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 55.4 (48.6-75.6) | 0.59 (0.56-0.66) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 872.7 (870.1-876.5) | not measured | 4.64 (4.62-4.65) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1670.0 (1668.0-1673.1) | not measured | 9.66 (9.64-9.67) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1956.0 (1949.9-1977.3) | not measured | 16.45 (16.27-16.50) |
| agent session, follow-up turns | 12758 (12756-12763) at the last turn | see below | up to 64 | 3 | see below | 61.7 (55.5-68.0) | 4.48 (4.18-4.90) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.1-69.1); earlier-turn tokens read again per session: 74 (74-75) (earlier turns that the cache did not hold; the 11896 (11895-11897) new tokens of the files are read once); first turn (nothing cached) first token: 3.51 s (3.25-3.57).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 189.2 | 3.254 | 49.0 |
| 1 | 3211 | 633 | 2578 | 571.5 | 4.55 | 59.6 |
| 2 | 4744 | 3204 | 1540 | 354.1 | 4.387 | 62.2 |
| 3 | 7988 | 4807 | 3181 | 672.3 | 4.772 | 62.1 |
| 4 | 10010 | 8046 | 1964 | 458.7 | 4.323 | 56.3 |
| 5 | 12763 | 10057 | 2706 | 596.2 | 4.579 | 64.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 217 s.
