# Community benchmark on NVIDIA RTX PRO 4000 Blackwell

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 4000 Blackwell, 24 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 9 285K (24 cores, 24 threads); installed RAM: 125.3 GiB; storage: not measured
- OS: Linux 6.17.0-1033-oem; GPU driver: 595.99.02; CUDA: 13.2
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:16384 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --pool-workers 15
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ADAPT_OVERLAP=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 89.1 (76.5-107.0) | 0.36 (0.30-0.43) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2465.8 (2408.6-2470.2) | not measured | 1.65 (1.65-1.69) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2874.4 (2873.4-2882.5) | not measured | 5.61 (5.59-5.61) |
| agent session, follow-up turns | 14593 (14574-14623) at the last turn | see below | up to 64 | 3 | see below | 96.2 (88.3-127.6) | 1.58 (1.31-1.75) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 2 (2-3) (earlier turns that the cache did not hold; the 13729 (13728-13731) new tokens of the files are read once); first turn (nothing cached) first token: 0.83 s (0.83-0.84).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 743.0 | 0.839 | 76.6 |
| 1 | 3759 | 637 | 3122 | 1880.7 | 1.683 | 95.1 |
| 2 | 5292 | 3822 | 1470 | 1110.9 | 1.345 | 94.3 |
| 3 | 8736 | 5356 | 3380 | 1963.1 | 1.746 | 94.5 |
| 4 | 11662 | 8799 | 2863 | 1815.9 | 1.602 | 105.5 |
| 5 | 14623 | 11725 | 2898 | 1856.5 | 1.587 | 96.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 74 s.
