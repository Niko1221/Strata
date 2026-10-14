# Community benchmark on 6x Tesla V100-PCIE-32GB (32K context)

Measured on 2026-10-13 by zwcf5200 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run. This arm ran the six cards at the install's 32,768-token context; the same machine at 262,144-token context is in `2026-10-13-community-6x-tesla-v100-pcie-32gb-qwen3-8-flash-next-unsloth-ud-q4-k-xl-262k/`.

## Hardware and software

- GPU and VRAM: 6x Tesla V100-PCIE-32GB, 192 GiB VRAM in total
- CPU: Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz (24 cores, 48 threads); installed RAM: 502.6 GiB; storage: not measured
- OS: Linux 5.15.0-191-generic; GPU driver: 580.178.04; CUDA: 13.0
- Strata: project 0.1.43.1, commit a0116b27; engine 0.1.43.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 32768 --kv int8 --ple-io ram --resident-budget-gib 71
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 70.6 (67.0-101.6) | 0.43 (0.38-0.49) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1281.4 (1235.2-1308.9) | not measured | 3.18 (3.11-3.29) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2421.7 (2404.8-2479.8) | not measured | 6.71 (6.55-6.76) |
| agent session, follow-up turns | 14475 (14454-14515) at the last turn | see below | up to 64 | 3 | see below | 86.5 (73.0-101.3) | 3.61 (2.64-3.97) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.5 % (68.5-68.6); earlier-turn tokens read again per session: 144 (74-145) (earlier turns that the cache did not hold; the 13634 (13634-13637) new tokens of the files are read once); first turn (nothing cached) first token: 2.12 s (2.11-2.15).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 294.7 | 2.106 | 83.7 |
| 1 | 3704 | 621 | 3083 | 866.3 | 3.623 | 98.9 |
| 2 | 5237 | 3767 | 1470 | 552.8 | 2.718 | 73.0 |
| 3 | 8681 | 5230 | 3451 | 885.8 | 3.966 | 77.8 |
| 4 | 11605 | 8674 | 2931 | 816.3 | 3.674 | 77.7 |
| 5 | 14515 | 11668 | 2847 | 807.5 | 3.613 | 96.6 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens

Per-request data, including the engine's own timing blocks: `results.json`.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 122 s.
