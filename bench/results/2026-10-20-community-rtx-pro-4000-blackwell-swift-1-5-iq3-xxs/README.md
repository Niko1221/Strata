# Community benchmark on NVIDIA RTX PRO 4000 Blackwell

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 4000 Blackwell, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i9-14900T (24 cores, 32 threads); installed RAM: 125.6 GiB; storage: not measured
- OS: Linux 6.18.57-gentoo-x86_64; GPU driver: 615.78.08; CUDA: not read
- Strata: project 0.1.46, commit c093ac26; engine 0.1.46
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 524288 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 524288 --rope-scaling yarn --rope-scale 2 --kv int8 --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 99.5 (86.1-118.8) | 0.30 (0.28-0.33) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2332.3 (2263.5-2333.1) | not measured | 1.75 (1.75-1.80) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2796.8 (2790.7-2808.5) | not measured | 5.77 (5.75-5.78) |
| agent session, follow-up turns | 14672 (14671-14674) at the last turn | see below | up to 64 | 3 | see below | 129.3 (99.0-143.6) | 1.73 (1.54-1.87) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.3-69.4); earlier-turn tokens read again per session: 2 (2-73) (earlier turns that the cache did not hold; the 13825 (13825-13826) new tokens of the files are read once); first turn (nothing cached) first token: 1.16 s (1.12-1.20).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 555.7 | 1.121 | 82.1 |
| 1 | 4088 | 625 | 3463 | 1908.9 | 1.843 | 102.7 |
| 2 | 5621 | 4152 | 1469 | 968.4 | 1.542 | 113.2 |
| 3 | 9054 | 5684 | 3370 | 1867.3 | 1.833 | 99.0 |
| 4 | 11812 | 9106 | 2706 | 1605.6 | 1.716 | 131.9 |
| 5 | 14674 | 11855 | 2819 | 1623.8 | 1.764 | 129.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 76 s.
