# Community benchmark on NVIDIA GeForce RTX 4090 Laptop GPU

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090 Laptop GPU, 16 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i7-13700HX (16 cores, 16 threads); installed RAM: 61.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02; CUDA: 13.2
- Strata: project 0.1.43.1, commit not a git checkout; engine 0.1.43.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 65536 tokens; parallel slots: 1
- Settings: default configuration (not calibrated; defaults `--pcie-frac` 0.55, `--spec-min-p` 0.5); thinking off for the suite; temperature 0
- Storage: NVMe SSD (ext4)

```text
strata --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 46.0 (41.7-60.4) | 0.72 (0.66-1.02) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 893.3 (874.0-898.3) | not measured | 4.52 (4.51-4.63) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1695.6 (1680.6-1699.6) | not measured | 9.51 (9.48-9.60) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1730.6 (1714.1-1734.9) | not measured | 18.58 (18.53-18.75) |
| agent session, follow-up turns | 14498 (14487-14532) at the last turn | see below | up to 64 | 3 | see below | 63.3 (44.4-87.5) | 4.41 (4.26-4.56) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.9); earlier-turn tokens read again per session: 4 (4-4) (earlier turns that the cache did not hold; the 13634 (13634-13635) new tokens of the files are read once); first turn (nothing cached) first token: 1.91 s (1.89-1.92).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 321.9 | 1.915 | 42.6 |
| 1 | 3704 | 624 | 3080 | 680.1 | 4.558 | 50.7 |
| 2 | 5237 | 3767 | 1470 | 342.7 | 4.331 | 61.1 |
| 3 | 8681 | 5300 | 3381 | 750.3 | 4.537 | 63.3 |
| 4 | 11605 | 8744 | 2861 | 640.7 | 4.501 | 82.2 |
| 5 | 14487 | 11641 | 2846 | 645.5 | 4.445 | 54.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 218 s.
