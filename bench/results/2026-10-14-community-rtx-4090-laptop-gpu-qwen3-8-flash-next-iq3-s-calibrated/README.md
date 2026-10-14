# Community benchmark on NVIDIA GeForce RTX 4090 Laptop GPU

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090 Laptop GPU, 16 GiB VRAM
- CPU: 13th Gen Intel(R) Core(TM) i7-13700HX (16 cores, 16 threads); installed RAM: 61.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02; CUDA: 13.2
- Strata: project 0.1.44, commit ed4773eb; engine 0.1.44
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 65536 tokens; parallel slots: 1
- Settings: after `./setup.sh --calibrate --model IQ3_S`, which kept `--pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97` (defaults before: `--pcie-frac` 0.55, `--spec-min-p` 0.5, engine-default expert tier); thinking off for the suite; temperature 0
- Storage: NVMe SSD (ext4)

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 65536 --kv int8 --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 48.2 (41.6-64.5) | 0.66 (0.54-0.94) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 734.8 (723.9-737.8) | not measured | 5.50 (5.47-5.59) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1420.8 (1415.9-1422.4) | not measured | 11.34 (11.33-11.38) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1441.9 (1437.7-1449.5) | not measured | 22.28 (22.16-22.35) |
| agent session, follow-up turns | 14560 (14552-14571) at the last turn | see below | up to 64 | 3 | see below | 57.1 (51.7-80.1) | 5.34 (5.17-5.54) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 4 (4-74) (earlier turns that the cache did not hold; the 13728 (13728-13728) new tokens of the files are read once); first turn (nothing cached) first token: 2.34 s (2.29-2.52).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 268.9 | 2.29 | 36.3 |
| 1 | 3750 | 630 | 3120 | 567.4 | 5.54 | 52.4 |
| 2 | 5283 | 3813 | 1470 | 284.0 | 5.226 | 55.6 |
| 3 | 8727 | 5346 | 3381 | 614.8 | 5.534 | 54.3 |
| 4 | 11616 | 8754 | 2862 | 533.8 | 5.393 | 70.8 |
| 5 | 14552 | 11653 | 2899 | 543.8 | 5.365 | 53.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 253 s.

## Compared with the same model before calibration

Same machine, suite and engine (0.1.44). Before = default settings, after = calibrated (this run). Single suite run each.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| short chat decode tok/s | 44.0 | 48.2 | +9.5% |
| short chat first token (s) | 0.78 | 0.66 | -15.6% |
| agent decode tok/s | 52.3 | 57.1 | +9.2% |
| 4K prompt tok/s | 696.3 | 734.8 | +5.5% |
| 16K prompt tok/s | 1372.8 | 1420.8 | +3.5% |
| 32K prompt tok/s | 1402.5 | 1441.9 | +2.8% |
