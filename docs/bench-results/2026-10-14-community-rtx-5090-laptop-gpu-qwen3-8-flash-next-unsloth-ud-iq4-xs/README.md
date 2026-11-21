# Community benchmark on NVIDIA GeForce RTX 5090 Laptop GPU

Measured on 2026-10-14 by **jie3000** with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 Laptop GPU, 24 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 9 275HX (24 cores, 24 threads); installed RAM: 255.7 GiB; storage: not measured
- PCIe link: Gen 5 x16 (nvidia-smi `pcie.link.gen.current` = 5, `pcie.link.width.current` = 16)
- OS: Windows 11 (10.0.26300); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads: not recorded; GPU power limit 150 W (default) / 175 W (max), laptop Dynamic Boost

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 8 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 49152 --resident-experts --vision --vram-reserve-mib 500 --pool-workers 20
```

Environment of the config: `STRATA_PREFILL_CPU_SHARE=auto`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 46.1 (39.6-63.9) | 0.61 (0.47-0.65) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1904.7 (1211.1-1910.5) | not measured | 2.14 (2.14-3.35) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2441.6 (2120.1-2465.7) | not measured | 6.61 (6.57-7.62) |
| agent session, follow-up turns | 14496 (14451-14523) at the last turn | see below | up to 64 | 3 | see below | 50.1 (38.9-69.8) | 1.96 (1.41-4.34) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.4-68.8); earlier-turn tokens read again per session: 3 (2-214) (earlier turns that the cache did not hold; the 13631 (13631-13636) new tokens of the files are read once); first turn (nothing cached) first token: 1.26 s (1.24-1.67).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 375.0 | 1.667 | 20.7 |
| 1 | 3701 | 621 | 3080 | 715.4 | 4.338 | 38.9 |
| 2 | 5234 | 3764 | 1470 | 585.5 | 2.54 | 47.0 |
| 3 | 8678 | 5298 | 3380 | 868.6 | 3.925 | 42.8 |
| 4 | 11589 | 8729 | 2860 | 770.9 | 3.747 | 69.8 |
| 5 | 14496 | 11653 | 2843 | 916.8 | 3.14 | 67.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 118 s.
