# Community benchmark on NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU, 8 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 7 255H (16 cores, 16 threads); installed RAM: 63.4 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 596.86; CUDA: 13.2
- Strata: project 0.1.43, commit not a git checkout; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq2_xs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 65536 --kv int8 --kv-resident 32768 --pool-workers 10
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 39.0 (33.2-49.6) | 0.90 (0.68-1.00) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 535.9 (534.8-543.5) | not measured | 7.55 (7.43-7.56) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 537.6 (532.3-539.6) | not measured | 29.92 (29.81-30.22) |
| agent session, follow-up turns | 14470 (14464-14505) at the last turn | see below | up to 64 | 3 | see below | 34.9 (24.1-51.5) | 6.58 (3.69-8.18) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.9); earlier-turn tokens read again per session: 3 (2-3) (earlier turns that the cache did not hold; the 13632 (13632-13633) new tokens of the files are read once); first turn (nothing cached) first token: 1.98 s (1.93-2.04).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 303.7 | 2.042 | 26.0 |
| 1 | 3701 | 621 | 3080 | 447.8 | 6.922 | 35.1 |
| 2 | 5234 | 3764 | 1470 | 396.0 | 3.759 | 32.5 |
| 3 | 8673 | 5293 | 3380 | 416.0 | 8.183 | 24.1 |
| 4 | 11571 | 8711 | 2860 | 444.5 | 6.497 | 39.0 |
| 5 | 14464 | 11619 | 2845 | 422.9 | 6.826 | 25.5 |

Per-request data: `results.json`. No engine log excerpt was generated because the benchmark used an already running server.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 278 s.
