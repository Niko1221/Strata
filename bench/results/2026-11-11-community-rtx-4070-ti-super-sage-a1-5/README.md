# Community benchmark on NVIDIA GeForce RTX 4070 Ti SUPER

Measured on 2026-11-11 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB VRAM
- CPU: AMD Ryzen 9 7950X 16-Core Processor (16 cores, 32 threads); installed RAM: 63.1 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 610.47; CUDA: not read
- Strata: project 0.1.48, commit 379f0950; engine 0.1.48
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: sage-a1.5
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 117.7 (107.7-124.7) | 0.14 (0.13-0.17) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3136.2 (2872.6-3136.9) | not measured | 1.33 (1.33-1.44) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 3179.0 (3145.7-3184.5) | not measured | 5.07 (5.06-5.12) |
| agent session, follow-up turns | 14511 (14506-14517) at the last turn | see below | up to 64 | 3 | see below | 126.1 (88.9-147.3) | 1.30 (0.71-1.40) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.5); earlier-turn tokens read again per session: 1 (1-3) (earlier turns that the cache did not hold; the 13728 (13728-13729) new tokens of the files are read once); first turn (nothing cached) first token: 0.67 s (0.67-0.68).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 925.8 | 0.675 | 93.5 |
| 1 | 4089 | 619 | 3470 | 2547.2 | 1.396 | 88.9 |
| 2 | 5598 | 4131 | 1467 | 2037.8 | 0.727 | 119.3 |
| 3 | 9009 | 5640 | 3369 | 2524.9 | 1.341 | 126.1 |
| 4 | 11752 | 9045 | 2707 | 2106.3 | 1.294 | 122.5 |
| 5 | 14517 | 11801 | 2716 | 2111.8 | 1.295 | 147.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 61 s.

## Notes for this run

- Build: `q36-r2` branch of the `xyehya/Strata` fork (commit `379f0950`), CUDA
  source build on Windows. This is the q36-engine fork the model's HF page
  targets, not an upstream release.
- Model: `ykarout/Strata-Sage-A1.5-GGUF` / `Sage-A1.5-Q4_K_M-ssmbf16.gguf`
  (sha256 `b10b9ed79ba5081dbce88f11bf067fcc060cf08af40437424b40653888c448c1`),
  base `InternScience/Sage-A1.5` (35B-A3B MoE, 262K context). That file is not
  identical to the official `InternScience/Sage-A1.5-Q4_K_M-GGUF` (sha256
  `a64fe0bbf936db3eb54f66f142c2b2eee569ade83673649f549a21daef8ca395`); see the
  ykarout repo's README (60 DeltaNet gate tensors kept in BF16).
- Agent-session numbers depend on a one-line local chat-template fix (a broken
  `preserve_thinking` condition baked into the GGUF made the agent cache hit
  0%; with the fix it is 69.47%). See Niko1221/Strata issue #5758.
- Launch arguments (the model page's recommended config): `--native
  <gguf> --max-context 262144 --prefill auto --mtp 2 --kv int8 --pool-workers
  15 --gqa-attn --hybrid-zero-copy --expert-prefetch 2 --expert-profile
  expert-profile-agent-v2.prof --decode-swaps 64 --snapshots 2 --spec-min-p
  0.5 --mtp-sampled q`. The expert profile ships in the ykarout repo.
- One parallel slot; the GPU was otherwise idle.
