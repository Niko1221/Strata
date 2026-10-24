# Community benchmark on NVIDIA GeForce RTX 5090

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090, 32 GiB VRAM
- CPU: 12th Gen Intel(R) Core(TM) i7-12700KF (12 cores, 20 threads); installed RAM: 95.8 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 617.42; CUDA: not read
- Strata: project 0.1.45.3, commit 3fa1b524; engine 0.1.45.3
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-qwen3.8-flash-next-iq3_s
- Context: 524288 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 140.6 (113.0-162.0) | 0.29 (0.26-0.38) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3107.1 (2800.5-3130.5) | not measured | 1.33 (1.32-1.46) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 5871.3 (5854.0-6026.9) | not measured | 2.77 (2.70-2.77) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 6111.9 (6109.0-6149.1) | not measured | 5.30 (5.26-5.30) |
| agent session, follow-up turns | 14612 (14604-14629) at the last turn | see below | up to 64 | 3 | see below | 194.4 (123.5-252.5) | 1.25 (1.14-1.59) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.3-69.4); earlier-turn tokens read again per session: 73 (3-73) (earlier turns that the cache did not hold; the 13746 (13745-13747) new tokens of the files are read once); first turn (nothing cached) first token: 1.16 s (1.11-1.18).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 564.5 | 1.108 | 94.9 |
| 1 | 4042 | 622 | 3420 | 2207.9 | 1.594 | 123.5 |
| 2 | 5575 | 4105 | 1470 | 1124.0 | 1.342 | 157.9 |
| 3 | 9008 | 5639 | 3369 | 2251.3 | 1.529 | 149.6 |
| 4 | 11755 | 9071 | 2684 | 1897.9 | 1.474 | 178.1 |
| 5 | 14604 | 11797 | 2807 | 2021.3 | 1.443 | 153.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Build and method notes (added by the reporter)

- The engine is **not** a release binary: it is a **local source build of commit `3fa1b524` with
  CUDA 13.1** (nvcc 13.1.115), sm_120 only, `STRATA_PORTABLE=OFF`. The GPU runtime libraries come
  from the `nvidia-cu13` wheels in the venv (cudart 13.0.96, cuBLAS 13.0.2.14); the driver is
  617.42 (CUDA UMD 13.4).
- The model is Swift 1.5 Qwen3.8-Flash-Next IQ3_S with a prepared native pack
  (`D:\Strata-data\packs\swift-iq3_s`) and the engine's expert profile; 524,288-token context,
  `--kv int8 --kv-resident 32768 --conversation-cache-mib 8192 --expert-cache auto --prefill auto
  --spec 4`, MTP draft layer at `D:\Strata-data\mtp\rt`, rope scaling yarn factor 2,
  `--spec-min-p 0.5 --pool-workers 3 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97`.
- Environment: `STRATA_HC_Q8=1`, `STRATA_EXCHANGE_ROTATE=1`, `STRATA_ATTN_DAILY=1`, `STRATA_EXPDEC=1`,
  `STRATA_HC_UPMIX=1`, `STRATA_QSA_SPLIT_ATTN=1`.
- **No calibration was applied**: this is the install's own run configuration, unchanged.
- The server was started by hand and the bench pointed at it with `--url`, because the bench tool's
  own "start the server" path raises a `NameError` (issue #3188). Model loading (114 s) is not timed.
- The same measurement with the 0.1.45 release build, and the old-vs-new comparison, are in
  [`../2026-10-19-community-rtx-5090-engine-ab-0.1.45-vs-0.1.45.3/`](../2026-10-19-community-rtx-5090-engine-ab-0.1.45-vs-0.1.45.3/).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 71 s.
