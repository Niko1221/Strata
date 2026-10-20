# Community benchmark on NVIDIA GeForce RTX 4060 Laptop GPU

Measured on 2026-10-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB VRAM
- CPU: Intel(R) Core(TM) i7-14700HX (20 cores, 28 threads); installed RAM: 31.7 GiB; storage: not measured
- OS: Windows 11 (10.0.26200); GPU driver: 580.97; CUDA: 13.0
- Strata: engine 0.1.45.3 (0.2.0-beta), source build from upstream main `3fa1b524` plus three local patches: three SSE keep-alive events in `serve/server.py` (for a client that drops idle streams), `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` in `setup.py`, and the MSVC `tq_avx2.cpp` unroll fix (Gyro only; Q2_0 does not use it). The engine log line reports the same version.
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-q2_0
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack q2_0 --native Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --prompt-cache-every 8192 --kv q4_0 --kv-resident 20480 --vram-reserve-mib 200 --vision --resident-experts --pool-workers 19
```

Environment of the config: `STRATA_PCIE_AUTO_CANON=1`, `STRATA_ADAPT_OVERLAP=1`, `STRATA_ADAPT_OVERLAP_RESIDENT=1`, `STRATA_DECODE_TIMING=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 37.4 (29.9-49.7) | 0.97 (0.64-1.08) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 266.5 (244.0-267.7) | not measured | 15.17 (15.09-16.54) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 282.2 (279.0-283.4) | not measured | 56.99 (56.78-57.64) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 289.4 (288.6-296.9) | not measured | 110.93 (108.05-111.19) |
| agent session, follow-up turns | 14514 (14504-14536) at the last turn | see below | up to 64 | 3 | see below | 27.5 (9.8-39.2) | 12.40 (10.31-14.47) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.2 % (69.2-69.4); earlier-turn tokens read again per session: 73 (4-74) (earlier turns that the cache did not hold; the 13746 (13744-13746) new tokens of the files are read once); first turn (nothing cached) first token: 7.43 s (7.14-7.94).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 77.9 | 7.942 | 13.1 |
| 1 | 4039 | 621 | 3418 | 239.2 | 14.473 | 30.7 |
| 2 | 5572 | 4102 | 1470 | 124.2 | 11.946 | 34.8 |
| 3 | 9005 | 5635 | 3370 | 263.2 | 12.91 | 25.7 |
| 4 | 11706 | 9023 | 2683 | 225.8 | 12.033 | 31.0 |
| 5 | 14536 | 11729 | 2807 | 228.7 | 12.396 | 9.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 829 s.

**Notes from the author.** 8 GiB VRAM is the smallest tier in `bench/results` (the nearest is a 12 GiB RTX 3060), and this host has 32 GB of RAM, the tier `docs/TEST_REQUESTS.md` asks for. The closest sibling by constraints is `bench/results/2026-10-06-community-rx-6800m` (12 GiB RX 6800M, 31.4 GB RAM, a CPU without AVX-512, Q2_0, `--resident-experts` with `--kv-resident`): it measured 9-27 tok/s decode and 50-114 tok/s prompt, against 24-28 (37.4 on the three short chats) and 266-289 here. The engine start of this run reported a RAM complement of 18.62 GiB of the 29.73 GiB of experts the VRAM cache does not hold (62.7%). On this machine the complement has varied between 13.15 and 18.62 GiB with what else was loaded, and an agent session at 13.15 GiB (44%) measured 24-28 tok/s - between 44% and 62.7% the coverage did not move Q2_0's decode. The expert cache is 1,487 slots because the 8 GiB card also holds the dense weights, the draft head and the streamed KV window.
