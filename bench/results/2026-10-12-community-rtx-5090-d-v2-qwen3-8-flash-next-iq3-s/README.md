# Community benchmark on NVIDIA GeForce RTX 5090 D v2

Measured on 2026-10-12 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges), plus the
repository's `tools/needle_bench.py` recall check at 32K (**3/3 needles found**, depths 10/50/90; `needles.json`).
Speeds are the engine's own timings; first-token times are measured by the client over HTTP.

This is the first community report on an **RTX 5090 D v2** (the 24 GB China-market variant).

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 D v2, 24,455 MiB VRAM, power limit 546.25 W (`nvidia-smi`);
  PCIe Gen 4 x16, the engine's startup transfer probe reported 28.4 GB/s host-to-device (`pcie_frac 0.55`)
- CPU: AMD Ryzen 5 5600X 6-Core Processor (6 cores, 12 threads, no AVX-512: the expert kernels run on AVX-2);
  installed RAM: 127.9 GiB (4 x 32 GB DDR4-3600); storage: KINGSTON SNV3S2000G 2 TB NVMe SSD (system drive)
- OS: Windows 11 Pro (10.0.26200); GPU driver: 616.56; CUDA 13 (the engine's bundled `cu13` libraries)
- Strata: project 0.1.43, official Windows release layout (`START-HERE.bat` install, not a git checkout); engine 0.1.43
- Background workloads: the same server was live-serving the interactive agent session that launched this
  benchmark (one request at a time). Two short-chat runs queued behind live agent requests and show TTFT
  outliers of 19.4 s and 23.9 s; the decode throughputs and the fresh-prompt rates are the engine's own
  timings and are not affected by that queueing. No other load was running.

## Model and configuration

- Model and quantization: `qwen3.8-flash-next-iq3_s` — ISTA-DASLab's GSQ-RCO **abliterated** IQ3_S of
  Qwen3.8-Flash-Next (`Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf`, with the PLE table
  of 320,001,536 rows read directly from shard `00002`), native expert pack `packs/iq3_s`, MTP draft layer
  fetched by the installer (`--mtp mtp/rt`)
- Context: 262144 tokens; KV int8; parallel slots: 1; vision on (GPU helper, 1024 image tokens, 1500 MiB reserve)
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0.
  From the engine's startup log: expert cache `auto` settled at **5104 slots / 9.71 GiB of VRAM**,
  profile-prefilled (24576 ranked pairs), no eviction; 46.84 GiB expert arena in host RAM
  (`cudaHostRegister PORTABLE` ok, large pages refused, 4 KB pages); MTP `--spec 4 --spec-min-p 0.5`,
  draft head over 106,299 tokens (the CJK-inclusive subset); prefill `auto`: 8,192-token chunks borrowing
  2187 cache slots (4.12 GiB); CPU expert pool: 5 workers + the host thread; `STRATA_ROUTE_TAIL_SKIP=7`
  on (default); 905 MiB of VRAM free with everything loaded

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 70.3 (60.7-91.1) | 0.49 (0.38-23.94) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 2327.3 (2314.0-2328.2) | not measured | 1.75 (1.75-1.76) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4571.7 (4557.1-4576.4) | not measured | 3.54 (3.54-3.56) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 4678.5 (4668.3-4685.9) | not measured | 6.89 (6.88-6.90) |
| agent session, follow-up turns | 14492 (14478-14503) at the last turn | see below | up to 64 | 3 | see below | 102.2 (76.6-152.8) | 1.70 (1.64-1.72) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.6-68.8); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13633 (13632-13634) new tokens of the files are read once); first turn (nothing cached) first token: 1.07 s (1.04-1.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 596.6 | 1.038 | 56.9 |
| 1 | 3707 | 627 | 3080 | 1824.1 | 1.708 | 76.6 |
| 2 | 5240 | 3700 | 1540 | 942.1 | 1.656 | 90.4 |
| 3 | 8684 | 5303 | 3381 | 1998.1 | 1.713 | 90.7 |
| 4 | 11593 | 8733 | 2860 | 1696.4 | 1.713 | 111.6 |
| 5 | 14478 | 11632 | 2846 | 1698.6 | 1.701 | 102.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (copied from the
server's own log; the benchmark attached to an already-running server).

## Correctness and limitations

The unchanged `tools/needle_bench.py` found **all three needles** at the `32k` length (depths 10%, 50%, 90%;
actual prompt lengths 32,804-32,805 tokens; per-request data in `needles.json`). The two 37 s runs include
queueing behind the live agent session mentioned above; the recall check measures recall on these three
inputs, not general model accuracy.

The conversation uses the model's own replies, so token counts differ a little between runs. Machine details
come from the server, `nvidia-smi` and the OS. Total time of the speed suite: 139 s. Not measured: long
outputs, sampled decoding, thinking, sustained thermal behavior, concurrency, and VRAM/RAM peaks during
inference (the engine's startup figures above are static allocations, not peaks).
