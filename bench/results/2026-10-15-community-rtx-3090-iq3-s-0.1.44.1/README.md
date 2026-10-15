# Community benchmark on NVIDIA GeForce RTX 3090

Measured on 2026-10-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3090, 24 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 9 285K (24 cores, 24 threads); installed RAM: 95.5 GiB DDR5 (a 6800 kit running at 5600); storage: WD SN8100 NVMe (the model's files)
- OS: Windows 11 (10.0.26300); GPU driver: 617.42; CUDA: 13.4 toolkit for the build
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1, a source build of that commit (CUDA 13.4, sm_86, Visual Studio 2026), no changes; `strata-base.exe` in the command below is that build
- Background workloads and power limits: other processes used 5-20% of the CPU during the runs (WSL and desktop apps; sampled during the extra measurements below, not during the suite); default power limits

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-base.exe --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 3 --mtp rt --max-context 262144 --kv int8 --kv-resident 65536 --conversation-cache-mib 8192 --conversation-cache-slots 24 --conversation-cache-min-free-mib 8192 --prompt-cache-every 4096 --ple-io direct --pcie-frac 1.00 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97 --pool-workers 15
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 81.8 (73.8-87.5) | 0.54 (0.52-0.58) |
| 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 2086.0 (1989.2-2096.7) | not measured | 1.97 (1.95-2.06) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2703.0 (2697.6-2713.1) | not measured | 5.99 (5.95-5.99) |
| 32K prompt, fresh | 32050 (32050-32050) | 0 (0-0) | 24 cap | 3 | 2780.6 (2740.7-2782.8) | not measured | 11.58 (11.56-11.74) |
| agent session, follow-up turns | 14637 (14612-14665) at the last turn | see below | up to 64 | 3 | see below | 93.0 (55.2-113.5) | 1.82 (1.65-2.07) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.9); earlier-turn tokens read again per session: 74 (4-74) (earlier turns that the cache did not hold; the 13730 (13729-13732) new tokens of the files are read once); first turn (nothing cached) first token: 1.37 s (1.28-1.39).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 647 | 0 | 647 | 488.7 | 1.369 | 68.6 |
| 1 | 3792 | 671 | 3121 | 1557.5 | 2.044 | 55.2 |
| 2 | 5325 | 3855 | 1470 | 844.0 | 1.773 | 86.2 |
| 3 | 8769 | 5388 | 3381 | 1699.3 | 2.037 | 56.6 |
| 4 | 11674 | 8812 | 2862 | 1479.1 | 1.984 | 105.3 |
| 5 | 14612 | 11713 | 2899 | 1540.8 | 1.925 | 84.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 124 s.

## Extra: engine settings and two requests at once (not from the suite)

These ran on the same PC the same day, but on **v0.1.43 with local changes** (hibernate, start-up and
conversation-cache patches that do not touch decode or prompt kernels), not on the build above. Each setting
was tested with its own server on a second port, 3 engine starts per value in ABBAAB order, so drift through
the session cancels out. Each start asked for 400 tokens (thinking off, temperature 0) after prompts of 1,661,
60,677 and 83,622 tokens of prose and a 7,044-token code-editing prompt (copy a function with comments). Medians
of the 3 starts, decode tok/s. Other processes used 4-13% of the CPU. Each A/B ran on top of the settings chosen
by the ones before it, so compare within a row only.

| Change | 1.7K | 7K edit | 61K | 84K | Prompt tok/s at 84K |
|---|---|---|---|---|---|
| `--prefill auto` -> `auto:16384` | 99.3 -> 99.4 | | 82.8 -> 82.5 | 82.1 -> 79.5 | 2206 -> 2440 |
| `--prefill auto:16384` -> `auto:32768` | 99.5 -> 99.9 | 85.1 -> 84.7 | 82.1 -> 81.8 | 81.7 -> 83.2 | 2433 -> 2671 |
| `--spec 4` -> `6` | 99.7 -> 91.7 | | 83.7 -> 81.9 | 83.2 -> 77.6 | no change |
| `--spec 4` -> `3` | 99.0 -> 92.9 | | 81.7 -> 85.2 | 78.9 -> 81.6 | no change |
| `--lookup-chain 0` -> `4` (spec 3) | 92.7 -> 90.0 | 83.1 -> 83.9 | 86.0 -> 86.2 | 79.4 -> 80.1 | no change |
| `--kv int8` -> `q4_0` | 99.9 -> 87.6 | | 83.6 -> 82.7 | 81.8 -> 82.2 | 2199 -> 2108 |
| `--kv-resident 32768` -> `65536` | 93.7 -> 89.3 | 82.5 -> 80.0 | 85.6 -> 87.9 | 79.1 -> 83.5 | no change |

What this machine runs now: `--prefill auto:32768`, `--spec 3`, `--kv-resident 65536`, which is the command above.
The prefill chunk of 32768 reads long prompts about 21% faster than `auto` here (2206 -> 2671 tok/s at 84K).
`--spec 3` and `--kv-resident 65536` cost 3-7% on short prompts and gain 3-6% at 61K-84K; the agent work this PC
does is mostly above 90K tokens of context, so they were kept. Not tested: contexts above 84K in these A/Bs.

**Two requests at once.** `"parallel": 2` with `--batch-mtp` against one request at a time (`--kv-resident 32768`,
`--spec 3`, same v0.1.43 build). Two ~100K-token conversations of different text were primed, then each asked for 400
tokens; 4 engine starts per setting in ABBA order:

| | One at a time | `"parallel": 2` |
|---|---|---|
| Both together, total | 70.3 tok/s (67.6-76.5) | 55.8 tok/s (53.6-57.5) |
| Both together, each request | 83.1 tok/s (the other waits) | 34.4 tok/s |
| One request alone | 83.2 tok/s (81.9-83.9) | 69.8 tok/s (68.2-71.1) |

On this 24 GB card the second slot is slower in every case, also for a single request. The second slot takes
GPU expert-cache room, so more experts come over PCIe (not measured separately here). Not tested: `"parallel": 2`
without `--batch-mtp`, short prompts, more than two slots.
