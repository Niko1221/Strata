# Community benchmark on NVIDIA GeForce RTX 4090 (48 GiB)

Measured on 2026-10-15 by jiasu-hezhip with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). What was tested: text-only API serving of the Unsloth UD-IQ4_XS file at the install's own settings, once with defaults and once with `--clients 4 --long`. Main limitation: speed only, no correctness check, and the server runs a single parallel slot, so the concurrency test measures queueing rather than batched decode.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4090, 48 GiB VRAM (48 GiB variant reporting 49140 MiB)
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 123.4 GiB
- Storage: NVMe WD PC SN810 SDCQNRY-1T00 953.9 GB (OS, model files and packs on this drive); PCIe link: Gen4 x16 (current = maximum)
- OS: Linux 6.8.0-124-generic (Ubuntu 22.04.5); GPU driver: 580.126.18; CUDA: 13.0 (toolkit used for the local engine build)
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1, compiled from source (no ready-made Linux engine at this version)
- Background workloads and power limits: not recorded (a gurobi solver service runs on the same host but was idle)

## Model and configuration

- Model repository and revision: unsloth/Qwen3.8-Flash-Next-GGUF @ 38bb39ee97821de2c9009abb7e93950eec396e66 (2026-09-30), quantization UD-IQ4_XS (~4-bit Unsloth Dynamic), GGUF filenames `Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf` (93.7 GB total, read in place from the user's download folder via `--native`)
- Vision encoder: not used (`images: off`; text-only requests)
- MTP draft layer: from the original Qwen/Qwen3.8-Flash-Next checkpoint (fetched via ModelScope), speculative decoding `--spec 4 --spec-min-p 0.5`
- Context: 131072 tokens; KV: int8 with `--kv-resident 32768`; expert cache: auto (all 55.4 GiB of experts resident in RAM, 39.8 GiB of them mirrored into the GPU's expert cache); prefill: auto; parallel slots: 1; sampling: temperature 0 for the suite; thinking off
- Custom packs or profiles: the setup's own `unsloth-ud-iq4_xs` pack (`--compat-bf16`, built by `tools/iq_pack.py`) and the stock `expert-profile.bin`

```text
strata --pack unsloth-ud-iq4_xs --native Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py` (first run: no arguments; second run: `--clients 4 --long`) against the HTTP API of a Strata server that was already running; model loading is not timed and no engine-start log excerpt exists. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests. `results.json` is the per-request data of the second (fuller) run; `results-earlier-run.json` is the same suite run about five minutes earlier with identical configuration (all rows except the 32K prompt and concurrency), as a cross-check — the two agree (e.g. short-chat decode 148.5 vs 152.3 tok/s, agent cache hit 68.8% vs 68.5%).

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K / 32K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency: 4 requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 152.3 (133.6-204.0) | 0.17 (0.16-0.19) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 4018.2 (4015.8-4020.2) | not measured | 1.01 (1.01-1.01) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 5243.1 (5205.9-5248.9) | not measured | 3.08 (3.08-3.10) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 5370.0 (5362.7-5373.1) | not measured | 5.99 (5.99-6.00) |
| agent session, follow-up turns | 14597 (14590-14611) at the last turn | see below | up to 64 | 3 | see below | 157.9 (135.2-181.6) | 1.06 (0.96-1.12) |

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.5 % (68.5-68.8); earlier-turn tokens read again per session: 143 (4-145) (earlier turns that the cache did not hold; the 13730 (13730-13731) new tokens of the files are read once); first turn (nothing cached) first token: 0.49 s (0.49-0.50).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1279.9 | 0.489 | 150.6 |
| 1 | 3756 | 634 | 3122 | 2956.7 | 1.07 | 174.1 |
| 2 | 5289 | 3819 | 1470 | 1558.7 | 0.958 | 144.9 |
| 3 | 8733 | 5352 | 3381 | 3092.5 | 1.11 | 137.1 |
| 4 | 11650 | 8788 | 2862 | 2727.3 | 1.067 | 181.6 |
| 5 | 14590 | 11691 | 2899 | 2771.8 | 1.064 | 139.1 |

### Concurrency: 4 clients

Total 114.8 tok/s (112.6-115.5); first token p50 1.952 s, p95 3.665 s; request latency p50 2.796 s, p95 4.5 s; errors 0. Server parallel slots: 1 (requests beyond that wait their turn).

## Correctness and limitations

Not measured: correctness (this is a speed suite; no expected-answer checks were run), the vision path (images off), and batched decode (the server kept its single parallel slot, so the 4-client test exercises queueing only). The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 46 s (first run), 77 s (second run). No failures or skipped cases.
