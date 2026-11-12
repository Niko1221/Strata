# Community benchmark: Radeon 780M (gfx1103), DeepSeek-V4.1-Flash-1017 IQ1_M on strata-dsv41

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon 780M (gfx1103, integrated, 12 CUs). 512 MiB UMA carveout; the engine uses GTT (system RAM) instead: `ttm.pages_limit=11534336` = 44 GiB. Driver: in-kernel amdgpu; ROCm userspace is the TheRock wheels 7.10.0a20251121 inside the container
- CPU: AMD Ryzen 7 H 255 (8 cores / 16 threads, Zen 4, AVX-512); installed RAM: 80 GB DDR5-5600 (48 + 32 GB, dual channel; 77.9 GiB seen by the host); storage: Samsung 980 PRO 1 TB (PCIe 4.0 x4, ext4) holds the model files
- OS: Proxmox VE 9.2 host, kernel 7.0.14-28-pve; the engine runs in a Debian 13 LXC container (44 GB memory limit) that has /dev/kfd and /dev/dri bound in
- Strata: commit `146df3c`, `strata-dsv41 0.1.58 build 146df3c46`, built from source for gfx1103 (`-DSTRATA_BUILD_DSV41=ON -DCMAKE_HIP_ARCHITECTURES=gfx1103`)
- Background workloads: none; nothing else used the GPU or the NVMe during the run (checked before starting). Host `MemAvailable` was low, about 1.6 GiB, because the expert store had filled up

## Model and configuration

- Model and quantization: DeepSeek-V4.1-Flash-1017 ISTA GSQ-RCO IQ1_M (1.75 bpw, 121 GB) + Engram IQ4_NL (111 GB) + DSpark Q8_0 drafter
- Context: 131072 tokens; parallel slots: 1
- Settings: engine arguments `--max-context 131072 --eos 1 --vision off --dspark on --dspark-conf 0.65 --reserve-mib 700 --experts auto --ssd-cache 24`; environment `STRATA_EXPERIMENTAL_GFX1103=1 DS4_FAST_ATTN=1 STRATA_DS4_HIPK=1` (the last two are not on by default for gfx1103); thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 7.3 (6.8-11.5) | 2.25 (2.15-2.68) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 73.5 (72.3-73.7) | not measured | 56.12 (56.01-57.07) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 125.3 (125.1-125.3) | not measured | 132.24 (132.16-132.39) |
| 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 3 | 136.6 (136.6-137.0) | not measured | 242.51 (241.91-242.52) |
| agent session, follow-up turns | 13277 (13271-13310) at the last turn | see below | up to 64 | 3 | see below | 7.4 (6.6-10.4) | 42.66 (26.38-47.41) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.0 % (70.0-70.1); earlier-turn tokens read again per session: 1 (0-2) (earlier turns that the cache did not hold; the 12427 (12427-12429) new tokens of the files are read once); first turn (nothing cached) first token: 14.39 s (14.36-14.78).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 40.7 | 14.783 | 8.9 |
| 1 | 3801 | 641 | 3160 | 66.7 | 47.406 | 7.4 |
| 2 | 5319 | 3864 | 1455 | 55.2 | 26.377 | 6.7 |
| 3 | 8428 | 5381 | 3047 | 64.4 | 47.348 | 8.5 |
| 4 | 10673 | 8491 | 2182 | 59.7 | 36.577 | 7.3 |
| 5 | 13310 | 10723 | 2587 | 60.4 | 42.869 | 10.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 2262 s.

## Notes from the submitter

The hardware, software, background and settings lines above were corrected by hand, because the tool ran inside a container and could not read the GPU, RAM size or the engine environment (it printed `AMD Radeon, 0 GiB VRAM` and 43 GiB RAM). All measured numbers, the tables and `results.json` are exactly what the tool wrote.

This is the second of two runs on the same server. The first one (`results-first-run-rsync-running.json`) had an rsync backup of the model folder reading from the same NVMe, which I only noticed halfway through. Medians of the two runs:

| Test | first run (with rsync) | this run |
| --- | ---: | ---: |
| 4K prompt, tok/s | 69.7 | 73.5 |
| 16K prompt, tok/s | 121.1 | 125.3 |
| 32K prompt, tok/s | 136.6 | 136.6 |
| short chat decode, tok/s | 7.7 | 7.3 |
| agent session decode, tok/s | 7.6 | 7.4 |
| agent cache hit | 70.0 % | 70.0 % |

So the backup cost the short prompts a few percent and did not show at 32K. The decode differences are within the spread of the runs.

The prompt speed depends on `STRATA_DS4_HIPK=1`: without it the same server did 47 tok/s at 7K and 58 tok/s at 19K tokens in my earlier A/B runs (98 and 121 with it) (see issue #5607). Decode needs `--dspark on`; with the default `auto` the drafter is not loaded on this iGPU.
