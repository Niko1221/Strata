# Community benchmark on RX 5700 XT (gfx1010)

Measured on 2026-10-13 by Gedankenn with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

A second RDNA1 machine against the numbers already in [OLDER_GPUS.md](../../../docs/OLDER_GPUS.md) (RX 5700 XT, Coder IQ1_M, 32K): that report was a 7600X, ROCm 7.14, ~115 / 144 / 149 prompt tok/s at 4K / 16K / 30K. This run is a 4600G with the Renoir iGPU hidden, ROCm 7.10 TheRock `gfx101X-dgpu` wheels, engine 0.1.43.1, setup opt-in `STRATA_EXPERIMENTAL_GFX1010=1`.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 5700 XT 8 GiB (gfx1010, Navi 10); the box also has a Renoir iGPU (gfx90c), hidden with `ROCR_VISIBLE_DEVICES=0`. The 5700 XT drives the desktop.
- CPU: AMD Ryzen 5 4600G (6 cores / 12 threads, AVX2); installed RAM: 46.4 GiB; storage: not measured
- OS: Linux 7.2.5-3-omarchy (Arch); amdgpu kernel driver; no system ROCm
- Strata: project 0.1.43.1, commit a0116b27; engine 0.1.43.1, HIP source build for gfx1010
- ROCm: TheRock wheels `rocm[libraries,devel]==7.10.0a20251120` from `https://rocm.nightlies.amd.com/v2/gfx101X-dgpu/` into `.venv`
- Expert cache at load: 762 experts, 1.47 GiB VRAM, 626 MiB free
- No hipBLASLt tuning table for gfx1010 (plain hipBLAS for dense prompt GEMMs)
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: Qwen3.8-Flash-Next Coder IQ1_M (`qwen3.8-flash-next-coder-iq1_m`)
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0
- Setup: `STRATA_EXPERIMENTAL_GFX1010=1 ./setup.sh --yes --family coder --backend hip --gpu 0 --port 8080 --no-start --vision no`

```text
strata --pack coder-iq1_m --native Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf --expert-profile expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 32768 --kv int8
```

Environment of the config: `ROCR_VISIBLE_DEVICES=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 18.0 (16.7-22.7) | 1.71 (1.51-1.84) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 163.7 (161.5-163.8) | not measured | 24.62 (24.59-24.94) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 162.1 (160.7-163.2) | not measured | 99.16 (98.44-100.02) |
| agent session, follow-up turns | 14417 (14416-14426) at the last turn | see below | up to 64 | 3 | see below | 17.4 (13.6-22.3) | 20.39 (10.44-32.22) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.4-68.7); earlier-turn tokens read again per session: 74 (4-144) (earlier turns that the cache did not hold; the 13634 (13633-13634) new tokens of the files are read once); first turn (nothing cached) first token: 5.56 s (5.55-5.60).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 110.9 | 5.561 | 10.5 |
| 1 | 3701 | 621 | 3080 | 150.0 | 20.609 | 17.5 |
| 2 | 5234 | 3694 | 1540 | 148.5 | 10.437 | 13.6 |
| 3 | 8633 | 5253 | 3380 | 137.9 | 24.577 | 16.7 |
| 4 | 11557 | 8626 | 2931 | 147.6 | 19.926 | 22.3 |
| 5 | 14417 | 11571 | 2846 | 146.5 | 19.496 | 21.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: needle recall, ctest on the card, 30K prompt, quality of answers. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server and the OS; background load and power limits are not recorded. The card drives the desktop (~600 MiB VRAM left). Total time of the suite: 812 s.
