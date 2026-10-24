# Community benchmark on 3x Tesla P100-PCIE-16GB

Measured on 2026-10-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges), against an already
running server. Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed
only in this suite; a separate correctness note is at the end.

`strata_bench.py` records one GPU and the driver's CUDA version (13.0); the corrections are below. `results.json` is the
tool's own output, unchanged.

## Hardware and software

- GPUs: **3x Tesla P100-PCIE-16GB** (sm_60, 48 GiB VRAM in total), power-limited to **140 W** (`nvidia-smi -pl 140`),
  layer split across all three. Links (`nvidia-smi`): GPU0 **PCIe 2.0 x4** (a chipset slot), GPU1 and GPU2 PCIe 3.0 x8
- Board: ASUS Prime X370-Pro; CPU: AMD Ryzen 5 2600 (6 cores, 12 threads); RAM: 64 GB DDR4 (4x 16 GB dual-rank, 2933 MT/s), 60.7 GiB usable;
  storage: 250 GB NVMe
- OS: Ubuntu 24.04, Linux 7.0.0-38-generic; GPU driver: 580.178.04
- Strata: 0.2.1 (commit 90767252), engine 0.1.46, **built from the root `Dockerfile` for sm_60 with CUDA 12.9.1**
  (`-DSTRATA_EXPERIMENTAL_SM60=ON`, `CUDA_ARCHITECTURES=60`, no image reader): the ready-made `0.2.1-cuda12` image has
  no sm_60 engine (#3503)
- Background workloads: the server was also serving a personal assistant agent, idle during the run

## Model and configuration

- Model and quantization: Qwen3.8-Flash-Next GSQ-RCO IQ3_S (`qwen3.8-flash-next-iq3_s`)
- Context: 262,144 tokens; parallel slots: 1; KV int8
- Settings: Docker (`strata launch` preset), `GPUS=0,1,2`, `STRATA_SET="resident-experts kv-resident=32768"`
  (the low-RAM resident mode plus KV streaming: IQ3_S at 256K does not fit 64 GB of RAM in the normal mode), MTP on
  (`--spec 4 --spec-min-p 0.5`), everything else setup's defaults. Engine report: 16,550 expert slots (31.8 GiB) of
  which 6,144 (11.1 GiB) primary, PCIe share 0.04
- Thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --url ...` against the HTTP API of the running Strata server; model loading is
not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the
conversation cache cannot help the fresh-prompt tests. Started with the cards at rest (56 / 44 / 45 °C).

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and
  continue from the model's own earlier answers, 64 new tokens per turn, 3 sessions

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 38.2 (31.6-51.0) | 0.74 (0.68-0.87) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 241.8 (240.5-242.6) | not measured | 16.66 (16.61-16.75) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 383.6 (380.5-386.5) | not measured | 41.92 (41.60-42.25) |
| agent session, follow-up turns | 14649 (14641-14661) at the last turn | see below | up to 64 | 3 | see below | 40.4 (32.2-48.3) | 15.84 (9.06-18.03) |

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.3-69.4); earlier-turn tokens read again per session: 4 (3-74); first
turn (nothing cached) first token: 5.37 s (5.37-5.42).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 114.2 | 5.371 | 33.1 |
| 1 | 4095 | 633 | 3462 | 212.7 | 16.321 | 40.8 |
| 2 | 5628 | 4158 | 1470 | 162.2 | 9.103 | 43.9 |
| 3 | 9061 | 5692 | 3369 | 207.5 | 16.284 | 33.4 |
| 4 | 11802 | 9094 | 2708 | 195.9 | 13.87 | 35.0 |
| 5 | 14649 | 11830 | 2819 | 200.9 | 14.075 | 40.4 |

The conversation cache works (earlier turns are not read again), so a follow-up turn's wait is the new text at
~160-210 tok/s: 9-16 s for 1.5-3.5K new tokens on this machine.

### Thermals

140 W per card, all three near the cap throughout (the layer split keeps every card busy). GPU0 reached 78-80 °C in
the last ~2 minutes of the 476 s suite, with SW thermal slowdown in 5 of 237 two-second samples; GPU1 max 67 °C, GPU2
max 72 °C. The decode numbers of the last agent session may be a few percent low because of it.

## Other measurements on this machine (own script, same server)

Not part of `strata_bench.py`; for context. Fixed prompts sent with `chat_template_kwargs: {enable_thinking: false}`,
cards from rest each time. The raw output of every run is in [own-script/](own-script/) (one line per request:
prompt and decode tok/s from the engine's timings, MTP drafts accepted/proposed, and the start of the answer).

| | Engine 0.1.42 (own sm_60 build) | Engine 0.1.46 (Strata 0.2.1, this report) |
| --- | --- | --- |
| Prompt, 4K / 16K / 32K tokens | 226 / 375 / 485 tok/s | 235 / 380 / 495 tok/s |
| 300-token story, 3 runs | 33.3-34.5 tok/s | 32.7-36.2 tok/s |
| 8 small Python tasks with unit tests | 8/8, 40-49 tok/s, 90 s | 8/8, 44-57 tok/s, 74 s |
| Tool call (one function) | correct | correct |

A harder set of 8 tasks (SemVer sort, cron next-run, JSON parser, regex matcher, OSA distance, CSV parser, bounded
asyncio runner, lexicographic shortest path; each checked against the stdlib or a brute-force reference on random
inputs) on engine 0.1.42: 7/8. For comparison on the same rig with llama.cpp b11540: Qwen3.6-35B-A3B UD-Q4_K_M 6/8,
Qwen3-Coder-Next UD-IQ4_XS 5/8, Laguna XS 2.1 Q4_K_M 4/8.

## Correctness and limitations

`strata_bench.py` itself checks no answers. The conversation uses the model's own replies, so token counts differ a
little between runs. Single machine, single day; the own-script comparison above is one run per engine.
