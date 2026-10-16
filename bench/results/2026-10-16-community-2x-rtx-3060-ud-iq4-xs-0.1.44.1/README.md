# Community benchmark on 2x NVIDIA GeForce RTX 3060

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: 2x NVIDIA GeForce RTX 3060, 12 GiB VRAM each; power limit 170 W per card (kept across reboots by a systemd service). The cards plateau near 90 W during decode and reach the limit only in prefill bursts
- PCIe: both cards CPU-direct at 16 GT/s x8 (the CPU's 20 lanes split x8/x8, the NVMe takes the remaining x4); PCIe RX peaks near 14 GB/s during expert streaming
- CPU: 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz (8 cores, 16 threads); installed RAM: 94.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02; CUDA: 13.2
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1 (release build)
- Background workloads: none; the machine is dedicated to this engine

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs (Unsloth UD pack, native experts)
- Context: 131072 tokens; parallel slots: 1
- Layer split: `auto` — the engine runs layers 0-21 on GPU 0 and layers 22-47 plus the resident expert cache (2939 slots, 6.62 GiB) on GPU 1; the rest of the experts tier in system RAM
- Settings: `--spec 4 --expert-cache auto --prefill auto --remote-expert-opt --kv int8 --kv-resident 32768 --vram-reserve-mib 500 --conversation-cache-mib 8192 --conversation-cache-slots 6 --conversation-cache-min-free-mib 2048 --vision --pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`, with `STRATA_STAGE_TRIM=1` and `STRATA_IO_PREFETCH=1` in the environment; the `--pcie-frac`, `--spec-min-p` and the three `--adapt` values come from `./setup.sh --calibrate` on this machine; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 45.2 (42.7-61.7) | 0.62 (0.57-0.71) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 799.4 (789.7-803.0) | not measured | 5.06 (5.04-5.12) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1216.1 (1215.3-1225.6) | not measured | 13.24 (13.14-13.25) |
| agent session, follow-up turns | 14559 (14537-14590) at the last turn | see below | up to 64 | 3 | see below | 52.1 (42.9-62.4) | 5.22 (4.69-5.48) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 74 (4-74) (earlier turns that the cache did not hold; the 13730 (13729-13730) new tokens of the files are read once); first turn (nothing cached) first token: 3.51 s (3.48-3.52).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 176.6 | 3.478 | 43.5 |
| 1 | 3750 | 628 | 3122 | 604.0 | 5.199 | 46.1 |
| 2 | 5283 | 3743 | 1540 | 327.1 | 4.739 | 45.9 |
| 3 | 8727 | 5346 | 3381 | 620.8 | 5.48 | 42.9 |
| 4 | 11613 | 8751 | 2862 | 552.4 | 5.216 | 53.2 |
| 5 | 14537 | 11638 | 2899 | 560.2 | 5.211 | 53.2 |

Per-request data: `results.json`. The suite reused a running server, so no engine-log excerpt is included.

## Correctness and limitations

Not measured: this is a speed suite; no recall (needle) or HumanEval run here, and no concurrency row. The conversation uses the model's own replies, so token counts differ a little between runs. Reproducibility: an earlier identical suite on the same server a few hours before this one reported 46.6 / 798.1 / 1224.5 / 52.8 tok/s (chat / 4K / 16K / agent, medians) with the same 68.6 % cache hit — every headline median within 3 %. The in-run decode ranges are wide (42.9-62.4 tok/s over agent turns) because decode speed follows which experts and KV blocks the tiers happen to hold at the moment of the turn. Total time of the suite: 192 s.
