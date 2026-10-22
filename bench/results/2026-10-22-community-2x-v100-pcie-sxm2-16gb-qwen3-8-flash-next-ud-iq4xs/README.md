# Community benchmark on 2x Tesla V100 16 GB (PCIe + SXM2), Qwen3.8-Flash-Next UD-IQ4_XS

Measured on 2026-10-22 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPUs: **2x Tesla V100 16 GB** — one retail PCIe card and one SXM2 module on a PCIe adapter board (both on PCIe Gen3; the SXM2 module runs its original SXM2 vBIOS, power-capped to 250 W profile). Enforced power limit: 250 W on each. A third card (TITAN V 12 GB) is installed in the box for side workloads but was explicitly pinned off for inference — only the two V100s served this suite
- CPU: Intel Core i9-9820X @ 3.30 GHz (10 cores, 20 threads); installed RAM: 128 GiB DDR4, trained to 2400 MT/s (the third GPU forces the lower training rate on this platform); storage: NVMe
- OS: Linux 7.0.0-38-generic; GPU driver: 580.178.04; engine built against CUDA 12.8 (nvidia-smi reports 13.0)
- Strata: project 0.1.47, commit 011c9e60, engine 0.1.47, local source build (CUDA12, archs [70], CUDA 12.8 toolkit, gcc 13 host compiler)
- Background workloads: this is a live agent host — the same server answers an interactive agent between benchmark runs, and a browser/office stack runs beside it. No other GPU workloads were active during the suite (the third card stayed idle)

## Model and configuration

- Model and quantization: **Unsloth UD-IQ4_XS** (`unsloth/Qwen3.8-Flash-Next-GGUF`, 3 shards, 93.7 GB; experts uniform UD-IQ4_XS 55.4 GiB, n-gram PLE read by direct per-row SSD reads, attention Q8_0), native-experts pack built with `iq_pack.py --compat-bf16`
  - NOTE: `results.json` shows the served id `qwen3.8-flash-next-iq4_xs` — that is this install's display label kept stable across client configs; the weights actually loaded are the Unsloth UD-IQ4_XS files named above
- Context: 262,144 tokens; parallel slots: 1; KV int8 with `--kv-resident 32768` (KV streaming on the layer split)
- Dual-GPU: `--layer-split auto`, expert cache auto; the engine startup on this build reports 4,149 VRAM expert slots / 9.40 GiB total across the split
- Live engine flags (from the running process this suite benched): `--pack --native <UD-IQ4_XS shard 1> --expert-profile --expert-cache auto --prefill auto --spec 4 --mtp --max-context 262144 --kv int8 --kv-resident 32768 --remote-expert-opt --vram-reserve-mib 768 --vram-reserve-later-mib 1105 --spec-min-p 0.5 --vision --pipeline-windows 2 --layer-split auto`
- Engine env on this box (a locally tuned combo, measured +11% chat decode vs the same build with defaults, +10% agent decode; see notes below): `STRATA_SM70_TABLE=1`, `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_CPU_IQ_ALL_EXACT1=1`
- Thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running, so no engine-log excerpt is bundled and model loading is not timed). One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 62.3 (54.0-69.3) | 0.49 (0.41-0.66) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 633.9 (630.6-634.4) | not measured | 6.37 (6.36-6.40) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1387.2 (1383.9-1388.6) | not measured | 11.61 (11.60-11.64) |
| agent session, follow-up turns | 14620 (14606-14626) at the last turn | see below | up to 64 | 3 | see below | 74.7 (66.8-81.8) | 6.54 (6.13-6.72) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.7 % (69.5-69.7); earlier-turn tokens read again per session: 3 (1-74) (earlier turns that the cache did not hold; the 13728 (13727-13732) new tokens of the files are read once); first turn (nothing cached) first token: 4.38 s (4.35-4.39).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 140.8 | 4.35 | 43.6 |
| 1 | 4101 | 633 | 3468 | 519.8 | 6.701 | 67.0 |
| 2 | 5634 | 4094 | 1540 | 252.1 | 6.137 | 71.4 |
| 3 | 9063 | 5694 | 3369 | 503.5 | 6.721 | 71.1 |
| 4 | 11827 | 9119 | 2708 | 414.9 | 6.556 | 80.4 |
| 5 | 14606 | 11889 | 2717 | 419.8 | 6.502 | 74.7 |

### Same box, previous quant (context for the 16 GB pair, not this run)

This machine previously ran IQ3_S weights on engine 0.1.45.3 / 0.1.46.1; the UD-IQ4_XS swap cost roughly -17% decode and -12 to -22% prefill at the same engine, in exchange for zero sub-IQ3_S expert bytes. The `pw2 + ADAPT trio` combo in the env block above was measured on the same build and is why those flags are live. Earlier dual-16 GB entries from this machine: `2026-10-04` (issue #2845 era, IQ3_S) — this folder supersedes nothing; it is the same rig one engine-generation and one quant later.

Per-request data: `results.json` (its machine block is sanitized by the tool to a single GPU — both V100s above ran, with the layer split active).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. This box is not deterministic at temperature 0 (int8 KV streaming + adaptive expert tiers), so cross-run comparisons should use medians of 3, not single runs. Total time of the suite: 291 s.
