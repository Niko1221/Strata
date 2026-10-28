# Community benchmark: 2x RTX 3090 on PCIe 3.0, dual Xeon E5-2620 v3 (Dell PowerEdge T630)

Measured 2026-10-24 to 2026-10-28 by Cory Jett. Tested: original Flash-Next IQ3_S on two RTX 3090s in a dual-socket
Haswell-EP tower (PCIe 3.0 x16, one card per socket, AVX2 only), layer split, tuned over 26 one-change-at-a-time
rounds for a coding-agent workload, plus one run of the draft standard protocol (custom track) on the final
configuration. Main limitations: apart from the standard run, our own scripts (fresh short prompts, a 32K prompt,
agent-shaped turns and incremental prompt reads, see Method) at the server's temperature 0.6, not greedy; no quality
suite beyond the standard run's sanity checks and a five-prompt greedy output comparison. A PCIe 3.0 data point for
the layer split, and one finding on the CPU share default under a split.

## Standard benchmark (draft protocol)

`strata bench run --profile <production profile> --no-start --track custom` against the already-running production
server (it had answered other requests before; nothing else was sent during the run), 2026-10-28 13:43 UTC. Folder:
[`../2026-10-28-standard-rtx-3090-plus-rtx-3090-5250594b/`](../2026-10-28-standard-rtx-3090-plus-rtx-3090-5250594b/README.md).
Status complete, 21 of 21 required requests, sanity checks 12 of 12 (instruction 4/4, retrieval 4/4, tool calls 4/4).
Medians of 3 (min-max), thinking on:

| Case | Prompt tok/s | Decode tok/s | TTFT s |
|---|---|---|---|
| short (~80 tokens) | n/a | 128.8 (126.7-130.4) | 0.53 (0.52-0.55) |
| prompt-4k | 1,126.7 (1,099.9-1,126.9) | 114.2 (110.4-118.5) | 3.66 (3.66-3.74) |
| prompt-16k | 1,610.4 (1,538.5-1,611.8) | 115.4 (109.2-135.9) | 10.24 (10.20-10.68) |

An earlier run the same morning with `--url` instead of `--profile` (same server, also clean) gave 112.9 / 116.0 /
119.3 tok/s decode and 1,169 / 1,547 tok/s prompt; it was discarded because without a run config the runner recorded
one of the two identical cards and none of the engine options.

## Hardware and software

- GPUs: 2x NVIDIA GeForce RTX 3090 24 GB, each PCIe 3.0 x16, one on each CPU socket (NUMA node 0 and 1). Power
  limit 250 W each (default 350 W). Engine start-up probe 7.8-9.4 GB/s host-to-device under `--interleave=all`.
- CPU: 2x Intel Xeon E5-2620 v3 @ 2.4 GHz (Haswell-EP, 6C/12T each, 24 logical). AVX2 + FMA, **no AVX-512**.
  `performance` governor.
- RAM: 256 GB DDR4-1866 ECC (8x 32 GB), four channels per socket, two NUMA nodes.
- OS: Ubuntu 26.10, kernel 7.3.0. NVIDIA driver 580.178.04 (open kernel modules).
- Strata: 0.2.7 official Linux release, engine 0.1.52 (CUDA 13). No source changes.
- Background: headless; the box serves only this model.

## Model and configuration

- Model: original Qwen3.8-Flash-Next IQ3_S as fetched by setup, plus the MTP head. The GGUFs are copied to
  `/dev/shm` at boot (interleaved over both NUMA nodes), so starts read them from RAM (load ~70-100 s).
- Both cards, `layer_split` 25 (what auto picks) with `--trim-stage-weights`; arena (no `--resident-experts`); the
  whole K/V in VRAM; 99.6-99.8% of decode expert lookups hit VRAM.
- Context 131,072; KV int8; `--prefill 16384`; MTP `--spec 4 --spec-min-p 0.7`; vision encoder on the CPU.
- Conversation cache 40 GiB, 16 slots, min 20,000 tokens. One slot.
- Environment: `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_PREFILL_CPU_SHARE_MAX=4096` (see Findings).
- Server: temperature 0.6, top_p 0.95, top_k 20; `reasoning_budget_tokens` 2048; `fit_max_tokens`.
- Started under `numactl --interleave=all`, `LimitMEMLOCK=infinity`.

```text
numactl --interleave=all strata serve --profile t630      # profile, paths shortened, key removed:
--pack .../packs/iq3_s --native /dev/shm/.../IQ3_S-00001-of-00002.gguf --ple-gguf /dev/shm/.../IQ3_S-00002-of-00002.gguf
--expert-profile data/expert-profile.bin --expert-cache auto --prefill 16384 --spec 4 --spec-min-p 0.7
--mtp .../mtp/rt --max-context 131072 --kv int8 --ple-io ram --vision --vram-reserve-mib 1400 --adapt-async 1
--conversation-cache-mib 40960 --conversation-cache-min-tokens 20000 --conversation-cache-slots 16 --kv-grow
--trim-stage-weights        "gpu": [0, 1], "layer_split": "25"
```

## Method

Every arm is a fresh server start after dropping the page cache, against the same profile with one change. Requests
through `POST /v1/chat/completions` at the server's sampling (temperature 0.6), the way our agents call it.

- **Short prompts:** fixed short coding prompts, thinking off and on; decode tok/s from the stream.
- **32K prompt:** a 32,650-token prompt of real source code with a unique first line per run (no prefix-cache hit);
  prompt tok/s, then decode after it.
- **Agent turns:** a ~61K-token context of real repo files, then four turns over it: two copy-heavy file edits, a new
  unit test, a refactor plan with thinking on. Decode tok/s = (completion tokens - 1) / (last token - first token).
- **Incremental reads:** one conversation over the same ~66K context grows by tool results of 300 to 4,000 tokens,
  twice round, `max_tokens` 1, thinking off; the time to the end of each answer is the read of the new tokens on top
  of the cached conversation. This is the shape of most coding-agent requests (on our box, 26% of busy time is
  prompt reading, most of it 0.5-3K-token increments).
- Medians of 2-3 runs per arm; production is measured at the start and end of each round, and a change counts only
  outside the spread between those two.

## Results

Production, latest round (two production arms, 2026-10-28):

| | Off | On | 32K read tok/s | Decode after 32K |
|---|---|---|---|---|
| production | 150.8 / 142.4 | 121.6 / 123.5 | 2,119 / 2,173 | 105.7 / 108.8 |

Agent turns, decode tok/s (two production arms):

| edit-file | edit-lock | new-test | plan (thinking on) |
|---|---|---|---|
| 145.7 / 146.1 | 153.4 / 161.8 | 106.0 / 104.8 | 101.2 / 102.7 |

Long prompts (engine 0.1.48, before the trim; 2 runs a size): 64K read 2,165-2,387 tok/s, 96K read 2,541-2,754 tok/s
(37-40 s), decode after 99-105 tok/s.

Incremental reads, seconds to read the new tokens and answer one token (median of 4):

| New tokens | 300 | 600 | 1000 | 1500 | 2000 | 2500 | 3000 | 4000 |
|---|---|---|---|---|---|---|---|---|
| CPU share off (0.2.7 default under a split) | 1.74-1.90 | 1.36-1.39 | 1.94-2.31 | 2.84-3.07 | 2.94-3.01 | 3.22-3.24 | 3.35-3.64 | 3.90-4.03 |
| CPU share on, max 4096 | 1.44 | 1.20 | 1.58 | 2.08 | 2.90 | 2.99 | 3.23 | 3.76 |

GPUs averaged ~220 W each under load and peaked at 71-73 C; ~410-430 W at the wall.

## Findings

- **The CPU share helps a PCIe 3.0 layer split.** Under the split ~78% of the experts are on the cards and every
  prompt chunk streams the rest; a 1.5K-token chunk touches nearly all of them, so a small read costs 2-3 s.
  `STRATA_PREFILL_CPU_SHARE=auto` is off by default under a layer split; setting it explicitly (max 4096) cut the
  1K read by ~0.6 s and the 1.5K read by ~0.9 s, -8 to -13% over 300-4,000 tokens, with the 32K read, decode and
  greedy output (five prompts, twice each) unchanged within production's own spread. Possibly worth turning on by
  default for a split on PCIe 3.0, or at least documenting.
- **The prefill help share does not add to it.** `STRATA_PREFILL_HELP=1` alone saved ~0.7 s at 1K, and a fixed 0.45
  share saved 0.3-0.7 s from 1.5K to 3K but lost 0.5 s at 4K; the default size rule was noise here. With the CPU
  share on, the help only runs above the CPU share's max, and lowering that max to let help take 1.5K-4K was no
  faster than the CPU share alone (a local build with a flat 0.45 rule up to 3K or 4K; not proposed).
- **`--trim-stage-weights` at the auto split point:** +5-6% decode, -5% on the 32K read (two confirmation rounds).
  Kept: our prompts are ~99% cached. Other split points (21, 23, 27) cost 10-17% of the 32K read.
- **NUMA:** with one card per socket, `numactl --interleave=all` beat `--preferred=0` on the 32K read on all three
  starts (2,254-2,326 vs 2,055-2,175 tok/s). Without numactl, about half of the starts put the pinned buffers on
  the far node: the start-up probe reads ~4-5 GB/s instead of 8-12 and the whole session is slower (up to -37%
  read, -20% decode on one card). A warning when the probe reads low would have saved us a round.
- **No gain here** (inside the spread, or slower): `--spec 6` and `8`, `--suffix-draft 2`, `--pipeline-windows 2`,
  `--pcie-frac 1.0` with `STRATA_PCIE_MIN1=1`, `STRATA_SPEC_ADAPT_SM86`, `STRATA_GR_UP_WIDE`, `STRATA_EXPDEC`,
  `STRATA_SPLIT_OVERLAP`, `STRATA_VERIFY_DEVICE_PLAN`, Foresight, `--conversation-cache-park-gain`, a 300 W power
  limit (the cards draw more, decode is not power-limited at 250 W), our own expert profile, deep CPU C-states off, and a tuned `STRATA_MMVQ_IL_ROWS` table for the 3090 (20-43% faster dense
  projections at 2-4 tokens in isolation, bit-identical, but nothing end to end). The worker pool on one socket
  (`STRATA_NUMA_POOL=0`) was 2-6% slower. A second slot costs a single client ~25%.
- **`--prefill 32768` reads a 32K prompt 25% slower** than 16384 (1,591 vs ~2,120 tok/s); not looked into.

## Files

Home paths, LAN addresses and host names replaced with placeholders; API keys were redacted when the configs were
written.

- `raw/r21/`: spec depth and suffix drafts, with agent turns (the agent-turns numbers above).
- `raw/r24/`, `raw/r26/`: incremental-read curves for the CPU share and the prefill help share (r26 includes the
  local-build arms `own-ctl`, `split3k`, `split4k`, `help3k-only`).
- `raw/r25/`: production against the CPU share at max 4096 (the production speed numbers above).
- Each round: `sweep.jsonl` (one line per arm, every run parsed), `sweep.log`, `speed-<arm>-<n>.txt` (each run's raw
  output), `config-<arm>.json` (redacted), `engine-<arm>.log`; `raw/t630-arms-r*.json` are the arm definitions.
- `scripts/`: `t630-sweep.py` (one arm per fresh start), `speed-checks.py`, `agent-turns.py`, `agent-and-speed.py`,
  `prefill-curve.py`, `prefill-table.py`. They read from a private repository (the agent-turns context and the 32K
  prompt are its source files), so they document the method rather than run as is.
- Not included: the raw files of the earlier rounds behind the trim, NUMA, long-prompt and opt-in-switch findings
  (2026-10-24 to 10-28); those numbers are summarized from our own log.
