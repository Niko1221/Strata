# Follow-up: 0.1.40.2 → 0.1.41, and int8 vs k8v4 KV — RTX 5090, interleaved protocol

Reported by gravitomagnetic (same machine/reporter as the 2026-10-02 folder in this directory).
Dates: 2026-10-08. Engine kits built locally from tags at CUDA 13.4 kit, arch 120 (see BUILD.json).

## Why an interleaved protocol

Our first runs today used the natural blocked order (A A A then B B B). The last leg of
each block decoded ~30% faster than its own first two legs — a warm-up drift (page cache,
expert-profile settling, clocks), not a version effect. Blocked comparisons at n=3 would
therefore rank versions by run order. We re-ran everything as **matched alternating pairs
(A B A B A B)** so each version occupies each position once. The drift vanished: across
six legs, cold-prefill spread was 0.1 s and greedy decode was identical run to run.
Interleaving is cheap insurance at small n; we'd suggest it for any version A/B on an
expert-streaming rig.

## Machine and configuration (identical for all legs)

- RTX 5090 32 GB, 122 GB system RAM, NVMe; Linux.
- IQ3_S GSQ-RCO pack, native experts, expert-profile; `--expert-cache 6500` floor
  (auto-fills to ~8.1–8.4k slots, 15.4–15.8 GiB), `--prefill auto`, `--spec 4`,
  `--spec-min-p 0.5`, MTP runtime head, `--max-context 1048576`, YaRN scale 4,
  `--kv-resident 32768`, vision on, `--vram-reserve-mib 995`, server `parallel: 2`
  (no `--batch-mtp`).
- Probe set per leg (fixed prompts, greedy where noted):
  1. xhigh cold prefill: 200,073-token prompt, 32-token answer (thinking on, temp 1.0/top_p 0.95)
  2. instruct decode: 256 tokens, count-to-300 (temp 0.7/presence 1.5)
  3. needle recall at ~200K context
  4. greedy decode: 256 tokens, count-to-300 (temp 0) — doubles as the repetition gate
  Engine-side boot lines (expert slots, KV tier) recorded from the serve log per leg.

## A/B 1 — engine 0.1.40.2 vs 0.1.41 (3 matched pairs, A B A B A B)

| probe | 0.1.40.2 (n=3) | 0.1.41 (n=3) |
|---|---|---|
| cold prefill 200K | 58.3 s (58.2–58.3) | 58.2 s (58.1–58.2) |
| instruct decode 256 tok | 2.4 s (2.4–2.5) | 2.4 s (2.4–2.5) |
| greedy decode 256 tok | 1.5 s (1.5–1.5) | 1.5 s (1.5–1.5) |
| count-to-300 repeat gate | 3/3 clean | 3/3 clean |
| needle at 200K | 3/3 | 3/3 |

Verdict: no regression from 0.1.41 on this regime; we keep it as the daily engine. The
stage-pin class flagged in the 0.1.41 notes did not appear in our greedy decode gate
(pinning is off by default in our config).

## A/B 2 — KV: int8 vs k8v4 on 0.1.41 (3 matched pairs, I K I K I K)

| probe | int8 (n=3) | k8v4 (n=3) |
|---|---|---|
| expert cache at boot | 8164/8164/8165 slots (15.48 GiB) | 8310/8311/8311 slots (15.76 GiB) |
| pinned-RAM K/V tier | 12.38 GiB | 9.56 GiB |
| cold prefill 200K (engine ms) | 57,980/57,911/58,057 | 59,904/59,837/59,855 |
| xhigh decode 256 tok | 115.9/117.3/115.0 tok/s | 115.7/118.3/115.0 tok/s |
| greedy decode 256 tok | 201.8/200.5/199.8 tok/s | 202.4/203.0/202.6 tok/s |
| count-to-300 repeat gate | 3/3 clean | 3/3 clean |
| needle at 200K | 3/3 | 3/3 |

Reading: the #1264 fix works — k8v4 streamed state is correct on sm_120 at 200K context
(needles found, zero repetition, three for three). The economics, however, do not pay on
this box: the 2.8 GiB pinned-RAM saving buys +146 expert slots, but at ~90% cache hit
those slots add nothing measurable to decode (tie within 0.1 s), while the streaming-KV
path costs a consistent ~3% on 200K prefill (≈1.9 s per cold prompt). We stay on int8.

## Raw data

Six version A/B JSONs (one per leg) attached. For the KV A/B, the client-side JSONs
share one timestamp and the last leg of each condition overwrote the earlier ones —
the per-leg KV numbers above are taken from the engine's own serve log (boot lines +
per-request timings), which records all six legs and is the stronger source anyway.
Blocked-order runs from earlier today that motivated the protocol change are not
included as comparison data.

## Standing offer

This machine remains a reference point for the IQ3_S-at-1M agent regime; happy to re-run
the exact legs (same protocol, interleaved) on any candidate build — KV, cache, or
prefill changes.
