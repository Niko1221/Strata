# Tuning A/B on NVIDIA RTX PRO 4000 Blackwell (2026-10-16)

What four switches are worth on this machine, measured with two instruments: an interleaved A/B
driver that restarts the engine per arm (`ab.py`, `ab2.py`, `ab3.py`, `ab4.py` in this folder),
and the project's own `tools/strata_bench.py` run before and after the change
(`2026-10-15-community-rtx-pro-4000-...` against `2026-10-16-community-rtx-pro-4000-...` beside
this folder).

Hardware, build and model are the ones of the 2026-10-16 community report: RTX PRO 4000 Blackwell
(24 GiB), Core Ultra 9 285K (24 cores, 24 threads, no AVX-512), 125.3 GiB RAM, Linux 6.17.0-1033-oem,
driver 595.99.02, engine 0.1.44.1 (project commit 296b59d2), unsloth UD-IQ4_XS, 131072 context,
`--kv int8 --kv-resident 32768`, one slot. Expert cache 7057 slots (15.91 GiB), 325 MiB of VRAM free
with everything loaded, decode expert-cache hit rate 88.5 %.

## The question

The install started with `STRATA_ROUTE_TAIL_SKIP=0` and no `STRATA_ADAPT_OVERLAP`. Both are upstream
defaults that someone turned off here: `STRATA_ROUTE_TAIL_SKIP=0` disables a skip that is **on by
default on CUDA serve** since 0.1.42 (`src/program/generate.cpp:8618`), and `STRATA_ADAPT_OVERLAP`
became opt-in on every backend in 0.1.44 (#2010) although it was the Linux CUDA default in 0.1.43.
The rest of this folder measures those two, then three more candidates, then two switches for
several clients at once.

## Method

Every arm starts a freshly loaded engine (about 12–15 s), warms up with one short answer, then sends
the three chat prompts of `tools/ab_engine.py` (256 new tokens, greedy, thinking off) plus one
generated document. Arms alternate their order from round to round; the tables give the median over
the rounds. The config's `env` overrides the process environment for the engine
(`serve/server.py:2797`), so every arm gets its own config copy under `arms/`. The driver binds each
test server to 127.0.0.1 and starts it without an API key.

Caveats: one machine, two rounds per arm (round 3: two, except the first arm which ran once), greedy,
speed only — no correctness and no KL check. `STRATA_ROUTE_TAIL_SKIP` changes the output slightly by
design: teacher-forced KL 0.003–0.006, top-1 agreement 97.7–98.2 % (docs/DETAILS.md).

## 1. `STRATA_ROUTE_TAIL_SKIP` x `STRATA_ADAPT_OVERLAP` (2x2), `ab.py`

Decode tokens/s pooled over the four requests, median of 2 rounds (both rounds in brackets):

| Arm | `ROUTE_TAIL_SKIP` | `ADAPT_OVERLAP` | Decode tok/s | vs today |
| --- | --- | --- | ---: | ---: |
| A_cur | 0 | off | 77.9 (77.8, 78.0) | x1.000 |
| B_tail | default (7) | off | 89.1 (88.8, 89.3) | **x1.144** |
| C_ovl | 0 | 1 | 79.1 (78.7, 79.3) | x1.015 |
| D_both | default (7) | 1 | **91.3 (91.3, 91.3)** | **x1.172** |

Both arms that were expected to engage said so at startup: `STRATA_ROUTE_TAIL_SKIP=7 is ON (default):
a missed expert that every token of a verify window routes at rank 7 or lower is skipped (+10..20%
decode...)` and `adaptive overlap (...): the adaptive tier's copies land while the next window runs
(same tokens, a layer waits for its own copies)`. The tail skip alone is +14.4 % here, in the same
band as the +12.9..15.6 % the upstream tables measured on an RTX 3060, an RTX A4000 and a Tesla P100
at 75–90 % hit rate (this machine: 88.5 %). The adaptive overlap adds +2.5 % on top of it and never
changes the tokens; it was worth far less alone (+1.5 %) because most of its waiting is already
covered once the missed experts are skipped.

Prompt reading is untouched: the ~8K-token document read at 2534, 2535, 2534 and 2561 tok/s.

## 2. Three further candidates, `ab2.py` (all on top of D_both)

| Arm | Setting | Decode tok/s | x | Expert cache | VRAM free | Decode hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | — | 91.3 | x1.000 | 7057 slots | 325 MiB | 88.5 % |
| aux | `--aux-cpus auto` | 91.3 | x1.000 | 7057 slots | 325 MiB | 88.5 % |
| res | `--vram-reserve-mib 896` | 89.7 | x0.982 | 6973 slots | 521 MiB | 86.6 % |
| kv | `--kv-grow` | 91.4 | x1.001 | 7057 slots | 325 MiB | 88.5 % |

`--aux-cpus auto` did engage (`strata aux cpus: threads that are neither pool workers nor the host go
to CPUs 16,17,...,23 (on; 22 threads placed so far)`) and still bought nothing: with 15 pool workers
on CPUs 0,2–15 and the host on CPU 1, the helper threads were not the bottleneck here. The upstream
+10 % was measured on a 2x RTX 3080 layer split with `--batch-mtp` and two slots, a different shape.
`--vram-reserve-mib` costs about 400 slots for a safety margin the engine asks for at 325 MiB free
(see 4.): 325 MiB is above the cliff, so it stays off. `--kv-grow` does not apply at all here — the
engine says so itself: `--kv-grow is off (one GPU, a profile, the whole K/V in VRAM, every expert in
RAM, no --batch, ...)`.

## 3. Several requests at once: `--slots`, `ab3.py`

Two requests sent at the same moment, medians of 2 rounds (slots 1 = the requests queue):

| Arm | Slots | Concurrent | Wall (s) | Aggregate tok/s | Per request tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| s1c1 | 1 | 1 | 3.21 | 79.7 | 90.6 |
| s1c2 | 1 | 2 | 6.14 | **82.9** | 93.6 |
| s2c2 | 2 | 2 | 9.11 | 56.2 | 31.2 |
| s2c1 | 2 | 1 | 4.52 | 56.6 | **63.5** |

A second slot costs 643 of the 7057 expert slots and takes the machine over the VRAM cliff: a single
request drops from 90.6 to 63.5 tok/s (x0.70) and two requests together decode 32 % slower than the
same two queued one after the other. The slots buy waiting time, not speed, on this card — the same
finding as the 12 GB RTX 5070 in docs/BATCHING.md, and the reason setup's own rule (a slot is offered
only where the cache still holds most experts) leaves this machine at one at a time.

## 4. `STRATA_HC_Q8` and the prefill chunk, `ab4.py`

| Arm | Setting | Slots | Chunk | Decode tok/s | 32K prompt tok/s | Prompt-path slots borrowed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | — | 7057 | 8192 | 90.4 | 2716 | 2013 (4.55 GiB) |
| hc0 | `STRATA_HC_Q8=0` | 6890 | 8192 | 86.0 | 2747 | — |
| pf16 | `--prefill auto:16384` | 7057 | 16384 | 90.5 | 2832 | 3344 (7.56 GiB) |
| pf32 | `--prefill auto:32768` | 7057 | 32768 | 91.2 | **2877 (+5.9 %)** | 6015 (13.57 GiB) |

`STRATA_HC_Q8=1` is worth keeping: turning it off frees 0.63 GiB (167 slots) and loses 4.9 % decode.
The larger prefill chunk raises the prompt speed (the agent client on this machine sends 43K–77K
token prompts) at no cost to decode, but the prompt path borrows the expert cache for the buffer:
85 % of it at 32768 against 47 % at 16384. `auto:16384` was applied as the balance; `auto:32768` is
+1.6 points more if the prompt path is what matters.

## Applied, and how to undo it

`strata-unsloth-ud-iq4_xs.json`, backed up as `strata-unsloth-ud-iq4_xs.json.bak-20261016-2236`:

```diff
  "env": {
    "STRATA_HC_Q8": "1",
-   "STRATA_ROUTE_TAIL_SKIP": "0"
+   "STRATA_ADAPT_OVERLAP": "1"
  },
-   "--prefill", "auto",
+   "--prefill", "auto:16384",
```

The change is worth **x1.172 decode** (77.9 -> 91.3 tok/s) in the interleaved A/B. The project's own
suite, run against the 2026-10-15 report from the day before (a cross-day comparison, so less
controlled than the A/B above):

| Metric | 2026-10-15 | 2026-10-16 | Change |
| --- | ---: | ---: | ---: |
| chat decode tok/s | 81.10 | 89.10 | +9.9 % |
| chat first token (s) | 0.40 | 0.36 | -9.8 % |
| 4K prompt tok/s | 2402.8 | 2465.8 | +2.6 % |
| 16K prompt tok/s | 2815.1 | 2874.4 | +2.1 % |
| agent decode tok/s | 88.50 | 96.20 | +8.7 % |

## Raw evidence in this folder

- `ab*.py` — the four drivers; `arms/` holds each arm's config copy and the engine log of every run
  (startup lines, slot count, hit rate, per-request timings), `rows.jsonl`, `rows_r2.jsonl` and
  `rows_r4.jsonl` the per-request raw data, `run*.log` the drivers' stdout.
- `bench-config.json` — the config the official suite ran with; it differs from the install config
  only in `"host": "127.0.0.1"`, so the test server is never reachable from the network.

Not measured here: a learned expert profile (`--expert-profile-save`), `--calibrate`, huge pages, and
any correctness or KL check of the two switches.
