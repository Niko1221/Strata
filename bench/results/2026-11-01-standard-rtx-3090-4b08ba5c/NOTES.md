# NOTES.md - host notes for the RTX 3090 standard-benchmark submission

Extra provenance file for `bench/results/<date>-standard-rtx-3090-<id>/`. The measured tables
live in the runner's own `README.md` / `results.json`; this file carries only what the runner
cannot see. Everything here was read on the machine on 2026-11-01 unless marked `not measured`.

## What ran

- `strata bench run --profile launch-iq3_s-128k --suite standard-v1 --optional-concurrency`
  with the revision flags in `REPRODUCE.md`. Two conforming runs, each on a fresh engine:
  the stock profile (`2026-11-01-standard-rtx-3090-4b08ba5c`) and the tuned keeper config
  (`2026-11-01-standard-rtx-3090-c11a6683`, three extra engine options in the run config's
  `user_args`: `--kv k8v4 --spec-min-p 0.70 --pcie-frac 0.17`). Two more arms (t2b) were
  run as measurements but are not submitted; see the tuning section for the full table.
  Exact commands, exit codes and folders: see `REPRODUCE.md`.
- The runtime was stopped before the run, so the runner started a **fresh engine** and stopped
  it again: `server_requests_before_run` must read 0. The engine was not reused from the
  agent session.
- Bench exit code: `0` (complete). Verdict re-confirmed 2026-11-01 with
  `strata bench report` against the results folder: "21 of 21 required requests finished,
  0 failed, 0 skipped". Engine load: 61 s from `engine started` (engine.log 03:20:18Z) to
  `run.started_at` 03:21:19Z, model hashing included; protocol time after load 2 min 3 s.

## Component versions (the release bundle, not a source build)

| Component | Version |
| --- | --- |
| terminal (`strata`, `~/.local/bin/strata`) | 0.2.4.2, build 0.1.49.2, `COMMIT dfa3f40...` as printed by `strata --version` |
| control (server wrapper `serve.server`, `strata_control.runtime`) | 0.1.51 |
| engine (inference) | 0.1.50-linux-x86_64-cuda-cuda13 |
| vision engine | 0.1.48-linux-x86_64-cuda-cuda13 (`strata-vision`, holds the mmproj on the GPU) |

`strata --version` prints `0.2.4.2 (build 0.1.49.2)`; the runner records only the terminal
commit in `software.strata_commit`. The engine/control versions above come from the component
directory names and the live process table, because `results.json` records them separately.

## Model identity

- Quantization repo: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at
  `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (`main` HEAD, 2026-10-25T10:02Z). Checked against
  the local files: the LFS sha256 oids of both IQ3_S shards and of
  `mmproj-Qwen3.8-Flash-Next-BF16.gguf` at that revision equal the local `.done` markers
  (`96a4995d...`, `6f3f798a...`, `51b73250...`).
- Base weights: `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540`
  (`main` HEAD at lookup time). **Not verified as the exact build input of the GSQ-RCO quant** -
  the quant README names Qwen3.8-Flash-Next as the source model but records no base commit.
- The quantization is labelled `IQ3_S` but is ISTA's GSQ-RCO non-uniform per-tensor scheme
  (`<quant_type>GQ</quant_type>`, `qtype 10`, `gqa_res "Q4_K"`, `gqa_act "Q5_K"`,
  `gqa_fck "Q6_K"` in the GGUF metadata), so it is not llama.cpp's uniform IQ3_S.
  `results.json` `model.files[].quantization` reads `IQ3_S` from the file names only.
- The quantization repo carries no tokenizer files; the tokenizer in use is the pack's copy
  derived from the GGUF (see `provenance/pack-provenance.sha256`).
- The server's effective chat template is byte-identical to
  `Strata-data/packs/iq3_s/tokenizer/chat_template.jinja`, sha256
  `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce` (9993 bytes). The runner
  writes the live template to `provenance/chat_template.jinja`; the two hashes should match.

## Custom packs and expert profile (hashes in `provenance/pack-provenance.sha256`)

- `Strata-data/packs/iq3_s` - prepared pack, `tools/iq_pack.py`, native experts, built from the
  IQ3_S shard; `dense.bin` 1538035200 bytes, `index.txt` records `generated_from` the shard path.
- `terminal/0.1.49.2/data/expert-profile.bin` - 196608-byte expert profile shipped with the terminal.
- `Strata-data/mtp/rt` - MTP draft pack (`experts.bin` 707788800, `dense.bin` 116099072,
  `draft_vocab.bin` 425196, `dense.txt`).
- No draft vocabulary override, no custom build options: release binaries.

## Engine options actually in force

Read from the live process (the runner records the same set in `config.json`):

```
components/engine/0.1.50-linux-x86_64-cuda-cuda13/strata --serve
  --pack Strata-data/packs/iq3_s
  --native Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  --ple-gguf Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  --expert-profile terminal/0.1.49.2/data/expert-profile.bin
  --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --mtp Strata-data/mtp/rt --max-context 131072 --kv int8 --vision
  --vram-reserve-mib 700 --batch 3
```
plus server `serve.server --parallel 3 --port 8080 --host 127.0.0.1` (control 0.1.51), which
gives `total_slots = 3` in `/props`. `--pcie-frac` is not set (engine default; the live
`/metrics` pcie fraction read 0.55). `--kv-resident` is not set, although the maintainer's
3090 guidance suggested 32768 - a difference worth noting rather than changing mid-run.

## Hardware (world-readable host reads, not protocol results)

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, driver 615.78.08 |
| Power | current limit 390 W (default 390 W, max 480 W), max SM clock 2130 MHz |
| PCIe at idle | 5.0 GT/s (Gen3) x16, max 16.0 GT/s (Gen4) x16 - idle link state, not measured during model load |
| CPU | AMD Ryzen 7 7800X3D, 8 cores, `ID_MODEL_FROM_TABLE=AMD Ryzen 7 7800X3D 8-Core Processor` |
| RAM | 65333200 kB total |
| Swap | zram 62.3 GiB (compressed RAM, **not** NVMe) |
| Storage | 2 x NVMe Samsung SSD 990 PRO 4TB |
| OS | CachyOS (`ID=cachyos`), kernel 7.3.1-1-cachyos, x86-64 |
| Kernel env | desktop host with KDE Plasma; kwin/plasmashell hold about 1.1 GiB VRAM |

## Memory behaviour during the run (host observation)

- Engine `VmRSS` before the run: 52869772 kB (50.4 GiB) with `VmSwap` 1771720 kB; vision engine
  241736 kB RSS. The 83.6 GB of shards is tiered: hot experts in the 24 GB of VRAM, the rest in
  DDR5, with zram as the compressed backing tier.
- Peak engine RSS during the run: **not captured** (no `/proc` sampler was wired into the
  runner; `engine.log` carries no `VmHWM`). Closest machine facts from the same run:
  the memory guard at start saw 4.9 GiB current against the 62.3 GiB limit, and the
  conversation cache parked up to 1.695 GB (`bytes=1695033584`, 4 parked conversations,
  32 evictions). Wire an RSS sampler into the tuning rounds' wrapper if the PR needs the number.
- Profile-locked load vs interactive load: same path and same numbers this round
  (61 s cold start through the managed runtime; the interactive start of the serving
  runtime at 03:24-03:25Z answered `/health` within the same order of magnitude).

## Background during the run

- KDE Plasma session (the operator's seat) stayed up; about 1.1 GiB VRAM held by the compositor.
- The agent harness and its supervisor were stopped with the runtime; the `strata monitor`
  process (it polls `/metrics`, no inference) was closed for the run.
- No browser, game, or other GPU client. Nothing was paged to NVMe; swap is zram.
- `nvidia-smi` utilization 1 %, 23901 MiB used immediately before the run.

## Rehearsal evidence (probe, 2026-11-01, against the live 0.1.50 engine)

The exact fixture `performance-short-1` payload was sent over loopback with streaming before
the run, to prove the settings contract is accepted by this engine: HTTP 200, 77 prompt tokens
(= the fixture's frozen 77), `cached_tokens` 0, MTP `draft_n` 290 / `draft_accepted` 195,
decode 61.8 tok/s on that single short request, first delta 0.676 s. This is **not** a protocol
result; it only shows the profile's flags reach a working engine.

## Known defects that touch this run

- Issue **#4574** (reproduced on control 0.1.51 / terminal 0.1.49.2): the managed runtime host
  can survive its server child, which then blocks a later `runtime start` with "profile is
  locked". The stop phase of this run is exposed to the same failure; the recovery used was
  `ldb-bench-3090.sh --restart` (kill the leftover, `strata runtime start --profile ... --wait 300`).
  It did **not** fire: the post-run `runtime start` returned `ok: true, state: ready` in
  the same log, and `BENCH-RUN-STATE.txt` shows `serving=yes` at 03:25:08Z with no
  `RESTARTED` line (the `--restart` path was never taken).
- Issue **#2936** decode regression on sm_86 in 0.1.44: not applicable (engine 0.1.50).

## Limitations of this submission

- One run, one engine load: no repeat-stop or loop-recovery statistics, no cold-model
  measurement (the expert cache warms and evolves through the suite, and the shared chat
  template prefix is reused), and the protocol's own caps mean a truncated request fails its
  sanity check rather than reporting a longer answer.
- The optional 4-client stage exceeds this server's 3 slots, so the fourth request queues;
  read the aggregate concurrency number as 3-in-flight, not 4.
- PCIe link speed during model load was not captured (root-only sysfs read at load time);
  the idle reading above is the only link fact in this file.
- The base-weights revision is unverified as the quant's build input, as stated above.

## Expected range for this host, from the maintainer's own 3090 guidance

Decode 75-85 tok/s, prefill 1500-2500 tok/s, expert-cache hit 95-99 % (IQ3_S, 128K, MTP on,
64 GB DDR5 CL26); the insiderllm 3090 benchmark of 2026-10-10 reported 78 tok/s.
Deltas after the run (medians from `README.md`, 2026-11-01):

| Metric | Guidance | Measured | Delta |
| --- | --- | --- | --- |
| decode, short | 75-85 tok/s | 72.8 (68.2-73.1) | 2.9 % under the lower bound |
| prefill, 4k | 1500-2500 tok/s | 2089.7 | inside |
| prefill, 16k | 1500-2500 tok/s | 2782.2 | 11.3 % above the upper bound |
| expert-cache hit | 95-99 % | 69.5-84.1 % during the performance phase (91.6-95.8 % on the short needle rounds; batch stage 68.2 % VRAM / 15.4 % PCIe / 16.4 % CPU) | well under |

The decode gap tracks the expert-cache hit gap: the 3902-slot cache could not cover this
model's routing at `--batch 3` (the slots cost 5.66 GiB VRAM the cache would otherwise
hold), and `engine.log` warns "only 319 MiB of VRAM free at the start of this request
(below 512 MiB)" (#2065) at the tightest moment. The sanity-phase decodes reach
62.9-93.5 tok/s, consistent with the guidance once the cache is warm and short.

## Tuning A/B, 2026-11-01 (four arms, one fresh engine and one protocol run each)

The stock run sits 2.9 % under the maintainer decode guidance, so the profile was tuned
one change at a time (Tier 2 arms build on the Tier 1 base alone). `--batch 3` and
`--parallel 3` never changed. Short decode medians (min-max), 16K decode, prefill and the
best concurrency round (aggregate tok/s over four simultaneous requests, 0 failed, all
rounds sanity 12/12):

| Arm | extra options | short dec | 16k dec | prefill 4k / 16k | conc. best |
| --- | --- | --- | --- | --- | --- |
| baseline | (none) | 72.8 (68.2-73.1) | 73.9 | 2089.7 / 2782.2 | 70.1 |
| **t1 (keeper)** | `--kv k8v4 --spec-min-p 0.70 --pcie-frac 0.17` | **76.4 (74.9-79.3)** | 70.8 | 2065.3 / 2590.1 | **81.6** |
| t2a | t1 + `--kv-grow` | 79.3 (68.4-91.2) | 78.6 (72.7-90.1) | 2060.7 / 2586.5 | 75.3 |
| t2b | t1 + `--pool-workers 9` | 39.1 (36.0-39.6) | 38.3 | 2014.5 / 2551.1 | 38.3 |
| t2c | t1 + `--batch-mtp` | 71.2 (67.1-80.1) | 71.8 | 2057.9 / 2557.8 | 74.5 |

**t1 is the keeper**: decode +3.6 tok/s over baseline (crosses the 75-85 guidance band),
concurrency +16.5 % best-round (median rounds 78.1 vs 68.8), at the documented k8v4 prefill
cost: 16K prefill 2782 to 2590 (-6.9 %, a little worse than the 2-5 % the 0.1.25 DETAILS.md
notes quote for k8v4). The mechanism is visible in `engine.log`: k8v4 frees VRAM and
`expert cache auto` grows from 3902 slots (baseline) to 4644.

**t2b anti-recommendation (measured, Issue 3887)**: `--pool-workers 9` on the 8-core
7800X3D halves everything (39.1 vs 72.8 short decode, -46 %). The engine's own warning is
in that arm's `engine.log`: "strata cpu pool: warning: 9 CPU expert workers are configured,
but this process can run only 8 CPU threads. The extra workers spin and take time from the
others. Recommended: --pool-workers 7 (or leave it out: the engine picks that by itself)."
The default 7 was already right on this box; never set workers above the physical core count.
The 9-worker figure circulates in community configs tuned for other CPUs.

**t2c anti-recommendation (measured, Issue 2065)**: `--batch-mtp` needs VRAM per slot for
draft state; on a box that already warns "only 319 MiB of VRAM free", it comes out of the
expert cache (4377 slots vs 4644 under t1) - the very thing holding decode up. Net result
is below the untuned baseline (71.2 vs 72.8 short). Do not enable on VRAM-constrained
single-GPU boxes.

**t2a is the honest alternative for single-stream work**: best 16K decode (78.6) and best
single short (79.3), but the spread is wide (one short repetition at 68.4, below baseline)
and concurrency is lower than t1. For this seat (agent harness, mostly one stream but a
shared endpoint) t1 was chosen for robustness; a single-stream long-context seat should
try `--kv-grow` on top.

**Do not chase `--kv-resident 32768`** from the maintainer 3090 guidance on this box: it
reserves 32K KV in VRAM up front at the 319 MiB-free wall, shrinking the expert cache;
t1's k8v4 already bought the same headroom the cheap way (3902 to 4644 slots).

The keeper config is applied as `user_args` in the run config (`--kv k8v4 --spec-min-p
0.70 --pcie-frac 0.17`), so it survives future setup rewrites; the t1 folder's
`config.json` records the merged effective options.
