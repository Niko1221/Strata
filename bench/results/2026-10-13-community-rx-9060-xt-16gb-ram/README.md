# Community benchmark: Coder IQ1_M on an RX 9060 XT with 16 GB of RAM and a SATA SSD

Measured on 2026-10-13 by [mandoof1](https://github.com/mandoof1). This report shows the Coder IQ1_M running
at a 131,072-token context on a Linux PC with **15.5 GiB of usable RAM**, half of the 32 GB the README lists for
the Coder. The experts are read from a SATA SSD (`--mmap-experts`). It lists the settings and three small local
engine patches that made it stable on this machine, a same-day comparison against stock 0.1.43.1 with setup's
own configuration, and earlier tuning measurements.

**Everyday speed vs the bench number.** In everyday agent use (coding prompts, long replies with thinking on,
a warm cache), decode runs at **12-17 tok/s with peaks of 25 tok/s**. That comes from my own measurements on
2026-10-09 to 2026-10-11 ([earlier tuning measurements](#earlier-tuning-measurements-not-repeated-today)). The
8.9 tok/s in today's results table is `strata_bench.py`'s short chat: three trivia questions with 100-token
replies, thinking off, minutes after a reboot. Short replies are dominated by the slow first tokens after each
prompt, and those prompts match the agent-learned expert profile less well (GPU hit rate 38-58% today, against
79% on agent prompts).

Main limitations: one machine and one model. The bench suite runs with thinking off, but day-to-day use is an
agent client with thinking on. The patches were built and tested only with HIP (gfx1200), not CUDA or SYCL.
No correctness check was run.

## What made it work

In order of impact on this machine:

1. **`GPU_PINNED_MIN_XFER_SIZE=1048576`** in the config's `env`. This fixed every stall and engine stop seen
   here: "verify: timed out", "no progress for 60 s", and 2-4 tok/s decode after a long prompt. This is the
   ROCclr USERPTR mechanism already described in TROUBLESHOOTING.md and #1705. Upstream sets it only with
   `STRATA_ARENA_MMAP=1`; the `--mmap-experts` path still needs it set by hand.
2. **The engine in a swap-off cgroup scope.** CachyOS has zram swap on by default. The launcher runs
   `systemd-run --user --scope -q -p MemorySwapMax=0 python serve/server.py ...`, so the server and the
   engine are never pushed to zram while the page cache holds experts.
3. **Static expert residency: `--adapt-every 100000`.** The default adaptive tier (every 4 windows) was the
   slowest configuration tried, and it ended the engine once.
4. **Longer stall limits:** `STRATA_WATCHDOG_S=180` and `STRATA_VERIFY_TIMEOUT_S=300`. The second needs local
   patch 1 below: the verify window's 20 s limit is hard-coded upstream. A SATA SSD under a page-fault storm
   can be slow for longer than 20 s without being stuck.
5. **An expert profile learned from agent prompts** instead of the shipped `expert-profile-coder.bin`
   (local patch 3, `STRATA_PROFILE_ROUTING_ONLY=1`). In real agent use the GPU's expert hit rate was 26% with
   the shipped profile. A profile learned from 4 agent prompts and tested on 4 different ones raised the hit
   rate from 56% to 79% and decode from 12.9 to 17.7 tok/s. The profile is included:
   [`expert-profile-agent.bin`](expert-profile-agent.bin).
6. **Smaller settings:** `--spec 5`, `--pcie-frac 0.6`, `STRATA_IO_PREFETCH=1`, `STRATA_ROUTE_TAIL_SKIP=7`
   (on HIP it is off unless set), and `--kv int8`.
7. **Storage and desktop:** `chattr +C` on the model folder (btrfs: no compression or copy-on-write for the
   experts file), and the browser closed. Brave alone used about 1.7 GB of RAM and page cache.

The full configuration, with paths replaced by `<STRATA>` and `<DATA>`:
[`strata-coder-iq1_m-128k.json`](strata-coder-iq1_m-128k.json).

## Hardware and software

| Component | Setup |
| --- | --- |
| GPU | AMD Radeon RX 9060 XT, 16 GB (15.92 GiB reported), gfx1200; also drives the desktop (Wayland) |
| PCIe | GPU reports 32 GT/s x16; the upstream port runs at 8 GT/s x16 (PCIe 3.0, the CPU's limit) |
| CPU | Intel Core i5-9400F, 6 cores / 6 threads, AVX2, no AVX-512 |
| RAM | 16 GB installed, 15.55 GiB usable; zram swap 15.5 GiB (CachyOS default) |
| Storage | Crucial MX500 500 GB, SATA (about 410 MB/s sequential read); btrfs, `chattr +C` on the model folder |
| OS | CachyOS, Linux 7.2.0-1-cachyos |
| ROCm | 7.2.4 (system install, `/opt/rocm`) |
| Strata | source build of `a0116b2` (0.1.43.1), HIP, `GPU_TARGETS=gfx1200`, Release, plus [`local-patches.diff`](local-patches.diff) |

Background: a desktop session only. The browser was closed. No power limits were set, and clocks and
temperatures were not recorded.

## Model and configuration

- Model: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF),
  revision `5348543e` (setup's pin), IQ1_M:
  `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` and `-00002-of-00002.gguf`. The pack was made by
  setup (`native_experts_sha256` `9ace1b92...74dd3b`), with the Q2_0 MTP runtime that setup made.
- Expert profile (tuned run): `expert-profile-agent.bin`, sha256
  `89141a8317c3efb8767fca6bdcfdd87c8a5ce069d3bb19a3541a18ac9f97bbb7`. The shipped
  `data/expert-profile-coder.bin` has sha256 `a514661124d5305effe9c1bcecff7dc069af398b2b19ff9d11628a2cad0a8915`.
- No vision encoder. One slot. Calibration and the experimental speed projection were not used.

| | Stock (baseline) | Tuned |
| --- | --- | --- |
| Engine | 0.1.43.1 at `a0116b2`, no patches | the same source plus `local-patches.diff` |
| Context | 16,384 (setup's choice for this PC) | 131,072 |
| Expert profile | `data/expert-profile-coder.bin` | `expert-profile-agent.bin` |
| Args changed | none (setup's own) | `--spec 5` (setup: 4), `--adapt-every 100000`, `--pcie-frac 0.6`, `--kv-grow`, `--expert-profile-save` (left out for the bench) |
| Env added | none (setup's hipBLASLt tuning file only) | `GPU_PINNED_MIN_XFER_SIZE=1048576`, `STRATA_IO_PREFETCH=1`, `STRATA_VERIFY_TIMEOUT_S=300`, `STRATA_WATCHDOG_S=180`, `STRATA_PROFILE_ROUTING_ONLY=1`, `STRATA_ROUTE_TAIL_SKIP=7` |
| Process | server started by the bench | the bench and its server in a `MemorySwapMax=0` scope |

Common args from setup: `--expert-cache auto --prefill auto --spec-min-p 0.5 --kv int8 --mmap-experts`, plus
the MTP runtime. The engine logs that `--kv-grow` is off in this configuration. The tuned engine filled
4,177 expert slots (7.93 GiB of VRAM), with 582 MiB of VRAM left free.

## Method

`python tools/strata_bench.py --config <config> [--long] --ready-timeout 1200`, run once per configuration,
tuned first. The tuned run used `--long`; the 16K context of the stock configuration has no room for the 16K
and 32K prompts. Everything else is the default suite: 3 runs per test, thinking off, temperature 0, model
loading not timed, one discarded warm-up. Page cache: the PC was rebooted about 5 minutes before the first run.
The baseline started right after the tuned run, so some experts may still have been in the page cache. The
baseline was stopped during its agent session (see below).

Memory: [`tuned-128k/mem-samples.txt`](tuned-128k/mem-samples.txt) and [`stock-16k/mem-samples.txt`](stock-16k/mem-samples.txt) sample
`MemAvailable`, zram swap in use, and VRAM and GTT in use (whole adapter) every 2 s, plus the SATA drive's
read rate from `/proc/diskstats`. A guard
would have ended the engine below 400 MiB available. It never triggered.

## Results (2026-10-13, same day, same machine)

Medians with the lowest and highest run in brackets. Speeds are the engine's own timings.

| Test | Stock 0.1.43.1, setup's config (16K) | Tuned (128K) |
| --- | --- | --- |
| short chat, decode tok/s | 2.6 (0.7-7.7), 7 of 9 requests returned | 8.9 (7.8-12.7), 9 of 9 |
| short chat, first token s | 49.8 (5.1-119.9) | 4.9 (4.0-6.0) |
| 4K prompt (4,019 tokens), prompt tok/s | 43.6 (13.8-67.0) | 178.4 (178.1-191.9) |
| 4K prompt, first token s | 92.3 (91.9-291.2) | 22.6 (21.2-22.7) |
| 16K prompt (16,058 tokens), prompt tok/s | does not fit the 16K context | 322.7 (313.0-327.2) |
| 32K prompt (32,049 tokens), prompt tok/s | does not fit | 373.2 (364.6-374.9) |
| agent session, 6 turns x 3 | the first turn failed (no output for 312 s); stopped | 18 of 18 turns; cache hit 68.7%; first token 18.8 s (15.6-21.8); decode 5.7 tok/s (3.5-8.0) |
| engine stops | 3: two during chat, one in the agent session | none |

The tuned run's generated report and per-request data are in [`tuned-128k/`](tuned-128k/) (`README.md`,
`results.json`, `engine-log-excerpt.txt`). Its GPU name reads "AMD Radeon" because the tool did not read the
model name on this machine.

**The stock run reproduces the problem this report is about.** The engine log
([`stock-16k/engine-log-excerpt.txt`](stock-16k/engine-log-excerpt.txt)) shows `verify: timed out at layer 19`
(#267) and `no step finished for 60 s, but the file tier is still being read`. In one request the OS took
3,365,575 major faults to read 13.9 GB from the SSD, so the experts came in about 4 KB at a time. While the
agent turn waited, the drive read about 35 MB/s, not its ~410 MB/s. The server ended the engine three times. I stopped the
stock run about 15 minutes into the agent session, after its first turn failed. `strata_bench.py` writes
`results.json` only at the end, so the stock numbers come from its console output
([`stock-16k/console.txt`](stock-16k/console.txt)).

The tuned run had no failures. Its agent-session decode (5.7 tok/s, 64-token replies, thinking off) is lower
than what I see in real use with longer replies (12-17 tok/s, earlier measurements below). The first tokens
after a prompt wait for the post-prompt slot refill from the SSD, and short replies weigh that more.

Memory: the lowest `MemAvailable` was 11.1 GiB (tuned) and 11.9 GiB (stock); the mapped experts count as page
cache. VRAM in use peaked at 15.8 GiB in both runs, and zram held at most about 1 GiB of other processes.

## Earlier tuning measurements (not repeated today)

These were measured from 2026-10-09 to 2026-10-11 with my own scripts, not `strata_bench.py`, on engines
0.1.41 and 0.1.42 with the local patches. They show where each setting came from. They are not comparable
one-to-one with the table above.

| Change | Context | Before | After |
| --- | ---: | --- | --- |
| `GPU_PINNED_MIN_XFER_SIZE=1048576` (0.1.41) | 16K | decode after a 7K prompt 2-4 tok/s, verify timeouts | 8-9 tok/s; short prompts about 16 tok/s; prefill 250-344 tok/s |
| agent-learned profile vs the shipped coder profile (0.1.41, learned on 4 agent prompts, tested on 4 others) | 64K | GPU hit 56%, decode 12.9 tok/s | GPU hit 79%, decode 17.7 tok/s (+37%), peak 25 |
| `STRATA_ROUTE_TAIL_SKIP=7` (0.1.42, agent test prompts) | 64K | 11.9 tok/s | 14.5 tok/s |
| `STRATA_ROUTE_TAIL_SKIP=7` (0.1.42, agent test prompts) | 128K | 10.6 tok/s | 12.1 tok/s (+14%) |
| `STRATA_ROUTE_TAIL_SKIP=7`, replay of a real 54K-token agent conversation (0.1.42) | 128K | 5.9 tok/s | 6.5 tok/s |

At 128K the engine has 4,177 expert slots, against 4,682 at 64K. On 0.1.42 the 64K configuration was 20-35%
faster when the extra context was not needed.

**Measured slower or unstable here, not kept:** the adaptive tier at its default (`--adapt-every 4`: slowest,
and one engine stop), `--resident-budget-gib` (slower, deadlocked once), `--vram-reserve-mib 300` (pin
failures with the desktop on the same card), `--spec 6`, `--prefill 2048`, `--pcie-frac 0.4` and `0.8` (no
change against 0.6), `STRATA_HIP_WMMA=1` (prefill is SSD-bound: 217 tok/s either way), `STRATA_ROUTE_PRIOR`
0.5 / 1.0 (no change in hit rate), `HSA_ENABLE_SDMA=0`, GFXOFF off, the performance CPU/GPU power modes, and the
kernel's `watermark_scale_factor` / `min_free_kbytes`. A 64K context instead of 16K costs about 7-10%. KV stays
int8; q4_0 and k8v4 were not kept.

## Local engine patches

[`local-patches.diff`](local-patches.diff) applies to `a0116b2` with `git apply`. Each patch is opt-in or
only adds an OS read-ahead hint. None of them is needed to start the engine; they make this machine stable
and faster:

1. `src/core/verify.cpp`: `STRATA_VERIFY_TIMEOUT_S` sets the verify window's per-layer wait limit (unset: the
   current 20 s).
2. `src/prefill/prefill.cpp` and `src/program/generate.cpp`: `madvise(MADV_WILLNEED)` on the mapped experts
   before the prefill stager copies them and ahead of the post-prompt slot refill (64 blobs ahead). After a
   long prompt the page cache has thrashed, and without the hint the OS faults the experts in 4 KB at a time.
   This one is always on in the patch; I did not measure it on its own.
3. `src/program/generate.cpp`: `STRATA_PROFILE_ROUTING_ONLY=1` makes `--expert-profile-save` rank experts by
   routing counts alone. With static residency (a large `--adapt-every`) no round folds the counts into
   `heat`, so the saved profile only repeated the experts already resident. The patch adds `drive.d.usage`
   and does not put resident experts first.

I am not proposing them as engine changes in this pull request. If any of them is wanted, I can open it on
its own with the measurements it needs.

## How to repeat on a similar PC

1. Run setup normally (it picks the Coder and `--mmap-experts` on a 16 GB PC). On btrfs, `chattr +C` the
   model folder before the download.
2. Optional: apply `local-patches.diff` and rebuild (`cmake --build build-hip --target strata`, then copy the
   binary to `engine/`). Without patch 1, leave out `STRATA_VERIFY_TIMEOUT_S`. Without patch 3, leave out
   `STRATA_PROFILE_ROUTING_ONLY`.
3. Merge the `env` and the changed args of [`strata-coder-iq1_m-128k.json`](strata-coder-iq1_m-128k.json)
   into your config. `GPU_PINNED_MIN_XFER_SIZE` is the one that matters most.
4. Start the server in a swap-off scope:
   `systemd-run --user --scope -q -p MemorySwapMax=0 .venv/bin/python serve/server.py --engine strata --config <config> --port 8080`.
5. Close the browser before long sessions.

## Not tested

CUDA and SYCL builds of the patches; Windows; an NVMe drive (the NVMe in this PC is a Windows drive without
room); thinking-on speed through this suite; needle or other correctness checks; more than one slot; a
second run order (baseline first).
