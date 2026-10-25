# Community benchmark: RTX 3090 + RTX 3060 on a Xeon W-2125, Strata v0.1.41 vs v0.2.3.1

Measured on 2026-10-23 by Flix2032. Speed only (prompt processing and decode); no quality evaluation. The same machine, model files, packs and workload were run on two engine versions. Main limitation: one request at a time, one machine, three cold runs per cell. Run with AI assistance (Claude Code); every number below comes from the attached per-run files, and the harness and commands were run on the machine.

## Hardware and software

- GPUs: RTX 3090 24 GB (PCIe 3.0 x16) and RTX 3060 12 GB (PCIe 3.0 x8), both sm_86. Power limits at the default (350 W and 170 W; not changed). CPU: Intel Xeon W-2125 (4 cores / 8 threads, Skylake-W, PCIe 3.0 only), one socket.
- RAM: 176 GB installed in 8 DIMMs of mixed size (2x16, 4x32, 2x8 GB), running at 2133 MT/s (the slowest DIMM sets it); 172 GiB visible; 64 GiB swap file (almost unused). Storage: SATA SSDs (Samsung 860 and another 512 GB SSD); which device holds the model files was not recorded.
- OS: Ubuntu 24.04.4, kernel 6.8.0-147, NVIDIA driver 615.71.09, CUDA 13.4, g++ 13.3, CMake 3.28.
- Strata, both built from source (CUDA, sm_86), one engine at a time, each with its own build folder and port:
  - **v0.1.41**: commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (2026-10-08), engine 0.1.41, `STRATA_Q6K_EXPERTS=OFF`, `STRATA_MMQ_KQUANTS=OFF`.
  - **v0.2.3.1**: commit `7ddc17a38fe7f31637e895ecfba58c2cd43d65ec` (2026-10-23), engine 0.1.48.1, built with the flags `setup.py` pins (`STRATA_Q6K_EXPERTS=ON`, `STRATA_MMQ_KQUANTS=ON`). **The two builds therefore do not have identical compile options**; I did not test v0.2.3.1 with them off, so I cannot say how much of any difference comes from them.
  - Python: 3.14 venv (v0.1.41), 3.12.3 venv (v0.2.3.1).
- Background workloads: none; the normal Strata service was stopped during each run. No power limits were changed.

## Model and configuration

- Model: Qwen3.8-Flash-Next (125B MoE) as two GGUF quantizations. Repository and revision of the downloads: not recorded. File hashes: UD-IQ4_XS sha256 values were recorded (`models.md`); IQ3_S hashes were not computed.
- IQ3_S: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf` (54.8 + 28.8 GB), pack `packs/iq3_s`, `native_experts.txt` v3.
- UD-IQ4_XS (Unsloth): `Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf` (about 94 GB), pack `packs/unsloth-ud-iq4_xs`, `native_experts.txt` v4.
- **The same model files, packs, MTP draft layer (`mtp/rt`) and expert profile (`expert-profile.bin`) were used for both engine versions.** v0.2.3.1 loaded the existing packs without a repack. No vision encoder.
- Common settings: context 262144, `--kv int8 --kv-resident 32768`, `--spec 4 --spec-min-p 0.5`, MTP draft layer, `--expert-cache auto --prefill auto`, reasoning and sampling as the server defaults, no calibration, no experimental speed projection.
- Three configurations per version (full redacted configs in each `data/<run>/config_redacted.json`):
  - **iq3**: IQ3_S, both GPUs, `CUDA_VISIBLE_DEVICES=0,1`, `--expert-cache-device1 auto --remote-expert-opt` (the 3060 as a helper expert cache, no layer split), `--conversation-cache-mib 24576 --conversation-cache-slots 12`.
  - **q4-3090**: UD-IQ4_XS, RTX 3090 only, `--resident-budget-gib 55`.
  - **q4-2card**: UD-IQ4_XS, both GPUs, same helper-cache flags as iq3, with `--resident-budget-gib` removed (it fails to load together with remote expert caches), no conversation cache.

Launch: `serve/server.py --engine strata --config <config>.json --port <8080|8081> --open`, with `api_key` set (removed from the attached configs).

## Method

- Harness: `tools/bench.py`, an existing script from a previous benchmark round on this machine, reused rather than rewritten. One one-line change (original kept as `bench.py.orig`): the regular expression that recognises the "prompt too large" error also matches `leaves no room to answer in the context`, because the engine's wording differs from what the script expected, so the 250K shrink-and-redo did not trigger without it. `tools/bench_one.sh` is the per-run driver (start server, wait for health, run `bench.py`, collect memory, log and swap data, stop).
- Prompts: a deterministic generated corpus (seed 7), sized to a target prompt length. Depths: 1K (warm-up/short prompt), then 150K, 200K and 250K prompt tokens. If a prompt overshoots, the harness shrinks it and redoes the run; those redone runs are not counted (`status = redo-size` in the CSVs).
- Output cap: 256 tokens. Three cold runs per depth, each a fully processed prompt (0 reused tokens: every cold prompt starts with a fresh random session-id line, so no cached prefix can match; the harness also redoes a run, status `redo-cached` or `redo-fast`, if the server reports cached tokens or an implausibly fast prefill). Then one follow-up that appends about 3K tokens to the same conversation to measure reuse (`--ckpt-interval 16384`).
- Timing: prompt tok/s = prompt tokens / time to first token (client side, streaming API); decode tok/s = (generated tokens - 1) / time from the first token to the last. Model loading is not included. TTFT is client side.
- Memory: RAM, swap and VRAM were sampled during each depth (`samples_*.csv`, `summary_*.json`); the "available RAM" minimum is the useful value because pinned expert memory is not counted as used. `pswpout` swap counters before and after each run are in `swap_counters.txt`. Logs were searched for swap, OOM, "stopped unexpectedly", restarts, CUDA errors and tracebacks (`log_findings.txt`), and `dmesg` for OOM-killer entries (`dmesg_oom.txt`).

## Results

Medians and ranges of the cold runs (all runs are in `data/<run>/runs_*.csv`):

| Configuration | Context | Actual prompt tokens (median) | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| current-iq3 | 1K | 983 | 0 | 256 | 3 | 511 (504-514) | 107.5 (83.1-110.2) | 1.9 (1.9-2.0) |
| current-iq3 | 150K | 152321 | 0 | 256 | 3 | 1849 (1751-1854) | 99.0 (94.2-103.5) | 82.4 (82.1-87.0) |
| current-iq3 | 200K | 199631 | 0 | 256 | 3 | 1745 (1740-1758) | 101.6 (97.4-101.8) | 114.4 (113.6-114.7) |
| current-iq3 | 250K | 250977 | 0 | 256 | 3 | 1757 (1744-1760) | 96.7 (88.4-103.9) | 142.8 (142.6-143.9) |
| current-q4-3090 | 1K | 971 | 0 | 256 | 3 | 318 (310-322) | 40.9 (30.6-41.5) | 3.1 (3.1-3.1) |
| current-q4-3090 | 150K | 152326 | 0 | 256 | 3 | 1125 (1124-1125) | 38.2 (36.6-40.6) | 135.4 (135.3-135.5) |
| current-q4-3090 | 200K | 199634 | 0 | 256 | 3 | 1101 (1101-1102) | 40.9 (36.3-41.2) | 181.3 (181.2-181.3) |
| current-q4-3090 | 250K | 250953 | 0 | 256 | 3 | 1094 (1094-1096) | 39.2 (30.9-41.3) | 229.4 (228.9-229.5) |
| current-q4-2card | 1K | 988 | 0 | 256 | 3 | 441 (440-442) | 83.6 (61.8-85.0) | 2.2 (2.2-2.3) |
| current-q4-2card | 150K | 152311 | 0 | 256 | 3 | 1595 (1577-1601) | 84.5 (70.0-87.3) | 95.5 (95.2-96.6) |
| current-q4-2card | 200K | 199626 | 0 | 256 | 3 | 1556 (1554-1557) | 81.2 (76.6-84.9) | 128.3 (128.2-128.5) |
| current-q4-2card | 250K | 250968 | 0 | 256 | 3 | 1531 (1530-1536) | 74.7 (67.4-83.3) | 163.9 (163.4-164.0) |
| v0.2.3.1-iq3 | 1K | 968 | 0 | 256 | 3 | 528 (519-530) | 106.6 (94.2-129.1) | 1.8 (1.8-1.9) |
| v0.2.3.1-iq3 | 150K | 152334 | 0 | 256 | 3 | 1761 (1741-1762) | 104.4 (97.9-105.4) | 86.5 (86.5-87.5) |
| v0.2.3.1-iq3 | 200K | 199629 | 0 | 256 | 3 | 1750 (1740-1756) | 107.7 (104.0-110.5) | 114.1 (113.7-114.7) |
| v0.2.3.1-iq3 | 250K | 250983 | 0 | 256 | 3 | 1762 (1730-1785) | 109.5 (104.1-109.6) | 142.4 (140.6-145.1) |
| v0.2.3.1-q4-3090 | 1K | 986 | 0 | 256 | 3 | 355 (351-357) | 50.2 (42.9-66.0) | 2.8 (2.8-2.8) |
| v0.2.3.1-q4-3090 | 150K | 152336 | 0 | 256 | 3 | 1123 (1123-1129) | 53.5 (50.5-61.4) | 135.7 (135.0-135.7) |
| v0.2.3.1-q4-3090 | 200K | 199634 | 0 | 256 | 3 | 1108 (1105-1109) | 59.3 (54.7-61.3) | 180.2 (180.1-180.7) |
| v0.2.3.1-q4-3090 | 250K | 250977 | 0 | 256 | 3 | 1103 (1102-1108) | 58.6 (43.6-62.1) | 227.6 (226.5-227.7) |
| v0.2.3.1-q4-2card | 1K | 988 | 0 | 256 | 3 | 427 (425-436) | 92.7 (65.0-97.1) | 2.3 (2.2-2.3) |
| v0.2.3.1-q4-2card | 150K | 152321 | 0 | 256 | 3 | 1586 (1582-1596) | 89.0 (78.4-92.7) | 96.1 (95.4-96.3) |
| v0.2.3.1-q4-2card | 200K | 199625 | 0 | 256 | 3 | 1555 (1550-1555) | 89.0 (82.6-95.1) | 128.4 (128.4-128.8) |
| v0.2.3.1-q4-2card | 250K | 250985 | 0 | 256 | 3 | 1536 (1535-1548) | 80.8 (66.9-90.4) | 163.4 (162.1-163.5) |

Follow-up that reuses the previous prompt (append to the user message, reused tokens from the response `usage`):

| Configuration | Context | Prompt tokens | Reused tokens (from usage) | TTFT s | Effective prompt tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| current-iq3 | 150K | 155140 | 147456 | 6.9 | 22344 | 78.8 |
| current-iq3 | 200K | 202226 | 196608 | 5.6 | 35864 | 78.0 |
| current-iq3 | 250K | 254807 | 245760 | 9.2 | 27786 | 75.8 |
| current-q4-3090 | 150K | 155138 | 147456 | 8.7 | 17917 | 25.3 |
| current-q4-3090 | 200K | 202232 | 196608 | 8.2 | 24704 | 22.8 |
| current-q4-3090 | 250K | 254775 | 245760 | 13.4 | 19060 | 25.9 |
| current-q4-2card | 150K | 155131 | 147456 | 6.1 | 25612 | 52.1 |
| current-q4-2card | 200K | 202223 | 196608 | 5.9 | 34225 | 47.5 |
| current-q4-2card | 250K | 254792 | 245760 | 9.8 | 26085 | 54.4 |
| v0.2.3.1-iq3 | 150K | 155156 | 147456 | 6.9 | 22519 | 82.4 |
| v0.2.3.1-iq3 | 200K | 202230 | 196608 | 5.7 | 35622 | 78.4 |
| v0.2.3.1-iq3 | 250K | 254813 | 245760 | 9.1 | 28000 | 78.4 |
| v0.2.3.1-q4-3090 | 150K | 155152 | 147456 | 9.0 | 17181 | 36.5 |
| v0.2.3.1-q4-3090 | 200K | 202230 | 196608 | 8.5 | 23754 | 35.4 |
| v0.2.3.1-q4-3090 | 250K | 254805 | 245760 | 13.2 | 19277 | 38.0 |
| v0.2.3.1-q4-2card | 150K | 155136 | 147456 | 6.9 | 22629 | 60.2 |
| v0.2.3.1-q4-2card | 200K | 202222 | 196608 | 6.5 | 30953 | 54.0 |
| v0.2.3.1-q4-2card | 250K | 254808 | 245760 | 10.5 | 24294 | 58.9 |

Version comparison (v0.2.3.1 against v0.1.41, medians of 3 cold runs):

| Quant / mode | Context | PP v0.1.41 | PP v0.2.3.1 | PP change | Decode v0.1.41 | Decode v0.2.3.1 | Decode change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ3_S, 2 cards (helper cache) | 150K | 1849 | 1761 | -4.8% | 99.0 | 104.4 | +5.5% |
| IQ3_S, 2 cards (helper cache) | 200K | 1745 | 1750 | +0.3% | 101.6 | 107.7 | +6.0% |
| IQ3_S, 2 cards (helper cache) | 250K | 1757 | 1762 | +0.3% | 96.7 | 109.5 | +13.2% |
| UD-IQ4_XS, 3090 only | 150K | 1125 | 1123 | -0.2% | 38.2 | 53.5 | +40.1% |
| UD-IQ4_XS, 3090 only | 200K | 1101 | 1108 | +0.6% | 40.9 | 59.3 | +45.0% |
| UD-IQ4_XS, 3090 only | 250K | 1094 | 1103 | +0.8% | 39.2 | 58.6 | +49.5% |
| UD-IQ4_XS, 2 cards (helper cache) | 150K | 1595 | 1586 | -0.6% | 84.5 | 89.0 | +5.3% |
| UD-IQ4_XS, 2 cards (helper cache) | 200K | 1556 | 1555 | -0.1% | 81.2 | 89.0 | +9.6% |
| UD-IQ4_XS, 2 cards (helper cache) | 250K | 1531 | 1536 | +0.3% | 74.7 | 80.8 | +8.2% |

Observations:
- Prompt processing is unchanged within about 1% everywhere (the 150K IQ3_S cell, -4.8%, is not repeated at 200K or 250K and may be noise).
- Decode is faster on v0.2.3.1 at every depth from 150K to 250K in every configuration (at the 1K warm-up depth the IQ3_S decode is 106.6 against 107.5 tok/s, within noise): about +5% to +13% for IQ3_S, +40% to +50% for UD-IQ4_XS on the 3090 alone, and +5% to +10% for UD-IQ4_XS on two cards. Decode varies 10-30% from run to run (the first cold run is often the lowest), so differences below about 10% are within noise; the 3090-only UD-IQ4_XS gain is well outside it. I did not isolate the cause (compile flags versus runtime defaults).
- Two cards with a helper cache against the 3090 alone for UD-IQ4_XS (v0.1.41, 250K): prompt 1531 vs 1094 tok/s, decode 74.7 vs 39.2 tok/s; v0.2.3.1: 1536 vs 1103 and 80.8 vs 58.6. Layer splitting was not used (an earlier test on this machine found it slower).
- Memory: no swap was used in any run (`pswpout` unchanged), no OOM-killer entries, no restarts. Minimum available RAM was 91 GiB for IQ3_S, 106-121 GiB for UD-IQ4_XS on v0.1.41, and 98-110 GiB for UD-IQ4_XS on v0.2.3.1; v0.2.3.1 used about 7 GiB more RAM for UD-IQ4_XS (resident experts 43.5 against 39.7 GiB). VRAM peaks: 3090 about 23.9 GB in every run, 3060 about 11.7 GB when used. These are peaks sampled during inference.

## Failures, skipped cases and first attempts (kept in `data/attempt1-*`)

- First pass of the v0.1.41 runs failed at 250K because the harness did not recognise the oversized-prompt message (see Method); both v0.1.41 runs were repeated after the one-line change. The partial results are kept and not used.
- The first v0.2.3.1 UD-IQ4_XS start failed (`port 8081 is already in use`) because the previous server had not released the port yet; it was repeated with a gap. No benchmark data came from it.
- Needle/recall and other quality checks were not run.
- Not tested: concurrent requests, other quantizations, other contexts below 150K (except the 1K warm-up), v0.2.3.1 with the old compile flags, a second machine.

## Notes for maintainers (small observations; not part of the measurements)

- On this CPU (Skylake-W: `avx512f/dq/cd/bw/vl`, no VNNI or VBMI) both versions log `this CPU has no AVX-512: the expert kernels run on AVX-2`. From reading `include/strata/kernels/cpu/expert.hpp`, the AVX-512 path requires VNNI and VBMI as well, so the message appears to mean "no usable AVX-512 path"; a CPU with AVX-512 F/BW/VL may find the wording confusing. This is a reading of the code plus the observed log line, not a measured effect on speed.

## Files

- `data/<run>/`: `runs_*.csv` (every run), `summary_*.json`, `samples_*.csv` (resource samples), `bench.log`, `version.txt`, `config_redacted.json`, `free_*.txt`, `vram_*.txt`, `swap_counters.txt`, `log_findings.txt`, `dmesg_oom.txt`, `health.txt`, `server.log` (client-side server output; the engine log is not attached).
- `tools/`: `bench.py`, `bench.py.orig`, `bench_one.sh`.
- `RAM-history.md`: summaries of earlier test sessions on this machine, grouped by installed RAM (40 GB to 176 GB). Background only; the measurements in this report are in the tables above.
