# Strata testing history by installed RAM (Dell Precision 5820, Xeon W-2125, RTX 3090 + RTX 3060)

Companion to this folder's `README.md` (era 4). The raw data for eras 1 to 3 (the `~/strata-src/...` and `~/ram-test-...` folders named below) is not attached to this pull request.

Draft summaries of the earlier test sessions, grouped by how much RAM was installed at the time. Everything below is taken from the saved reports and data of those sessions (folders named at the end of each section); I did not re-run anything for this file. Where a report left something unmeasured or unexplained, it says so.

## At a glance

| Era | Dates | Installed RAM (channels) | DIMM speed | Engine | Main models tested |
|---|---|---|---|---|---|
| 1 | 2026-10-06 | 40 GB: 5 x 8 GB (8/16/8/8 GB per channel) | 2400 MT/s | v0.1.40.1 | Coder IQ1_M, llama.cpp Qwen3.8-27B for comparison |
| 2 | 2026-10-09 to 10-10 | 72 GB: 5 x 8 GB + 1 x 32 GB (16/16/8/32) | 2400 MT/s | v0.1.41 | IQ2_XS, IQ3_S, IQ3_S with the 3060 in several roles |
| 3 | 2026-10-10 to 10-11 | 104 GiB, 8 slots filled (32/16/40/16 GiB) | 2133 MT/s | v0.1.41 | IQ3_S helper cache, UD-IQ4_XS, quality comparison |
| 4 | 2026-10-23 | 176 GB, 8 slots filled (48/40/48/40 GiB; 172 GiB visible) | 2133 MT/s | v0.1.41 and v0.2.3.1 | IQ3_S, UD-IQ4_XS (see this folder's `README.md`) |

Memory bandwidth by era (own STREAM/scan tools, Strata stopped):

| | 40 GB | 72 GB | 104 GiB | 176 GB |
|---|---|---|---|---|
| Fast 4-channel region (read) | about 34.5 GiB at about 51 GB/s | about 34.5 GiB at about 50-51 GB/s | about 66.5 GiB at about 45-50 GB/s | about 162 GiB at about 44-49 GB/s |
| Slower region(s) | about 8 GiB single channel at 18.7 GB/s | about 15.5 GiB at about 37 GB/s, then about 23 GiB single channel at 18.7 GB/s | about 32 GiB at about 31 GB/s (2-way), about 8 GiB single channel at about 16 GB/s | last about 13 GiB at about 31 GB/s (2-way) |
| STREAM Triad, 4 threads | 21.8 GB/s (28% of the 76.8 GB/s peak) | not run | 38.3 GB/s (56% of the 68.3 GB/s peak) | 38.1 GB/s (official STREAM, median of 5; 37.8 with 8 threads) |
| Own `ramtune bw 4 1` read | not run | 35.8-38.3 GB/s | 48.4-49.3 GB/s | 48.3-48.7 GB/s |

The DIMM speed fell from 2400 to 2133 MT/s in era 3 because two 16 GB Hynix sticks (HMA42GR7MFR4N-TF) are rated 2133 and hold every channel to that speed. Slot-to-channel pairing (1+5, 2+6, 3+7, 4+8) was inferred from the EDAC capacities, not read from the board.

## Era 1: 40 GB (2026-10-06), engine v0.1.40.1

**What was tested:** a Coder IQ1_M model on the RTX 3090 alone (128K then 256K context), then a two-GPU layer-split attempt, then a baseline of the hardware and a llama.cpp comparison. Folders: `~/strata-src/bench-results/20261006-134933`, `-145021`, `-152009`, and `-161803-ram` (RAM scan).

**Hardware baseline:** the 40 GB were five 8 GB DIMMs, one channel carrying two of them. STREAM Triad was 21.8 GB/s with 4 threads and barely rose with more threads, so memory, not cores, was the limit. The 3090 copies host-to-device at 8.14 GB/s (PCIe Gen3 x16) and the 3060 at 6.20 GB/s (Gen3 x8).

**Results (Coder IQ1_M, 3090 only, 256K context, mean of the cold runs):**

| Depth | Cold prefill tok/s | Code decode tok/s |
|---|---|---|
| 50K | 2,483 | 95.9 |
| 100K | 2,501 | 95.1 |
| 150K | 2,418 | 75.8 |
| 200K | 2,329 | 88.4 |
| 250K | 2,254 | 71.6 |

- Memory: about 28 GB in use at peak, minimum available RAM about 6.5-7.3 GB, swap 0.75 GB. The swap value was already there at the start of the phase (baseline swap 869 MB) and did not grow.
- **Two-GPU layer split was much slower:** at 128K it read prompts at 843 tok/s (50K) and 781 tok/s (100K) against about 2,495 on one GPU (-66% and -69%), decode fell 7-12%, and the prefix cache never hit (0 cached tokens). The second card only started with a VRAM reserve of 1000 MiB or more (default 700 and 500 failed to load). Every two-GPU config split the model as layers 0-46 on the 3090 and layer 47 on the 3060; other splits were not tried in that session. The 250K two-GPU run stopped at a safety gate before finishing.
- **llama.cpp comparison (Qwen3.8-27B UD-Q4_K_XL, a different dense model on both GPUs):** prompt reads 952 tok/s at 50K and 784 at 100K, decode 45 and 39 tok/s, with 98-99% prefix reuse. It says nothing about quality.

## Era 2: 72 GB (2026-10-09 to 10-10), engine v0.1.41

**What was tested:** the update from v0.1.40.1 to v0.1.41, then IQ2_XS, IQ3_S and a series of configurations that put the RTX 3060 to use. Folders: `~/strata-src/bench-results/20261009-215805-iq2xs`, `20261010-010814-iq3s`, `20261010-022058-iq3s-2gpu`.

**RAM scan:** the extra 32 GB stick did not widen the fast region (still about 34.5 GiB at 50 GB/s), but it added about 15.5 GiB at about 37 GB/s; the single-channel tail grew to about 23 GiB at 18.7 GB/s, so a normal allocation lands partly in the slow tail.

**Results on the 3090 alone, 256K context (cold prefill tok/s / code decode tok/s):**

| Depth | IQ2_XS | IQ3_S |
|---|---|---|
| 50K | 2,249 / 112.6 | 1,857 / 79.5 |
| 100K | 2,273 / 110.0 | 1,910 / 64.9 |
| 150K | 2,251 / 109.7 | 1,929 / 70.7 |
| 200K | 2,080 / 100.6 | 1,863 / 68.8 |
| 250K | 2,039 / 99.0 | 1,802 / 71.2 |

- IQ2_XS: about 38.8 GB in use, 28.5 GB minimum available, no swap. IQ3_S: about 52.7 GB in use, **14.6 GB minimum available and 628 MB of swap** (a little, with no visible slowdown). Both passed the 250K needle tests (3 of 3).
- IQ3_S setup took about 11 minutes including the download; 8,660 expert slots in VRAM.
- **Using the 3060 (IQ3_S):** the best setting was **C5b**: the 3090 runs the model and the 3060 holds an extra expert cache (`--expert-cache-device1 auto --remote-expert-opt`). It left prefill unchanged (1,859 / 1,907 tok/s at 50K / 100K) and raised decode **+37% at 50K (108.7 vs 79.5 tok/s) and +58% at 100K (102.8 vs 64.9)**, with swap at 129 MB. A default layer split cut prefill by 77-85% and broke prefix reuse; a manual split that gives the 3060 many layers fixed prefill (+7-12%) but did not improve decode and made follow-up TTFT 2-3x worse. Smaller contexts gave the 3060 no extra experts. This is the finding that layer splitting was detrimental.
- Run-to-run decode spread was 10-30%, so only the large gains count.

## Era 3: 104 GiB (2026-10-10 to 10-11), engine v0.1.41

**What was tested:** the RAM tests after the first upgrade, an overnight evaluation of IQ3_S against UD-IQ4_XS (speed and quality), and a RAM-stability run. Folders: `~/ram-test-20261010-1943`, `~/strata-src/upgrade-eval-20261010-2124`.

**RAM:** all 8 slots filled, 104 GiB, 2133 MT/s. The 4-way region grew to about 66.5 GiB at about 45-50 GB/s; about 32 GiB ran at about 31 GB/s and the last about 8 GiB at about 16 GB/s. Whole-RAM average of the scan 39.8 GB/s by capacity (36.2 for pages spread evenly). **memtester on 95.5 GiB, 1 loop: all tests ok**, exit code 0, in 23,361 s; EDAC corrected and uncorrected errors 0 on every DIMM and the kernel log was unchanged.

**Strata results (200K shown; the report's 50K and 100K figures are in `speedtable.md`):**

| Config | Cold prefill tok/s at 50K / 100K / 200K | Code decode tok/s at 50K / 100K / 200K | Minimum available RAM | Swap |
|---|---|---|---|---|
| IQ3_S helper cache (C5b), 72 GB (from era 2) | 1,859 / 1,907 / 1,861 | 108.7 / 102.8 / 94.0 | about 13.5 GB | peak 129 MB |
| IQ3_S helper cache (C5b), 104 GiB | 1,963 / 1,923 / 1,839 | 99.3 / 106.1 / 91.8 | about 45.7 GB | flat at about 270 MB (no growth) |
| UD-IQ4_XS, 3090 only, 104 GiB | 1,098 / 1,098 / 1,086 | 33.1 / 37.6 / 30.5 | about 51 GB | flat at about 265 MB |
| UD-IQ4_XS helper cache, 104 GiB | 1,625 / 1,596 / 1,547 | 66.9 / 84.9 / 67.6 | about 36 GB | flat at about 260 MB |

- **The RAM upgrade did not change speed for IQ3_S** (prefill and decode matched the old-RAM numbers within the 10-30% decode spread). **The gain was headroom:** minimum available RAM rose from about 13.5 GB to about 45.7 GB and swap stopped growing.
- UD-IQ4_XS loaded and worked, but its helper-cache prefill was 17% slower than IQ3_S at 50K and 100K, which failed the replacement criterion (10%), so IQ3_S stayed as the server. Without the helper cache UD-IQ4_XS was too slow to use (decode 28-41 tok/s).
- The first UD-IQ4_XS helper-cache load failed (`--resident-cpu-experts does not support remote expert caches`): `--resident-budget-gib` conflicts with remote expert caches, so the working config ("variant A") drops it. That cost about 30 minutes because the wait loop only watched for health.
- Quality (HumanEval+, MBPP+ and others, not speed): the only difference above the noise floor was MBPP+ (+2.9 points for IQ4_XS, p = 0.013). The helper cache did not measurably change quality on a 164-task spot-check (92.7% vs 94.5%, p = 0.375). Needle recall was 15/15 for every quant.
- The old `llm.service` (llama.cpp) was disabled in this session so Strata serves port 8080; the unit backup is in the upgrade folder.

## Era 4: 176 GB (2026-10-23)

**RAM:** channels now 48/40/48/40 GiB, 2133 MT/s, 172 GiB visible. The 4-way region is about 162 GiB at about 44-49 GB/s; only the last about 13 GiB (2-way) runs at about 31 GB/s. STREAM Triad 38.1 GB/s (4 threads), 37.8 (8 threads); own tool read 48.3-48.7 GB/s and non-temporal triad 50.4-51.6 GB/s, the same as the 104 GiB layout. EDAC corrected and uncorrected errors: 0 on all 8 DIMMs. **memtester has not been run on this configuration yet** (planned for later). Folder: `~/ram-test-20261023-1937`.

**Strata (v0.1.41, 3 cold runs, means; see this folder's `README.md` for v0.2.3.1 and the full tables):**

| Config | Cold prefill tok/s at 150K / 200K / 250K | Code decode tok/s at 150K / 200K / 250K | Minimum available RAM | Swap |
|---|---|---|---|---|
| IQ3_S helper cache | 1,818 / 1,748 / 1,754 | 98.9 / 100.3 / 96.3 | about 91 GiB | none (0 pages out) |
| UD-IQ4_XS, 3090 only | 1,125 / 1,101 / 1,095 | 38.5 / 39.5 / 37.2 | about 121 GiB | none |
| UD-IQ4_XS helper cache | 1,591 / 1,556 / 1,532 | 80.6 / 80.9 / 75.1 | about 106 GiB | none |

- Against era 3 (104 GiB), IQ3_S at 200K is 1,748 prefill and 100.3 decode (104 GiB: 1,839 and 91.8) and UD-IQ4_XS with the helper cache is 1,556 and 80.9 (1,547 and 67.6): speed is in the same range, so more RAM again added headroom, not speed. The 200K IQ3_S prefill is about 5% lower than era 3's; the harness differed a little (3 cold runs against 2, a changed oversized-prompt check), and I did not look for a cause.
- Swap: none at all this time, against 129-628 MB in era 2 and a flat 260-270 MB (carried over, no growth) in era 3.

## Cross-era view: what each RAM step bought

1. **40 to 72 GB:** more capacity let IQ3_S (52.7 GB in use) run, but with only 14.6 GB left and 628 MB of swap on the 3090-only config. The fast region did not grow.
2. **72 to 104 GiB:** the fast 4-channel region doubled (about 34.5 to about 66.5 GiB), and available RAM during IQ3_S rose from about 13.5 to about 45.7 GB, with swap no longer growing. Speed did not change. The clock fell from 2400 to 2133 MT/s, and the peak bandwidth with it (76.8 to 68.3 GB/s), but the measured read bandwidth in the fast region stayed at about 48-50 GB/s.
3. **104 to 176 GB:** the fast region grew again (about 66.5 to about 162 GiB), available RAM during IQ3_S reached about 91 GiB, and swap stayed at zero. IQ3_S speed was in the same range (decode 96-100 tok/s; the 200K prefill about 5% lower than era 3).
4. **What decided speed was never the amount of RAM:** it was the second GPU's role (helper cache, +37-58% decode in IQ3_S, doubling decode for UD-IQ4_XS), the engine version (v0.2.3.1: decode +5-50%) and the quant.

## Limits of these summaries

- The sessions used slightly different harness settings (2 or 3 cold runs, depths 50K/100K/200K or 150K/200K/250K, different engine versions in era 1), so only the 200K IQ3_S figures overlap exactly; treat cross-era differences under about 10% (decode 10-30%) as noise.
- The era 1 and 2 reports were written by earlier sessions; I read their summaries and tables but did not re-check the raw CSVs.
- Era 1 used a Coder IQ1_M model, a different model from the IQ3_S and UD-IQ4_XS ones, so its numbers do not compare with them.
- Slot-to-channel mapping is inferred. memtester for the 176 GB configuration is pending.
