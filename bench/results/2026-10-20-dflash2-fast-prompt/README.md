# DFlash2: fresh long-prompt latency and measured profile selection

Measured 2026-10-20 on RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, 125.7 GiB RAM, Linux, CUDA toolkit 13.4 / driver 610.57.04, SM89. This is a configuration study, not another accepted kernel change.

## Result

The selected profile reduces native client request time from **11.42 to 7.76 seconds (-32.04%)** compared with the first batched-prefill profile. Its decode throughput is **51.95 versus 50.35 tok/s for same-day MTP (+3.18%)**. Every measured selected-DFlash2 decode sample exceeds every MTP sample. Whole request latency is effectively tied with MTP: 7.762 versus 7.755 seconds, with overlapping observed ranges. No overall latency win over MTP is claimed.

| Drafter/profile | Prefill | Expert slots | Block / floor | Prompt read median, s | Decode median [range], tok/s | Client request median [range], s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| DFlash2 Q8_0, first batched profile | 256 | 2980 | 7 / 0.70 | 9.510 | 50.24 [49.91–50.37] | 11.421 [11.416–11.434] |
| DFlash2 Q8_0, larger prefill | 512 | 2900 | 7 / 0.70 | 5.909 | 50.21 [50.19–50.29] | 7.821 [7.815–7.824] |
| DFlash2 Q8_0, larger again | 1024 | 2700 | 7 / 0.70 | 6.354 | 44.99 [44.90–45.00] | 8.488 [8.484–8.493] |
| **DFlash2 Q8_0, selected** | **512** | **2900** | **6 / 0.85** | **5.909** | **51.95 [51.82–51.98]** | **7.762 [7.756–7.794]** |
| MTP, same prefill/cache budget | 512 | 2900 | up to 3 / 0.50 | 5.844 | 50.35 [50.26–50.39] | 7.755 [7.749–7.762] |

All rows use `--short-read 64`, one warmup and three measured repeats of the same 2425-token prompt, 96 output tokens and greedy seed 1234. Prompt/conversation snapshots, CPU prompt sharing, PCIe sharing, adaptive swaps and suffix/lookup drafting are off. Context/resident capacity is 8192; the allocation reserve is 480 MiB. Both speculative arms enable `STRATA_MMVQ_IL_Q8_HEAD=1` and `STRATA_DFLASH_HEAD_IL=1`; MTP receives the same shared-target-head optimization. Forward budgets differ and MTP width was not independently tuned. The selected profile was chosen on this fixture; ranges describe three observations, not confidence intervals.

Client request time includes prompt reading and generation through the native protocol. It excludes process/model startup, HTTP and tokenization. The MTP control was measured before the final DFlash2 threshold confirmation, on the same day with the same executable. It was not rerun or retuned after that confirmation.

The target remains the NVFP4 pack, BF16 embedding, FP8 PLE, full Q8_0 target output head and INT8 KV from the [previous report](../2026-10-20-dflash2-long-tuned/README.md). Accepted engine source is `aab69ad03f8c225402ff0972e5b5a4bc0147fec1`; the source/report head before this study is `27b65cc8b4c42c83da65bbed605c8ab6bba8171b`. Every main comparison uses the **exact previous executable**, SHA-256 `7eb8b4c9e9ddbe1a305ce34f6f5bcc5cf8edb7d1f8fb225614ef7f171c92bdb1`. Discarded prototypes were restored before these measurements. No engine default or kernel is changed by this report.

## What was slowing the prompt read

The preceding controlled decode benchmark used `--short-read 4096`. That sends this whole 2425-token prompt through decoder windows, bypassing batched prefill. Raising `--prefill` alone therefore did not help. The earlier selected profile took 52.636 seconds per client request, including 50.956 seconds of prompt reading, while decoding at 56.77 tok/s.

Restoring `--short-read 64` enables the normal batched prompt path. **64 is already Strata's default.** The roughly 85% request-time reduction against that 52.636-second controlled configuration is not a new speedup of default Strata. The further measured change from batched prefill 256 to the selected 512 profile is the 32.04% reduction reported above. It also changes expert slots, block length and probability floor; this is a profile comparison, not an isolated prefill-kernel speedup.

Increasing prefill to 1024 and reducing the expert budget further was slower. The 512 profile needs fewer rejected verifier rows after retuning: selected DFlash2 accepts 63/65 proposals in 33 windows; MTP accepts 64/73 in 32. More accepted proposals alone do not imply better throughput.

The initial `short-read 64` probe kept 3188 expert slots and reached 11.259 seconds, but logged only 37 MiB free before its next request. It is not the recommended profile. The selected 2900-slot run reports 963 MiB free at startup and 828 MiB at its last logged verifier capture. These are logged snapshots, not continuous VRAM monitoring.

## Output controls and limits

All warmed/measured selected DFlash2 outputs match its target-only control exactly, **96/96 token IDs per request**, with no near-tie exemption. Every same-day MTP output matches that same control. The prefill-256 and prefill-512 controls also agree. Controls use temperature 1, top-k 1, top-p 1, min-p 0 and seed 1234, offer zero proposals and preserve the actual target expert placement of their profile.

The selected target-only control differs from the older 3188-slot, decoder-read control at 12 of the 96 positions (the sequence changes from position 83). Changing prompt arithmetic and CPU/GPU placement can change greedy rounding. No output identity across those configurations is claimed. Within each reported final configuration, comparison against its own target-only control passes.

Not run: a held-out prompt suite, other model families/quantizations in the selected profile, other hardware, Windows, HIP/SYCL/Metal, concurrent serving, a full 128K prompt, or an independent MTP-width sweep. The PR remains a draft. Keep the earlier 3188-slot / decoder-read profile when reproducing its 56.77-vs-55.31 decode table; do not mix those cells with this profile's latency table.

## Exploration retained

`exploratory/` retains complete commands, logs and token arrays for the Q8 floor sweep, Q5_0 sweep, PCIe sharing, 12 CPU tasks, adaptive swaps, prefill 512 with decoder reading still forced, the low-headroom 3188-slot probe and cached prefill-256/512 grids. Cached grids only select candidates; the table above is remeasured with caches off.

PCIe shares 0.4/0.5/0.75 change token output relative to that sweep's zero-share reference and do not improve speed. Adaptive swapping also fails its target-only output comparison and is not selected. The early Q5/PCIe/12-task grids used a binary containing an inactive joint-gate prototype; [its rejected patch](exploratory/rejected-joint.patch) is retained for provenance. They are not pooled into any main table. Other discarded kernel prototypes and their raw local logs remain outside the checkout; none supplies a claimed speedup here.

## Reproduce

Use the existing local model artifacts referenced by the preceding report; no downloads are needed. Slot counts are specific to this machine and are asserted at startup.

```sh
python3 bench/results/2026-10-20-dflash2-fast-prompt/run.py dflash2-il \
  --draft-quant Q8_0 --block 6 --floor .85 --expert-cache 2900 --reserve 480 \
  --prefill 512 --short-read 64 --tasks long --only-selected --repeats 3 \
  --output /tmp/strata-fast-prompt
python3 bench/results/2026-10-20-dflash2-fast-prompt/run.py mtp \
  --floor .5 --q8-head-il --expert-cache 2900 --reserve 480 \
  --prefill 512 --short-read 64 --tasks long --only-selected --repeats 3 \
  --output /tmp/strata-fast-prompt
python3 bench/results/2026-10-20-dflash2-fast-prompt/analyze.py /tmp/strata-fast-prompt
```

[profile.json](profile.json) records the selected settings. [summary.json](summary.json) retains individual measured values and summaries; [checks.json](checks.json) records control comparisons and executable identity. Main raw evidence is in [tuned512/](tuned512/), [prefill512/](prefill512/), [prefill256/](prefill256/) and [prefill1024/](prefill1024/). Model artifact provenance is unchanged from the preceding report.


## Setup calibration

The explicit `./setup.sh --calibrate` path for DFlash2 now searches fresh-request
profiles after the existing decode sweep. It compares prefill 256/512/1024 with
`short-read 64`, then jointly measures loaded-artifact block widths and probability
floors on the two best allocation profiles plus the original baseline. It records
actual expert residency and separates prompt/decode time. The objective gives
short and long workloads equal weight and normalizes generation to the same output
budget, so early EOS cannot win by producing a shorter answer.

Startup headroom must be at least 1024 MiB. Calibration can raise the existing
reserve or reduce an explicit expert cap; it never lowers a requested reserve or
increases an explicit cap. These are startup checks, not a guarantee against every
later allocation. The manually measured 2900-slot profile above has slightly less
headroom; calibration intentionally remeasures its own placement instead of copying
that machine-specific slot count.

Search measurements cannot authorize a new profile by themselves. The winner must
beat the original baseline in all three alternating pairs of independent restarts,
with more than 3% median improvement on additional prompt cases. Rejected
confirmation restores the original calibration; cancellation leaves the config
unchanged. If no complete safe baseline can be measured, the extra phase is marked
skipped and the earlier decode result remains available. Cached setup results
include the full measurements and bind to hardware/model/context/artifact and the
known head/prefill environment flags. A calibrated block change no longer hides the
result during reinstall. This is opt-in calibration, not a change to MTP or
uncalibrated engine defaults.

See [calibration-validation/](calibration-validation/) for exact commands, source
hashes, test output and bounded GPU smoke evidence. A full production calibration
with 128 output tokens, 2048/4096-token documents and three confirmation pairs was
not run; the main performance table comes from the independent native benchmark
above, not the reduced calibration smoke.
