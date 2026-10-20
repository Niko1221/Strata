# DFlash2 long-case confidence-gate diagnosis

Metadata correction (2026-10-20): the shared target head is Q8_0 (GGML type 8, 2560 × 248320, 675430400 bytes). The older engine hard-coded Q5_K in its log. This corrects the format label; the original timings and raw logs are unchanged. See the [follow-up](../2026-10-20-dflash2-long-tuned/README.md).

Measured on 2026-10-20: RTX 4070 Ti SUPER 16 GiB, Ryzen 9 5900X, Linux, CUDA 13.4 / SM89. Same local NVFP4 target, BF16 embedding, FP8 PLE, Q8_0 shared output head and official Q4_0 drafter as the [preceding experiment](../2026-10-19-dflash2-pr-performance/README.md).

## Cause and fix

The earlier -21.1% long-case comparison used DFlash2 block 3 with probability floor 0, against MTP's setup floor 0.5. It did not compare both setup defaults. The zero-floor DFlash2 run verified uncertain suffixes that the target then rejected.

There was also a manual-CLI inconsistency: selecting a fixed DFlash2 block without specifying `--spec-min-p` left the floor at zero. Setup already explicitly selects 0.5. The fix gives manual DFlash2 launches the same 0.5 default, independently of forward width. Explicit floors, including zero, still win. Existing setup configurations explicitly using 0.5 do not change.

The confirmation uses the patched binary, checks its native INFO reports floor 0.50 with a fixed block and no CLI floor, and interleaves explicit zero requests with requests inheriting the default. This isolates the gate in one process at identical actual expert placement. It measures a default-setting fix, not a faster tensor kernel.

| Q4_0 configuration | Code median tok/s [range] | 2425-token prompt median tok/s [range] |
| --- | ---: | ---: |
| Block 3, floor 0 | 46.18 [46.11–46.74] | 42.57 [41.59–43.03] |
| Block 3, default floor 0.5 | 48.04 [47.66–48.45] | 48.29 [48.09–48.41] |
| Gain from gating | +4.0% | +13.4% |

Three measured repeats per cell, 96 generated tokens, one warmup per task; greedy seed 1234. Same two prompts from the preceding experiment (27 and 2425 input tokens), context/resident capacity 8192, INT8 KV, prefill 256, auto expert cache, reserve 512 MiB. CPU prompt/PCIe sharing, adaptive swaps, lookup drafting and snapshots are off. The Q4_0 process holds 3279 target expert slots. Floor order reverses on alternating repeats. Changing whether confidence is requested clears the drafter's graph cache; that transition cost remains included. Decode excludes prefill/startup.

Every measured and exploratory threshold/block sequence has the same 96 output IDs per task in this process. This is fixture evidence, not a universal equality claim.

## What the profile shows

For the long case, both settings accept 62 proposals in 34 verification windows. The gate reduces offered proposals from 99 to 79, rejected proposals from 37 to 17 and total verified target rows from 133 to 113. The verifier therefore avoids 20 rows without losing any accepted proposal.

Median native milliseconds per window:

| Stage | Floor 0 | Default 0.5 |
| --- | ---: | ---: |
| Target verification | 62.09 | 54.30 |
| CPU expert execution (inside verification) | 47.94 | 40.55 |
| Draft context update and proposal | 4.19 | 4.15 |

The measured saving is predominantly target verification. This is not evidence of attention growing without bound: the official checkpoint uses a 2048-cell sliding window, and the CUDA runtime limits its attention launch to that window plus the proposed rows. No attention mask, KV layout, weight or target arithmetic was changed.

The cold first diagnostic pair (`diagnose.json`) showed 38.27→47.79 tok/s (+24.9%). It is retained as diagnosis, not pooled into the warmed confirmation medians above.

## Same-day MTP control

MTP was run afterward with the same target and prompt settings, setup floor 0.5, up to three proposals, warmups and three measured repeats. Both arms use automatic maximum VRAM residency: MTP fits 3134 target expert slots and Q4_0 fits 3279. This is the practical memory/cache comparison, not equal expert counts. The MTP and DFlash2 96-token sequences also match each other on both fixtures.

| Drafter | Code median tok/s [range] | Long median tok/s [range] |
| --- | ---: | ---: |
| MTP, floor 0.5 | 38.33 [35.82–38.36] | 51.57 [51.04–51.91] |
| Q4_0, floor 0.5 | 48.04 [47.66–48.45] | 48.29 [48.09–48.41] |
| Q4_0 versus MTP | +25.3% | -6.4% |

The long-case gap is reduced, not eliminated. MTP accepts 63/72 proposals in 33 windows and verifies 105 target rows; gated DFlash2 accepts 62/79 in 34 windows and verifies 113. MTP's median verification/draft costs are 53.96/2.42 ms per window, versus 54.30/4.15 for DFlash2. One additional verification window and the more expensive drafter account for the remaining measured latency. A confidence gate does not make the two draft models equally accurate or equally cheap.

This table is a fresh control; do not divide these numbers by the previous day's measurements. The same-day zero-floor DFlash2 is 17.5% behind MTP on this long fixture, versus 6.4% behind with floor 0.5. No four-task aggregate is inferred from this two-task follow-up.

## Additional probes

One exploratory long-case request per setting: block 3/floor 0.7 = 50.27 tok/s (61/70 accepted/offered); block 2/floor 0.5 = 48.90 (57/64); block 7/floor 0.5 = 40.36 (67/120). A wider trained pass did not solve this fixture: it admitted more bad proposals. These single samples do not justify another default change.

## Validation and limits

CUDA extension-ON and extension-OFF builds passed. Three ON CTests passed (`dflash2_parity`, `dflash_policy_test`, `dflash_load_test`); the two policy/loader OFF CTests also passed. Native startup confirms the fixed-block default is 0.50 and explicit CLI zero remains 0.00. Target-only controls (`temperature=1 top_k=1`, zero proposals) match all 192/192 fixture IDs at exactly the same 3279 expert slots as the DFlash2 performance process. All repeats, explored widths/floors and same-day MTP outputs match those fixture IDs. `checks.json` retains these checks.

The first output-control startup provided only 3275 slots for the requested 3279. Its placement assertion stopped before any request; its log and traceback are retained. For the successful unmeasured controls, the reserve was lowered from 512 to 480 MiB while keeping exactly 3279 actual slots. This changes no target/draft math. Performance runs retain their original 512 MiB reserve; the failed startup is not a timing sample.

This follow-up covers Q4_0 and two prompts only; BF16/Q8_0/Q5_0 and a full long-context length sweep were not rerun. Prompt length and task content differ in the two fixtures, so this does not establish that length alone causes the acceptance difference. HIP/SYCL/Metal/Windows and concurrent or multiple-GPU execution remain untested. Keep the pull request draft pending the missing backend validation and scope agreement.

An initial build command named a nonexistent `dflash2_parity_test` target; the corrected `dflash2_parity` build passed. No failed engine requests are omitted from the confirmation. All individual timings, IDs, native INFO and profiler lines are retained.

## Source and reproduction

Source parent: `86af7286254d1efb148cf0aa8c9a96868d4e9f09`. The tested `src/program/generate.cpp` SHA-256 and both before/after executable hashes are in `provenance.json`; the source change is limited to the default-floor condition and its comment. The Q4_0 GGUF hash was rechecked against the preceding artifact manifest. No weights or executables are included.

From the repository root, with the existing local model artifacts and built engine, use a fresh results directory:

```sh
STRATA_LONG_RESULTS=/tmp/strata-long-recheck .venv/bin/python bench/results/2026-10-20-dflash2-long-prompt/confirm.py --mode dflash2
STRATA_LONG_RESULTS=/tmp/strata-long-recheck .venv/bin/python bench/results/2026-10-20-dflash2-long-prompt/confirm.py --mode mtp
STRATA_LONG_RESULTS=/tmp/strata-long-recheck .venv/bin/python bench/results/2026-10-20-dflash2-long-prompt/controls.py
STRATA_LONG_RESULTS=/tmp/strata-long-recheck .venv/bin/python bench/results/2026-10-20-dflash2-long-prompt/analyze.py
```

The portable helpers accept `STRATA_LONG_RESULTS`; the measured run used the sibling `Strata-data/dflash2-long-prompt` directory. `diagnose.py` is the initial exploratory helper and additionally requires the saved before-fix binary `strata-before` in its output directory. Exact engine argv/environment are retained in the command JSON files. Build argv and validation outcomes are in `provenance.json`; CTest outputs are included. The full extension-OFF compiler log remains local; its final build output is summarized here.
