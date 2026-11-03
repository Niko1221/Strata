# Community benchmark: served DFlash2 on a modified 0.1.55 engine, six mixed cards

> **Read this first.** These runs measure my own branch on `c62d5e3a` (engine 0.1.55), not current main. Upstream main has since moved to 0.2.10, engine 0.1.56, with its own served DFlash2 on one card and on a layer split, and batch slot drafts. Main now starts DFlash2 with `--spec 6 --draft-conf 0.35`. I have not run that code, so these numbers do not say how main performs on these cards. `provenance/branch-0.1.55.diff` is what was built. I replaced the data folder in the logs with `/models/` and removed the machine's name; nothing else in the raw files changed.

Measured 2026-11-02 on one machine, on commit `cc75837c`: upstream main `c62d5e3a` (0.2.9, engine 0.1.55) merged with branch `dflash2`, which adds DFlash2 serving,
conversation reuse with it, and batch slots beside it. **Follow-up turns and parallel requests now work with DFlash2 as with MTP. On this machine it is no faster than MTP overall. The branch does not slow MTP.** No default change is proposed.
An earlier set of runs on the previous base is in [the 0.1.54 report](../2026-11-02-community-6gpu-mixed-dflash2-iq3_s/README.md).

## Results in short

| Question | Answer on this machine | Measured |
|---|---|---|
| Does the branch change MTP against upstream main? | No sign of it | All decode prompts 53.9, 53.8 tok/s on the branch; 53.7, 53.9 on main. Reads at 128K 1,498, 1,498 against 1,499, 1,499. Same VRAM |
| Do follow-up turns reuse conversations with DFlash2? | Yes, as with MTP | 99.80 to 99.83% of tokens against 99.72 to 99.83%; first token 0.54 s against 0.52 to 0.54 s |
| Is DFlash2 faster than MTP? | No | All decode prompts 51.8 against 53.9 tok/s (Swift 1.5), 52.1 against 52.9 (original) |
| Is it faster on anything? | Forced tool calls, in all four pairs | +1.0 to +3.4% |
| Is it slower on something? | Chat, prose and code, in all four pairs | -2.6 to -6.5%, -1.5 to -5.2%, -0.9 to -1.3% |
| Does it slow a fresh prompt read? | Yes, by a near-fixed amount | +0.5 s at 4K, +0.8 s at 32K, +1.0 s at 18.7K and 128K |
| Do wider windows let DFlash2 win? | Only against MTP at the same width; both are slower than at `--spec 4` | Code at `--spec` 4, 6, 8: MTP 55.4, 51.2, 47.5; DFlash2 54.8, 54.2, 49.9 |
| Do Swift 1.5 weights cost speed? | No | Decode and reads within 1.9%, in either direction; drafts accepted 71.7 to 72.7% |
| Do saved rings help? | A little, and not on every prompt | 87.8% against 85.4% of drafts accepted after a switch |
| Is the output affected? | No sign of it | 12 of 12 sanity checks for both; see [below](#do-answers-change) |

The eight main runs generated **129,282 output tokens**; all 16 runs of this kind generated 215,584 and read 6.07 million fresh prompt tokens. Standard benchmarks are extra. Every decode cell is 3,072 tokens.

## Hardware and software

- GPUs in split order: RTX 2080 Ti 11 GB, RTX 3060 12 GB, 3 x Tesla P100 16 GB, RTX 3060 12 GB. Driver 580.178.04. The drafter runs on the last RTX 3060.
- Xeon W-2150B (10 cores, 20 threads; the engine's default of 9 pool workers plus the host), 188 GiB RAM, Ubuntu 24.04.5, kernel 6.8.0-147. Models on an NVMe SSD.
- Branch image: `cc75837c`, root `Dockerfile`, CUDA 12.9.1, `CUDA_ARCHITECTURES=60;75;86`, `BUILD_VISION=1`.
- Baseline image: `c62d5e3a` with only the P100 fix of PR #4274 (`include/strata/kernels/dp4a.hpp`), same recipe, `BUILD_VISION=0`. The branch also contains that fix.
- Nothing else used the GPUs. No throttle reasons were logged. One run was affected by other CPU work; see [Failures and limits](#failures-and-limits).

## Model and configuration

As in the [earlier report](../2026-11-02-community-6gpu-mixed-dflash2-iq3_s/README.md#model-and-configuration): original IQ3_S (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `ed59f920`, sizes and SHA-256 checked) and Swift 1.5 IQ3_S (sizes match `setup.py`, hash not checked). [config/](config/) has the files. `--spec 4`, `--spec-min-p 0.5`, context 178,176, `--kv int8`, `parallel: 2`, a 16 GB and 8-entry conversation cache, picture reader off. The drafter was chosen only with `DRAFTING`.

## Method

Scripts: [scripts/](scripts/) (`matrix2.sh` takes its start from `matrix-base.sh`). Each run is a freshly started engine, and nothing was retried. Order: main MTP, four branch configurations, the same four reversed, main MTP. Then two runs without saved rings, four at wider windows, and the draft standard benchmark for each drafter on the original weights.

A full run has fresh reads (about 4K, 32K, 128K, 3 each), decode (code, prose, JSON, code edits, chat and a forced tool call at temperature 0 with thinking off, and a thinking prompt with temperature 1, 512 tokens, 3 each; all reach the cap, so they do not measure time to a finished answer), one prompt in three ring states, and a session with six follow-ups, a side request, a second conversation, parallel rounds and a cancel. Speeds are the engine's `timings`; first-token times are the client's clock.

Raw data: [raw/](raw/), [standard/](standard/). Tables come from [analysis.txt](analysis.txt), [analysis-baseline.txt](analysis-baseline.txt), [analysis-windows.txt](analysis-windows.txt).

## Results

### MTP: the branch against upstream main

Swift 1.5, same flags. Main ran first and last.

| | Main, first | Branch, round 1 | Branch, round 2 | Main, last |
|---|---:|---:|---:|---:|
| All decode, median tok/s | 53.7 | 53.9 | 53.8 | 53.9 |
| Code | 55.3 | 55.4 | 55.4 | 55.5 |
| Prose | 42.1 | 40.8 | 42.5 | 41.2 |
| JSON | 58.0 | 58.2 | 57.9 | 57.6 |
| Code edits | 53.7 | 53.9 | 53.8 | 53.8 |
| Chat | 42.8 | 42.9 | 41.7 | 41.7 |
| Tool call | 56.5 | 56.2 | 56.3 | 57.3 |
| Thinking | 51.0 | 51.7 | 48.8 | 51.3 |
| Read 4K, tok/s | 370 | 370 | 370 | 370 |
| Read 32K | 1,082 | 1,082 | 1,081 | 1,081 |
| Read 128K | 1,499 | 1,498 | 1,498 | 1,499 |
| First token, follow-ups | 0.54 s | 0.52 s | 0.53 s | 0.51 s |
| Tokens reused, follow-ups | 99.82% | 99.82% | 99.83% | 99.83% |
| Drafts accepted | 72.1% | 72.5% | 71.2% | 72.0% |

VRAM after loading was the same to the MiB on all six cards in all four. Their engine arguments were the same.

### Fresh prompt read

Tokens per second, median (range) of 6 reads.

| Prompt | Swift MTP | Swift DFlash2 | Original MTP | Original DFlash2 | DFlash2 extra time |
|---:|---:|---:|---:|---:|---:|
| 4K | 370 (365-424) | 355 (348-398) | 370 (368-423) | 351 (342-397) | +0.46 s, +0.57 s |
| 32K | 1,082 (1,081-1,086) | 1,053 (1,052-1,056) | 1,080 (1,080-1,085) | 1,052 (1,047-1,053) | +0.80 s, +0.82 s |
| 128K | 1,498 (1,497-1,500) | 1,482 (1,480-1,485) | 1,496 (1,495-1,498) | 1,478 (1,476-1,480) | +0.92 s, +1.01 s |
| 18.7K (session, n=2) | 20.86 s | 21.82 s | 20.83 s | 21.78 s | +0.96 s, +0.96 s |

The extra time levels off at about 1 s. That fits the drafter's context update for its 2,047 rows, but this was not isolated: unconfirmed.

### Decode

Engine tokens per second, median (range), n=6.

| Prompt | Swift MTP | Swift DFlash2 | Original MTP | Original DFlash2 |
|---|---:|---:|---:|---:|
| Code | 55.4 (53.9-55.7) | 54.8 (53.7-55.6) | 55.0 (54.0-55.4) | 54.4 (54.0-55.3) |
| Prose | 41.6 (40.6-42.7) | 40.2 (40.1-42.1) | 42.7 (42.2-44.0) | 41.2 (39.8-44.0) |
| JSON | 58.0 (57.4-58.7) | 58.3 (57.8-59.0) | 59.1 (58.3-60.4) | 57.8 (57.4-58.4) |
| Code edits | 53.8 (51.5-54.4) | 51.3 (49.4-53.9) | 52.7 (51.5-54.3) | 52.1 (49.9-53.2) |
| Chat | 42.8 (41.1-43.6) | 39.7 (38.0-41.9) | 41.9 (40.1-43.1) | 40.6 (40.5-42.1) |
| Tool call | 56.2 (55.8-57.6) | 58.1 (54.9-58.4) | 57.0 (56.2-58.3) | 58.2 (57.7-60.1) |
| Thinking, sampled | 51.4 (46.9-55.6) | 49.5 (46.5-52.3) | 49.9 (45.1-53.0) | 49.0 (47.0-51.3) |
| All (n=42) | 53.9 | 51.8 | 52.9 | 52.1 |

DFlash2 against MTP, same weights and same round (median of 3 each):

| Pair | Code | Prose | JSON | Edits | Chat | Tool | Thinking |
|---|---:|---:|---:|---:|---:|---:|---:|
| Swift, round 1 | -0.9% | -1.5% | +0.5% | -4.6% | -6.5% | +3.4% | -4.8% |
| Swift, round 2 | -1.1% | -5.2% | +0.3% | -4.6% | -6.0% | +2.5% | +2.3% |
| Original, round 1 | -1.3% | -2.1% | -1.4% | +0.6% | -3.3% | +1.0% | -3.0% |
| Original, round 2 | -1.1% | -3.5% | -2.7% | -0.9% | -2.6% | +3.0% | +0.8% |

Drafts accepted, all six runs per cell:

| Prompt | Swift MTP | Swift DFlash2 | Original MTP | Original DFlash2 |
|---|---:|---:|---:|---:|
| Code | 79.1% | 80.1% | 78.7% | 79.3% |
| Prose | 50.2% | 50.9% | 53.8% | 52.4% |
| JSON | 86.1% | 89.1% | 88.3% | 88.0% |
| Code edits | 79.5% | 78.3% | 79.1% | 78.1% |
| Chat | 58.1% | 53.1% | 57.1% | 58.2% |
| Tool call | 81.3% | 85.1% | 82.5% | 87.3% |
| Thinking, sampled | 72.8% | 70.7% | 70.1% | 70.7% |
| All | 71.8% (13,474 of 18,764) | 71.7% (13,390 of 18,688) | 72.3% (13,490 of 18,652) | 72.7% (13,538 of 18,620) |

### Wider windows

`--spec 4` limits a drafter's window to 4 rows, so DFlash2 offers at most 3 of its 7 drafts. Swift 1.5, decode prompts, one run each for 6 and 8.

| `--spec` | Drafter | All | Code | Prose | JSON | Edits | Chat | Tool | Thinking | Tokens per window | Accepted | ms per window |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | MTP | 53.9 | 55.4 | 41.6 | 58.0 | 53.8 | 42.8 | 56.2 | 51.4 | 2.68 | 71.8% | 53.2 |
| 4 | DFlash2 | 51.8 | 54.8 | 40.2 | 58.3 | 51.3 | 39.7 | 58.1 | 49.5 | 2.65 | 71.7% | 53.9 |
| 6 | MTP | 49.9 | 51.2 | 37.4 | 55.1 | 50.4 | 38.6 | 53.4 | 47.5 | 3.12 | 62.7% | 66.9 |
| 6 | DFlash2 | 49.4 | 54.2 | 36.8 | 56.5 | 48.7 | 36.7 | 58.1 | 47.2 | 3.10 | 62.5% | 66.6 |
| 8 | MTP | 45.8 | 47.5 | 36.4 | 53.1 | 47.3 | 36.5 | 46.3 | 41.8 | 3.23 | 56.2% | 74.2 |
| 8 | DFlash2 | 45.2 | 49.9 | 34.8 | 57.0 | 45.2 | 34.4 | 53.5 | 42.8 | 3.23 | 56.4% | 73.3 |

A least-squares fit of request time gives 12 to 19 ms a window plus 11 to 13 ms a row. At `--spec 4` a token costs about 20 ms, so an extra row pays only when it is accepted more than about 60% of the time. Going to `--spec 6` adds about 13 ms a window and 0.44 tokens. That fits what the table shows: DFlash2 is better than MTP deep in a window on code, JSON and tool calls, but deep rows do not pay here. Cards with cheaper rows could differ. It is a model, not a direct measurement.

### Standard benchmark, draft protocol, original weights

`strata-qwen38-thinking-v1-draft.1`, one run per drafter, n=3 per case, thinking on. **Track `custom`, draft protocol, modified engine.** These are the clean reruns; the earlier pair is also in [standard/](standard/).

| Case | MTP prompt tok/s | DFlash2 prompt tok/s | MTP decode tok/s | DFlash2 decode tok/s | MTP first token | DFlash2 first token |
|---|---:|---:|---:|---:|---:|---:|
| short | n/a | n/a | 47.5 (47.1-50.0) | 45.9 (41.1-49.8) | 1.81 s | 2.46 s |
| prompt-4k | 428.7 | 404.3 | 42.9 (42.6-45.8) | 44.3 (41.7-45.4) | 9.58 s | 10.16 s |
| prompt-16k | 856.5 | 808.7 | 40.3 (39.5-45.7) | 43.8 (40.4-44.4) | 19.18 s | 20.31 s |

Sanity: 12 of 12 for both. The decode differences are within these ranges. On engine 0.1.54 the same benchmark had DFlash2 about 10% slower on two of these cases, so with n=3 and sampling, neither result is evidence of a real difference. The prompt read difference repeats.

### Follow-up turns and parallel requests

| | Swift MTP | Swift DFlash2 | Original MTP | Original DFlash2 |
|---|---:|---:|---:|---:|
| Tokens reused in follow-ups | 99.83% | 99.80% | 99.72% | 99.83% |
| First token, median (range) | 0.52 s (0.46-1.06) | 0.54 s (0.45-2.86) | 0.54 s (0.45-2.47) | 0.54 s (0.49-0.66) |
| Three short requests on two slots, median wall (600 tokens) | 10.80 s | 10.91 s | 10.91 s | 11.09 s |
| Two long conversations at once, median wall | 7.15 s | 5.56 s | 5.82 s | 7.67 s |

The slow follow-ups (2.47 s, 2.86 s) read about 180 to 240 tokens again from a checkpoint because a replayed reply did not match the saved tokens. Both drafters had them. The long wall times mainly follow from how often that happens in a run. In the 0.1.54 runs DFlash2 was slower on round 1 of that case; here it was 5.8 to 9.1 s against 7.3 to 8.6 s for MTP, so that does not repeat.

### Saved rings

Same 17,608-token prompt, 300 tokens, temperature 0, after an unrelated long conversation took the ring.

| Ring state | Weights | Requests | Drafts accepted | Tok/s |
|---|---|---:|---:|---|
| Saved copy restored | Both | 8 | 1,721 of 1,961 = 87.8% | 58.9-61.6 |
| No saved copy, zeroed | Both | 4 | 851 of 996 = 85.4% | 56.1-61.5 |
| Saved copy restored | Original | 4 | 88.5% | 60.7-61.6 |
| No saved copy, zeroed | Original | 2 | 89.7% | 61.5 |
| Saved copy restored | Swift 1.5 | 4 | 87.0% | 58.9-59.6 |
| No saved copy, zeroed | Swift 1.5 | 2 | 81.5% | 56.1-56.6 |

With both reports together: 88.1% (3,437 of 3,903) against 85.3% (1,705 of 1,999). The gain is small and varies; this prompt repeats almost the same output with little need for distant context. Restoring took 14 to 66 ms in the earlier report; here see [analysis.txt](analysis.txt).

### Do answers change?

At temperature 0, repeats differ with either drafter. Of 33 outputs per run, two runs of one configuration matched on 15 or 16 for three configurations, and 33 of 33 for original DFlash2. Main against main matched on 24, main against branch on 15 to 24. Between the drafters, 8 to 10 of 27 matched in a round, and across all runs both drafters reached a same exact answer on 13 of 28 prompts (Swift) and 10 of 28 (original). This is no sign of DFlash2 changing answers, and no proof: the engine does not repeat exactly here. The cause was not investigated.

## Failures and limits

- **A contaminated run, excluded.** A compile job that was started by mistake overlapped the first DFlash2 standard benchmark (19:44 to 19:49 UTC). Its prompt-4k read fell to as low as 369 tok/s. It is kept as `standard/contaminated-std-dflash2*`, not used, and both drafters were rerun on a quiet machine. `standard/first-std-mtp*` is the MTP run from before that; it agrees with its rerun.
- No failed requests, and no engine, GPU or non-finite errors in any kept run.
- Other kinds of hardware were not tested. The drafter runs on an RTX 3060, a class where the engine already warns that MTP was faster.
- Not measured: time to a finished answer, Swift's shorter thinking, answer quality, sustained load, more than two slots, conversations past about 20,500 tokens, wider windows on the original weights, a fresh `--pipeline-windows` comparison, PR #4937's `--batch-dflash2` (its head `761fe6c0` does not compile: `settle_mtp_drafts` at `src/program/generate.cpp:16686` is not declared).
