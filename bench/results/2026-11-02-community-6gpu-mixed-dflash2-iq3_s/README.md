# Community benchmark: DFlash2 against MTP, original against Swift 1.5 IQ3_S, six mixed cards

> **Read this first.** These runs measure my own branch on `c62d5e3a` (engine 0.1.54), not current main. Upstream main has since moved to 0.2.10, engine 0.1.56, with its own served DFlash2 on one card and on a layer split, and batch slot drafts. Main now starts DFlash2 with `--spec 6 --draft-conf 0.35`. I have not run that code, so these numbers do not say how main performs on these cards. `provenance/branch.diff` is what was built. I replaced the data folder in the logs with `/models/` and removed the machine's name; nothing else in the raw files changed.

Measured 2026-11-02 on one machine. This compares two drafters and two sets of weights on a six-card layer split, with DFlash2 serving
requests, conversations reused and two requests at once. **DFlash2 was not faster than MTP here. Original and Swift 1.5 weights ran at the same
speed.** The build is a modified engine, not a release. No default change is proposed.

## Results in short

| Question | Answer on this machine | Measured |
|---|---|---|
| Is DFlash2 faster than MTP? | No. 2 to 3% slower over all decode prompts | Median 51.5 against 53.1 tok/s (Swift), 51.8 against 52.8 (original) |
| Is it faster on anything? | Forced tool calls, in all four pairs | +1.9 to +3.9% |
| Is it slower on something? | General chat and code edits, in all four pairs. With thinking on and model-card sampling, on two of three cases in one run | Chat -3.2 to -6.9%, edits -1.2 to -3.2%. Standard benchmark about -10% on `short` and `prompt-16k` |
| Does DFlash2 slow a fresh prompt read? | Yes, by a nearly fixed amount | +0.4 to +0.5 s at 4K, +1.0 to +1.1 s at 18.7K and 128K |
| Does a follow-up turn cost more with DFlash2? | No | 99.7 to 99.8% of tokens reused with either drafter; first token at about 0.5 s |
| Do Swift 1.5 weights cost speed against the original? | No | Decode within 1%, prompt read within 0.3%, drafts accepted within 0.1 points, with both drafters |
| Do wider windows let DFlash2 win? | On code, JSON and tool calls it beats MTP at the same wider window, but both are slower than at window 4 | Code at window 4, 6, 8: MTP 55.9, 49.7, 44.7 tok/s; DFlash2 55.7, 53.9, 50.2 |
| Does saving the DFlash2 ring between conversations help? | A little | 88.4% against 85.1% of drafts accepted after a switch; speed within noise |
| Do answers change? | No sign of it | A repeat of the same configuration matches about as often as the two drafters match each other; 12 of 12 sanity checks for both |

Over the eight main runs the server generated **128,259 output tokens** and read about **4.7 million fresh prompt tokens**. The fixed decode
prompts are 21,504 tokens per configuration (7 prompts, 512 tokens, 3 repeats, 2 runs).

## Hardware and software

- GPUs in split order: RTX 2080 Ti 11 GB, RTX 3060 12 GB, 3 x Tesla P100 16 GB, RTX 3060 12 GB. Driver 580.178.04. The drafter runs on the last RTX 3060.
- Xeon W-2150B, 188 GiB RAM, Ubuntu 24.04.5, kernel 6.8.0-147. Models on an NVMe SSD (LVM).
- Strata source: base commit `870aedd9` (0.2.8, engine 0.1.54) plus the uncommitted changes in [provenance/branch.diff](provenance/branch.diff), which add DFlash2
  serving, conversation reuse and batch slots. Built with the root `Dockerfile`, CUDA 12.9.1, `CUDA_ARCHITECTURES=60;75;86`, one image for every run.
  The tree hash is the same before the build and after the runs. Upstream main has moved 169 commits since this base.
- Nothing else used the GPUs. No throttle reasons were logged and the hottest card reached 68 C (`raw/*.gpu.csv`).

## Model and configuration

- Original: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision `ed59f920`, IQ3_S, both shards checked against the sizes and SHA-256 in `setup.py`.
  Pack made with `tools/iq_pack.py` ([log](prepare-iq3-orig.log)).
- Swift 1.5: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`. Its sizes match `setup.py`'s Swift 1.5 table; its hash was not checked here.
- Same for both ([config/](config/)): context 178,176, `--kv int8`, `--spec 4`, `--spec-min-p 0.5`, `--prefill auto`, `--prefill-pipe 6144`, `--adapt-*` as listed, `layer_split: auto`,
  `parallel: 2`, a 16 GB and 8-entry conversation cache, picture reader off. Both use the same MTP files and the same DFlash2 pack (`incoai` revision `a22a1744`, Q8).
- The drafter was chosen only with the `DRAFTING` environment variable, through `strata serve --profile`. `/metrics` confirmed it on every run.
- Experts in VRAM after loading: 19,148 slots with MTP, 19,231 with DFlash2 (37,634 and 37,812 MiB).

## Method

Scripts: [scripts/](scripts/). Each run is a freshly started engine. The four configurations ran in one order and then in the reverse order (`r1`, `r2`). No request was retried. Speeds are the
engine's `timings`; first-token times are the client's clock. Each run has these parts:

1. **Fresh reads.** About 4K, 32K and 128K tokens, 3 times each, with a unique first line so that only the template is reused. Output cap 32.
2. **Decode.** Code, prose, JSON, code edits, chat and a forced tool call at temperature 0 with thinking off, and a thinking prompt with temperature 1, `top_p` 0.95, `top_k` 20. Cap 512, 3 times each.
   All 42 per configuration reached the cap, so these say nothing about time to a finished answer.
3. **The ring.** One 17,608-token prompt with a fixed assistant reply, 300 tokens at temperature 0, sent read fresh, continued in place, and twice more after an unrelated long conversation.
4. **Session and parallel.** Six follow-up turns of an 18,726-token conversation, a short side request, a second long conversation, switching between them, 3 rounds of the two long conversations at once and of three short requests at once on two slots, then a cancelled request and the turn after it.

Controls: two more runs with `STRATA_DF2_RING_CACHE=0` (no saved rings, parts 3 and 4 only), and one draft standard benchmark for each drafter on the original weights.

Raw data: [raw/](raw/) has every call (`.jsonl`), every output (`.jsonl.text`), engine and server logs, and VRAM and GPU samples. [analysis.txt](analysis.txt) is the output of the two analysis scripts.
Request logs and their engine lines were matched in order; drafts accepted exist only for requests decoded alone, because batch windows do not draft.

## Results

### Fresh prompt read

Tokens per second, median (range) of 6 reads. Time is the median read time.

| Prompt | Swift MTP | Swift DFlash2 | Original MTP | Original DFlash2 | DFlash2 extra time |
|---:|---:|---:|---:|---:|---:|
| 4K | 370 (368-423) | 357 (351-398) | 369 (368-422) | 352 (315-397) | +0.40 s, +0.53 s |
| 32K | 1,081 (1,078-1,082) | 1,051 (1,050-1,052) | 1,079 (1,078-1,081) | 1,051 (1,049-1,051) | +0.85 s, +0.79 s |
| 128K | 1,499 (1,497-1,502) | 1,481 (1,480-1,482) | 1,495 (1,494-1,498) | 1,478 (1,477-1,483) | +1.08 s, +1.02 s |
| 18.7K (session, n=2) | 20.84 s | 21.83 s | 20.85 s | 21.84 s | +0.99 s, +0.98 s |

The extra time is close to fixed, about 1 s once a prompt has more than the drafter's 2,047 context rows. That fits the drafter's context update for those rows, but this was not isolated: unconfirmed.

### Decode

Engine tokens per second, median (range), n=6, 3,072 tokens per cell.

| Prompt | Swift MTP | Swift DFlash2 | Original MTP | Original DFlash2 |
|---|---:|---:|---:|---:|
| Code | 55.9 (53.8-56.4) | 55.7 (54.1-56.6) | 55.2 (54.4-56.8) | 54.3 (53.5-54.5) |
| Prose | 42.5 (41.5-43.4) | 42.2 (40.0-42.6) | 43.4 (41.2-44.0) | 42.0 (40.9-42.4) |
| JSON | 57.8 (57.1-59.7) | 58.1 (57.4-58.8) | 59.0 (58.8-60.6) | 57.9 (57.5-58.6) |
| Code edits | 52.6 (52.4-54.4) | 51.0 (50.2-52.0) | 52.8 (51.8-52.8) | 51.3 (50.8-51.7) |
| Chat | 42.1 (41.2-43.8) | 39.4 (37.5-42.3) | 40.9 (40.0-45.3) | 39.3 (39.0-41.0) |
| Tool call | 56.9 (56.0-57.8) | 58.7 (57.1-59.4) | 57.0 (55.8-57.9) | 58.5 (55.9-59.2) |
| Thinking, sampled | 52.8 (48.4-53.4) | 49.0 (46.1-56.4) | 49.2 (46.7-50.7) | 51.5 (45.8-52.9) |
| All (n=42) | 53.1 | 51.5 | 52.8 | 51.8 |

DFlash2 against MTP, same weights and same round (median of 3 each). Differences that change sign are noise:

| Pair | Code | Prose | JSON | Edits | Chat | Tool | Thinking |
|---|---:|---:|---:|---:|---:|---:|---:|
| Swift, round 1 | -1.1% | -1.9% | +0.2% | -3.2% | -6.9% | +3.9% | -5.1% |
| Swift, round 2 | +0.4% | +1.2% | +1.0% | -3.2% | -6.2% | +2.4% | -3.1% |
| Original, round 1 | -1.8% | -1.9% | -1.4% | -1.2% | -3.2% | +1.9% | +7.5% |
| Original, round 2 | -1.6% | -4.8% | -2.4% | -2.3% | -4.8% | +2.6% | -4.2% |

Drafts accepted, all six runs per cell:

| Prompt | Swift MTP | Swift DFlash2 | Original MTP | Original DFlash2 |
|---|---:|---:|---:|---:|
| Code | 80.7% | 81.9% | 79.9% | 78.4% |
| Prose | 52.5% | 53.3% | 54.0% | 54.2% |
| JSON | 87.2% | 88.7% | 89.0% | 88.0% |
| Code edits | 79.0% | 76.9% | 78.8% | 77.5% |
| Chat | 57.9% | 52.8% | 57.1% | 54.1% |
| Tool call | 82.5% | 86.4% | 82.8% | 86.3% |
| Thinking, sampled | 72.2% | 71.2% | 70.6% | 72.2% |
| All | 72.6% (13,527 of 18,635) | 72.2% (13,438 of 18,618) | 72.7% (13,455 of 18,518) | 72.3% (13,493 of 18,664) |

The DFlash2 drafter was trained on the original weights. Its acceptance on Swift 1.5 is 72.2% against 72.3% on the original, so there is no sign of a cost from the fine-tune.

### Standard benchmark, draft protocol, original weights

`strata-qwen38-thinking-v1-draft.1`, one complete run per drafter, n=3 per case, thinking on, seeds 1 to 3. **Track `custom`; this is a draft protocol and a modified engine.** Folders: [standard/](standard/).

| Case | MTP prompt tok/s | DFlash2 prompt tok/s | MTP decode tok/s | DFlash2 decode tok/s | MTP first token | DFlash2 first token |
|---|---:|---:|---:|---:|---:|---:|
| short | n/a | n/a | 50.5 (48.3-52.9) | 45.6 (43.7-49.1) | 1.82 s | 2.41 s |
| prompt-4k | 428.0 | 404.1 | 43.8 (41.2-44.1) | 43.3 (40.3-43.5) | 9.60 s | 10.17 s |
| prompt-16k | 857.9 | 810.8 | 45.5 (42.2-48.7) | 40.4 (39.2-44.4) | 19.15 s | 20.26 s |

Sanity: 12 of 12 for both. Requests that reached the cap: 7 of 9 with MTP, 6 of 9 with DFlash2.

### Follow-up turns

Tokens reused, then first-token time, in the two runs of each configuration. All follow-ups together:

| | Swift MTP | Swift DFlash2 | Original MTP | Original DFlash2 |
|---|---:|---:|---:|---:|
| Tokens reused | 99.83% | 99.82% | 99.77% | 99.72% |
| First token, median (range) | 0.52 s (0.46-1.07) | 0.55 s (0.49-1.10) | 0.54 s (0.49-2.49) | 0.54 s (0.51-3.18) |

The slow ones (2.49 s, 3.09 s, 3.18 s) are turns that reused about 18,788 of 19,032 tokens. The server rendered a reply that did not match its saved tokens and read on from an earlier checkpoint. Both drafters had them.

### Requests at once

Batch slots do not draft in this build with either drafter, the same as MTP's default.

| | Swift MTP | Swift DFlash2 | Original MTP | Original DFlash2 |
|---|---:|---:|---:|---:|
| Three short requests, two slots, median wall time (600 tokens) | 11.07 s | 11.21 s | 11.40 s | 11.24 s |
| Two long conversations, medians across rounds | 5.51 s | 8.55 s | 5.88 s | 5.66 s |

The long row depends on tokens reused and reply lengths, which differ by run. **One pattern is real and unexplained:** in round 1 of the long case, where each conversation adds about 170 tokens, DFlash2 took 9.1 to 9.7 s in all six of its runs and MTP 7.3 to 8.3 s in all four of its runs.
The Swift DFlash2 median of 8.55 s also comes partly from `r2-swift-dflash2`, where two rounds read about 180 to 230 tokens again from a checkpoint.

### Saved rings

DFlash2's context ring holds its last 2,064 rows. This build saves a copy (about 81 MiB) when another conversation takes the ring. Same prompt, 300 tokens, temperature 0:

| Ring state | Requests | Drafts accepted | Tok/s, median |
|---|---:|---:|---:|
| Saved copy restored, after a switch (`r1`, `r2`) | 8 | 1,716 of 1,942 = 88.4% | 59.7 |
| No saved copy, zeroed, after a switch (`c1`) | 4 | 854 of 1,003 = 85.1% | 58.6 |

Follow-ups in the session after a switch: 83.0 to 87.0% with saved copies, 81.3 to 85.1% without; 51.8 to 55.7 and 53.2 to 54.4 tok/s. Restoring took about 14 to 66 ms.
So the saved copy adds about 3 points of acceptance and no clear speed. The prompt here is self-contained: work that depends on distant context could gain more. That was not measured.

### Wider windows

`--spec 4` limits a drafter's window to 4 rows, so DFlash2 offered at most 3 of its 7 drafts. Later the same day, Swift 1.5 only, the decode prompts of the method above (21 requests, 10,752 tokens per run), one run per cell. Window 4 is the two matrix runs. Source: [analysis-windows.txt](analysis-windows.txt), `raw/s4-*`, [scripts/spec_runs.sh](scripts/spec_runs.sh).

| `--spec`, `--spec-min-p` | Drafter | All, median tok/s | Code | Prose | JSON | Edits | Chat | Tool | Thinking | Tokens per window | Rows per window | Accepted | ms per window |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4, 0.5 | MTP | 53.1 | 55.9 | 42.5 | 57.8 | 52.6 | 42.1 | 56.9 | 52.8 | 2.70 | 3.34 | 72.6% | 53.3 |
| 4, 0.5 | DFlash2 | 51.5 | 55.7 | 42.2 | 58.1 | 51.0 | 39.4 | 58.7 | 49.0 | 2.67 | 3.31 | 72.2% | 53.8 |
| 6, 0.5 | MTP | 49.7 | 49.7 | 38.1 | 56.1 | 51.5 | 38.6 | 53.0 | 47.4 | 3.13 | 4.34 | 63.7% | 66.7 |
| 6, 0.5 | DFlash2 | 50.1 | 53.9 | 36.6 | 54.8 | 49.5 | 37.8 | 56.3 | 46.1 | 3.12 | 4.35 | 63.3% | 66.8 |
| 8, 0.5 | MTP | 46.4 | 44.7 | 35.7 | 53.6 | 48.7 | 35.8 | 49.1 | 44.3 | 3.33 | 5.02 | 58.0% | 75.9 |
| 8, 0.5 | DFlash2 | 45.5 | 50.2 | 36.8 | 57.0 | 45.3 | 35.1 | 52.5 | 44.0 | 3.35 | 5.10 | 57.2% | 74.7 |
| 8, 0.3 | MTP | 40.0 | 40.4 | 27.3 | 50.6 | 40.7 | 27.3 | 49.1 | 35.7 | 3.36 | 6.53 | 42.8% | 92.1 |
| 8, 0.3 | DFlash2 | 38.6 | 46.0 | 23.8 | 51.5 | 37.1 | 24.0 | 51.0 | 37.2 | 3.38 | 7.05 | 39.4% | 95.5 |

A least-squares fit of request time to windows and rows gives about 12 to 20 ms a window plus 10 to 12 ms a row at the first three settings (at the last, 6 to 16 ms a window). At window 4 a token costs about 20 ms, so a draft pays only when it is accepted more than about 60% of the time. Going from window 4 to 6 added about one row (13 ms) and 0.45 tokens per window. This is the likely reason neither drafter gains from wider windows on this machine. It is a model fit, and cards with cheaper rows may behave differently.

### Do answers change?

At temperature 0, repeats of one configuration can differ: 17 or 18 of 33 outputs were identical between its two runs, whichever the configuration. The same prompts matched between the two drafters in 9 to 14 of 27 cases. On the ring prompt, MTP and DFlash2 produced some of the same exact outputs (`fd6f25` and `36fb15` on the original; `30324f` and `e7857f` on Swift).
This is no sign that DFlash2 changes answers. It is also not a proof, because the engine does not repeat exactly here with either drafter; the cause of that was not investigated.

## Failures and limits

- **No failed requests** and no engine, GPU or non-finite errors in any of the 12 runs.
- Two sets of runs on one machine, one day. A different card order, a faster last card or a different `--spec` could change this. The 4K read varies most: the first of a run was faster than the later ones (423 against 368 tok/s on Swift with MTP); the cause was not investigated.
- The drafter runs on an RTX 3060, the class where the engine already warns that MTP was faster. This says nothing about other cards.
- Not measured: time to a finished answer (decode prompts reach the cap), Swift 1.5's shorter thinking, answer quality, sustained load, conversations past about 20,000 tokens on this build, more than two slots, wider windows on the original weights, `--batch-dflash2` from PR #4937 (its head does not compile: see below).
- PR #4937's head `761fe6c0` was built with the same recipe and stopped at `src/program/generate.cpp:16686`: `settle_mtp_drafts` is not declared. Log kept on the test machine. So that design is not measured here.
- HIP compiled for gfx1201 and gfx1151 on ROCm 7.2.2 (compile only, no AMD hardware). SYCL and Metal not built. No claim for those backends.
