# Layer split on 2 x RTX 2080 Ti (2026-09-30)

**Setup:**
- **PC:** 2 x RTX 2080 Ti 11 GB (Turing, sm_75, both PCIe gen3 x16, 12.6 GB/s host->device each), Core i7-7700K
  (4C/8T, AVX2), 64 GB RAM, Linux, CUDA 12.0, the experimental sm_75 build (`-DSTRATA_EXPERIMENTAL_SM75=ON`).
- **Model:** Qwen3.8-Flash-Next IQ2_XS (48 layers x 512 experts = 24,576 pairs, 33 GiB of experts), 32K context,
  int8 KV, `--spec 4`, MTP drafter, `--expert-cache auto`, `--prefill auto`, greedy.
- **Harness:** `split_bench.py` in this folder: 3 decode prompts (story, LRU-cache code, TCP explanation) x 2 at 450
  tokens, then needle prompts of 2.7K / 8K / 16K / 29K tokens, through `serve/server.py`. Raw results in `data/`.
- **Engines:** 0.1.24 (`3ce2523`) as the baseline; branch `multi-gpu-split` for the rest.

## What was wrong with the split on this PC

1. **Only 7 of the 33 GiB expert arena was pinned.** The 8 GiB cap for a split is a WDDM workaround, but it applied on
   Linux too. The PCIe share of the missed experts then covered only the first ~10 layers, and the prompt path staged
   ~80% of the streamed experts through host copies. One GPU pinned all of it.
2. **Every card reserved its own 2048-token prompt buffers** (~1.5 GiB a card for the whole session) where one card
   borrows cache slots, and prompts ran in 2048-token chunks instead of 6144.
3. **The second card's reserve counted the drafter twice** (1 GiB): 1.3 GiB of CUDA1 stayed free all session.

## Speed

| Run | Decode tok/s | ms / window | Decode hits | Prompt 2.7K / 8K / 16K / 29K tok/s |
|---|---:|---:|---:|---|
| 0.1.24, one GPU | 33.9 | 72.4 | 63.7% | 618 / 712 / 795 / 801 |
| 0.1.24, two GPUs (auto, K=31) | 35.8 | 67.6 | 68.8% | 391 / 497 / 537 / 535 |
| branch, two GPUs (auto, K=25), run 1 | 47.3 | 51.6 | 82.4% | 609 / 826 / 1,126 / 1,290 |
| branch, two GPUs (auto, K=25), run 2 | 45.3 | 54.7 | 80.1% | 627 / 829 / 1,120 / 1,290 |

Run 1 kept an extra 256 MiB free on the later card, run 2 128 MiB; the branch computes it as 64 MiB of slack plus what
the drafter's bind adds (53 MiB for the 40,525-token draft subset used here). Between identical runs a
request's decode rate moves by up to ~10% (the code answer's text, and with it the routing, differs slightly).

Where a decode window's host time went (`STRATA_SPLIT_TIMING`, cumulative over the run):

| Run | Wait for the GPUs | CPU pool + plan | Staging + commit |
|---|---:|---:|---:|
| 0.1.24, one GPU | 19.5 ms | 52.2 ms | 1.6 ms |
| 0.1.24, two GPUs | 19.3 ms | 44.1 ms | 1.6 ms |
| branch, two GPUs | 20.5 ms | 31.6 ms | 1.6 ms |

Decode on this PC is bound by the CPU pool: the second card pays through the experts its cache holds and through the
PCIe share, not through GPU time.

**VRAM after the runs** (reserve 700 MiB): 0.1.24 left CUDA0 988 MiB and CUDA1 1,968 MiB free; the branch leaves 814
and 876 MiB.

### After the rebase onto 0.1.27

The same harness with the branch rebased onto 0.1.27, and 0.1.27's draft subset (106,299 tokens: the drafter's head
is 138 MiB instead of 53, and the second card's reserve now follows it). The needle prompts are shorter at the low
end because the repository's files they are built from changed.

| Run | Decode tok/s | ms / window | Decode hits | Prompt 1.9K / 8K / 16K / 29K tok/s |
|---|---:|---:|---:|---|
| one GPU | 35.7 | 69.2 | 64.3% | 457 / 747 / 834 / 832 |
| two GPUs (auto, K=25) | 47.3 | 53.4 | 79.4% | 459 / 825 / 1,140 / 1,304 |

The split: +33% decode, prompts level at 1.9K and +10% / +37% / +57% at 8K / 16K / 29K. VRAM after the runs: CUDA0
808 MiB free, CUDA1 878 MiB (reserve 700).

## K sweep (branch; one decode round, 16K prompt)

| K | Decode tok/s | Hits | Prompt 16K tok/s |
|---|---:|---:|---:|
| 18 | 41.4 | 81.8% | 962 |
| 25 (auto) | 45.3-47.3 | 80-82% | 1,120-1,126 |
| 31 | 45.2 | 81.0% | 1,026 |
| 36 | 42.2 | 79.6% | 929 |

Auto's pick is the best measured. From the real profile, the planner picks K=25 for any routing exponent from 0.8
to 1.2: the choice follows the caches' sizes.

## Correctness

| Check | Result |
|---|---|
| Needles at 2.7K-29K tokens, every run | all found |
| Two GPUs, fixed cache (`--adapt-every 0 --pcie-frac 0 --suffix-draft 0`): a story and a code answer (200 tokens each) before and after a 16K prompt that lent 2,687 slots on CUDA0 and 2,599 on CUDA1 and refilled them | byte-identical; needle found |
| The same after the rebase onto 0.1.27, two GPUs and one GPU (2,640 slots lent) | byte-identical; needle found |
| One GPU and 2,000 experts on the other card (`--expert-cache-device1`, the whole arena pinned) | starts, answers coherently, 8K needle found |

A slot lent without being marked non-resident would have fed the prompt path's buffers to the experts (the needle
lost); a slot refilled on the wrong card or with the wrong expert would have changed the answers after the prompt.

## Reproduce

From the repository root (the harness runs from any folder; results go to `runs/NAME/` in the folder you run it from):

```sh
H=bench/results/2026-09-30-layer-split-2080ti/split_bench.py
python $H one --gpu 0 --exe <0.1.24 engine binary>
python $H split --gpu 0,1 --exe build/strata
python $H k31 --gpu 0,1 --layer-split 31 --decode-reps 1 --lengths 16k
python $H lend --gpu 0,1 --lend-check
```

`--config` defaults to `strata-iq2_xs.json` at the repository root; `--exe` defaults to the config's engine.
