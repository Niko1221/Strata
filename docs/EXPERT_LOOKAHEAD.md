# Experts in VRAM ahead of time: how much faster could decode be?

Status: measurement only, 2026-10-19. Nothing in Strata changed. No feature, no default change, no speed claim for
Strata as it ships.

## In short

Most of the time Strata spends on a token goes to the CPU computing experts that are not in the graphics card's
memory. We asked: if the right experts were already in VRAM when they are needed, how much faster would it be?

- We tried it on an RTX 5070 with Swift 1.5 IQ2_XS. We filled the VRAM expert cache with exactly the experts the
  next answer would use. Decoding ran **3.0× faster for prose (12.7 → 38.1 tok/s) and 4.1× faster for code
  (10.4 → 42.3 tok/s)**. The answers were the same token for token.
- To get there for real, something has to guess which experts the next 16 to 64 tokens will use and copy them in
  the background. Copying is not the problem: a perfect guesser would copy 39 to 84 experts per token, 2 to 4 ms over
  PCIe, well inside the time a token takes.
- Whether a real guesser can be good enough is not known. These numbers are the ceiling, not a result.

This came out of the NCM study (pull request #3247): NCM predicts a rough picture of a whole block of future text,
and the set of experts a block uses is that kind of picture. No NCM model was used here.

## Machine and settings

NVIDIA GeForce RTX 5070 (12,227 MiB), AMD Ryzen 9 7900X, 61 GiB RAM, Linux 7.0.0-38, driver 610.43.02. Model: Swift
1.5 IQ2_XS (a derivative of Qwen3.8-Flash-Next), 48 layers, 512 experts per layer, 10 routed per token. Existing
engine binary 0.1.45 (SHA-256 in `engine.sha256`), checkout e88ee39d.

All engine runs use the saved run config with diagnostic overrides: context 4,096, KV residency 0,
`--mmap-experts`, greedy, and the greedy reference mode: no MTP, `--spec 2 --spec-corrupt 1 --suffix-draft 0`. The
native IQ pack refuses `--spec 0`, so this mode checks one dummy draft per step. It is **not** a plain-decode rate and
not the saved 64K configuration. VRAM expert cache: 4,866 slots (6.53 GiB), pre-filled from
`data/expert-profile.bin`.

Raw files: [`benchmarks/expert-lookahead-2026-10-19-rtx5070/`](benchmarks/expert-lookahead-2026-10-19-rtx5070/).
Tool: [`tools/expert_lookahead_bench.py`](../tools/expert_lookahead_bench.py).

## 1. Where a token's time goes

64 greedy tokens of the prose prompt, `--stats` (`breakdown/plainstats.stdout`):

| Part | ms per token |
| --- | ---: |
| Total, wall clock | 59.98 |
| CPU expert pool (experts not in VRAM) | 46.94 |
| Everything else (GPU layers, waiting for the GPU, host) | about 13 |

The VRAM cache served 42,470 of 60,960 routed experts (69.7%). The other 30.3% were computed on the CPU, and that is
78% of the time.

## 2. The oracle run: the right experts already in VRAM

`tools/expert_lookahead_bench.py oracle-profile` writes a profile whose first entries are exactly the experts the
continuation routes to, taken from that prompt's own routing trace (`inputs/*.oracle8.routing.bin`), followed by the
normal profile's ranking. The engine pre-fills its cache from the top of a profile, so these experts are in VRAM
before decoding starts. `--adapt-min-gain 1e9` stops the adaptive tier from swapping them out.

Only as many tokens as fit in 4,866 slots were run: 32 for prose (4,279 experts), 24 for code (4,708). Three arms per
prompt, each run twice in the order normal, control, oracle, then oracle, control, normal:

| Prompt | Arm | tok/s (two runs) | ms per token | CPU expert pool, ms/token | VRAM hits |
| --- | --- | --- | ---: | ---: | ---: |
| Prose, 32 tokens | normal profile, adaptive tier on | 12.68, 12.69 | 78.81 | 66.2 | 58.4% |
| Prose, 32 tokens | normal profile, adaptive tier off (control) | 12.73, 12.57 | 79.06 | 66.5 | 58.4% |
| Prose, 32 tokens | **oracle profile**, adaptive tier off | **38.21, 37.88** | **26.29** | 14.1 | 88.5% |
| Code, 24 tokens | normal profile, adaptive tier on | 10.34, 10.48 | 96.05 | 82.9 | 46.8% |
| Code, 24 tokens | normal profile, adaptive tier off (control) | 10.46, 10.37 | 96.00 | 82.8 | 46.8% |
| Code, 24 tokens | **oracle profile**, adaptive tier off | **41.67, 43.02** | **23.62** | 11.4 | 88.4% |

Oracle against normal: **3.00× for prose, 4.07× for code**. All 12 runs' output IDs equal the prompt's greedy
reference. Control equals normal, so the gain comes from which experts are in VRAM, not from turning the adaptive
tier off.

Why hits stop at 88%: the reference mode checks one dummy draft token per step, and the dummy routes to experts that
are not in the trace, so not in the oracle profile. Those misses cost the 11 to 14 ms of CPU work left. Without them
about 12 ms per token would remain, roughly 80 tok/s. That last figure is an estimate, not a measurement.

## 3. What a real prefetcher would have to copy

A prefetcher has to copy experts into VRAM during decoding, not before it. `simulate` replays the same two traces
(the last 64 generated tokens) against the 4,866-slot cache with perfect knowledge of the next L tokens and the best
possible eviction (Belady). It counts the experts that must be copied per token (`simulate.txt`; after the first
8 tokens, so the cold start is left out):

| Looks ahead | Prose, copies per token | Code, copies per token | PCIe time per token at 28.4 GB/s |
| ---: | ---: | ---: | ---: |
| 1 (copy on demand) | 118.1 | 173.8 | 5.82 / 8.57 ms |
| 16 | 59.1 | 96.8 | 2.91 / 4.77 ms |
| 32 | 44.8 | 84.4 | 2.21 / 4.16 ms |
| 64 | 39.3 | 79.5 | 1.94 / 3.92 ms |

PCIe speed: `tools/h2d_bench.cu` measured 28.4 GB/s host to device from pinned memory, one 1.4 MB expert blob in
0.075 ms (`pcie-h2d_bench.txt`). From 16 tokens ahead the copying is a fraction of the ~13 ms of other work per token,
so it can overlap. At this rate PCIe has room for about 265 copies per token, so a prefetcher may copy several times
more experts than needed. Missing a needed one is what costs time.

Limits: experts mapped from the GGUF file are not pinned memory and would need a staging copy (the engine read them at
13.8 GB/s per reading thread); a simulation over 64 tokens of two prompts.

## 4. Distinct experts in a window of T tokens

From the same traces, consecutive generated tokens grouped into windows (`unions.txt`). Routing per token does not
depend on the window, so grouping afterwards is valid. The 4- and 8-token values equal the speculative pilot's
traces.

| Tokens | Prose, distinct experts per layer | Code |
| ---: | ---: | ---: |
| 1 | 10.0 | 10.0 |
| 4 | 25.4 | 27.7 |
| 8 | 40.8 | 45.0 |
| 16 | 61.6 | 70.0 |
| 32 | 84.7 | 106.3 |
| 64 | 109.6 | 157.7 |

## 5. The cost of checking T tokens at once

A verify window is limited to 8 tokens (`kVerifyMaxT`). To see what larger windows would cost, a 1,023-token prompt
(the start of `docs/HOW_IT_WORKS.md`) was read with the prompt path in chunks of T tokens, `--max-new 1`, each size
twice in the order 4, 16, 32, 64, 64, 32, 16, 8, 4 (`check-cost/prefill_results.json`):

| T | ms per chunk (two runs) | tokens/s if every token counts |
| ---: | --- | ---: |
| 4 | 163.9, 160.8 | 24.6 |
| 8 | 221.8 | 36.1 |
| 16 | 320.7, 320.8 | 49.9 |
| 32 | 465.9, 463.9 | 68.8 |
| 64 | 648.2, 648.9 | 98.7 |

A first, cold run at T = 8 took 263.1 ms per chunk (`check-cost/prefill8.cold-trial.*`) and is left out of the table.
The prompt path is a different code path from the verifier: it copies experts to the GPU instead of computing misses
on the CPU. At T = 8 it took 0.95 to 1.19 times the verifier's measured round time (186.3 ms prose, 232.5 ms code in the
NCM pilot). Treat larger T as an estimate of what a verifier of that size would cost.

## What this does and does not show

It shows that on this machine decode time is mostly the CPU computing missed experts, and that removing those misses
gave 3.0 to 4.1 times the speed with the same output, in short runs.

It does not show:

- that any real predictor can find the right experts in time. Each missed share costs: about 17 ms per token per 10%
  of routed experts missed, from the normal and oracle runs above (1.74 ms per point for both prompts; an estimate);
- the speed over long answers. Normal runs of 24 to 32 tokens were slower (79 to 96 ms per token) than the 64-token
  run (60 ms) because the cache had not settled, so the gain over long answers is likely smaller;
- the cost of copying during decoding (the oracle experts were loaded at start-up, 1.2 s, outside the timing);
- other machines, the saved 64K configuration, MTP or prompt lookup on top, HIP, SYCL or Metal.

## How to repeat

The first three need no GPU and use the committed traces:

```sh
B=docs/benchmarks/expert-lookahead-2026-10-19-rtx5070
python3 tools/expert_lookahead_bench.py unions $B/inputs/prose.oracle8.routing.bin $B/inputs/code.oracle8.routing.bin
python3 tools/expert_lookahead_bench.py simulate $B/inputs/prose.oracle8.routing.bin $B/inputs/code.oracle8.routing.bin
python3 tools/expert_lookahead_bench.py oracle-profile $B/inputs/prose.oracle8.routing.bin --tokens 32 --out /tmp/oracle-prose.bin
python3 tools/expert_lookahead_bench.py oracle-profile $B/inputs/code.oracle8.routing.bin --tokens 24 --out /tmp/oracle-code.bin
```

The two profiles should match `oracle/oracle-profiles.sha256`. With the model installed and the GPU free:

```sh
python3 tools/expert_lookahead_bench.py oracle-runs --out /tmp/el-oracle \
  --prompt prose=$B/inputs/prose.prompt.ids:$B/inputs/prose.oracle.ids:/tmp/oracle-prose.bin:32 \
  --prompt code=$B/inputs/code.prompt.ids:$B/inputs/code.oracle.ids:/tmp/oracle-code.bin:24
python3 tools/expert_lookahead_bench.py prefill-sweep --pack "$DATA/packs/swift-iq2_xs" --out /tmp/el-prefill
nvcc -O2 -o /tmp/h2d_bench tools/h2d_bench.cu && /tmp/h2d_bench 0
```

The recorded runs were made from scratch scripts with fixed paths. The tool builds the same argument lists (checked
for all 12 oracle runs and the chunk sweep), writes byte-identical oracle profiles and prints the same unions and
simulation numbers. Paths in the recorded files are rewritten: `$STRATA` is the checkout, `$DATA` the Strata data
folder, `$PILOT` the folder of the speculative pilot's inputs (now `inputs/`), `$OUT` the scratch output folder.
