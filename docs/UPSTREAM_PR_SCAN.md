# Upstream PR scan: what would make Strata faster on an 8 GB laptop

**Status:** screening report, not a change. No engine code touched.
**Scanned:** all 754 open pull requests on `Niko1221/Strata` on 2026-10-17 (search API), verified against
`upstream/main` at `296b59d2`.
**Corrected:** same day, after re-checking the tree directly: two PRs were wrongly listed as already in `main`,
and #1358's merge state was wrong. See [Corrections to this scan](#corrections-to-this-scan). The scan's own
numbers are unchanged.
**Target host:** the PC in [Q36_MEMORY_COMPACTION.md](Q36_MEMORY_COMPACTION.md) - RTX 4060 Max-Q Laptop
(8188 MiB), Ryzen 7 7735HS, 62 GB, Ubuntu, no swap, NVMe.
**Model/path:** `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, `--kv int8 --mtp 2`, 32K context, `--family qwen36`.
The experts do not fit in 8 GB either way, so the engine runs **0 expert slots** and decode is the CPU pool
([Q36_INT8_KV.md](Q36_INT8_KV.md)).

This is a research answer to "of the 754 open PRs, does any make this machine faster?". It ranks the ones
that are both **not already in `main`** and **applicable to this hardware path**. Every number below is the
one the PR itself reports, on the hardware the PR names - none of it was re-measured on this PC.

## Summary

- The 754 are mostly **not** candidates for this box: other vendors (AMD/HIP, Intel/SYCL, Metal), other
  NVIDIA generations (sm_60/61/70/75/86 work), benchmark reports with no code change, Windows/WSL-only
  behaviour, multi-GPU/layer-split features, and duplicates of each other.
- **A visible share of the open PRs are already merged into `main`** after the history rewrite and would be
  no-ops if applied. They look like top hits from their own text. Section
  [Already in main](#already-in-main-do-not-apply) lists the ones checked.
- On the `qwen36` path this PC actually runs, **almost nothing upstream helps beyond the two local PRs**
  (#2742 int8 K/V + single staging, #2823 mapping release). The path is RAM-bandwidth-bound at one token
  (the Q4_K/Q5_K rows are at the RAM bound on all three CPU kernels - see the Q5_K note in `UNSLOTH_Q4.md`
  and commit `21bf82c1`), so CPU micro-optimisations buy little.
- The main-engine (Flash-Next) path has more to offer, because a 64 GB box cannot hold its 71.7 GiB of
  experts and it therefore streams. The strongest single item is the **prefill VRAM loan** (#1581).

## How this was screened

Five themed passes over the PR bodies (expert placement/cache, CPU/host, K/V and VRAM, CUDA kernels and
scheduling, file/disk IO and tooling), then the shortlist was re-checked against `upstream/main` by looking
for each PR's flag or algorithm in the current tree:

```
git fetch upstream
git grep -n '<FLAG_OR_SYMBOL>' upstream/main
git apply --check /tmp/prdiff_<n>.patch      # in a worktree at upstream/main
```

A flag that `git grep` finds in `upstream/main` means the feature landed by another route and the PR is
stale. This caught several otherwise-plausible picks.

## Recommendations - new and applicable

Ranked by expected effect on this box. "Main engine" means `src/program/generate.cpp` + `expert_source` /
`verify` / `prefill`; "q36" means the separate driver in `src/qwen35/q36_engine.cu`, which deliberately
keeps `generate/verify/prefill` untouched.

| PR | Applies to | What it changes | Reported measurement (hardware from the PR) | State |
|---|---|---|---|---|
| [#1581](https://github.com/Niko1221/Strata/pull/1581) CUDA prefill VRAM loans | main, CUDA+VMM, single GPU | Lends decode-only VRAM (verifier, native head, MTP) to the expert cache for a prefill, then restores it; optional repeat for later long messages | RTX 2080 Super Max-Q **8 GB**, 31,258-token prompt, IQ3_XXS: PP 422.5 -> 865.2 t/s (Q4_0, +104.8%), 435.8 -> 854.7 (INT8, +96.1%, TG -7.4%). The ~2x is the **whole stack including SM75-only kernels**; the PR assigns the loan no isolated %. Loan separately qualified on an RTX 4070 Ti (Ada/sm_89) | open, CONFLICTING; opt-in |
| [#1838](https://github.com/Niko1221/Strata/pull/1838) `STRATA_RAM_ADAPT` | main | The RAM-held expert copy follows the conversation's routing instead of a fixed profile | RTX 3060 12 GB / 32 GB RAM, IQ2_XS, 47 replayed requests: decode 43.26 -> 50.40 t/s (+16.5%); file-tier blob reads per token 14.15 -> 2.67 (-77.7%) | open, CONFLICTING; opt-in |
| [#1949](https://github.com/Niko1221/Strata/pull/1949) AVX2 dot accumulators in registers | main (CPU), **not** the q36 Q4_K/Q5_K pack | Bug fix, no flag: the router `bf16_rows_dot_multi` and the `q5_1`/`q8_0` expert dots held sums in a runtime-indexed `__m256 acc[]` that GCC and clang spill; four named accumulators instead | Ryzen 7 5700X3D (Zen 3): router 0.400 -> 0.185 ms (**2.1x**); Q5_1 down 190 -> 155 us, Q8_0 down 135 -> 104 us at 4 tokens. Parity 0 failures, hashes identical. `q4k_dot` is deliberately left alone | open, APPLIES; not opt-in |
| [#2775](https://github.com/Niko1221/Strata/pull/2775) per-client expert memory | main | Keys routing counts on the client's first 64 prompt tokens, so switching programs no longer flushes the VRAM cache | RTX 5070 Ti, Q2_0, 61-75K: agent step after another program +21% tok/s; already-adapted +11% | open, MERGEABLE; opt-in |
| [#1262](https://github.com/Niko1221/Strata/pull/1262) `STRATA_HC_REQ8` | main (HC projections) | Requantizes the four hyper-connection projections to int8 + fp32 scale per 32 at load | RTX 2060 SUPER **8 GB**: dense load 1,416 -> 891 MiB, expert cache 2,044 -> 2,441 slots (+397) | open, CONFLICTING; opt-in |
| [#1989](https://github.com/Niko1221/Strata/pull/1989) `STRATA_HC_HOST` | main (HC), CUDA | Moves the 1.17 GiB hyper-connection down/up block to pinned mapped host memory | RTX 5070 Ti Laptop 12 GB: slots 2,494 -> 3,221 (+29%), CPU expert time/window 13.4/15.4 -> 11.5/12.8 ms (-15%); agent decode 11.58 -> 11.47 ms/token (-0.9%, in noise) | open, CONFLICTING; needs `STRATA_HC_Q8=1` |
| [#1181](https://github.com/Niko1221/Strata/pull/1181) O(n) verify-window plan | main | Rewrites the window plan's quadratic scans as three linear passes; output-identical | 5,103 -> 1,576 ns/window-layer = **245 -> 76 us of GPU spin-wait per 48-layer window**; 4,020-case differential test | open, CONFLICTING; default (not a flag) |

### Main-engine CUDA decode/prefill, second tier

Small and cumulative; all opt-in unless stated. Measured on Ada-or-newer or every CUDA card unless noted.

- [#2058](https://github.com/Niko1221/Strata/pull/2058) `STRATA_EXPDEC` (verify-window gate/up, q8_1 rows in
  shared memory, bit-exact) and [#2059](https://github.com/Niko1221/Strata/pull/2059) `STRATA_QSA_ROWS`
  (block scores, bit-exact) stack: gate/up 0.73-0.84x; +2.9% decode together (2080 Ti/3060, 131K).
- [#1888](https://github.com/Niko1221/Strata/pull/1888) `STRATA_QSA_SPLIT_ATTN` (draft, sm_80+) **or**
  [#2075](https://github.com/Niko1221/Strata/pull/2075) `STRATA_ATTN_DAILY` (bit-exact, every CUDA card):
  decode attention. #1888 +4.2% decode / +13.1% at 120K (4060 Ti + 5080); #2075 2.15x on the attention
  kernel (3060, 131K). Pick one - they address the same cost.
- [#1367](https://github.com/Niko1221/Strata/pull/1367) `STRATA_PROMPT_ATTN_IMMA` (sm_80+, explicitly
  compiled for sm_89): int8-K/V prompt attention on INT8 tensor cores, 1.35-1.53x per launch, closer to
  FP32 than v2 (RTX 5090).
- [#1525](https://github.com/Niko1221/Strata/pull/1525) prefill dense fusions, same bits: -4.0% on a 32K
  prompt (RTX 5090). On by default.
- [#2234](https://github.com/Niko1221/Strata/pull/2234) `STRATA_BATCH_ATTN_ROWS` / `STRATA_BATCH_GDN_GROUPS`:
  one launch for all rows - only matters with `--batch` (RTX PRO 6000: c=4 268 -> 312 tok/s).
- [#2318](https://github.com/Niko1221/Strata/pull/2318) `--kv q2k`: K 2-bit / V 4-bit, 448 B/cell vs q4_0's
  576 (-22%). EXPERIMENTAL and quality-costly (the PR's own text: prompt KL 3.8-5.3x q4_0's); on this path
  the local int8 K/V already captures the bigger win.

## The `qwen36` path (what this PC runs)

Almost nothing here. The q36 driver computes the router with `f32mm` on the GPU and has its own K/V and
expert pool, so the main-engine routing and VRAM PRs above do not reach it.

- [#2698](https://github.com/Niko1221/Strata/pull/2698) is the **same int8-K/V idea as local #2742** (author
  measured RTX 3060 12 GB: 128K decode 33.9 -> 54.8 tok/s, experts in VRAM 0 -> 31/256). Competing
  duplicate of work already on this branch; do not take both.
- [#2751](https://github.com/Niko1221/Strata/pull/2751) `STRATA_POOL_FUSED` touches the **shared** CPU pool
  (`src/kernels/cpu/pool.cpp`) that q36 uses: -8% to -11% per layer on a Ryzen 7 3800X, but measured on
  compute-bound IQ gate/up rows. On this pack the Q4_K/Q5_K rows are RAM-bound at one token, so expect less.
  Cheap to A/B.
- [#2515](https://github.com/Niko1221/Strata/pull/2515): `setup.py` pins `-DSTRATA_Q6K_EXPERTS=ON` and
  `-DSTRATA_MMQ_KQUANTS=ON`, which fixes the exact load failure this model produces when
  `build/CMakeCache.txt` predates the option's default flip:
  `blk.34.experts are Q4_K/Q6_K, which the native expert kernels do not take`.
- [#2653](https://github.com/Niko1221/Strata/pull/2653): per-PC calibration of `STRATA_PREFILL_CPU_SHARE_MAX`
  (the share only arms below 1,024 tokens by default). Modest; prefill here is GPU-side.
- [#2321](https://github.com/Niko1221/Strata/pull/2321) / [#1820](https://github.com/Niko1221/Strata/pull/1820)
  route-resident 256/8 (+9.7% decode on Qwen3.6-35B-A3B in the PR) live in the **main engine's**
  `src/core/verify.cpp`. The q36 driver does not include that path; skip unless it is ported.

## Practical order

1. **Measure first.** [#2660](https://github.com/Niko1221/Strata/pull/2660) `tools/strata_profile.py` says
   where the time goes; [#2423](https://github.com/Niko1221/Strata/pull/2423) adds per-request PCIe expert
   bytes. Note #2660's parsers target main-engine log lines and may need extending for `strata-q36:`.
2. **Ubuntu prerequisite.** If `nvcc` is older than 12.5 (the distro `nvidia-cuda-toolkit` 12.0 is accepted
   for sm_89), take [#1358](https://github.com/Niko1221/Strata/pull/1358) first or no VMM feature
   (#1581, #1117, #2399) compiles.
3. **Then A/B in this order:** #1581 (prefill) -> #1838 (RAM tier) -> #1949 (free CPU win on the main
   engine) -> #1262 / #1989 (VRAM back to experts) -> the decode-kernel set.
4. Most are **CONFLICTING** against current `main` (#1581, #1838, #1262, #1989, #1181, ...); **#1949**,
   **#2751**, **#2515**, **#2660**, **#2698** and **#2775** are mergeable today. #1358 is not (corrected below);
   it is 6 lines in one file, so it can be redone by hand.
5. Follow `AGENTS.md`: one change at a time, against a same-day build, with the numbers recorded. The
   headline #1581 number bundles SM75 kernels this card cannot use.

## Implementation plan

Recorded 2026-10-17, for the 8 GB laptop in [Q36_MEMORY_COMPACTION.md](Q36_MEMORY_COMPACTION.md) on branch
`docs/upstream-pr-scan-8gb`. Nothing here has been built or measured on this PC: every expected effect is the
PR author's number on the PR author's hardware, and is marked as such. This plan revises
[Practical order](#practical-order) in one way - the cheap, mergeable, bit-exact items go first, because they
cost little if they turn out to be flat, and the biggest headline item (#1581) carries the most caveats.

**Blocker found while writing this plan:** the host this plan was written on has no `nvcc` and no `nvidia-smi`
in the shell, and no step can be built until that is settled. That is step 0.1.

### The one command

Every phase is measured through the served API (the path this PC actually runs) and read off one number: the
decode rate the server reports as `timings.predicted_per_second` (`pps`). It needs the model and a built image
- no other tool. The `--bench-pp 32768 --bench-tg 32` bench is not the baseline: at a full 32K prompt it aborts
with `cudaMalloc(&E.stage[b], ...) failed: out of memory` unless `STRATA_Q36_SINGLE_STAGE=1` is set, and then
hits `an illegal memory access` in the hybrid CPU-expert upload (`q36_engine.cu:1878`). Serve mode has neither
failure.

```bash
MODEL=/home/george/apps/llamacpp/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

# start the server (the phase image, the 32K ceiling, the user's switches)
IMAGE=strata:pr-ph0 STRATA_Q36_SINGLE_STAGE=1 CONTEXT=32768 \
  ENGINE_ARGS_EXTRA="--kv int8 --mtp 2" API_KEY=secret \
  MODEL_FILE="$MODEL" docker/run.sh serve

# one measurement, in another shell (build-q5k/ab_client.py sends the same request):
python3 build-q5k/ab_client.py http://127.0.0.1:11434 secret "<PROMPT>" 192
```

The number is the response's `timings.predicted_per_second` (decode only; prefill time is excluded), read
alongside `timings.prompt_n` (the context the decode actually ran at) and `timings.predicted_n`. Decode rate
falls as the context grows, so a number without its `prompt_n` is not comparable - see the baseline below.
`prompt_n` is the real context, not the `CONTEXT=32768` ceiling. `--mtp 2` is the draft count. A phase's number
is a win only if it beats `strata:pr-ph0` at the same `prompt_n`; the only things that change phase to phase are
the image tag and, for the opt-in PRs, one switch.

### Phase 0 - prerequisites (no engine change)

| Step | Do | Done when |
|---|---|---|
| 0.1 | Confirm the toolkit: `nvcc --version` | `nvcc` >= 12.5, or 12.0 for sm_89 from the distro `nvidia-cuda-toolkit`. Below that, #1358 first |
| 0.2 | Build the branch and tag the image | `IMAGE_TAG=strata:pr-ph0 docker/build.sh` returns |
| 0.3 | Merge [#2660](https://github.com/Niko1221/Strata/pull/2660) `strata_profile.py` and [#2515](https://github.com/Niko1221/Strata/pull/2515) setup pins | merged; neither changes a tok/s |
| 0.4 | Record the baseline | the serve-mode decode `pps` by `prompt_n`, in [Baseline](#baseline-measured-2026-10-17) below |

Phase 0 has no expected speed-up; step 0.4 is the number every later phase is compared against. #2660 and
#2515 are merged here without testing: #2660 only says where the time goes, and #2515 only makes the pack
load.

### Baseline (measured 2026-10-17)

The Phase-0 number, measured on `strata:pr-ph0` (built from this branch after the #2660/#2515 merges) on the
target PC (RTX 4060 Max-Q 8 GiB, Ryzen 7 7735HS, 62 GB), `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`,
`--kv int8 --mtp 2`, `STRATA_Q36_SINGLE_STAGE=1`, `CONTEXT=32768`, `max_tokens=192`, through the served
API. Decode rate (`pps`) vs the context the decode actually ran at (`prompt_n`):

| `prompt_n` (context) | decode `pps` (tok/s) |
|---|---|
| 4,106 | 17.6 |
| 16,391 | 12.3 |
| 24,581 | 10.4 |
| 32,561 | 8.1 |

Decode falls smoothly as the history grows. The "~10 tok/s" the memory-compaction plan called "the user's 32K
run" is decode at **~24-25K context**, not a full 32K - `CONTEXT=32768` was the ceiling, not where the rate was
measured. A later phase compares against the same `prompt_n`, not the same ceiling. The 32K point drifts
8.1-8.8 tok/s across repeats (sustained load, thermal); the shorter-context points are stable. The short-context
sanity check is 18.9 tok/s at 32 tokens, matching [Q36_INT8_KV.md](Q36_INT8_KV.md)'s 18.3-19.6.

### Phase 1 - cheap, safe, unblocking (all mergeable today)

| Step | PR | Why this box | Measure | Expected |
|---|---|---|---|---|
| 1.1 | [#2515](https://github.com/Niko1221/Strata/pull/2515) setup pins | this pack is UD-Q4_K_XL, and the PR fixes exactly the failure it produces: `blk.34.experts are Q4_K/Q6_K, which the native expert kernels do not take` | merge (done in Phase 0) | correctness: the pack must load |
| 1.2 | [#1949](https://github.com/Niko1221/Strata/pull/1949) AVX2 accumulators | bit-exact bug fix, no flag, no default changed | merge; the q36 run is expected flat | 2.1x on the router and 190 -> 155 us Q5_1 (Zen 3, author's). Our q36 Q4_K/Q5_K pack is **not** covered |
| 1.3 | [#2751](https://github.com/Niko1221/Strata/pull/2751) `STRATA_POOL_FUSED` | the only perf PR here that touches the shared CPU pool q36 uses | the one command, switch off then on | -8..-11% per layer (Ryzen 7 3800X, author's), compute-bound IQ rows. Our rows are RAM-bound, so expect less |

All three are settled (2026-10-17, `strata:pr-ph1` on this box, `--kv int8 --mtp 2`, 32K, `max_tokens=192`):

- **1.1 (#2515) merged.** Done in Phase 0; the pins make the Q4_K/Q6_K pack load. No tok/s to read.
- **1.2 (#1949) merged.** Bit-exact (the smoke prompt's output hash is unchanged), no flag, no default change. It
  leaves `q4k_dot` (the q36 pack's expert dot) alone, so the q36 run is flat by construction; it is a 2.1x fix to
  the main engine's CPU router and its q5_1/q8_0 expert dots.
- **1.3 (#2751) rejected.** Measured A/B, `STRATA_POOL_FUSED` off vs on: decode 10.6 -> 10.7 tok/s at 24,581
  tokens and 8.8 -> 8.6 tok/s at 32,561 (both inside the run-to-run thermal noise; the output hash is identical).
  The Q4_K rows are RAM-bound, so the fusion's overlap of compute-bound gate/up with RAM-bound down has nothing to
  hide - the PR's own -8..-11% was on compute-bound IQ rows. Flat here, so it does not go in.

### Phase 2 - VRAM and prefill (largest effect, most caveats)

| Step | PR | Caveat that decides the outcome | Expected |
|---|---|---|---|
| 2.1 | [#1581](https://github.com/Niko1221/Strata/pull/1581) prefill VRAM loan | CONFLICTING and opt-in; needs 0.1. The PR's headline bundles SM75-only kernels this sm_89 card cannot use, and the loan itself has no isolated measurement. Watch for the TG regression the PR saw | PP 422.5 -> 865.2 t/s (2080 Super Max-Q 8 GB, author's) is an **upper bound**, not an expectation. TG -7.4% (INT8) |
| 2.2 | [#2775](https://github.com/Niko1221/Strata/pull/2775) per-client expert memory | Mergeable, opt-in. Only pays off when more than one program shares the server | +21% tok/s agent step after a program switch (RTX 5070 Ti, author's); +11% already-adapted |

2.1 is the one prefill number. This report records no switch name for #1581, so read it from the PR before
running; the same build runs both arms (switch off, then on). 2.2 is server behaviour and is flat in the
bench run, so it is merged without a tok/s to read.

```bash
MODEL=/home/george/apps/llamacpp/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
LOAN_ENV=${LOAN_ENV:?set this to the switch the PR documents}

# 2.1 (#1581): read pp (prefill), not tg; watch tg for the -7.4% the PR saw
IMAGE=strata:pr1581 MODEL_FILE="$MODEL" docker/run.sh engine \
  --bench-pp 32768 --bench-tg 32 --kv int8 --mtp 2
env "$LOAN_ENV=1" IMAGE=strata:pr1581 MODEL_FILE="$MODEL" docker/run.sh engine \
  --bench-pp 32768 --bench-tg 32 --kv int8 --mtp 2
```

- **2.1 PASS:** the `pp` with the switch on is higher than with it off, and `tg` did not fall by more than the
  PR's own -7.4%. Quote 422.5 -> 865.2 only as the author's number on a 2080 Super Max-Q.
- **2.2:** merged; no bench number (a single-program run cannot show it, and this box's bench is one).

### Phase 3 - only if we run the Flash-Next main engine

The q36 driver has its own router, K/V and expert pool, so none of these reach it. On a 62 GB box the main
engine streams its 71.7 GiB of experts, which is where they apply. One at a time, in this order: #1838
(`STRATA_RAM_ADAPT`), then #1262 / #1989 (VRAM back to experts), then #1181 (O(n) plan, output-identical, on
by default), then the second-tier decode kernels. [#1888](https://github.com/Niko1221/Strata/pull/1888) and
[#2075](https://github.com/Niko1221/Strata/pull/2075) address the same cost - take one, never both.

These do not run on our Qwen3.6 pack, so there is no tok/s to read here until a Flash-Next pack is present.
The measurement stays the same shape - build the image, run the engine once, read the decode number - but the
main engine takes a `--pack`, so copy the run from a recorded Flash-Next config rather than the q36 line
above. Until that pack exists, the phase is recorded as untested, not skipped.

### Phase 4 - re-screen the two mis-parked PRs

[#1614](https://github.com/Niko1221/Strata/pull/1614) `--kv-pool-tokens` and
[#1548](https://github.com/Niko1221/Strata/pull/1548) / [#2224](https://github.com/Niko1221/Strata/pull/2224)
`STRATA_PCIE_BALANCE` are not in the tree and were parked in error (see
[Corrections](#corrections-to-this-scan)). They have not had the screen the rest of this report got, and no
value claim is made for either. This is a reading step, not a run: for each, write one row in the report's
table format - what it changes, which files, opt-in or default, whether it reaches the q36 driver. Only after
that does either get a tok/s to read.

### What this plan does not do

- It does not re-measure any PR number on this PC. Phase 0.4 is the first number that will be ours.
- It does not plan [#2698](https://github.com/Niko1221/Strata/pull/2698) (the same int8 K/V idea as the local
  #2742 already on this branch), [#2318](https://github.com/Niko1221/Strata/pull/2318) `--kv q2k` (the local
  int8 K/V already captures the bigger win, and the PR's own text reports prompt KL 3.8-5.3x q4_0's), or
  #2321 / #1820 (main-engine `verify.cpp`; the q36 driver does not include that path).
- It does not commit to any order beyond Phase 1: if the baseline run (0.4) shows the time is somewhere this
  report did not screen for, that finding comes before the rest.

## Already in main (do not apply)

Verified present in `upstream/main` at `296b59d2`; these PRs are stale after the history rewrite:

- #1101 prefill Stager sleep (`src/prefill/prefill.cpp`), #1125 `q8k_quant_avx2`, #1237 pinned stage buffers
  (as `STRATA_STAGE_PIN`), #1107 `STRATA_PREFILL_GROUP_GATHER`, #2042 `STRATA_EXCHANGE_ROTATE`,
  #1517 `STRATA_ADAPT_NOWAIT`, #1089 `bench_eviction`, #2073 `STRATA_PLE_AUTO_MMAP`, #1166 `--host-core`,
  #1597 `--conversation-cache-min-tokens`.
- #2321 is **partly** merged: the 256/8 route-resident kernel exists in `src/kernels/cuda/native_router.cu`,
  but the `verify.cpp` defaults/guard for it do not.

## Corrections to this scan

Applied the same day, after re-checking the branch's own tree. `upstream/main` in this checkout is exactly
`296b59d2`, the commit the scan used, so the search is against the same source.

- **#1548 / #2224 `STRATA_PCIE_BALANCE` and #1614 `--kv-pool-tokens` are not in `main`.** Neither symbol
  appears in `src/`, `include/`, `serve/`, `tools/`, `sycl/`, `setup.py` or `CMakeLists.txt`, and a community
  report in `bench/results/2026-10-08-community-2x-rtx3080-20gb-ud-q4-k-xl-512k-pool/` also records both as
  not in `main`. They were parked under "already in main" in error: applying them would not be a no-op. No
  value claim is made here - they have not had the screen the rest of this report got, and they are Phase 4
  of the [implementation plan](#implementation-plan).
- **#1358 is not cleanly mergeable.** This scan said it was; the GitHub API on the same day reports
  `mergeable: false`, `mergeable_state: dirty`. The patch is 6 added / 2 removed lines in `src/core/vmm.cpp`,
  so the conflict is small and can be redone by hand. It is still the prerequisite for the VMM features.
- **Merge state re-checked live** for the rest of the shortlist: #1949, #2751, #2515, #2660, #2698 and #2775
  are mergeable; #1581, #1838, #1262, #1989, #1181 and #1358 are conflicting. The scan's own table entries
  agreed with all of these.

## Duplicates and rejected

- Duplicates: #1117 = resubmission of #1030; #1548 = refile of the closed #1038 (it is the live copy of that
  change, not a rejection - see Phase 4 of the [implementation plan](#implementation-plan)); #2399 resubmits
  #2201 and bundles #2398; #1107 replaced #1033; #1089 reposts #991; #2698 vs local #2742.
- Wrong vendor or arch: AMD/HIP/gfx (#2398, #2399, #2522, #2078, #1360, #2628, ...), Intel/SYCL/Arc
  (#2394, #2485, #2486, #2555, #2583, ...), Metal/Apple, Volta/Turing/Pascal device code
  (#2069, #2145, #2220, #2221, #1742, #1744), Windows/WSL-only (#1911, #1324, #1325, #1520, #2672, #1498).
- Multi-GPU / layer split only: #1461, #1458, #2522, #1223, #2425.
- Benchmark reports, no code change: #2004, #2479, #1496, #2581, #2643, and the rest of the community
  reports. They are baselines, not changes.
- RAM/conversation housekeeping with no decode-throughput claim on this box: #2243 (refuses when any
  expert is read from files - not this streaming case), #2501, #1896, #2253.

## What this report does not claim

- None of the above was built or measured on this PC. The numbers are the authors' on their hardware.
- Applicability is judged from the files each PR touches and the flags present in `upstream/main`; a PR can
  still conflict, need the other half of a stacked pair, or behave differently on sm_89.
- The "already in main" check is a flag/symbol search plus an `git apply --check`; a partial merge is
  possible and is called out where seen (#2321). It also produced two wrong calls (#1548/#2224 and #1614),
  corrected above: a search can find a look-alike symbol, or the feature can be absent under the name the PR
  chose.
