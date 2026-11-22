# `STRATA_L3_PF=1` (next-layer expert prefetch into the CPU caches) — community report, RTX 3090 + Ryzen 9 5950X (Linux)

Results and method for the pull request that adds `STRATA_L3_PF`. The change is in the pull request; this folder holds the
measurements.

**Machine.** RTX 3090 24 GB (sm_86, driver 620.70.06, 320 W, PCIe 4.0 x8), Ryzen 9 5950X (16 cores, two CCDs with 32 MiB of
L3 each, AVX2), 96 GB DDR4, Ubuntu 24.04, Linux 7.0, GCC 13.3, CUDA 13.1, Ninja Release builds for sm_86. Qwen3.8-Flash-Next
Huihui abliterated v2 UD-Q4_K_XL (native pack, experts in RAM), with the server's arguments: `--expert-cache auto --prefill auto
--spec 4 --mtp --max-context 262144 --kv int8 --kv-resident 32768 --spec-min-p 0.70 --pcie-frac 0.10 --adapt-every 1
--adapt-swaps 160 --adapt-decay 0.97 --ple-io mmap --conversation-cache-mib 8192 --vram-reserve-mib 500`, a learned expert
profile, `STRATA_HC_Q8=1 STRATA_ROUTE_TAIL_SKIP=0`. About 5,350 experts in VRAM, a decode hit rate near 70%.

**Builds.** All from source on this machine, the same day:

| Name | Revision | Switch |
| --- | --- | --- |
| 0.1.64 | upstream main `6e38f588a` (0.2.18, engine 0.1.64) | none |
| 0.1.64 + this change | the pull request's branch | `STRATA_L3_PF=1` |
| 0.1.59 | `87dfca0e7` (0.2.13, engine 0.1.59.1), the base of PR #5712 | none |
| 0.1.59 + this change | the same four commits on `87dfca0e7` | `STRATA_L3_PF=1` |
| 0.1.59 + #5712 | PR #5712 at `eca66e81a`, `-DSTRATA_CPU_PREFETCH=ON` | `STRATA_CPU_PREFETCH=4` |

**Method.** Arms interleaved, a fresh engine or server per run, 3 rounds.

- *Agent session*: a recorded 34-turn coding-agent session, replayed straight into the engine (`--serve`, `GEN` lines) as
  the exact token ids of each turn's prompt (142K to 174K tokens; each turn reuses the previous prompt), greedy, each turn
  capped at the number of tokens it generated when recorded. Decode tok/s = all decoded tokens / all decode time; the mean of
  the per-turn speeds is given as well. The prompts are private and not included; `data/agent-session.json` has every turn's
  prompt length, reuse, tokens, prompt and decode time and expert hits.
- *Chat prompts*: the server (`--engine strata`, the build's own server) and 8 fixed prompts (stories, code, explanations),
  600 tokens each, greedy. Decode tok/s = tokens / time over the 8 requests' `done:` lines. `data/chat-prompts.json` has each
  request.
- Welch t over the 3 rounds.

## Against upstream main (0.1.64)

| Workload | 0.1.64 | 0.1.64 + `STRATA_L3_PF=1` | Gain |
| --- | --- | --- | --- |
| Agent session, total decode tok/s | 63.03 / 62.10 / 60.56 | 65.63 / 66.81 / 64.31 | **+6.0%** (t 3.6; pairs +4.1 / +7.6 / +6.2%) |
| Agent session, mean of per-turn tok/s | 60.16 / 59.03 / 57.58 | 62.52 / 63.39 / 61.83 | +6.2% (t 4.2) |
| Chat prompts, tok/s | 66.76 / 67.98 / 67.37 | 70.54 / 71.58 / 71.79 | **+5.8%** (t 7.5; pairs +5.7 / +5.3 / +6.6%) |

0.1.64 runs `STRATA_POOL_DEP` (one batch per native layer call) by default; with `STRATA_L3_PF=1` the layer calls run as
the two phases with fixed row slices. The baseline above includes `STRATA_POOL_DEP`.

## Against PR #5712 on its base (0.1.59)

| Workload | 0.1.59 | + #5712 | + this change | This change vs #5712 |
| --- | --- | --- | --- | --- |
| Agent session, total decode tok/s | 61.21 / 60.09 / 60.45 | 62.30 / 62.61 / 61.87 (+2.8%, t 4.3) | 65.47 / 65.51 / 65.05 (+7.9%, t 13.2) | **+5.0%** (t 11.9; pairs +5.1 / +4.6 / +5.1%) |
| Agent session, mean of per-turn tok/s | 58.31 / 56.95 / 57.68 | 59.19 / 59.39 / 59.11 (+2.7%) | 61.91 / 62.59 / 61.68 (+7.6%) | +4.8% (t 10.0) |
| Chat prompts, tok/s | 66.13 / 66.40 / 67.58 | 67.94 / 68.42 / 69.15 (+2.7%, t 3.2) | 70.38 / 70.77 / 71.55 (+6.3%, t 7.5) | **+3.5%** (t 4.9; pairs +3.6 / +3.4 / +3.5%) |

#5712's counters over a whole agent-session run: about 1.1 million experts selected, 52% of them then routed, half of the
prefetch batches cancelled by the next batch. Both read up to 4 experts per layer. The differences in the designs that this
measurement does not separate: #5712 warms each worker's share of the bytes while the compute phases still hand out row
ranges dynamically, so a worker often computes rows another worker read (on this CPU, possibly into the other CCD's L3);
this change gives each worker the same rows to read and to compute. #5712 predicts with a Q4 copy of the router and
softmax-weighted votes; this change uses the BF16 router and rank votes over the tokens' top k.

## Variants (0.1.61 + this change, agent session, mean of per-turn tok/s, one run each)

Top 2 61.14, top 3 61.04, top 4 61.16 to 63.40 (3 runs), top 6 61.60, top 8 61.29; the top 2 of each token row 62.12.
Top 4 with `--pcie-frac 0` 61.56 or 0.05 61.49; `--pcie-frac 0` without the prefetch 58.85. The fixed row slices alone
(`STRATA_POOL_STATIC=1`): 65.17 / 60.66 / 64.30 tok/s on the chat prompts against 64.71 / 64.65 / 64.64 without, no gain.
The CPU time per missed expert dropped from about 74 to 60 us (`kq_window_bench`: 74 us from DRAM, 26 us from the caches).

## Output

On 0.1.64, a greedy one-shot run (`--tokens-file`, `--max-new 200 --adapt-every 100000 --pcie-frac 0 --window-hashes`)
three times on 0.1.64, twice on this change with the switch off and once with it on: the same 200 tokens every time. The
draft windows' sizes differ from run to run on every build (0.1.64 against itself too); every window with the same position
and drafts had the same hash in every pair of runs. `pool_tasks_test` (468 comparisons, including the fixed slices with and
without a prefetch being stopped), `pool_determinism` (218,736 checks), `pool_test`, `pool_cores_test`, `pool_plan_test` and
`router_dot_parity` pass on this branch (`pool_test` skips its AVX-512 VNNI case on this CPU).

## Not tested

HIP, SYCL, Metal; Windows; NUMA or two-socket machines (the switch is off with the NUMA split); any other GPU or CPU; other
models or packs; the file-backed expert source (all runs used the page-locked RAM arena).
