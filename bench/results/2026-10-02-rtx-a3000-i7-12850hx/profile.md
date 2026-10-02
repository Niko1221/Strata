# RTX A3000 12 GB + i7-12850HX: where a decode token goes

A wall-clock split of one decode round into CPU expert pool / GPU / MTP / host, on the same host as
[README.md](README.md), from the engine's own `--stats` accumulators. The goal is to size the host-side work
(fewer CPU misses, a faster CPU pool, fewer synchronisations) before writing it.

## What was run

Production engine **0.1.35** (`strata-engine-0.1.35`), Qwen3.8-Flash-Next Swift 1.5 IQ3_XXS native pack
(48 layers, top-10), the calibrated host settings from [README.md](README.md):

```
--expert-cache auto --prefill auto --spec 4 --mtp <rt> --max-context 524288 --kv int8
--kv-resident 32768 --rope-scaling yarn --rope-scale 2 --vram-reserve-mib 1800
--pcie-frac 0.35 --spec-min-p 0.7 --pool-workers 8  --stats
```

with the fixed `bench/e2e.sh` prompt, run standalone (the serve unit stopped so the dGPU and the 44 GiB hugetlb
pool are otherwise idle; the arena logs `hugetlb 2 MB pages`, 2107 expert-cache slots). The same prompt through
the running server (`/metrics`, `request.decode_tok_s`) gives the same fresh-decode rates to within ~2%, so the
standalone harness is representative.

`--stats` reports the verify window (the spec path's round) in **ms/round**; a round emits
`1.15`–`2.10` tokens depending on how many drafts are accepted. Percentages below are of the round's
wall clock, `ms/token × tokens/round`.

## The split

At the **fresh-decode** hit rate the fixed prompt actually gets (`hit_rate 0.354`, ≈ the server's fresh
requests), one round is:

| Term (`--stats`) | ms/round | share | note |
| --- | ---: | ---: | --- |
| GPU — "wait for rings" | 26.7 | **34%** | replay of the window up to the router's doorbell |
| CPU expert pool — "pool" | 41.6 | **53%** | the host computing the CPU misses |
| MTP drafter | 3.1 | 4% | one sync per draft token |
| host | 1.7 | 2% | |
| verify commit | 0.03 | ~0% | the `cudaStreamSynchronize(copy_)` arm |
| embed/sample/round overhead | 4.9 | 6% | residual vs `ms/token` |
| **round** | **78.1** | | 26.9 tok/s here |

The same run at **max-new 64** (`hit_rate 0.339`) splits 19.6 / 29.6 / 2.6 / 2.0 / 0.03 ms → pool 52%, GPU 35%,
MTP 5%. Two 42-token repeats agree to ~5% (pool 41.6 vs 37.6, GPU 26.7 vs 30.9; 26.9 vs 26.3 tok/s).

The **warm/degenerate** end is different: the 200-token run falls into a repeated `<|im_start|>` tail, hits
**75.6%**, and the pool drops to **9.6 ms/round** (drain 8.9) while the GPU wait rises to 23.1 ms/round — the GPU
is then the largest term. So the pool's share is set by the cache hit rate, which on real serve traffic here ran
**0.35–0.65** (34–40 tok/s), matching the fresh-decode split above.

**Pool internals** (fresh, M42a): gate/up 28.2 ms/round, down 11.9 ms/round, quantize 0.2 ms — the i-quant
gate/up is ~68% of the pool, the down path ~29%, at **18–22 GB/s over the rows**. That is below the
pinned/host-loop figures in `include/strata/kernels/cpu/pool.hpp` (36.3 / 26.9 GB/s), so the pool is not running
at its pinned best here.

## The 8th pool worker is an E-core

`--pool-workers 8` with the default `--pool-affinity all` does **not** land on 8 P-cores. Linux hybrid detection
orders the physical cores CPU-index order and pins worker `i` to `worker_cores[i]`, so 8 workers take cores
**2,4,6,8,10,12,14,16** — seven P-core primaries plus the first E-core (CPU 16). Confirmed by reading
`/proc/<pid>/task/*/status` during a run. The E-core is the tail of every split-row drain;
`--pool-affinity p-cores` avoids it but starts only `p_cores - 1 = 7` workers.

## PCIe is not bandwidth-bound in use

At the fresh hit rate the verify window sends **4.06 distinct missed experts per layer** to the GPU over PCIe
(`0.84` at 75% hits), i.e. ~195 experts/round × ~1.7 MiB ≈ **0.3 GiB/round**, ~4 GB/s at 78 ms/round. The link
is x16 Gen4. The probe's job is therefore only to choose `pcie_frac` (how many misses move to the GPU), not to
measure a rate the decode then saturates — consistent with issue
[#485](https://github.com/Niko1221/Strata/issues/485), whose median-of-bursts fix (#487) is the right shape.

## What this native pack cannot measure

The IQ (native) pack runs **verify windows only** — `generate.cpp:3184` skips the per-layer graph capture when
`expert_layout().native`, and the pack requires `--native/--spec/--prefill`. So on this model:

- `--gpu-stages` → `session_replay_stages: not captured with the split`
- `--gpu-only-full` → `session_replay_full: not captured with post graphs`
- `--no-pool` + `--spec` → `--spec needs the device residency table ...` (the R4 hit path that supplies it is
  disabled with the pool)

`/metrics`'s `ram_blobs`/`file_blobs`/`file_mb` are **0** for every request on the pinned-arena configuration
(they are populated for the mmap/resident-file modes, not the `MAP_HUGETLB` arena). The verify-window
"wait for rings" term is the GPU floor available here; the graph-replay floor (26.32 ms, `generate.cpp`'s
`--no-pool` comment) is the non-native path's. `ncu` is not installed on this host, so kernel-level counters
were not collected.

## What it sizes

- **Pinning the verify host thread is the top lever.** The pool is ~50% of the round at the real hit rate;
  recovering the pinned 36.3 vs 26.9 GB/s (1.35×) on it is worth roughly **+13–15% tok/s**. The measured
  18–22 GB/s says there may be more than 1.35× to win.
- **256-bit AVX-VNNI is bounded.** It shortens only the down path (~29% of the pool) plus part of the i-quant
  gate/up; a ~1.5–2× down path is ~5–8% of the round.
- **The per-window copy-stream sync is already ~0 on this host.** The `commit` term is 0.03 ms/round. The MTP
  drafter's per-draft syncs live inside the 4–5% MTP term and are the only part of that item with headroom here.
- **The E-core tail is real** (above); the calibrated 8-worker count is already worth ~6%.

## Raw

The `--stats` block per run (fixed prompt above; `prompt`/`output` token lines elided):

```
# fresh, max-new 42 (M42a)
verify window   wait for rings 26.732  pool 41.589  host 1.682  commit 0.032 ms/round; CPU experts 8.85 distinct / 12.38 routed per layer
pool multi      gate/up 28.233  quantize 0.231  down 11.933 ms/round; 18.4 GB/s over the rows phases; CPU pool call 41.560 ms/round
dispatch        plan 0.115  activation quantize 0.559  jobs 0.453  run 40.413 ms/round
mtp             3.141 ms/round drafting (19 rounds)
decode          42 tokens in 1561.4 ms  ->  26.90 tok/s
  per token     37.175 ms/token
  the CPU expert pool  41.560 ms/token over 48 layers (20 positions, 960 dispatches)
  R4 expert-cache hits 6513 of 18401 = 0.3539

# fresh, max-new 64 (M64)
verify window   wait for rings 19.628  pool 29.592  host 2.009  commit 0.027 ms/round; CPU experts 7.49 distinct / 10.16 routed per layer
pool multi      gate/up 19.617  quantize 0.172  down 8.887 ms/round; 21.9 GB/s; CPU pool call 29.566 ms/round
mtp             2.624 ms/round drafting (37 rounds)
decode          64 tokens in 2159.1 ms  ->  29.64 tok/s
  R4 expert-cache hits 9499 of 28028 = 0.3389

# warm/degenerate tail, max-new 200 (N1)
verify window   wait for rings 23.146  pool 9.572  host 0.216  commit 0.026 ms/round; CPU experts 2.13 distinct / 2.74 routed per layer
pool multi      gate/up 6.267  quantize 0.051  down 2.604 ms/round; 20.3 GB/s; CPU pool call 9.556 ms/round
mtp             2.094 ms/round drafting (173 rounds)
decode          200 tokens in 6483.6 ms  ->  30.85 tok/s
  R4 expert-cache hits 70942 of 93789 = 0.7564
```
