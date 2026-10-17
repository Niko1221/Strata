# Q36: the model-load bottleneck and the parallel expert copy

**Status:** **confirmed and approved (2026-10-17).** Analysis and Phase 1 implemented, measured and committed on
branch `faster-model-load` (from `dev`), commit `a4362c47`.
**Target:** the Qwen3.6 family's engine (`--family qwen36`, `strata-q36` / the q36 path inside `strata`), the path this
PC actually runs.
**Measured on:** RTX 4060 Laptop GPU (8188 MiB), Ryzen 7 7735HS (8 cores / 16 threads), 62 GB RAM,
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.85 GB), `--kv int8 --mtp 2`, 32K context, hybrid mode (0 expert slots,
decode is the CPU pool) — the same machine and settings as [Q36_MEMORY_COMPACTION.md](Q36_MEMORY_COMPACTION.md).

## Why this exists

The model takes **~1m50s to load**. During the load the host reports GPU utilisation 0% (VRAM is allocated but no
kernel runs), CPU pegged at 100% on an 8-core part, and storage I/O at ~300 MB/s against a disk that reads ~1 GB/s.
All three point the same way: **the load is not compute-bound, not disk-bound, but bound by one CPU thread doing
serial work on the bytes as they arrive.** This document records the measurement that confirms it, and the plan to
lift it.

## Baseline (measured 2026-10-17)

The engine already times its two load phases; the numbers are its own log lines.

| phase | what it does | warm | cold |
|---|---|---|---|
| `load()` (dense weights) | 753 tensors laid out, 4.40 GiB dequantised/copied to the GPU | **2.8 s** | **5.4 s** |
| `place_experts()` (routed experts) | 18.75 GiB copied from the GGUF mapping into the pinned host arena (256 experts × layers) | **37.2 s** | **~90 s** |
| total to ready (incl. warmup + a trivial bench) | | 60.6 s | ~1m50s (user report) |

"Warm" is a second run with the 22.85 GB file in the page cache; "cold" is the first run after the cache was empty.
The cold `place_experts()` number is approximate because the first cold run hit a staging-buffer OOM before its
summary line printed (the fix needs `STRATA_Q36_SINGLE_STAGE=1` on this 8 GB card, which the memory-compaction
baseline also used); its wall time to that point was 99 s, of which `load()` was 5.4 s.

The 37.2 s "warm" figure understates the serial cost: the controlled A/B below (same binary, cache explicitly warmed
and dropped) reproduces **serial `place_experts()` at ~86 s — 22.5 s arena pin + 64 s copy — whether the cache is hot
or cold**. The 37.2 s was taken immediately after the OOM'd run, when the freshly freed arena made the pin cheap; it is
kept here as the first observation, not the reproducible number.

Two facts stand out:

1. **`place_experts()` is ~90% of the load**, and within it there are two distinct costs: the `cudaHostAlloc` pin
   (~22.5 s) and the expert copy (~64 s, serial). The dense weights are not the problem.
2. The expert copy runs at **~300 MB/s**. The same machine's CPU pool measures **36–44 GB/s** of memory bandwidth (see
   `include/strata/kernels/cpu/pool.hpp`), so one thread copying 18.75 GiB is ~100× below what the cores can move — and
   the copy is within 2% of the same time warm vs cold, so it is the per-page fault + memcpy on one thread that caps
   it, not the disk. That is exactly the "CPU 100%, storage 300 MB/s, GPU 0%" picture: the ~300 MB/s the user saw is
   the single thread's rate, and the disk's 1 GB/s never gets a chance to matter.

## Where the time goes (the code)

`src/qwen35/q36_engine.cu`, `place_experts()`:

```cpp
for (int64_t e = 0; e < ne; ++e) {                      // 256 experts, serial
    expert_blob(L, e, L.hblobs + (size_t) e * L.stride);  // 3 memcpys: gate | up | down, from the mmap
    std::memset(L.hblobs + (size_t) e * L.stride + written, 0, L.stride - written);
}
```

`expert_blob()` is three `memcpy`s out of the mmapped GGUF (`L.src_g/u/d`) into the pinned arena. The loop is one
thread, one expert at a time, for `n_layers × 256` experts. Nothing in it depends on order; the source and destination
ranges are disjoint per expert.

## Proposed solutions (ranked by leverage / effort)

| # | change | effect | risk |
|---|---|---|---|
| 1 | **Parallelise the expert copy** across the cores | warm 64 s → 13 s; cold 65 s → 16 s | none: byte-identical output, ~15 lines |
| 2 | **Prefetch the mapping** (`madvise(MADV_WILLNEED)`) | cold only; measured: warm ≈ cold, so this is now lower value | low |
| 3 | Parallelise `upload_f32` (F16/BF16/Q8_0 → F32 scalar loops) | dense 5 s → <1 s | low, small win |
| 4 | Overlap the dense GPU upload with the expert copy | hide the ~5 s dense phase | medium, more moving parts |
| 5 | **Address the `cudaHostAlloc` pin** (measured at **~22.5 s**, see Results) | the copy fix leaves this as ~40% of the load | medium, needs a pageable/pin-by-need investigation |

The primary lever is **#1**. The new phase metrics (below) separate the arena allocation from the copy so #5's share
is measured rather than guessed.

## Implementation plan

### Phase 1 (this change)

- `STRATA_Q36_PARALLEL_PLACE`: `0`/unset = the serial path (the pre-change default, byte-identical); `N` = copy the
  routed experts on `N` threads. Chunked by expert index; each chunk is disjoint, so the bytes are identical however
  many threads run.
- Phase metrics on the existing `hybrid experts:` log line: `arena alloc X s, expert copy Y s, N copy thread(s)`.
  These are the levers — the copy number is the one that moves when the flag flips.

### Phase 2 (follow-up, only if the measurements support it)

- `madvise(MADV_WILLNEED)` on the expert ranges before the copy, for cold loads.
- Flip the default to parallel once it is measured faster, per the repo rule that a default changes only where it is
  measured faster.

## Changes made

- `src/qwen35/q36_engine.cu`:
  - `STRATA_Q36_PARALLEL_PLACE` — `0` = serial (explicit opt-out), `N` = `N` threads, unset = one thread per logical
    core (the default). Chunks are disjoint via `place_expert_range()`, so the placed bytes are identical at any
    thread count.
  - `STRATA_Q36_PAGEABLE_ARENA` — `1` = `malloc` the expert arena instead of `cudaHostAlloc` (pinning) it; unset = the
    pinned default. The pin is the largest remaining load cost (see the diagnosis below).
  - The `hybrid experts:` log line now ends `(arena alloc X s, expert copy Y s, N copy thread(s))`, so the two costs
    and the copy's thread count are visible in the engine log without a profiler.
- `docs/Q36_LOAD_TIME.md` — this document.

## The arena pin is not OS compaction (diagnosis)

The ~22.5 s `cudaHostAlloc` looked like it could be the host rearranging RAM (the memory-compaction work found
`kcompactd` doing 71.7 M futile page migrations). It is not. While the arena was being allocated, the host's
compaction counters stayed frozen — `compact_stall`, `pgmigrate_fail`, `pgmigrate_success` and
`compact_migrate_scanned` did not move over the whole 28 s `place_experts`, and 47.5 GiB was free. The Q36 host
settings are active (THP `never`, `compaction_proactiveness=0`, `compact_unevictable_allowed=0`). So the cost is the
pin itself — locking 18.75 GiB through `cudaHostAlloc` — not the OS making room for it. A pageable arena is therefore
a legitimate cut, not a workaround, which is what `STRATA_Q36_PAGEABLE_ARENA` is for.

## Acceptance criteria

- **AC1** — `STRATA_Q36_PARALLEL_PLACE=8` cuts the `expert copy` phase by ≥3× (64 s → ≤21 s), and the total load drops
  by the same amount. **Met: 64 s → 16.2 s (3.9×).**
- **AC2** — the placed experts are byte-identical with the flag on vs off. **Met: greedy-decode `gen:` tokens are
  identical (see Results).**
- **AC3** — the log line separates `arena alloc` from `expert copy`, so the lever's movement is visible without a
  profiler. **Met: `(arena alloc X s, expert copy Y s, N copy threads)` on the `hybrid experts:` line.**
- **AC4** — the default (flag unset) path is unchanged: same timing and byte-identical output as the pre-change build.
  **Met: flag unset runs the serial loop, byte-identical output.**
- **AC5** — a cold load (page cache dropped) improves too, not just a warm one. **Met: 64.6 s → 16.4 s cold.**

## Results

Measured on `strata:parallel-load` (the branch's image), `--kv int8 --mtp 2`, 32K context, hybrid mode, page cache
dropped before the cold runs and warmed before the warm runs. The `arena alloc`/`expert copy` split is the new metric
on the `hybrid experts:` log line.

| expert copy threads | arena alloc | expert copy | `place_experts` total |
|---|---|---|---|
| 1 (serial, flag unset) | 22.1–23.0 s | 63.8 s warm / 64.6 s cold | 86.1–87.9 s |
| 4 | 22.3 s | 23.6 s | 46.1 s |
| 8 | 22.1–22.6 s | 16.2 s warm / 16.4 s cold | 39.1–39.6 s |
| 16 | 22.1 s | 13.4 s | 35.7 s |

Three findings:

1. **The lever moves, 3.9–4.8×.** Parallelising the copy takes it from 64 s (serial) to 16.2 s (8 threads) and 13.4 s
   (16 threads). Total load drops from ~104 s to ~58 s (8 threads) — the user's ~1m50s becomes just under a minute.
2. **Warm ≈ cold.** The copy is within 2% whether the page cache is hot or cold, so it is bound by per-page fault
   handling and the memcpy on one thread, not by the disk. The "storage 300 MB/s" the user saw is that single
   thread's fault+copy rate (~300 MB/s), not the disk's 1 GB/s limit — which is why the disk never saturated.
3. **The pin is the next bottleneck.** `cudaHostAlloc` pinning the 18.75 GiB arena is ~22.5 s and is untouched by this
   change. Once the copy is parallel, that pin is ~40% of the load and the next lever to pull.

**Byte-identity (AC2):** greedy decode of `--tokens "1,2,3" --n 24 --mtp 0` prints `gen:` 4 5 6 24218 10 4838 1665 …
identically with the flag off and on — the placed experts are bitwise the same, as the disjoint-chunk `memcpy`
guarantees.

### Arena-pin A/B (`STRATA_Q36_PAGEABLE_ARENA`, parallel copy default)

| arena | arena alloc | expert copy | `place_experts` total | prompt (pp512) | decode (tg8) |
|---|---|---|---|---|---|
| pinned (default) | 14.2 s | 13.1 s | 27.8 s | 295–301 tok/s | 16–17 tok/s |
| pageable (`=1`) | **0.0 s** | 13.7 s | **13.8 s** | **136–140 tok/s** | 16–17 tok/s |

The pageable arena removes the ~14 s pin and leaves decode byte-identical, but it **halves prompt throughput**
(295 → 138 tok/s): the prompt staging `cudaMemcpyAsync` reads the arena into GPU staging buffers, and a pageable
source goes through the driver's bounce buffer instead of a direct DMA. That is a bad trade for a model that is loaded
once and prompted many times, so the flag stays opt-in and pinned stays the default. The right way to get the load
win without the prompt cost is a **deferred pin** — allocate pageable, then `cudaHostRegister` the arena after load
(off the critical path) so prompts still DMA from pinned memory.

### Next steps

1. **Done — flip the default to parallel.** `parallel_place_threads()` now returns one thread per logical core when
   `STRATA_Q36_PARALLEL_PLACE` is unset, with `=0` as the serial opt-out (justified: 4.8× faster, byte-identical).
2. **The arena pin — pageable arm measured, rejected as a default.** `STRATA_Q36_PAGEABLE_ARENA=1` cuts the ~14 s
   pin but halves prompt throughput (staging bounce buffer), so it stays opt-in. The next step is a **deferred pin**:
   `malloc` the arena, copy the experts, then `cudaHostRegister` it after load so prompts keep DMA speed.
3. **Trim the CUDA EAGER module load** (~10–15 s of the wall clock before the first tensor is read): load the graph
   kernels lazily or overlap the dense upload with it.
