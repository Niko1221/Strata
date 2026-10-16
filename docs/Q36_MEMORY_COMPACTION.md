# Q36: the model mapping, the expert arena, and kcompactd

**Status:** plan, no engine change implemented yet. Branch `q36-memory-compaction` (from `dev`).
**Target:** the Qwen3.6 family's engine (`--family qwen36`), the path this PC actually runs.
**Host settings:** applied on the target PC 2026-10-17, **not persisted yet** - see
[Host settings](#host-settings-applied-2026-10-17).

| phase | switch | goal | state |
|---|---|---|---|
| 0 | `STRATA_MEM_REPORT` | the memory/compaction counters in the engine's own log | planned |
| 1 | `STRATA_Q36_RELEASE_MAPPING` | release the GGUF mapping after load | planned |
| 2 | `STRATA_Q36_PAGEABLE_ARENA` | stop pinning the 18.75 GiB expert arena | planned |
| 3 | `STRATA_Q36_ARENA` | large pages for the arena, if it must stay pinned | planned |
| 4 | - | padding-only memset of the arena | planned |
| 5 | - | flip the defaults that measured better | planned |

Measured on the machine this plan is for: **RTX 4060 Max-Q Laptop (8188 MiB), Ryzen 7 7735HS, 62 GB**,
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.85 GB), `--kv int8 --mtp 2`, 32K context, hybrid mode
(the model's 18.75 GiB of experts fit in 8 GiB VRAM neither way, so **0 expert slots: decode is the CPU pool**,
[Q36_INT8_KV.md](Q36_INT8_KV.md)).

## Why this exists

`kcompactd` works continuously while the engine runs, and the host reports **71,744,161 failed page migrations
against 2,373,960 successful ones** (`/proc/vmstat`) with `compact_stall` at only 146 — no shortage, just
non-stop futile work. Two different memory objects in the engine drive that, and they need different fixes:

| object | how it is created | evictable? | size |
|---|---|---|---|
| the GGUF mapping | `mmap(PROT_READ, MAP_PRIVATE)`, the `GgufModel` held by `Model::file` for the process lifetime | **yes** — clean file pages | ~22.85 GB |
| the expert arena | `cudaHostAlloc` in `place_experts` | **no** — pinned/unevictable | ~18.75 GiB |

The mapping is read exactly once. The only reader of it for experts is `expert_blob()`, and every caller of that
is inside `place_experts()`; the dense weights read it through `upload_f32()` at init. After that nothing needs
it again, but it stays mapped, so ~22.85 GB of 4 KB file pages (this host has `thp_file_alloc = 0`) sit in the
middle of physical memory for the whole run.

The arena is the larger cause and cannot be reclaimed at all while it is pinned. On the decode path the CPU pool
reads it; the GPU only DMAs *out* of it for prompt staging and slot refills.

## Host settings (applied 2026-10-17)

These are **host** settings, not repo changes. They were applied at runtime on the target PC
(`kernel 7.0.0-38-generic`, Ubuntu 24.04) and **none of them is persisted** - the config files below were not
created, so a reboot reverts every one of them.

| setting | before | now | what it stops | revert with |
|---|---|---|---|---|
| `vm.compaction_proactiveness` | 20 | **0** | the compactor's periodic background pass | `sysctl -w vm.compaction_proactiveness=20` |
| `vm.compact_unevictable_allowed` | 1 | **0** | compaction trying to migrate the mlocked/pinned arena - the 71.7 M failures | `... =1` |
| `vm.extfrag_threshold` | 500 | **1000** | the allocator reaching for *direct* compaction | `... =500` |
| `vm.min_free_kbytes` | 67584 (66 MiB) | **1048576** (1 GiB) | high-order requests falling through to compaction | `... =67584` |
| `vm.nr_overcommit_hugepages` | 0 | **10000** | nothing by itself; it lets Phase 3's `MAP_HUGETLB` succeed on demand instead of reserving 19 GiB up front | `... =0` |
| THP `enabled` | `always` | **`never`** | every new mapping wanting 2 MiB blocks the fragmented free lists cannot supply | `echo always > /sys/kernel/mm/transparent_hugepage/enabled` |
| THP `defrag` | `madvise` | **`never`** | synchronous compaction on huge-page faults | `echo madvise > .../defrag` |
| `kernel.numa_balancing` | 0 | 0 (unchanged) | - | - |

Deliberately left alone: `vm.nr_hugepages` stays 0 (Phase 3 uses the overcommit pool), `vm.swappiness` is moot
(this PC has no swap), `vm.overcommit_memory` stays 0.

**Baseline caveat:** the 71.7 M / 2.4 M migration figures were read *before* these settings were applied. Every run
from here re-reads them, so the before/after is a delta on the same counters, not a comparison against the number
in this document.

**To keep the settings across a reboot:**

```bash
sudo tee /etc/sysctl.d/90-strata-compaction.conf >/dev/null <<'EOF'
vm.compaction_proactiveness = 0
vm.compact_unevictable_allowed = 0
vm.extfrag_threshold = 1000
vm.min_free_kbytes = 1048576
vm.nr_overcommit_hugepages = 10000
EOF
printf 'w /sys/kernel/mm/transparent_hugepage/enabled - - - - never\nw /sys/kernel/mm/transparent_hugepage/defrag - - - - never\n' \
  | sudo tee /etc/tmpfiles.d/90-strata-thp.conf
sudo sysctl --system
```

**The interaction to remember.** If THP is ever put back to `madvise` (the distro default), the engine will
*request* `MADV_HUGEPAGE` on a ~20 GB arena (`src/core/pinned.cu`), which is the #2126 failure the source already
documents: *"148K failed compactions, 100% system time, the load never ended."* With `madvise` you must also set
`STRATA_NO_ARENA_THP=1`. `never` avoids the question; the Q36 path never requests THP anyway.

**The trade-off, recorded honestly.** `vm.compact_unevictable_allowed=0` stops the futile scanning but also means
compaction will not defragment *around* the pinned arena. That is the right direction while the engine holds
18.75 GiB pinned; it is the wrong direction once the arena is pageable (Phase 2) and something else needs
high-order blocks.

## What this plan does not touch

- the expert kernels, the router, the placement policy, the K/V path;
- the **default** behaviour of any phase until Phase 5, and only there where a phase measured better
  (AGENTS.md: a new feature is opt-in and the default path stays byte-identical).

## Quality gates

A gate container is made once. The `nvidia/cuda:13.0.0-devel-ubuntu24.04` image is already on this host and has
`make`, `g++` and `nvcc`, but not `cmake`, `ninja` or `python3`:

```bash
docker run --name strata-gate-tmp nvidia/cuda:13.0.0-devel-ubuntu24.04 \
  bash -c 'apt-get update && apt-get install -y cmake ninja-build python3 python3-numpy git'
docker commit strata-gate-tmp strata-gate:cu130 && docker rm strata-gate-tmp
```

**G1 - CPU suite** (every phase; no GPU, ~minutes):

```bash
docker run --rm -v "$PWD":/src -w /src strata-gate:cu130 bash -lc \
  'cmake -S . -B build-gate-cpu -G Ninja -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON &&
   cmake --build build-gate-cpu -j"$(nproc)" &&
   ctest --test-dir build-gate-cpu --output-on-failure'
```

**G2 - CUDA build + the parity suite on the card** (every phase; needs `--gpus all`):

```bash
docker run --rm --gpus all -v "$PWD":/src -w /src strata-gate:cu130 bash -lc \
  'cmake -S . -B build-gate-gpu -G Ninja -DSTRATA_ENABLE_CUDA=ON \
     -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_BUILD_TESTS=ON &&
   cmake --build build-gate-gpu -j"$(nproc)" &&
   ctest --test-dir build-gate-gpu --output-on-failure'
```

**G3 - the script gates:**

```bash
python3 -m unittest tools.test_docker_run
python3 tools/check_isa_guard.py
```

**G4 - the container image**, one per phase, tagged by phase:

```bash
IMAGE_TAG=strata:q5k-ph0 docker/build.sh
```

**G5 - smoke** (`run.sh` has its own smoke path):

```bash
IMAGE=strata:q5k-ph0 SMOKE=1 MODEL_FILE=/home/george/apps/llamacpp/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  docker/run.sh serve
```

**G6 - the run the user does**, after every phase:

```bash
IMAGE=strata:q5k-phN STRATA_MEM_REPORT=1 STRATA_Q36_<phase switch>=1 \
  STRATA_KQ256=1 STRATA_Q36_SINGLE_STAGE=1 CONTEXT=32768 \
  ENGINE_ARGS_EXTRA="--kv int8 --mtp 2" API_KEY=secret \
  MODEL_FILE=/home/george/apps/llamacpp/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf docker/run.sh serve
```

and, in another shell, while it decodes:

```bash
watch -n1 "grep -E 'Mlocked|Unevictable' /proc/meminfo; echo ---; \
           grep -E 'compact_daemon|pgmigrate|compact_stall' /proc/vmstat; echo ---; \
           ps -eo pcpu,etime,comm | grep -iE 'compact|strata' | head"
```

`IMAGE=` must be named every time: `docker/run.sh` defaults to `strata:local`, which is not the phase image.

## Measurement protocol (the same for every phase)

1. Same prompt, same `CONTEXT=32768`, `--kv int8 --mtp 2`, `--mtp 2`'s window shape unchanged.
2. Report **decode tok/s as the server reports it** (`predicted_per_second`), with `prompt_n`, `pred_n` and the
   generated-token count, three repeats, median. The A/B harness already does this
   (`build-q5k/ab_client.py`), and it must be run the same way each time.
3. Report the context the decode actually ran at. **This is where the 18 vs 10 tok/s question came from**:
   18.3-19.6 tok/s was measured at a 32-token prompt, and this card measures 15.4 at 4K and 13.6 at 32K
   ([Q36_INT8_KV.md](Q36_INT8_KV.md)) - the same engine, different context. A phase that changes nothing will
   look like a 20% regression if the two runs are at different contexts.
4. Record, per phase: `Mlocked`, `Unevictable`, `MemAvailable` before/during/after; the `/proc/vmstat`
   compact deltas; and whether `kcompactd` is still pegging a core.
5. One variable per phase. If a phase changes two things, split it.

---

## Phase 0 - make the numbers visible (no behaviour change)

**Goal.** Every later phase is judged on memory counters that today must be read from another shell. Put them in
the engine's own log, so the user's run and the A/B harness carry them.

**Change.**
- New `include/strata/platform/mem_report.hpp` + `src/platform/mem_report.cpp`:
  `mem_snapshot()` reads `/proc/meminfo`, `/proc/vmstat` and
  `/sys/kernel/mm/transparent_hugepage/enabled`; `mem_report(FILE*, const MemSnapshot* prev, const char* when)`
  prints the absolute values and the delta since `prev`.
- `parse_meminfo()` / `parse_vmstat()` are separate from the file reads, so the test feeds them fixture text.
- `src/qwen35/q36_engine.cu`: one `mem_report` after `place_experts()` (so `Mlocked` includes the arena) and one
  on QUIT, both behind `STRATA_MEM_REPORT=1`. Off = no output, no behaviour change.
- `src/platform/mem_report_test.cpp` + a `CMakeLists.txt` entry: parses two fixtures, checks the delta arithmetic,
  the GiB formatting, and that a counter missing from an older kernel does not crash.

**Gates:** G1, G2, G3, G4 (`strata:q5k-ph0`), G5.
**Then:** ask the user to run the baseline with `STRATA_MEM_REPORT=1` **and no phase switch**, at 32K, and report
tok/s + the memory report + what `kcompactd` does. This is the baseline every later phase is compared against.
**Rollback:** delete the two call sites; the switch is off by default.

---

## Phase 1 - release the GGUF mapping after the load

**Goal.** Return ~22.85 GB of reclaimable cache and ~45 MB of page tables, and remove 5.6 million 4 KB pages
from the compactor's walk.

**Change.**
- `include/strata/artifact/gguf_reader.hpp`:
  `GgufFile::release_pages()` -> `posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED)` (the fd is already kept open),
  `GgufModel::release_pages()` over the shards, and `GgufModel::released()`.
  `munmap` alone is not enough (the page cache keeps the pages) and `MADV_DONTNEED` alone is not enough
  (it drops the PTEs, not the cache) - both are needed, which is what the two calls do.
- `src/qwen35/q36_engine.cu`: `release_model_mapping(Model&)` called after the last reader of
  `src_g/src_u/src_d`, behind `STRATA_Q36_RELEASE_MAPPING=1` (default 0 = today's behaviour).
- A `file_released_` flag asserted in `expert_blob()`, so a future re-placement path fails loudly instead of
  dereferencing a dead mapping.
- `src/artifact/gguf_reader_test.cpp`: extend the existing test - map a temp GGUF, `release_pages()`, confirm the
  file still reads correctly by path and the object destructs cleanly.

**Why it is safe.** `expert_blob()` is the only reader of the expert tensors and every call is inside
`place_experts()`; the dense `upload_f32()` reads happen at init. Both are before the release point.

**Gates:** G1, G2, G3, G4 (`strata:q5k-ph1`), G5.
**Then:** the same 32K run with `STRATA_Q36_RELEASE_MAPPING=1`.
**Expected (to be measured, not claimed):** `Cached` down by ~22 GB; `Mlocked` unchanged (~19 GiB) - the arena is
still pinned, so this phase alone should **not** stop `kcompactd`.
**Rollback:** `STRATA_Q36_RELEASE_MAPPING=0`.

---

## Phase 2 - a pageable expert arena

**Goal.** The arena is the unevictable mass and probably the reason the compactor cannot finish. Test whether
pinning all 18.75 GiB is actually buying anything on the decode path.

**Change.** `place_experts()` already has the fallback (`cudaHostAlloc` failing falls back to `std::malloc`).
Add `STRATA_Q36_PAGEABLE_ARENA=1` to take the `malloc` branch deliberately, and one log line naming which branch
was taken and the arena size.

**Trade the phase exists to measure.** The prompt path `cudaMemcpyAsync`es out of the arena, and the driver stages
copies out of pageable memory - so **prefill may get slower**. Decode reads the arena on the CPU only, so it should
not move. That trade is the whole question.

**Gates:** G1, G2, G3, G4 (`strata:q5k-ph2`), G5.
**Then:** the 32K run with `STRATA_Q36_PAGEABLE_ARENA=1`, plus a prompt-heavy run, and report both.
**Expected:** `Mlocked` -> ~0; the compact deltas should stop growing. If decode is unchanged and prefill is not
badly hurt, this is the phase that fixes `kcompactd`.
**Rollback:** `STRATA_Q36_PAGEABLE_ARENA=0`.

---

## Phase 3 - large pages for the arena (only if Phase 2 hurts)

**Goal.** If the arena must stay pinned, make it out of pages the compactor never touches, and cut the pool's TLB
pressure at the same time.

**Change.** Route the Q36 arena through the existing allocator in `src/core/pinned.cu` (it already tries
`MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB|MAP_HUGE_2MB`, then a 2 MiB-aligned `MADV_HUGEPAGE` mapping, then 4 KB) and
`cudaHostRegister()` the result. Behind `STRATA_Q36_ARENA=pin|large|pageable` (`pin` = today).
The host setting `vm.nr_overcommit_hugepages` is what lets the hugetlb arm succeed without reserving 19 GiB up
front; the engine already prints `MAP_HUGETLB unavailable (needed N 2 MiB pages, vm.nr_hugepages=0)` when it fails.

**Risk: the highest in this plan.** It changes how the prompt staging and slot refills get their source memory.
It is a separate phase precisely so a regression is attributable.

**Gates:** G1, G2, G3, G4 (`strata:q5k-ph3`), G5, and the full 32K run plus a prefill benchmark.
**Rollback:** `STRATA_Q36_ARENA=pin`.

---

## Phase 4 - stop zeroing 18.75 GiB that is about to be overwritten

**Goal.** `std::memset(L.hblobs, 0, ...)` is followed immediately by `expert_blob()` writing every expert into it.
The zeroing is only needed for the padding **between** blobs (`stride > L.L.bytes`) and the `kMmqTail` tail.

**Change.** Zero the padding only: for each expert, `[e*stride + L.L.bytes, (e+1)*stride)`, plus the tail after the
last blob. Prove it with a test that both forms leave identical bytes, and with `native_expert_parity` unchanged.

**Value:** a few seconds off every load, and less startup memory churn. Lowest priority of the four.

**Gates:** G1, G2, G3, G4 (`strata:q5k-ph4`), G5. **Rollback:** the phase is small enough to revert whole.

---

## Phase 5 - defaults

Only for the phases that measured better on this machine, flipped one at a time, each with the numbers and what
they were measured on (AGENTS.md). The A/B switches stay. The doc gets the measured table, the way
[Q36_INT8_KV.md](Q36_INT8_KV.md) has one.

## Risks and open questions

- **Phase 2/3 change what the prompt path copies from.** Both are opt-in and phase-separated for that reason.
- **The mapping release assumes no re-placement.** The `file_released_` assert is the guard; if a future
  adaptive/placement path needs the map again, the release must move or be disabled.
- **`kcompactd` may not be the cause of anything.** If the counters flatline after Phases 0-2 and tok/s does not
  move, the compactor was a symptom, and the bottleneck is back in H3-H6 (pool bandwidth, per-layer serialisation,
  host-side plan/actq, staging).
- **`16 tok/s` vs `18 tok/s` vs `10 tok/s`** is not settled by this plan. Every run below must report the context
  it decoded at, or the numbers cannot be compared.

---

## Appendix: the evidence behind this plan

Everything here is measured on the target PC, not inferred.

| what | value | where from |
|---|---|---|
| failed vs successful page migrations | 71,744,161 / 2,373,960 (~30:1) | `/proc/vmstat`, before the host settings |
| pages the compaction daemon has scanned | 107,916,788 migrate + 90,449,289 free | `/proc/vmstat` |
| allocation stalls | 146 | `/proc/vmstat` - hence no visible memory shortage |
| file huge pages ever allocated | 0 | `/proc/vmstat` `thp_file_alloc` |
| mlocked/unevictable at idle (engine not running) | 104 kB | `/proc/meminfo` |
| expert arena | ~18.75 GiB, one `cudaHostAlloc` | `src/qwen35/q36_engine.cu`, [DETAILS.md](DETAILS.md) |
| GGUF mapping | 22.85 GB, `MAP_PRIVATE PROT_READ`, kept for the process lifetime | `include/strata/artifact/gguf_reader.hpp`, `src/qwen35/q36_engine.cu` |
| readers of the mapping for experts | `expert_blob()` only; every caller is inside `place_experts()` | `src/qwen35/q36_engine.cu` |
| decode at 32-token context | 18.3-19.6 tok/s | `build-q5k/ab_results2.txt` |
| decode at 4K / 32K context | 15.4 / 13.6 tok/s | [Q36_INT8_KV.md](Q36_INT8_KV.md) |

**Open, and deliberately not part of this plan:** whether the CPU pool is on the critical path at all (its
~6.6 GB/s of expert traffic at 10 tok/s is far below any plausible DRAM ceiling on this box), whether the per-layer
barrier serialises CPU and GPU, and where the host's per-layer `plan`/`actq` time goes.
`STRATA_DECODE_TIMING=1`, `--stats` (`pool phases`) and `--gpu-only-full` are the instruments
(see [DETAILS.md](DETAILS.md)). If the compaction counters flatline after Phases 0-2 and tok/s does not move, that
work is the next plan.

