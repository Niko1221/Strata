# Linux: lock weights without locking host checkpoints

The Linux `strata-dsv41` engine has two independent, opt-in settings. The main engine has its own separate pair,
described below. Set them in the environment of the engine
or of the server that starts it. They are read once per process; restart the engine after changing them.

| Setting | Default | `1` requests |
| --- | --- | --- |
| `DS4_MLOCK_WEIGHTS` | `0` | Linux `mlock()` on CPU weight storage |
| `DS4_MLOCK_CONTEXT` | `0` | Linux `mlock()` on engine-owned host checkpoint and RAM-slot payloads |

Only `0` and `1` are accepted. An invalid value prints a warning and uses `0`. These are not aliases for the
GPU driver's pinning settings. No `mlockall()`, process-wide locking, forced swapping, limit changes, or kernel
tuning is performed. With both unset, neither category is explicitly locked by this feature.

For **each of the four variables independently**, zero and unset retain pre-patch allocation,
initialization, retention and legacy locking choices. They add no new locks. Zero is not an unlock
switch and does not remove GPU-required pinning. Core's existing legacy PLE soft-limit increase
is an explicitly approved exception to the no-limit-changes rule; see the main-engine section.

For weights resident in RAM and pageable host conversation payloads:

```sh
export DS4_MLOCK_WEIGHTS=1
export DS4_MLOCK_CONTEXT=0
strata-dsv41 --pack /path/to/model/maya-ds4 --mmap-experts off --serve
```

`--serve` above is the engine's stdin/stdout protocol, not an HTTP listener. When using the Python HTTP server,
set these variables in that server's environment so its engine child inherits them.

Pageable means the kernel **may** swap an allocation when appropriate, not that it must. Swap does not provide
room for locked pages. On a machine with 125 GiB usable RAM, leave room for the OS, GPU-pinned host allocations
and other workloads; the locked working set must fit physical RAM. This feature does not change swappiness,
MGLRU, working-set protection, checkpoint formats, slot retention, or inference math.

## Allocation coverage

Weights selected by `DS4_MLOCK_WEIGHTS=1`:

- EXL3 CPU expert arenas, including their NUMA parts and spare slots.
- GGUF copied/packed expert arenas (`--mmap-experts off`), the RAM tier, and spare slots. A lock belongs to an
  allocation, not to an expert ID, so replacing an expert in the same slot keeps its new contents locked.
- Mapped expert source ranges. GGUF tensors use the actual packed row sizes and strides; EXL3 uses each expert's
  file region. A layer whose CPU experts all have dedicated copies does not require duplicate locks of its mapped
  sources: cache swaps return evicted GPU experts through a spare copied slot. A layer with any CPU file-tier experts
  selects all its fallback tensors, including GPU experts that may return to the CPU. Those sources are locked at
  load rather than during routing. This can require more RAM than just the current CPU expert working set. The
  evictable RAM tier keeps its mapped fallback sources selected too, because an evicted expert may be read there again.
- The expert store's committed chunks, including later commits. Lock ownership ends before decommit or unmap.
  Enabled stores prepare at startup within the existing budget and RAM floor. Later growth is prepared by
  one worker, outside the demand mutex. Only completed chunks are published. Demand never waits for preparation:
  it reuses an evictable ready slot or returns to the existing mapped/file fallback. A lock failure publishes
  pageable backing with a failure diagnostic, not a successful lock. Shutdown joins the worker before unmapping.
  Shrink skips a concurrent preparation; subsequent demand can request regrowth. No worker or new pregrowth is
  introduced when weight locking is zero/unset.
  Mapping/identity queries and slot-metadata readers share the publication mutex, including while vectors
  grow or shrink. A returned slot pointer still requires the existing pin/visibility lifetime contract.
- Engram blocks prepared at cache initialization, within its configured row-cache budget (not the whole table).
  Insertion, including block boundaries and eviction, does not allocate or lock payload storage when enabled.
  If backing allocation fails, the cache reports and uses only completed blocks; zero ready rows means uncached
  reads. A failed optional lock retains usable pageable backing. The disabled path still grows lazily.
- Embedding and Markov host weight tables; host vision weights and small
  vision tensors; image-marker weight tensors.

Dense weights kept only in GPU allocations are not passed to `mlock()`. File ranges whose views have been dropped
are not treated as live CPU storage. An expert not held in any CPU allocation (only on disk or GPU) is not a locked
CPU allocation; a subsequently allocated store/cache slot is covered by its own lock.

Context selected by `DS4_MLOCK_CONTEXT=1`:

- Host copies of window rings and compressor groups in the three checkpoints and additional restore points.
- The same checkpoint payloads for every parallel conversation, including restored/copied checkpoints.
- Server-owned in-memory conversation slots and shared-prefix snapshots, containing their serialized KV, index
  keys, attention state and checkpoint payloads.

These vectors acquire locks when their backing allocations grow. Copies own separate locks; moves transfer ownership.
Deletion releases locks before freeing storage. Selected vector payloads use separate anonymous mappings when locking
is enabled, so weight and context payloads cannot share a malloc page. The disabled path keeps ordinary allocation.

With `DS4_MLOCK_CONTEXT=0`, none of those host conversation payloads is explicitly locked, including future growth.
Weight locking never selects them. Disk-backed slots keep their existing format, writes and eviction policy. A
temporary buffer used to read a disk slot is not locked by this feature.

## Important limits

- **Active KV, window rings and attention state in the CUDA/HIP DS4 path live in GPU allocations.** These cannot use
  Linux swap, regardless of either setting. This change does not implement CPU KV offload. Their host checkpoint
  and parked-slot copies remain pageable with `DS4_MLOCK_CONTEXT=0`.
- Existing GPU driver-pinned DMA staging, host-mapped synchronization buffers, embeddings and vision tables remain
  subject to the driver's existing residency rules even when both variables are `0`. The switches neither unpin
  those buffers nor make driver-required staging pageable.
- Small token-history vectors, protocol/slot identity metadata, image-row API buffers and caller-owned buffers
  passed to the original `Model::slot_save(std::vector<uint8_t>&, ...)` API are not selected. They remain pageable,
  including with `DS4_MLOCK_CONTEXT=1`. The server uses the new ownership-safe context-buffer overload for its
  persistent payloads. Thus context locking does **not** currently cover every persistent metadata allocation.
- The main engine uses separate `STRATA_` controls below. Neither family reads the other family's settings.
  Backend-specific paths in `metal/`, `sycl/` and `src/split/` have not been comprehensively audited or validated.
- Non-Linux platforms report the feature unavailable and retain their existing allocation behavior.
- Real-model CUDA/HIP load/unload, cache-swap correctness and inference identity still require GPU validation.
  Do not treat the CPU tests below as proof of full-model residency or inference parity.

## Main Strata engine

For the main engine, use this pair instead. These are **not DS4 aliases**:

```sh
export STRATA_MLOCK_WEIGHTS=1
export STRATA_MLOCK_CONTEXT=0
# Start the main engine or its Python server normally.
```

The settings are read once. Only `0` and `1` are accepted; invalid values warn and use the unset policy.
An empty value also uses the unset policy. Neither setting is a process-wide unlock control.

| Main weight setting | Linux behavior |
| --- | --- |
| Unset | No new weight locks. Existing optional arena locks and `--ple-io ram` remain available. |
| `0` | Same legacy arena and PLE choices as unset; no new weight locks. |
| `1` | Requests tracked allocation-scoped locks instead of the overlapping legacy locks. |

`STRATA_ARENA_LOCK=0` controls the legacy path, not an explicit `STRATA_MLOCK_WEIGHTS=1` request.
**Approved legacy exception:** with `STRATA_MLOCK_WEIGHTS` zero/unset, mapped PLE requested
with `--ple-io ram` (`io.lock=true`) retains its pre-patch soft-limit increase. Only when the
table span exceeds the current soft limit, and the hard limit permits more, it requests the
smaller of the hard limit and the table span plus 1 GiB. It never raises the hard limit.
Ordinary mmap and direct PLE modes do not request this legacy increase. Invalid/empty values
use the disabled policy as documented above.

With `STRATA_MLOCK_WEIGHTS=1`, no PLE path raises limits, even with `--ple-io ram`.
Insufficient limits produce diagnostics and usable pageable fallback. DS4 and the shared
locking utility never change limits. This exception restores only existing legacy behavior;
it is not authorization for new limit-raising paths or system-wide tuning.

Selected CPU weights include expert arenas (ordinary, shared, by-need and deferred), resident complements and
exchange buffers, mapped expert tensor spans, mapped read-only expert arenas, host embedding/output-head weights,
mapped PLE tables and persistent direct-reader row caches. Locks end before their storage is released.
Mapped fallback selection includes tensors that later expert exchanges may need. This can duplicate resident
copies and increase physical RAM requirements. A requested `1` is not proof of complete coverage.

`STRATA_MLOCK_CONTEXT=1` selects parked K/V segments and checkpoint GDN, PLE and indexer payloads, including
copies, growth, retained backing, checkpoint-pool reuse and stage parts. Cache accounting includes retained
capacity, page rounding, lock-owner header pages and stage-image directories. Disk-read preflight bounds the
number of these allocations using file size and runtime limits; open limits can produce conservative estimates.
Logical session-file bytes and the v1 format are unchanged.

With context locking enabled, snapshot admission reserves checkpoint-pool ownership before estimating and consumes that same
storage during capture. It distinguishes a final resident bound from an additional-allocation peak.
The bound includes retained capacities (even empty payloads), checkpoint/stage directories, moved
checkpoints without charging another payload copy, retained K/V and old-plus-new replacement backing.
Exact reservations prevent uncharged geometric directory growth. Async host staging is bounded too;
device staging is not host RAM. The estimate is conservative, not a prediction of RSS or peak VRAM.
Pool refill is joined before planning and is not launched inside capture. Subsequent replenishment
checks its own RAM floor per checkpoint. Rejected plans restore selected running checkpoints and
release reserved pool storage; failed capture or final cache refusal also restores moved checkpoint ownership.
Selected-context cache directory allocation failures retain the image for rollback. Capture cannot publish
a partial image. Final cache byte checks remain. Disabled cache insertion order is unchanged.

Context zero/unset retains the pre-patch allocation and pool lifecycle instead: ordinary resize/directory
growth, legacy identity-vector assignment, admission before taking pool storage, no wait for refill during
planning, unused storage returned to the pool, and immediate post-capture background refill without the new
per-checkpoint RAM gate. Weight locking alone does not change these choices. The optional helper estimates
also bound baseline geometric growth rather than requiring exact reservations on this path. Those bounds
are conservative for the existing standard-library growth, not measurements of RSS or latency.

With context `0` (also the unset default), ordinary host checkpoint/conversation payloads are not selected by
this feature. **Authoritative CPU KV pools that the GPU must access are still driver-pinned.** This change does
not make those pools pageable; that requires a different access/offload design. GPU VRAM cannot use Linux swap.
Small identity/token metadata and legacy caller-owned buffers are not comprehensively selected.

The deployment and `/proc` checks below apply to either engine. Use the correct environment names and the actual
engine child PID. Successful-byte counters cover explicit tracked locks, not all driver pinning or huge pages.

## Limits and failures

The startup log gives both effective variables, soft/hard `RLIMIT_MEMLOCK` in bytes (or `unlimited`), and Linux
availability. A load summary reports successfully locked weight/context bytes and failure counts. The engine's
`INFO` lines expose current weight/context locked-byte totals without scanning `/proc` or printing per token.
Totals are **tracked locked virtual page bytes**, not unique physical RAM. Each acquisition has an immutable
engine owner. Within one engine/category overlapping ranges count once. Different categories or engines may
count the same virtual page; do not sum these totals as unique process memory. Separate file mappings may
also alias the same physical pages, which address-based accounting cannot identify. GPU-driver pinning is
not included. One process-wide registry prevents releasing one owner from unlocking another owner's pages.
INFO fields are named `weights_locked_virtual_bytes` and `context_locked_virtual_bytes`.
Counters also track successful nonempty acquisition operations (including fully overlapped acquisitions)
and failed attempts/reasons. Zero-sized and disabled acquisitions count neither successes nor failures.

A failed request reports engine, category, size, errno and its meaning. Further messages for the same engine/category/errno are
suppressed, but failure counts still advance. Failed requests fall back to pageable memory; no pages from that request
are reported locked. A partial syscall failure rolls back newly selected pages without unlocking other owners' pages.
New enabled locking paths do not change resource limits or terminate inference. Legacy PLE may first
request the approved soft-limit increase above, and still falls back if locking fails. Weight residency is only guaranteed for acquisitions
that succeeded: check the failure log and actual mappings rather than trusting the requested setting alone.

For an interactive shell, inspect the limits, then raise the soft limit within the existing hard limit:

```sh
ulimit -Sl
ulimit -Hl
ulimit -l unlimited
```

Shell limits are usually in KiB. If the hard limit prevents this, an administrator must configure an appropriate
login/PAM limit (and the user must start a new login session). The new locking paths never raise it automatically;
the approved legacy PLE exception above is the only automatic increase. These are setup instructions, not changes
performed by this implementation task.

For a systemd service, use an administrator-managed override (replace the service name with yours):

```ini
# systemctl edit strata.service
[Service]
Environment=DS4_MLOCK_WEIGHTS=1
Environment=DS4_MLOCK_CONTEXT=0
LimitMEMLOCK=infinity
```

Then run `systemctl daemon-reload` and restart the service. A bounded `LimitMEMLOCK` covering the selected allocations
is also suitable. A container needs an adequate memlock limit from its runtime; a service override cannot bypass an
outer restriction. No broad capability grant is required merely to lock within a sufficient unprivileged limit.

## Verify on Linux

Use the **engine child's PID**, not just the Python HTTP server's PID:

```sh
pid=$(pgrep -n -x strata-dsv41)
grep -E 'VmLck|VmRSS|VmSwap' /proc/"$pid"/status
grep -E 'Locked|Swap|Rss' /proc/"$pid"/smaps_rollup
grep -E '(^[0-9a-f]+-|^Locked:|^VmFlags:)' /proc/"$pid"/smaps
grep 'Max locked memory' /proc/"$pid"/limits
```

In `smaps`, successfully `mlock()`ed mappings have `lo` in `VmFlags` and nonzero `Locked:`. Compare weights-only
against an unset baseline, then create, extend, park, restore and delete conversations. With context locking off,
new host snapshots should not acquire `lo`; weight mappings should keep it. With context locking on, snapshot
payload mappings should acquire `lo`. `VmLck` is a process total, not a category breakdown. Existing driver pinning
and explicit huge pages may not be accounted for like ordinary `mlock()` pages. Reading `smaps` may require the same
user or administrative access. Do not change sysctls or force page-out to run this check.

## Tests and validation status

Latest independent-review repairs (2026-11-07): fresh CPU **102/102**, fatal UBSan **102/102**,
and ASan/UBSan lifetime **56/56**. The actual-store publication regression passed **25 consecutive
GCC TSan runs**. It queries mapping state directly during worker publication, from initially unavailable
RAM through shrink/regrowth and pending shutdown. Test-only simulated mapping is not native registration.
Removing only the query mutex in a temporary header reproduces the original `gpu_mapped()` race with
TSan exit 66; removing both query mutexes also exposes the identity-query race. The checked-in source
retains both fixes. TSan/ASan intercepted locking must not be used as residency evidence. An exploratory
TSan run of unchanged residency cases failed their lock assertions; those tests pass without those
interceptors in normal/fatal-UBSan builds and were not weakened.

Core host tests compare allocation capacities against the verified pre-patch contract at HEAD
`54828c7efbcdf9474697e70922d851f6707f4b32`, not another patched binary. They exercise the production
pool class, unused backing, baseline geometric growth, selected-context bounds and failed-move rollback.
Selected-context publication tests also inject cache directory allocation failures and check refusal rollback.
Both cache-failure cases also passed 25 consecutive GCC TSan runs (75 executions with the store test).
A companion source check pins baseline pool/refill-task hashes and checks context-only server wiring;
it is static evidence, not a native server journey. The user subsequently supplied CUDA **13.5.37**
at `/usr/local/cuda-13.5`; `/usr/local/cuda-13.5/bin` is now in the shell PATH. `nvcc --version`
and fresh native configuration succeed; earlier missing-nvcc results below are historical.
Fresh native **Core and DS4 builds both passed**, with native experts enabled and setup's pinned
llama.cpp source, targeting sm_86, sm_89, sm_120 and PTX 120. CLI help/version smoke checks passed.
Native builds emit warnings (including Core template-stub and shift-count diagnostics); they were
not suppressed and their GPU-runtime impact remains unvalidated. This is not a warning-free build.
No HIP/SYCL compiler or GPU device is available. See [handover](../handover.md) for native build results,
commands, failed/blocked checks and exact artifact paths. Ready for controlled production-machine
validation, with matching driver/GPU functional tests as the next gate, **not normal production deployment**.
Older results below are historical.

CPU-only checks (no model download):

```sh
cmake -S . -B /tmp/opencode/strata-mlock-build \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_MEMORY_LOCK_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build /tmp/opencode/strata-mlock-build -j 4
ctest --test-dir /tmp/opencode/strata-mlock-build --output-on-failure
```

The locking tests run all four combinations and unset defaults in separate processes. They check real `mlock()`
effects through `status`/`smaps`, anonymous and file-backed storage, overlapping ownership, partial-failure rollback,
growth/copy/restore/move/deletion, concurrent owners, expert-store shrink/regrow/read/cleanup, Engram blocks, and zero
soft-limit fallback. Only the test process temporarily lowers its own soft limit, then restores it.

For the synthetic fixture (one 4 KiB weight buffer and two 4 KiB context buffers), both `VmLck` and
`smaps_rollup.Locked` measured 0 / 4096 / 8192 / 12288 bytes for `0,0` / `1,0` / `0,1` / `1,1`, respectively.
These are allocation tests, not full-model measurements. All four combinations also passed with UndefinedBehaviorSanitizer,
and a subprocess check confirmed category/errno failure messages were not repeated.

Current validation (2026-11-07): see [handover](../handover.md) for final counts and commands. The CPU suite now
also runs the five conversation regressions in all four main-engine modes. It covers exact byte budgets,
eviction, retained capacity, stage parts, pool-copy growth, many small disk-read payloads, overflow and cleanup.
The file tests check the existing v1 golden size/hash in each mode; this is not an inference-identity check.

The R1–R4 rework adds `memory_lock_owners_test` (all 16 combined engine settings, both startup
orders), demand-syscall instrumentation, legacy-on/off arena/PLE tests and
`snapshot_capture_bound_test`. The last test links the actual capture and estimation sources,
copies synthetic host state through the private CUDA shim, and compares actual retained backing
against preflight, including pooled and moved checkpoints and growth. Separate helper tests cover
retained K/V and multiple stage-image directories. Neither is a real GPU/server capture journey.
Engram/store tests cover floor-limited preparation, exhausted ready capacity, shrink/regrowth,
allocation/registration failure and worker shutdown, without claiming production latency gains.
Final rework host results: **98/98 CPU tests**, **98/98 fatal-UBSan tests**, and **52/52
ASan/UBSan lifetime tests**. ASan results are not locking/residency evidence. The native build
remains blocked; these results are not production readiness. The legacy PLE exception is now approved
and restored. Its additional child-process regressions check zero/unset versus enabled behavior,
ordinary mmap/direct exclusion, hard-limit clamping, fallback and parent-limit isolation.
Fresh post-decision builds in `/tmp/opencode/strata-mlock-ple*` again passed **98/98 CPU**,
**98/98 fatal UBSan**, and **52/52 ASan/UBSan lifetime** cases. Child tests use real Linux
`getrlimit`, `setrlimit`, `mlock` and `/proc` checks; no running-shell or system-wide limits
were changed. Native CUDA configuration is still blocked by missing nvcc.

The main-engine test compiles actual arena host code, but **GPU registration is mocked**. The private CUDA shim
is guarded by `STRATA_MLOCK_HOST_TEST`; its include path is limited to test/object-check targets. Object-only
host compilation of GPU-dependent core sources is not native SDK, linkage or inference validation.

Earlier native CUDA configuration was blocked by missing `nvcc`. The user has since supplied CUDA 13.5.37
at `/usr/local/cuda-13.5`, available in PATH, and native configuration succeeds. With authorization, setup's
source-fetch helper obtained the pinned llama.cpp/ggml source at `third_party/llama.cpp` for the default
native-experts build; no packages or models were installed. Both native Core/DS4 targets and CLI smoke
checks passed. See the latest handover for commands and remaining native compiler warnings.
The actual CUDA device probe returns error 35 with zero devices; no HIP/SYCL compiler or exposed GPU was found.
Real-model load/unload, dynamic expert exchanges, server checkpoint/restore/delete, deterministic outputs
and model-scale `/proc` residency remain unvalidated.
**The production acceptance criterion has not been demonstrated.**

## Changed files

- `include/strata/platform/memory_lock.hpp`: shared interval ownership, `mlock()`/`munlock()`, counters, diagnostics,
  page-isolated allocator and uninitialized cache storage.
- `include/strata/platform/memory.hpp`: exposes the shared allocation-scoped utility.
- `src/dsv41/ds4_mlock.hpp`: DS4-only environment parsing and category allocator policies.
- `src/dsv41/ds4_model.cu`, `ds4_model.hpp`: weight selection, context checkpoint payloads, compatible slot-save
  overload, load summary and current `INFO` diagnostics.
- `src/dsv41/ds4_main.cpp`: startup and ownership-safe RAM-slot/shared-prefix payloads in server mode.
- `src/dsv41/ds4_store.hpp`: lock ownership attached to committed expert chunks, released before decommit/unmap.
- `src/dsv41/ds4_engram_io.hpp`: allocation-scoped locking of cached weight rows without extra default zero-fill.
- `src/dsv41/ds4_vision.cu`, `ds4_vision.cuh`: host vision weight ownership and small weight buffers.
- `tests/dsv41/memory_lock_test.cpp`, `CMakeLists.txt`: independent CPU-only tests and the environment matrix.
- `docs/DS4_MEMORY_LOCKING.md`, `docs/DETAILS.md`: configuration, coverage, verification and validation limits.
