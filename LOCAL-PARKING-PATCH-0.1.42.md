# Local parking patch on Strata 0.1.42

This tree is upstream `v0.1.42` (`61b3fb5d`, also `upstream/main`) plus the parking fixes from this PC's
0.1.40.1 build that 0.1.42 does not already contain. It replaces the `fork-local-baseline` branch, which
ran the engine before.

Branch: `fork-0.1.42` (from tag `v0.1.42`).
Owner of the original fix work: `D:\AI_envs\aider_olymp\project_strata_ivanov\strata_fork_main`
(branch `fork-main`), whose full write-up is `CONVERSATION_PARKING_ROOT_CAUSE.md` there.

## The failure this addresses

```
strata serve: conversation cache: allocation failed; skip parking
```

It appeared intermittently at 52k-97k-token conversations while the reported cache usage was 4.6-5.5 GiB,
well inside the configured 16 GiB, and with free physical RAM. Nothing about the conversation was lost; the
park was skipped and the next request re-read the whole prompt.

### What the log says (`strata-iq3_s.log`, 25 353 lines, `engine 0.1.40`)

Counted over the whole file by `..\project_strata_ivanov\strata_patches\log_stats_042.py`:

```
park attempts: 2481 succeeded, 291 refused -> 10.5% refused
refusal families, matched against the branches of the build that wrote this file:
    203  "conversation cache: allocation failed; skip parking"
    88   "skip parking (physical RAM admission; need N MiB plus 4096 MiB floor, or telemetry unavailable)"
     0   "skip parking (snapshot N MiB exceeds available budget)"
     0   "skip parking (physical RAM floor after capture, or telemetry unavailable)"
```

Every refusal belongs to exactly one of two branches, and both are visible in
`git show v0.1.40:src/program/generate.cpp`:

1. **88 refusals at the RAM gate, L6464.** The gate asks the OS for free bytes and requires
   `additional + 4096 MiB floor` to fit; `additional` is `estimate - retained` (L6463). Capacity was already
   granted by `make_room(estimate, held)` at L6454 - that is the cache-budget eviction path, and it refused
   **zero** times in this log. So none of these refusals were ever about the 16 GiB budget or the 4 slots; they
   were the OS saying it had no room for `new bytes + 4 GiB floor`. Note for anyone reading that line: in
   v0.1.40 `or telemetry unavailable` is a *static literal* in the format string, printed on every refusal -
   it does not say whether telemetry worked. 0.1.42 prints the measured figure instead.
2. **203 `bad_alloc` caught at L6495 (the capture) or L6506 (the wrapper around the checkpoint split).** The
   admission gate had already passed. Both handlers print one identical, region-less, cause-less line, which is
   why the failure could never be attributed to a stage. The other three `bad_alloc` catches in the file are
   session save/restore and do name themselves.

### The mechanism behind the 88

In v0.1.40 `conversation_available_memory()` returns `status.ullAvailPhys` - free *physical* only
(`git show v0.1.40:src/core/conversation_memory.cpp` line 40). The condition that failed was therefore:

```
free physical RAM  >=  (new bytes) + 4096 MiB
```

Across those 88 refusals the requested new bytes were min 531 MiB, median 2441 MiB, max 3165 MiB, and the line
carried `4096` as the floor in all 88: the gate asked for a 5.8-7.2 GiB cushion on free physical RAM every
time, in one configuration.

The same machine, measured while the engine is running (values read directly, not estimated):

```
physical total                        79.8 GiB
free physical right now               11.2 GiB
this process: private commit           87.5 GiB, working set 10.7 GiB
page file configured                 65536 MB; system commit limit 143.8 GiB, 41.7 GiB available
startup line: "expert arena: locked 40290 MiB via working-set minimum + VirtualLock"
```

A 40 GiB arena held by a working-set minimum plus `VirtualLock`, plus WDDM device allocations, is what that
87.5 GiB of private commit is: the process commits far past physical RAM on purpose. With that in place free
physical measured 11.2 GiB just now, and the gate's own demand on this workload reaches 7.2 GiB - a margin of
a few GiB on a resource that moves by gigabytes per request while a 100k-token prompt is read and decoded. I
did not sample free physical at the instant of each historical refusal (the log does not record it), so the
per-refusal dip is not measured; what is measured is that the gate and the demand were within a few GiB of
each other on this machine, and that 88 requests lost that race. That is why "Windows still has substantial
free RAM" never prevented a refusal: the check is not about whether the machine has RAM, it is about a fixed
4 GiB cushion on the one resource the engine itself had already committed.

Both explanations people reach for are excluded by one measurement each:

```
                          median      p90        max
cache bytes, refused:     4.62 GiB    7.52 GiB   12.09 GiB
cache bytes, succeeded:   4.49 GiB    6.94 GiB   12.09 GiB   <- same maximum, overlapping
slots occupied, refused:  1:77  2:66  3:81  4:67              <- every slot count fails
```

Token count correlates but is not a threshold: 7/339 under 50k (2%), 106/1293 at 50k-100k (8.2%), 178/1140
above 100k (15.6%) - a rising rate on one code path, not a cutoff. A 108k-token conversation parked in 182 ms
two lines before a 79k-token one was refused.

**Why no configuration change could have moved this.** The settings that were tuned are the budget
(`--conversation-cache-mib 16384`), the slots (`--conversation-cache-slots 4`) and the floor
(`--conversation-cache-min-free-mib 4096`). In the build that produced this log the budget branch never fired,
and the slot count only appears in the *success* distribution. The floor is the one number in the failing
condition of the 88 - and it is a subtracted margin on a measurement, so lowering it moves the threshold but
leaves the cause; raising it makes more requests fail. The 203 are thrown by the copy itself and were not
reported with a size or a cause, so no setting could be shown to relate to them at all.

`engine/BUILD.json` of the process that wrote this log reads `"version": "0.1.40", "source": "release"` - the
stock release binary, 140 MB, still containing the old string and none of the new ones. No parking fix has run
on this machine yet.

## What 0.1.42 already fixes, so it was not re-applied

`src/core/conversation_memory.cpp` line 286 admits on
`min(ullAvailPhys, ullAvailPageFile)` under a `#1607` comment: *the commit limit (RAM + page file) refuses
an allocation while physical RAM is still free*. That is the primary cause found on this machine, and
upstream now measures it. Raising the page file or lowering `--conversation-cache-min-free-mib` cannot fix
that cause on its own, which is why the earlier configuration experiments on this PC did not either.

`src/program/generate.cpp` also gained the physical-RAM admission gate with an evict-until-it-fits loop and
`--memory-limit-mib`. Those are upstream's, kept as-is.

## What was still broken in 0.1.42, and the port

13 edits: 5 source files (+178/-21 lines) and 2 test files (+85). Verified by `git show --numstat` on the
local commit `0b0ae6be` on branch `fork-0.1.42`.

### 1. The eviction loop bounds itself by what cannot be freed — `conversation_cache.hpp`, `generate.cpp`

Upstream's loop (`v0.1.42` only - `git show v0.1.42:src/program/generate.cpp` line 7802; v0.1.40 has no such
loop, which is why the 0.1.40 log above shows no evictions at all) is bounded by `conversations.slots()` and
gated on `conversations.size() > 0`, and it calls `evict_oldest()` unconditionally. Two categories of memory
are charged in `bytes()` but are **not** `evict_oldest()` material: the retained K/V (`reuse_`, see
`conversation_cache.hpp` line 207, `bytes() { return bytes_ + reuse_.bytes(); }`) and entries holding a pinned
shared prefix (`evict_oldest()` skips them, line 284). So a park asking for more than the evictable total
spends conversations one at a time, evicts everything it can, and still refuses at the end - the 0.1.40.1 fork
log that this analysis started from shows `evicted 3, 0 still parked` for exactly such a refusal. This
(defect B in `strata_fork_main/CONVERSATION_PARKING_ROOT_CAUSE.md`) is upstream's code, not the port's.

Added `ConversationCache::evictable_bytes()` (mirrors `evict_oldest()`'s own filter), and the loop now:

* refuses immediately when the new bytes exceed what could be freed (`hopeless`);
* stops as soon as one eviction failed to make the short resource larger (`stalled`) — the Windows case,
  where a locked working set and WDDM device allocations keep commit charged while parked entries are dropped;
* skips the loop entirely when the park needs no new bytes (`additional == 0`, a park reusing all of its
  K/V): no parked conversation is what the floor is asking for, so the eviction round is pure loss.

The refusal line now names the reason: `no new bytes were requested` / `freeing parked conversations did not
return memory to the OS` / `freeing every parked byte that could be freed (N MiB) could not have fitted it,
so nothing was evicted` / `the conversations still parked would leave it short by N MiB`.

### 2. A budget refusal threw away the retained K/V — both `generate.cpp`

`park_current_body` starts with `auto reuse = conversations.take_reuse();`. A refusal from
`make_room()` (the cache's own byte budget) returned without putting it back, so the *next* park of the same
conversation re-copied the entire conversation instead of reusing it. `ConversationCache::put_reuse()` hands
it back, bounded by the same budget `retain()` uses. On a 74k-token session this is the difference between
reusing the K/V and re-allocating about 6.4 GiB.

### 3. The failure line named no region and no cause — both `generate.cpp`

Four `catch (const std::bad_alloc&)` handlers printed the same textless line. Two of them sit before any
admission gate (the checkpoint split, the stage byte estimates), so the line could not be attributed to a
stage. Now:

* the capture handler says `allocation failed at the capture (N of M MiB was new)` and adds `e.what()`;
* the outer handler says `skip parking after the checkpoint split (bad_alloc: ...)`;
* each gained a `catch (const std::exception&)` so an internal error is no longer reported as an allocation
  failure — item 9 in the investigation brief ("an unrelated internal error incorrectly translated");
* `conversation_memory_probe()` (new, `conversation_memory.hpp`/`.cpp`) re-asks the machine for the failed
  size right in the handler and reports which of the three cases it is: `commit is exhausted`,
  `the address space itself cannot be reserved`, or `the same size allocates now, so the failure was
  transient or came from elsewhere`. A zero-byte request is not probed and says so.

### 4. Refusals did not say which resource was short — `generate.cpp`, `ram_note()`

On Windows the guard takes the smaller of free RAM and free commit. The line printed only the minimum. It now
prints both, plus this process' own commit and working set:

```
skip parking (memory admission; need 1536 MiB plus 4096 MiB floor, 2210 MiB available, source meminfo;
physical RAM has 19120 MiB free, the system commit (RAM plus page file) has 2210 MiB free;
this process commits 71512 MiB (72040 peak), working set 58210 MiB; evicted 0, 2 still parked;
freeing every parked byte that could be freed (4510 MiB) could not have fitted it, so nothing was evicted)
```

That shape — tens of GiB free RAM, a few GiB of commit — is the whole story on two RTX 5060 Ti cards under
WDDM with a 40 GiB locked expert arena: the arena and the driver's device allocations are charged against
the commit limit, not against RAM.

## Not ported, on purpose

* The 0.1.40.1 `MemoryShort`/`admit_by_evicting`/`conversation_memory_shortage` helpers. 0.1.42 rewrote
  this area around `HostMemoryReading`; the two implementations would have fought over the same decision.
  Only the pieces 0.1.42 still lacks were ported, in its types.
* `serve/` Monitor conversation-id column and the run-config files from the old branch. They do not apply
  cleanly to 0.1.42 (`serve/server.py`, `test_server.py`, `app.js`, `index.html` all moved); if wanted they
  need re-authoring against the new files, not a cherry-pick.
* `AGENTS.md`: the uncommitted local block is stashed (`git stash list` → `agents-md-before-042`) because
  0.1.42 added its own section at the same spot. Pop it and keep both sections.

## Build and run

The old `build_cuda`/`build_tests` caches are configured for the previous path and cannot be reused
(CMake refuses the re-rooted cache). Configure fresh, reusing the already-fetched llama.cpp:

```
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE=C:\Python310\Lib\site-packages\cmake\data\bin\cmake.exe
%CMAKE% -S . -B build_cuda_new -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON ^
  -DSTRATA_PORTABLE=OFF -DSTRATA_WERROR=OFF ^
  -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=D:/AI_envs/aider_olymp/project_strata/build_cuda/_deps/strata_llamacpp-src
%CMAKE% --build build_cuda_new -j 6
```

Scripts that do this in one step: `..\project_strata_ivanov\strata_patches\build_cuda_042.bat` (engine),
`build_tests_042.bat` (CPU test build, `-DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON`),
`syntax_042.bat` (`cl /Zs` syntax-only check of the CUDA park path).

The model files stay in `project_strata_temp`; this tree is the source and build tree.

## Tests run on this tree

```
ctest --test-dir build_tests                        -> 100% tests passed out of 43   (2 skipped: CPU-only build)
build_tests\conversation_cache_test.exe             -> conversation_cache_test: 4225 checks passed
build_tests\conversation_memory_test.exe            -> conversation_memory_test: 32 checks passed
build_tests\conversation_split_failure_test.exe     -> stages 3: allocation-failure cases preserved: 16
syntax_042.bat  (cl /Zs, CUDA park path)            -> CL_EXIT=0
build_cuda_042.bat (full CUDA engine)               -> [263/263] BUILD_EXIT=0, build_cuda_new\strata.exe -> "strata 0.1.42"
```

Verified in the built binary, against the running 0.1.40.1 one:

| string | new `build_cuda_new\strata.exe` | running `project_strata_temp\engine\strata.exe` |
|---|---|---|
| `allocation failed; skip parking` (the textless line) | absent | present |
| `allocation failed at the capture` | present | absent |
| `skip parking after the checkpoint split` | present (both park copies) | absent |
| `memory admission` | present | absent |
| `commit is exhausted` | present | absent |

The new regression tests are in `src/core/conversation_cache_test.cpp` (+11 checks: `evictable_bytes()`
counts exactly what `evict_oldest()` can free, pinned entries excluded; `put_reuse()` round-trips, refuses an
over-budget candidate, ignores an empty one) and `src/core/conversation_memory_test.cpp` (+7 checks: the probe
says nothing for 0 bytes, confirms a size the machine has room for, and its text distinguishes commit
exhaustion from address-space failure). Both fail without the patch, which is the point of them.

## Independent review and the changes it asked for

An independent senior reviewer (`.openhands/agents/senior-reviewer.md`, conversation
`a36c2bcd-bc38-43a5-a55f-7096f56df07b`) read the diff. It accepted the diagnosis and the port as sound, and
raised three concrete notes. All three were checked against the code and acted on:

1. **`put_reuse()`'s "only if nothing is held" guard.** The reviewer called the `reuse_.kv.empty()` condition
   dead weight: a park reaches `put_reuse()` only after `take_reuse()` has cleared `reuse_`, and every other
   caller of `put_reuse()` (session restore in `generate.cpp`, the engine's own paths) also runs on a cleared
   candidate. Left in, it would silently drop a hand-back in a future call site that does hold one. The guard
   is removed; the doc comment now states the invariant instead of enforcing it. The test for this changed from
   "a second candidate does not replace the one held" to "a newer candidate replaces the one held, as with
   `retain()`", and the round-trip and "an empty candidate is not stored" checks now assert against the newer
   candidate's size.
2. **A UTF-8 BOM had appeared at the start of `src/program/generate.cpp`.** It came from an editor, not the
   patch. The file is otherwise pure ASCII (3 non-ASCII bytes, all the BOM; `v0.1.42` had 0), the CMake build
   adds no `/utf-8` flag, and MSVC accepts a BOM - so this changed no behaviour, but it made the local file
   differ from upstream for no reason. The BOM is stripped: the file now has 0 non-ASCII bytes, matching
   upstream, and the SYCL copy was already BOM-free with 0 non-ASCII bytes. No charset flag was added.
3. **The probe leaked its reservation on the path that matters.** `conversation_memory_probe()` only reached
   `VirtualFree` on success, so every failed *commit* left the `MEM_RESERVE` range held. The reviewer noted
   this as a comment-worthy detail; it is cheap to make correct instead, so the failure paths now release the
   reservation before returning (`VirtualFree` after a failed commit on Windows, `munmap` after a failed
   `mprotect` on Linux). Repeated refusals no longer each keep the probed size of address space.

Two review points needed no change. The reviewer asked whether the SYCL copy's skip-parking wording was as
precise as the CUDA one; `git grep` on the current tree shows the SYCL text already says "not an allocation",
names `bad_alloc`, and covers the checkpoint split, so nothing was edited there. The reviewer also confirmed
the stalled-eviction loop and the hoisted `put_reuse` lifetime are correct as written.

Re-run after these changes (same commands as above):

```
ctest --test-dir build_tests                        -> 100% tests passed out of 43   (2 skipped: CPU-only build)
build_tests\conversation_cache_test.exe             -> conversation_cache_test: 4225 checks passed
build_tests\conversation_memory_test.exe            -> conversation_memory_test: 32 checks passed
build_cuda_042.bat / cmake --build build_cuda_new --target strata -> [7/7] BUILD_EXIT=0, "strata 0.1.42"
```

`build_cuda_new\strata.exe` after the re-review still contains `allocation failed at the capture`,
`skip parking after the checkpoint split`, `could not have fitted it`, `freeing every parked byte` and
`reserves but will not commit`, and still does not contain the old `allocation failed; skip parking`.
The non-ASCII audit of both park copies reports `non_ascii_bytes=0` and `bom=False`.

## Limitations

* **No end-to-end run yet.** Measured on this PC at the time of writing: the user's live server is
  `serve\server.py --port 8080` (PID 7092) with its engine child (PID 9432) running
  `project_strata\engine\strata.exe`, which is the 0.1.40 binary still containing the textless
  `allocation failed; skip parking`. Both RTX 5060 Ti are at ~15.0 GiB of 16.3 GiB each and 11.3 GiB of
  79.8 GiB RAM is free, so a second engine cannot be started alongside it. The 50k-97k-token park/restore
  demonstration needs that process stopped (the user's call) and this build started with the same arguments;
  until it is run, no e2e result is claimed here. Installing the new engine means copying
  `build_cuda_new\strata.exe` over `engine\strata.exe` while the server is down - not done in this session.
* **SYCL is not compiled.** Both park copies were edited (`src/program/generate.cpp` and
  `sycl/src/program/generate.cpp` - `git grep park_current_body v0.1.42` finds exactly these two), but no Intel
  oneAPI toolchain is installed on this machine, so `sycl/src/program/generate.cpp` was reviewed by hand and not
  compiled. The edits there are the same three as the CUDA side minus the probe, and use no SYCL-specific API.
  The header fixes do reach the SYCL build: `sycl/CMakeLists.txt:110` adds `${STRATA_ROOT}/include` to the
  include path, `sycl/include/strata` has no `core/` override, and the engine target lists
  `src/core/conversation_memory.cpp` (L221), which `strata_resolve` falls back to the shared root copy for
  (L98). What the SYCL park copy still lacks is only the CUDA side's richer admission wording (it prints
  "physical RAM admission" without the probe/eviction detail); the behaviour fixes - `evictable_bytes()` in
  `make_room` and the `put_reuse()` hand-back - are shared code. HIP has no separate park copy.
* The Windows commit pressure itself is not removed by this patch — the engine still commits tens of GiB for
  the locked arena. What changed is that the park decision now reads the right number, stops spending
  conversations it cannot spend, keeps the K/V it already had, and says which limit refused.
