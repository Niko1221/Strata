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

Upstream's loop is bounded by `conversations.slots()` and gated on `conversations.size() > 0`, and it calls
`evict_oldest()` unconditionally. Two categories of memory are charged in `bytes()` but are **not**
`evict_oldest()` material: the retained K/V (`reuse_`, see `conversation_cache.hpp` line 207,
`bytes() { return bytes_ + reuse_.bytes(); }`) and entries holding a pinned shared prefix
(`evict_oldest()` skips them, line 284). So a park asking for more than the evictable total spends
conversations one at a time, evicts everything it can, and still refuses at the end — the log shows
`evicted 3, 0 still parked` for a refusal that could not have been won.

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

## Limitations

* **No end-to-end run yet.** The model is loaded in `project_strata_temp` (its engine still runs), and this PC
  has ~10 GiB free with it up, so a second engine cannot start alongside it. The 50k-97k-token park/restore
  demonstration has to be done by stopping that process and starting this build; until it is, no e2e result is
  claimed here.
* **SYCL is not compiled.** Both park copies were edited (`src/program/generate.cpp` and
  `sycl/src/program/generate.cpp` - `git grep park_current_body v0.1.42` finds exactly these two), but no Intel
  oneAPI toolchain is installed on this machine, so `sycl/src/program/generate.cpp` was reviewed by hand and not
  compiled. The edits there are the same three as the CUDA side minus the probe, and use no SYCL-specific API.
  HIP has no separate park copy.
* The Windows commit pressure itself is not removed by this patch — the engine still commits tens of GiB for
  the locked arena. What changed is that the park decision now reads the right number, stops spending
  conversations it cannot spend, keeps the K/V it already had, and says which limit refused.
