# Local parking patch on Strata 0.1.43

This tree is upstream `upstream/main` (`7695859a`, tag `v0.1.43`) plus the parking fixes this PC's 0.1.40.1 build
already had and 0.1.43 does not. The patch was written against `v0.1.42` and rebased onto 0.1.43; it applies with
no conflicts, and 0.1.43 did not fix any of what it addresses - the loop it replaces is still upstream's, at
`git show v0.1.43:src/program/generate.cpp` 7915-7930, and the unattributed failure line is still at 7966 and
7975 there.

Branch: `fork-0.1.42`, rebased onto `park-0.1.43` (a branch at `upstream/main`). Rebased with `git rebase --onto
park-0.1.43 v0.1.42 fork-0.1.42`; all ten commits applied cleanly.
Owner of the original fix work: `D:\AI_envs\aider_olymp\project_strata_ivanov\strata_fork_main`
(branch `fork-main`), whose full write-up is `CONVERSATION_PARKING_ROOT_CAUSE.md` there.

Upstream's 0.1.43 does touch this cache in one place: `ConversationCache::clear()` (#879, non-finite logits) drops
every parked conversation and the retained K/V. It is a poison purge, not an admission path, so it needs none of
this patch's handling - and it is consistent with it, since `clear()` zeroes `reuse_` and `evictable_bytes()` never
counted it.

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

## What upstream already fixes, so it was not re-applied

`src/core/conversation_memory.cpp` admits on
`min(ullAvailPhys, ullAvailPageFile)` under a `#1607` comment (v0.1.43 line 286 of the unpatched file): *the
commit limit (RAM + page file) refuses
an allocation while physical RAM is still free*. That is the primary cause found on this machine, and
upstream now measures it. Raising the page file or lowering `--conversation-cache-min-free-mib` cannot fix
that cause on its own, which is why the earlier configuration experiments on this PC did not either.

All of this held in 0.1.42 and still holds in 0.1.43: nothing in the 72 commits between the tags changed the
park path, the admission helper, or `ConversationCache` apart from the `clear()` noted at the top.

`src/program/generate.cpp` also gained the physical-RAM admission gate with an evict-until-it-fits loop and
`--memory-limit-mib`. Those are upstream's, kept as-is.

## What was still broken, and the port

7 files, +529/-35, read with `git diff --numstat upstream/main` on this tree: the two headers and
`conversation_memory.cpp` (the admission helper), both `generate.cpp` copies, and the two test files.

### 1. The eviction loop bounds itself by what cannot be freed — `conversation_cache.hpp`, `generate.cpp`

Upstream's loop (the `while` at `git show v0.1.43:src/program/generate.cpp` lines 7915-7930; it is new since
v0.1.40, which is why the 0.1.40 log above shows no evictions at all) is bounded by `conversations.slots()` and
gated on `conversations.size() > 0`, and it calls `evict_oldest()` unconditionally. Two categories of memory
are charged in `bytes()` but are **not** `evict_oldest()` material: the retained K/V (`reuse_`; `bytes()` is
`bytes_ + reuse_.bytes()` in `conversation_cache.hpp`, and `evict_oldest()` never touches `reuse_`) and entries
holding a pinned shared prefix (`evict_oldest()`'s `find_if` picks only `!e.pinned()`). So a park asking for more than the evictable total
spends conversations one at a time, evicts everything it can, and still refuses. That last claim is read off the
code, not off the log - the 0.1.40 log cannot show it, because that engine never ran the loop. Section 1b runs
both loops against one state and asserts the difference. This is defect B in
`strata_fork_main/CONVERSATION_PARKING_ROOT_CAUSE.md`, and it is upstream's code, not the port's.

Added `ConversationCache::evictable_bytes()` (mirrors `evict_oldest()`'s own filter), and the loop now:

* spends nothing when the new bytes exceed what could be freed (`hopeless`) - and says so only if a reading has
  already refused the request, because the reading, not the arithmetic, is what decides (bullet below);
* stops as soon as one eviction failed to make the short resource larger (`stalled`) — the Windows case,
  where a locked working set and WDDM device allocations keep commit charged while parked entries are dropped;
* skips the loop entirely when the park needs no new bytes (`additional == 0`, a park reusing all of its
  K/V): no parked conversation is what the floor is asking for, so the eviction round is pure loss.

The refusal line now names the reason: `no new bytes were requested, so evicting parked conversations could not
change the answer` / `freeing parked conversations did not return memory to the OS, so it stopped` / `freeing
every parked byte that could be freed (N MiB) could not have fitted it` / `the conversations still parked would
leave it short by N MiB`. No arm claims how many were evicted - the line already prints `evicted %zu, %zu still
parked`, and an added clause would go stale when the loop evicted and then failed (review point 2).

The rules are not inline in the caller any more. `evict_for_admission()` (new, `conversation_memory.hpp`,
templated on the cache so the real `ConversationCache` can be driven without a model) runs the round and returns
`{admitted, evicted, hopeless, stalled, freeable, short_by}`; `src/program/generate.cpp` only prints what it
answered. Two things that changed with the move, both deliberate:

* the answer is always taken from a reading, never from the arithmetic alone - `evict_for_admission()` samples
  first and reports `admitted`, where the first cut of this patch decided `hopeless` and `additional == 0`
  without sampling, so a refusal line could have been printed for a park the machine would have taken;
* `short_by` and `freeable` describe the round that refused, so the text cannot describe a different attempt
  than the one being reported. Within a round, `freeable` is the total at the point the round decided it could
  not win, and `short_by` is taken from the newest reading that round obtained - after every eviction the
  helper re-samples before it can end, so no eviction is ever reported against an older reading than itself.

The move itself was then checked rather than asserted: `conversation_cache_test.cpp` runs the loop exactly as
upstream still has it (`git show v0.1.43:src/program/generate.cpp` 7915-7930) and
`evict_for_admission()` over a twelve-case table of readings (flat, improving, improving-then-fitting, worsening,
a request above the whole cache, an empty cache, a roomy reading, one entry with one byte needed,
`additional == 0` twice, and telemetry unavailable twice), each case with its own cache and its own reading
program. The decision fields - admitted, evicted, stalled, and what is left parked - agree on all twelve, and so
do the numbers a refusal line prints on the nine that entered the loop. They differ on exactly the three the old
code skipped by arithmetic before sampling - the `hopeless` veto, `additional == 0`, and an empty cache (which its
arithmetic also called hopeless, at 0 freeable bytes): there the helper records a `short_by` the old code left at
0. That difference cannot reach the log, because those three take the `hopeless` / `no new bytes were requested`
arms, which print `freeable` (identical) and no shortfall clause. So the refusal line the two versions print is
the same in all twelve; what differs is only how many times the machine is asked.

The SYCL copy (`sycl/src/program/generate.cpp` line 7500) has **no eviction loop** - it calls
`conversation_memory_admit(available_host_bytes(), ...)` once and refuses. So item 1's loop rules apply to the
CUDA path only. What both copies share is the code, not the wording: `evictable_bytes()` and `put_reuse()` are
cache members used by `make_room()`, and the `put_reuse()` hand-back on a budget refusal is in both park copies.
The four named admission causes and the `ram_note()` figure live in `src/program/generate.cpp` only; SYCL still
prints its two generic lines, `physical RAM admission ... or telemetry unavailable` (line 7502) and `physical RAM
floor after capture, or telemetry unavailable` (line 7525).

### 1b. What the two loops do to the same cache - the regression test

`src/core/conversation_cache_test.cpp` parks three 8 MiB conversations and runs both loops against that one state
with a reading that never changes (the Windows case: commit stays charged while parked entries are dropped). What
the two loops are asserted to do on that state:

```
old loop (upstream v0.1.43, size()/slots() bound): admitted=0 evicted=3 still_parked=0
new loop (evict_for_admission):                    admitted=0 evicted=1 stalled=1  2 still parked
```

The test prints nothing; those are the values its checks assert, not an output it produced. Nothing is retained in
that state on purpose - the upstream loop reads neither `evictable_bytes()` nor the retained K/V, and
`evict_oldest()` never touches `reuse_`, so a retained candidate would not change what that loop does. The
retained-K/V cases are asserted separately, against the helper.

The first line is the failure this fix was written against, now asserted against both loops rather than argued
from the source: the old loop spends every conversation it can and still refuses, and the log cannot show it
because the 0.1.40 engine that produced the log had no loop. `evict_for_admission()` is additionally tested for:
the flat reading stopping after one eviction and asking the machine twice rather than once per entry; a request
above the freeable total being refused with nothing spent and `freeable != bytes()` (retained K/V is charged but
is not freeable); evictions that *do* return memory being accepted and the round stopping at the first reading
that fits; a park that requests no new bytes leaving the cache untouched while still honouring the floor, with
the following attempt admitted once the floor is free; and a cache holding one unpinned and one pinned entry,
where a request only the pinned entry could cover spends nothing at all, and a request the freeable entries can
just cover spends only the unpinned one and leaves the pin parked.

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
prints both, plus this process' own commit and working set. The line below is verbatim from the end-to-end run
recorded further down ("End-to-end run on the built binary") - the single refusal in 164 parks:

```
conversation cache: skip parking (memory admission; need 3966 MiB plus 4096 MiB floor, 6900 MiB available,
source meminfo; physical RAM has 6900 MiB free, the system commit (RAM plus page file) has 36771 MiB free;
this process commits 90556 MiB (95414 peak), working set 11672 MiB; evicted 0, 1 still parked;
freeing every parked byte that could be freed (2152 MiB) could not have fitted it)
```

That is the `hopeless` arm, and it is the case the fix exists for: 3966 MiB was requested and everything the
cache could free was 2152 MiB, so no amount of evicting could have made the park fit. Upstream's loop would have
dropped that conversation anyway and still refused, and the line would have carried no number at all. Note what
this line does *not* blame: here the short resource is free RAM (6900 MiB) while commit is plentiful
(36771 MiB) - the opposite of the 0.1.40 mechanism described above. The reporting is the same path either way;
it names whichever one actually refused.

The 0.1.40 refusals printed no numbers at all: counting the archived log this patch was written against
(`strata-iq3_s.log`, 33374 lines, `engine 0.1.40`) finds 293 `skip parking` lines - 203 of the bare textless
`allocation failed; skip parking`, and 90 of the budget refusal ending in a static `or telemetry unavailable`.

## Not ported, on purpose

* The 0.1.40.1 `MemoryShort`/`admit_by_evicting`/`conversation_memory_shortage` helpers. 0.1.42 rewrote
  this area around `HostMemoryReading`; the two implementations would have fought over the same decision.
  Only the pieces 0.1.42 still lacks were ported, in its types.
* `serve/` Monitor conversation-id column and the run-config files from the old branch. They do not apply
  cleanly to 0.1.42 (`serve/server.py`, `test_server.py`, `app.js`, `index.html` all moved); if wanted they
  need re-authoring against the new files, not a cherry-pick.
* `AGENTS.md`: 0.1.42 added its own section where the local block sat, so the local block was re-added after
  it rather than replacing it - kept short, pointing here for detail. My own commit had written that block as
  UTF-8 with a BOM and two mojibake em-dashes; `f9f48a8f` removed the BOM, used plain `-`, and spaced the
  heading. The file is now ASCII with the repository's CRLF endings.

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
build_tests\conversation_cache_test.exe             -> conversation_cache_test: 4243 checks passed
build_tests\conversation_memory_test.exe            -> conversation_memory_test: 32 checks passed
build_tests\conversation_split_failure_test.exe     -> stages 3: allocation-failure cases preserved: 16
syntax_042.bat  (cl /Zs, CUDA park path)            -> CL_EXIT=0
syntax_sycl_042.bat                                 -> SYCL_EXIT=9009 (no oneAPI toolchain on this machine)
build_cuda_042.bat (full CUDA engine)               -> [263/263] BUILD_EXIT=0, build_cuda_new\strata.exe -> "strata 0.1.42"
```

Run again after the `evict_for_admission()` move, same commands: `ctest` 43/43, `conversation_cache_test` 4258
checks, `syntax_042.bat CL_EXIT=0`, and `build_cuda_042.bat` rebuilt to `BUILD_EXIT=0` with `strata 0.1.42` and the
same string table below. The 4225 -> 4258 difference is the loop test: 4225 was the count with the loop rules still
inline in `generate.cpp` and no way to drive them.

Run a third time, after the review notes of the second review pass were acted on (the equivalence table compared
its reporting fields as well, the inert retained-K/V state was dropped from the old-loop copy, one pinned/unpinned
round was added, and the header's dead `!evict_oldest()` bound was documented):

```
cmake --build build_tests                           -> [9/9] BUILD_EXIT=0
ctest --test-dir build_tests                        -> 100% tests passed out of 43   (2 skipped: CPU-only build)
build_tests\conversation_cache_test.exe             -> conversation_cache_test: 4281 checks passed
build_tests\conversation_memory_test.exe            -> conversation_memory_test: 32 checks passed
build_tests\conversation_split_failure_test.exe     -> stages 3: allocation-failure cases preserved: 16
syntax_042.bat  (cl /Zs, CUDA park path)            -> CL_EXIT=0
syntax_sycl_042.bat                                 -> SYCL_EXIT=9009 (no oneAPI toolchain on this machine)
build_cuda_042.bat (incremental CUDA engine)        -> [9/9] BUILD_EXIT=0, build_cuda_new\strata.exe -> "strata 0.1.42"
```

A count attributed to one block, not inferred: with a temporary `printf` at the top of that block the binary
printed `BLOCKSTART 4225` and `4281` at the end, so the loop-level cases of 1b are 56 checks; the file is
unchanged otherwise (the probe was removed and the tree rebuilt and re-run).

Clean build and diff checks, on the same tree:

```
cmake -S . -B build_tests_clean (fresh dir)         -> CFG_EXIT=0
cmake --build build_tests_clean -j 8                -> [131/131] CLEAN_BUILD_EXIT=0, 0 errors
                                                       1 warning: dequant.cpp(41) C4456 - upstream, untouched here
build_tests_clean\conversation_cache_test.exe       -> 4281 checks passed
build_tests_clean\conversation_memory_test.exe      -> 32 checks passed
ctest --test-dir build_tests_clean                  -> 100% tests passed out of 43 (2 skipped)
git diff --check                                    -> clean (no whitespace errors)
non-ASCII / BOM audit of the three changed files    -> test and header 0 non-ASCII bytes, no BOM, CRLF;
                                                       the doc keeps its existing em-dashes, no BOM, CRLF
```

After the last comment-only edit to the test file, `build_tests_clean` was recompiled (`BUILD_EXIT=0`, no new
warnings) and re-run: 4281 / 32 checks, `ctest` 43/43. The `[131/131]` figure is the from-scratch pass; the
figures above it are what the final source produces in either directory.

### Re-run after rebasing onto upstream 0.1.43

The rebase changes no source of this patch (the net diff against `upstream/main` is the same seven files as
against `v0.1.42`); the check counts move only because 0.1.43 added its own tests to `conversation_cache_test.cpp`
(#879's `clear()`, five checks), so 4281 + 5 = 4286. Fresh configure, fresh directory:

```
cmake -S . -B build_043 -G Ninja (fresh dir)          -> CFG_OK
  note: STRATA_BUILD_CONVERSATION_TESTS=ON is required - the parking tests are behind that option,
        and STRATA_BUILD_TESTS=ON alone configures with no test targets at all.
cmake --build build_043 -j 8                          -> BUILD_EXIT=0, 85 objects, 1 warning
                                                        (dequant.cpp(41) C4456 - upstream, untouched here)
build_043\conversation_cache_test.exe                 -> conversation_cache_test: 4286 checks passed
build_043\conversation_memory_test.exe                -> conversation_memory_test: 32 checks passed
build_043\conversation_split_failure_test.exe         -> stages 1/2/3 allocation-failure cases preserved: 8/12/16
ctest --test-dir build_043                            -> 100% tests passed out of 43 (2 skipped: CPU-only)
git diff --check upstream/main HEAD                   -> clean
cmake --build build_cuda_120 -j 6                     -> CUDA_EXIT=0, no errors
build_cuda_120\strata.exe --version                   -> strata 0.1.43
  string audit of that exe: "allocation failed; skip parking" ABSENT; "allocation failed at the capture",
  "skip parking after the capture", "skip parking after the checkpoint split", "memory admission",
  "could not have fitted it", "freeing every parked byte", "reserves but will not commit" all present.
```

Two reference fixes came with the rebase, for the reason AGENTS.md records: a branch-local SHA stops naming the
blob it points at as soon as the branch moves. The test comment and two doc passages that cited `3ae6a9ec` now
cite `git show v0.1.43:src/program/generate.cpp` 7915-7930, and `git cat-file -t` confirms both `3ae6a9ec` and
the numstat commit `0b0ae6be` are no longer reachable from any branch.

Verified in the built binary, against the running 0.1.40.1 one:

| string | new `build_cuda_new\strata.exe` | running `project_strata_temp\engine\strata.exe` |
|---|---|---|
| `allocation failed; skip parking` (the textless line) | absent | present |
| `allocation failed at the capture` | present | absent |
| `skip parking after the checkpoint split` | present (both park copies) | absent |
| `memory admission` | present | absent |
| `commit is exhausted` | present | absent |

The new regression tests are in `src/core/conversation_cache_test.cpp` (+67 checks over `v0.1.42`'s own test file,
measured by compiling that file against the patched headers: 4214 upstream, 4225 with the `evictable_bytes()` and
`put_reuse()` block, 4281 with the 56 loop-level checks of 1b) and
`src/core/conversation_memory_test.cpp` (+7 checks: the probe
says nothing for 0 bytes, confirms a size the machine has room for, and its text distinguishes commit
exhaustion from address-space failure). Both are checked against the patch, not just claimed to need it: compiled
against the unpatched `v0.1.42` headers they do not even build (`..\project_strata_ivanov\strata_patches\prepatch_compile_042.bat`,
which puts `git show 61b3fb5d:include/strata/core/*.hpp` on the include path first):

Re-run against the tree as it stands now (same script, same unpatched headers):

```
conversation_cache_test.cpp(394) error C2039: 'evictable_bytes' is not a member of ConversationCache   (83 errors, CACHE_CL_EXIT=2)
conversation_memory_test.cpp(69) error C3861: 'conversation_memory_probe' identifier not found         (33 errors, MEMORY_CL_EXIT=2)
```

The cache test fails on `evictable_bytes`/`put_reuse`, which only the patch adds - that error count went from 7
to 49 when the loop test of 1b first arrived, and to 83 once the equivalence table and the pinned round call
`evictable_bytes()` more often, because all of it calls functions only the patch adds. Line numbers move with the
test file; the member errors do not.

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

Two review points needed no change, and both were checked rather than assumed. The reviewer wrote that the SYCL
inner handler said "after the checkpoint split" where it should say "at the capture"; the blob it reviewed
(`git show a8db42e6:sycl/src/program/generate.cpp`) already has the same split as the CUDA side - the inner
handler prints "allocation failed at the capture", the outer `park_current` wrapper prints "after the checkpoint
split" - and no commit between `a8db42e6` and HEAD touches that file, so the note describes a state that does not
exist. The reviewer also confirmed the parts most at risk of being wrong: the `evictable_bytes()` bound is
"correct and free of an off-by-one", hoisting `floor`/`retained`/`additional` above the `try` moves no work in or
out of the guarded region and reports the same value admission used, the `put_reuse()` hand-back cannot
double-count or leak, no real allocation failure is masked by the `catch (const std::exception&)` arm
(`bad_alloc` precedes it in both handlers), and the probe cannot throw. On the diagnosis it agreed in terms of
the log: with the arena already 100 % committed behind 39.4 GiB of `VirtualLock`, `ullAvailPageFile` sits near
4 GiB while `ullAvailPhys` still reads free - "so evicting parked conversations cannot restore the short
resource, which is exactly the stalled case".

### What the reviewer left unproven

These are the reviewer's own stated uncertainties, kept here because they bound what this patch claims:

* It is not proven that the observed 52k-97k refusals were commit-limited rather than heap-metadata or
  fragmentation exhaustion, nor that they came from the capture rather than the estimate/split region - the new
  wording is what decides that on the next run.
* A successful `VirtualAlloc` commit is not identical to a successful CRT `new[]` (heap bookkeeping, the failing
  call's own transient copies), so "the same size allocates now" is an inference; the message's own "transient or
  came from elsewhere" wording hedges it.
* The stalled heuristic's sensitivity to `ullAvailPhys` noise is unquantified.
* Whether `conversation_snapshot_save` can throw anything that is not `bad_alloc` was not verified; if it cannot,
  the "not an allocation" arm is dead code rather than triage.
* `put_reuse()` re-parks K/V that a later `limit_reuse()` or park may reject, so a handed-back candidate holds
  cache budget between turns. Bounded by `budget_ - bytes_`, the same rule `retain()` follows, but not exercised
  end-to-end here.

Two further observations needed no action: the probe leaves the reserved region uncommitted on the failure path
(now fixed anyway, point 3 above), and `if (probe.error)` prints nothing if a step ever fails with
`GetLastError() == ERROR_SUCCESS`, in which case the four text branches already carry the meaning.

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

## Second review pass on the `evict_for_admission()` move

A second read of the refactor (same reviewer role, conversation
`67eb3fed-f0dd-42d6-a2aa-631dc5367193`) found no behaviour change: it walked the helper against the inline loop
as it stood then (the branch commit `3ae6a9ec`, since rebased away; upstream still has the same loop at
`git show v0.1.43:src/program/generate.cpp` 7915-7930) and reported the two equivalent on every reachable input,
including `additional == 0`,
the `hopeless` case, a flat reading, and `sample()` returning `nullopt`; it confirmed the loop terminates without
the dropped `slots()` bound, that `admitted` is never set without a reading saying so, and that there is no
unsigned underflow in `additional + floor - *available`. It raised five points to fix and six nice-to-haves. Each
was checked against the code before acting on it.

Acted on:

1. **The `hopeless` bullet claimed an immediate refusal.** True - the bullet predated the move. The helper samples
   first (`conversation_memory.hpp:117-122`); `hopeless` only decides what a refusal says, not whether there is
   one. The bullet now says so.
2. **"The refusal line can print a shortfall measured before an eviction."** Checked and refuted. The helper takes
   its reading immediately after each eviction (`:131`) and refreshes `short_by` from it (`:136`); the only break
   that skips a sample (`:129`) happens *before* an eviction, so no eviction is ever reported against a reading
   older than itself. This is now pinned rather than argued: the equivalence test compares the old loop's
   `short_by` and `freeable` with the helper's field by field and requires them equal on every table case that
   entered the loop. The earlier version of this document claimed the opposite - that the old loop reported a
   stale shortfall - and that claim was wrong; it is removed. What *is* true, and what the text now says, is that
   `freeable` is a snapshot taken when the round decided it could not win, and in the one arm that prints it
   (`hopeless`) nothing has been evicted.
3. **"Measured output of the test" for a test that prints nothing.** Retitled as what the checks assert.
4. **A retained-K/V state in the old-loop copy of the test.** Inert: that loop reads neither `evictable_bytes()`
   nor `reuse_`, and `evict_oldest()` (`conversation_cache.hpp:290-297`) never touches it. Removed, with a
   comment saying why the state is deliberately not retained there.
5. **"What both park copies share is ... the named-cause failure lines."** Overreach - the four named causes and
   `ram_note()` are in `src/program/generate.cpp`; the SYCL copy still prints its two generic lines
   (`sycl/src/program/generate.cpp:7502, 7525`). Rewritten to say what is shared (the cache members, the
   `put_reuse()` hand-back) and what is not (the wording).
6. `!conversations.evict_oldest()` in the loop guard is indeed dead for `ConversationCache`: `evictable_bytes() > 0`
   means an unpinned entry exists (`conversation_cache.hpp:305-310`) and `evict_oldest()` picks exactly such an
   entry (`:291`). It stays - it is what bounds the round for a cache type whose two predicates disagree, which
   is what the template allows - and is now marked as such in the code rather than left to inference.
7. Nice-to-haves taken: the equivalence table now also compares the reported numbers, not only the decision; a
   round against a cache holding one unpinned and one pinned entry was added, which is where the `hopeless` bound
   is shown to spend nothing at all - including the one entry that could pay part of the request - and where a
   request the freeable entries can just cover spends only the unpinned one and leaves the pin parked; the
   top-level `const` on the helper's parameters was dropped.

Already covered, no change needed: the point that a refused park should be asserted to leave the cache untouched
is asserted at `test:525-527` ("the parked conversation is still there for the next request"). Not taken: making
`session_save_reclaim.hpp` a caller of the helper (it also reclaims the checkpoint chain and `retained_bytes()`,
so it solves a strictly broader problem and would lose behaviour), and adding an explicit "no telemetry" flag to
`EvictionRound` so the CUDA and SYCL refusals agree on that wording - that is a caller-visible change to the
refusal text, out of scope for this patch; the SYCL wording is recorded above instead.

## End-to-end run on the built binary (this PC, 2026-10-11 23:42 to 2026-10-12 02:46)

The user's production server ran a build of this tree:
`D:\AI_envs\_olympapp_local\server_strata\strata-0.1.42-sm120.exe`, configured from
`project_strata\build_cuda_120` (`CMAKE_HOME_DIRECTORY` = this tree; `generate.cpp.obj` postdates the last edit
to `src/program/generate.cpp`), logging to `server_strata\strata-sm120.log`. It reports
`session is up (engine 0.1.42)` on the arguments the failures were reproduced on
(`--max-context 262144 --kv int8 --kv-resident 32768 --conversation-cache-mib 16384
--conversation-cache-slots 4 --conversation-cache-min-free-mib 4096 --remote-expert-opt --layer-split auto`,
two RTX 5060 Ti, split auto). A string audit of that exe: the textless `allocation failed; skip parking`
absent; `allocation failed at the capture`, `skip parking after the checkpoint split`, `memory admission`,
`could not have fitted it`, `freeing every parked byte`, `reserves but will not commit` all present.

Counted over that log (1685 lines, of which the first 1595 are one session; the second session had just started
and parked nothing) with:

```
python - <<'PY'   # over the log, per session boundary at 'engine started'
pat = r"parked (\d+) tokens.*?parked=(\d+) bytes=(\d+).*?reused_kv_bytes=(\d+)"
PY
```

```
park attempts: 216, of which 176 at >= 50k tokens and 232826 tokens at the largest
refusals:      1 of 216 (0.46%)         <- the 0.1.40 build refused 291 of 2772, 10.5%
"allocation failed; skip parking": 0    <- 203 in the 0.1.40 log
restores: 130, largest 232494 tokens; 129 of the 216 parks reused K/V, up to 3.29 GiB in one park
evict-to-admit rounds: 12 (11 evicted one conversation, one evicted two; largest request admitted 4006 MiB)
park wall time: 62.9 ms to 1620.1 ms
peak cache bytes: 8.65 GiB against the 16 GiB budget; parked entries never exceeded 4
budget refusals (make_room): 0; refusal arms seen: 'hopeless' 1, 'stalled' 0, 'short_by' 0
```

The one refusal, at log line 737, is quoted in item 4 above; it is the `hopeless` arm saying the request
(3966 MiB) exceeded everything freeable (2152 MiB) - a refusal upstream would also have made, after first
throwing away a conversation that could not have paid for it. Parking continued normally after it; no later park
was affected by it.

What this run does and does not show. It shows the admission path deciding correctly at token counts far beyond
the 52k-97k that produced the original failures, the new stop-early bound keeping evictions to one or two
conversations across 12 admit-after-eviction rounds, the retained-K/V hand-back working (129 parks reused K/V),
and a refusal that names its own cause with numbers. It does not exercise the `stalled` arm (0 occurrences:
evicting did return memory on this machine on this run), the probe in a `bad_alloc` handler (no allocation
failed), or the budget-refusal hand-back in item 2 (no budget refusals; the 16 GiB budget was never reached).
Those three remain covered by the native tests only. This is one Windows/WDDM multi-GPU data point; HIP and SYCL
remain unvalidated at runtime.

## Limitations

* **SYCL is not compiled.** Both park copies were edited (`src/program/generate.cpp` and
  `sycl/src/program/generate.cpp` - `git grep park_current_body v0.1.43` finds exactly these two), but no Intel
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
