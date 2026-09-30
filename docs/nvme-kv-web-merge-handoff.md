# Handoff: merging `nvme-kv-web` into `nvme-kv-cache`

**Purpose:** land the NVMe KV cache web page (plan `docs/nvme-kv-cache-web-plan.md`, steps 1-8, all shipped on
`nvme-kv-web` at `7e4ceea`) onto the tier's own branch, and redeploy. The merge is small and mechanical in
`kv_delta.cpp` and a judgment call in `kv_delta_host_test.cpp`; **the hard part is not the merge, it is that two
upstream commits change what the page's numbers are and what its words say.** Read §3 and §4 before you resolve
anything, because a resolution that keeps my `TierActivity` lines and ignores the accounting change produces code
that passes its tests and a page that lies.

Written by the implementer of steps 1-8, from the worktree, after running every gate the branch claims.
Facts below are measured, not estimated. Read `docs/nvme-kv-cache-web-design.md` §6 and `docs/nvme-kv-cache-design.md`
§6 for the numbers themselves; this document is about the merge.

---

## 0. Read-me-first

1. `src/platform/kv_delta.cpp` — `KvDeltaStore::sweep()` is the only place both sides wrote over the same lines
   (the one real conflict). `kv_delta_enforce_cap()` changed on MY side only (it returns `TierActivity` and now
   consumes `sweep()`'s return), so it merges onto mine - but re-read it, because its loop condition is what the
   accounting change in §4 bites.
2. `include/strata/platform/kv_delta.hpp` — `TierActivity` (mine), `blocks_per_chunk` / `kDeltaBlocksPerChunk`
   (upstream, `c70c119`). These two do not interact textually, which is why the header auto-merges; they DO
   interact in the test fixtures.
3. `src/platform/kv_delta_host_test.cpp` — the five conflict hunks, all of them boundary-value collisions.
4. `src/program/generate.cpp` — **upstream has not touched this file**; the whole `KV` line is mine alone, so
   there is nothing to resolve there. Verify with `git log --oneline 7c779de..nvme-kv-cache -- src/program/generate.cpp`.

## 1. The state of the two branches

```
base (merge-base)            7c779de        both sides fork here
nvme-kv-cache    +5 commits  69eae44  (docs: restore-perf handoff)
                                      6648be7  kv-delta: the sweep RECOMPUTES the tier's byte total from the disk
                                      f665d14  tools: the delta p0's fork-sharing threshold counts shared span-chunks
                                      c70c119  kv-delta: 256-token sealed chunks (64 lcm-blocks per chunk)
                                      5bc4555  kv-delta: fdatasync per record
nvme-kv-web     +12 commits   7e4ceea  docs: the GPU gates were run
                                      bf69e20  serve: attach_cache() replays the startup line
                                      d75566d  tools: three oracles asserted a tier they never asked for
                                      801b138  setup, docs: setup option, C11 closed        (step 8)
                                      f6ce321  serve, web: Cache column, About card, RSS    (step 7)
                                      da0b893  web: the Cache tab                           (step 6)
                                      2ccc944  serve: GET /cache + /metrics cache block     (step 5)
                                      53fbdc1  serve: wire the tiers into the request path  (step 4)
                                      84736c1  serve: kvcache.py (unwired)                  (step 3)
                                      9a41f77  kv-delta/nvme/serve: the KV line             (step 2)
                                      7e79c16  kv-delta/nvme: TierActivity                  (step 1)
                                      8b4abc1  docs: the web design + plan
```

Conflict preview, no working-tree change (git ≥ 2.38):

```bash
git merge-tree --write-tree nvme-kv-cache nvme-kv-web          # tree 4e1a59dd16ad9e5e3c65d8754dd18385bc3e9a8f
git show 4e1a59d:src/platform/kv_delta.cpp | sed -n '1180,1202p'
```

Verdict: **2 conflicted files, 1 hunk + 5 hunks.** `include/strata/platform/kv_delta.hpp`,
`tools/nvme_delta_p0_test.sh`, `src/program/generate.cpp`, all of `serve/`, `setup.py` and the docs merge cleanly.
**My side changed no store format and no version** - steps 1-2 are reporting-only, and `NvmeHeader` (208 B, version
3) is untouched on both branches. Upstream's side DID grow `DeltaManifestHeader`: `c70c119` adds
`blocks_per_chunk` at `offsetof == 256` (static-asserted), recorded **per manifest**, with readers treating `0` as
the pre-grouping layout - so the existing 381 GB store still opens under the merged binary, which is the property
§7's rollback rests on.

## 2. The two conflicts, and what to keep

### 2.1 `kv_delta.cpp` — one hunk, and the answer is "both"

At the tail of `KvDeltaStore::sweep()` (`nvme-kv-cache:src/platform/kv_delta.cpp:1134-1185`, the recompute comment
at `:1174` and `total_ = kept` at `:1179`), upstream replaced the subtraction with a recompute; I replaced the same
subtraction with reporting. They are not alternatives.

```cpp
    // upstream (6648be7) — KEEP, this is the cap's correctness
    total_ = kept;
    for (const fs::directory_entry& de : fs::directory_iterator(dir_, ec)) { ... total_ += file_size ... }  // the log- manifests
    // mine (7e79c16) — KEEP, reporting only, it touches nothing
    act.swept = (int64_t) swept;
    act.swept_bytes = bytes;
```

Drop my `total_ -= bytes;` line — that IS the double-subtraction `6648be7` fixed. Everything else in the function
(`TierActivity act;` at the top, `kept`/`bytes`/`swept` in the loop, `return act;`) auto-merges. The same goes for
`kv_delta_enforce_cap()`, which merges onto my version and ends with
`const TierActivity swept = delta.sweep(); act.swept = swept.swept; act.swept_bytes = swept.swept_bytes;` - the
reporting is mine, the recompute inside `sweep()` is upstream's, and §4 is about what that combination means.

### 2.2 `kv_delta_host_test.cpp` — five hunks, all one pattern

Every hunk is the same collision: **upstream moved these fixtures to the 64-block chunk layout (dumps at 256 /
400 / 600 tokens, 1-2 chunk files), my step-1 assertions wrote against the pre-grouping layout (10 / 14 / 18 / 20
tokens, 4-5 chunk files) and added `TierActivity` checks on top.** The rule:

> Take **upstream's token boundaries, file counts and `sealed()` arithmetic**, then re-apply **my reporting
> assertions on top of them** — and re-DERIVE the expected values instead of copying mine.

The hunks (line numbers in the merged blob `4e1a59d:src/platform/kv_delta_host_test.cpp`):

* **779** — `ck_eq(L1, 300, …)` (theirs) vs `ck_eq(L1, L_BOUNDARY, …)` plus my `image_bytes` check (mine).
  Keep BOTH: take their literal or my symbolic boundary for the length, but my
  `image_bytes == slurp(v3_path).size()` assertion is the page's `staging_bytes` proof and must stay
  (`generate.cpp` passes that out-param through).
* **1001** — the supersede fixture. Theirs dumps 256→600 and counts 2 chunk files; mine dumps 10→18, counts 4, and
  carries the `dropped` / `dropped_bytes` / `written` reporting. Keep their 256/600 and their 2; re-derive `written`
  as "the new manifest + State + only the chunks THIS turn sealed". My *formula* (`entries()[0].bytes -
  dir_bytes("/chunks")` measured before) is layout-agnostic and survives; my literal `4` does not.
* **1064** — fork B: `boundary_checkpoint(S, 400, 400)` (theirs) vs `(S, 14, 12)` + `&ab` (mine). Theirs for the
  boundary, mine for the out-param.
* **1105** — fork C + `kv_delta_enforce_cap` + sweep. Theirs asserts 1 surviving chunk (`sealed(400) = 256 = 1`);
  mine asserts 5 and adds the `evicted` / `evicted_bytes` / `swept` / `swept_bytes` reporting. Theirs for the
  counts; my four reporting assertions survive mostly as written because three of them compare against the **disk
  walk** (`delta_records()` before/after) rather than a constant. My `ck_eq(ev2.swept, 1, …)` argument ("every chunk
  B sealed is below C's boundary too") is the same argument upstream's line makes, so it holds under grouping — but
  re-derive it, do not trust it.
* **1217** — the eviction-LRU fixture: `entries()[0].L == 600` (theirs) vs `== 18` (mine) plus my `evicted == 0` /
  `swept == 1` / `swept_bytes` block. Their 600, my reporting block; it compares against `delta_records()`, so it
  moves on its own.

Two helpers exist only on my side and must come across with the hunks: `delta_records(dir)` (record count + bytes
from the disk) and `dir_bytes(dir)`. Both are disk-truth measures, which is precisely why they outlive layout
changes — that was the point of writing the assertions that way.

The check-count in the recorded results will change: `kv_delta_host_test` on this branch prints **156,081 checks
passed**; after the merge it will not be that number, and the old number must not be quoted as current.

## 3. What the merge changes about the page's NUMBERS

`c70c119` moved sealed chunks from 1 lcm-block to **64 blocks** (`kDeltaBlocksPerChunk = 64`, recorded per
manifest as `blocks_per_chunk`, readers treat `0` as the old layout). "Only the file count drops 64×" — and the
file count is exactly what the Cache tab's store table and my measured divergence are made of. Every figure below
was measured on the **pre-merge** binary and layout, and must be re-measured before it is quoted again:

* **cap accounting vs disk footprint, live store**: `6,170,642,908` vs `3,267,409,228` = **1.89×**. Both sides
  change; re-measure.
* **fork sharing**: 644 chunk FILES for 1,176 references (my live store: 3,607 chunks / 19 manifests). Chunk files
  drop ~64× and the refs per conversation collapse; `f665d14` already moved the oracle's threshold to "shared
  span-chunks".
* **store table row counts**: 5 v3 snapshots + 19 manifests + 3,607 chunks + 19 states = 3,650 files. The chunks
  term collapses; the v3 / delta / state terms are untouched.
* **per-turn cascade write**: 112.8-115.8 MiB for turns of 15-217 new tokens, 555 MiB for a 30k turn. The bytes are
  the same material and the KV line's `dump_bytes` is unaffected; the record count differs.
* **promote cost, live**: 1,453 tokens resumed in 581 ms, 140,319,780 bytes read, `staging_bytes` 140,293,564.
  Unchanged by the merge (upstream did not touch the restore path), but re-verify after redeploy.
* **oracle boundaries**: `RESUME 4104` / `RESUME 3928` / boundary 2355 are pre-merge-layout results. The `delta_p0`
  re-run will print different boundaries and file counts; the claim that must still hold is P1's hash equality
  across the two tiers.

The doc lines to update after re-measuring: `docs/nvme-kv-cache-design.md` §6 (the live-run block, the 1.89× and
the 644/1,176 figures), `docs/nvme-kv-cache-web-design.md` §5's fact-table line (already corrected here to
"lcm(page_size, idx_block) = 4 tokens; a sealed chunk = 64 of them" - see §8's last bullet: it said 1,024, which
was never true and was never rendered), and the same doc §6's `kv_delta.cpp:921-970` citation (my branch's line
numbers; on the merged tree the accounting walk is at `nvme-kv-cache:src/platform/kv_delta.cpp:938-990`, the
per-entry sum at `:990`, the append at `:1084`, the sweep recompute at `:1174-1185`).

## 4. What the merge changes about the page's WORDS — do not skip this

`6648be7` means the engine's `delta_bytes` is no longer a monotone over-count. The books are:

* **at open**: `total_` = every `chunks/`+`states/` file once (including the residue a coming sweep will delete),
  **plus** each entry's own bytes (manifest + State + every referenced chunk) → a chunk shared by three manifests
  is counted four times. This is what my tooltip describes, and it is still exactly right.
* **on append**: `total_ += e.bytes` (`:1084`) — the same per-reference counting.
* **at every sweep**: `total_` is **reset to the disk** — surviving chunk+state files + manifest sizes
  (`nvme-kv-cache:src/platform/kv_delta.cpp:1174-1185`).

So `delta_bytes` is a **sawtooth**: it drifts above the footprint as shared refs accumulate, and a sweep snaps it
back to the truth. My text says "the cap must not lie about the disk" as if the over-count were permanent; the
honest sentence is that the over-count is the *safety* direction (over-estimate → evict early) and the sweep is
where the books meet the disk again. The three places that carry the old wording:

1. `serve/web/app.js:137` — the "Store used" card tooltip ("A chunk shared by three manifests counts once per
   manifest, on purpose, so the cap cannot lie about the disk").
2. `serve/kvcache.py:285` — the same sentence in the module docstring.
3. `docs/nvme-kv-cache-design.md` §6 "The page's metric definitions" — "`entries_bytes` / `delta_bytes` are CAP
   ACCOUNTING, not a disk footprint … deliberately".

**The two books are still two books** — do NOT "fix" this by merging them. The serve-side walk counts what the
volume holds including v3 snapshots, foreign-magic files, stale-version heads and `.tmp-` residue, none of which
`delta_bytes` ever includes; and the engine's number counts per-reference, which the walk never does. What changes
is the reason and the shape: they are different *measurements*, and they **agree exactly when a sweep has just
run** — which is a coincidence the page must not present as an invariant. There is a host test worth adding while
you are in there: `sweep()` then compare `delta.total_bytes()` to a disk walk of the delta subtree, and assert the
equality the recompute promises.

**The cap's decisions change too, and it is not my regression.** `kv_delta_enforce_cap()` auto-merges onto my
`TierActivity` version and consumes `delta.sweep()`'s return, so its `while` loop now tests
`v3.total_bytes() + delta.total_bytes()` against books that were just **reset to the disk**. `6648be7`'s own message
says the old build drifted low and therefore UNDER-evicted; a truthful total evicts **earlier and more often**. So
expect the recorded `steps123` step-3 figures (`6 files remain`, `evict=1 evict_bytes=178042216`), the `evict=`
fields on live `KV` lines, and `short_tests`' store-growth counts to come back **different after the merge**. The
assertions that must still hold are the shape ones - `< 7 files`, `evict>=1` with non-zero bytes, and the `sweep ==
disk walk` equalities - not my literals. Attribute the delta to `6648be7` in the merge commit message, or the next
reader will blame `TierActivity`.

One future collision to note rather than fix: `69eae44`'s `docs/nvme-delta-restore-handoff.md` item **B (streaming
restore)** is explicitly "the choreography B must reproduce **without assembling**" the image. When B lands,
`staging_bytes` (`delta_restore`'s `image_bytes` out-param, mine) stops being the whole snapshot, and the page's
"Promote staging" row, the About card's "a promote stages the whole snapshot in RAM at once", and C10's ~2 GiB
transient claim all need revisiting. B's implementer should grep `staging_bytes` and `Promote cost` before
shipping.

## 5. What does NOT change (verified, so you do not have to re-prove it)

* **No byte-format change on my side**: steps 1-2 add out-params and return values; `NvmeHeader` (208 B, version
  3) and the delta record family are untouched, and upstream's `blocks_per_chunk` rides the manifest header at
  `offsetof == 256` (static_asserted). The production store (224 conversations, 381.70 GiB) opens with either
  binary.
* **`generate.cpp` is mine alone** — the `KV` line, `TierCounters`, `print_kv()` behind `if (have_kvstore)`, the
  `transfer=1`-before-`ERR` ordering, the `start=1` line after `READY`. No conflict, nothing to re-derive.
* **A tier-off engine prints nothing new** (`print_kv()` is guarded), and the old `serve/server.py` request loop has
  **no `else`** in its line chain (`69eae44:serve/server.py:322-339`), so an unrecognised `KV …` line is skipped
  rather than emitted as a token. Binary-before-code is harmless; code-before-binary gives a Cache tab that
  reports `enabled` with null engine facts. **Deploy the two together.**
* The serve side (`kvcache.py`, `server.py` wiring, `/cache`, `/metrics`, the web app, `setup.py`) has no upstream
  counterpart at all — those files are not touched on `nvme-kv-cache`.

## 6. Gates, in the order that costs least

Host, no GPU, run these first:

```bash
cd /local/strata/.worktrees/<merged-tree-worktree>
export PATH=/local/strata/.venv/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DSTRATA_ENABLE_CUDA=ON \
      -DSTRATA_BUILD_CONVERSATION_TESTS=ON          # already configured in .worktrees/nvme-kv-web/build
ninja -C build
ctest --test-dir build -R "conversation|nvme|delta|transfer|cache|memory|validation" --output-on-failure   # 7/7
/local/strata/.venv/bin/python -m unittest serve.test_server serve.test_mcp                                  # 80 + 24
node --check serve/web/app.js
```

`kv_delta_host_test` is the one that will fight you; `serve.test_server` must stay **80 tests green** because
nothing on the serve side depends on the tier's internals.

GPU gates — these need the card, i.e. a **second production outage (~35-45 min for the set)**. The scripts resolve
`ROOT` before any `cd /local/strata`, so always pass the binary under test explicitly, or you will be testing the
deployed one:

```bash
E=/local/strata/.worktrees/<merged>/build/strata
NVME_ENGINE=$E bash tools/nvme_steps123_test.sh          # expect: 3 KV per 3 DONE, one start=1, evict>=1 with bytes
NVME_ENGINE=$E bash tools/nvme_failure_contract_test.sh  # expect: ALL PASS, KV transfer=1 BEFORE the ERR line
NVME_ENGINE=$E bash tools/nvme_p0_test.sh                # expect: ALL PASS (RESUME, hash equality, negative control)
NVME_ENGINE=$E bash tools/nvme_delta_p0_test.sh          # expect: ALL PASS incl. P1 across the two tiers
STRATA_TESTS_BASE=... STRATA_TESTS_LOG=... STRATA_TESTS_STORE=... python tools/short_tests.py
python tools/needle_bench.py --url http://127.0.0.1:<port>
```

Carried through the merge by `d75566d`, and easy to lose because `f665d14` edits the same file: **three v3 oracles
now pass `--kv-delta 0`** (`steps123`, `failure_contract`, and `delta_p0`'s `runv3`). Without them the delta tier —
the DEFAULT since Phase 6 — takes the cascade, no `kv-*.bin` is written, and the oracle reports a PASS-shaped
silence on an empty store. After merging, grep to confirm: `grep -c "kv-delta 0" tools/nvme_*_test.sh` should read
**2 in each of the three files** (the flag, and the comment saying why it is load-bearing), and `runv3`'s comment
("delta off") must match its flags.

Then, on the merged binary + merged serve code, re-do the live check that caught the ordering bug (§8) and
re-measure the numbers in §3: a cold turn, an engine restart, the same conversation back (`src=delta`,
`resume>0`, `reused_from_disk>0`), then `/cache`'s `promotable.bytes` vs `on_disk_bytes`.

## 7. Redeploy

Production is `/local/strata` (branch `nvme-kv-cache`, currently `69eae44`), supervised by
`/tmp/serve-loop.sh` (a `while true` loop, 10 s between attempts, `STRATA_WATCHDOG_S=300`), serving
`--engine strata --config /local/strata/strata-iq3_xxs.json --host 0.0.0.0 --port 8080`; the engine is
`/local/strata/engine/strata`, refreshed the usual way: `cp build/strata engine/strata` after a matching
`md5sum`. Config args today: `… --max-context 262144 --kv int8 --prompt-cache 4 --kv-nvme /local/strata/kvstore
--kv-nvme-max 200 --kv-delta 1`.

Sequence:

1. Merge and get §6's host gates green **before** touching the GPU; the outage is for the engine reload and the
   oracles, not for compiling.
2. Stop the loop FIRST, then the server, then the engine — killing only the server just restarts it in 10 s with
   the old binary: `kill <loop-pid>` then the server, then `strata --serve` by PID.
3. Deploy binary **and** the serve tree together (`serve/*.py`, `serve/web/*`, `tools/*`). `setup.py` is inert at
   runtime; bring it too so the knob is documented where the install lives.
4. Start the loop, wait for `ready:` (~90 s at 262k), then verify the line reaches the page — this is the one
   command that proves the whole chain is deployed:
   ```bash
   curl -s localhost:8080/metrics | /local/strata/.venv/bin/python -c \
     "import json,sys; c=json.load(sys.stdin)['cache']; print(c['enabled'], c['promotable'], c['ram_tier'])"
   ```
   With zero requests you must see real `entries` / `delta_entries` / `bytes` (the `start=1` line) and
   `ram_tier: {checkpoints: None, live_tokens: None}` — the RAM tier rides the request line, so nulls there are
   correct, not a bug. Then run one real conversation and confirm a `KV` line landed (`/metrics`
   `requests[-1].cache`).
5. Rollback is a checkout + `cp` of the previous `engine/strata` and the old `serve/`; the store needs no
   migration in either direction (`blocks_per_chunk` is per-manifest and `0` means the old layout).

Budget the outage honestly and say so before starting: the tier's RAM state (checkpoints, the live session) is
lost every time, and users' next turn re-reads. That is the only user-visible cost of this whole operation.

## 8. Things this branch found that are worth not losing in the squash

* **The ordering bug needed a live server.** `main()` attaches the `KvCache` after the engine is constructed, and
  the pump thread had often already consumed the `start=1` line — so a tier-on server reported `enabled: true`
  with every store fact null until a request happened to arrive. Every unit test attached the cache by hand
  *before* the line, so every unit test passed. Fixed by `StrataEngine.attach_cache()` (`bf69e20`). If you resolve
  `serve/server.py` by hand, keep the `store_kv` field and the replay; the test is
  `serve.test_server.KvPump.test_a_startup_line_read_before_the_cache_exists_is_replayed_at_attach`.
* **`restart()` must keep the cache** (`53fbdc1`, tested by `RestartKeepsTheCache`): a transfer failure is the
  path that restarts the engine, and losing the tier wiring there loses the only record of the failure.
* **`short_tests.py` can never gate on the `KV` line** — the server gives the engine's *stderr* to the log and
  keeps *stdout* on a pipe (`serve/server.py:159-160`). The plan's test matrix said otherwise and was corrected.
* **A number I wrote into the design as fact had never been measured.** The §5 mock-up's Store block said
  "Chunk BLOCK lcm(page_size, idx_block) = **1,024** tokens". The live engine says otherwise three ways over:
  `appended 7590 chunks … T 0->30363` and `363 chunks … T 0->1453` (both 4.0 tokens per block), and both
  oracles' `sealed()` values (`sealed(20) = 20 -> 5` on mine, `sealed(400) = 256 -> 1` at 64 blocks per chunk on
  upstream's). **BLOCK = 4**; a sealed chunk is 64 blocks = 256 tokens. It is corrected in the mock-up here and
  the page never rendered it, so no user ever saw the wrong number - but the failure mode is the one worth
  remembering: a mock-up figure becomes a documentation figure becomes a belief. If the page ever grows a BLOCK
  row, take it from the engine (the `KV` line or a walk field), never from a literal.
* **`needle_bench` 3/3 at 32K with the tier on is NOT a tier win** — three depths are three different prompts,
  all three reported `src=none`, and the speed-up was the RAM prompt-cache. The page exists to make that
  distinction visible; keep that sentence in §6 of the tier doc.

## 9. Still open after all of this

* The manual both-themes / narrow-breakpoint pass on the Cache tab (`serve/web/app.css` reuses only existing
  tokens and collapses at the Monitor's own 1000 px / 640 px breakpoints, but nobody has *looked at it*).
* The tier-off `needle_bench.py` stdout diff (the plan's "a tier-off server is byte-identical" row) — never run;
  it needs a second tier-off pass with the same prompts.
* About's two deliberately-absent rows: a **weight fingerprint** (not a field of anything the serve side has —
  `kv_delta_weights_fp` exists in C++ but is never printed on the `KV` line, so a `weights=` field would be the
  cheap way to give the page one) and the **stale-file count** (a walk field, so it is on the Cache tab and not
  in About).
* `setup.py`'s new NVMe knob was never executed end-to-end (only `py_compile`, `--help`, and the block sliced out
  and exercised over 8 scenarios). First real install with it is a live check.
* Step 8's C11 close-out says "settled as a build". The remaining metric gaps are real and documented: no
  per-conversation identity (§7), no `dump_tokens` on the line (so cascade rows show `–` for tokens), and no
  store-mutation surface by design (§8).
