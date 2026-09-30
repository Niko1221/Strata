# NVMe-backed KV cache for warm sessions — design & test record

**Status: implemented, tested live, merged behind `--kv-nvme`.** An automatic cold tier that keeps
whole conversations on NVMe so a returning session resumes with low TTFT instead of a full
re-prefill. Saving and restoring are entirely server-driven - the client never calls save/restore
(this was the explicit requirement; ninfer's client-initiated model and llama.cpp's
`--slot-save-path` hooks were rejected for exactly this reason).

---

## 1. What the tier is

Three tiers, with the pinned host arena as the single source of truth and NVMe strictly below it
(NVMe never touches the GPU):

```
VRAM slots (hot)  <->  pinned host arena (warm, primary)  <->  NVMe snapshots (cold)
   kv_stream            KvHostPools                            KvNvmeStore (this work)
   clock eviction       (pre-existing)                         src/platform/kv_nvme.cpp
```

Everything strata needed already existed except the NVMe tier and the automatic behavior: the host
copy (`KvHostPools`), the clock-evicted VRAM slots (`kv_stream`), the running-state checkpoint
(`ConvCheckpoint`), and the longest-prefix resume in the serve loop.

## 2. The snapshot (v3 format, 208-byte header + payload + 8-byte digest footer)

One whole-session file per conversation turn, keyed by the **exact token prefix** (the consumed
`ids` are stored verbatim and matched with the same `starts_with` semantics the serve loop already
uses - no content hashing, no radix tree; a one-live-session server has no concurrent-prefix dedup
payoff, and exact keys make branching unambiguous).

Persisted set (the complete state a continuation needs):
- attention KV host copy, cells `[0, L)`, every format (fp16 / int8 codes+scales / q4_0), identity
  layout `[page][kv_head][page_size][head_dim]` - a contiguous prefix per array;
- indexer positional state per QSA layer: `idx_pooled` (rows `[0, L/idx_block + 1)`: the completed blocks plus
  the spare row the kernels keep equal to `dead` - a live array too small for that is refused, not truncated),
  `idx_tail`, `idx_dead`, `idx_block_pos` - the last three taken from the **turn-boundary checkpoint** when one
  exists, because the live `idx_block_pos` names a block completed by tokens past the boundary (v3; see §3);
- running state: the 36 GDN recurrences + conv history, the PLE history, and `ple_prev`;
- the MTP drafter's host KV copy (`mtp_host` arrays recorded in the header);
- header: ids length, image keys, cvec flag, KV format, the **shared core's 18-field geometry key**
  (`conversation_geometry_key` - the same array `SavedConversation::geometry` holds, so the RAM tier and this one
  refuse the same mismatch) plus the three runtime shapes the segment walk needs (`page_size`, `idx_block`,
  `max_cells`), and `mtp_host`; a restore **refuses** any mismatch, never converts, and refuses a file of another
  format version by name before it walks a single segment;
- footer: FNV-1a over the payload (everything after the header). The restore verifies it, so a
  flipped or truncated byte is refused ("integrity check failed"), never decoded.

Restore is **atomic**: the whole file is read and validated (sizes walked segment-by-segment, then
the digest) before anything is applied - a corrupt snapshot fails without touching the session.

## 3. The automatic behavior

- **Cascade** - at every `DONE` (device synced, synchronous: ~0.1-0.6 s for typical sessions, vs
  minutes of re-prefill) the engine stores the snapshot **at the chat turn boundary**: the longest
  `ConvCheckpoint` supplies the running state as of the end of the prompt, and the KV/pooled arrays
  are truncated to that position. This is the load-bearing design decision: a chat client re-sends
  history **without the model's hidden reasoning tokens**, so a full consumed-state snapshot could
  never full-prefix-match the next turn (the first live HTTP test showed every request falling back
  to re-prefill). The next request's prompt extends exactly the turn-boundary prefix, so the match
  is exact. Dumps are idempotent (exact key + imgs -> skip, recency refreshed) and the previous
  dump of the same growing conversation is superseded; a failed dump unlinks its partial file.
- **Promote** - in the serve loop's resume selection, after the RAM checkpoints: the **longest**
  stored entry whose ids (and image keys, and cvec state) are a prefix of the incoming prompt is
  restored into the arena (`pread` into the pinned host copies, H2D for the indexer/gdn/ple
  arrays), `kv_stream_reset` makes every VRAM slot refill on demand, the drafter ring is refilled,
  and the request continues from `resume`.  A promote that the tier **refuses** drops the entry and falls back to
  the clean path (`session_zero` + full read); a promote that fails a **transfer** stops the engine.  See §5 - the
  two used to be the same sentence, and the fallback that sentence described has never actually run.
- **Cap** - `--kv-nvme-max GB` evicts the least recently stored snapshots (the last entry is never
  evicted; the store is never emptied). Snapshots accumulate one per conversation turn
  (cross-restart supersession needs a conversation identity a single-session engine does not have;
  the cap bounds it).

Config: `--kv-nvme DIR` (enables the tier and forces the streamed-KV floor `--kv-resident`
`qsa_kv_resident_min()`), `--kv-nvme-max GB` (default 100, 0 = unlimited). `--nvme-dump`/
`--nvme-restore PATH` remain as hidden single-file debug spikes.

**Under an active layer split (0.1.21) the tier is inert, by refusal.** A split session's later stages hold
their own running state (`ConversationCheckpoint::stage_parts`, one checkpoint per stage, composed by the serve
loop with per-stage saves), and the envelope carries the primary stage only - a snapshot written from a split
engine could never be restored, because the file has nothing to put back into the later stages. So the dump side
refuses a split session outright (`nvme_dump_at` rejects a checkpoint with non-empty `stage_parts`) and the serve
loop does not promote (logged once per process): restoring into the primary session alone would leave the later
stages' running state zeroed while the caller believes it mounted a whole conversation. Refusing rather than
growing the format also keeps every stored snapshot - including the 106 converted in §5.3's wake - valid. Disk
support for split sessions would be a version bump with the stage blobs as first-class segments, and is not
built.

## 4. Correctness rules (each one earned the hard way)

1. **Store bytes verbatim, never recompute on restore** - the expert path rounds differently
   depending on residency; the engine's decode is also run-to-run nondeterministic (verified with
   the pre-tier binary), so the hard gate is bit-exact *state* restore, not bit-identical
   *continuations*.
2. **The hybrid trap** (llama.cpp #26676/#25913): strata is QSA full-attention + GDN recurrent.
   The GDN/PLE running state MUST be restored, never recomputed, and the positional KV is only
   valid while its cells hold the entry's tokens. Strata's own split - positional KV is rewindable,
   running state is checkpointed - is what makes a correct restore possible at all.
3. **Turn-boundary keying** (see §3): the matchable prefix is the prompt the client will re-send.
4. **A spill is a re-keyed dump, never a copy of a parked image** - `nvme_dump_at` is the only path to disk and
   it writes the boundary's ids, the boundary's pictures and the boundary's running state, all describing
   position `L`.  A parked conversation's consumed ids include the model's hidden reasoning tokens, so copying a
   parked image verbatim produces a snapshot no next turn can match.  A picture at or past `L` is refused for the
   same reason: it describes a token the snapshot does not hold.
5. **Refuse, never convert** across format/geometry changes - including a file of another format version, which is
   refused by version (naming the one found) before any segment is walked. The one conversion that exists is the
   offline v2→v3 store migration (§6, `tools/nvme_v2_to_v3.cpp`), which is a verified byte-mapping, not an
   in-engine reinterpretation: the engine itself still refuses, always.
6. **Payload digest** - hashed during the write with the same hasher that feeds the file (a plain
   write inside the payload silently desynchronizes the digest from the bytes; this exact bug
   shipped briefly and was caught by the corruption test).
7. **The spare pooled row is part of the snapshot, and a restore re-publishes it.**  A prefix of `L` cells owns
   `L / idx_block + 1` pooled rows - the completed blocks plus the spare at `L / idx_block`, which the writers
   keep equal to `dead` (`qsa.cu:213`).  A turn-boundary dump reads that row from a live array a longer turn has
   already overwritten, so `nvme_restore` re-publishes `dead` into it, as the shared core's
   `conversation_checkpoint_restore` does.  `STRATA_STATE_HASH` spans the same rows, so the DONE-vs-restore
   comparison can actually see the difference.
8. **An image is part of the key, and the entry must carry what the file holds.**  Two defects lived exactly
   here and none of the text-only tests could see them: `NvmeEntry::imgs` was never populated by the scan, so
   every image comparison in the tier compared against an empty vector; and a turn-boundary dump stored the
   *live* image list (every picture, including ones at `start >= L`) while keying on the boundary, so the resume
   match could never equal it - any conversation with a picture produced an unreachable snapshot.  The boundary
   now supplies its own filtered list (`at->imgs`), `nvme_dump_at` refuses an image record outside `[0, L)`, and
   the entry carries the file's list, re-checked at restore.

## 5. The failure contract (C7)

Three classes, named by **one enum in both tiers**: `strata::core::ConversationRestore { restored, invalid,
transfer_failed }` - the shared core's, which `nvme_restore` and `KvNvmeStore::restore` now return instead of a
`bool`.  A serve loop with two failure vocabularies eventually gets two policies, and the two tiers already
behave differently for the same event.

### 5.0 First, the finding this contract replaced

**The clean-reset fallback this document has been describing since §3 has never run.**  A failed promote left the
CUDA last error unconsumed, and the fallback's own first step reads it:

`resume = 0` → `session_zero` (`generate.cpp:3025-3026`) → `qsa_state_zero` (`layer.cpp:657`) → `kv_stream_reset` for
every streamed layer (`layer.cpp:676`) → `check("reset")` (`kv_stream.cu:199-202`), which prints
`kv_stream: reset: <error>` and calls **`std::exit(1)`**.  `--kv-nvme` forces streamed KV (`generate.cpp:1099-1103`),
so that chain always runs when the tier is on.  `cudaGetLastError()` returns the last error **and clears it**; left
unread it is sticky, so the error a *restore* caused was reported by the *reset*, and the process died there.
This tree had already been taught the same lesson by a `cudaHostRegister` failure whose unread error made an
unrelated kernel launch report "out of memory" (`pinned.cu:170-183`), and `graph.cpp:110-118` states the rule for
captures.

So today's behaviour was **neither our claimed recovery nor their fatal rule**: it was an unintended exit at an
unrelated point, with the diagnosis misattributed to the reset.  The fallback only looked like it worked because
nobody ran a failing restore - every recorded test corrupts the *file* (a refusal, which never reaches the
transfer pass) and none injects a failing *copy*.

### 5.1 `invalid` - recoverable: drop the snapshot, re-read the prompt

Every refusal the tier can make **before the apply pass begins**: bad magic, a format version it does not write,
the geometry key, header sizes that would size an impossible read, a truncated file, a segment walk that does not
account for the file, the payload digest, a live array too small for the snapshot, a null target buffer.  Also the
store's own stale-index case (the file applied cleanly but no longer matches the entry the scan built), which is
recoverable for the opposite reason - see §5.3.

*Why it is safe, not merely convenient:* `kv_nvme_host_test` asserts each of these paths makes **zero `cudaMemcpy`
calls and zero CUDA calls of any kind**, leaves every session buffer at a poison sentinel, hands the caller no
ids/imgs, and leaves no CUDA error pending.  Nothing was written, so the clean path has nothing to undo.

*Operator sees:* `strata serve: nvme promote refused (<reason>); reading the prompt instead`, then `RESUME 0`.
The snapshot file is deleted - a file the tier will not read is not worth keeping.

### 5.2 `transfer_failed` - fatal: the engine stops

A `cudaMemcpy` in the apply pass or in the spare-row re-publish, or either `cudaDeviceSynchronize` (the one that
now opens the apply pass, and the one that closes it).

*Why it is fatal:* the apply pass is a **loop**, so a failure in its middle leaves the session half-written by
construction.  The fixture shows the mix rather than arguing it - fail the second device copy and the GDN state is
the snapshot's while the PLE history is not; fail the last one and every segment landed while
`pooled[L / idx_block]` still holds the snapshot's stale value and `dead` holds the value it must become, i.e. a
session that looks restored and is one invariant short.  The shared core's own fixture asserts the same shape for
its restore (`conversation_validation_test.cpp:167-169`).  And nothing has yet shown the CUDA context still
answers, which is exactly the proof §5.4 says a recovery owes.

*Operator sees:* `strata serve: nvme promote FAILED (transfer): <segment + byte count>`, then
`… the snapshot is half-applied and nothing proves the CUDA context still answers - not attempting a clean reset.
The snapshot is left on disk; stopping this engine.`, and the client gets `ERR restoring a stored conversation
snapshot failed: <reason>` before the process exits with 1.  The snapshot is **not** dropped: a transfer failure
says nothing about the file.  This is the same consequence the RAM tier already gives a failed checkpoint restore
(`generate.cpp:3044-3045`) and the same one issue #57's core calls `transfer_failed`.

*Why stopping is recovery, not defeat:* under `serve/server.py` the engine runs behind a supervisor that notices a
dead engine and starts it again on the next request (`serve/server.py:686-695`).  A **new process** is a new CUDA
context, a new pinned arena and new captured graphs - the state the in-process reset could not prove it reached.

### 5.3 A stale-format store - recoverable, and an operator action

A store full of snapshots this build cannot read is **not** a startup failure and **not** a transfer failure: it
is §5.1 reached before any file is even opened for restore.  `KvNvmeStore::open` counts stale-version files
separately from malformed ones, skips them, keeps them on disk, and reports the version it found.  Nothing is
promotable, so every request re-prefills.  *Operator sees:*
`strata serve: kv-nvme: N snapshot(s) of format version 2 in DIR: this build writes version 3 and refuses older
files. They stay on disk and are skipped, so nothing can be promoted from them and every request re-prefills - to
keep them working, re-dump them with the binary that wrote them; to stop the skip, remove them and let the store
rebuild`.  Both options are stated because both are legitimate: 113 GB of v2 snapshots is either a corpus to
re-dump with the old binary or a store to delete and let refill - what it must not be is an ambiguity.

**This case was then actually executed, end to end** (2026-09-29): the production store held 106 v2 snapshots
(199.27 GiB), and "re-dump with the binary that wrote them" is impossible - the old binary writes v2 - so the
store was migrated with an offline converter, `tools/nvme_v2_to_v3.cpp`. The payload bytes are the same layout
with one formula difference (v2 wrote one spare pooled row per QSA layer that no reader can reach), so the
converter verifies the source's own footer, solves the one unknown (the gdn+ple prefix block) from the file's own
size, swaps in the shared core's 18-field geometry key from a reference v3 dump, copies every segment verbatim
except that unreachable row, and recomputes the footer. Nothing was guessed: the drafter term and the prefix
block were fitted - all 106 files and four fresh v3 references (L 12 … 58,513) solve to exactly
gdn + ple = 118,038,528 B with zero spread, and the converter's walk must land on both footers to the byte or it
refuses. Verification: the old binary restoring the v2 original and the new binary restoring the converted v3
print IDENTICAL `gdn`/`ple`/`tail`/`kv`/`mtp`/`stale`/`ple_prev` hash fields (the two that differ are exactly the
two v3 adds: `pooled` now spans the spare row, `dead` is a new field); the largest file (3.8 GB, L=242,357)
restores cleanly; all 106 converted, zero refused. The originals are renamed aside in
`/local/strata/kvstore-v2-backup/` (same filesystem, so a rename, not a copy) and should be deleted only after a
few days of production promotes from converted files. The production store now scans as 106 sessions / 199.27
GiB with no version refusals, and the live server has promoted from converted files.

### 5.4 What a future clean reset must prove before it may exist

A clean reset after a transfer failure is not forbidden forever; it is unproven today.  Before it may exist, all
three of these have to be demonstrated **on device**:

1. that a real failed host-to-device `cudaMemcpy` leaves a **non-sticky** context error - i.e. that the next
   launch, the next `kv_stream_reset` and the next graph replay succeed rather than reporting the old failure;
2. that the **captured graphs** survive it (they are captured once, before the serve loop - `session.cpp:164`,
   `graph.cpp:106` - so the question is whether replaying them after a failed copy is sound, not whether they can
   be re-captured);
3. that the **pinned arena** and the streamed page table are back in the residency contract, which §5.1's
   `session_zero` path does when nothing was written and does not obviously do when something was.

The mechanism would be a **device-usability probe** - consume the error, `cudaDeviceSynchronize()`, and a bounded
write-and-read-back through a scratch device buffer - run before the tier reports `transfer_failed`, with a clean
reset allowed only when it passes.  **It is not implemented, and it is unvalidated on this machine**: the GPU
holds a live ~24 GB engine, the host fixture's copies are `memcpy`, and an unvalidated guard in front of an
unvalidated recovery is worse than a clean exit.  The one piece of the proof this tier *can* show today is the
restore's own final `cudaDeviceSynchronize()`: a success there means the device answered after the last write,
which is why the store's stale-index case is recoverable and a mid-apply failure is not.

## 6. Test record

Oracles and harnesses (all exit non-zero on failure):
- `tools/nvme_p0_test.sh` - bit-exact restore: `STRATA_STATE_HASH` (refactored into
  `state_hash_line()`) printed straight after a restore (`STRATA_NVME_HASH`) must equal the
  dumper's DONE hash; identical-request processes compared; deliberate-corruption negative control
  (must be refused). GPU-needed, ~7 min.
- `tools/nvme_steps123_test.sh` - cascade/supersede (one growing file per conversation), promote
  after a process restart, LRU cap eviction. ~8 min.
- Live-server HTTP tests (`/v1/chat/completions`, streaming) - the needle test and the short
  correctness suite (driver scripts were run ad hoc; assertions listed below).
- `kv_nvme_host_test` (ctest, `STRATA_BUILD_CONVERSATION_TESTS=ON`) - the tier's format, resume and **failure**
  rules with no CUDA context: a synthetic session whose device arrays are host buffers, CUDA linked and never
  initialised, the copies wrapped into `memcpy` and the runtime's **last-error state modelled** (returned once,
  then cleared) so a copy or a sync can be made to fail at a chosen call number.  Asserts the turn-boundary key
  (10 ids, not the consumed 26), the refusal of a picture at or past `L`, the promote of a boundary snapshot by a
  request that drops the reasoning tokens (and that a consumed-state snapshot is NOT promotable by that request),
  `pooled[L / idx_block] == dead` after a restore with the completed rows untouched and row `L / idx_block + 1`
  never written, `block_pos` / tails taken from the boundary checkpoint, the drafter-ring window, and §5's
  three classes: 9 pre-apply refusals (`invalid`, zero CUDA calls, every buffer still at the poison sentinel, no
  error left pending), 6 injected transfer failures (`transfer_failed`, no error left pending, and the session
  shown to be half-applied), a failed dump leaving no entry and no partial file, and a store of version-2 files
  opening cleanly with 0 entries while the same files at version 3 promote.  **249 checks** (18 refusals, 6
  transfer failures).  It does NOT prove the bytes are really device memory, that `kv_stream_reset` refills slots
  (its launch is a no-op), whether a real failed copy leaves a sticky context error, or that a promoted session
  generates the same tokens - those remain the GPU oracles' job.

Results:
- **P0 (v3, on device, 2026-09-29)**: ALL PASS - restore-exactness (post-restore hash == dumper's hash; the
  hash lines now include `dead` and the spare pooled row, the fields v3 added), resume engages
  (`RESUME 4107`), the corruption negative control is refused and correctly classed (`refused`, never
  `transfer`). This was the oracle's first execution against this branch; it found a harness bug (ROOT was
  resolved after `cd`, so the script sourced another tree's header constants) which is fixed.
- **Steps 1-3 (v3, on device)**: ALL PASS - one growing file per conversation (supersede), a promote after a
  process restart read only 13 fresh tokens, the byte cap evicted correctly. First execution also found and
  fixed an unset-`LD_LIBRARY_PATH` abort under `set -u`, and the scripts now accept `NVME_ENGINE` so a worktree
  build can be the engine under test (previously they always ran the checkout's own binary).
- **Image path (v3, on device, `tools/nvme_image_promote_test.sh`)**: ALL PASS - the first end-to-end exercise
  of the image path anywhere (every earlier test was text-only, which is exactly how two image defects - §4.8 -
  survived). An image conversation dumps at the turn boundary with `n_imgs == 1` and the key inside the user
  turn; a second process re-sending the conversation PROMOTES it (19 of a 73-token conversation); the same grid
  with different embeddings at the same position does not promote. No vision weights needed: `--vision` takes a
  GENI request carrying an embeddings file, so synthetic rows exercise the same ImgKey plumbing - the tier keys
  on the hash of the grid and the rows, not the picture.
- **Failure contract (v3, on device, `tools/nvme_failure_contract_test.sh`)**: ALL PASS - the first execution of
  a failing *transfer* anywhere (§5.0 explains why that gap hid the never-running fallback). `invalid`: a stored
  snapshot with 64 corrupted GDN bytes is refused at promote, the entry is dropped (file unlinked), the engine
  re-reads the prompt and keeps serving. `transfer_failed`: the test-only `STRATA_TEST_FAIL_CUDA` hook (default
  inert, asserted inert by the host fixtures and by P0's clean restore) breaks the named transfer - gdn, the
  spare-row re-publish, or the pre-apply sync - and the startup path exits 1 naming the class, while the
  operator-facing PROMOTE path prints `nvme promote FAILED (transfer) … not attempting a clean reset`, the `ERR`
  line, and leaves the snapshot on disk. A fresh process then serves normally: the new process IS the
  supervisor-restart recovery.
- **Restore staging, measured (`tools/nvme_restore_rss_probe.sh`)**: restoring a 964 MiB snapshot (58,513
  tokens) peaks at 45,605 MiB RSS over a 43,548 MiB steady-state engine - a ~2 GiB transient on a 62 GiB host,
  roughly twice the file size. This is the number that sizes the chunked-`pread` fix in §7.
- **No base regression from the tier**: `tools/needle_bench.py` (1k/32k, tier off in both configs) is identical
  across the pre-tier and converge binaries - 6/6 found, same timings.
- **GPU parity suite**: 26/28 with a free GPU; the two failures are environmental/upstream (`ple_parity` needs a
  Q2_0 fixture this machine lacks; `cuda_device_selftest` enforces an sm_120 policy in `device_info()`, which
  only the selftest calls - the engine itself runs on this sm_89 card). Both documented, neither ours.
- **Live needle test** (v2 era, three ~110k-token sessions, needle ~900 tokens in, rotation): all three
  needle turns promoted from NVMe (`reused 110,370/110,281/110,391 + 39/42/42 fresh read` vs
  ~110 s full prefill cold); TTFT 110 s -> **1.3-1.9 s** warm-cache / ~4 s cold-cache (the ~1.8 GB
  snapshot read dominates); needles retrieved (one empty reply was the reasoning budget exhausting
  `max_tokens`, reproduced and ruled out as a cache issue).
- **Short correctness suite** (~3k-token sessions): multi-turn chaining avoids prefill on every
  turn; branching creates isolated snapshots and both branches retrieve the same needle;
  shared-prefix sessions answer their own needles with no cross-contamination; a 3-session
  rotation x 2 rounds is 6/6 correct; an identical repeat turn is served without prefill and adds
  exactly one per-turn snapshot; a corrupted snapshot is refused ("integrity check failed"), the
  request falls back to a full re-prefill and still answers correctly.  **Read that last result as §5.1, not as
  §5.2**: a corrupt file is refused before the tier writes anything, so this suite has never exercised a
  transfer failure - which is why the never-running fallback §5.0 describes survived all of it.
- **The `KV` line on a live server - RUN, ALL PASS** (web-plan step 2, C11's engine half; run on this machine's
  4090 with the production engine stopped and `NVME_ENGINE` pointed at the tree under test, 2026-09-30):
  `tools/nvme_steps123_test.sh` asserts one `KV start=1 entries=` line after `READY`, one `KV src=` line per
  `DONE` line, and that the capped process prints a `KV` line with `evict>=1` and non-zero `evict_bytes` - and
  it passed: 3 `KV src=` lines for 3 `DONE` lines in each of the three processes, exactly one `start=1` each,
  and the cap turn reported `evict=1 evict_bytes=178042216`.  `tools/nvme_failure_contract_test.sh` asserts the
  refused promote reports `src=none refused=1 transfer=0`, and that a dying engine's `KV … transfer=1` lands on
  stdout strictly BEFORE its `ERR` line and the exit 1 - and it passed, on all three CUDA hooks: the promote-path
  line carried `transfer=1 … total_transfer=1` on stdout line 4, the `ERR` line was line 5.
- **Three oracles had been silently asserting a tier they never asked for** (found by running them, not by
  writing them): `nvme_steps123_test.sh`, `nvme_failure_contract_test.sh` and `nvme_delta_p0_test.sh`'s `runv3`
  all read v3 snapshot files (`kv-*.bin`) while passing only `--kv-nvme` - and the delta tier has been the
  DEFAULT cascade since Phase 6 (`kv_delta = 1`, `generate.cpp:298`), so with `--kv-nvme` alone a turn appends
  chunks, states and a manifest and writes NO snapshot.  steps123's helper died on the empty glob and Step 2
  then read `RESUME 0` off a store it believed held a snapshot; failure_contract had nothing to corrupt; and
  delta_p0's `--nvme-restore ""` died with rc=2 while the §5.2 comparison's other half never existed.  Each now
  names its tier with `--kv-delta 0` (`tools` commit).  **The lesson is the oracle's, not the tier's: a gate that
  cannot find its fixture must fail loudly, and this one reported a PASS-shaped silence for months.**
- **Why `tools/short_tests.py` cannot be a `KV` gate**: the server hands the engine's **stderr** to the log file
  (`serve/server.py:159-160`) and keeps **stdout** on a pipe, so a log tail never sees a `KV` line.  The two shell
  oracles above capture engine stdout directly; the serve-side assertions live in `serve/test_server.py` (80 host
  tests), which read the facts off `/metrics` and `/cache` instead of off a log.
- **Byte and decision neutrality - RUN, ALL PASS** (the line must not have changed what the tiers write or choose).
  `tools/nvme_p0_test.sh`: `RESUME 4104` and the post-restore `STATE_HASH` equal to the dumping process's `DONE`
  hash (bit-exact dump/restore), negative control fires, a corrupt file refused as `invalid` and not as a
  transfer.  `tools/nvme_delta_p0_test.sh`: P1 holds - the v3 snapshot and the delta manifest key the SAME 2355
  tokens and their restored-prefix hashes are equal across the two tiers; forks: 10 chunk FILES for 28
  references (content-shared - the 64-block grouping collapsed the old 644-file/1176-ref shape ~64×, and
  `f665d14` moved this oracle's threshold to shared span-chunks accordingly); the cascade wrote 113.5 MiB for
  turns adding 16 new tokens, flat within 3.5 MiB (3.1 % of the floor); crash points C1..C5 each relaunched,
  promoted and swept exactly.  14 PASS, 0 FAIL (re-run on the merged binary, 2026-09-30).
- **The web page against a real engine - RUN** (the worktree's `serve.server` + worktree binary on port 8090,
  a scratch store, the production engine off the GPU; this is the first time the whole chain - store counters,
  `KV` line, `_pump`, `KvCache`, `/metrics`, `/cache` - was exercised on silicon):
  - a cold 1,458-token turn: `src=none`, the cascade wrote 140 MB (`dump_ms=600`);
  - the same conversation after an engine restart, i.e. a real promote seen through the page: `src=delta
    resume=1453 promote_ms=581 promote_bytes=140319780 staging_bytes=140293564`, the prompt read in 652 ms
    instead of 1751 ms, `totals.reused_from_disk` 1453, and the events table holding a promote row AND a cascade
    row whose `tokens` is `null` (the line carries no dumped-token count, so the page renders `–`, never 0);
  - §5.1 seen through the page: a promote refused because a chunk `short_tests` had corrupted is *shared* into
    this conversation's manifest - `refuse` + `sweep` rows, `src=none resume=0`, the prompt re-read (1702 ms),
    the answer served correctly, the orphan swept as 0.11 GiB.  A corrupt chunk in one conversation is a refused
    promote in another, which is exactly why the cap accounting counts a shared chunk once per manifest;
  - **the two books re-measured on the merged binary, live, post-redeploy** (after the gate run's test traffic
    grew the store): cap accounting `delta_bytes` 25,008,877,848 vs the walk's delta subtree 25,008,877,848 -
    **byte-exact**, over 5 v3 snapshots + 84 manifests + 3,808 chunks + 84 states (3,981 files).  This is not
    luck and not an invariant: since `6648be7` the sweep runs after EVERY turn's dump (`kv_delta_enforce_cap`
    sweeps unconditionally, `generate.cpp`'s cascade tail), so on the live serve path the sawtooth sits at its
    snap-back point and the books meet the disk at page-refresh time.  The pre-merge 1.89× figure was the old
    binary's drift, measured before the recompute existed - kept here only as the history of why the two
    quantities are labeled apart.  They remain TWO measurements (the walk counts what the volume holds,
    including v3 snapshots and files the engine refuses to promote; the books count per-reference between
    sweeps) and are never merged into one "cache size".  A promote seen live through the page on the same
    redeploy: `src=delta resume=3018 promote_ms=560 promote_bytes=164173592 staging_bytes=164172720` - the
    restore path is untouched by the merge, re-verified; and a live small turn's cascade write: 118.9 MB
    (~113.4 MiB) for a turn adding a handful of tokens, the State floor.
- **What the live run also caught, in the wiring not the tiers**: a tier-on server reported `enabled: true` with
  every store fact NULL until the first request's line arrived, because `main()` attaches the `KvCache` after the
  engine is constructed and the pump had often already consumed the `start=1` line.  No unit test could see it -
  every test attached the cache by hand, before the line.  Fixed by `StrataEngine.attach_cache()`, which replays
  the state the pump kept (`serve` commit bf69e20); verified live with ZERO requests, `ram_tier` still honestly
  null because `checkpoints`/`live` ride the request line, not the store line.
- **`tools/needle_bench.py` with the tier on - RUN**: 3 of 3 needles FOUND at 32K (depths 10/50/90).  Note what
  the `KV` line says about those runs, because it is the reason the line exists: all three are `src=none` with a
  555 MiB cascade each - three needles at three depths are three DIFFERENT prompts, so nothing on disk matched -
  and the 2nd/3rd were faster for the RAM prompt-cache, not for the tier.  A page that showed only wall-clock
  would have credited the disk with a saving it did not make.

### The page's metric definitions (so the page and these oracles count the same things)

The Cache tab is the consumer of these numbers, so its definitions (`docs/nvme-kv-cache-web-design.md` §6) are
binding here too - a number on the page must be a number an oracle already measures:

- **"Warm" is two numbers and the page names both.** *Stored prefixes* = `entries + delta_entries` - snapshots,
  not conversations (§7: no cross-restart identity, so one five-turn conversation is five prefixes).  *RAM tier*
  = `checkpoints` + `live`.
- **`entries_bytes` / `delta_bytes` are CAP ACCOUNTING, a SAWTOOTH, not a disk footprint.** At open, the
  delta store counts every `chunks/`+`states/` file once - including the residue a coming sweep will remove -
  and then adds each entry's own `bytes` (manifest + State + its chunks), so a chunk shared by three manifests
  is counted four times (`kv_delta.cpp`, deliberately: the over-estimate is the *safety* direction - the cap
  over-evicts rather than letting the disk grow past it).  On append the same per-reference counting applies;
  at every sweep the total is RECOMPUTED from the disk, which is where the books meet the disk again.  So the
  number drifts above the footprint as shared references accumulate and snaps back at each sweep - it equals
  the footprint exactly only right after a sweep, a coincidence the page must not present as an invariant.
  The serve-side walk (`serve/kvcache.py scan()`) reports what the volume actually holds, including files this
  engine refuses to promote.  The page labels the two *cap accounting* and *on disk* and never merges them into
  one "cache size".
- **Store fill** = `(entries_bytes + delta_bytes) / cap`, shown beside `disk_free_bytes` from
  `shutil.disk_usage(dir)`: the cap is a policy the user set, free space is what actually stops the tier.
- **Overhead is three costs, not one**: `dump_ms` (server occupancy at `DONE`), `promote_ms` (the TTFT price paid
  instead of a re-prefill), and `staging_bytes` plus the engine's own process RSS (the ~2 GiB transient measured
  above, now visible as a bump in the hardware series).
- **Endurance** is cumulative bytes written since the engine started, quoted with §10.2's ratio -
  `(total_tokens × 16 KB + 118 MB) / (new_tokens × 16 KB + 118 MB)`, ~6.5× at the production average turn - **not**
  the handoff's 33×.
- **Time saved** is derived and labelled as derived: `saved_s ≈ resume_from_disk / fresh_prefill_tok_s −
  promote_ms`, with `fresh_prefill_tok_s` measured from this server's own cold turns, not from the 110 s figure
  above.

## 7. Known limitations / follow-ups

- **Restore reads the whole file into RAM** (atomicity) - a streamed `pread` directly into the
  pinned buffers with size-then-digest validation would halve promote time and drop the transient
  buffer. This is the path to sub-second TTFT for 128K sessions (currently ~2-4 s cold). **Measured**
  (§6): a 964 MiB snapshot costs ~2 GiB of transient RSS over the engine's 43.5 GiB steady state.
- **Layer-split sessions are not snapshot-able** (§3) - the envelope carries the primary stage only, the
  dump refuses a split session, and the serve loop does not promote under a split. Disk support would be a
  version bump with the stage blobs as first-class segments.
- **The v2 originals of the converted store live in `/local/strata/kvstore-v2-backup/`** (106 files,
  ~200 GiB) until deleted. Delete only after a few days of production promotes from converted files; the
  converter and its verification are in the branch history if they are ever needed again.
- **Metrics (issue #57's C11) - closed as a build** (web-plan steps 1-7, `docs/nvme-kv-cache-web-design.md`):
  the tiers return what they did (`TierActivity`, step 1); the serve loop prints one `KV key=value` line per
  request before `DONE`, plus the store's own `KV start=1` state after `READY` (step 2); `serve/kvcache.py`
  parses it, walks the store directory and feeds `GET /cache` and the `cache` block of `/metrics` (steps 3-5);
  the web Cache tab, the Monitor's Cache column and the About card render it (steps 6-7). A log reader is no
  longer the only way to tell the three failure classes apart, and the live-server `KV` gates have since been RUN
  on silicon, ALL PASS - including a promote, a refusal and a sweep seen through `/metrics` (§6). What the page
  does **not** do: it cannot mutate the store - `/cache` is a GET, the only POST paths
  are `/settings` and the two chat routes, and the engine's stdin takes `QUIT`/`STOP`/`GEN`/`GENI` only - and it
  has no conversation identity across a restart, so its "stored prefixes" are prefixes, not conversations - the
  next bullet.
- **Per-turn snapshot accumulation** (no cross-restart supersession) - bounded by the cap; a
  conversation identity would enable per-conversation supersession.
- **Sparsity** (Step 4, default skip): only the blocks the QSA selection can reach need storing;
  a correctness project (it interacts with the indexer's top-k selection), only worth it if
  profiling shows capacity or promote time is the bottleneck.
- **Determinism**: engine decode is run-to-run nondeterministic (pre-existing in 0.1.13's MMQ
  path); judge restores by state-hash equality and answer coherence, never by token equality
  across runs.
- **A transfer failure ends the request and the process** (§5.2).  The supervisor in `serve/server.py:686-695`
  restarts the engine on the next request, so the cost is one restart (~1-2 min), not a lost conversation - but
  the only route to recovering a promote in process is the device-usability probe §5.4 describes, and that probe
  is unvalidated on this machine.  Until a GPU run shows what a real failed host-to-device copy does to the
  context, "the engine stopped" is the honest answer.

## 8. Prior art this design was checked against

vLLM's Automatic Prefix Caching (radix-tree content addressing - adopted in spirit: automatic,
prefix-keyed; simplified to exact keys for a single-session server) and its tiered KV offloading
(host-primary, secondary tiers, cascade/promote, LRU, ref-counts - adopted; multi-session host LRU
rejected as a rewrite of the single-session arena); llama.cpp's `--slot-save-path` (rejected:
client-initiated) and its hybrid-model restore no-op bug (#26676/#25913 - the cautionary tale
behind rule §4.2). The full reviews that shaped the v1 (glm-5.3 design critique and implementation
review) are in the branch history: commits `55e3337` (initial) and `0c7e6f7` (turn-boundary fix)
reference them; they were removed from `docs/` at completion.

## 9. The shared conversation core (issue #57) — the convergence record

This tier did not grow up alone. Issue #57 (`jeremiahritchey`) proposed one shared capture/restore core for
conversation state - RAM tier 1, this NVMe tier as optional tier 2 below it - and this branch adopted it. This
section is the settled record of that convergence; it replaces the running "collision ledger" a working file
(`docs/nvme-kv-cache-design.md`, now deleted) kept while the work was in flight.

### 9.1 What was adopted, and what each side contributed

The shared core (`include/strata/core/conversation_cache.hpp`, `conversation_snapshot.hpp`,
`conversation_memory.hpp`, `src/core/conversation_state.cpp`, `conversation_snapshot.cpp`,
`conversation_checked.hpp`) is imported **verbatim** from `feat/conversation-cache-shared-core` except for header
comments, the export of `conversation_geometry_key` (the disk adapter keys its files on the same array the RAM
tier's `SavedConversation` holds - one geometry identity, not two), and 0.1.21's `used` / `stage_parts` port.
The core's own fixtures (1,805 checks: cache 35, memory 23, validation 780, transfer 1,020) pass here unchanged,
which is what makes "one vocabulary in both tiers" more than a rename: the core's `ConversationRestore` enum is
the failure contract's vocabulary (§5), its `conversation_state_sizes` is the one place running-state byte counts
are computed, and its `conversation_checkpoint_{save,restore}` are what our serve-loop wrappers call.

What the NVMe side contributed that the RAM core did not have: the disk envelope itself (versioned format,
geometry key, digest footer, atomic publication, refusals instead of conversions), **turn-boundary keying** -
the RAM image parks the *consumed* state including the model's hidden reasoning tokens, which a chat client
re-sending history can never prefix-match, so any spill must re-key at the turn boundary (§4.4) - and the
failure contract's on-device proof (§5). The answer to the core author's open question - *how would #52 consume
a snapshot without unbounded promotion staging?* - is: drop the whole-file buffer, `pread` straight into the
pinned host pools (they are device-mapped, so KV needs no `cudaMemcpy` at all), admit before the read against
the bounded chunk, and hold one `open()` fd across validate → apply with `fstat` before/after plus
write-temp-then-rename publication for TOCTOU. The staging cost of not doing it is now measured (§6: ~2 GiB
transient for a 964 MiB snapshot); the fix itself is still §7's first item.

### 9.2 The collision ledger, settled

Each item below was a real incompatibility between the two designs when the work started. Status is current.

- **C1 - turn-boundary keying**: settled as §4.4 (the spill rule) plus two defects the audit found in OUR tier:
  `NvmeEntry::imgs` was never filled and a boundary dump stored the live image list - any conversation with a
  picture was unreachable (§4.8). Still open: the *policy* question of whether a RAM-tier eviction also spills
  through `nvme_dump_at`, and what spilling a parked (rather than live) session costs.
- **C2 - checkpoint blobs**: settled - the shared checkpoint carries `dead` and `block_pos`, so a turn-boundary
  checkpoint describes the indexer completely; 0.1.21 added `used`/`stage_parts` on top (§3).
- **C3 - the spare pooled row**: settled - the "the `b == n_bid` path masks it" hope was false (two of three
  readers score row `n_bid` out of the pool), so restore re-publishes `dead` into row `L / idx_block` (§4.7).
- **C4 - pooled-row formula**: settled - the shared core's `L / idx_block + 1`, stated once
  (`strata::kernels::qsa_pooled_rows`) and used by the envelope, the fingerprint and the core; our `+ 2` wrote
  one row no reader can reach. Refuse-instead-of-truncate came with it.
- **C5 - whose `dead` / `block_pos` the envelope owns**: settled - the boundary checkpoint's. The live
  `idx_block_pos` names a block completed by tokens past the boundary, so reading it off the device wrote a
  running-state value that did not describe the keyed prefix (§2). This is what bumped the format to v3.
- **C6 - the drafter ring's home**: settled - the disk adapter refills it right after its apply pass, and the
  serve loop keeps only the resumes that never touched the adapter (`from_nvme` gate). Collapse condition: this
  is correct *while the adapter reads straight into the pinned pools*; it folds into
  `conversation_kv_restore` (which already calls `kv_ring_restore`) the moment the adapter adopts that wholesale.
- **C7 - the failure contract**: settled as §5, executed on device (§6).
- **C8 - one state hash**: settled - `state_hash_line()` is the single implementation, matching the shared
  formula (`dead`, the spare row via `qsa_pooled_rows`, `ss.max_cells`), used by the DONE line and
  `STRATA_NVME_HASH` alike. A hash that could not see the row C3 fixes would have been a weak oracle for C3.
- **C9 - the disk-envelope boundary**: settled for the format (v3, §2) - see the inventory below. A historical
  note: the conversion boundary forbids serializing C++ structs; `NvmeHeader` still *is* one (fixed-width
  fields, layout pinned by `static_assert`s on size and offsets), and the header sits outside the digest with
  its layout pinned at compile time instead. That is the honest remaining divergence from the strict boundary.
- **C10 - physical-RAM admission**: NOT settled. The restore still reads the whole file (§7, measured ~2 GiB
  transient); the core's `conversation_available_memory` / `conversation_memory_admit` are imported dormant
  (`conversation_memory.cpp` builds only the memory fixture, not the engine).
- **C11 - metrics**: settled as a build (§7): `TierActivity` out-params, the per-request `KV` line and the
  `KV start=1` store state, `serve/kvcache.py`'s parse + store walk behind `GET /cache` and `/metrics`, and the
  web Cache tab. The page reads and never mutates (`/cache` is a GET; no POST path reaches the store), and it
  still has no conversation identity across a restart. Its live-server gates are RUN, ALL PASS (§6).

### Where step 2 leaves NvmeHeader

What step 2 inherited, what v3 is now. The v2 header violated the shared core's disk boundary four ways: a raw
C++ struct `memcpy`ed into the envelope (implicit ABI, no stated offsets/widths/endianness); no versioned
envelope (exact-equality version check, every segment length re-derived from the live engine at restore, so a
sizing change surfaced as "layout mismatch" after the fact); the payload digest covering only `[sizeof(header),
at)` - the one thing that could move every segment was the one thing the footer could not see; and a derived
8-field geometry tag that was neither the core's 18-field key nor a disk schema. v3 answers each: the header is
208 bytes with `static_assert`s pinning its size and the offsets of `L`, `geometry`, `page_size` and `mtp_host`
(a reordered field fails the build instead of re-mapping segments); version 3 is refused by name before any
segment is walked; the geometry is the shared core's 18-field key verbatim with `page_size` / `idx_block` /
`max_cells` carried as separate *validated* runtime shapes (a reader must check them, not re-derive them); and
the oracle scripts derive the header offset and format version from the same constants the asserts pin
(`tools/nvme_header_layout.sh`). What remains deliberately un-done is the full segment table (offset+length per
segment as fixed-width integers) and field-by-field encoding - the layout is still implicit, just compile-time
pinned; that is the next envelope revision, and it is not needed until a format change actually happens.

### 9.3 Steps, for the record

The five alignment steps (rebase onto the shared core's base; adopt the core boundary; turn-boundary spill
semantics; fixtures and one state hash; the failure contract) were executed serially on this branch, then the
whole branch was rebased onto 0.1.21 and the GPU-phase validation was run (§6). The step-by-step running
commentary lived in the deleted working file; what it found that is still true lives in §4, §5 and this section.

## 10. The delta tier (the handoff's delta record family, Phases 0-6)

`docs/nvme-delta-cache-handoff.md` is the build record; this section is what it converged to. The delta tier is
a NEW record family beside the v3 snapshots - the snapshot format stays v3, `NvmeHeader` is untouched, nothing
converts:

    <kv-nvme dir>/
      kv-<pid>-<seq>.bin               # v3 snapshots (unchanged; age out under the shared LRU)
      delta/
        chunks/<16-hex key>.bin        # sealed, block-aligned, content-addressed
        states/<16-hex digest>.bin     # per-turn State record (the ragged tail + the running state)
        log-<pid>-<seq>.manifest       # one per conversation head; superseded by unlink

### 10.1 What it changes

- **The DONE cascade's serialization**: instead of `KvNvmeStore::dump` rewriting a whole v3 snapshot, the delta
  path (`delta_dump_at`, §5.9 of the handoff) writes ONLY the sealed chunks the previous head does not already
  cover, plus one State record, then moves the manifest head (new manifest fsynced and renamed BEFORE the old
  head is unlinked - the commit point; the crash matrix's C4). Chunks are exact contiguous slices of the v3
  segments, keyed by an FNV chain over the conversation's ids through the chunk's last token - so forks share
  the shared prefix's files by construction and stale chunks are unreachable.
- **The restore** reassembles the exact v3 image (chunks + State + the manifest's ids/imgs/header fields) and
  runs `nvme_restore_image` - the v3 apply pass extracted line-for-line from `nvme_restore` - so every validated
  property of the v3 path is inherited, not re-derived. THE INVARIANT (§5.2): for the same session and boundary,
  the reassembled image is byte-identical to what `nvme_dump_at` writes; asserted by the host fixture's `cmp`
  oracle and on device by `tools/nvme_delta_p0_test.sh` (STATE_HASH over the restored prefix).
- **The failure contract is unchanged and mapped** (§5.13): every manifest/chunk/State problem - magic, version,
  geometry, format, BLOCK-vs-lcm, weights, digests, missing or short records - is `invalid` (recoverable, zero
  CUDA calls, nothing written); only the apply pass can report `transfer_failed`. A corrupted record degrades to
  refuse-and-drop, and the conversation's next dump REPAIRS the store (the re-dump re-derives every chunk and
  rewrites them under their content keys - measured: 1513 fresh records, 0 bad after S3's corruption).
- **Eviction/GC**: ONE byte cap across BOTH tiers (`kv_delta_enforce_cap`), LRU by mtime; eviction of a delta
  conversation is just its manifest's unlink; the sweep (mark-and-sweep, marks read from the manifests ON DISK
  - the in-memory list can be behind a C4/C5 dump that committed its manifest and failed after) reclaims exactly
  the unreferenced set. Fork sharing survives eviction by construction (the P2-2 regression test).
- **The limits (§5.15, stated not hidden)**: a boundary past the drafter ring (`T > max_cells`) refuses and the
  whole-snapshot v3 fallback writes it (log line `nvme delta: boundary N exceeds the drafter ring (M) - whole
  snapshot`); layer-split sessions are inert (both tiers); one writer per store; whole-file staging at restore
  (the accepted C10 cost).

### 10.2 Measured, and WHERE THE HANDOFF'S ESTIMATE WAS WRONG

`tools/nvme_delta_p0_test.sh` (ALL PASS) and `tools/short_tests.py` (19/19 with `--kv-delta 1`) are the gates.
The 2026-09-30 re-run of both on the `KV`-line build: ALL PASS - delta_p0 14 checks (the "19/19" above counts
short_tests' VERDICT line as a check; that suite prints 18 assertions and then the verdict), and the §5.2
cross-tier P1 comparison only passes once `runv3` is told to turn the delta tier off (§6).
The numbers that matter:

| quantity | v3 cascade | delta cascade |
|---|---|---|
| snapshot/state at 2,355 tokens | 154 MB (= 36 MB KV + **118 MB running state**) | State record 112.7 MB + manifest |
| per-turn write, 217 new tokens | ~154 MB (whole rewrite) | **115.8 MB** (112.7 MB State + 3.0 MB chunks) |
| per-turn write vs session length | grows with the prefix | FLAT (2.6% spread over 4 turns) |
| chunk files, 3 conversations sharing a prefix | 3 × whole snapshots | 644 files for 1,815 references (588/589 shared) |

**The handoff's §2 estimate ("~34 GB per window, 33x") omitted this model's RUNNING STATE**: the GDN recurrence
state is ~112 MB per turn, and §5.5 carries it per turn BY DESIGN (the C5 rule: every running-state byte comes
from ONE source, the boundary checkpoint). The write reduction is therefore NOT 33x - it is
`(total_tokens x 16 KB + 118 MB) / (new_tokens x 16 KB + 118 MB)`: ~6.5x at the production average turn
(49k-token sessions, ~1.5k new tokens), ~2x on short sessions, growing toward the KV-only ratio as sessions
grow. The tier still wins (and the SSD-endurance picture with it), but the honest ratio must be used in any
endurance claim. A follow-up that would move the number: chunking the GDN/PLE state itself is NOT possible
(it is a recurrence, not an append-only log); the lever is writing the State record LESS OFTEN (e.g. every N
turns or at eviction - the maintainer's original "write when evicted from RAM" reading), trading a longer
promote tail for the per-turn State write.

### 10.3 Wiring

`--kv-delta N` (default 1), beside `--kv-nvme` (which stays opt-in - a plain engine runs nothing new). The serve
loop: the DONE path routes to the delta dump only on its own conditions (a turn boundary, no layer split,
`T <= max_cells`), everything else to the v3 cascade; the promote is the longest match across BOTH tiers
(`kv_nvme_match` reused verbatim per tier); the weight-set fingerprint (§5.8: each model shard's path, size,
first and last 64 KiB) is checked at match time. `NvmeEntry.kind` (0 = v3, 1 = delta manifest) dispatches the
restore; the failure handling is byte-for-byte the serve loop's old rule.
