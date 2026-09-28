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
   refused by version (naming the one found) before any segment is walked.
6. **Payload digest** - hashed during the write with the same hasher that feeds the file (a plain
   write inside the payload silently desynchronizes the digest from the bytes; this exact bug
   shipped briefly and was caught by the corruption test).
7. **The spare pooled row is part of the snapshot, and a restore re-publishes it.**  A prefix of `L` cells owns
   `L / idx_block + 1` pooled rows - the completed blocks plus the spare at `L / idx_block`, which the writers
   keep equal to `dead` (`qsa.cu:213`).  A turn-boundary dump reads that row from a live array a longer turn has
   already overwritten, so `nvme_restore` re-publishes `dead` into it, as the shared core's
   `conversation_checkpoint_restore` does.  `STRATA_STATE_HASH` spans the same rows, so the DONE-vs-restore
   comparison can actually see the difference.

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
- **P0**: restore-exactness PASS (post-restore hash == dumper's hash, deterministic across runs);
  resume engages; negative control refuses. Token-equality across runs is informational only
  (engine decode nondeterminism, pre-existing - reproduced on the old binary).
- **Steps 1-3**: one growing file per conversation; promote after restart resumed 4235 tokens and
  read 13 fresh in ~350 ms; cap evicted to 0.83 GiB under 1 GB.
- **Not re-run since the v3 header.**  Step 3 widened the header to 208 bytes and step 4 changed the state-hash
  field set, so the two results above are the v2-format record.  Both scripts now take the offset and the version
  from `kNvmeHeaderBytes` / `kNvmeFormatVersion` through `tools/nvme_header_layout.sh` (and refuse a snapshot of a
  version they do not write), but neither has been run on this branch: the GPU holds a live ~24 GB engine.
- **Live needle test** (three ~110k-token sessions, needle ~900 tokens in, rotation): all three
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

## 7. Known limitations / follow-ups

- **Restore reads the whole file into RAM** (atomicity) - a streamed `pread` directly into the
  pinned buffers with size-then-digest validation would halve promote time and drop the transient
  buffer. This is the path to sub-second TTFT for 128K sessions (currently ~2-4 s cold).
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
