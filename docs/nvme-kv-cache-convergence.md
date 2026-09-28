# Aligning the NVMe cold tier (#52) with the shared conversation-snapshot core (#57)

**Status: internal working record. Not a PR, not a GitHub comment, not an agreed design.**

This branch (`nvme-kv-cache-converge`) exists to test whether our NVMe tier can sit on top of the
shared capture/restore core that `jeremiahritchey` proposed in issue #57, before anyone commits to a
public position.

## The three efforts

| Issue | Tier | Where |
|---|---|---|
| #41 `@midhatn` | one primary conversation parked in RAM around auxiliary calls | superseded if the combined design covers it |
| #52 `@maedoc` (us) | whole-session snapshots on **NVMe**, survive restarts | `src/platform/kv_nvme.cpp`, `include/strata/platform/kv_nvme.hpp`, `docs/nvme-kv-cache-design.md` |
| #57 `@jeremiahritchey` | several conversations parked in **host RAM**, LRU, longest exact prefix | `jrich/feat/conversation-cache-shared-core` @ `3657b8f`, based on upstream `b38c183` (0.1.18) |

Niko's proposed shape (issue #57 comment `5870584130`): **one capture/restore core**, RAM as tier 1,
NVMe as an optional tier 2 below it, sized by the conversation rather than by `--max-context`.

## What their core is

```
include/strata/core/conversation_cache.hpp     ConversationImageKey, ConversationCheckpoint,
                                               ConversationKv, SavedConversation,
                                               ConversationCache (byte budget + slots + LRU + prefix match)
include/strata/core/conversation_snapshot.hpp  conversation_state_sizes
                                               conversation_checkpoint_{save,validate,restore}
                                               conversation_kv_{bytes,save,validate,restore}
                                               conversation_snapshot_{bytes,save,validate,restore}
                                               enum ConversationRestore { restored, invalid, transfer_failed }
include/strata/core/conversation_memory.hpp    conversation_available_memory()  (MemAvailable / GlobalMemoryStatusEx)
                                               conversation_memory_admit(available, allocation, floor)
src/core/conversation_checked.hpp              overflow-checked add()/product()
```

Their `generate.cpp` retypes `ConvCheckpoint` -> `strata::core::ConversationCheckpoint` and
`ConvStateSizes` -> `ConversationStateSizes`, and reduces our `checkpoint_save` / `checkpoint_restore`
to one-line wrappers over the core.

## Where the two designs already agree

1. **The payload is the same payload.** `ConversationKv{k, v, k_scale, v_scale, pooled}` + format +
   page-rounded `cells` + identity layout is what `nvme_dump`/`nvme_restore` already write and read.
2. **The pinned host arena is the source of truth.** Their `pools()` reads `st.host.*` whenever
   `kv_mode != 0`, and `valid()` *requires* `st.host.present()`. That is our founding rule: NVMe never
   touches the GPU.
3. **Residency contract matches.** Their `conversation_kv_restore` calls `kv_stream_reset(st.map, nullptr)`
   per layer; so does ours.
4. **Refuse, never convert.** `image.geometry != geometry_key(g)` rejects; so does our header tag check.
5. **No flag collision:** `--conversation-cache-mib/-slots/-min-free-mib` vs `--kv-nvme DIR` / `--kv-nvme-max GB`.

## Where they collide (the work)

### C1 - turn-boundary keying (our contribution, and a trap in their spill model)

Their RAM image parks `live` = *consumed* tokens: the prompt **plus the model's generated and hidden
reasoning tokens**. `conversation_prefix()` requires those ids to be a prefix of the incoming prompt.
A chat client re-sends history **without** the hidden reasoning tokens, so a parked image spilled
verbatim to disk can never full-prefix-match the next turn.

We already hit and fixed exactly this: commit `0c7e6f7` / `nvme_dump_at`, which re-keys the snapshot at
the **chat turn boundary** (the longest `ConvCheckpoint`). Our first live HTTP test showed every
request falling back to re-prefill before that fix.

**Rule for any spill path: a spill re-keys at the turn boundary. It does not copy the parked image.**

**Made explicit in step 3** (the rule was already the rule; what was missing was where it is written down): the
disk-adapter API states it on `nvme_dump_at` itself - it is the ONLY path to disk, it writes the boundary's ids,
the boundary's pictures and the boundary's running state, and all three describe position `L`; a picture at or
past `L` is refused rather than written.  `docs/nvme-kv-cache-design.md` §4.4 states it as a correctness rule.
The envelope now says the same thing in its own fields: the geometry key, `L`, the ids, the image records and the
running-state segments are all the boundary's (C5), and the pooled row count is a function of `L` alone (C4).
What remains open is C1's *policy* question below - whether a RAM-tier eviction also spills through `nvme_dump_at`
- not its semantics.

### C2 - `ConvCheckpoint` gains `dead` and `block_pos`

Theirs: `gdn, ple, tails, dead, block_pos`. Ours: `gdn, ple, tails`. Our `NvmeRunning{gdn, ple, tails}`
hand-off into `nvme_dump_at` must grow to carry the indexer spare key and block position, or a
turn-boundary snapshot's indexer state is incomplete.

### C3 - spare-row reconstruction at the resume point

`idx_dead` is the **cell-0** key (`qsa.cu:187`, written only when `pos == 0`), constant for the whole
sequence - so dumping the live `idx_dead` at a turn boundary is correct.

But `pooled[n_bid]` (the row of the currently *incomplete* block) is supposed to hold `dead`
(`qsa_select.cu:35` reads `dead` directly for `b == n_bid`). Our `nvme_dump_at` dumps `idx_pooled` rows
`[0, T/idx_block + 2)` from the **live** array, where row `T/idx_block` has since been overwritten by a
completed block from tokens past `T`. Their core re-publishes `pooled[ids.size()/idx_block] = dead` on
restore. Probably masked by the `b == n_bid` path, but it is exactly the class of omission their audit
caught, so it needs a fixture before we claim ours is clean.

**Settled: their re-publish rule is adopted, and the "already masked" hope is false.**  The `b == n_bid` masking
exists in ONE reader - `qsa_select.cu:35` reads `dead` instead of the pool for the block in progress, so that
kernel's highest pooled read is `n_bid - 1`.  The other two score row `n_bid` out of the pool itself:
`qsa_index_kernel` guards `b > n_bid` and then reads `pooled[b * idx_dim]` (`qsa.cu:253`, `:260`), and the native
scorer gates `row <= full` (`native_qsa_score.cu:74`).  At resume `n_bid = n_kv / idx_block = L / idx_block`
(`qsa.hpp:156`, `layer.cpp:813`) - exactly the row a boundary snapshot holds stale, because the pooling kernel
rewrote it with the completed block's key (`qsa.cu:211`) when tokens past the boundary finished that block.  So
the invariant `pooled[n_bid] == dead`, which the writers maintain at every completion (`qsa.cu:213`,
`native_qsa_indexer.cu:93`), must be RESTORED, not assumed: `nvme_restore` re-publishes `dead` into row
`L / idx_block` of every QSA layer, the same rule `conversation_checkpoint_restore` applies
(`conversation_state.cpp:186`).  For a full-`L` dump the row already equals `dead`, so it is a no-op there.
**Proved in step 4**: `kv_nvme_host_test` dumps a non-block-aligned boundary (10 tokens, `idx_block` 4) whose
`idx_dead` differs from every pooled row, reads the STALE value back out of the file, and asserts that after a
restore the row holds `dead` while the completed rows are untouched and the row after the spare was never written.
Deleting the re-publish makes the fixture fail.

### C4 - pooled row count

Ours: `pooled_rows = min(L / idx_block + 2, idx_pooled_rows)`. Theirs: `upto / idx_block + 1`.
One formula must win before either format is written down.

**Settled: theirs.**  `L / idx_block + 1` is the completed block rows plus the SPARE row at `L / idx_block`, and
the spare row is load-bearing: `qsa.cu:213` / `native_qsa_indexer.cu:93` keep `pooled[n_bid] == dead`, and
`qsa.cu:260` and `native_qsa_score.cu:74` read row `n_bid` straight out of the pool.  Our `+ 2` wrote one row
past that - what the live array happens to hold at dump time - which no reader can reach, because every pooled
reader gates on `n_bid` (`qsa.cu:253`, `qsa_select.cu:33`, which reads `dead` for `b == n_bid` so its highest
pooled read is `n_bid - 1`).  The writer does touch row `n_bid + 1` when a block completes, so a stale value
there is overwritten before any read.  Their refuse-instead-of-clamp rule comes with the formula: a live pooled
array too small for the snapshot is now a refusal naming both counts, not a silently short segment.  Stated once
in `kv_nvme.cpp` (`snapshot_pooled_rows`), used by the dump and the restore alike; part of format version 3.

### C5 - where `idx_dead` / `idx_block_pos` live

Ours: per-QSA-layer segments in the KV body of the file. Theirs: per-checkpoint blobs (live + every
checkpoint), reconstructed on restore. The disk envelope must pick one.

### C6 - drafter ring restore

Theirs: inside the core (`kv_ring_restore(...)` for `kv_mode == 2`). Ours: in the serve loop
(`mtp.kv_restore(resume)`) plus host-array dumps recorded by `NvmeHeader::mtp_host`. Pick one home.

**Settled: the disk adapter, with a collapse condition.**  `nvme_restore` refills the drafter's ring itself
(`refill_drafter_ring`, the same `[b1 - n_slots, b1)` blocks the serve loop computed) as part of applying the
snapshot it read, and the `--nvme-restore` spike's `mtp.kv_restore(rL)` is gone.  Why not their core: reaching it
means routing the drafter's KV through a `ConversationKv` staging vector, which contradicts the answer we give to
their open question (read straight into the pinned pools, bounded staging).  Why not the serve loop: the refill is
part of the residency contract for bytes the tier just wrote, and a call-site rule is exactly what a new spill or
restore path forgets.

**Collapse condition - this is a deliberate third home, not an accident.**  B is correct *while the adapter reads
straight into the pinned pools*.  The moment the adapter adopts `conversation_kv_restore` wholesale (C10 / the
staging work), `refill_drafter_ring` must fold into it, because the core would then be writing those pools.

What the serve loop kept: `mtp.kv_restore(resume)` still runs for the resumes that never touched the adapter - a
RAM checkpoint (`from_live == false`) or the live session.  It cannot be deleted, and the promote path cannot be
told apart from a live resume by `from_live` (both set it true), so a promote now sets `from_nvme` and the serve
loop's rule is gated on it.  Editing `conversation_checkpoint_restore` to cover every resume was rejected: it
would change the semantics of a file that is a verbatim copy of their core.

Cells past the promoted `L` are left in the ring on purpose.  The ring table is static (`block -> block %
n_slots`, `kv_stream.hpp:76-79`), so the slots a LONGER previous turn could have clobbered are exactly the blocks
below `b1` - which is what the refill repopulates - and the drafter's attention reads only cells below the one it
is writing (`n_kv = pos + 1`, `layer.cpp:813`), so every cell past `L` is written before anything can read it.

### C7 - failure contract

On a failed promote we currently `kvstore.drop()` + `resume = 0` + `session_zero` + full re-read - a
**clean-reset fallback**. Their contract says a transfer failure is **fatal** and "a future clean-reset
recovery needs its own proof that the CUDA context remains usable". Our fallback is only defensible if
we can prove it.

**Settled in step 5: their rule, because our fallback never ran.**  The claim that needed proving could not be
proved, and the claim that was being relied on turned out to be false.  What a failed promote actually did:
`resume = 0` -> `session_zero` (`generate.cpp:3025-3026`) -> `qsa_state_zero` (`layer.cpp:657`) -> `kv_stream_reset`
(`layer.cpp:676`) -> `check("reset")` (`kv_stream.cu:199-202`) -> **`std::exit(1)`**, because `nvme_restore`
handled a copy failure without consuming the CUDA error, and `cudaGetLastError()` hands that error to the next
caller.  The process died inside the fallback's own first step, printing the *restore's* error under the label
"reset".  That is neither our documented recovery nor their fatal rule: it is an unintended exit at an unrelated
point with a misattributed diagnosis, and `pinned.cu:170-183` is this tree having already paid for the same trap
once.  The fallback only looked like it worked because every recorded test corrupts a **file** - a refusal, which
never reaches the transfer pass - and none injects a failing **copy**.

**The contract, in one enum both tiers use** (`strata::core::ConversationRestore`, which `nvme_restore` and
`KvNvmeStore::restore` now return instead of a `bool`), written out in `docs/nvme-kv-cache-design.md` §5:

| class | what it covers | consequence | operator sees |
|---|---|---|---|
| `invalid` | every refusal **before the apply pass**: magic, format version, geometry, header sizes, truncation, the layout walk, the payload digest, a live array too small, a null target buffer | **recoverable** - drop the entry, `resume = 0`, full re-read | `nvme promote refused (<reason>); reading the prompt instead`, then `RESUME 0` |
| `transfer_failed` | a `cudaMemcpy` in the apply pass or in the spare-row re-publish, or either `cudaDeviceSynchronize` | **fatal** - `ERR …` + exit 1; the snapshot is left on disk | `nvme promote FAILED (transfer): <segment, bytes>` + `not attempting a clean reset` |
| stale-format store | a directory of snapshots this build cannot read, skipped at scan | **recoverable**, before any transfer, with an operator action | `N snapshot(s) of format version 2 in DIR: … re-dump them with the binary that wrote them; to stop the skip, remove them and let the store rebuild` |

Why the middle row is fatal and the outer two are not, as evidence rather than preference:

- **The recoverable rows are provably untouched.**  `kv_nvme_host_test` asserts each pre-apply refusal makes zero
  `cudaMemcpy` calls and zero CUDA calls of any kind, leaves every session buffer at a poison sentinel, hands the
  caller no ids/imgs, and leaves no CUDA error pending.  The clean path has nothing to undo.
- **The fatal row is half-applied by construction.**  The apply pass is a loop; the fixture injects a failure at a
  chosen copy and reports what landed - fail the second device copy and the GDN state is the snapshot's while the
  PLE history is not; fail the last one and every segment landed while `pooled[L / idx_block]` still holds the
  stale value and `dead` holds the value it must become, i.e. a session that looks restored and is one invariant
  short.  The shared core's own fixture asserts the same shape for its restore
  (`conversation_validation_test.cpp:167-169`), so this is not a difference between the tiers - it is the same
  exposure, named.
- **The stale-store row is not swept into the fatal rule.**  A v2 store is skipped at scan: `open()` succeeds, 0
  entries become promotable, 0 bytes count against the cap, the files stay on disk, and the request re-prefills -
  asserted against the same three files at version 3, which do promote.  113 GB of v2 snapshots is either a corpus
  to re-dump with the binary that wrote it or a store to delete and let refill; what it must not be is ambiguous.
- **The one proof this tier can produce today** is its own final `cudaDeviceSynchronize()`: a success there means
  the device answered after the last write.  That is why `KvNvmeStore::restore`'s stale-index case (the file
  applied cleanly but no longer matches its entry) is `invalid` while a mid-apply failure is not.

**What a future clean reset must prove before it may exist** (their sentence, made concrete; design doc §5.4):
on device, that a real failed host-to-device `cudaMemcpy` leaves a **non-sticky** context error; that the captured
graphs - captured once, before the serve loop (`session.cpp:164`, `graph.cpp:106`) - survive replay after one; and
that the pinned arena and the streamed page table are back inside the residency contract.  The mechanism would be
a **device-usability probe** (consume the error, sync, a bounded write-and-read-back through a scratch device
buffer) run before the tier reports `transfer_failed`.  **Not implemented, and unvalidated on this machine**: the
GPU holds a live ~24 GB engine and the host fixture's copies are `memcpy`.  An unvalidated guard in front of an
unvalidated recovery is worse than a clean exit.

**Why stopping is recovery, not defeat.**  Under `serve/server.py` the engine runs behind a supervisor that
notices a dead engine and starts it again on the next request (`serve/server.py:686-695`).  A new process is a new
CUDA context, a new pinned arena and new captured graphs - precisely the state the in-process reset could not
prove it reached.  The fatal rule is recovery by the only route that is actually proven here.

**Fixed on the way, because it is the same trap**: every CUDA failure path in the tier now consumes the error it
handled (`kv_nvme.cpp`, `consume_cuda_error`), the dump side included.  That does not make a transfer failure
recoverable - a sticky context error comes straight back from the next call - it makes the REPORT true, and it is
the floor below which no clean reset is worth discussing.

### C8 - two copies of the state hash

We refactored `STATE_HASH` into `state_hash_line()` so `STRATA_NVME_HASH` can print it immediately
after a restore (that is our bit-exactness oracle). They patch the inline version in `main()` and add
`dead=%016llx`, change the pooled row count, and switch `ss.qsa_states[0].max_cells` -> `ss.max_cells`.
`tools/nvme_p0_test.sh` compares the dumper's DONE hash with the post-restore hash, so both tiers must
share one formula or their evidence and ours stop being comparable.

**Settled in step 4: their field set, in our one function.**  `state_hash_line()` is the only copy in this tree and
both call sites (the DONE line, the `STRATA_NVME_HASH` line) already go through it; what changed is that it now
hashes what their `main()` hashes:

| field | before (ours) | now (theirs) |
|---|---|---|
| `dead` | not hashed | `idx_dead` per QSA layer, printed as `dead=%016llx` |
| pooled span | `L / idx_block` rows | `qsa_pooled_rows(L)` = `L / idx_block + 1` rows |
| stale clamp | `ss.qsa_states[0].max_cells` | `ss.max_cells` |
| PLE | `z.ple` bytes even with no history | `z.ple` only when `ss.ple_hist` exists |
| fp16 KV | read the int8 array names (null in fp16 mode) | the fp16 pools at `head_dim * 2`, no scale arrays |
| `STATE_HASH_GDN` | divided by `n_gdn_layers()` unguarded | guarded on `n_gdn_layers() > 0` |

**The pooled span, decided: `[0, L / idx_block + 1)`.**  The row at `L / idx_block` is the one C3 re-publishes, and
the fingerprint exists to compare a dumped session with a restored one; a span that stopped at the completed rows
could not see the difference between a restore that re-publishes the spare row and one that leaves it stale, which
makes it a weak oracle for the very fix C8 is being used to prove.  It is also the row the file carries (C4), so
the DONE-time live array and the restored array now hold the same set of rows.  The span is not a second formula:
`state_hash_line`, `kv_nvme.cpp: snapshot_pooled_rows` and the shared core's `conversation_snapshot.cpp: layout`
all read `strata::kernels::qsa_pooled_rows`.  `kv_nvme_host_test` asserts that count at the aligned and the
non-aligned boundary, that the file's pooled segment is exactly that many rows (and NOT the `+2` layout the tier
rejected), that the restore re-publishes the last of them, and that the row after it was never written.  What it
does NOT assert is the fingerprint itself: `state_hash_line` lives in the program layer and needs a live session,
so the fixture pins the one count the fingerprint spans, and the hash line stays the GPU oracle's job.

### C9 - they forbid what our file format currently does

Their boundary says the disk adapter must **not** serialize C++ structs, pointers or native vector
layouts, and must define a versioned envelope. Our `NvmeHeader` is a raw `memcpy` of a C++ struct
(fixed-width fields, so mostly fine, but implicit ABI). Our magic / version / geometry tag / FNV-1a
payload digest / atomic publication discipline is what they do **not** have and need from us.

### C10 - physical-RAM admission is missing on our side

`nvme_restore` does `std::vector<uint8_t> buf(fsize)` - the whole snapshot (1.8 GB in the needle test)
in RAM, with no admission check. Their `conversation_available_memory()` + `conversation_memory_admit()`
is the guard our own known limitation needs.

### C11 - metrics gap

Their acceptance list wants hits, reused tokens, capture/restore time, occupancy, evictions, admission
skips, disk bytes and time. Our promote/dump messages are stderr-only; `serve/server.py` and
`serve/telemetry.py` parse nothing NVMe-related.

## Where step 2 leaves NvmeHeader

Step 2 (`114300f`) moved the tier onto the shared core's **types** and left its **format** exactly where step 1
left it.  This is the inventory that `include/strata/platform/kv_nvme.hpp` (the `NvmeHeader` comment) points at,
and the list of what step 3 has to change.  Nothing here is new design: it is what C9 says our format does,
written against the code that does it.

**What is on disk today.**  `struct NvmeHeader` (`include/strata/platform/kv_nvme.hpp:33-47`): `magic`,
`version = 2`, `L`, `n_imgs`, `cvec`, `kv_format`, eight geometry fields, `mtp_host`.  Then ids, image records,
the per-layer KV / pooled / tail / dead / `block_pos` segments, the drafter arrays, and the FNV-1a footer.

**The four ways it breaks the boundary their core draws.**  Their rule (C9) is that a disk adapter must not
serialize C++ structs, pointers or native vector layouts, and must define a versioned envelope.

1. **A raw C++ struct is copied into the envelope.**  `nvme_dump_at` writes `&h, sizeof h`
   (`src/platform/kv_nvme.cpp:200`); `nvme_restore` (`:313`) and `KvNvmeStore::open` (`:438`) read it back the
   same way.  There is no field-by-field encode/decode, so the file is a picture of one compiler's struct rather
   than a described record.
2. **The ABI is implicit.**  The file states no offsets, widths, endianness or padding; they come from the
   translation unit that wrote it.  Every field is fixed-width, so on the toolchains we build with the struct
   happens to be padding-free - but nothing checks that, and a reordered or widened field would silently
   re-map every segment after it.  Worse, the payload digest deliberately covers only
   `[sizeof(NvmeHeader), at)` (`:386-388`), so the one thing that can move the whole layout is the one thing the
   integrity footer cannot see.
3. **There is no versioned envelope.**  `version` is compared for exact equality with `2` (`:314`, `:438`);
   there is no segment table, no per-segment size or offset, and no older-reader rule.  The restore
   **re-derives** every segment length from the live engine (`sizes_of()`, `qsa_real_shapes()`,
   `idx_pooled_rows`, `max_cells`) and then compares the walk's end with the file size (`:369-382`).  That is
   why a sizing change surfaces as *"layout mismatch ... idx_pooled_rows / PLE / drafter ring changed?"*
   instead of as a version negotiation: the format cannot describe itself.
4. **Its geometry tag is a derived projection, not the shared core's geometry key.**  The eight fields are
   computed, not read: `n_qsa` / `n_gdn` come from `n_layers` + `qsa_interval`, `idx_dim` is `idx_key_dim`, and
   `page_size` / `idx_block` are `qsa_real_shapes()` granules - runtime shapes, not model identity.  Their
   `SavedConversation::geometry` is `geometry_key(g)`'s 18 raw fields (`src/core/conversation_state.cpp:20-25`)
   and is labelled *"Runtime compatibility only; NOT a model/weights identity or disk schema"*
   (`include/strata/core/conversation_cache.hpp:50`).  So **neither** key is a disk schema today, and they
   answer different questions: ours asks *can this engine read this file*, theirs asks *is this the same
   conversation object in this process*.  Step 3 must not stack a third key on top of the two.

   **Settled: one key, theirs, with the runtime shapes recorded beside it.**  `NvmeHeader::geometry` IS
   `strata::core::conversation_geometry_key(g)` - the same 18 fields in the same order, compared as one array - so
   the derived projection is gone and the two tiers refuse the same mismatch.  `page_size`, `idx_block` and
   `max_cells` stay in the header as SEPARATE fields: they are not model identity and their key does not contain
   them, but a disk reader must VALIDATE them rather than re-derive them from a live engine (re-deriving is what
   turned a sizing change into "layout mismatch").  To make it one key rather than two, `conversation_geometry_key`
   left the anonymous namespace of `conversation_state.cpp` and is declared in `conversation_snapshot.hpp` - the
   only edit to that imported file, and its 18 fields and their order are unchanged.  `NvmeHeader` is now 208
   bytes, pinned by `static_assert`s on its size and on the offsets of `L`, `geometry`, `page_size` and `mtp_host`,
   because the integrity footer cannot see the header that describes the layout.  Format version 3.

**What step 3 must turn it into.**

- a **segment table**: an explicit byte offset and length per segment, written as fixed-width integers at fixed
  offsets, so a reader validates sizes instead of re-deriving them from a live engine;
- a **version and minimum-reader rule** inside the envelope, with the header itself covered by the integrity
  digest;
- **one geometry identity for the file** - their 18-field key, or a documented superset of it - replacing the
  derived tag, so the two tiers refuse the same mismatch;
- **one pooled-row formula** (C4) and **one home for `dead` / `block_pos`** (C5), written down as format fields
  rather than left as whichever array the writer happened to reach for;
- the **turn-boundary key stated as a format property** (C1): the ids, the images and the running state all
  describe position `L`, and the envelope says so.

Step 2 moved no field, no offset and no sizing formula.  `src/platform/kv_nvme.cpp:36-39` is what proves the one
record type that changed *name* (`std::pair<int64_t, uint64_t>` -> `ConversationImageKey`) did not change
layout.

## The answer we would give to their open question

They ask: *"how #52 would consume the snapshot without unbounded promotion staging?"*

1. **Drop the whole-file buffer.** `pread` straight into the pinned host pools (they are device-mapped,
   so KV needs no `cudaMemcpy` at all - our restore already does plain `std::memcpy` into `st.host.*`,
   which is cheaper than their `cudaMemcpyDefault` per array). Sequence: header + geometry walk ->
   segment-size walk against the live engine -> digest pass -> apply pass. Transient staging becomes one
   chunk, not the whole snapshot.
2. **Admit before the read**, using their helper, against the *bounded chunk* rather than the whole file.
3. **TOCTOU**: hold one `open()` fd across validate -> apply, `fstat` before and after (size, mtime,
   inode), and publish via write-temp-then-`rename` so a promoted file is immutable for its lifetime.
   That answers "the file can change before the apply pass" without a whole-image reservation.
4. **Spill = re-keyed dump, not a copy of the RAM image** (C1). `nvme_dump_at` is the spill primitive.

## The five alignment steps (run serially on this branch)

1. Rebase our 6 tier commits from `236d5f2` (0.1.17) onto `b38c183` (0.1.18) - their base.
2. Adopt the shared core boundary: take their `conversation_*` headers and `conversation_state.cpp` /
   `conversation_snapshot.cpp`, retype our checkpoint types, keep our persistence as the disk adapter.
3. Make the turn-boundary spill semantics explicit (C1, C3, C4, C5, C6).
4. Add the missing fixtures (C2, C3, C8): non-block-aligned turn boundary, distinct `idx_dead`,
   `block_pos` at the boundary, one shared state-hash formula.
5. Reconcile the failure contract (C7): prove the clean-reset fallback or adopt their fatal rule.
   **Done in step 5 - and the proof came out the other way: the fallback never ran, so their fatal rule is
   adopted for transfer failures, with the pre-apply refusals and a stale-format store kept recoverable.**

## Where step 2 stands, and what step 3 inherits

Step 2 is `114300f` plus the corrections that finish it:

| commit | what it settled |
|---|---|
| `114300f` | the core is imported; `ImgKey` / `ConvCheckpoint` / `ConvStateSizes` are aliases of the shared types; `checkpoint_save` / `checkpoint_restore` / `conv_state_sizes` are wrappers over it; the turn-boundary hand-off is one `ConversationCheckpoint*` instead of three raw pointers; the RAM policy stays dormant |
| `cbced52` | "Where step 2 leaves NvmeHeader" above - the section `kv_nvme.hpp` already pointed at |
| `6585361` | a turn-boundary snapshot wrote the **live** image list instead of the boundary's |
| `49d0b7d` | `NvmeEntry::imgs` was never filled, so every image comparison in the tier compared against nothing |
| `2b54acd` | the core's diagnostic string was discarded at all four wrappers, and `sizes_of` discarded its `bool` with it |
| `41f8777` | the image-record `static_assert` pinned the record's size, not its layout |
| `740b29d` | the `block_pos` envelope comment named a field shape that does not exist |

**Why the two image bugs were fixed here instead of left to C1.**  They are C1's subject matter, not C1's open
decision.  C1's rule is already written down - *a spill re-keys at the turn boundary; it does not copy the parked
image* - and the image list is part of the key rather than a format choice: the resume match compares the next
request's pictures below `e.L` against what the entry holds, so a snapshot whose pictures describe a longer
prefix than its ids do is unreachable, full stop.  Neither fix moves a field, an offset or a sizing formula, and
`114300f` is what made the first one a one-line change: the boundary arrives as one `ConversationCheckpoint`, and
that checkpoint already carries the filtered list.  Leaving them would have put step 3 on top of a cold tier that
silently does nothing for every conversation containing a picture.

**Open items step 3 owns - do not lose these:**

- **C1's fixture, not C1's rule.**  Nothing in this tree proves turn-boundary keying end to end, and both image
  bugs above survived because of that.  The fixture step 4 must add: a boundary that is **not** `idx_block`-
  aligned, an image whose `start` is **at or past** the boundary (the exact shape that hid both), and a second
  request that re-sends the prompt without the model's reasoning tokens - asserting a **promote**, not merely a
  successful dump.  Until it exists the image path of the tier is untested: `tools/nvme_p0_test.sh`,
  `tools/nvme_steps123_test.sh` and `tools/needle_bench.py` are all text-only, and all three need the GPU this
  branch may not take.
  **Added in step 4** (`src/platform/kv_nvme_host_test.cpp`): all three shapes, the promote asserted through
  `kv_nvme_match` (the serve loop's own rule), and the negative control that a consumed-state snapshot is not
  promotable by that request.  What is still unproven is the end-to-end part only a GPU run can show - that the
  promoted session continues from those bytes with the model's real weights.
- **C1's remaining policy**: whether `nvme_dump_at` is also the spill primitive for a *RAM-tier* eviction (their
  `ConversationCache` evicts on a byte budget; our dump is driven by DONE), and what spilling a parked rather
  than a live session costs.
- **C4, C5, C6** - one pooled-row formula, whose `dead` / `block_pos` the envelope owns, and where the drafter
  ring is restored.  These are the format decisions the `NvmeHeader` section is the input to.
  **All three are settled in step 3** (C4: their `L / idx_block + 1`; C5: the boundary checkpoint's copies;
  C6: the disk adapter, with a collapse condition), and C3's re-publish rule with them.  They are written into
  their sections above and into format version 3.
- **C7, C8, C9, C10, C11** are untouched by step 2.
- **The RAM policy stays imported dormant, on purpose.**  `ConversationCache`, `conversation_prefix`,
  `SavedConversation`, `conversation_checkpoint_validate`, `conversation_snapshot_*`, `conversation_kv_*`,
  `conversation_available_memory` and `conversation_memory_admit` are compiled or present and called by nothing
  outside the core - the only non-core mention in the tree is a comment.  Only
  `conversation_state_sizes` and `conversation_checkpoint_{save,restore}` are live on our side, and both are
  called from `generate.cpp` and `kv_nvme.cpp`.  `src/core/conversation_memory.cpp` was deliberately not
  imported, so `conversation_memory.hpp` is a declaration with no definition behind it.  Wiring the rest in is a
  step-3 decision, not a step-2 leftover.

## Where step 4 stands

| commit | what it settled |
|---|---|
| `41180bd` | **C8** - `state_hash_line()` now prints their field set (`dead`, the spare pooled row, `ss.max_cells`, no PLE bytes without a PLE history, the fp16 KV branch reading the fp16 pools), and the pooled-row count is one named function, `strata::kernels::qsa_pooled_rows`, read by the envelope, the fingerprint and (unchanged) the shared core |
| `036237e` | the shell oracles read the header offset and format version from `kNvmeHeaderBytes` / `kNvmeFormatVersion` through `tools/nvme_header_layout.sh`, and refuse a snapshot of a version they do not write.  Also fixed: `nvme_p0_test.sh`'s snapshot-ids step indexed an argument that was never passed - an IndexError on every run, invisible because the script was edited and not run |
| `8db2c35` | their CPU-only fixtures ported verbatim and passing: `conversation_cache_test`, `conversation_memory_test`, `conversation_validation_test` (which is where their `conversation_checked` overflow checks live) and its wrapped `conversation_transfer_test`.  `conversation_memory.cpp` imported for the fixture only |
| `c34d45a` | `src/platform/kv_nvme_host_test.cpp` - the tier's own host fixture - and `kv_nvme_match`, the serve loop's promote rule moved into the tier's header so the fixture asserts the code the server runs |

**The fixture step 3 owed exists now.**  A boundary that is not `idx_block`-aligned (10 tokens, `idx_block` 4), a
picture at token 14 - inside the consumed prefix, at or past the boundary - a `dead` value no pooled row has,
`block_pos` recorded at the boundary against a live array naming 24, and a second request that re-sends the prompt
without the model's reasoning tokens.  It asserts a **promote** (`kv_nvme_match` returns the boundary entry and the
snapshot restores), and it asserts the negative: the same session dumped without a boundary is keyed on the
consumed 26 ids and that same request can never reach it.  The spare-row check is there too - after a restore,
`pooled[L / idx_block] == dead`, the completed rows are untouched, and row `L / idx_block + 1` was never written.

**How it runs with no GPU.**  The fixture links `strata_engine` - so it drives the real `nvme_dump_at` /
`nvme_restore` / `KvNvmeStore` - and never initialises CUDA: GNU link wrapping turns `cudaMemcpy` and
`cudaMemcpyAsync` into `memcpy`, `cudaGetLastError` into "no error" (which is what keeps `kv_stream.cu`'s `check()`
from exiting on the no-device launch failure) and `cudaDeviceSynchronize` into a no-op; `main` sets
`CUDA_VISIBLE_DEVICES=-1` before any CUDA call.  The same wrapping is what their `conversation_transfer_test` uses.
It is a fixture over **the bytes and the decisions**, not over the device: it cannot show that the arrays are
device memory, that `kv_stream_reset` refills slots, or that a promoted session generates the same tokens.

**What ran here** (no model, no GPU; the engine's 23.9 GiB context untouched throughout):

| check | result |
|---|---|
| `ctest -R 'conversation_\|kv_nvme_host_test'` | 5/5 pass: `kv_nvme_host_test` 113 checks (9 refusals), `conversation_cache_test` 35, `conversation_memory_test` 23, `conversation_validation_test` 780, `conversation_transfer_test` 1020 |
| `bash /tmp/converge-build.sh` | BUILD_OK after every commit |
| `python -m unittest discover -s serve` | 23 tests OK |
| `bash -n` on the three shell files | clean; `tools/nvme_header_layout.sh` exercised against synthetic v3 and v2 snapshots (v3 read, v2 refused, a missing header refused) |
| four mutations of the tier | each caught: dropping the spare-row re-publish, dropping the image filter, widening `qsa_pooled_rows` to `+2`, keying the dump on the consumed ids |

**What was skipped, and why.**  `tools/nvme_p0_test.sh`, `tools/nvme_steps123_test.sh` and `tools/needle_bench.py`
need the model and the GPU; they were edited, not run.  The ctest parity suite is not built in this configuration
(`STRATA_BUILD_TESTS=OFF`, and `tests/` is absent from the published source), which is why the new fixtures got
their own switch, `STRATA_BUILD_CONVERSATION_TESTS`.  Their `conversation_snapshot_test` (it calls `cudaMalloc`)
and their HTTP / isolation / soak tools (they drive a running server) were not ported.

**Unverified until a GPU is free.**  That the v3 envelope round-trips against a real engine - the P0
restore-exactness gate now compares two lines that include `dead` and one more pooled row, and neither has been
printed since step 3; that `tools/nvme_p0_test.sh` and `tools/nvme_steps123_test.sh` still pass with the offset and
version they now derive from the header (the scripts' recorded results are the v2-format record); that a promoted
session continues bit-exactly from a boundary snapshot rather than merely restoring; what `kv_stream_reset` does to
the streamed layers' slots after a restore; and whether the existing v2 snapshots in `/local/strata/kvstore` are
re-dumped acceptably by a v3-writing binary.

## Where step 5 stands

| commit | what it settled |
|---|---|
| `51620f1` | **C7 part 1** - `nvme_restore` / `KvNvmeStore::restore` return the shared core's `ConversationRestore` instead of a `bool`; the apply pass now opens with a `cudaDeviceSynchronize()` (as `conversation_snapshot.cpp` does); every CUDA failure path consumes the error it handled and names the segment that failed; `kv_nvme_host_test` 113 -> 247 checks, with the CUDA last-error state modelled and copy / sync failures injected at a chosen call number |
| `ddfa219` | **C7 part 2** - the serve loop branches on the class: `transfer_failed` is `ERR …` + exit 1 (no drop, no re-prefill), `invalid` keeps drop + `resume = 0` + full re-read; the startup `--nvme-restore` message names which class it was; the stale-store scan message states both operator actions |
| (this commit) | the contract written into `docs/nvme-kv-cache-design.md` §5 (new section; Test record / limitations / prior art renumbered to §6 / §7 / §8) and into the C7 section above |

**The result of this step is a negative one, and it is stated that way on purpose**: the clean-reset fallback the
design document had been describing since §3 has never run.  Every failing restore aborted inside the fallback's
own first step - `session_zero` -> `qsa_state_zero` -> `kv_stream_reset` (`layer.cpp:676`) -> `check("reset")`
(`kv_stream.cu:199-202`) -> `std::exit(1)` - with the restore's error printed under the label "reset".  Nobody
noticed because no test had ever injected a failing copy: every recorded corruption test corrupts a **file**, and
a file refusal never reaches the transfer pass.

**What the fixture now proves rather than argues** (`src/platform/kv_nvme_host_test.cpp`, 249 checks: 18 refusals,
6 injected transfer failures):

- 9 pre-apply refusals - `invalid`, **zero `cudaMemcpy` calls and zero CUDA calls of any kind**, every session
  buffer still at the poison sentinel, no ids/imgs handed to the caller, no CUDA error left pending;
- 6 transfer failures - `transfer_failed`, no error left pending, and the session shown to be half-applied: fail
  the 2nd device copy and the GDN state is the snapshot's while the PLE history is not; fail the 5th and layer 0
  landed while layer 1 did not; fail the 12th (the last spare-row re-publish) and every segment landed while
  `pooled[L / idx_block]` still holds the snapshot's stale value and `dead` holds the value it must become;
- the two syncs - a failure of the one that opens the apply pass writes nothing, a failure of the one that closes
  it leaves every write unconfirmed;
- the store reports the same classes, and its own stale-index case is asserted to be the recoverable one with the
  evidence spelled out (12 copies, 2 syncs, the new snapshot's last token in the PLE window);
- a failed dump leaves no entry, no partial file, and the session byte-identical - the dump side needs no failure
  contract because it only reads;
- a store of version-2 files: `open()` succeeds, 0 entries, 0 bytes against the cap, the operator is told the
  version found and both options, the files stay on disk, and the request that promotes from those same files at
  version 3 gets nothing.

**Mutation-tested** (each built, run, reverted; tree clean): dropping `consume_cuda_error()` -> "left a CUDA error
pending (invalid argument) for the next caller to misread"; dropping the pre-apply sync -> the pre-apply case
reports the post-apply message; classifying a transfer failure as `invalid` (today's behaviour) -> "the first copy
failing is a transfer failure".

**What ran here** (no model, no GPU; the engine's 23.9 GiB context untouched throughout):

| check | result |
|---|---|
| `bash /tmp/converge-build.sh` | BUILD_OK after every commit |
| `ctest -R 'conversation\|kv_nvme'` | 5/5 pass: `kv_nvme_host_test` 249 checks, `conversation_cache_test` 35, `conversation_memory_test` 23, `conversation_validation_test` 780, `conversation_transfer_test` 1020 |
| `python -m unittest discover -s serve` | 23 tests OK |
| three mutations of the tier | each caught, listed above |

**Still unverified, and now sharper than before.**  The GPU oracles (`tools/nvme_p0_test.sh`,
`tools/nvme_steps123_test.sh`) have never been run against the v3 header or the new hash field set - and the P0
corruption negative control now has a class to check: a corrupt file must report `nvme promote refused`, not
`FAILED (transfer)`.  What `kv_stream_reset` does to a real page table after a restore is still unobserved.  And
the one question this step could not answer at all - what a real failed host-to-device `cudaMemcpy` does to the
CUDA context - is the precondition for ever adding the clean reset §5.4 describes.

## Hard constraints for this branch

- **No GitHub writes.** No comments, no PRs, no pushes to `origin`. Push only to `maedoc` if asked.
- **The GPU is busy.** A live `--serve` engine holds ~24 GB on the RTX 4090. Do not run GPU tests
  (`tools/nvme_p0_test.sh`, `tools/nvme_steps123_test.sh`, `tools/needle_bench.py`) or load a model.
  CPU-only validation: `ctest` parity selftests, component fixtures, `python -m unittest`.
- Do not touch `/local/strata/kvstore` (113 GB of live snapshots) or the running server.
