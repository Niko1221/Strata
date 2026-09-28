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

### C8 - two copies of the state hash

We refactored `STATE_HASH` into `state_hash_line()` so `STRATA_NVME_HASH` can print it immediately
after a restore (that is our bit-exactness oracle). They patch the inline version in `main()` and add
`dead=%016llx`, change the pooled row count, and switch `ss.qsa_states[0].max_cells` -> `ss.max_cells`.
`tools/nvme_p0_test.sh` compares the dumper's DONE hash with the post-restore hash, so both tiers must
share one formula or their evidence and ours stop being comparable.

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
- **C1's remaining policy**: whether `nvme_dump_at` is also the spill primitive for a *RAM-tier* eviction (their
  `ConversationCache` evicts on a byte budget; our dump is driven by DONE), and what spilling a parked rather
  than a live session costs.
- **C4, C5, C6** - one pooled-row formula, whose `dead` / `block_pos` the envelope owns, and where the drafter
  ring is restored.  These are the format decisions the `NvmeHeader` section is the input to.
- **C7, C8, C9, C10, C11** are untouched by step 2.
- **The RAM policy stays imported dormant, on purpose.**  `ConversationCache`, `conversation_prefix`,
  `SavedConversation`, `conversation_checkpoint_validate`, `conversation_snapshot_*`, `conversation_kv_*`,
  `conversation_available_memory` and `conversation_memory_admit` are compiled or present and called by nothing
  outside the core - the only non-core mention in the tree is a comment.  Only
  `conversation_state_sizes` and `conversation_checkpoint_{save,restore}` are live on our side, and both are
  called from `generate.cpp` and `kv_nvme.cpp`.  `src/core/conversation_memory.cpp` was deliberately not
  imported, so `conversation_memory.hpp` is a declaration with no definition behind it.  Wiring the rest in is a
  step-3 decision, not a step-2 leftover.

## Hard constraints for this branch

- **No GitHub writes.** No comments, no PRs, no pushes to `origin`. Push only to `maedoc` if asked.
- **The GPU is busy.** A live `--serve` engine holds ~24 GB on the RTX 4090. Do not run GPU tests
  (`tools/nvme_p0_test.sh`, `tools/nvme_steps123_test.sh`, `tools/needle_bench.py`) or load a model.
  CPU-only validation: `ctest` parity selftests, component fixtures, `python -m unittest`.
- Do not touch `/local/strata/kvstore` (113 GB of live snapshots) or the running server.
