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

### C5 - where `idx_dead` / `idx_block_pos` live

Ours: per-QSA-layer segments in the KV body of the file. Theirs: per-checkpoint blobs (live + every
checkpoint), reconstructed on restore. The disk envelope must pick one.

### C6 - drafter ring restore

Theirs: inside the core (`kv_ring_restore(...)` for `kv_mode == 2`). Ours: in the serve loop
(`mtp.kv_restore(resume)`) plus host-array dumps recorded by `NvmeHeader::mtp_host`. Pick one home.

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

## Hard constraints for this branch

- **No GitHub writes.** No comments, no PRs, no pushes to `origin`. Push only to `maedoc` if asked.
- **The GPU is busy.** A live `--serve` engine holds ~24 GB on the RTX 4090. Do not run GPU tests
  (`tools/nvme_p0_test.sh`, `tools/nvme_steps123_test.sh`, `tools/needle_bench.py`) or load a model.
  CPU-only validation: `ctest` parity selftests, component fixtures, `python -m unittest`.
- Do not touch `/local/strata/kvstore` (113 GB of live snapshots) or the running server.
