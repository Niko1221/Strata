# NVMe-backed KV cache for warm sessions — design (draft v1)

**Goal:** keep many conversations warm so a returning session has low TTFT, by adding a
cold **NVMe tier** below strata's existing pinned-host KV tier. Saving/restoring is
**automatic** (server-driven), not client-initiated.

**Scope v1:** persist the **full** attention KV (12 QSA layers) **and** the running state
(36 GDN recurrences + conv history, QSA indexer tails, PLE history) per session. Optimize
for sparsity (only the blocks attention actually reads) in a later phase.

---

## 0. AGREED v1 (2026-09-28, after the glm-5.3 critique — see `docs/nvme-kv-cache-critique.md`)

The critique verified the design against the code and corrected it. These decisions are
locked and supersede the conflicting parts of §4–§11 below:

1. **Host = live session only; NVMe = the persistence tier.** No multi-session host LRU (the
   pinned `KvHostPools` is one arena for the live session, baked into the captured kernels).
   Promote = NVMe → the existing arena in place. (Demotes old §4/§9 host-LRU idea.)
2. **Synchronous dump** at `DONE` (~0.1–0.6 s at 3–5 GB/s, trivial vs ~110 s re-prefill).
   No background write thread in v1 (it would race the next request's prefill on the same
   arena). Async only later with a private staging buffer, if measured to matter.
3. **Force streamed mode under `--kv-nvme`** (a `qsa_set_kv_resident` floor) so the host copy
   always exists; closes the `kv_mode 0` gap.
4. **Persist the MTP drafter KV** on restore (one layer, cheap, keeps decode speed).
5. **Keying = exact token prefix** (store `ids` verbatim, match with the existing
   `starts_with`), NOT content-addressed block hashing. Refuse (never convert) a restore after
   a `--kv` format or context-size change (entry carries a format+geometry tag).
6. **`fsync` on dump**; the store holds conversation content in plaintext on disk (accepted).

**Complete persisted set (corrected — the old §5.2/§8 set was incomplete):**
per QSA layer: KV host copy cells `[0, L)` (all formats) + `idx_pooled` + `idx_block_pos` +
`idx_dead` + `idx_tail`; plus `gdn`, `ple`, the **MTP drafter host KV copy**, and the header
(`ids`, `imgs`, cvec flag, kv format, geometry tag, L). Missing `idx_pooled`/`idx_dead`/
`idx_block_pos`/drafter KV yields *fluent-but-wrong* tokens, not a crash — the silent class.

**Smallest correct v1:** one whole-session snapshot file per session, dumped synchronously
from the pinned host arena on `DONE`, promoted by `pread` into that arena + `kv_stream_reset`
+ `checkpoint_restore`, gated on format/geometry tags. No new CUDA kernels — plain file I/O
on mapped pinned memory.

**Falsifiable P0 oracle (already in the code):** `STRATA_STATE_HASH`
(`generate.cpp:3040`) hashes gdn/ple/tail/pooled/kv/mtp. P0 = bit-exact STATE_HASH equality +
identical greedy continuation between a never-evicted run and an NVMe-restored run, pinned
with `--adapt-swaps 0`, plus a deliberate-corruption negative control.

---

## 1. Requirements (from the owner)

1. **Automatic** save/restore — the client must not call save/restore (this was ninfer's pain).
2. Persist **full attention KV + running state** — whatever it takes to keep warm-session TTFT low.
3. **Full session first**; sparsity later.
4. Keying/coherence: follow what production servers (vLLM) already solved.
5. Design now, implement second.

---

## 2. Prior art (what production already proved)

- **vLLM Automatic Prefix Caching (APC / RadixCache).** KV is fixed-size blocks,
  **content-addressed** by hash of `(prefix_hash, block_tokens)` in a **radix tree keyed on
  the token prefix**. A new request walks the tree, reuses the longest cached prefix, LRU
  evicts free blocks, ref-counts protect in-use ones. Fully automatic.
  - Hybrid models (full-attention + sliding-window/mamba): per-group cache-hit rules,
    intersect the hits. **Directly relevant to strata's QSA + GDN hybrid.**
- **vLLM Tiered KV Offloading** (blog 2026-09-10; RFC #38260; PR #40020).
  - **All KV flows through host DRAM (primary tier); NVMe/object/remote are secondary tiers.**
  - Save: GPU→host async DMA (GPU freed immediately), then **cascade** host→all secondary tiers.
  - Load: scheduler checks host first; on miss queries secondary tiers in order; **promote**
    chunk to host async (scheduler gets `RETRY` while promotion is in flight).
  - **Canonical layout:** each page = one block of one layer, all heads contiguous → offset
    addressing, shareable across configs.
  - Host tier is a real **LRU/ARC** cache, not a staging buffer. `ref_cnt` guards async transfers.
  - Filesystem secondary tier: `root_dir`, `n_read_threads`/`n_write_threads` (~16 each).
  - Capacity rule: <64 conversations = HBM fine; 64–128 = CPU offload; **>128 = storage offload**.
- **llama.cpp.** `--slot-save-path` + `/slots/{id}/save|restore` is **client-initiated** (the
  pain we are avoiding). Its automatic form is `--clear-idle` ("save and clear idle slots on
  new task") + `--cache-reuse N` + `--kv-unified`.
  - ⚠️ **#26676 / #25913:** slot restore is a **no-op for hybrid/recurrent models** — restored
    KV cells are unusable at their positions because the recurrent state was never persisted.
    **Strata is hybrid (QSA + GDN): we MUST persist the recurrent state alongside KV, and keep
    position consistency.** Strata already separates positional KV (rewindable) from running
    state (checkpointed) — the right foundation.

---

## 3. What strata already has (build on this, don't reinvent)

- **`KvHostPools`** (`kv_stream.hpp`): the authoritative K/V already lives in **pinned,
  device-mapped host memory** in streamed mode. This *is* vLLM's host primary tier — already built.
- **`kv_stream_resolve`** (`kv_stream.cu`): clock/second-chance eviction over VRAM slots,
  PCIe zero-copy refill from the host copy. This *is* the GPU↔host tier boundary — already built.
- **`ConvCheckpoint`** (`generate.cpp:501`): already saves/restores the **running state**
  (GDN, PLE, indexer tails) D2H/H2D, ~118 MB/session. Already built.
- **Prefix-match resume** (`generate.cpp:2557`): the serve loop already finds the longest
  checkpoint whose token prefix matches the incoming prompt (`starts_with`) and prints
  `RESUME n` / `REUSED n`. Already built.

**The only missing tier is NVMe below the host copy, and the only missing behavior is
automatic cascade-on-idle / promote-on-hit.** Everything else is a wiring job.

---

## 4. Target architecture (3 tiers)

```
VRAM slots (hot)  <->  pinned host arena (warm, primary)  <->  NVMe (cold, secondary)
   kv_stream            KvHostPools (exists)                  KvNvmeStore (NEW)
   clock eviction       LRU/ARC over sessions                 content-addressed blocks
```

Principle copied from vLLM: **NVMe never talks to the GPU directly.** Save = host→NVMe;
load = NVMe→host→(slots on demand). The host copy stays the single source of truth.

---

## 5. Data model

### 5.1 Content-addressed blocks (the keying answer, from vLLM)
- Unit = a **block** = one page = `page_size` (4) cells × all kv-heads × K+V, per layer —
  already strata's natural granule (`kv_block_bytes`).
- A session's identity = the **token prefix** it consumed. We key the session by a rolling
  hash of its token ids (and image keys, as `ConvCheckpoint` already tracks `ImgKey`).
- A block's on-disk key = `hash(prefix_tokens_of_block)` so **shared prefixes across
  sessions deduplicate automatically** (two conversations with the same system prompt share
  those blocks). This is the radix-tree idea, flattened to a hash index for v1.

### 5.2 Session index (NVMe)
```
kvstore/
  index.bin        # session_id -> {prefix_hash, n_cells, block_keys[], running_state_key, mtime, lru}
  blocks/<hash>.bin# one block, canonical layout [layer][block][kv_head][page_size][head_dim]
  state/<sid>.bin  # the ConvCheckpoint running-state blob (GDN/PLE/tails)
```
Canonical layout matches vLLM's "one block of one layer, heads contiguous" so a restore is a
single contiguous DMA per block, not a scatter.

### 5.3 What a "warm session" costs on NVMe
- 128K int8 KV ≈ **1.66 GiB/session**; q4_0 ≈ **0.9 GiB**.
- 1.7 TB free NVMe → **~1000 sessions at 128K int8** (more with q4_0). Running state ~118 MB each.

---

## 6. Automatic save (no client calls)

Trigger points, all server-side in the `--serve` loop:
1. **On `DONE`** (a request finishes): the live session's KV blocks + running state are the
   just-computed truth. **Cascade** them to NVMe keyed by the session's prefix hash.
   - Reuse the existing `checkpoint_save` for running state; add `kv_dump_to_nvme` for the
     KV blocks (read from `KvHostPools`, which is already the authoritative copy).
2. **On session switch / eviction** (the `--clear-idle` pattern): when the live KV pool is
   about to be overwritten by a different conversation, spill the outgoing session's blocks
   to NVMe first.
3. **Async, off the token path:** writes run on a background I/O thread (like vLLM's
   `n_write_threads`), so decode/TTFT is not blocked. GPU/host freed immediately; NVMe lags.

Idempotency: a block already on NVMe with the same prefix hash is skipped (dedup).

## 7. Automatic restore (no client calls)

On a new request, the serve loop already computes the longest matching prefix. Extend it:
1. **Host hit** (session still in the pinned arena): current behavior, instant.
2. **NVMe hit** (session evicted from RAM but on disk): **promote** its blocks
   NVMe→host (`n_read_threads`), restore the running state via `checkpoint_restore`,
   re-point the `page_table`/residency map, set `resume = matched_cells`. Continue from
   `resume` — **no re-prefill**.
3. **Miss:** current behavior (re-prefill the fresh tail).

Promotion is async; the scheduler proceeds once the matched prefix is resident (vLLM's
`RETRY` model). For v1 we can block on promotion of just the matched prefix (it's ~0.4–0.6 s
for 128K at 3–5 GB/s vs ~110 s re-prefill — already a ~200× TTFT win).

---

## 8. Hybrid correctness (the llama.cpp #26676 trap)

Strata = QSA (full attention, positional, rewindable) + GDN (recurrent, non-rewindable).
- **Positional KV:** valid to restore as long as the restored cells' positions match the
  session's token prefix. Guard: a restored block is usable only if `prefix_hash` matches AND
  the cell positions are `[0, matched_cells)`. (Strata's `qsa_layer` already asserts position
  is inside the RoPE table — extend that guard to the restore path.)
- **Running state (GDN/PLE/tails):** MUST be restored from `state/<sid>.bin`, not recomputed —
  this is exactly what llama.cpp forgot and why its hybrid restore is a no-op. Strata's
  `ConvCheckpoint` already does this correctly; keep it mandatory on every restore.
- **Coherence rule (already in strata):** a checkpoint is valid only while the positional cells
  below it hold its tokens. Keep that invariant when promoting from NVMe.

---

## 9. Eviction & policy

- Host tier: LRU/ARC over sessions (reuse the `kv_stream` clock machinery one level up).
- NVMe tier: LRU by `mtime`; capacity-capped; evict oldest cold sessions first.
- `ref_cnt` on blocks during async promote so a session in flight isn't evicted (vLLM pattern).
- Dedup shared-prefix blocks so N conversations with a common system prompt cost ~1 copy.

---

## 10. Where it lives in the code

- **New:** `src/platform/kv_nvme.{cpp,hpp}` + a small CUDA/host copy helper — the
  `KvNvmeStore` (store/load/promote/evict), next to `direct_file.cpp`.
- **Extend:** `include/strata/kernels/kv_stream.hpp` — add `kv_dump_to_nvme` /
  `kv_promote_from_nvme` beside the existing host-copy functions.
- **Extend:** `src/program/generate.cpp` serve loop — call cascade on `DONE`/switch, and on a
  prefix match prefer NVMe promote over re-prefill. Reuse `ConvCheckpoint` for running state.
- **Config:** `--kv-nvme DIR` (enable + path), `--kv-nvme-max GB` (capacity),
  `--kv-nvme-threads N`. Off by default (parity preserved).

---

## 11. Phased plan (revised per critique)

- **Step 0 (P0 spike, falsifiable):** manual `--nvme-dump`/`--nvme-restore` of ONE session.
  New `src/platform/kv_nvme.{cpp,hpp}`. Assert `STRATA_STATE_HASH` bit-equality + identical
  greedy continuation vs a never-evicted run (`--adapt-swaps 0`), plus a negative control
  (skip one memcpy → hash must differ). **Gate: bit-exact or stop.**
- **Step 1:** automatic **synchronous** cascade on `DONE` behind `--kv-nvme DIR` (idempotent:
  skip if an entry with the same `ids` exists). Gate: TTFT regression ≤ a few hundred ms.
- **Step 2:** automatic **promote on prefix match** in the resume selection block
  (`generate.cpp:2725`): longest NVMe hit → `nvme_restore` → `kv_stream_reset` →
  `mtp.kv_restore(resume)` → existing fresh-tail path. Test: A → B → A, assert `RESUME` +
  identical continuation. Gate: return-to-A TTFT ≤ ~1 s for 128K vs ~110 s re-prefill.
- **Step 3:** capacity `--kv-nvme-max` + LRU by mtime; *then* consider async with a private
  staging buffer only if measured. Gate: no torn writes in a 10k-iteration branch stress.
- **Step 4 (default skip):** radix/prefix dedup + sparsity — only if profiling proves capacity
  or promote time is the bottleneck (unlikely for a one-live-session server).

### Status (2026-09-28)
- **Step 0: DONE and PASSING** (`tools/nvme_p0_test.sh`): restore-exactness (post-restore
  `STRATA_STATE_HASH` == the dumper's DONE hash), `RESUME` engages, greedy continuation
  identical to both the same-process live continuation and a full re-prefill, negative control
  detects corruption. Note: batched-prefill vs decode parity was confirmed bit-exact (A' == C),
  and the DONE-hash gate for resumed runs is informational only (the consumed-window `L`
  semantics differ between resumed and fresh paths; token equality is the meaningful gate).
- **Housekeeping (not steps):** deploy `build/strata` -> `engine/strata` + restart; compare the
  DONE-hash gate at matched accounting (cosmetic); commit the Step 0 work (kv_nvme module,
  `state_hash_line` refactor, test harness, this doc + the critique).

### Steps 1-3 + review response (2026-09-28, later)

Steps 1-3 implemented and passing (`tools/nvme_steps123_test.sh`: ALL PASS - one growing file per
conversation; promote after restart resumes with 13 fresh tokens read in ~350 ms; LRU cap evicts
to 0.83 GiB under a 1 GB cap). The glm-5.3 implementation review (`docs/nvme-kv-impl-review.md`)
was addressed:

- **P1-1** p0 oracle offsets fixed for the v2 header (96 -> 104) and the oracle RE-RUN: restore
  exactness, resume and the negative control all pass on the current binary.
- **P1-2** a failed promote now forces `resume = 0; from_live = false` (clean session_zero path):
  no half-applied state can ever be decoded from.
- **P1-3** `--kv-nvme` now forces the streamed-KV floor (`qsa_kv_resident_min()`) as agreed.
- **P1-4** corrupt store files can no longer crash the server: L/n_imgs/file-size bounds before any
  allocation, 64-bit file sizes (`ftello`/`_ftelli64`), a 64 GiB sanity cap, try/catch in the scan,
  and skipped files are counted and logged.
- **P2-1** promote picks the LONGEST match. **P2-3** the spike dump syncs the device. **P2-4** imgs
  are memcpy'd (no misaligned cast). **P2-5** a failed dump unlinks its partial file. **P2-6** the
  exact-skip compares imgs. **P2-8** layout mismatches are reported as such, not "truncated".
  **P2-9** Windows `_commit` fsync. **P2-10** flag validation + `--help` entries; both scripts exit
  non-zero on failure.
- **P2-2 DEFERRED with rationale:** dropping any stored strict-prefix entry would destroy the only
  cache a BRANCHED conversation's own continuations can match (a prompt matching the shorter entry
  need not match the longer one). Supersede stays per-process (safe); the cap bounds accumulation.
- **P2-7 documented:** the last entry is never evicted even over cap (the store is never emptied);
  recency refresh is in-memory only (the file mtime is untouched).

**Determinism finding (important):** the engine's decode is run-to-run nondeterministic on this
build - two identical plain runs of the same prompt diverge mid-generation, and the OLD
pre-kv_nvme binary shows the same behavior on the same prompt (same divergence point). The NVMe
restore itself is fully deterministic and bit-exact: the post-restore `STRATA_STATE_HASH` is
identical across repeated restored runs. The P0 token-equality comparison is therefore
informational only; the hard gates are restore-exactness + resume + the negative control. This
confirms the verbatim-bytes rule: recomputing KV on restore would compound the divergence instead
of preserving the session.

## 12. Risks / open questions

- **Determinism:** KV computed via the GPU expert cache "rounds differently from the CPU"
  (log note). A restored KV must be the *same* KV that produced the session — store the KV
  bytes verbatim, never recompute on restore.
- **Position/RoPE consistency** on restore (guard in §8).
- **Write amplification:** cascading every DONE to NVMe is heavy; batch + dedup + only spill
  sessions likely to return (LRU hint).
- **NVMe endurance:** ~1.7 TB, ~1 GB/session writes; monitor write load.
- **Interaction with `--kv-resident` streaming:** the host copy is the source of truth only in
  streamed mode; for fully-resident sessions we must also snapshot the host copy.

---

## 13. Sources

- vLLM Automatic Prefix Caching — docs.vllm.ai (APC example; hybrid KV cache manager design doc).
- vLLM "Tiered KV Cache Offloading" blog (2026-09-10); RFC #38260; PR #40020 (`vllm/v1/kv_offload/tiering/`).
- LMCache offloading docs (CPU/FS/Mooncake/Redis destinations).
- llama.cpp `--slot-save-path` / `--clear-idle` / `--cache-reuse` (tools/server/README.md);
  discussions #13606, #15530, #20572; **bug #26676 / #25913 (hybrid restore no-op)**.
