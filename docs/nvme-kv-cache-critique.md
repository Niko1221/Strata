I have completed a full review of the design doc against the actual code seams. Here is the complete artifact.

---

# Critique: `docs/nvme-kv-cache-design.md` (NVMe cold tier for warm sessions)

Reviewer scope: read-only. All claims below were verified against the code, not the doc's own prose.

---

## A. VERDICT

**Sound in direction, over-scoped in keying and tiering.** The core idea — an NVMe tier *below* the pinned-host KV copy, host stays source of truth, automatic save/restore, running state persisted alongside KV — matches what strata already has and what the hybrid architecture requires. But the doc's "everything else is a wiring job" claim (§3) is wrong in three specific places: the snapshot set it defines is **incomplete** (indexer pooled keys and the MTP drafter's KV are missing), the host-tier-as-multi-session-LRU (§4/§9) does not fit the single-tenant pinned arena the code actually has, and the async cascade (§6.3) races the next request on that same arena. vLLM's architecture is multi-tenant block-granular; strata serves ONE session at a time through one fixed arena. A much smaller v1 (whole-blob snapshot keyed by exact token prefix, synchronous dump) gets the full TTFT win and is genuinely testable.

### Top 3 strengths

1. **It builds on real, verified seams.** `KvHostPools` really is the authoritative copy in streamed mode with the identity `[block][kv_head][page_size][head_dim]` layout (`include/strata/kernels/kv_stream.hpp`, `src/core/layer.cpp:611-628`); `ConvCheckpoint`/`checkpoint_save`/`checkpoint_restore` really do persist the GDN/PLE/indexer-tail running state (`src/program/generate.cpp:617-676`); the serve loop really does longest-prefix resume with `starts_with` including image keys (`generate.cpp:2705-2734`). The doc does not invent infrastructure that isn't there.
2. **The hybrid-correctness section (§8) targets the right invariant.** The code's own split — positional KV "rewindable", running state checkpointed, "a checkpoint is only valid while the positional cells below it still hold its tokens" (`generate.cpp:2332` comment block) — is exactly the llama.cpp #26676 failure mode the doc calls out. Restoring GDN state via the existing `checkpoint_restore` and gating KV on prefix+position is the correct rule.
3. **The verbatim-bytes rule (§12) is correct and load-bearing.** KV bytes depend on which expert engine computed each token (GPU cache vs CPU pool round differently, per the doc and the engine's own activation-contract notes). Any design that recomputes or requantizes on restore would silently diverge. Storing bytes verbatim also makes the restore bit-testable via `STRATA_STATE_HASH`.

### Top 5 risks / gaps

1. **P1 — the snapshot set is incomplete.** §5.2/§8 say "full attention KV" + `ConvCheckpoint` running state. That misses: **`idx_pooled`** (pooled indexer keys, one per block — positional, written by the indexer kernel, `src/core/layer.cpp:584`, `src/kernels/cuda/qsa.cu:211`), **`idx_block_pos`** (`layer.cpp:585`, written `qsa.cu:216`), **`idx_dead`** (`layer.cpp:582`, `qsa.cu:187,213` — "CONSTANT for the sequence"), and the **MTP drafter's own host KV copy** (`src/core/mtp.cpp:513-519`; `mtp.kv_restore(resume)` at `generate.cpp:2766` refills the drafter's ring *from its host copy*, which an NVMe-evicted session no longer has). Restoring KV without `idx_pooled` does not crash: the indexer scores garbage/uninitialized pooled rows, the top-k cell selection shifts, and the model produces fluent-but-different tokens — the *silent* failure class this engine's own comments warn about (e.g. the L100 stale-`parts` bug note in `layer.hpp`). Note that `ConvCheckpoint` gets away with omitting these only because the live session's cells are never evicted today.
2. **P1 — the host tier cannot be an LRU/ARC over sessions (§4, §9).** `KvHostPools` is one pinned, device-mapped allocation per QSA layer sized to the *live* session's `max_cells` (`layer.cpp:611-628`), and those host pointers are baked into the captured resolve/copy kernels (`kv_stream.cu` resolve+copy are launched per call but the pointers are struct fields consumed by the graphs' fixed-address contract). There is exactly one host copy. vLLM's "host primary tier, LRU over blocks, `ref_cnt`" assumes many concurrent sequences sharing block-granular host memory. For strata the honest v1 host tier is: *host = live session only; NVMe = everything else; promote = NVMe → the existing arena in place.* P2 as written is a rewrite of the host tier, not a wiring job.
3. **P1 — "async cascade, host freed immediately" (§6.3) is unsafe here.** In vLLM, GPU blocks are freed and the host copy is ref-counted. In strata the host arena is the *next request's* arena. The serve loop is sequential (`DONE` printed, then the next `GEN` line is parsed, `generate.cpp:3055-3063`), but a background write thread reading the pinned host pools would race the next request's prefill, which writes those same host buffers from device kernels (`src/kernels/cuda/qsa.cu:137`, `kv_q8.cu:61`, `kv_q4.cu:113`, and the prefill path `prefill/kernels.cu:356`) — a branch that rewrites cells ≥ its `resume` tears the in-flight dump. v1 must either dump synchronously (0.05–0.6 s for typical 8–128K sessions at 3–5 GB/s — cheap vs 110 s re-prefill) or stage into a private buffer. The design specifies neither.
4. **P2 — content-addressed block hashing (§5.1) is the wrong key for v1 and under-specified.** Strata's existing match is *exact token-prefix equality including `ImgKey`s and gated on the control-vector state* (`starts_with`, `generate.cpp:2705-2712`; `cvec_cached` invalidation `generate.cpp:2717-2726`). A block hash gives you probabilistic identity, needs a hash-collision story, needs a format/geometry/model-weights tag (int8 vs q4_0 vs f16 — `QsaState::kv_int8/kv_q4` — plus `n_head_kv/head_dim/page_size/idx_block`), and its dedup payoff requires *many concurrent sessions sharing prefixes*, which a one-live-session server does not have. The capacity rule quoted from vLLM (">128 conversations = storage") is about concurrent sequences, not strata's sequential sessions.
5. **P2 — kv_mode 0 has no host copy.** Default `--kv-resident 0` means fully-resident VRAM and `KvHostPools` all null (`kv_stream.hpp`, `layer.cpp:605`). §7's "read from `KvHostPools`" only works in streamed mode. The doc defers this to a §12 bullet, but P0/P1 as written silently produce empty dumps for small-context (non-streamed) sessions. Cheapest fix: `--kv-nvme` forces streamed mode (a tiny `qsa_set_kv_resident` floor), or dump via D2H staging.

Also worth noting (report-only): write amplification and NVMe endurance (§12) are real but bounded by "dump only on DONE/evict, skip when the entry's prefix is unchanged"; and the doc never mentions that the NVMe tier buys *process-restart* persistence for free, which is arguably the bigger operational win.

---

## B. CORRECTNESS TRAPS specific to this hybrid engine

**Position/RoPE consistency.** RoPE is applied to K *at write time* (`kv_append_*` stores the rotated key), and queries rotate against the `cos_tab`/`sin_tab` built once from `max_cells` (`layer.hpp`, `QsaState`). So restoring K bytes restores the rotations, and positions are consistent *provided* the restored cells land at exactly `[0, matched_cells)` in an arena whose `max_cells` and RoPE table match. Because v1 keys on exact token-prefix equality, position follows from prefix length; the guard should be an explicit assert (`matched_cells ≤ max_cells`, stored geometry == live geometry) rather than the doc's vaguer "prefix_hash matches AND positions are [0, matched)".

**GDN running-state coupling.** GDN state is a function of *all* tokens, not of any attention window — there is no rewind. `checkpoint_restore` (`generate.cpp:660`) is mandatory on every promote, *and* the PLE token window it fixes (`ss.ple_prev`, lines 670-673) must match the restored prefix length. The inverse coupling matters too: restoring GDN state but not the drafter's KV ring produces *bad drafts* (slower decode) rather than wrong output, since drafts are always verified — but restore `mtp.kv_restore(resume)` still needs the drafter's host copy present to do anything (§A gap 1).

**The rewindable/non-rewindable invariant.** The engine's rule: a checkpoint is valid only while the positional cells below it hold *its* tokens. On a branch (same prefix, new tail), the serve loop already enforces this in memory by dropping every checkpoint that is not a prefix of the new prompt (`generate.cpp:2735-2739`). The NVMe tier must inherit the same semantics: **what must be invalidated on a branch is every *running-state* entry whose token prefix is not a prefix of the new conversation** (the KV blocks below the branch point stay valid — they are immutable cells — but with whole-prefix keying they are simply orphaned and LRU-evicted later, which is correct and needs no invalidation logic).

**Determinism.** KV bytes depend on expert residency at compute time. Two consequences: (a) restore must be verbatim — never requantize, never recompute (§12 is right); (b) the P0 "identical continuation" test is only *falsifiable* with the VRAM expert set pinned (`--adapt-swaps 0`), exactly as the existing `STRATA_CKPT_REREAD` check requires (`generate.cpp:2752-2759` comment). Without that pin, a mismatch proves nothing and a match is luck.

**The ready-made oracle.** `STRATA_STATE_HASH` (`generate.cpp:3040-3115`) already hashes `gdn`, `ple`, `idx_tail`, `idx_pooled`, the per-cell KV of all 12 layers, and the MTP drafter's KV over `[0, L)`. This is a strictly stronger restore check than greedy-token equality, and it is already written. P0 should assert hash equality, not just token equality. (It does not hash `idx_dead`/`idx_block_pos` — add them or dump them under the same env flag; see E.)

---

## C. KEYING / COHERENCE OPTIONS

| | (i) whole-blob keyed by conversation | (ii) content-addressed block hash (radix) | (iii) llama.cpp slot save |
|---|---|---|---|
| Identity | exact token prefix (+ ImgKeys, cvec flag, format tag) | `hash(prefix, block_tokens)` | client-supplied slot id |
| Client calls | none (server derives key from the prompt it already reads) | none | **required — rejected by requirement 1** |
| Restore I/O | one sequential read per session | block-granular random reads | one read |
| Branch semantics | orphan + LRU (no logic) | ref-counts, partial reuse | manual |
| Matches existing seam | yes — `starts_with`/`ConvCheckpoint` exactly | no — new index, new collision story | no |
| Dedup | none | shared prefixes dedup | none |
| Implementation size | small | large (index, hash, eviction of shared blocks, ref counts) | medium |

**Recommendation for v1: (i), keyed by the exact token prefix.** Store the `ids` vector itself in the entry (like `ConvCheckpoint` does — compare with the existing `starts_with`, don't trust a filename hash), plus `imgs`, the cvec flag, the kv format, and a geometry tag. Justification: it is the ConvCheckpoint model extended with the positional state, so the resume seam (`generate.cpp:2725-2766`) is reused nearly verbatim; sequential whole-file I/O is the optimal NVMe pattern (no per-block random access, no `blocks/<hash>.bin` directory of millions of files); and it is trivially testable. The dedup sacrifice is bounded: at ~1.6 GiB per 128K session and 1.7 TB free, ~1000 whole sessions fit even with zero dedup; shared system prompts costing N copies is acceptable at this scale. Do P3 (radix/hash dedup) only if profiling after P2 shows NVMe capacity or promote time is actually the bottleneck — for a one-live-session server it likely never is. Option (iii) is correctly rejected by the doc (client-initiated).

---

## D. ALTERNATIVES the doc didn't consider

1. **The pinned host arena *is* the staging buffer — no new CUDA plumbing.** Dump = `pwrite`/`fwrite` directly from `st.host.k_q/v_q/k_scale/v_scale` (or `k_q4/v_q4`, or `k_pool/v_pool`); promote = `pread` into the same fixed pinned addresses. The host layout is already the "canonical layout" §5.2 wants (`kv_stream.hpp` identity layout). The doc's `kv_dump_to_nvme`/`kv_promote_from_nvme` beside the host-copy functions in `kv_stream.hpp` (§10) implies kernel work where plain host-side file I/O on mapped pinned memory suffices — the only device call a promote needs is the existing **`kv_stream_reset`** (`kv_stream.cu` reset_kernel), which makes every block non-resident so `kv_stream_resolve` refills slots from the now-restored host copy on demand. §7's "re-point the page_table/residency map" overstates the work: a reset is the whole job.
2. **`kv_ring_restore` / `kv_stage_from_host` need not be touched.** `kv_ring_restore` (`kv_stream.cu:223`) already handles the drafter's ring refill once its host copy is repopulated; `kv_stage_from_host` is the prefill stage's path. The NVMe tier should be invisible to both.
3. **A simpler v1: one snapshot file per session (per-layer segments), not a block store.** `kvstore/<prefix-hash>/snapshot.bin` = header (ids, imgs, cvec, format, geometry, L) + per-QSA-layer segment (K/V runs + `idx_pooled` + `idx_block_pos` + `idx_dead` + `idx_tail`) + `gdn` + `ple` + drafter host-copy segment. One sequential write, one sequential read, no index file. This collapses §5.2's `index.bin`/`blocks/`/`state/` triple into one artifact.
4. **Mirror the existing checkpoint cadence instead of inventing one.** The serve loop already takes checkpoints at the turn boundary (`turn_at`, `generate.cpp:2886-2888`) and every `prompt_cache_every` tokens mid-prompt (`sp.on_chunk` → `checkpoint_at`, `generate.cpp:2371-2375`). The NVMe cascade can dump exactly the *final consumed state* (the equivalent of `live`, set at `generate.cpp:3050`) — that is the resume point the next identical conversation will match.
5. **Force streamed mode under `--kv-nvme`.** Eliminates the kv_mode-0 gap (§A gap 5) for the cost of a small resident-cells floor via the existing `qsa_set_kv_resident`.

---

## E. INCREMENTAL, TESTABLE IMPLEMENTATION PLAN

Guiding rule: **P0 must be able to fail.** Every step below has a bit-level oracle (`STRATA_STATE_HASH`) and a behavioral oracle (greedy continuation), plus a stop gate.

### Step 0 — P0 spike: manual dump/restore of ONE session (falsifiable)

- **Files:** new `src/platform/kv_nvme.cpp` + `include/strata/platform/kv_nvme.hpp` (beside `direct_file.cpp`); `src/program/generate.cpp` for two hidden flags. No changes to `layer.hpp`/`session.hpp`/`kv_stream.*` — `QsaState` and `SessionState` already expose everything (`st.host`, `idx_pooled`, `idx_block_pos`, `idx_dead`, `idx_tail`, `ss.gdn_state`, `ss.ple_hist`).
- **Functions:** `bool nvme_dump(const char* path, const SessionState& ss, const MtpDrafter& mtp, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, bool cvec, const ModelGeometry& g, int fmt)` — called after `cudaDeviceSynchronize()` (the `checkpoint_save` contract, `generate.cpp:644`); `bool nvme_restore(...)` — `pread` each segment into the pinned host pools / H2D for the device-side indexer arrays, `kv_stream_reset` per layer, then the existing `checkpoint_restore`-equivalent for gdn/ple/tails (reuse it by materializing a `ConvCheckpoint`).
- **Wiring:** `--nvme-dump PATH` in the DONE block (~`generate.cpp:3055`); `--nvme-restore PATH` before request processing (next to `session_zero` at `generate.cpp:2741`), setting `live`/`live_imgs`/`live_ok` so the existing `starts_with` fast path takes over.
- **The not-a-no-op test (the whole point):**
  1. Process A: `--serve`, prompt P, `GEN 200` → record all `T` tokens; run with `STRATA_STATE_HASH=1` and `--adapt-swaps 0`; dump at DONE.
  2. Process B: fresh process, `--nvme-restore dump`, same prompt P, `GEN 200` with the same pinning.
  3. **Assert:** the `STATE_HASH L=... gdn= ple= tail= pooled= kv= mtp=` line is *identical* between the two processes at the same L (bit-exact restore — catches a missing `idx_pooled`, a torn read, a format mismatch), and all 200 greedy tokens are identical (continuation — catches the llama.cpp no-op class, where state hash equality could still hold at restore time but the session diverges after).
  4. Add `idx_dead`/`idx_block_pos` to the hash line (small edit at `generate.cpp:3085`) so the spike cannot pass by luck on those.
  - Negative control to prove the test can fail: run B *without* restoring gdn state (comment out one memcpy) and confirm the hash line differs. If it doesn't, the test is decorative.
- **Gate:** both asserts pass on streamed (kv_mode 1, int8 and q4_0) sessions ≥ 32K; hash equality demonstrated; only then proceed. If bit-exactness cannot be achieved, stop — the whole design's verbatim-bytes premise is broken.

### Step 1 — Automatic cascade on DONE (single session, synchronous)

- **Files:** `generate.cpp` DONE block; `kv_nvme.cpp` gains a tiny directory store (list of snapshot files + their `ids`/`imgs`/cvec held in host memory).
- **Behavior:** behind `--kv-nvme DIR`; after `DONE`, `cudaDeviceSynchronize` + dump the consumed state; skip if an entry with the same `ids` already exists (idempotency §6). **Synchronous in v1** — see §A gap 3.
- **Test:** serve a conversation, then serve a *branching* conversation in the same process; assert the store holds both entries and neither restore path regresses; assert dump time ≪ re-prefill time (log it).
- **Gate:** TTFT regression ≤ a few hundred ms per DONE; no correctness change with the store enabled vs disabled (same greedy tokens for a fixed prompt).

### Step 2 — Automatic promote on prefix match

- **Files:** `generate.cpp` resume selection block (`2725-2766`): after the `checks` scan, scan the NVMe entries with the same `starts_with`; on the longest hit: `nvme_restore` → set `resume`/`from_live = true` (treat as live, since the host copy is now populated) → `mtp.kv_restore(resume)` → the existing fresh-tail read path continues. `kv_stream_reset` per layer after the host copy is repopulated.
- **Test (the eviction test):** request conversation A (long), then conversation B (long, to overwrite the arena), then conversation A again with a longer tail. Assert: `RESUME <full A prefix>` printed, no re-prefill of A, and the greedy continuation of A's new tail is identical to the same tail appended in a never-evicted single-conversation run (state-hash equality at the resume point).
- **Gate:** measured TTFT for the return-to-A case ≤ ~1 s for a 128K session vs ~110 s re-prefill; correctness asserts green.

### Step 3 — Capacity, LRU, background writes (only if needed)

- Directory cap `--kv-nvme-max GB`, LRU by mtime, evict-oldest; *then* consider a background write thread — but only with a private staging buffer (or a dump restricted to blocks below the *next* request's resume, which is unknowable at DONE, so: private staging) because of §A gap 3. Re-measure whether async is worth 1.6 GiB of extra pinned RAM.
- **Test:** N+1 conversations over a cap of N; assert the oldest is evicted, the newest restores; assert no torn write under an interleaved branch (a stress loop with random branches while dumps run).
- **Gate:** no torn-write failures in 10k-iteration stress; measurable TTFT benefit from async vs the synchronous Step 1, else keep synchronous.

### Step 4 (optional, default skip) — Prefix dedup / sparsity

Only after profiling shows capacity or promote time is the bottleneck (§C). Sparsity (P4) additionally interacts with the indexer's top-k *selection*, which changes which cells are read — a correctness project, not a storage project; keep it out of scope until P2 measurements exist.

---

## F. OPEN QUESTIONS for the owner

1. **Single-tenant host tier — confirm.** Is "host = live session only, NVMe = the persistence tier" acceptable as the v1 architecture? The doc's P2 host LRU over sessions is a rewrite of the pinned arena (`layer.cpp:611-628`) for a one-session server; I recommend explicitly descoping it.
2. **Sync vs async dump.** Is a ~0.1–0.6 s synchronous dump at DONE acceptable? If not, are we willing to pay ~1.6 GiB pinned RAM for a private staging buffer, or restrict dumps to immutable prefixes below the last turn boundary?
3. **Snapshot completeness — sign off the set.** KV (all formats) + `idx_pooled` + `idx_block_pos` + `idx_dead` + `idx_tail` + gdn + ple + drafter host copy + ids + imgs + cvec flag. Anything else that should ride along (e.g. sampler/penalty state — I believe no: those are per-request, but confirm)?
4. **Non-streamed sessions.** Force streamed mode under `--kv-nvme`, or implement the kv_mode-0 D2H dump path? (Forcing streamed is cheaper and always correct.)
5. **kv_mode-0/small-context and format changes.** What happens on a restore attempt after a `--kv` format change (int8 → q4_0) or a context-size change? My recommendation: refuse (entry keyed on format+geometry tag), never convert.
6. **Durability/privacy.** `fsync` on dump (crash consistency) or accept a possibly-truncated snapshot (detected by a length/hash check in the header)? And note the store persists full conversation content (tokens + derivable KV) in plaintext on disk — is that acceptable deployment-wise?
7. **Is P3 dedup ever worth it here?** With one live session and ~1000 whole-session capacity, I see no case unless strata becomes multi-session; confirm before spending the index/collision/refcount budget.
8. **Drafter KV on restore.** Persist it (my recommendation — it's one layer, cheap, and keeps decode speed) or accept slower drafting on the first window after a promote?

---

## Review summary (concise)

- **Verdict:** direction sound; design over-engineered in keying (block-hash radix) and host tiering (multi-session LRU), and incomplete in its snapshot set. Merge verdict for the *design doc*: **OK with notes** — must fix §5/§8 (add `idx_pooled`/`idx_block_pos`/`idx_dead`/MTP drafter KV to the persisted set), demote P2/P3, and specify the dump/promote concurrency model before implementation.
- **Smallest correct v1:** whole-session snapshot file keyed by exact token prefix (ids stored verbatim, matched with the existing `starts_with`), dumped synchronously from the pinned host arena on DONE, promoted by `pread` into that arena + `kv_stream_reset` + `checkpoint_restore`, gated on format/geometry tags.
- **The falsifiable P0:** `STRATA_STATE_HASH` bit-equality + identical greedy continuation between a never-evicted run and an NVMe-restored run, pinned with `--adapt-swaps 0`, plus a deliberate-corruption negative control. The oracle already exists in the code (`generate.cpp:3040-3115`).