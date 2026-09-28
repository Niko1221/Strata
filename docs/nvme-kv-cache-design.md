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
  and the request continues from `resume`. A failed promote drops the entry and falls back to the
  clean path (session_zero + full read) - a half-applied snapshot can never be decoded from.
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

## 5. Test record

Oracles and harnesses (all exit non-zero on failure):
- `tools/nvme_p0_test.sh` - bit-exact restore: `STRATA_STATE_HASH` (refactored into
  `state_hash_line()`) printed straight after a restore (`STRATA_NVME_HASH`) must equal the
  dumper's DONE hash; identical-request processes compared; deliberate-corruption negative control
  (must be refused). GPU-needed, ~7 min.
- `tools/nvme_steps123_test.sh` - cascade/supersede (one growing file per conversation), promote
  after a process restart, LRU cap eviction. ~8 min.
- Live-server HTTP tests (`/v1/chat/completions`, streaming) - the needle test and the short
  correctness suite (driver scripts were run ad hoc; assertions listed below).

Results:
- **P0**: restore-exactness PASS (post-restore hash == dumper's hash, deterministic across runs);
  resume engages; negative control refuses. Token-equality across runs is informational only
  (engine decode nondeterminism, pre-existing - reproduced on the old binary).
- **Steps 1-3**: one growing file per conversation; promote after restart resumed 4235 tokens and
  read 13 fresh in ~350 ms; cap evicted to 0.83 GiB under 1 GB.
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
  request falls back to a full re-prefill and still answers correctly.

## 6. Known limitations / follow-ups

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

## 7. Prior art this design was checked against

vLLM's Automatic Prefix Caching (radix-tree content addressing - adopted in spirit: automatic,
prefix-keyed; simplified to exact keys for a single-session server) and its tiered KV offloading
(host-primary, secondary tiers, cascade/promote, LRU, ref-counts - adopted; multi-session host LRU
rejected as a rewrite of the single-session arena); llama.cpp's `--slot-save-path` (rejected:
client-initiated) and its hybrid-model restore no-op bug (#26676/#25913 - the cautionary tale
behind rule §4.2). The full reviews that shaped the v1 (glm-5.3 design critique and implementation
review) are in the branch history: commits `55e3337` (initial) and `0c7e6f7` (turn-boundary fix)
reference them; they were removed from `docs/` at completion.
