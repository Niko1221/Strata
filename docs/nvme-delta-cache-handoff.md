# Handoff: NVMe delta cache — append-only KV chunk log

**Purpose:** continuation brief for implementing the delta tier on top of the live
v3 snapshot tier. Design-first, review-first. **This revision (2) is a full rewrite:
it corrects stale facts from revision 1, folds in a critique of that proposal, and
adds the complete implementation spec and phased plan the executor (glm-5.3-flash)
can follow mechanically.**

---

## 0. Read-me-first for the implementer (glm-5.3-flash)

You are implementing an append-only, content-addressed chunk log next to the
existing whole-snapshot NVMe tier. The correctness bar is **bit-exact** restore.
The design is settled (§5); your job is to build it in the order of §6 and to stop
at every gate.

**Before writing any code, read these files in full:**

1. `docs/nvme-kv-cache-design.md` — the single design record for the live tier.
2. `include/strata/platform/kv_nvme.hpp` — the interface you will sit beside.
3. `src/platform/kv_nvme.cpp` — dump/restore walk you must reproduce segment-for-segment.
4. `src/platform/kv_nvme_host_test.cpp` — the host-fixture scaffolding you will copy.
5. `tools/nvme_p0_test.sh` and `tools/nvme_steps123_test.sh` — the oracles you will extend.

**Hard rules (violating any of these is a failed build, not a style issue):**

- R1. **Never modify the bytes `nvme_dump_at` / `nvme_restore` write or read.** The
  one exception is the mechanical Phase-3 extraction of `nvme_restore` into
  `nvme_restore_image` (line-for-line move, no logic changes).
- R2. **Never weaken a correctness gate:** the failure contract
  (`strata::core::ConversationRestore { restored, invalid, transfer_failed }`), the
  layout-before-integrity diagnostic ordering, the pre-apply validation pass, the
  image rule (pictures below `L`, compared at match time), the layer-split refusal,
  the `static_assert`s that pin on-disk structs.
- R3. **The delta tier must refuse, never convert** anything it cannot read.
- R4. The single acceptance oracle for the whole writer/reader pair is §5.2's
  **byte-identity invariant**: for the same session and boundary, the reassembled
  image must be byte-identical to what `nvme_dump_at` writes. When a test fails,
  `cmp` the two byte streams and read the first differing offset against the layout
  tables in §5.4–§5.6 before changing any code.
- R5. Follow the repo's documentation conventions: every struct and function gets a
  comment saying WHY (the repo's comments are load-bearing; see `kv_nvme.hpp`).
  Keep new comments shorter than the existing ones but never empty.
- R6. Stop at the first failing gate in §6 and triage with the table in that phase
  before continuing. Do not "fix forward" past a red test.

**Build and test loop (this machine):**

```bash
# build the engine and host fixtures
cd /local/strata/build && .venv/bin/ninja strata
# run the NVMe host fixtures (no GPU, no model)
.venv/bin/ctest -R kv_nvme_host_test --output-on-failure
# deploy the engine when a phase needs the server
cp build/strata engine/strata
```

New host fixtures register exactly like `kv_nvme_host_test` (see `CMakeLists.txt`
lines ~659-665): `add_executable`, `target_include_directories`, the same
`-Wl,--wrap` link options, `add_test`, `set_tests_properties(... ENVIRONMENT
"CUDA_VISIBLE_DEVICES=-1" ...)`.

---

## 1. What is live today (baseline — CORRECTED from revision 1)

Revision 1 described the v2 world. The repo has moved; these are the current facts.

Branch `nvme-kv-cache`, HEAD `a951094`, **pushed** (tip of `maedoc/nvme-kv-cache`
equals local HEAD — revision 1's §7 push problem is resolved). Base is
**v0.1.21** (`origin/main` f1b1d96), not v0.1.17. The design record is
`docs/nvme-kv-cache-design.md` (the convergence file was merged into it; every
source reference points there).

The live tier (`include/strata/platform/kv_nvme.hpp`, `src/platform/kv_nvme.cpp`,
serve integration in `src/program/generate.cpp`):

- **Snapshot format v3:** 208-byte `NvmeHeader` (magic `0x5E564D45`, `L`, imgs,
  `cvec`, `kv_format`, the shared core's **18-field `conversation_geometry_key`**
  verbatim, `page_size`, `idx_block`, `max_cells`, `mtp_host`) + ids + imgs + GDN/PLE
  running state + per-layer KV/pooled/tail/dead/block_pos + MTP drafter ring + 8-byte
  FNV-1a footer over the payload. Layout pinned by `static_assert`s; the shell
  oracles read `kNvmeHeaderBytes` via `tools/nvme_header_layout.sh`.
- **Turn-boundary dumps** (`nvme_dump_at`): snapshot at the longest
  `ConvCheckpoint` (end of prompt); every running-state byte comes from ONE source
  (the boundary checkpoint when given, the live arrays otherwise).
- **Failure contract (C7):** `nvme_restore` returns the shared core's
  `ConversationRestore`. `invalid` = refused before any CUDA call (recoverable:
  drop the entry, re-read the prompt). `transfer_failed` = a copy/sync failed
  mid-apply (FATAL: the session is half-applied by construction; the engine stops;
  the file is NOT dropped).
- **Image rule (correctness rule 8):** `NvmeEntry::imgs` is populated at scan; the
  match compares the request's pictures below the entry's `L`; a dump refuses a
  picture at or past `L`.
- **Layer-split rule (0.1.21):** a split session is never dumped or promoted — the
  envelope carries the primary stage only.
- **`KvNvmeStore`:** scan at engine start, exact-match idempotent skip, supersede
  (this process's previous dump, strict prefix), atomic restore, LRU byte cap.
- **The v2→v3 store conversion already happened:** all 106 v2 snapshots (199.27 GiB)
  were converted in place by `tools/nvme_v2_to_v3.cpp`; the originals sit in
  `kvstore-v2-backup/`. The production store scans as 106 sessions / 199.27 GiB.
  **There is no v2 coexistence question left to design for.**
- **Validated:** P0 oracle ALL PASS on device under v3, steps123 ALL PASS,
  image-path oracle, failure-contract oracle (249 checks), HTTP suite 19/19
  (`/tmp/short_tests.py`), live production serving, needle_bench parity.
  Restore staging measured: ~2 GiB transient for a 964 MiB snapshot (C10, bounded,
  accepted).

**Terminology (important):** "v3" is the SNAPSHOT format version
(`kNvmeFormatVersion = 3`) and stays frozen. The delta cache is a **new record
family** (chunks / state records / manifests) living in a subdirectory of the same
store. Do not bump `kNvmeFormatVersion` and do not touch `NvmeHeader`.

---

## 2. The problem the delta tier solves: write cost / SSD endurance

Measured from the production log (1,426 turns, ~2.5-day window):

| metric | value |
|---|---|
| tokens dumped (one full snapshot per DONE) | 70.4M |
| bytes written | **~1.13 TB → ~450 GB/day** |
| 1 TB consumer SSD (600 TBW) | ~4 years |
| append-only delta equivalent | **~34 GB same window (~14 GB/day), 33× less** |

Cost scales with **total session length**, not new tokens: a 110k-token session
writes ~1.7 GB per turn (16 KB/token). Sessions are growing (observed up to 137k),
so the burn rate grows. Secondary costs: ~0.3–0.5 s of server occupancy at DONE
(after the reply streams, before the next request), and O(total) restore reads.

Restore reads are O(prefix) in BOTH designs — the delta tier's win is entirely on
the write side (and on forks sharing parent chunks). Do not trade read volume for
write volume; reads are already the accepted cost.

---

## 3. Measured evidence that shaped the design

Ran against the live store (209 snapshots, 148 sessions ≥ 30k tokens, 11.5M tokens):

- **Cross-session prefix dedup is tiny:** pairwise common prefixes ~1,839 tokens
  median; total dedupable < 1% (~2 GB per write round). pi injects per-session-unique
  content within the first ~2k tokens, so streams diverge almost immediately.
- **Intra-session delta is the prize:** 33× today, approaching 100× at 200k sessions.
- **Fork sharing** is small in volume but free by construction once chunks are
  content-addressed; it dissolves deferred item P2-2 (strict-prefix supersede
  destroys a branched conversation's only cache).

Conclusion: design for delta/endurance/latency; dedup is a free rider.

---

## 4. Critique of revision 1's proposal (what this revision fixes)

Revision 1 was directionally right (append-only content-addressed chunks, small
per-turn state, manifest head moves) but had these defects, all now resolved in §5:

1. **Name collision.** It called the delta tier "v3". v3 is now the shipped snapshot
   format. Fixed: the delta tier is a separate record family; the snapshot format
   stays v3.
2. **Stale baseline.** 104-byte header / v2 coexistence / unpushed `d548f56` / v0.1.17
   base — all obsolete (§1).
3. **`cvec` missing from the chunk key.** A conversation stored with the control
   vector enabled must never be restored without it. Fixed: cvec is key material
   (§5.7). (A mid-conversation cvec toggle then behaves as a fork — correct.)
4. **Images missing from the manifest.** The resume match compares pictures below
   `L`; a manifest without an image list cannot be matched. Fixed: manifest carries
   ids AND imgs, and `kv_nvme_match` is reused verbatim.
5. **No failure-contract mapping.** Revision 1 never said what class a missing or
   torn chunk is. Fixed: §5.13 — every chunk/manifest problem is `invalid`
   (recoverable, before any CUDA call); only the apply pass can `transfer_failed`.
6. **Pre-apply validation vs chunked apply.** Naive chunk-at-a-time validate-then-apply
   breaks the tier's atomicity claim ("every refusal provably touches nothing").
   Fixed: restore validates ALL chunks and the manifest, then assembles the exact v3
   image, then runs the existing validation+apply pass unchanged (§5.10).
7. **Crash-consistency ordering unspecified.** Fixed: §5.9 — chunks and state are
   fsynced before the manifest; the manifest is fsynced before the old head is
   unlinked; a torn manifest/chunk is detectable and sweepable, never promotable.
8. **MTP ring is not append-only past `max_cells`.** The drafter ring wraps
   (`block -> block % n_slots`), so cells are overwritten and a naive per-range chunk
   would not reconstruct the ring. Fixed: the delta path requires `T ≤ max_cells`
   (the observed common case); longer boundaries fall back to a whole v3 snapshot
   dump (§5.15). Wrap-aware chunking is explicitly out of scope.
9. **Ragged chunks break immutability.** The last (partial) chunk's content changes
   every turn; content-addressing it would rewrite a "immutable" file. Fixed: **only
   sealed, block-aligned chunks are content-addressed.** The ragged tail
   (partial KV pages, the in-progress pooled row, tail MTP cells) lives in the
   per-turn State record (§5.5). This also kills the "supersede = head move with
   garbage ragged chunks" wrinkle.
10. **Parent refs for forks are unnecessary.** Content addressing already implies
    sharing; a manifest is a flat list of chunk keys. Dropped.
11. **`BLOCK` as a tuning constant was a false decision.** Chunks must not straddle
    KV pages or indexer blocks, so `BLOCK = lcm(page_size, idx_block)`, computed at
    runtime from `qsa_real_shapes()`, recorded in the manifest (§5.3).
12. **Weight-set fingerprint underspecified.** Fixed: §5.8 — exact bytes, computed
    once per process at store open.
13. **"~300–400 lines" was an underestimate** by ~3× once the repo's conventions
    (host fixtures, oracles, static_asserts, load-bearing comments) are honored.
    §6 budgets honestly.
14. **The migration story was moot** (v2 already converted) and **the test oracle was
    missing the obvious one:** byte-identity of reassembly against `nvme_dump_at`
    (§5.2), which inherits every validated property of the v3 path for free.

---

## 5. Design spec — the delta record family

### 5.1 Directory layout

Everything lives under the existing store directory; the v3 snapshot files stay
exactly where they are (`kvstore/kv-<pid>-<seq>.bin`):

```
<kv-nvme dir>/
  kv-<pid>-<seq>.bin               # v3 snapshots (unchanged, aging out via LRU)
  delta/
    chunks/<16-hex key>.bin        # sealed, immutable, content-addressed
    states/<16-hex digest>.bin     # per-turn State records, content-addressed
    log-<pid>-<seq>.manifest       # one per conversation head; superseded by unlink
```

`KvDeltaStore::open` creates `delta/` and its subdirectories on demand. A missing
`delta/` is not an error (the tier simply starts empty).

### 5.2 THE invariant (the whole design hangs on this)

> **Byte-identity invariant.** For any session and boundary `T` for which the delta
> path applies (§5.15), assembling (sealed chunks in key order) + (the head State
> record at `T`) + (the manifest's ids/imgs/header fields) must produce a byte
> stream **identical** to what `nvme_dump_at(path, …, at=boundary)` writes for the
> same session and boundary — header, payload, and footer digest included.

Consequences (all testable by `cmp`):

- Every correctness property of the validated v3 path (segment order, pooled-row
  formula, spare row, C5 boundary sources, image filtering, digest coverage) is
  inherited without re-derivation.
- Restore can reuse the v3 validation+apply pass verbatim (§5.10).
- The fork case is free: a fork's reassembly is its own manifest's chunks — shared
  parent chunks are the same bytes.

The implementation cost of the invariant: chunk and State payloads are laid out as
**exact contiguous slices of the v3 snapshot's segments** (§5.4–§5.6).

### 5.3 Runtime shapes and BLOCK

```
shapes = qsa_real_shapes()                 // page, idx_block — same source as Sizes
BLOCK  = lcm(shapes.page_size, shapes.idx_block)   // tokens per sealed chunk
sealed(T)  = (T / BLOCK) * BLOCK           // token length covered by sealed chunks
rows_per_chunk = BLOCK / shapes.idx_block  // sealed pooled rows per chunk
```

`BLOCK` is computed once, recorded in every manifest, and validated at open/restore.
It is a multiple of both `page_size` (so sealed KV slices are whole pages) and
`idx_block` (so sealed pooled-row slices are whole rows). No tuning constant; if a
future engine changes shapes, old manifests refuse (never convert) and age out.

### 5.4 Chunk record (`delta/chunks/<hex16>.bin`)

```cpp
inline constexpr uint32_t kDeltaChunkMagic    = 0x4B4E4843;  // "CHNK"
inline constexpr uint32_t kDeltaStateMagic    = 0x54415453;  // "STAT"
inline constexpr uint32_t kDeltaManifestMagic = 0x474F4C44;  // "DLOG"
inline constexpr uint32_t kDeltaFormatVersion = 1;

struct DeltaChunkHeader {          // 48 bytes, padding-free, pinned by static_asserts
    uint32_t magic   = kDeltaChunkMagic;
    uint32_t version = kDeltaFormatVersion;
    uint64_t key = 0;              // the chunk key (§5.7); also the file name
    int64_t  a = 0;                // first token position; a % BLOCK == 0
    int64_t  b = 0;                // a + BLOCK (sealed chunks only)
    int64_t  layers = 0;           // n_qsa_layers, recorded for the walk
    int64_t  payload_bytes = 0;    // everything between header and footer
};
// file = header || payload || footer
// footer = uint64_t FNV-1a over payload
```

**Payload layout** (for chunk covering tokens `[a, a+BLOCK)`; every quantity
below is identical for every sealed chunk of a store, given one geometry — the
reader sizes it from the manifest's shapes, never from the chunk header):

| order | content | bytes |
|---|---|---|
| for each qsa layer `i` in `[0, n_qsa_layers)` | | |
| — for each kv array `k` in `[0, kv_array_count(st_i))` | KV host-array slice, pages `[a/page, (a+BLOCK)/page)` | `(BLOCK/page) * n_head_kv * page * w_k` |
| — pooled rows | `idx_pooled` rows `[a/idx_block, (a+BLOCK)/idx_block)`, D2H copy | `rows_per_chunk * idx_key_dim * 4` |
| then MTP drafter arrays `k` in `[0, kv_array_count(mtp))` (only while `a < mtp max_cells`; §5.15) | drafter host-array slice, same page range | `(BLOCK/page) * n_head_kv * page * w_k` |

Array order and widths are the v3 walk's own: `kv_host_arrays(st, head_dim, k)` in
`k` order (`kv_array_count`: 2 for q4/fp16, 4 for int8). The KV arrays are
host-resident (streamed mode is required for the tier, same as v3); the slice is a
plain contiguous range of the pinned host array (pages are contiguous
`n_head_kv * page * w` blocks). The pooled rows come off the device with one
`cudaMemcpy` per layer per chunk — consume any CUDA error on failure
(`consume_cuda_error()` discipline, kv_nvme.cpp).

### 5.5 State record (`delta/states/<hex16>.bin`)

One per turn, content-addressed by its own payload digest. Payload for boundary
`T` with sealed prefix `S = sealed(T)`:

| order | content | bytes |
|---|---|---|
| `gdn` | checkpoint's `at->gdn` | `z.state.gdn` |
| `ple` (if `ss.ple_hist`, same presence rule as v3) | `at->ple` | `z.state.ple` |
| for each qsa layer `i` | | |
| — for each kv array `k` | tail KV pages `[S/page, ceil(T/page))` | `(ceil(T/page) - S/page) * n_head_kv * page * w_k` |
| — tail pooled rows | `idx_pooled` rows `[S/idx_block, qsa_pooled_rows(T))` (includes the in-progress + spare rows) | `(qsa_pooled_rows(T) - S/idx_block) * idx_key_dim * 4` |
| — `tail` | `at->tails` slice for layer i | `z.state.tail` |
| — `dead` | `at->dead` slice | `z.state.dead` |
| — `block_pos` | `at->block_pos` slice | `z.state.block_pos` |
| MTP drafter arrays `k` | tail pages `[S/page, ceil(T/page))` (only if `T <= mtp max_cells`) | as above |

Every source is the boundary checkpoint's blob — the C5 rule, unchanged. Header:
`{ u32 magic = kDeltaStateMagic; u32 version; i64 payload_bytes; }` (16 bytes);
footer: `uint64_t FNV-1a` over payload; the digest is the file name and the
manifest's `state_key`.

Size: the ragged tail is < BLOCK tokens (~4 MB at BLOCK 256 × 16 KB/token) plus
~KBs of running state. Per-turn write: new sealed chunks + this + the manifest.

### 5.6 Manifest (`delta/log-<pid>-<seq>.manifest`)

```cpp
struct DeltaManifestHeader {      // 256 bytes, padding-free, static_asserts pin it
    uint32_t magic   = kDeltaManifestMagic;
    uint32_t version = kDeltaFormatVersion;
    int64_t  L = 0;               // boundary length T (the resume key)
    int64_t  block = 0;           // BLOCK (validated against the live lcm)
    int64_t  n_chunks = 0;        // sealed(T) / BLOCK
    int64_t  n_imgs = 0;
    int32_t  cvec = 0;
    int32_t  kv_format = 0;
    int64_t  page_size = 0, idx_block = 0, max_cells = 0, mtp_host = 0;
    int64_t  geometry[18] = {};   // conversation_geometry_key(g), verbatim
    uint64_t weights_fp = 0;      // §5.8
    uint64_t state_key = 0;       // the head State record's digest
    int64_t  pid = 0, seq = 0;    // the file name, for debugging
    int64_t  reserved = 0;
};
// file = header || body || footer
// body  = ids (int32 * L) || imgs (ConversationImageKey * n_imgs)
//       || per chunk j: { u64 key; int64_t a; }   (16 bytes * n_chunks; b = a + block)
// footer = uint64_t FNV-1a over body
```

Manifest rewrite per turn is a full rewrite of a small file (~0.8 MB at 200k tokens
— 0.02% of the KV bytes; do not optimize it). `mtp_host` = number of drafter arrays
written, exactly as the v3 dump counts it.

`NvmeEntry` gains one field: `int kind = 0;` (`0` = v3 snapshot, `1` = delta
manifest). `kv_nvme_match` is reused **verbatim** — it already compares ids, imgs
below `L`, cvec, and picks the longest match. Delta entries are scanned from
manifests at `open()` exactly as v3 entries are scanned from snapshots.

### 5.7 Chunk key derivation (prefix property)

```
tag  = FNV1a( FNV_OFFSET_BASIS,
              geometry_key bytes || kv_format || cvec || weights_fp || block )
c_0  = tag
c_j  = FNV1a( c_{j-1}, ids[((j-1)*BLOCK) .. (j*BLOCK)) )        // j = 1..n_chunks
key(chunk covering [j*BLOCK, (j+1)*BLOCK)) = c_{j+1}
state_key = FNV1a( tag, state_payload_bytes )
```

A chunk's key commits to the full token prefix through its own last token, so it is
reachable only by prompts sharing that prefix (forks share; stale chunks are
unreachable). 64-bit keys: birthday collision at 10^6 chunks is ~10^-8, and a
collision is caught by the §5.2 footer check (a refused restore, never corruption) —
document this in the header comment.

### 5.8 Weight-set fingerprint

Computed **once per process** at store open:

```
weights_fp = FNV-1a over: resolved model file path bytes || file size (int64)
             || first 64 KiB of the file || last 64 KiB of the file
```

Two 64 KiB reads on NVMe at startup; no false positives across same-geometry
different-weights models (the v2 review's open item). Checked at **match time**
(a delta entry whose `weights_fp` differs from the live one is not a candidate —
same treatment as cvec), never per-chunk. On mismatch log once per process:
`strata serve: kv-delta: N stored conversations belong to a different weight set - skipped`.

### 5.9 Writer algorithm (the DONE path)

At DONE with the boundary checkpoint `at` (the existing selection logic in
`generate.cpp` is unchanged — the delta tier replaces only the *serialization*):

1. `T = at ? at->ids.size() : live.size()`. Refuse exactly as `nvme_dump_at` does:
   `kv_mode == 0`, split session (`stage_parts` non-empty), images outside `[0, T)`.
   Additionally: **if `T > mtp_state.max_cells`, fall back to the v3 whole-snapshot
   dump** (`KvNvmeStore::dump`) — §5.15.
2. Find this conversation's previous head: the store's tracked previous dump of
   this process (the `last_ids_` discipline), i.e. the manifest whose ids are a
   strict prefix of the new ids. If none, or the ids regressed (not a prefix — a
   fork or a rewrite), all chunks are new.
3. **Reuse** every sealed chunk of the previous head whose `(j+1)*BLOCK <= prev_T`.
   Everything from `floor(prev_T / BLOCK)` on is new (the chunk containing the old
   ragged tail is re-derived; its old State record and old manifest become garbage).
4. Write each new sealed chunk: gather payload slices (host arrays: memcpy from the
   pinned pool; pooled rows: one `cudaMemcpy` D2H per layer), write
   `chunks/<hex>.bin`, `fsync`. **Failures here abort the dump, leave no manifest
   move, and leave only sweepable garbage.** Consume CUDA errors.
5. Write the State record from the checkpoint blobs; `fsync`.
6. Write the new manifest (`log-<pid>-<newseq>.manifest`), `fsync`, **then** unlink
   the previous head manifest. Crash between the two fsyncs: the new head is durable
   and the old is either present (scanned as a separate conversation; harmless) or
   gone — never a torn head, because manifest files themselves are footer-checked.
7. Register the entry; `enforce_cap()` (§5.12).

Write volume per growing turn: `16 KB × new_tokens` + tail State (~4 MB) +
manifest (~KB). Idempotent re-dump (same ids) refreshes mtime only.

### 5.10 Restore algorithm (Phase A: correctness-first)

1. Entry matched by `kv_nvme_match` over the combined entry list (§5.11).
2. **Validate everything, no CUDA calls:** manifest magic/version/geometry/
   kv_format/block-vs-lcm/max_cells rule/weights (match-time), footer digest;
   every chunk file exists, header `key`/`a`/`b` match the manifest record, size
   matches, payload digest matches; the State record likewise. Any failure →
   `ConversationRestore::invalid` with a message naming the record and the reason.
3. **Assemble** the v3 image in one `std::vector<uint8_t>` (the same whole-file
   staging the v3 restore already does — C10's bounded, accepted cost), interleaving
   chunk slices and State slices per the v3 segment order (layer loop:
   kv arrays → pooled → tail → dead → block_pos), then the MTP sealed+tail pages,
   then the footer digest computed over the assembled payload.
4. Run the **existing** validation+apply pass on the assembled image
   (`nvme_restore_image`, the Phase-3 extraction of `nvme_restore`): layout walk →
   layout-drift diagnostics → digest → apply → `STATE_HASH` gate, failure classes
   unchanged.

Phase B (optional, later, behind measurement): pread chunks directly into the
pinned host pools and device arrays with per-chunk digest validation, skipping the
staging buffer. Do NOT attempt Phase B until Phase 5 is green — Phase A's restore
read volume is the same order as v3's (O(prefix)), which is the accepted cost.

### 5.11 Match and promote

Promote selection becomes: RAM checkpoints → **longest match across both tiers** →
cold read. Concretely in the serve loop:

```cpp
const NvmeEntry* bd = kv_nvme_match(delta.entries(), ids, req_imgs, cvec, resume);
const NvmeEntry* bs = kv_nvme_match(store.entries(), ids, req_imgs, cvec, resume);
const NvmeEntry* best = (bd && (!bs || bd->L > bs->L)) ? bd : bs;
// restore via the store matching best->kind; failure handling UNCHANGED (§5.13)
```

The existing failure handling in `generate.cpp` (transfer → stop; invalid → drop,
re-read) is reused verbatim, dispatching on `kind`.

### 5.12 GC, supersede, eviction

- **Supersede** = the §5.9 manifest head move (unlink old head after the new one is
  durable). The old head's exclusive chunks/states become garbage.
- **GC = mark-and-sweep.** No refcounts on the write path. Sweep at (a) `open()`
  after the scan, and (b) cap pressure, after eviction. Sweep = collect the union of
  chunk/state keys referenced by all live manifests, scan `chunks/` and `states/`,
  unlink unreferenced files. Log one line: `kv-delta: swept N orphan chunks (X GiB)`.
- **Eviction** = LRU by manifest mtime across BOTH tiers under ONE byte cap:
  `total = v3 snapshot bytes + delta (chunks+states+manifests)`. Evicting a delta
  conversation = unlink its manifest; its chunks are reclaimed by the next sweep.
  Evicting a v3 snapshot = today's behavior.
- Never convert, never delete a chunk referenced by a live manifest (the sweep's
  union is computed BEFORE any unlink).

### 5.13 Failure-contract mapping (do not invent new classes)

| failure | class | consequence |
|---|---|---|
| manifest magic/version/geometry/format/block/weights/cvec mismatch, footer digest, malformed sizes | `invalid` | drop the manifest entry, re-read the prompt (serve code unchanged) |
| missing chunk/state file, header mismatch, size mismatch, payload digest mismatch | `invalid` | same — and the manifest is untrustworthy, drop it |
| assemble-time OOM / short read | `invalid` | same |
| any `cudaMemcpy`/sync inside the apply pass | `transfer_failed` | **engine stops; nothing is dropped** — existing serve code, unchanged |
| dump-side failure (chunk write, D2H copy) | dump fails | no manifest move; garbage swept later; the conversation is simply not cached this turn |

Diagnostic ordering rule preserved: layout/size facts BEFORE digest verdicts, in
every message (R2; the `2fde2e2` discipline).

### 5.14 Coexistence and migration

- v3 snapshots and delta manifests coexist in one store, one cap, one LRU.
- **No conversion of the 106 stored v3 snapshots.** Delta writes start empty; v3
  files age out under the shared LRU. (A converter would re-write 199 GB to save
  nothing — refuse-never-convert also means never-convert-for-conversion's-sake.)
- A conversation may have both a v3 file and a delta manifest mid-transition; the
  longest-match rule resolves it, and supersede in each tier only ever touches that
  tier's own previous head of the same process.

### 5.15 Limitations (stated, not hidden)

- **`T > mtp max_cells` falls back to v3 whole-snapshot dumps.** The drafter ring
  wraps; wrap-aware chunking is out of scope. If the fallback fires in production
  (log line: `nvme delta: boundary N exceeds the drafter ring (M) - whole snapshot`),
  revisit.
- **Layer-split sessions are inert** for both tiers (existing rule).
- **Single writer** per store directory (as today). Chunks are created with
  `O_EXCL`-style semantics (write to `chunks/.tmp-<pid>-<seq>`, `fsync`, rename to
  the content-addressed name) so two processes cannot interleave a torn chunk.
- **Whole-file staging at restore** (Phase A) — the measured, accepted C10 cost.

### 5.16 Resolved decisions (supersedes revision 1's §6 open items)

| revision-1 open item | resolution |
|---|---|
| BLOCK 256 vs 512 vs page-aligned | runtime `lcm(page_size, idx_block)`, recorded per manifest (§5.3) |
| MTP chunked vs snapshotted | chunked while `T ≤ max_cells` (contiguous ring cells); v3 fallback beyond (§5.15) |
| GC refcount vs mark-and-sweep | mark-and-sweep at open + cap pressure (§5.12) |
| State per-turn vs CoW-deduped | per-turn, content-addressed by digest; dedup is free when identical |
| weight fingerprint bytes | path + size + first/last 64 KiB FNV (§5.8) |

---

## 6. Implementation plan — phased, gated, mechanical

Budget (honest): ~500 lines `src/platform/kv_delta.cpp`, ~250 lines
`include/strata/platform/kv_delta.hpp`, ~600 lines host fixture, ~80 lines in
`generate.cpp`, ~30 lines `CMakeLists.txt`. If a phase runs 2× over budget, stop and
re-read §5 — you have drifted, not the design.

Every phase ends with: build green, its tests green, the EXISTING tests still green
(`ctest -R kv_nvme_host_test`, and from Phase 3 on-device, `tools/nvme_p0_test.sh` +
`tools/nvme_steps123_test.sh`), and one commit whose message states what the phase
proved (the repo's commit-message convention: outcome first, evidence second).

### Phase 0 — hygiene (no engine code)

1. Commit this document (`docs/nvme-delta-cache-handoff.md`).
2. Move the three untracked oracles into `tools/` and commit:
   `/tmp/short_tests.py`, `/tmp/fnvaudit.c` (+ binary), per the §7 note.
3. Verify push state: `git log --oneline -1 maedoc/nvme-kv-cache` must equal local
   HEAD; push if not.
4. Ask the owner about committing `tools/make_profile_blend.py`,
   `data/expert-profile-personal.bin` (§8) — do not decide this unilaterally.

**Gate:** clean `git status` for everything this task owns.

### Phase 1 — chunk/state/manifest records + host fixture

Files: `include/strata/platform/kv_delta.hpp`, `src/platform/kv_delta.cpp`,
`src/platform/kv_delta_host_test.cpp`, `CMakeLists.txt` (register the fixture
exactly like `kv_nvme_host_test`, including the `-Wl,--wrap` options and
`CUDA_VISIBLE_DEVICES=-1`).

Implement: the constants and three structs of §5.4–§5.6 with `static_assert`s
pinning size/alignment/offsets (copy the `NvmeHeader` discipline); pure byte-math
functions (`delta_shapes`, per-segment slice sizes and offsets for a given
geometry/shapes — all pure, no I/O); `delta_write_chunk(path, header, payload)` /
`delta_read_chunk(path, expected key/a/b, payload out, err)` with FNV footer
verification; same for State records; `fnv1a` reuse from `kv_nvme.cpp` (expose it
or duplicate the 4-line function in an anonymous namespace — prefer exposing).

Host-test cases (all CPU-only):
- round-trip: write/read a chunk and a State record byte-identically;
- truncated tail (chop N bytes) → refused, `invalid`-class error string;
- digest corruption (flip one payload byte) → refused;
- wrong `key`/`a`/`b` expectation → refused;
- `static_assert` compile check is itself a test (build failure = gate failure).

**Gate:** `ninja strata && ctest -R kv_delta_host_test` green; `ctest -R
kv_nvme_host_test` still green. Triage: byte offsets wrong → §5.4 table; digest
wrong → hash covers payload only, never header/footer.

### Phase 2 — the writer: `delta_dump_at` + byte-identity oracle

Add to `kv_delta.cpp`: `bool delta_dump_at(const KvDeltaStore::Head* prev, const
std::string& dir, const SessionState&, const QsaState& mtp, const ModelGeometry&,
const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
bool cvec, const ConversationCheckpoint* at, std::string& err)` implementing §5.9.
Gather: host-array slices are `memcpy` (the arrays are already host-pinned); pooled
rows and (in the host fixture, wrapped) device slices go through the same
`cudaMemcpy` call sites the fixture wraps. Write via temp-name + fsync + rename.

Host-test cases, with the synthetic session scaffolding copied from
`kv_nvme_host_test.cpp` (same wrapped-CUDA stand-in):
- **THE oracle:** for T ∈ {small, mid, exactly BLOCK, BLOCK+ε, several BLOCKs}:
  dump v3 with `nvme_dump_at` to file A; dump delta to a store dir; reassemble
  (Phase 2 reassembly can be a test-local helper — Phase 3 productizes it); `cmp`
  A vs reassembly. **This single check subsumes the segment tables.**
- growth: dump at T1, then extend the session, dump at T2 > T1 → only new sealed
  chunks on disk (count files and bytes; sealed set of dump 1 ⊆ dump 2);
- idempotent re-dump at the same T → zero new files;
- fork: dump at T1; mutate the tail tokens; dump at T2 with a different suffix →
  shared sealed chunks identical (same file names), both manifests valid;
- dump-side failure (fail a wrapped `cudaMemcpy`) → no manifest written, no head
  moved, garbage chunks only.

**Gate:** all green, existing fixtures green. Triage: identity mismatch → `cmp -l`
the two files; the first differing byte, mapped through §5.4–§5.6, names the
mis-ordered segment. Chunk-file reuse wrong → you reused the ragged chunk (sealed
means `(j+1)*BLOCK <= prev_T`).

### Phase 3 — the reader: `nvme_restore_image` extraction + `delta_restore`

1. **Mechanical extraction** in `kv_nvme.cpp`: move the body of `nvme_restore`
   after the whole-file read into
   `nvme_restore_image(const uint8_t* data, size_t n, SessionState&, QsaState&,
   const ModelGeometry&, ids, imgs, cvec, L, err)`; `nvme_restore(path, …)` becomes
   open/size-check/read + call. Line-for-line; no logic changes; keep the size cap
   (64 GiB) in the file wrapper and the image variant.
2. `delta_restore(entry, …)` in `kv_delta.cpp`: §5.10 — validate manifest, chunks,
   state; assemble; call `nvme_restore_image`.
3. Host-test cases:
   - restore round-trip on the fixture: the restored session's arrays equal the
     dumper's (the fixture's existing comparison discipline);
   - **cross-check:** `nvme_restore(file A)` and `delta_restore(store)` produce
     identical array contents and identical `L/ids/imgs`;
   - delete one chunk file → `invalid`, message names the chunk;
   - truncate one chunk → `invalid`;
   - corrupt manifest footer → `invalid`;
   - wrong-prefix restore (request ids diverge inside a sealed chunk) → never
     matched (the `kv_nvme_match` negative control, already covered by its own
     fixture — re-run it over delta entries);
   - weights_fp mismatch → not a candidate.

**Gate:** host fixtures green AND `tools/nvme_p0_test.sh` +
`tools/nvme_steps123_test.sh` green on device (the extraction regression gate —
server must be STOPPED first, §9). Triage: P0 breaks after extraction → you changed
logic, not just structure; diff the moved lines.

### Phase 4 — `KvDeltaStore` + serve wiring

`kv_delta.hpp`: the store class (`open` scans manifests → `NvmeEntry` with
`kind=1`; `dump` = §5.9 with head tracking; `restore`; `drop`; `sweep`;
`total_bytes`). `NvmeEntry` gains `int kind` in `kv_nvme.hpp` (default 0; v3 paths
untouched).

`generate.cpp`:
- option `--kv-delta N` (0/1, **default 0** in this phase), reusing `o.kv_nvme`
  + `/delta`;
- compute `weights_fp` at startup (§5.8) from the model path;
- DONE path: delta on && not split && `T ≤ mtp max_cells` → `delta_dump` else
  today's `kvstore.dump`;
- promote: the two-match longest-wins of §5.11; failure handling unchanged,
  dispatched on `kind`;
- cap: combined totals; evict LRU across both; sweep after eviction.

Host-test cases: scan of a mixed dir (v3 + delta) → correct entries/kinds; cap
eviction across tiers; sweep removes exactly the orphans; supersede unlinks only
the old head.

**Gate:** host fixtures green; a manual engine run with `--kv-delta 1` over 3 HTTP
requests shows, in the log: one delta dump (`nvme delta: appended N chunks (X
bytes) T prev→new` — add that instrumentation line), one `nvme promote: resumed N
tokens from delta/log-…manifest`, and `reuse` lines unchanged in shape.

### Phase 5 — on-device oracles + soak

New `tools/nvme_delta_p0_test.sh` (copy `nvme_p0_test.sh`'s structure, honor
`NVME_ENGINE` and `tools/nvme_header_layout.sh` conventions where applicable):
1. bit-exact: dump via delta, restore, `STRATA_STATE_HASH` equals the dumper's DONE
   hash (the P0 discipline);
2. negative control: corrupt one chunk byte → promote refused with the chunk named
   in the message, prompt re-read (`RESUME 0`), engine alive;
3. fork: two conversations sharing a long prefix → the second one's chunks are
   mostly shared (log the shared-byte count), both restore bit-exactly;
4. cascade: a 5-turn conversation → per-turn log line's bytes ≈ new tokens ×
   16 KB (+ ~4 MB tail), NOT total × 16 KB;
5. crash safety: for each C1..C6, `STRATA_DELTA_FAIL_AT=Ck` on the dump, relaunch,
   assert the WAL-matrix row for that point (clean scan, valid promote, sweep
   exactness — the table in the cross-cutting section above);
6. re-run `tools/nvme_p0_test.sh`, `tools/nvme_steps123_test.sh`,
   `tools/nvme_image_promote_test.sh`, `tools/nvme_failure_contract_test.sh`
   unchanged (regression);
7. `/tmp/short_tests.py` (now `tools/short_tests.py`) 19/19 with `--kv-delta 1`.

Then: deploy (`cp build/strata engine/strata`, restart via §9's procedure), soak
the live server with delta on for a day, and collect the endurance numbers from the
instrumented log lines (sum bytes/day; compare against §2's ~450 GB/day baseline).

**Gate:** every oracle ALL PASS; a day of live serving with ≥ 20× write reduction
and unchanged reuse percentage.

### Cross-cutting: the WAL correctness matrix (applies across phases 2, 4, 5)

"The log is correct" is seven properties; each has a named test. Do not improvise new
coverage — build these.

| # | property | test |
|---|---|---|
| P1 | playback fidelity: any live manifest reassembles byte-identically to the v3 snapshot of its prefix | the §5.2 `cmp` oracle (host), `STRATA_STATE_HASH` (device) |
| P2 | crash safety at every interruption point | crash-point enumeration below, via `STRATA_DELTA_FAIL_AT` |
| P3 | supersede semantics (idempotent, growth = new chunks only, forks share) | phase-2 host cases + the differential fuzz |
| P4 | eviction never breaks sharing | the fork-eviction case below |
| P5 | sweep deletes exactly the unreferenced set | every crash case asserts sweep exactness; the fuzz asserts it per-op |
| P6 | every disk defect is `invalid`, zero CUDA calls, sentinel buffers untouched | phase-3 host cases (the fixture's existing discipline) |
| P7 | TOCTOU: external deletion degrades to refuse-and-drop | delete a file after scan, before restore |

**Crash-point enumeration (deterministic, never `kill -9` luck).** The write protocol
is 6 steps; the post-crash disk state of each is constructible on demand:

```
C1  chunk temp written (not fsynced)      C4  new manifest fsynced + renamed   <- the commit point
C2  chunk fsynced + renamed               C5  old head manifest unlinked
C3  State record fsynced + renamed        C6  in-memory only (cap, entry)
```

Add a fault hook to `delta_dump_at` — `STRATA_DELTA_FAIL_AT=<C1..C6>` aborts at that
step — following the `[STRATA_TEST_FAIL_CUDA]` precedent in `kv_nvme.cpp` (debug
device, documented the same way). Expected recovery, each row asserted by BOTH the
host fixture and `tools/nvme_delta_p0_test.sh` on the real binary and real store
(dump with fault at C_k -> relaunch -> clean scan -> valid promote -> sweep):

| crash at | live heads | garbage | assert |
|---|---|---|---|
| C1/C2 | old head (or none) | partial sealed chunks | open clean; old head reassembles; sweep removes exactly the partials; scan ignores `.tmp-*` names |
| C3 | old head | chunks + state | same, plus the state is an orphan |
| C4 | BOTH old and new head | none | open clean; both heads reassemble byte-identically; longest-match picks the new; the duplicate ages out — this is the interesting row, test it explicitly |
| C5 | new head only | old head's exclusive chunks | final state; sweep reclaims exactly those |

Two structural facts the tests PROVE rather than assume: C4-before-C5 means the new
head is complete before the old disappears; and immutable digest-checked chunks mean
a half-written chunk can never sit under a live manifest (it is garbage or a refusal).

**Differential fuzz (the playback workhorse).** In the host fixture: ops = {extend by
a random amount, fork at a random prefix, re-dump same T, crash@C_k (random), evict
with the cap set to a random fraction, sweep, external-delete a random file}; a
reference model tracks live manifests -> expected chunk-key sets and expected
reassembly bytes. After EVERY op assert: (a) disk scan == the model's live set;
(b) every live manifest reassembles byte-identically; (c) every chunk/state file is
either referenced by a live manifest or removable by sweep — no third kind;
(d) sweep output == exactly the unreferenced set. ~1000 seeds x ~200 ops, seed printed
for reproduction. Sequences no hand-written case enumerates live here.

**Eviction/GC cases (hand-written, synthetic store with injected mtimes/sizes):**

1. fork sharing survives eviction — manifests A and B share most chunks; evict A,
   sweep: shared chunks STILL PRESENT (B references them); evict B, sweep: gone. This
   is the test that catches a naive "evict = delete the conversation's chunks" bug —
   the P2-2 regression the design exists to fix.
2. two-tier single cap: mixed v3 + delta store; drive the cap down; eviction order is
   global oldest-mtime; accounting counts chunks + states + manifests + v3 bytes.
3. boundary: cap smaller than one conversation -> everything evicted, store still
   opens and scans clean.
4. recency: an idempotent re-dump refreshes mtime, so the active conversation is
   never the eviction victim.

**Honest limits (state them in the test file header, the `kv_nvme_host_test` way):**
these tests prove every *file-level* interruption state is safe; they cannot prove
the SSD actually persisted on fsync (a power-cut property). That is acceptable
because the posture degrades, never corrupts: a torn-on-disk chunk fails its digest
to `invalid` -> refuse -> re-prefill. The wrapped `cudaMemcpy` limits are inherited
from the existing fixture's WHAT-IT-CANNOT-PROVE header.

### Phase 6 — default-on, docs, close-out

1. Flip `--kv-delta` default to 1 (v3 dump remains the fallback path).
2. Fold this document's design sections into `docs/nvme-kv-cache-design.md` as the
   delta-tier section (the repo keeps ONE design record); keep this handoff as the
   build record or delete it per the owner's preference.
3. Endurance report: one table (v3 baseline vs delta, from the soak).
4. Ask the owner about `kvstore-v2-backup/` deletion (200 GB) — after a week of
   clean delta promotes.

---

## 7. Repo / machine state (updated)

- **Pushed:** `maedoc/nvme-kv-cache` tip = local HEAD `a951094` — resolved.
- **Store:** 106 v3 sessions / 199.27 GiB under the 200 GB cap; `kvstore-v2-backup/`
  holds the converted-from originals (200 GB; deletion is Phase 6).
- **Untracked, this task:** this document; `tools/make_profile_blend.py`;
  `data/expert-profile-personal.bin` (196,632 B, live in the server config);
  `/tmp/short_tests.py`, `/tmp/fnvaudit{,.c}` (Phase 0 commits these).
- **Untracked, other work:** `HANDOFF-converge-rebase.md`,
  `HANDOFF-decode-tuning.md`, `DECODE-TUNING-4090.md`, `tools/decode-probe.sh`,
  `bench-*`, `routing.bin` (1.84 GB, static) — not part of this task.
- **Live config** `strata-iq3_xxs.json`: `--max-context 262144`, `--expert-profile
  /local/strata/data/expert-profile-personal.bin`, `--kv-nvme /local/strata/kvstore`,
  `--kv-nvme-max 200`, `--prompt-cache 4`, `--spec 4`,
  `--mtp /local/strata/mtp/rt`, `--kv int8`.
- Log: `strata-iq3_xxs.log`; promote/reuse line shapes in §9.

## 8. Expert-profile result (adjacent, completed — unchanged from revision 1)

- Trace: 427,784 decode positions, 48 layers, k=10, fixed 88-byte records;
  24,510/24,576 pairs routed.
- Shipped profile covers 75.9% of this user's routings; personal top-10,559 85.6%
  in-sample; split-half: shipped 77.64%, trace 83.51%, **blend `--weight 0.05`
  84.06%** (best; tool default 0.25 over-weights base).
- Built: `.venv/bin/python tools/make_profile_blend.py /tmp/routing-final.bin
  --weight 0.05 --slots 10559 --out data/expert-profile-personal.bin`.
- Trap: `tools/make_profile.py` base-first is a no-op on a < 24,576-pair card —
  use the blend tool.
- Live: decode ~100–110 tok/s on big sessions (was ~78–90).

## 9. Operational gotchas (this machine — READ BEFORE ANY SERVER WORK)

- Build: `cd /local/strata/build && .venv/bin/ninja strata`; deploy `cp
  build/strata engine/strata`.
- Server: detached `/tmp/serve-loop.sh` relaunching `.venv/bin/python -m
  serve.server --engine strata --config strata-iq3_xxs.json --port 8080`. **Kill the
  loop first** (`pkill -f 'serve-loop\.sh'`), then the server (`pkill -f
  'serve\.server'`), in **separate** command blocks — a pattern matching the
  block's own cmdline kills the block. Never combine pkill with relaunch text.
- **Do not run P0/oracle tests while the server holds the memlock budget.** Stop
  server → run tests → relaunch.
- Long `sleep` blocks get killed by the harness (exit 143/137); poll in short
  chunks.
- Log lines to know: `nvme promote: resumed N tokens from …`; `prompt N tokens =
  X reused + Y read in Z ms` (`Y == N` means cold); `ready: … context 262144
  tokens` confirms context; new in Phase 4+: `nvme delta: appended N chunks (X
  bytes) T prev→new` and `nvme promote: resumed N tokens from delta/log-…manifest`.
- Store is scanned **once at engine start**; external file removal leaves stale
  in-memory entries until restart (drop-on-failure handles it gracefully).
- `strata generate: --kv-nvme needs the KV host copy` in the log means the engine
  started without the streamed-KV flags — the delta tier (like v3) requires them.

## 10. First actions next session

1. Phase 0 (hygiene: commit this doc + the three oracles; verify push).
2. Phase 1 as written — do not start with the serve loop; the first two phases are
   entirely host-side.
3. If §5.2's oracle fails in Phase 2 and the layout tables don't explain the
   differing byte, stop and hand back to the owner with the two files and the first
   differing offset — that is a design bug, not an implementation bug.
