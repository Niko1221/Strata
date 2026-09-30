# Handoff: delta restore performance — parallel read (A) and streaming restore (B)

**Purpose:** the delta tier's promote is ~14k tok/s on a 142k-token conversation; the v3 tier's equivalent
restore ran ~50k tok/s. This document specifies the two work items that close the gap, in order, with gates.
A lands first and is reused by B. **This revision is written by the implementer of the delta tier (Phases 0–6,
all green); the facts below are measured from the live store, not estimated.**

---

## 0. Read-me-first

Before writing any code, read:

1. `include/strata/platform/kv_delta.hpp` and `src/platform/kv_delta.cpp` — the record family, the writer
   (`delta_dump_at`), the reader (`delta_restore`), the store (`KvDeltaStore`).
2. `src/platform/kv_nvme.cpp` — `nvme_restore_image`: the walk → applies → sync → apply → spare row → ring
   refill → sync choreography the delta restore reuses by assembling a v3 image. **This choreography is the
   thing B must reproduce without assembling.**
3. `src/platform/kv_delta_host_test.cpp` — the fixture: the byte-identity oracle, the cross-check, the fault
   matrix, the fuzz. Every gate below extends this file.
4. `tools/nvme_delta_p0_test.sh` — the on-device oracle (bit-exactness via STATE_HASH).

**Hard rules (violating any is a failed build):**

- **R1.** The v3 restore path's bytes and decisions are frozen. B adds ONE new entry point to `kv_nvme.cpp`
  (`nvme_restore_apply`, the apply-pass choreography extracted line-for-line from `nvme_restore_image`'s tail —
  the second permitted extraction, same discipline as `nvme_restore_image` itself: nothing moves, nothing
  changes, only the function boundary).
- **R2.** The failure contract is untouched: every pre-apply refusal is `invalid` (recoverable, zero CUDA
  calls, nothing written); only the apply pass can report `transfer_failed`. On a worker thread this means:
  the WORKERS touch no CUDA state at all — reads, digests and host memcpys only; the CUDA choreography stays
  on the caller's thread.
- **R3.** The byte-identity invariant (§5.2 of the delta handoff) is unchanged: B changes WHERE the restored
  bytes land, never WHAT the restore leaves in the session. The STRATA_STATE_HASH oracle is the gate.
- **R4.** The applies list B builds must be in the v3 walk's ORDER and NAMES, because the fault-injection hook
  (`STRATA_TEST_FAIL_CUDA`) skips segments by name and the host fixture counts them
  (`device_applies = 2 + n_layers × 5`: gdn, ple, then per layer pooled / tail / dead / block_pos, then one
  spare-row re-publish per layer). A streamed restore that reorders or renames breaks the fault matrix.
- **R5.** Stop at the first failing gate. No fix-forward.

**The build/test loop and the server discipline** are the delta handoff's §0/§9: `ninja strata`,
`ctest -R kv_delta_host_test`, the oracles with the server STOPPED, `pkill` patterns never matching their own
block, and the live server at :8080 runs `--kv-delta 1` with `STRATA_WATCHDOG_S=300`.

---

## 1. Where the restore time goes (measured, 142k-token promote ≈ 10 s ≈ 14k tok/s; v3 ≈ 50k)

**Status: A, B1 and B2 are landed on `nvme-kv-cache` (2026-09-30); the table below is the BEFORE picture, kept
for the record.**  The measured before/after pair is under it.

**Before, re-measured on the merged binary (Phase 0's own line, on-device, 2026-09-30; `GEN`-driven
143,495-token conversation, 560 chunks, 2.15 GiB read, server stopped):**

```
strata serve: kv-delta restore timing: manifest 0.5 ms, read+digest 3115.3 ms, assemble 2378.8 ms,
                                                  apply 1887.8 ms, rss peak 50606 MB (entry 46202 MB)
```

≈ 7.4 s wall ≈ 19k tok/s, and the promote's own transient = 50,606 − 46,202 ≈ **4.4 GB** — the §1 cost model
below is confirmed phase-by-phase (read+digest and assemble are the two ~2.5 s blocks A and B attack; the
~1.9 s apply is the inherited v3 choreography, unchanged).

**After B2 (the streaming restore, landed):**

```
strata serve: kv-delta restore timing: manifest 0.5 ms, read+digest+place 2637.7 ms, stage 0.0 ms,
                                                  apply 24.2 ms, rss peak 46597 MB (entry 46257 MB), T 143495, 560 chunks, 2.15 GiB read
```

≈ 2.7 s wall ≈ **53k tok/s** (the v3 tier's ~50k is the bar) - and the `KV` line itself agrees on device:
`KV src=delta resume=143495 promote_ms=2668 promote_bytes=2309000692 staging_bytes=356179248`.

**On the RSS target.**  This document's expected end state guessed "~100 MB (the bounded staging)"; what the
path actually allocates at 143k tokens is **356,179,248 bytes**, and the breakdown is worth recording because
two thirds of it is not the conversation's KV at all:

| term | bytes | what it is |
|---|---|---|
| State record | 120,139,080 (115 MiB) | the tail pages + tail rows + gdn/ple/tails/dead/block_pos, read once |
| pooled-row staging | 219,547,308 (~209 MiB) | `n_layers × qsa_pooled_rows(T) × idx_key_dim × 4` |
| 4 worker buffers | 16,492,860 (~16 MiB) | one chunk payload each, reused |

The process's own transient peak was **~340 MB** (VmHWM 46,597 vs 46,257 at entry) - the assembled path cost
**~4.4 GB**, so this is a 13x reduction and the staging is bounded and flat in the CHUNK COUNT (which is what the
parallel read bought), but it is NOT yet ~100 MB: the pooled-row staging is O(conversation length) because R4
requires ONE pooled apply per layer covering all `qsa_pooled_rows(T)` rows, so every layer's whole row span has to
be resident in host memory at apply time.  Removing THAT would mean per-chunk pooled applies (560 x n_layers
H2D copies instead of n_layers), which changes the applies list's shape and count and therefore needs the
owner's decision, not a quiet implementation change - it is the natural next item after this work.

**Bit-exact against the v3 tier at this scale**: the v3 cascade's own snapshot of the same boundary (L=143,495), restored by
`--nvme-restore`, prints the SAME `STATE_HASH` line as the streamed delta promote - all nine fields, including
`stale` and the `ple_prev` window (`docs/nvme-kv-cache-design.md` §5.2).

| step | v3 restore | delta restore (today) |
|---|---|---|
| read | one sequential 2.2 GB file | 557 separate 4 MB chunk files (open/read/close + digest each) |
| integrity | one FNV pass (~2.2 GB) | FNV per chunk (~2.2 GB) **+** FNV over the assembled image again (~2.3 GB) |
| staging | read buffer (2.2 GB) | chunk payloads (2.2 GB) **+** the assembly memcpy (2.3 GB) |
| apply | memcpy into the pools + the H2D copies | identical (inherited) |
| RSS peak | ~2.2 GB | ~4.6 GB |

Three compounding costs: an extra full copy of the conversation, an extra full hash pass (FNV-1a is a scalar
~1 GB/s byte loop — 4.5 GB of hashing ≈ 2–4 s), and scattered small-file reads. The RSS spike to ~4.6 GB is
what a monitor reports as "OOM"; it is the C10-style staging bound the maintainer's "bounded read staging"
wants gone.

**Instrumentation comes FIRST** (Phase 0): `STRATA_DELTA_RESTORE_TIMING=1` makes `delta_restore` print one
line per phase — `manifest scan ms / chunk read+digest ms / assemble ms / apply ms / rss peak MB` — so A and
B are measured, not felt. Without this, do not start A.

---

## 2. Phase A — parallel read + verify + assemble

**Structure.** The restore keeps today's shape (validate → read chunks → assemble → `nvme_restore_image`) but
the middle stages fan out over a small worker pool:

1. **Validate everything, no CUDA** (unchanged): the manifest, every chunk's key/range/size against the
   manifest's records and the live slice math.
2. **The worker fan-out**: N workers (N = 4, `std::thread` + an atomic index + a join barrier — no new
   dependency) each take the next chunk index: one `open`/`read` of the 4 MB payload, the FNV digest check,
   and the **assembly memcpys into the final image buffer** — every slice's destination offset comes from the
   same pure walker the writer uses (§5.4–§5.6), the workers' writes are disjoint, and the digests are
   independent. The chunk payload buffer is then REUSED for the next chunk (the worker keeps one 4 MB buffer,
   not one per chunk).
3. **The assembled image's footer** is written by the main thread after the join (it covers the whole payload
   and `nvme_restore_image` re-verifies it — unchanged).
4. **Failure**: any worker's refusal (digest, size, short read) aborts the restore as `invalid`; the error
   names the chunk and the reason (the same strings `delta_read_chunk` produces — the workers call the same
   `delta_read_chunk`-class helper, not a copy).
5. **No CUDA in the workers** (R2): the memcpys are host-to-host; `nvme_restore_image` runs exactly as today,
   on the main thread, after the join.

**Why A is worth landing even though B supersedes the assembly**: A's worker pool, its per-chunk
read+digest helper, and its failure plumbing are EXACTLY what B's streaming path needs — B replaces A's
"memcpy into the assembly buffer" with "memcpy into the destination arrays" and deletes the assembly stage.
A is B's first half, landed and gated on its own.

**Host-test cases (Phase A):**
- the existing cross-checks stay green (the restore leaves the session identical to the v3 restore's);
- a torn/corrupt chunk under parallelism still refuses with the chunk NAMED and zero CUDA calls
  (the refusals must be deterministic despite the thread pool: the FIRST failure in CHUNK ORDER wins —
  collect errors, join, report the lowest index's);
- a timing assertion is NOT a gate (machines vary) — the TIMING line is evidence, not an oracle.

**Gate A:** `ninja strata && ctest -R kv_delta_host_test` green; the delta p0 unchanged-green; the timing line
shows the read+assemble phase reduced (evidence for the record); one commit.

---

## 3. Phase B — streaming restore: the chunks land in their destinations

**The insight that makes B safe**: the writer sliced every chunk payload FROM the session's arrays as exact
contiguous ranges (§5.4–§5.6). The restore can put them back where they came from — no assembly buffer, no
second hash pass, and the RSS drops from ~4.6 GB to the per-chunk working set (~4 MB) plus the staged
device-segment bytes (~20 MB total). **This is also the maintainer's "bounded read staging" requirement,
satisfied properly.**

**The destination map** (which bytes go where, per the writer's own cut):

- **HOST slices, memcpy'd directly as each chunk is read** (no staging, no apply):
  - each chunk's KV pages → `st.host.*` at page offset `(a/page) × n_head_kv × page × w` — page-aligned by
    construction;
  - each chunk's MTP drafter pages → the drafter's host arrays, the same arithmetic;
  - the STATE record's KV tail pages and the drafter's tail pages → the same arrays at `[S/page, …)`.
- **DEVICE segments, staged small, applied in the v3 order** (the applies list):
  - `gdn`, `ple` (from the State record);
  - per QSA layer: the pooled rows — the chunks' rows `[0, S/idx_block)` **concatenated in chunk order** with
    the State's tail rows `[S/idx_block, qsa_pooled_rows(T))`, so each layer gets ONE pooled apply, exactly
    the v3 walk's shape; the `tails`, `dead`, `block_pos` slices (from the checkpoint blobs in the State).
- The staged device-source bytes total ~20 MB for a 142k conversation (the pooled rows are 512 B/row/block;
  gdn/ple are the big ones at the model's sizes) — this is the restore's ENTIRE transient allocation besides
  the per-chunk read buffer.

**The two layout traps the grouping already taught us** (both bit the K=1-masked code once):

1. The CHUNK's payload layout and the STATE's payload layout have DIFFERENT internal strides (a chunk's
   drafter section strides by `span/page` pages per array; the State's drafter tail strides by `st_pages`).
   The streaming path reads each section with ITS OWN offsets — the same separate-offset discipline the
   assembly's drafter loop now uses.
2. The chunk's pooled-row count scales with the span (`span/idx_block` rows), not with BLOCK. The streaming
   path sizes every pooled slice from the span.

**The apply pass** (R1's one new extraction): `nvme_restore_apply` in `kv_nvme.cpp` — the tail of
`nvme_restore_image` from "the apply pass begins with a sync" through the final sync, line-for-line, taking
the applies list and the layout facts as parameters. `nvme_restore_image` calls it unchanged; the delta
streaming path calls it with its own applies list. The STRATA_TEST_FAIL_CUDA hook moves WITH the code (the
fault matrix keeps its names and counts — the fixture asserts `device_applies` after B exactly as before).

**What B does NOT do**: the assembled v3 image no longer exists, so `nvme_restore_image`'s own layout walk and
digest verdict cannot run — **the streaming path's validation must be a strict superset**: the manifest's
header/count/lattice checks (unchanged), every chunk's existence/size/key/range/digest (unchanged), the slice
math against the live arrays (unchanged), and the sizes of the State's sections against
`delta_state_payload_bytes` (unchanged). Anything that would have failed the image walk must fail here, as
`invalid`, before the first CUDA call. The diagnostic-ordering rule (layout facts before digest verdicts)
holds per chunk.

**The STRATA_STATE_HASH gate is the correctness oracle**: a streamed restore must leave the session
byte-identical to the assembled restore's — the host fixture's existing cross-check
(`nvme_restore(file) == delta_restore(store)`, arrays compared byte-for-byte) is the whole proof.

**Host-test cases (Phase B):**
- the cross-check stays green with streaming ON (arrays, running state, drafter copy, PLE window);
- the restore's RSS: assert the staging allocation is bounded (the fixture can count the bytes the path
  allocates, or `/proc/self/status` VmPeak delta — a stated bound, not a wall-clock guess);
- every Phase-3 refusal case re-run under streaming: missing chunk, truncated chunk, corrupt manifest footer,
  wrong weights, zero CUDA calls, nothing written;
- the fault matrix: `STRATA_TEST_FAIL_CUDA` skips the streamed applies by the SAME names/counts
  (`device_applies` asserted);
- `delta_read_manifest`'s mixed-K store: the streaming path reads BOTH the legacy manifests
  (`blocks_per_chunk` 0 → K=1) and the grouped ones (K=64) — the live store holds both, and nothing converts.

**Gate B:** the fixtures green; `tools/nvme_delta_p0_test.sh` green on device (server stopped, §9); the
timing line shows the restore at or better than the v3 tier's; one commit.

---

## 4. Budget and order of work

| phase | lines (honest) | gate |
|---|---|---|
| 0 — the timing instrumentation | ~60 | the timing line exists and is printed under the env flag |
| A — the parallel fan-out | ~250 + the fixture | the cross-checks + the deterministic refusals; the timing evidence |
| B1 — the `nvme_restore_apply` extraction (line-for-line) | ~120 moved | the existing fixtures green UNCHANGED (the extraction must not move a byte) |
| B2 — the streaming path + the applies builder | ~350 + the fixture | the cross-checks, the refusals, the fault matrix, the RSS bound |
| B3 — the device oracle re-run | 0 | `nvme_delta_p0_test.sh` ALL PASS; the timing evidence |

A is B's first half on purpose: land A, gate it, then B2 swaps A's "memcpy into the assembly" for "memcpy
into the destination" and deletes the assembly stage. If B2 grows past ~2× its budget, stop and re-read the
destination map above — the drift is in the layout offsets, not the design.

**Expected end state**: the promote's wall time for a 142k conversation ≈ the v3 tier's (~3 s) or better, the
restore's RSS ≈ 100 MB (the bounded staging), and the maintainer's "bounded read staging" checked off.

---

## 5. Repo / machine state (for this work)

- Branch `nvme-kv-cache`, head `f665d14` + the sweep-accounting fix `6648be7` — pushed (`maedoc-https`).
- The live server runs `--kv-delta 1` with `STRATA_WATCHDOG_S=300`; the store holds mixed manifests
  (`blocks_per_chunk` 0 and 64) and ~67k legacy 61 KB chunks alongside the 4 MB grouped ones — the streaming
  path must read both, and the legacy chunks age out via supersede + sweep (never convert).
- The fdatasync-per-record and the 256-token grouping are the two ops fixes behind the current write profile
  (the per-turn write is ~113 MB State + new-token chunks; the fsync storm is gone — do not reintroduce
  per-record `fsync` in any read-side change).
- The stall trap (`/tmp/stall-trap.sh`) watches for host-loop freezes; the §9 server procedure applies to
  every device run.
- Untracked, this task's neighbors: `PR-COMMENT-delta-tier.md` (the maintainer-status comment, the user
  posted it), the profile-blend files (the owner's call, parked).

## 6. First actions

1. Phase 0 (the timing instrumentation) — measure today's phases, commit the numbers into this document's
   §1 table as "before".
2. A as written — entirely host-side.
3. B1 (the extraction) with the fixtures green UNCHANGED — that is the proof the extraction moved nothing.
4. B2/B3 with the §9 discipline; hand back to the owner with the timing evidence and the two files' hashes
   if the STATE_HASH oracle disagrees — that is a design bug, not an implementation bug.
