# L2 Session Persistence

## Overview

L2 session persistence extends Strata's conversation cache beyond host RAM.
When a long conversation no longer fits in the PARKED RAM cache (L1), it can
be saved to disk (L2) as a compressed session file and restored later —
including after an engine restart.

This document describes the design and internal architecture. For setup and
usage, see [L2 Guide](l2-guide.md).

## Problem

OpenAI-compatible agent clients resend the full message history every request.
When multiple long conversations alternate:

1. The RAM cache evicts older conversations to make room.
2. A returning conversation not in RAM requires full prefill.
3. Existing session files write and read **whole snapshots**, amplifying I/O
   when only a small part changed or only a prefix is needed.

## Cache levels

```
L0 / Active     — current request's state in the engine (GPU + host)
L1 / PARKED     — inactive conversations in bounded host RAM
L2 / Persistent — session files on disk (this feature)
```

The engine owns all KV state and transfers at every level. An external
controller (see `l2/reference_controller.py`) only decides *when* to save
and restore; it never touches KV bytes.

## v2 file format

Session format v2 stores data in independently compressed blocks organized
into segments:

```
Header:    magic, version, model/config fingerprints, payload size
Segments:  counts, live fields, checkpoint fields, K/V parts
Directory: segment records, block offsets/sizes/modes/hashes
Footer:    format marker + directory offset
```

Properties:

- Each block is LZ4-compressed (or stored if incompressible) and hashed.
- The counts segment provides all sizes before allocation, enabling
  bounded reads on untrusted or corrupted files.
- Checkpoint selection retains the deepest point, pinned points, head
  anchors, and depth-fill points — enabling prefix restore at multiple
  depths within one file.
- Format v1 files remain readable; v2 writes require explicit opt-in
  (`STRATA_SESSION_V2=1`).

## SAVEINC: incremental save

Instead of rewriting the whole file, SAVEINC appends only the changed data:

1. **Peek** the old file's live token count/hash, checkpoint hashes, and
   per-part byte watermarks.
2. **Validate** that the new live state exactly extends the old one.
3. **Plan** which checkpoints to reuse (by hash match) vs. capture fresh.
4. **Append** the counts, live tail, fresh checkpoints, K/V part tails,
   and a new directory/footer.
5. **Commit** after flush + structural verification.

Any mismatch (identity, divergence, shrink, budget) declines the incremental
save and falls back to a full save.

**Limitation:** the current in-place append is not power-loss atomic.
A crash mid-append can leave the file unreadable. Recovery needs a journal,
dual footer, or copy-on-write design (see Known Limitations below).

## Restore modes

### RESTOREP: full parked restore

Streams a v2 file directly into a PARKED entry:

- Estimates the image cost from counts and checks the pool budget.
- Reads, decompresses, and verifies blocks into the snapshot.
- Does NOT replace active state or perform GPU transfer.
- The next ordinary request uses the existing `best/take/borrow` path.

### RESTOREPP / preload: checkpoint-prefix restore

With the incoming request's prompt token IDs:

1. Reads checkpoint ID hashes from the directory.
2. Selects the deepest checkpoint exactly matching a prompt prefix.
3. Reads only that checkpoint's state and K/V prefix.
4. Publishes it as a PARKED entry for the normal request to borrow.

Read cost is proportional to the borrowed prefix, not the file size.

### PROBE: read-only availability query

Returns how many tokens the cache (live + PARKED) would give back for
this prompt, without running the request or modifying any state.

## Asynchronous writer

The writer pool moves SSD append/flush work off the inference FIFO:

```
Request thread:  capture → deep-copy → enqueue → return
Writer thread:   append → flush → verify → publish generation → notify
```

- Bounded by a byte budget; over-limit falls back to synchronous.
- Per-file monotonically increasing generation numbers.
- Commit tracking: pending → committed/failed, queryable via `/slots`.
- Alias roll-forward and subsequent restores wait for the relevant
  generation to commit.

**Limitation:** memory accounting approximates metadata as 64 MiB;
fresh checkpoints may exceed this.

## PARKED control plane

The engine exposes a minimal control API over the existing conversation cache:

```
CACHE STATS      — budget, slots, entries, bytes, evictions, pinned, guarded
CACHE LIST       — per-entry: serial, tokens, bytes, checkpoints, pin state
CACHE PIN/UNPIN  — protect an entry from eviction (short-lived, for preload)
CACHE DROP       — remove an entry by serial
CACHE TOUCH      — move an entry to newest position
```

These map to HTTP endpoints:

```
GET  /v1/cache/stats
GET  /v1/cache/list
POST /v1/cache/pin       {"serial": N}
POST /v1/cache/unpin     {"serial": N}
POST /v1/cache/drop      {"serial": N}
POST /v1/cache/touch     {"serial": N}
POST /v1/cache/preload   {"filename": "...", "messages": [...], "pin": true}
```

## Session identity

The engine validates restore correctness through:

- Model and configuration fingerprints (fail-closed on mismatch).
- Exact token prefix matching.
- Image identity and control-vector compatibility.

A reference controller may additionally use message-level hashing
(family/chain) for snapshot selection, but this is advisory — the engine
always performs the authoritative validation.

## Reference controller

`l2/reference_controller.py` demonstrates the full orchestration loop in
~300 lines:

1. Group requests by stable system+tools prefix (family).
2. Identify branches by chained message hashes (chain).
3. On conversation switch → SAVEINC the outgoing session.
4. On returning conversation → preload the snapshot as PARKED.
5. Forward the generation request (engine's take/borrow handles the rest).
6. Track dirty state and update the snapshot index.

This is a demonstration, not a production gateway.

## Benchmark

`l2/bench_rotation.py` measures end-to-end effectiveness:

- Builds three synthetic conversations (two long, one short).
- Rotates between them (tests PARKED reuse).
- Creates a fork that shares a prefix (tests checkpoint borrow).
- Optionally restarts the engine (tests disk restore).

Reports per-request reuse rates, latency percentiles, and failure count.

## Known limitations

1. **Power-loss safety:** In-place SAVEINC is not atomic. A crash mid-append
   can corrupt the previous good file. Recovery design (journal, dual footer,
   or COW) is needed before production use.

2. **POSIX-only I/O:** The file layer uses `open/pread/pwrite/fsync/ftruncate`
   directly. Windows requires a separate implementation.

3. **Default checkpoint tolerance:** `conversation_snapshot_restore()` currently
   enables tolerant checkpoint validation globally. This should be scoped to
   the L2 opt-in path only, preserving upstream fail-closed semantics by default.

4. **Writer memory accounting:** Approximates metadata as 64 MiB. Fresh
   checkpoints may exceed this, causing the queue budget to undercount.

5. **Single-slot only:** Not compatible with `--layer-split`, `--peer-device`,
   or `--batch`.

6. **No concurrent inference:** This extends conversation caching, not
   parallel request execution.

7. **Platform validation:** Tested on Linux/HIP (AMD W7900) only. CUDA,
   SYCL, Metal, and Windows have not been validated.
