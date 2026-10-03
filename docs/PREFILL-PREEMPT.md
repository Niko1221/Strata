# PREFILL-PREEMPT.md - cooperative prefill preemption at chunk boundaries

A long prompt used to own the engine from its first chunk to its last (`Prefill::run`'s chunk loop runs to the
end, and the server's FIFO holds one HTTP request's whole lifetime, decode included): a ~208K-token cold prefill
(~60 s) made every request that arrived behind it wait for all of it. This adds an opt-in scheduler turn: a
prefilling request may **park** at a state-consistent chunk boundary, a queued request runs to completion, and the
parked request is restored and continues as if nothing had happened. The engine still runs ONE sequence at a
time - `A -> park -> B -> restore A` - never `A + B` concurrently, and decode is untouched.

`--prefill-preempt` turns it on (default off). With it off, every code path is the one that shipped before.

## Phase 1 audit - what one prefill chunk leaves behind, and what a boundary must copy

`Prefill::run` processes `[pos0, pos0 + n)` in chunks of `m.T`. At the end of each chunk the serve path's
`on_chunk` runs with the compute stream synchronized (`prefill.cpp`, the `on_chunk || on_stage_chunk` block), so
"chunk c done" is a real point: every kernel that contributes to the state of positions `[pos0, pos0 + T)` has
landed. The audit question is: *if we copy the session's state here, does the copy hold exactly what processing
chunk c + 1 needs?* Everything below is per one completed chunk boundary at absolute position `P`.

**Running state** ("everything so far", cannot be rewound - a checkpoint copies these, ~118 MB):

| state | where | committed at the boundary? |
|---|---|---|
| 36 GDN recurrences + conv histories | `ss.gdn_state` (device) | yes - the chunk's kernels are synced |
| PLE normalized history | `ss.ple_hist` (device) | yes |
| PLE token window `ple_prev` | host ints | **no - see below** |
| each QSA layer's indexer tail | `ss.qsa_states[i].idx_tail` (device) | yes |
| the indexer's spare key `idx_dead` | device | yes, but B rewrites it (below) |
| `ple_token` | host int | yes (4 bytes, copied for completeness) |

**`ple_prev` is the one trap, and it is the bug the audit was for.** `Prefill::run` keeps the window in a local
`prev[2]` and writes `ss.ple_prev` only after the WHOLE loop (`ss.ple_prev[0] = prev[0]` after the `for`). At a
mid-prompt boundary the session's `ple_prev` still names the tokens before the prompt, not before `P`. The
conversation checkpoints never hit this because `checkpoint_restore` re-derives the window from the checkpoint's
own token list (`L >= 2 ? ids[L-2] : -1`). The park path does the same: the snapshot records `done`, and the
restore sets `ple_prev` from the prompt's tokens `[done-2, done)` - which is exactly what an uninterrupted run's
`prev[2]` holds there. (Suspending also runs `run`'s tail, which commits `ss.ple_prev` for the boundary; both
mechanisms agree.)

**`idx_dead` is sequence state, not a constant.** The header says "CONSTANT for the sequence" and the indexer
kernel writes it only at position 0 (`rms_norm` of cell 0's raw key, rotated) - constant per the token AT
position 0, which differs between conversations. B running from 0 rewrites it with B's token-0 key; A's restore
must copy A's. Copied per layer (idx_dim floats).

**Positional state** (cells written once per position; copying it is what makes room for B, because the KV arena
is one branch of history and B's prompt rewrites it from cell 0):

| state | source of truth | park copies |
|---|---|---|
| 12 QSA layers' K/V pools | `kv_mode` 0: the device pools; 1: the pinned host pools (the VRAM slots are a cache; writers keep both current) | blocks `[0, ceil(P/page_size))` of every pool run (K, V, scales - the `runs_of` layout) |
| each layer's residency map | `kv_mode` 1: `KvStreamMap` on device | nothing - the RESTORE resets it (`kv_stream_reset`): B's decoding re-pointed the slots at B's blocks, so every one of A's blocks must miss and re-stream from the restored host copy. Writers keep host current, so the host copy is enough |
| each layer's pooled indexer rows | `idx_pooled` (device) | rows `[0, (P-1)/idx_block + 2)`: one row per completed block plus the spare row (a copy of `dead`) |
| each layer's `idx_block_pos` | device int | copied (debug/reader metadata) |
| the MTP drafter's K/V | a ring (`kv_mode` 2) over its host copy, or a resident pool | the ring's blocks `[b0, b1)` from the host copy (b1 = `ceil(P/page_size)`, b0 = `b1 - n_slots`); restore memcpies them back and refills the ring with `MtpDrafter::kv_restore(done)` - the same call a conversation-cache resume uses |

Not copied, on purpose: the residual stack `ss.block.R` and every verify-window scratch (recomputed per token);
`cos_tab`/`sin_tab` (constants); `step`/`host_step`/`host_pos` staging (rewritten per token); the drafter's
`prompt_len_` (set again per request); the suffix drafter and the draft policy (reset per request / process-wide
tuning); the expert cache, its residency table and the prompt path's borrowed slots (the park refills the lent
slots first, as the cancelled path always did); `checks`/`live` bookkeeping (below).

**Drained before parking** (no async work may outlive the request): the chunk-end `cudaStreamSynchronize(m.cs)`
has run (the `on_chunk` contract); the expert issuer thread is joined (inside the chunk loop); the PLE read-ahead
future for the next chunk is a `std::async` future whose destructor blocks on scope exit - waited, then
discarded (the resume re-gathers; MVP-simple); the copy stream is synchronized in `run`'s tail, which the
suspend break runs; the lent expert-cache slots are refilled (`refill`, same as the cancel path). Only then are
the snapshot copies enqueued, after a `cudaDeviceSynchronize` - the same guard `checkpoint_at` uses.

## Why the park cannot live in the conversation cache's `checks`

A `ConvCheckpoint` copies only the running state and stays valid *because* its positional cells below it still
hold its tokens - the serve loop erases any checkpoint whose tokens stopped being a prefix of what the session
holds. A parked A breaks that premise on purpose: B rewrites the cells. So the park is its own record holding a
FULL copy (running + positional), kept OUT of `checks`; it is dropped when finished, cancelled, or when the
engine dies (a restart starts from nothing, so a client that resumes into a fresh engine gets a clean
`ERR no parked request`, never a foreign state). Exactly one parked request exists at a time (MVP): while the
slot is occupied, no new preemption happens - the running request finishes first. Repeated preemption of the
SAME request still works: `A -> B -> A (1 chunk) -> C -> A ...`, which is the fairness shape the tests walk.

## Resume correctness argument

Restoring puts back every byte the audit lists, for positions `[0, P)`; `cur` becomes A's prompt again; the
remaining read is `sp.run(ids + P, n - P, P)` - the same call shape, chunk shapes and callback path as any other
prompt segment (mount, checkpoints at `prompt_cache_every`, short-read windows for its tail) - and decode is the
untouched single-sequence loop. Deterministic config (greedy, `--adapt-swaps 0`, fixed expert sets) must
therefore produce, after `A -> park -> B -> A`, the same tokens and the same `STRATA_STATE_HASH` fingerprints
(gdn, ple, tails, pooled, kv, mtp, stale, ple_prev) as uninterrupted A - that is the acceptance test
(`tools/prefill_preempt_test.py`), not a review claim.

## Protocol (engine)

One line per request, as before, plus an optional request id as a key - every line the engine emits for an
id-carrying request carries the same suffix, so output can always be demultiplexed:

    GEN <max_new> [k=v ...] <ids>          k may be id=N
    GENI <max_new> <file> [k=v ...] <ids>
      -> RESUME <reused>[ id=N]            (before reading, as before)
      -> PP <done> <total> <ms> <tok/s>[ id=N]
      -> REUSED <reused>[ id=N]
      -> SUSPENDED <done> <total>[ id=N]   (parked: state valid engine-side, prompt NOT fully read)
      -> T <tok>[ id=N] ...
      -> DONE <generated> <total> <pms> <dms> <finish> <acc> <off> <reused> <hits> <look>[ id=N]
      -> ERR <msg>[ id=N]
    RESUME[ id=N]                          (client: continue the parked request)
    CANCEL[ id=N]                          (client: drop the parked request; DONE ... cancel)
    STOP / QUIT                            (as before: STOP ends the RUNNING request; a parked one takes CANCEL)

Requests without `id=` behave exactly as before (no suffixes, no suspension - the engine only arms preemption
for id-carrying text requests), so an old server drives a new engine and the other way round.

## Policy (v1, conservative)

Suspend only when: the feature is on (`--prefill-preempt`); the running request is a text request that carried
`id=`; the engine is single-GPU (a layer split's stages are chunks apart - explicitly unsupported, fail closed);
the request is in the batched prompt path; a further GEN/GENI line is queued on stdin; the prompt position
reached is >= `--prefill-preempt-min-tokens` (default 8192: never park for a tiny prompt, where the snapshot
costs more than the wait); the park slot is free (or holds this same request being re-parked); and the request
has been preempted fewer than `--prefill-preempt-max` times (default 16). Not less than one full chunk runs
between suspensions by construction (the check is at chunk ends). Decode is never preempted. Snapshot admission
precedes any state change: the snapshot is built entirely from copies, so a failed allocation just clears the
yield request and A continues - the session is never destroyed to discover the copy did not fit.

## Server side

`python -m serve.server --prefill-preempt` splits the whole-request FIFO into ownership periods: the HTTP thread
holds the lock from `GEN` to `SUSPENDED`, releases it for the queued request, re-acquires and sends
`RESUME id=N`. Only the FIFO holder ever talks to the engine (the invariant the single pipe relies on), and the
id suffix demultiplexes the shared line queue. Cancellation of a parked request sends `CANCEL`; engine death
while parked fails the request cleanly on resume (the restarted engine has no such id). `/metrics` gains
`prefill_preemptions`, `prefill_resumes`, `prefill_preempt_snapshot_ms`, `prefill_preempt_restore_ms` and
`max_queue_wait_ms`; parked requests show `state: parked` in `/status`.

## Tests

* Group A (no GPU): the scheduler state machine over a fake engine - no contention, preempt on queue, FIFO
  order, cancel queued/parked, engine death while parked, shutdown with one parked (`serve/test_server.py`).
* Groups B/C/D/E (GPU, real model): `tools/prefill_preempt_test.py` - A/B/A output parity (greedy, 128 tokens)
  and `STRATA_STATE_HASH` equality across suspend boundaries (early/middle/penultimate/final partial chunk),
  with and without KV streaming (`--kv-resident 0` vs a context past the resident window), repeated
  suspend/restore cycles, and cancellation of the parked request followed by a healthy request.
* Groups F/G/J/L ride the same engine harness (conversation cache on and off via `--prompt-cache`; the snapshot
  is admission-gated, so a failed park never destroys A); group M (layer split) is the fail-closed check
  (`--prefill-preempt` refuses multi-GPU at startup).
