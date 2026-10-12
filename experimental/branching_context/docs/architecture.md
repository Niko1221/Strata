# Architecture and proof obligations

Read `spec/branching-context-api.md` first. This file translates the theory into boundaries Codex must implement and test. Where implementation details differ, the specification wins unless changed in a recorded, versioned design decision.

## Only three state categories

1. **Snapshot** `S=(f,parent,delta,D,boundaries)`: immutable, content-addressed, retained input history and complete decoder/controller state. Canonical history is `H(S)=H(parent)||delta`. A controller-only snapshot has empty `delta`.
2. **Branch** `B=(name,root,revision,tip)`: a movable named reference, changed only via transactional CAS. Forking a snapshot creates a branch with the *same* tip, including RNG position; no magic new seed.
3. **Checkpoint** `J=(snapshot-or-prefix-identity,e,state)`: optional materialization of model computation. It may be evicted, offloaded, regenerated, or converted only through a separately proven operation. It is never authoritative history.

An **evaluation** is an independent immutable result attached to a snapshot ID and evaluator contract, not a fourth mutable conversational state. An **execution lease** is a temporary private workspace, not a new logical snapshot.

## Laws to prove

- `H(fork(S))=H(S)`, `D(fork(S))=D(S)`.
- `Continue(Restore(S_P),A) ≈ Fresh(P||A)` for *every* tested suffix, including empty.
- `Continue(Restore(S_P),B) ≈ Fresh(P||B)` irrespective of A's creation, processing, eviction or evaluation.
- `rewind` changes only the branch's pointer and monotonically increases its revision.
- `Generate(Generate(S,m),n) ≡ Generate(S,m+n)` when the same generation scope is active and stops are handled identically.
- `EvictAllNumericalState(S)` leaves `H(S)` and `D(S)` unchanged; cold replay reproduces the same model/decoder semantics.
- `Evaluate(S)` leaves `S` and every branch unchanged. CAS adoption moves exactly one authorized pointer, never rewrites any snapshot.

These are relational correctness properties, not unit-output examples. Every one gets both a positive and an adversarial/negative case.

## Canonical boundary

Published `S_t` means every token in `H(S_t)` has **already been consumed**. The next-logits frontier `ell_t` predicts the token after `x_t`. Internally sampled-but-unconsumed tokens are normalized before committing. No side channel may carry a hidden token across a snapshot boundary.

A complete model state may require KV pages, positions/masks, rotary configuration, window state, frontier logits, recurrent/convolutional state, memory-compression state, and adapter-specific metadata. A checkpoint needs *everything* required for subsequent computation. The decoder state independently needs policy, grammar progress, stop progress, RNG algorithm/state/draw count, repetition history, output-decoder bytes and generation status.

## API ownership

```
Source of truth (file store / immutable records):
  create_root(...) -> SnapshotID
  append(parent, input_delta, next_decoder_state, boundaries) -> SnapshotID
  fork(snapshot_id, name) -> BranchHead
  head(name) -> BranchHead
  move(name, expected_revision, new_tip, operation_id) -> BranchHead
  evaluate(snapshot_id, evaluator_contract) -> EvaluationID

Rebuildable backend/materialization:
  materialize(snapshot_id, execution_profile) -> StateLease
  next_logits(lease) -> logits
  ingest_exact(lease, token_ids) -> StateLease
  export_checkpoint(lease) -> CheckpointID
  evict(checkpoint_id) -> None
  stats(lease) -> counters

Sampler/controller:
  start_generation(D, policy, stream_id) -> D'
  choose(logits,D) -> (token,D')
  stop_or_continue(D',token) -> D''
```

The reference `src/context_time_travel/store.py` implements only immutable object storage and CAS. Its `append()` accepts the full next decoder state supplied by a caller—**it does not prove the state is correct**. Codex must implement a verified controller before claiming real generation. The reference has no message-boundary schema, retention/GC, evaluation engine, access control, or numerical checkpoint implementation. These are intentional *unimplemented* capabilities, not passing gates.

## Exactly where mutable data can exist

- Inside an uncommitted execution lease: yes, owned by that worker. Changes may be discarded on error.
- Inside a published snapshot or shared checkpoint: **never**. A lease must COW a tail page before a sibling's writes can touch it.
- In the branch reference index: only within a transactional, revision-checked update; retry receipt and move are one commit.
- In a checkpoint cache: derived index/eviction state may change, but neither source tokens nor immutable logical snapshots may change.

Any backend that cannot expose immutable shared pages can be wrapped with deep copy/replay. That can pass G1/G2, but cannot claim efficient G3 hot branching.

## Logical and physical implementation are separate measurements

A correct branch may still be slow. Track independent counters:

```
prefix_tokens_prefilled       # work re-executed; not guessed from elapsed time
shared_prefix_pages          # actual physical pages shared
private_tail_pages           # allocations and copies caused by COW
kv_bytes_unique              # physical bytes, deduplicated
checkpoint_load_bytes        # copy/IO cost
replay_tokens                # exact consumed tokens reconstructed
```

If an engine cannot report these, add instrumentation or declare G3 blocked. Latency only is not evidence of sharing; a warm allocator or fast GPU may mask full-prefix copying.

## Crash and retry model

A root/snapshot is written durably *before* any branch pointer may reference it. On POSIX local filesystems: write temporary immutable object, fsync it, publish atomically, fsync directory; branch index + receipt + audit event as one atomic transaction. A crash after commit but before reply creates an unknown client outcome; retry the same `operation_id` to recover the prior result.

Atomic filesystem **rename** does not itself prove power-loss durability. Test real process kills at multiple injection points; record filesystem type and whether the test modeled process death or true power loss. Do not imply POSIX filesystem guarantees apply to NFS, object stores, or Windows.

## Performance expectation

For a saved prefix of `n` tokens and branches `i=1..k` with suffix lengths `m_i`, ideal unique cache work/storage resembles `n + sum(m_i)` token-equivalents, rather than `k*n + sum(m_i)`. This is an aspiration conditional on full-prefix sharing and checkpoint retention, not a guarantee. A partial last page can require a bounded copy. Shared immutable prefix state may have head-specific or recurrent overhead.

## Rollover and unsupported state

If no model state checkpoint survives, reconstruct from the canonical inputs + controller. If a model's state cannot be reconstructed (e.g. lost multimodal artifact or required weights), say **not resumable** with precise cause. Never pretend the old KV is history.

Exactly different models, adapter stacks, RoPE/YaRN semantics, or model contracts must not share numerical checkpoints even when tensor dimensions line up. Matching output text is insufficient compatibility evidence.
