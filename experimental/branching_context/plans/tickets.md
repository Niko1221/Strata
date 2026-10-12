# Codex task tickets — cumulative, test-first

The user asked for a *completely tested and robust* branching-context API, not another one-off cache demo. Treat each gate as mandatory certification, with explicit evidence. Suggested increments below are small enough to be independently reviewed. See `docs/test-matrix.md` for **85 individually named mandatory real-backend tests**.

## G0: Reference and oracle plumbing (included reference tests; still review)

- [ ] **T00** Read the full mathematical spec and confirm the exact meaning of model state, decoder state, physical cache state, and frontier logits. Write down ambiguities; never assume a GGUF session save covers all fields.
- [ ] **T01** Run all 51 provided model-free tests (9 original toy, 32 logical store, 10 numerical comparator). Verify they are not skipped. Save result in `evidence/runs`.
- [ ] **T02** Implement a deliberate mutant campaign: wrong prefix, unconsumed last token, reset RNG, ABA CAS, bad hash, shared-page tail writes. Show tests go red without lowering tolerances.
- [ ] **T03** Record the actual hardware/backend to integrate and pin exact executable and model hashes. If no backend selected, keep status G1 BLOCKED.

## G1: One truthful real-model path

- [ ] **T10** Load one pinned test model and freeze the exact input token fixtures; include short and long prefixes, distinct suffixes and position/cache-boundary fixtures.
- [ ] **T11** Write independent full-history/no-cache oracle tests (`G1-001..G1-005`) *before* any branch adapter. Verify uncached path cannot read SUT KV state.
- [ ] **T12** Implement adapter materialization/ingest/frontier with explicit consumed count and positions. Write tests `G1-006..G1-014`.
- [ ] **T13** Implement fully serialized decoder state and sampler semantics (PRNG cursor, grammar, EOS, penalties, output decoding). Write tests `G1-015..G1-022`.
- [ ] **T14** Implement semantic and execution profile checks with hard failures, pinned metadata and frozen numerical tolerances (`G1-023..G1-027`).
- [ ] **T15** Run entire G0+G1 suite on actual checkpoint and save all result/fixture/logit fingerprints in `evidence/profiles/G1.json`.

## G2: Recovery without fake time travel

- [ ] **T20** Implement optional checkpoint serialization/materialization as a rebuildable accelerator; allow cold reconstruction from exact histories. Test G2-001..005.
- [ ] **T21** Implement proper before/after publication fsync and atomic transactional branch update + receipt. Add process-crash injection (G2-006..011).
- [ ] **T22** Implement abort-safe leases and resource/oom cleanup (G2-012..014).
- [ ] **T23** Validate compatibility, random/grammar resume, input availability, and external tool side-effect boundary (G2-015..019).
- [ ] **T24** Repeat G0+G1+G2 under restart and corrupted checkpoint fixtures; save actual recovery evidence.

## G3: Honest hot prefix reuse

- [ ] **T30** Instrument real page/refcount ownership, computation counters and deduplicated KV bytes (G3-001..002 and G3-015).
- [ ] **T31** Prove COW isolation for all partial/full page boundaries, quant metadata and hybrid state if supported (G3-003..006).
- [ ] **T32** Measure eviction, offload, memory pressure and bounded copy/reconstruction work; no unsupported numeric conversion (G3-007..010 and G3-014).
- [ ] **T33** Validate fork and resume latency, leak-free teardown, randomized concurrent interleaving and longest supported contexts (G3-011..013 and G3-016).
- [ ] **T34** Keep G3 BLOCKED if physical counters or real sharing are unavailable; never substitute a fast full prefill and call it zero-copy.

## G4: Branch-and-evaluate demonstration

- [ ] **T40** Implement isolated evaluation record with immutable provenance: evaluator code/version, exact snapshot ID, fixtures, metric, seeds and side-effect policy.
- [ ] **T41** End-to-end: P → A/B/C; rewind A → D; evaluate A/B/C/D; adopt winner by revision CAS; evict and restore winner; verify all four fresh histories (G4-001..011).
- [ ] **T42** Include stale adoption, evaluator error, and re-run checks; prove no candidates or RNG streams changed.

## G5: Release stress/hardening

- [ ] **T50** Publish compatibility and unsupported-feature matrix. Test every advertised combination (G5-001).
- [ ] **T51** Run >=1000 reproducible randomized state-machine programs with independent reference outcomes and failure shrinking (G5-002).
- [ ] **T52** Test upgrading adapters/checkpoint schema and cold replay; long soak with memory/leak/latency histograms; sanitizers for native code (G5-003..005).
- [ ] **T53** Implement/test multi-user authorization, history/secret protection, reachability-based retention/GC under concurrent leases (G5-006..009).
- [ ] **T54** Test ambiguous commit recovery, complete mutant campaign and rollback (G5-010..012).
- [ ] **T55** Produce one final release evidence ledger showing every gate PASS, 0 missing/skipped/xfail, model/profile matrix, numerical error maxima, resource counters, known non-goals. Do not claim production confidence until this exists.

## First real-model demonstration acceptance criteria

```
P = exact tokenized prefix, length chosen around a real KV page boundary
S = Create(P)
A, B, C = Fork(S, distinct branch names)
Advance(A, [known explicit suffix a]); assert logits == Fresh(P||a)
Advance(B, [known explicit suffix b]); assert logits == Fresh(P||b)
Advance(C, [known explicit suffix c]); assert logits == Fresh(P||c)
Rewind(A, S); Advance(A, [known explicit suffix d]); assert logits == Fresh(P||d)
assert old A is still reachable and unaffected
Evict(all numerical KV); cold Restore(B); assert logits == Fresh(P||b)
Evaluate(each); assert unchanged snapshots and RNG states
Move(winner branch via revision CAS); assert exactly one atomic update + receipt
Check counters: hot forks did not redo P, shared pages remain immutable
```

Every `==` on real logits in this pseudocode means numerical equality **within previously fixed tolerances**, while history IDs/inputs, controller bytes, CAS behavior, and compatible deterministic sampled-token traces may be exactly checked.

## What is not sufficient

- Nine toy tests alone; **they never exercise a transformer**.
- A static prefix-cache hit demonstration with no rewind, RNG restoration or independent reference.
- A checkpoint file that omits sampled-but-unconsumed last token, next logits or grammar state.
- Tests that compare two branched runs through the same broken KV path.
- All-green screenshots without machine-readable run metadata, real model hashes and reproducible fixtures.
- Rewriting tests/tolerances to conform to a broken backend.
