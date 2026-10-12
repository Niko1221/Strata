# Certification and proof: fail-closed release standard

Robustness is an evidence claim, not a percentage of green lines. `scripts/verify.py` enforces cumulative gate tests and requires per-gate evidence profiles for G1–G5. Each new gate must keep all earlier tests passing.

## Acceptance gates

**G0 — reference semantics.** Toy attention example; immutable snapshot DAG; operation-id retry and ABA prevention; corruption rejection; process-concurrent CAS; failure injection around atomic rename; numerical comparator catches perturbations. Entirely model-free. Execute the included reference tests. G0 is **not** GPU proof.

**G1 — real model.** Fresh, uncached oracle; pinned checkpoint and tokens; explicit cache-origin/position matrix; frontier logits before and after every appended token; known distinct suffixes; mid-generation RNG+grammar restore; chunk/scheduling invariant; cache identity hazards; unsupported model state rejected. All results must be reproducible.

**G2 — persistence & recovery.** Actual model state complete; dispose processes; rebuild after full cache eviction; optionally offload/checkpoint with checksums; corrupt/partial checkpoint safely rejected; rename/receipt crash points; mid-decode kill, OOM, stale CAS/retries; validated histories survive. Process kill ≠ simulated sudden power loss—distinguish them in evidence.

**G3 — efficiency & resource safety.** Demonstrate physical shared pages or equivalent bounded overhead under branches, including incomplete tail pages. Instrument prefix work, unique KV bytes, COW count, checkpoint load work, and latency. No stale aliases, leaks, UAF, race-induced corruption or unbounded growth under retention limits. Eviction correctness remains tested.

**G4 — evaluated exploration.** Fork three futures; reproducible metric evaluation in isolation (or explicit declared stochastic evaluator state); no evaluator writes to history, tools, or sibling leases; CAS adopt winner; race/adoption retry; reproduce selected future; audit trails.

**G5 — release qualification.** Full replay across supported hardware/backends/cache types; randomized state-machine sequences, fault injection, sanitizer/toolchain diagnostics where available, long-context and multi-user soak, access checks, sensitive checkpoint storage, retention/GC, upgrades and rollback; report unsupported combinations rather than silently lowering guarantees.

## Mandatory negative testing

At minimum, show that each invariant has an actual **mutant that fails** an independent test:

1. B inherits A's suffix (branch contamination).
2. Sampled last token unconsumed (frontier off by one).
3. RNG draw pointer resets when Generate is chunked.
4. Same seed and token count imply same KV even when earlier prefix differs.
5. Shared partial page modified by a sibling.
6. Old CAS request succeeds after branch moved P→A→P (ABA).
7. Retry replays branch update twice after lost acknowledgement.
8. Corrupt checkpoint accepted without fallback or explicit refusal.
9. Offload/cache eviction deletes retained authoritative history.
10. Evaluator mutates candidate/controller or affects sibling order.
11. Different model/RoPE/adapter state reuses stale checkpoint.
12. Unexpected backend error publishes a half-consumed snapshot.

Mutation testing can use injectable adapters or deliberately broken fixtures; never ship production code toggles that make it easy to enable incorrect behavior in a real run.

## Failure-handling rules

- Missing real backend, independent oracle, required state-export primitive, instrumentation, exact required test, or evidence profile → **BLOCKED**, not PASS.
- Unknown token/position semantics, truncated/missing historical artifact, incompatible execution profile → explicit structured error; no implicit compatibility mode.
- Checkpoint corruption → reject checkpoint, restore by replay only when the canonical retained input history and all required resources are present. Never trust corrupt numbers.
- Branch move or journal uncertainty → query/retry operation receipt, not blind second move.
- Offloaded KV page missing → replay or report unavailable, but never remove history or substitute a different token set.
- Any changed test threshold or oracle implementation → rebaseline independently and require versioned review.

## Test report required for each run

`evidence/runs/<date>-<gate>.json`:

```
{
  "gate": "G1",
  "status": "PASS|FAIL|BLOCKED",
  "source_tree_sha256": "...",
  "model_sha256": "...",
  "backend_revision": "...",
  "execution_profile_sha256": "...",
  "fixtures_sha256": "...",
  "oracle_implementation": "separately named true uncached path",
  "numerics": {"atol": 0, "rtol": 0, "observed_max_error": 0},
  "test_count": 0,
  "skipped": 0,
  "expected_failures": 0,
  "failed": 0,
  "counters": {},
  "commands": []
}
```

Example **schema only**, not a claim of execution. For G3 record memory allocations, reused prefix tokens, actual KV pages, partial-tail copies, transfer bytes and timing distributions rather than one successful output. Model evidence must use actual measured values, not placeholders.

## Independence standard

A test is not independent when it compares two calls through the same cached path, relies on the same serializer for "fresh" and "restored" state, compares only final text, or merely verifies a snapshot hash of its own output.

A correct test should catch a real error deliberately injected into (or emulated at) the SUT. The independent full-prefill oracle must compute every expected prefix/suffix outcome from exact input tokens, without looking up checkpoints produced by the SUT.

## Full coverage does not mean a proof of zero bugs

The goal is to make unsupported claims and broad classes of regressions difficult to hide, while retaining reproducible artifacts. Infinite input and scheduling spaces cannot be exhaustively enumerated. Thus "completely tested" here means **every specified gate/edge-case obligation executed with measurable evidence**, not mathematical proof of bug-freedom.
