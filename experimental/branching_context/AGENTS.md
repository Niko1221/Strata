# Codex operating contract: Context Time Travel

## Mission

Implement a local-first, robust, fully **evidence-tested** branching-context API according to `spec/branching-context-api.md`. Execute all validation gates. No claim of production robustness is permitted without G0–G5 evidence.

## Nonnegotiable invariants

1. **Source of truth:** parent-linked immutable input/history snapshots plus controller state. Checkpoints, GPU buffers and KV pages are rebuildable materializations only.
2. **Correct branch:** from snapshot `P`, branch A with suffix A behaves as `P||A`, and B as `P||B`. Never `P||A||B`.
3. **Public boundary:** all recorded inputs consumed; next logits predict exactly the next token. Pending-sampled-but-unconsumed tokens cannot hide in a published snapshot.
4. **Snapshot immutability:** no operation, including tail-page append, eviction, quantization, loading, evaluation or generation, mutates an existing logical snapshot or shared checkpoint.
5. **Complete controller:** RNG algorithm/state (draw cursor, not only seed), grammar, penalties, stopping, sampler order and pending output decoder state survive snapshot/restore.
6. **Atomic branch move:** CAS on monotonic revision, atomic retry receipt, no ABA stale-write vulnerability. A failed operation cannot partially move a reference.
7. **Eviction:** deleting every numerical checkpoint never deletes an explicitly retained history; restoration may cost more compute.
8. **Compatibility:** reject mismatched model contract, positional semantics, adapters, execution formats, corrupt hashes, incomplete checkpoint, or inaccessible root. No silent fallback that changes function.
9. **Evidence:** a fresh full-prefill oracle is independent of the system under test. Verify logits, controller, physical use/copy counters, durability, integrity and effects separately.
10. **No fake green:** skipped/xfail/unsupported suites do not count as passes; no lowered tolerances, relaxed tests, fake counters or mock engine represented as real inference.

## Design discipline

- Single canonical model of an immutable snapshot, movable branch reference, and optional checkpoint. Do not add aliases or representations unless an actual boundary requires them.
- Favor native filesystem + stdlib first. POSIX advisory lock and fsync for atomic reference transactions where supported; platform guarantees must be measured, not asserted.
- Build smallest backend-neutral control plane. Model-specific state lives behind a checked adapter interface.
- Do not add an HTTP API/UI before demonstrating correctness locally.
- Do not introduce separate "save token" versus "save KV" user-facing states; normalize backend pending-token internals at the adapter boundary.
- If a backend cannot do immutable physical prefix sharing, a deep-copy/rebuild fallback is acceptable for semantic G1/G2 but **not** G3. Report this honestly.
- Verify any claimed upstream method/signature against the installed header and commit, not a search snippet.

## Mandatory workflow

For each ticket in `plans/tickets.md`:

1. State the invariant being enforced and the reference result expected.
2. Add a failing deterministic test including an adversarial counterexample.
3. Run it and record the failure (red).
4. Make the smallest implementation change (green).
5. Run `python3 scripts/verify.py --gate G0`, then all currently completed gates.
6. Run relevant fault injection/mutation test and independent oracle.
7. Record evidence (commands, code hashes, model/backend revisions, seed, results, resource counters, failures) under `evidence/runs/` and update `evidence/STATUS.md`.
8. If test cannot be implemented, stop that gate as BLOCKED instead of inventing stub success.

## First task

Read all of `spec/branching-context-api.md`. Run G0 reference verification. Identify target backend and real state-saving API by direct inspection. Implement the G1 oracle first; only then integrate cache restoration and acceleration. Keep a no-cache, full-history reference path completely independent from the cache path being tested.

## What to report at each gate

- Gate/status PASS, FAIL, or BLOCKED (never "done" based solely on code presence).
- Tests actually executed, test count, failures/skips/xfails.
- Relevant SHA-256 IDs, exact model/backend/build/cache layout and host.
- Worst-case next-logit differences; reproducible bad fixture if failing.
- Prefix work reused, pages shared/copied, bytes and latency measured (where relevant).
- Known unsupported states/capabilities and next ticket.

## Scope control

No broad refactors to preserve a flawed abstraction. On a conflict, identify the incorrect model assumption and propose the simpler redesign. Never modify the existing `spec/branching-context-api.md` to make a failing test pass; propose an explicit versioned spec change with a migration/compatibility explanation.
