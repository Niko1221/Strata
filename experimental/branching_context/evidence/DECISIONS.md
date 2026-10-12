# Recorded implementation decisions

## October 12 UTC: Luna development harness

- Created local `feat/branching-completion` from `6dd5f9df`, preserving the
  published upstream-port branch. The handoff adds orchestration, tests and
  evidence; it does not change inference or the 85 registered assertions.
- Used a stdlib queue, explicit ticket states and an append-only hash-chained
  development ledger. This ledger is independent from canonical product
  history. No database or alternate conversation representation was added.
- Reused the existing ticket result collector and cumulative verifier. The
  harness records actual unittest IDs, fingerprints and artifacts, and refuses
  skips, xfails, missing artifacts and stale validation. A ticket is not a gate.
- Recorded an actual failing assertion for import errors being accepted as
  red evidence, then fixed it. The initial broader regression also exposed a
  missing `tests/` path case in changed-test detection; that defect was fixed.
  A second recorded red assertion exposed stale observations that could not
  be revalidated; fresh executed evidence now permits revalidation.
- Tested kernel process identities and lock contention on Windows and Linux.
  Windows owned job objects and Linux owned process groups terminated actual
  descendant processes on timeout. Ambiguous/inaccessible/live processes
  prevent duplicate execution. Abrupt Linux driver death can leave a live
  child; it must be observed rather than restarted.
- Kept 1800-second command and 7200-second ticket validation limits. The queue
  keeps the original exhaustive assertions and disables the long campaign.
  Unsupported builds/isolation/persistence capabilities remain explicit.
- Rechecked llm-49 and created a fresh owned scratch copy for bounded checks.
  Existing builds, profiles and measurements were not overwritten. Checked
  422 native/build source hashes and 11 unchanged model-file hash receipts;
  no new model bytes were hashed and no backend build was performed.
- Executed the same final 18 harness tests on Windows and Linux. Executed
  Linux G0/demo/mutation smoke and unchanged G3-004. The final model case used
  Q8_0/FP16 and existing separately linked SUT/oracle binaries; eight comparisons
  were exact, 28 byte audits passed and cleanup was observed. No numerical
  gate admission was inferred from this case.
- Audited the meaning of shared allocation counts: 24 root allocations remained
  shared by each initial child after append, but none reported filled-prefix
  bytes in the short fixture. This does not prove populated full-prefix sharing
  or total-RAM savings. Future sharing checks must distinguish allocated pages
  from live prefix payload, rather than counting unused capacity as saved KV.
- Reproduced native Windows reference failures without changing the store or
  tests. G0 ran 11 tests with two `fcntl` import errors; demo and mutants could
  not import the store. The Windows portability tickets remain necessary.
- Captured host-specific raw paths/bindings in ignored private evidence. Public
  summary records source/build/model identities and measured scope. No push,
  submission, merge or long-running certification was performed.

## October 12 UTC: C01 checkpoint history binding

- Added the SHA-256 identity of the complete consumed token sequence to each
  in-memory checkpoint. Materialization drops a hot entry whose identity does
  not match the immutable snapshot and replays the canonical history. The
  existing disk manifest continues to validate history, model contract and
  execution profile before accepting a serialized checkpoint.
- The adversarial test first failed with an `AssertionError` because the
  checkpoint carried no input identity. On the pinned Q8_0/FP16 Linux-CUDA
  profile, the green test deliberately associated a checkpoint from a
  different, same-length history and asserted that the backend reprocessed
  all 33 canonical tokens. G0 and the four reference mutants also passed.
- Direct native trace: 99 forward tokens total (three full 33-token prefills),
  354189456 exported running-state bytes, final sampled tracked-VMM counter
  251658240 bytes, 0 COW pages, 0 copied bytes. Eleven unchanged model hash
  receipts were reused and 0 model bytes were hashed. These counters describe
  this isolation/replay case and do not establish a RAM or latency benefit.
- C01's green test did not invoke the bound fresh-process oracle and made no
  numerical comparison. The existing exact G3-004 observations remain the
  only reported numerical result; neither they nor C01 qualify the profile.
- One initial remote red launch could not stage its test because the target
  directory was absent. A separate scratch attempt accidentally copied the
  fixed implementation before its red phase and was recorded as `FAIL`; it is
  retained as a failed attempt and is not part of C01 validation. The final
  clean-source scratch produced the accepted red and green evidence. A 30-second
  client observation ended while green remained live; the same recorded child
  was inspected and allowed to finish (84.31 seconds), with no duplicate run.
- Final C01 run records and the remote ledger snapshot are preserved at
  `evidence/runs/luna/remote/C01-20261012/`. The native engine files and pinned
  SUT/oracle binaries were unchanged, so this ticket did not require a backend
  rebuild. Ticket validation does not change any gate status.

## October 12 UTC: C02 versioned controller validation

- Controller state now checks schema 1, the declared HMAC-SHA256 RNG version,
  the exact sampler processing order and an in-range 128-bit draw cursor before
  either token selection or forced-token consumption. Exhausted cursors return
  an integrity error instead of overflowing inside `int.to_bytes`.
- The first two red attempts were preserved but not accepted: unittest subtests
  reported expected `OverflowError`s as errors, then the runner could not match
  subtest IDs to the registered red assertion. The final deterministic test
  aggregated all outcomes into the required assertion ID and recorded RED.
  Green rejected all eight combinations (four corruptions through `choose`
  and `consume`); G0 and four reference mutants passed.
- The model-free green took 0.183 seconds. No model was loaded; no SUT/oracle
  forward, GPU work, numeric maximum or resource measurement was produced.
  Linux bindings were still checked by the driver. The backend source and
  binaries were unchanged, so no backend rebuild was required.
- C02 run records and the hash-chained remote ledger snapshot are preserved at
  `evidence/runs/luna/remote/C02-20261012/`. Ticket validation is not G1
  certification; the 85 registered cases remain governed by the original
  assertions and tolerances.

## October 12 UTC: C03 pinned-root retention

- `ConversationCache::drop_superseded` now retains an existing image when it
  owns a pinned checkpoint. A background child carrying the same root
  checkpoint cannot delete the parked main prefix. Unpinned turn-back images
  remain eligible for superseded cleanup and eviction.
- The C03 red run failed because parking the background child removed the
  pinned root. The green CPU-only C++ probe confirmed two parked images after
  the fork, eviction of the unpinned child, and refusal to evict the pinned
  root. The same test evicted derived Python cache state, confirmed canonical
  root/child history remained readable, and faulted before publishing a
  canceled child; no object appeared. G0 and the four reference mutants passed.
- The probe compiled the changed header with the host C++ compiler; the native
  CUDA, HIP and SYCL engine builds were not run. No transformer forward,
  populated KV-sharing, resource, numerical or latency measurement was made.
  Full backend build evidence remains in X01, and this ticket is not G2/G3
  certification.
- C03 evidence and the hash-chained remote ledger snapshot are preserved at
  `evidence/runs/luna/remote/C03-20261012/`.

## October 12 UTC: R01 immutable host-buffer segments

- Changed `ConversationBuffer` copies to share `shared_ptr`-owned segments.
  Mutable `visit` and an in-place partial resize detach before writing. A
  shared final segment has no writable spare capacity, so append allocates a
  private segment instead of changing another snapshot's logical extent.
  `shared_bytes()` reports referenced shared capacity and `copied_bytes()` is
  incremented by the number of payload bytes actually copied on detach.
- The clean-scratch red compiled the adversarial probe and failed its direct
  pointer-sharing assertion against the original deep-copy implementation.
  Green observed 4096 shared segment bytes across three siblings and exactly
  4096 COW-copied bytes after one sibling changed one byte; the ancestor and
  untouched sibling retained their original byte. A same-final-token prompt
  with a different earlier prefix received no cache match. The test emitted
  `R01_METRIC shared_prefix_bytes=4096 cow_copied_bytes=4096`.
- The final green took 0.779 seconds on llm-49/r4090. The bound Q8_0/FP16
  profile (`b62135ea`), SUT (`9d045e70`) and separate oracle (`d2e6b715`) were
  fingerprinted by the driver. The test used only the CPU native header probe;
  it did not run the SUT, invoke the fresh oracle or compare logits. Thus it
  provides no R01 numerical maximum and no numerical admission. G0 and all
  four reference mutants passed. Ticket validation only; no registered gate
  was certified, and CUDA/HIP/SYCL builds remain queued under X01.
- Early R01 scratch attempts were kept as evidence: the first probe used a
  nonexistent templated cache type and failed to compile; the next red was a
  valid pointer assertion, but an initial green exposed a stale segment
  accessor, then a too-short prefix fixture. The driver refused another red
  after `IMPLEMENTED` because that test edit changed the ticket fingerprint.
  No ledger was reset: the corrected test was run red-first in a new owned
  scratch cloned from validated C03, then implemented and validated there.
- Final R01 run directories and hash-chained ledger snapshot are preserved at
  `evidence/runs/luna/remote/R01-20261012/`. The source edit changes native
  headers, so full backend builds and their dependent reviews are still
  outstanding.

## October 12 UTC: R02 published-tail isolation

- Changed host segments into immutable payload views. `freeze()` marks every
  segment in a captured snapshot immutable, even when its reference count is
  one. A later mutable visit splits the view around the write range and copies
  only the bytes being changed; the published prefix and suffix retain their
  original payload. Snapshot capture freezes K, V, scale and pooled buffers
  only after all transfers succeed and immediately before publishing the image.
- The test was written first and red failed with an `AssertionError` because
  the original buffer had no publication/freeze operation. Green recorded
  `R02_METRIC immutable_unique_copy=17 private_tail_copy=23`: a 17-byte write
  to a unique frozen payload copied 17 bytes, and a 23-byte write to a fork
  copied 23 bytes while the parent's prior value stayed unchanged. Green took
  0.581 seconds; G0 and the four reference mutants passed.
- Driver bindings checked the Q8_0/FP16 profile (`b62135ea`), pinned SUT
  (`9d045e70`) and separately linked fresh oracle (`d2e6b715`), but this CPU
  probe did not load either binary, call the oracle or compare logits. It did
  not execute the original exhaustive G3-003 sweep, physical CUDA page audits,
  quantization metadata isolation or GDN/PLE/frontier integration assertions.
  Those registered assertions were not edited or weakened. Ticket validation
  does not certify G3, and native CUDA/HIP/SYCL builds remain outstanding.
- Red/green/regression run directories and the append-only ledger snapshot are
  preserved at `evidence/runs/luna/remote/R02-20261012/`.

## October 12 UTC: R03 unique host-storage accounting

- Changed `ConversationCache` byte admission to count a shared segment payload
  once across parked snapshots and retained reuse buffers. Each buffer
  directory and each snapshot's independent metadata remain charged. `put`
  calculates the candidate's additional unique allocation bytes against the
  current cache; eviction and clear update the measured total by removing
  owners, not by subtracting a guessed per-entry payload estimate.
- Added direct storage metrics that enumerate payload allocation identities,
  capacities and live `shared_ptr` owner counts, and report the existing
  byte-copy counter. The native probe first failed because two entries were
  charged for the same 4096-byte payload. Green measured 4644 total bytes for
  both snapshots, two owners / one allocation, then one owner after eviction,
  then zero bytes and owners after clear. Green took 0.913 seconds; G0 and four
  reference mutants passed.
- The probe did not measure process RAM, induce OS pressure/OOM, exercise RAM
  or disk restoration, load the model or make a numerical comparison. The
  driver checked the Q8_0/FP16 profile, pinned SUT and separate oracle
  fingerprints, but the CPU test did not call them. The registered G2-013 and
  G3-008/009/012/014/015 cases remain unexecuted; no gate was certified.
  Existing `conversation_memory.cpp` OS telemetry was left intact because it
  samples available RAM and cannot establish allocation identity for shared
  host segments.
- R03 run directories and its hash-chained ledger snapshot are preserved at
  `evidence/runs/luna/remote/R03-20261012/`.
- A post-review failure-atomicity change reserves the segment-view directory
  before erasing/replacing a published view, so an allocation failure leaves
  the old layout intact. Its first probe rerun exposed an invalid `self`
  receiver in the non-template method and failed to compile; that run is under
  `R03-20261012/post-reserve-retry/`. The corrected green took 0.886 seconds,
  and G0/mutants passed again. The final driver check is current with the new
  source fingerprint; `events-final.jsonl` preserves this later ledger state.

## 2026-10-12: primary review after Luna stopped

- Preserve Luna's source and tests in `6e651a15`; work in a separate owned remote
  copy and retain all red runs. Ticket labels represent measured component scope,
  not completed native integration or gate certification.
- Restore the legal idle forced-token transition. An idle controller may contain
  only its status; generation controllers still require the versioned schema.
  Three tests passed, including nine exact fresh-process model comparisons.
- Fix the real ASan use-after-free by capturing a segment's shared payload and
  offsets before reserving its directory. Four injected allocation failures
  preserved bytes and reported zero completed copy bytes.
- Preserve unpublished workspace value-copy semantics required by the original
  R01 probe. Published snapshots retain bounded written-range COW. The unchanged
  R01/R02/R03 probes and two new safety tests passed; tests were not weakened.
- Rebuild actual CUDA binaries before attributing host-sharing behavior to model
  inference. The old pinned binaries establish controller-boundary results only.
  Complete C01's native cache integration before advancing dependent slot work.

## 2026-10-12: capture admission and exact native lookup

- A read-only callback on a mutable buffer must receive const bytes without
  detaching its payload. Mutable callbacks still take the COW path. New actual
  native tests first failed and then passed; original R01/R02/R03 probes pass.
- Preserve the existing standalone capacity-admission contract. It charges
  independent K/V buffers conservatively; the cache's shared allocation walk
  and direct storage metrics count actual payload allocations once. Do not
  present the conservative admission charge as measured physical memory.
- Admission includes detached payloads and changed view directories on a
  published rewrite, including same-size and shrinking captures. Three native
  admission/read/accounting cases passed; G0 51 and four mutants passed.
- Keep the unchanged native snapshot test failure. Its unconditional retained
  pointer assertion conflicts with a rewritten shared prefix. No production
  switch, ancestor mutation or test edit was introduced to produce a green run.
- The rebuilt pinned CUDA binaries passed 17 exact fresh-process comparisons,
  including G3-004, but this is bounded evidence with an unqualified profile.
  Normal conversation-cache and batch-slot integration remains outstanding.
- Add exact live-history cache lookup rather than changing request-prefill
  matching, which intentionally excludes the final prompt token. Check all
  tokens, image keys and steering state. This is a derived cache primitive;
  canonical snapshots and controllers remain in the existing immutable store.
  The primitive's one native CPU case passed after a recorded fixture-setting
  correction and a fresh red/green cycle. It is not an inference integration.

## 2026-10-12: cold-to-resident canonical identity

- Retain `saved.input_identity` when replacing checked disk state bytes with a
  resident native image. Dropping it turned a subsequent hot lookup into a miss
  and replayed the prefix. This changes one constructor argument, without new
  representations or relaxed integrity checks.
- The real regression passed after a clean red assertion: actual payload size
  and manifest hash, exact fresh-process logits, unchanged canonical record,
  and zero forward-token work on both cold and subsequent hot materialization.
  G0 51 and four mutants passed. Existing pinned CUDA binaries were used.

## 2026-10-12: actual native RAM-cache adapter

- Use the existing `SavedConversation` numerical checkpoint and
  `ConversationCache`, with an optional consumed next-logit frontier. Session
  files still omit that frontier and use the existing outer manifest. No
  canonical history/controller representation was added.
- RAM capture/restore is private to a fixed-model native process. Exact full
  history lookup checks all tokens; mismatches return without changing device
  state. Restore validates geometry and allocates replacement history/frontier
  before touching private device state. Native RNG/grammar are not inferred.
- Admit capture under a 1 GiB / eight-entry derived cache. Capacity charges
  include running state, frontier and metadata; K/V allocation metrics enumerate
  actual shared payload identities. The existing copy metric counts host COW
  clones, not device-to-host capture traffic or process RSS. No RAM savings or
  populated long-prefix sharing is claimed by the short test.
- Real Q8_0/FP16 CUDA sibling tests made six fresh comparisons with maximum
  error 0.0, restored twice with zero prefix forward work, preserved a mismatched
  history as a read-only miss, and released host payloads on cache clear.
  Unchanged native support checks, G0 and four mutants passed. The cold identity
  regression also passed on the newly pinned build.
- Clear native RAM entries during full numerical-checkpoint eviction. The next
  ticket connects adapter restoration to automatic canonical materialization;
  ordinary concurrent serving remains separate work.

## 2026-10-12: automatic canonical RAM restoration

- Populate the existing native numerical cache at a consumed `_save` boundary.
  On a canonical materialization miss, restore the exact complete history or
  replay it. The no-native-cache adapter always reports a miss. Canonical
  history and controller records remain the sole source of truth.
- A detected input-identity misbinding clears native history acceleration before
  replay. The original misassignment test still measures full canonical replay;
  its assertions were not edited.
- Replace exact unpinned duplicate live histories only when neither has an
  intermediate checkpoint history. Keep the pinned-prefix rule. A real native
  RAM regression exposed the duplicate; a native CPU counterexample and the
  unchanged real regression both pass after the fix.
- The canonical API passed nine exact fresh comparisons; the RAM adapter passed
  its original six comparisons on the rebuilt native code. Zero forward work
  was measured for a RAM restore, while changed-prefix and complete-eviction
  misses each computed nine tokens. No physical host-sharing or speedup claim
  follows from those work observations.
- Preserve tested binary copies before rebuilding. New profiles point to their
  retained SUT/oracle copies, so a subsequent build does not overwrite them.

## 2026-10-12: one owned workspace and FIFO branch references

- Bind execution leases to their entering thread and check ownership before
  every operation, including release. Real CUDA regression confirmed rejected
  foreign operations do not consume tokens or clear the active lease.
- Use a deque of immutable jobs containing existing canonical branch heads.
  Keep controllers in the store. Execute one job at a time and expose a result
  after revision-checked durable adoption. Two forced futures matched fresh
  forwards at every consumed boundary.
- Prioritize one request at a time, as explicitly requested by the user. This
  is local API scheduling, not ordinary serving or concurrent-kernel validation.

## 2026-10-12: cancellation releases pending work only

- Remove a pending job by object identity under the FIFO lock. A repeated
  cancellation returns false. An active job retains its execution owner.
- Check the captured branch revision before numerical work and adopt only
  after the complete job. Failed publication may leave durable immutable
  orphan boundaries; it cannot advance the movable branch reference.
- Keep the assertion-red test byte-identical. Additional cache-pressure
  coverage lives in a separate regression module; the harness correctly
  refused a changed red-test module before that separation.

## 2026-10-12: compare execution operators before numerical admission

- Preserve the original rejected normal-batch fixture and all new full vectors.
  A diagnostic PASS proves the measurements ran; it does not admit equivalence.
- Include a one-token control. Its nonzero discrepancy eliminates batched
  prefix processing as the sole cause and directs investigation to ordinary
  verifier initialization and expert placement.
- Keep the separate pinned token-level oracle and existing tolerances unchanged.
  Do not reinterpret unchanged top-1 predictions as matching logits.

## 2026-10-12: repair the shared inference defect before admission

- Treat independent process/state isolation as necessary but insufficient for
  an oracle. Compare against ordinary model execution to catch shared driver
  mistakes. The one-token control exposed omitted resident experts even though
  the branch and fresh drivers had agreed exactly with each other.
- Bind the real ExpertDispatch to its verifier GPU plan and report actual
  resident/CPU/PCIe work. Preserve the old binaries and evidence; use a new
  execution identity for the correction. No tolerance or reference equation
  changed. No ordinary serving default changed.
- Pin the existing PCIe-share option explicitly in the corrected profile, since
  an automatic placement choice should not silently change the numerical
  operator. The measured mmap fixture used no PCIe groups; do not claim it did.

## 2026-10-12: remeasure prefill after correcting expert dispatch

- Add observation ticket N02-BASELINE before further N02 implementation. It
  runs the existing full-vector diagnostic unchanged, using the corrected
  fresh executable and profile, the preserved normal binary and historical
  counterexample. No acceptance assertions or tolerances change.
- The dedicated bindings now bind profile, model receipt and native source to
  the corrected build, rather than the historical profile used for the
  controlled dispatch red. All vectors and actual forward counts are retained.
- Ordinary prefill and token decode use different arithmetic paths: the source
  explicitly documents BF16 versus FP32 activations for several projections
  and automatic CPU expert sharing in eligible ordinary prefill. These are
  inspected differences, not yet isolated causes of the measured error.

## 2026-10-12: preserve the short prefill failure and unsupported control

- Reduce the historical fixture with diagnostic-only length/chunk selectors;
  preserve the original default matrix and all acceptance assertions.
- Retain the two-token top-1 disagreement. Do not dismiss it as long-context
  accumulation or use a token-level fallback to claim prefill equivalence.
- Record ordinary native-pack no-prefill rejection as a missing comparison
  primitive. Do not bypass its guard or add an engine flag to manufacture a
  passing control. Intermediate state instrumentation remains a possible
  follow-up; the observed rejection does not prove the underlying cause.
- N02 is explicitly blocked at this boundary. Independent evaluation,
  persistence and hardening work can proceed; full serving parity cannot.

## 2026-10-12: ordinary initialization as an independent diagnostic control

- Add an EXCLUDE_FROM_ALL CUDA diagnostic target, using the ordinary verifier
  construction and expert-plan wiring already in generate.cpp. The private
  fresh/branch startup paths are not used. Preserve their pinned binaries.
- The diagnostic supplies forced tokens to T=1 run/commit calls and reports
  actual consumed work, positions and expert entries. This bypasses optimized
  prefill only in the diagnostic executable; it does not qualify prefill.
- Compare all logits at every short consumed prefix before examining internal
  buffers. The exact control results distinguish the previously suspected
  multi-token driver initialization issue from the observed prefill-path error.
- Verify default isolation by rebuilding ordinary strata and hashing both
  binaries: they are byte-identical, stronger than a source-only assertion.

## 2026-10-12: compare populated state without defining another snapshot API

- Keep the dump diagnostic-only with no restore operation. Read actual session
  buffers and translate the first logical KV page through its page table;
  capture only the first populated cell per head, not spare allocation bytes.
- Split GDN recurrence and convolution using the inspected native geometry.
  Capture PLE and live indexer components separately. Reject unsupported layouts
  instead of pretending this bounded diagnostic supports every checkpoint.
- Establish readback noninterference by comparing final full-logit hashes against
  both previous uninstrumented runs. All bytes match. The diagnostic exposes
  divergence; its passing test does not assert numerical equivalence.
- Source inspection shows different projection arithmetic before the first GDN
  convolution: prefill native_proj consumes FP16 activations; the verifier uses
  its native quantized projection path. This is a localization hypothesis,
  not yet an isolated causal measurement or justification to loosen tolerances.

## Projection diagnosis, 2026-10-12

Keep projection capture in the separate CUDA diagnostic target. The production
executable hash is unchanged. Preserve the observed pre-projection discrepancy
as an unresolved arithmetic-path issue; do not alter tolerances or attribute it
to convolution storage. Ordinary token-step control now exists and agrees on
the tested fixtures; the former missing-control blocker is resolved, while
normal batched-prefill equivalence remains unresolved. See
[projection-comparison-validation.json](projection-comparison-validation.json).

## Bind N03 to executed admission, 2026-10-12

Remove the obsolete hardcoded oracle pathname and default source revision. Use
the oracle executable from the measured admission receipt, reject a changed binary
hash, and require explicit source provenance. Keep the calibration matrix and
tolerance rule unchanged; token-level repeatability does not admit batched prefill.

## Private publication only, 2026-10-12

The user suspended GitHub/Strata publication pending their discussion with Niko.
Work is now on private/branching-context-prototype, tracking the existing
box-git:/srv/data/git/strata-private.git remote. The WIP calibration commit
3cd3479e8de97e36d168f7210ba17b83e9d3d334 was pushed there and the remote SHA
verified. Do not push further work to GitHub or update the existing draft PR
until the user explicitly authorizes publication again. Calibration and the
remaining implementation queue continue privately.

## Serving-path check, 2026-10-12

The omitted-expert repair belongs to branch_driver and fresh_driver; the ordinary
serving runner already sets drive.d.plan = ver.plan_sink(). The private build
passed the existing seven-request A/B/A serving smoke without a further wiring
change. Treat this as bounded serving/cache regression evidence, not proof that
batched prefill matches the token-step numerical operator. No test assertion
was changed. Evidence is private-serving-validation.json.

## 2026-10-12 primary host and supporting evidence

User selected llm-60 as primary and llm-49 for agreement checks. Preserve each hardware/build calibration separately; do not pool repeats into a falsely complete profile. Saved llm-60 exact ordinary-step and evaluation noninterference measurements are supporting observations only, not frozen-profile gate admission. No GitHub publication.

## 2026-10-12 E02 evaluator process boundary

Use actual Linux bubblewrap namespaces for evaluator execution, with read-only runtime/input mounts and disposable scratch space. Passing detached Python dictionaries alone would not isolate filesystem or network effects. llm-49 executed namespace and adversarial test probes successfully. Do not substitute an unrestricted subprocess when isolation is unavailable; unsupported hosts fail closed. Preserve candidate history/controller representation and compose evaluation provenance separately.

## 2026-10-12 selection and adoption

Read only hash-checked evaluation records. Require explicit mandatory check names, all recorded mandatory checks PASS, no unresolved checks, and a finite available score. Reject mixed evaluator/fixture/environment specifications and contradictory judgments of one snapshot. Order eligible candidates by declared metric, fewer appended canonical inputs since task root, stable snapshot ID, then evaluation ID. Reuse Store.move revision CAS and retry receipts for adoption. No alternate branch representation.

## 2026-10-12 host-specific evaluator isolation limitation

Preserve the failed llm-60 E02 test and AppArmor denial logs. Do not weaken evaluator isolation or disable host policy to convert it to PASS. llm-49 remains the available isolated-evaluation host; llm-60 remains the primary inference host. This limits G4 host qualification, not the correctness of an already measured GPU forward.

## 2026-10-12 authorized llm-60 namespace policy adjustment

Per explicit user authorization for the test box, applied runtime sysctl kernel.apparmor_restrict_unprivileged_userns=0. No persistent config or other global protection was changed. AppArmor remains enabled and the unchanged evaluator sandbox tests passed. Reverse with sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=1. Prefer llm-60 for G4 again after baseline qualification.

## 2026-10-12 local principal boundary

The trusted local owner authenticates a principal and binds a restricted Store/model handle with for_principal. Grants are separate mutable authorization metadata keyed by canonical root ID, not per-user history copies. Bound handles cannot impersonate another principal or administer grants. This is API authorization, not isolation against arbitrary Python executing as the filesystem owner; P02 addresses separate OS users. Revocation is checked at operation entry; an already admitted operation is not retroactively canceled. Evaluation access metadata contains only a snapshot ID so denied reads need not deserialize the report payload.

## 2026-10-12 at-rest policy

Use explicit POSIX 0700 directories and 0600 publication/staging files, independent of caller umask. Reject symlink directories and directories owned by another OS principal. Store policy states owner-only plaintext; no encrypted-storage claim or unverified key handling. Existing legacy payloads are protected by owner-only root traversal; their modes are not silently recursively rewritten. Registered sensitive-storage acceptance must exercise actual native checkpoint files under a separate OS principal.

## 2026-10-12 retention lease lifetime

Use shared OS locks on snapshot-named lease files. The kernel releases an active lease on process death; stale lock files alone do not pin history. Register leases under the reference transaction lock so a future collector can synchronize discovery with entry. Named historical pins and evaluation roots are separate retention metadata over the canonical snapshots. This does not yet authorize or implement physical history deletion.

## 2026-10-12 explicit collection eligibility

Default history retention is preserved: unreferenced does not imply permission to delete. release_history explicitly marks a snapshot eligible; collect_history is a separate owner administration operation. Mark branch roots/heads, retry-receipt targets, named pins, evaluation candidates/task roots, active leases and every unreleased object, then their ancestors. Delete eligible unreachable descendants before ancestors. Serialize history publication and reference moves with the same transaction lock. Numerical cache eviction never invokes history release or collection.

### Collection concurrency evidence (2026-10-12)

Evaluation retention was measured at the actual public evaluate read-lease boundary using a Store subclass only to synchronize processes; scoring still executed through bubblewrap. After publication, the immutable report retained its candidate without the reader process. Rewind races accept either legal serialization and require a valid root head, revision and idempotent receipt. These bounded CPU schedules do not replace registered GPU validation.

### Checkpoint upgrade boundary (2026-10-12)

Kept incompatible checkpoint loads as explicit errors. Tightened versioned manifest parsing rather than inferring compatibility from tensor shapes or silently importing unknown fields. Existing explicit numerical eviction is the cold-replay route under the same model contract; canonical history and references are untouched. No format converter or automatic semantic migration is claimed. CPU checks pass; native replay qualification remains pending.

### Rollback schema refusal (2026-10-12)

JSON schema versions must be integers, not values that merely compare equal in Python. In particular a Boolean legacy version must not trigger reference migration. Reject before changing reference records. Process-kill receipt recovery is measured separately from power-loss durability and from a future two-release executable rollback test.

### E01 report/index regression correction (2026-10-12)

A recursive basename assertion predating P01 conflated a canonical evaluation report with its derived authorization index. Kept production representation unchanged. Corrected only that CPU assertion to validate both distinct artifacts and reject additional matches; canonical report hash and corruption tests remain. This fixes an obsolete test assumption, not a numerical/backend failure. Preserved failed run and exact source overlay.

### Supported matrix filesystem scope (2026-10-12)

Measured both workspace and temporary-store mounts. llm-49 workspace is ext4 but default /tmp is tmpfs; llm-60 uses ext4 for both. Earlier llm-49 CPU tempfile runs are tmpfs evidence, not ext4 durability evidence. Keep ext4 as the declared draft storage target and place future secondary-host matrix test stores there explicitly. No host-independent or power-loss claim follows from these runs.

### Frozen baseline operator descriptions (2026-10-12)

Preserved the measured calibration and pinned execution identity. Their operator descriptions differ in prose, although full commands and actual executable hashes match. Finalizer recognizes only this exact historical pair after binary/command validation, retains the baseline description separately, and rejects an unknown/batched description in an executed negative test. No tolerance, raw numerical result, reference equation or model/backend identity changed.

### Draft validation priority restored (2026-10-12)

The user clarified that full gate certification is not required for the initial prototype. Interrupted the long page-boundary G1 campaign without modifying its tests or tolerances. Preserve it as incomplete, not failed numerical evidence or a passing gate. Continue focused real-GPU checks of the implemented workflows before adding more features.

### Serial comparison interpretation (2026-10-12)

Use pristine upstream source 7695859a for the stock baseline, not a cache-disabled modified binary. Keep ordinary-serving comparison separate from the experimental immutable API. Retain the measured slower total and faster reused calls without changing cache budgets or fitting numerical tolerance. Stock and modified serving agree on generated IDs; the independent fresh oracle checks full API logits. These facts do not establish batched-prefill numerical equivalence. All work remains local/private.
