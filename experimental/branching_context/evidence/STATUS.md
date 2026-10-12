# Branching-context implementation handoff status

The Luna handoff harness is implemented on local branch
`feat/branching-completion`, based on publication commit `6dd5f9df`. The queue
contains 34 tickets, including a disabled exhaustive-certification ticket.
The first implementation tickets now bind numerical checkpoints to exact
canonical histories, validate controller versions and draw cursors, keep
pinned main-prefix images during background-fork cache cleanup, and share
immutable host-buffer segments with bounded copy-on-write for published tail
ranges and cache-wide unique-payload budget accounting. Concurrent-slot
integration, G4 and G5 remain queued.

## Executed bounded checks, October 12 UTC

| Check | Observed result |
| --- | --- |
| Harness regressions, Windows | 18 passed; zero failures/errors/skips/xfails |
| Same harness revision, Linux | 18 passed; zero failures/errors/skips/xfails |
| Reference G0, Linux | 51/51 passed; zero skips |
| Reference demo / mutation smoke, Linux | Demo passed; four mutants rejected |
| Registered real-CUDA G3-004 | One case passed; eight exact comparisons, max error 0.0, 28 byte audits |
| Hot append work | Six appends, one forward token each, zero prefix recomputation |
| Resources in the short CUDA case | 216 COW pages / 452984832 bytes; sampled tracked-VMM peak 704643072 bytes |
| Ownership cleanup | Eight handles/images released, one context closed, 27 leases acquired/released |
| Native Windows reference checks | G0 failed: 11 tests ran, two import errors; demo and mutation smoke also failed on missing `fcntl` |
| C01 checkpoint/history binding | One real-CUDA ticket test passed; a deliberately misassigned same-length checkpoint missed and replayed 33 canonical tokens |
| C02 controller identity | One model-free test passed; eight corrupted controller/RNG cases were rejected before sampling or input consumption |
| C03 pinned-root retention | Native CPU policy test passed; a background child no longer supersedes a pinned root, and only the unpinned child was evicted |
| R01 host-prefix ownership | Native CPU probe passed; siblings shared 4096 segment bytes, one changed byte caused 4096 actual bytes copied, and exact-prefix mismatch was rejected |
| R02 published-tail isolation | Native CPU probe passed; a 17-byte write to a unique published page and a 23-byte sibling tail write copied exactly those bytes |
| R03 unique host-storage accounting | Native CPU probe passed; 4,096 shared payload bytes charged once, owners moved 2 → 1 → 0 through eviction and clear |

The model case used existing pinned `ed5c4557` binaries, Q8_0 weights, FP16 KV
and the token-level operator on llm-49/r4090 (RTX 4090). The harness checked 422
native/build source files, actual binary hashes, 11 still-valid model hash
receipts and oracle symbols. It did not rebuild the backend or rehash model
bytes. The final case took 75.9314 seconds on a shared host. No throughput or
overall-RAM-benefit claim follows.

Each initial child retained 24 root physical allocations after append. In this
short partial-page fixture none of those shared allocations had a nonzero
`filled_prefix_bytes` in the audit. Populated tail allocations were copied.
This checks immutable retained bytes and measured COW, not populated full-prefix
sharing across long branches. That sharing/accounting qualification remains
outstanding.

The C01 run used the existing pinned SUT (`9d045e70`) and separately linked
oracle binding (`d2e6b715`) with the Q8_0/FP16 profile (`b62135ea`). The C01 test
did not invoke the oracle or compare logits, so it has no C01 numerical maximum
and does not qualify G1. Its direct native trace counted 99 forward tokens:
three 33-token prefill operations, including the 33-token replay after the
wrong checkpoint association. It reused 11 prior model-file hash receipts and
hashed 0 model bytes. The final native counter sample reported 251658240
tracked VMM bytes, 354189456 exported running-state bytes, 0 COW pages and 0
copied bytes; this isolated replay test does not measure sharing benefit.

C02 validated against the versioned controller contract without invoking the
model, SUT or oracle. Its test exercised four corruptions (unknown RNG ID,
unknown sampler order, negative draw cursor and cursor beyond the 128-bit PRF
range) through both sampling and input consumption. All were rejected as
`IntegrityError`; G0 and the four reference mutants passed. This run produced
no model, numerical, GPU-resource or throughput measurement.

C03 exercised the native cache header with a CPU C++ probe, plus canonical
store eviction and cancellation before publication. The pinned main prefix
remained available after a background child was parked and evicted; canonical
root and child histories remained readable, while the canceled child created
no object. The changed native header was compiled by the probe, but the full
engine was not built for CUDA, HIP or SYCL. Physical sharing, allocation bytes,
latency and overall RAM savings were not measured; backend builds are queued
under X01.

R01 changed `ConversationBuffer` copies to share owned segments. Mutable visits
and partial resize detach shared segments before exposing or changing payloads;
the class reports shared segment capacity and cumulative bytes copied by COW.
Its direct probe observed 4096 shared bytes across three sibling buffers and
4096 copied bytes when one sibling changed one byte. The exact-prefix cache
probe matched a three-token root only when the full prefix agreed. Green took
0.779 seconds; G0 and reference mutants passed. The driver checked the
Q8_0/FP16 profile plus the existing SUT and separately linked oracle hashes,
but this CPU test made no model forward or numerical comparison. The 4 KiB
fixture is an ownership check, not a whole-workload RAM-savings or throughput
measurement. No CUDA, HIP or SYCL engine build was performed; X01 remains
queued.

R02 made buffer segments explicit views over owned payloads. Snapshot capture
freezes every K/V, scale and pooled segment, including unique buffers. A later
mutable visit splits the view at the written range and copies only that range,
keeping neighboring prefix/suffix bytes on the published payload. The CPU
probe measured 17 bytes copied for a write to a unique frozen buffer and 23
bytes for a sibling tail write; the parent byte stayed unchanged. Green took
0.581 seconds; G0 and reference mutants passed. Driver bindings included the
Q8_0/FP16 profile, SUT and separate oracle, but this test did not load them or
compare logits. It did not run physical CUDA page audits, scale/indexer/GDN/PLE
integration checks, or exhaustive G3-003 coverage. Those original assertions
remain unchanged and unqualified; no backend build or gate certification was
performed.

R03 changed cache byte accounting to deduplicate shared payload allocations
across parked snapshots and retained reuse images. Direct storage metrics
report unique payload capacity, allocation count, current `shared_ptr` owner
count, and cumulative COW-copied bytes. The probe measured 4,644 cache bytes
for two snapshots sharing one 4,096-byte payload; the owner count fell from
two to one when the first entry was evicted and to zero after clear. Remaining
bytes were per-snapshot directories and metadata. The final green after a
pre-reservation safety fix took 0.886 seconds; G0 and reference mutants passed.
An intermediate rerun preserved a compile typo in that reservation fix. No
host RAM-pressure telemetry, OOM injection, model restore, disk roundtrip, GPU
work or numerical comparison was run. The registered G2/G3 methods remain
unexecuted; this ticket is not gate certification. `conversation_memory.cpp`
telemetry was unchanged because this probe measures cache allocations through
their actual payload owners, not an OS available-memory estimate.

## Gate boundaries

| Gate | Status for this handoff |
| --- | --- |
| G0 | PASS on Linux; FAIL on native Windows because the store cannot import |
| G1 | BLOCKED for certification: current numerical profile not qualified; full suite unexecuted |
| G2 | BLOCKED for certification: current full crash/restart/OOM/eviction suite unexecuted |
| G3 | BLOCKED for certification: one of 16 cases executed; no complete physical/resource qualification |
| G4 | BLOCKED: 11 registered methods and full workflow absent |
| G5 | BLOCKED: 12 registered methods and release hardening outstanding |

There are 85 registered cases and 62 existing method bodies. The 18 harness
tests and C01-R03 implementation tests are separate from the registered suite
and do not increase registered implementation coverage. G4 still lacks its 11
registered methods; G5 still lacks its 12. Ticket validation is not gate
certification.
The runner preserves old validation, exposes stale fingerprints and never
converts a ticket outcome into a gate PASS. The long campaign remains disabled.

## Evidence and restart

[harness-validation.json](harness-validation.json) records actual test IDs,
counts, fingerprints, numerical/resource measurements and limits. Private raw
evidence is in `evidence/runs/luna/`, including the Windows import failures,
two intended red failures, the append-only ledger and
`remote/evidence-final.zip`. Archive SHA-256:
`7a7a3b235f33e01d0ac281d022f0165c378125d7c4abda276b8c50b7e65bc535`.

The Windows ledger has H00 and B00 validated. The owned Linux ledger records
H00, B00, B01, B02, C01, C02, C03, R01, R02 and R03 validated; on the final
R03 source fingerprint, R03 is current while R01 and R02 are stale and need
post-build regression. Its recorded next ticket was **S01**; the takeover
review below supersedes that recommendation. C01,
C02 and C03 run reports and hash-chained ledger snapshots are under
`evidence/runs/luna/remote/C01-20261012/` and
`evidence/runs/luna/remote/C02-20261012/` and
`evidence/runs/luna/remote/C03-20261012/` and
`evidence/runs/luna/remote/R01-20261012/` and
`evidence/runs/luna/remote/R02-20261012/` and
`evidence/runs/luna/remote/R03-20261012/`. The local Windows ledger remains
separate; do not rerun remote reference/model evidence to make it appear local.
The remote location is recorded in `evidence/runs/luna/remote/location.json`:
`/home/dflanag3/branching-g3/luna-handoff-20261012T015402-76cb1e6d`.
Do not repeat the completed reference/model checks merely because the local
Windows ledger does not contain the Linux runs. Inspect the Linux ledger on
its own host with the recorded bindings.

Read [the Luna handoff](../plans/luna/HANDOFF.md) and launch with
[LAUNCH.txt](../plans/luna/LAUNCH.txt). No Codex CLI session, push, PR, service,
or long certification campaign was launched by this handoff preparation.

## Primary takeover, 2026-10-12

Luna's execution turn stopped before takeover. Commit `6e651a15` preserves
its implementation and tests. The owned execution directory is now
`/home/dflanag3/branching-g3/primary-queue-20261012T031643` on llm-49 / r4090.

- C02-REPAIR: PASS, three tests, no failures/errors/skips/xfails, 74.878 seconds.
  Nine fresh-process full-vocabulary comparisons had maximum absolute error
  0.0. Forced idle feeds, consumed-token publication, interrupted publication,
  grammar/RNG resume and complete numerical-checkpoint eviction were checked.
  The existing CUDA binaries and unqualified Q8_0/FP16 profile were used.
- R03 repair: PASS for two actual C++ sanitizer/allocation-failure tests,
  two unchanged R01/R02 probes and the unchanged R03 accounting probe.
  ASan exposed a segment reference invalidated by vector reservation; capturing
  its payload and offsets before reservation removed the use-after-free.
  Four failed allocations left the payload and completed-copy counter intact.
  The original R01 workspace-copy assertion also failed during review; restoring
  unpublished value-copy behavior fixed it while retaining published range COW.
- G0: PASS, 51/51 on Linux. Four reference mutants rejected. No registered
  tests, equations or tolerances were edited. No new G1-G5 gate PASS is claimed.

[takeover-validation.json](takeover-validation.json) records executed test IDs,
reports, artifact hashes and limitations. Raw archive SHA-256 is
`38173e40e26e661ccc6ec66a2f2528590a6c61e661302e2206fb7b3524c0a2ed`.
All earlier failures remain in the ledger and raw evidence.

C01's focused checkpoint-identity check does not establish normal native cache
integration. The next work is a new CUDA engine build and completion of that
integration before relying on S01's prerequisites. CUDA configuration succeeded
with the changed source. Compilation of 15 targets succeeded in 127.188 seconds.
Nine native executables then passed, but the unchanged conversation snapshot
test failed its incremental allocation-admission assertion. A new test reproduced
under-admission for published shrink/same-size/growth rewrites; including detach
payloads and view-directory allocations made that focused test pass. The rebuilt
snapshot test now reaches a retained-pointer stability failure. The unchanged
validation and transfer tests also fail their exact estimate-versus-image-bytes
assertion. These integration failures remain unresolved; native validation is
FAIL, not PASS. Tests and their assertions remain unchanged. The first build's
commands, source/binary hashes and logs are in the execution directory's
`native-validation-primary/progress.json`; the admission red log is retained
under `evidence/runs/luna/remote/PRIMARY-20261012/`.
HIP and SYCL compilers were not found on the execution host's PATH; those builds
remain unverified. The long certification campaign remains disabled.

## Native repair follow-up, 2026-10-12

R03-ADMISSION passed three unchanged-after-red native tests, the three original
R01/R02/R03 probes, G0 51/51 and four reference mutants. Const callbacks now read
without COW. Standalone capture admission retains its original conservative
per-buffer charge; cache-wide allocation counting still deduplicates actual
payload identities. Published rewrites charge detached payloads and directories.
The unchanged native validation and transfer executables now pass 972 and 1548
checks. The snapshot executable still fails `growth never reallocates the
retained payload`: the test requires pointer retention even when `dirty=3` or
`dirty=0` rewrites that prefix. Sharing the old address for different bytes would
mutate the ancestor. This failure is retained and is not counted as a PASS.

The rebuilt actual CUDA SUT and separate fresh-process oracle passed three
controller-boundary tests and registered G3-004. There were 17 full-vocabulary
comparisons, maximum absolute error 0.0, 175 oracle-computed tokens and 28
retained-image-after-generation byte audits. Six hot appends computed one token
each; measured COW was 216 pages / 452984832 bytes. The maximum tracked VMM bytes
sampled after those appends was 578813952; this excludes model arenas and host
memory and is not a process-wide peak. The profile remains unqualified.
[current-native-validation.json](current-native-validation.json) records the
executed IDs, exact source/binary/model/profile hashes and limits. Raw archive
SHA-256: `56fd895e88e79e31899e5ae5e7fe1b2d30ee492fa2dbd2e89de6d9c095a5310d`.

C01-EXACT adds and validates a read-only native cache lookup for a completely
consumed history. Legacy request matching remains unchanged. One CPU lookup
test, three original host probes, G0 and mutants passed. Its first new fixture
omitted its steering setting; that failed run was retained, the fixture setup
was corrected, and a fresh red/green cycle preserved the same assertions.
This lookup is not yet connected to the native inference driver. It was added
after the CUDA model measurements; those measurements pin the preceding source.

[queue-followup-validation.json](queue-followup-validation.json) records both
follow-up tickets, including red and failed runs. Their ledger lives in the
owned `primary-queue-20261012T031643-admission` execution directory. Raw archive
SHA-256: `5bda1950cafaed03c8e05c8a27a230f843cb5b1c8e7f0423cb9192d7f6ab8c1d`.
Next: connect exact native cache lookup and checked snapshot restore to the
canonical API, then run forced sibling/fresh-oracle and cache-miss tests.

## C01-COLD, 2026-10-12

PASS: `test_cold_resident_identity_and_no_replay` on llm-49 / r4090.
The checked disk checkpoint's full-history identity now survives replacement
with a resident native image. The saved payload measured 119371292 bytes,
including 118378012 native-state bytes; its SHA-256 matched the manifest.
The independent nine-token fresh forward matched exactly (maximum logit error
0.0). Both the cold restoration and the next hot materialization computed zero
prefix tokens. G0 51/51 and four reference mutants passed. No skips or xfails.

[cold-identity-validation.json](cold-identity-validation.json) records executed
IDs, commands, binary/profile hashes and measured bytes. Raw archive SHA-256:
`94a55bdb528ce357db21bd150afc7b3edf6996b97ac3580d852a9fe4dfe2ba32`.
Initial command-environment and test-cleanup errors were retained separately
from the clean red assertion. Assertions were unchanged between clean red and
green. This test used the existing pinned Q8_0/FP16 binaries and an unqualified
profile; it does not certify G1 or G2. Native RAM-cache inference integration
remains the next ticket.

## C01-RAM, 2026-10-12

PASS for the new native RAM-cache adapter test on llm-49 / r4090. Actual
`ConversationCache` capture stores numerical state and consumed frontier;
`BranchSession` validates complete history and runtime geometry before restoring
private device state. Controller/RNG/grammar remain in the canonical store.
Six fresh-process comparisons (two different forced two-token suffixes, before
the suffix and after each consumed token) matched exactly. Both RAM restores
computed zero prefix tokens. A changed full prefix returned a miss without
changing the active consumed position. Cache clear released all K/V payload
allocations. Measured cache capacity charge was 119373844 bytes, including
313344 K/V payload bytes; this excludes allocator/control-block overhead.

New CUDA binaries were built and fingerprinted. Four unchanged native
executables passed 434 session, 972 validation, 1548 transfer and 4219 cache
checks. The cold-to-resident regression passed again on these binaries.
G0 51/51 and four reference mutants passed, without skips or xfails.
The profile is unqualified; this is not G1-G5 certification.

[native-ram-validation.json](native-ram-validation.json) records executed IDs,
source/build/model hashes and measurements. Raw archive SHA-256:
`f1401dd2392e86a6ef3abcd539c4337883b4e8f7f91a2c91ee5bcd686ed96df9`.
The native RAM cache is private to one fixed-model process and one active
lease. The adapter is exercised from checked canonical history leases, but
automatic canonical cache-miss routing and ordinary concurrent conversation
slots are still outstanding. Full checkpoint eviction now clears native RAM
entries as well as hot handles and disk checkpoints.

## Automatic canonical materialization, 2026-10-12

C01-AUTO and C01-DUPLICATE passed their executed checks. `BranchingModel`
now parks consumed numerical histories and checks the native RAM cache when a
hot/disk checkpoint is absent. A RAM miss replays the canonical history. The
adapter without a native history cache explicitly returns a miss. Detecting a
misbound checkpoint clears native history acceleration before rebuilding.

Nine fresh-process comparisons through the canonical API had maximum absolute
error 0.0 and 87 oracle-computed tokens. A hot-cache eviction restored from RAM
with zero prefix forward work. A different full prefix missed and computed its
nine tokens. Removing every hot/RAM/disk numerical checkpoint left the immutable
root intact and replayed its nine tokens. Each forced sibling append computed
one token. G0 51/51 and four reference mutants passed.

Regression testing exposed repeated parking retaining duplicate unpinned live
histories. Exact duplicates with no intermediate checkpoint history now replace
the old entry; pinned roots and differing history/image/steering keys remain.
The original RAM test first failed with two entries; after the native rebuild
it passed unchanged, including six exact fresh comparisons and actual allocation
release. The original C01 binding and C01-COLD identity tests also passed.
Four unchanged native executables passed 4219 cache, 972 validation, 1548 transfer
and 434 session checks. Original R01/R02/R03 probes passed.

[automatic-ram-validation.json](automatic-ram-validation.json) records executed
IDs, errors retained during development, source/build/model fingerprints and
measurements. Raw archive SHA-256:
`5b0f90aee8b05bcc4bc5bf381a9dc2b4e9ca1654c402f69117edafc942706cb5`.
The latest tested SUT and oracle copies are preserved in `native-auto-primary/bin`
under the owned execution directory. The profile remains unqualified. No G1-G5
PASS, ordinary serving parity, concurrent-kernel support, host-prefix sharing
benefit or latency improvement is claimed. Next: S01 slot ownership/scheduling.

## Serialized slots and ownership, 2026-10-12

S01-OWNER and S01 passed two real-CUDA implementation tests on llm-49/r4090,
Q8_0 weights / FP16 KV. Foreign-thread reads, writes and release attempts fail
before changing the lease; its original owner continues and closes normally.
The FIFO executes two canonical branch references sequentially, preserves
independent RNG cursors, and returns only after a durable CAS branch move.
Seven fresh-process full-history comparisons had maximum absolute error 0.0
and 70 oracle-computed tokens. FIFO execution measured 2.0762215069844387
seconds for two jobs; this is not an end-to-end latency benchmark. Green runs
had zero failures, errors or skips. Linux G0 passed 51/51 and four mutants
were detected. Assertion-red runs remain in the evidence.

[serialized-slot-validation.json](serialized-slot-validation.json) records
executed IDs and backend/build/model artifact fingerprints. Raw archive SHA256
`c76ce63086385e0253dbde9965c9ae4d2b47d49bf8254bc10b87bae523cd735d`.
The user confirmed serialized execution is the first prototype priority.
Ordinary serving integration, long contexts and profile qualification remain
unexecuted; no G1-G5 PASS is claimed. Next: cancellation and slot reuse.

## S02 cancellation and recovery component, 2026-10-12

`test_s02_invariant` passed on llm-49/r4090 after its recorded assertion red.
Queued cancellation releases only the identified pending job without token
consumption. Slot reassignment then succeeds. Stale revisions reject before
GPU work, preserve retry receipts, and clear queue activity. Injected interruption
before the second snapshot publication leaves the branch on the original
generation-ready history with its original RNG cursor; later work succeeds.
Complete hot/RAM/disk checkpoint eviction replays the nine-token root and leaves
root and sibling canonical bytes unchanged. Five fresh forwards were exact
(maximum error 0.0; 48 oracle tokens), zero green failures/errors/skips.
G0 passed 51/51; all four reference mutants were detected.
[slot-recovery-validation.json](slot-recovery-validation.json) contains executed
reports, retained red and numerical measurements. Real cache-admission pressure also passed: ten distinct retained histories
forced the eight-entry cache to evict numerical states. At measurement it
held 954,990,752 capacity-accounted bytes and 2,506,752 KV payload bytes
across 288 allocations. The oldest history replayed nine tokens and matched
its fresh oracle exactly. These counters are not process RSS. Final green,
G0 and mutation runs passed; S02 is VALIDATED for the recorded local API
component scope. Every-frontier process kills, GPU OOM and randomized
concurrent interleaving remain outstanding acceptance qualification. The
publication injection is not a process-kill or GPU OOM measurement.

## Fresh normal-prefill diagnosis, 2026-10-12

N01's diagnostic test passed after its recorded red. Eight new processes
measured the historical 127-token counterexample and a one-token control on
llm-49/r4090, Q8_0 weights / FP16 KV. Actual counters report the complete
history: normal prefill consumed n-1 tokens and the final verifier consumed
one; the independent token-level oracle consumed n tokens from an empty
session. All 248,320 logits per forward are retained.

The current normal chunk-1 vs chunk-128 error at length 127 is
1.3281426429748535; chunk-31 vs chunk-128 is 1.259131669998169.
Normal vs token-level maximum error is 2.688899040222168. Even the one-token
control differs by 1.1226537227630615, with no batched-prefix work. Top-1
matched in these six comparisons. These are absolute logit errors, not
percentages. The historical 1.5612177848815918 counterexample is preserved.

G0 passed 51/51 and four reference mutants were detected. No tolerance was
fitted or admitted. N01 validates diagnosis only; normal batched equivalence
remains unresolved. Source inspection identifies differing expert-dispatch
initialization as a hypothesis for the one-token discrepancy, not yet a
measured cause. [normal-prefill-validation.json](normal-prefill-validation.json)
records full-vector hashes, work counters, binaries and model fingerprints.
Raw archive SHA256: `189e983b183b1ed156c87c8f1d9eaa1595a204622d5a4547ba120ad6f52cdc67`.

## Corrected omitted resident-expert work, 2026-10-12

The private branch and fresh drivers had a shared inference defect: neither
attached `ExpertDispatch.plan` to the verifier. CPU dispatch classified resident
experts as GPU work and zeroed their CPU rows, while the verifier received an
empty GPU plan. Earlier exact branch/oracle comparisons therefore validated
isolation relative to the same defective operator; they do not prove ordinary
Strata model equivalence and cannot qualify the corrected build. Their raw
measurements remain historical evidence.

The driver now attaches the actual verifier plan, honors the parsed PCIe mode
and share, and clears the borrowed plan on return. A real one-token regression
first failed four comparisons by 1.1226537227630615. The corrected branch and
fresh drivers matched ordinary Strata exactly for both explicit share settings
(0 and 0.55): four maximum errors of 0.0. Each private forward reported two
resident expert entries and 478 CPU expert entries. The mmap configuration
performed zero PCIe expert groups in both settings, so PCIe execution itself
is not validated by these runs.

Corrected pinned executables are under `native-dispatch-primary/bin` on llm-49.
The new execution profile ID is
`397394a2a9a0103b9db198f3d55aba4689f0a44377d8335942c9c8a42e4ecdd1`.
The fresh executable contains no BranchSession, BranchPages or branch-driver
symbols (`nm -C`, executed). The final supporting suite passed eight tests, including one controller-only
test: 34 exact full-logit comparisons, of which 30 use fresh full-history
forwards (296 oracle tokens) and four compare private drivers to ordinary
one-token execution. Linux G0 passed 51/51 and all four reference mutants were
detected. Green suites had no failures or skips. A prior regression invocation
failed on two missing measurement-path environment settings; it is retained
alongside the successful rerun. RAM hits required zero prefix work; changed
prefix and complete-eviction misses each replayed nine tokens.

[dispatch-correction-validation.json](dispatch-correction-validation.json)
records executed IDs, final run selection, corrected build/model fingerprints
and measurements. Raw archive SHA256:
`fbbb697bfae7e95d28dd4b8ee2ae05a65a73fbc7d3eb235bcd12f79fe0f140d4`.
Normal batched-prefill equivalence and numerical admission remain unresolved.
HIP/SYCL builds were not executed; no G1-G5 gate PASS is claimed.

## Corrected full-history prefill diagnosis, 2026-10-12

N02-BASELINE completed eight new full-history forwards on the corrected
execution profile. One-token ordinary-versus-fresh comparisons were exact
for chunks 1, 31 and 128. At length 127, the respective maximum absolute
logit errors were 1.6786870956420898, 1.6211271286010742 and
1.595560759305954. Top-1 matched in each comparison; that does not establish
logit equivalence. Actual counters total 512 consumed forward tokens, with
126 prefill plus one verifier token on each ordinary 127-token forward.

The observation took 266.93 seconds, with no process failures or skipped
forwards. It executed zero registered acceptance tests. No new memory benchmark
was collected. N02 remains unresolved, not a gate PASS; this observation
introduces no tolerance or implementation change. See
[corrected-prefill-validation.json](corrected-prefill-validation.json) for
binary/model fingerprints, complete fixture, errors and full-vector hashes.
Raw archive SHA256:
`c161acb1eb491f48455d956429fbdacc9525c259175a995a62f007b56466a40f`.

## Two-token prefill counterexample, 2026-10-12

N02-SHORT executed 16 fresh full-history forwards and 12 full-vocabulary
comparisons at lengths 1, 2, 3 and 4. One-token controls stayed exact. The
smallest tested failing prefix is [100,117]: all ordinary prefill chunk
settings (1,31,128) differ from the corrected fresh driver by
4.67782187461853 absolute logits; top-1 is 2972 versus 9304. This is not an
accepted tolerance. The run took 81.21 seconds and computed 40 tokens.

N02-CONTROL attempted the existing ordinary --prefill 0 option. One-token
execution matched exactly, but the native Q8 pack rejected the two-token
ordinary request with exit 2: this path requires prefill. That failed run is
preserved, not skipped or counted as PASS. It took 13.51 seconds.

N02 and its ordinary control are BLOCKED pending a matched ordinary token-only
control or intermediate prefill-state instrumentation. The cause is not yet
proved. This is a backend investigation blocker, not a claim that source
changes cannot solve it. Independent queue work remains available. No G1-G5
PASS, baseline admission or memory measurement is claimed. The diagnostic
length/chunk selectors change only measurement inputs, not engine semantics.
See [short-prefill-validation.json](short-prefill-validation.json) for exact
fixtures, comparisons, failed control, work counts and fingerprints.

After these diagnostic changes, Linux G0 passed 51/51 and all four reference
mutants were detected. Log hashes are included in the short-prefill evidence.

## Ordinary token-step control implemented, 2026-10-12

N02-STEP passed its unchanged regression after the recorded exit-2 red.
The separate CUDA diagnostic target `strata-step` uses ordinary generate-path
verifier initialization and expert dispatch, then consumes the requested
history with T=1 run/commit calls. It does not link the branch or fresh driver
(symbol audit executed). It adds no production serving option.

Eight real forwards at lengths 1,2,3,4 produced four exact comparisons against
the independently linked fresh driver: maximum absolute error 0.0, 20 total
consumed tokens, matching expert counters and position frontiers. The two-token
ordinary control and fresh driver each executed 10 resident and 950 CPU expert
entries. Green took 31.49 seconds, with one test and no failures/skips. G0
passed 51/51 and four reference mutants were detected.

Both CUDA targets built. The rebuilt ordinary `strata` binary is byte-identical
to the preserved normal binary (SHA256
`e5a575d5a7c96a14a7365b6e1ac6bf8e35ef8832b9c9424a1871585a9dc6414d`).
The diagnostic binary SHA256 is
`7b65ed5e0da33bfc055a75ec266e1d256120dcf7c84c2f2d12ded304840989e3`.
See [ordinary-step-validation.json](ordinary-step-validation.json) for the
source/build/model identities, executed runs and measurements.

The missing ordinary token-only comparison primitive now exists. N02's prefill
mismatch remains unresolved; direct state-buffer comparison after the first
token is still missing. No batched-equivalence, memory-benefit or G1-G5 PASS
claim follows. HIP/SYCL were not built; the new target is CUDA-only.

## First consumed state compared, 2026-10-12

N02-STATE added read-only instrumentation to the separate CUDA diagnostic,
with a missing-dump assertion red followed by the unchanged passing test.
After consuming token 100, each path exported 169 components totaling
118081584 bytes: recurrence, convolution, PLE, indexer state and only populated
resident FP16 KV rows. 157 components differ. All 12 indexer block cursors
match; both PLE token windows are [-1,100]. These observations do not establish
that every positional/mask invariant is correct.

The earliest observed divergence is already in GDN ordinal 0: recurrence max
absolute difference 0.0029393713921308517, convolution 0.028350234031677246.
QSA ordinal 0 K-head-0 differs by 0.005859375; PLE history by
0.04052877426147461. The largest component difference is GDN 35 convolution,
0.5618761777877808. These are state-value differences, not logit tolerances.

Both instrumented final logit files match their prior uninstrumented files
byte-for-byte. The ordinary executable remains byte-identical after rebuilding.
G0 passed 51/51; four mutants were detected. The diagnostic test took 13.07
seconds. No acceptance gate, memory-efficiency or prefill-equivalence PASS is
claimed. The dump is deliberately limited to a one-token boundary and resident
FP16 KV; other layouts are rejected. The first responsible arithmetic operation
still needs isolation. See [state-comparison-validation.json](state-comparison-validation.json).

## First projection measured, 2026-10-12

N02-PROJECTION passed its unchanged diagnostic test after a missing-capture
assertion failed. The first-layer mixed input already differs by
0.0020284652709960938 (2531 values); QKV output differs by
0.028350234031677246 (10240 values). The latter equals the first convolution
history maximum difference. Divergence therefore precedes the QKV projection;
these measurements do not establish a convolution-storage defect or isolate
the responsible mixer operation.

All 340 previously captured state and final-logit files remained byte-identical.
The rebuilt ordinary executable also remained byte-identical. The diagnostic
passed in 13.336 seconds; G0 passed 51/51 and four reference mutants were
detected. No gate admission or batched-prefill equivalence is claimed.
See [projection-comparison-validation.json](projection-comparison-validation.json)
for executed run IDs, payload sizes, hashes and limitations. The raw archive
was transferred and its SHA-256 verified locally.

## Corrected operator calibration started, 2026-10-12

N03 recorded an assertion red for the absent frozen baseline. Executed dual-engine
admission passed with nine simultaneous samples and peak combined GPU allocation
17136 MiB. The calibration now selects the actual admission oracle and verifies its
SHA-256, with an explicit source revision instead of the obsolete default. All six
existing model-free calibration-input tests passed. The full nine-length, three-repeat
matrix is running as q8-operator-20261012T073213 on llm-49; no frozen tolerance or
numerical gate PASS exists yet. See n03-admission-validation.json.

N03 interim regression runs N03-0e14968ee33349a18f5981555d9e2385 (G0) and
N03-90f64d663ce3478097a938e99035688a (mutants) passed. These use the original
unqualified profile binding; final frozen-profile validation is pending. A direct
audit of the first seven completed forwards checked saved logit SHA-256, finite
values, exact input histories, and consumed/forward counters. The calibration
process remained live; no rerun was launched.

## Private build serving smoke, 2026-10-12

At the user's request, the normal persistent serving engine was exercised through
StrataEngine with real Q8_0 weights and FP16 KV, using the unchanged existing
conversation_cache_parity.py reuse assertions. Seven requests passed in 124.62
seconds: two baseline requests with parked-conversation cache disabled, and five
candidate requests covering A, B, A continuation, B return, and checkpoint return.
A continuation restored 33 prompt tokens after B; subsequent B/A returns reused
28/48 tokens. Generated IDs and required main-model state fingerprints agreed.
The baseline still uses live-prefix reuse and is not a completely uncached oracle.

This used the ordinary binary SHA-256 e5a575d5a7c96a14a7365b6e1ac6bf8e35ef8832b9c9424a1871585a9dc6414d,
not either private driver. The ordinary runner already binds its expert plan.
No additional serving wiring fix was needed for this smoke. No HTTP listener
was started. Full-vocabulary logit equivalence, batching and broad production
qualification remain unproven. Calibration yielded between requests and its
hold was removed after the serving processes exited. Evidence:
[private-serving-validation.json](private-serving-validation.json).

## E01 brought forward during calibration, 2026-10-12

With user authorization to advance independent work during N03, evaluation
provenance storage now has an executed CPU-only assertion red and unchanged
green test. Reports use content-addressed canonical JSON in a separate evaluations
directory, with hash verification, no-overwrite publication and ancestor checks.
The tests cover reopen, detached reads, source/score identity, malformed records,
tampering and unchanged snapshot/reference bytes. G0 passed 51/51 and four
reference mutants were detected. The registered G4-008 method is written but
unexecuted; it requires a frozen profile and actual model/oracle continuations.
E01 is preparatory work, not a validated queue ticket or a gate PASS.
See evaluation-records-prework.json. N03 remains the active validation ticket.

## 2026-10-12 llm-60 supporting GPU checks

Two real-backend supporting unittest methods passed, zero failures/skips: ordinary forced-step versus fresh oracle (eight forwards, four exact comparisons, 20 consumed tokens) and evaluation-record noninterference (33-token prefix plus three-token suffix, two fresh oracle requests/69 tokens). Maximum absolute error was zero. Snapshot bytes and branch head stayed unchanged after record publication/reopen. Remote raw artifacts were copied and every file SHA-256 verified. See `llm60-supporting-validation.json`. This does not qualify G4 or complete N03/E01: baseline calibration is still running and registered acceptance remains pending. Normal batched-prefill equivalence remains unresolved.

## 2026-10-12 E02 isolated evaluation prework

Executed deterministic E02 red assertion on llm-49, then passed the unchanged test with a Linux bubblewrap evaluator worker. Candidate inputs recursively read-only, recorded fixture bytes provided without executing tools, host store unmounted, network namespace isolated, independent RNG seeded per evaluation. Test probes blocked host-file writes/network access and distinct ERROR/UNKNOWN/FAIL/TIMEOUT outcomes. G0 passed 51/51 and all four reference mutants were detected. See `evaluator-isolation-prework.json`. No registered G4 acceptance run or ticket VALIDATED claim; GPU/controller noninterference and report composition remain pending.

## 2026-10-12 E02 report integration

Isolated scoring now composes into the existing immutable evaluation record and persists through Store.put_evaluation. Executed assertion red then unchanged green for report/snapshot identity, source/fixture/runtime fingerprints, reopen, foreign task-root rejection, unchanged controller/history/branch reference. Both E02 CPU tests passed, G0 51/51, mutants 4/4. Four registered G4 methods (003/004/005/010) are written but unexecuted pending frozen calibration. No gate PASS or queue VALIDATED claim. Evidence: `evaluator-record-composition.json`.

## 2026-10-12 E03 validated-candidate selection prework

Executed red then unchanged green for validity-first selection, all-failed/unknown/no-report cases, missing checks, unresolved checks, stable ties, incompatible specifications, conflicting judgments, retry receipt and stale CAS rejection. All six A/B/C input orders select C (8), excluding B (10 with FAIL). G0 51/51; reference mutants 4/4. Added registered G4-006/007 with real model histories and an actual spawned-process CAS race; not yet executed. Evidence: `evaluation-selection-prework.json`. This remains prework while N03 calibration runs.

## 2026-10-12 E04 four-future workflow preparation

Executed the missing-real-demonstration red assertion on llm-49. Added run_real to the existing demo plus four registered G4 methods: A/B/C and rewound D, independent persisted evaluation, validity-first C adoption, saved-checkpoint eviction, native engine restart, fresh oracle at every suffix boundary, and direct-counter cold reconstruction of winner/losers. Syntax checked only after implementation. Green and all four registered cases remain UNEXECUTED pending frozen calibration. All 11 registered G4 methods now exist; this is method coverage, not passing acceptance. Evidence: `evaluated-futures-prework.json`.

## 2026-10-12 evaluator host prerequisite and E04 regressions

E04 addition preserves the standalone toy demo; executed toy command passed, G0 51/51 and mutants 4/4 on llm-49. Real E04 green remains unexecuted. On llm-60 the unchanged E02 isolation test FAILED (one failure, zero skips): expected PASS but sandbox returned ERROR. Kernel audit confirms AppArmor unprivileged_userns denies bwrap setpcap/net_admin; network namespace creation fails. Evaluator-dependent G4 is BLOCKED on llm-60 under current host policy. llm-49 has executed working isolation; prioritize llm-60 for inference and run evaluator-dependent acceptance on llm-49 after its own baseline qualification. No security policy was changed. Evidence: `llm60-evaluator-prerequisite.json`.

## 2026-10-12 N03 finalization validation

Added a finalizer to the existing profile builder that preserves the pinned build identity while attaching calibration evidence. Executed red then unchanged green for rejection of a saved incomplete real run and the same incomplete rows falsely labeled FROZEN. No profile output is permitted on either failure. Full path verifies model/build/admission fingerprints, all 27 raw frontier files and exact token inputs, direct counters, 18 recomputed errors and the predeclared tolerance rule. Completed-run path remains unexecuted until calibration finishes. G0 51/51 and mutants 4/4 passed. Evidence: `calibration-finalization-check.json`.

## 2026-10-12 P01 authorization red

Executed test_p01_invariant on llm-49: expected AssertionError because Store.for_principal is absent. Test covers unauthorized known-ID snapshot/history/fork/reference/append/evaluation access, grant escalation, root-specific read grants, grant persistence/revocation and zero snapshot payload bytes read on denied operations. Implementation remains pending; no G5 method/pass claimed. Evidence: `authorization-prework.json`.

## 2026-10-12 llm-60 evaluator prerequisite resolved

User explicitly authorized global protection changes on this test box. Changed only runtime kernel.apparmor_restrict_unprivileged_userns from 1 to 0; AppArmor remained enabled (Y), no reboot or calibration restart. Both unchanged E02 CPU methods passed (2 tests, zero failures/skips), including host filesystem/network isolation and immutable report composition. llm-60 is available again for evaluator-dependent G4 after its frozen baseline; earlier BLOCKED host status is superseded, original failure retained. This is a host prerequisite PASS, not G4 acceptance. Evidence: `llm60-evaluator-host-fix.json`.

## 2026-10-12 P01 principal-bound store prework

Unchanged P01 CPU red now passes: same underlying immutable store, durable root grants, restricted identity binding, denied reads before snapshot payload access, fork/move/append/evaluation checks and revoke-after-reopen. Added model access checks before cache restore/forward/checkpoint access. G0 51/51; mutants 4/4; both evaluator CPU regression methods passed. Added registered G5-006 testing hot-cache denial and direct work counters; unexecuted pending frozen profile. History deletion authorization remains pending its GC API. Evidence: `authorization-prework.json`.

## 2026-10-12 P02 sensitive storage permissions

Executed red with umask 0: newly created store directory was 0777 rather than required 0700. Unchanged test now passes with private directories and 0600 files, including grants, evaluations, references and interrupted staging. Real reads as nobody (uid 65534) fail with PermissionError for history/evaluation/staged bytes. P01 regression also passes; G0 51/51; mutants 4/4. Native checkpoint directories now use the same private directory helper. Registered G5-007 written but unexecuted. Declared policy is owner-only plaintext, explicitly not encryption. Evidence: `sensitive-storage-prework.json`.

## 2026-10-12 P03 retention and read leases

Added named historical pins, branch/evaluation reachability and POSIX flock-backed active read leases. Materialization and evaluation hold a lease. A real spawned child proved active pin visibility and automatic release on termination. Initial test cleanup hung in multiprocessing.Event.set after killing the waiter; preserved interrupted traceback, removed that cleanup operation without changing assertions, and reran corrected test red against 214cc3e8 then green. Three P01-P03 tests, two evaluator regressions, G0 51/51 and mutants 4/4 passed. No history deletion/GC yet; GPU lease integration unexecuted. Evidence: `retention-leases-prework.json`.

## 2026-10-12 P04 explicit history collection prework

Executed P04 red then unchanged green. History is retained by default; release_history makes an identity eligible, and owner collection sweeps only released unreachable records in the requested root. Test preserved live-head ancestors, named pins, evaluation candidates and a real child-held lease; collected the released orphan and later the released lease snapshot. Create/append/fork/lineage/evaluation publication now share a reentrant reference transaction lock with collection. G0 51/51; four P01-P04 tests; two evaluator regressions; mutants 4/4 passed. Crash/fork/append schedule tests and GPU cases remain outstanding. Evidence: `history-collection-prework.json`.

## 2026-10-12 P04 process fault validation

Two process-based tests passed on llm-49: eight competing fork/append-versus-collection schedules preserved valid histories; os._exit(42) after first physical unlink preserved remaining ancestry and a committed retry receipt, and a reopened collector completed cleanly. Reversing deletion order in an isolated source copy was rejected by the unchanged crash test (AssertionError), demonstrating the test detects parent-first deletion. Added G5-008/009 real GPU materialization/collection tests; unexecuted. Concurrent evaluation/rewind schedule coverage remains to expand. No gate PASS. Evidence: `history-collection-processes.json`.

### P04 evaluation and rewind schedules (2026-10-12)

Four CPU process tests passed on llm-49: eight fork/append races, four rewind races, collector crash recovery, and collection while public evaluate holds its actual history lease and after report publication. An isolated mutant removing the evaluator lease failed the new test. G0 passed 51/51 and reference mutation smoke caught 4/4. No native code changed. Registered GPU qualification remains pending the frozen calibration. Evidence: `history-collection-evaluation-rewind.json`.

### Registered G5-009 concurrency coverage (2026-10-12)

Expanded the registered case to synchronize collection, fork, rewind and real isolated evaluation across four processes while a GPU materialization holds the candidate. It requires unchanged canonical bytes, correct revision/receipt, valid evaluation and post-concurrency fresh-oracle agreement for candidate/root. This revision is syntax-checked only, not executed: frozen calibration remains pending. CPU process evidence above is supporting evidence, not a substitute for this registered run. Manifest: `registered-collection-concurrency.json`.

### U01 checkpoint compatibility prework (2026-10-12)

Strict manifest validation now rejects undeclared fields, unknown schema versions, Boolean versions/sizes, mismatched contracts/profiles/positions and invalid payload identities before native deserialization. Executed assertion red then unchanged green on llm-49; isolated Boolean-version mutant caught. G0 51/51 and reference mutants 4/4 pass. G5-003 is written but unexecuted: reject incompatible saved state, explicitly evict derived checkpoints, cold replay and verify canonical IDs/controller bytes/revision plus fresh logits and work counters. No numerical conversion or gate PASS claimed. Evidence: `checkpoint-upgrade-prework.json`.

### U02 rollback and receipts prework (2026-10-12)

Executed four red assertion failures for numeric/Boolean schema acceptance; strict reference marker, branch, order and legacy migration version checks now pass unchanged tests. Actual child processes exited 73 before/after branch rename; retry produces one receipt/revision, payload mismatch and stale revision reject, canonical snapshot bytes unchanged. Five collection regressions, G0 51/51 and reference mutations 4/4 pass. Registered G5-010/G5-012 written but unexecuted. Actual two-release executable rollback remains outstanding; format simulation is not that measurement. Evidence: `rollback-receipts-prework.json`.

### llm-60 current CPU regression run (2026-10-12)

Copied 138 hash-verified source/test files at d3d67a24 to isolated `python-validation-d3d67a24`, leaving running numerical calibration untouched. First combined run: 13 pass/1 fail, because E01 recursively counted P01 authorization index as a report. Corrected the CPU test to require one canonical report, one snapshot-only access index and no other matching files; retained hash/corruption assertions. Re-execution passed 14/14, G0 51/51 and reference mutations 4/4. Original failure and source overlay hash preserved in `llm60-current-cpu-validation.json`. No GPU gate claim.

### Q01 explicit draft matrix (2026-10-12)

Added `docs/support-matrix.json`: one pending Q8_0/FP16/T=1/Linux/ext4/CUDA/token-ID/serialized-lease target with ten explicit unsupported, unmeasured or failed categories. CPU declaration test red then unchanged green; llm-60 G0 51/51 and reference mutants 4/4. Registered G5-001 written, unexecuted. Host filesystem observation found llm-49 `/tmp` is tmpfs although its workspace is ext4: prior default temporary CPU stores there exercised tmpfs, and future ext4 matrix runs must set TMPDIR to an ext4 test directory. llm-60 workspace and `/tmp` are ext4. Evidence: `supported-matrix-prework.json`.

### Q02 generator/reducer substep (2026-10-12)

Seeded generator emits append/fork/reseed/generate/rewind/checkpoint/evict/evaluate/retry/release/GC instructions referencing prior results. CPU test checks 10 reproducible 40-step programs and a dependency-sensitive reduction counterexample. Executed red then unchanged green; mutant accepting a different failure signature was caught. llm-60 G0 51/51 and reference mutants 4/4 pass. This is only the generator/reducer: real interpreter, independent logical oracle and GPU campaign remain incomplete. Zero real state-machine sequences executed; registered >=1000 requirement unchanged. Evidence: `state-machine-helper-prework.json`.

### Q02 independent logical oracle substep (2026-10-12)

Added a test-only graph model using tuples/dictionaries/work lists, without Store/backend imports: sibling token histories, immutable node mapping, branch revisions, idempotent receipts, pins/releases and ancestor reachability. Executed assertion red then green on Windows; both Q02 CPU tests also pass on llm-60. Removing receipt roots from the isolated reference model was detected. Real operation interpreter/numerical campaign still pending; zero registered sequences executed. Evidence: `state-machine-logical-oracle.json`.

### Q02 real interpreter written (2026-10-12)

Added actual BranchingModel/FreshOracle operation execution with independent graph comparisons, per-boundary logits, immutable-byte checks, branch/receipt checks, isolated evaluation, checkpoint/evict/release/GC, and persisted operation/error/counter reports. Singleton-support generation checks exact token and draw frontier; reseed checks key/cursor. Missing interpreter assertion red executed; implementation is syntax-checked only and real green remains pending frozen profile. Campaign/failure reduction orchestration and registered >=1000 method are not complete. Evidence: `state-machine-real-interpreter.json`.

### llm-60 numerical baseline frozen (2026-10-12)

Completed all 27 fresh-process forwards and 18 repeat comparisons. All comparisons have maximum absolute error 0, freezing atol=0/rtol=0 by the unchanged 2x-baseline rule. Finalizer independently read/hash-checked all raw float32 frontiers, token fixtures, model receipts, commands, binaries and consumed-work counters. Original finalization failed on descriptive operator text; exact known descriptions are now accepted only after identical executable hashes/commands, preserving execution/model IDs and both descriptions. Unknown description rejection, N03 profile test, G0 51/51 and reference mutants 4/4 pass. Cumulative G1 is RUNNING on current primary checkout; calibration alone is not gate PASS. Evidence: `llm60-frozen-calibration.json`.

### G1 campaign deliberately interrupted for draft scope (2026-10-12)

Stopped the verified G1 process tree after 28 completed numerical comparisons, max absolute error 0.0. The first registered boundary test did not finish; zero completed registered methods and no G1 PASS. G0 had passed 51/51. Original stdout/stderr, traces, measured bytes/counters and SIGINT receipt preserved. Prioritized focused GPU checks consistent with the requested rough prototype validation. Evidence: `llm60-g1-interrupted.json`.

### Real eleven-operation smoke PASS (2026-10-12)

llm-60 completed append/fork/reseed/generate/rewind/checkpoint/evict/evaluate/retry/release/GC: 11 operations, 14 fresh-oracle boundaries, 464 oracle tokens, maximum absolute error 0 with frozen atol=0/rtol=0. Native work consumed 68 tokens; 33 leases acquired/released; zero integrity mismatches and VMM failures. Test elapsed 537.328 seconds, initially overlapping G1. Retained store/report/traces on host; local raw report hashes recorded. One deterministic smoke sequence only, not registered G5-002 or gate PASS. Evidence: `llm60-state-machine-smoke.json`.

### Bounded serial speed comparison (2026-10-12)

Built untouched upstream base 7695859a and ran eight serial requests on each of untouched serving, modified serving and the immutable branching API on llm-60/Q8_0/FP16 KV. All 24 requests completed, output tokens agreed across all paths, and eight API consumed frontiers matched independent fresh forwards with max error 0 at frozen tolerance 0. Actual disk reload read 119010952 bytes; complete eviction forced 38 forward tokens for the 36-token B history plus two generated tokens. Summed request times: stock 42.122s, modified serving 42.181s, API 50.345s. Startup separately 3.463s, 3.233s, 2.596s. Warm API requests improved but the complete mixed hot/cold sequence was slower. Different prefill operators and cache budgets prevent an isolated cache-only speedup claim. One fixed-order short fixture; no performance certification. Harness regression 1/1, G0 51/51, reference mutants 4/4. Evidence: `llm60-serial-speed.json`.
