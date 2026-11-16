# Acceptance producer preparation, 2026-11-16

Preparation scope is four bounded adapters in acceptance.py, the source Harness
runner acceptance-harness.mjs, and synthetic boundary tests test_acceptance.py.
Base is b0df9b67f78e75ef78e9b8a8dec5b9e45751d3cb on the primary C main checkout.
No commit or push is performed by this worker. Other work is preserved.

The canonical PC prompt, PC/project AGENTS, common engineering, implementation
start, skill selection, security, application, Python and TypeScript SOPs were
read. Applied skills are codex-session-style-and-standards and
milspec-codex-standard. Codebase-inspection was inspected and found to cover LOC
analysis, so its pygount procedure was not applicable. This record maps the
requested acceptance boundaries to implementation and tests; it makes no formal
MILSPEC compliance claim.

## Contract and invocation

All modes use native-input.py's parser/run_controller and release.py's current
guard. Default invocation is read-only: `acceptance.py ci`, `harness`, `nsis`, or
`final-native`. Execution additionally requires `--execute --reviewed-commit
FULL_SHA --attempt rN`. Clean primary main, current source before/after, a fresh
attempt, contained Windows Job and drained success are enforced by the existing
controller. The adapter retains RAM 3 / commit 4 / H 10 / C 5 GiB floors, compiler
jobs 5 and test jobs 1. Trusted PowerShell paths extend the existing filtered
environment; ambient credentials/tuning are not copied to children.

The producer writes the existing `receipt.json`, then only after a successful
drain writes `acceptance.json`. The public consumer API is
`validate_acceptance(path, expected_sha, mode, commit) -> dict`. Callers must pass
the absolute acceptance.json path and its independently frozen SHA-256. Schema is
`strata.acceptance.v1`; phase is `acceptance-{mode}`. Required identity fields
include sourceCommit, sourceRoot, reviewedSourceRoot, mode, scope and
releaseQualified=false. inputs/results map logical names to absolute path and
sha256. receiptSHA256, ownedJobSHA256 and workerLogSHA256 bind the observed local
Job and retained log; worker-command.json records the exact adapter invocation.
The consumer checks exact raw-receipt equality, source/phase, drained Job, resource
samples, output truncation and all bound artifact hashes. It does not accept a
filename, an ok flag, a name count, or stale source as evidence of completion.

CI reads actual `gh run list/view --repo zapabob/Strata` metadata at the exact
reviewed source and main branch. Nine workflow file/name identities and expanded
job multisets are frozen with current YAML byte hashes: Checks (13), Docs (1),
Backend compile checks (5), Native preview CPU checks (2), Portable monitor client
(4), Minimal serving runtime (1), Config snapshot (2), Deploy GitHub Pages (1),
Metal build (1). All 30 jobs must be completed/success. Missing, duplicate,
unexpected, skipped/cancelled or ambiguous workflow runs are refused. No workflow
is dispatched. Intake and Docker publishing are excluded. This is explicitly
remote-state observation; the local Job runs the metadata reader, not hosted CI.

Harness verifies managed resources/harness commit
571902028a75c64c1d767f4ae5fd099485f0ddf0, current budget patch and import closure,
and freezes its inventory before/after. Seven current source test files comprise
the CI group plus agent-compaction and agent-modes. Literal source test names,
files, runtime, commands and input/result hashes are recorded. Actual TAP requires
every expected name, full counters and no failures, cancellations, skips or TODOs.
Node 24.22.3 uses concurrency 1, private scratch/home directories and the production
test-fs-guard/test-environment helpers. Synthetic local model endpoints are test
fixtures; no real model, runtime acquisition or persistent service is requested.

NSIS invokes production tools/test_nsis_upgrade.ps1 with the trusted PowerShell 7
path and explicit installed makensis, using a fresh attempt-local fixtures tree.
All six scenarios require the actual expected cleanup, sentinel, exit and
uninstall observations. The production script/installer, tools, result JSON,
recorded source hash and command log are bound. No real app installation occurs.

Final-native requires each external input as `--NAME-receipt ABSOLUTE_PATH` and
`--NAME-receipt-sha256 SHA256`, for NAME native, component, native-job,
component-job, gpu-binding, gpu-report, gpu-guard and gpu-junit. Native/component
receipts must be current successful attempts under the approved release root;
their explicit Job pins must identify those same attempts. It uses the production
native verifier, component signature verifier and PE closure checks. Fresh native
configured inputs/profile/build metadata, the nine measured compiled-library
hashes, separate installed toolkit PE dependency hashes, both executable hashes
and actual version probes are checked. Signed component
files are individually bound for later consumers. It performs no build/sign/GPU
replay itself.

GPU carry is restricted to the four exact approved historical evidence hashes
from the brief. JUnit must contain the ten exact required run cases without
errors/failures/skips. The historical guard, command log, admission, sampled
resource minima and VRAM reserve are checked. The historical single sourceDigest
and sourceUnchanged attestation are labelled as recorded; no invented historical
sourceBefore/sourceAfter hashes or whole-tree HEAD are asserted. Both actual
product binary before/after hashes must equal the final signed component binaries,
and current source_digest must match the historical digest. A different final
binary refuses reuse and requires the parent to arrange fresh bounded GPU proof.
Generic spill-default two-IQ2 failures remain retained; only the explicit bound
Windows profile is checked. Model quality, full context and concurrency remain
unqualified and releaseQualified remains false.

## Verification and remaining work

Preparation verification passed after the existing resource floors recovered.
During preparation the parent observed RAM below 3 GiB and this worker independently
observed 231,664 KiB free. No local test, fixture or build process was started while
below the floor. Existing processes were not stopped by this worker. The parent
later confirmed resource recovery after the user's authorised ordinary stop/unload
and explicitly returned the serial test slot.

All Python commands used the existing offline uv/private Python 3.12 and
_docs/upstream-sync-20261116/check.py. The outer command was
`C:/Users/downl/.local/bin/uv.exe run --no-project --offline --python
C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python -B
_docs/upstream-sync-20261116/check.py --seconds 300 ATTEMPT COMMAND...`.
Final pytest command was `python -B -m pytest -q
_docs/stable-060-20261116/test_acceptance.py
--junitxml=H:/Strata/evidence/upstream-sync-20261116/acceptance-preparation-tests-v3.xml`.
It passed 68 tests in 4.93 seconds, without failures/errors/skips. The full Job
drained successfully with exit/native-exit 0, containment-before-resume true,
termination-confirmed true, active-after 0 and zero dropped output bytes. Sampled
minimum RAM was 15,713,783,808 bytes, commit headroom 37,246,377,984 bytes and
H free 84,435,247,104 bytes; last C free was 34,045,227,008 bytes.

The initial pytest v1 result was 1 failed / 67 passed: the TAP directive regex
incorrectly matched the `# todo 0` summary. It now recognises SKIP/TODO directives
only on test-result lines. The retained v1 receipt/log/JUnit and successful v2/v3
evidence remain separate. The final v3 receipt SHA-256 is
827fc2fbee492e0bf8b4ba3f2eea69d600c13a12edede48f888ca5e90d78b440;
its command log SHA-256 is
3ff4cf47a887f78d44546663dca95ef39adb2baf3715a41667eead5ae4640fc9;
JUnit SHA-256 is dad7e7399cfe5e6b582a481a0bc38f2585a6a86043b7b52843f3b85f4899e0d3.

Black with line length 110 and its final --check passed. Flake8 with maximum line
length 140 initially found three long string literals; splitting the literals
resolved all three, and final lint passed. Guarded Node 24.22.3 --check passed for
acceptance-harness.mjs. Each check has a distinct acceptance-preparation-* receipt
under H:/Strata/evidence/upstream-sync-20261116/receipts. Existing uv warnings about
dangling ~astapi_users/~ip metadata were retained and not repaired. No dependency
was installed or resolved online. Default read-only CLI plans for all four modes
were exercised by the synthetic tests.

Actual CI collection, actual Harness execution, actual NSIS fixtures and actual
final-native verification are NOT_RUN in this preparation task. The parent runs
them only after final clean source freeze and independently reviews the adapters.
Current stale cuda-build native-build.json is not treated as an attestation of
current executable bytes; native-input must produce fresh metadata later.

The private handoff directory is
H:/Strata/private-backups/cleanup-20261116/acceptance-producers-v1. It preserves the
brief, before/after status, base, scoped diff, final report, source copies and
verification receipt/log digests. All new source files were absent at the base;
no pre-existing implementation is overwritten. Rollback is omission/reversion of
these scoped additions after parent review; no runtime rollback is needed.

Self-review-event: qa_defect (initial TAP parser false positive) and ticket_completed
(preparation only), review-only under agent-self-review-evolution. The positive TAP
fixture caught the parser defect before a production phase could run; its failure
and correction are preserved. The existing floor gate correctly prevented test
execution during the capacity block. No governance/memory/skill/SOP update is
justified by this bounded implementation defect.

## Independent review correction, round 1

The independent v1 review requested changes for F1/P1, F2/P1 and F3/P2. Its frozen
report SHA-256 is 4594e15c0e9be3b338616ff196fadf0b254883bead424a5a9bd7bbc29b4682a5.
The v1 frozen directory is unchanged. Only the original owned additions are in
scope; no shared release/controller implementation was edited.

F1 corrected an implementation error: toolkit_libraries inventories CUDA runtime
DLLs, not the nine compiled outputs in the approved resource profile. The adapter
now requires precisely strata_kernels.lib, strata_core.lib, strata_engine.lib,
strata_prefill.lib, strata_qwen35.lib, strata_mmq.lib, strata_pf_own.lib,
strata_mmq_w4a4.lib and strata_kq.dll in measuredLibraries, without missing,
duplicate or unexpected entries. Each digest is compared to the corresponding
current BUILD/Release artifact and recorded under inputs compiled:NAME. The
approved profile path is fixed, and production cuda_spill_baseline.bindings/load
checks source digest, compiler key, vendored dependency and schema. Toolkit DLLs
remain a separate PE dependency inventory, with no arbitrary nine-DLL condition.
Neither profiles nor tolerances are altered or remeasured.

F2 corrected the incomplete consumer contract. Generic nonempty hash maps are no
longer sufficient. A shared validate_proof runs after the real Job drains and
again inside validate_acceptance. It requires mode-specific artifacts at their
owned paths and revalidates CI run/job metadata and exact gh commands, Harness
source test names/files, runtime/bundle/patch identity, TAP and launch commands,
NSIS's six observations and production-source/fixture command, or native
component/builder/profile/compiled-output/binary/GPU equality evidence. Native
consumption uses read-only production signature and profile parsers; it performs
no executable probe, build or GPU replay. Scope strings remain explicit.

The same existing run_owned implementation still provides containment and resource
admission. A narrow wrapper records the exact argv/cwd/attempt/reservation hash it
actually passed to that implementation into owned-job.json.acceptanceExecution.
Inside the real owned worker, worker-context.json records its PID and Windows
process creation FILETIME, its exact argv and reservation hash. Consumption checks
both observations against owned_root_pid/root_creation_filetime and the exact
mode, reviewed source, explicit attempt and every external receipt path/SHA
argument. Thus a successful Job from another command or attempt cannot satisfy
this contract merely by being placed beside a receipt. The common guard and its
floors are unchanged; the wrapper adds evidence fields only.

F3 now rejects --execute before controller admission unless exactly one explicit
--attempt is supplied. Both --attempt rN and --attempt=rN forms are supported;
read-only plan defaults are preserved. Tests prove the controller is not called
when the explicit argument is absent.

The synthetic suite now has complete per-mode consumer fixtures and rejects
generic/missing evidence, rehashed failed output, wrong command/mode/source/attempt,
replayed worker PID/creation/reservation, mismatched native external-input argv,
and tampered/missing/extra compiled-library evidence. Toolkit-only and extra-toolkit
cases distinguish the two inventories. The native fixture mocks only the existing
production signature boundary because it uses deliberately synthetic components;
it exercises actual production profile binding/schema readers against synthetic
source/compiler/dependency files. No real signing key or GPU is used.

First guarded fix-suite execution passed 106 tests with no failures/skips in
64.62 seconds; its Job exited 0, was contained before resume, confirmed termination
and active-after 0. The final run, acceptance-fix1-tests-v2, includes explicit scope
and native metadata binding checks and passed all 106 tests in 12.24 seconds with
zero failures/errors/skips. Its guard was FINISHED with exit/native exit 0,
contained-before-resume true, termination-confirmed true, active-after 0 and zero
dropped output bytes. Sampled minima were RAM 13,366,550,528 bytes, commit headroom
36,064,698,368 bytes and H free 63,445,929,984 bytes; last C free was
34,025,205,760 bytes. Unchanged resource floors remained satisfied.

The final test command was the same offline uv/Python312/check.py wrapper recorded
above, with name acceptance-fix1-tests-v2 and child `python -B -m pytest -q
_docs/stable-060-20261116/test_acceptance.py
--junitxml=H:/Strata/evidence/upstream-sync-20261116/acceptance-fix1-tests-v2.xml`.
Its guard receipt SHA-256 is
9daab3fcf28655a65e670f50c60885f0fe03499da6181b54f4cf1c2c496e9821;
command log SHA-256 is
9952ec13ff6dd3f4f4327ad8278ebc420848f130e8f99afb8d9511b4ce27de42;
JUnit SHA-256 is 010e898f364c657521297b5c0e8139238f6caf7a56af69573f554e2eda2c24be.
Final guarded flake8 --max-line-length 140 and Black --line-length 110 --check both
passed. The Harness JS helper is byte-identical to v1
(8c5aa7da23daeab8c5cc70ba4a594b4f4d502882a36706cc1c75b9e676ab9d0c), so its v1
Node syntax evidence is retained without an unrelated rerun.

Initial fix lint found one Black/flake8 E203 slice-format conflict;
two explicit list removals preserve the test behavior and satisfy both tools.
All attempts, including this lint failure, are retained in the v2 evidence set.
Actual production CI/Harness/NSIS/native acceptance remains NOT_RUN.

The fix handoff is H:/Strata/private-backups/cleanup-20261116/acceptance-producers-v2,
with before-fix source copies, independent review, incremental diff, full staged
patch, final sources/report, command receipts/logs/JUnit and SHA-256 manifest.
Independent scoped re-review is required before commit or actual acceptance.

Self-review-event: rework_returned/qa_defect, review-only. The v1 tests established
hash integrity but allowed a generic fixture to stand in for a complete Harness
proof, and conflated compiled outputs with toolkit dependencies. The correction
is confined to explicit semantic contracts and substantive synthetic fixtures in
this adapter. No global skill/SOP/AGENTS or memory change is made. A fixture's
synthetic success remains test evidence only, never a production acceptance.
