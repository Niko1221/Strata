# Pinned official 0.2.15 intake for Hakua 0.6.0

Status: source applied and scoped local validation complete; independent review
is pending. Release packaging, installation, model/GPU validation and publication
are NOT_RUN in this intake. Candidate compilation is recorded separately below.
Installed and published 0.5.9 evidence is unchanged.

The fork input is d552b56ab0218a613dc6c287947022adf4bd4164. The previous
official input is baf16363b4403ef0ed4f538f5d5ed11c43fb9535. The fixed delivery
cutoff is c77572806996349b49995e4f98d41b5294d36a21, exactly 45 official
commits ahead according to the GitHub comparison API. Later official commits
are outside this intake. Fork app, Terminal, Control and engine versions stay
0.6.0; installed and published stable evidence remains 0.5.9 until actual new
delivery verification succeeds.

`_docs/upstream-sync-20261116/intake-c775.py` adapts the previous immutable-blob
diff3 helper. It prepares private copies of old official, current fork and new
official blobs, preserves physical original bytes, and records every changed
path. Application requires a complete per-path reason and reviewed result hash,
unchanged originals and no unresolved conflict markers. It does not stage,
commit, push, alter installed profiles or run a release. The exact reviewed
command vectors, results and guarded receipts are retained in the review evidence.

Preservation requirements include private workflow learning/application, `/btw`,
`/deepresearch`, RustXNG, sidebar CRUD, memory isolation, verified PrismML and
TurboQuant adapters, runtime payload integrity, effective context capacity and
startup ownership, bounded native expert memory, production feed trust and
saved user choices. Generic legacy llama-server discovery must be removed while
explicitly supported adapters remain available. The pinned seven-case sd.cpp
RoPE oracle and scalar-float fix remain strict. Existing Windows native test
repairs and Windows snapshot limits remain in force.

The actual `tools/cuda_spill_baseline.py::source_digest` result is
`1c7811ccd9ebe8a853469b468ea196757a6d95332fe030b3e5f8d1512aedeaef`.
The prior digest `8bc0448a35e217052eb0537285c7e38e724f0422a02f33f30554d2fb07d6a848`
and its GPU observations remain historical. Neither their pins nor outcomes
were rewritten. The changed source requires separate native qualification.
Generic IQ2 spill failures remain distinct from the explicitly bound Windows
profile pass.

Applicable controls: the canonical pre-implementation context, PC/project
AGENTS, common engineering, implementation-start, skill-selection, security,
deep-research, application and relevant language SOPs. Skills applied are
milspec-codex-standard, deepresearch-defense-standard, git-workflow-and-versioning
and documentation-and-adrs. No formal MILSPEC certification is claimed.

The exact fetch and 121-path diff3 preparation completed. Forty paths had
textual conflicts. Per-hunk rulings preserve independent fork versions and
features while incorporating the official delta. Eight root README installer
version conflicts retain factual fork delivery text; their final delivery
rewrite remains a separate task. The apply step verified every physical
original hash before any production write and left the Git index untouched.

Automatic llama-server discovery through environment, PATH, Homebrew, build
folders and generic GGUF components is removed. Explicit PrismML/TurboQuant
selection retains source/payload verification, KV compatibility, private memory,
startup ownership, RAM protection and effective `/props` context capacity.
Lifecycle unit tests use isolated synthetic 0.4.0 packages; these do not qualify
real models or replace the separately pinned PrismML 0.4.1 acceptance input.

The pending runtime-state fix validates file privacy before treating an empty
lease as still being written. Drafter feed lookup now respects the previously
accepted sequence as well as existing signature checks. Saved drafting choices
and the fork's merge-into-current-file persistence remain intact.

RoPE verification keeps the strict seven-case independent sd.cpp C++ oracle,
compiled with the same compiler, flags and math library as the implementation.
Historical recorded hashes remain glibc-only. The incoming float64 reference
runs separately with its unchanged `2*ulp32(v)+abs(t)*2**-22` tolerance. The
official portability change at `88d404026c57df7777d8c3c67e2f406f02f1a089`
intentionally limits historical hashes to glibc while retaining float64 checks
on Windows and macOS. The fork's volatile scalar fix and pinned oracle originals
remain unchanged. Exact equivalence is scoped to the measured toolchain/libm;
this does not assert that all C libraries have identical last-bit results.

The first guarded Python attempt was interrupted by RESOURCE_PRESSURE with
exit 125 when available RAM fell below 3 GiB; its owned Job was drained. It
has no completed pytest result or passing JUnit. The user's explicit operation
exception is implemented as `--allow-low-ram` in the local check runner. Default
admission remains 3 GiB. An opted-in invocation records its effective RAM floor,
authorization, nominal-floor breach count and minimum observed RAM; commit,
disk, timeout, output and owned-Job checks remain required. Production model RAM
guards and saved settings are unchanged. Two synthetic guard tests passed;
subsequent completed results are recorded below.

The second Python attempt timed out after 600 seconds without a completed
pytest/JUnit result. Host validation first reported 173 passes and 16 failures;
an affected replay reported 20 passes and eight failures. These exposed a
missing filesystem import, fixture assumptions changed by removal of generic
backend discovery, and a clean merge that erroneously threw after a successful
adapter start. Corrections retain all lifecycle and privacy assertions. The
complete 21-module Host replay then passed all 189 tests, with no failures or
skips, in 130.347 seconds. Its owned Job exited successfully and drained.
An intervening replay was refused before launch at the unchanged 4 GiB commit
headroom floor. All failed, interrupted and refused attempts remain retained.

The named Python replay completed 416 passes, one failure and eight existing
skips in 752.85 seconds. The added privacy test's mock rejected its parent
directory before reaching the intended state-file check. Scoping the mock to
that file preserves real directory checks. Two skips need unavailable owner
release archives; six are POSIX-only semantics. The affected replay passed all
31 tests in 1.94 seconds, including operation-guard regressions.

Progress fixtures keep the original 23 GB, reader and 1 GB logical sizes.
On Windows, Python's CRT truncate writes zero bytes when extending a file,
so sparse marking alone still timed out in a retained attempt. The test-only
helper now uses typed Windows file-end APIs after sparse marking and restores
the stream position; POSIX behavior is unchanged. Actual files retain their
exact sizes, sparse attribute, at most 1 MiB allocation and zero-readable holes.
All original progress, rounding and rename assertions remain.
The Windows behavior follows [CPython's FileIO implementation](https://raw.githubusercontent.com/python/cpython/v3.12.0/Modules/_io/fileio.c)
and [Microsoft's CRT documentation](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/chsize-s?view=msvc-170).

Final static checks passed for 52 Python files (AST and fatal flake8), 30 Node
modules, the snapshot workflow with actionlint, whitespace and TypeScript
without emission. A retained first static attempt caught a misplaced local
variable in the orchestration helper; correcting it did not execute source
application again. Clang 22.1.8 targeting Windows compiled and ran the pool
tests: 3,778 segment and 21 buffer checks passed. The 34 CPU Python checks for
RoPE, tokenizer and image code passed without skips, including the strict
seven-case oracle and unchanged float64 comparison. Exact numerical results
are for this compiler, flags and Windows math library, not all platforms.

The parent controller also completed the five-job incremental MSVC/CUDA
candidate compilation, with the native source digest unchanged before/after.
That uncommitted working-tree build is separate from final release qualification;
it performs no GPU or CTest run and does not refresh the prior GPU proof.

The first new snapshot run completed 12 passes and one failure in 401.66 seconds.
The official saved-word test assumed its three Qwen model sets; this fork's
matrix also includes DeepSeek. The test now retains every machine/model and
restart comparison while requiring Qwen's saved Auto word only for its existing
families. DeepSeek keeps its independent DSpark behavior and must have no Qwen
draft flags. No model-family behavior or production allowlist was broadened.
The affected five-case replay passed in 282.10 seconds. Together with the eight
unaffected cases that passed in the first run, all thirteen new cases have
completed passing coverage. The first run remains a failure in the evidence.
These local subsets do not establish the full expanded hosted snapshot's timing;
exact-source hosted CI remains a separate required check.

Snapshot jobs remain exactly Ubuntu 24.04 and Windows 2022. The upstream matrix
timeout scale is integrated with Linux 900 seconds/30 minutes and Windows
1800 seconds/45 minutes, retaining Node 24.22.3. The delivery acceptance file
pins the resolved workflow bytes; no earlier CI result is reused at this source.
Release-controller exception propagation and its typed evidence validators
remain separate follow-up work, as do actual native rebuild/profile/GPU replay.
That follow-up must cover `release.py::admit` and its owned runner,
`native-input.py::parser/run_controller`, delivery admission,
`acceptance.py::successful_job/validate_gpu` and the real-model validator's
calls to `successful_job`. Their current 3 GiB checks remain unchanged here;
an operation exception cannot be treated as zero nominal-floor violations.

The signed official v0.2.15 feed (stable sequence 127) carries Intel Arc's SYCL
0.1.59 archive from v0.2.13. Its URL, size and digest match the primary release
asset metadata. Looking only at v0.2.15 attachments had incorrectly suggested
that no package existed. The capability table now records package presence as
untested, retaining historical measurement attribution and explicitly leaving
archive execution/current Qwen3.8 compatibility unqualified. This does not add
SYCL to the fork's Windows CUDA distribution. Source-only test coverage remains
as a synthetic absent-package case, including rejection of contradictory data.


## Independent review fix round 1

The independent SPEC/QUALITY review requested two P2 corrections. ChatModelPicker now shows real imported models for an explicitly selected PrismML or TurboQuant adapter, and lets a user return from a remote API model to the already loaded import without reloading it. Unloaded generic imports under Automatic stay unavailable. The existing server payload/source verification, lifecycle and capacity gates are unchanged.

Native install now rechecks both job.state and engineAbort.signal.aborted immediately after the initial engine probe resolves, before starting the engine step or sending a confirmed model install. It also checks both after awaiting a required engine step. The signal check covers Stop while the cancellation RPC is still pending.

The new mounted React tests exercise local-to-remote-to-local selection for both adapters, the existing confirmation action for installed imports, Automatic refusal, and returning to an already loaded import after changing the next-load setting. The fake-Control tests defer the initial engine probe and cover usable/unusable results with Stop completed or its cancellation RPC still pending. They assert no confirmed model install, no engine update check/update, no published install progress, and a final cancelled job. Their temporary profiles and browser contexts are isolated; these are component/fixture checks, not a real-model qualification.

Recorded attempts remain separate. Red-v1 had 7 passes and 9 failures, including the intended regressions and two post-test asynchronous diagnostics from the unfixed continuation. Green-v1 had 25 passes and 4 failures caused by a fixture counting legitimate background refresh engine reads as operation probes. The fixture now checks actual install/update/progress effects. Red-v2 against the exact frozen v1 production bytes again had 7 passes and 9 failures; its fixture cleanup drains the buggy polling continuation, removing those asynchronous diagnostics. Green-v2 had 27 passes and 2 fixture-detail failures because an aborted operation returning while Stop awaited its RPC records undefined instead of the initial null. Both mean no result; the final assertion accepts either while requiring the install/update/progress effects to remain absent. Production fix bytes did not change during these fixture corrections.

Final covering green-v3 passed all 29 cases with zero failures, cancellations or skips in 51.4465562 seconds. The five modules are chat-model-picker, fix0215-drafting-off, model-picker, engine-step and no-llama-server. The last two Automatic refusal cases exercise the packaged Host against environment and PATH discovery. The scoped static attempt passed three JavaScript syntax checks, the Host TypeScript noEmit check and production-file whitespace validation. All six round-one owned Jobs are FINISHED with native and wrapper exits recorded, termination confirmed and active process count zero. These attempts recorded zero nominal RAM breach samples; prior intake breaches remain retained. The explicit operation-only RAM exception does not change default or shipped floors.

Covering command form (the private receipt records the exact verified interpreter path): `uv run --no-project --offline --python <verified-Python-3.12-executable> python _docs/upstream-sync-20261116/check.py --allow-low-ram --seconds 300 c775-round1-green-v3 node _docs/upstream-sync-20261116/host-focused.mjs chat-model-picker fix0215-drafting-off model-picker engine-step no-llama-server`. Both red attempts use the same wrapper with their own attempt name and only the first two modules. Intermediate green attempts use the same five-module selection. Static runs through the same guarded wrapper using the frozen private static.py command list.

Correction to the earlier diff-completeness statement: v1's intake.patch omitted 25 additions even though its source-after inventory contained them. The separately pinned added-files supplement supplies those exact new-file diffs and a byte-verified isolated apply. V1 artifacts are unchanged. V2 supplies full and round-one incremental raw-byte patches, asserts every expected changed/addition path, applies each into a private Git index and compares every resulting file byte with the frozen source. The original freezer's preventive improvement and earlier eight LF-to-CRLF diagnostics remain recorded for later branch triage; this round makes no broad EOL rewrite.

This round changes two production files, one incoming test fixture and one new mounted test, plus this report. All unrelated v1 source files retain their recorded physical hashes. Independent scoped re-review is still required. No commit, push, model run, GPU operation, installation, signing or release action was performed by this fix round. Parent-owned later native/GPU evidence remains separately identified; the original intake observations above are not relabelled as new-source release qualification. Exact-source hosted CI remains pending parent integration.
