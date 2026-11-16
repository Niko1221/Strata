# Q38 final reader and version repair

Scope: the two Important findings in the final Q38 integration review at
ee603a5a9cee0046dbe6343352d35609236fdca0. The Windows reader now owns a terminal
stop flag and cancels pending synchronous reads until the thread has actually
exited. A cancellation that finds no pending read does not establish completion.
RAII joins the reader before its protocol and handle owners can be destroyed.
Framing, the 4 MiB input bound and protocol queue/cancellation semantics remain.
The runner reports `strata-q38 own-native 0.6.0 CUDA13 SM120`; production packaging
retains its whitespace-delimited stable version check for all three runners.

The CPU fixture uses the production reader with real Windows pipes. Boundary
events force QUIT and fatal/main-stop to cancel before the next read starts,
then release that read while the writer stays open. Both observed one missed
cancellation followed by one successful cancellation and terminated. EOF,
STOP/GEN reset, normal QUIT and exception unwinding also passed (five cases).
A private one-shot negative control timed out at the same QUIT boundary after
five seconds, with the writer open. Its expected failure is retained separately.

Precommit receipts live under
`H:/Strata/evidence/upstream-sync-20261116/receipts/q38-final-*`:
`reader-build-r1` and `reader-test-r1` passed; `version-unit-r1` passed two tests.
The `negative-test-r1` receipt failed as intended (CTest timeout, exit 8).
Two initial `reader-precommit-r1/r2` command attempts failed with cmd syntax
before compilation; they are retained. CMake's existing Visual Studio path
then compiled the fixture successfully. These are dirty-tree development
checks, not final clean-source build evidence.

Selected controls: canonical CLAUDE-FABLE-5 prompt applied only compatibly with
higher-priority instructions; common engineering, implementation start, skill
selection, security, application, LLMOps, C++ and Python SOPs; milspec-codex-standard,
debugging-and-error-recovery and code-review-and-quality skills. Python uses the
existing project-specific owned-Job check.py wrapper and its pinned interpreter
(the custom-wrapper exception), without dependency changes. Every command retains
the 3 GiB RAM, 4 GiB commit, H 10 GiB/C 5 GiB disk floors and five compiler jobs.

Fresh clean-source compilation is bound to source commit
`8b0b3c74514d26e5ca73496f0b268d66a0cb99a5`. The actual `--clean-first` build under
`H:/Strata/evidence/native-dense-final-060-20261117/build` passed with CUDA
13.3.73, MSVC 14.44.35207, SM120 and five jobs. All fourteen CTest groups passed
in 16.36 seconds, including the five reader cases and existing numerical, codec,
KV, language and protocol checks. CPU CLI/admission passed all 54 cases. The
separate actual `--version` probe passed. Build/CTest/CLI/version guard receipts
are `q38-final-build-r1`, `q38-final-ctest-r1`, `q38-final-cli-r1` and
`q38-final-version-r1`; all report native exit 0, Job active-after 0, no dropped
output and zero RAM-floor breaches. Minimum build RAM was 11139133440 bytes.

The 665600-byte executable has SHA-256
`426e8caf4813ab4c1a7802c3e5595f9aa899abcc0f11b7cb4917c5fcccf383b6`.
The new `final-build-manifest.json` binds 387 native source entries and hashes to
`4b05ea823da3606f311a2d1702f532278b56e35c7069f8f50565f58d78c07b2c`.
`BUILD_MANIFEST-final-full-r1.json` binds 1253 files including 628 observed compiler
read dependencies, no missing observed paths, compiler tools, SDK/static-library
inputs, generated profile/configuration bytes, and retained guard receipts. Its
SHA-256 is `49a0fa93e8c25d2a1e7cbf2f0fe3f5c6367bacdcfd3b7680d7d0b895a4b22be4`.
Both manifests and inventory scripts are under that fresh H evidence directory.
The previous 6ee/706867da executable and its manifests were not overwritten.

The packaging adapter and Node verifier select those actual build identities.
The current binding commit is deliberately not substituted for compiled source A.
The delivery regression stages the actual PE/CRT payload, executes its version
probe through the production validator, and verifies its portable record with
the actual Node consumer. Focused delivery Python tests passed 23/23 and Node
identity tests passed 4/4 (`q38-final-delivery-python-r1` and
`q38-final-delivery-node-r1`).

Root then ran the fresh legacy build, resource recapture and ten serial GPU
cases against clean consumer B `221ee683df6c8727146f6af2810ac47183cafe37`.
Observed completion was handed back before any final tracked edit. Receipts
`q38-final-legacy-build-r1`, `q38-final-legacy-capture-r1` and
`gpu-q38-final-legacy-r1` all passed with Job active-after 0. All eleven recorded
legacy outputs were byte-identical before/after the no-op build. All 451 resource
entries and library hashes matched the preceding profile; only the native source
digest changed to `1f992dc100ee7806b00563b7c4568cc03c30a476a8ae6a2c7377dc097887a899`.
The selected baseline is byte-identical to the measured candidate, SHA-256
`863c2dc3fc077de7b12000a9d043a841059dabe772465e3fae90d3b4e2c51122`.
Its private comparison SHA-256 is
`9861b9aae9a374d7ced7d5ddc82ac58212cfb06c2a72df20bdce565c28121a5d`.

acceptance.py now pins the four actual final legacy proof artifacts. The direct
fixture checks their new paths, source digest, exact candidate/comparison hashes,
unchanged outputs, actual consumer B and the observed 1200-second outer deadline.
The GPU suite itself ran 442.135 seconds, with ten passes, zero failures/errors/
skips, minimum available RAM 11170017280 bytes and minimum free GPU memory
10807 MiB. These are root-produced synthetic numerical/lifecycle observations.
The final worker CPU check `q38-final-legacy-bindings-r1` passed 42 tests with
120 deselected, plus focused Ruff, whitespace and AST checks. The AST comparison
proves production validation changed only in four proof pins. No resource amount,
compiler/ggml identity, library hash or resource floor was relaxed. Old captures
and reports remain intact. The one final scoped independent review follows this
source/binding repair; root still owns later release operations.

Exact commands are retained in each named receipt's `command` array and adjacent
`.command.txt`. The final command shapes were `cmake --build <build> --config
Release --parallel 5 --clean-first`, `ctest --test-dir <build> -C Release
--output-on-failure -V -j 1`, `python src/qwen38/test_cli.py <exe> <build>`, and
`<exe> --version`, all through the existing guarded check.py wrapper. The two
freeze scripts only read/hash source and write new manifests after all build
commands drained; they use the metadata-inspection Python exception.

Self-review-event: type=qa_defect scope=q38-native-shutdown category=qa-coverage.
The prior tests covered protocol state but not the input thread's pre-read gap,
and mock version output hid the real runner spelling. The narrow prevention is
the deterministic production pipe test and actual staged-PE version test above.
agent-self-review-evolution was applied in review-only mode; no governance or
resource policy was rewritten. TypeScript SOP applies to the Node verifier pins.
Historical whole-file Black drift and external NVIDIA C4819 warnings are retained;
this record does not claim a warning-free build or repository-wide lint pass.

No new model-quality claim is made here. Historical Japanese
Soya 4/4 and PQ 2/4 remain attributed to compiled 6ee42543. Full-window performance,
general quality and GPU concurrency stay unqualified; releaseQualified=false.
The allocation diagnostic Minor and existing vendor/formatting warnings remain
deferred. Installation, signing, publication and original profiles are outside
this repair's authority. Root owns the later delivery and independent review.
