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

Fresh clean-source compilation, full existing Q38 numerical/codec/KV/language
CTest, CLI admission, provenance rebinding and root-owned legacy recapture follow
this source commit. No new model-quality claim is made here. Historical Japanese
Soya 4/4 and PQ 2/4 remain attributed to compiled 6ee42543. Full-window performance,
general quality and GPU concurrency stay unqualified; releaseQualified=false.
The allocation diagnostic Minor and existing vendor/formatting warnings remain
deferred. Installation, signing, publication and original profiles are outside
this repair's authority. Root owns the later delivery and independent review.
