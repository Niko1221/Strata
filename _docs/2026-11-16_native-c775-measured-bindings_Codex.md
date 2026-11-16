# Native c775 measured evidence preparation

This preparation selects the complete actual measured Windows CUDA profile and
the already-executed ten-case GPU proof for source candidate 0.6.0. Stable
installation and publication remain 0.5.9. Base main is
332aa8bab2e346f6a7651a172f7aff70856ab143. Final native build, component signing,
installation and publication are NOT_RUN in this task; releaseQualified=false.

The selected profile is byte-identical to the captured candidate, SHA-256
45c1fdb68fbe506c2790643c3295d59aa01f160cd1976ef9031469cd6ce028e5,
under H:/Strata/private-backups/cleanup-20261116/native-c775-profile-v1.
All 451 nonzero resource entry amounts are unchanged, with no added or missing
entries. Source binding changes from
8bc0448a35e217052eb0537285c7e38e724f0422a02f33f30554d2fb07d6a848 to
1c7811ccd9ebe8a853469b468ea196757a6d95332fe030b3e5f8d1512aedeaef.
The measuredLibraries row for strata_kq.dll also changes from
53ae1df150f2f25f66cf9bc27f29b0f7eff81ad1c3c41c8466a5fe718aad6883 to
7f42796411387ef9d7d0d9cca8000bf3009bb85990a9f21633c6b10e6bab2bf8.
The new DLL hash matches capture inputs and current actual DLL bytes. Therefore
the earlier source-only description was incomplete; complete candidate bytes
are selected. Compiler/configuration and vendored ggml bindings are unchanged.

Generic default spill checking retains its two IQ2 STACK failures at 24 versus
16 bytes. The preserved native-c775-default-strict-v1 receipt has exit 1, two
failures and drained Job 0. The explicit bound-profile receipt
native-c775-profile-bound-v1 has exit 0 and drained Job 0. Neither generic
allowances nor parser behavior changes in this task.

The four acceptance GPU pins select gpu-c775-v1 binding, report, guard and JUnit
bytes. The actual earlier run passed ten numerical/lifecycle cases, failed none
and skipped none, with serial concurrency one, a 900-second root deadline and
drained Job 0. Its actual native executable hashes are
94e13464087b8944665dc226d798df4942cb74199022fed20947101b53d7f99c and
51d4a189cb6ab53641008a6ecbecd89af6f0a37ec33d45da3515bf4779a77b60.
The recorded d552 location HEAD identifies dirty-candidate provenance; it does
not identify a committed build. Acceptance still requires current source digest
and both executable hashes to match. Changed final executable bytes require a
fresh actual proof, rather than manual hash replacement.

The GPU guard self-describes a RAM exception, but its child-command record does
not authenticate an explicit outer flag/reservation. This proof uses only the
nominal 3 GiB RAM path: admission was 16959631360 bytes and sampled minimum was
14365597696 bytes. The new tests reject below-nominal admission and observation
despite that self-description. Reviewed RAM-policy implementation and all
commit, disk, Job and output/deadline gates remain preserved.

Focused verification passed 31 tests with no skips; 110 unrelated cases were
deselected. Those tests read frozen proof and current measured binaries without
executing GPU work. They validate complete profile selection, production
source/compiler/ggml binding and library equality, and rejection of changed
identity, increased/unlisted resources, stale source, either changed executable,
malformed guards, resource floors and changed artifact bytes. Existing exact
ten-case XML rejection coverage remains in place. Full adjacent regression
passed 312 tests with no skips in 207.39 seconds. Formatting preserved identical
ASTs and restored untouched statements; final focused verification passed the
same 31 cases in 21.53 seconds. Fatal Ruff checks, AST checks, whitespace and
unchanged production-gate checks passed. Each owned Job finished with observed
exit/native 0 and active count 0. Exact commands and raw evidence are in the
task report and freeze.

Applicable controls are the canonical PC prompt, home/Codex/project AGENTS,
common engineering, implementation-start, skill-selection, application and
Python SOPs. Applied skills are milspec-codex-standard,
documentation-and-adrs and review-only agent-self-review-evolution. The existing
custom guarded runner is used through offline uv with the pinned project
interpreter; no dependency resolution or installation occurs.

All owned source before/after bytes, scoped patch including this added note,
raw-byte patch-application comparison, exact commands, receipts, JUnit and
verification manifest are retained under
H:/Strata/private-backups/cleanup-20261116/native-c775-acceptance-v1.
Rollback restores only these four scoped source files from frozen before bytes;
old captures and proofs are retained. Independent SPEC and QUALITY review are
required before root commits. The worker does not stage, commit or push.

Self-review-event: ticket_completed. The evidence correction about the changed
DLL is preserved explicitly. No governance or memory update was needed. These
synthetic checks do not establish real-model quality, full-window inference,
performance, GPU concurrency or release qualification.
