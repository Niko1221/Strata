# Task 3 compiler-resource follow-up

Status: DONE_WITH_CONCERNS. The final exact-WW4 repeat passed. The
default strict guard remains failed for two IQ2 instances. No release approval,
baseline acceptance, model-quality or performance qualification is implied.

Only src/prefill/moe_fused_iq.cu and tests/cuda/prefill_fused_q36_test.cpp are
production/test edits. The former delays the next raw weight load only for
Q36 NVFP4 gate/up WW4/NT512/F16false; IQ2 scheduling is restored to original.
The latter adds mandatory exact-Q36 IQ2_S/IQ4_NL, IQ2_XS/IQ4_NL and NVFP4/NVFP4
checks, a forced --spill-ww4 mode and correctly allocated/scaled NVFP4 tails.
The independent reference, numerical ceilings and old six pairs are retained.
New pairs are CUDA-only; HIP's old six-pair default remains intact.

One broad lifetime candidate was tried, then narrowed to NVFP4 at root's
direction. Candidate2 was omitted. Final full resources: NVFP4 REG128 STACK16
LOCAL0 SHARED1024, spill stores/loads12/12bytes; IQ2_S and IQ2_XS REG128 STACK24
LOCAL0 SHARED1024, spill stores/loads24/24bytes each. All167 library resource
entries were compared; only NVFP4 STACK differs from the original32bytes.

Both affected numerical targets built with five jobs. The complete strict
guard native-spill-nvfp4-strict-v1 failed only on the two IQ2 STACK24 cases,
with all other libraries within their unchanged allowances. Exact-Q36 WW4
numerical checks passed all three formats, and registered Q36 nine-pair plus
original IQ/NVFP4 checks passed3/3, zero skips. Full logs and receipts are under
H:/Strata/evidence/upstream-sync-20261116/native-spill-v1 and ../receipts.

Details, source snapshots, commands, initial failure and limits are in
_docs/2026-11-16_upstream-native-spill_Codex.md. No CMake, Python product,
dependency, allowance, baseline, launch-size or register-limit changes occurred.
Private GPU runner only exists in the owned evidence directory. No commits,
subagents, downloads, installs or publication actions occurred.

Root's revised direction defers product relink/native-build.json refresh until
independent review of a possible explicit source/compiler-bound Windows profile.
That proposed exception must remain separate from the unresolved strict default.
HIP/SYCL/Metal compilation and real-model inference are NOT_RUN. Scope of
numerical results is synthetic inference arithmetic, not full-context serving.

Final repeat native-spill-exact-final-v1 passed all three mandatory formats
after the final CUDA-only fixture guards, exit0 and Job active0. Fresh GPU
admission14528MiB, minimum12140MiB, 53 samples; GPU ownership returned to root.
Final production SHA-256:
0a8f448ed738a97bf0d04487b05c44373c576e52e078428c6816ea2d0971ea2d.
Final Q36 fixture SHA-256:
5460b4dfe4a91dd0998f8fd69f4dc64687ae5b4b42d886036cf35ce3f6be33b8.
Source inventory and review diff are retained as native-spill-v1/source-sha256-v2.json
and native-spill-final-v2.diff. Root owns the next independent review and any
separately approved profile/relink work; no additional codegen experiment ran.
