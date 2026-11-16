# Q36 compiler-resource investigation, 2026-11-16

## Scope

Task 3 follow-up to the strict `cuda_spill_guard` failure at integration HEAD
`4282fce334e0d6c1843c74bca5d4063bd784bc05`. This follows the bounded instructions
in `.superpowers/sdd/2026-11-16_upstream-sync-plan_Codex/task-3-spill-brief.md`.
The three failing sm_120 instances are Q36 `native_kernel<T,true,4,512,false>`:
NVFP4 T40 STACK32, IQ2_S T22 STACK24 and IQ2_XS T17 STACK24. The unchanged
allowance is STACK16/LOCAL0. No allowance, launch shape, register limit, format,
model/default, CMake, dependency or release change is authorised in this slice.

Canonical prompt, PC/project instructions and previously selected common,
implementation-start, C++/Python and LLMOps SOPs apply. Skills remain
`milspec-codex-standard` and `debugging-and-error-recovery`. No formal
certification or performance qualification is inferred from these checks.

## Candidate 1

The only edited production file is `src/prefill/moe_fused_iq.cu`. Within the
existing CUDA-only native kernel body, a compile-time condition limits the
change to geometry 2048/512, gate/up, WW4, NT512, F16false and types 40/22/17.
The next raw unit is read at the decode point immediately before its existing
put, after the current-stage MMA. The previous early load of stage1 and trailing
load of stage s+2 are disabled for those instances. Activation copies, barriers,
decoded-buffer selection and all arithmetic remain unchanged. Weight staging
is only WW2, so the affected instances read the same global raw units.

Private evidence root:
`H:/Strata/evidence/upstream-sync-20261116/native-spill-v1`.
Original and candidate source copies are `moe_fused_iq.before.cu` and
`moe_fused_iq.candidate1.cu`. Their SHA-256 values are respectively
`b22e8453281cdbcc40e4fabc58d588107c52765ddd9ac967c949c243220499fe`
and `af12a2c76cd1b6ccb0aab3e79d6ca972af12f83a83331ad64b7804a8f7ddfb32`.

All execution used `uv run --offline --no-project --python
C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python
_docs/upstream-sync-20261116/check.py` with fresh receipt names. Resource floors
remain RAM3/commit4/H10/C5 GiB and at most five compiler jobs. No GPU experiment
has been run for this candidate.

The before table is `receipts/native-spill-before-table-v1.command.txt`, from
`tools/check_cuda_spills.py --list --cuobjdump <CUDA13.3>/bin/x64/cuobjdump.exe
<BUILD>/Release/strata_pf_own.lib`. Full mangled and demangled symbols and
REG/STACK/LOCAL/SHARED are retained for all 167 entries.

`native-spill-build-candidate1` passed in about63seconds, rebuilding
`strata_pf_own` with five jobs, CUDA13.3/MSVC14.44/sm_120. Diagnostic-only
`NVCC_APPEND_FLAGS=-Xptxas=-v` was set inside that child command. It did not
change the parent environment or persisted CMake settings. The complete ptxas
output is in its command log. All owned processes were drained.

`native-spill-candidate1-table-v1` shows only one resource-row difference over
all167entries: Q36 NVFP4 WW4 STACK32 to16. Its REG128/LOCAL0/SHARED1024 remain
unchanged. IQ2_S and IQ2_XS remain REG128/STACK24/LOCAL0/SHARED1024. Original
geometry entries have unchanged resource values; this is not bytewise binary
or numerical equivalence proof. The delta is `candidate1-resource-delta.json`.
Ptxas reports NVFP4 spill stores/loads12/12bytes and IQ2_S/IQ2_XS20/20bytes each.

`native-spill-candidate1-guard-v1` ran the complete unchanged strict CTest guard:
FAIL, exactly the two IQ2 instances remain over16. Every other checked library
has zero violations. The original three-failure receipt and this two-failure
receipt remain preserved. Candidate1 is therefore not an accepted solution.

## Final scoped change and numerical fixture

The parent explicitly omitted candidate2: delaying NVFP4 tail reads cannot fix
the remaining IQ2 instances. Following that direction, the final source applies
candidate1 only to NVFP4. IQ2_S and IQ2_XS retain their original scheduling.
No second tuning candidate was built. No measured baseline was created and the
default strict guard and its allowances remain unchanged.

The final fixture adds required IQ2_S/IQ4_NL, IQ2_XS/IQ4_NL and NVFP4/NVFP4
pairs without optional skipping. It preserves the original six pairs, their
default tile selection, tolerances and reference shapes. `--spill-ww4` selects
the three new pairs and sets TILE64, NT512, F16=0, WS=0 and NVFP4w4a8 before
initialising the kernel dispatch. It rejects `--only` and `--no-ref` in this mode
so none of the three required reference cases can silently be omitted.

NVFP4 blobs now allocate a 16-byte tail after the down matrix, with independent
nonuniform gate/up/down scales drawn from the original-geometry fixture's
range. Its block bytes use valid finite UE4M3 scales and random E2M1 codes.
The independent ggml dequantization scales each projection before the double
sum reference. MMQ is excluded only for NVFP4 because this fixture's MMQ helper
does not apply its per-expert tail. The existing no-MMQ dense ceilings and int8
arithmetic agreement limits are unchanged; no threshold was relaxed.

`native-spill-nvfp4-build-v1` passed, rebuilding both Q36 and original-geometry
numerical executables with five jobs. It recorded ptxas verbose diagnostics:
NVFP4 STACK16/spill stores12/loads12, restored IQ2_S and IQ2_XS each
STACK24/spill stores24/loads24. `native-spill-nvfp4-table-v1` has exactly the
same full resource table as candidate1: only NVFP4's STACK changed from the
original, while all three still use REG128/LOCAL0/SHARED1024.

`native-spill-nvfp4-strict-v1` ran the full strict guard and FAILED with exactly
the two remaining IQ2 violations. All other libraries passed their allowances.
This failed guard is retained explicitly; no configured profile or baseline is
used to hide it.

The private `native-spill-v1/gpu-exact.py` runner performs the exact WW4 run and
then registered Q36/default plus original-geometry IQ and NVFP4 tests serially.
It uses fresh 8192 MiB GPU0 admission per phase and continuously monitors the
unchanged 768 MiB reserve inside the outer owned-Job resource guard. The root
confirmed exclusive experiment ownership; a fresh GPU check reported14451MiB
free before launch.

`native-spill-numerical-v1` PASSED. The forced WW4 run passed all three formats
with reference shapes256/33/1/130/517, zero-token rows, unused experts, ragged
routing and 2048/8192-token synthetic timing/sample checks. It admitted14234MiB
and observed at least12250MiB free. Registered regressions then passed3/3 with
zero skips: original-geometry IQ102.61seconds, Q36 default nine-pair134.55seconds
and original-geometry NVFP4 (including edges). Total CTest time273.15seconds;
this phase admitted14440MiB and observed at least10526MiB free. The owned Job
ended with zero active processes. Full output is retained in its command log
and `native-spill-v1/q36-and-original-LastTest.log`, not just truncated JUnit.
These timings identify executed cases, not a speedup or latency qualification.

A final test-only portability correction guards the three new format entries
to CUDA and rejects `--spill-ww4` explicitly on HIP. The six existing HIP pairs
are preserved instead of introducing an unsupported mandatory NVFP4 case there.
Windows/CUDA test behavior and kernel source are unchanged by these preprocessor
guards. `native-spill-fixture-final-build-v1` PASSED after that change, without
the diagnostic NVCC environment variable. `native-spill-exact-final-v1` then
PASSED all three required formats on the final fixture, exit0 with Job active0.
Fresh GPU admission was14528MiB and minimum observed free memory12140MiB over
53 samples, above the unchanged768MiB reserve. The command-log SHA-256 is
`16c2306dac824b153783aa5a47b5f87469fe4d42da732874f00ea67c6c00a1b2`.
GPU ownership was returned to the parent after this final run.

Final exact source snapshots and review diff are retained in
`native-spill-v1/source-sha256-v2.json`, `prefill_fused_q36_test.final-v2.cpp`
and `native-spill-final-v2.diff`. The final production source SHA-256 is
`0a8f448ed738a97bf0d04487b05c44373c576e52e078428c6816ea2d0971ea2d`;
the final Q36 fixture SHA-256 is
`5460b4dfe4a91dd0998f8fd69f4dc64687ae5b4b42d886036cf35ce3f6be33b8`.

## Outstanding gate

The strict16-byte IQ2 resource defect remains unresolved. The parent will only
consider proposing an explicit Windows source/compiler-bound per-entry
measurement profile after exact numerical checks and independent review.
That is a separate acceptance decision and does not repair the strict default.
Product relink and authoritative native identity refresh are deferred to the
parent after that review, per its revised direction. No release claim is made.
The current source is uncommitted, with the original copy retained for a scoped
rollback. Real-model weights are absent and no downloads are authorised.
HIP/SYCL/Metal builds and performance qualification are NOT_RUN. The modified
kernel body is inside the existing CUDA-only architecture conditional; HIP
implementations and all SYCL/Metal files are unchanged, but unrun backend
compilations are not claimed bytewise equivalent.
