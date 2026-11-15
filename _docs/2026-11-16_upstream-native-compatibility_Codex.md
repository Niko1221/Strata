# Native compatibility corrections, 2026-11-16

## Scope and acceptance

This is a focused Task 3 correction in the uncommitted integration of fork
`6f13c7555a7d50289f0e3587ab780627c65a8000` and official
`09b603886925994d0d4461857b01321ef3aa431c`, on
`H:/Strata/worktrees/upstream-sync-20261116`. It addresses the two actual failures
in `gpu-tests-v1`, retaining that receipt. It does not qualify the whole merge,
release, real-model serving, full context, HIP, SYCL, Metal, or model quality.

Acceptance is the unchanged fork fail-closed ownership contract plus precise
segment metadata after a partial shrink, no usable device slot while cleanup is
pending, explicit cleanup/reopen recovery for deliberately skipped calls, and
Q36 synthetic parity with the format flags that its engine already establishes.
The upstream normal uniform/sized shrink/grow address, byte and VRAM assertions
remain unchanged. RAM 3 GiB, commit 4 GiB, H 10 GiB, C 5 GiB and GPU 768 MiB
reserve floors remain unchanged. At most five compiler jobs and one GPU experiment
at a time are permitted. No model downloads or normal-profile changes are made.

Canonical pre-implementation prompt and the PC/project instructions were read.
Applied SOPs: common MILSPECLLMOps engineering standard, implementation start gate,
skill selection, Application Development, LLMOps, C++ and Python; security rules
govern bounded command execution. Skills: `milspec-codex-standard` and
`debugging-and-error-recovery`. These records establish traceability, not formal
certification against a military standard.

## Decisions and implementation

The fork intentionally differs from upstream after a segment unmap/release
failure. It retains ownership, reports `mapped_bytes() == -1`, exposes zero live
slots and blocks resize until cleanup succeeds. The merged `shrink` already
updates `mapped_segs_` after each successful unmap; that correction is retained.
No production branch interprets the test hook as permission to continue serving.
Both `device_slot` overloads now reject an invalid cache. The header documents
this rule and the shrink failure contract. Normal valid-cache addressing stays
unchanged, including the existing full-capacity bounds.

The segmented GPU regression independently opens caches for release of segment 9
and unmap of segment 6. It checks invalidity, unknown mapped bytes, zero slots,
blocked resize, null device pointers, clearing residency, and close/reopen before
refill and successful shrink/grow. A hook means the selected driver call was
never made; it cannot prove that real CUDA failures are recoverable. The hook
header now makes that distinction explicit.

The CPU diagnostics extract actual implementation functions from the current
source. Release-9 checks leave `mapped_segs_ == 9`, mapping flag 9 clear, handle 9
retained and higher handles clear. Unmap-6 checks leave `mapped_segs_ == 7`, flag
6 set and handle 6 retained, without releasing handle 6. Exact descending call
order, no driver calls on blocked resize and no double-unmap during repeated
failed cleanup are checked. The real refusal-hook declaration is used. A disabled
nibble stub supplies the new upstream profile-fill API without claiming coverage
of nibble packing.

Caller verification executes the verbatim shrink/error block from `generate.cpp`
with the actual residency-trim helper. It observes all residency entries cleared,
then upload and relend callbacks with zero slots, and executes the source's
next-request unknown-ownership guard, which returns failure. GPU upload/loan
callbacks are stubs; this does not execute the full request loop or establish
real-model recovery. `generate.cpp` is not edited.

Q36 failure was Windows-only test setup: it enabled `STRATA_PF_FUSED` but omitted
`STRATA_PF_FUSED_KQ`, `STRATA_PF_FUSED_Q6K`, `STRATA_PF_FUSED_Q8GU` and
`STRATA_MMQ_KQ`, which were already enabled in the POSIX test and Q36 engine.
The Windows test now sets those flags too. No kernel, format assertion, numerical
tolerance, optional-pair rule, model path or production default is changed.

Changed files are `src/core/expert_cache.cpp`,
`include/strata/core/expert_cache.hpp`, `include/strata/core/expert_cache_refuse.hpp`,
`tests/core/expert_cache_segmented_test.cpp`, `tests/cuda/prefill_fused_q36_test.cpp`,
`tests/perf131k/diagnostics_cpu.cpp` and `tests/perf131k/test_diagnostics.py`.
No CMake or unrelated source ownership changes are required.

## Verification receipts

All new executed commands use the existing owned-Job wrapper
`_docs/upstream-sync-20261116/check.py`, launched by
`uv run --offline --no-project --python C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python`.
Exact argument vectors, resource samples, exit status and command-log hashes are
in `H:/Strata/evidence/upstream-sync-20261116/receipts`.

`native-diagnostics-v2` failed compiling the extracted profile-fill function:
upstream added nibble symbols absent in the old CPU fixture. Its compiler log is
retained. After supplying the explicitly disabled nibble stub,
`native-diagnostics-v3` passed the selected pytest case, running both the VMM and
blob executables. This uses the existing frozen MSVC receipt for CPU stubs, not
a product CUDA compilation. Both owned Jobs ended with zero active processes.

`native-fixes-build-v1` passed the affected Release targets with `cmake --build
H:/Strata/evidence/upstream-sync-20261116/cuda-build --config Release --parallel 5
--target expert_cache_segmented_test prefill_fused_q36_test
expert_cache_slot_guard_test expert_cache_per_layer_test`. It used the parent's
fresh CUDA 13.3 / MSVC 14.44 / sm_120 portable build with Spin8, native cache and
MMQ enabled. Wall time was about 62 seconds; the owned Job ended with zero active
processes. Existing compiler warnings remain in the command log.

After parent coordination, `native-gpu-fixes-v2` passed all four registered tests
without skips in 97.06 seconds. The exact helper arguments were
`python _docs/upstream-sync-20261116/gpu-tests.py --name native-gpu-fixes-v2
--test expert_cache_segmented_test --test prefill_fused_q36_test
--test expert_cache_slot_guard_test --test expert_cache_per_layer_test` inside
the wrapper with a 1200-second deadline. It ran serially. GPU admission was
14409 MiB free; the minimum observed was 11795 MiB, above the unchanged 768 MiB
reserve. The owned Job returned exit 0 and zero active processes. The GPU is no
longer owned by this experiment.

Q36 took 94.49 seconds and exercised all six format pairs, five reference shapes
per pair and the 2048/8192-token synthetic timing sections. No optional pair was
skipped. Full output is preserved in
`H:/Strata/evidence/upstream-sync-20261116/native-gpu-fixes-v2-LastTest.log` because
CTest's JUnit includes only the first 1024 output bytes of a passing test.
Timing numbers on this shared desktop are not performance qualification.

`native-diagnostics-full-v1` ran the final CPU diagnostic source: 10 passed and
1 failed. The new extracted-source fixture passed. The untouched
`test_native_consumer_option_reaches_owned_runner` imports the historical native
pilot fixture, which tries to hash the absent frozen binary
`H:/Strata/build/p01-b0-20261025/bin/strata.exe`. It returned
`INPUT_BINDING_FAILED`, `FileNotFoundError`, code 78 before launching any child.
The original result remains at
`H:/Strata/worktrees/p01-a-20261025/_scratch/perf131k/b2-fixtures/ba2a9076cff34fb6a417cd8f4d61178b/native-4096-off/result.json`.
No binary/model reconstruction or input-binding bypass was attempted. This is
an environment-bound failure, not an all-green suite claim.

`git diff --check` passed for the seven scoped source/test files. The final
source hashes and correction-only diff are retained beside the receipts.

## Limits and rollback

The only production behavior added by this correction is rejecting device-slot
access when `valid()` is false; valid cache addressing is unchanged. Reverting
these scoped changes before publication is the rollback path. The parent owns
the full integration and release decision. Original failed GPU evidence remains
alongside follow-up evidence. Clang-format/clang-tidy and non-CUDA backends are
not run in this focused Windows/MSVC task; neither analyser is discoverable on
PATH. Compiler warnings are retained in logs; passing compilation is not a
warning-free claim. The next action is independent review of the staged scoped
changes and the parent-owned integration/release checks. The broader CPU suite
retains the missing historical-binary limitation above.
