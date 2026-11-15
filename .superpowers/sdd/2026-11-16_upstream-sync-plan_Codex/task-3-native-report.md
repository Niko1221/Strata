# Task 3 native correction handoff

Completed the focused correction of the two failures in `gpu-tests-v1` on
`codex/upstream-sync-20261116`; no commit or subagent was created. Other workers'
changes remain preserved. Full implementation and commands:
`_docs/2026-11-16_upstream-native-compatibility_Codex.md`.

Production change: both ExpertCache device-slot overloads return null for an
invalid cache. The fork cleanup-pending contract and precise upstream
mapped-segment progress are retained. Failure tests now require zero usable
slots, unknown mapping ownership, blocked resize, cleanup and reopen. CPU tests
verify exact release9/unmap6 ownership and order, repeated cleanup, and the
verbatim generate caller blocks with explicitly stubbed upload/loan callbacks.
Q36 Windows parity setup now enables the four format variables already set by
POSIX and the engine; no kernel or tolerance changes.

Evidence under `H:/Strata/evidence/upstream-sync-20261116`: affected-target build
`native-fixes-build-v1` passed with five jobs. GPU `native-gpu-fixes-v2` passed
4/4, zero skipped, 97.06 seconds; all six Q36 pairs ran. Admission/minimum free
VRAM were 14409/11795 MiB. Full passing output was copied to
`native-gpu-fixes-v2-LastTest.log`. All owned Jobs are inactive.

CPU focused `native-diagnostics-v3` passed. Final full
`native-diagnostics-full-v1` had 10 passes and one environment-bound failure:
the unchanged historical native-pilot fixture hashes a removed frozen
`H:/Strata/build/p01-b0-20261025/bin/strata.exe`, returning code78 before launch.
Earlier `native-diagnostics-v2` compile failure is retained and explained in the
log; the upstream nibble profile API needed an explicit disabled CPU stub.

No CMake, production generate caller, kernel, dependency, release or unrelated
file change is included. No real-model serving, full-context generation,
performance, HIP/SYCL/Metal or new release qualification is claimed. Static
analyser tools were unavailable on PATH. Parent should review the staged
correction, retain the historical-fixture failure, then continue its integration
and release gates.
