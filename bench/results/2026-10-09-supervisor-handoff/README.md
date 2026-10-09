# v0.1.41 adaptive merge and supervisor handoff qualification

Hardware: Windows, Ryzen 9 7940HS, RTX 4070 Laptop 8 GiB, 64 GiB DDR5-5600.
Model: ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S. Context is configured to
65,536; main requests use high reasoning, with vision enabled. These runs do
not constitute a new near-64K prompt qualification or a speed comparison.

Upstream base: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (v0.1.41).
Adaptive native SHA-256:
`9ec67fc32f0a0fae56947ba1f8064cfe85b62b62bd67a431b6c00150f68eb353`.

## Component checks

- CUDA sm89 build passed, including AVX512 expert kernels.
- Eight targeted native regressions passed. HIP and SYCL were not built here.
- Final lease server suite: **945 tests run, 938 passed, 7 skipped**, 256.194 s.
  This includes 19 dedicated lease state/lifecycle/HTTP regressions.
- The standalone broker's 22 tests passed, including real loopback HTTP
  authentication, redirect refusal, bounded response size, supervisor grants,
  worker serialization, cancellation, late readiness and release failures.

Final server SHA-256:
`dcf1c12c113932763e6030abb68d8d0eaf3e9c4c9a64855d7f5ae93b9887c3aa`.
Lease module SHA-256:
`024a44d82f1d926d25cdee5c8a808c253ffe8b43d8fa26d4f9cacb652242cf2d`.

Companion Hermes runtime commit:
`c09e4cacedd04733d94f3d3c7af341367bb6ac26` on base `1e0c7730d791`.
Its final focused campaign passed **101 tests: 45 new and 56 existing**, with
3 existing platform skips. Six repository checks passed. Two additional Windows
failure groups were reproduced on the pristine base as well as the candidate
(4 approval-batch and 14 subprocess-environment failures); they are not counted
as passing. This is not a full Hermes-suite claim.

Review found and fixed a lifetime gap in the new credential declaration API:
an already-captured callback could outlive plugin unload. Callback and private-key
name capture is now coherent, and names remain protected through that in-flight
dispatch and cleanup. Real-child concurrent-unload and profile-isolation tests
cover the fix. This was a defect in the proposed implementation, not an upstream
Hermes vulnerability report or a research-derived improvement.

## Existing reactive path, rerun after the merge

This earlier frozen server revision predates the supervisor lease endpoint:
`62543dc65f683e88058aca21830fd5fef2096bce417cb8002531815427bedbb7`.
It uses the same adaptive native binary above, 32 GiB configured resident RAM,
64K context, high reasoning and CPU vision. The HTTP case explicitly enabled
idle parking before execution; its final configuration hash was recorded.

| Check | Observed result |
| --- | --- |
| Actual Hermes compiler/tool feedback | 218.893 s agent interval; passing real four-job C++ build, 2,096 cases plus 104 independent checks, and a tool-only nonce in the final assistant answer |
| Lifecycle | Both native and vision processes exited and returned with new identities; no owned process remained after cleanup |
| Admission and cancellation | Held incoming request stayed blocked for 5.732 s; streaming wait/cancellation checks passed |
| Sampled minimum headroom | 9.380 GiB available RAM; 310 MiB native free VRAM |
| HTTP image | Actual red-square image answered correctly in 13.153 s |
| HTTP tool result | Function arguments 17 and 25, followed by real result continuation to 42, in 16.395 s |
| HTTP sampled minimum headroom | 10.177 GiB available RAM; 342 MiB native free VRAM |

As in the prior idle-pressure test, a separate bounded **3 GiB RAM holder** and
extended tool-response wait exercise the release/reload path. The small compile
finishes before reactive unloading. Its duration and the holder's allocation
must not be described as compiler memory requirements or compilation speedup.
The five recorded API calls include independent admission/cancellation probes;
they are not five main agent turns.

## Supervisor handoff acceptance

The final Hermes revision above completed the resource lifecycle and produced
verified build feedback, with **one disclosed instruction-following deviation**.
The [sanitized result](supervisor-result.json) records the settings, relative
timeline, sampled resources and failed-checker history.

| Check | Final-code observation |
| --- | --- |
| Supervisor and worker | One accepted supervisor plan, one delegated worker, one actual child compiler invocation |
| Before compilation | Both model processes confirmed dead; fresh ready acknowledgment; about 51.422 GiB RAM available when the tool started |
| Lease lifetime | Two renewals; release after the real build and declared 15-second foreground hold |
| Credential boundary | Control credential absent from the actual terminal child |
| Build and feedback | Four compiler jobs, 2,096 checks and 104 independent checks passed; final assistant output matched actual tool output and its fresh nonce |
| Return to inference | Both model processes returned with new identities; normal final assistant response independently corroborated in API output and the session database |
| Reasoning and time | Seven actual high-reasoning main requests: five supervisor, two worker; 371.107 s agent interval including hold/reload/prefill and the extra read |
| Vision | Actual red-square image answered correctly on the same runtime, 12.978 s |
| Minimum sampled headroom | 11.056 GiB available RAM; 292 MiB native CUDA/DXGI free VRAM |
| Cleanup | No owned process survivors |

The supervisor made an extra read-only grep of its worker's delegation log to
recover the exact result. The test prompt broadly prohibited other commands, so
**exact prompt compliance is false**. The original harness also incorrectly
treated any supervisor terminal use as recompilation. Its failed result is
preserved. A separate read-only check verified the sole root command, linked
exit status, exclusive child-build claim, actual lease traffic, fixture identity
and final output: all 18 factual lifecycle/result gates passed. No rerun or
runtime edit was used to relabel that recorded run.

The same-boot function/result smoke was skipped when the original harness stopped;
the separate GPU component below passed it on the unchanged Strata source/native
revision. An earlier pre-retention-fix Hermes run took 356.990 s and needed a
different checker correction for legitimate `tool_call` wrapping. It is not
substituted for this final revision or treated as a matched speed baseline.

The C++ fixture was already correct and immutable. This tests resource handoff
and real tool feedback, not general generated-code repair. The existing
[104-case fixture](../2026-10-08-adaptive-background/unique_sorted_checks.cpp)
and protocol/plugin tests are public; the complete host-specific model harness
and its private profile/session data are not presented as a portable benchmark.

## Separate GPU tool component

On the final server/native revision above, an authenticated full-unload lease
requested 4 GiB total free RAM and 2,048 MiB total free VRAM. Both model processes
were independently confirmed stopped before a bounded CUDA helper ran. This was
a direct protocol/component check, **not a Hermes GPU-agent run**.

| Check | Observed result |
| --- | --- |
| Lease progression | First sampled unloading at +1.999 s, first ready at +4.568 s, release at +5.375 s from acquisition |
| Real GPU operation | 512 MiB allocated; filled with `0xA5`; first and last 64 bytes copied back and verified |
| Actual CUDA free VRAM | 7,068 MiB before allocation, 6,556 MiB during it, 7,068 MiB after `cudaFree` and synchronization |
| Helper | Exited successfully in 0.328 s; control credential absent from its environment |
| Return to inference | New native and vision process identities; high-reasoning arithmetic returned exactly 42 in 54.672 s including reload/admission |
| Tool compatibility | Actual `add(a=17,b=25)` and result continuation to exactly 42 with normal stop |
| Sampled minimum headroom / cleanup | 12.110 GiB available RAM / 340 MiB native free VRAM; no owned survivors |

The 54.672-second return illustrates why full unload should be reserved for work
that needs the released capacity. It is a single observed reload/request interval,
not an isolated reload-cost estimate or a matched performance comparison. No
physical OOM was induced, and the small GPU operation is not a game/FPS test.

## Interpretation and provenance

The feature adds advance notice of known heavy tool work to the existing
reactive controller. Full unload can increase successful completion time by
losing warm model state. There is no matched comparison here showing a faster
agent, no guarantee against other applications allocating memory after a sample,
and no proof of uninterrupted progress under permanent resource exhaustion.

The [protocol](../../../docs/TOOL_RESOURCE_LEASES.md),
[standalone plugin](../../../integrations/hermes-resource-feedback/README.md)
and [implementation provenance](../../../docs/ADAPTIVE_PROVENANCE.md) describe
the control boundary and actual source reuse. MARS influenced information
sharing and admission separation; vLLM sleep/wake is actuator prior art. Their
code and published speedups are not imported. The older
[idle-pressure packet](../2026-10-08-idle-pressure) retains its original
revision identities and results.

Private prompts, user paths, session databases and control credentials are not
included in this public packet. Source hashes and test counts describe their
specific revisions rather than combining results from different snapshots.

## Cross-platform CI follow-up

The [first public CI run](https://github.com/midhatn/Strata/actions/runs/37877811304)
passed both native CPU jobs but found three test-fixture portability/scheduling
issues: an implicit Windows commit-sensor assumption on Linux, an EOF emitted by
a fake process still described as waiting for READY, and Windows DOS short-name
versus resolved temporary-directory spelling. These were corrected in tests;
the measured server, lease module and native binary did not change.

The affected modules passed 65 tests locally (64 passed, 1 platform skip), and
the corrected cancellation case passed 100 consecutive repetitions. A separate
optional-commit-sensor regression was added, raising the full suite to 946 tests.
These targeted results do not substitute for the new cross-platform CI run.
