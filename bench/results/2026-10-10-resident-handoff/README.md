# Resident supervisor handoff: bounded Windows qualification

These measurements use the v0.1.41-based C1 implementation at
`efb21996fea649622cb84fdedd5765f425dbbe5f`. They are **not v0.1.42 performance
results**. The later source integration and its validation must be reported
separately. Raw local transcripts, credentials and private paths are excluded.

The objective is to reduce the time before an agent can continue after a tool
finishes. If the requested tool resources are already available, `auto` can
retain the native and vision processes. Otherwise it attempts live cache relief
and can fall back to the existing verified unload/reload lifecycle. Strict
`relieve` refuses admission when resident relief cannot meet the request.

## Matched component comparison

Hardware: Windows, Ryzen 9 7940HS, RTX 4070 Laptop 8 GiB, 64 GiB DDR5-5600.
Model: ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S; high reasoning; configured
65,536-token context; CPU vision enabled. The complete allowlisted settings,
hashes, token usage and individual measurements are in
[component-results.json](component-results.json).

Each run makes a deterministic lookup, executes a known-correct C++ fixture
using four compiler jobs, and repeats the same lookup. Both model answers must
be `42`; all 2,096 fixture checks must pass. Startup is measured separately and
excluded from the cycle. Each mode has two runs, in unload/resident/resident/
unload order; this is a small qualification sample, not a broad benchmark.

The configured `spec=4` resolves to native `mtp_max=4`, `lookup=3`, and a
six-row verification window (`spec=6` in INFO); it is not six MTP draft tokens.
Configured KV residency of 16,384 is raised by the native QSA floor to 20,480.
Both modes request 6 GiB RAM and 250 MiB VRAM for tool admission. The legacy
unload request retains a 6 GiB execution RAM floor; auto explicitly uses 3 GiB.
Neither floor binds in these runs (sampled available RAM stays above 9 GiB).
Per-arm admission, execution, and return allowances are included in the JSON.

| Mode | Run | First request | Tool admission | Compiler | Return request | Complete cycle |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Unload | 1 | 23.516 s | 2.907 s | 2.422 s | 65.453 s | 94.515 s |
| Resident auto | 1 | 15.625 s | 1.094 s | 2.422 s | 4.937 s | 24.296 s |
| Resident auto | 2 | 23.547 s | 0.469 s | 2.421 s | 4.907 s | 31.562 s |
| Unload | 2 | 23.516 s | 3.032 s | 2.437 s | 65.922 s | 95.078 s |

The mean **return request** fell from 65.688 to 4.922 seconds, 92.5% less waiting
for this deliberately cache-sensitive continuation. Resident runs preserved both
process identities and reused 1,124 of 1,129 prompt tokens. Unload runs replaced
both identities and re-read the prompt. The minimum sampled native free VRAM
was 322 MiB and minimum available RAM was 9.365 GiB across these four runs.
All owned processes were confirmed gone after each test.

The first HTTP request varied by roughly eight seconds outside native timing
counters. Its cause is unproven; that variation must not be attributed to
resident handoff. Compilation time was essentially unchanged. This is evidence
for faster continuation, not an 85–93% faster compiler or generation kernel.

Both modes explicitly set `resume_vram_working_mib: 0` for this already-allocated,
bounded workspace, while retaining a 320 MiB effective base reserve and a
250 MiB sampled safety guard. This does **not** qualify zero additional working
space for arbitrary long prompts or images. The conservative default remains
256 MiB additional return allowance; its fallback requires separate validation.

## Earlier failed case and correction

The first resident candidate completed the compiler but failed return admission:
the 320 MiB base plus default 256 MiB working allowance required 576 MiB free.
The native prompt floor only allowed the expert cache to shrink from 672 to
640 MiB, increasing free VRAM from about 340 to 372 MiB. The request timed out
before model dispatch. CUDA and Windows budget telemetry agreed; this was not
a counter discrepancy or a successful speed result.

The corrected `auto` path can park and verify both processes have exited, clear
the temporary resident controls, and use guarded reload. It never replays the
completed tool. Strict resident mode still returns a bounded refusal rather
than silently changing lifecycle policy. An uncertain process exit retains the
resource barrier instead of admitting a second process.

Four additional component arms passed on the same frozen candidate:

| Scenario | Observed behavior | Admission | Return request |
| --- | --- | ---: | ---: |
| RAM relief | Released 1,134 MiB of resident expert RAM; kept both processes; 2,096 compiler checks passed | 6.703 s | 5.000 s |
| Strict resident, 2 GiB GPU headroom requested | Refused before starting the compiler; kept both processes; inference worked after release | 1.360 s | 5.203 s |
| Auto, 2 GiB GPU headroom requested | Resident relief was insufficient; confirmed unloading before compiler admission; 2,096 checks passed | 3.438 s | 67.593 s |
| Default 256 MiB return allowance | Kept both processes during compilation, then observed `resident_return_auto_unload` and guarded reload; 2,096 checks passed | 0.406 s | 69.718 s |

The fallback tests corroborated replacement of both native and vision process
identities, correct `42` continuations and no owned survivors. The minimum
sampled native free VRAM across these four arms was 322 MiB; available RAM
remained above 9.472 GiB. Each scenario has one qualification run, not a repeated
performance distribution. The strict arm is an expected refusal test, not a
successful compiler task.

## Actual Hermes supervisor and worker

The frozen candidate above also passed one real Hermes CLI qualification on
Hermes commit `c09e4cacedd04733d94f3d3c7af341367bb6ac26`. The model selected a
supervisor resource plan, delegated to one sequential worker, and ran exactly
one child compiler command. The broker acquired and explicitly started the
grant, renewed it during a declared hold, and released it after owned compiler
descendants had exited. AUTO selected `none`, retaining both model processes.
The allowlisted record is [hermes-resident03-results.json](hermes-resident03-results.json).

All 2,096 fixture checks and 104 independent checks passed. The supervisor's
normal final answer included the actual compiler result and a fresh tool-only
nonce, corroborated by linked tool results and API records. All six recorded
tool-bearing supervisor/worker requests used high reasoning. The control
credential was absent from the real terminal child.

The agent interval was 217.406 seconds; the complete harness took 291.968
seconds, including model startup, vision, independent checks and cleanup.
Actual compilation/testing took 2.110 seconds. Its 17.596-second foreground
wrapper included a **deliberate 15-second hold** to exercise lease renewal;
that hold is not compiler work. Minimum available RAM was 8.229 GiB and sampled
native free VRAM was 322 MiB. No owned process survived final cleanup.

Vision answered the actual red-square image correctly **before** the agent
workflow, with the same vision process still present afterward. An actual
`add(17, 25)` function call, tool result, and correct `42` continuation passed
**after** the workflow. This is not a claim of a post-build image test.

There were no instruction-following deviations or failed acceptance gates in
this run. It used the same explicit zero additional return-VRAM allowance as
the bounded component profile. It validates orchestration of a supplied correct
fixture, not autonomous code creation or repair. It is one qualification, not a
matched speed comparison against an earlier Hermes prompt or v0.1.42.

## Limits

- The component sections are direct API tests. The separately labeled Hermes
  section is a real bounded agent workflow; neither tests model-authored code.
- Configured 64K is not a test with 64K of occupied context. The component
  requests contain 1,129 input and 73 generated tokens.
- A correct lookup and a deterministic compiler fixture do not establish
  general agent competence, universal no-OOM behavior or desktop responsiveness
  under every external workload.
- CPU vision was loaded, but these component arms contain no image.
- Only this CUDA/Windows single-owner configuration was measured; HIP, SYCL,
  multiple replicas and helper-GPU lifecycle ownership are not qualified here.
- The benchmark fixture used GCC 16.2.0 with `-march=znver4`, not LLVM. Enabling
  this target permits appropriate CPU instructions; it does not prove every
  operation used AVX-512.

Implementation provenance and the sources actually adapted are documented in
[ADAPTIVE_PROVENANCE.md](../../../docs/ADAPTIVE_PROVENANCE.md). Measurements here
do not imply endorsement by the cited projects or authors.
