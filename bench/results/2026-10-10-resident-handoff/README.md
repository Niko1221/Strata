# Resident supervisor handoff: bounded Windows qualification

The matched comparison below uses the v0.1.41-based C1 implementation at
`efb21996fea649622cb84fdedd5765f425dbbe5f`. The separately labeled v0.1.42
qualification uses `1cb95593f5dd10954ba4ade6c957589ef820267b`; it is not a
matched release-speed comparison. Raw local transcripts, credentials and
private paths are excluded.

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

## v0.1.42 integration and component qualification

The updated source includes upstream v0.1.42 at `61b3fb5d`. Its native binary
SHA-256 is `d39a23a174d668cf3a7f0983457741b080b52c6797323cbd614fd16b774a52f9`.
The allowlisted [qualification record](v0142-qualification.json) contains exact
source/server/harness hashes, per-arm settings, cleanup and test counts.

The port preserves the new seven-field restart metadata, CPU affinity,
queue/KeepAwake ownership and SAVE retry. A real lazy-engine regression test
covers the runtime identity change. Adaptive replicas/helper GPUs and live
memory with asynchronous expert swaps are rejected before allocation because
this controller does not own those lifetimes. Default non-adaptive support is
not removed.

Final server validation ran 1,128 tests: 1,119 passed and nine expected skips,
with no resource or unraisable warnings. The broker passed 54 tests. CUDA sm89
passed 14 selected native tests and two bounded IQ3_S fused-reference pairs;
the CPU-only build passed 19 selected tests. HIP/SYCL are unqualified. These
counts overlap focused suites and are not additional independent trials.

The two fresh component arms kept tail skipping disabled, PLE I/O explicitly
`direct` and PCIe fraction 0.37. High reasoning, IQ3_S, configured 64K context,
CPU vision and the supplied four-job C++ fixture stayed fixed.

| v0.1.42 arm | Additional return VRAM | Tool admission | Compiler | Return request | Complete cycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resident AUTO | Explicit 0 MiB | 0.437 s | 2.422 s | 4.719 s | 23.563 s |
| AUTO with default return allowance | Default 256 MiB | 1.109 s | 2.422 s | 70.750 s | 98.672 s |

Both answered `42` before and after the tool and passed all 2,096 checks.
The first arm retained both model identities and reused 1,124 prompt tokens.
The default arm retained them during compilation, then observed
`resident_return_auto_unload`, confirmed both old processes had exited and
completed guarded reload. Minimum sampled native free VRAM was 324/340 MiB,
and available RAM was 9.179/9.400 GiB respectively. Cleanup had no survivors.

This validates both lifecycle paths on the update. It does not establish a
v0.1.42 decode/prefill gain, a safe zero allowance for arbitrary tasks, or a
default end-to-end speedup. The conservative additional allowance is still
256 MiB; the low latency profile is explicitly bounded to already-allocated
buffers. Candidate04 was preparation-only and never executed.

The first actual v0.1.42 Hermes run is **failed overall**, retained in
[hermes-resident05-results.json](hermes-resident05-results.json). All 17
resource-lifecycle checks and the worker's 2,096 compiler cases passed, but only
10 of 12 final workflow gates passed. The supervisor received the complete,
untruncated worker result and then redundantly read the worker log with an
additional terminal command. It did not compile twice; it did violate the
explicit one-command/no-supervisor-terminal test rule. No source or checker
change relabels this attempt as successful.

The agent interval was 230.957 seconds; sampled minima were 9.216 GiB RAM and
322 MiB native VRAM. Vision before the workflow passed. The independent 104
cases and post-agent function continuation **did not run**, because the harness
stopped at the failed gate. Owned-process cleanup still passed. This is agent
instruction-following variability, not evidence of a failed lease or a
demonstrated engine regression; the cause of the redundant verification is not
established.

One unchanged-protocol repeat is retained in
[hermes-resident05-r2-results.json](hermes-resident05-r2-results.json). It also
failed overall: 17/17 lifecycle and 10/12 final workflow checks passed. The
supervisor tried an unavailable search tool, then read the worker log despite
having complete feedback. The worker compiled only once and passed 2,096 cases.
The later independent and function-continuation stages again did not run.
There was no third unchanged repeat. These two attempts are lifecycle successes
and strict workflow failures. The separately patched Hermes qualification below
does not relabel either attempt.

The repeat agent interval was 265.892 seconds; sampled minima were 9.423 GiB
RAM and 322 MiB native VRAM, with clean owned-process teardown. A separate
mocked HTTP/sampler test ran for 0.585 seconds during the repeat (about 1.55
seconds command wall); this timing is not presented as uncontended. In both
attempts, `verify_on_stop` was configured but did not inject a verification
nudge: there were no changed-code paths or synthetic follow-up messages.
Standing verification guidance was present, but its causal role is unproven.

The first remote CI run passed Windows Python and both native CPU jobs, but
Ubuntu exposed a race in the inherited sampler test: it counted every server's
thread while unrelated samplers could finish stopping. The test-only correction
in `e2529d11` verifies its own bound sampler and closes the server in `finally`.
The original failure is preserved in
[run 38073230611](https://github.com/midhatn/Strata/actions/runs/38073230611);
the corrected source is checked by
[run 38073750104](https://github.com/midhatn/Strata/actions/runs/38073750104).
This is a test-ownership fix, not a performance improvement or an imported
research technique. The measured runtime and native hashes above are unchanged.

## Hermes delegated-evidence correction and complete .42 workflow

The separate [Hermes PR #136266](https://github.com/NousResearch/hermes-agent/pull/136266)
clarifies when complete, consistent worker check evidence can satisfy supervisor
verification and adds compact terminal outcome/exit/truncation metadata linked
to the actual tool call. Missing, failed, truncated, conflicting or unsupported
evidence still needs follow-up; explicitly required independent checks remain.
The stop-verification hook is unchanged. This is not a task-verified flag.

Paired Hermes source `873419f6bfc7c19cd18406d4da91027b4cbcf65e` passed the original
strict task on the same frozen Strata runtime, native binary and supplied fixture.
The [allowlisted result](hermes-evidence01-results.json) records all **17 lifecycle
and 12 workflow gates passing**, one sequential worker, exactly one child compiler
command and no additional supervisor command. The build passed 2,096 fixture
cases; all 104 independent cases, the initial vision check and the real post-agent
function-call/result continuation also passed. Both model process identities
remained unchanged through the final continuation, and cleanup left no survivors.

The agent interval was **215.382 s** and the full harness **287.691 s**, including
startup, independent checks and cleanup. The compiler took 2.179 s; the foreground
wrapper deliberately held the lease for an additional 15 seconds to test renewal.
Minimum sampled available RAM was **4.572 GiB** and native free VRAM **322 MiB**.
All six tool-bearing model requests used high reasoning. This is one bounded
success, not a matched speedup or a general instruction-following guarantee.

The harness now continues independent correctness/function checks after a
workflow-only failure while retaining failed overall status. Resource, ownership
or containment failures still abort. Nine checker regressions passed, and no gate
was relaxed. The earlier failed records and their skipped stages remain intact.

The Hermes patch has 20 passing regression tests; the untouched base fails 17 of
those tests. Its broader focused suite passed 188 tests with four skips and three
optional-schema/provider dependency failures reproduced on the untouched base.
All 11 repository checks passed. External API read-back behavior was not exercised
by this local compiler fixture. The standalone upstream PR contains no middleware
dependency; this paired validation includes the existing resource middleware.

The test retains the bounded explicit zero additional return allowance described
above. The separately installed adaptive package now pins this tested Hermes
source, while its default additional return allowance remains 256 MiB. No Strata
kernel/configuration change caused this agent improvement.

## Separate projection-disabled workflow

A fresh run disabled experimental speed projection in all configuration and
request paths and removed native control-vector loading arguments. Native
`INFO cvec=0` was checked before the first request and after the last; all ten
recorded requests explicitly disabled projection. The same frozen runtime,
paired Hermes `873419f6`, supplied correct fixture and strict acceptance gates
were retained. Earlier projection-enabled successes and failures remain intact.

The [separate allowlisted result](hermes-no-projection01-results.json) passed all
**17 lifecycle and 12 workflow checks**, 2,096 compiler-fixture cases, 104
independent cases, CPU vision and the real function/result continuation. There
was exactly one sequential worker and one compiler command, with no redundant
supervisor command. Both model identities were retained and cleanup was verified.

The agent interval was **217.872 s** and whole harness **292.981 s**. Sampled
minimum available RAM was **9.882 GiB** and native free VRAM **328 MiB**. AUTO
selected `none` because sufficient headroom was already available; this run does
not demonstrate forced memory relief or pressure-triggered unload. It retains
the same explicit zero **additional** return allowance for this bounded shape;
the general default remains 256 MiB. These separate single runs do not establish
a causal speed or memory improvement from disabling projection.

## Projection-disabled default return allowance

The [separate component result](default256-return-projection-off-results.json)
also passed with the ordinary **256 MiB additional VRAM return allowance**.
Both high-reasoning lookups returned exactly 42, and the four-job compiler passed
2,096 fixture checks. Native `cvec=0`, absent control-vector arguments and explicit
projection-disabled requests were checked before and after the model reload.

AUTO selected `none` at tool admission and retained both model processes during
compilation. On inference return it requested the normal 320 MiB floor plus the
256 MiB allowance, or **576 MiB**. Cache trimming reached 640 MiB from 672 MiB,
but sampled free VRAM reached only 374 MiB before the prompt-workspace floor
prevented further release. The existing fallback then verified full unloading
and guarded reloading; both old processes were replaced and the answer passed.

The post-tool HTTP request took **71.375 s including reload**; the complete
model/tool/model cycle took **99.625 s**. Minimum sampled available RAM was
**9.789 GiB** and native free VRAM **342 MiB**. Cleanup left no owned survivors.
This is a direct API lifecycle test, not an additional full Hermes run, a speedup
comparison, or proof that the ordinary allowance permits resident continuation.
The earlier bounded zero-additional-allowance measurements remain separate.

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
