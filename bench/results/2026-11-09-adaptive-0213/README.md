# Adaptive draft: upstream 0.2.13 compatibility

This records conflict maintenance of the existing draft, not a new feature release or a performance result. The merge preserves both public parents: draft `06ba725d8b9bb42b457302f71dd85204e9760c32` and upstream `504b143342a9fe1aa8320a7b31e11f5c0ac265c3` (after the v0.2.13 tag). No installed model, engine, or hardware settings were changed for these checks.

## Integration decisions

- Keep the upstream CMake backend layout. Move the draft's existing CUDA driver linkage and test targets into the corresponding backend modules.
- Keep upstream drafting, log-probability, request-ID, deferred-tool, vision and shutdown paths. Guard the new drafter startup retries with the existing admission cancellation owner, so a guarded attempt cannot silently start a replacement process outside that owner.
- Preserve the draft's single-engine restriction for the new explicit split-serving configuration. Ordinary split serving without adaptive features is unchanged.
- Preserve the complete probabilistic proposal packet during verification. Shortening its `T` or `n_q` would incorrectly reinterpret an offered token as a NULL tail. Bound the accepted commit prefix afterward using the existing output-window helper, before EOS handling and log-probability reporting. The added deterministic counterexample and prefix matrix test this integration; they do not execute the GPU verifier.
- Distinguish an ordinary encoder-start failure from model ownership or uncertain teardown. Tool-returned pictures retain upstream's omission-note behavior for the former; `EngineDied`, `EngineStuck`, `ModelBusy` and request cancellation still propagate unchanged. User-supplied image errors retain the existing `EngineDied` error family.

## Host validation

Windows, Python 3.12.9, Node 24.14.0 and GNU C++ 16.2.0. No GPU model was loaded.

| Check | Observed result |
|---|---|
| 36-module combined server run listed below | 607 tests in 310.151 s: one error, two skips. The error exposed the encoder-start distinction above; this run remains recorded as failed. |
| Six affected vision/admission/lease modules after that correction | 89 tests in 13.949 s, all passed. |
| Final tool-image module, including cancellation and exact ownership-exception identity | Five tests in 2.058 s, all passed. |
| Added deferred-tool forwarding/reservation test | Passed separately; also included in the subsequent 89-test run. |
| Resource-controls JavaScript tests | 12 passed. |
| Python syntax, JavaScript syntax, scoped merge whitespace/conflict check | Passed. |

The combined run used these modules with `python -m unittest`:

```text
serve.test_adaptive_0142_integration serve.test_adaptive_0144_integration
serve.test_adaptive_0145_integration serve.test_adaptive_commit
serve.test_adaptive_upstream serve.test_background_policy
serve.test_background_service serve.test_c1_lease_races
serve.test_c1_lease_service serve.test_coadaptive serve.test_coadaptive_reserve
serve.test_coadaptive_route_wire serve.test_idle_parking_service
serve.test_live_memory serve.test_memory_policy serve.test_memory_policy_http
serve.test_memory_policy_lease serve.test_parking_profile_identity
serve.test_request_parking serve.test_request_parking_service
serve.test_resource_lease serve.test_resource_lease_service
serve.test_resource_presets serve.test_service_resources
serve.test_drafting_retry serve.test_logprobs serve.test_logprobs_parallel
serve.test_split serve.test_responses serve.test_vision_elastic
serve.test_vision_tool_image serve.test_vision_start_error
serve.test_control_cancel serve.test_context_overflow
serve.test_restart_waiters serve.test_stream_ownership
```

The post-correction run was:

```sh
python -m unittest serve.test_vision_tool_image serve.test_vision_start_error serve.test_vision_elastic serve.test_idle_parking_service serve.test_c1_lease_service serve.test_adaptive_0145_integration
node tools/test_resource_controls.cjs
```

Earlier preliminary runs exposed stale fake-engine slot state and the added deferred argument, plus the need to preserve uncertain encoder teardown. Those were corrected before the combined run. The complete 607-test selection was not repeated after the final narrowly scoped encoder correction; the passing affected modules are reported separately.

## Native host checks

CPU-only CMake configuration succeeded with CUDA, HIP, SYCL and the native expert library disabled. Six separately built host executables passed:

| Program | Observed result |
|---|---|
| `serve_window_test` | 15,588 assertions, including full-proposal verification with a bounded commit prefix |
| `serve_stop_bound_test` | 2,736 cases |
| `gated_proposal_test` | 276 assertions, including the counterexample to trimming a probabilistic proposal; no GPU or actual main-window copy test |
| `live_prefill_policy_test` | 9,794 assertions |
| `background_control_test` | Passed |
| `request_stop_test` | Passed |

## Remaining qualification

No complete CUDA, HIP, SYCL or Metal build was performed for this merge. There was no current-head GPU model, numerical comparison, live-memory resizing, compiler/lease handoff, physical interruption or performance qualification. Earlier successful builds and earlier failed complete suites remain tied to their recorded revisions. This maintenance does not replace them or establish that the draft is ready for review. The feature remains opt-in and the PR remains draft.
