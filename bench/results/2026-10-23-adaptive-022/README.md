# Adaptive public candidate: v0.2.2 and v0.2.3.1 qualification

This report concerns the public-only adaptive branch for PR #1493. It excludes the separate local checkpoint bank, startup-budget integration and Windows default-token DACL correction. Private combined-runtime results do not qualify this source.

The published predecessor is `667deb1594ffd975b83f2b8e9c2da3beaf34e439`. Merge `be8a6183773a4f93a082d0d906f3d514440c1f95` incorporates upstream v0.2.2 `011c9e600a93bbc9fd451c563501b5831c32bbca` without rewriting that history. Commit `037314f905130d6607331c3df6245bdde365266c` validates adaptive startup against effective user arguments. Commit `e39dbb99dcc5c3a27834017f6c913588f9f5348b` adds the five-line idle-HTTP fixture synchronization described below. Report-only commits after these do not change executable source.

## Environment and commands

Windows 11, Python 3.11.9. Tests ran from the repository root with isolated temporary directories and bounded owned child-process jobs. The full run's environment excluded inherited `STRATA_*`, `HERMES_*` and `OPENAI_*` variables. The commands below are the test commands; the local containment wrapper is not required by the product.

```text
python -B -m unittest discover -s serve -t .
python -B -m unittest serve.test_idle_parking_service -v
```

## Observed results

**Failed:** The complete server run on `037314f9` executed 1,402 tests in 408.886 seconds: two failures, one error and 21 skips. This run remains failed; the focused checks below are not a replacement complete-suite pass.

One failure was `test_request_reservation_covers_load_to_prepare_and_generation_gap`: the HTTP client could receive the response before the server's `finally` released its preparation reservation. The fixture now waits up to three seconds for that release, preserving the existing response, reservation and unload assertions. The change affects only test synchronization.

**Passed:** All 42 `serve.test_idle_parking_service` tests passed in 4.920 seconds on `037314f9` with exactly that five-line correction, subsequently committed unchanged as `e39dbb99`. No runtime source was changed by the correction.

The other failure and error were classified against exact upstream v0.2.2 in separate disposable interpreters. Each case ran these same three existing tests:

```text
serve.test_lifeline.Lifeline.test_a_child_ends_when_its_server_is_killed
serve.test_lifeline.Lifeline.test_the_child_is_recorded_while_it_runs_and_forgotten_after
serve.test_lifeline.Lifeline.test_a_server_that_exits_normally_leaves_nothing_behind
```

| Source and token defaults in the test interpreter | Result |
| --- | --- |
| Exact unmodified v0.2.2, ordinary defaults | 3 passed |
| Exact unmodified v0.2.2, after `adopt_user_defaults()` | 1 passed, 1 failure, 1 error |
| Public candidate `037314f9`, after `adopt_user_defaults()` | Same failure and error |
| v0.2.2 with only token-default `FA` changed to `GA`, after adoption | 3 passed |

The relevant `serve/lifeline.py`, `serve/test_lifeline.py` and unpatched `strata_control/private_files.py` were byte-identical between the base and candidate. The failed cases lose readable child identity and the associated ownership record. All four isolated jobs emptied. This is the independent Windows issue addressed in [PR #3399](https://github.com/Niko1221/Strata/pull/3399); its correction is not folded into this adaptive branch. The result does not establish broader Windows ACL behavior or a complete-suite pass with that correction.

## Upstream v0.2.3.1 merge

Merge `e124281321603ce4b37dcf2337b49b82b898e037` incorporates upstream `7ddc17a38fe7f31637e895ecfba58c2cd43d65ec` into published candidate `8f99733c628b53e2ba8d0639baddacea46479b42`, preserving both histories. Its resolved source tree is `b37e52d25927fda75f8e8fc606553afc888a05dd`.

The two conflict files were `serve/server.py` and `serve/web/app.js`. The resolution keeps upstream's richer vision startup diagnostics, tool-picture handling, drafting configuration and in-memory API key, while retaining adaptive admission, confirmed child shutdown and resource-control refresh. Focused regressions check that failed admission or uncertain vision shutdown cannot silently become an omitted picture, and that a changed API key refreshes resource controls even when browser storage fails. The native adaptive contributions in `CMakeLists.txt`, `include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp` and `src/program/generate.cpp` merged automatically; a source comparison found no lost added/deleted contribution lines. This comparison is not native build evidence.

**Passed:** The resolved source completed the following server/web modules, each in a fresh interpreter, using `python -B -m unittest <module> -v`:

| Module | Tests collected |
| --- | ---: |
| `serve.test_adaptive_0142_integration` | 25 |
| `serve.test_adaptive_0144_integration` | 10 |
| `serve.test_adaptive_0145_integration` | 10 |
| `serve.test_adaptive_upstream` | 3 |
| `serve.test_idle_parking_service` | 42 |
| `serve.test_vision_tool_image` | 4 |
| `serve.test_vision_start_error` | 7 |
| `serve.test_vision_token_limit` | 8 |
| `serve.test_tool_call_envelope` | 11 |
| `serve.test_tool_call_unclosed` | 6 |
| `serve.test_native_tool_parser` | 8 |
| `serve.test_responses` | 41 |
| `serve.test_web_api_key` | 9 |
| `serve.test_draft_layer` | 4 |

Total: 188 collected, 187 passed and one optional `jsonschema` check skipped, in 39.078 seconds including wrapper and interpreter overhead. Every module exited zero. The owned job emptied and all 54 retained child handles signaled exit.

**Passed separately:** `python -B -m unittest tests.terminal.test_drafting -v` ran 40 tests in 7.746 seconds. Its launcher exited zero and its owned job emptied. This CPU-only check does not claim retained handles for every descendant: the CLI adopts the upstream token defaults described above, which deny the observer's required child-query rights.

**Harness failures, preserved separately:** Two preliminary runs stopped at child-handle observation, before a complete final result. A controlled comparison on this source confirmed that opening a child with synchronization, limited-query and termination rights succeeds before `adopt_user_defaults()` and fails with Windows access-denied error 5 after adoption. Both controlled cases exited and their job emptied. Module isolation prevents a startup test's token mutation from contaminating later web tests; the terminal check uses the narrower cleanup evidence stated above. Neither preliminary run is counted as passing, and no token-default correction was added to this branch.

These focused checks do not replace the earlier failed complete server run or qualify a model workload. The public branch still excludes the private checkpoint bank and startup-budget changes; upstream's new in-memory conversation-cache sizing is separate from that private implementation.

## Remaining validation

**Not run:** Exact-source CUDA native build/test qualification for this candidate; HIP, SYCL and Metal builds or hardware execution; a public-only real-model lease/compiler/reload workflow; visible Desktop, long-history compression and media acceptance. A prepared command or a successful private prototype run is not evidence for these checks. Native results will be added only after an actual bounded run and cleanup verification.

There is no performance comparison or universal memory-safety/progress claim. Capacity observations are not operating-system reservations. The branch remains a draft. The tracker-reset `/reopen` format check is distinct from the all-touched-backend build requirement before requesting review.
