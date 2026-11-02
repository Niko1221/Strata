# Lifetime usage: v0.2.9 conflict resolution and field evidence

Revalidated on 2026-11-02. Server/UI revision `33b9c6241862ea0e32c841a1e94d234ed090b0e9`
merges upstream `c62d5e3a15b40ea51c96aae9e8d36928965f53ab` (v0.2.9) into existing PR #2979.
The merge preserves the original contribution history. Browser regressions are in `b37a743d`; later commits add evidence/documentation only.

## Compatibility changes

- Keep upstream's asynchronous language/chat startup, with remembered Custom dates initialized before the first poll.
- Keep upstream's load/unload response guard in both scheduled and date-triggered metric refreshes.
  A rejected stale response returns to the existing poll loop; it does not start an extra loop.
- Keep the new upstream documentation structure and add the statistics section alongside the serving features.
- Usage remains opt-in and at the bottom of Python Monitor. No native engine, model, scheduler,
  production configuration, automatic switching or idle-unload setting was changed.

## Real deployment evidence (historical implementation, not a new-version benchmark)

A **read-only** aggregate snapshot from the existing Windows deployment was collected at
**2026-11-02 11:06:10 UTC**. This deployment predates this v0.2.9 port. Its counters can include
multiple sequential model profiles, development work and tests; they are not attributed exclusively
to SC117 or a single task. The two categories must not be mixed when calculating rates or date windows.

| Counter | Recorded by the ledger | Undated import from retained logs | Headline total |
| --- | ---: | ---: | ---: |
| Output tokens | 15,624,110 | 6,396,046 | **22,020,156** |
| Requests | 15,307 | 8,530 | **23,837** |

Tracked input totals: 1,923,713,023 prompt tokens, of which 1,898,672,411 were reused;
25,040,612 required reading. Reused input is **not newly generated output**.

[Allowlisted raw aggregate counters](production-counters.json) include the timing and draft counters.
[Reproducible arithmetic/load-save-reload check](check_evidence.py) verifies these exact sums and
that the new Usage implementation preserves the published aggregate snapshot through two round trips
in a temporary directory. [Observed result](evidence-check.txt).
Hourly/daily buckets were deliberately not exported, so this evidence cannot independently validate historical date ranges.

The [earlier user-supplied production screenshot](production-history.png) shows 20,097,558 total
output tokens = 13,701,512 tracked + 6,396,046 imported. It was taken before the new snapshot,
not generated using test fixtures. EXIF/XMP metadata was removed; decoded pixels were verified identical. It illustrates why the permanent headline, import label and
separate range totals matter. Its capture time was not independently established.

Only aggregate counters and this statistics-only screenshot are published. No prompts, answers,
tool arguments, credentials, machine paths or per-request history are included. These observations
show practical use over many requests; they do not prove zero lost counts, zero overhead, model
quality, a cache speedup or v0.2.9 production stability. Those require separate measurements.

## Validation

Commands are run from the checkout root in isolated test environments. HTTP and browser checks
use mock engines; the PC production services were not stopped, upgraded or used for test inference.

### Passed

- Windows 11 (10.0.26100), Python 3.13.1: **412 tests, OK, 3 skips**, 152.171 s.
  `python -m unittest serve.test_server serve.test_usage serve.test_usage_durability serve.test_monitor serve.test_prometheus serve.test_runconfig -q`
  ([raw log](windows-tests.txt)). Unix-only signal cases are skipped on Windows.
- Mac and Windows: `python docs/usage-v020/check_http.py`: eight requests per host, enabled/disabled
  statistics, streaming/non-streaming Chat Completions and Responses. Both arms produce 72 mock output
  tokens; API, session and saved totals match; reload/custom range checks pass. Disabled mode creates
  no ledger or usage metric. Normalized payloads match. [Mac](http-mac.txt), [Windows](windows-tests.txt).
- Chromium/Node v24.3.0: **7 new usage browser tests** in `tools/webtest/usage_test.mjs`, plus **5 existing
  load/unload tests** in `tools/webtest/browser_test.mjs 'load/unload:'`, all passed.
  Run with `WEBTEST_MODULES=<folder containing node_modules/puppeteer-core>`.
  [Usage log](browser-usage.txt), [load/unload log](browser-load.txt). Tests use the repository's mock HTTP server.
- All four `node serve/test_web_*.js` files pass ([log](js-tests.txt)); `node --check serve/web/app.js`
  and `git diff origin/main --check` pass. Compare whitespace against current main, not the old PR base.
- `python docs/usage-v029/check_evidence.py` preserves all published aggregate values through a temporary
  store round trip; production is only read, never overwritten.

### Failed and investigated

- macOS / Python 3.13.0: `python -m unittest discover -s serve -p 'test_*.py' -q`:
  **1,326 tests, 1 failure, 21 skips**, 427.109 s ([full log](mac-suite.txt)).
  `test_gpu_roles_server.Roles.test_every_card_off` expects `SystemExit`, but upstream's
  `card_roles_lines` explicitly permits CPU-only operation on Darwin. This function is unchanged by this PR.
  The exact single test fails both on the candidate ([log](mac-candidate-control.txt)) and on an unmodified
  archive of upstream `c62d5e3a` ([log](mac-upstream-control.txt)). The full Mac run is **not green**;
  this unrelated upstream platform assertion is left out of this feature patch.
- The first attempt to detach the isolated Windows test launcher did not survive the SSH session;
  no tests ran and its log was empty. Retried with the SSH session kept alive; both checks then returned 0.
  No production process was stopped.

### Not run / limits

- Native CUDA/HIP/SYCL/Metal builds or model inference: no native source changed; no new performance claim.
- Physical power loss or a new persistence-latency benchmark; requests killed before settlement remain excluded.
- Historical per-request/date reconstruction; only the aggregate snapshot was exported.
- Separate desktop/TUI statistics UI; this PR changes the Python server's `serve/web` Monitor.
- Maintainer scope approval remains pending given overlapping #2370/#2400/#2204 proposals; the PR stays draft.

Earlier beta tests remain in [their dated report](../usage-v020/README.md); they are not substituted
for the v0.2.9 checks above. Diagnostic logs redact usernames and local temporary paths, leaving failures intact.
