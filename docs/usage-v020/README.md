# Lifetime usage: 0.2.0-beta revalidation

Tested on 2026-10-19–20 (Europe/Berlin), for existing PR #2979. Server code:
`c7df4b95b8a30ce3bde61c3972c1ff0cc545a95d`, based on upstream
`e88ee39d1d6be6e3364183af3b39b62a7ed6f211` (0.2.0-beta).
The six original #2370 commits retain their authorship. #2370 was still closed and unmerged when checked.
This report does not claim maintainer agreement on the consolidated proposal.

## Result and beta adaptation

The existing patch applies to beta with an import conflict resolved by retaining both imports. The ledger,
opt-in settings and bottom-of-Monitor card are retained. No native engine source or production configuration
was changed; no model was loaded for this revalidation. Production Maya and Strata were not stopped.

The beta port exposed a lifecycle issue: after `server.main()` returned in an embedded process, the usage
store stayed locked until process exit. Reproduced with real temporary ledger locks and mocked server
lifecycles in three subcases: normal stop, ready-file publication failure and HTTP bind failure. The fix
closes the ledger on those paths and unregisters its exit hook. All three subcases now pass.

## Commands and observed results

```sh
python -m unittest serve.test_server serve.test_usage serve.test_usage_durability serve.test_monitor serve.test_prometheus serve.test_runconfig -q
python docs/usage-v020/check_http.py
node --check serve/web/app.js
git diff --check
```

The Windows tests ran from an isolated source directory, using the existing isolated beta-test Python
3.13.1 environment. Mac used Python 3.13.0 in the isolated beta-test venv on macOS 27.0.1 / Apple M5 Pro.
No GPU inference was involved; the PC is Windows 11, Ryzen 9 5900X, 128 GB RAM, RTX 3090.

### Passed

- Windows: **394 tests run, OK, 3 skipped**, 133.889 s. The three skips are Unix signal cases.
  [Raw log](windows-tests.txt).
- Ledger tests cover settled requests followed by idle time, abrupt process exit, concurrent writers,
  competing process ownership, atomic replacement failure/retry, corrupt-file preservation, fractional
  timing, undated imports, tool-call token accounting and disconnect settlement.
- HTTP probe on both Mac and Windows: eight requests per host, Chat Completions and Responses with
  streaming off/on, and statistics disabled/enabled. Each four-request arm generated 72 mock tokens.
  Enabled API counters matched every persisted counter and session total; reload preserved them;
  a custom date range returned the correct output total. Disabled mode created no ledger or lock file
  and added no usage metrics field. Reply payloads were equal after removing protocol IDs and timestamps.
  [Mac log](http-mac.txt), [Windows log](http-windows.txt), [probe](check_http.py).
  This compares semantic payloads, not byte-identical independent HTTP streams or native inference.
- Lifecycle regression: all three previously failing subcases pass. [Before](lifecycle-before.txt),
  [after](lifecycle-after.txt).
- JavaScript syntax and Git whitespace checks passed.
- Browser inspection at 1280×720 confirmed speed/hardware/cache first, Usage at the bottom, working
  Lifetime/7-day/Custom selections and a permanent all-time total. Screenshots use synthetic usage data
  and actual preview-host hardware telemetry; they are not inference measurements.
  [Top](monitor-top.png), [Usage bottom](usage-bottom.png).

### Failed and investigated

- First Mac attempt used the system interpreter without `regex`: 296 tests ran with 146 errors,
  42.906 s. Retained as [environment setup failure](mac-missing-dependency.txt); rerun used the prepared venv.
- Mac venv suite: **394 tests, one error**, 131.287 s. The failing upstream case is
  `serve.test_server.VisionArgs.test_min_tokens_is_passed_only_when_set`: its mocked process lacks `pid`.
  Running that exact test on an unmodified archive of upstream `e88ee39d` reproduced the same error.
  [Candidate log](mac-tests.txt), [unmodified upstream control](mac-upstream-control.txt).
  The ledger tests passed; the entire Mac suite is not reported as green.
- Initial lifecycle regression failed in all three subcases, before the cleanup fix (linked above).

### Not run and limits

- No CUDA/HIP/SYCL/Metal build or native-model run; these source trees were unchanged. No claim of
  inference speed improvement or new backend compatibility. Older native integration results in
  [the historical report](../usage-stats-validation.md) are not substituted for beta validation.
- No physical power-loss test. Counts from an active request killed before settlement remain excluded.
- No new persistence latency benchmark; the historical October 17 numbers remain labelled as historical.
- The card is in the Python server's Monitor (`serve/web`); the separate new desktop/TUI UI was not extended
  or validated by this patch. Phone layout was not re-tested in this round.
- Scope still needs maintainer agreement given overlapping #2370/#2400/#2204 work. Keep this PR draft for
  that discussion. No production deployment is part of this revalidation.

Raw logs have local usernames and machine paths replaced with placeholders. Failures remain intact.
