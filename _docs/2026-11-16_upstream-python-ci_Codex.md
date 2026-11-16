# Python and Terminal CI corrections, 2026-11-16

The correction starts at main `32c07800e0c384ab11cdede5dfd3b0be82fadb60`. It addresses the eighteen Linux failures and the same eighteen plus two Windows failures in Checks run [43777967951](https://github.com/zapabob/Strata/actions/runs/43777967951). The final local selection passes 139 tests without skips. This is focused Windows evidence, not a replacement for a new hosted run on Linux, macOS and Windows.

## Observed failures and requirements

The original Linux job completed with 4212 passed, 18 failed, 120 skipped and 636 passing subtests in 1799.24 seconds. Windows completed with 4210 passed, 20 failed, 118 skipped and 639 passing subtests in 3117.72 seconds. Both produced completed pytest results; neither failure was a suite timeout. The Windows-only pair expected an exact one-second timeout sentence from real sockets, while the measured wait rounded to two seconds.

The separate Windows Config snapshot job in [run 43777967970](https://github.com/zapabob/Strata/actions/runs/43777967970) reached 52 pytest progress dots without F/E before tools/check killed its child at 900 seconds. Its 0/0 summary was missing final JUnit counts. The same source's Linux snapshot passed 64 tests in 7m52s. The earlier local Windows snapshot passed 60 tests in 1328.28 seconds before the final upstream hotfix. That historical timing supports a larger bounded Windows budget; it does not prove that the current 64-test Windows snapshot completes.

The change must retain the fork's DeepSeek behavior, Qwen-only KVA conditions, fail-closed process ownership checks, normal network deadlines, and visible launch choices. No golden snapshot was edited or regenerated in this correction. Product versions and release qualification remain unchanged.

## Implementation

`packaging/package_terminal.py` now carries the three DeepSeek scripts run by path: pack_gguf, smoke and tune. Its existing dependency traversal also carries their local imports. The bundle regression invokes each script's --help from an otherwise empty working directory with no checkout PYTHONPATH. `packaging/make_model_release.py` classifies DeepSeek archives separately and includes their unchanged names and sizes in the kept artifacts of both case A and case B. DeepSeek archives never enter the Qwen shape-engine build inputs. The existing case A/B retention assertions remain.

On POSIX, `strata_control/procs.py` enumerates PIDs, matches the executable, and only then reads one candidate ownership snapshot. Unrelated command lines are not fetched. Windows keeps its inexpensive name filter and coherent candidate snapshot. Unknown, denied, Boolean, nonfinite or invalid ownership fields still cannot establish orphanhood. Existing live-parent, init adoption, denied-field and PID identity regressions remain in the focused selection. The privacy fixture explicitly exercises POSIX and asserts that no full snapshot is requested for a foreign executable; the existing Windows fixtures supply real-shaped name/info/as_dict data.

`strata_cli/launch/screens.py` counts a storage path as the shortened row it actually renders. A long path alone no longer forces compact mode. Compact mode also shortens the path in the middle within the remaining width, preserving its beginning and final directory. Pilot checks cover ordinary and compact 80x24 layouts, a 200-character destination, the ellipsis, and all three visible actions. Existing smaller-window and keyboard-action checks remain.

`strata_control/app_policy.py` again generates the three retained DeepSeek vector configurations and the DeepSeek name/family negative drafting cases. The already committed generated files remain byte-for-byte unchanged. Python checks and five real Node policy tests confirm the generator and Host agree on the existing coverage.

The startup AST fixture now provides threading.RLock. Closed-app worker tests construct and close a real Textual app, invoke the worker on a real thread for success, cancellation, LaunchError and unexpected error, and inspect each RunOutcome without updating the removed screen or leaking a coroutine. A live-app case checks delivery through its event loop. Production exception handling was not broadened. Feed carry coverage now parses the restored dsv41 member while retaining its URL, entry-schema and refusal checks. The KV fixture follows make_kv_layout and alloc_kv to the default FP32 layout and shared FP16 copies; 47104 bytes/token and the default context cap remain unchanged. Speed-switch fixtures retain the restored DeepSeek default and all six legacy entries, while preserving Qwen applicability and environment reset assertions.

Real-socket wait tests allow the displayed integer to be one or two seconds while retaining the 2.5-second completion bound, progress sentence, host/reason, connection counts and retry-memory behavior. A deterministic NetFailure fixture checks rounding immediately below and at 1.5 seconds. Production network limits and rounding are unchanged.

## Workflow bounds and runtime pins

Only Windows Config snapshot uses --timeout-scale 2, giving its selected snapshot child 1800 seconds, and a 45-minute job. Linux retains the original command, its 900-second child limit and a 30-minute job. This is a planned hosted budget correction, not a claim of a new Windows snapshot pass.

The six application setup-node entries in check.yml, snapshot.yml and terminal-minimal-server.yml now pin 24.22.3, matching apps/host/package.json's declared minimum and the private runtime. The preceding hosted logs used 24.19.0 despite the broad 24 selection. Local verification actually used v24.22.3. A new remote run with the pin remains pending. The standalone scripts-test.yml Node 22 compatibility job is unchanged. The cost is a possible CI cache miss/download and maintaining the explicit patch pin; no application default or resource guard changes.

## Commands and results

All local commands used uv offline with the existing Python 3.12 interpreter and `_docs/upstream-sync-20261116/check.py`. The guard retained 3 GiB available RAM, 4 GiB commit headroom, 10 GiB H free and 5 GiB C free, compiler jobs five and test concurrency one. No dependencies were installed. Every owned receipt finished with termination confirmed and zero active Job processes.

The first focused red selection reproduced eight failures in 57.68 seconds. The initial correction selection recorded 129 passed, six skipped and one failure in 103.37 seconds: the filtered local PATH selected the Windows WSL bash launcher, whose /bin/bash was absent. The source fixture was retained. Later runs explicitly prepended the installed Git bin and usr/bin directories inside the guarded test child, allowing its real shell syntax/context checks to run.

The adjacent selection recorded 40 passed, one skipped and one failure in 109.32 seconds. Its new compact fixture initially selected a tiny 80x22 layout, where the existing design intentionally omits the third summary row. The final regression uses an actual 80x24 compact layout induced by configuration warnings and assigns the chosen storage path after wizard navigation. The unchanged original storage test and the compact test then passed with all SaveChoices cases: six passed in 34.56 seconds. These failed intermediate records are preserved.

The final full selection of the thirteen affected test modules passed **139 tests, zero failed and zero skipped in 109.17 seconds**. The exact command was:

```powershell
uv run --no-project --offline --python C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python _docs/upstream-sync-20261116/check.py --seconds 600 python-ci-fix-final-v4 python -c "import os,sys,pytest; os.environ['PATH']='C:/Program Files/Git/bin;C:/Program Files/Git/usr/bin;'+os.environ['PATH']; sys.exit(pytest.main(sys.argv[1:]))" tests/qwen36/test_startup_log_analysis.py tests/terminal/test_app_policy.py tests/terminal/test_bundle_tools_paths.py tests/terminal/test_docker_files.py tests/terminal/test_feed_carry_dsv41.py tests/terminal/test_launch_worker_closed_app.py tests/terminal/test_make_model_release.py tests/terminal/test_orphan_scan.py tests/terminal/test_procs.py tests/terminal/test_own_path_fit.py tests/terminal/test_speed_switches.py tests/terminal/test_tuiqa_wizard.py tests/terminal/test_net_launch_waits.py -q --tb=short -p no:cacheprovider --junitxml=H:/Strata/evidence/upstream-sync-20261116/python-ci-fix-final-v4.xml
```

The final static command, stored verbatim in receipt python-ci-fix-static-v2, parsed all sixteen changed Python files, checked the six Node pins and platform bounds, and ran:

```text
C:/Users/downl/go/bin/actionlint.exe -shellcheck= -pyflakes= .github/workflows/check.yml .github/workflows/snapshot.yml .github/workflows/terminal-minimal-server.yml
python -m flake8 --isolated --select=E9,F63,F7,F82 <the sixteen changed Python files>
node --version
node --test --test-concurrency=1 tests/policy-generated.test.mjs
```

All passed; Node printed v24.22.3 and its five tests passed. The Node command ran from apps/host. Actionlint's optional external shellcheck/pyflakes integrations were disabled; the Python fatal selectors ran separately. Black and a whole-repository style pass were not run, preserving the existing project formatting. The first static helper's raw-byte comparison of an untouched checkout file to a Git blob failed on pre-existing CRLF/LF checkout conversion; the corrected helper compares normalized text. No source changed for that correction. The whitespace check passed with Git's cr-at-eol setting, retaining the two files' existing CRLF format.

## Frozen evidence

The private review bundle python-ci-fix-v1 contains a baseline-scoped patch, exact physical source copies and SHA-256 values in source.json, every intermediate receipt/log, final JUnit files, and the originating hosted artifacts. No credentials or environment dumps are included. Source hashes below identify the production changes; workflow and test hashes are in the complete manifest.

| Source | SHA-256 |
| --- | --- |
| packaging/package_terminal.py | 114e850e39c62aac1303d80c84920d5bc17291f3f92948e9e2aaad616a249461 |
| packaging/make_model_release.py | 68bab7e76b2f5cff558215f930585db8e8466e54d3a67e08fe75b1a62bc414fc |
| strata_control/procs.py | 2281abc51c212d2caa013683b5234f9114c9415facf0c0003bd3c0192d497049 |
| strata_control/app_policy.py | 221ab35aa4785d46ddb933aaee8d7d824b69a11a9ff766fc6cab3d5508f7e74f |
| strata_cli/launch/screens.py | 6b71e4505a2704c0ce481bfd6494c79c94abc6b3864c4a6725d005dfa53de30d |

The final command log SHA-256 is `8a88f872233e4295695a741e06ddd08e665d9b66b0c66fe53edbef569606d590`; its guard receipt is `06c85140c85a79e98003e5f4e6be8127c20b46c127eb5dc2fc0fd7686d647f5b`; its JUnit is `cb1a9e007931331a8e82abe3b237ff93fa6902c3bcf0b23a1b99905490efe200`. The final static command log is `49bb8b4adbe0cb11be36f6fd131bbd2e0c9803e74c3bf473147a43ba62e3c16a`.

## Limits and next gate

Fresh hosted Linux/macOS/Windows checks and the enlarged Windows Config snapshot are NOT_RUN for this correction. The full 4300-test suite, native compilers, GPU, models, downloads, installers and release actions were not run. The independent reviewer must assess the scoped patch and evidence before the parent commits and pushes. That source commit can be reverted as a unit; no model, key, installed profile or release asset was changed. This record does not establish release qualification.

Applicable controls were the canonical CLAUDE-FABLE-5 context, home/Codex/project AGENTS, the common engineering, implementation-start, skill-selection, security, application, frontend and Python SOPs. Skills used were milspec-codex-standard, debug-failing-test, gh-fix-ci for the preceding evidence inspection, and frontend-ui-engineering for the existing Textual layout. Formal MILSPEC certification is not claimed.
