# Q38 delivery consumer implementation log

The shared final-native producer/consumer now binds current Q38 provenance separately from the unchanged historical two-binary GPU equality. The app release runtime probe uses explicit Python -B so isolated imports do not change staged Control through bytecode writes.

The consumer previously compared all three native-binding executables to the two historical GPU binaries. acceptance.py now shares the explicit current-Q38 identity and reviewed source verifier between producer and consumer. Missing/tampered Q38 metadata and unexpected binaries are rejected. test_acceptance.py covers the complete synthetic roundtrip and eleven negative cases. tests/terminal/test_delivery_q38_current_proof.py covers retained actual Q38 bytes, consistently re-pinned malformed source provenance and the actual bundled-Python probe without bytecode writes. verify-release.mjs changes only the explicit -B flag; no feed-controller or inventory-filter change was needed.

Canonical prompt, AGENTS, common/start-gate/skill-selection/security/application/Python SOPs and MILSPEC/incremental-implementation skills were applied. Self-review identified a concrete missing producer-consumer/no-write test gap and added the narrow regressions. No governance or memory update was made.

The final command was executed with existing Python312 through uv and the established check.py owned Job:

```powershell
uv run --no-project --offline --python C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe python _docs/upstream-sync-20261116/check.py --seconds 300 q38-delivery-consumer-fix-final-r2 python -B -m pytest _docs/stable-060-20261116/test_acceptance.py tests/terminal/test_delivery_q38_current_proof.py -k 'final_native or final-native or actual_three_binary or actual_release_runtime' -q -p no:cacheprovider --junitxml=H:/Strata/evidence/upstream-sync-20261116/q38-delivery-consumer-fix-final-r2.xml
```

Result: 21 passed, 165 deselected, zero skips/failures/errors, 9.49 seconds. Owned Job exit0/native0, drained0, containment and termination confirmed. Minimum RAM11810852864/commit32159113216/H72730013696 bytes and final C17467232256 bytes, no breaches or configured process cap. RAM3GiB/commit4GiB/H10GiB/C5GiB floors are unchanged. Final JUnit SHA-256402bfa4e82ea33e007f3a9aff96cf310e69cafa9b8cffe52814e758ffd8cf6ff; guard receipt7f87aac5acb185068a694e2353e23ef098f401c796348b4189acfaff0a72c1f1.

Guarded Flake8 with --max-line-length140 and the existing Black-compatible E203 exclusion passed all three Python files. Guarded Black --line-length110 --check passed the new terminal test. Formatter-only changes outside the repair were restored. git diff --check passed. Existing uv no-project/dangling dist-info and Git LF/CRLF warnings were observed; no environment cleanup was done. Exact argv/results/output remain in the q38-delivery-consumer-* receipts under H:/Strata/evidence/upstream-sync-20261116/receipts.

Frozen Q38 compiled source8b0b3c74514d26e5ca73496f0b268d66a0cb99a5 and executable426e8caf4813ab4c1a7802c3e5595f9aa899abcc0f11b7cb4917c5fcccf383b6 are unchanged. Old source-bound failures remain preserved. The local actual-byte tests skip when their explicit local inputs are unavailable; no recorded case skipped on this PC. Whole suites and subsequent final-source delivery phases remain NOT_RUN here. Rollback is the scoped inverse commit diff.

No GPU/model/build/install/publication action was performed. Parent owns independent review and final-source phase reruns. No broader qualification or formal MILSPEC certification is claimed.
