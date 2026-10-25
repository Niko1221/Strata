# 0.2.0-beta CPU re-test (2026-10-20)

Pinned release: `3d1fb48090e8f2cefc22d7b5845fb7f5b71f33fa`.
PR repair updated onto upstream main `f57c0adcb31c00f3f216550c004dac4e713a9f70`; the two target files in that main equal the release before applying A. The official `_strata_replica_selected` callback is preserved. No native code is changed.

| CPU protocol control | Passed / 18 |
|---|---:|
| Official release | 9 |
| Close only | 16 |
| Full A | 18 |

These use the repository's real StrataEngine reader/pump and a synthetic native producer; two contracts use real OS pipes. No model or GPU ran. The six selected serving modules passed 102/102 in separate interpreters, including the official callback and telemetry contracts. This is not the full repository suite or a performance measurement.

Reproduce from the PR checkout with Strata's Python runtime requirements installed:

```sh
python -m unittest -v serve.test_replica_lifetime
python tools/replay_replica_lifetime.py --output ./replica-v020-controls-new
python -m unittest -v serve.test_replicas
python -m unittest -v serve.test_parallel
python -m unittest -v serve.test_request_accounting
python -m unittest -v serve.test_request_hardening
python -m unittest -v serve.test_queue_keepalive
```

The output directory must not exist. The replayer defaults to the fixed 0.2.0-beta commit; fetch it in shallow checkouts or use `--baseline-file` with its captured `serve/replicas.py`. The local archive checkout used that latter option. `controls.json` records the actual counts and hashes.

The earlier audit's close-only run was 15/18; its sanitized log is retained here. The later pinned-dependency run was 16/18. The extra multi-caller test can also expose shared statistics. The replayer therefore requires both forced statistics failures, permits only that additional statistics failure, rejects all control errors, and still requires every fixed contract (18/18). It does not retry to select a count or relax a fixed contract.

Two setup failures are retained: a partial archive initially omitted the unmodified strata_cli package; then the tokenizer's regex dependency was missing. Supplying the pinned runtime closure/dependencies resolved imports without product/test assertion changes. Final environment is in environment.json. Raw local logs were path-redacted for publication; historical synthetic request/SSE bytes are unchanged.

GPU scope: the existing GPU evidence in the parent directory is historical engine 0.1.44.1 qualification. It must not be described as 0.2.0-beta / engine 0.1.45.3 GPU validation. CUDA/HIP/SYCL/Metal builds, model inference, GPU performance and long-soak testing of this refresh are NOT_RUN. They are not needed for the CPU protocol reproduction; no cross-backend hardware claim is made.
