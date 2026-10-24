# Publication validation — 2026-10-22

Parent: `5c25ecdc4760fd3650d3d76b437535ce28084527`. CUDA 13.4, RTX 4070 Ti SUPER SM89, Linux. Validation applies to the six source/documentation files committed immediately before this report. All execution changes remain opt-in. The default convolution uses the original epilogue-free template; default top-k and selector dispatch remain the original paths; unset verifier split override retains `(T + 1) / 2` at all three boundaries.

Passed again before publication:

```
cmake --build build-dflash2 --target strata dflash2_parity dflash_load_test dflash_policy_test -j4
build-dflash2/dflash2_parity
STRATA_DFLASH2_REGISTER_TOPK=1 STRATA_DFLASH2_LATTICE_SELECTOR=1 build-dflash2/dflash2_parity
build-dflash2/dflash_load_test
build-dflash2/dflash_policy_test
git diff --check
```

Both parity invocations printed PASS. Loader and policy printed `ok`. Fused convolution epilogues are exercised directly by the parity executable, independently of the runtime environment flag. Expanded selector checks include exact IDs/confidences, invalid values and workspace fallback. Build emitted existing unused-variable warnings in verify.cpp and generate.cpp.

Failed: no failures in these publication checks. Earlier excluded/negative performance experiments are documented in the length report and retained evidence; the requested +32.23% long-prompt advantage was not achieved.

Not run: HIP, SYCL, Metal, Windows, other GPUs/models, concurrent serving or a new full-model performance comparison after this rebuild. The parallel throughput study used the immutable tested executable recorded in its command JSONs, with the same source patch retained in the length report. No Q/K/V or gate/up stream prototype is included; only the candidate-transition lattice is newly parallelized. Existing SIDE_MIXER and DF_BRANCH switches were measured without changing their implementations or defaults. No general end-to-end speedup is claimed.

Absolute local workspace prefixes in the published evidence have been replaced by `<WORKSPACE>`; hashes and numeric results are unchanged. Local runner working directories remain described in the reports. No weights or binaries are included.
