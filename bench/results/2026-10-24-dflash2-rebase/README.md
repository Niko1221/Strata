# DFlash2 rebase and gfx906 stream guards, 2026-10-24

Code tested: `ddbd3b945f600a72c8d84bc575c2081ac99f450b`, rebased onto upstream main
`96ca4383`. The following evidence commit adds only this report and logs.
Linux, RTX 4070 Ti SUPER, CUDA 13.4.92, GCC 14.2, SM89.
Workspace paths in logs are replaced with `<workspace>`.

The original 65 commits were replayed. Upstream now has a separate pack-based
`--draft dflash2` implementation. Its kernel source and FP32 prefill callback
are retained. The opt-in GGUF extension uses `dflash2_extension.cu`, a distinct
parity target, BF16 callback and boundary-layer list. The installer retains
upstream's MTP fetch helper, `--mtp off` download handling, library path refresh
and positional `mtp` argument in low-RAM helpers.

Both guards in `src/core/dflash2.cpp` now include `STRATA_HIP_GFX906`.
This selects HIP stream creation and the HIP single-stream default for gfx906.
The existing single-card/CLI restrictions on the pack-based path are unchanged.
The supplied gfx906 build report was not reproduced on this NVIDIA machine.

## Passed

Commands ran from the repository root using existing configured build directories:

```sh
cmake -S . -B build-dflash2 -DSTRATA_ENABLE_DFLASH2=ON
cmake --build build-dflash2 --target strata dflash_load_test dflash_policy_test dflash2_extension_parity dflash_taps_test dflash2_parity -j4
ctest --test-dir build-dflash2 -R '^(dflash_load_test|dflash_policy_test|dflash2_extension_parity|dflash_taps_test)$' --output-on-failure
build-dflash2/dflash2_extension_parity --bench
cmake -S . -B build -DSTRATA_ENABLE_DFLASH2=OFF
cmake --build build --target strata dflash_load_test dflash_policy_test dflash2_parity -j4
ctest --test-dir build -R '^(dflash_load_test|dflash_policy_test)$' --output-on-failure
.venv/bin/python -m unittest discover -s tools -p 'test_*dflash*.py'
.venv/bin/python tools/test_setup_choices.py
.venv/bin/python tools/test_setup_amd.py
.venv/bin/python tools/test_setup_golden.py
.venv/bin/python tools/test_calibrate_dflash2.py
```

CUDA ON: 4/4 CTests passed. CUDA OFF: 2/2 passed. Both engine builds passed;
the final incremental build logs are included. The extension benchmark/parity
command passed. The upstream `dflash2_parity` target compiled in both builds;
its model-based execution was not run.
Python: 57 DFlash tests ran (one skipped), 54 setup-choice tests, 52 AMD setup
tests, 5 golden tests and 15 calibration tests. All completed successfully.
The AMD setup tests do not compile or execute HIP kernels.

## Failed checks and warnings

No build or test failure in the commands above. Compiler and Python resource
warnings are retained in the logs. `git diff --check origin/main` reports existing
trailing whitespace in historical benchmark logs and rejected experiment patches;
those raw records were left intact.

The GitHub integration returned 403 when updating the PR description and when
converting the PR to Draft. Those metadata changes were not applied.

## Not run and limits

HIP/gfx906 and SYCL compilation/execution: their compilers are unavailable.
Metal, Windows, model generation, full default-path token equality, full test suite
and performance comparisons were not run. Historical performance tables are not
measurements of the rebased revision. This patch does not establish runtime
correctness on AMD or unlock gfx906 serving. Missing backend builds remain required
before review under CONTRIBUTING.md; the PR should remain a draft until validated.
