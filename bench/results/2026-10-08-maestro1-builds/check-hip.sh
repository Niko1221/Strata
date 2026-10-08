#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/workspace/strata-validation"
validation_root="$PWD"
rocm_root="$validation_root/toolchains/rocm/lib/python3.10/site-packages/_rocm_sdk_devel"
rocm_libraries="$validation_root/toolchains/rocm/lib/python3.10/site-packages/_rocm_sdk_libraries_gfx110X_dgpu"
export LD_LIBRARY_PATH="$rocm_root/lib:$rocm_libraries/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$validation_root/tmp"
export PATH="$validation_root/toolchains/build-tools/bin:$PATH"
ctest --test-dir build-hip --output-on-failure \
  -R '^(shared_memory_budget_test|q8_oracle_contract_test|q2_portable_reference_test|expert_multi_test)$' \
  > logs/hip-cpu-tests.txt 2>&1
build-hip/strata --help > logs/hip-help.txt 2>&1
ldd build-hip/strata > logs/hip-ldd.txt
sha256sum build-hip/strata > logs/hip-binary-sha256.txt
printf 'HIP CPU checks and help passed\n' > logs/hip-checks-result.txt
