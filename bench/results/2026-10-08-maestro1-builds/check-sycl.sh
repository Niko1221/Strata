#!/usr/bin/env bash
set -eo pipefail
cd "$HOME/workspace/strata-validation"
validation_root="$PWD"
oneapi_root="$validation_root/toolchains/intel-root/opt/intel/oneapi"
source "$oneapi_root/tbb/2023.1/env/vars.sh"
source "$oneapi_root/umf/1.1/env/vars.sh"
source "$oneapi_root/tcm/1.5/env/vars.sh"
source "$oneapi_root/compiler/2026.1/env/vars.sh"
source "$oneapi_root/mkl/2026.1/env/vars.sh"
source "$oneapi_root/dpl/2022.13/env/vars.sh"
set -u
export TMPDIR="$validation_root/tmp"
ldd build-sycl/strata > logs/sycl-ldd.txt
build-sycl/strata --help > logs/sycl-help.txt 2>&1
sha256sum build-sycl/strata > logs/sycl-binary-sha256.txt
printf 'SYCL help passed\n' > logs/sycl-checks-result.txt
