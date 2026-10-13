#!/usr/bin/env fish
# Build the HIP engine/regression tests, then run long prompts plus same-chat follow-ups.

if test (count $argv) -lt 1
    echo "Usage: tools/hip/validate_mixed_prefill.fish <local-server-config.json> [build-dir] [--image <local-image>] [--ple-batch 0|1]" >&2
    exit 2
end

set -l script_dir (dirname (status --current-filename))
set -l repo (realpath "$script_dir/../..")
set -l config (realpath $argv[1])
set -l build_dir "$repo/build-hip"
if test (count $argv) -ge 2
    set build_dir (realpath $argv[2])
end
if not test -f "$config"
    echo "Config not found: $config" >&2
    exit 2
end
if not test -d "$build_dir"
    echo "HIP build directory not found: $build_dir (configure it first; see docs/AMD_HIP.md)" >&2
    exit 2
end

set -l extra_args
if test (count $argv) -ge 3
    set extra_args $argv[3..-1]
end

set -l jobs 2
if set -q STRATA_BUILD_JOBS
    set jobs $STRATA_BUILD_JOBS
end

cmake --build "$build_dir" --target strata hip_multigpu_prefill_dispatch hip_multigpu_prefill_gemm_dispatch -j "$jobs"
if test $status -ne 0
    exit $status
end

ctest --test-dir "$build_dir" --output-on-failure -R '^hip_multigpu_prefill_(dispatch|gemm_dispatch)$'
if test $status -ne 0
    exit $status
end

set -l python python3
if set -q STRATA_PYTHON
    set python $STRATA_PYTHON
else if test -x "$repo/.venv/bin/python"
    set python "$repo/.venv/bin/python"
end
if not test -x "$python"; and not command -q "$python"
    echo "Python not found: $python" >&2
    exit 2
end
"$python" "$repo/tools/hip/validate_mixed_prefill.py" --config "$config" --engine "$build_dir/strata" $extra_args
exit $status
