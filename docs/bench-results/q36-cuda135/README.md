# CUDA 13.3 and 13.5 build comparison

This is a proposed build baseline for Qwen3.6, generic Qwen3.8 and the separate
DeepSeek V4.1 native executable. It does not qualify model startup, speed, quality
or 98304/131072-token generation. See `docs/HAKUA_RELEASE.md` for release evidence.

Freeze one committed source HEAD and the ggml pin in `workload.json`. Use two new
private build directories, the same MSVC compiler and Release flags, architecture
120 and one compile job. Reproduce the existing release CMake cache settings.
Enable `STRATA_BUILD_DSV41=ON` in both arms: the release reference has it OFF.
This also sets ggml BMI2 OFF. Record these deliberate differences from the release
receipt. Keep all other effective settings equal and save both generated caches.
The old release binary is not a same-source comparison arm.

Run the passive collector with an explicit toolkit and private output path:

```powershell
py -3.12 tools/qwen35/q36_collect_environment.py --build-dir <private-build> --nvcc <absolute-nvcc.exe> --dependency <pinned-ggml-checkout> --binary <built-executable> --output <new-private-manifest.json>
```

`collect_environment()` returns schema version 1. Each observation has a value,
status and reason; unknown values stay null. The default reads only selected OS,
CIM, Git and build-file fields and runs `nvcc --version`. It never imports a CUDA
library, calls `nvidia-smi`, runs the engine or initialises a device. CUDA API
versions, GPU UUID, PCIe link and device capabilities remain NOT_RUN until a
separately authorised probe. Windows display driver fields are passive CIM facts,
not `cudaDriverGetVersion` or a CUDA compatibility decision. Do not substitute a
GPU model's advertised specification for an unmeasured value.

With an explicitly selected already-running process, `--pid` hashes the module
files in a passive Windows process-module snapshot. It never launches that process.
An on-disk DLL, an import table and an actually loaded DLL are separate evidence.
Save generated project/link inputs and PE imports too: CMake's static CUDA runtime
default can coexist with an explicit `CUDA::cudart` import library. Absence of a
cudart DLL is not proof of runtime version. Keep compile flags, linker inputs and
the compiler metadata with binary hashes. Do not run `--version` on an engine
until its initialisation behaviour and GPU ownership have been checked.

Build and execute only `q36_expert_store_test` and `q36_transfer_lifetime_test` for
the initial CPU checks. Hash the exact test executable before execution and save
its exit status. These tests do not use CUDA. Do not run the whole CTest suite,
which includes real-GPU tests. Python collector regressions run with:

```powershell
py -3.12 -m unittest discover -s tests/qwen35 -p test_collect_environment.py -v
```

Keep at least 3 GiB available physical RAM and 4 GiB commit headroom. Monitor during
compilation and stop only the owned compile tree if a gate is crossed. Use a
process-local toolkit path; do not change machine PATH, drivers or installed apps.
Do not change source while either arm compiles. Review compiler spill differences
against source/compiler-bound baselines; do not remove a resource guard to pass.

Keep results private, preferably outside the checkout under a dedicated build root.
The proposed `results/<run-id>/` convention also requires a private, ignored output
location; do not add evidence with `git add -A`. Manifests contain local paths,
device identity and build details and must not be committed. The collector hashes
tracked diffs rather than recording their contents and does not read untracked
files, arbitrary environment variables, model metadata or credentials.

A later runtime comparison must freeze the same driver, selected model/pack hashes,
layer CRCs, expert quantisation, context, MTP and single session/GPU. Freeze the
selected family's actual router count from model metadata and its effective
precision/layout from that metadata and the existing engine settings. The
`runtime.families` contracts apply independently; unknown fields and reasons block
runtime qualification until the actual values are recorded.

Qwen3.6 retains top-8 routing and FP32 attention K/V for comparison and rollback;
confirm the selected model metadata agrees. Its FP16, Q8_0 and NVFP4 candidates
are family-specific proposals. Each opt-in format needs engine capability,
rounding, scale and layout checks before use.

Generic Qwen3.8 retains its selected model's router count and existing effective
FP16 QSA, grouped-64 INT8, rotated Q4_0 or K8V4 contract as applicable. Do not assign
the Qwen3.6 FP32 baseline or proposed formats to this family. Its grouped-64 INT8
is distinct from GGUF Q8_0. DeepSeek V4.1 retains its own metadata-derived router
count, FP8 window ring, compressed E2M1/E4M3 KV, E2M1/UE8M0 index keys and BF16
working rows. Its compressed latent rows and index keys have separate precision,
scale and layout contracts. No additional formats are selected for these two
families; proposals require their own capability/layout checks. See
`docs/NATIVE_CUDA_PRECISION_SCOPE.md` for the existing family contracts.

If a driver update becomes necessary, measure the three distinct arms in
`driver_change_arms` rather than attributing the driver's difference to the
compiler. Preserve the previous binary, config and DLLs so returning to the old
launch path remains possible.
