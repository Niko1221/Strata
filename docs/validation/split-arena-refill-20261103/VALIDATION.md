# Submission checks, 2026-11-03

Base: `99866a47a2b8a21b131aad162a584ee9da92f8f0`. Engine changes are limited to the compiler-gated additions
in `src/program/generate.cpp`. The benchmarked source is preserved in the archived patch; the submitted guards
also exclude gfx906 and Metal. No new model requests or production restarts were performed for this submission.

## Passed

- `python3 docs/validation/split-arena-refill-20261103/recheck.py`: archive checksums and both original campaign
  accounting analyses pass. All six long-context comparisons regenerate the reported values.
- `git diff --check`: clean.
- The Debian 12 / GCC 12 / CUDA 13.1 builder ran `pr-validation/check-cuda-source.py` from the evidence archive.
  It reused the recorded CUDA translation-unit compiler flags, preprocessed both sources with `-E -P`, and
  compiled the submitted `generate.cpp` to an object with `-c`. The original include tree is identical to this
  base revision. This is a fresh host-side CUDA translation-unit compilation, **not** a full new CUDA link or
  kernel execution.
  - Macro disabled, baseline and submitted preprocessed SHA-256:
    `2f90c833b5ac0c2ca8d4f3cd80b158240de89019ccff60b2a2d7f97630464e3c`.
  - Macro enabled on NVIDIA CUDA, measured and submitted preprocessed SHA-256:
    `80c3b8881e875f71f0e5bcd39f20f996d369ad98cccf5fbb3d12edb28d8378d1`.
  - Three compiler probes confirm the actual submitted guard excludes `STRATA_USE_HIP`,
    `STRATA_HIP_GFX906`, and `STRATA_GPU_METAL`. These are guard checks, not backend builds.
- CPU-native configure/build completed with GCC 15.2.0, CMake, CUDA/HIP/SYCL disabled,
  `STRATA_NATIVE_EXPERTS=ON`, `STRATA_BUILD_TESTS=ON`, `STRATA_BUILD_CONVERSATION_TESTS=ON`, and the recorded
  llama.cpp checkout. Command shape:

  ```sh
  cmake -S . -B build-cpu-native \
    -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR="$GGML_ROOT" \
    -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
  cmake --build build-cpu-native -j6
  ctest --test-dir build-cpu-native --output-on-failure --timeout 60 -j6
  ```

  CTest: **80 passed, 4 skipped, 1 failed** out of 85 registered tests. This is not a green suite; the failure
  and skips are listed below. CPU tests do not exercise the CUDA-only refill path.
- The earlier measured CUDA build compiled/linked `strata` and `strata_kq`; its exact build log and artifact
  fingerprints are archived. The measured binary passed a 256-window diagnostic with zero residency-table
  mismatches, then the bounded short/long workloads documented in README. These are historical checks on the
  measured source, not a new fault-injection or sustained-runtime result on the submission.

## Failed / blocked

- CPU-native `isa_guard_check` failed: `iq512_reference_gu_rows` and `iq512x_rows` contain wide-ISA instructions
  but are not in the checker's allowed-public-function list. `CMakeLists.txt`, CPU kernels/headers and the ISA
  checker are unchanged from the base. A separate pristine-baseline full build was not rerun; the logs establish
  the failure in this configuration, not a complete regression diagnosis. No unrelated ISA-check changes are
  included in this PR.
- An earlier CPU configuration with `STRATA_NATIVE_EXPERTS=OFF` configured but failed linking several pool/expert
  tests on unresolved AVX1 functions (`act_quant_q8_1_avx1`, `q2_0_gguf_rows_multi_avx1`, and its legacy variant).
  Its CTest invocation recorded 52 passed and 7 not run due to missing executables. This unsuccessful configuration
  is retained separately; the later native-enabled build does not erase it.
- CPU-native CTest skipped `expert_multi_test`, `native_hadamard_test`, `tq_cpu_parity`, and `tq_cpu_parity_avx2`.
  These skips are not passes. Refer to the full log for capability/fixture messages.
- Rejected benchmark preflights and the thermally interrupted long campaign are preserved in the archive and
  described in README. They are not included in accepted performance averages.

## Not run / unresolved

- Full new CUDA link and GPU model execution after guard hardening. Preprocessing verifies that the NVIDIA CUDA
  translation unit is identical to the one used in the historical measurements; it does not replace new execution.
- HIP, gfx906, SYCL and Metal builds; Windows runtime; mixed GPUs or more than two cards.
- Numerical parity/KL, semantic-quality evaluation, repeated A/B against production `NOWAIT=1,LAG=2`,
  sustained soak, cancellation/EOF during refill, and injected CUDA failures.
- A fresh independent static review found no Critical defect. It identified the backend-gate hole (fixed by
  exclusions above), and an **unconfirmed teardown concern**: direct error returns/EOF can bypass the explicit
  all-card adaptive drain. Possible implicit CUDA host-unregister/staging synchronization was not established
  as an all-path guarantee. No memory corruption was reproduced. Before merge or deployment, add/verify an
  experiment-only all-owning-stream scope drain and exercise those exits.

Keep the PR **Draft**. The static review does not certify runtime safety or performance. Targeted maintainer
feedback on scope, teardown and the minimum backend/runtime checks is requested; no default change is proposed.
