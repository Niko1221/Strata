# Q38 -> official engine API port (status 2026-11-19)

## Built successfully with CUDA 13.5 ONLY
`strata_q38` and all three dependencies linked under CUDA 13.5.37 / sm_120
(`BUILD_EXIT=0`, `H:/Strata/builds/q38-vs-135h`):

- `strata_q38.lib` 2,965,420 bytes (entry `strata::qwen38::run`)
- `strata_q38_ops.lib` 1,771,454 bytes (CUDA kernels)
- `strata_q38_kv.lib` 825,284 bytes
- `q38_spin8.lib` 226,662 bytes

Evidence: `H:/Strata/evidence/upstream-oneway-20261119-build/verification.json`
and `H:/Strata/builds/q38-elevated-build3.log`.

## Done (code, on main)
- `Qwen38Dense` registered in `model_family.hpp` (arch `qwen35`, dense 64 layers).
- `generate.cpp` family switch calls `strata::qwen38::run` for `Qwen38Dense`.
- `engine/q38/include/strata/qwen38/engine.hpp` entry declaration.
- `engine/q38/src/engine.cu`: `run()` entry + official `ENGINE_REQUEST`
  completion line alongside the existing `DONE`/`TIMING` lines.
- `engine/q38/CMakeLists.txt` + `strata_gpu_engine.cmake` link; C++17 forced
  on nvcc; `tqcodec/src` include added.
- Merge defects fixed: `.gitignore` no longer ignores `engine/`; fork tests
  relocated to `engine/tests/`.

## Build environment notes (for reproducibility)
CUDA 13.5 is a standalone toolkit here; VS has only 13.2/13.3 BuildCustomizations.
Under UAC elevation the 13.5 props/targets/Version.props/xml and
`Nvda.Build.CudaTasks.v13.5.dll` were copied into VS BuildCustomizations,
`CUDA_PATH_V13_5` set, and the copied targets patched for two CMake/MSBuild
incompatibilities (CudaToolkitDir default, MSB4023 host-runtime metadata -> MD).
Toolkit source files were NOT modified. MSBuild was invoked directly with
`/nodeReuse:false`.

## Full strata.exe link — DONE
`strata.exe` (68.4 MB) links with `strata_q38` under CUDA 13.5-only:
`BUILD_EXIT=0`, `H:/Strata/builds/q38-full-final/Release/strata.exe`.
`strata --version` -> 0.1.62; `strata --families` ->
`qwen4exp qwen35moe qwen35` (qwen35 = Q38's GGUF arch, reported after the
`--families` fix on main `05c22664b`). Evidence:
`H:/Strata/evidence/upstream-oneway-20261119-build/full-strata-verification.json`.

## Resident --serve E2E — DONE
Real Qwen3.8-27B GGUF (11.73 GB) on RTX 5060 Ti, resident `--serve`:
`READY 1024 stop` with `engine=qwen38_dense`, `PP`/`T` streaming, 8 tokens
generated, and the official `ENGINE_REQUEST` completion line
(`prompt 4 tokens = 0 reused + 4 read in 211.200 ms ..., 8 generated in
240.358 ms ...`), clean `QUIT` with exit 0. Evidence:
`H:/Strata/evidence/upstream-oneway-20261119-build/serve_e2e_verified.json`.
This is a protocol E2E, not a quality/parity measurement.

## Accepted constraints (will NOT be done — no key / not required)

- **Signing: not applicable.** The user confirmed the fork's ed25519 private
  key does not exist at all (not just "unavailable this session"). Only public
  keys are in the repo (`strata_control/update_keys.json`). Therefore
  `engine-component.json` stays `signature: null` and Windows Authenticode is
  absent. No key was generated or faked. This is a final, accepted constraint —
  not a pending task. `STRATA_REQUIRE_SIGNED_COMPONENTS=1` will refuse this
  component; that is the expected state.
- **`family_layout.hpp` qwen35-dense table: intentionally NOT added.** Q38 uses
  its own reader (`admission.hpp`) and is dispatched in `generate.cpp`'s family
  switch *before* the shared geometry/layout path runs, so inference never
  touches this table. Adding a dense branch would require weakening
  `family_layout.hpp:84` (`n_expert <= 0`), which validates every official MoE
  model — risk > benefit.

## Known issue (not fixed in 0.6.6)
- Normal-profile picker: `/api/native` returns `models:4` but all `ready:false`,
  so the menu reports "no local models downloaded" (v0.6.4 had
  `q38-dense-20acacc6c8778c27` ready). Cause: staged Control 0.6.3 ready-detection
  vs 0.6.6 native model registry. Fixing needs a signed Control rebuild — which
  is impossible without the key, so it stays as a documented regression.

Performance parity vs Prism llama.cpp was not achieved and was accepted by the
user under policy A (`releaseQualified=false`); no performance claim is made.
