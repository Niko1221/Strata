# Q38 -> official engine API port (status 2026-11-19)

Done in the isolated branch `codex/upstream-oneway-20261119`:

- `Qwen38Dense` registered in `engine/include/strata/artifact/model_family.hpp`
  (arch string `qwen35`, dense 64 trunk layers, no experts).
- `engine/src/program/generate.cpp` family switch calls
  `strata::qwen38::run(argc, argv)` when the model resolves to `Qwen38Dense`.
- `engine/q38/include/strata/qwen38/engine.hpp` declares the entry point.
- `engine/q38/src/engine.cu` now defines `strata::qwen38::run` (a thin
  `STRATA_Q38_STANDALONE` main wrapper is kept for the old CLI), and emits
  the official `ENGINE_REQUEST` completion line in addition to the existing
  `DONE`/`TIMING` lines.
- `engine/q38/CMakeLists.txt` builds `strata_q38` with CUDA 13.x and sm_120;
  `engine/CMakeLists.txt` and `engine/cmake/strata_gpu_engine.cmake` wire it
  into the official `strata` target behind `-DSTRATA_ENABLE_Q38=ON`.
- `.gitignore` no longer ignores `engine/` (it was a stale fork build-output
  rule); the full official engine source is now tracked.
- Merge-side fix: fork-added tests under `tests/core|prefill|hip|quant|program|
  cuda|platform|gdn|ngram` were moved to `engine/tests/...` so the official
  CMake can find them; `tools/oneway_upstream_merge.py` carries the same rule.

Verified so far:

- `cmake -S engine -B ... -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_Q38=ON`
  configures successfully (exit 0) with CUDA 13.5 nvcc.
- Full compile of `strata` is in progress; no result is claimed yet.

Remaining before this can ship (NOT done):

1. Finish the `strata` build and fix any compile/link errors from the entry
   point integration.
2. `engine/include/strata/core/family_layout.hpp`: add the qwen35-dense
   geometry + tensor layout table from `include/strata/qwen38/admission.hpp`
   (64 trunk layers + optional MTP, `expert_count == 0`). The Q38 engine uses
   its own reader, so this is for official tooling/validation parity, not
   inference.
3. Real resident `--serve` E2E with the Qwen3.8 GGUF through the official
   Terminal/app (`terminal/serve/server_parts/engines.py` parsing the new
   completion line).
4. Signed engine-component packaging (fill `files` digests in
   `engine-component.json`) and the official Terminal/app test suites.
5. Re-check the 133 retained conflict copies, prioritising host server files.

No runtime, parity or E2E result is claimed by this port.
