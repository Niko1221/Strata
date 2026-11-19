# Q38 -> official engine API port (status)

Done by `port_q38_to_official_api.py`:
- `Qwen38Dense` registered in `model_family.hpp` (arch string `qwen35`).
- opt-in `add_subdirectory(q38)` in `engine/CMakeLists.txt`.
- schema-2 `engine-component.json` + `BUILD.json` template.

Required before this can ship (NOT done, NOT verified):
1. `engine/include/strata/core/family_layout.hpp`: add the qwen35-dense
   geometry keys and the `blk.N.*` tensor layout table from the fork's
   `include/strata/qwen38/admission.hpp` (64 trunk layers + optional MTP,
   expert_count == 0).
2. `engine/q38/q38_serve.cpp`: adapt the fork's existing `READY`/`PP n total`/
   `T id` stdin/stdout protocol to the official resident `--serve` protocol
   expected by `terminal/serve/server_parts/engines.py` (the `prompt N
   tokens = ...` progress line and the `Engine.generate(...)` contract).
3. Switch the qwen38 entry point at the top of `engine/src/main.cpp`.
4. Build with `cmake -S engine -DSTRATA_ENABLE_Q38=ON -DSTRATA_ENABLE_CUDA=ON`
   and run the official Terminal + app E2E suites.
5. Sign the component with the fork public key and fill the `files` digests
   in `engine-component.json`.

No compile, runtime or parity result is claimed by this port script.
