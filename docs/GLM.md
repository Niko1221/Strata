# GLM-5.3-Flash with NVIDIA

Strata can also run GLM-5.3-Flash using the GLM engine from
[Project Maya](https://github.com/mw00/project-maya). Qwen remains the default model family.
This integration targets native Linux with one or two NVIDIA GPUs; native Windows support is experimental.
WSL2, AMD and Intel are not supported for GLM. Full-model inference has not been tested on either platform in
this checkout. Windows CUDA compilation has not been verified here either.

## What you need

- Linux or 64-bit Windows 10/11 on an x86-64 CPU with AVX2, Python 3.10 or newer, a C++ compiler and a CUDA toolkit.
- On Linux, install the distribution's C++ build tools. On Windows, CUDA 12.8 with Visual Studio 2022 Build Tools
  and the "Desktop development with C++" workload is the recommended starting point. Setup checks that the
  installed Visual Studio version is compatible with the selected toolkit; Visual Studio 2026 cannot replace
  2022 for CUDA 12.8.
- An NVIDIA GPU with compute capability 7.0 or newer. Volta/V100 needs CUDA 12.x; newer GPUs can use CUDA 13.
- A nominal 32 GB of RAM or more and a fast NVMe SSD. Setup accepts at least 30 GiB of usable RAM because
  hardware reservations reduce what Linux reports. Experts that do not fit in VRAM or RAM are read from the SSD.
- Allow about 108 GB free for a fresh installation with images (96.5 GB of weights, 1.14 GB of vision files,
  and 10 GB reserved for the pack/build), plus at least 5 GB free in the source checkout.

These requirements follow the [Maya v1.2.0 reference](https://github.com/mw00/project-maya/blob/5932f601373f53fc021f75dc55159a722c772571/README.md).
They are not a performance measurement of this Strata integration.

## Install

Select the `glm` family in the existing Strata installer. The model is
[Maya-S v2 IQ2_XXS](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF), about 96.5 GB in three GGUF shards.
The installer checks each shard against its pinned SHA-256 before packing it.

To install without starting the server:

```sh
./setup.sh --family glm --download-model --no-start --yes
```

To use files already on your NVMe SSD:

```sh
./setup.sh --family glm --gguf-dir /path/to/gguf --no-start --yes
```

On Windows, use the existing Strata launcher with the same options:

```bat
START-HERE.bat --family glm --check
START-HERE.bat --family glm --gguf-dir D:\Maya-data --no-start --yes
```

To authorize a fresh download, use `START-HERE.bat --family glm --download-model --no-start --yes`.
Windows support is experimental. Set the Windows page file to "System managed" (Advanced system settings >
Performance > Advanced > Virtual memory). Windows charges GPU allocations and pinned RAM against its commit
limit too; the engine uses the smaller of available physical RAM and available commit when sizing the RAM tier.

`--yes` alone does not authorize a model download; use `--download-model` or answer the download question.
`--no-vision` skips the image encoder. `--check` checks the GLM requirements without installing or starting it.
The GGUF directory must be writable because the pack is written beside the shards. Keep the shards after packing:
the engine reads experts and the MTP weights from them.

GLM uses a source build with `STRATA_ENABLE_GLM=ON` and `STRATA_NATIVE_EXPERTS=ON`.
A normal Qwen ready-made engine does not contain this optional path.
GLM on Windows defaults to the shared CUDA runtime. An explicit
`-DCMAKE_CUDA_RUNTIME_LIBRARY=Static` uses the static runtime consistently instead.
The installer writes `strata-glm-maya-s-v2-iq2_xxs.json` and `run-glm-maya-s-v2-iq2_xxs.sh` (`.bat` on Windows).
Run that script when you want to start the model.

## Chat, APIs and images

GLM uses the same Strata web app and OpenAI- and Anthropic-compatible APIs. The loaded model is reported by
`/v1/models`; use the same base URL and API key as other Strata models. Reasoning effort supports off, low,
medium and high. Requests are processed one at a time on the GLM path.

The engine keeps expert tiers in VRAM, RAM and the model files. With two GPUs it can split layers and use its
MTP block for speculative decoding. Images use the GLM encoder and borrow GPU memory from the engine while
encoding; the borrowed cache memory is returned afterwards. The Monitor shows the GLM expert tier statistics.

Keep the default host `127.0.0.1`. Listening on another address requires a nonempty `--api-key`, including when
starting from a saved configuration.

## Prompt memory and read-ahead

The v1.2.0 prompt path sizes its GPU scratch budget at about 6% of VRAM, bounded to 1-2 GiB. It starts at an
8,192-token chunk and reduces the chunk until its scratch fits. Its pinned SSD landing buffer uses about 2% of
currently available RAM per GPU, bounded to 12-64 expert slots by default. If pinning fails, it halves the
requested slots down to 12; if even that fails, it falls back to processing tokens individually.

From 1,024-token chunks, it reads the next MoE layer's disk-only experts while the current layer computes.
These controls go in the configuration's `env` object:

- `STRATA_GLM_PREFILL_MB`: GPU scratch budget in MiB; overrides the automatic budget.
- `STRATA_GLM_PREFILL_CHUNK`: requested chunk size, still reduced to fit the budget.
- `STRATA_GLM_PREFILL_LAND`: requested landing-buffer slots, at least 12; allocation failures still reduce it.
- `STRATA_GLM_PREFILL_PRED_T`: chunk size from which to read ahead; `0` disables read-ahead.

The engine prints the chosen chunk, scratch and pinned-buffer sizes. Larger buffers use memory that could
otherwise cache experts. No speed improvement has been measured for this Strata port.

## Build and validation

The CUDA parity tests cover the GLM router, hyper-connections, KDA, DSA and feed-forward kernels. The Python tests
cover pack metadata, tokenizer, installer configuration and API adaptations without downloading weights.
Run the offline GLM checks with:

```sh
.venv/bin/python -m unittest tools.test_setup_glm tools.test_glm_pack tools.test_glm_oracle serve.test_glm
.venv/bin/python tools/test_glm_prefill.py
```

On a Linux/NVIDIA development machine with the build dependencies installed:

```sh
cmake -S . -B build-glm-tests -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_GLM=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR=third_party/llama.cpp -DCMAKE_CUDA_ARCHITECTURES=native \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-glm-tests -j2
ctest --test-dir build-glm-tests --output-on-failure -R 'glm_|router_sigmoid|native_split'
```

For a V100, use CUDA 12 and add `-DSTRATA_EXPERIMENTAL_SM60=ON` to the configure command.
The runner tests generate their small GGUF and oracle fixtures locally. They do not fetch weights.
`glm_prefill_parity` creates a synthetic Q4_0 pack and compares five prompt chunks and their continuation with
token-at-a-time logits at a maximum absolute error below `2e-3`. It checks disk reads with 12/64 landing slots
and next-layer read-ahead off/on. `glm_prefill_memory_test` checks slot sizing, allocation retries and fallback
without a GPU; actual CUDA allocation cleanup still needs device validation.
Live text/image inference and one/two-GPU MTP still need validation with the full model on a Linux/NVIDIA machine.
No speed or quality results from Project Maya are presented as measurements of this port.

Initial-port validation on Windows on 2026-10-07 (before the v1.2.0 update): 327 server/API tests passed with mock engines; 152 additional
installer/pack/tokenizer/oracle/config/API tests completed with seven skips. Three C++ cache/native-helper tests
passed. The CPU image encoder, including its GLM projector, built with MSVC 19.51; JavaScript syntax checks passed.
CUDA compilation and device parity tests were not run because this environment has no supported Linux/CUDA
toolchain. No full model was downloaded or started.

Fresh v1.2.0-update validation on Windows on 2026-10-07: 590 Python test cases completed, with 583 passed and
seven skipped. Four C++ cache/native-helper/prefill-memory tests passed. The synthetic Q4_0 fixture generated and
packed successfully on CPU, and the CPU image encoder built with MSVC 19.51. Python and JavaScript syntax and
Git whitespace checks passed. Native Windows GLM configuration reached CUDA discovery, then failed with
"No CUDA toolset found"; CMake also rejected GLM with Intel/SYCL and 32-bit Windows as expected.
CUDA compilation, Q4_0 device parity, actual CUDA allocation-failure cleanup and
full-model inference remain unverified. No model weights were downloaded and no production engine/server was
started.

`STRATA_GLM_SLOW=1` selects the diagnostic CPU-expert path. Its older GPU expert pool has not been ported;
use the default fast path for the GPU/RAM/SSD tiers.

## Source and licenses

The port uses Strata `main` at `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`. Its initial Maya reference was
`444030a3f2afc01515d4d067d663377329efb5b5`; the prefill and Windows update follows Maya v1.2.0 at
[`5932f601373f53fc021f75dc55159a722c772571`](https://github.com/mw00/project-maya/commit/5932f601373f53fc021f75dc55159a722c772571).
Project Maya and Strata use the MIT license; the copyright notice remains in [LICENSE](../LICENSE).
ggml/llama.cpp retains its [MIT notice](../third_party/ggml/LICENSE), and the dashboard font retains its
[OFL notice](../serve/web/fonts/OFL.txt). GLM weights and the Maya quant retain the model publisher's license.
