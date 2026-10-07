# GLM-5.3-Flash on Linux with NVIDIA

Strata can also run GLM-5.3-Flash using the GLM engine from
[Project Maya](https://github.com/mw00/project-maya). Qwen remains the default model family.
This integration targets Linux with one or two NVIDIA GPUs. Windows, WSL2, AMD and Intel are not supported
for GLM. The port has not been tested with the full model on Linux/NVIDIA hardware in this checkout.

## What you need

- Linux on an x86-64 CPU with AVX2, Python 3.10 or newer, a C++ compiler and a CUDA toolkit.
- An NVIDIA GPU with compute capability 7.0 or newer. Volta/V100 needs CUDA 12.x; newer GPUs can use CUDA 13.
- A nominal 32 GB of RAM or more and a fast NVMe SSD. Setup accepts at least 30 GiB of usable RAM because
  hardware reservations reduce what Linux reports. Experts that do not fit in VRAM or RAM are read from the SSD.
- Allow about 108 GB free for a fresh installation with images (96.5 GB of weights, 1.14 GB of vision files,
  and 10 GB reserved for the pack/build), plus at least 5 GB free in the source checkout.

These requirements follow the [Maya reference](https://github.com/mw00/project-maya/blob/444030a3f2afc01515d4d067d663377329efb5b5/README.md).
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

`--yes` alone does not authorize a model download; use `--download-model` or answer the download question.
`--no-vision` skips the image encoder. `--check` checks the GLM requirements without installing or starting it.
The GGUF directory must be writable because the pack is written beside the shards. Keep the shards after packing:
the engine reads experts and the MTP weights from them.

GLM uses a source build with `STRATA_ENABLE_GLM=ON` and `STRATA_NATIVE_EXPERTS=ON`.
A normal Qwen ready-made engine does not contain this optional path.
The installer writes `strata-glm-maya-s-v2-iq2_xxs.json` and `run-glm-maya-s-v2-iq2_xxs.sh`.
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

## Build and validation

The CUDA parity tests cover the GLM router, hyper-connections, KDA, DSA and feed-forward kernels. The Python tests
cover pack metadata, tokenizer, installer configuration and API adaptations without downloading weights.
Run the offline GLM checks with:

```sh
.venv/bin/python -m unittest tools.test_setup_glm tools.test_glm_pack tools.test_glm_oracle serve.test_glm
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
The runner tests generate their small GGUF and oracle fixtures in the build directory. They do not fetch weights.
Live text/image inference and one/two-GPU MTP still need validation with the full model on a Linux/NVIDIA machine.
No speed or quality results from Project Maya are presented as measurements of this port.

Local validation on Windows on 2026-10-07: 327 server/API tests passed with mock engines; 152 additional
installer/pack/tokenizer/oracle/config/API tests completed with seven skips. Three C++ cache/native-helper tests
passed. The CPU image encoder, including its GLM projector, built with MSVC 19.51; JavaScript syntax checks passed.
CUDA compilation and device parity tests were not run because this environment has no supported Linux/CUDA
toolchain. No full model was downloaded or started.

`STRATA_GLM_SLOW=1` selects the diagnostic CPU-expert path. Its older GPU expert pool has not been ported;
use the default fast path for the GPU/RAM/SSD tiers.

## Source and licenses

The port uses Strata `main` at `d5ea7133741e67743c0e886bb426c0ce8d69cf6c` and Project Maya at
[`444030a3f2afc01515d4d067d663377329efb5b5`](https://github.com/mw00/project-maya/commit/444030a3f2afc01515d4d067d663377329efb5b5).
Project Maya and Strata use the MIT license; the copyright notice remains in [LICENSE](../LICENSE).
ggml/llama.cpp retains its [MIT notice](../third_party/ggml/LICENSE), and the dashboard font retains its
[OFL notice](../serve/web/fonts/OFL.txt). GLM weights and the Maya quant retain the model publisher's license.
