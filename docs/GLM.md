# GLM-5.3-Flash with NVIDIA

Strata can also run GLM-5.3-Flash using the GLM engine from
[Project Maya](https://github.com/mw00/project-maya). Qwen remains the default model family.
This integration targets native Linux with NVIDIA GPUs; native Windows support is experimental.
The GLM engine can split layers across up to 16 visible GPUs; multi-GPU inference is unverified here.
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

These minimum requirements are for Maya-S v2. Larger model files need additional SSD space.
The source reference is [Maya v1.0.14](https://github.com/mw00/project-maya/blob/327cfa60d46f9d8eea33a1b38e76284646fbad86/README.md).
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

The current GLM prompt path windows routed expert rows to reduce scratch memory. On one GPU it starts at
32,768-token chunks and borrows up to 40% of free VRAM after a 1 GiB reserve, with a 1-8 GiB budget. With two
GPUs it starts at 8,192 tokens and about 6% of VRAM, bounded to 1-2 GiB. With more GPUs it starts at 512 tokens
for the layer pipeline. Each chunk is reduced until its scratch fits. Its pinned SSD landing buffer uses about
3% of available RAM per GPU, bounded to 12-96 expert slots by default. On Windows the available commit limit
also bounds the RAM estimate. If pinning fails, it halves the requested slots down to 12; if even that fails,
it falls back to processing tokens individually.

From 1,024-token chunks, it reads the next MoE layer's disk-only experts while the current layer computes.
These controls go in the configuration's `env` object:

- `STRATA_GLM_PREFILL_MB`: GPU scratch budget in MiB; overrides the automatic budget.
- `STRATA_GLM_PREFILL_CHUNK`: requested chunk size, still reduced to fit the budget.
- `STRATA_GLM_PREFILL_LAND`: requested landing-buffer slots, at least 12; allocation failures still reduce it.
- `STRATA_GLM_PREFILL_PRED_T`: chunk size from which to read ahead; `0` disables read-ahead.
- `STRATA_GLM_PREFILL_ATTN`: `f32` selects the F32 prompt-attention calculation for comparison; otherwise the
  engine selects tensor cores when the compiled kernel and the GPU's shared memory allow it.
- `STRATA_GLM_PREFILL_ATTN_CHECK`: when present, also runs the F32 calculation and prints the difference from
  tensor-core attention; this adds work and memory use and is intended for debugging.

The engine prints the chosen chunk, scratch and pinned-buffer sizes. Larger buffers use memory that could
otherwise cache experts. No speed improvement has been measured for this Strata port.

Prompt attention uses FP16 operands with F32 accumulation on tensor cores and prefetches the next selected
latent rows. The default fast engine keeps its DSA latent cache in FP16 for both prompts and token decoding,
including MTP. This halves the latent-cache storage, not the whole attention or engine memory allocation.
`STRATA_GLM_PREFILL_ATTN=f32` still reads that FP16 cache; `STRATA_GLM_SLOW=1` keeps the diagnostic F32 cache.
The fast path's reduced precision still needs device parity and full-model quality checks in this checkout.

## Local diagnostic report

Use `START-HERE.bat --family glm --report` on Windows or `./setup.sh --family glm --report` on Linux to write
`strata-glm-report.txt` beside the installer. It collects this PC's hardware, available RAM/commit, recorded build
details, installed GLM settings and recognized speed/memory messages from existing engine logs. It works when
the toolkit, model or logs are missing, and does not download, build or start anything.

The report stays on this PC. It excludes API keys, conversations and raw log tails, and masks personal paths.
Only recognized numeric engine messages and supported settings are retained, so an unrelated error may not
appear in the report. The file is ignored by Git; inspect it before attaching it to an issue.

## Changes reviewed through Maya v1.0.14

The NVIDIA GLM port includes typed dense GEMV dispatch, windowed MoE prefill, prefill CPU/PCIe balancing,
next-layer expert reads, Turing register-accumulator attention, contiguous CPU work pieces and the Windows,
Volta and Turing compile guards. The F32 attention fallback and synthetic Q4_0 test path remain.
Native GLM tool calls use `NAME<arg_key>...</arg_key><arg_value>...</arg_value>` inside `<tool_call>`;
complete and streamed responses keep the existing OpenAI/Anthropic API formats. Qwen remains the default.

Model selection is explicit: `--family glm --model Maya-M` or `--family glm --model GSQ-RCO-3.5bit`.
Their files are about 116.1 GB and 137.1 GB respectively; these are file sizes, not RAM recommendations.
Maya-S v2 remains the GLM installer default. GSQ-RCO-3.5bit has no MTP weights.
Every model and image-file URL uses an immutable Hub revision
and every file is checked by SHA-256. `--yes` does not authorize downloads.

For an installed GLM configuration, `START-HERE.bat --family glm --calibrate` (Linux: `./setup.sh`) measures
CPU-lane threads and PCIe share only when requested. It needs the model and a working CUDA build, starts
an engine for the measurement and saves settings only after an interleaved improvement check. No calibration
was run with a model here. `STRATA_GLM_CPU_SPLIT` controls contiguous CPU work pieces (default 48).
`STRATA_GLM_PREFILL_WINDOW` and `STRATA_GLM_PREFILL_SUB` control expert windows and sub-batches.

Text conversation slots on SSD are opt-in in Strata: set `STRATA_GLM_SLOTS` to a count (maximum 64).
`STRATA_GLM_SLOT_MIN` defaults to 1,024 tokens, `STRATA_GLM_SLOT_GB` to 16 GiB and `STRATA_GLM_SLOT_DIR` to
`<pack>/slots`. Each engine owns a private run directory and removes only its own files on normal exit.
Slots with images are excluded; the cache preserves an 8 GiB free-space reserve. A crashed process may leave
its directory behind. Slot save/restore, including multiple GPUs, still needs CUDA parity validation.

The Maya AMD/ROCm extension is not enabled here: its HIP headers, architecture-specific tuning and GLM GEMM
linkage differ from Strata's backend. Installer and CMake continue to reject GLM with HIP/Intel.
Maya's separate launcher, benchmark command and extra long-segment tokenizer cache are not copied;
Strata keeps its installer, diagnostics and upstream piece cache. Server request-body limits, backlog,
stall recovery and multiple API keys arrived from Strata 0.1.41 and are reused.

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
The prefill test also compares F32 and automatic attention selection. `glm_batch_attention_f32` and
`glm_batch_attention_wmma` compare synthetic attention against a CPU softmax reference, with FP16 cache guards,
negative/empty cell lists and partial tiles. The tensor-core test skips GPUs with insufficient hardware support
and fails if a supposedly supported run silently selects F32. These new device tests have not been run here.
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

The v1.0.4 port also aligns the packed latent-cache starts to 16 bytes and rounds its storage up. Its F32 fallback
uses 16-cell tiles, fitting below 48 KiB of shared memory, so GPUs such as Turing do not need the tensor-core
kernel's larger shared arena. These are Strata compatibility adaptations; their device execution is pending.

Fresh v1.0.4-update validation on Windows on 2026-10-07: 596 Python test cases completed, with 589 passed and
seven skipped. This includes 22 GLM installer/report cases. Four C++ CPU tests passed; the prefill-memory helper
now includes 19 checks, covering packed FP16 sizing/alignment and the unchanged F32 diagnostic layout. The Q4_0
fixture generated and packed on CPU, the CPU image encoder built with MSVC 19.51, and Python/JavaScript syntax
and Git whitespace checks passed. CUDA configuration again stopped with "No CUDA toolset found". The new
attention test prepares 81,920 context values per mode; the prefill test prepares eight configurations and
16,384 continuation-logit comparisons. Neither device test was executed here. Tensor-core compilation/parity,
Turing fallback, full-model quality and one/two-GPU MTP remain pending. No weights were downloaded or production
engine/server started.

Fresh v1.0.14-update validation on Windows on 2026-10-08: 1,116 distinct Python cases completed, with 1,106
passed and 10 skipped (605 server cases: 596 passed/9 skipped; 444 installer cases passed; 67 additional
pack/oracle/tokenizer/calibration cases: 66 passed/1 skipped). The initial full-server run exposed an existing
Windows short-name versus long-name path comparison in a test; path normalization fixed that test, and the
final full run passed. Five C++ CPU tests passed, including 19 prefill-memory checks and 12 device-list checks.
The new Q2_K/Q3_K AVX-512 kernel and synthetic test compiled with MSVC 19.51; arithmetic was skipped because
this CPU lacks AVX-512. The synthetic Q4_0 pack generated on CPU (184 tensors, 15 routed expert tensors,
5 disk-backed MoE layers). The CPU image encoder built; Python/JavaScript syntax and Git whitespace checks
passed. CMake rejected GLM with SYCL and without CUDA. Native Windows GLM configuration stopped at
"No CUDA toolset found". No weights were downloaded and no production engine/server was started.

Prepared attention parity now covers F32, WMMA, Turing register WMMA and Ampere mma.sync separately. Device
compilation/parity, pinned-allocation cleanup, SSD slot round-trips, multiple GPU pipelines, model calibration,
full-model text/image/MTP quality and performance remain unverified. No upstream speed measurements are
presented as measurements of this Strata integration.

`STRATA_GLM_SLOW=1` selects the diagnostic CPU-expert path. Its older GPU expert pool has not been ported;
use the default fast path for the GPU/RAM/SSD tiers.

## Source and licenses

The port includes Strata 0.1.41 `main` at `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. Its initial Maya reference was
`444030a3f2afc01515d4d067d663377329efb5b5`; the prefill and Windows update used
[`5932f601373f53fc021f75dc55159a722c772571`](https://github.com/mw00/project-maya/commit/5932f601373f53fc021f75dc55159a722c772571).
The previous reference was Maya v1.0.4 at
[`cfd2f45b506be6c02d715e3198b3ff9719de25fc`](https://github.com/mw00/project-maya/commit/cfd2f45b506be6c02d715e3198b3ff9719de25fc).
The current selective reference is Maya v1.0.14 at
[`327cfa60d46f9d8eea33a1b38e76284646fbad86`](https://github.com/mw00/project-maya/commit/327cfa60d46f9d8eea33a1b38e76284646fbad86).
Maya renumbered its earlier releases: the previous v1.2.0 is now v1.0.2, the report update is v1.0.3, and the
tensor-core/FP16-cache update is v1.0.4. This does not change Strata's own version number.
Project Maya and Strata use the MIT license; the copyright notice remains in [LICENSE](../LICENSE).
ggml/llama.cpp retains its [MIT notice](../third_party/ggml/LICENSE), and the dashboard font retains its
[OFL notice](../serve/web/fonts/OFL.txt). GLM weights and the Maya quant retain the model publisher's license.
