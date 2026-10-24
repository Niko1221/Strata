# Run the experimental Muse Spark reference CLI

<!-- Updated: 2026-10-22 -->

`strata-muse` executes Muse Spark's text graph from existing GGUF files, using
memory-mapped weights and bounded decoded row tiles. Build it explicitly; it is
**off by default** and does not change the normal [Qwen installation](MODELS.md).

This is a numerical reference path, not a new model in the installer or chat/API
server. It accepts token IDs, not chat messages. It does not use Strata's Qwen
expert cache, speculative decoder, multi-GPU scheduler, image/audio encoders or
reasoning/tool parser. Full-checkpoint correctness and useful 816B performance
need separate validation; a synthetic fixture cannot establish either.

## Build and check a synthetic model

Use the normal [build environment](DETAILS.md), with the Python packages in
[`requirements.txt`](../requirements.txt), including NumPy, and CMake/Ninja from
[`requirements-build.txt`](../requirements-build.txt) or your system packages.
No model download is needed for these commands. Run from the repository root:

```sh
cmake -S . -B build-muse \
  -DSTRATA_ENABLE_MUSE=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF
cmake --build build-muse --target strata-muse muse_model_test --parallel 2
ctest --test-dir build-muse -R '^muse_model_test$' --output-on-failure
python -m unittest discover -s tools/muse -p 'test_*.py'
python -m tools.muse.fixture --output-dir /tmp/muse-fixture
build-muse/src/muse/strata-muse \
  --model /tmp/muse-fixture/model.gguf \
  --tokens 1,5,8,3,7,2,9 --ctx 32 --steps 0 --backend cpu \
  --logits-file /tmp/muse-fixture/native-cpu.json
python -m tools.muse.compare \
  --reference /tmp/muse-fixture/reference.json \
  --actual /tmp/muse-fixture/native-cpu.json
```

GGML's row converters come from Strata's existing pinned dependency. Keep
`STRATA_NATIVE_EXPERTS=ON` (the default). An existing matching llama.cpp checkout
can be selected with `-DSTRATA_GGML_DIR=/absolute/path/to/llama.cpp` to avoid a
build-time fetch. This does not change the dependency pin.

## Check the CUDA adapter

Build another directory with `STRATA_ENABLE_CUDA=ON`. For an RTX 3060, CUDA
architecture 86 is appropriate:

```sh
cmake -S . -B build-muse-cuda \
  -DSTRATA_ENABLE_MUSE=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-muse-cuda --target strata-muse muse_model_test --parallel 2
ctest --test-dir build-muse-cuda -R '^muse_model_test$' --output-on-failure
build-muse-cuda/src/muse/muse_model_test --cuda
build-muse-cuda/src/muse/strata-muse \
  --model /tmp/muse-fixture/model.gguf \
  --tokens 1,5,8,3,7,2,9 --ctx 32 --steps 0 --backend cuda --gpu 0 \
  --logits-file /tmp/muse-fixture/native-cuda.json
python -m tools.muse.compare \
  --reference /tmp/muse-fixture/reference.json \
  --actual /tmp/muse-fixture/native-cuda.json
```

Select the architecture and device for your hardware. CUDA accelerates tiled
matrix-vector products through cuBLAS; attention, routing, normalization and row
dequantization remain on the CPU. This is not an optimized GPU inference engine.
No Muse HIP, SYCL or multi-GPU path is provided.
With a HIP engine build, Muse uses its CPU path; the gfx906 compatibility mode
does not enable the NVIDIA adapter. Build Muse separately from the SYCL project.

The CTest above uses the CPU backend. The explicit `muse_model_test --cuda`
check runs CUDA logits, four greedy steps, EOS/EOT stops and synthetic quantized
rows against CPU results. It uses logical device 0; set `CUDA_VISIBLE_DEVICES`
to a GPU UUID to check one physical card. The separate NumPy comparison checks
the full synthetic F32 logits against an independent implementation.

## Supply your own GGUF

The serialized architecture must be `muse-spark`, with the tensor names and
converter-folded scales used by
[llama.cpp's Muse Spark graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/muse-spark.cpp).
[Meta's model](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)
and its conversions have their own license and access requirements.

Pass the first shard with `--model`; every sibling shard must be present and
unchanged throughout execution. The CLI never downloads files. Missing tensors,
invalid geometry, truncated shards, unsupported row encodings and scaled RoPE
produce errors rather than falling back to a Qwen graph. Matrix rows are decoded
with the pinned GGML converter; recognizing an encoding does not imply a native
Strata quantized kernel or full-checkpoint validation.

Use the checkpoint's tokenizer and chat template to obtain token IDs outside
this CLI. `--steps N` generates up to N greedy token IDs and stops at metadata
EOS/EOT. `--ctx` must hold the forced input plus requested generation. Output is
JSON with:

- `tokens`: the supplied forced input IDs;
- `logits`: one full next-token logit vector per forced input;
- `generated_tokens`: generated IDs, including an encountered EOS/EOT.

The default context is 128. Unknown/duplicate flags, invalid token IDs and
context overflow fail with a nonzero exit. Output must not alias any input
shard. Execution/write failures can leave incomplete JSON; check the exit code
before using it.

## Memory limits

Weights stay file-backed; no full model or expert layer is copied into a GPU
cache. KV buffers use a local-layer ring and bounded global-layer history. The
reference rejects calculated KV plus activation/tile scratch above 1 GiB.
Headers, mapped-file/page-cache residency, allocator and library overhead are
**not** a total-process memory guarantee.

The CUDA adapter explicitly allocates two 4 MiB buffers and 256 output bytes.
CUDA context and cuBLAS overhead are additional, so available VRAM must exceed
those buffers. These limits do not make a large checkpoint fit the disk or
establish acceptable storage-backed inference speed. Coordinate resource use
before testing beside an existing model service.

## Understand the numerical check

The fixture contains locally generated, nonuniform F32 weights: hidden64,
head16, four query/two KV heads, latent16, FF32, four experts/top2, vocabulary64,
two layers, context32 and window3. Its local/RoPE and global/NoPE pattern is
`[true,false]`. Seven forced tokens cross the local-window boundary and yield
448 next-token logits.

The oracle is independently implemented in NumPy and reads back serialized F32
values. It covers embedding normalization, gated GQA, local masking/RoPE,
alpha/beta residuals, full-stream sigmoid routing with selection-only correction
bias, latent projections, per-expert input-scaled normalization before weighted
accumulation, final logit scaling and softcap. Norm deltas, Q/router scales and
residual gates are already folded in GGUF; they must not be folded twice.

Mutation tests verify that omitting or reordering these operations is detected.
The comparator requires matching tokens and full matrix shape, rejects nonfinite
or malformed values and exits 1 on disagreement. Its default elementwise bound
is `abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`.

The fixture has no tokenizer, real model weights or multimodal inputs. It does
not validate a published quantized checkpoint, model quality, full context or
performance. Record those results separately in the issue or pull request.
