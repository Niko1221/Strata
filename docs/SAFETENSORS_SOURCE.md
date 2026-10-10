# Read-only safetensors source (standalone, opt-in)

This is the first part of a native NVIDIA NVFP4 checkpoint backend. It reads
the original `model.safetensors.index.json` and its shards without torch,
CUDA, GGUF conversion, or generated model files. It is a container reader,
not a complete inference path or a generic model-architecture adapter.

Configure this standalone project explicitly; the root build, existing engine
sources, backend selection, setup and runtime defaults are unchanged:

```sh
cmake -S native_safetensors -B build-safetensors-source -DCMAKE_BUILD_TYPE=Release
cmake --build build-safetensors-source --config Release
ctest --test-dir build-safetensors-source -C Release --output-on-failure
```

A C++20 compiler and CMake 3.24 or newer are required. The tests additionally
need Python 3 (standard library only); use `-DBUILD_TESTING=OFF` to omit them.
No model downloads or GPU initialization occur in the tests. With a
multi-configuration generator, executables are under `Release/` or `Debug/`.

```sh
build-safetensors-source/strata-safetensors-source headers /path/to/model
build-safetensors-source/strata-safetensors-source read /path/to/model TENSOR 0 16
```

`headers` reads metadata only. `read` is a diagnostic byte-range read, limited
to 4096 bytes. Both emit JSON; failures exit nonzero and emit JSON on stderr.
Original weights are opened read-only. Tensor descriptions retain the
checkpoint's physical row-major shapes and absolute byte ranges. Logical
shapes and packed FP4 interpretation belong to a later model adapter.

The reader rejects duplicate JSON keys, invalid UTF-8, excessive nesting or
header size, shape/offset overflow, unsupported dtypes, overlaps, gaps,
trailing payload, index/shard disagreement, unsafe filenames and resolved
symlink/junction escapes. It supports scalar and empty tensors, unaligned
data, Unicode Windows paths, and 64-bit file seeks. This initial format
requires an index; it rejects unindexed `.safetensors` files in the model
directory and does not guess which shards belong to a checkpoint.

`WeightSource::read_many` takes caller-owned destination spans, validates
the complete batch before I/O, sorts by source location, and coalesces
adjacent reads within one weight family using bounded staging. Descriptors
must belong to that source. Calls are synchronous and not thread-safe.
The family byte counters measure requested source bytes, not physical SSD
traffic or OS page-cache misses. `seal_expert_reads` refuses subsequent
expert and MTP reads. `seal_resident_reads` closes all startup handles and
refuses subsequent payload reads while keeping metadata and counters valid.

The MIT-licensed nlohmann/json 3.12.0 header and its license are vendored in
`third_party/nlohmann`; no dependency is fetched during configuration.

Validation on Windows with MSVC 19.44.35228 and Python 3.13.15:

- Release and Debug build the standalone library, inspector and C++ source test.
- The C++ test checks scatter order, coalescing, family accounting, descriptor
  ownership, batch validation, sealing, and reuse of already-owned sample bytes.
- Twelve Python fixture tests exercise valid and adversarial containers,
  including a sparse file larger than 4 GiB. Eleven pass; the symlink-escape
  fixture is skipped because this Windows account cannot create symlinks.
- The same Release inspector parses the original NVIDIA Qwen3.8-Flash-Next
  NVFP4 checkpoint's 11 shards and 299,545 tensors. It reads 40,402,984 header
  bytes; every family's payload-read counter remains zero in `headers` mode.

No CUDA, HIP or SYCL engine/backend file is changed or built by this project.
Linux execution has not been validated in this contribution. These tests
establish reader behavior; they do not establish inference speed or accuracy.

The complete experimental CUDA integration, lossless NVFP4 layout adapter,
resident CPU/GPU experts, MTP, cache checks and measured limits are available
in [strata-safetensors](https://github.com/spideytznn/strata-safetensors).
They build on [Strata](https://github.com/Niko1221/Strata) and
[sergqwer's NVFP4 implementation](https://github.com/sergqwer/strata-nvfp4).
They are deliberately outside this first contribution's runtime scope.
