# Jetson AGX Orin port validation, 2026-10-07

Hardware: Jetson AGX Orin Developer Kit, 32 GB nominal RAM; Linux aarch64,
JetPack 6.2 / L4T R36.4.4, GCC 11.4, nvcc 12.6.68, CUDA runtime and driver
12060. `/proc/meminfo` reports 32,187,883,520 physical bytes. The GPU reports
sm_87 and the same total allocation ceiling. Managed memory is supported;
concurrent managed access is not. Mapped host registration and VMM are supported.

Revision: working tree on `feat/jetson-orin-arm64`, based on
`d5ea7133741e67743c0e886bb426c0ce8d69cf6c`. ggml/llama.cpp is pinned to
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. These are correctness checks;
there is no model throughput benchmark or validated model configuration yet.
Power mode, thermals and storage throughput were not measured.

Build commands and remaining work are in [the plan](../../../docs/JETSON_ORIN_PLAN.md).
The engine was built with CUDA enabled, `CMAKE_CUDA_ARCHITECTURES=87`,
`CMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc`, Release, and two build jobs.
CPU tests used the same pinned ggml source and native ARM CPU backend.

| Check | Result | Evidence |
| --- | --- | --- |
| ARM multi-token, tasks, affinity, router, budget | Pass | [CPU log](cpu-tests.txt) |
| Worker park/wake stress, 20 seconds | 3187 batches, zero missed jobs | [Stress log](pool-stress.txt) |
| q4_0/q4_0, q4_K/q5_1, q8_0/q8_0 synthetic expert CPU/GPU parity | Zero failures | [Parity log](native-expert-parity.txt) |
| DMA, native multi-token, streamed KV, pinned shared arena, device arena | Five tests pass | [Memory log](memory-tests.txt) |
| Coder IQ1_M real weights, expert 7 of each of 48 layers | Zero failures | [Real-weight log](coder-real-expert-parity.txt) |
| File expert source, cgroup limits, exhausted/unknown RAM, VMM ranges | Three tests pass | [Source log](file-source-tests.txt) |
| Resident/file expert exchange, with/without pinned storage and rotation | Exact data preserved through 64 exchanges per mode | [Rotation log](file-source-rotation.txt) |
| Startup with headroom exceeding physical capacity | Refuses before model files or weights are loaded | [Startup log](startup-refusal.txt) |
| Injected whole-arena/slice registration failures, error cleanup and pageable copy | Pass | [Fallback log](pinned-fallback.txt) |
| Actual expert cache VMM shrink/grow, stable address, data retention, cudaMalloc, budget refusal | Two tests pass | [Cache log](expert-cache-tests.txt) |
| Optional Q8_0 activation contract, zero/tie/random-scale inputs | Matches pinned ARM ggml over 5120 blocks | [Contract log](q8-contract.txt) |
| ARM64 Python 3.12 dependency wheel availability | All applicable pinned requirements available as wheels; installation and pip check pass | [Wheel log](python-wheels.txt) |
| Q2_0 GGUF rows, 1–8 tokens and 1/3/10/40 blocks, independent FP64 reference | Zero failures | [Reference log](q2-reference.txt) |
| Native expert synthetic K/Q quantization suite | Eleven cases pass | [Suite log](native-suite.txt) |
| IQ3_XXS/IQ4_NL, IQ2_S/Q2_0, IQ3_S/IQ4_NL, IQ4_XS/Q2_0 synthetic experts | Zero failures | [IQ log](iq-parity.txt) |
| CUDA probe with GPU visibility disabled | Returns error, no invented device | [Probe log](no-gpu-probe.txt) |
| Coder tokenizer header extraction | 16 round-trip strings, zero failures | [Tokenizer log](tokenizer-check.txt) |
| Mock-engine API, security and quiet cancellation | 290 tests pass | [Server log](server-tests.txt) |
| Desktop setup regression fixtures, x86 CPU simulated | 110 tests pass | [Setup log](setup-regression.txt) |

For the small cache resize test, `STRATA_UMA_HEADROOM_GIB=0` permits its eight
MiB allocation even when this machine has less than the normal six GiB reserve
available. The separate refusal process sets headroom to 1024 GiB and checks
that allocation is rejected. These settings are test inputs, not deployment
recommendations. The refusal test skips on discrete GPUs.

Server tests use a mock engine on loopback; they do not prove actual model
inference, API integration with the CUDA engine, or inference cancellation.

The optional CPU vision encoder builds natively on ARM and reaches argument
parsing. The CUDA encoder also builds and reaches argument parsing; image inference
remains unverified.

Read-only nvmap diagnostics found PID 4730 (`ninfer-serve`, systemd service
`ninfer-qwen38-27b.service`) holding 26,603,488 KiB of shared GPU allocations.
This explains the roughly two GiB available during these tests. No existing
service was stopped during this checkpoint.

Tokenizer extraction used the already-downloaded header of shard one, while
its data download continued. This validates tokenization only. Both complete shards subsequently passed
SHA256 verification; model weight loading remains unverified.

The first downloaded Coder shard matches the pinned LFS SHA256 and byte count
([hash log](model-shard1-hash.txt)). Shard two also matches its pinned LFS SHA256 and byte count
([second hash log](model-shard2-hash.txt)). Both model shards are complete.

A 23.42 GiB file-backed Coder expert pack and 1.37 GiB canonical dense arena
were prepared from the verified shards ([preparation log](pack-preparation.txt)).
With the existing ninfer service still running, actual Coder startup correctly
refuses the insufficient shared budget before loading weights
([startup log](coder-startup-memory.txt)). Successful inference still needs
that memory released; this refusal is not a successful model run.

The MTP download is complete: all 31 tensors match the pinned checkpoint
([verification](mtp-verification.txt)); packing and runtime-file conversion are complete; MTP inference remains
outstanding. The vision projector is checked separately against its pinned
repository SHA256 ([verification](projector-verification.json)).

An x86_64 cross-build of the CPU kernels and router test succeeds with GCC
11.4. No x86 runtime or desktop GPU validation was performed on this ARM host.

A local loopback-only validation configuration is prepared at
`/home/calvin/models/strata-orin-validation/strata-orin-validation.json`. It uses
the built CUDA engine, a 4096-token context and file-backed experts. This is a
test configuration, not a measured deployment default.

The locally built ARM64 engine and CPU vision encoder are installed in
`engine-cuda12/`. The setup build-cache check recognizes their current source
hashes and ARM64 metadata without rebuilding
([installation check](local-installation.txt)). The validation configuration
now points at that installed engine.

After sufficient shared RAM is available, start the actual-model API test with:

```sh
.venv/bin/python -m serve.server --engine strata \
  --config /home/calvin/models/strata-orin-validation/strata-orin-validation.json \
  --host 127.0.0.1 --port 18081
```

This command has not yet passed real-model startup. MTP, longer contexts,
repeated requests, cancellation and image inference still require hardware
validation. No existing server configuration was replaced.

MTP preparation now streams completed GGUF tensors to disk and avoids the
list-plus-concatenation duplicate of each expert tensor. Synthetic Q2/Q4/Q8
fixtures retain exact tensor bytes ([check](mtp-streamed-fixture.txt)). Actual
Q2 MTP packing completes in 203.1 seconds, producing 0.889 GB of tensor data;
maximum child RSS is 1,878,516 KiB, including mapped source pages. Minimum
system available RAM observed at one-second intervals is 1,382,080,512 bytes
([packing log](mtp-pack.txt)). This is a preparation measurement with the
existing server and compilers running, not an inference memory benchmark.
Runtime conversion produces 707,788,800 bytes of expert data and 116,099,072
bytes of dense data ([conversion log](mtp-runtime.txt)). A separate validation
configuration, `strata-orin-validation-mtp.json`, enables MTP with the shipped
CJK draft vocabulary. Neither validation configuration has run model inference.

All 512 prepared MTP experts pass the ARM generic CPU execution check against
the scalar INT8 activation oracle ([parity log](mtp-all-experts-parity.txt)).
The diagnostic retains legacy “VNNI” labels; on this ARM build it exercises
the portable CPU implementation. This check does not test MTP GPU execution
or full draft-token acceptance.

[The reference helper](reference_logits.cpp) compiles against the pinned ARM
llama.cpp CPU library. It consumes explicit token IDs, decodes one position at
a time, checks finite logits, and writes the same binary header/FP32 row layout
with a vocabulary/row-count header. Its argument checks passed at this checkpoint;
model evaluation followed on October 8. Build it after the CPU reference libraries are built:

```sh
g++ -O2 -std=c++17 \
  bench/results/2026-10-07-jetson-orin/reference_logits.cpp \
  -I/tmp/strata-orin-build/_deps/strata_llamacpp-src/include \
  -I/tmp/strata-orin-build/_deps/strata_llamacpp-src/ggml/include \
  -L/tmp/strata-orin-reference/bin \
  -Wl,-rpath,/tmp/strata-orin-reference/bin -lllama -lggml \
  -o /tmp/strata-orin-reference/reference-logits
```

Run it with `SHARD1 OUTPUT TOKEN_ID...` once enough shared RAM is available.
Compare the final reference row against Strata using the same IDs,
`--max-new 1`, `--prefill 1`, and the environment variable
`STRATA_DUMP_FIRST_LOGITS=OUTPUT`. That hook writes raw FP32 logits without
a header. The native-pack path does not implement `--dump-logits`.

The installer now bounds ARM Linux build parallelism by `MemAvailable`:
it leaves two GiB aside and allows two GiB per job, limited by half the CPU
count. These are build heuristics, not measured compiler peak requirements.
With little available RAM or a missing reading it chooses one job. All 15
Jetson setup cases pass, including command selection and desktop preservation
([Jetson setup log](setup-jetson.txt)); the 110 desktop regression fixtures
also pass after this change.

The upstream ARM CPU `llama-completion` reference executable builds and
launches its help command ([configuration](reference-config.txt),
[build](reference-build.txt)). It uses the same pinned llama.cpp source with
CUDA disabled. Neither the reference executable nor the explicit-ID logit
helper has evaluated the real model yet.

A [56×56 synthetic image](vision-fixture.png) is prepared for CPU/GPU encoder
comparison ([fixture SHA256](vision-fixture.json)). After shared RAM is freed,
use `--min-tokens 16 --max-tokens 16 --threads 8`, the verified projector and
first model shard, and send `ENC <absolute image path> <output>` followed by
`QUIT` on standard input. Run once without `--gpu` and once with it. Encoder
outputs have five int32 header fields (magic, tokens, grid width, grid height, embedding width) followed by FP32 embeddings. Check shape,
finiteness and numerical differences before claiming vision support. No image
inference has been run yet.

The CUDA vision executable builds successfully ([build log](vision-cuda-build.txt))
and contains 143 SM87 CUDA ELF images ([image list](vision-cuda-images.txt)).
It is installed in `engine-cuda12/strata-vision`, with the CPU-only executable
preserved as `strata-vision-cpu` ([installation](vision-installation.txt)).
Setup no longer reuses a CPU-only encoder for a GPU vision request; both reuse
modes are covered by the 15 Jetson setup cases. Actual image inference remains
unverified.
