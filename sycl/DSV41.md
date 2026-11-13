# DeepSeek V4.1 Flash on SYCL

This opt-in target ports Strata's current `src/dsv41` engine to SYCL. The source
comes from Project Maya; its notices and MIT license remain in
`third_party/project-maya`. Other engine backends are unchanged.

The initial target is a local DeepSeek-V4.1-Flash-1017 GGUF pack, including its
Engram tables and a compatible DSpark GGUF. EXL3 packs and the experimental PCIe
stream lane are rejected. Multi-GPU, vision, long contexts and serving acceptance
are not established by the component tests below.

## Build and pack

Use an environment with oneAPI, oneMKL, Level Zero headers and a C++20 compiler.
Development uses `intel/omix:0.4.1-devel-ubuntu24.04`; model files are mounted
read-only. Use the user's selected device and verify its PCI address before a run.

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S sycl -B build-sycl -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON
cmake --build build-sycl --target strata-dsv41 dsv41_numerics_parity \
  dsv41_weights_parity dsv41_mailbox_parity -j
python3 tools/dsv41/pack_gguf.py --model /models/UD-IQ1_M \
  --dspark-gguf /models/dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
  --out /work/model-pack
```

The stock packer writes metadata pointing at the existing weights. It does not
copy them. The engine takes token IDs; use the pack's tokenizer and chat template,
as the server does. `--pack-info` inspects the pack without loading a device.

## Backend behavior

The port retains top-6 routing, encoded weights, CPU cold experts, Engram reads,
the GPU expert tier, and the current engine's DSpark verification/restore logic.
Portable kernels decode GGUF blocks; dense BLAS calls use oneMKL. CUDA WMMA
fragments are not treated as Intel XMX fragments. Small expert groups retain the
upstream F32 tiles. For supported GGUF experts with at least 32 token rows, the
SYCL default decodes their original weights once to FP16 and uses oneMKL GEMM
with FP32 accumulation. `DS4_FAT_ROWS=0` restores the tile-only path; an explicit
positive threshold overrides 32. This changes accumulation order, not routing
or the encoded weight source, and needs the full-model checks below.

Leave `--prefill-chunk` unset for normal serving. The upstream automatic rule
amortizes cold-expert uploads across the prompt.
Attention projections use a 1024-row slab on this discrete-GPU backend when no
slab override is supplied. Allocation retries can restore the smaller slab;
`DS4_ATTN_SLAB=0` explicitly uses the upstream base slab.

Existing `mmap` allocations cannot be registered for GPU reads on this backend.
`DS4_STORE_ZC`, `DS4_STREAM_ZC`, `DS4_COPY_PIN` and
`STRATA_DS4_PIN_BY_NEED` must stay off. CPU readers may read the weight mappings;
GPU uploads use staging allocations. `DS4_STREAM` / `STRATA_DS4_STREAM` must also
stay off. The NVIDIA one-GPU registered-memory path is disabled in the SYCL copy.

Mailbox allocations use the existing uncached Level Zero host allocator. At the
CPU result handoff the host finishes queued resident GPU work, then waits for the
CPU response with a 60-second timeout. The CPU expert worker overlaps that queued
work. The GPU consumes host results through explicit uncached reads. This avoids
an unbounded CUDA-style GPU spin on cached host memory. `DS4_SYCL_TRACE=1` logs
mailbox issue/receive/answer sequences for diagnosis.

Chunked prefill completes the preceding chunk and the next input upload before
reusing its Engram/BLAS scratch buffers; CPU read-ahead still runs independently.
The GGUF flow hands ring ownership over on the host: uploads finish before
compute reads their slots, and compute finishes before an upload overwrites
them. Staging is also refilled only after its own upload event completes. The
host waits for the relevant slot or batch event, preserving independent queued
work; it does not enqueue GPU-side cross-queue event waits. The model preserves the backend's `UR_L0_USE_COPY_ENGINE=0` default before creating queues, honoring an
explicit current/legacy setting. That generic setting does not control V2's
native copy-offload hint and does not replace the ring ownership handoffs. While
waiting for routing, a host completion task lets the main thread continue
pumping the expert copy ring without querying driver queue/event status.

The numeric port also preserves synchronous weight-copy lifetimes, uses portable
E4M3 conversion, and retains the parentheses of nested rounded arithmetic. Small
device codebooks have process lifetime to avoid freeing them after dpct queues
have already been destroyed. BLAS warm-up uses private scratch: borrowed chunk
row buffers are not available before the prompt starts.

## Functional checks

```sh
ctest --test-dir build-sycl -R 'dsv41_' --output-on-failure
build-sycl/dsv41_weights_parity /work/model-pack /models/UD-IQ1_M
```

The numeric test checks all E4M3 encodings, midpoint ties, neighboring floats and
seeded float patterns against an independent CPU oracle. The weight test reads
17 real rows per format from the pack, checks dequantization against ggml, and
checks one-, four- and eight-column products against double-precision sums. For
the busy-expert formats it separately checks the FP16 row decoder against ggml
and 64-column oneMKL products against independent double-precision sums. It also
checks the IQ4_NL Engram decoder. The mailbox test checks top-6 routing,
the full 5120-value request payload and CPU response visibility.

Full-model checks must additionally cover plain greedy decoding, DSpark with
every draft verified (`--dspark-conf 0`), rejected-draft restore, and chunked
prefill. Component passes alone do not establish model correctness or performance.
`dsv41_rollback_parity PACK PROMPT_IDS_FILE` checks 16 deliberately rejected
draft windows against a plain greedy trajectory from the same loaded model and
fixed expert placement. It checks positions and subsequent tokens, and reports
full-vocabulary logit differences separately.
