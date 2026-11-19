# Vulkan expert-streaming tier for Strata (Intel Arc B390 / Panther Lake)

**Running the model today:** `strata launch --backend vulkan` now uses `engine/`,
the native engine. It runs the whole forward pass itself — resident dense weights
plus MoE experts streamed from the GGUF — and generates tokens without llama.cpp
(~5.5 tok/s in a short warmed window on the B390, one token at a time). The faster
`bridge/` adapter, which drives llama.cpp's Vulkan backend at ~31 tok/s but does
not stream experts, is still available with `--vulkan-impl bridge`. See `docs/VULKAN.md` for the user
guide, and `bridge/README.md` / `engine/README.md` for the two binaries.

This directory contains the **first working vertical slice** of a Vulkan backend
for Strata: the *storage* side of the idea described in
[180B Parameters, 32GB of RAM](https://huggingface.co/blog/moe-expert-streaming).

MoE experts are **not loaded at startup**. They are read from the GGUF on the
NVMe only when the router asks for them, into a **bounded VRAM slot pool** with
LRU eviction and a Q4_K/Q6_K/Q8_0 matvec run by a Vulkan shader. Speculative
next-layer prefetch was measured and is off by default: the routing corpus shows
only 2.9% overlap between consecutive layers.

## What works (verified)

Hardware: **Intel Core Ultra X7 358H + integrated Arc B390**, `xe` driver,
Mesa ANV Vulkan 1.4, 32 GB of shared RAM, YMTC PC411 NVMe.

Test model: `/home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf`
(41 layers, 256 experts per layer). `ffn_gate_exps` / `ffn_up_exps` are Q4_K
(589,824 bytes per expert). `ffn_down_exps` is **mixed**: Q6_K (860,160 bytes per
expert) in 21 layers and Q4_K (589,824 bytes per expert) in the other 20 — the
shader is chosen per tensor, not per model.

```
cmake -S engine -B build-engine-vulkan -G Ninja -DSTRATA_ENABLE_VULKAN=ON
cmake --build build-engine-vulkan
./build-engine-vulkan/vulkan/strata-vulkan-stream-test \
  --model /home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf \
  --shader engine/vulkan/shaders/expert_q4_k_gemv.comp.spv \
  --down-shader engine/vulkan/shaders/expert_q6_k_gemv.comp.spv \
  --silu-shader engine/vulkan/shaders/expert_silu_mul.comp.spv \
  --router-gemv-shader engine/vulkan/shaders/router_gemv.comp.spv \
  --router-topk-shader engine/vulkan/shaders/router_topk.comp.spv \
  --experts 256 --slots 64 --threads 8 --rows 64 --forward 8
```

Result (layer 0, Q6_K down):

```
Q4_K gate matvec:   gate     max_abs=1.37e-06 max_rel=0.00189 bad=0/16384
down matvec (type 14):   down     max_abs=2.09e-07 max_rel=0.00196 bad=0/16384
full expert forward (8 experts):
  gate     max_abs=1.4e-06   bad=0/4096
  up       max_abs=1.01e-06  bad=0/4096
  silu*up  max_abs=1.49e-07  bad=0/4096
  down     max_abs=2.51e-08  bad=0/16384
stats: hits=0 misses=536 prefetches=256 MB=538.5 io_ms=39
TOTAL bad_rows=0/61440
PASS
```

The same test with `--layer 5` (Q4_K down, type 12) also passes: the shader is
chosen from the tensor's `ggml_type` field, so one GGUF can mix both formats.
Correctness is checked row by row against ggml-compatible dequantization (Q4_K
and Q6_K) read straight from the file. In the `forward` phase every stage (gate,
up, SiLU*up, down) is compared separately. The slot pool is sized from the
largest tensor (Q6_K, 860,160 bytes), so it covers both formats.

`bash engine/vulkan/asan_check.sh` rebuilds and runs all six test binaries with
AddressSanitizer (output in `/tmp/opencode/strata-vulkan-asan/`). All pass with
no ASAN reports. This closes the heap-buffer-overflow that a stale
`/tmp/opencode/router-asan` binary still shows: `read_spv` used brace
initialisation (`std::vector<char> data{n}` narrowed `n` to one char, then
`fread` wrote n bytes into it). Current sources use parentheses; the script
keeps that verified.

Phase 5 adds the **shared expert** (`ffn_gate_shexp` / `ffn_up_shexp` /
`ffn_down_shexp` plus the F32 `ffn_gate_inp_shexp` scalar gate). The same
matvec and SiLU×up shaders compute `sigmoid(dot(gate_inp_shexp, x)) *
down(silu(gate(x)) * up(x))`, streamed through the same slot pool with
`Key{..., shared=true}`. Layer 0: 0/2048 bad rows, max abs 3.73e-08; layer 5
(Q4_K down): 0/2048, max abs 1.86e-08. The test also shows the wrong readings
are observable: silu-on-up 33% / 14% apart, a per-dimension gate 47% / 46%
apart, and dropping the scalar gate 98% / 111% apart (layers 0 / 5).

The final MoE step is the **combine**: `out[j] = sum_e w[e] * parts[e][j] +
shared[j]`. `shaders/moe_combine.comp` runs it on Vulkan, with a test (`dim=256,
k=10` as in the parity file and `dim=2048, k=8`) that checks the two rival
readings first (unweighted routed 95% / 80% apart, weighted shared 94% / 77%
apart), then the kernel (rel 1.04e-08 / 8.10e-09) and a null shared vector that
really omits it (99.8% / 99.3% apart). The native engine's `GEN` path uses it.

The greedy **sampler** kernels (`shaders/argmax_rows.comp`,
`shaders/row_top_prob.comp`) are ported too. `argmax_rows` uses strict `>` so
ties go to the lowest index, skips NaN, and returns 0 for a row with no value
above -inf; `row_top_prob` is `exp(v_max - v_max) / sum exp(v_i - v_max)` with
NaN skipped. Tested on the verify-window fixtures: vocab sizes 248320, 40525,
4097 and 33, 1..8 rows, ties/NaN/-inf, three launches on the same buffers —
0 failures.

A **BF16 GEMV** (`shaders/bf16_gemv.comp`) is also available for the
projections the native forward will need (`ssm_alpha`, QSA indexer, PLE).
Tested at the parity file's real shapes ([2560,48], [2560,128], [2560,512],
[2560,2560]): rel_l1 6.3e-08 ... 7.2e-08 against a double host reference over
the same bf16 bits. The activation contract is observable — routing the
activation through fp16 first changes the result by 0.14% ... 0.16%. The
parity file's naive-vs-split agreement check has no direct Vulkan equivalent
(one reduction strategy is ported); the tight reference bound covers the same
property for this kernel.

## Structure

| File | Role |
|---|---|
| `shaders/expert_q4_k_gemv.comp` | Vulkan matvec: Q4_K dequantization times the activations, addressed through a **dynamic storage-buffer offset** (any slot, no descriptor rebinding) |
| `shaders/expert_q6_k_gemv.comp` | Vulkan matvec for Q6_K `ffn_down_exps`; same interface, selected by the tensor's `ggml_type` |
| `shaders/expert_q8_0_gemv.comp`, `src/q8_0_gemv_test.cpp` | Q8_0 matvec (34 B / 32 elements = 272 B / 256) selected for `ggml_type == 8`, plus a model-free synthetic-GGUF integration test. |
| `shaders/expert_iq{3_xxs,2_xs,1_m}_gemv.comp`, `shaders/iq_tables.glsl`, `tools/gen_iq_tables.py`, `src/iq_gemv_test.cpp` | IQ3_XXS / IQ2_XS / IQ1_M F32-activation dequant matvec, selected for ggml types 18 / 17 / 29. ggml's lookup grids are generated from `ggml-common.h` into `iq_tables.hpp` (CPU oracle) and `iq_tables.glsl`; the synthetic-GGUF test checks 64 rows × 2048 columns per format with 0 bad rows. |
| `shaders/expert_silu_mul.comp` | Vulkan `silu(gate)*up` elementwise activation; first kernel of the native port moved off the CPU |
| `shaders/router_gemv.comp` | F32 router GEMV (`ffn_gate_inp.weight`, 256×2048 here) |
| `shaders/router_topk.comp` | Softmax → stable argsort → top-8 → clamp/renorm, matching the SYCL router parity spec |
| `src/router.cpp`, `include/strata/vulkan/router.hpp` | `Router` API: `gemv()` and `topk()` |
| `shaders/moe_combine.comp`, `src/moe_combine.cpp`, `include/strata/vulkan/moe_combine.hpp` | Vulkan routed+shared combine: weighted routed sum, unweighted shared expert, null `shared` omits it |
| `src/moe_combine_test.cpp` | Combine test with the parity file's two rival-reading traps (no model file needed) |
| `shaders/bf16_gemv.comp`, `src/bf16_gemv.cpp`, `include/strata/vulkan/bf16_gemv.hpp` | BF16 GEMV for projections: bf16 bits in, F32 out, 64-lane workgroup reduction |
| `src/bf16_gemv_test.cpp` | BF16 GEMV test at four real shapes, with the bf16-vs-fp16 activation trap (no model file needed) |
| `shaders/argmax_rows.comp`, `shaders/row_top_prob.comp`, `src/sampler.cpp`, `include/strata/vulkan/sampler.hpp` | Greedy sampler: argmax with lowest-index ties and NaN skipping, plus the top token's probability |
| `src/sampler_test.cpp` | Sampler test on verify-window fixtures (vocab 33..248320, 1..8 rows, ties/NaN/-inf, repeated launches) |
| `ExpertIndex::find_router`, `ExpertStream::prefetch_route` | Indexes the real F32 `ffn_gate_inp.weight` matrices and prefetches gate/up/down for the router's top-8 experts; a `pending_` map dedupes in-flight reads |
| `src/router_test.cpp` | Router correctness test (no model file needed) |
| `src/expert_stream.cpp` | Minimal GGUF parser, `[layer][kind][expert] -> offset` index, slot pool, LRU, `loading` lock, `pread()` thread pool, speculative prefetch, four pipelines (Q4_K/Q6_K/Q8_0 matvec + SiLU), the shared-expert forward and the CPU references |
| `include/strata/vulkan/expert_stream.hpp` | API: `ExpertIndex`, `ExpertStream::ensure/prefetch/matvec/silu_mul/shared_forward`, stats, `q4_k_dot_reference` / `q6_k_dot_reference` / `q8_0_dot_reference` |
| `src/expert_stream_test.cpp` | Integration test on the real GGUF: Q4_K matvec, down matvec, full gate -> SiLU -> up -> down forward, shared expert with structure traps |

The stream test's last phase is `router-guided prefetch`: it reads the model's
real layer-(N+1) router matrix, routes a deterministic hidden vector to top-8,
calls `prefetch_route()`, and checks that all 8 experts are served from the
cache with no extra misses (8/8, 0 misses on layers 0 and 5). This uses real
router weights but not a real model forward. Traces captured from real prompts
are checked in under `traces/`; `check_prefetch.py` scores them and reports 2.9%
consecutive-layer overlap and 35.7% same-layer overlap across tokens.

Router test (build as above, then run `strata-vulkan-router-test`): exact expert
IDs on random, all-equal, 12-way-tie, dominant and peaked logits; worst relative
weight error 2.4e-7; GEMV on a 256×2048 F32 matrix within 6.3e-6. It follows
`engine/sycl/src/kernels/router_top10_parity.cpp`, including the smallest-index
wins tie rule and the 2^-14 renormalisation clamp.

## Why Vulkan here

All the extensions we need are present on this machine's B390:
`VK_KHR_buffer_device_address`, `VK_EXT_descriptor_buffer`,
`VK_EXT_external_memory_host`, `VK_KHR_timeline_semaphore`,
`VK_EXT_subgroup_size_control`, `VK_KHR_cooperative_matrix`.

On an iGPU the memory is shared: the slot pool is `HOST_VISIBLE | HOST_COHERENT`,
so `pread()` writes **straight into VRAM** (zero copy). No oneAPI, Level Zero,
Docker or custom kernel driver is needed — just Vulkan and C++17.

## What is NOT done yet (important)

The engine in `engine/` generates tokens end to end through Strata. What is still
open:

0. QSA sparse attention, KV quantization, gated residual and n-gram PLE — those
   are for Qwen3.8-Flash-Next, not this model. Dense GQA, GDN, the router,
   the shared-expert FFN, the routed+shared combine, BF16 GEMV and the greedy
   sampler are ported and used by the native forward;
1. prompt batching and MTP: the native forward is one token at a time with fp32
   activations and host-side sampling. `GEN` no longer calls libllama;
2. timeline semaphores and transfer-queue overlap for submissions (the known
   iGPU bugs are in issue #515). `strata doctor --vulkan` probes the device and
   sizes the expert cache from shared memory; `strata launch --backend vulkan`
   points at `engine/`, and `--vulkan-impl bridge` keeps the old path.

Done in this iteration: the Q6_K shader for `ffn_down_exps`, pipeline selection
by `ggml_type` (the test GGUF mixes Q6_K and Q4_K down experts), a full
forward test with a CPU comparison of every stage, a Vulkan SiLU×up shader so
the whole gate → SiLU → up → down path runs on the GPU (0 bad rows on layers
0 and 5), the two MoE router kernels with a parity test, a native engine
skeleton (`engine/`) verified through Strata's server, the shared expert and
the routed+shared combine with the same structure traps as the SYCL parity
test, a BF16 GEMV for projections and the greedy sampler kernels.

The module can also be linked as a **library** (`strata_vulkan_stream`). The
native engine in `engine/` links it and supplies the remaining forward kernels;
the bridge adapter in `bridge/` takes the other route through llama.cpp.

