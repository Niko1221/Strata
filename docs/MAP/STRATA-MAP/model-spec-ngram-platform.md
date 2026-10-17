# Strata map: model, speculative, n-gram and platform code

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root. Scope: `src/qwen35/`, `src/spec/`, `src/ngram/`, `src/artifact/`, `src/platform/` and their headers under `include/strata/`.

## 1. Overview

**Watch out:**

- src/qwen35 is NOT the Flash-Next forward.
- It is a separate driver (`strata-q36`) for **Qwen3.6-35B-A3B (GGUF arch qwen35moe)**. It loads GGUF weights directly, and `strata` switches to it for a qwen35moe model (src/qwen35/q36_engine.cu:1-18, src/qwen35/q36_main.cpp:1-2).
- The Flash-Next per-layer forward, with GR/hyper-connection `hc_*` tensors and `qsa_layer`/`gdn_layer`, is in **engine-core.md** (src/core/layer.cpp:226, src/core/layer.cpp:1025, src/core/layout.cpp:153-161). src/spec holds the speculative-decode *policy* (cost model, MTP vs n-gram lookup choice, suffix drafter).
- Flash-Next's `src/program/generate.cpp` uses that policy (see prefill.md, server.md). src/ngram holds the SSD-streamed n-gram/PLE table reader. src/artifact holds only two small GGUF CLIs; the real readers and dequantizers are header-only in include/strata/artifact/. src/platform has unbuffered async file I/O and memory pinning/residency.
- src/platform has no GPU-architecture defaults or PCIe probing.
- CPU pinning lives in `include/strata/platform/aux_cpus.hpp:1-4`.

## 2. Files

| path | lines | purpose |
|---|---|---|
| src/qwen35/q36_engine.cu | 2542 | Full Qwen3.6-35B-A3B engine: GGUF load, GDN+attention+MoE layer, MTP spec, hybrid CPU/GPU experts; serve/bench CLI |
| src/qwen35/q36_main.cpp | 5 | `main` -> `strata::qwen35::run` |
| src/qwen35/q36ref.cu | 832 | Exactness-first reference (1 token at a time, fp32, plain or Strata kernels), checked vs llama.cpp |
| src/spec/controller.cpp | 73 | `CostModel::step_ms` + `Controller::choose/observe`: picks MTP vs lookup and k from expected tokens/ms |
| src/spec/draft_policy.cpp | 161 | `DraftPolicy`: learned window cost (EMA), lookup acceptance by match bucket, MTP/lookup chaining |
| src/spec/draft_source.cpp | 36 | Registry of extra draft sources; `propose_from_sources` keeps the longest match |
| src/spec/suffix_drafter.cpp | 98 | Hashed-trigram suffix matcher over token history (n-gram/lookup drafting) |
| src/spec/controller_test.cpp | 74 | unit test (tests-bench-docs.md) |
| src/spec/draft_policy_test.cpp | 119 | unit test (tests-bench-docs.md) |
| src/spec/suffix_drafter_test.cpp | 158 | unit test (tests-bench-docs.md) |
| src/ngram/ple_reader.cpp | 548 | `PleReader`: issue/collect async 4 KiB SSD reads of n-gram table rows + row cache |
| src/ngram/ple_reader_test.cpp | 514 | test |
| src/artifact/dequant.cpp | 123 | `strata-dequant` self-test CLI (only built with STRATA_DEQUANT_SELFTEST) |
| src/artifact/gguf_reader.cpp | 67 | GGUF dump CLI, `--check` |
| src/artifact/gguf_reader_test.cpp | 129 | test |
| src/platform/direct_file.cpp | 371 | `DirectFile`: unbuffered async reads (Win IOCP branch src/platform/direct_file.cpp:70-261; POSIX branch src/platform/direct_file.cpp:262-369), STRATA_IO_THREADS |
| src/platform/memory.cpp | 260 | lock_resident(_ranges), GPU shared-memory budget (Win DXGI), read-ahead/madvise, proc_io_sample |
| src/platform/memory_test.cpp | 96 | test |

## 3. Where it does X

- **Layer type (attn vs linear-attn).** include/strata/core/geometry.hpp:75 `is_qsa_layer`: `layer % qsa_interval == qsa_interval-1`. The default interval is 4 (include/strata/core/geometry.hpp:26), so layers 3,7,... use full attention and the rest use GDN. For Flash-Next, src/core/layout.cpp:153-161 checks the split is 36 GDN / 12 QSA. q36 sets `y.attn = is_mtp || is_qsa_layer(g,l)` at src/qwen35/q36_engine.cu:931, which means the MTP block is always an attention layer. src/qwen35/q36ref.cu:11-19 documents qwen35moe as GDN at l%4!=3 and attention at l%4==3, with 40 layers.
- **Per-layer forward order (q36).** `Engine::forward` at src/qwen35/q36_engine.cu:1511: iq_embed_rows (gpu-cpu-kernels.md), rms(attn_norm), then `layer()` per layer, then `head_rows` for the logits plus a greedy pick. `layer()` at src/qwen35/q36_engine.cu:1338 does mixer(h) -> `k_add_rms` (x+=y; h=rms(x)*post_norm) -> router -> experts -> `k_moe_out`. `k_moe_out` adds the MoE output and fuses the next layer's attn_norm rms ("later layers: from k_moe_out", src/qwen35/q36_engine.cu:1521).
- **GDN mixer**:
  - src/qwen35/q36_engine.cu:1345-1375. qkv and z use MMVQ/MMQ `apply`. alpha/beta use `k_gdn_ab` (T<=kGemmT) or f32mm + `k_gdn_gb`.
  - Then `k_conv_seq` runs.
  - Prompt chunks use `k_gdn_seq_cols` + `k_gdn_norm`; decode, verify and commit use one `k_gdn_seq`, so verify rows bit-match decode rows.
  - Last comes `mm(ssm_out)`.
  - The comment at src/qwen35/q36_engine.cu:1361-1365 explains why Strata's fused single-token GDN is deliberately NOT used: 1-ulp FMA differences would break MTP exactness.
- **Attention mixer.** src/qwen35/q36_engine.cu:1376-1437. wq/wk/wv -> `k_attn_prep` does q/k norm, NEOX RoPE over n_rot=64 dims, the gate split and the KV write. Prompt chunks: FP16 cuBLAS strided-batched QK^T, then `k_attn_softmax16`, then PV, then `k_gate_mul`. Decode/verify: split-K `k_attn_part` over 256-cell chunks, then the `k_attn_merge` LSE merge with the sigmoid gate. Last comes `mm(wo)`. GQA is 16 query heads over 2 KV heads.
- **MoE invocation**:
  - src/qwen35/q36_engine.cu:1442-1506. For one token, a single f32mm over `router_ext`, which is the router plus the shared-expert gate row ([n_expert+1][N]).
  - Otherwise router only.
  - Then `K::native_router_topk` (gpu-cpu-kernels.md) does top-8 of 256.
  - There are three paths.
  - Hybrid rows: `moe_hybrid_gpu` src/qwen35/q36_engine.cu:1627 / `moe_hybrid_cpu` src/qwen35/q36_engine.cu:1642.
  - Prompt chunk with MMQ: route_count/scan/scatter then `MMQ::quantize` and the MMQ gate+up / SwiGLU / down (prefill.md prefill/moe_mmq).
  - Otherwise grouped native expert blobs [gate|up|down] (`native_expert_grouped`, gpu-cpu-kernels.md).
  - The shared expert uses a sigmoid gate.
- **Expert placement / hybrid VRAM.** `place_experts` src/qwen35/q36_engine.cu:1934 fills VRAM slots from the mapped GGUF; the rest go to host blobs plus the CPU pool (STRATA_Q36_CPU_POOL, gpu-cpu-kernels.md cpu). `rebalance` src/qwen35/q36_engine.cu:1702 swaps hot experts in using per-expert route counts. `stream_begin/kick/done` src/qwen35/q36_engine.cu:1571-1625 double-buffer the hybrid layers' experts for large prompts.
- **Speculative decode in q36 (MTP)**:
  - `spec_step` src/qwen35/q36_engine.cu:1831.
  - `forward_mtp` src/qwen35/q36_engine.cu:1540 is eh_proj over [enorm(e)|hnorm(h)] -> one attention+MoE layer -> shared_head_norm -> main head.
  - It drafts D tokens chained from its own hidden state.
  - Then one verify `forward(T=D+1, verify=true)` runs from a graph `gv[T]`.
  - The accepted prefix is `a`, `commit(a+1)` replays GDN conv/state for the kept rows only (src/qwen35/q36_engine.cu:1555), and pos advances by a+1.
  - Greedy-only acceptance (exact match).
- **Spec policy (Flash-Next, used by src/program/generate.cpp)**:
  - `Controller::choose` src/spec/controller.cpp:36 compares expected tokens/ms for MTP and lookup windows.
  - The cost comes from `CostModel::step_ms` src/spec/controller.cpp:8: dense cost + CPU-miss cost scaled by distinct experts + sync + draft.
  - The priors are MTP 0.86 per draft and lookup {0.35,0.6,0.8,0.92} by match (src/spec/controller.cpp:19-24).
  - `DraftPolicy` (src/spec/draft_policy.cpp:60 choose, src/spec/draft_policy.cpp:117 chain) learns round cost by window size and buckets lookup acceptance by match length.
- **N-gram / lookup drafting.** `SuffixDrafter::propose` src/spec/suffix_drafter.cpp:64 hashes the trigram ending at the current token, scans the stored positions for the longest backward match (min 3, `max_match_`), and returns the continuation after it. `propose_after` src/spec/suffix_drafter.cpp:88 temporarily appends pending tokens. `propose_from_sources` src/spec/draft_source.cpp:13 also queries the registered `extra_draft_sources()`.
- **N-gram/PLE table from SSD**:
  - `PleReader::issue` src/ngram/ple_reader.cpp:422 dedups, sorts and submits page reads through `platform::DirectFile`.
  - `collect` src/ngram/ple_reader.cpp:485 waits for them.
  - The header (include/strata/ngram/ple_reader.hpp:3-4) says the table has 320,001,536 rows of 90 B (IQ4_NL), about 26.8 GiB, and needs 16 rows per token.
  - Reads are issued at token-id time and hidden behind embedding + layer 0.
  - The bounded clock-eviction row cache is "NOT THE TABLE".
  - Consumer: src/kernels/ngram.cpp:211 (gpu-cpu-kernels.md).
- **Platform I/O.** `DirectFile::submit/wait` src/platform/direct_file.cpp:206, src/platform/direct_file.cpp:220 (Win) and src/platform/direct_file.cpp:336, src/platform/direct_file.cpp:355 (POSIX); `io_threads` src/platform/direct_file.cpp:63. src/platform/memory.cpp `lock_resident` src/platform/memory.cpp:23 (Win) / src/platform/memory.cpp:164 (POSIX), `gpu_shared_memory_budget` src/platform/memory.cpp:112 (Win DXGI; the POSIX stub is at src/platform/memory.cpp:190), `read_ahead_enabled` src/platform/memory.cpp:205. Consumers: src/core/expert_source.cpp:530, src/core/pinned.cu:368, src/core/pinned.cu:492 (engine-core.md).
- **Serve protocol (q36).** `serve` src/qwen35/q36_engine.cu:2197 prints `READY <ctx> stop` and runs a stdin reader thread (STOP flag). It speaks serve/server.py's READY/GEN/T/DONE/STOP/PP protocol (server.md).

## 4. Types, env vars, CLI flags

**Types:**

| Name | Where / notes |
|---|---|
| `Mat` | src/qwen35/q36_engine.cu:726 |
| `Layer` | src/qwen35/q36_engine.cu:732 (attn flag, GDN/attn/MoE weights, blobs/stride/hblobs/n_slots) |
| `Model` | src/qwen35/q36_engine.cu:757 (has_mtp, top_k=8, eps 1e-6, rope_base 1e7, n_rot 64) |
| `Engine` | src/qwen35/q36_engine.cu:1035 |
| `Sampling` | src/qwen35/q36_engine.cu:2134 |
| `CostModel` | include/strata/spec/controller.hpp:25 |
| `Controller` | include/strata/spec/controller.hpp:49 |
| `DraftPolicy` | include/strata/spec/draft_policy.hpp:22 |
| `SuffixDrafter` | include/strata/spec/suffix_drafter.hpp:19 |
| `DraftSource` | include/strata/spec/draft_source.hpp:21 |
| `PleReader` / `ReaderStats` | include/strata/ngram/ple_reader.hpp:46; include/strata/ngram/ple_reader.hpp:30 |
| `platform::DirectFile` / `Completion` / `LockResult` | include/strata/platform/direct_file.hpp:23; include/strata/platform/direct_file.hpp:17; include/strata/platform/memory.hpp:16 |

**Env vars:**

| Name | Where / notes |
|---|---|
| `STRATA_Q36_NO_MMQ` | src/qwen35/q36_engine.cu:1168 |
| `STRATA_Q36_BLAS_F32` | src/qwen35/q36_engine.cu:1288 |
| `STRATA_Q36_STREAM_ALL` | src/qwen35/q36_engine.cu:1583 |
| `STRATA_Q36_FIXED_SLOTS` | src/qwen35/q36_engine.cu:1703 |
| `STRATA_Q36_NO_MTP_CPU` | src/qwen35/q36_engine.cu:1961; src/qwen35/q36_engine.cu:1986 |
| `STRATA_Q36_KEEP_PACKED` | src/qwen35/q36_engine.cu:1964 |
| `STRATA_Q36_PLACE_DEBUG` | src/qwen35/q36_engine.cu:2439 |
| `CUDA_MODULE_LOADING` | forced to EAGER (setenv, no overwrite) at src/qwen35/q36_engine.cu:2403-2410 |
| `STRATA_Q8_PACKED` | non-Windows only; forced to 1 at src/qwen35/q36_engine.cu:2403-2410 |
| `STRATA_MMQ_KQ` | forced to 1 at src/qwen35/q36_engine.cu:2403-2410 |

**Platform:**

| Name | Where / notes |
|---|---|
| `STRATA_IO_THREADS` | src/platform/direct_file.cpp:64 |
| `STRATA_READ_AHEAD` | src/platform/memory.cpp:207 |

**Compile flags:**

| Name | Where / notes |
|---|---|
| `STRATA_Q36_CPU_POOL` | CMakeLists.txt:640 |
| `STRATA_DEQUANT_SELFTEST` | CMakeLists.txt:246 |

**CLI (`run` src/qwen35/q36_engine.cu:2346-2400):**

| Name | Where / notes |
|---|---|
| `--model/--native` | src/qwen35/q36_engine.cu:2346-2400 |
| `--max-context/--ctx` | src/qwen35/q36_engine.cu:2346-2400; 32768 |
| `--chunk/--prefill` | src/qwen35/q36_engine.cu:2346-2400; auto=0 |
| `--serve` | src/qwen35/q36_engine.cu:2346-2400 |
| `--tokens` | src/qwen35/q36_engine.cu:2346-2400 |
| `--n/--max-new` | src/qwen35/q36_engine.cu:2346-2400; 64 |
| `--force` | src/qwen35/q36_engine.cu:2346-2400 |
| `--dump-full-logits` | src/qwen35/q36_engine.cu:2346-2400 |
| `--bench-pp` | src/qwen35/q36_engine.cu:2346-2400 |
| `--bench-tg` | src/qwen35/q36_engine.cu:2346-2400 |
| `--no-graph/--no-token-graph` | src/qwen35/q36_engine.cu:2346-2400 |
| `--no-warmup` | src/qwen35/q36_engine.cu:2346-2400 |
| `--mtp` | src/qwen35/q36_engine.cu:2346-2400; N (numeric only; a dir arg is ignored) |
| `--expert-vram` | src/qwen35/q36_engine.cu:2346-2400; GiB |
| `--expert-slots` | src/qwen35/q36_engine.cu:2346-2400 |
| `--cpu-threads/--pool-workers` | src/qwen35/q36_engine.cu:2346-2400 |
| `--hybrid-rows` | src/qwen35/q36_engine.cu:2346-2400; 128 |
| `--gpu` | src/qwen35/q36_engine.cu:2346-2400; ignored here |
| `--greedy` | src/qwen35/q36_engine.cu:2346-2400; no-op |

In --serve, unknown `--x` flags are skipped with a warning.

## 5. Gotchas / constants / surprises

- qwen35/ targets Qwen3.6-35B-A3B GGUF (Q8_0/K-quants), **not NVFP4 Flash-Next**. None of it reads NVFP4.
- It is a CUDA source (`cublas_v2.h`, `cuda_runtime.h`). HIP builds force-include include/strata/hip_compat/cuda_runtime.h (cmake/hip_backend.cmake:100-112).
- Hardcoded EOS ids 248044, 248046 (`is_eos` src/qwen35/q36_engine.cu:2194). Attention chunk = 256 cells (kChunkCells). MoE is 256 experts / top-8. GQA is 16/2, head_dim 256, and the RoPE base is 1e7 (src/qwen35/q36ref.cu:17-18).
- The MTP block is always loaded if present, even without --mtp, so VRAM/slot tables and greedy text don't depend on the flag (src/qwen35/q36_engine.cu:924-927).
- The spec controller priors come from another machine. src/spec/draft_policy.cpp:9-11 cites RTX 5070 curves, and the MTP 0.86 acceptance is "flyweight's measurement" (src/spec/controller.cpp:21-23).
- `kStaleRounds = 300` re-probes old window sizes. The shape table `kShape` covers up to kMaxT=8.
- The first prompt after load runs at about half speed, so `warmup()` src/qwen35/q36_engine.cu:1895 runs a throwaway chunk + decode + spec step.
- On POSIX, `gpu_shared_memory_budget` is a stub (src/platform/memory.cpp:190). The DXGI budget is Windows-only.
- There are no hyper-connections and no PLE layer math in this area. Hyper-connection/GR is in engine-core.md src/core/layer.cpp:1388-1449. "PLE" here only means the SSD n-gram table reader.
