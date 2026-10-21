# Gyro-S on a GeForce GTX TITAN X (Maxwell, sm_52): a GPU profile and what it says

Every number here was measured on this machine, in this folder, with the tools named next to it.

| | |
| --- | --- |
| Card | NVIDIA GeForce GTX TITAN X, 12,288 MiB, 24 SMs, compute capability 5.2, driver 580.178.04 |
| CPU / RAM | 12 logical CPUs (6 physical), 15.7 GiB RAM, SK hynix SC311 SATA SSD |
| Toolkits | CUDA 12.9 at `/opt/cuda` (host compiler `g++-14`), Nsight Systems 2025.2.1; Nsight Compute 2025.2.1 is installed but its counters are refused on this card without root (`ERR_NVGPUCTRPERM`) |
| Engine | 0.1.46.1, `build-cuda12` (`-DCMAKE_CUDA_ARCHITECTURES=50 -DSTRATA_EXPERIMENTAL_SM60=ON`), CUDA 12.9 |
| Model | Qwen3.8-Flash-Next-Gyro-S (TQ1_0; the experts are TQK6 gate/up and TQK7 down, in a Hadamard-rotated basis) |
| Config | the repo's `strata-gyro-s.json`: MTP on, `--spec 4`, `--expert-cache auto` (6,160 slots, 6.02 GiB of VRAM), `--mmap-experts`, `--resident-budget-gib 6`, `--prefill auto` (an 8,192-token chunk), `--max-context 8192`, `--kv int8` |

The tools:

* `tools/gyro_ab.py` - the A/B harness: it starts the engine with a config, evicts the page cache, warms the
  caches up with a full request, sends a fixed prompt a fixed number of times, and writes every number the
  engine reported (the per-window phase split included) into `results.json`. The page cache is evicted by reading
  a 16 GB file through it: the harness writes that file once to `/home/mo/.cache/strata-page-cache-evict.bin`
  (16 GB on the disk, not in RAM - a tmpfs file would be the very thing that fills the memory) and reads it back
  before each run. Delete it when the measurements are done; the harness writes it again when it is missing.
* `tools/nsys_kernels.py` - reads the SQLite that `nsys stats` writes and turns a window of the trace into the
  per-kernel table, the copy table and the GPU's busy and idle time.
* `nsys-tools/strata.nsys-rep` - the trace: one real request through the server under Nsight Systems, with
  `--cuda-graph-trace=node` so the kernels inside the engine's CUDA graphs are named. The file itself is not in git
  (the repo's rule keeps profiler captures out: `*.nsys-rep` in `.gitignore`). `nsys-tools/strata-nsys.sh` runs the
  same profile again, and `nsys stats --report cuda_gpu_kern_sum` writes the SQLite the tables are derived from
  (`tools/nsys_kernels.py` reads it).
* `nsys-tools/decode-window.md`, `nsys-tools/prompt-window.md` - the two windows of that trace.
* `maxwell-instr-rates.cu` - this card's instruction rates, measured (see below).
* `CANDIDATES.md` - the ranked candidate list with the evidence for each entry.

## 1. The profile: what the GPU did

Two windows from the same trace, the model load excluded (the trace is 105.3 s; the request is the last 80 s of
it: a 1,024-token prompt read, then 201 tokens decoded).

| window | length | GPU busy | GPU idle | real compute | PCIe copies |
| --- | --- | --- | --- | --- | --- |
| prompt (1,024 tokens read) | 27.0 s | 9.77 s (36 %) | 17.2 s (64 %) | 9.8 s | 0.3 s |
| decode (201 tokens out) | 53.0 s | 49.2 s (93 %) | 3.8 s (7 %) | **9.0 s (17 %)** | 0.29 s |

The decode's "GPU busy 93 %" is not compute: 40.8 s of it is one kernel, `wait_flag_ge_kernel` (10,800 launches,
one thread each, 3.8 ms on average) - the doorbell spin while the host fetches the experts the card does not
hold. The engine's own timers (`STRATA_DECODE_TIMING=1`) say the same from the other side: of a 505.7 ms
window, **waitCPU 405.2 ms**, the VRAM expert hits 38.8 ms, the GPU's own layer work about 50 ms, the draft
7.2 ms.

So on this card the decode is not compute-bound and not PCIe-bound: it waits for the expert data to come off
the SSD. The drive delivers 0.5 GB/s (the engine's own probe) and the decode needs 20.8 GB of expert bytes
for 201 tokens - 104 MB per token - so it spends 41.2 s of its 53 s in `decode waited ... in 5787 fetches`
(7.1 ms per fetched blob). The copies themselves are nothing: 2.47 GB H2D in 210 ms, 0.4 % of the window.

### The kernels of the decode window (53 s, the whole GPU time)

| kernel | source | launches | total ms | avg us | what it is |
| --- | --- | --- | --- | --- | --- |
| `wait_flag_ge_kernel` | `src/kernels/cuda/verify_kernels.cu:1169` | 10,800 | 40,774 | 3,775 | the doorbell: the GPU's one thread waits for the host |
| `native_gu_multi_kernel<145>` | `src/kernels/cuda/iq_kernels.cu:1804` | 7,200 | 1,849.5 | 256.9 | the TQK6 gate/up expert rows (145 = `GGML_TYPE_TQK6`) |
| `native_mmvq_multi_kernel<Q5KTraits>` | `src/kernels/cuda/native_mmvq.cu:1173` | 16,380 | 1,720 | ~105 | the Q5_K dense weights (the projections) |
| `native_down_multi_kernel<146>` | `src/kernels/cuda/iq_kernels.cu:1967` | 7,200 | 1,445.8 | 200.8 | the TQK7 down expert rows (146 = `GGML_TYPE_TQK7`) |
| `gr_up_multi_kernel` | `src/kernels/cuda/fused_gr.cu:367` | 7,843 | 567.3 | 72.3 | the GDN hyper-connection's up projection (REG 40, 6.2 KB of shared memory) |
| `gr_down_staged_kernel` | `src/kernels/cuda/fused_gr.cu:963` | 7,455 | 419.4 | 56.2 | the GDN hyper-connection's down projection, the staged one (REG 83, **40 KB** of shared memory per block) |
| `native_mmvq_multi_kernel<Q6KTraits>` | `src/kernels/cuda/native_mmvq.cu:1173` | 2,325 | 334.1 | 143.7 | the Q6_K dense weights |
| `native_q6_k_mmvq_kernel` | `src/kernels/cuda/native_mmvq.cu:674` | 199 | 299.9 | 1,507.2 | the token embedding: 248,320 rows of Q6_K, one call per token |
| `fetch_blobs_kernel` | `src/kernels/cuda/verify_kernels.cu:591` | 3,600 | 250.7 | 69.6 | the expert fetch the doorbell waits for |
| `bf16_f32_mmvf_multi_kernel` | `src/kernels/cuda/native_bf16.cu:78` | 6,835 | 150.1 | 22.0 | the BF16 dense weights, dequantized and multiplied (no tensor cores here: the FP32 path) |
| `gpu_stamp_kernel` | `src/kernels/cuda/verify_kernels.cu:1527` | **78,450** | 140.6 | 1.8 | the doorbell's bookkeeping: 390 launches per token |
| `native_q5_k_mmvq_kernel` | `src/kernels/cuda/native_mmvq.cu:255` | 2,520 | 131.2 | 52.0 | the head: 10,240 rows of Q5_K |
| `gr_norm_split_kernel` | `src/kernels/cuda/fused_gr.cu:813` | 7,843 | 78.9 | 10.1 | the GDN's norm |
| `attn_chunk_kernel_pre75` | `src/kernels/cuda/qsa_decode_attn.cu:545` | 1,089 | 88.3 | 81.1 | the QSA attention, the pre-sm_75 variant |
| `gdn_step_norm_multi_kernel` | `src/kernels/cuda/verify_kernels.cu:286` | 2,340 | 83.6 | 35.7 | the GDN step's norm |
| `gdn_step_commit_layers_kernel` | `src/kernels/cuda/verify_kernels.cu:198` | 65 | 67.0 | 1,030.3 | the GDN step's commit (one call per window) |
| `route_multi` | `src/kernels/cuda/native_router.cu:100` | 3,120 | 65.0 | 20.8 | the router: the 512-expert top-10 per layer |
| `hadamard_rows_kernel<float>` | `src/kernels/cuda/iq_kernels.cu:1904` | 10,800 | 46.6 | 4.3 | the Hadamard rotation of the trellis experts |
| `copy_rows_from_mapped_kernel` | `src/kernels/cuda/elementwise.cu:238` | 3,600 | 41.9 | 11.6 | the copies of the expert rows into the device buffers |
| `native_quantize_q8_1_kernel` | `src/kernels/cuda/native_mmvq.cu:156` | 12,112 | 38.9 | 3.2 | the activations quantized to q8_1 for the experts |
| `quantize_q8_1_kernel` | `src/kernels/cuda/iq_kernels.cu:2510` | 10,800 | 36.3 | 3.4 | the same, the other path |
| `doorbell_publish_res_kernel` | `src/kernels/cuda/elementwise.cu:329` | 3,600 | 34.1 | 9.5 | the doorbell's result publication |
| `swiglu_entries_kernel` | `src/kernels/cuda/iq_kernels.cu:1924` | 7,200 | 24.3 | 3.4 | the SwiGLU of the expert rows |
| `attn_merge_kernel` | `src/kernels/cuda/qsa_decode_attn.cu:699` | 1,089 | 19.8 | 18.2 | the QSA attention's merge |
| `gdn_ab_multi_kernel` | `src/kernels/cuda/verify_kernels.cu:145` | 1,368 | 21.6 | 15.8 | the GDN's a/b step |
| `sampler_greedy_kernel` | `src/kernels/cuda/sampler.cu:119` | 264 | 16.3 | 61.7 | the sampling of the tokens and the MTP drafts |
| `block_scores_multi_kernel` | `src/kernels/cuda/qsa_select.cu:1032` | 900 | 8.6 | 9.5 | the QSA block scores (the router's second stage) |
| `kv_append_q8_kernel` | `src/kernels/cuda/kv_q8.cu:33` | 1,090 | 7.2 | 6.6 | the K/V append (the int8 K/V) |
| `append<false>` | `src/kernels/cuda/native_qsa_indexer.cu:120` | 900 | 5.1 | 5.7 | the QSA indexer's append |
| `mtp_select_kernel` | `src/kernels/cuda/verify_kernels.cu:659` | 189 | 2.2 | 11.6 | the MTP draft head's selection |

The MTP draft head's own GPU work is the sampler and the selection together: 18.5 ms of the 53 s window (0.03 %).
The draft costs the *host* 7.2 ms per window, not the GPU.

### What Maxwell does not have, and where it bites

Four things, all measured on this card (the SASS with `cuobjdump`, the instruction rates with
`maxwell-instr-rates.cu`):

1. **No `__dp4a`** (it arrived with compute capability 6.1). The engine's four-`vmad` fallback
   (`include/strata/kernels/dp4a.hpp`) stands in, and `vdp4a` does not exist here either (`ptxas`: "Not a name
   of any known instruction"). The binary has **zero** `IDP4A`. The rates: VMAD 268 instructions per SM per
   clock, IMAD 493, FFMA 806 - so the integer pipes are a third to two thirds of the FP32 rate, and the trellis
   expert kernels (which are integer) run at 18 GB/s against the card's ~336 GB/s of memory bandwidth. The dot
   products are 18 % of their instructions; the trellis *state* extraction is the rest.
2. **No `cp.async`** (it arrived with compute capability 8.0). The binary has **zero** `CP.ASYNC` and **zero**
   `LDGSTS`: the staged hyper-connection read compiles its plain-copy path on this card
   (`src/kernels/cuda/fused_gr.cu:912` gates the async copy on `__CUDA_ARCH__ >= 800`). The engine's start-up
   message does not know that - it says "the down projection's activations staged ahead by cp.async" on a card
   that has none (`src/kernels/cuda/fused_gr.cu:2578`).
3. **No tensor cores.** The dense projections' FP16-operand GEMMs have no tensor-core path here, and cuBLAS's
   Maxwell kernels for them are slow - which is what the FP32 route in this pull request is for.
4. **48 KB of shared memory per block** (96 KB per SM, the sm_50 limit). The GDN hyper-connection's staged down
   projection takes **40 KB** of it per block (30 KB in its other variant), so an sm_50 SM holds two of them: the
   registers (83 per thread) would allow three. The shared memory is what caps that kernel family, and it is the
   one hot family where the limit bites.

### The kernels of the prompt window (27 s)

| kernel | source | launches | total ms | what it is |
| --- | --- | --- | --- | --- |
| `mul_mat_q<(ggml_type)145>` and `<146>` | the MMQ instances of `third_party/ggml-tq/` (the trellis prompt kernels, via the overlay in `cmake/strata_ggml_tq_overlay.cmake`) | 2,266 | 5,530 | the trellis prompt path: the experts' rows for the whole chunk |
| `maxwell_sgemm_fp16_32x128_tn`, `_64x64_tn`, `maxwell_sgemm_128x64_tn`, `sgemm_128x128x8_NT_vec` | cuBLAS | 556 | 1,934 | the dense projections' GEMMs (the FP16-operand ones are the slow Maxwell kernels) |
| `dequant_kernel<13,H16>` | `src/kernels/cuda/dequant_bf16.cu:176` | 252 | 236.0 | the BF16 weights' dequantization |
| `gdn_rec_cols_pipe_kernel` | `src/prefill/gdn_rec_parity.cu:57` (the same code the engine builds) | 36 | 164.6 | the GDN recurrence, in shared memory |
| `copy16_group_kernel` | `src/prefill/moe_mmq.cu:83` | 4,669 | 150.6 | the group copies of the MMQ path |
| the rest (the GDN's gates, the norms, the attention, the copies) | `src/prefill/`, `src/kernels/cuda/` | ~12,000 | 1,460 | - |


## 2. Why Maxwell is slow here, from the SASS and the instruction rates

`cuobjdump -sass build-cuda12/strata`:

* `native_gu_multi_kernel<145,false,false>`: REG 64, **STACK 64 bytes per thread**, 1,476 instructions: 160
  `VMAD`, 245 `XMAD` (IMAD), 240 `IADD`, 93 `LDG`, 14 `F2F`, **no `IDP4A`**.
* `native_down_multi_kernel<146,true>`: REG 64, SHARED 2,880 B, 1,848 instructions: 176 `VMAD`, 312 `XMAD`,
  97 `LDG`, **no `IDP4A`**.

`maxwell-instr-rates.cu` on this card (24 SMs at 1.08 GHz), instructions per SM per clock:

| instruction | rate | note |
| --- | --- | --- |
| FFMA (FP32) | 806 | the card's full-rate pipe |
| F2F (F16 -> F32) | 660 | the trellis codebook's conversion |
| IMAD (INT32) | 493 | 0.61 of the FP32 rate: Maxwell has 64 INT32 per SM against 128 FP32 |
| VMAD (the video pipe) | 268 | 0.33 of the FP32 rate |
| the 4-`vmad` dp4a fallback | 1,073 per group | 4 VMADs per 4-byte dot product |
| the byte-wise dp4a | 482 | 4 byte extractions and 4 IMADs |

Three facts follow, and all three are measurements, not theory:

1. **No `__dp4a` on this card** (it arrived with compute capability 6.1). The engine's fallback
   (`include/strata/kernels/dp4a.hpp`) does the byte dot product with four `vmad` instructions.
2. **No `vdp4a` either**: `ptxas` refuses it for sm_52 ("Not a name of any known instruction"), so the
   four-`vmad` form is the only exact one here. Measured against the alternative: the byte-wise form takes
   128.3 ms where the four-`vmad` form takes 54.0 ms (`tools/sm60_dp4a_check.cu` on this card: 1.07e9 cases,
   0 mismatches, **2.37x**). The fallback is the right choice on Maxwell as it is on Pascal.
3. **The trellis expert kernel is integer-bound, and far from the memory roof.** `native_expert_bench` on
   the Gyro experts: the gate/up rows run at **18 GB/s** against the card's ~336 GB/s of memory bandwidth, and
   the SASS is integer (160 `VMAD`, 245 `IMAD`, 240 `IADD` against 10 `FFMA`). The dot products are 18 % of
   the instructions; the trellis *state* extraction (the shifts, the masks, the adds) is the rest. So a faster
   dot product would not make this kernel much faster: the state machine is the cost.

## 3. What was tried, and what the measurements said

The harness: `tools/gyro_ab.py`, three repeats per variant, the page cache evicted before the engine starts
(a 16 GB file read through it, no root needed), one warmup request of the same fixture, the medians of the
repeats reported cold (the first) and warm (the rest). The two base runs agree on the prompt read to 0.2 %
(89.95 and 89.75 tok/s) and to 11 % on the decode (8.35 and 9.0 tok/s), so a change below about 15 % on the
decode is not a result.

| candidate | what it changes | result | kept? |
| --- | --- | --- | --- |
| `STRATA_IO_PREFETCH=0` (the file tier's read-ahead off) | the drive reads only what is needed | the prompt read 89.95 -> 46.25 tok/s (-49 %), the decode 10.05 -> 4.25 tok/s (-58 %) | **no** |
| `--prefill 2048` (a smaller prompt chunk) | the prompt path borrows fewer slots | the prompt read 89.95 -> 79.05 tok/s (-12 %); the lend region still gets 0 of 1,024 slots, because the 6 GiB RAM budget is already full | **no** |
| `--resident-budget-gib 8` | 2 GiB more of the experts page-locked in RAM | the prompt read 89.95 -> 89.80 tok/s, the decode within the noise | **no** (no change) |
| `--resident-budget-gib 4` | 2 GiB less in RAM, more for the page cache | the prompt read 89.95 -> 93.80 tok/s (+4.3 %) - inside the noise | **no** (inside the noise) |
| `STRATA_ADAPT_OVERLAP_RESIDENT=1` | the tier's swaps land between verify windows | the prompt read 89.95 -> 91.85 tok/s (+2.1 %) - inside the noise | **no** (inside the noise) |
| `STRATA_EXP_MODE=2/4/5/6/7/8` (the other expert layouts) | the experts' kernel layout | every mode is slower than the default on this card: mode 2 1,138 -> 1,255 us, mode 4 -> 1,340, mode 5 -> 1,364, mode 6 -> 1,373, mode 7 -> 1,387, mode 8 -> 1,408 (all bitwise equal to mode 0) | **no** |
| a faster byte dot product for sm_50 | the 4-`vmad` fallback | the byte-wise form is 2.37x slower here; `vdp4a` does not exist on sm_52 | **no** (the fallback is already the best) |
| `STRATA_GR_FAST=1` (the AMD-style hyper-connection kernels) | the GDN's read | `fused_gr_bench` on this card: at the decode's window size (1-3 tokens) the old one is faster - 116.3 vs 122.3 us at 1 token, 133.0 vs 141.4 at 2, 162.4 vs 162.4 at 3; the fast one only wins from 5 tokens on (262.7 -> 246.7) | **no** (the decode's windows are 2-3 tokens) |
| `STRATA_SM50_TABLE=1` (the interleaved small-batch matvec, a table for Maxwell) | the dense weights' MMVQ: a warp takes 2 or 4 rows, so the weights and the activations are reused across rows | the GPU's own stage timers (`STRATA_VERIFY_PROFILE=1`), the medians of 3 repeats: the dense-projection stage 8.92 -> 7.76 ms per window (**-13 %**, better in 3 of 3 repeats), the VRAM-hit stage 36.7 -> 34.5 ms (-6 %); the *request* does not move (the decode's tok/s is inside the noise), because the GPU waits 190-310 ms per window for the experts while it computes about 40 ms. `mmvq_il_parity` on this card: every case, T 1-8, rows 1/2/4 and the table, bitwise | **yes** (opt-in; the answers are the same) |
| `STRATA_DENSE_F32=1` (the dense projections' GEMMs as `cublasSgemm` on FP32 copies) | cuBLAS's FP16-operand kernels, which are the slow Maxwell ones | `gemm_f32_parity` on this card: 5,201 chunk sizes **bitwise**; the route is 1.18-1.65x faster for the big shapes (qkv 10,240x2,560 at T 7,040: 166.5 -> 120.0 ms; query 12,288x2,560 at T 7,041: 203.4 -> 149.8; gate 6,144x2,560: 101.0 -> 67.8; out 2,560x6,144: 127.5 -> 77.1) and neutral from T 560 to 2,048 (0.97-1.06x); through the server the prompt read goes from 91.5 to 95.7 tok/s (the medians of three repeats; the two base runs agree to 0.2 % on that number) | **yes** (opt-in, see below) |

The two changes that measured faster are both opt-in, and they are worth a different amount in different places:

- **`STRATA_DENSE_F32=1`** (the dense projections' FP16 GEMMs as `cublasSgemm` on FP32 copies) is the one that moves
  the *request* on this card: the prompt read goes from 91.5 to 95.7 tok/s, because the prompt's cuBLAS time
  (1.93 s of the 9.8 s of compute) falls by about a third. The change that makes it possible on this card is one
  line in `src/prefill/gemm.cu`: the route was gated to compute capability 6.x, and it now also accepts 5.x (the
  cards whose cuBLAS has no tensor-core path for this call). The default path is unchanged - the switch is still
  `STRATA_DENSE_F32=1` - and the build is clean (`cmake --build build-cuda12`).
- **`STRATA_SM50_TABLE=1`** (the interleaved small-batch matvec, a table for Maxwell) makes the *compute* faster:
  the GPU's dense-projection stage 13 % faster (better in 3 of 3 repeats) and the VRAM-hit stage 6 % faster. It does
  not move the request on this card, because the GPU is idle 80 % of the time: it waits 190-310 ms per window for
  the experts while it computes about 40 ms. On a card where the experts fit, that is where it would show. The
  change is the table and its switch in `src/kernels/cuda/native_mmvq.cu` (the repo's documented procedure:
  `mmvq_il_parity --bench --emit-table` on the card, then the table), and the interleaved kernels were compiled
  for this architecture but never enabled on it (`il_arch_ok` stopped at compute capability 6).

The answers are the same in both cases: `gemm_f32_parity` says every chunk size of the GEMM route is bitwise,
`mmvq_il_parity` says every case of the interleaved kernels is bitwise, and the end-to-end check
(`parity-check.py`, the same prompt through the server with and without the switch, greedy) says the same:
`parity-answers-dense_f32.json` and `parity-answers-sm50-table.json` hold the two answers.

The other candidates were dropped with their numbers:

* `STRATA_IO_PREFETCH=0`: the drive's read-ahead is worth 49 % of the prompt read and 58 % of the decode.
* `--prefill 2048`: the prompt's lend region gets nothing (0 of 1,024 slots), because the 6 GiB RAM budget is
  already full of the ranked experts; the prompt read is 12 % slower.
* `--resident-budget-gib 8` and `4`: no change and +4.3 % (inside the noise).
* `STRATA_ADAPT_OVERLAP_RESIDENT=1`: +2.1 % (inside the noise).
* `STRATA_EXP_MODE=2/4/5/6/7/8`: every other expert layout is slower on this card (mode 2 +10 %, mode 8 +8 %;
  mode 8 is 35 % faster on a P40).
* A faster byte dot product: the four-`vmad` fallback is already 2.37x faster than the byte-wise form here, and
  `vdp4a` does not exist on sm_52.

## 4. What the profile says to do next, in the order the profile gives it

The decode's compute, largest first: the trellis expert kernels 3.30 s, the dense weights' MMVQ 1.72 s, the GDN
hyper-connection 1.01 s, the head and the embedding 0.43 s, the attention 0.11 s, the MTP draft head 0.02 s.

1. **The trellis gate/up and down expert kernels** (3.30 s): integer-bound, with no `__dp4a` and no `vdp4a` on this
   card, and the dot products are only 18 % of their instructions - the trellis *state* extraction is the rest. So
   a faster dot product is not where the win is; the state extraction is. (Measured: the byte-wise dot product is
   2.37x slower than the four-`vmad` one here, and every other expert layout is slower too.)
2. **The dense weights' MMVQ** (1.72 s) and the head and the embedding (0.43 s): the interleaved matvec with a
   table for Maxwell is now in (`STRATA_SM50_TABLE=1`), 13 % faster on the dense-projection stage.
3. **The GDN hyper-connection** (1.01 s): the staged down projection takes 40 KB of shared memory per block, so an
   sm_50 SM holds two of them where the registers would allow three - the shared memory is the binding constraint
   on this kernel family, and the card's 48 KB per block is the limit. A smaller staging tile is the thing to try
   here. The staged read's other trick, `cp.async`, does not exist on this card at all (the binary has zero
   `CP.ASYNC`), so on Maxwell it is only the "split" variant - and the engine's start-up message should not claim
   otherwise.
4. **The doorbell's bookkeeping** (`gpu_stamp_kernel`, 78,450 launches = 390 per token): 0.14 s of GPU time, but
   the host pays for every launch, and on a card where the GPU waits anyway, fewer round trips is the same idea as
   more overlap.

**But on this card the first thing to change is not a kernel:** the model does not fit. Gyro-S's experts are
18 GiB; the card has 12 GiB of VRAM and the PC 15.7 GiB of RAM. The engine's own start-up says so: the
8,192-token prompt chunk lends 3,391 slots to the prompt path and **0 of them fit in RAM**, and only 327-559 MiB
of VRAM are free at the start of a request (the engine warns: "when the card is this full the driver can page GPU
memory to system RAM, decode falls to a few tok/s"). A card with 24 GB, or a PC with 32 GiB of RAM, would put the
decode on the compute path, and the list above would matter then. The MTP draft head's VRAM (802 MiB) and the
expert cache (6.02 GiB) compete for the last of the VRAM; the draft acceptance is 66-85 % here, so the head is
earning its keep, but the trade is worth measuring on a card where the VRAM is not the limit.

## 5. What was not measured

* No hardware counters (ncu): this card refuses them without root, so there are no occupancy, memory
  throughput or stall numbers here. The occupancy statements come from the SASS and the launch shapes.
* No other card: everything above is this one TITAN X on this PC. The P40 numbers in
  [docs/OLDER_GPUS.md](../../../docs/OLDER_GPUS.md) are a different card and a different model.
* The HIP, SYCL and Metal engines were not built or run: this is a CUDA measurement, and this PC has no ROCm
  and no Intel toolchain. The one engine change is inside a CUDA-only guard
  (`#if defined(STRATA_EXPERIMENTAL_SM60)` in `src/prefill/gemm.cu`), so the HIP and the release CUDA 13 builds do
  not compile the changed lines at all - but no build of them was made here to prove it.
* The kernel candidates that were rejected were rejected on this card, for this model, with this quant. The
  trellis state extraction and the dp4a fallback would matter differently on a card with `__dp4a`.
