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
| `native_mmvq_multi_kernel<Q5KTraits>` | `src/kernels/cuda/native_mmvq.cu:1173` | 16,380 | 1,720 | ~100 | the Q5_K dense weights |
| `native_down_multi_kernel<146>` | `src/kernels/cuda/iq_kernels.cu:1967` | 7,200 | 1,445.8 | 200.8 | the TQK7 down expert rows (146 = `GGML_TYPE_TQK7`) |
| `native_q6_k_mmvq_kernel` + `native_q5_k_mmvq_kernel` + `native_mmvq_multi_kernel<Q6KTraits>` | `native_mmvq.cu` | 2,462 | 764 | - | the Q6_K/Q5_K dense weights, the head, the 248,320-row embedding |
| `gr_up_multi_kernel` + `gr_down_staged_kernel` | `src/kernels/cuda/` | 12,460 | 1,010 | - | the GDN hyper-connection projections |
| `fetch_blobs_kernel` | `src/kernels/cuda/` | 3,600 | 250.7 | 69.6 | the expert fetch the doorbell waits for |
| `gpu_stamp_kernel` | `src/kernels/cuda/verify_kernels.cu` | **78,450** | 140.6 | 1.8 | the doorbell's bookkeeping: 390 launches per token |
| `attn_chunk_kernel_pre75` | `src/kernels/cuda/qsa_decode_attn.cu:545` | 1,089 | 88.3 | 81.1 | the QSA attention, the pre-sm_75 variant |
| `attn_merge_kernel` | `src/kernels/cuda/qsa_decode_attn.cu` | 1,089 | 19.8 | 18.2 | the QSA attention's merge |
| `block_scores_multi_kernel` | `src/kernels/cuda/qsa_select.cu:1032` | 900 | 8.6 | 9.5 | the QSA block scores (the router's second stage) |
| `kv_append_q8_kernel` | `src/kernels/cuda/kv_q8.cu` | 1,090 | 7.2 | 6.6 | the K/V append (the int8 K/V) |
| `append<false>` | `src/kernels/cuda/native_qsa_indexer.cu` | 900 | 5.1 | 5.7 | the QSA indexer's append |
| `sampler_greedy_kernel` + `mtp_select_kernel` | `src/kernels/cuda/` | 453 | 18.5 | 41 | the MTP draft head's own GPU work (the sampler and the draft's selection): 0.03 % of the window - the draft costs the *host* 7.2 ms per window, not the GPU |
| `hadamard_rows_kernel<float>` | `src/kernels/cuda/` | 10,800 | 46.6 | 4.3 | the Hadamard rotation of the trellis experts |
| `bf16_f32_mmvf_multi_kernel`, `gdn_step_*`, `route_multi`, `copy_rows_from_mapped` | `src/kernels/cuda/` | ~30,000 | 800 | - | the router, the GDN steps, the copies |

### The kernels of the prompt window (27 s)

| kernel | launches | total ms | what it is |
| --- | --- | --- | --- |
| `mul_mat_q<(ggml_type)145>` and `<146>` | 2,266 | 5,530 | the trellis prompt kernels (the MMQ instances from `third_party/ggml-tq/`) |
| `maxwell_sgemm_fp16_32x128_tn`, `_64x64_tn`, `maxwell_sgemm_128x64_tn`, `sgemm_128x128x8_NT_vec` | 556 | 1,934 | cuBLAS, the dense projections' GEMMs |
| `dequant_kernel<13,H16>`, `gdn_rec_cols_pipe_kernel`, `copy16_group_kernel`, the rest | ~15,000 | 2,300 | the dequantization, the GDN recurrence, the group copies |

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
| `STRATA_DENSE_F32=1` (the dense projections' GEMMs as `cublasSgemm` on FP32 copies) | cuBLAS's FP16-operand kernels, which are the slow Maxwell ones | `gemm_f32_parity` on this card: 5,201 chunk sizes **bitwise**; the route is 1.18-1.65x faster for the big shapes (qkv 10,240x2,560 at T 7,040: 166.5 -> 120.0 ms; query 12,288x2,560 at T 7,041: 203.4 -> 149.8; gate 6,144x2,560: 101.0 -> 67.8; out 2,560x6,144: 127.5 -> 77.1) and neutral from T 560 to 2,048 (0.97-1.06x); through the server the prompt read goes from 91.5 to 95.7 tok/s (the medians of three repeats; the two base runs agree to 0.2 % on that number) | **yes** (opt-in, see below) |

The dense-GEMM route was the only change that measured faster, and it is worth about 0.5 s of a 27 s prompt
read on this card: the prompt's cuBLAS time (1.93 s of the 9.8 s of compute) falls by about a third. It is
opt-in, as on the P40, because the equality belongs to the cuBLAS build, not to the code.

The change that makes it possible on this card is one line in `src/prefill/gemm.cu`: the route was gated to
compute capability 6.x, and it now also accepts 5.x (the cards whose cuBLAS has no tensor-core path for this
call). The default path is unchanged - the switch is still `STRATA_DENSE_F32=1` - and the build is clean
(`cmake --build build-cuda12`). The answers are the same: `gemm_f32_parity` says every chunk size is bitwise,
and the end-to-end check (`parity-check.py`, the same prompt through the server with and without the switch,
greedy) says the same.

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

## 4. What the profile says to do next

1. **The card is too small for this model.** Gyro-S's experts are 18 GiB; the card has 12 GiB of VRAM and the
   PC 15.7 GiB of RAM. The engine's own start-up says so: the 8,192-token prompt chunk lends 3,391 slots to
   the prompt path and **0 of them fit in RAM**, and only 327-559 MiB of VRAM are free at the start of a
   request (the engine warns: "when the card is this full the driver can page GPU memory to system RAM,
   decode falls to a few tok/s"). A card with 24 GB, or a PC with 32 GiB of RAM, would put the decode on the
   compute path, and then the kernel work in section 2 is what matters.
2. **The kernels to work on, when the data path is fixed** (in the order the profile gives):
   the trellis gate/up and down expert kernels (3.3 s of the decode's 9.0 s of compute, integer-bound, and
   the state extraction is the cost - not the dot products), the dense MMVQ (1.7 s, the same dp4a story), and
   the dense projections' GEMMs (1.9 s of the prompt's compute, where the FP32 route is now available).
3. **The MTP draft head's VRAM** (802 MiB) and the expert cache (6.02 GiB) compete for the last of the VRAM;
   the draft acceptance is 66-85 % here, so the head is earning its keep, but the trade is worth measuring on
   a card where the VRAM is not the limit.

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
