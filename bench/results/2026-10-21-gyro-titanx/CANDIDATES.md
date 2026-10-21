# Gyro-S on the TITAN X: what the GPU profile says, and what to try next

Machine: NVIDIA GeForce GTX TITAN X (sm_52, 12,288 MiB, 24 SMs), driver 580.178.04, 15.7 GiB RAM,
SATA SSD (SK hynix SC311), Linux 6.2, engine 0.1.46.1 (build-cuda12, CMAKE_CUDA_ARCHITECTURES=50,
STRATA_EXPERIMENTAL_SM60=ON). Model: Qwen3.8-Flash-Next-Gyro-S (TQ1_0; experts TQK6 gate/up, TQK7 down).
Config: the repo's `strata-gyro-s.json` (MTP on, `--spec 4`, `--resident-budget-gib 6`, `--kv int8`).

Profile: Nsight Systems 2025.2.1 (`/opt/cuda/nsight_systems/bin/nsys`), `--trace=cuda,nvtx
--cuda-graph-trace=node`, one real request through the server. Trace: `nsys-tools/strata.nsys-rep`;
tables: `nsys-tools/decode-window.md`, `nsys-tools/prompt-window.md`; the tool that made them:
`tools/nsys_kernels.py`.

## 1. The answer: compute or transfer?

Two windows, both from the same trace, the model load excluded:

| window | length | GPU busy | GPU idle | real compute | PCIe copies |
|---|---|---|---|---|---|
| prompt (1,024 tokens read) | 27.0 s | 9.77 s (36 %) | 17.2 s (64 %) | 9.8 s | 0.3 s |
| decode (201 tokens out) | 53.0 s | 49.2 s (93 %) | 3.8 s (7 %) | **9.0 s (17 %)** | 0.29 s |

The decode's "GPU busy 93 %" is misleading: 40.8 s of it is ONE kernel,
`wait_flag_ge_kernel` (10,800 launches, one thread each, average 3.8 ms) - the doorbell spin. The GPU
sits in that spin while the host fetches the experts the card does not hold. The engine's own timers say
the same thing from the other side (`STRATA_DECODE_TIMING=1`): of a 505.7 ms window, **waitCPU 405.2 ms**,
VRAM hits 38.8 ms, the GPU's own layer work about 50 ms, the draft 7.2 ms.

So on this machine the answer is neither "compute" nor "PCIe transfer": it is **waiting for the expert
data to arrive from the SSD**. The drive delivers 0.5 GB/s (the engine's own probe) and the decode needs
20.8 GB of expert bytes for 201 tokens - 104 MB per token - so it spends 41.2 s of its 53 s in
`decode waited ... in 5787 fetches` (7.1 ms per fetched blob, the drive at its limit). The copies
themselves are nothing: 2.47 GB H2D in 210 ms.

The compute that does exist (9.0 s of the decode) is:

| kernel | source | launches | total ms | avg us | what it is |
|---|---|---|---|---|---|
| `native_gu_multi_kernel<145>` | `src/kernels/cuda/iq_kernels.cu:1804` | 7,200 | 1,849.5 | 256.9 | the TQK6 gate/up expert rows (145 = `GGML_TYPE_TQK6`) |
| `native_mmvq_multi_kernel<Q5KTraits>` | `src/kernels/cuda/native_mmvq.cu:1173` | 16,380 | 1,720 | ~100 | the Q5_K dense weights |
| `native_down_multi_kernel<146>` | `src/kernels/cuda/iq_kernels.cu:1967` | 7,200 | 1,445.8 | 200.8 | the TQK7 down expert rows (146 = `GGML_TYPE_TQK7`) |
| `native_mmvq_multi_kernel<Q6KTraits>` + `native_q6_k_mmvq_kernel` + `native_q5_k_mmvq_kernel` | `native_mmvq.cu` | 2,462 | 764 | - | the Q6_K/Q5_K dense weights, the head, the 248,320-row embedding |
| `gr_up_multi_kernel` / `gr_down_staged_kernel` | `src/kernels/cuda/` | 12,460 | 1,010 | - | the GDN hyper-connection projections |
| `bf16_f32_mmvf_multi_kernel`, `gdn_step_*`, `route_multi`, `fetch_blobs_kernel` | `src/kernels/cuda/` | ~30,000 | 800 | - | the router, the fetch, the GDN steps |
| `gpu_stamp_kernel` | `src/kernels/cuda/verify_kernels.cu` | **78,450** | 140.6 | 1.8 | the doorbell's bookkeeping: 390 launches per token |
| `attn_chunk_kernel_pre75` | `src/kernels/cuda/` | 1,089 | 88.3 | 81.1 | the attention, the pre-sm_75 variant |

The prompt phase (9.8 s of compute in 27 s) is: the trellis prompt kernels
`mul_mat_q<(ggml_type)145|146>` (the MMQ instances from `third_party/ggml-tq/`) **5.53 s**, cuBLAS
`maxwell_sgemm_fp16_32x128_tn` + `_64x64_tn` + `maxwell_sgemm_128x64_tn` + `sgemm_128x128x8_NT_vec`
**1.93 s**, the rest 2.3 s.

## 2. Why the Maxwell path is slow (from the SASS, not from theory)

`cuobjdump -sass build-cuda12/strata`:

* `native_gu_multi_kernel<145,false,false>`: REG 64, **STACK 64 bytes per thread**, 1,476 instructions:
  160 `VMAD`, 245 `XMAD`, 240 `IADD`, 93 `LDG`, **no `IDP4A`**.
* `native_down_multi_kernel<146,true>`: REG 64, SHARED 2,880 B, 1,848 instructions: 176 `VMAD`,
  312 `XMAD`, 97 `LDG`, no `IDP4A`.

Three facts about this card, all measured here:

1. **No `__dp4a`.** It arrived with compute capability 6.1; this card is 5.2. The engine's fallback
   (`include/strata/kernels/dp4a.hpp`) does the byte dot product with four `vmad` instructions instead
   of one `IDP4A` - 4x the integer work of the integer-heavy quant kernels, in 53 call sites
   (`STRATA_DP4A` in `iq_kernels.cu` and `native_mmvq.cu`).
2. **No `vdp4a` either.** `ptxas` refuses it for sm_52 ("Not a name of any known instruction"), so the
   four-`vmad` form is the only exact one available here (the repo's own choice). Measured, not assumed.
3. **Integer is half rate on Maxwell.** 64 INT32 per SM against 128 FP32, and no tensor cores at all, so
   the quant kernels (which are integer) run at half the rate they would on Pascal, while the dense
   FP16-operand GEMMs have no tensor-core path to fall back on.

Plus: `native_gu_multi_kernel` spills 64 bytes per thread of stack, and at 64 registers with 256-thread
blocks an sm_50 SM holds 4 blocks (32 of its 48 warps) - 67 % occupancy at best.

## 3. The candidates, ranked by the gain they can bring to a request

Measured request: 80 s (27 s prompt + 53 s decode) for 1,024 prompt tokens and 201 decoded tokens.
The A/B harness (`tools/gyro_ab.py`) measures the medians of 3 repeats with a cold page cache and a
warmup request, and the engine's own per-window phase split.

| # | candidate | class | what it changes | why sm_50 hurts | expected gain | how it is measured |
|---|---|---|---|---|---|---|
| D1 | `--prefill 2048` (a smaller prompt chunk) | data | the prompt's chunk size, so the prompt path borrows fewer expert slots and they stay in the 6 GiB RAM copy | the engine's own start-up warning: with the 8,192-token chunk 3,391 slots are lent to the prompt path and **0 of them fit in RAM**, so the prompt re-reads 12 GiB from the drive | the prompt phase 27 s -> 10-15 s (the engine's own note: up to 3x faster prompts, #669) | A/B on the prompt workload, same fixture, 3 repeats |
| D2 | `--resident-budget-gib 8` | data | 2 GiB more of the 18 GiB of experts page-locked in RAM, so fewer come from the file tier | the drive is at 0.5 GB/s and the decode needs 104 MB of experts per token | the decode's 41 s of file wait -> less, but the page cache loses 2 GiB; the trade is unknown | A/B on the decode workload, 3 repeats |
| D3 | `STRATA_IO_PREFETCH=0` | data | the file tier's read-ahead | the smoke run's prompt read 38.6 GB from the disk for 24.1 GB of expert reads (60 % overhead), and 47 % of the read-ahead blocks were never used | 0-20 % of the prompt phase | A/B on the prompt workload |
| D4 | `STRATA_ADAPT_OVERLAP_RESIDENT=1` | data | the tier's swaps land between verify windows instead of blocking one | the resident RAM mode has the overlap off by default | small; the swaps are 26,425 over a whole run | A/B on the decode workload |
| K1 | `STRATA_DENSE_F32=1` on sm_50 (`src/prefill/gemm.cu:764`, `dense_f32_on()` refuses cc < 60) | kernel | the dense projections' FP16-operand GEMMs as `cublasSgemm` on FP32 copies | cuBLAS picks `maxwell_sgemm_fp16_*` on this card (the trace shows it) and has no tensor-core path; on a P40 the FP32 route measured 8.4-9.8 against 4.3-7.6 TFLOPS | the prompt's cuBLAS 1.93 s -> ~1.1 s | `gemm_f32_parity` (bitwise check with this cuBLAS 12.9) + A/B on the prompt workload |
| K2 | a faster byte dot product for sm_50 (`include/strata/kernels/dp4a.hpp`) | kernel | the 4x `vmad` fallback that stands in for `__dp4a` | no `IDP4A`, no `vdp4a`, integer at half rate; it is in 53 call sites, so it is in the trellis gu/down (3.3 s) and the dense MMVQ (1.7 s) | 0-20 % of 5.9 s of decode compute | `native_expert_bench` and the MMVQ bench on this card, then A/B |
| K3 | no stack spill in `native_gu_multi_kernel` (STACK 64 B/thread) | kernel | the kernel's register/local use | 64 registers + a 64-byte stack frame at 256 threads | up to 10 % of the gu kernel (0.19 s) | `native_expert_bench`, then A/B |
| K4 | fewer `gpu_stamp_kernel` launches (78,450 = 390 per token) | kernel | the doorbell's per-token bookkeeping | 1.8 us of GPU time each, but each is a launch the host pays for | up to 0.5 s of host time per request | A/B, and the launch count from the trace |
| K5 | `attn_chunk_kernel_pre75` | kernel | the attention kernel for pre-sm_75 cards | the pre-75 path is a different kernel by construction | 0.09 s; not worth it | only if a bigger item falls through |

**What the profile says about the order of work.** The data-path candidates (D1-D4) are worth more than
all the kernel candidates together on this machine, because the GPU is idle 77 % of the decode. The
kernel candidates are still worth doing - the prompt phase is 36 % compute, and a card with more VRAM
(where the experts fit) would put the decode on the compute path too - but on *this* card a 2x faster
kernel set would make the request about 8 % faster, while a fixed prompt read would make it 15-20 %
faster.

## 4. What each candidate turned out to be

The harness: `tools/gyro_ab.py`, three repeats per variant, the page cache evicted before the engine starts (a
16 GB file read through it, no root needed), one warmup request of the same fixture, the medians of the repeats
reported cold (the first) and warm (the rest). The two base runs agree on the prompt read to 0.2 % (89.95 and
89.75 tok/s) and to 11 % on the decode (8.35 and 9.0 tok/s), so a change under about 15 % on the decode is not
a result.

| candidate | result | kept? |
|---|---|---|
| D1 `--prefill 2048` | the prompt read 89.95 -> 79.05 tok/s (-12 %): the lend region still gets 0 of its 1,024 slots, because the 6 GiB RAM budget is already full of the ranked experts | no |
| D2 `--resident-budget-gib 8` | the prompt read 89.95 -> 89.80 tok/s, the decode inside the noise | no (no change) |
| D3 `STRATA_IO_PREFETCH=0` | the prompt read 89.95 -> 46.25 tok/s (-49 %), the decode 10.05 -> 4.25 tok/s (-58 %): the read-ahead is worth much more than the waste it does | no |
| D4 `STRATA_ADAPT_OVERLAP_RESIDENT=1` | the prompt read 89.95 -> 91.85 tok/s (+2.1 %) - inside the noise | no (inside the noise) |
| K1 `STRATA_DENSE_F32=1` on sm_50 | `gemm_f32_parity` on this card: 5,201 chunk sizes bitwise; the route is 1.18-1.65x faster for the big shapes; the prompt read 91.5 -> 95.7 tok/s through the server, the same answers | **yes** |
| K2 a faster byte dot product for sm_50 | the byte-wise form is 2.37x slower here (128.3 ms against 54.0 ms, 0 mismatches), and `vdp4a` does not exist on sm_52: the four-`vmad` fallback is already the best exact form | no |
| K3 no stack spill in `native_gu_multi_kernel` | the mode comparison shows the kernel's layout is the bigger lever, and every other layout is slower on this card; the 64-byte stack frame is not worth a rewrite of the trellis decode on its own | no (deferred) |
| K4 fewer `gpu_stamp_kernel` launches | 78,450 launches in 53 s, 1.8 us of GPU time each (0.14 s of the window): the GPU time is not the problem, and the launches are graph nodes | no (deferred) |
| K5 `attn_chunk_kernel_pre75` | 0.09 s of a 53 s decode | no |
| new: the other expert layouts (`STRATA_EXP_MODE=2/4/5/6/7/8`) | every one is slower on this card: 1,138 -> 1,255 / 1,340 / 1,364 / 1,373 / 1,387 / 1,408 us for an 18-group call at 2 tokens (mode 8 is 35 % faster on a P40, 8 % slower here), all bitwise equal to mode 0 | no |

The measured instruction rates (`maxwell-instr-rates.cu`, instructions per SM per clock on this card): FFMA 806,
F2F 660, IMAD 493, VMAD 268. The trellis expert kernel is integer-bound, and the dot products are 18 % of its
instructions - the trellis *state* extraction is the rest. So the kernel to work on, on a card where the data
path is not the limit, is the state extraction.

## 5. What is not in this list

* No change to the model's placement policy, the router or the answers: the candidates above are
  engine settings and kernel code, and every one of them is checked for answer parity before it is kept.
* No change to the other backends: a header shared with the HIP build is out of scope unless the user
  allows building HIP too, so K2 and K3 would have to stay inside the CUDA sources.
* ncu's hardware counters are refused on this GeForce card without root (`ERR_NVGPUCTRPERM`, measured),
  so there are no occupancy or memory-throughput counters here: the occupancy statements above come from
  the SASS and the launch shapes, not from counters.
