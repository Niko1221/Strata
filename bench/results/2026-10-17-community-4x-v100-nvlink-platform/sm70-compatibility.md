# What a V100 (sm_70) build of Strata 0.1.44.1 actually contains: kernel inventory and the SM70 gate table

Date: 2026-10-17 (UTC). Method: **reading only** - the binary of the working build, its source tree, the working
service logs and the documentation. Nothing was installed, rebuilt (except a compile check), stopped, or measured on a GPU.

Object: `<HOME>/strata185/Strata` (the git SHA is unavailable - `git` is not installed on the machine),
engine `<HOME>/strata185/Strata/engine-cuda12/strata`, sha256 `bdfec9f5…` (full form in
`raw/inventory/2026-10-16T232834Z/strata-build-artifacts.txt`), version 0.1.44.1, toolkit `<HOME>/strata185/cuda-12.8`.

Raw data: `raw/build-audit/2026-10-17T002952Z/` (primary `cuobjdump` dumps),
`raw/build-audit/2026-10-17T003101Z/` (the full kernel inventory + grep sets), script `scripts/audit-kernel-inventory.py`.

---

## 1. Facts about the working build

| What was checked | Result |
|---|---|
| device-code sections in `engine-cuda12/strata` | **62 cubin + 62 PTX, all `sm_70`** (`cuobjdump --list-elf`, `--list-ptx`) |
| `.target` in the PTX | 62 x `.target sm_70` - **no virtual architectures (`-virtual`), no driver JIT is needed** |
| architecture in `--dump-resource-usage` | a single `arch = sm_70` |
| number of `__global__` kernels in the binary | **2084** |
| translation units the kernels are attributed to | **55 of 56** (2028 of 2084 kernels; 56 remain unattributed) |
| kernels whose declaration was found in the source tree | 1648 of 2084 (the rest are template instantiations declared by macros; their TU was determined from the mangling) |
| registers per kernel | max **255**, median **48**; 139 kernels above 128 registers |
| shared memory per kernel | max **41 104 B** (`fused_gr.cu`), median 64 B; 17 kernels above 16 KiB |
| the `strata-vision` engine | a separate 8 MB binary, also sm_70 (not used in the working profile) |

The V100 per-block shared-memory limit (the CUDA specification for CC 7.0: 48 KiB opt-in) is not exceeded by any kernel; the engine raises it with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` (example: `src/kernels/cuda/fused_gr.cu:944`).

The top 5 files by number of compiled kernels (inventory, `kernel-tu-summary.csv`):

```
native_mmvq.cu        449   iq_kernels.cu      210   native_bf16.cu   128
fused_gr.cu            75   verify_kernels.cu   59   kernels.cu        48
```

and the llama.cpp component: `third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh` - 704 MMQ kernels through templates (part of the same binary, not a separate library).

## 2. "Component -> SM70 requirement -> check result"

Checked: the presence of a kernel in the binary (the inventory), the presence of a fallback in the code (compile-time and runtime gates), and agreement with the upstream table in `docs/NVIDIA_V100.md`.

| Component | Requirement | What the working build contains | Status |
|---|---|---|---|
| MMQ / MoE experts (ggml) | Volta has no int8 tensor cores -> the `dp4a` path | `mmvq_*` / `mul_mat_q*` kernels are compiled; the path is chosen at `third_party/.../mmq.cuh:252-255, 279-281` (`__CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA` -> the dp4a path) | present |
| dense projections | FP16 tensor-core GEMM (cuBLAS/CUTLASS) | host calls to cuBLAS, no separate kernel names (checked in code: `src/prefill/gemm.cu`) | present, from code |
| BF16 projections (hyper-connection, router, indexer) | Volta has no BF16 GEMM -> conversion to FP16 | `src/prefill/gemm.cu:467-475`: **on by default for CC 7.0-7.4**, off for 7.5; `STRATA_BF16_TC=0/1/2` is the switch; `bf16_to_f16_kernel`, `native_bf16.cu` (128 kernels), `dequant_bf16.cu` (48) are in the binary | on by default |
| QSA prompt attention (int8 / FP16 / K8V4 KV) | `mma.m8n8k4` - Volta only | `prompt_attn_v70_kernel<0>`, `<1>`, `<3>` - **3 instantiations in the binary**; code `src/kernels/cuda/qsa_prompt_attn.cu:743-1128`, guard `__CUDA_ARCH__ >= 700 && < 750`, outside that range `__trap()` (`:770`); the host selects it with `volta = cc[dev] < 75` (`:1482`) | **an explicit SM70 path of its own**, not JIT emulation |
| QSA prompt attention, the common path (sm_75+) | `mma.m16n8k8/k16`, `cp.async` | `STRATA_PA_SM80` (`:33-41`); below 750 the instruction compiles to `__trap()` (`:50`) and the host refuses such a device -> the old/`v70` kernel runs | fallback present |
| QSA decode/verify attention | the pre-sm_75 kernel | `STRATA_ATTN_PRE75_BUILT` is built only with `STRATA_EXPERIMENTAL_SM60` (`src/kernels/cuda/qsa_decode_attn.cu:285-294`); in the binary the TU `qsa_decode_attn.cu` has 13 kernels | built into the binary |
| QSA block scores | TF32 MMA needs sm_80 | `block_scores_tc_kernel` **is compiled**, but the runtime gate `cc_major < 8` falls back to the warp kernel (`src/kernels/cuda/qsa_select.cu:1143-1165`); A/B switches `STRATA_QSA_WARP`, `STRATA_SELECT_SIMT` | fallback; on this card the warp path is used |
| cluster kernels (top-k, sampler) and PDL | sm_90 + PTX >= 90 | gates `cc_major >= 9 && fa.ptxVersion >= 90 && fa.binaryVersion >= 90` (`qsa_select.cu:1252`, `verify_kernels.cu:1464`, `sampler.cu:1219`); `griddepcontrol.wait/launch_dependents` compile only at `__CUDA_ARCH__ >= 900` (`include/strata/kernels/pdl.hpp:63-71`) | **not used on SM70** (by design, the path is replaced) |
| `kPdlPrefetch` | prefetch without PDL edges | `pdl.hpp:53-58`: false only below 700 -> on sm_70 it is **on** | works |
| `__restrict__` in PDL locals | below 700 an LDG.CI is needed | `pdl.hpp:42-48`: for 700…899 the macro is empty | correct for Volta |
| `__dp4a` | CC >= 6.1 | `include/strata/kernels/dp4a.hpp:27,42`: a fallback only below 610 -> native `__dp4a` on sm_70 | native instruction |
| `__nanosleep` | CC >= 7.0 | `dp4a.hpp:48-51`: a fallback only below 700 -> native on sm_70 | native instruction |
| GDN chunked scan | needs CC >= 8 and an SM test | `src/prefill/kernels.cu:1039` -> `cudaErrorNotSupported` on sm_70, and the caller uses the ordinary recurrent kernel (`gdn_rec_*`, `fused_gdn.cu` in the inventory) | not available on SM70 (replaced) |
| `gr_down_multi` "small tile" | CC 7.5 only | `src/kernels/cuda/fused_gr.cu:936`: `small_tile = (cc == 75)` -> on sm_70 the tile is 2560 | correct choice (not 1280) |
| hyper-connection up/norm "latency-hidden" | built for CUDA as well | `fused_gr.cu:31-33` `STRATA_GR_FAST_BUILD=1` for non-HIP; `gr_up_fast_kernel`, `gr_up_q8_fast_kernel`, `gr_norm_fast` in the inventory; switch `STRATA_GR_FAST` (`:1939-1941`) | built in |
| dense 2-4 column GEMV | the "interleaved" variant with a Volta rows table | `native_mmvq_il_kernel` - **60 instantiations**, `swiglu_il_kernel` - 1 (`src/kernels/cuda/native_mmvq.cu:2663-2692`, `STRATA_MMVQ_IL*`) | built in |
| the expert mode from gfx906 | opt-in `STRATA_EXP_MODE` | `src/kernels/cuda/iq_kernels.cu:3591-3593` (`-1` = the card's default); the TU `iq_kernels.cu` has 210 kernels | available; the specific mode was not tested on this card |
| Pascal paths (`bf16_to_f32`, `cublasSgemm`) | sm_6x only | `gemm.cu:490-498, 543-551, 662-666` | not relevant to SM70 |

**The answer to "are there places that need sm_75 or newer".** Yes, they exist in the code (MMA `m16n8k8/k16`, `cp.async`, the TF32 scorer, cluster kernels, PDL, the GDN chunked scan). None of them is a dead end: each is closed either by a compile-time replacement (`__trap()` plus the host path refusing the device) or by a compute-capability runtime gate that calls the fallback kernel. Separately, **an SM70 path of its own exists** (`prompt_attn_v70_kernel` on `mma.m8n8k4`, `native_mmvq_il`, `gr_*_fast`, BF16->FP16 on by default for 7.0-7.4) - the build is not "a universal build with JIT filling the gaps", it is built for Volta.

## 3. How SM70 is switched on

1. **Build.** `CMakeLists.txt:117-121`: architectures `70` and `60` are accepted **only** with `STRATA_EXPERIMENTAL_SM60=ON`; the same place notes "CUDA 13 dropped sm_60/sm_70", so a CUDA 12.x is required (here a private 12.8, `CMakeLists.txt:144-153`). The option is declared at `CMakeLists.txt:129-131`. This matches `docs/NVIDIA_V100.md` (the build command with `-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`).
2. **The runtime device check.** `src/core/device.cu:283` - the ordinary engine requires CC >= 7.5; `:302-313` - with `STRATA_EXPERIMENTAL_SM60` the floor becomes 6.0, otherwise an exception "Strata needs compute capability …". The working log has `GPU 0: Tesla V100-SXM2-16GB, compute capability 7.0` - the check passed.
3. **Replacing two intrinsics.** `dp4a.hpp` (see the table) - exactly what `CMakeLists.txt:131-133` states.
4. **Emulating "virtual" architectures.** `include/strata/core/emulate.hpp:3-7` (`STRATA_EMULATE_CC=75|80|86|89`) is needed only when device code is built as PTX for an older architecture and loaded by the driver. **Not the case here**: all 62 PTX are `.target sm_70`, the cubins are real.

## 4. What the engine actually uses on this machine

From the working environment (`raw/inventory/2026-10-16T232834Z/strata-process-env.txt`) exactly one `STRATA_*` variable is set in the server process - `STRATA_STAGE_TRIM` (value not published). None of `STRATA_PROMPT_ATTN_OLD`, `STRATA_BF16_TC`, `STRATA_QSA_WARP`, `STRATA_GR_FAST`, `STRATA_EXP_MODE`, `STRATA_DF_PDL` is set, so the defaults are in effect: **the SM70 paths from the table above are on** (prompt attention on `m8n8k4`, BF16->FP16, the interleaved GEMV, fast GR), and the sm_80+/sm_90+ paths are off by the gates.

Switches useful for A/B work on one machine:

| Variable | Effect | Where |
|---|---|---|
| `STRATA_PROMPT_ATTN_OLD=1` | the old decode path instead of `prompt_attn_v70_kernel` | `src/prefill/prefill.cpp:2921` |
| `STRATA_BF16_TC=0/1/2` | off/on for the FP16 path for BF16 | `src/prefill/gemm.cu:522` |
| `STRATA_QSA_WARP=1\|select` | run the pre-sm_80 kernels on any card | `qsa_select.cu:1155`, `qsa_prompt_attn.cu:1477` |
| `STRATA_SELECT_SIMT=1` | an FP32 tiled scorer instead of the warp kernel (a different summation order) | `qsa_select.cu:1162` |
| `STRATA_MMVQ_IL*`, `STRATA_MMVQ_IL_ROWS` | the interleaved GEMV and its rows table | `native_mmvq.cu:2663-2692` |
| `STRATA_GR_FAST=0/1` | fast vs plain GR | `fused_gr.cu:1939-1941` |
| `STRATA_EXP_MODE` | the expert mode | `iq_kernels.cu:3591-3593` |
| `STRATA_GDN_CHUNKED=2` | remove the SM test (on a card with fewer than 128 SMs it is expected to be slower) | `kernels.cu:1038` |
| `STRATA_DF_PDL` | PDL in a verify window (on sm_70 it is off by the gate anyway) | `verify_kernels.cu:1461` |

## 5. Indirect signs of compatibility

* In the working service log (33 251 lines) and in the server's stdout there is **not one** occurrence of `no kernel image is available`, `invalid device function`, `CUDA error`, `not supported` (checked with `grep -c`, the count is 0).
* All 62 cubins are sm_70; there are no module-load errors.
* This is **not** proof of numeric correctness: no load errors only means the needed kernels exist and run. Numeric correctness is `correctness-report.md`.

## 6. How this compares with other SM70 stacks on the same hardware

Reading report documents only, no measurements, no settings changed.

| | a 1Cat-LLM stack | a QPN2/QPN8 variant on 1Cat-vLLM | this Strata build |
|---|---|---|---|
| stack | 1Cat-LLM (a vLLM fork for V100/SM70) v1.5.1, wheel | QPN2/QPN8 kernels on 1Cat-vLLM 1.2.2 | Strata 0.1.44.1, its own CUDA stack |
| device code | Triton/TileLang **JIT** on the card + a `FLASH_ATTN_V100` path | its own QPN2/QPN8 kernels over vLLM | **AOT cubin sm_70** (62), no JIT needed |
| model/profile | `Qwen3.8-27B-FP8` + MTP (262144 window) / NVFP4 | `Qwen3.8-27B-NVFP4` | `Qwen3.8-Flash-Next IQ3_S` (a native GGUF pack) |
| four cards | TP4 (NCCL over NVLink) | TP4 | Strata's layer split / expert placement |
| relevance | shows that an NVLink TP4 path on this machine is already exercised by another stack; our P2P numbers (38.77 GB/s per pair, 77.50 GB/s total) are an upper bound of what the transport can give | the same | the subject of this audit |

The difference is not "is SM70 supported" but how code for SM70 is obtained: JIT at start-up versus a cubin built in advance. For this machine the second matters: Strata has no JIT stage at start, so start-up latency and "the kernel is missing" risks are absent by construction.

## 7. Limits of this audit

1. The translation unit of 56 of the 2084 kernels is not established (mangling without a TU marker); 436 kernels are template instantiations whose declaration was not found for all of them.
2. The table of "what actually runs" is derived from code gates and defaults, not from call tracing: profiling the kernels that run would need a profiler.
3. The numeric correctness of the SM70 paths (including the `m8n8k4` attention) was not checked here; the upstream has the `qsa_prompt_attn_parity` tool (`docs/NVIDIA_V100.md`), building it is a candidate for the stand.
4. The performance of the SM70 paths was not measured here; the upstream figures (22.8 -> 21.2 ms per verify window, +15…18 % prompt reading) are someone else's measurements on another machine and appear here only as a hypothesis.
5. The git SHA of the tree was not recorded - this audit is tied to the sha256 of the binary and the tree.

## 8. Artifacts

```
scripts/audit-kernel-inventory.py                          the inventory script (read-only)
raw/build-audit/2026-10-17T002952Z/                       primary cuobjdump dumps (list-elf, elf-symbols, resource-usage)
raw/build-audit/2026-10-17T003101Z/kernel-inventory.csv   2084 rows: kernel -> TU -> regs/shared/local/stack
raw/build-audit/2026-10-17T003101Z/kernel-tu-summary.csv  56 rows: a summary per translation unit
raw/build-audit/2026-10-17T003101Z/cuobjdump-*.txt        the dumps the inventory was built from
raw/build-audit/2026-10-17T003101Z/arch-grep-1.txt        grep set: __CUDA_ARCH__ thresholds, STRATA_EXPERIMENTAL_SM60
raw/build-audit/2026-10-17T003101Z/arch-grep-2.txt        grep set: compile-time and runtime gates, env switches
```
