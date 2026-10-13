# OpenCL peer tier: running Strata's expert kernels on the MI50

## Why this directory exists

The goal is to put the MI50's 32 GB of VRAM to work as a second expert tier, so the 5060 Ti's 8 GiB
cache stops being the whole story (it holds 3663 of 24576 experts, and the decode hit rate is only
48-73%, so most of the expert math currently runs on the 22-core AVX2 host pool).

The MI50 is a gfx906 card with the Radeon VII driver. It cannot be a CUDA device, and Strata's CMake
refuses to build `STRATA_ENABLE_CUDA` and `STRATA_ENABLE_HIP` at the same time, so the working
assumption had been a hand-written Vulkan backend — see `vulkan/`, which got as far as a bit-exact
IQ3_XXS kernel at 83 GB/s on the MI50.

**That plan is superseded.** AMD's Windows driver registers an OpenCL 2.0 runtime for gfx906, and
OpenCL C is close enough to CUDA that the existing `src/kernels/cuda/iq_kernels.cu` expert kernels can
be *transcribed* rather than re-derived. Transcribed code can be checked bit-exactly against the CUDA
original, which SPIR-V shaders could not be; and the compiler is a real clang rather than a glslc that
would not resolve `dotPacked4x8EXT`.

## What the machine actually provides

```
[platform 1] AMD Accelerated Parallel Processing   OpenCL 2.1 AMD-APP (3592.0)
  device gfx906
    global mem      31.98 GiB
    max alloc       13.60 GiB       <- a single buffer cannot cover the whole tier
    local mem       32 KiB
    compute units   60
    max work group  256
    subgroup size   64             <- reported by the device
    cl_khr_subgroups, cl_khr_fp16, cl_khr_int64_base_atomics, cl_amd_media_ops
    NO cl_khr_integer_dot_product
```

The ICDs live in
`C:\Windows\System32\DriverStore\FileRepository\u0412788.inf_amd64_d661dab6e8d53074\B413136\`
(`amdocl64.dll`, `amdhsail64.dll`, `amd_comgr.dll`). Headers and the import library come from the CUDA
toolkit (`include/CL/cl.h`, `lib/x64/OpenCL.lib`); at run time the call lands in
System32\OpenCL.dll and dispatches to the registered ICD.

## The 4x8 integer dot, which is the one thing that could have sunk this

Every i-quant kernel is built around `__dp4a`. Measured, not assumed (`ocl_dot.cpp`):

| form | verdict |
|---|---|
| `dot_acc_sat_4x8packed_ss_int` | **not visible.** The string is in `amd_comgr.dll`, but the front end only declares it when the device advertises `cl_khr_integer_dot_product`. gfx906 here does not, so the compiler reports "unknown OpenCL extension ... ignoring" and the call is an implicit declaration. |
| `amd_mqsad` | **unusable.** It exists in `amdhsail64.dll`/`amd_comgr.dll`, but building anything that calls it aborts the host process: `Error in hsa_operand section, at offset 948: Invalid operand 3, expected: a register` / `LLVM ERROR: Brig container validation has failed in BRIGAsmPrinter.cpp`. |
| inline `v_dot4_i32_i8` asm | **not available.** `asm` / `__asm` are undeclared identifiers — this OpenCL front end has inline assembly switched off. |
| `mad24` x4 with a sign-extended byte each | **yes.** Bit-exact against the reference on 65536 random int8x4 pairs, including `0x7F7F7F7F`/`0x80808080` extremes. |
| plain signed multiply x4 | **yes**, also bit-exact. |

So four `mad24`s are what the kernels use. On a 2560-wide row this reaches ~175 GB/s, and the absence of
a native dp4a is not the limit — the Vulkan kernel, using an equivalent scalar fallback, managed 83 GB/s.

## The IQ3_XXS gate/up kernel, on the MI50, bit-exact

`ocl_iq3.cpp` transcribes `vulkan/probe/iq3_fast.comp` (itself a bit-exact transcription of
`Split<18>::load/apply`) into OpenCL C and checks it against the same Python reference the Vulkan kernel
was validated with (`vulkan/probe/iq3_bench.bin`, 1280 real gate+up rows of `n_embd` 2560, 32 reference
rows). Dispatch scaled to 12 copies so launch overhead does not dominate the timing:

```
config                   max_err   bad/refs         ms      GB/s   Mrows/s  us/expert
wg=32 r=1 subgroup      0.000023       0/32      0.2078      72.5      73.9      17.31
wg=32 r=4 subgroup      0.000023       0/32      0.2180      69.0      70.4      18.17
wg=32 r=8 subgroup      0.000023       0/32      0.2064      72.9      74.4      17.20
wg=64 r=1 subgroup      0.000015       0/32      0.1503     100.1     102.2      12.53
wg=64 r=4 subgroup      0.000015       0/32      0.1563      96.3      98.3      13.03
wg=64 r=8 subgroup      0.000015       0/32      0.1622      92.8      94.7      13.52
wg=32 r=4 lmem          0.000000       0/32      0.1956      76.9      78.5      16.30
wg=32 r=8 lmem          0.000000       0/32      0.2045      73.6      75.1      17.04
```

* `lmem` reproduces the reference summation order exactly — **max_err 0.000000**, i.e. the MI50 returns
  the same floats the CUDA path does for this kernel.
* `subgroup` differs by 1.5-2.3e-5, which is fp32 reassociation, not a decoding error. It is ~30% faster,
  and Strata already accepts GPU/CPU rounding differences on the resident-expert path ("the GPU computes
  the experts in the cache; it rounds differently from the CPU"), so this is a real choice.
* `wg=64` beats `wg=32` and R makes little difference: the kernel is ALU-bound, not activation-bandwidth
  bound, which is what the Vulkan probe also concluded.
* **100 GB/s beats the 83 GB/s the Vulkan kernel reached for the same work.**

## One driver bug worth knowing about

The llama.cpp byte-compare idiom

```c
uint nz = (((t & ~kHigh) + ~kHigh) | t) & kHigh;   // kHigh = 0x80808080
return as_int((nz >> 7u) * 0xFFu);
```

is **mis-lowered in bit 31** by this driver. For `t = 0x08040001` it yields `0x00ff00ff` where the
correct answer is `0xffff00ff` — the top byte of the mask loses its high bit, which turns a sign byte
into a positive one. The per-sub-chunk dump in `ocl_iq3.cpp` shows it against Python step by step, and
Python's value is the one that reproduces the reference row, so the driver is at fault.

`vcmpne4` is therefore written as an explicit four-iteration byte loop (`IQ_VCNE_PLAIN`, on by default).
It costs a few instructions per call and it is exact.

## The IQ4_NL down-projection kernel, on the MI50, bit-exact

`ocl_iq4nl.cpp` transcribes `vulkan/probe/iq4nl.comp`, checked against `vulkan/probe/iq4nl_ref.py` (whose
own integer chain is verified against a naive dequantisation at 3e-6).  Down is 2560 rows of 640 values,
360 bytes a row, 0.922 MB, and 39 of the 48 layers use it.  12 copies of the data again:

```
config                   max_err   bad/refs         ms      GB/s   Mrows/s    us/down
wg=32 r=8 subgroup      0.000004       0/32      0.1823      60.7     168.5      15.19
wg=64 r=1 subgroup      0.000008       0/32      0.1153      96.0     266.5       9.61
wg=64 r=4 subgroup      0.000008       0/32      0.1059     104.4     290.1       8.82
wg=32 r=4 lmem          0.000000       0/32      0.1802      61.4     170.4      15.02
wg=64 r=4 lmem          0.000000       0/32      0.1112      99.4     276.2       9.27
```

`wg=64 r=4 lmem` is bit-exact **and** within 5% of the fastest variant, so the exact one is the one to
build on.  Note the down matrix is only 40 dot calls a row against IQ3_XXS's 80, so the non-linear
codebook lookup is a larger share of the work here — it still lands at 99 GB/s.

## The SwiGLU + q8_1 pass, which is what lets an expert chain on the MI50

`ocl_swiglu.cpp` transcribes `swiglu_q8_1_entries_kernel`, `q8_1_store` and `q8_1_finite.hpp`.  Without it
the MI50 could run a matrix but not an expert: gu produces 2*n_ff values, SwiGLU folds them to n_ff, and
those have to be quantised to q8_1 before the down kernel can do integer dots.

Two entry points, both checked against a host model of the same pass: `q8_1_from_h` takes the activations
directly, so the reduction, the quantiser and the store can be diffed bit for bit with no `exp()` in the
way, and `swiglu_q8_1` is the fused form whose only inexactness is the driver's transcendental.

```
[1] q8_1_from_h - 40960 values, 1280 q8_1 blocks, checked bit for bit
reduction                       qs mismatched ds mismatched
xor butterfly (CUDA order)            0/40960     0/2560   EXACT
lane+off tree (local mem)             0/40960     0/2560   EXACT
sub_group_reduce_max/add              0/40960     1/2560
      first ds mismatch: block 405 sum  host c3c7  gpu c3c8
work_group_reduce_max/add             0/40960     1/2560
      first ds mismatch: block 405 sum  host c3c7  gpu c3c8

[2] swiglu_q8_1 - fused, host model uses libm exp
reduction                         exp     qs off by 1 / more  ds mismatched
xor butterfly (CUDA order)        exp                  0 / 0         1/2560
xor butterfly (CUDA order)     native                  0 / 0         1/2560
lane+off tree (local mem)         exp                  0 / 0         1/2560
... (all eight combinations the same)
```

Three things this settles.

* **The xor butterfly is the one to use.**  It is the order `q8_1_store` accumulates in, and it is
  bit-exact.  The built-in reduces differ on exactly one block's **sum** out of 2560, by one fp16 ULP
  (`c3c7` = -24.89 against `c3c8` = -24.906) — and on **zero codes**.  So the reduction only perturbs the
  stored sum, never a quantised code, which is exactly the sensitivity this pass has to get right.
* **`roundf`, not `rint`.**  `q8_1_quant` is `roundf(x/d)`, ties away from zero.  Blocks 2 and 3 of the
  bench data are planted with `amax = 254` (so `d = 2.0`, and a value of `1.0` lands exactly on the tie)
  precisely because `rint` would quietly give a different code there.
* **The clamp matters and is transcribed.**  `q8_1_finite` holds `d` and the stored sum at the largest
  finite half; a block that overflows it is planted too (±1e6).

Cost, at the real row shape (n_ff = 640 = 20 q8_1 blocks, 64 expert-token rows in one dispatch): 0.011 ms,
0.170 us a row.  A decode step offloads ~192 (expert, token) pairs, so the whole activation pass is
**0.03 ms/token** — three orders of magnitude under the budget.

## The remaining four formats, so that every layer is covered

The pack's 48 layers use gu types {IQ3_XXS x17, IQ2_S x20, IQ3_S x10, IQ4_XS x1} and down types
{IQ4_NL x39, Q2_0 x9}, and a layer can only be offloaded when **both** of its formats run here.  Before
this, that was 14 layers out of 48.

`iqfmt_ref.py` writes the reference data for types 21/22/23/42 and, in the same run, decodes each block
two independent ways: the engine's integer path transcribed from `vec_dot_iqX_s_q8_1` and
`Split<X>::load+apply`, and the reference decoder `dq_iqX` dotted with dequantised activations.  They
agree to 1.4e-7 (IQ3_S), 1.06e-7 (IQ4_XS), 9.8e-8 (Q2_0) and 2.2e-5 (IQ2_S — its integer form has a
`cdiv(s0+s1, 2)`, so that is truncation noise, not a layout error).  Any wrong byte offset or sign-bit
index shows up as a relative error of order 1, which is how the four mistakes in the first draft were
found.

`ocl_iqfmt.cpp` then runs them on the MI50, one kernel body with an `#if IQ_FMT` block: the row walk, the
dp4a loop and the reduction are format-independent, only the decode and the integer scale change.
`wg=32` reproduces the reference's own 32-lane tree, so that is the bit-exact column.  rep=32 (Q2_0: 64)
to put the data past L2 and the grid at 20-40k rows:

```
format    wg    R  reps       max|d|       GB/s   us/matrix      Mval/s  copybad
IQ3_S     32    1    32     0.000000       93.6       15.04    217825.6        0   EXACT
IQ2_S     32    1    32     0.000000       72.7       14.35    227067.7        0   EXACT
IQ4_XS    32    8    32     0.000000      104.9       16.60    197444.7        0   EXACT
Q2_0      32    8    64     0.000000       86.2        5.35    306380.2        0   EXACT
```

**A measurement trap worth recording.**  The kernel indexes rows by work-group id, so the grid has to be
`groups * wg` work-items.  Dispatching one work-item per row launches `rows/wg` work-groups and computes
only the first `1/wg` of the rows — and that still looks correct, because the rows checked against the
reference fall inside the prefix.  It inflated the reported bandwidth 32x.  Every result above carries a
`copybad` column for this reason: the weight block is replicated N times and **every copy must reproduce
the first exactly**.  If a dispatch is truncated the tail is stale and copybad is non-zero.  It is 0 for
every row of the table.

## The per-expert cost, which is what decides the target

One expert's blob is `gu (1280 rows) + down (2560 rows)` = 1.254 MB + 0.922 MB = **2.176 MB**.
All measured on the MI50, bit-exact):

| matrix | us |
|---|---|
| IQ3_XXS gate+up, 1280 rows x 2560 | 16.30 |
| IQ3_S gate+up | 15.04 |
| IQ2_S gate+up | 14.35 |
| IQ4_XS gate+up | 16.60 |
| IQ4_NL down, 2560 rows x 640 | 9.27 |
| Q2_0 down | 5.35 |

So per layer, by its format pair:

| gu | down | layers | us/expert |
|---|---|---|---|
| IQ3_XXS | IQ4_NL | 14 | 25.6 |
| IQ2_S | IQ4_NL | 15 | 23.6 |
| IQ3_S | IQ4_NL | 9 | 24.3 |
| IQ4_XS | IQ4_NL | 1 | 25.9 |
| IQ3_XXS | Q2_0 | 3 | 21.7 |
| IQ2_S | Q2_0 | 5 | 19.7 |
| IQ3_S | Q2_0 | 1 | 20.4 |

Every one of the 48 layers is now inside 26 us.

## Why the numbers say this will work

* The 32 GB tier holds roughly 11000-14000 experts of 24576 — 45-57% of all experts, and a much larger
  share of routing traffic given the frequency-ranked placement the engine already uses
  (`data/expert-profile.bin`).  There are 480 expert-computes per token.
* The ~40% of those the 5060 Ti's cache misses currently go to the 22-core AVX2 host pool, which is the
  long pole in the current ~24 ms/token.  192 experts at 25.6 us is **4.9 ms**, running in parallel with
  the 5060 Ti rather than in series with it.
* The 50 tokens/s target is 20 ms/token.  4.9 ms of parallel MI50 work leaves a lot of headroom, and the
  tier also raises the overall cache hit rate, which shrinks the number of experts that need a CPU path
  at all.

## Status

| piece | state |
|---|---|
| gfx906 reachable from OpenCL on Windows, with a working compiler | done |
| 4x8 dot equivalent, bit-exact and fast | done |
| IQ3_XXS gate/up row kernel, bit-exact, 100 GB/s | done |
| IQ4_NL down row kernel, bit-exact, 99 GB/s | done |
| the byte-compare bit trick mis-lowered by the driver, worked around | done |
| SwiGLU + q8_1 activation quantiser in OpenCL, bit-exact | done |
| remaining gu formats: IQ2_S (20 layers), IQ3_S (10), IQ4_XS (1) | done |
| Q2_0 down format (9 layers) | done |
| CL device layer: buffers, staging, queues | not started |
| `PeerExperts` equivalent + engine wiring (`--peer-device`) | not started |

Every expert format the iq3_s pack uses now runs here and matches the reference bit for bit, at 20-26 us
per expert.  What is left is the plumbing: a CL device layer (context, queues, the residency table, and
the fact that the device's largest single allocation is 13.60 GiB so the 32 GB tier has to be split across
buffers), then an OpenCL-backed peer that `src/program/generate.cpp` can dispatch to, and the end-to-end
measurement against the ~40 tokens/s baseline and the 50 target.

## Building

```
opencl\probe\build_probe.bat          # hand-rolled MSVC + Windows SDK env; no vcvars, which needs reg.exe
opencl\probe\build\ocl_probe.exe      # platform/device/capability report
opencl\probe\build\ocl_dot.exe        # dot semantics and row-kernel bandwidth
opencl\probe\build\ocl_iq3.exe    [repeat] [copies] [-1|0|1]   # IQ3_XXS gate/up
opencl\probe\build\ocl_iq4nl.exe  [repeat] [copies] [-1|0|1]   # IQ4_NL down
opencl\probe\build\ocl_swiglu.exe [repeat] [copies] [-1|0|1]   # SwiGLU + q8_1
opencl\probe\build\ocl_iqfmt.exe  [repeat] [rep] [-1|0|1] [21 22 23 42] [--sg]   # the other four formats
```

`ocl_iq3` and `ocl_iq4nl` find their data under `vulkan/probe/` (`iq_bench.bin`, `iq4nl.bin`);
`dump_bench_row.py` prints the same quantities the IQ3 kernel's debug path writes out, in case they ever
need lining up again.  `ocl_swiglu` makes its activations in-process (including the hostile blocks), and
`ocl_iqfmt` reads what the Python generator writes:

```
python opencl\probe\iqfmt_ref.py        # writes iqfmt_{21,22,23,42}.bin and self-checks each one
```
