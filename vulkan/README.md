# A Vulkan backend for Strata (probe stage)

> **Superseded — see `../opencl/README.md`.**
>
> Everything below was written on the assumption that Vulkan is the only compute API covering Windows +
> NVIDIA + gfx906 in one process. That assumption is wrong: AMD's Windows driver registers an OpenCL 2.0
> runtime for gfx906, so the MI50 is reachable from an API whose language is close enough to CUDA that the
> existing `.cu` expert kernels can be *transcribed* instead of re-derived — and a transcription can be
> checked bit-exactly against the CUDA original, which a SPIR-V shader cannot.
>
> This directory is kept because it is not wasted work: it established the device enumeration, proved the
> IQ3_XXS algorithm and the split `d`/`aux`/`qs` layout, produced the reference data and the Python model
> that the OpenCL kernels are now validated against (`probe/iq_bench.bin`, `probe/iq3_ref.py`,
> `probe/iq4nl_ref.py`), and measured the MI50 at 83 GB/s. The OpenCL transcription of the same kernel
> reaches 100 GB/s and is bit-exact, so the Vulkan shader path is no longer being pursued.

**Goal.** Let a GPU that has no CUDA and no usable HIP join a Strata run. The card this starts from is an
Instinct MI50 (gfx906, 32 GB) on Windows: ROCm does not ship gfx906 for Windows and AMD's HIP path there covers
gfx1030/gfx11/gfx12 only (`docs/AMD_HIP.md`), so Vulkan is the one compute API left for it. Vulkan is also the
only API in the tree that both vendors expose at once, so a Vulkan device can sit next to the CUDA card in one
process - the mixed NVIDIA + AMD case `docs/MULTI_GPU.md` lists as not supported.

**Where the engine stands today.** The GPU side of Strata is CUDA sources (`src/kernels/cuda/*.cu`, `src/core/*.cu`).
Ports are not a runtime abstraction: `include/strata/platform/hip_compat/` maps the CUDA runtime and BLAS names to
HIP and the same `.cu` sources are recompiled (`STRATA_HIP_GFX906`), and `sycl/` is a separate ported tree for Intel
Arc. Nothing in the tree uses Vulkan, and no Vulkan backend can be reached by compiling the existing sources - the
CUDA->Vulkan gap is not a rename of an API, it is a different execution model (SPIR-V modules, descriptor sets,
explicit barriers, no cuBLAS, no CUDA graphs).

**Why the secondary-device path is the way in.** The engine already moves expert work across devices through
**pinned host memory, without peer-to-peer** (`docs/SECOND_GPU.md`: the CUDA1-3 helper caches, and the layer
split's host hand-off). A helper device only has to do three things: hold a set of expert rows in its VRAM, compute
those rows when a routing list names them, and get the results back through mapped host buffers. That is a much
smaller contract than the primary device's (dense weights, KV/state, the 48 captured layer graphs, MTP, the prompt
path). A Vulkan expert-helper device is therefore a tractable first target, and a mixed CUDA-primary +
Vulkan-helper run is exactly the configuration the project says it does not have.

## What is measured so far

Measured on this machine (2026-10-06): Xeon E5-2696 v4, 128 GB DDR3L-1866, Windows 10 22H2,
RTX 5060 Ti 16 GB on PCIe 3.0, Instinct MI50 32 GB (Radeon VII driver) on PCIe 3.0.
Vulkan SDK 1.4.357, MSVC 19.44, Khronos validation layer on (no errors reported in the runs below).

**Step 1 - `probe/device_probe.cpp`: both cards are visible in one process.**

| # | device | Vulkan | VRAM | compute queue | subgroup |
|---|---|---|---|---|---|
| 0 | NVIDIA GeForce RTX 5060 Ti | 1.4 | 15.9 GiB | yes | 32 |
| 1 | AMD Radeon VII (MI50) | 1.3 | 32.0 GiB | yes | 64 |

The MI50 reports its full 32 GB and a compute queue although it has no display output and runs a consumer
driver; the subgroup sizes differ (32 vs 64), which is the wave32/wave64 split the gfx906 HIP path handles with
its own kernels.

**Step 2 - `probe/gemv_probe.cpp`: one expert-GEMV kernel, bit-exact on both.**

`probe/gemv.comp` computes `out[row] = scale[row] * dot(w_q[row][:K], x_q[:K])` - 4096 rows x 1024 bytes of int8
with a fp32 row scale, the shape of an expert row computation. One workgroup per row, 32 lanes, shared-memory
tree reduction: no subgroup builtins and no specialization constants, so one SPIR-V module serves wave32 and
wave64 and the older AMD driver. 50 dispatches back to back, one command buffer each, results copied back per
dispatch.

| device | vs CPU reference | ms/dispatch | weights read |
|---|---|---|---|
| RTX 5060 Ti | max_err 0.0000, 0 of 4096 rows wrong | 0.025 | 160 GB/s |
| Radeon VII (MI50) | max_err 0.0000, 0 of 4096 rows wrong | 0.058 | 72 GB/s |

Both are far below the cards' memory bandwidth (5060 Ti ~448 GB/s, MI50 ~1 TB/s HBM2), so this kernel is
compute-bound: it does the four byte products per word with sign-extension arithmetic. That is the portable
baseline, not the target - Strata's own expert kernels use a packed dot (`dp4a` or the gfx906 `g^(m)` form) and
the tree's MMVQ layout. The number to beat for the helper role is the **CPU pool** on this machine (E5-2696 v4,
AVX2, no AVX-512: the CPU expert path is the fallback every routed-but-uncached expert lands on).

**Step 3 - `probe/iq3_gu.comp`: an IQ3_XXS expert row, bit-exact on both cards.**

This is the format the IQ3_XXS model files actually hold, so it is the first kernel that matters
for the goal rather than for the probe. `iq3_gu.comp` transcribes the engine's
`Split<18>::load`/`apply` from `src/kernels/cuda/iq_kernels.cu` - the codebook lookup, the
`unpack_ksigns` sign bytes, `vcmpne4`/`vsub4`, the `(ls*sumi + sumi/2)/2` scale - as scalar
integer arithmetic. Each of the format's 98-byte blocks carries 256 weights in 8 chunks of 32:
eight codebook indices per chunk, and one `aux32` word holding that chunk's signs and 4-bit
scale. 32 lanes compute one row with a stride-32 walk over the chunks and a tree reduction,
which is the `row_dot<18>` / `warp_sum` structure, so the summation order matches too.

Expected error is exactly zero, because both sides are int32 arithmetic with the same
truncation - so a nonzero error is a real transcription difference, not float noise.

| device | vs the engine's integer path | rows/64 | ms per 64 rows | weights read |
|---|---|---|---|---|
| RTX 5060 Ti | max_err 0.000000 | 0 wrong | 0.056 | 0.9 GB/s |
| Radeon VII (MI50) | max_err 0.000000 | 0 wrong | 0.050 | 1.0 GB/s |

Getting there needed the reference to be as pedantic as the kernel: `ref_int` in
`probe/iq3_ref.py` rounds every operation to fp32 and sums in the lanes-then-tree order, because
a plain running sum in Python doubles disagreed in the last bit. The reference is checked two
ways - `ref_int` against the naive dequantization `w = grid * sign * d * (2*ls+1)/4` - so a
mistake in reading the block format would have shown up before the shader was ever built.

**The speed numbers are not a result yet.** At ~1 GB/s the kernel is three orders of magnitude
from either card's memory bandwidth: `dp4a` is scalar rather than `dotPacked4x8EXT`, and every
byte of `qs` is fetched with its own read. That is the deliberate correctness-first version -
the shape of an expert row is proven, and the optimisation has a reference to check against.

## Two bugs the probe paid for

Both are recorded because they are what a Strata Vulkan backend will hit again:

- **The pipeline layout must name the descriptor set the shader uses.** A layout with `setLayoutCount = 0`
  while the SPIR-V declares `set = 0, binding = 0..3` is not an install-time error: the dispatch ran, read
  whatever the driver left bound, and returned wrong numbers for all 4096 rows.
- **A command buffer must not be re-recorded while it is pending.** Submitting one buffer 50 times in a loop
  without a fence is a spec violation that raised `VUID-vkResetCommandBuffer-00045` and faulted the GPU
  (`VkResult -4`, DEVICE_LOST). Windows logged `nvlddmkm stopped responding and has successfully recovered`
  (Display/Event 4101) at the same second - a TDR, and the next runs failed earlier until the driver settled.
  The fix is one recorded command buffer per iteration, submitted together.

## Plan

1. **An MMVQ-equivalent expert kernel in Vulkan** for the formats the helper would hold (IQ and the K-quants),
   benched against the CPU pool on this machine - the gate for whether a Vulkan helper is worth wiring in.
   *In progress:* the IQ3_XXS gate/up row is transcribed and bit-exact on both cards (Step 3). Still missing:
   the down type the IQ3_XXS files use (Q2_0 or IQ4_NL), the SwiGLU + q8_1 pass that joins gate/up to down,
   and the speed work - correctness first, then `dotPacked4x8EXT` and vectorised loads.
2. **A Vulkan device layer**: allocation, a pinned-host staging ring, the transfer contract `docs/SECOND_GPU.md`
   already describes, and a queue per device in one process.
3. **Wire it in as a helper expert cache** beside the CUDA primary (`--expert-cache-deviceN` style), reusing the
   host scheduling and the per-request counters the log already prints.
4. Later, if the helper pays: the layer split for a Vulkan card, and finally the primary path (dense weights,
   KV, the prompt path) - the part that needs Vulkan replacements for cuBLAS and the captured graphs.

## Build and run

Double-click **`probe/run_probe.bat`**: it builds both programs and runs step 1 and step 2 in order,
leaving both tables on screen. Or by hand:

```bat
cd vulkan\probe
build_probe.bat
build\vk_probe.exe        rem step 1: device list
build\vk_gemv.exe         rem step 2: expert GEMV on every device
build\vk_gemv.exe 1       rem ... on one device only (Vulkan index)
```

Needs the Vulkan SDK (path in `build_probe.bat`), MSVC, and CMake+Ninja (`build_probe.bat` sets the MSVC and
Windows SDK environment by hand; it does not call `vcvars64.bat`, which needs `reg.exe`).
