# Intel GPU (SYCL / Level Zero)

Strata's Intel backend compiles the same engine as the CUDA and HIP builds. A small rewrite
(`tools/xpu/rewrite_cuda.py`) turns `<<<>>>` launches and `__shared__` declarations into SYCL,
and `include/strata/xpu_compat` supplies the runtime, the fp16 helpers and the warp intrinsics.
NVIDIA-only inline PTX (tensor-core MMA, `cp.async`, thread-block clusters) takes the portable
path already used on AMD. Prefill GEMM goes through oneMKL. Graphs are replayed eagerly: the
same kernels run, with more host overhead than a CUDA graph.

This is a first port. It is not bit-identical to the NVIDIA build. Shuffle and `__dp4a` are
correct subgroup / scalar implementations, not XMX kernels yet.

Measured on one Intel Arc Pro B60 (device 0xe211, 23.3 GB), Qwen3.8-Flash-Next Q2_0, expert
cache auto (14,570 slots, 18.8 GiB), context 1024, `--spec 2 --kv int8`: prefill 127 tokens at
**77.8 tok/s** (time to first token 2.1 s); greedy decode with the speculative verify window at
**10.2 tok/s** (64 tokens). Correctness gates in-run: slot 0 verified bit for bit, the verify
window's per-layer doorbells all ring, exit 0.

Where the remaining time goes (STRATA_XPU_PROFILE, per-kernel GPU time): ~53% is the quantized
expert GEMV family (`native_mmvq_multi`, `native_gu_multi`, `native_down_multi`, `gr_*`) — all
scalar `__dp4a` emulations, the XMX/`joint_matrix` rewrite is the next lever; ~27% is doorbell
parked time (the CPU serving between rings); submission overhead is gone since graph launch is
one `ext_oneapi_graph` call (~1.2 µs/op amortised vs ~31 µs/op for the eager loop). This host's
PCIe link measures 3.6 GB/s pinned (the 3090 bench box: 23-26 GB/s, x16), so the engine keeps
its PCIe tier small and streams experts from the files instead.

The Intel-specific part that cost the most time: the doorbell handshake. The verify-window graph
contains spin kernels that poll host-mapped flags the CPU raises between layers. On this stack a
pure volatile read loop over host USM can spin forever on a stale GPU-cache line: neither volatile
nor system-scope atomic loads snoop the CPU's stores. The fix is one line in
`xpu_compat/intrinsics.hpp`: `__nanosleep` (which every `strata_spin_pause()` call site hits each
poll iteration) issues a `sycl::atomic_fence(seq_cst, system)` that refreshes the line. Without it
the engine times out at the first flag wait, every run.

## Build

On a machine with the Intel GPU compute runtime and oneAPI (icpx + oneMKL):

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-xpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_XPU=ON -DSTRATA_BUILD_TESTS=OFF
cmake --build build-xpu --target strata -j$(nproc)
```

`build-xpu/strata` is the engine. Device 0 is the first SYCL GPU. Two cards are not used
together yet (`cudaDeviceCanAccessPeer` reports no).

## What is not in this build

- Tensor-core / XMX kernels (the portable kernels run instead)
- Registering an existing host mapping (`cudaHostRegister` fails; the engine keeps the arena
  resident and copies)
- Multi-GPU layer split
- The one-click installer (`setup.py` still knows `cuda` and `hip` only)
