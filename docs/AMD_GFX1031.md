# RX 6700 XT (gfx1031) on Linux with system ROCm: what it took, with the numbers

One PC's report, for #524 / #881. docs/AMD_HIP.md lists gfx1031 as "runs but unvalidated"; here is a validation run
and the two changes it needed. Nothing here changes the default path for any other card.

Hardware and software — every number below was measured on this machine
- RX 6700 XT 12 GB (device id 0x73DF, which setup maps to gfx1031), 64 GB RAM, Intel Xeon E5-1620 v3 (4 cores, 8
  threads), Ubuntu, system ROCm 7.1.1 in /opt/rocm (not TheRock's venv), rocBLAS 5.1.1.
- Strata 0.1.45, HIP engine built from source, model Qwen3.8 Swift 1.5 iq3_s, 2046 KV slots, 12.39 of 12.87 GB VRAM
  with the model resident.

Two separate faults on one card

1. The engine did not compile (fixed locally, still uncommitted). src/kernels/cuda/iq_kernels.cu has
   `__shared__ alignas(16) int4 s_u[...]` in expdec_gu_kernel (1858) and expdec_down_kernel (2045); HIP clang answers
   "'alignas' attribute cannot be applied to types", the #778 pattern. The file already carries the answer for
   uint32_t arrays — STRATA_SHARED_ALIGN16_U32 puts `__attribute__((aligned(16)))` after the declarator under
   __HIPCC__ and keeps alignas elsewhere. The fix is the same macro for int4 (STRATA_SHARED_ALIGN16_I4) and those two
   sites: 4 lines added, 2 changed, layout identical. nvcc accepts the CUDA form, which is why CUDA builds never see it.

2. The engine loaded cleanly and died on the first long prompt in a way that reads as out of memory. It is not memory:
   64 GB RAM, 12.39 of 12.87 GB VRAM used at the time. The log line is the cause —

       rocBLAS error: Cannot read /opt/rocm/lib/rocblas/library/TensileLibrary.dat: No such file or directory for GPU arch : gfx1031

   The prompt's dense matrix products go cublasGemmEx -> rocblas_gemm_ex (src/prefill/gemm.cu), and that path only runs
   above the prefill token threshold, so a 367-token prompt is fine and a 12K-token one kills the process. ROCm 7.1.1's
   rocBLAS ships Tensile libraries for gfx1030 only: `ls /opt/rocm/lib/rocblas/library | grep -c gfx1031` is 0.

   Both halves below are needed, and neither works alone:
   - build the engine for the family's gfx1030 as well as gfx1031: `-DCMAKE_HIP_ARCHITECTURES="gfx1030;gfx1031"`.
     275 s on this 8-thread CPU; engine/strata 53.7 MB against 32.5 MB for one arch, and no run-time cost.
   - run the card as gfx1030: `HSA_OVERRIDE_GFX_VERSION=10.3.0` in the run config's "env" (serve passes cfg["env"] to
     the engine, serve/server.py:2867; docs/AMD_HIP.md:666 does the same for gfx906). The log then reads
     `GPU 0: AMD Radeon RX 6700 XT (gfx1030)`.
   The override on a gfx1031-only binary SEGFAULTS — HIP then goes looking for gfx1030 code objects — and the fat
   binary alone leaves rocBLAS asking for gfx1031. Not a fix, tried: a ROCBLAS_TENSILE_LIBPATH directory of symlinks
   renaming the gfx1030 files to gfx1031. The abort stops and every rocblas_gemm_ex returns status 6 (internal_error)
   and writes nothing.

What the gfx1030 kernels do on this card — rocblas_gemm_ex probe, N = 10240 x T = 1024 x K = 2560 (a shape the prompt
path uses), checked against a CPU reference
    f16 in, f32 out (the default route)   7.0 TFLOPS, rel err 2.5e-4
    f16 in and out (STRATA_HIP_PROMPT_F16) 21.3 TFLOPS, 4.3e-4
    bf16 in, f32 out                      6.6 TFLOPS, 1.9e-3
    f32 in and out                       11.8 TFLOPS, 6.6e-7
Probe gotcha: rocblas_gemm_ex takes alpha/beta as the compute type (f32), not the input type; a __half alpha reads as
a denormal, the output comes out ~zero and looks like a broken kernel.

End to end through the OpenAI endpoint — the same 11,103-token prompt in one server session, every read from the log
    default route (3 reads):        33.9 s / 32.8 s / 34.4 s reading (327.6 / 338.1 / 322.4 tok/s), 37 tokens in
                                    1.09 s / 1.64 s / 1.68 s (22.0-34.1 tok/s)
    STRATA_HIP_PROMPT_F16=1 (1 read): 17.9 s reading (620.9 tok/s) = 1.90x, 37 tokens in 1.26 s (29.3 tok/s)
A 13,023-token prompt reads fine too (38.6 s, 337.6 tok/s), and the second copy of the same 11,103-token prompt hits
the prefix cache: 11,098 of 11,103 tokens reused, 5 read, in 102 ms.
STRATA_HIP_PROMPT_F16=1 is the engine's own tip on gfx103x and stays opt-in: it is not bit-identical to the default (an
FP16 rounding of the prompt GEMMs). Decode does not use these GEMMs, which is why decode is unchanged.

The built-in suite (`tools/strata_bench.py --long`, three runs per test) in both arms is in
[../bench/results/2026-10-19-community-rx-6700-xt/](../bench/results/2026-10-19-community-rx-6700-xt/): prompt
339.2 / 352.5 / 356.2 tok/s becoming 595.9 / 692.9 / 707.5 at 4K / 16K / 32K tokens (1.76x / 1.97x / 1.99x), TTFT
11.90 / 45.61 / 90.05 s becoming 6.80 / 23.23 / 45.39 s, chat decode 21.7 becoming 21.0 tok/s.

The setup.py change (43 added, 4 changed, no new defaults)
- `HIP_ROCBLAS_SIBLINGS = {"gfx1031": "gfx1030"}` and `HIP_ROCBLAS_AS = {"gfx1031": "10.3.0"}` next to AMD_ARCHS, with
  the reason and these numbers in the comment above them.
- `hip_build_archs(gpu)`, used by build_engine_hip: the arch list grows by the sibling, so the code objects are in the
  binary and BUILD.json's "archs" carries both — setup's "engine already built for this PC" test keeps passing.
- `hip_report_arch(arch)` in the HIP config block: writes HSA_OVERRIDE_GFX_VERSION into cfg["env"], says so, and looks
  the hipBLASLt / rocBLAS tuning tables up under the name the engine will report (the engine picks the table of the
  arch it sees).
- HSA_OVERRIDE_GFX_VERSION added to SETUP_ENV, so a setup run again refreshes it instead of leaving it behind.
- tools/test_setup_gfx1031.py — 8 tests, no GPU, no ROCm, no downloads: `python -m unittest tools.test_setup_gfx1031`.
  It asserts a gfx1031 PC builds gfx1030+gfx1031, a split keeps every card, every other arch builds only its own with
  say() never called, the tables resolve to the arch the engine reports, and carry_over keeps the user's
  STRATA_HIP_PROMPT_F16 while treating the override as setup's own.
- Checked: the 8 new tests pass, and tools.test_setup_config, _rerun_size, _amd, _choices, _configs pass.
  tools.test_setup_engine_hash fails 2 tests on this PC with and without this change (it fetches
  .../releases/latest/download/, which 404s here), so those two are not from this change.

Not tested here: Windows, TheRock's venv ROCm, a layer split with a 6700 XT in it, the vision encoder,
golden-answer parity (the F16 route is not bit-identical), any quantisation other than Swift 1.5 iq3_s, the other
gfx1031 cards (6750 XT, 6800M), long sessions, and ROCm older than 7.1.1.

Reproduce on a 6700 XT with system ROCm
    cmake -S . -B build-hip -DCMAKE_HIP_ARCHITECTURES="gfx1030;gfx1031" <setup.py's other -D flags>
    ninja -C build-hip strata && cp build-hip/strata engine/strata    # then stamp BUILD.json with both archs
    # run config: "env": {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}  (add "STRATA_HIP_PROMPT_F16": "1" for the faster
    # prompt route; the numbers shift slightly)
