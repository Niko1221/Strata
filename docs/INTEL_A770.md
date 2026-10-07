# Strata on an Intel Arc A770 (DG2, 16 GB)

How the SYCL port was brought up on an Arc A770 (Alchemist, Xe-HPG, `xe` driver) after being written for the Arc Pro B70 (Battlemage),
what had to change, what was measured, and the bugs found on the way. The port itself is described in [INTEL.md](INTEL.md); every
a short catalogue of the driver, runtime and
compiler problems (with workarounds and what is worth reporting to Intel) is in [INTEL_A770_ISSUES.md](INTEL_A770_ISSUES.md).

## Bring-up and measurements, 2026-10-03

The SYCL port builds and generates correct text on an Arc A770 (Xe-HPG, 16 GB, xe driver), not only the Arc Pro B70 it was
written for. Measured on a Xeon E5 v4 host (22 cores, 121 GB RAM, PCIe 3.0 x16), kernel 7.0 with the xe driver, the Coder
IQ1_M, a 19-token prompt, 128 greedy tokens, 8,192 context. The output matches the sample in "The engine end to end".

**How to build and run it.**

```
cmake -S sycl -B build-sycl-aot -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_AOT=dg2-g10 \
      -DSTRATA_SYCL_NO_XMX=ON -DSTRATA_SYCL_LARGE_BUFFERS=ON
IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1 cmake --build build-sycl-aot --target strata   # full build ~25-40 min

UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 ZES_ENABLE_SYSMAN=1 \
build-sycl-aot/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> --expert-profile data/expert-profile-coder.bin \
    --stream-experts --expert-cache auto --vram-reserve-mib 1024 --prefill auto --spec 2 --suffix-draft 0 ...
```

This is the host-plan path: the CPU computes part of the experts that miss VRAM beside the GPU (8.0-8.7 tok/s). For a GPU-only
run add `STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1`: 9.75 tok/s, the faster of the two (see "Why the GPU-only path was
slow").

The serving image builds the CPU code with `STRATA_GGML_NATIVE=OFF`, `GGML_AVX2=ON`, `GGML_FMA=ON`,
`GGML_F16C=ON` and `GGML_AVX512=OFF`. Its x86-64 CPU baseline requires AVX2, FMA and F16C; it is not a generic
x86-64 image. It enables `STRATA_Q6K_REPACK=1` for the measured A770 speedup below; `docker run -e STRATA_Q6K_REPACK=0`
restores the unpacked layout.

**What differs from the B70, and what each difference needed.**

| the A770 | what went wrong | the change |
|---|---|---|
| no fp64 (the engine accumulates some norms in `double` on purpose) | the build failed with "Double type is not supported" | build with IGC's fp64 emulation (`IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1`) |
| IGC (dg2) crashes on `joint_matrix` | `ocloc` died with a floating point exception on `xmx_gemm_iq` and `qsa_prompt_attn_xmx` | `-DSTRATA_SYCL_NO_XMX=ON` (both are opt-in and slower than the default paths) |
| the oneAPI Level Zero adapter does not answer free-memory queries on xe | `--expert-cache auto` saw 0 GiB free and made 0 slots | `dpct::get_memory_info` asks Level Zero Sysman directly when the SYCL aspect is missing |
| one allocation is capped at 4,095 MiB, host and device | the 9 GiB expert cache and the 14 GiB pinned mirror failed to allocate | `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` for the device; the mirror in chunks of at most 3 GiB |
| IGC addresses a buffer with 32-bit offsets unless told the buffer may exceed 4 GiB | a resident expert 5.96 GiB into the cache dequantised to NaN while the same expert in a small buffer was fine; the prompt path then poisoned the residual from layer 1 on and the model produced only token 0 | `-DSTRATA_SYCL_LARGE_BUFFERS=ON` adds `-ze-opt-greater-than-4GB-buffer-required` to the AOT options |
| a work-group barrier after some work-items have returned never completes | the native router (`route<NE>`) hung the first decode window, GPU busy, until the xe driver reset the compute engine | a sub-group barrier instead (upstream's fix, `tools/fixups.py`) |
| `DPCT_CHECK_ERROR(queue.ext_oneapi_empty())` returns 0, not the queue's answer | "layer N never rang (graph finished)" after 2 ms on any slow layer | the queue's answer is used (upstream has the same fix) |
| a kernel polling `sycl::malloc_host` memory never sees a host store made after it started | the GPU's bounded wait for each flag the host raises gave up, on every layer, although the host had raised all of them; the window went on with garbage | the polled words come from `strata::host_malloc_polled` (`zeMemAllocHost` with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`, `sycl_queue.hpp`); a wait that still gives up now stops the engine and names the flag (`STRATA_WAIT_TIMEOUT=warn` only reports, `STRATA_SPIN_MAX` sets the bound) |

The A770 has one compute queue and one copy queue (the B70 has more), 512 EUs, 64 KiB of local memory per work-group.

**Speed.** Decode is 10.4 tok/s GPU-only and 8.0-8.7 on the host-plan path (llama.cpp's SYCL backend, same GGUF, same card:
9.4-9.8), prompt reading 11-23 tok/s on a 19-token prompt. 4,836 of the 12,288 experts fit in VRAM; the other 7,452 are mirrored in pinned host memory and read by the
GPU over PCIe.

| setting | decode |
|---|---|
| `--spec 4` (default drafts, 8% accepted) | 5.88 tok/s |
| `--spec 2` | 7.67 |
| `--spec 2 --suffix-draft 0` (no drafts) | 7.11 |
| `--vram-reserve-mib 512` (5,102 slots) | 7.33 (against 7.11 at 1,024) |
| `--expert-cache-per-layer` | 6.34 |
| host handshake path, `--adapt-every 0` (CPU pool takes misses) | 7.06 |
| (measured before the handshake fix below, the GPU's waits then gave up) host path with adaptive swaps, 128 tokens | 9.80, 9.77, 6.73 (three identical runs) |
| (before the fix) the same, 256 tokens / `STRATA_ADAPT_TUNED=1` / `--adapt-every 1` / `--spec 3` with drafts | 7.66 / 7.02 / 6.53 / 7.63 |
| GPU-only (`STRATA_VERIFY_NO_HOST=1`), two runs | 7.14, 7.14 |

The three "before the fix" rows were fast when they happened to work because the device plan then never had to wait on the host.
The measurements after the fix are in the second table below.

A round costs about 140 ms whether it carries one token or four; this alone does not establish the decode bottleneck. In eager mode with
`STRATA_VERIFY_PROFILE=1` the 36 GDN layers spend 47 ms in the expert stage (VRAM hits and mirror reads), 17 ms in the first
hyper-connection read and 14 ms in the second read plus the router, against 0.27 ms per layer all-in on the B70. The card
reads host memory at 9.4 GB/s from a kernel and 9.4 GB/s by DMA on this host (a 2 GiB copy), so PCIe 3.0 is a ceiling but
not the whole cost.

**The host-to-GPU handshake (fixed).** The engine's design is that the host raises flags (the plan, the copies, the CPU's
results) that a kernel on the GPU polls. On the A770 a polling kernel never saw a host store to `sycl::malloc_host` memory: 60
million system-scope atomic loads (and every other load variant) saw nothing, and a wait gave up after its bound however long
the bound was (checked at 100x: 2 million polls, about two seconds). Every layer then paid the whole bound and went on with
the data the host was supposed to write still missing: garbage, which the recurrent GDN state keeps for good. Memory from
`zeMemAllocHost` with the uncached or write-combined flag was seen within microseconds in every trial; the same call with no
flags was not reliable. The polled words are now allocated that way (upstream's `host_malloc_polled`; every polled buffer, not only the flags). A wait that still gives up now stops the engine with the flag's name
(`STRATA_WAIT_TIMEOUT=warn` only reports; `STRATA_SPIN_MAX` sets the bound, default 20,000 polls, about 22 ms here; do not set it
to hundreds of millions: a spin that long can trip the xe driver's hang detection).

With the handshake working, the host-plan path is the fast one: 8.0-8.6 tok/s over 128 tokens, five runs, correct output (it
produced zeros from the first token before). The CPU computes about 4-5 experts per layer beside the GPU.

| setting (128 tokens, `--spec 2 --suffix-draft 0`) | decode |
|---|---|
| host plan, adaptive swaps (the default), resident experts mirrored too | 8.05, 8.64, 8.26, 7.82, 8.14, 8.03, 8.36 |
| the same with `STRATA_MIRROR_ALL=0` | 7.31, 7.45 |
| device plan + host handshake, `STRATA_MIRROR_ALL=0` | 7.28 |
| GPU-only (`STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1`) | 7.14 |
| host plan, `--adapt-every 0` (64 tokens) | 5.65, 5.97 (and one run that produced wrong tokens, see below) |

**Keeping the experts in use in VRAM.** The static profile (`data/expert-profile-coder.bin`) puts 4,836 experts in VRAM and with
swaps off the CPU computes about 10 of the 16 experts a two-token window routes per layer (a VRAM hit rate of about a third). The
adaptive swaps move the experts the prompt actually routes to into VRAM and bring that down to about 2 (a hit rate of about 87%).
Their settings were swept on the host-plan path, 256 tokens, `--spec 2 --suffix-draft 0`:

| setting | CPU experts per layer | swapped | decode |
|---|---|---|---|
| default (`--adapt-every 4 --adapt-swaps 96 --adapt-decay 0.7`, 4,836 slots) | 2.1 | 2,621 | 8.69 tok/s |
| `--adapt-swaps 192` | 2.6 | 4,340 | 8.18 |
| `--adapt-decay 0.92` | 3.3 | 4,474 | 8.13 |
| `STRATA_ADAPT_TUNED=1` (every 2, 192 swaps, decay 0.92) | 3.2 | 8,859 | 8.34 |
| `--vram-reserve-mib 512` (5,102 slots) | 3.8 | 5,630 | 7.86 |
| tuned + 5,102 slots | 2.1 | 5,198 | 8.27 |

No tested setting improved on the default; the differences are within the observed run-to-run variation (about 0.4 tok/s).
A round took about 100 ms of GPU time with either 2 or 4 CPU experts per layer. These runs do not isolate the remaining bottleneck.

**Why the GPU-only path was slow, found with unitrace.** `unitrace -d` (build it from `intel/pti-gpu`, tools/unitrace, as
`sycl/tools/Dockerfile.unitrace` does) prints the device time per kernel; the decode rounds are the difference between a short
and a long run. The GPU was 88% busy: 141 ms of kernel time in each 161 ms round, of which 78 ms (55%) was the routed-expert
kernels (`native_gu_port`, `native_down_port`), 0.85 ms per layer-token for the 17 MB of expert weights it touches (from VRAM
that is 0.03 ms): they were reading the mirrored experts over PCIe at about 9 GB/s. With `STRATA_VERIFY_NO_HOST` nothing on the
host sees the routing, so the adaptive tier (which lifts the hit rate from a third to 87%) had nothing to rank. The plan kernel
now counts the routed experts on the GPU (`resident_plan_set_usage`) and the engine feeds the counts to the adaptive tier:

| GPU-only, per decode round | before | after |
|---|---|---|
| kernel time | 141 ms | 100 ms |
| expert kernels (gu_port + down_port) | 78 ms | 31 ms |
| `native_mmvq_q6k_wide` (129 calls) | 17.7 ms | 17.7 ms |
| `fused_gr_read_multi` (hyper-connection reads) | 24 ms | 24 ms |
| `gdn_step_norm_multi` | 4.2 ms | 6.7 ms |
| decode, 128 tokens | 7.14 tok/s | 9.76, 9.74 tok/s (9.77 at 256 tokens) |

What is left is spread out: the Q6_K projections run at about a quarter of the card's memory bandwidth (137 us per call for
about 21 MB), `gr_up_multi` takes 159 us for a small projection, and the GPU idles for about 16% of the round between kernels.

**Kernel work on the GPU-only path: what was tried (A770, 2026-10-04).** After the expert kernels, the biggest item in the 100 ms
round is `native_mmvq_q6k_wide` (the Q6_K projections and the LM head: 129 tensors, about 1.9 GB per round, 17.7 ms, about 106
GB/s against a measured 426 GB/s streaming ceiling with 16-byte loads). Measured with `mmvq_bench` on the model's real shapes
(2560x10240, 2560x6144, 6144x2560, 2560x12288, and the 2560x248320 head; 2 columns, GB/s of weights):

| variant | 10240 | 6144 | 6144x2560 | 12288 | head |
|---|---|---|---|---|---|
| default (aligned-load `a2` kernel), no large-buffer option | 164 | 134 | 154 | 171 | 243 |
| the same linked with `-ze-opt-greater-than-4GB-buffer-required` (`STRATA_SYCL_LARGE_BUFFERS=ON`) | 142 | 113 | 130 | 149 | 220 |
| misaligned loads (`STRATA_MMVQ_A2=0`) | 117 | 103 | 61 | 123 | 146 |
| weight loads issued ahead, `STRATA_MMVQ_PF=2 / 3 / 4` | 129 / 105 / 64 | 113 / 95 / 59 | 127 / 110 / 70 | 136 / 110 / 65 | 167 / 132 / 76 |
| 2 / 3 / 4 rows per sub-group, aligned loads | 51 / 41 / 33 | 60 / 49 / 41 | 99 / 77 / 54 | 53 / 39 / 30 | 53 / 43 / 34 |
| the shared kernel (`STRATA_MMVQ_WIDE=0`) | 62 | 60 | 61 | 63 | 65 |
| blocks repacked at a 224-byte stride (`q6k_align_bench`) | 186 vs 162 | | 142 vs 102 | 162 vs 101 | 267 vs 135 |

- The repacked 224-byte stride is available as an opt-in (type tag 114, below). The other variants in this table did not
  improve on the aligned-load kernel in these runs.
- `dp4a.cpp` reported 730 G dp4a/s for `strata::dp4a` and 2,654 G/s for `__builtin_IB_dp4a_ss`. This is not an isolated
  instruction-throughput measurement: correlated operands let the emulated form share byte extraction, only the sum escapes,
  and the opaque builtin changes compiler optimisation. Swapping the builtin into Q6_K gave the same reported GB/s in that
  build. Going from 1 to 2 columns raised kernel time 1.6x for the same weight bytes. Neither observation establishes whether
  dp4a throughput limits this kernel. Load latency, instruction issue and occupancy remain hypotheses; ISA inspection
  (`ocloc disasm`, including dp4a instructions, sends and spills) and controlled measurements are still missing.
  A random-input comparison found 12 differences in 4 million inputs near int32 accumulator overflow; it does not establish
  equivalence for all inputs.
- The large-buffer AOT option reduced throughput in the table. A whole-decode comparison with a cache under 4 GiB measured
  6.84 vs 6.73 tok/s (1.6%, same output tokens). That single comparison does not establish the value of splitting the cache
  into smaller arenas to avoid the option.
- XMX's value for decode is unresolved. Earlier fork notes reported int8 `joint_matrix` expert variants 1.4-3x slower than
  dp4a and one GPU hang, but no reproducible configuration is recorded here, so those reports cannot establish a general
  limit. On DG2, the fp16 `joint_matrix` kernels and `xmx_int8_bench` fail during IGC compilation with a floating point
  exception. That prevents a comparison on this A770 build; it is not evidence that XMX cannot help decode or prefill.

**Q6_K served from a repacked layout (opt-in).** Q6_K blocks occupy 210 bytes. Set `STRATA_Q6K_REPACK=1` before loading
weights to pad them to 224 bytes (6.7% more Q6_K storage) for aligned weight loads. The switch is read once in
`native_mmvq_serving_type`; `native_dense.cpp` and `native_head.cpp` use that decision for allocation and packing.
Unset, `0`, or any value other than `1` keeps tag 14 and its existing `STRATA_MMVQ_WIDE` selection. The old
`STRATA_Q6K_STRIDE224` switch is replaced by this opt-in. The serving image enables it explicitly.

The repacked tag 114 uses `native_q6_k_mmvq_s224`, bypassing tag 14's wide-kernel selection. It launches at most 4 columns
at a time; the original single-launch 5/6-column experiment measured 0.14x. The prefill dequantiser (`dequant_bf16.dp.cpp`)
also accepts tag 114. `mmvq_bench --selftest` now checks tag 114 against unpacked tag 14 with wide mode both off and on,
using per-output tolerance `abs(candidate - reference) <= 1e-3 + 1e-5 * abs(reference)`. Non-finite results or mismatches
return a non-zero exit status. The fixed cases cover columns 1-6, 17 output rows and 1, 3, 4, 5 and 10 blocks per row,
with deterministic synthetic weights and finite scales. FP32, BF16 and FP16 prefill dequantisation are compared for exact
finite values at row offsets 0 and 2. This checks layout parity, not agreement with an independent mathematical oracle.
The expanded checks have not yet been run on a GPU; the earlier benchmark only printed differences and did not enforce
parity, so it did not establish bit-identical results. With `STRATA_SYCL_PARITY=ON`, run:

```sh
cmake --build build-sycl-aot --target mmvq_bench
ctest --test-dir build-sycl-aot -R '^q6k_repack_parity$' --output-on-failure
```

Normal benchmark invocations also run the fixed parity cases before checking and timing the requested shape.
Historical A770 timing measurements (not rerun with the expanded checks):
`mmvq_bench` against the old kernel in the same build, 2 columns: 1.20x (2560x10240), 1.26x (2560x6144), 1.32x (6144x2560),
1.20x (2560x12288), 1.12x (the 248,320-row head), about 1.3x at 4 columns, 0.93x at 1. Whole GPU-only decode, 128 tokens, same
tokens: **10.37 and 10.36 tok/s against 9.77 and 9.76** (+6.2%); load time unchanged (25-26 s of wall time either way; the
repack of about 1.9 GB is a fraction of a second); the weights grow from 2,019 to 2,105 MiB.

**The hyper-connection read (`fused_gr_read_multi`), looked at and left alone.** It is about 24 ms of the 100 ms round (norm,
down, reduce and up kernels back to back; `gr_bench` on the A770: 153-220 us for 1-6 tokens, the B70's notes say 76-108 us at 6).
`unitrace` on `gr_bench` puts the up kernel at the top, but its token-mixed average overstates it. An ablation of the up
kernel at 2 tokens (total 166 us): skipping its weight loads saves 35 us (that is its memory work, about 190 GB/s), skipping the
prologue, the epilogue loads or the stores about 4-5 us each, skipping all three loads leaves 95 us, which is the other three
kernels and the gaps between the launches. The tile width (`UPM_COLS`) gives 166 / 304 / 237 / 152 us at 16 / 8 / 4 / 2 (bit
identical); `STRATA_GR_V3=1` has no effect on the port's path, `STRATA_GR_DOWN_SLICED=0` is 1.5x slower, `STRATA_GR_DOWN_DIRECT=0`
is the same. The sliced down kernel spills about 38 registers (the link log says so for SIMD32 at 128 registers) and
`gdn_step_norm_multi` also warns. A real gain would come from merging the norm and down stages and shortening the chain of four
launches, worth about 6 ms of the round; not done.

**Adaptive swaps (fixed).** The default `--adapt-every 4` used to zero the output after the first swap. The swap itself was
sound: the device residency table matched the host's and the swapped-in slots were byte-identical to the GGUF. An evicted expert
had never been mirrored in pinned host memory, so the GPU could not read it and the device plan waited for the host, whose flags
the GPU could not see (above). Now the waits work, and mirroring the resident experts too (after the real misses, within the RAM
cap; `STRATA_MIRROR_ALL=0` turns it off) is a speed matter: the swap costs 0.3 ms per round with it and 6.6 without, and the
decode is 8.0-8.6 against 7.3-7.5. It costs about 9 GiB more pinned memory on the 16 GB card. Separately,
`GgufExpertSource::blob()` published a ring slot before its read had finished; that race is fixed too but was not the cause.

**Re-checked on engine 0.1.38-sycl (2026-10-04),** after merging upstream's b70 up to 25277f9 into this branch: GPU-only decode 10.37 and
10.36 tok/s (128 tokens, the same tokens as before), host plan 8.66; the previously reported test failures remain unresolved
(listed below); `STRATA_GR_DOWN_MAX4=1` makes no difference here. `xmx_int8_bench`, upstream's int8 DPAS
prototype, makes IGC for dg2 die with a floating point exception, like the fp16 `joint_matrix` kernels, so it is built only
without `STRATA_SYCL_NO_XMX`; upstream measured it 0.38-0.98x of dequant + oneMKL on the B70 and not worth an engine port.

**Against llama.cpp in llama-bench's shapes (2026-10-04).** The same Coder IQ1_M GGUF on the same A770 (xe driver, host Xeon E5 v4),
the card free, one session. llama.cpp b11223 SYCL through `llama-bench -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ncmoe 48
-ot per_layer_token_embd.weight=CPU -lm none -ub 2048 -r 2` (the experts on the CPU, the best flags found for it). Strata GPU-only
(`STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1`, `--stream-experts --expert-cache auto --vram-reserve-mib 1024 --kv int8
--spec 2 --suffix-draft 0`), engine 0.1.38-sycl with this branch, driven through its CLI because it has no llama-bench: real text
from this repo's docs, tokenised with the model's tokenizer; each shape is its own process with its own model load.

| tok/s | llama.cpp | Strata | |
|---|---:|---:|---|
| pp512 | 121.6 +- 0.8 | 115 (the first 512-token chunk; 113 with the 512th token) | 0.95x |
| tg128 | 9.58 +- 0.04 | 10.89 (10.88, 10.91) | 1.14x |
| pp512 at depth 8192 | 117.6 +- 1.5 | 131 (134, 128) | 1.11x |
| tg128 at depth 8192 | 8.54 +- 0.78 | 10.46 (10.46, 10.46) | 1.22x |
| pp20000 (cold) | 223.7 | 312 (64.0 s) | 1.40x |
| tg64 at depth 20000 | 6.95 | 8.35 | 1.20x |

How the Strata numbers were taken: pp512 is a 512-token prompt in one 512-token chunk (`--prefill 512`); pp512 at depth 8192 is the
difference between prefilling 8,704 and 8,192 tokens in 512-token chunks (the last chunk, at depth 8192); tg is the decode rate
over the generated tokens after a prompt of that depth (so it includes the adaptive tier's first swaps, as llama-bench's tg128 includes
its own warm-up); the 20k prompt uses the engine's own chunking. The first chunk of a prompt costs about 0.5 s more than the others,
which is why pp512 at depth 0 is below the steady 132 tok/s of the longer prompts; llama.cpp's pp512 shows no such start.

**Open.**
- Host plan with `--adapt-every 0` produced wrong tokens once (from the second token, the CPU computing 10 experts per layer) in
  three runs; two repeats were correct. Not explained.
- Uncached host memory cost 42 ms per window on the GPU-only path in a build that made only the flag words uncached (the plan and result rows were left plain). Not explained; upstream's allocator makes every polled buffer uncached, which was not re-measured separately.
- Per-kernel timestamps read zeros (`%globaltimer`), so the profile above is host-timed and per stage.
- The prompt path needs `iq_dequant_f16` and the oneMKL GEMMs; `conversation_snapshot_test` segfaults (not investigated).
- `kv_hybrid_parity` and `qsa_prompt_attn_parity` fail with `STRATA_SYCL_NO_XMX=ON` because they require the XMX kernel.
- `s2_expert_grouped_parity` fails on the unused s2 path. The earlier claim of five failing tests did not identify a fifth;
  a complete named A770 test log is still needed.
- DG2 IGC compilation crashes on `xmx_gemm_iq`, `qsa_prompt_attn_xmx` and `xmx_int8_bench`; these paths are excluded by
  `STRATA_SYCL_NO_XMX=ON`, not validated by it.
- The expanded `q6k_repack_parity` checks above await a GPU run.

## Long prompts decode to token 0 with a borrowed cache over 4 GiB (root cause found 2026-10-04)

- **Symptom.** In serve mode with an expert cache over 4 GiB, a prompt of about 1.5k tokens or more answers with token 0 forever
  ("!!!!"), every draft accepted. Short prompts are fine. `--no-prefill-borrow` fixes it; so does a cache under 4 GiB.
- **Cause.** The prompt path carves its working buffers out of the cache's tail slots. In a big cache those are more than 4 GiB
  into the allocation, and oneMKL's bf16/f16 to fp32 GEMM (`dpct::blas::gemm`, which `Gemm::bf16/f16/native` call) returns
  zeros when an input matrix (activations or weights) starts 4 GiB or more into an allocation. The output position does not
  matter. Small GEMMs (chunks up to about 1024 tokens) use another kernel and are correct, which is why the loan size seemed to matter.
  The zeros turn into NaN a few operations later.
- **Not the cause.** 32-bit offsets in Strata's own kernels (the large-buffer AOT flag covers them), memcpy/memset, plain
  kernels, and fp16 oneMKL GEMMs of 256^3, all correct at every offset up to 8.2 GiB. `SYCL_PROGRAM_COMPILE_OPTIONS` does not change
  it: oneMKL's kernels are prebuilt. Zeroing the loan, and giving the KV staging its own allocation, do not help.
- **Reproduce.** `sycl/tests/repro/mkl_gemm_4gib.cpp` (build line in its header).
- **Mitigation.** `--no-prefill-borrow` uses separate prompt buffers. With an 8.2 GiB
  cache and its own prompt buffers, decode was 13.7-14.3 tok/s and 30k-token prompts were correct.
- **Cache layout fix (GPU validation pending).** The SYCL cache now reverses slot addresses automatically when its allocation
  exceeds 4 GiB. Profile rank stays the slot index: the coldest slots are at the lowest addresses. Loans start at offset zero
  and include at most `floor(3.9 * 2^30)` bytes, including whole-slot padding. A larger request uses a smaller chunk or the
  existing own-buffer fallback. `STRATA_CACHE_REVERSE=1` forces this layout for small caches; `STRATA_CACHE_REVERSE=0` disables
  it. The warning now checks whether the actual loan extends past 4 GiB. The large-buffer compiler option is still needed
  for Strata's own kernels to read experts at higher addresses.
- **Host test.** `cmake -S sycl/tests -B /tmp/strata-cache-tests && cmake --build /tmp/strata-cache-tests && ctest --test-dir /tmp/strata-cache-tests --output-on-failure`
  checks uniform and sized layouts, loan limits, and fill/loan/refill addressing without SYCL or a GPU.
