# Intel Arc A770: driver, runtime and compiler issues found running Strata

What went wrong on an Arc A770 (Alchemist, DG2-G10, 16 GB) with Intel's compute stack while bringing up the SYCL port, what was
established about each problem, and what works around it. It is a list of things to report or avoid, not a how-to; the bring-up story and measurements are in [INTEL_A770.md](INTEL_A770.md). Most entries
came with a measurement; where one is a report from another contributor or an inference, it says so.

## Configuration these were seen on

| | |
|---|---|
| GPU | Arc A770 16 GB, PCI `8086:56a0`, AOT target `dg2-g10`; Resizable BAR on (16 GiB BAR2) |
| Host | PCIe 3.0 x16 root port, an AVX2-only Broadwell-class Xeon, 120+ GiB RAM |
| OS / kernel | Ubuntu 26.04 LTS, Linux 7.0 (`7.0.0-34-generic`), `linux-firmware 20260319` |
| Kernel driver | `xe`, forced (below) |
| Compute stack (in a container) | oneAPI DPC++/C++ 2026.1.0-235, oneMKL 2026.1.0-236, NEO (`libze-intel-gpu1`, `intel-opencl-icd`) 26.35.39758.10, IGC 2.41.5, Level Zero loader 1.32.0, gmmlib 22.10.0 |
| Run environment | `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`, `ZES_ENABLE_SYSMAN=1`, `IGC_EnableDPEmulation=1`, `OverrideDefaultFP64Settings=1`, `SYCL_CACHE_PERSISTENT=0` |

The workload is a large MoE model whose experts are streamed from pinned host memory over PCIe to a 9 GiB cache in VRAM, with a
GPU kernel that polls flags the host raises. That pattern finds problems an ordinary SYCL program does not.

## Using the `xe` driver

`xe` does not claim the A770 by default (its PCI table has no `56a0`) and `i915` takes it first. Force it with kernel parameters,
and make them the default boot entry (a one-shot entry means a reboot after a power cut silently returns to `i915`):

```
# /etc/default/grub.d/xe.cfg
GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT xe.force_probe=56a0 i915.force_probe=!56a0"
sudo update-grub && sudo reboot
readlink /sys/class/drm/card0/device/driver      # .../xe
```

Why bother: on llama.cpp's SYCL backend (b11223, 9B Q6_K with MTP) decode was 40-48 tok/s on `xe` against 27-32 on `i915`; prompt
processing was about 3% faster and VRAM use identical. Strata has only been run on `xe`.

Differences you will notice on `xe`:

- `intel_gpu_top` does not work (it needs the `i915` PMU). Read the card from sysfs (`.../tile0/gt0/freq0/act_freq`, `gtidle/idle_status`,
  hwmon) and per-process VRAM from `drm-resident-vram0` in `/proc/<pid>/fdinfo` (root only).
- Kubernetes' Intel device plugin advertises `gpu.intel.com/xe` (and `gpu.intel.com/i915` falls to 0); a pod requesting the old name will not start.
- Observed, not investigated: HuC does not load under `xe` on this card (`huc_info` says so). Compute is unaffected.

## Issues, by component

### Kernel driver (`xe`)

1. **A long device spin can wedge the card until reboot.** A kernel polling a flag for hundreds of millions of iterations trips the driver's
   hang detection; it times the queue out and resets the GT, and with a window of about 2,400 graph nodes the reset cascades. The card stayed wedged twice.
   Work-around: bound every spin (here about 20,000 polls, 22 ms) and fail loudly. A 200-million-poll test was aborted before it could do it again.
2. **A work-group barrier after some work-items have returned never completes.** The first decode window hung, GPU busy, until `xe` reset the
   compute engine. This is undefined behaviour in SYCL, but the symptom is a driver reset rather than an error. Removing the barrier fixed it.

### Level Zero and the compute runtime (NEO)

3. **Allocation caps.** One device or host allocation is capped at 4,095 MiB unless `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`, which
   lifts the device cap (9 GiB arenas work). A single *host* allocation stays capped near 4 GiB even with it; the pinned mirror is therefore a list of chunks of at most 3 GiB.
4. **A kernel polling `sycl::malloc_host` memory never sees a host store made after the kernel started.** 60 million system-scope atomic loads (and every
   other load variant) saw nothing, and the wait gave up however long its bound. The data the host was meant to supply stayed missing and the model produced garbage.
   Memory from `zeMemAllocHost` with the *uncached* or *write-combined* flag was seen within microseconds in every trial; the same call without
   flags was not reliable. (Making the plan and result rows uncached as well cost 42 ms per decode window in one build; not explained.)
5. **Level Zero v2 adapter: a host wait on an event stops releasing another queue's barrier that lists it.** The GPU then waits forever
   (a prompt that streamed more than 16 experts per layer, and a PLE upload past about 4K tokens). The v1 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) does not do this.
   Work-around: the copy queue writes a sequence number into pinned host memory and the host polls it.
6. **Free-memory queries fail on `xe`.** The adapter did not answer them, so `--expert-cache auto` saw 0 GiB free and made no slots. Asking Level Zero Sysman
   directly (`ZES_ENABLE_SYSMAN=1`) works.
7. **Pinned-host to pageable copies hang the copy engine** (reported by another contributor on a B70, not seen here; the port routes host-only copies through the CPU).
8. **`SYCL_CACHE_PERSISTENT=1` segfaults (exit 139)** on the first compile in the runtime used here.
9. **First-run JIT stall.** With an empty kernel cache the first request of a JIT-compiled program stalled 185-283 s while the driver compiled kernels
   (17 s with NEO's own cache persisted). Ahead-of-time compiled kernels do not pay it.
10. **Per-kernel timestamps read zeros** (`%globaltimer`), so GPU-side profiling has to be host-timed.
11. **Over-subscribed VRAM is not refused.** On `i915` the driver evicted other processes' buffers to system RAM; Level Zero reports running out of
    VRAM as `UR_RESULT_ERROR_OUT_OF_HOST_MEMORY`. Do not share the card.

### Graphics compiler (IGC) and DPC++

12. **No fp64 on the A770.** An AOT build that uses `double` fails with "Double type is not supported on this platform". `IGC_EnableDPEmulation=1`
    plus `OverrideDefaultFP64Settings=1` emulates it, and it is needed at compile time (the CI image sets it) as well as at run time.
13. **IGC crashes on `joint_matrix` (XMX) kernels for DG2.** `ocloc` died with a floating-point exception on three kernels (`xmx_gemm_iq`,
    `qsa_prompt_attn_xmx`, `xmx_int8_bench`); fp16 `joint_matrix` was the common factor. Three int8 variants that did compile were 1.4-3x slower than
    scalar dp4a and one hung the GPU. The port builds with `-DSTRATA_SYCL_NO_XMX=ON`. No existing report was found.
14. **Buffers over 4 GiB are addressed with 32-bit offsets unless told otherwise.** `-ze-opt-greater-than-4GB-buffer-required` is needed for the
    device code; without it a resident expert 5.96 GiB into the cache dequantised to NaN while the same expert in a small buffer was fine. In a standalone test the
    failing case was exactly "base pointer plus a large offset computed inside the kernel".
15. **fp32 divide and sqrt are not correctly rounded by default** (OpenCL allows 2.5 ulp). A quantiser that must match a CPU reference byte for byte
    had 303k mismatches until `-cl-fp32-correctly-rounded-divide-sqrt` reached `ocloc` (it must be passed to the AOT backend explicitly).
16. **The device compiler does not honour type punning through a wider type** (a `uint16_t*` read as `int2`); extract with shifts.
17. **Warp-style code needs `-fsycl-default-sub-group-size=32`;** the default on this hardware is 16.

### oneMKL

18. **The mixed-precision GEMM (bf16 or f16 in, fp32 out, through `dpct::blas::gemm`) returns zeros when an input matrix starts 4 GiB or more
    into a device allocation.** The output position does not matter; small GEMMs take a different kernel and are correct, so it only shows at larger sizes
    (about 1.5k-token chunks here). `SYCL_PROGRAM_COMPILE_OPTIONS` does not help because the kernels are prebuilt. Reproducer: `sycl/tests/repro/mkl_gemm_4gib.cpp`.
    **Not yet reported to Intel.** The zeros become NaNs a few operations later, which is how it first appeared (the model emitted token 0 forever).
    Work-around in Strata: lay the expert cache out in reverse so the buffers handed to the GEMM start in the first 4 GiB.
19. **A decode with the asynchronous commit hangs after 100 to 600 windows.** The default of `set_commit_async` (the commit graph is launched,
    an event is recorded after it and replaced every round, and the host later waits on it) stopped for good in every configuration tried: adaptive swaps off,
    host plan, another reserve, the shared-expert fork off (`STRATA_SH_STREAM=0`), `UR_L0_DISABLE_EVENTS_CACHING=1`. The process
    sleeps with every thread idle, the card stays at full clock, and the kernel log is silent (no xe reset or timeout); the process cannot be
    killed until the pod is deleted. In the xe debugfs (`/sys/kernel/debug/dri/<pci>/gt0/hw_engines`, `.../gt0/stats`) the compute engine
    `ccs0` shows `RING_HEAD` different from `RING_TAIL`, an `ACTHD` that does not change between samples and `IPEHR` 0x0e00d003 (an
    `MI_SEMAPHORE_WAIT`), while the copy engine `bcs0` is idle; `gtidle/idle_residency_ms` does not advance. So the compute stream is parked
    in a semaphore wait that nothing signals.
    **Cause, narrowed:** the default adapter on this stack is Level Zero **v1** (`sycl::platform` name "...Level-Zero", no "V2"; `SYCL_UR_USE_LEVEL_ZERO_V2=0` changes
    nothing). A standalone program (in-order queues with profiling, as dpct creates them: replay a 2,000-node graph, wait on the host, replay a
    100-node graph, record a barrier event with `ext_oneapi_submit_barrier()` and drop it, run a kernel on a second queue) hangs within about 100 rounds on v1; it finishes 6,000 rounds with
    `SYCL_UR_USE_LEVEL_ZERO_V2=1`, with `UR_L0_USE_DRIVER_INORDER_LISTS=1`, or with out-of-order queues, and it hangs with `UR_L0_DISABLE_EVENTS_CACHING=1`.
    Recording no event at all also finishes. A hang of the same kind is reported in intel/llvm#18424 (A770, v1 adapter, discarded events). The mechanism inside the adapter or driver is not known.
    **Work-arounds, in the engine** (3,000-token decode on an 8,191-token prompt, then on a 19,999-token prompt, identical tokens to the synchronous build for the first):
    the synchronous commit (`STRATA_COMMIT_SYNC=1`, now the default of the SYCL port; `STRATA_COMMIT_ASYNC=1` selects the asynchronous one) finished every run;
    the asynchronous commit finished with `UR_L0_USE_DRIVER_INORDER_LISTS=1` (four runs, two prompt sizes) and with `SYCL_UR_USE_LEVEL_ZERO_V2=1` (one run).
    The synchronous commit is not slower on a 128-token decode (19.1 against 18.4 tok/s on a short prompt, 15.6 against 14.7 on an 8k prompt, against the asynchronous commit with the in-order-list variable).
    The two modes do not always produce the same tokens: on one short prompt the synchronous run diverged at token 25 from every asynchronous run, which agreed with each other; draft acceptance differs (88 of 110, 94 of 102).

## Platform limits worth knowing

- PCIe 3.0 x16 is the ceiling for streaming experts (about 9.4 GB/s measured from a kernel and by DMA on the same host).
- The A770 has one compute queue and one copy queue, 512 EUs and 64 KiB of local memory per work-group.
- Resizable BAR should be on; the card's own PCIe bridges show `2.5 GT/s x1` when idle, which is not the link to the host.

## To check whether you hit the same things

1. `readlink /sys/class/drm/card0/device/driver` is `xe`, and the kernel command line carries both force-probe parameters.
2. The container sees `gpu.intel.com/xe` (or `/dev/dri`) and runs with the environment in the table above.
3. A prompt of 3,000 or more tokens answers sensibly. If it answers `!!!!`, suspect item 18 (or 14, on a build without the large-buffer flag).
4. Decode stalls with the card at 100% and no error point at items 1, 2, 5 or 19.
