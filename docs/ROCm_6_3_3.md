# ROCm 6.3.3 / gfx906 — system-install report (2x MI50)

**Tree:** upstream `main` @ `2af30ccaa` (engine 0.1.67) **plus** the three-line `hip_compat` P2P mapping in PR #7088 — without it the compat build stops on undeclared `cudaDevP2PAttrNativeAtomicSupported` / `cudaDeviceGetPCIBusId` in `src/core/peer_experts.cpp`. **Cards:** 2x AMD Instinct MI50 32 GB (gfx906) | **System:** Ubuntu 24.04, kernel 6.8, ROCm 6.3.3 at `/opt/rocm-6.3.3` (clang 18), 2x Xeon E5-2630 v3 (16C/32T), 256 GB DDR4

**Power state:** every capture below was taken at **225 W per card**, which on these boards is `power1_cap` = `power1_cap_max` = `power1_cap_default`. There is no higher setting: `pp_od_clk_voltage` does not exist for either card under ROCm 6.3.3, so 225 W is a firmware ceiling, not a tunable, and no cap sweep is possible on this hardware. Read `power1_cap` at run start rather than assuming it.

**Power telemetry note:** this ROCm 6.3.3 build does not report average power (`rocm-smi`: `get_power_avg`, "Not supported on the given system"); instantaneous draw comes from `power1_input`, and the cap from `power1_cap` under `/sys/class/drm/cardN/device/hwmon/`.

---

## 1. Building on a system ROCm 6.3.3 (no container)

The 6.3.x line still ships the gfx906 libraries, so the build works from a plain system install. CMake fetches the pinned ggml commit itself, so no `STRATA_GGML_DIR` is needed:

```sh
cmake -S engine -B build-906 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm-6.3.3 \
  -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER=/opt/rocm-6.3.3/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.3/lib/llvm/bin/clang++
ninja -C build-906 strata
```

Notes that cost time to find:

- `-DCMAKE_PREFIX_PATH` is not optional in this layout; without it the `hipblas-common` lookup fails.
- A **failed first configure poisons the build directory** — the RPATH error that follows is a symptom, not a cause. Delete the directory rather than adding `-DCMAKE_BUILD_RPATH*` flags.
- The build carries the upstream warning set and is not `-Werror`.
- Two source-level blockers must be applied first on this tree: the `__shared__ alignas(16) int4` class (#4151, fixed upstream as `804003950`) and the `hip_compat` P2P names (#7088). With those, `ninja -C build-906 strata` links, a 28.6 MB binary.

## 2. Same-day run (2026-11-25, 225 W = the ceiling)

Identical request throughout: 122-token prompt, 256 generated, temperature 0, `--batch 2`, `--spec 4`, `--max-context 262144`, `--kv-resident 32768`, `--pool-workers 15`; one warm-up request discarded; decode tok/s from the engine's own per-request line, cross-checked against client wall time. Clocks, power and junction temperature sampled every 0.5 s.

| config | engine decode tok/s (3 runs) | median | client wall median |
| --- | --- | --- | --- |
| layer split, 2x MI50 (`gpu [0,1]`, `layer_split auto`) | 24.8 / 17.9 / 23.2 | **23.0** | 22.96 |
| single MI50 (`gpu [0]`) | 17.3 / 18.3 / 15.4 | **17.3** | 17.07 |

**The split arm is bimodal, and this is the honest reading of it.** Fast/slow/fast reproduced across two sessions on different days. The slow run is not distinguishable from the fast ones by any GPU signal: all runs reached the top DPM level (1725 MHz), peak draw 141–192 W of the 225 W ceiling, junction 46–55 °C, expert-cache hit rate flat at 96.3–96.4 %, zero file-tier reads with 0 major faults, KV streaming 99.7–99.9 %, draft acceptance flat at 82–87 %. Two candidates remain, both host-side: the pageable resident complement (27.27 GiB, and every start logs `NUMA placement FAILED: the complement is not page-locked`) taking a memory event mid-decode, or the peer-expert handoff occasionally not engaging. **Do not quote a split speed-up on this evidence** — the median advantage is smaller than the spread of the winning arm.

The single-GPU arm also moved between days (21.50 median on 2026-11-24, 17.07 on 2026-11-25) at an identical 7,499-slot expert cache, with healthy clocks and the cap not binding. Unexplained; treat single-GPU absolute numbers as day-dependent, while the two arms remain comparable to each other because they ran back to back in the same conditions.

Memory placement, measured at load:
- **Split:** CUDA1 took 25.94 GiB of experts (`--vram-reserve-mib` is gone in 0.2.x, replaced by the elastic segment scheme, and the freed reserve goes to expert slots); 810 MiB (CUDA0) / 908 MiB (CUDA1) free after load.
- **Single:** resident RAM mode — 62.78 GiB of experts pageable in RAM, 7,499 slots (21.89 GiB) in VRAM, 810 MiB free after load.

Boot to serving: **~12.2 min from a cold page cache** (2.5 s startup, 270.4 s profile pre-fill, 458 s for the 62.78 GiB pageable complement at ~87 MB/s), against **~90 s warm** for the split engine and ~1.7 min warm for single. This box's system disk is DRAM-less, so cold boots measure the disk rather than the engine; warm the cache before timing anything.

## 3. Historical: the 0.1.45 tree (October)

The previous revision of this report documented the 0.1.45 base (`477ac13c`) in full: the HIP compat layer's per-file port list, serve-mode A/Bs, the dual-GPU layer split, and a **12 h continuous resident-serve soak on the 256K-context / 2-batch-slot split config — no crashes, no restarts, no degradation**. Its warm decode numbers: 21.7–22.6 tok/s split, 17.2–18.4 tok/s single. Those numbers were taken under a lower cap and are historical; they are not comparable to §2 and are not restated as current.

Of that tree's engine fixes, 0.2.16/0.2.17 absorbed the shared `alignas(16)` class (#4151, fixed upstream as `804003950`), the f16-range reset check (#4152, `04dd856ea`), and setup's gfx906 recognition (#4153). qwen3.6 left the gfx906 build after 0.1.51, which retired the rest.

## 4. Cross-validation on three architectures

The compat work behind this report was built and smoke-run on three different architectures, so the gfx906 result is not a single-machine curiosity. Those three legs were built from the branch carrying the gfx900/gfx906 fixes at `75718ba21` (`d46f5f268` plus seven commits). The numbers in §2 are from `main` @ `2af30ccaa` on gfx906, which is five commits later plus the #7088 mapping.

| card | arch | build | what ran |
| --- | --- | --- | --- |
| 2x Instinct MI25 | gfx900 | 172/172 | inkling lane runs; the code object disassembles to 0 matrix-core and 0 dot-product instructions, so the scalar fallback is the code that ran |
| 2x Instinct MI50 | gfx906 | 178/178 | inkling gives identical greedy sequences across processes; Hy4-Flash produces coherent prose (2.34 tok/s on one card) with `hy4_no_wmma()` true |
| Tesla V100 | sm_70 | 169/169 | Qwen3.6 lane runs, coherent: 1,301.7 tok/s prefill, 88.6 tok/s decode; `--families` lists both qwen4exp and qwen35moe |

Two things this established that the gfx906 build alone could not. The CUDA build graph is undisturbed — both model families are present on sm_70, where Qwen3.6 is native and no `STRATA_HIP_GFX906` exclusion applies. And on gfx900 the scalar fallback is not simply available but is the code that executes, checked by disassembly rather than assumed.

A pre-sm_75 CUDA build of this tree needs `-DSTRATA_EXPERIMENTAL_SM60=ON` and a CUDA 12.x toolkit; CUDA 13 dropped sm_70 and fails the compiler test at `compute_70`. That is a property of the tree, not of this branch: a control build of stock `d46f5f268` fails identically.

## 5. Not tested

- **No soak on this tree.** The 12 h soak in §3 ran on 0.1.45; nothing of that length has been run on engine 0.1.67, so leak, drift and sustained-heat behaviour of the current tree are unmeasured.
- Multi-GPU beyond `layer_split auto` on this pair; the lane-select and prefill-split paths.
- SYCL builds and Mac/Metal. The CUDA leg was built and run on sm_70 (§4), but no CUDA build was made on this box.
- `-Werror`.
- The standard benchmark protocol (`docs/BENCHMARK_PROTOCOL.md`) — not yet run on gfx906.
