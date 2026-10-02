# Strata ROCm/gfx1100 Optimisation Plan

Target: RX 7900 GRE (RADV NAVI31, gfx1100, wave32, 16 GB VRAM).
Each item is self-contained; apply independently.

---

## 0. Verification pass (read this first)

Every item below was checked against the code and, where the claim was quantitative, measured on the
target card. **Four of the eight original items did not survive.** The numbers in this document come
from:

- `hipGetDeviceProperties` on the actual device: `AMD Radeon RX 7900 GRE`, `gcnArchName=gfx1100`,
  **40 CUs**, `warpSize=32`, `maxThreadsPerMultiProcessor=2048`, `sharedMemPerMultiprocessor=64 KB`,
  `clockRate=2052 MHz`, 17.2 GB.
- Device instruction counts per translation unit, by compiling each `src/kernels/cuda/native_*.cu`
  with the exact `strata_kernels` HIP line from `build-hip/build.ninja` (`DEFINES`/`INCLUDES`/`FLAGS`,
  plus `-x hip -S`), with and without the flag under test, and counting emitted
  `s_*`/`v_*`/`ds_*`/`buffer_*`/`global_*`/`flat_*` instructions.
- Register allocation and spill counts, from the `.amdhsa_next_free_vgpr` directive in the same
  assembly and a count of `scratch` references.
- The running workload's own log, `strata-coder-iq1_m.log`.

The single most consequential finding: **the two items the original plan led with both target
`native_flash_attn.cu`, which never executes in this workload.** `native_flash_attn_short_step`
rejects `max_context > 256` (`src/kernels/cuda/native_flash_attn.cu:199`), and the logged sessions run
at contexts of 4210–9071 tokens. Measured contexts from the log: `prompt 4210 tokens`,
`prompt 8830 tokens`, `prompt 9071 tokens`. So any gain attributed to that file is zero, and the real
hot path is the generic `qsa_decode_attn_step` (`src/core/layer.cpp:942`).

### What changed from the original plan

| Item | Original claim | Verdict |
| --- | --- | --- |
| OPT-1 | `--use_fast_math` on 5 files, 15–30% on attention | **Flag does not exist for HIP.** Rewritten as OPT-1 with the real flag and a measured file list. Applied, but then found to silently delete input validation — see OPT-1's blocking defect. |
| OPT-2 | `__launch_bounds__(128, 4)`, 10–25% | **Measured regression.** Dropped; the real defect it brushed against is now OPT-2b. |
| OPT-3 | Verify hipBLASLt is linked | **Already satisfied.** No code change. Kept as a re-verification checklist. |
| OPT-4 | `cudaMemcpy2DAsync` split, 0.3–0.5 ms/token | Real but ~10x overstated. Gain restated. |
| OPT-5 | Verify graph capture is always on | **Already satisfied.** Optional log line kept. |
| OPT-6 | `std::atomic<bool>` in MoE combine | **Impossible as written** — host-side object. Withdrawn. |
| OPT-7 | `__nanosleep` shim is inert | Conclusion survives; the stated justification is false. Comment corrected. |
| OPT-8 | `__dp4a` → `sudot4` is scalar | Accurate. No action. |

---

## OPT-1: `-ffast-math` for the HIP native kernels that actually contain transcendentals

**File:** `CMakeLists.txt`
**Location:** the `elseif(STRATA_ENABLE_HIP)` block inside the `if(STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN)` /
kernels section, immediately after the existing `set_source_files_properties(... "-ffp-contract=off")`
call for `elementwise.cu` / `quantize_act.cu` (currently line 257).

**Correction — the original flag is an nvcc flag and will not compile.**
`--use_fast_math` is rejected by HIP clang outright:

```
clang++: error: unknown argument: '--use_fast_math'
```

The HIP/clang equivalent of nvcc's `--use_fast_math` is `-ffast-math`. Applying the original snippet
verbatim also fails at configure time, because its fifth file does not exist:
`src/kernels/cuda/native_qsa_gate.cu` is not in the tree. `set_source_files_properties` errors out on a
missing source.

**Problem.** The CUDA branch (`CMakeLists.txt:239-253`) puts `--use_fast_math` on thirteen `native_*`
files. The HIP branch does not, so on AMD the transcendentals in these kernels lower to full software
sequences instead of the hardware instructions gfx1100 has. On a probe kernel, `expf`/`rsqrtf`/`cosf`/
`sinf`/`logf`/`tanhf`/`1/sqrtf` cost **453 device instructions without `-ffast-math` and 93 with it** —
a 4.9x reduction. The only hardware instructions the non-fast build emits are `v_exp_f32`,
`v_rsqrt_f32`, `v_log_f32`, `v_rcp_f32`, `v_sqrt_f32`; `-ffast-math` adds `v_cos_f32` and `v_sin_f32`
and inlines the rest.

**Action.** Add, after the existing `-ffp-contract=off` block:

```cmake
  elseif(STRATA_ENABLE_HIP)
    # Keep Q8_K's multiply and magic-bias rounding as separate FP32 operations on HIP; contraction to an
    # FMA changes tie results. The embedding gather also relies on separately rounded operations.
    set_source_files_properties(src/kernels/cuda/elementwise.cu src/kernels/cuda/quantize_act.cu
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    # gfx1100 has hardware exp/rsq/sin/cos; without fast math they lower to software sequences.  This is
    # the HIP spelling of the CUDA branch's `--use_fast_math` (an nvcc flag that HIP clang rejects).
    # ONLY the files that actually contain transcendentals: measured device instruction counts, and the
    # files omitted here are omitted because they gain nothing (see OPTIMISATION.md section 0).
    set_source_files_properties(src/kernels/cuda/native_rope.cu
      src/kernels/cuda/native_qsa_indexer.cu
      src/kernels/cuda/native_gr_postops.cu
      src/kernels/cuda/native_gdn_preprocess.cu
      src/kernels/cuda/native_ple_postops.cu
      src/kernels/cuda/native_gr_norm.cu
      src/kernels/cuda/native_qsa.cu
      src/kernels/cuda/native_router.cu
      PROPERTIES COMPILE_OPTIONS "-ffast-math")
  endif()
```

**Deliberately NOT in that list,** with the measurement that excludes each:

| File | instrs before → after | Why excluded |
| --- | --- | --- |
| `native_flash_attn.cu` | 8692 → 8642 (−0.6%) | No upside, and it never runs above 256 context anyway. |
| `native_qsa_score.cu` | 365 → 363 (−0.5%) | No upside, and `tests/hip/native_qsa_score.cpp:80` compares **bit-exactly** against a host `std::fma` reference. |
| `native_mmvq.cu` | 286054 → 286303 (**+0.1%**) | The 286k-instruction decode kernel gets marginally *worse*. |
| `native_bf16.cu` | 7619 → 7619 (0%) | Integer/packed math only, no transcendentals. |
| `native_moe.cu` | 98 → 98 (0%) | Seven multiply-adds; nothing to gain. |
| `native_gdn.cu` | 220 → 216 (−1.8%) | Below noise. |

**Rationale for the included files**, by what they gain and which call site drives it:

| File | instrs before → after | Δ | Driver |
| --- | --- | --- | --- |
| `native_rope.cu` | 635 → 187 | **−70.6%** | `cosf`/`sinf`/`powf`, `native_rope.cu:54,91` |
| `native_qsa_indexer.cu` | 2199 → 811 | **−63.1%** | `rsqrtf` ×3 and `cosf`/`sinf`, lines 77,86-87,106-107,139,167 |
| `native_gr_postops.cu` | 516 → 417 | −19.2% | sigmoid `expf`, lines 36,46 |
| `native_gdn_preprocess.cu` | 492 → 407 | −17.3% | transcendentals in preprocessing |
| `native_ple_postops.cu` | 942 → 840 | −10.8% | transcendentals in post-ops |
| `native_gr_norm.cu` | 334 → 304 | −9.0% | `rsqrtf` in the norm |
| `native_qsa.cu` | 610 → 558 | −8.5% | `rsqrtf` + sigmoid `expf`, lines 65,75 |
| `native_router.cu` | 2589 → 2369 | −8.5% | router scoring transcendentals |

Note the original plan named `native_flash_attn.cu` as the top win and did not mention
`native_qsa_indexer.cu` at all. The indexer is the second-largest win in the whole set.

**Do not add `-ffp-contract=off` alongside it.** The original snippet paired the two flags. That would
be wrong here: the CUDA branch enables `--use_fast_math` *without* disabling contraction, because
CUDA's default is already FMA-contracting and the pinned sm120a oracle binary's numerics include
contraction. Adding `-ffp-contract=off` on the HIP side would move HIP *away* from the oracle. The
existing `-ffp-contract=off` on `elementwise.cu` / `quantize_act.cu` is a separate, file-specific
Q8_K tie-rounding requirement (see the comment at `CMakeLists.txt:255-256`) and is not a general
house rule.

**BLOCKING DEFECT — `-ffast-math` silently deletes this tree's input validation. Do not ship as-is.**

Applying the flags above compiles and links, and clang warns loudly, but the warning is load-bearing
rather than cosmetic. Seven files validate their host-side arguments with `std::isfinite`, and
`-ffast-math` makes those checks undefined behaviour — clang is entitled to fold them to `false`, and
it does. The `throw std::invalid_argument` on the other side of the guard goes dead.

Measured, host codegen for the guard predicate `!std::isfinite(x) || x <= 1.0f`:

| flags | emitted | inf rejected? | NaN rejected? |
| --- | --- | --- | --- |
| none | bit-pattern check (`cmpl $0x7F800000`) | yes | yes |
| `-ffast-math` | `ucomiss` + `setbe` — the isfinite half is **gone** | **no** | **no** |

Runtime confirmation, same predicate, values `1.5 / 0.5 / +inf / -inf / NaN / 1e30`:

| value | correct | under `-ffast-math` |
| --- | --- | --- |
| `+inf` | reject | **accept** |
| `NaN` | reject | **accept** |

This is not fixable with flags. Every decomposition of `-ffast-math` was measured, and there is no
combination that keeps the guard:

| flag | guard | probe instrs |
| --- | --- | --- |
| none | OK | 453 |
| `-ffast-math` | DELETED | **93** |
| `-funsafe-math-optimizations` | OK | 398 |
| `-freciprocal-math` | OK | 417 |
| `-ffp-contract=fast` | OK | 454 |
| `-fno-math-errno` | OK | 453 |
| `-ffinite-math-only` | DELETED | 443 |
| `-ffast-math -fno-finite-math-only` | DELETED | — |
| `-ffast-math -fno-finite-math-only -fhonor-nans -fhonor-infinities` | DELETED | — |
| `-funsafe-math-optimizations -freciprocal-math` (combined) | DELETED | — |

The guard's lowering is gated on clang's overall fast-math state, so restoring `-fhonor-nans` /
`-fhonor-infinities` afterwards does not bring it back. Any speedup worth having costs the guard.

**The fix, verified.** Replace the `isfinite` idiom with a magnitude range. A range check with finite
bounds rejects `+inf` (too large), `-inf` (too small) and `NaN` (fails both comparisons) without any
FP-assumption-dependent builtin, and it is correct under `-ffast-math`:

| value | `!isfinite(x) \|\| x<=1` under fast math | `!(x > 1.0f && x < 1e9f)` under fast math |
| --- | --- | --- |
| `1.5` (valid) | accept | accept |
| `0.5` | reject | reject |
| `+inf` | **accept (wrong)** | reject |
| `-inf` | reject | reject |
| `NaN` | **accept (wrong)** | reject |
| `1e30` | accept | reject |

Note the last row: the range form is *stricter* than today's guard, additionally rejecting absurdly
large finite values. That is the right direction for these parameters (a RoPE `freq_base` and an RMS
`epsilon` are both small), but it is a behaviour change and belongs in its own reviewed change, not
smuggled in under OPT-1.

The seven sites, all host-side argument validation:

- `native_rope.cu:79` — `freq_base`
- `native_qsa_indexer.cu:206`, `:207`, `:226` — `epsilon` and `freq_base`
- `native_qsa.cu:111` — `epsilon`
- `native_gr_norm.cu:87` — `epsilon`
- `native_gdn_preprocess.cu:133` — `epsilon`

**And one device-side hazard beyond the guards.** `native_router.cu` uses `-INFINITY` as a max-reduction
sentinel in device code at lines 56 and 87. Under `-ffinite-math-only` the compiler assumes no infinities
exist, so the sentinel initialisation and the `max()` chain that depends on it are unreliable. That is a
different failure mode from the host guards and is not fixed by the range-check change above — it needs
`native_router.cu` dropped from the list, or the sentinel reworked (e.g. to the most negative finite
value, or a separate validity mask).

**Consequence for the file list.** Only two of the eight files are clean as they stand:

| file | gain | `-ffast-math` safe today? |
| --- | --- | --- |
| `native_gr_postops.cu` | −19.2% | **yes** — no isfinite guard, no inf sentinel |
| `native_ple_postops.cu` | −10.8% | **yes** — no isfinite guard, no inf sentinel |
| the other six | −8.5% to −70.6% | no — guard or sentinel must be fixed first |

So OPT-1 ships in two stages: land the two clean files now, and take the six others once the guard
change is reviewed on its own. The two biggest wins — `native_rope` (−70.6%) and `native_qsa_indexer`
(−63.1%) — are both in the blocked half, so stage one alone is worth much less than the full item.

**Note this is a pre-existing bug in the CUDA build too.** `CMakeLists.txt:239-253` already applies
`--use_fast_math` to `native_rope.cu`, `native_qsa_indexer.cu`, `native_qsa.cu`, `native_gr_norm.cu`,
`native_router.cu` and `native_gdn_preprocess.cu` — the same files, with the same guards. So the CUDA
backend has been running with dead `isfinite` validation all along, and nvcc's equivalent of
`-ffinite-math-only` has the same effect. Fixing the guards is therefore a bug fix for **both**
backends, not a concession to HIP, and it is worth doing regardless of which of the two gets fast math.

**Parity consequence.** This is a deliberate numerical change on the HIP
backend. It is consistent with what `native_rope.cu` already assumes — its comment at line 90 reads
"Match pinned host-side float powf before device fast powf/trigonometry", i.e. approximate device
trigonometry is already the intended contract there — but it is still a change, and:

- `-ffast-math` also turns `fma(x, y, 0)` into a plain multiply. That is bit-exact for all finite
  inputs and for inf/nan; it differs only in signed zero.
- **No registered HIP test covers any of the eight files above.** `ctest` covers `hip_intrinsics`,
  `hip_handoff`, `hip_native_qsa_score`, `hip_prefill_native_batch`, `hip_prefill_hipblaslt_gemm`,
  `hip_prefill_mmq_parity`, `hip_ple_iq4`, `hip_expert_cache_staging` — of which only
  `hip_native_qsa_score` touches this area, and that file is excluded above. So `ctest` passing after
  this change is **not** evidence that it is correct. Gate it on a decode benchmark instead.

**Expected gain.** Large on the RoPE and indexer kernels (−60% to −70% instructions each), which run
once per head-row per QSA layer; single-digit percent on the rest. The original plan's "15–30% on
attention and GR hot paths" is not supportable — attention is excluded from this item entirely.

---

## OPT-2: `__launch_bounds__(128, 4)` on flash attention — WITHDRAWN, measured regression

**File:** `src/kernels/cuda/native_flash_attn.cu`, line 49.
**Status: do not apply.** The existing `__launch_bounds__(128, 1)` is correct.

Measured, same compiler and flags, only the annotation varying:

| bounds | device instrs | `scratch` refs | `amdhsa_next_free_vgpr` |
| --- | --- | --- | --- |
| `(128, 1)` (current) | 9858 | 465 | 256 |
| `(128, 2)` | 9858 | 465 | 256 |
| `(128, 4)` (proposed) | **9916** | **507** | 256 |

`(128, 4)` asks for 4 concurrent blocks per CU = 512 threads, which on gfx1100's 512-VGPR file needs
≤128 VGPRs. The kernel needs 256 (`amdhsa_next_free_vgpr 256`, driven by `float2 qreg[16]` plus
`float2 vkq[16]` at lines 66-67). The hint is unsatisfiable, so the register allocator's only response
is to spill more: +58 instructions and +42 scratch references, for no occupancy change.

The original plan's supporting facts are also wrong. It states gfx1100 supports "1024 threads per CU";
`hipGetDeviceProperties` reports `maxThreadsPerMultiProcessor = 2048`. And the claim that
`minBlocksPerSM=1` caps occupancy at 12.5% is not the binding constraint anyway — see OPT-2b.

---

## OPT-2b: `native_flash_attn` leaves 40% of the card idle (the real defect behind OPT-2)

**File:** `src/kernels/cuda/native_flash_attn.cu`, line 206.
**Priority: low, and gated on OPT-1's context-length finding being resolved first.**

`native_flash_attn_short_step` launches `attend<<<24, dim3(32, 4), ...>>>`. The grid is 24 blocks
(`shapes.n_head == 24`, validated at line 196). The device has **40 CUs**. Twenty-four blocks cannot
occupy more than 24 CUs, so **16 CUs — 40% of the silicon — are idle for the whole kernel**, and no
`__launch_bounds__` value can change that: the grid, not the per-CU occupancy, is the limit.

Restructuring the kernel to use more blocks (splitting the 256-wide head dimension, or the KV range,
across a second grid axis) is the only fix. It is a real rewrite with real correctness risk, and it is
**not worth doing yet**: per OPT-1's finding this kernel does not run at this model's context lengths
at all. Fix the context gate, or confirm the workload will move below 256, before spending effort here.

---

## OPT-3: hipBLASLt linkage and tuning table — ALREADY SATISFIED, re-verification only

**Files:** `cmake/hip_backend.cmake`, `src/prefill/gemm.cu`, `src/prefill/hipblaslt_tuning.hpp`
**Status: all four checklist items are already done. No code change.**

Verified:

1. **hipBLASLt is present.** `hipblaslt_DIR` resolves to
   `.venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib/cmake/hipblaslt`, `roc::hipblaslt` is a
   real target, and `STRATA_HIPBLASLT_AVAILABLE` is `ON` (`cmake/hip_backend.cmake:22-27`).
   `libhipblaslt.so.1.2` is on disk; `HIPBLASLT_VERSION_MAJOR/MINOR = 1/2`.
2. **The tuning table is selected at runtime.** `strata-coder-iq1_m.json:50` sets
   `STRATA_HIPBLASLT_TUNING` to `tools/hip/gfx1100-hipblaslt-100200.txt`, and `serve/server.py:497`
   forwards `cfg["env"]` into the engine's environment. `src/prefill/gemm.cu:122-123` requires that
   variable to be set, so without the config entry prefill silently degrades to hipBLASEx.
3. **The table matches arch and version.** Its header is `STRATA_HIPBLASLT_TUNING_V1 gfx1100 100200`;
   100200 encodes hipBLASLt 1.2.0, which is the installed version, and the arch is the runtime
   `gcnArchName`. `tools/hip/gfx1100-hipblaslt-100100.txt` is the stale 1.1.0 table — do not point at it.
   Confirmed live in `strata-coder-iq1_m.log:32`:
   `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`.
4. **The mismatch warning already exists.** `src/prefill/gemm.cu:149-152` prints
   `prefill gemm: <reason>; using hipBLASEx` and returns `nullptr`. The original plan asked for this;
   it is implemented.

**Action.** None. Re-run step 3's check after any ROCm upgrade — a pip-wheel bump changes
`HIPBLASLT_VERSION` and silently invalidates the table, at which point `tune_hipblaslt` must be
re-run and the config's path updated. That coupling is the actual risk here, not a missing link.

---

## OPT-4: replace the `cudaMemcpy2DAsync` q/gate split with a kernel

**File:** `src/core/layer.cpp`, line 922 (inside `qsa_layer()`).

**Problem.** `cudaMemcpy2DAsync` copies the first half of each head's `2*head_dim` block out of `q_full`
into `qcur`: `dstPitch = head_dim*4`, `srcPitch = head_dim*2*4`, `width = head_dim*4`,
`height = n_head`. With the shapes this model uses (`n_head = 24`, `head_dim = 256`) that is
**24 rows of 1 KB — 24 KB total**, not the "128 separate 1 KB copies" the original plan claimed. It
runs once per QSA layer, 12 layers per token, and it materialises a 24 KB round trip through VRAM that
the very next operation (`normalize_rotate`) immediately reads back.

**Action.** Either of the two options the original plan offered works; the second is better.

Option A — a dedicated kernel. The shape is 24 rows × 256 floats, so one block per row of 256 threads
is a clean fit:

```cpp
__global__ void split_head_half(const float* __restrict__ src, float* __restrict__ dst, int head_dim) {
    const int d = threadIdx.x;
    if (d < head_dim) dst[blockIdx.x * head_dim + d] = src[blockIdx.x * 2 * head_dim + d];
}
```

Option B — fold the split into `normalize_rotate`'s norm call. The RMS norm at
`src/core/layer.cpp:844` reads `data` row-by-row and writes it back; a strided-source variant that
reads row `h` of `q_full` at offset `h*2*head_dim` and writes row `h` of `qcur` at `h*head_dim` removes
the intermediate buffer entirely, saving both the launch and the 24 KB round trip. Prefer this if the
norm kernel's addressing can be parameterised without disturbing the other four `normalize_rotate`
call sites (`b.kcur`, `b.q_idx`, and the GR paths) that pass contiguous rows.

**Expected gain — much smaller than the original plan's 0.3–0.5 ms/token.** That figure is off by
roughly an order of magnitude. This sits inside a captured graph, so the per-launch overhead the plan
reasoned from is already amortised; what remains is one 24 KB D2D transfer and, on ROCm, the
`hipMemcpy2DAsync` bookkeeping. Order of **tens of microseconds per token** across 12 layers, not
hundreds. Take it as a free cleanup, not a headline win, and measure before and after with the same
prompt rather than trusting either number.

---

## OPT-5: graph capture in the production path — ALREADY SATISFIED

**File:** `src/program/generate.cpp`
**Status: verified on. No bug found.**

- `no_capture` defaults to `false` (`generate.cpp:168`); it is set only by the explicit `--no-capture`
  flag (`generate.cpp:978`).
- Neither `serve/server.py`, `serve/__init__.py`, `chat.py`, nor `run-coder-iq1_m.sh` passes
  `--no-capture`. The only references in the serving path are the flag's own help text and the
  guard clauses that *refuse* invalid `--no-capture` combinations (`generate.cpp:2472-2491`).
- `session_capture` is called whenever `!o.no_capture && !native_pack` (`generate.cpp:2456`), and the
  log confirms graphs are live — `strata verify: captured the 6-token window`, and likewise for the
  5/4/1/2/3-token windows.

**Optional hardening (the one part of the original item still worth doing).** The plan asked for a
startup line making accidental fallback visible, e.g. `"graphs: N captured, 0 direct launches"`. The
capture path already logs per-window ("captured the 6-token window"), so the marginal value is a
single summary count. `src/core/session.cpp` would need to return the captured/replayed graph counts
to `generate.cpp` to print it. Low priority; do it only if you want the invariant asserted in one place
rather than inferred from four log lines.

---

## OPT-6: `std::atomic<bool>` in MoE combine — WITHDRAWN

**File:** `src/kernels/cuda/native_moe.cu`, line 33.
**Status: the described problem cannot exist. No change.**

`std::atomic<bool> enabled{false}` is declared in the file's anonymous namespace and is touched only by
the two host functions `native_moe_combine_set_enabled` (line 58) and `native_moe_combine_enabled`
(line 59). The `combine` kernel (lines 34-48) never reads it — the kernel body is a plain weighted sum
over `parts`/`weights` with an optional `shared` addend, and takes no flag argument.

The original plan's cost model assumed a device-side atomic compiled to `buffer_store_dword` +
`s_waitcnt` + `s_barrier` per launch. There is no such access: this is a host object, read once per
launch on the CPU, with no GPU instruction at all. The proposed `atomicExch` is a device intrinsic and
cannot be substituted for a host `std::atomic` anyway.

If a device-side "is the native path enabled" flag is ever genuinely needed, that is a new piece of
work, not a fix to this line.

---

## OPT-7: `__nanosleep` shim — conclusion holds, justification was wrong

**File:** `include/strata/hip_compat/intrinsics.hpp`, line 123.
**Status: comment-only, and the comment must correct the record.**

```cpp
#define __nanosleep(cycles) __builtin_amdgcn_s_sleep(1)
```

The shim sleeps one cycle whatever the argument says, so a caller asking for `__nanosleep(N)` spins N
times more often than intended. That conclusion is right.

The original plan justified skipping this by claiming Strata's only spin-wait is host-side
(`graph.cpp::wait_ms` using `_mm_pause()`). That is false — **three device-side spin-waits use the
shim**:

- `src/kernels/cuda/elementwise.cu:211` — `while (*flag != want) __nanosleep(100);`
- `src/kernels/cuda/verify_kernels.cu:426` — `while (*flag < value) __nanosleep(100);`
- `src/kernels/cuda/verify_kernels.cu:475` — `while (*flag < value) __nanosleep(100);`

These are the CPU/GPU handoff doorbells, and `tests/hip/handoff.cpp` exercises them deliberately. The
behaviour is still correct — a 1-cycle sleep is a busier spin than a 100-cycle one, not a wrong answer
— which is why the original plan's conclusion happened to land correctly. But "inert" was the wrong
word, and the reasoning should not be left in the tree, because it would justify ignoring the next
kernel that spins.

**Action.** Replace the shim with a comment recording both facts: the argument is ignored, and the three
call sites above are the current users. If a fourth caller appears, note that it will spin N times more
often than written.

---

## OPT-8: `__dp4a` → `sudot4` is scalar-only — informational, no action

**File:** `include/strata/hip_compat/intrinsics.hpp`

`__builtin_amdgcn_sudot4` executes on the scalar ALU. For decode GEMV this does not matter: the
workload log shows decode at 27–41 tok/s with a 58–64% expert-cache hit rate and 98–99% KV block hit
rates, i.e. firmly memory-bound, so ALU issue is not the constraint. For compute-bound prefill MMQ it
would cap throughput at one dot product per cycle per lane. Only relevant if prefill becomes
compute-bound; a `v_madmk`-based alternative is the escape hatch then. No action now.

---

## Build verification

The original snippet was wrong in three ways: it passed `-DSTRATA_ENABLE_HIP=ON` twice, omitted the
compiler and ROCm root (ROCm is not on `PATH` on this machine — it comes from the pip
`_rocm_sdk_devel` wheel), and omitted `CMAKE_BUILD_TYPE` and `CMAKE_MAKE_PROGRAM`. This configures and
generates cleanly:

```bash
ROCM=$PWD/.venv/lib/python3.13/site-packages/_rocm_sdk_devel
cmake -B build-hip -G Ninja \
  -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=$ROCM/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM \
  -DCMAKE_PREFIX_PATH=$ROCM \
  -DHIP_PLATFORM=amd \
  -Dhip_DIR=$ROCM/lib/cmake/hip \
  -Dhipblas_DIR=$ROCM/lib/cmake/hipblas \
  -Dhipblaslt_DIR=$ROCM/lib/cmake/hipblaslt
cmake --build build-hip --target strata
```

`HIP_PLATFORM=amd` is required and easy to miss: without it `hip-config.cmake:144` aborts with
`Unexpected HIP_PLATFORM:` and an empty value.

Run the HIP tests:

```bash
cmake --build build-hip --target hip_intrinsics hip_handoff hip_native_qsa_score
./build-hip/hip_intrinsics && ./build-hip/hip_handoff && ./build-hip/hip_native_qsa_score
```

**Do not treat a green `ctest` as the acceptance gate for OPT-1.** As noted in OPT-1, none of the
registered tests exercise the eight files that change. Benchmark decode before and after on the same
prompt, and watch for log drift. The current baseline from `strata-coder-iq1_m.log` is:

- decode 26.8–41.3 tok/s across sessions (128 generated tokens per request)
- prefill 594–619 tok/s on cold 4210- and 8830-token prompts
- 58–64% decode expert-cache hit rate, 97.8–99.3% KV block hit rate

The original plan's "expect ≥20% improvement on the attention-dominated path" is not the right
expectation: this workload is not attention-dominated, and the attention kernel it names does not run.
Measure the decode tok/s delta and set expectations from that.
