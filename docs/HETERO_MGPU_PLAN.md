# Heterogeneous multi-GPU execution plan (MAIN + DRAFT, pipelined prefill)

Status: **design (Fáze 1) — no code changed yet.** Branch `sm61-1080ti`, base `2ab8e23`.
First real application: *MAIN GPU = main model (model + KV + verification), DRAFT GPU = the
MTP drafter (weights + own KV + drafting)*, plus using both GPUs during prefill. The roles
must be swappable (`main-device=0 draft-device=1` and the reverse), per-device in capability
decisions, and opt-in — every existing mode (single GPU, layer split, expert-cache device1,
CPU/RAM offload, mmap streaming) keeps working untouched.

Everything below was read out of the working tree on 2026-10-01; file:line references are to
that tree. Three analysis passes produced it: device/multi-GPU, MTP/spec path, prefill/KV.

---

## 1. What exists today (facts, not wishes)

### 1.1 Device abstraction — minimal, and mostly unused

- `DeviceInfo` (`include/strata/core/device.hpp:21-30`) carries ordinal, name, cc_major/minor,
  free/total bytes, driver/runtime versions, SM count, HIP arch string. It is queried only
  twice in the whole tree (both `strata-device`, ordinal 0) and dropped — there is **no
  persistent per-device record**. No `sharedMemPerBlock`, no dtype/kernel support matrix,
  no P2P topology.
- Kernel selection is already per-device at runtime, but via disjoint lazy caches:
  `static int cc_major[64]` tables inside `qsa_select.cu:510-522` and
  `qsa_prompt_attn.cu:680-692`, plus per-device
  `cudaFuncAttributeMaxDynamicSharedMemorySize` opt-ins (`qsa_select.cu:532`,
  `qsa_prompt_attn.cu:622/653`, `qsa.cu:675-682`, `fused_gr.cu:334-355`). The ggml prefill
  backend keeps a full per-device struct (`prefill/ggml_cuda_host.cu:70-89`). So
  "different kernels per device in one process" already happens in practice — it is just not
  *planned* anywhere: no plan-level record of "device X lost its MMA path".
- The cc<75 runtime guard (`device.cu:138-145`, wrapped by `STRATA_EXPERIMENTAL_SM60`) runs
  only where `device_info()` is called — effectively nowhere on the multi-GPU path. On CUDA,
  per-device CC enforcement on the split path happens only under `STRATA_USE_HIP`
  (`generate.cpp:1297-1311`).
- `DeviceArena` (`device.cu:150-196`) is unused in production; all real allocations are bare
  `cudaMalloc` (weights, expert slots, sessions, drafter arena).
- Device visibility is fixed by the Python launcher (`serve/server.py:531-558`,
  `CUDA_VISIBLE_DEVICES`), not by the engine.

**Hard device0=primary assumptions to unwind:** stage 0 is always ordinal 0
(`OnDevice on0(0)` at `generate.cpp:2150`, `session_bytes(...,0,hi0)`, plan string
`generate.cpp:3665-3668`); helper devices are positional (`remote_dev[3] = {1,2,3}`,
`generate.cpp:1262`; `RemoteExperts::preflight` rejects `device < 1`,
`remote_experts.cpp:69`); `kDrafterMib`'s 1000 MiB reserves the *last stage* only
(`generate.cpp:2018`); VRAM reporting sums tiers correctly
(`generate.cpp:4040-4087`) but per-device free VRAM is only priced in the layer-split
`stage_room()` search (`generate.cpp:2019-2028`).

### 1.2 Existing multi-GPU mechanisms (two, and they are the building blocks)

**A. Layer split** (`--layer-split auto|K..` + `--split-device`; `GpuStage`,
`generate.cpp:661-679`): each stage owns its device's dense-weight slice, session carve,
`ExpertCache`, `Verifier`, `Prefill`, two streams, an event, the residency table. Stages are
executed strictly sequentially per verify window, handing the residual through **portable
mapped pinned host buffers** (`cudaHostAllocMapped | cudaHostAllocPortable`,
`generate.cpp:3618-3640`; consumed by `copy_from_mapped`, `verify.cpp:356-361,749-757`).
Every stage transition ends in a full `cudaStreamSynchronize` (`verify.cpp:1050-1053,1191`).
`--layer-split auto` is already heterogeneity-aware (per-stage VRAM, SM count × clock cost
model, `generate.cpp:2019-2112`) — the one genuinely capability-pricing piece of the code.

**B. Expert-cache tiers** (`--expert-cache-device1/2/3 N` → `RemoteExperts`,
`remote_experts.cpp`): a second GPU holds its own non-evicting `ExpertCache` of ranked
(layer, expert) pairs; a decoded layer's remote-owned experts are classified, the
activation block is memcpy'd into pinned staging, quantized and computed **on the helper's
stream**, results land in a pinned mapped buffer and are host-memcpy'd back; the CPU pool
skips those rows. One blocking `cudaStreamSynchronize` per layer on the token critical path
(`remote_experts.cpp:316-326`).

**Synchronization reality: no P2P, no cross-device events.** Nothing in the tree calls
`cudaDeviceCanAccessPeer` / `cudaDeviceEnablePeerAccess`; `docs/MULTI_GPU.md` and
`docs/SECOND_GPU.md` state P2P is deliberately not required — all cross-GPU traffic goes
through pinned/mapped host memory, and most of it is fenced by a blocking
`cudaStreamSynchronize` on the exporting side. Intra-device event discipline exists
(`overlap_main.cpp:126-145`, `generate.cpp:3885-3968`); cross-device events have no
precedent here yet. On the dev rig (2× GTX 1080 Ti, PHB-only topology — no direct PCIe
switch link) P2P is not expected to be available anyway.

### 1.3 The MTP draft path (full anatomy)

Weights (`--mtp DIR` → `MtpDrafter::load`, `mtp.cpp:142-302`), files
`/mnt/models2/strata-q2_0-pack/mtp-rt/`:

| File | Size | Contents |
|---|---|---|
| `dense.bin` | 111 MiB | dense projections (q8_0), hc/router (bf16), norms (f32) |
| `experts.bin` | 675 MiB | **all 512 routed experts, fully VRAM-resident** |
| `draft_vocab.bin` | 158 KiB | 40,525-token draft-head vocab subset |
| `dense.txt` | index | tensor table |

Plus the draft head rows gathered out of the main model's head blob (`dhead_`, 68 MiB,
`mtp.cpp:404-412`) and its **own separate QSA K/V state** (one draft layer ↔ the whole
48-layer main model; only `max_cells` and the RoPE tables are shared by reference —
`mtp.cpp:206-238`, `layer.cpp:580-585`). Measured total: **~870 MiB VRAM** (logged "802 MiB
+ 68.0 MiB head"), reserved as `kDrafterMib = 1000` (`generate.cpp:2018`).

Devices: the drafter is already written device-agnostic — `device_` captured once at
`load` (`mtp.cpp:144`), **every** public entry (`bind/prefill/draft/draft_first/kv_restore`)
opens `OnDevice(device_)`, it owns its own non-blocking stream (`mtp.cpp:294`), and there is
no `cudaSetDevice` anywhere in `mtp.cpp`. Placement is only implicit today
(`generate.cpp:2220` puts it on the layer-split last stage; single-GPU = CUDA0).

Hidden-state handoff: the drafter consumes the main model's **final multi-stream residual
`R`** (`hc × n_embd` = 4×2560 floats = 40 KB/row), produced by the verifier's last-layer
`gr_write` (`verify.cpp:734`), bound via `mtp.bind(wt, &head, ver.final_R_all(), ...)`
(`generate.cpp:5382` / `3672`), and read by a **kernel** on the drafter's stream:
`copy_from_mapped(Rin_, window_R_, T·HC·N, cs_)` (`mtp.cpp:631`). Ordering is guaranteed
only by the **host**: `Verifier::run` ends in `cudaStreamSynchronize` (`verify.cpp:1050`)
and `commit` syncs again (`verify.cpp:1181-1183`) before `mtp.draft(...)` is called. No
cross-stream events exist in the spec loop. `draft_first` even *writes back* into the
verifier's `R_` rows (`mtp.cpp:826-832`) — harmless today, a real hazard once devices split.

Draft loop (`mtp.draft`, `mtp.cpp:768-826`): 1 round graph (catch-up cells + full MTP-layer
run + `mtp_select` writing draft 0 and its probability to mapped host) then **one captured
graph per chain step, each followed by `cudaStreamSynchronize`** (up to `max_t−2` ≈ 2 steps
at `spec=4`). Every step is a full single-row MTP layer: 512-resident-expert router +
head GEMV over 40,525 rows — the per-step sync is genuinely expensive; this is where most
of the measured host ~190-206 ms/verify-round hides besides the verifier's 8+ syncs and
48 doorbell spins.

Verification/acceptance/rollback: the verify window runs all `T` positions through all
layers in **one captured graph** (flat chain, no tree; `verify.cpp:322-795`); acceptance is
a host-side greedy prefix match (`generate.cpp:4759-4760`); rollback is **not a KV
truncate**: `ver.commit(a+1)` replays GDN conv state for accepted tokens, restores the
indexer tail from a snapshot and re-appends only accepted keys (`h_commit_` carries `-1`
reject markers, `verify.cpp:884-932`), PLE history rewinds to its snapshot — and rejected
K/V cells are simply **overwritten** when those positions come round again (stated in
`verify.hpp:12-18`, `mtp.hpp:13-19`). The drafter's K/V state is part of the persistent
session/conversation-checkpoint format (`mtp.kv_state()` read by
`conversation_snapshot_*`, `generate.cpp:3717-3724,4343-4373`) and has an existing
"rewrite up to position N" operation in ring mode (`kv_ring_restore` / `kv_restore(upto)`,
`mtp.cpp:669-677`).

### 1.4 Prefill, KV, loading

- **Chunked prefill is the default** (`--prefill 2048` in production; `auto` scans
  chunk sizes against borrowable slots, `generate.cpp:3377-3383`). Segments round up to 256
  (`generate.cpp:3365-3372`).
- **"Prefill borrow"**: prompt-path chunk buffers are carved from the expert-cache VRAM
  tail for the duration of the prompt (`Prefill::init(..., borrow)`, `prefill.hpp:56-60`),
  borrowed slots get refilled after; disabling it shrinks the VRAM tier permanently and
  measured ~4× slower prompts (6.5 s vs 24.7 s). Treat as mandatory, do not break.
- **Cross-GPU prefill pipelining already exists and measured +18-20 %**: layered
  `Prefill::set_stage` + `std::async` next-stage chunk ("next card reads chunk c while
  this one reads c+1", `prefill.cpp:1788-1806`, two pinned handoff buffers per stage).
  What does **not** exist: a parallel *draft* prefill on another GPU. `Prefill::on_chunk`
  → `mtp.prefill` runs synchronously on the main device after each chunk
  (`prefill.cpp:1808-1815`), and `Prefill::draft_kv` explicitly refuses cross-device
  (`prefill.hpp:41-44`, `prefill.cpp:692`).
- **KV**: paged QSA pools with on-device page table, int8/q4_0/k8v4 formats,
  per-QSA-layer indexer state; allocated once at `--max-context`. **No truncate primitive**
  anywhere; rollback relies on the snapshot-tail + re-append machinery above.
- Loading/streaming (`weights.cpp`, `expert_source.cpp`): the pack is dense VRAM arena +
  `experts.bin` either fully pinned in RAM (arena), mmap'd (`--mmap-experts`, page-cache
  dependent — current production mode), or resident-pinned complement
  (`--resident-experts`). `--shared-expert-arena` exists (MAP_SHARED file).
- Decode per-token path: one captured graph per verify window on one non-blocking stream
  per stage + a copy-engine stream for PCIe expert fills (`verify.hpp:219-222`); the CPU
  expert pool interleaves per (layer, group) driven by mapped-memory doorbells.

---

## 2. Design principles (locked)

1. **Roles, not ordinals.** Nothing in the new code may conclude from a device index.
   A `DeviceRole` is assigned to devices by configuration or a capability-based selector;
   `MAIN`/`DRAFT` today, extensible to `CACHE`, `OFFLOAD`, later roles without changing
   call sites.
2. **Per-device capability record, persisted.** A process-lifetime `DeviceCaps` table
   (per visible device) is the single source for cc, smem, dtype/kernel support, P2P,
   topology. Kernel-selection sites get a record lookup instead of disjoint `static`
   caches. No "one device supports X ⇒ all use X".
3. **P2P is an optimization, never a requirement.** The pinned/mapped-host handoff (the
   existing, measured pattern of layer-split handoff and prefill stage handoff) is the
   baseline transport. A `can_peer()` probe may upgrade it later; code must run with the
   probe reporting 0 (which it does on this rig).
4. **Correctness before concurrency.** New mode launches with the same host-ordered,
   synchronised handoffs as today; overlap is added only per-measurement afterwards.
5. **All new modes opt-in.** The default and every existing flag path stay bit-identical
   (greedy determinism, `chat_golden.json`, selftests).
6. **Step discipline.** Each implementation step: explore → one small change → build →
   test → (runtime measure) → fix/re-test → commit/checkpoint — before the next step.
   A step that fails is fixed or reverted, not walked past.

---

## 3. Corrections to the original plan before coding

- **Fáze 8 Varianta A ("draft prefill independently of the main model") is not possible
  as stated**: the MTP block consumes the main model's final residual `R` to build its K/V
  and select drafts (`mtp.prefill(R_rows, ...)`, `prefill.cpp:1827`; its QSA K/V is
  appended per row from those residuals). There is no token-only draft prefill path in this
  architecture. The realizable options are:
  - **B1 (preferred): pipelined draft prefill.** Copy the chunk's `R` rows (chunk 2048 ⇒
    **80 MiB** = 2048 × 40 KB) plus next-token ids to the draft GPU through the existing
    pinned-host handoff pattern, and run `mtp.prefill` on the draft GPU's own stream
    **while the main GPU prefills the next chunk** (the `set_stage` + `std::async` pattern
    exists). Net PCIe volume ~80 MiB/chunk ≈ 7-13 ms over the copy stream — acceptable;
    it must be measured, not assumed.
  - **B2 (degenerate): keep draft prefill on MAIN (today's behavior), but overlap it with
    the next chunk's *attention* instead of sitting after the whole chunk** — smaller
    change, no cross-device activation stream.
  - **C: expert prefetch into a DRAFT-role GPU's cache during prefill** (device-stream
    agnostic API exists: `ExpertCache::admit/fill_slot(slot, host_blob, stream)`) —
    independent bonus, useful in the current device1-expert-cache config.
- **"DRAFT keeps MTP permanently in VRAM" is already true** — 675 MiB of experts + dense +
  head are resident from load; no eviction path exists. Fáze 5 is about *placement and
  handoff*, not residency.
- **Draft KV rollback**: the overwrite-on-revisit semantics plus `ver.commit`'s replay
  already implements "rollback to A B C" for the *main* KV. For the *draft* side the
  equivalent is: confirmed-pointer = last accepted position; speculative cells beyond it
  are overwritten next round (they only exist in the drafter's own QSA state and, in ring
  mode, `kv_restore(upto)` restores the ring from the host copy). So Fáze 6 becomes:
  make the confirmed/speculative split **explicit and per-device-checkable** (a position
  watermark + per-ring restore), not a new full KV cache.
- **The session/checkpoint format reads `mtp.kv_state()` from whatever device the drafter
  is on** — a DRAFT-role move must wrap these accesses in `OnDevice` (they use
  `cudaMemcpyDefault`; they will work cross-device but are synchronous in the current
  device's context — must be explicit, not accidental).

---

## 4. Phased plan (each phase = several small gated steps)

### Fáze 2 — Device capability discovery
New `DeviceCaps` (extend `DeviceInfo`): name, cc, VRAM free/total, `sharedMemPerBlockOptin`,
warp size, dp4a / tensor-core(mma) support flags, per-device smem opt-in results,
`cudaDeviceCanAccessPeer` matrix (probed on demand, cached), PCIe link speed/width from
attributes where available. Populated at startup for **every visible device** (not just 0),
logged, and exposed by `strata-device`. Kernel-selection sites read the record (first
consumers: the `static int cc_major[64]` caches in `qsa_select.cu`/`qsa_prompt_attn.cu` —
replace with a shared accessor; behavior-identical).
Gate: build, unit test of the caps record, `strata-device` prints the full matrix on the
2×1080 Ti rig (expect `can_peer == 0` both ways).

### Fáze 3 — Explicit roles (`main-device` / `draft-device`)
New execution strategy, **separate from layer split** (opt-in flag group; layer split and
roles are mutually exclusive by validation). `--main-device N`, `--draft-device M`. Inside,
generalize the 7 hard device0 sites from §1.1 just enough for a two-role setup: main model
stays a (possibly single) `GpuStage`; the drafter takes `OnDevice(draft_dev)` instead of
`last_st->dev`, binds a per-device head copy, and the `kDrafterMib` reservation follows the
role, not the last stage.
Gate: both assignments (0→1 and 1→0) build, run and produce identical greedy output;
step order: (a) role plumbing types + config; (b) drafter device override; (c) head copy
per device; (d) e2e both directions.

### Fáze 4 — Role validation
At startup, against the `DeviceCaps` table: MAIN must have VRAM for (weights slice + cache
+ session) as today; DRAFT must have VRAM for `mtp.vram_bytes()` ≈ 970 MiB + head + draft
KV and support every kernel/dtype the drafter path launches (cc floor, smem opt-ins for
the drafter's kernels). Incompatible ⇒ precise error listing the unsatisfied requirement
per device. Gate: negative tests (undersized budgets via an env/flag override).

### Fáze 5 — MTP execution path on the DRAFT GPU
Core change set, in sub-steps: (1) `window_R_` becomes a portable mapped-pinned host
handoff (the `generate.cpp:3618-3640` pattern) instead of a cross-device pointer; verify's
last stage additionally mirrors the final residual rows into that host buffer; (2) remove
`draft_first`'s write-back into verifier `R_` (it only matters cross-device — route that
path through the handoff too); (3) the drafter's stream keeps its own capture; the host
ordering stays synchronous (Fáze 9 later).
Gate: greedy stream identical to the single-GPU baseline on the dev prompt set;
`STRATA_DECODE_TIMING` breakdown before/after; no VRAM change on MAIN beyond the
160 KB/round handoff pin.

### Fáze 6 — Draft KV checkpoint/rollback (explicit watermark)
Make the draft-side confirmed/speculative split explicit: position watermark
`confirmed ≤ last accepted`, speculative region above it; per-round append as today
(overwrite semantics), plus an explicit `kv_restore(upto)`-equivalent path usable when the
ring is active, and a snapshot of the draft KV page rows touched by rejected positions
only where a test needs byte-exact re-derivation. Unit test: propose A B C D E, verify
A✓B✓C✓D✗ → state consistent with A B C, next-round re-append reproduces the
pre-speculation bit pattern (`STRATA_STATE_HASH` equality on the main state; hash on draft
state before/after).

### Fáze 7 — Main↔draft sync, minimal traffic + instrumentation
Formalize the per-round crossing set: tokens (`T` ints), hidden `R` (40 KB/T), acceptance
`a`, KV watermark — nothing else. Add counters: bytes, transfer count, transfer latency
(µs), host sync latency; printed under `STRATA_DECODE_TIMING`. P2P upgrade path: if
`can_peer`, replace the R-handoff with a device-to-device copy (same abstraction); test
with the probe forced off.

### Fáze 8 — Prefill on both GPUs
Implement B1 (§3): per-chunk `R` rows + ids shipped to the draft GPU on a copy stream,
`mtp.prefill` executed there concurrently with the *next* main chunk (double-buffered
handoff, event-guarded like prefill's ring); fallback = today's synchronous on-main
behavior (`--no-draft-prefill-parallel`). Measure: prompt wall-time, draft-GPU idle%
(timing counters), PCIe bytes. C (expert prefetch during prefill) as a separate small step.
Gate: draft KV byte-identical to synchronous prefill (hash), prompt end-to-end not
slower, TTFT not worse.

### Fáze 9 — Async pipeline (verify ∥ draft)
Only after Fáze 5-8 are stable: batch the draft chain (the 3 sequential launch+sync
round-trips → single graph with all steps, host-min_p check off or via a second smaller
graph), and overlap the draft round with the *previous* verify's tail (commit graph can
run while the drafter's round graph is still in flight — independent devices/streams).
Measure `mtp.ms_draft` and round wall-time deltas; greedy output identical.

### Fáze 10 — `auto` role selection
Capability-based: both roles scored on (VRAM headroom vs role requirement, cc/kernel
coverage, draft-expert residency fit, P2P, measured SMs/clock for the MAIN's heavy
layers); `auto` picks only when one assignment strictly dominates; otherwise it prints
the ranked options and demands explicit config. Never "biggest VRAM = MAIN" alone.

### Fáze 11 — VRAM analysis per role
Report, from actual allocations (not theory): MAIN per device (dense slice, KV+indexer,
expert cache, staging, borrow reserve) and DRAFT per device (dense, experts, head,
draft KV + speculative region, arena, staging pairs, the 1000 MiB reserve). Verify fit
against 8/11/12/16/24 GB by *running* with adjusted budgets (context/chunk flags) and
recording the real peak via the `cudaMemGetInfo` checkpoints the engine already makes.

### Fáze 12 — Benchmark, honestly
Sweep: single GPU / layer split / MAIN+DRAFT; prefill, decode, acceptance rate, effective
tok/s, per-role utilization. **On this rig (2× identical 1080 Ti) only the symmetric
assignment can be measured; asymmetric heterogeneous claims are NOT measurable here** and
the docs must say so. Prepared so that e.g. an RTX 4090 + RTX 2070 owner runs the same
sweep untouched (configs + a bench script entry).

### Fáze 13 — Regression gate
Selftests + `chat_golden.json` untouched by default runs; explicit keep-list: single GPU,
layer split, expert-cache device1, resident/mmap modes, session cache, coupled sampling,
CPU pool, and the Pascal build (`STRATA_EXPERIMENTAL_SM60`).

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Cross-device `window_R_` used by a kernel today (`mtp.cpp:631`) — silent corruption if roles split while the pointer stays device-local | Fáze 5 step 1 replaces it with a pinned handoff before any device assignment is exposed; debug assert comparing devices |
| `draft_first` writes the verifier's `R_` rows (`mtp.cpp:826-832`) | removed/rerouted in Fáze 5 |
| Mapped staging pairs lack `cudaHostAllocPortable` (`mtp.cpp:69-77`) | add Portable for any buffer shared cross-device |
| Session snapshot code assumes the drafter device reachable synchronously | wrap in `OnDevice`; test checkpoint save/restore after the move |
| Draft prefill on the DRAFT GPU stalls TTFT if the handoff is not double-buffered | event-guarded double buffer; measure B1 vs B2; fall back to B2 |
| Host sync storm (8+ syncs/round) grows with the new handoff | instrumentation first (Fáze 7), batching only in Fáze 9 |
| sm61 rig cannot falsify heterogeneous-role claims | explicit measurement-honesty section (Fáze 12); bench script for other owners |