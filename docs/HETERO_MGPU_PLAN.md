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
2×1080 Ti rig. DONE 2026-10-01 (commit fe665a5) - and the expectation above was WRONG:
canAccessPeer reports YES both ways on this rig, so peer access is *available* here even on
the (claimed) PHB topology - which only makes the optional P2P upgrade path cheaper; the
design still never requires it. Caps measured: sm_61, 49152 B smem opt-in, dp4a yes, mma no,
cp.async no, GPU0 free ~0.41 GiB (production engine resident on both cards).

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
---

## 6. Corrections recorded 2026-10-01 (user) — baseline and test-rig reality

1. **The work targets upstream `Niko1221/Strata`** (the x99 tree origin), not just the
   `hireymage/Strata` fork. Upstream `main` has moved past the v0.1.30 base this analysis
   was written against: **0.1.32 at `c499bd1` = +150 commits / +55,759 lines over
   `08ea0e0^`**, touching `device.cu`, `verify.cpp`, `mtp.cpp`, `prefill.cpp`,
   `generate.cpp`, `session.hpp`, `layer.cpp` and more. **The Fáze 2+ line references
   above are valid for the sm61-1080ti tree (2ab8e23) but must be re-validated after the
   branch is rebased onto 0.1.32** — the analysis conclusions (mechanisms, handoff
   patterns, no-P2P discipline) need a re-check against the newer tree, not a blind carry.
2. **Direct heterogeneous-role testing is not possible in this homelab, on either rig:**
   - **Intel 8700K (Windows)**: the only Turing card (runs upstream `main` fine, cc >= 7.5)
     — **one card, so no second device exists to pair with it**; heterogeneous roles can
     never be exercised there.
   - **x99**: two GTX 1080 Ti (Pascal, PHB topology) — both assignments of a two-role
     setup are testable, but only on the `STRATA_EXPERIMENTAL_SM60` build; **upstream
     `main` still refuses cc < 7.5** (`device.cu` guard, unwrapped there).
   Consequence for Fáze 12: every heterogeneous-perf claim stays unmeasured here by
   construction (symmetric 1080 Ti pair only); the bench script must be runnable
   untouched by an owner of a mixed rig (e.g. RTX 4090 + RTX 2070).
---

## 7. Re-validation on 0.1.32 (2026-10-01, commit worktree `sm61-hetero`)

The `sm61-1080ti` commits were ported onto upstream `c499bd1` (0.1.32) in the worktree
`/home/hozzy/src/Strata-rebase` (branch `sm61-hetero`). Port notes:

- `33d9a9f` (sm61-enable) **superseded by upstream #236**: 0.1.32's CMake adds
  `STRATA_EXPERIMENTAL_SM60=1` for `strata_core` and `device.cu` takes `kMinCc = 60`.
- cc12d0a's logic re-applied by hand as the 0.1.32 `iq_pack.py` rewrote the pack writer
  (tmp+rename+sidecar atomic publish); port = `SafeMemmap` on all three memmap sites, per-layer
  fsync, resume on `experts.bin.tmp`.
- 0.1.32's `open_sized()` still zeros `layer_next_` after `open()` rebuilt it — the open_sized
  fix (7d0f9f9) remains required; ported as `db6a67c`.

Anchor re-check (v0.1.30 → 0.1.32): `window_R_` kernel handoff `mtp.cpp:631 → 634`;
`draft_first` R_ write-back → `mtp.cpp:825`; draft-chain syncs unchanged in shape
(mtp.cpp:679-826); `draft_kv` cross-device refusal → `prefill.cpp:741`; static per-device cc
caches → `qsa_select.cu:697` + `qsa_prompt_attn.cu:982`; `OnDevice on_mtp` →
`generate.cpp:2499`; `kDrafterMib` → `generate.cpp:2287` (comment updated to "839 MiB"); P2P
still absent everywhere (0 hits). The design's Fáze 1 conclusions survive 0.1.32.

**New for Fáze 9**: upstream 0.1.32 already has an async-commit path (`E-6`,
`g_commit_async` / `STRATA_VERIFY_ASYNC_COMMIT`, `commit_done_` event at `verify.cpp:351`;
synchronisation kept as fallback by default and always under a split, `verify.cpp:1329-1338`).
The drafter already reads only this window's final rows while commit completes in the
background. Fáze 9's remaining scope is therefore mostly the draft-chain batching (the
3 sequential launch+sync in `mtp.cpp`) rather than commit overlap.

Build/test on 0.1.32 (sm61, -DCMAKE_CUDA_ARCHITECTURES=61): 237/237 targets; ctest 50/53 PASS
(12 new upstream tests all pass). The 3 failures are pre-existing baseline, not rebase
regressions: `ple_parity` (upstream test fixture absent), `kv_hybrid_parity` (mode-5
`qsa_prompt_attn` refuses hybrid pools — fails identically on the 0.1.30 build; our
production uses int8 KV, unaffected), `expert_multi_test` (E5-2678 v3 has no AVX512-VNNI/
VBMI). Config note: vendored `third_party/ggml` in 0.1.32 is an incomplete checkout (no
CMakeLists); configure with `-DSTRATA_GGML_DIR=<llama.cpp checkout with pinned 3cf0325>`.

## 8. Fáze 2 krok-záznam (2026-10-01 večer)

- Commit fe665a5: DeviceCaps (+ device_caps(), peer_access_matrix()) in
  include/strata/core/device.hpp + src/core/device.cu; strata-device enumerates every visible
  device with its caps and prints the peer matrix. HIP path kept (gcn arch + guards);
  STRATA_EMULATE_CC honoured via cc_major_of/cc_minor_of/smem_optin_of.
- Build 237/237, ctest 50/53 (the same 3 pre-existing baseline failures; no new ones).
- Gate result on 2×1080 Ti: full matrix printed; canAccessPeer YES both ways (the earlier
  "expected 0" prediction falsified - peer IS available over this topology).
- Next (Fáze 2b): first consumers - point the static int cc_major[64] caches at the caps
  record (behaviour-identical), then Fáze 3 explicit roles.

- Fase 2b done: `int device_cc_major(int)` - one shared per-ordinal cached accessor
  (STRATA_EMULATE_CC honoured); both static cc-caches replaced by it, with the call-site A/B
  override (STRATA_QSA_WARP = select / attn) applied as POLICY on top of the fact, exactly as
  before (behaviour-identical: same kernels chosen, same failure paths). Build clean; ctest
  unchanged (50/53, same 3 baseline).

## 9. Fáze 3 gate-záznam (2026-10-02) — the portable-mapped hand-off is green

The mapped-hand-off step (verifier's window residual carried through a host mapped mirror;
cross-device kernel reads only of mapped host memory) CLOSED Fáze 3: the drafter runs on another
GPU end to end, with the P2P path still never used.

Two root findings from the gate runs (both now in code comments):

1. **P2P on this rig is not only deadlocked, it is unnecessary**: `cudaDeviceCanAccessPeer`
   answers yes, but plain `cudaMemcpyDeviceToDevice / cudaMemcpyDefault` across two non-peer
   GPUs already works (driver stages through host) - measured with a standalone probe
   (uva_probe.cu on the rig, 2026-10-02).  The design's premise holds: the peer path is
   never required.
2. **`cudaPointerGetAttributes` from a foreign context answers `devicePointer = (nil)`** for
   another device's arena memory - the RoPE-table localization copied a nil source and failed
   'invalid argument', and the fallback shared the main device's tables, which the draft
   kernels then read cross-device and crashed with an illegal access (uva_probe2.cu).
   Fix: localize_rope copies through the raw UVA pointers, never through the queried
   devicePointer.  The same rule applies anywhere else we copy across devices.

Gate result (125B pack, 95-token prompt, 60 new tokens, greedy, --spec 4, both 1080 Ti):
the role plans 0->1 AND 1->0 (the reversed one loads the whole main stage on the second GPU) and
the degenerate 0->0 WITH role flags are all rc=0 and **bit-identical** to the flagless baseline
(60/60 tokens, same speculation statistics 23 rounds of 6, 37/65 drafts accepted).  The
regression gate (Fáze 13 harness) compares the engine's "output :" dump lines.

Harness note (bench-hetero.sh): the one-shot CLI takes --mtp DIR (the serve config's "mtp" key
names the same directory; there is no --rt flag), and a native (IQ) pack additionally needs
--native SHARD1, --ple-gguf, --prefill CHUNK and the --expert-profile residency table for --spec.
## 10. Fáze 4 krok-záznam (2026-10-02) — the roles' capability view and the ambient-device contract

The roles' plan now reads its own premises out loud at startup (generate.cpp, after the
role range check): one informational line per role device (name, cc, dp4a or the software
fallback, free VRAM) plus the peer-matrix entry for the pair, reported only - the mapped
mirror needs no peer pair, the kernel paths self-select per device.  No new refusals.

The step also captured a real hazard: `caps_from` (device.cu, the `device_caps()` record's
per-device probe) `cudaSetDevice`s the probed ordinal and never handed the ambient device
back.  The roles view's three probes left the LAST probed ordinal current, and the engine's
later device picks read the wrong device - the verifier's banner printed "GPU 1" for
`--main-device 0`, the expert-cache auto sizing came off the wrong card (4579 instead of
the certified 5011 slots), and the run died with "prefill copy_i32: an illegal memory
access".  Fix: a CurrDeviceGuard in caps_from brackets the probe (destructor restores, the
throw paths included).  `device_info` stays unguarded on purpose - the engine's init path
may legitimately consume the set device; only the informational query changed its contract.

Gate (125B pack, 95-token prompt, 60 new tokens, greedy, --spec 4): 0->1 and 1->0 both
rc=0 and BIT-IDENTICAL to the certified baseline (the same 23 rounds of 6, 37/65 accepted;
regression gate PASS both.  The banner and the 5011-slot cache match the certified log
again).  ctest 50/53 (the 3 pre-existing).  Harness note: ctest with -j 4 collides while
the production engine holds the RAM - run the suite serially near a loaded machine; a
single test alone passes at any time.
## 11. Fáze 6/7/8/10/11 krok-záznam + the determinism fix (2026-10-02)

Commits on `sm61-hetero-0134` (tree `/home/hozzy/src/Strata-rebase`): 041e21b (Fase 6: the drafter
ring restore launches on the ring owner device), 5032a98 (Fase 7 instrumentation + THE DETERMINISM
FIX), 7a15ea5 (Fase 8: `--draft-prefill-parallel`), cc4646d (Fase 10: `--auto-roles`), ae5c180
(Fase 11: per-role VRAM report). ctest stays 50/53 throughout (the same 3 pre-existing parity
failures: `ple_parity`, `kv_hybrid_parity`, `expert_multi_test`).

### 11.1 THE ROLES' NONDETERMINISM (pre-existing, only main=0 draft=1) — root cause and fix (5032a98)

Symptom: a roles 0->1 run gave either A=23 rounds/37 of 65 or B=24/36/69 (or 25/35/72); the same
GPU0-only and 1->0 runs were always A. Method: CRC instrumentation over the mapped mirror, the
drafter window and the per-round drafts (STRATA_DBG_MIRROR/DRAFT, reverted before the commit) plus
one-Knob bisections. Findings, in order:

- The hand-off itself is EXACT: the mirror's crc equals the drafter's window crc in EVERY window of
  both attractor runs, and the divergence starts at pos=101 = the first window AFTER the first
  adapt round ((rounds+1) % 4) on IDENTICAL window inputs (state, tokens, drafts all bit-equal).
- The suspect is main-side: `adapt()` runs the experts' H2D swap copies on `adapt_stream` (async),
  while the residency table (d_res) reached the devices only in `apply_pending` - a NON-BLOCKING
  call at the window boundary that DROPS while a copy is in flight. Two consequences:
  (a) the evicted expert stays marked resident -> a window can read its slot mid-overwrite;
  (b) the admitted expert switches from the CPU pool to the GPU-resident path at whichever window
  the copy had landed by - and the GPU expert path and the CPU pool path round DIFFERENTLY (the
  pre-existing parity gap; the same one behind the 3 failing ctests) - so the stream depends on the
  race. The 0->0 and 1->0 runs had no jitter in that timing, hence always A.
- Bisect proof: `--adapt-swaps 0` (adapt runs, no copies) is deterministic (D-attractor 26/37/75 x2);
  an early res_upload()-only attempt still left B alive (it fixed (a) only).

Fix: upload the table right after the copies are SUBMITTED (evictions visible at once, and
swapped-in experts stay non-resident until they land) and make the window-boundary call BLOCKING
(`apply_pending(true)`), bounded by one ~1.4 MB copy (~1 ms). Both decode loops (serve, one-shot)
and the chat loops got it.

### 11.2 THE NEW GATE REFERENCE (the old certified stream is VOID)

The previously certified reference (A, md5 1515024ed68a8a2797f98710df847ca7) was measured UNDER
the race: deterministic by schedule, not by construction. WITH the fix:

- baseline (no roles flags, drafter same-device, 4400 slots): 22 rounds/38 of 62, bit-identical x3;
- roles 1->0 (4400 slots): 22/38/62, bit-identical x3, BIT-IDENTICAL to the baseline;
- roles 0->1 (auto, 5011 slots): 26 rounds/34 of 75, bit-identical x3;
- roles 0->1 with FORCED `--expert-cache 4400`: 22/38/62, BIT-IDENTICAL to the baseline.

That last row is the real proof of the phases: under EQUAL expert residency the drafter's placement
changes nothing token for token; directions otherwise differ only through residency (the auto slot
count differs because the drafter's 68+870 MiB sit on the MAIN's GPU when the roles are the same
device but on the DRAFT's otherwise). Gate logs: /tmp/bench-gate0 (fc9b4c93e2c2), /tmp/bench-gate10
(22fe41487d75), /tmp/bench-gate01 (39d97c814cd8), /tmp/bench-gate01n (b172785529af).

Fase 13 gate rule that follows:

1. every plan is bit-exact across repeats (now enforced >= x2);
2. directions are compared only under EQUAL residency - either the same effective slot count, or a
   forced `--expert-cache N` shared by all plans;
3. the reference streams are the 22/38/62 ones above (NOT the 0.1.34 re-certification's md5).

### 11.3 Fase 8 measurements (the pipelined prompt fill)

- The 95-token probe prompt: no change (5.30 vs 5.30 s prefill; the drafter catches up inside one
  chunk). The flag's output is bit-identical with and without the pipeline.
- A 4,099-token prompt (3 chunks of 2,048, mmap-experts): 655.6 -> 590.7 and 587.5 s without the
  flag vs 469.3 and 656.7 s with it. The prefill is I/O-BOUND (the experts' mmap reads alone: 388 -
  577 s = 78-98 % of the window), and the run-to-run spread covers the drafter's share, so the
  pipeline's gain is NOT PROVEN on this rig's pack - the honest number is "variance-noise-dominated";
  a fair claim needs a cached/pinned-expert profile (the flag itself is correct and off by default).

### 11.4 Fase 7/11 numbers worth quoting

- decode timing (serve probe, 125B, roles 0->1): 938.91 ms/window, of which verify 929.86 (host
  experts 715.96 = the CPU pool), commit/emit 0.14, draft 4.42, mirror 1.66 - the mapped mirror
  hand-off costs ~1.7 ms per ~939 ms window (~0.18 %); there is no measurable hand-off tax.
- per-role VRAM after the binding (Fase 11): main 633 MiB free of 11163, draft 9619 MiB free of
  11165 (the drafter is tiny; the main's VRAM is the expert-cache budget).
## 12. Fáze 9 krok-záznam (2026-10-02) — the batched draft chain (`--draft-chain-batch`, d21f369)

The design's insight still held on 0.1.34 with one simplification: the chain's steps ALREADY
propagate their data on the device (`mtp_select` leaves the next step's residual and token in
`Rin_[0]` / `tok_[0]`), so the per-step waits only ever bought the HOST-side min-p cut.  The opt-in
flag restructures `draft()` to stage every step's cell (`p + a + j`, known before any launch) up
front, capture the step graphs while the stream is idle, launch the round graph and all of them
back to back, and apply the cut from the mapped probabilities after ONE wait - the same rule as the
per-step loop (the count of drafts whose probability and every one before it reaches min_p).  A
step past the cut runs harmlessly: it writes only ring cells the chain rewrites when it reaches
them again, and its drafts/probabilities are zeroed before the caller sees them.  Roles-independent
(a same-device run gains too); STRATA_DECODE_TIMING's draft figure and the one-shot's
`mtp ... ms/round drafting` measure it.

Gate (125B, greedy 60 tokens, the F12 reference EXTRA - reconstructed from the pack's
sm61-config.json and verified bit-identical to the a306853 reference streams before A/B): every
row bit-identical to its flag-off reference - same-device (22/38/62), roles 0->1 (26/34/75), roles
1->0, and `--spec-min-p 0.20` (an ACTIVE cut: 23/39/64, window sizes T4:21) - determinism x2-x3
per row; ctest 50/53 (the same 3 pre-existing).  Drafting 4.63 -> 4.14 ms/round same-device,
4.40 -> ~4.1 ms/round roles (noise ±0.4 over 25 rounds) - the drafter is ~0.5% of a ~939 ms window
(Fase 7/12), so the batching is measured, not felt.  Gate logs: /tmp/bench-f9-ref-sd,
/tmp/bench-f9-sd, /tmp/bench-f9-01, /tmp/bench-f9-01on, /tmp/bench-f9-10on, /tmp/bench-f9-minp-{off,on}.

PHASE STATE after this: 1-8, 9, 10, 11, 12, 13 COMPLETE.  The hetero plan's implementation list is
done; what remains is the PR strategy (issue to the author first, then a hetero-only PR) and the
production tree's promotion decision.
## 13. Fáze 14: the third-party test harness (`tools/hetero-test.sh`)

Everything above was proven on OUR rig (2x GTX 1080 Ti). For the upstream issue to carry weight it
must travel to OTHER rigs: different GPUs, different expert-cache ceilings, possibly one GPU. The
harness turns the F13 gate rule into a one-command, dependency-free probe:

    tools/hetero-test.sh BUILD_DIR "BASE_FLAGS" OUT_DIR [MAX_NEW=60] [REPEATS=2]

- BUILD_DIR holds `strata` + `strata-device` (the tester's own build of the branch);
- BASE_FLAGS is the tester's own model load (quoted, the engine's own demanded set -- the pack,
  the (optional) MTP, quantization flags...); the harness only APPENDS the roles/batching flags;
- the probe input is a deterministic random token-id list (seeded; the builds have no tokenizer,
  so ids ARE the input) -- every variant runs the same ids with `--max-new`, temperature off.

Variants: V0 baseline (same GPU), V1 roles 0->1, V2 roles reversed, V3 +`--draft-prefill-parallel`,
V4 +`--draft-chain-batch`, V5 both, V6 `--auto-roles`; on ONE GPU with `--mtp`, V7 = the batch
alone. Then the exactness row V1f: V1 at V0's auto slot count (`--expert-cache N`) -- the
EQUAL-residency run that isolates "placement changed the output?" from "residency changed the
output" (two-GPU rigs only, and only when BASE does not already force a numeric cache).

The report (`OUT/report.md`, markdown, ready to paste): per run the accepted/of-totals, tok/round,
ms/round drafting, expert slot count and the GREEDY stream md5; verdicts -- repeats identical
(determinism); every auto-residency variant compared bit-exactly against a reference of the SAME
residency (V1 if the slot counts match, else V0, else V1f), otherwise SKIPPED with the slot counts
stated; V1f vs V0 as the placement-exactness proof; the residency-dividend NOTE when V0's and V1's
auto slot counts differ (that difference IS the "is it worth it" signal on the tester's rig).

Validated on the source rig (0.1.34 sm61-hetero-0134): all verdicts PASS, V1f == V0 bit-exact --
the same outcome the F13 gate measured by hand.