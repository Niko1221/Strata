# NVFP4 peer-prefill (config C) — scale wiring: WORKING NOTES

Goal: make the peer GPU's MoE products numerically correct for NVFP4 packs.
Today the peer's MMQ runs with **no per-expert scales** (`grp_tail` = null, no
`s_gate/s_up`, no `s_down`, no `y_scale`), so its output rows are wrong.
`STRATA_PF_PEER_NVFP4=1` (prefill.cpp 1694–1704) opens the path anyway for
measurement — this note is the plan to make it *correct*.

## FINAL DESIGN (review round 3) — read this first

**`s_down` is applied IN PLACE with `mmq::scale_down_rows`, not via a
`row_sd` table.** As soon as `peer_now` is set, BOTH GPUs scale their own Dm
rows in place right after the down-proj:

- peer:  `mmq::scale_down_rows(P.Dm_b[b], N, rel, ngx, P.grp_tail, nr, ps);`
- primary: `mmq::scale_down_rows(m.Dm + r0 * N, N, dn.bounds, ngx, m.grp_tail, nr, m.cs);`

Consequences (this is why the design was chosen):
1. **`kernels.cu` and `kernels.hpp` are NOT touched at all.** The four
   accumulation kernels (`peer_gather_add`, `peer_scatter_add`,
   `moe_combine_peer`, `moe_combine_peer16`) never see a scale — the rows
   they read are already scaled. No new params, no OOB traps, no rebuild of
   the kernel ABI.
2. **Pfad C fixed**: `moe_combine_peer[16]` (4092–4095) has no `row_sd`
   param at all, so in sums mode GPU 0's OWN rows lost their `s_down`
   (it only ever went into `m.row_sd`, which only `moe_combine` reads).
   In-place scaling fixes GPU 0 and GPU 1 at once.
3. **Pfad B fixed** (the old 31 non-finite logits — see "Puzzle solved"):
   with `peer_now`, `moe_combine` gets `row_sd = nullptr`, so the
   never-written `m.row_sd[rows_local .. T*K)` (QSA scratch garbage) is
   never read.
4. No `P.row_sd` member, no back-copy over PCIe, no ps-vs-s_out race.

Scope: **only `prefill.cpp` is edited** (plus the `STRATA_PF_PEER_NVFP4`
gate in `generate.cpp` that lets the experiment reach `set_peer` at
`--peer-prefill-rows 0`). Line numbers below refer to the v0.1.44.1 tree.

## Runtime setup (what "config C" is)

- Launch: `run-nvfp4-nvfp4.sh` → serve/server.py + `strata-nvfp4-nvfp4.json`,
  engine `--peer-device 1`, NVFP4 pack, **no P2P env** → peer route =
  **host route (mapped pinned) + SUMS mode** (`pp->sums` default when !p2p &&
  compact, prefill.cpp ~1780).
- Gate (user-side scripts, NOT in the repo): `recall-test.py` with
  `STRATA_PF_PEER_NVFP4=1` must pass both needles; then `c-an.sh` for perf.
- Build: `docker exec strata2 sh -c 'cd /workspace/build && make -j$(nproc) strata'`
  then copy `build/strata` → `engine/strata`, restart (`tools/safe-restart.sh`).

## The three scale pieces (NVFP4 MMQ)

1. **`grp_tail`** (float, **`MMQ_GROUP * 4`** = one group's 16 experts ×
   {s_gate, s_up, s_down, 0}, indexed by GROUP SLOT `q * 4`, NOT by expert
   id): the 16-byte tails live at the end of each expert blob
   (`bd + f.tail_off`) and reach the group buffer only through
   `gather_native`'s `tail` / `tail_dst` args (primary fallback loop 3896–
   3899: `f.tail_off ? b + f.tail_off : nullptr, f.tail_off ? m.grp_tail +
   i * 4 : nullptr`; grouped: `gather_native_group(..., f.tail_off,
   m.grp_tail, zero)` 3891–3893). `mmq::Product` has **NO `.tail` field** —
   the scales reach the math only via `swiglu_scaled` (s_gate/s_up) and
   `scale_down_rows` (s_down). **The peer's blobs are already on the peer
   GPU** (`P.peer->slot_ptr(l, e)` 3665, or its ring `P.pstage` 3662) →
   **NO upload from the primary**; the peer just passes the tail args in its
   own `gather_native` calls (compact 3668–3670, non-compact 3750–3751).
2. **`s_down` — IN PLACE via `mmq::scale_down_rows(d, ld, bounds, n, tails,
   nrows, stream)`** (moe_mmq.cu 303; kernel 270–278: `r = blockIdx.x` from
   **0**, RELATIVE bounds, scales row `d + r * ld` by `tails[4q+2]`). So
   `d` = the group's Dm base and `bounds` = the same relative table the dn
   Product got: compact `P.Dm_b[b]` + `rel`; non-compact `P.Dm + r0*N` +
   `dn.bounds`; primary `m.Dm + r0*N` + `dn.bounds`.
   The OLD mechanism — `down_row_scales` → `m.row_sd` table, applied by
   `moe_combine` at read time (kernels.cu 1588/1611) — stays ONLY for the
   no-peer case; with `peer_now` it would be read by kernels that have no
   `row_sd` param at all (`moe_combine_peer`/`peer16` 2073/2097,
   `peer_scatter_add`/`peer_gather_add` 2088/2112) → GPU 0's own rows would
   silently lose `s_down` in sums mode. CPU-pool rows are unaffected: they
   carry s_down already (row_sd 1.0 at 4075, s_down multiplied in at 3437),
   and with a peer active `moe_combine` gets nullptr anyway (step 8).
3. **`y_scale`** (float, per quantized row): activation scales for w4a4.
   Default NVFP4 mode is **W4A8** (activations q8_1) → primary passes
   `y_scale = mmq::fp4_activations(type) ? m.Xscale : nullptr` (3954/3968) =
   null at runtime. BUT the w4a4 kernel `exit()`s on null y_scale
   (moe_mmq.cu) → still allocate a buffer as guard. **Indexed by LOCAL dest
   row** (`blockIdx.x` in quantize.cu, `y_scale[j]` in MMQ) → size = rows in
   the call, NOT cap_rows: `G` in compact mode, `R` in layer-sized mode.
   The primary passes `m.Xscale` / `m.Hscale` to `quantize` **unconditionally**
   (3499 / 3962) and sets `gu.y_scale` / `dn.y_scale` from `fp4_activations`
   (3954 / 3968) — the peer mirrors exactly that (members `P.Xscale`,
   `P.Hscale`).

## PeerPrefill (struct at prefill.cpp 597) — what exists today

- compact (default, also forced when !p2p): per-GROUP buffers `Xq_g, Hq_g,
  GU_g, H_g, Dm_b[2]`, `G = T_max` (alloc ~1740–1770); non-compact: `Xq, Hq,
  GU, H, Dm` sized `R = cap_rows` (~1772–1778).
- host route: `host_x / host_rows / host_src / host_bounds` mapped pinned;
  `back_at / back_rows` (656) = the rows this layer sends back. (No scale
  ever travels this way in the final design — everything is in place.)
- peer compute block ~3685–3774: compact quantize+MMQ at 3684–3695
  (`mmq::quantize(P.mixed, P.src + r0, P.Xq_g, mmq_gt, N, N, nr, ps)` — no
  y_scale arg; gu/dn descriptors `P.grp_gu`/`P.grp_d`, no tail, no
  `.y_scale`); non-compact at 3742–3770.
- `mmq_gt` / `mmq_dt` (3074–3075) are **locals of the layer loop** — the peer
  block is inside the same loop and already uses them; keep that.

## The patch — FINAL (review round 3), prefill.cpp ONLY (APPLIED, gate passed)

*(Line numbers are this tree's; the review's hunks ~68/~313/~744/~823/~846/
~876/~1469/~1481/~1529/~1550/~1638/~1725 map to 74 / 637 / 1654 / 1755 / 1777
/ 1819 / 3668 / 3684 / 3742 / 3763 / 3970 / 4097 — same code, different
revision numbering. The review's diff was verified hunk-by-hunk against this
tree on 18 Oct: all anchors and the `scale_down_rows` semantics (relative
bounds, d = group base, moe_mmq.cu 270–278) match.)*

1. **No-MMQ stub block (74–81):** add
   `void scale_down_rows(float*, int64_t, const int32_t*, int, const float*, int64_t, void*) {}`
   after the `down_row_scales` stub (the real one exists: header 48,
   moe_mmq.cu 303 — only the stub is missing).
2. **`PeerPrefill` members (after 640 `grp_gu/grp_d`):**
   `float *Xscale = nullptr, *Hscale = nullptr, *grp_tail = nullptr;`
   — and **NO `row_sd` member** (in-place design).
3. **`bind_stage_helper` (after 1655 `P.grp_d = h.grp_d;`):**
   `P.Xscale = h.Xscale; P.Hscale = h.Hscale; P.grp_tail = h.grp_tail;`
   (the helper stage's `h.Xscale/h.Hscale/h.grp_tail` are always allocated
   when `mp.any`, 1297–1303, and 1631 already requires the MMQ plan — no
   extra null-check needed).
4. **Allocs:**
   - compact block (after 1757 `pp->Dm_b[1] = ...`):
     `pp->Xscale = (float*) take((size_t) Gr * 4); pp->Hscale = (float*) take((size_t) Gr * 4);`
   - non-compact block (after 1782 `pp->Dm = ...`): same with `R`;
   - common (after 1819 `pp->grp_d = ...`):
     `pp->grp_tail = (float*) take(MMQ_GROUP * 4 * sizeof(float));`
5. **Compact peer loop:**
   - gather (3668–3670): add tail args
     `f.tail_off ? bd + f.tail_off : nullptr, f.tail_off ? P.grp_tail + q * 4 : nullptr`
     (exactly the primary's 3896–3899 pattern);
   - quantize (3684): trailing `, P.Xscale`; after the gu field block
     (3688): `gu.y_scale = mmq::fp4_activations(mmq_gt) ? P.Xscale : nullptr;`
   - swiglu (3690): `if (lay.native && f.tail_off) mmq::swiglu_scaled(P.GU_g,
     P.H_g, nr, 640, rel, ngx, P.grp_tail, 0, ps); else` + old `swiglu` call;
   - quantize H (3691): trailing `, P.Hscale`;
   - after `dn.ld_dst = N;` (3698): `dn.y_scale = mmq::fp4_activations(mmq_dt)
     ? P.Hscale : nullptr;`
   - **after `P.run_ctx->run(dn, ps);` (3699): `if (lay.native && f.tail_off)
     mmq::scale_down_rows(P.Dm_b[b], N, rel, ngx, P.grp_tail, nr, ps);`**
     — in place, on `ps`, BEFORE the `ev_grp`/`ev_dm` records below ⇒ the
     sums kernels and `P.s_out` only ever see scaled rows. The
     `peer_gather_add`/`peer_scatter_add` calls (3700–3710) are UNCHANGED.
6. **Non-compact peer loop:**
   - gather (3750–3751): tail args as in 5;
   - quantize (3742): trailing `, P.Xscale`; `gu.y_scale` after 3765;
   - swiglu (3767): `swiglu_scaled(P.GU + r0 * 1280, P.H + r0 * 640, nr, 640,
     P.bounds + j0, ngx, P.grp_tail, r0, ps)` / else old call;
   - quantize H (3768): trailing `, P.Hscale`; `dn.y_scale` after 3773;
   - **after `P.run_ctx->run(dn, ps);` (3774): `if (lay.native && f.tail_off)
     mmq::scale_down_rows(P.Dm + r0 * N, N, dn.bounds, ngx, P.grp_tail, nr,
     ps);`** (`dn.bounds` = the group's RELATIVE table, 3728–3731 — matches
     the kernel's `r = blockIdx.x` from 0).
7. **Primary compute lambda (3970–3971):** branch on `peer_now`
   (`peer_now` is set at 3524, before this lambda runs; captured by `[&]`):
   `if (peer_now) mmq::scale_down_rows(m.Dm + r0 * N, N, dn.bounds, ngx,
   m.grp_tail, nr, m.cs); else mmq::down_row_scales(m.row_sd + r0, dn.bounds,
   ngx, m.grp_tail, nr, m.cs);`
8. **Combine (4097–4098):** `moe_combine`'s row_sd arg gains `!peer_now && `
   → with a peer active, rows are already scaled (7) and the never-written
   `m.row_sd[rows_local .. T*K)` is never read (Pfad-B fix).

### kernels.cu / kernels.hpp — NO CHANGES (round-3 decision)

The earlier plan added `row_sd` params to `peer_gather_add`/
`peer_scatter_add`/`moe_combine_peer`/`moe_combine_peer16`. Superseded: with
in-place `scale_down_rows` the rows those kernels read are already scaled,
so the kernel ABI stays untouched. (The identified gap — `moe_combine_peer`
(2073)/`peer16` (2097) have no `row_sd` and silently drop GPU 0's s_down in
sums mode — is fixed by step 7 instead of by new kernel params.)

## Gotchas / decisions already made

- **MODE MATTERS (verified 18 Oct):** runtime = host route + **SUMS** +
  compact. Sums mode: the peer folds its (already-scaled) rows via
  `peer_gather_add`/`peer_scatter_add` (3703/3708); the primary's own rows
  go through `moe_combine_peer[16]` (4092–4095), which has **no row_sd
  param** → those rows MUST be scaled in place (patch step 7).
  Rows mode (`STRATA_PF_PEER_SUMS=0`, or P2P): the peer's already-scaled
  rows are copied into `Dm[rows_local + r0 ..]` per group (3715/3718) and
  the primary's rows are scaled in place → `moe_combine` must get
  `row_sd = nullptr` (patch step 8). Both modes: nothing scale-related
  crosses the PCIe link.

- **W4A8 is the default NVFP4 mode** → `y_scale` is null at runtime; the
  buffers exist so the w4a4 kernel's null-`exit()` guard can't fire, and so
  the w4a4 switch needs no further wiring.
- **`grp_tail` never crosses the PCIe link**: the peer owns its blobs
  (resident slots / its own ring) and extracts the tails itself in
  `gather_native`. It is a **group-slot** buffer (`MMQ_GROUP * 4`), reused per
  group — never index it by expert id.
- **`swiglu_scaled` is not optional** for native+tail layers: without it the
  gate/up scales are simply missing (plain `swiglu` ignores `grp_tail`).
  Compact passes `bounds = rel, row0 = 0`; non-compact `bounds = P.bounds +
  j0, row0 = r0` — mixing those up silently mis-scales every row.
- **`scale_down_rows` takes the GROUP base + RELATIVE bounds** (kernel:
  `r = blockIdx.x` from 0, `d + r * ld`): compact `P.Dm_b[b]` + `rel`,
  non-compact `P.Dm + r0*N` + `dn.bounds`, primary `m.Dm + r0*N` +
  `dn.bounds`. Passing absolute Dm or absolute bounds silently mis-scales.
- **`m.row_sd` is QSA-scratch-aliased** (`m.region`, 1271/1283/1303): only
  `[0, rows_local)` is ever written. NEVER read it when a peer was active
  (step 8) — and never size a peer buffer from `G` alone for anything that
  is indexed by absolute peer rows.
- **`row_sd` tables are dead in the peer design**: no `P.row_sd` member, no
  back-copy, no kernel params. The `down_row_scales` path survives only for
  `!peer_now` (single-GPU), where `moe_combine` reads `m.row_sd` as before.
- **`bind_stage_helper` must bind the new members** (step 3) — the
  layer-split helper path reuses the helper stage's buffers and would run
  with null scales otherwise.
- **`y_scale` is LOCAL-row indexed** (quantizer writes `scale[blockIdx.x]`,
  MMQ reads `y_scale[j]`) — do NOT index by `ids[row]`; size = rows-in-call
  (compact `Gr`, non-compact `R` — matches each path's quantize calls).
- The peer's `Dm` rows are in **the peer's own row order** (peer rows are
  laid out local-first as one contiguous block, 3332–3364; compact's
  per-group products sit at `[r0, r1)` of peer-Dm space, `Dm_b[b]` alone is
  group-based `0..nr` — which is exactly what `scale_down_rows` expects).
- CPU-pool rows carry s_down already → row_sd = 1.0 there (4075); that path
  is untouched by this patch (it only matters when `!peer_now`... and with
  peer active step 8 makes `moe_combine` ignore row_sd entirely).
- `mmq_gt/mmq_dt` are loop locals (3074–3075), fine to use in the peer block.
- Fused-native path (`fused_nat`, 3080) requires `no_peer` — no fused changes
  needed for the peer.

## Puzzle solved: path B's 31 non-finite logits (old experiment)

Why did rows mode (`!P.sums`) show 31 non-finite logits before any scale
wiring? `m.row_sd` lives in the shared scratch `m.region` (carve: region at
1271, `m.sel_ids`/`m.sel_scores` from `b` at 1283–1284, `m.row_sd` from `c`
at 1303 — `b` and `c` are sub-allocators of the SAME region, and the region
is `max(gdn, qsa, moe)` reused between the attention and the MoE half).
GPU 0 only ever wrote `m.row_sd[0 .. rows_local)`; for the peer's rows
`[rows_local .. T*K)` `moe_combine` (4097–4098, `row_sd != nullptr` on NVFP4)
read **uninitialised QSA garbage** (`sel_ids` int bits reinterpreted as
floats → huge/NaN multipliers). So it was never the peer's math — the
primary's combine read junk for the peer's rows.
⇒ Fix (round 3): with `peer_now`, the primary scales its own rows IN PLACE
(step 7) and `moe_combine` gets `row_sd = nullptr` (step 8) — the junk range
is never read, and no scale data travels back over PCIe.

## Status log (keep updating!)

- 18 Oct: exploration done; backups made; **no source edits yet**.
- 18 Oct (review round): user review confirmed the 4 kernels.cu targets and
  fixed 4 plan bugs — (1) no grp_tail upload, extract via `gather_native`
  tail args (Product has no `.tail`; buffer is `MMQ_GROUP * 4`, group-slot
  indexed); (2) `swiglu_scaled` added to the plan (compact: rel bounds +
  row0 0); (3) `Xscale`/`Hscale` always to `quantize` + `y_scale` on
  gu/dn; (4) **`row_sd[r - r0]` in `peer_gather_add_kernel`** (else garbage
  from group 2, illegal access past `r ≥ G`) and shifted pointer at the
  `peer_scatter_add` call; + `bind_stage_helper` binds the new members.
  All verified against this tree; README updated accordingly.
- 18 Oct (review round 2): row_sd design changed per review — **full layer
  size `Rt`** (1784), each group writes at `+ r0` ⇒ gather kernel reads
  `row_sd[r]` absolute (no `- r0`, no OOB past `G`), scatter gets
  `P.row_sd + P.bounds_host[j]`, and path B's ps-vs-s_out race disappears.
  Review also EXPLAINED path B's old 31 non-finite logits: `m.row_sd` shares
  `m.region` with the QSA `sel_ids`/`sel_scores` scratch (1271/1283/1303);
  the peer's rows `[rows_local..T*K)` read uninitialised QSA garbage — see
  "Puzzle solved". Both verified in this tree.
- 18 Oct (review round 3, FINAL DESIGN): user supplied the complete
  prefill.cpp diff — **in-place `mmq::scale_down_rows` replaces every
  row_sd-table idea**: peer scales `P.Dm_b[b]`/`P.Dm + r0*N` after dn (new
  steps 5/6), primary branches on `peer_now` (step 7), `moe_combine` gets
  `!peer_now &&` on its row_sd arg (step 8), sums kernels UNTOUCHED,
  **kernels.cu/kernels.hpp NOT edited at all**. New members: only
  `Xscale/Hscale/grp_tail` (no `P.row_sd`!). Stub `scale_down_rows` added
  to the no-MMQ block (step 1). Whole diff re-verified against this tree
  (anchors, scale_down_rows semantics moe_mmq.cu 270–278/303, peer_now
  scope 3096/3524, take() byte semantics 1750–1782).
- 18 Oct: **PATCH APPLIED** — all 8 steps landed in prefill.cpp (stub 81,
  members 642, bind_stage_helper 1658–1660, allocs 1763/1790/1829, compact
  loop 3679–3718, non-compact loop 3761–3802, primary branch 3998–4003,
  combine nullptr 4130). `gather_native`'s 11th arg `zero_bytes` defaults
  to 0 (header 87) — the peer calls omit it, matching the group-tail
  handling already done by the existing `cudaMemsetAsync(..., MMQ_TAIL)`.
  kernels.cu/kernels.hpp untouched, as designed.
- 18 Oct: **GATE PASSED — the peer's NVFP4 rows are correct.** Built clean
  (`ninja strata` in /workspace/build, 0 warnings), deployed
  `engine/strata` = 2061d366 (backup `engine/strata.pre-nvfp4peer` =
  3bcb2ace), restarted with `STRATA_PF_PEER_NVFP4=1 STRATA_PEER_NO_P2P=1`
  via run-nvfp4-nvfp4.sh. Route line confirmed C ("no P2P: through mapped
  host memory in FP16"). `recall-test.py`: BOTH needles OK at 24933 prompt
  tokens, exact answer. **0 non-finite logits** since the peer start (was:
  every cold ≥20k prompt died in path B). Perf `pfmeasure-log.py
  nvfp4peer 3`: **median 2718 tok/s** (2717/2718/2729, all HTTP ok; an
  84k-token prompt ran at 2638) — slightly ABOVE the 2683 C-reference: the
  scale math (2 tiny row-scale kernels + swiglu_scaled per group, all on
  ps) costs nothing measurable. GPU1 tx 2-4 GB/s = the fp16 sums back over
  the host route, as expected.
- FALLBACK: `/home/florian/p2p-lab/nvfp4peer-zurueck.sh` (deploys
  strata.pre-nvfp4peer, starts without the peer env); a2-zurueck.sh (A2
  anchor) unchanged.
- STATUS: **DONE.** NVFP4 + peer-prefill (config C) is numerically correct
  and the fastest measured config. This patch can go into the mainline.

## Companion experiment: sums mode ON the P2P route (`STRATA_PF_PEER_SUMS_P2P=1`)

The same working session also wired sums mode onto the **P2P** route (second
commit on this branch): forward keeps P2P, only the way BACK changes - the
peer adds its rows into one weighted sum per token and sends T x N FP16
through mapped host memory instead of R x N FP32 rows over the P2P link.
Gate: `pp->p2p && pp->compact` + the env flag; the mapped buffers are the
sums' own (host_w / host_sum16 / host_pair / host_adds), and the route
banner gained a "forward over P2P, one weighted sum per token back" line.

Measured on THIS box (two PCIe GPUs, the P2P link at 5.5 GB/s - slower than
PCIe to RAM): correct (recall gate passes both needles, 0 non-finite logits)
but **2259 tok/s vs 2683 on the plain host route** - the host route wins
here, which is also why config C is the deployed default. It stays opt-in
because the trade flips on boxes where P2P beats PCIe (NVLink bridges,
newer boards): there the forward keeps the fast link and the return moves
~4x less data than the FP32 rows. Anyone on such a box: try
`STRATA_PF_PEER_NVFP4=1 STRATA_PF_PEER_SUMS_P2P=1` (no `STRATA_PEER_NO_P2P`)
and compare against the plain P2P route - the banner line tells which route
is live. The non-compact twin is NOT covered (it always copies its rows
back over P2P as well, so the sums would save nothing).

