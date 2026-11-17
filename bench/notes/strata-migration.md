# Strata V620 Migration & Benchmark Record

AMD Radeon Pro V620 (gfx1030) + Threadripper PRO 3975WX, running Qwen3.8-Flash-Next locally via Strata.
Started 2026-10-12. Rewritten 2026-11-17 for clarity: current state first, history compressed.

**Where the detail went.** This file is the working reference. Every experiment keeps a recreate recipe
(§5). Full unabridged detail — per-phase tables, logs, theories, all numbers — is in git history:
`git show c8d3f3b2:bench/notes/strata-migration.md` (pre-cleanup snapshot, 1,282 lines), and in the
result JSONs under `~/git/llm-benchmark/results/` (names cited per experiment).

---

## 1. Current state (2026-11-17)

### Machine

- Azure NVv5-class VM: Threadripper PRO 3975WX (32 cores), **V620 32 GB (gfx1030/RDNA2, 40 CU)**,
  207 GB RAM. ROCm 10.0 at `/opt/rocm`. `/var/lib/ollama` (sdb, 984 GB) holds the models, ~215 GB free.
- Power cap ~180 W (deliberate, user-set; card is pinned there of 265 W spec). Junction settles ~85 °C
  under load; sclk wanders 1520–2470 MHz. `rocm-smi --setperfdeterminism` is NOT supported on this card.
- memlock limit is 8 MiB, so the PLE table mlock fails — harmless but noisy; raising needs sudo (pending).

### What is serving right now

- **UD-IQ1_M lane** (Unsloth Dynamic v3.1, 1-bit, 63.3 GiB) on port 8080, swapped in 2026-11-09 for a
  real-work trial. Model id `qwen3.8-flash-next-ud-iq1m`. Config `/tmp/ud-iq1m-serve.json`
  (daily-equivalent knobs: parking-ON 8192, all 9 env vars, spec-5 + T9 draft-vocab recipe, wt-stack binary).
- Speed: fastest lane measured on this box (see §4). Quality: Unsloth's ladder says ~79.7% top-1 recovery —
  the trade was accepted for a trial; reverting is one config flip (below).
- Wrapper: `~/bin/strata-serve` (`stop` / `logs` / `status`; bare = foreground). Logs:
  `~/engines/logs/daily.log`, engine log per config `"log"` key.

### Rollback / fallbacks

- **Daily driver config** (untouched): `~/strata-v620-iq3_s-T9-draftvocab.json` — GSQ-RCO IQ3_S,
  backups `.bak-pre-v2/.bak-pre-mmq/.bak-pre-v4` next to it. Restoring = point the server at it again.
- **Binaries**: `~/git/wt-stack/build-hip/strata` (daily driver v4 — see version chain, §4);
  `~/git/wt-mmq/build-hip/strata` (v3, instant fallback); `~/engines/backup/strata-0.1.48.1-f2`,
  `~/engines/backup/strata-0.1.47-base`.
- **Parked UD lanes**: `unsloth-iq1m/` (68 GB), `unsloth-iq2xs/` (78 GB) under `/var/lib/ollama/gguf/`;
  packs `~/git/Strata-data/packs/{iq1m,iq2xs}`; configs `/tmp/ud-iq{1m,2xs}.json`.

### Version chain (what the daily driver is)

| # | What | Commit / branch |
|---|---|---|
| 1 | 0.1.48.1 (10-25 baseline) | `7ddc17a3` |
| 2 | 0.1.55 + F-2 (11-02) | `ff9d98b0`, worktree `~/git/wt-0155` |
| 3 | 0.1.55 + F-2 + MMQ-patched ggml (11-04) | worktree `~/git/wt-mmq`, branch `v620-daily-0155-f2` |
| 4 | **0.2.10 + F-2 + MMQ (11-05) — current** | `434dfe91` on `ffa80d73`, worktree `~/git/wt-stack`, branch `v620-stack-0210-f2-mmq` (pushed to fork) |

The MMQ part is our llama.cpp branch `v620-mmq-vop3p` (fork `andrewprock/llama.cpp`): 3 commits on the
engine's exact pin `3cf03257f` — VOP3P dot chains for q8_0 (patch 6) and IQ2_XS/IQ2_S (patch 7), plus the
`ggml_cuda_dp4a_z` helper (`56ccab041`). Built via `FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP` (0.1.55 has
no `STRATA_GGML_DIR`; 0.2.10 pins the same pin at CMakeLists:2110, so it drops in unchanged).

### Models on disk (as of 11-17)

- **IQ3_S GSQ-RCO** (daily weights): `/var/lib/ollama/gguf/IQ3_S/` (78 GB, 2 shards) — live daily shards,
  never delete under a running engine.
- **unsloth-iq1m/**, **unsloth-iq2xs/** — parked UD v3.1 lanes (benched 11-06, §4).
- `IQ3_S-abl/` (80 GB, abliterated; pack `packs/iq3_s-abl`), `maya-models/` (238 GB, user-managed),
  `mtp-draft/` (2.8 GB).
- Ollama service (systemd) is separate: 9 models, ~162 GB, user-managed (`ollama rm` is the clean way).
- Deleted lanes (history): DSV41 IQ2_XS (275 GB, deleted 11-06 at user request), unsloth UD-IQ4_XS +
  huihui abl-v2 + IQ3_XXS + coder IQ1_M + UD-Q4_K_XL/Q6_K_XL (k-quant expert dead end, deleted 10-16/11-03).

---

## 2. Canonical procedures

### Bench protocol (use this for any new A/B)

1. **Build clean.** Flags (setup's defaults — hand-built arms silently lose ~25% PP without them):
   `-DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_Q6K_EXPERTS=OFF
   -DCMAKE_HIP_ARCHITECTURES=gfx1030` + Release. Gotchas: the option is `STRATA_ENABLE_HIP` (not
   `STRATA_HIP`); `CMAKE_HIP_ARCHITECTURES` (not `HIP_ARCHITECTURES`). **After any flag/base change:
   `rm -rf build` and rebuild clean** — incremental builds over stale objects produced two false
   "regressions" (10-18, 10-25).
2. **Soak.** 1 warm-up pass after any engine restart, discarded (cold→settled PP is −5% @16k, −2% @64/128k).
   Counted passes: 3, take the median (med-of-N for N>1). For A/Bs: interleave boots when possible and
   reverse the second cycle; same-day, same thermal epoch. Never compare across sessions without
   soak-equalization.
3. **Parking.** Daily runs parking-ON (8192 MiB — session switching costs 1 s vs 18–100 s re-prefill).
   Benches run parking-OFF: `--conversation-cache-mib 0`. The bench never revisits a conversation, so
   parking-ON charges each tier ~0.2–1.0 s eviction — a bench artifact, not a code delta.
4. **Harness.** `~/git/llm-benchmark/bench_strata_tiers.py` — NO CLI args, env-configured:
   `STRATA_BASE`, `STRATA_MODEL`, `STRATA_API_KEY`, TIERS=16k,32k,64k,128k, `NUM_CTX=135168`,
   `NUM_PREDICT=1024`, temp 0. Output auto-named `strata_tier_pass_<ts>.json` in `results/`.
   Acceptance/`drafts` is NOT exposed via `/v1/status` — read `drafts accepted X of Y` from the serve log.
5. **Server.** Start from the worktree:
   `setsid nohup .venv/bin/python serve/server.py --engine strata --config <json> --port 8080 … & disown`.
   `--engine strata` is MANDATORY (0.1.55+ otherwise starts a bare engine — cost us a day, 10-31).
   Readiness = `grep ^ready:` in the log. Logs are append-mode: for fresh-boot checks use a byte offset.

### Measurement laws (learned the hard way)

- **Noise floors**: warm PP ±0.3%, TG ±3–4%. TG medians also drift day-to-day on an unchanged binary
  (demonstrated: 92.6 → 87.9 @16k overnight). **Treat single-pass TG deltas inside ±4% as noise**; never
  claim a TG win or loss from one run. `gen_tokens` varies 388–619 per pass (model stops early sometimes).
- **Real-text rule (E483)**: routing-sensitive switches must be measured on real text; synthetic prompts
  can flip the sign (happened with PREFILL_STREAM_MIN: +2.8% synthetic, −1.6%/+9.9% real).
- **Thermal epochs**: cool-start rows read ~2–3% above settled medians. Same-day A/B: compare first-pass
  to first-pass or re-soak between arms (S26 lesson).
- **`STRATA_PREFILL_TIMING=1`** phase timers over-count cross-stream overlap (sums run ~113% of timeline).
  Fine for attribution, never for totals. Keep it in bench configs, out of the daily config.
- **Flag-vs-binary check**: before citing a flag's effect, verify the binary actually contains the gate
  (`strings <binary> | grep <FLAG>`); an env var against an older binary is silently inert (bit us 10-24).
- **`pkill -f` self-matches the calling shell** — use bracketed patterns (`server[.]py`) or `pkill -x`,
  and never put the pkill and the restart in the same shell command (restart gets SIGTERM'd; hit twice).
- PP/TG conventions: PP = prompt-processing tok/s, TG = generation tok/s; numbers are med-of-N medians.

### Quality gates for any engine/config change

Temp-0 smoke must be **byte-identical** vs the incumbent (96-token prompt), or parity-verified via the
in-server check (`STRATA_HC_UPMIX_CHECK=2` + `STRATA_HC_UPMIX=1` — note: without `STRATA_HC_UPMIX=1`
the check silently runs nothing on gfx1030). Recall harness: `~/git/llm-benchmark/halo_needle.py`
(15/15 protocol, §5 row 12).

---

## 3. Decisions ledger

One line each; evidence numbers are med-of-N tier results unless noted. Full detail: §5 rows + git history.

**Adopted (in the daily driver now):**

| What | When | Evidence | Knob / where |
|---|---|---|---|
| rocBLAS tuning table (#1167) | 10-14 | PP +1.2–2.7% every tier, TG neutral | `STRATA_ROCBLAS_TUNING` (absolute path), `lib_dirs: [/opt/rocm/lib]` |
| SIMT1 QSA scorer (#1151) | 10-14 | 64K +2.7%, 128K +8.6% PP; 16K −1.8% (accepted) | `STRATA_SELECT_SIMT=1` |
| DPP wave-reduce | 10-14 | PP +0.3–1.0%, TG neutral; now upstream | in tree (`48d6aa11` lineage) |
| PLE_BATCH=1 | 10-19 | PP +11.6–14.0% every tier; TG +2.3–6% | `STRATA_PLE_BATCH=1` (see caveat, §4) |
| Calibration recipe (#1992) | 10-22 | TG +~13% vs uncalibrated same build | `--pcie-frac 0.00`, `--spec-min-p 0.70`, `STRATA_SH_STREAM=1` |
| T9 draft vocab | 10-23 | chat decode +5.2%, agent decode +6.8%, PP zero-cost | `--mtp-draft-vocab …/draft_vocab_en.bin` |
| Spec 5 + mtp-window 8192 | 10-15 | TG +6.2–8.5%, PP flat | `--spec 5 --spec-min-p … --mtp-window 8192` |
| F-2 skip-widen | 10-20 → promoted 10-24 | −3.2% GPU timeline @33k, bitwise-identical | `STRATA_HC_MIX_H16=1` |
| CPU_SHARE + HC_PERSIST (config v2) | 11-03 | short-TTFT −12%; TG +2.3–4.8% | `STRATA_PREFILL_CPU_SHARE=1`, `STRATA_HC_PERSIST=1` |
| MMQ-patched ggml (v3) | 11-04 | PP +0.9–1.4% every tier, TG wash | llama.cpp branch `v620-mmq-vop3p` |
| Stack 0.2.10 (v4) | 11-05 | PP +0.4% vs v3, 8/8 pairs; +4.7% vs 10-25 | worktree `wt-stack` |

**Measured, rejected / retired (do not re-litigate without new evidence):**

| What | Verdict | One-line reason |
|---|---|---|
| SIMT2 (`STRATA_SELECT_SIMT=2`) | retired 11-03 | loses PP every tier, −9.9% @128k, grows with context |
| V620 rocBLAS self-tune | retired 10-14/11-03 | tier wash ±0.2%; per-shape wins don't survive integration |
| DENSE_MMQ, TOPK_STREAM=0 | retired 11-03 | gains = session drift; noise at every tier |
| `STRATA_IO_THREADS`, `STRATA_PIPED_CONCURRENT`, `STRATA_PREFILL_STREAM_MIN` | retired 11-03 | real-text re-screen negative (E483 rule) |
| DOT2 attention score path | abandoned 10-19 | 1.9× in micro-bench, 1.48× SLOWER in engine (per-block fixed cost); opt-in `STRATA_ATTN_DOT2` stays in tree |
| reduce12 warp_sum port | abandoned 10-19 | 1.71× slower on gfx1030; opt-in stays in tree |
| fused_gr/gr DPP extension | abandoned 10-14 | only 2/19 sites convertible; hash identity never pinned |
| gr_upmix GEMM port (S28/S29) | dead 10-19 | SIMT GEMM is 38% of packed peak vs rocBLAS 77%; real prize was skipping widen → became F-2 |
| #2226 HC_PERSIST for decode-only | neutral single-card 10-14 | re-enabled via v2 config for prefill-side win (see above) |
| PR-A SH_STREAM default (as pushed) | dropped 10-24 | PP −2.3–2.5% on its target base; decode gate variant loses the win too |
| k-quant experts (Q4_K_XL/Q5/Q6/Q8) | dead end 10-15 | no GPU expert kernels; engine hard-fails; confirmed at scale in 10-16 matrix |
| `OLLAMA_KV_CACHE_TYPE=q8_0`, `use_mmap=false`, `--spec 6` | rejected 10-14 | −27% TG; +53 GB RAM no gain; worse cold |
| DFlash2 drafter | not adopted 11-05 | MTP wins upstream head-to-heads; drafter = separate 1.2 GB block, Qwen-only |
| INT8 compute conversions | dead end 10-14 (ISA) | INT8 = ¼-rate dot = FP32 parity on RDNA2; no WMMA/MFMA; packed FP16 is the only 2× path |

---

## 4. Performance record

### Tier evolution (PP/TG tok/s, med-of-N; PP is the stable metric, TG noisy)

⚠️ **PLE-flip caveat**: every row through 2026-10-19 ran `STRATA_PLE_BATCH=0` (per-token PLE arithmetic —
an unfixed-bug workaround inherited from the fork era). Post-flip rows are ~+12–14% PP at equal code;
compare pre-flip rows to post-flip rows only with that in mind.

| Stage | 16K | 32K | 64K | 128K |
|---|---|---|---|---|
| Ollama 0.34.2 (Q4_K_M) | 168.3/15.3 | 171.3/14.8 | 160.5/14.1 | 140.4/11.4 |
| Ollama 0.40.3 (Q4_K_M) | 188.4/16.2 | 185.5/17.1 | 172.6/15.9 | 149.8/15.2 |
| Strata fork (Q4_K_M) | 290.7/45.3 | 288.8/47.4 | 286.0/48.9 | 280.3/46.4 |
| 0.1.43 Q4_K_M | 527.7/51.1 | 540.5/49.1 | 536.9/50.8 | 516.3/49.6 |
| 0.1.43 IQ3_S | 793.1/65.1 | 807.6/68.9 | 789.3/68.2 | 737.9/67.8 |
| 0.1.43 Coder IQ1_M | 870.8/81.1 | 863.7/78.2 | 837.1/72.4 | 775.6/69.6 |
| 0.1.44.1 + #1167 + #1151 (IQ3_S) | 823.0/73.9 | 834.0/76.8 | 834.7/76.5 | 822.6/72.9 |
| 0.1.44.1 + wave-reduce (IQ3_S) | 828.8/72.7 | 841.6/75.5 | 842.3/74.0 | 827.5/74.2 |
| 0.1.44.1 UD-IQ4_XS, spec-5 | 835.5/73.2 | 857.8/73.7 | 869.4/73.3 | 861.0/70.0 |
| 0.1.48.1 + F-2, parking-OFF baseline | 923.2/92.2 | 944.9/91.5 | 953.8/88.7 | 943.8/86.2 |
| 0.1.53 new main ± F-2 (arms a/b) | 902.5 / 912.3 | 943.5 / 957.5 | 946.8 / 962.1 | 933.5 / 949.0 |
| 0.1.55 ± F-2 (arms c/d, med-of-6) | 910.9 / 927.3 | 931.9 / 948.5 | 940.1 / 957.2 | 931.1 / 947.7 |
| 0.1.55 + F-2, simt2 (retired) | 918.7/91.0 | 932.0/88.2 | 914.4/87.2 | 858.2/86.3 |
| v2: 0.1.55 + F-2 + CPU_SHARE + HC_PERSIST | 944.6/94.6 | 957.9/94.5 | 961.9/93.0 | 951.4/89.3 |
| v3: v2 + MMQ-patched ggml | 956.8/92.6 | 973.3/95.6 | 973.5/92.9 | 960.2/89.2 |
| **v4: 0.2.10 + F-2 + MMQ (current daily)** | **966.2/90.4** | **986.0/90.7** | **982.3/94.0** | **965.8/89.4** |

### Quant lanes (2026-11-06, same wt-stack binary, parking-OFF, med-of-3)

| Lane (disk size) | 16K | 32K | 64K | 128K | Status |
|---|---|---|---|---|---|
| IQ3_S daily (78 GB, ~3.1 bpw) | 966.2/90.4 | 986.0/90.7 | 982.3/94.0 | 965.8/89.4 | daily |
| UD-IQ2_XS (78 GB, ~2.5 bpw) | 979.7/96.0 | 984.5/96.6 | 980.6/94.3 | 964.6/93.0 | parked |
| **UD-IQ1_M (68 GB, ~1.6 bpw)** | **1000.6/100.9** | **1004.4/98.1** | **999.7/97.6** | **982.2/92.1** | **serving since 11-09** |
| DSV41 IQ2_XS (deleted 11-06) | 209.4/19.2 | 231.7/19.2 | — (KV > VRAM) | — | was: PP record-class, TG unusable |

Expert-bytes law on gfx1030: decode speed tracks expert bytes per token monotonically
(1-bit < 2.5-bit < 3.1-bit). UD-IQ1_M = first 1,000+ PP and first 100+ TG on this box. Quality ladder
(Unsloth's own numbers, not ours): IQ1_M 79.7% top-1 recovery / KLD 0.315; IQ2_XS has NO published
number; IQ3_S-class ≈ 85%+ (UD-IQ3_XXS 85.4%). Caveat: IQ2_XS carries one Q5_0-expert layer that
computes at 2.6–3.0% dot error on 0.2.10 (0.2.11 fixes it to ~1.2%); IQ1_M has no such layer.

### Records and regressions worth remembering

- Best PP per tier: 1000.6 / 1004.4 / 999.7 / 982.2 (UD-IQ1_M, 11-06). Best TG: 100.9 / 98.1 / 97.6 / 93.0.
- The 0.1.48→new-main "16k regression" (10-24/25) was the conversation cache evicting between tiers, not
  code; a "bisect" attempt died of single-boot noise (±1.3% > effect size) — interleaved pairs are mandatory.
- 0.1.48.1 prefill regression vs 0.1.47: bounded to ≤16k, −2.4% calibrated (10-24); never bisected (low value).
- 128k KV compaction cycle wall: 16.4 → 3.0 min across the record (5.5×).

---

## 5. Experiment index (enough to recreate; full data in git `c8d3f3b2`)

Format: **what / result / verdict / recreate**. "Old doc" = the pre-rewrite snapshot above.

1. **Ollama → Strata migration** (10-12/13). Result: Strata 0.1.43 ≈ 2–4× Ollama on every tier.
   Recreate: Ollama configs via `bench_ollama_tiers.py`; Strata via v043 configs.
2. **PR #1167 rocBLAS table** (10-14). +1.2–2.7% PP. ADOPTED. Recreate: env `STRATA_ROCBLAS_TUNING`
   pointing at `tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt` (shipped, build-locked); loader rejects
   version-mismatched tables silently (system 5.1.1 rejected; re-tune:
   `cmake --build build-hip --target tune_rocblas && LD_LIBRARY_PATH=/opt/rocm/lib ./build-hip/tune_rocblas
   --tuning-out <out>.txt`). Recreate A/B: configs `strata-iq3_s-pr1167{,-noTable}.json` (old doc).
3. **PR #1151 SIMT1 scorer** (10-14). 128K +8.6% PP (bigger than author's +3.2% — single card, no split).
   ADOPTED. Recreate: `STRATA_SELECT_SIMT=1`; mechanism line `qsa select: FP32 tiled scorer`.
4. **DPP wave-reduce** (10-14, `48d6aa11`). +0.3–1.0% PP micro→tier; bit-identical. ADOPTED (upstream now).
   Recreate: old doc "wave-reduce A/B" + result JSONs `…20261014_08*`.
5. **fused_gr/gr extension** (10-14). ABANDONED: 17/19 sites convert bit-identically, 2 ternary sites in
   `fused_gr.cu` produce hash diffs with unpinned root cause. Revisit only with a hot-path profile AND an
   ISA diff of those two kernels. Recreate: old doc bisect section.
6. **rocBLAS V620 self-tune** (10-14). 56/112 shapes pick different solutions on V620 vs the shipped
   table's 6900 XT, but tier deltas ±0.2% = noise. KEEP SHIPPED TABLE (portability datapoint).
   Recreate: `tune_rocblas` command above; protocol = ≥3 soak passes per arm, med-of-3.
7. **Thermal/power study** (10-14). ~180 W cap governs clocks; cold→settled −5% @16k / −2% @64/128k.
   Knowledge → §2 protocol. `setperfdeterminism` unsupported.
8. **RDNA2 ISA survey** (10-14). INT8 = ¼-rate dot (FP32 parity), packed FP16 (`v_pk_fma_f16`) the only 2×
   path, no WMMA/MFMA on gfx1030; dp4a shim routes correctly. Knowledge; kills INT8-conversion ideas.
9. **UD-IQ4_XS lane** (10-15). PP +3–4%, TG −8–12% vs IQ3_S (bandwidth tax). Lane deleted 11-03.
   Recreate: unsloth GGUF + `iq_pack.py --compat-bf16`, config clone with `--vram-reserve-mib 600`.
10. **Spec-5 A/B** (10-15). TG +6.2–8.5%, PP flat; no long-context acceptance decay on prose.
    ADOPTED (`--spec 5 --mtp-window 8192`). Recreate: old doc table; acceptance from serve log only.
11. **Halo ladder replication** (10-16). The Halo report's 128k acceptance collapse did NOT replicate here
    (0.79–0.85 at 128k). Knowledge: machine/fixture-specific upstream. Recreate:
    `~/git/llm-benchmark/halo_ladder.py`, data `halo_ladder_runs.json`.
12. **Needle recall test** (10-16). 15/15 both lanes (IQ4_XS spec-5, IQ3_S spec-4). The recall harness.
    Recreate: `~/git/llm-benchmark/halo_needle.py` (parametrize per lane; needs `STRATA_API_KEY`).
13. **6-lane keep/drop matrix** (10-16). IQ3_XXS best TG; Q4_K_XL ~60% slower (k-quant dead end confirmed);
    kept IQ4_XS/IQ3_XXS/abl at the time. Note IQ3_S base was re-adopted 10-18 as the standard lane; the
    other lanes were deleted 11-03. Recreate: `run_matrix.py` + configs `*-spec5.json` (old doc).
14. **huihui abl-v2** (10-18). Performance parity; byte-diff PROVED the README wrong — expert
    down-projections ablated in 43/48 layers. Lane deleted 11-03. Recall 15/15, refusal probe 0/6.
15. **Rebase + branch A/B matrix** (10-18/19). Local gfx1030 work rebased onto `477ac13c`; decode +13%
    = fork default + wave-reduce; PP parity across arms. Recreate: `tools/ab_engine.py --rounds 3`.
16. **STRATA_PLE_BATCH flip** (10-19). +10.8% PP @33k probe, +11.6–14% tiers, TG +2–6%. ADOPTED.
    The flag was a stale workaround for upstream #541 (batched-prompt stall, no fix upstream, but zero
    stalls here since the 0.1.4x prefill rewrite). If a stall ever reappears: `STRATA_PF_STEP_SYNC=1`
    to diagnose; revert to 0 as last resort. Baseline caveat lives in §4.
17. **S-series qsa-attention work** (10-18/19). Findings: `qsa attn` is instruction-issue-limited at ~23%
    of FP32 peak; the two "fixes" (DOT2, reduce12) both LOST in-engine (per-block fixed cost — micro-bench
    projections don't transfer to small blocks). Opt-in flags remain in tree. Recreate:
    `tests/hip/qsa_rows_bench.cpp` + `tests/hip/dot_score_bench.cpp` (build with repo shim flags, stop
    the server first — bench needs ~30–150 MB VRAM).
18. **S27–S29 hc-chain analysis** (10-19). gr_upmix port dead (rocBLAS GEMM 6.03 ms/launch at 77% packed
    peak vs SIMT 38%); the widen pass is the waste → led directly to F-2. Recreate:
    `tests/hip/upmix_gemm_bench.cpp`; `tools/hip/tune_rocblas` over engine shapes (all `keep_default`).
19. **F-2 skip-widen** (10-20 → 10-24). −1042 ms / −3.2% GPU timeline @T=32005, bitwise-identical
    (parity harness 21 shapes + in-server check). Reconciliation: per-phase sums over-count cross-stream
    overlap (~4.6 s baseline), so Δphases −2242 ms vs Δtimeline −1042 ms is fully explained — no missing
    time. ADOPTED; committed as `392586db` lineage, cherry-picked onto every later base.
    Recreate: parity `tests/hip/mix_h16_parity.cu`; A/B via `STRATA_HC_MIX_H16=0/1` same binary.
20. **Calibration** (10-22, on 0.1.47; re-verified on 0.1.48.1+F-2 10-24). Recipe:
    pcie-frac 0.00, spec-min-p 0.70, SH_STREAM=1. Runs via `./setup.sh --calibrate` → `tools/calibrate.py`
    (686 s). NOTE: never run while the daily server is live (restarts the model repeatedly).
    `strata launch --tune quick|thorough` is the Terminal interface (never used here).
21. **T9 draft vocab** (10-23). +5.2/+6.8% decode, zero prefill cost. ADOPTED. Rejected in same sweep:
    spec 4, adapt-overlap, pcie-frac 0.35, T7+T9 combo, vram-reserve 200 on agent, no mtp-window.
    Recreate: `--mtp-draft-vocab /home/drewster/git/Strata-main/data/draft_vocab_en.bin`.
22. **PR-A SH_STREAM default** (10-23/24). PP −2.3–2.5% on 0.1.47 base (opposite sign vs 0.1.44.1 —
    attribution does not carry across bases). DROPPED. Upstream later shipped per-PC calibration (#1992).
23. **Parking discovery** (10-25). Conversation cache evict = ~1 s after big sessions; parking ON daily /
    OFF bench (§2). Recreate: `/tmp/probe-nopark.sh`, `/tmp/agent-loop.py` patterns (old doc).
24. **New-main A/Bs rounds 1–2** (10-29, 11-01). Interleaved, parking-OFF; F-2 wins med-of-6 (6/6 sweep)
    → promoted wt-0155. Server.py `--engine strata` gotcha found here (§2).
25. **SIMT1 vs SIMT2** (11-03). simt2 loses PP, worse with context; 128k PP identical across boot orders
    (not thermal). RETIRED; `STRATA_SELECT_SIMT=1` kept.
26. **6-knob screen** (11-03). CPU_SHARE + HC_PERSIST ADOPTED (v2); rocBLAS retune / DENSE_MMQ /
    TOPK_STREAM retired. Recreate: `/tmp/env_screen.py` pattern — 1 boot per candidate, soak + 2 counted.
27. **Env-var screens** (11-03/04). IO_THREADS/PIPED_CONCURRENT/PREFILL_STREAM_MIN: synthetic positive,
    real-text negative → RETIRED. Establishes the E483 real-text rule (§2).
28. **MMQ Vop3p port** (11-04). Patches 6+7 on pin `3cf03257f` (+ helper commit `56ccab041`); iq3_s kernel
    30.9→31.7 TFLOPS author-side; here PP +0.9–1.4% every tier, TG wash. ADOPTED (v3).
    Recreate: llama.cpp branch `v620-mmq-vop3p` + `FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP`.
29. **Stack v4** (11-05). 0.2.10 + F-2 (clean cherry-pick `434dfe91`) + MMQ; PP +0.4% vs v3, 8/8 pairs;
    captures upstream's own +2–3%. ADOPTED. Recreate: wt-stack + build flags (§2).
30. **DFlash2 vs MTP** (11-05). DFlash2 = separate 1.2 GB drafter, Qwen-only; upstream head-to-heads favor
    MTP. KEPT MTP. Recreate: `--draft dflash2 --draft-model PACK` on 0.1.55+.
31. **DSV41 IQ2_XS eval** (11-06). First gfx1030 DSV41 run anywhere: PP 209–231 (ties best documented
    single card), TG 19.2 (reasoning model, 4K KV = 1.2 GB → 64k+ impossible single-card). Curiosity lane;
    deleted 11-06 at user request. Recreate: `dsv41` engine via config `"exe": strata-dsv41`,
    `-DSTRATA_BUILD_DSV41=ON`; pack recipe in old doc/`/tmp/dsv41-*.log` era notes.
32. **UD-IQ2_XS / UD-IQ1_M benches** (11-06). See §4 quant-lane table. Recreate: configs
    `/tmp/ud-iq{2xs,1m}.json`, packs `packs/{iq2xs,iq1m}`, `HF_HUB_DISABLE_XET=1` for fast downloads.
33. **0.2.11 review** (11-06). No gfx1030 changes (all new perf work gated to NVIDIA/gfx12/split-engine);
    llama.cpp pin unchanged; F-2 cherry-picks clean; server.py fixes worth having (streamed tool-args,
    branch races, 60s no-progress disconnect). UD v3.1 re-uploads + Q5_0 dot fix noted (§4 caveat).
    Recreate: pull main, re-run the review checklist in the session log.

---

## 6. Second V620 runbook (on arrival)

Layer split is the play for PP (not helper mode). Evidence from others: 2×V620 report (`bench/results/2x-v620.txt`,
untracked): 4K 893.4 / 16K 1,380.3 PP, decode 68.1/70.7; 0.1.41 2×6900 XT: split ≈ +50% mid-tier PP, decode
+32–39%; #3930 (2×2080 Ti): decode 35.2→51.0 with `STRATA_DECODE_TIMING` window 70→38 ms. Nothing measured
on THIS box with two cards yet — all projections.

1. Stop the daily server. Edit config: `"gpu": [0,1]`, `"layer_split": "auto"`. Boot and WATCH the log
   split lines (which layers went where).
2. Bench parking-OFF vs the v4 row (§4). Sanity: prefill flat from 4k→16k = second card idle — stop and debug.
3. Then one at a time: `--pipeline-windows 2`, `STRATA_SPLIT_OVERLAP=1`, re-run calibration (never while
   serving daily). Re-check `STRATA_HC_PERSIST` (author's gains were on a split config).
4. DSV41 multi-GPU is native (2-card boot collapses `tensor parallel 2 -> 1`); 128k context fits 2 cards
   only without the drafter. DSpark economics flip at 2 cards (drafter fits; ~18% expert-cache coverage).

---

## 7. Open items

1. **Second V620** — runbook above; migration-doc TODO closes on arrival.
2. **0.2.11 stack rebase** — F-2 verified to apply clean; expected single-card delta ~0; win = server.py
   fixes. Decision: build `0.2.11 + F-2 + MMQ` and A/B vs v4, or hold.
3. **#3381 RDNA2 prompt-attention port** — diff at `~/strata-upstream/pr-3381-prompt-attn-rdna2.diff`
   (373 lines) does NOT apply to 0.1.55 or 0.2.10 (`prefill.cpp:3435` drift); needs a manual port.
   Author claims QSA prompt chunk −29%.
4. **memlock raise** — needs interactive sudo; PLE mlock fails at 8 MiB.
5. **Upstream drafts** — two drafts in `~/strata-upstream/` (rocBLAS-retune-neutral, simt2-vs-simt1);
   PR descriptions pending for the three pushed fork branches.
6. **UD-IQ1_M trial verdict** — keep serving (current) or revert to IQ3_S daily based on real-work quality.
7. **STRATA_DECODE_TIMING=1 bench boot** — still the candidate predictor for split-TG attribution (runbook §6, #3930).

---

## 8. Pointers

- **Worktrees**: `~/git/Strata` (main @ 0.2.11 `9f7d977d`), `~/git/Strata-main` (parked
  `parked/pr-a-gfx103x-default`), `~/git/wt-0155` (detached `ff9d98b0`), `~/git/wt-mmq`
  (`v620-daily-0155-f2`), `~/git/wt-stack` (`v620-stack-0210-f2-mmq` — this doc's home).
- **Forks**: `andrewprock/Strata` — branches `v620-daily-0155-f2`, `v620-stack-0210-f2-mmq`;
  `andrewprock/llama.cpp` — branch `v620-mmq-vop3p`. gh CLI authed (https).
- **Configs**: daily `~/strata-v620-iq3_s-T9-draftvocab.json` (+3 backups); UD lanes `/tmp/ud-iq{1m,2xs}.json`
  (+`-serve` variants); bench arms `/tmp/ab-*.json`, `/tmp/iq3s-*.json`.
- **Packs**: `~/git/Strata-data/packs/{iq3_s,iq1m,iq2xs,iq3_s-abl}`; MTP head `~/git/Strata-data/mtp/rt`.
- **Results**: `~/git/llm-benchmark/results/strata_tier_pass_*.json` (timestamps = run time).
- **Backups**: `~/engines/backup/{strata-0.1.47-base,strata-0.1.48.1-f2}`; this doc's pre-rewrite copies:
  `~/strata-migration.md.bak-20261117-pre-rewrite`, `~/strata-migration.md.bak-20261117-post-redundancy`,
  `~/strata-migration-redun.diff`; git history: `c8d3f3b2` (original) → `ba97795e` (dedup) → this rewrite.
- **Env vars in every lane config (the 9)**: `STRATA_WATCHDOG_S=300`, `STRATA_PLE_BATCH=1`,
  `STRATA_HIP_PROMPT_F16=1`, `STRATA_SELECT_SIMT=1`, `STRATA_SH_STREAM=1`, `STRATA_ROCBLAS_TUNING=<abs>`,
  `STRATA_HC_MIX_H16=1`, `STRATA_PREFILL_CPU_SHARE=1`, `STRATA_HC_PERSIST=1`.
