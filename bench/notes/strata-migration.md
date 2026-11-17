# Strata Migration — Benchmark Record (2026-10-12 → 2026-10-20)

**CURRENT STATUS (start here, 2026-10-20):** F-2 skip-widen (`STRATA_HC_MIX_H16=1`) is implemented,
verified (bitwise parity, 21 shapes + in-server), A/B-measured (**-1042 ms / -3.2% end-to-end GPU
timeline at T=32005**, med-of-3, same-day), and committed on branch `f-hc-skip-widen-f2`
(commit `392586db`). Read the F-2 sections at the bottom of this file for the full story,
including the timing reconciliation (phase timers over-count cross-stream overlap).
**Awaiting user decision:** promote the flag to the production config now, or run a bracketing
re-run first — see "Open decisions & pending tasks (as of 2026-10-20)" near the end for the
options, the task list, and the branch inventory for the planned PRs.

## Machine
Azure NVv5-class VM: Threadripper PRO 3975WX (32C), **AMD Radeon Pro V620 32 GB (gfx1030/RDNA2)**, 207 GB RAM, ~480 GB free on /var/lib. ROCm 10.0 at /opt/rocm.

## Models on disk
- **Q4_K_M (the "tuned" model)**: Ollama blob `/var/lib/ollama/models/blobs/sha256-89059c88...` (103 GiB, contains PLE table, NO MTP blocks)
- **IQ3_S GSQ-RCO** (base model): `/var/lib/ollama/gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`
- **IQ3_S abliterated**: `/var/lib/ollama/gguf/IQ3_S-abliterated/` (downloaded, pack NOT built yet)
- **Coder IQ1_M**: `/var/lib/ollama/gguf/coder/IQ1_M/` (58 GiB, 2 shards) — **Strata pack already built**
- **unsloth UD-Q4_K_XL** (103.7 GiB, 4 shards) + **MTP drafts** (`MTP/mtp-...-{Q4_K_M,shared-Q4_K_M}.gguf`): `/var/lib/ollama/gguf/unsloth-q4/` — **never testable on the GPU lane** (k-quant experts, engine-blocked; see Q6_K_XL row below). The llama.cpp draft-pairing failure note stays historical.
- **unsloth UD-IQ4_XS** (87.2 GiB, 3 shards; shard 1 is a 10 MiB metadata stub, shard 2 carries a 26.8 GiB quantized PLE table): `/var/lib/ollama/gguf/unsloth-iq4xs/UD-IQ4_XS/` — **tested 2026-10-15, see results below**
- **unsloth UD-Q6_K_XL** (157.5 GiB, 6 shards): `/var/lib/ollama/gguf/unsloth-q6/UD-Q6_K_XL/` — **DEAD END: engine-blocked.** K-quant experts (Q6_K/Q8_0, ggml types 14/8) have no GPU expert kernels; `generate.cpp` hard-fails with "no GPU kernels for" (only the i-quant family is servable on the GPU lane). Pack `packs/q6xl` exists, orphaned. Same block applies to the UD-Q4_K_XL shards and any Q5/Q8_0/BF16 expert lane — the bpw ceiling for GPU expert serving is IQ4-class.
- Strata tokenizer packs (one per quant format — the old "iq3_s pack works for all quants of this family" note was WRONG, each format's expert layout needs its own pack): `packs/iq3_s`, `packs/iq3_xxs`, `packs/iq4xs`, `packs/q4km` (k-quant, --compat-bf16), `packs/q6xl` (orphaned)

## Engines / trees
- `~/git/Strata` — fork lineage (local main @ 99f3dbd, engine ≈0.1.38). NOT user-modified (inherited fork).
- `~/git/Strata-main` — **git worktree at upstream origin/main a0116b27 (v0.1.43)**; engine **built**: `build-hip/strata` (flags: `-DSTRATA_ENABLE_HIP=ON -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030`). ⚠️ correct flag is CMAKE_HIP_ARCHITECTURES (HIP_ARCHITECTURES silently ignored → wrong-arch binary). NOTE 2026-10-14: the tree is actually v0.1.44.1 now (dcc6ed9f) — engine numbers below still match the v0.1.43 record.
- `/tmp/strata-pr1167` (PR #1167 worktree, 2026-10-14) — detached at PR head db64983 (v0.1.43.1 + rocBLAS tuning table), built with the SAME flags (incl. STRATA_PREFILL_MMQ=ON — first build had it OFF; re-matched before benchmarking). VOLATILE (/tmp) — rebuild from `git fetch origin db64983` + `git worktree add` if gone.
- `.venv` symlinked from fork tree into worktree. `bench.sh` lives only in the worktree.
- Ollama **0.40.3** (systemd, `/etc/systemd/system/ollama.service.d/override.conf`; `OLLAMA_KV_CACHE_TYPE=q8_0` line **commented out** — rejected).
- llama.cpp: `~/git/llama.cpp-rdna2/build/bin` (build 11169-era). llama.cpp **cannot do MTP** for this model (GGUF lacks MTP blocks; unsloth draft → `output_hc_norm.weight not found` vs Ollama blob).

## Harness (all in ~/git/llm-benchmark)
- `bench_strata_tiers.py` — TIERS=16k,32k,64k,128k; STRATA_MODEL env; nonce anti-cache; POSTs /v1/chat/completions, reads /v1/status `last_timings`. Same fixtures as other lanes (`llm-context-bench` prompts `regular-{tier}.txt`, NUM_CTX=135168, temp 0, max 1024).
- `bench_ollama_tiers.py` (EXTRA_OPTIONS JSON env), `bench_llamacpp_tiers.py`.
- Launchers: `run_strata_043.sh` (IQ3_S + 0.1.43), `run_strata_q4km_043.sh` (Q4_K_M + 0.1.43), `run_strata_coder_043.sh` (Coder + 0.1.43), `run_strata_server.sh` (Q4_K_M + fork engine, legacy), `run_llamacpp_mtp.sh` (broken draft pairing).
- Strata configs: `Strata-main/strata-iq3_s-v043.json` (0.1.43 IQ3_S: ple-io ram, spec 4, int8 KV, kv-resident 32768, expert-cache auto, **STRATA_HIP_PROMPT_F16=1 REQUIRED**), `Strata-main/strata-q4km-v043.json` (spec 3, F16 on), `Strata-main/strata-coder-iq1_m-v043.json` (spec 4, F16 on), `Strata/strata-q4km.json` + `Strata/strata-iq3_s.json` (fork-era).
- PR #1167 configs (2026-10-14): `llm-benchmark/strata-iq3_s-pr1167.json` (PR engine + `STRATA_ROCBLAS_TUNING=…/tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt`) and `strata-iq3_s-pr1167-noTable.json` (same engine, no table) — both are the v043 config with `exe` pointed at the PR build; run via `server.py --config <file> --port 8080` then `bench_strata_tiers.py`.
- Daily-driver config `strata-iq3_s-v043.json` carries: `STRATA_ROCBLAS_TUNING` (#1167), `STRATA_SELECT_SIMT=1` (#1151). `STRATA_HC_PERSIST` from #2226 was TESTED and REMOVED (neutral on single card; re-add for a layer split).
- **Server state (2026-10-16)**: Strata 0.1.44.1 + #1167 + #1151 + #2226 merged (`b5962db4`) + gfx103x wave-reduce (`48d6aa11`) — **IQ3_S-abliterated lane on :8080** running `strata-iq3_s-abl-v043.json` (user testing it now). IQ4_XS lane default `strata-iq4xs-spec5.json` (`--spec 5 --mtp-window 8192`, adopted 2026-10-15); IQ3_S `strata-iq3_s-v043.json` unchanged (spec 4). Ollama on :11434.

## Headline results
**16K tier, IQ3_S lane (same config):** fork engine 408.5 PP / 61.6 TG → **0.1.43: 795.7 (cold) / 810.9 (warm) PP, 67.1/71.5 TG, 27 s wall**. Upgrade = 2× PP, +9–16% TG. Harness-vs-harness engine delta: 4K PP +86%, 16K PP +110%, TTFT −46/−52%.

**Q4_K_M lane, fork engine:** 16K 290.7 PP / 45.3 TG; 4-tier pass: PP 290.7→280.3 flat, TG 45.3→46.4 flat (NO long-context decay), 128K wall 487 s. vs Ollama 0.40.3: +87% PP, +205% TG at 128K.

**Cumulative 2026-10-14 session (IQ3_S lane, 0.1.43 → 0.1.44.1 + #1167 + #1151):** 16K 793.1→823.0 PP (+3.8%), 32K 807.6→834.0 (+3.3%), 64K 789.3→834.7 (+5.8%), **128K 737.9→822.6 PP (+11.5%), wall 188→169 s (−19 s)**; TG +5–12% (from 0.1.44's shared-expert stream, not the PRs). Source: master table rows + tier JSONs in results/.
**Q4_K_M lane, 0.1.43 (full pass 2026-10-13):** 16K 527.7/51.1, 32K 540.5/49.1, 64K 536.9/50.8, 128K 516.3/49.6 — zero decay holds, 128K wall 268 s (−45% vs fork). vs Ollama 0.40.3 at 128K: +244% PP, +226% TG. Config: `strata-q4km-v043.json` (spec 3, F16 on).

**Ollama 0.34.2 → 0.40.3** (Q4_K_M): PP 168→188 (16K), TG 15.3→16.2; 128K PP 140→150, TG 11.4→15.2. Total pass 29.5→27.3 min.

**FULL 4-TIER PASSES ON 0.1.43 (2026-10-13):** Coder IQ1_M: 870.8/81.1, 863.7/78.2, 837.1/72.4, 128K 775.6/69.6 (wall 181 s). IQ3_S: 793.1/65.1, 807.6/68.9, 789.3/68.2, 128K 737.9/67.8 (wall 188 s). Q4_K_M: 527.7/51.1, 540.5/49.1, 536.9/50.8, 128K 516.3/49.6 (wall 268 s). All zero-to-mild decay; Coder is the fastest lane at every tier.

**MASTER EVOLUTION TABLE (PP/TG tok/s, wall per tier) — canonical record:**

| Stage | 16K | 32K | 64K | 128K |
|---|---|---|---|---|
| Ollama 0.34.2 (Q4_K_M) | 168.3/15.3 (122s) | 171.3/14.8 (218s) | 160.5/14.1 (442s) | 140.4/11.4 (985s) |
| Ollama 0.40.3 (Q4_K_M) | 188.4/16.2 (110s) | 185.5/17.1 (200s) | 172.6/15.9 (410s) | 149.8/15.2 (917s) |
| Strata fork (Q4_K_M) | 290.7/45.3 (45s) | 288.8/47.4 (75s) | 286.0/48.9 (142s) | 280.3/46.4 (487s) |
| Strata 0.1.43 Q4_K_M | 527.7/51.1 (40s) | 540.5/49.1 (71s) | 536.9/50.8 (132s) | 516.3/49.6 (268s) |
| Strata 0.1.43 IQ3_S | 793.1/65.1 (28s) | 807.6/68.9 (48s) | 789.3/68.2 (91s) | 737.9/67.8 (188s) |
| Strata 0.1.43 Coder IQ1_M | 870.8/81.1 (24s) | 863.7/78.2 (44s) | 837.1/72.4 (86s) | 775.6/69.6 (181s) |
| Strata 0.1.43.1 + PR #1167 table (IQ3_S) | 823.7/65.3 (26s) | 827.0/70.9 (46s) | 804.6/68.5 (90s) | 753.6/68.0 (184s) |
| Strata 0.1.44.1 merged + PR #1167 table (IQ3_S) | **839.4/68.5 (26s)** | **840.8/75.7 (45s)** | **815.7/74.8 (88s)** | **759.2/74.5 (182s)** |
| 0.1.44.1 merged + #1167 + #1151 SIMT=1 (IQ3_S, med of 2) | 823.0/73.9 (26s) | 834.0/76.8 (46s) | **834.7/76.5 (86s)** | **822.6/72.9 (169s)** |
| 0.1.44.1 merged + #1167 + #1151 + #2226 (IQ3_S) [1] | — | — | — | — |
| 0.1.44.1 merged + #1167 + #1151 + #2226 + gfx103x wave-reduce (IQ3_S, med of 2) | 828.8/72.7 (27s) | 841.6/75.5 (45s) | **842.3/74.0 (85s)** | **827.5/74.2 (168s)** |
| 0.1.44.1 daily driver, settled-state med-of-3 (IQ3_S) [2] | 803.4/77.3 (26s) | 819.2/73.2 (46s) | 827.5/73.2 (86s) | 820.3/72.1 (169s) |
| 0.1.44.1 **UD-IQ4_XS** lane (unsloth, 4.1 bpw) [3] | 829.2/67.6 (26s) | 851.2/67.4 (46s) | **863.9/66.9 (84s)** | **856.2/63.6 (162s)** |
| 0.1.44.1 UD-IQ4_XS lane, spec 5 + mtp-window 8192 — **lane default** [4] | 835.5/73.2 (26s) | 857.8/73.7 (45s) | 869.4/73.3 (82s) | 861.0/70.0 (162s) |
| 0.1.48.1 + F-2 parking-OFF baseline (IQ3_S) [5] | 923.2/92.2 | 944.9/91.5 | 953.8/88.7 | 943.8/86.2 |
| 0.1.53 new main, arm-a (IQ3_S) [6] | 902.5/92.3 | 943.5/92.1 | 946.8/90.1 | 933.5/86.9 |
| 0.1.53 new main + F-2, arm-b (IQ3_S) [6] | 912.3/92.3 | 957.5/88.1 | 962.1/85.6 | 949.0/86.0 |
| 0.1.55 new main, arm-c (IQ3_S) [7] | 910.9/92.4 | 931.9/90.6 | 940.1/88.5 | 931.1/86.4 |
| 0.1.55 new main + F-2, arm-d (IQ3_S) [7] | **927.3/90.3** | **948.5/90.0** | **957.2/86.4** | **947.7/86.2** |
| 0.1.55 + F-2, simt1 = daily re-measure (IQ3_S) [8] | 927.5/91.7 | 951.1/92.1 | 961.1/89.0 | 952.7/86.7 |
| 0.1.55 + F-2, simt2 (IQ3_S) [8] | 918.7/91.0 | 932.0/88.2 | 914.4/87.2 | 858.2/86.3 |
| 0.1.55 + F-2, daily driver v2: CPU_SHARE + HC_PERSIST (IQ3_S) [9] | 944.6/94.6 | 957.9/94.5 | 961.9/93.0 | 951.4/89.3 |
| 0.1.55 + F-2 + MMQ-patched ggml, daily driver v3 (IQ3_S) [10] | 956.8/92.6 | 973.3/95.6 | 973.5/92.9 | 960.2/89.2 |
| 0.2.10 + F-2 + MMQ-patched ggml, daily driver v4 (IQ3_S) [11] | 966.2/90.4 | 986.0/90.7 | 982.3/94.0 | 965.8/89.4 |
Footnotes:
[1] #2226 is a decode-only change; agent-session A/B neutral (see results); no tier row measured.
[2] Settled state: card hot for 1.5 h (2026-10-14 ~18:00). The −2 to −3% delta vs cool-start rows is the thermal epoch, not a regression (see thermal note below).
[3] 2026-10-15, 1 soak + med-of-3.
[4] 2026-10-15, 1 soak + med-of-3; this is the lane default.
[5] Canonical parking-OFF baseline: 2026-10-25, 1 soak + med-of-3.
[6] Interleaved A/B pair (2026-10-29), parking-ON; arm-b's TG regression vs the baseline is recorded in the session section.
[7] Med-of-6, parking-OFF (2026-11-01). F-2 verdict: +1.8% PP at every tier (all 6 passes win), TG cost ≤ 2.4% — kept. Daily driver promoted to c62d5e3a + F-2 (wt-0155). Wall times for these rows live in the session sections.
[8] SIMT A/B (2026-11-03): interleaved, 2 cycles with reversed boot order, 1 soak + 2 counted per boot, med-of-4 per arm, parking-OFF. simt2 loses PP at every tier (−0.9/−2.0/−4.9/−9.9%), the gap grows with prompt length; TG flat. simt2's 128k PP was 858.2–861.4 across all six passes and identical from both boot positions — not thermal. Retired; explicit STRATA_SELECT_SIMT=1 kept (upstream 0.1.56+ defaults simt2 on gfx103x). Contradicts the RX 6900 XT community result; V620 report drafted (~/strata-upstream/).
[9] Knob screen session (2026-11-03, fast path — 1 boot per candidate, soak + 2 counted, or TTFT loops): V620 rocBLAS retune RETIRED (tier PP wash ±0.2%; short-prompt TTFT +2.8% slower; the tuner must be pointed at /opt/rocm/core-10.0 — the distro-linked first run stamped 5.1.0 and the engine refused the table at boot). PREFILL_CPU_SHARE=1 ADOPTED (short-prompt TTFT −12%: 1054 vs 1198 ms, n=12 each; inert at ≥1024-token chunks). HC_PERSIST=1 ADOPTED (screens +2.3%/+4.8% TG; combo confirm: 8/8 counted TG passes above the control median at every tier, PP flat 32k–128k). DENSE_MMQ RETIRED (its +1% PP also appeared on the decode-only knob = session drift). TOPK_STREAM=0 RETIRED (128k: +0.7% PP / +0.2% TG = noise). memlock raise pending (sudo; PLE table mlock still fails at the 8 MiB limit). Daily config v2 promoted (parking-ON + CPU_SHARE + HC_PERSIST; backup .bak-pre-v2). The simt1 row doubles as a day-over-day reproducibility check vs [7]: PP +0.0–0.5%, TG +0.6–3.0%. Combo row above is med-of-4 (2 boots × 2 counted passes, parking-OFF, 2026-11-03): TG +3.2/+2.6/+4.5/+3.0% vs the same-day simt1 control at every tier, PP +1.8/+0.7/+0.1/−0.1%; vs the 10-25 baseline PP +2.3/+1.4/+0.8/+0.8% and TG +2.6/+3.3/+4.8/+3.6% — above baseline at every tier on both metrics.
[10] Patched-ggml adoption (2026-11-04): the daily engine (ff9d98b0, 0.1.55 + F-2) rebuilt against Vop3p's RDNA2 MMQ patches on the engine's exact llama.cpp pin 3cf03257f — patch 6 (q8_0 MMQ dot chains start as VOP3P on RDNA2 via a ported ggml_cuda_dp4a_z helper; iq3_s kernel 30.9→31.7 TFLOPS on the author's E378 b8) + patch 7 (IQ2_XS/IQ2_S q8_0_16 chains) — together they cover every expert type of the IQ3_S pack. The helper comes from patch 5's common.cuh hunk (patches 6/7 call it; committed separately 56ccab041 in /tmp/llamacpp-mmq); RDNA2 is defined by ggml's vendors/hip.h on gfx103x, no build flag needed. Build: wt-mmq worktree + FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP (0.1.55 has no STRATA_GGML_DIR); the Gyro TQ overlay ran intact. Quality gate: 96-token temp-0 smoke byte-identical vs wt-0155 (the author also reports byte-identical logits). A/B (soak discarded + med-of-4 per arm, interleaved boots base→mmq→mmq→base, parking-OFF combo config): base 947.9/91.7 · 959.7/91.5 · 963.2/91.0 · 951.3/88.9 — PP within ±0.3% of the [9] combo row, a clean day-over-day reproducibility check; mmq 956.8/92.6 · 973.3/95.6 · 973.5/92.9 · 960.2/89.2 — PP +0.94/+1.42/+1.07/+0.94% at 16k/32k/64k/128k (tight within-arm spreads), TG never worse (TG medians wander ±3% day-to-day, so the +1–4.4% TG readings are recorded as noise-adjacent, not claimed as a win). Daily config exe → wt-mmq (backup .bak-pre-mmq); wt-0155 kept as instant fallback. Queued next: #3381 RDNA2 prompt-attention port — diff saved (~/strata-upstream/pr-3381-prompt-attn-rdna2.diff) but does NOT apply to 0.1.55 (src/prefill/prefill.cpp:3435 drift from the 2026-10-19-era base); needs a manual port. Author's claim for it: QSA prompt-attention chunk 500→355 ms (−29%), PP −3.5% at 34K/133K, quality-gated.
[11] Stacked-arm adoption (2026-11-05): daily driver promoted from 0.1.55 to upstream main ffa80d73 (v0.2.10, 708 commits ahead) + F-2 cherry-picked clean onto main (new commit 434dfe91; the F-2 diff needed no conflict resolution) + the same v620-mmq-vop3p llama.cpp branch via FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP — 0.2.10 pins the identical llama.cpp pin 3cf03257f (CMakeLists:2110), so the three MMQ commits drop in unchanged. Worktree wt-stack, branch v620-stack-0210-f2-mmq pushed to the fork. Quality gate: temp-0 smoke byte-identical vs the wt-mmq daily. A/B (soak discarded + 2 counted per arm, interleaved stack→ours→ours→stack, parking-OFF, same day): stack 966.2/90.4 · 986.0/90.7 · 982.3/94.0 · 965.8/89.4 vs ours-v3 961.5/87.9 · 981.7/95.0 · 978.3/90.7 · 962.1/89.9 — PP +0.49/+0.44/+0.41/+0.38% with stack ahead in 8/8 individual pairings; TG mixed signs inside the ±3–4% noise band = wash. Context: upstream main alone measured 967.0–989.6 PP (1 pass, same morning) vs the 0.1.55 base's 947.9–963.2 — main's own gains ≈ +2–3% at 16k–32k; the stack captures those, F-2, and the MMQ patches together. Daily config exe → wt-stack (backup .bak-pre-v4); wt-mmq kept as fallback. 0.2.10 accepted the shipped rocBLAS table and every v2 config arg unchanged; engine PID 1748479 at promotion. PP record per tier now: 966.2/986.0/982.3/965.8.
Note: the SIMT=1 and wave-reduce rows are medians of 2 sequential warm passes per arm (A1/B1/B2/A2, 2026-10-14), not interleaved A/B pairs — their 16K/32K cells carry ±2% sequential-pass variance; the 64K/128K gains of the SIMT row (+2.7/+8.6%) are well outside it. #1167-only rows are single warm passes.
Settled-state row (thermal, see rocBLAS A/B section): same build+config as the wave-reduce row above it, measured after 1.5 h of continuous load — the −2 to −3% delta vs the cool-start rows is the thermal epoch, NOT a regression. All future comparisons must be settled-state vs settled-state (or soak-equalized).
128K compaction cycle wall: 16.4 -> 15.3 -> 8.1 -> 4.5 -> 3.1 -> 3.0 min (5.5x). Every Strata row beats Ollama's 16K row at 128K.

**16K lane snapshot on 0.1.43 (all lanes measured with the same fixtures):**

| Lane | PP (tok/s) | TG (tok/s) | Wall |
|---|---|---|---|
| **Coder IQ1_M** | **863.8** | **80.6** | **24s** |
| IQ3_S | 811.8 | 71.7 | 28s |
| Q4_K_M | 531.8 | 49.4 | 40s |
| Ollama 0.40.3 (Q4_K_M) | 188.4 | 16.2 | ~90s |

**`STRATA_HIP_PROMPT_F16` A/B on 0.1.43 (IQ3_S @ 16K):**

| | PP (tok/s) | TG (tok/s) | Wall |
|---|---|---|---|
| F16 **on** | **811.8** | 71.7 | 28s |
| F16 **off** | 433.6 | 72.4 | 44s |

Was neutral on the fork engine; 0.1.43's prompt path requires it — +87% PP. TG unaffected (decode path unchanged).

**PR #1167 rocBLAS tuning-table A/B (2026-10-14, IQ3_S lane, same fixtures):** upstream PR <https://github.com/Niko1221/Strata/pull/1167> — ships `tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt` (112 rows, 76 with a solution; opt-in via `STRATA_ROCBLAS_TUNING` env), serves ONLY the `STRATA_HIP_PROMPT_F16=1` prompt GEMMs; decode untouched.

| Tier | PR build, no table (A) | + table (B) | Δ PP | wall A→B |
|---|---|---|---|---|
| 16K | 801.9/64.5 | **823.7/65.3** | +2.7% | 27→26 s |
| 32K | 814.2/69.8 | **827.0/70.9** | +1.6% | 47→46 s |
| 64K | 794.8/69.5 | **804.6/68.5** | +1.2% | 90→90 s |
| 128K | 741.9/68.9 | **753.6/68.0** | +1.6% | 187→184 s |

Result JSONs: `llm-benchmark/results/strata_tier_pass_20261014_{044402,045055}.json` (A, B). One warm pass per arm. **Arm A reproduced the recorded 0.1.43 IQ3_S row within noise** (baseline sanity ✓). All four PP gains above the ±0.3% warm-PP noise floor; TG within ±3%. Gains smaller than the isolated-A/B ones because tier prefill includes GDN/attention/PLE/expert-transfer (~75% of GPU timeline).

**Isolated prompt A/B (`tools/ab_engine.py`, 3 rounds, medians):** 1,121-tok prompt 277.1 → 316.0 tok/s (**+14.0%**); 6,345-tok prompt 642.6 → 665.2 (**+3.6%**); short chats & decode identical. Mechanism @6K in `STRATA_PREFILL_TIMING`: `hc read` (hyper-connection down N=320/K=10240) 1330 → 770 ms; total GPU timeline −3.4%.

**Version gate (verified):** the table is pinned to one rocBLAS build; the engine refuses any mismatch with a one-line log (defaults run silently — no perf cliff). Our server loads rocBLAS from `lib_dirs: ["/opt/rocm/lib"]` = **5.6.0.8d1ae90e** = shipped table's build → accepted out of the box. The system-wide `/usr/lib/x86_64-linux-gnu/librocblas.so.5` is **5.1.1** and is rejected — binaries/tests that don't get `lib_dirs` run 5.1.1 (re-tune in ~5 min: `cmake --build build-hip --target tune_rocblas && LD_LIBRARY_PATH=/opt/rocm/lib ./build-hip/tune_rocblas --tuning-out <arch>-rocblas-<ver>.txt`). Re-check after any ROCm update. NOTE 2026-10-14: a self-tuned V620 table measured NEUTRAL vs the shipped one (see A/B section) — after a ROCm update the re-tune is a correctness requirement (table/build match), not an expected speedup.
**Verdict: ADOPT.** PP +1.2–2.7% at every tier, zero TG cost, inert without the env var. Not yet merged upstream: PR is dirty vs v0.1.44.1 (expected conflicts: `setup.py`, `docs/AMD_HIP.md`; `gemm.cu` core applies cleanly). Logs: `/tmp/ab-pr1167{,-1k}.log`; self-tuned 5.1.1 table (unused by server): `/tmp/gfx1030-rocblas-5.1.1-v620.txt`.

**PR #1151 FP32 tiled QSA scorer A/B (2026-10-14, IQ3_S lane, same fixtures):** upstream PR <https://github.com/Niko1221/Strata/pull/1151> (Vop3p, single commit `3138986e`, rebased on 0.1.43.1) — enables `STRATA_SELECT_SIMT=1` on HIP: the QSA block scorer moves from the warp kernel to the FP32 tiled scorer CUDA uses below sm_80; 0.587 vs 2.69 ms per 256 queries at 128K (6900 XT, 4.6x), byte-identical scores for every tiling, selections match warp 256/256. Opt-in (different summation order, as on CUDA). Merged into `f-gfx103x-sh-stream` as `3394ea27` (only conflict: `docs/AMD_HIP.md` bullet order — kept both).
- A/B medians of 2 warm passes per arm (A = off, B = on): **16K −1.8%** (838.2→823.0; n=2, scorer segment itself faster — 257→134 ms — so the dip is variance/secondary, watch it), 32K −0.4% (noise), **64K +2.7%** (812.5→834.7), **128K +8.6%** (757.6→822.6; wall 183→169 s, −13 s).
- Mechanism, from `STRATA_PREFILL_TIMING`: `qsa select` 19,348→5,109 ms at 128K (3.8x), 4,472→1,505 ms at 64K (3.0x), 1,144→567 ms at 32K, 257→134 ms at 16K; signature `qsa select: FP32 tiled scorer (STRATA_SELECT_SIMT=1), tiles 32/128/16/8 (gfx103x)`.
- vs the author's +3.2% at 128K (2x 6900 XT, 10/10 pairs): we get +8.6% — bigger because we run one V620 (no cross-GPU select split) and QSA select is 11.0% of our 128K GPU timeline.
- TG neutral (413–582 tok range across arms; scorer is prompt-only).

**Verdict #1151: ADOPT for long-context work; kept `STRATA_SELECT_SIMT=1` in the daily-driver config.** Cost: ~−1.8% PP at 16K (within today's 815.7–841.7 spread; re-check if 16K becomes the hot tier). Logs: tier JSONs `results/strata_tier_pass_20261014_{052023,052658,053239,054406}.json` (A1/B1/B2/A2).
**PR #2226 persistent HC read A/B (2026-10-14, IQ3_S agent sessions, same fixtures):** upstream PR <https://github.com/Niko1221/Strata/pull/2226> (single commit `6ce12b4a`, base 0.1.43.1) — `STRATA_HC_PERSIST=1`: the verifier's hyper-connection read (1–8 tokens) as one 80-block kernel with two grid barriers instead of three launches. Merged into `f-gfx103x-sh-stream` as `b5962db4` (docs-only conflict). Startup check passed on the V620: `80 blocks on 36 multiprocessors, 0 token counts timed out, max relative difference 1.8e-07`.
A/B (`tools/agent_turns_bench.py`, 2 arms × 3 rounds × 9 fixtures × 5 turns = 69 paired turns, interleaved): **mean on/off decode ratio 1.006** — medians 60.5 (off) vs 60.9 (on) tok/s, 22/69 turns faster, no context bucket above 1.00 (short 0.997, ~4–12K 0.996, ~12–40K 0.999, >40K 0.998). NEUTRAL on this box.
Why no gain: the author's −4.2%/−2.8% ms-per-verify-window was measured on a layer-split config (`--pipeline-windows 2`, spec 3/0.7); our single-card window composition differs. Kernel microbenchmark (read 1.35–2.10×, 38.9 vs 55.1 µs at 2 tokens) is real but decodes twice: kernel → window share → TG %. PP unchanged by design (persistent read not on the prefill path — their own numbers: 848/850, 1,636/1,632).
**Verdict #2226: KEEP MERGED, FLAG OFF on single card** — no measured benefit, so the daily driver runs the byte-identical default read (`STRATA_HC_PERSIST` removed from `strata-iq3_s-v043.json`). **Re-enable if we ever do a 2-card layer split** (author's config matches; their split also showed +42% PP from the 2nd card: 1,155→1,636 tok/s at 32K, though that includes their faster uncapped 6900 XTs — ours single-card is 834 at 32K, so ~1.96× vs our row is card-count ×1.42 and card/config ×1.39). Log: `/tmp/ab2226/log.txt` (JSONL, 138 rows).

**gfx103x native DPP wave-reduce A/B (2026-10-14, IQ3_S lane, same fixtures):** ported from edwinbrowwn's llama.cpp-rdna2 gfx1030 native paths — `src/kernels/cuda/wave_reduce.cuh` (new) routes the 32-lane `warp_sum`/`warp_max` in `native_mmvq.cu`, `router_top10.cu` (both kernels' per-wave maxes) and `native_flash_attn.cu` through `__builtin_amdgcn_wave_reduce_fadd_f32`/`_fmax_f32` (DPP chain: `v_add_f32_dpp`/`v_max_f32_dpp` + `ds_swizzle_b32`) on the gfx103x code objects; every other arch (CUDA, gfx906 wave64, gfx1100/1200) compiles the same shuffle loops as before, so its code objects are byte-identical (fallback keeps the explicit `width=32` shuffle — matters for gfx906). Bit-identical output: same xor butterfly (0/32,768 mismatches over random warps, sum and max, standalone probes); the swapped down-loops are lane-0-consumer and `fmaxf` is order-independent; parity tests for every touched kernel pass. Left untouched: pair reductions, 8/16-lane and 64-lane loops, the double-precision router bracket sum, the CUDA-only warp-a-block quantize kernel, fused_gr/gr/gdn/sampler (follow-up candidate, needs per-site lane-0 audit).
Committed as `48d6aa11` on `f-gfx103x-sh-stream`.
**Measurements (V620, ROCm 10.0.0, daily-driver config):** micro: ~2× cheaper per reduction (0.09→0.05 ns per warp sum, 0.11→0.05 per max; 65,536 warps × 4,096 chained reductions). `q8_gemv_bench` (quantize_q8_1 + native_mmvq — the only bench that links the touched TUs), med of 2 runs: 2–6 token windows −1% to −10% per shape (QSA k/v 2560→512 6-col 53.3→49.5 µs; shared down 640→2560 6-col 9.7→9.0 µs; GDN gate 6-col 63.7→61.1 µs); 1-token neutral; head 2560→248320 neutral (one config, 4-col, +2.4%). Full tiers, 2 sequential warm passes per arm (A1/B1/B2/A2 — baseline first stashed, rebuilt, engine proc restarted from the reverted binary): **16K 820.3→828.8 PP (+1.0%), 32K 834.2→841.6 (+0.9%), 64K 836.7→842.3 (+0.7%), 128K 824.9→827.5 (+0.3%); TG neutral** (all tiers within the ±3% floor). PP gains are above the ±0.3% warm floor at 16/32/64K; 128K sits at it. `router_chain_bench` does not link the touched TUs (byte-identical binary before/after — an earlier mmvf 'regression' from it was the same binary measured twice).
**Verdict wave-reduce: ADOPT, now in the daily driver.** Result JSONs: `results/strata_tier_pass_20261014_{080125,080653,081343,081909}.json` (B1/B2/A1/A2).

**Extension attempt (fused_gr/gr/gdn/sampler) — TESTED AND ABANDONED (2026-10-14).** Full site audit first: gdn.cu and sampler.cu dropped immediately (gdn's reductions are double accumulators — the DPP builtin is f32-only; sampler's are pair float+int and lane-0 broadcasts). Remaining: `gr.cu` `warp_sumf` + all inline sum/max sites of `fused_gr.cu`, all per-warp-guarded and warp-uniform. After the swap, `fused_gr_bench` (which self-checks against the old kernel bitwise, T 1–6 × apply/inject) reported "bitwise equal" everywhere BUT its output hash differed from baseline on all 24 configs — a contradiction worth chasing.
**Bisect (hash identity against baseline, ~6 builds):** `gr.cu` bit-identical (24/24); `warp_max` bit-identical (24/24); all ~17 `warp_sum` sites bit-identical EXCEPT two ternary sites in fused_gr.cu (`s[k] = k < T ? warp_sum(acc[k]) : 0.0f`, lines 316/878) — with only those two on the old loop and everything else native, 24/24 hashes match. Root cause was never pinned down: the loop is warp-uniform (site 316 executes unconditionally; site 878 after a `!active` early return that is warp-uniform) and the two sites differ only in the ternary guard, so the working theory is that the predicated-DPP form of the intrinsic (or its 0xff-fill-uninitialized-acc interaction in the T<MAX_T tail) compiles differently than the probe's plain xor butterfly — the probes and every other site use converged whole-warp data; these two feed the reduction from a partially-written register array. NOT fixed, NOT shipped: the two sites keep the shfl loop, which is itself a correct bit-identical fix, but with only 2 of ~19 sites convertible the change is not worth carrying.
**Also ruled out on the way:** timing anomalies (−8.9%/−21% at T5/T6) in the first "new vs base" run were cold-clock artifacts — baseline ran first after a rebuild; `gr.cu`-only-stash proved gr timing/hashes identical; same-binary run-to-run hashes are deterministic (24/24), so hash diffs are real code-object differences, never timing.
**Decision:** do NOT revisit unless (a) a profile shows fused_gr/gr reductions are a hot path AND (b) someone pins the ternary-site root cause first (dump the gfx1030 ISA of the two kernels old-vs-native and diff the reduction block). The adopted `48d6aa11` scope (mmvq/router/flash-attn) is unaffected — that code is in the daily driver and stays.

**rocBLAS tuning-table portability A/B — shipped (6900 XT) vs V620 self-tuned (2026-10-14, IQ3_S daily-driver build):** the shipped #1167 table's solution ranking was authored on a 6900 XT, so we self-tuned on the V620 (~5 min, `tune_rocblas --tuning-out /tmp/gfx1030-rocblas-5.6.0-v620.txt`; 112 rows, 74 with a solution) and found **56/112 shapes pick a different solution** on our card. Given thermal sensitivity (below), the protocol was: per arm 6 sequential passes after an engine restart, first 3 = thermal soak (discarded), median of last 3; same day, both arms at the same settled ~180 W / ~85 °C equilibrium (per-pass tj/pwr/sclk sampling confirms identical epochs).

| Tier | A: shipped table (med-of-3) | B: V620-tuned (med-of-3) | Δ PP |
|---|---|---|---|
| 16K | 803.4/77.3 | 804.5/77.2 | +0.1% |
| 32K | 819.2/73.2 | 821.0/76.8 | +0.2% |
| 64K | 827.5/73.2 | 829.0/76.0 | +0.2% |
| 128K | 820.3/72.1 | 822.2/72.7 | +0.2% |

**Verdict: KEEP the shipped #1167 table** — every delta below the ±0.3% warm-PP floor. Isolated per-shape solution choice does not survive integration into the overlapped prefill path (GDN/attention/PLE/expert-transfer are ~75% of the timeline). Useful portability data point for the #1167 thread if we ever comment. Counted-pass JSONs: `results/strata_tier_pass_20261014_{170646,171213,171741,174005,174538,175109}.json` (B4–B6, A4–A6); A arm = the daily-driver config untouched (strata-iq3_s-v043.json), so the server landed back on it. Self-tuned table left at /tmp (volatile; regenerate in 5 min if ever needed).

**Thermal methodology findings (2026-10-14):** the ~180 W power cap is the de-facto clock governor — sclk wanders 1520–2470 MHz under load (user observed 1600–2400+), junction settles ~85 °C. Cold→settled PP drop is real and large: ~5% at 16K, ~2% at 64K/128K (B soak ran 849.6→829.0 @64K over three passes). The record's "first run after restart = throwaway" is this clock/power ramp, not a distinct phenomenon. `rocm-smi --setperfdeterminism` is **NOT supported on the V620** (firmware locks the overdrive/perf-management family: invalid-arguments at any value; manual sclk levels are only 500/2570 MHz, nothing in between). **Protocol (revised 2026-10-15, user decision): standard run = 1 soak pass (discarded) + 3 counted passes, median of the 3** — applies to all runs from now on; the older ≥3-soak rule (below) was used for the rocBLAS A/B, where sub-noise deltas demanded deeper soaking. Compare arms only within one thermal epoch or with equal soak; never compare across sessions without soak-equalization. Session-level note: cool-start rows read ~2–3% above settled-state medians — that is the thermal epoch, not a regression.
**RDNA2 native data-type survey + INT8 question (2026-10-14, ISA-level, no benchmark):** verified against the RDNA2 ISA guide AND empirically on ROCm 7.15's `llvm-mc -mcpu=gfx1030`. Native: FP32 FMA, FP16 (packed `v_pk_fma_f16` = the only 2× path), INT32/16 packed, INT8 **only via dot** (`v_dot4_i32_i8`, ¼-rate → FP32-parity throughput), FP64 1/16. NOT native: BF16 (software only), WMMA/MFMA, FP8/FP4 (those start at RDNA3/CDNA). Consequently: (a) "convert key portions to INT8" is a measured dead end — INT8 compute = FP32 parity on gfx1030, and the daily driver already runs native `v_dot4c_i32_i8` in its MMQ/iq kernels (14,312 sites disassembled from the shipped `native_mmvq.cu.o`); PREFILL_MMQ (the INT8-activation path) was already measured no-gain; (b) the current `hip_compat/intrinsics.hpp` dp4a shim already routes gfx1030 to `__builtin_amdgcn_sdot4` correctly (the gfx906-era `strata_hip.h` shim is NOT on the gfx1030 include path); (c) the only unexploited native-type candidate left is packed FP16 (`v_pk_fma_f16`) in scalar FP16 sites — needs a disassembly audit + full parity protocol before touching (wave-reduce method applies; FP summation order risk class).


**UD-IQ4_XS lane first run (2026-10-15, unsloth 4.1 bpw, engine 0.1.44.1 daily-driver build):** downloaded 87.2 GiB (811 s, ~110 MB/s), pack `packs/iq4xs` via `iq_pack.py --compat-bf16` (unsloth quantizes the hc/attn/ffn projections to Q8_0; 195 tensors rounded, 1.20 GiB dense.bin, expert + PLE bytes unchanged in the GGUF; shard 2 carries the 26.8 GiB quantized PLE). Config `strata-iq4xs-v043.json` = daily-driver knobs + `--vram-reserve-mib 600` (200 was not enough for the full-vocab draft head: 213 MiB needed, first load refused it; 600 MiB reserved costs ~4% of expert-cache slots). Loader: **9,911 resident experts** (vs 12,393 on IQ3_S, −20%), 462 MiB VRAM free, session up, no guard complaints. **Bench, protocol = 1 soak (discarded) + 3 counted, med-of-3** (user decision: 1+3 is the standard for all runs from now on):

| Tier | IQ4_XS PP/TG | IQ3_S settled | Δ PP | Δ TG |
|---|---|---|---|---|
| 16K | 829.2/67.6 | 803.4/77.3 | +3.2% | −12.5% |
| 32K | 851.2/67.4 | 819.2/73.2 | +3.9% | −7.9% |
| 64K | 863.9/66.9 | 827.5/73.2 | +4.4% | −8.6% |
| 128K | 856.2/63.6 | 820.3/72.1 | +4.4% | −11.8% |

Counted spreads: PP ≤±0.6% per tier, TG ±3% (documented floor). TG −8–12% is the predicted bandwidth mechanism (1.19× bigger expert reads, −20% cache slots). **PP +3–4% is the surprise** (IQ3_XXS was only +0.4–0.8% PP on the same comparison); candidate mechanisms: (a) real — MMQ prompt-path dequant cheaper per MAC for IQ4_XS's block layout, or (b) thermal residue — IQ4_XS got 1 soak pass vs IQ3_S's 3-pass full soak, and its counted passes still trend upward pass-over-pass (64K: 862.8→863.9→868.2). User's call: **numbers accepted as-is, lanes stay comparable for now**; if a future decision hinges on the PP delta, re-run IQ3_S same-epoch with the 1+3 recipe. Quality side-by-side not yet run. Result JSONs: `results/strata_tier_pass_20261015_{060051(soak),060610,061128,061646}.json`; server log `strata-iq4xs-043.log`.

**Spec-5 A/B on the UD-IQ4_XS lane (2026-10-15, replication of the Strix Halo report's spec arm):** community report (`bench/results/2026-10-09-community-strix-halo-laptop-70w/`, Radeon 8060S laptop @70 W) found `--spec 5 --spec-min-p 0.5 --mtp-window 8192` beats `--spec 4` on decode (+8% @32K) and that spec 6 hurts. We replicated the arm on our lane: same-day, soak-equalized (each arm 1 soak + 3 counted, med-of-3), same fixtures. Config `strata-iq4xs-spec5.json` = the v043 IQ4_XS config with `--spec 5` + `--mtp-window 8192` (spec-min-p 0.5 already set). Load check: 9,926 expert slots (vs 9,911), 454 MiB VRAM free, draft head 802 MiB — no reserve bump needed.

| Tier | spec 4 (PP/TG) | spec 5 + mtp-window (PP/TG) | Δ TG |
|---|---|---|---|
| 16K | 833.6/68.9 | 835.5/73.2 | +6.2% |
| 32K | 855.4/67.9 | 857.8/73.7 | +8.5% |
| 64K | 868.2/68.1 | 869.4/73.3 | +7.6% |
| 128K | 860.7/64.5 | 861.0/70.0 | +8.5% |

PP flat (±0.1–0.3%, at/below the noise floor — prompt path untouched). Acceptance from the serve logs: spec 4 ~307/335 (0.916) vs spec 5 ~317/363 (0.873) — deeper drafts accept less per offer but net tokens/round rises. **No long-context acceptance decay on our fixtures**: acceptance at 128K (0.87–0.88) ≈ 16K; the Halo's C++-source collapse (0.761→0.635, decode 58→42 tok/s) has not appeared on llm-context-bench prose — recheck with their exact prompts (see replication plan). One 16K pass generated 557 tokens (late EOS) — median protects. **Verdict: ADOPT spec 5 + mtp-window 8192 for the IQ4_XS lane; server left running it as the lane default.** Spec 6 not re-tested (both record rows say it loses; the Halo confirms 43.2 tok/s @32K). Baseline result JSONs: `results/strata_tier_pass_20261015_{135831(soak+cool-start),140346,140901,141418}.json`; spec-5 JSONs: `results/strata_tier_pass_20261015_{142137(soak),142652,143207,143722}.json`; server log `strata-iq4xs-spec5.log`. Note: draft acceptance is NOT exposed via `/v1/status last_timings` (harness prints drafts None/None) — read the serve-log `drafts accepted X of Y` lines instead.

### Experiment #1 — Halo prompt ladder on V620 (2026-10-16) — DONE
Their exact method: first 8,192/32,768/131,072 tokens of llama.cpp `src/**/*.cpp` (sorted, 183 files, 3,459,818 chars → 1,146,863 tokens, round-trip exact), engine-level fresh process per run, `--prefill auto:16384`, spec 5 + window, greedy, 256 tokens (cap reached in every run), 4 runs/tier, medians. Runner: `~/git/llm-benchmark/halo_ladder.py`; data: `~/git/llm-benchmark/results/halo_ladder_runs.json`.

| Tier | PP med | TG med | Acceptance | tok/round |
|---|---|---|---|---|
| 8K | 906.0 | 45.5 | 0.713–0.720 | 3.56 |
| 32K | 926.1 | 46.8 | 0.757–0.829 | 4.26–4.41 |
| 128K | 893.9 | 44.0 | 0.789–0.851 | 2.91–3.02 |

Findings: (1) **The Halo's 128K acceptance collapse did NOT replicate** — our acceptance at 128K (0.789–0.851) is the highest of the ladder, decode falls only −3% from 32K→128K (44.0 vs 46.8); theirs fell −27% (57.8→42.0) with acceptance 0.761→0.635. Spec 5 survives long C++ context on our lane; the collapse is fixture/machine-specific to their record, not a law. (2) Our cold-cache TG (~45–47) sits well under our warm-server lane TG (70–75): with `--expert-cache auto` (9,926 slots), decode-miss experts stream over PCIe in a fresh process — the Halo paid none of this (all 24,576 experts resident in unified memory). The apples-to-apples row vs their 70 W box IS this cold number: their 59.2/57.8/42.0 vs our 45.5/46.8/44.0 — we lose ~23% at 8K/32K but are AHEAD at 128K despite streaming, because their acceptance collapsed and ours did not. (3) Engine-level PP (894–926) is a different measurement layer than server-side PP (835–870) — includes `auto:16384` chunking and no server overhead; not directly comparable to our table rows.

## Keep/drop matrix — all lanes, spec-5 recipe (2026-10-16)

Full 6-lane matrix, one evening, sequential, same build (0.1.44.1 + #1167 + #1151 + wave-reduce `48d6aa11`). Every lane ran the **adopted MTP recipe** (`--spec 5 --spec-min-p 0.5 --mtp-window 8192`; spec-5 configs cloned from each lane's v043 config). Protocol: fresh server per lane, 1 soak + 3 counted, med-of-3, standard fixture ladder (llm-context-bench regular-{tier}, temp 0, 1024 max tokens). Driver: `~/git/llm-benchmark/run_matrix.py`; per-lane records `results/matrix_<lane>_*.json`; medians `results/matrix_final_20261016.json`.

| lane (spec-5) | 16K PP/TG | 32K PP/TG | 64K PP/TG | 128K PP/TG |
|---|---|---|---|---|
| IQ3_S | 802.6/82.0 | 819.5/78.6 | 826.6/79.3 | 818.7/75.5 |
| IQ3_XXS | 810.0/84.2 | 825.7/83.5 | 831.6/87.1 | 823.5/79.8 |
| IQ3_S-abl | 804.5/84.9 | 822.0/82.6 | 828.5/81.2 | 821.5/75.3 |
| UD-IQ4_XS | 829.8/69.0 | 851.0/70.7 | 862.5/71.7 | 854.3/66.7 |
| UD-Q4_K_XL | 504.5/52.7 | 525.8/53.0 | 529.4/54.2 | 513.8/52.5 |
| IQ1_M coder | 789.5/87.2 | 827.9/85.7 | 813.9/82.3 | 764.7/78.6 |

Readings:
- **Spec-5 replicates across lanes.** IQ3_S spec-4 (2026-10-15) vs spec-5 (this matrix): TG 77.3→82.0, 73.2→78.6, 73.2→79.3, 72.1→75.5 (+5–8%), PP unchanged. No lane regressed from the recipe.
- **IQ3_XXS is the throughput winner** among Flash-Next base lanes: best TG at every tier (79.8–87.1) with PP equal-or-better than IQ3_S. Smallest expert reads of the GSQ set (42.9 GB vs 50.3).
- **IQ3_S is dominated by IQ3_XXS on both axes** — candidate for retirement unless quality testing says otherwise.
- **Abliteration is throughput-free:** abl ≈ base IQ3_S at every tier (slightly ahead at 16–64K).
- **IQ4_XS keeps the PP crown** (+3–4% over the GSQ lanes) and pays the known TG bandwidth tax; recall parity with IQ3_S already on record (needle 15/15 both).
- **Q4_K_XL is decisively out:** 504–529 PP / 52–54 TG — ~60% slower decode, ~40% slower prefill than every i-quant lane, spec 5 notwithstanding. Confirms the k-quant-expert dead end at scale. Note: this lane's config model_name is `qwen3.8-flash-125b-tuned-q4km` — the blob may be the 125B-tuned checkpoint, not plain Flash-Next; either way it is not competitive.
- **IQ1_M coder has the best short-context TG** (87.2 @16K) but PP sags at 128K (764.7) and it is a different checkpoint (coder tune, IQ1_M extreme quant) — quality verdict pending; do not keep/drop on speed alone.

Keep/drop recommendation pending per-lane quality checks: **keep IQ3_XXS, IQ4_XS, abl (if uncensored wanted), coder (pending quality); drop Q4_K_XL** (frees ~238 GiB on sdb: 110.9 GiB blob + 127 GiB unsloth-q4 shards) **and IQ3_S** (dominated).

Matrix disposition (executed 2026-10-16): **kept** IQ3_XXS, UD-IQ4_XS, IQ3_S-abl; **dropped** IQ1_M coder, UD-Q4_K_XL, IQ3_S — server switched to `strata-iq4xs-spec5.json` on :8080, `~/.pi/agent/models.json` provider renamed `flayer-llama` → **`flayer-strata`** with exactly the three kept models. Deleted: `gguf/coder` (66 GiB incl. its pack), `gguf/IQ3_S` (78 GiB), `gguf/unsloth-q4` (127 GiB), packs `iq3_s`/`q4km`/`q6xl`. **Retained pending a decision:** the 110.9 GiB ollama blob `sha256-89059c88…` — it is shared by two ollama tags (`qwen3.8-flash-125b-tuned:latest`, `qwen3.8-flash-next:125b-a6b-q4_K_M`), the latter still served by the ollama stack and listed in pi's ollama provider; deleting it needs sudo or `ollama rm` (frees a further ~111 GiB). Disk after cleanup: 400 GiB free on sdb (was 130).

Ops notes: coder server start failed once (port 8080 still held by the dying previous server → `server.py` exits on port-in-use); `run_matrix.py` now waits for the port to actually free before starting the next server. 

## Strix Halo replication plan (opened 2026-10-15)
Source: `Strata-main/bench/results/2026-10-09-community-strix-halo-laptop-70w/` (Radeon 8060S laptop, 70 W, gfx1151, ROCm 7.14.1, v0.1.41). Their method: engine-level runs, fresh process per run (cold expert cache), prompts = first 8,192/32,768/131,072 tokens of llama.cpp `src/**/*.cpp` tokenized with `tools/strata_tokenizer.py`, 256 gen tokens, greedy, `--stats`, medians of 4–6 runs.

Mapped to our V620 (IQ4_XS lane, spec 5 default):

| # | Experiment | Status | Notes |
|---|---|---|---|
| 1 | Their exact C++-source prompt ladder 8K/32K/128K, fresh process + `--tokens-file` + `--stats`, 256 gen tokens | **DONE 2026-10-16 — see "Experiment #1" above.** Cold fresh-process medians 906/45.5, 926/46.8, 894/44.0; acceptance collapse did NOT replicate (0.79–0.85 at 128K). The owed warm-server C++ run is **optional** — superseded in practice by the keep/drop matrix's warm server-tier numbers (prose fixtures); run it only if the cross-machine C++ decode row ever matters again. | Their collapse (0.761→0.635, 58→42 tok/s) was the open question — settled: machine/fixture-specific, not a law. Cross-machine row: their 70 W box wins 8K/32K decode (59.2/57.8 vs our 45.5/46.8 cold), we win 128K (44.0 vs 42.0). |
| 2 | `--spec 5 --mtp-window 8192` vs spec 4 | **DONE — ADOPTED (see Spec-5 A/B above)** | +6.2–8.5% TG, PP flat. Spec 6 not re-tested (both records say it loses). Depth-vs-window decomposition declined by user (suffix-window counts similar across arms — depth likely carried the gain). |
| 3 | §6 arm: `--prefill 16384 --lookup-chain 3 --mtp-q4 all` | **DONE 2026-10-16 — NEUTRAL, not adopted.** 32K engine-level, 1 soak + 3 counted vs same-day base ladder: PP 927.3 vs 926.1 (+0.1%), TG 46.6 vs 46.8 (−0.4%), tok/round 4.19–4.45 vs 4.26–4.41 — all inside the ±3% noise floor. Lookup-chain widens the verify window 7→8 and adds no measurable gain or loss. Their §6 prefill rec (16384) is already in our lane as `auto:16384` via the base arm — effectively pre-adopted. Data: `halo_ladder_s6_runs.json`. |
| 4 | Server concurrency: 4 clients × ~6K prompts, parallel 2/4, ± `--batch-mtp` | **PARKED (user decision 2026-10-16)** — no concurrent-client workload today (no sub-agents). If it ever runs, run it on the spec-5 config (batching interacts with the verify path we changed). Watch their 41 s first-answer cliff. |
| 5 | Their needle method (5 codenames, depth 10–90%, ~83K-token code prompt, 3 seeds) on both lanes | **DONE 2026-10-16 — 15/15 BOTH LANES.** Runner `~/git/llm-benchmark/halo_needle.py` (their protocol rebuilt: codenames as C++ comments at exact token depths of the ladder corpus's first 83,000 tokens; FALCON/GRANITE swapped for MAGPIE/DOLOMITE — the originals collide with llama.cpp arch enums; `enable_thinking: false` via `chat_template_kwargs`). IQ4_XS spec 5: 15/15, walls 94/95/96 s (median 95). IQ3_S spec 4: 15/15, walls 99/100/101 s (median 100). Halo reference 15/15 at 93–105 s. **Closes the IQ4_XS-vs-IQ3_S quality question at recall level: no measurable difference** — the PP/TG throughput trade is the only thing separating the lanes. Server-based, as served (lane defaults). Data: `halo_needle_iq4xs.json`, `halo_needle_iq3s.json`. |

Not applicable: GTT 104→112 GiB + `amd_iommu=off` (UMA-only; we are discrete HBM), ROCm 7.14.1 lever (version gate — we are pinned to ROCm 10.0 / rocBLAS 5.6.0.8d1ae90e per the #1167 table). Their 70 W power-limit story maps to our deliberate 180 W cap — comparison is power-normalized, not a lever.

**Reference boxes:** community V620 report (engine 0.1.38, IQ3_S): PP 424.9 (4–9K), decode 59.5 — our fork was within ±4% (sanity ✓). User's other box (`/work`, config with `--vision`, `--vram-reserve-mib 200`, `STRATA_HIP_PROMPT_F16=1`, spec 4, max-ctx 131072): **16K PP 981.5, chat decode 52.1** — ahead on PP vs our 0.1.43 (861), tie on decode. **EXPLAINED (user-confirmed):** our V620 is deliberately down-powered (user set power cap on purpose; card runs pinned at ~180 W of 265 W spec, junction 60-63 °C, unthrottled). Gap is the power budget, not software — do NOT chase further.
## Validated findings (do not re-litigate)
- **Rejected**: `OLLAMA_KV_CACHE_TYPE=q8_0` (−27% TG @128K on CPU-heavy split — dequant-bound, not bandwidth-bound); `use_mmap=false` (+53 GB RAM, −1 GPU layer, no gain); STRATA_PREFILL_MMQ=1 (no gain); `--spec 6` (warm-equal, worse cold); MMQ/ring/io-threads (community: +0.1%).
- **Kept**: int8 KV + kv-resident 32768 (frees VRAM→expert cache); `ple-io ram` (cold-start immunity, no SSD reads); **STRATA_HIP_PROMPT_F16=1 on 0.1.43 (required)**; spec 4 in v043 configs (fork used spec 3, warm-equal); expert-cache auto; **STRATA_ROCBLAS_TUNING on PR #1167 build (validated 2026-10-14, see results section)**.
- **Noise floors**: warm PP ±0.3%, TG ±3% (16K). **Soak after any engine restart** per the current protocol (revised 2026-10-15: 1 soak pass discarded + 3 counted, median; the rocBLAS A/B used the older ≥3-soak rule — see thermal findings 2026-10-14; cold→settled PP −5% @16K, −2% @64/128K). Never compare across sessions/thermal epochs without soak-equalization — cool-start medians read ~2–3% above settled-state.
- llama.cpp `-ot experts=CPU` split: 96 PCIe round-trips/token at decode → TG 8.8 (vs 16.2 Ollama). Whole-layer split (-ngl 14) never completed.
- Speed ladder: quant/engine levers exhausted → next step-change is hardware (2nd V620 **on order** — expert cache per card 62% of Q4_K experts; Coder-class models get ~full cache on ONE card).

## Open items (decided, not done)
1. ~~`STRATA_HIP_PROMPT_F16` A/B on 0.1.43~~ **RESOLVED — knob is REQUIRED**: F16 off → PP 433.6 (−47%!), TG unchanged; on → 811.8/71.7. Was neutral on fork engine; 0.1.43's prompt path needs it. Explains most of the /work box's PP edge.
2. ~~`--spec 4` vs 3~~ **MOOT** — v043 config already runs spec 4 (matches /work box).
3. ~~**Q4_K_M + Coder lanes on 0.1.43**~~ **DONE** — Q4_K_M 531.8/49.4 warm @16K (+83% PP vs fork); Coder 863.8/80.6 warm @16K.
4. ~~Full 4-tier decay curves on 0.1.43~~ **DONE** — all three lanes complete, zero-to-mild decay (see master table).
5. ~~IQ3_S-abliterated pack not built~~ **DONE 2026-10-16.** Pack `packs/iq3_s-abl` via `iq_pack.py --gguf <abliterated shard 1> --out …` (no `--compat-bf16` — GSQ-RCO quants; index: 1079 tensors, 302 native, 1.54 GiB dense.bin; native_experts v3 with the abliterated shard's own offsets, same 50.3 GB expert total as stock IQ3_S). Config `strata-iq3_s-abl-v043.json` = IQ3_S v043 clone with abliterated pack/GGUF paths + `--mtp` pointed at the repo's own shipped draft `/var/lib/ollama/gguf/IQ3_S-abliterated/strata/rt` (786 MB, stock upstream MTP head — the README states the abliteration does NOT apply to the MTP block). Loads clean, 12,300 resident slots, full-vocab draft head; VRAM free 74 MiB → the same LOW-warning zone the IQ3_S base runs in (engine suggests `--vram-reserve-mib 638`, not applied — same 200 recipe as v043 so the lanes stay comparable). First sanity gen correct. pi models.json updated: `qwen3.8-flash-next-iq3_s-abl` + the previously-missing `qwen3.8-flash-next-ud-iq4_xs` added to the `flayer-llama` provider.
6. ~~llama.cpp lane: `-ngl 14` + `--spec-type ngram-cache`; UD-Q4_K_XL main + unsloth draft pair~~ **MOOT (2026-10-16)** — the Q4_K_XL lane was dropped in the keep/drop matrix and its GGUF/blob shards deleted; llama.cpp lane remains at `-ngl 14` + ngram-cache only, no draft-pair experiment. Note: llama.cpp `--spec-type ngram-cache` is unrelated to Strata MTP spec depths — never compare the two numbers directly.
7. Second V620 arrival → `--gpus 0,1` layer-split test (docs/MULTI_GPU.md; config `"gpu": [0,1]`, `layer_split auto`).
8. If user wants: delete local bench report folders in `Strata-main/bench/results/` (user declined community submission).
9. **PR #1167 adoption — DONE (2026-10-14):** merged db64983 into `f-gfx103x-sh-stream` (`8759a414`); only conflict was `setup.py` SETUP_ENV (kept 0.1.44's unsloth keys + added `STRATA_ROCBLAS_TUNING`); `docs/AMD_HIP.md` auto-merged. Rebuilt `build-hip/strata`; added `STRATA_ROCBLAS_TUNING` to `strata-iq3_s-v043.json` env. Retest pass: **839.4/68.5, 840.8/75.7, 815.7/74.8, 759.2/74.5** — PP +0.7–1.9% and TG +4.9–9.5% over the PR-worktree pass; the TG jump is from 0.1.44's own changes (gfx103x shared-expert stream default, `dcc6ed9f`), NOT the PR (decode untouched, PR-worktree A/B proved TG-neutral). Attribution caveat: no 0.1.44-no-table arm was run — if strict split needed, toggle the env var and re-run.
10. ~~Optional: second warm pass per arm; optional Coder/Q4_K_M lanes with the PR build~~ **MOOT (2026-10-16)** — both lanes dropped in the keep/drop matrix. The PRs' gains were quant-independent A/B'd on IQ3_S/IQ4_XS; no re-verification owed.
11. Optional: comment on both PRs with our V620 numbers (per docs/COMMUNITY_BENCHMARKS.md: same-day same-build comparisons — #1167 A/B was interleaved; #1151 was sequential passes). For #1167 the portability result is also worth posting: 56/112 shapes pick different solutions on a V620 vs the author's 6900 XT, yet tier medians differ by only +0.1–0.2% (sub-noise) — the table is portable.
12. ~~Optional: Coder/Q4_K_M lanes on the merged build~~ **MOOT (2026-10-16)** — see #10; both lanes were dropped after the final matrix.
13. **Wave-reduce adopted — DONE (2026-10-14):** `48d6aa11` (wave_reduce.cuh + native_mmvq/router_top10/native_flash_attn), full tier A/B +1.0/+0.9/+0.7/+0.3% PP, TG neutral; details + JSONs in results section.
14. **Wave-reduce extension (fused_gr.cu, gr.cu, gdn.cu, sampler.cu) — TESTED AND ABANDONED (2026-10-14).** gdn (double accumulators) and sampler (pair reductions) never had a convertible site; in gr + fused_gr every site except two ternary ones converts bit-identically, and the two exceptions would have to keep the shfl loop with the root cause unpinned (full bisect + theory in the results section). 17/19 sites is not worth carrying — abandoned, working tree reverted to `48d6aa11` and rebuilt. Revisit only with a hot-path profile AND an ISA diff of the ternary kernels (see Decision paragraph in the results section).
15. **rocBLAS V620 self-tune — DONE, NEUTRAL (2026-10-14):** kept the shipped table; see the A/B section. Thermal soak protocol adopted record-wide (≥3 discarded passes per restart).

## Environment/infra notes
- pi settings: `compaction.modelOverrides["openrouter/z-ai/glm-5.3-flash"] = {reserveTokens: 917503}` added (~128K compaction floor). Session model: z-ai/glm-5.3-flash.
- Azure box quirks: `pkill -f "pattern"` self-matches the calling shell — use bracketed patterns (`server[.]py`) or `pkill -x`. AND: never put the pkill and the server restart in the SAME shell command — the restart gets SIGTERM'd with the process group (hit twice 2026-10-14, even bracketed); kill first, then `setsid nohup … &` in a separate call.
- Canonical benchmark-build flag set (root cause: the 2026-10-18 missing-PP finding): `-DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_Q6K_EXPERTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1030` + Release. Gotcha: the option is `STRATA_ENABLE_HIP`, not `STRATA_HIP` — configure rejects `STRATA_PREFILL_MMQ` without it.
- Standing A/B convention (S-series, 2026-10-19 onward): each bench session ran with the server stopped and restarted after (verified healthy); sections below note only deviations from this. Engine lane untouched except explicitly named opt-in flags.
- User prefers: editing root files themselves via sudoedit; being asked before big config changes; tables with time columns (actual/elapsed/projected) and quant-labeled lanes.
- Result JSONs are stamped per-run; environment blocks pinned (ollama_version, quant, engine) — keep doing that for comparability.
- pi model routing as of 2026-10-16: provider `flayer-strata` on :8080 (UD-IQ4_XS / IQ3_XXS / IQ3_S-abl); ollama provider on :11434 unchanged. The abliterated lane's pi id is `qwen3.8-flash-next-iq3_s-abl`; its matrix row was measured with the spec-5 config (`strata-iq3_s-abl-v043-spec5.json`).

## Record close — 2026-10-16

Session goal reached: **all Flash-Next lanes on disk were inventoried, packed, spec-5-ized, and measured in one same-day matrix; lanes were kept or dropped on those numbers; the serving stack and pi routing now match the decision.**

Final state:
- **Kept (3):** UD-IQ4_XS (default, spec 5), IQ3_XXS (throughput king, spec 5), IQ3_S-abliterated (uncensored, spec 5 config ready; matrix row measured with it). Configs `*-spec5.json` + packs all present.
- **Dropped (3):** IQ1_M coder, UD-Q4_K_XL, IQ3_S (base) — data deleted (≈271 GiB freed; sdb 130→400 GiB free). Strata packs for dropped lanes removed; their spec-5/043 config JSONs left in place as historical record (tiny).
- **Undecided (one):** 110.9 GiB ollama blob `sha256-89059c88…` (shared by `qwen3.8-flash-125b-tuned:latest` and `qwen3.8-flash-next:125b-a6b-q4_K_M` ollama tags; the latter still active in pi's ollama provider). Deleting needs `ollama rm` or sudo.
- **Live now:** UD-IQ4_XS spec 5 on :8080, provider `flayer-strata`.

Quality evidence status for the kept lanes: recall-level parity was proven only for IQ4_XS vs IQ3_S (needle 15/15 both). **IQ3_XXS and abliterated were kept on throughput + availability grounds — neither has a dedicated quality test yet.** If either becomes a daily driver, run the needle harness (`halo_needle.py`, trivially parametrized per lane) plus a harder reasoning check before trusting it.

Next session's natural starting points: (a) second V620 → layer-split test (#2226 re-enable rides along); (b) quality harness for IQ3_XXS/abl; (c) the parked concurrency experiment if sub-agents arrive.

## New lane — huihui-ai abliterated-v2 UD-IQ4_XS (2026-10-18)
Source: <https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF>. Repo publishes a full quant matrix, but only the `unsloth-UD-IQ4_XS` set is servable on our GPU lane (Q4_K_XL/Q5/Q6_K_XL/Q8_0 re-quants are k-quant-expert-blocked like before; the `swift-IQ2_XS/IQ3_S/IQ3_XXS` files are ukisai Swift-1.5 DRAFT models, not main models; `mmproj-*-BF16.gguf` are vision projectors for the /work box's `--vision`, not used here). README claims v2 = "ablation emphasis adjusted" vs v1 and "none of the experts have been ablated".

**Byte-level diff vs base unsloth UD-IQ4_XS (sha256, all 3 shards + per-tensor):** shard 1 (metadata stub) byte-identical to base → same tensor layout/offsets; shards 2+3 differ. Per-tensor hash over shards 2/3: 1,085/1,224 tensors identical, 139 differ (19.7 GiB). Differing: `ssm_out` (573.8 MiB), `attn_output` (191.2), `ffn_down_shexp` (79.7) and **`ffn_down_exps` in 43 of 48 layers (19.35 GiB)**. Identical: ALL `ffn_gate_exps`/`ffn_up_exps` (33.2 GiB), the whole 27.4 GiB PLE table, `attn_qkv`, `attn_q`, `attn_gate`, `ffn_gate_inp`, `hc_*`, embeddings, output head. **The README's "no experts ablated" claim is wrong at byte level** — the expert DOWN-projections carry real weight changes in 43/48 layers (not quantizer noise: gate/up identical bytes elsewhere imply identical BF16 sources; the differing pattern is deliberate). Lane mechanics unaffected: identical sizes → identical expert reads and cache layout.

**Setup (all 6 steps done):** downloaded 3 shards 88 GiB → `/var/lib/ollama/gguf/huihui-iq4xs-ablv2/` (~14 min, ~100 MB/s), sha256 verified against HF LFS oids exactly. Pack `packs/iq4xs-abl-v2` built with `iq_pack.py --gguf <shard1> --out … --compat-bf16` (needs `STRATA_GGUF_PY=~/git/llama.cpp-rdna2/gguf-py`; same profile as base pack: 1,079 tensors, 303 native, 195 compat-bf16 = 1.20 GiB, arena 1.38 GiB, 13 s build). Config `strata-iq4xs-ablv2-spec5.json` = live spec-5 config with swapped pack/gguf paths, `model_name: qwen3.8-flash-next-ud-iq4_xs-ablv2`; MTP draft reused from base (`/home/drewster/git/Strata-data/mtp/rt` — v2 repo ships no draft; v1 precedent: ablation does not touch the MTP block). Load: **9,926 expert slots + 454 MiB VRAM free — identical to base lane** (confirms layout). Sanity gen correct (17×23=391, thinking off).

**Tier bench (2026-10-18, 1 soak + 3 counted, med-of-3, same fixtures/protocol; soak 015443, counted 015956/020514/021033):**

| Tier | abl-v2 PP/TG | base IQ4_XS spec-5 (10-15) | matrix row (10-16) |
|---|---|---|---|
| 16K | 825.4/70.7 | 835.5/73.2 | 829.8/69.0 |
| 32K | 847.1/72.6 | 857.8/73.7 | 851.0/70.7 |
| 64K | 858.6/71.3 | 869.4/73.3 | 862.5/71.7 |
| 128K | 850.3/68.5 | 861.0/70.0 | 854.3/66.7 |
PP −0.5 to −1.2% vs the base rows, TG within the ±3% floor of both → **performance parity** (no regression, no gain — consistent with the identical layout). Cross-session caveat: compared against two prior-session rows; soak protocol equal, but not same-day.

**Quality:** halo_needle 15/15 (median wall 97 s; base 95 s, abl-v1 n/a) — recall parity. Refusal probe (6 mild-to-moderate prompts, thinking off): **0/6 refusals** — fully uncensored behavior (`results/abl_v2_refusal_probe.json`). No side-by-side vs abl-v1 (that lane's server was swapped out; its GGUF is still on disk under `IQ3_S-abliterated/`) or vs base — same-day swap comparison possible if a keep/drop decision between v1/v2 ever hinges on it.

**Server state now:** `strata-iq4xs-ablv2-spec5.json` on :8080 (replaced the base UD-IQ4_XS lane; both configs coexist — flip `--config` to switch). pi `flayer-strata` provider NOT yet updated (still lists the 3 old models) — add `qwen3.8-flash-next-ud-iq4_xs-ablv2` before using it from pi.

**Harness fixes (both needed because the API key went into the configs after the matrix session — matrix-era bench runs predate key enforcement, hence the 24 401s in the old lane log):** `bench_strata_tiers.py` and `halo_needle.py` now send `Authorization: Bearer $STRATA_API_KEY` when set; `bench_strata_tiers.py` environment block now records `quant: MODEL` instead of the stale hardcoded IQ3_S string. Disk: sdb 162→75 GiB free after the download.


## Rebase verification + branch A/B matrix (2026-10-18 → 19)

**Rebase outcome.** `f-gfx103x-sh-stream` rebased onto upstream `477ac13c`: 4 local commits → 2 survived (`01373a42` gfx103x SH_STREAM default-on Linux; `0b3fc4d6` DPP wave-reduce in mmvq/router/flash-attn). `2278cce4` (rocBLAS table #1167) and `3138986e` (QSA SIMT #1151) dropped **empty — upstream merged evolved versions of both** (base ships the table + auto-enable, a newer `STRATA_SELECT_SIMT=2`, HC-persist). `docs/AMD_HIP.md` conflict resolved by keeping the patch's gfx103x text + upstream's newer `--calibrate` sentence. HIP ctest 23/23 after freeing VRAM (earlier `hip_gdn_chunked` failure = OOM against the live server, not a regression).

**The missing 800 PP — root cause found, do not re-litigate.** Hand-built cmake arms (~660 PP on the 6345-token fixture) were missing setup.py's HIP flags; `STRATA_PREFILL_MMQ=ON` (setup default, CMakeLists default OFF) is the dequant→F16-vs-MMQ prompt path and alone accounts for ~25% PP. Historical 817s and all daily-driver builds had it. Benchmark arms MUST be built with setup's flags: `-DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_Q6K_EXPERTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1030`. (Also explains the earlier abl-lane "670→867" — not a commit interaction.) Docs gap worth reporting upstream: hand-built benchmark binaries silently lose ~25% PP.

**Standard lane restored.** Plain GSQ-RCO IQ3_S re-downloaded from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (2 shards, sha256-verified vs HF manifest), `packs/iq3_s` rebuilt (1,079 tensors, 302 native). All A/B arms = `strata-iq3_s-v043-spec5.json` with only `exe` swapped (ABL/ablv2 lanes are NOT the standard; iq3_s-abl A/B earlier = smoke test only). Cost: removed both ollama flash-next 125B tags (110.9 GiB shared blob) → sdb 48→160 GiB free; re-downloadable if ever needed.

**A/B matrix (med-of-2 counted rounds after discarding round 0 as soak; `tools/ab_engine.py --rounds 3`; standard lane, temp 0):**

| Arm | long PP | long decode | vs base decode |
|---|---|---|---|
| A′ = main `477ac13c` (correct flags) | 865.1 | 45.9 | — |
| C1′ = +fork default (`01373a42`) | 866.3 | 49.6 | +8.1% |
| C2′ = +wave-reduce (`0b3fc4d6`) | 861.4 | 48.7 | +5.6% |
| B = both (= branch) | 861.1 | 52.2 | +13.0% |
| OLD′ = pre-rebase tree (correct flags) | 855.5 | 52.4 | parity with B |

- PP identical across all arms (<0.7% spread) — **neither commit changes PP; no interaction**. Fork decode +8% matches the original 10-14 measurement (+7.8–8.4%); wave-reduce +5.6% alone / composes to +13% with the fork.
- C1 (default-on) ≡ A2 (env-forced `STRATA_SH_STREAM=1`) — defaults plumbing verified.
- Rebase receipt: OLD′ ≈ B on every fixture → **nothing lost in the rebase**.
- End-to-end greedy output is nondeterministic within one binary (token counts vary across rounds) — exactness claims must come from ctest, not answer diffs.

**Tier bench, branch B (standard IQ3_S lane, 1 soak + 3 counted, med-of-3, `bench_strata_tiers.py`, results `~/git/llm-benchmark/results/strata_tier_pass_20261019_04*`):**

| Tier | PP | TG |
|---|---|---|
| 16k | 804.0 | 81.4 |
| 32k | 820.9 | 78.2 |
| 64k | 827.7 | 76.5 |
| 128k | 821.5 | 76.8 |

Per-tier spread <0.5% PP. 1.5–2.3% below the Oct-14 rows is cross-session (no same-night base tier pass); same-night A/B matrix shows PP parity, so do not quote a regression. TG consistent with fork-on.

**Builds/binaries kept:** `/tmp/strata-main-base/build/strata` (A′), `/tmp/strata-prA/build/strata` (C1′), `/tmp/strata-prB/build/strata` (C2′), `/tmp/strata-old/build/strata` (OLD′), branch `build-hip/strata` (B). Worktrees: `/tmp/strata-main-base` (477ac13c), `/tmp/strata-prA` (01373a42), `/tmp/strata-prB` (477ac13c+0b3fc4d6), `/tmp/strata-old`. Delete after PRs merge.

**Next up (not done):** PR-A + PR-B drafting (evidence = matrix above); same-night base tier pass if a controlled tier receipt is wanted; Phase-2 V620 rocBLAS re-tune (config stub `strata-iq3_s-v620tune.json` + Oct-14 `/tmp/gfx1030-rocblas-5.6.0-v620.txt` exist); `tools/calibrate.py` gfx103x-aware default-base check; `--soak` opt-in patch for ab_engine. Server at record time: **B on :8080, standard IQ3_S lane, loaded and idle.**

## STRATA_PLE_BATCH flipped 0→1 — per-token PLE was an unfixed-bug workaround (2026-10-19)

**Provenance (reconstructed from pi session logs + config forensics).** Oct 4, 06:53: a session hit a prefill stall in the fork-era log (`no progress for 60 s ... reading the prompt (batched): layer 1 of the prompt chunk from token 38532`, 5×, watchdog), matched it to upstream **issue #541** ("AMD/HIP (R9700): batched prompt read stalls at a repeatable token; `STRATA_PLE_BATCH=0` avoids it") and set `STRATA_PLE_BATCH=0` in `~/git/Strata/strata-iq3_s.json` as the workaround. Oct 11–15: the env block was cloned verbatim into strata-q4km ("inherited from the IQ3 config") and then all 15 v043/spec5 lane configs — the rationale never traveled with the flag and the record's "Kept" env list never mentioned it.

**What the var actually gates.** Two features: (a) *prefill* chunk-level PLE — engine default ON unless '0', shipped since 0.1.13's rewrite (`7cf709e5`, Sep 28, measured RTX 5070 IQ3_S 383→1208 tok/s); (b) *decode* S25 verify-window projections — requires explicit '1' (`d92c9feb`, bit-identical by construction via multi-row matvec). gfx1151 ships '1' via arch defaults; the gfx1030 arch-defaults note "PLE_BATCH neutral" was a decode-side measurement; gfx1030 defaults only set SH_STREAM. So our configs were overriding the shipped default on every card, not just opting out of a gfx1030 feature.

**A/B (2026-10-19, 33,441-token probe, med-of-3, nonce'd; A = config '0', B = '1'):**

| | A per-token | B batched |
|---|---|---|
| Prefill PP (median) | 870.7 tok/s | **965.0 tok/s (+10.8%)** |
| ple GPU phase | 3954–3958 ms (10.3%) | **131–135 ms (0.4%)** |
| ple host time | ~3419 ms | 5–35 ms |
| Decode TG | 65.9 tok/s | 67.5 tok/s (within noise) |

Mechanism: per-token path re-reads ~52 MB BF16 key weights per token (~1.7 TB ≈ 3.5 s) + 66,882 tiny MMVF launches (~118 µs/token). Outputs: 8-token spot check identical; **512-token outputs diverge after 349 chars** (synonym-level rephrase; A emits EOS at 489, B runs to cap). Cause: the prefill batched path rounds activations to bf16 (+rocBLAS accumulation order) — on this f16_io config it is a f32→bf16→f16 double round-trip (~8 mantissa bits lost). It is the engine's shipped default arithmetic everywhere, NOT bit-identical (unlike decode S25). The "bit-identical" in d92c9feb covers only the decode window path.

**Stall status.** #541 has no fix commit upstream (`c3c03474`, Oct 3, added diagnostics only: STRATA_PF_STEP_SYNC; STRATA_KV_HOST_DMA is the gfx1201-family workaround). Current branch-B build: 3 × full 33K prefills + decode with '1', **zero stalls**. The Oct 4 stall predates the 0.1.4x prefill architecture. If a stall ever reappears: watchdog names it; STRATA_PF_STEP_SYNC=1 for diagnosis; revert to '0' only as last resort.

**Decision (2026-10-19).** `STRATA_PLE_BATCH` 0→1 in all 15 lane configs (both indents verified, none left at '0'). `strata-iq3_s-v043-spec5-plebatch.json` A/B copy deleted (now identical to base). **Baseline caveat: every recorded tier row through 2026-10-19 was captured under per-token PLE arithmetic — future rows are not directly comparable to the tables above; expect roughly +9–10% PP per tier.** Re-run the tier sweep before quoting new evolution-table numbers. Probe artifacts: `/tmp/probe_req.json`, `/tmp/probe_out_A*.json` / `/tmp/probe_out_B*.json`; engine log `strata-iq3_s-spec5.log` (timing lines at 33,441 tokens).

## Prefill PP candidate list at 33K (post-flip handoff, 2026-10-19)

Baseline for this list: the **per-token era** (pre-PLE-flip) timing line — 33,441 tokens, GPU timeline 38.3 s,
870.7 tok/s median (log: `strata-iq3_s-spec5.log`, timing lines at "33441 tokens"). After the PLE flip the same
request is 34.4 s / ~971 tok/s with `ple 131 ms (0.4%)`; all other phases moved <±1%, so the ms numbers below
still hold except `ple`. Probe: `/tmp/probe_req.json` (regular-32k fixture + nonce), `STRATA_PREFILL_TIMING=1`.
Post-flip tier sweep NOT yet run — re-baseline before quoting new table rows.

| Phase | ms (33K, pre-flip) | Verdict / next step |
|---|---|---|
| gemm gate/up (MMQ) | 5997 (15.6%) | **Closed.** ~80% of the int8 `v_dot4` silicon ceiling; no MFMA/WMMA on gfx1030 (rocWMMA needs RDNA3+). Don't touch. |
| qsa attn | 4705 (12.3%) | **Open, #1 by size.** SIMT attention ~18% of FP32 peak (~3.9 TFLOP/s). Fix depends on profiling: run experiment #3 with `tests/hip/`-adjacent `qsa_rows_bench.cpp` to separate compute-bound from gather-latency-bound BEFORE touching code. Kernels: `prompt_attn_kernel`/`prompt_attn_i8_kernel`/`prompt_attn_i8v3_kernel` in `src/kernels/cuda/qsa_prompt_attn.cu` (callers use attn_batch=32, cap=0). The WMMA twin (`prompt_attn_wmma_kernel`) is gfx11/gfx12-only (`hip_wmma_usable()` rejects gfx1030). Also `qsa select` grows superlinearly (132 ms @16K → 5.1 s @128K; FP32 tiled block scorer sweeps a growing block pool) — the 128K-tier context tax, drives PP drift 891→861 across tiers. |
| ple | ~4700 (~12%) | **DONE 2026-10-19** (flip documented above). Was: per-token re-read of ~52 MB BF16 key weights (~118 µs/token) + 66,882 tiny MMVF launches; batched = GEMMs + `native_ple_postops_batch`, 131 ms. |
| gdn | 4630 (12.1%) | **Partly open.** Recurrence 2396 ms at ~3.9 TFLOP/s is a SIMT chunked scan — hard without WMMA, deprioritize. `gdn out proj` 1815 ms IS a plain GEMM: check its shape against `tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt` for untuned/default rows. Sub-kernels (rocprof): `gdn_rec_cols_pipe` 1.73 s ×180, `gr_write_norm_rs` 0.90 s ×470, `gr_mix_r` 0.79 s ×483, `gdn_out_norm` 0.66 s ×180. |
| hc read | 3820 (10.0%) | **Two sub-candidates.** (a) GEMMs: down+up ≈ 2.57 s; isolated rate 2.44 ms/call (both rocblas_gemm_ex and hipblasGemmEx — API theory dead, probe `/tmp/gemm_probe2.cpp`); in-engine ~5.5 ms/call down / ~3.5× up — likely contention with concurrent expert streaming (price of overlap, possibly not recoverable). Deprioritize. (b) Elementwise side: hc mix 795 + hc inject 431 + norm/silu ~30 ms — the fusion idea (#2), still untested, worth ~1.2 s if halved. hc read re-materializes R and `gated` between 3 launches and 2 elementwise passes; `STRATA_HC_PERSIST` (#2226) proved 1.35–2.1× on this same read on gfx1030 but only for decode, NOT prefill. Code: hc section `src/prefill/prefill.cpp` ~2936–3060, now instrumented with sub-phases (hc q8/norm/down/silu/up/inject/mix — opt-in via STRATA_PREFILL_TIMING, default path untouched). |
| qsa proj | 2696 (7.0%) | Not yet examined. GEMM-shaped (rocBLAS Cijk_* family): check shape vs the tuning table for untuned rows. |
| dequant | 1491 (3.9%) | Small, coupled to the Q8 expert path. Low priority. |
| router+shared / combine | ~2000 | Small slices (`router+shared` 1049 + `combine` 946). Low priority. |

**Phase→kernel attribution** (rocprof trace `/tmp/rocp/flayer/3645178_results.db`, taken pre-flip; query: join
`rocpd_kernel_dispatch_00002ad6_8ccf_7ccf_b394_1511961f3242` `kernel_id` → `rocpd_info_kernel_symbol_...`.`id`;
35.3 s kernel time over 421,539 dispatches; engine GPU timeline is trustworthy — ~10 s host-side setup sits
before the first timing mark):
- `mul_mat_q<ggml_type*>` (MMQ) ≈ 8.2 s = gemm gate/up + gemm down (MoE experts, 48×512 top-10, n_ff 640).
- rocBLAS `Cijk_Alik_Bljk_HHS_BH_MT128x{256,128}x16_*` ≈ 9.6 s = hc down/up + qsa proj + gdn out proj.
- `attn_chunk_kernel_pre75` 4.09 s ×12,552 = qsa attn.
- `bf16_f32_mmvf_kernel<256>` 2.3 s ×66,882 = PLE per-token (2/token; grids 2,621,440 and 655,360) — gone post-flip.
- Draft-model prefill ran per-token on the same stream, ~1.15 s after each of 4 chunks — that WAS the ple phase.
- `kv_append` 0.54 s ×63 (8.6 ms each, 128K token buckets); `weighted_rms_norm` 100k × 2.9 µs (launch overhead only).

**Reference numbers:** geometry n_embd=2560, working width D=10240 (HC=4 hyper-connections), 48 layers all MoE,
512 experts/layer top-10, 36 GDN + 12 QSA (qsa_interval=4), NG_HIST=9. rocBLAS anchor 37.7 TFLOP/s measured on
RX 6900 XT (80 CU, 300 W) vs V620's 180 W cap — "percent of peak" figures indicative only. Config flags that
matter for any new arm: setup.py flags (`-DSTRATA_PREFILL_MMQ=ON` etc. — see the "missing 800 PP" note above),
`STRATA_HIP_PROMPT_F16=1`, `STRATA_ROCBLAS_TUNING=...gfx1030-...txt`, engine args `--prefill auto --spec 5`.

## Post-flip tier sweep — PLE_BATCH=1 baseline (2026-10-19)

Server: B on :8080, standard IQ3_S lane (strata-iq3_s-v043-spec5.json, PLE_BATCH=1), idle since the
A/B probe; GPU 23 °C at start. Protocol: 1 soak + 3 counted, med-of-3, `bench_strata_tiers.py`, temp 0,
max_tokens 1024. Runs: soak `strata_tier_pass_20261019_080722.json` (DISCARDED — cool-start, read ~2.5%
above settled as the record predicts); counted `081233` / `081806` / `082313`.

| Tier | PP med-of-3 | vs pre-flip row | TG med-of-3 | TG vs pre-flip |
|---|---|---|---|---|
| 16k | 916.7 | **+14.0%** (804.0) | 84.8 | +4.2% (81.4) |
| 32k | 928.2 | **+13.1%** (820.9) | 81.1 | +3.7% (78.2) |
| 64k | 930.8 | **+12.5%** (827.7) | 81.1 | +6.0% (76.5) |
| 128k | 917.0 | **+11.6%** (821.5) | 78.6 | +2.3% (76.8) |

- PP gain is ABOVE the +9–10% the flip caveat predicted, and gains shrink slightly with tier (14.0→11.6%) —
  same direction as the qsa-select superlinear growth, not the PLE phase itself (batched PLE is 131 ms
  tier-independent). PP spread across counted passes 0.3–1.6% (128K tightest, <0.3%).
- **TG is up 2.3–6.0% vs the pre-flip row — likely REAL, not noise.** Same binary (`build-hip/strata` = branch B),
  same day, same 1+3 protocol. The flip also enabled decode S25 verify-window projections (requires explicit
  '1'; previously OFF under '0' on gfx1030, since arch-defaults only set SH_STREAM). d92c9feb's multi-row
  matvec is bit-identical by construction, so this is a pure decode-side speedup on top of the PP gain.
  Corroborated by the probe (+2.4%, was dismissed as noise) and tier-consistent direction. Caveat: 4-hour
  session gap from the 04:xx pre-flip row; treat +2–6% as probable, not proven.
- Probe (965.0 @ 33,441 tok) vs 32K tier (928.2 @ 33,524 tok): ~3.8% gap. Probe ran as a fresh single
  request in its own session; tier runs chase 16K→128K, churning expert cache before 32K. Normal harness
  behavior — don't compare probe numbers to ladder medians.
- **Evolution table: these are the new baseline rows. All pre-flip rows (incl. Oct-14) are now superseded;
  do not quote them next to these without the +9–14% PLE-flip caveat.**

## 2026-10-18 — S23 experiment #3 run: qsa_rows_bench (qsa attn issue-vs-gather discrimination)

Built `tests/hip/qsa_rows_bench` per the record's experiment-#3 spec (`STRATA_HIP=1 make` in the usual
build tree; engine lane left serving). Ran with the server stopped — the binary needs ~150 MB VRAM and the
loaded spec-5 lane leaves only ~1.1 MB free, so `kill <server.py pid>` → bench → restart with the exact
recorded command line (`.venv/bin/python serve/server.py --engine strata --config
strata-iq3_s-v043-spec5.json --port 8080`, cwd Strata-main, nohup, log /tmp/strata-server.log). Server back
UP in ~10 s and idle; engine 0.1.45 reloaded the same IQ3_S lane cleanly.

Results (V620, gfx1030, `--kv int8` lane semantics, med-of-3 internal reps; NQ=2048, 32K ctx unless noted):

| run | ms/pass | TFLOP/s | % V620 peak (21.7) | logical reads |
|---|---|---|---|---|
| QA_SAME=1 (all cache-hit) | 20.49 | 5.04 | 23.2% | 216.5 GB/s |
| QA_SAME=0 (production drift) | 20.84 | 4.95 | 22.8% | 212.8 GB/s |
| QA_SAME=3 (no sharing) | 21.19 | 4.87 | 22.4% | 209.3 GB/s |
| QA_SAME=0, fp16 pool | 20.28 | 5.09 | 23.5% | 424.3 GB/s |
| QA_SAME=0, nq=32 | 0.31 | 5.18 | 23.9% | 222.6 GB/s |
| QA_SAME=0, nq=128 | 1.24 | 5.22 | 24.0% | 224.1 GB/s |
| QA_SAME=0, nq=512 | 4.97 | 5.19 | 23.9% | 222.9 GB/s |
| QA_SAME=0, ctx=16384 | 20.69 | 4.99 | 23.0% | 214.4 GB/s |
| QA_SAME=0, ctx=65536 | 20.79 | 4.97 | 22.9% | 213.4 GB/s |

**Answer to experiment #3's question: the qsa attn kernel is INSTRUCTION-ISSUE-LIMITED, not
gather-latency-bound.** Evidence:

1. QA_SAME sweep is flat within 3.4% (20.49 vs 21.19 ms). If gather latency were the limiter, QA_SAME=1
   (perfect reuse, all loads L2-resident) would separate sharply from QA_SAME=3 (churn). It doesn't — the
   kernel's own header comment ("limited by instruction issue, not memory") is now measured, not assumed.
2. nq scales perfectly linearly at constant TFLOP/s (~5.0–5.2, 23–24% of V620 peak) from nq=32 to 2048.
   Grid is `(ceil(cap/CHUNK), n_head_kv)` — the engine's 32-query batches already saturate the grid, so
   batching beyond 32 buys nothing for this phase (matches the header's prediction).
3. fp16 pool format is only 3% faster at 2× the logical read traffic — the limiter is not the load path
   either.
4. Per-query time is ~9.75 µs constant. Selection width caps at ~2051 cells/query (recent 512 + ~1539
   older), so per-pass work is constant; the ctx axis in this bench is degenerate (16K, 32K, 64K all hit
   the same cap → identical w-sum 4200448 and identical timing). This is why the engine's `qsa attn` phase
   was flat-ish in the 33K vs 128K timelines.

**Implication for the #1 open prefill phase (4.7 s of the 33K timeline):** any fix has to cut instructions
per cell, not loads. Concretely: (a) v_dot2_f32_f16 / v_dot4_i32_i8 intrinsic paths for the score pass
(RDNA2 has both, the WMMA path is gfx11-only but scalar dot intrinsics are not), (b) unroll/hoist the
dequant ladder so the inner loop is FMA-only, (c) merge the chunk+merge two-pass structure if the merge
step is a measurable fraction. The 5× headroom implied by 23% of peak is the same-sized prize the record's
candidate list flagged; a 2× win here saves ~2.3 s of prefill at 33K.

Engine lane restored and verified; no spec or code changes were made — this was a measurement-only run.

## 2026-10-19 — S23 experiment #4: dot-intrinsic score pass, gfx1030 micro-bench (tests/hip/dot_score_bench.cpp)

Question: the record's #1 prefill fix is "cut instructions per cell" in the QSA attention score pass. RDNA2
has scalar dot instructions (v_dot2_f32_f16, v_dot4_i32_i8) that the WMMA kernels cannot use (gfx11-only),
and the KV pool is already int8 codes + per-64 fp16 scales. Do the intrinsics actually issue faster than
the current 8-FMA-per-head inner loop on V620?

Method: new micro-bench `tests/hip/dot_score_bench.cpp` mirrors the exact loop shape the engine runs on
gfx1030 — attn_chunk_kernel<KV_MODE=1,false,false,false> (256 threads, CHUNK=64 cells/block, warp scores
cells warp/warp+8/..., per lane per cell: load8_q8 dequant of 8 K dims, G=12 heads × (2× float4 q-LDS +
8 FMA), warp_sum reduce). Three variants, same loop skeleton, same row indirection, differing only in
score math:
- fma — exact mirror (fp32 q tile 12 KB, 2 float4 LDS reads/head, 8 FMA/head)
- dot2 — q tile as half2 (6 KB, one b128 LDS read/head), 4× __builtin_amdgcn_fdot2 (v_dot2c_f32_f16,
  fp32 accumulator) per head; K codes→fp16, per-64 k_scale folded after the dot
- dot4 — q tile as per-head int8 codes (3 KB, one b64 LDS read/head), 2× __builtin_amdgcn_sdot4
  (v_dot4c_i32_i8, exact i32 accumulator) per head; K codes used raw (no dequant), i32 partial × k_scale
  per lane, per-head q_scale folded into the write (k_scale is per-lane per-64-group, q_scale is
  per-head-uniform, so q_scale factors out past the warp_sum reduce)

Build note: plain `hipcc` (system /usr/include HIP 7.15) rejects the kernel's 32-bit shfl mask
(64-bit-mask static_assert); the engine builds with /opt/rocm/core-10.0/.../clang++ plus the repo shim
`-I include/strata/hip_compat -include include/strata/hip_compat/cuda_runtime.h` — use those flags.
ISA check (clang --cuda-device-only -S): v_dot2c_f32_f16 and v_dot4c_i32_i8 both present; the accumulate
("c") forms are the same RDNA2 instructions with the accumulator operand folded.

Results (V620, 36 SMs, grid 4096 blocks × 256 threads, best-of-7 hipEvent):
- fma  0.6048 ms  (2.31 ns per 64-cell block-pass)
- dot2 0.3188 ms  1.90× vs fma
- dot4 0.3250 ms  1.86× vs fma

Reading: the dot instructions are NOT throughput-starved on gfx1030 — the instruction-count cut converts
almost fully to wall-clock. The 1.9× exceeds the pure instruction-count projection (~1.3×) because the
smaller q tiles also cut LDS traffic and bank conflicts (fp32 b128 reads have an 8-way conflict pattern
at the kernel's lane*8 stride; half2 4-way, int8 b64 2-way).

Numerics: dot2 rounds K/q to fp16 (~1e-3 rel); dot4 quantizes q per head to int8 (~0.4%). Both change
score inputs vs the fp32 path, so any integration must go through the existing parity harnesses and be
opt-in behind a flag; the default path stays byte-identical (AGENTS.md).

Projection for the 33K prefill timeline: qsa attn prefill measured ~4.7 s of 33K (experiment #3). A 1.9×
on the score loop is not a 1.9× on the phase (V-side accumulation, softmax and merge are unchanged), but
the score loop is the dominant part of the phase, so the plausible phase-level win is ~1.5-1.7×, i.e.
~1.5-2 s of the 33K prefill. Next steps if pursued: (1) prototype in qsa_decode_attn.cu behind an
opt-in env flag (e.g. STRATA_ATTN_DOT4=1), (2) parity vs the current path, (3) end-to-end
qsa_rows_bench + tier sweep. Second knob if the phase-level win disappoints: the warp_sum reduce is 120
of the ~256 per-lane-per-cell instructions — the gfx906 reduce12 lane-specialization (16 shuffles for 12
heads instead of 12×(5 shfl+add)) is currently gfx906-gated and could be evaluated separately.

VRAM note: the standalone bench needs ~30 MB; measurement-only.

## 2026-10-19 — S24 experiment #4 follow-through: DOT2 integrated into the engine, parity OK, runtime regression — verdict: abandon DOT2

Method: integrated the dot2 score branch into `src/kernels/cuda/qsa_decode_attn.cu` behind
`STRATA_ATTN_DOT2=1` (opt-in; default path byte-identical per AGENTS.md). Template flag `bool DOT2=false`
on `attn_chunk_body` / `attn_chunk_kernel` / `attn_chunk_rows_kernel` with static_asserts
(`!DOT2 || KV_MODE==1`, `!DOT2 || !LANE_CELL`, `!DOT2 || !QUERY_SWIZZLE`); shared q tile became a
`union qt { float sq[G][HD]; __half2 sq2[G][HD/2]; }`; DOT2 score branch reads the q tile as `uint4`
from LDS and issues 4x `__builtin_amdgcn_fdot2` per head, `warp_sum(s*sc)`, `if (lane==0) sp[h][c]`.
Dispatch wired at both sites (`launch_chunks_batch` and the rows-window dispatch) under
`__HIP_PLATFORM_AMD__`, kv_mode==1 only, gated off when the parity harness names a kernel.

Parity: `tests/hip/qsa_rows_bench.cpp` got an attn checksum print (sum of the attn output buffer after
the timing loop). Default path run twice -> bit-identical (sum -2.103918665e+03). DOT2 run -> sum
-2.103562091e+03, differs ~2e-4 relative — the expected fp16 q rounding, no sign the plumbing is wrong.

ISA check (build-hip object, /tmp/attn_gfx1030.s): DOT2 kernel
`attn_chunk_kernelILi1ELb0ELb0ELb0ELb1EE` = 33354 instrs, 96 `v_dot2c_f32_f16`, 34 VGPRs, 0 spills —
statically nothing is wrong.

End-to-end A/B (qsa_rows_bench, ctx 32768, nq 512, int8, server stopped):
- default 5.10 ms/pass (5.06 TFLOP/s)
- DOT2 7.78 ms/pass (3.32 TFLOP/s) — 1.48x SLOWER, the opposite of the micro-bench's 1.90x faster.

Attempted fix (hoist): moved the 12 per-head `qt.sq2` reads into a register array `uint4 qraw[G]` before
the head loop so they do not serialize behind the previous head's bpermute reduce + `sp` write. Rebuilt,
re-timed: 7.78 ms — no change. The ISA shows the 12 head reads were not the blocker.

Bench re-run with the engine's real structure: patched `tests/hip/dot_score_bench.cpp` so all three
variants use the engine's post-loop LDS pattern (`if (lane==0) sp[h][c] = s*SCALE` per cell, then after
the cell loop `__syncthreads(); for h: r += sp[h][t & (CHUNK-1)]`) instead of keeping scores in
registers. Results (V620, 4096 blocks, best-of):
- fma  0.5664 ms (1.00x)
- dot2 0.3918 ms 1.45x   (was 1.90x register-accumulated)
- dot4 0.3704 ms 1.53x   (was 1.86x)

So even the 1.45x did not survive contact with the real kernel (7.78 vs 5.10 ms). The residual gap is
structural, not per-cell math: the real kernel launches one block per (chunk, kv-head, query) — cap 32768
x 2 KV heads -> 512+ blocks of only 64 cells each, so per-block fixed cost (q-tile fp16 conversion of
12x256 halves, srow/page setup, `__syncthreads`) is a much larger share than in the 4096-block bench.
The DOT2 path also adds work the default does not (q->fp16 pair conversion at block start, K codes ->
half2 per cell) that the default's fp32 tile never pays. At cap 2051 the fixed cost dominates even more
(attn 26.6 -> 49.1 us at T=1).

Verdict: abandon DOT2 for the engine. The `STRATA_ATTN_DOT2=1` opt-in stays in the tree (off by default,
harmless, parity-verified) but is not worth enabling. The default path is unchanged. Second knob next:
the warp_sum reduce is ~120 of the ~256 per-lane-per-cell instructions; the gfx906-gated
`reduce12` lane specialization (16 shuffles for 12 heads instead of 12x(5 shfl+add)) does not change
numerics or add per-block setup, so its bench projection should transfer more faithfully than DOT2's did.


## 2026-10-19 — S25 experiment #5: reduce12 lane specialization (gfx906 port) — verdict: 1.71x SLOWER on gfx1030, abandon

`src/kernels/cuda/qsa_decode_attn.cu` was extended with `STRATA_ATTN_REDUCE12=1`, gating
`reduce12_gfx906`'s lane-specialized warp_sum on any HIP card (previously gfx906-only). Dispatch sites:
chunk kernel (line ~1073) and rows kernel (line ~1236), both `kv_mode == 1 && !lane_cell` only.
Default path is byte-identical (flag off = old branch taken).

A/B on qsa_rows_bench (ctx 32768, nq 512, int8, 4 invocations each with reps=1, first discarded as soak):

- default  med-of-3: 5.20 ms (soak 5.20, passes 5.20/5.23/5.08)
- REDUCE12 med-of-3: 8.89 ms (soak 8.87, passes 8.88/8.90/8.89)
- checksums bit-identical to default (sum -2.103918665e+03) — parity holds

Slowdown, not speedup: 8.89/5.20 = 1.71x. The 16-shuffle lane specialization is a net loss on gfx1030.
Why (hypothesis, not measured): the specialization serializes the 12 heads' reductions behind each other
inside one lane instead of keeping them independent as 12x(5 shfl+add), and gfx1030's shuffle throughput
per warp is lower than gfx906's per-lane shuffle model assumed. Also per-block LDS write→read ordering
(the same structural issue that killed DOT2's projection) is likely compounded because the specialized
reduce holds scores in fewer lanes for longer.

Verdict: abandon reduce12 for the engine. The `STRATA_ATTN_REDUCE12=1` opt-in stays in the tree (off by
default, harmless, parity-verified). Default path is unchanged.


## 2026-10-19 (evening) — S26: full tier sweep, default path, after DOT2/REDUCE12 integration

Server restarted with the same cmdline (no STRATA_ATTN_DOT2 / STRATA_ATTN_REDUCE12 set, so the run
exercises the default kernel path) and the tier sweep re-run: 1 soak (discarded, cool-start) +
3 counted passes, med-of-3. Same protocol as the 08:07–08:23 baseline earlier today.

```
           PP tok/s            TG tok/s
tier      tonight   vs AM     tonight   vs AM
16k       892.7     -2.6%     81.8      -3.5%
32k       913.7     -1.6%     77.8      -4.1%
64k       921.6     -1.0%     79.0      -2.6%
128k      914.5     -0.3%     77.5      -1.4%
```

The deltas are negative but the run shows clear thermal drift: within the evening run the first
counted pass is the fastest and each later pass is slower (16k PP 929.3 → 892.7 → 890.7, a 4.3%
spread; 32k 942.0 → 913.7 → 912.2, 3.3%). The morning baseline's same-tier spread was only 0.3–1.6%,
so the evening deltas are within the run's own drift envelope and are not evidence of a regression
from the DOT2/REDUCE12 code, which is off by default. The first evening pass at every tier (929.3,
942.0, 937.9, 920.7 PP) is actually ABOVE the morning medians — on a cool GPU the default path
measures no slower than this morning's baseline.

Per-pass detail (counted passes only):

```
pass      16k PP/TG     32k PP/TG     64k PP/TG     128k PP/TG
1st       929.3/84.8    942.0/80.3    937.9/77.6    920.7/76.0
2nd       892.7/81.8    913.7/77.8    921.6/81.3    914.5/77.6
3rd       890.7/78.7    912.2/76.3    921.1/79.0    913.8/77.5
```

JSONs: strata_tier_pass_20261019_172623 (1st), _173115 (2nd), _173610 (3rd); soak _172049 discarded.

Verdict: the default path is unchanged and matches the morning baseline on a cool GPU. The evening
run is consistent with thermal throttling under sustained tier-ladder load, not with any code effect.
For any future same-day A/B, run both arms back-to-back and only compare the first pass of each arm
to each other, or re-soak between arms.

## 2026-10-19 (late evening) — S27: hc sub-phase prefill timing measured (STRATA_PREFILL_TIMING)

The hc sub-phase instrumentation in `src/prefill/prefill.cpp` (S23, commit 09819e4c) was built into the
engine (build-hip relinked 22:47) and measured. The report goes to the engine's own stderr log
(`strata-iq3_s-spec5.log` via server.py's `stderr=self.log`, line 843) — NOT /tmp/strata-server.log, which
is why the earlier sweep (S26) showed 0 hits. Each tier's prefill prints one report line. One 16k pass
at 22:53 and the tail of a full sweep (22:53–22:57) provided the numbers below.

Breakdown per tier (GPU timeline ms, hc sub-phases; "hc read" = sum of the hc sub-phases per the fold()):

```
tier     tokens   total     hc total  (pct)   down    up     inject  mix    norm+silu+q8
16k      16817    17407     1941      (11.2%)  668    628    220     408    16
32k      33516    34700     3847      (11.1%)  1329   1260   428     796    30
64k      66974    70226     7777      (11.1%)  2687   2566   844     1614   61
128k     133880   144429    15667     (10.8%)  5429   5219   1659    3231   122
```

Scaling is clean-linear (all hc phases double when n doubles). The other top phases for context:
gemm gate/up 17.1–17.7%, gdn 13.5%, qsa attn 13.0–14.4%, qsa proj 7.7–7.9%, gemm down 8.5–8.7%.

The S23 fusion candidate ("hc mix 795 + hc inject 431 + norm/silu ~30 ms, worth ~1.2 s if halved") is
now measured directly: mix + inject + norm/silu at 32k = 796 + 428 + 30 = 1254 ms — the ~1.2 s estimate
was right. Halving it saves ~630 ms of 34.7 s (~1.8% PP at 32k). The bigger hc phases are down (3.8%)
and up (3.6%) — those are GEMMs (rocBLAS), and `gr_upmix` (up+mix fused, WMMA) is gfx11-only, so a
gfx1030 port would be a SIMT rewrite competing with a tuned rocBLAS GEMM — much higher risk than the
elementwise fusion.

Next: read the hc section (`src/prefill/prefill.cpp` ~2936–3060) to map the launches and buffers,
then size the elementwise fusion (inject+mix, and possibly norm+silu) as a gfx1030-able change.



## 2026-10-19 (late evening, cont.) — hc geometry + traffic analysis: gr_upmix is the target, not the tiny phases

From the pack index (`/home/drewster/git/Strata-data/packs/iq3_s/index.txt`): 48 layers (`blk.0–47.`), 22
tensors/layer. hc geometry: D=2560 (hidden), hc width D*HC=10240 (4 streams x 2560), LR=320 (down 10240->320,
up 320->10240), inject 4 outputs (one scalar per stream). So each layer has 2 hc chains (attn + ffn) =
**96 hc chains per prefill pass**.

Traffic math on the measured phases (32k = 33516 tokens, mix 796 ms over 96 launches = 8.3 ms/launch):
gr_mix_r reads R (4 streams fp32, 40 KB/tok) + gated (40 KB/tok), writes mixed (5 KB/tok) + mixed16/lo (~15
KB/tok) ≈ 100 KB/tok moved per launch. Per launch: 33516 x 100 KB = 3.4 GB in 8.3 ms = ~410 GB/s ≈ 80–90%
of V620 bandwidth. **mix_r is already near memory-bound — fusing inject/norm/silu into it would save
nothing.**

The real target is the **gated materialization round-trip** between the up-GEMM and mix: the unfused chain
does lo16 read -> up GEMM writes gated (40 KB/tok) -> mix reads gated (40 KB/tok) + R (40 KB/tok) -> mixed
write (20 KB/tok). `gr_upmix` (gfx11-only WMMA) fuses up+mix: one GEMM whose epilogue reads R and writes
mixed directly, skipping the entire 80 KB/tok gated round-trip. Per chain: unfused up+mix ≈ 141 KB/tok,
fused ≈ 61 KB/tok — a ~57% cut of the up+mix pair. Scaling to the measured numbers (up 1260 + mix 796 =
2056 ms, weighted ~80/141): **estimated ~900–1160 ms saved at 32k, i.e. ~2.6–3.3% PP** — an order of
magnitude bigger than the inject+mix elementwise fusion idea from S27 (~630 ms).

Cost on gfx1030: gr_upmix's GEMM part is T x 320 x 10240 = 2.2e11 FLOPs (fp16) at 32k. rocBLAS does this
in ~13.1 ms/launch (1260 ms/96) ≈ 63% of the ~26.5 TFLOPS fp16 dot2 peak. A SIMT gfx1030 port would need
a hand-rolled fp16 GEMM (dot2, like the abandoned S24 DOT2 work but as the GEMM mainloop) to reach ~80%
of peak before the fusion epilogue's extra traffic pays for the difference. Risky — same per-block-fixed-
cost trap that killed DOT2 — but the win ceiling (1 s at 32k) is 5x the elementwise fusion's.

Verdict for the queue: the hc read fusion task is properly sized now. The gfx1030 gr_upmix port is the
high-ceiling/high-risk option; the inject+mix fusion (~630 ms ceiling) is the low-risk fallback if the
SIMT GEMM can't reach rocBLAS efficiency. Next concrete step if pursuing: micro-bench a SIMT fp16 GEMM
for the [33516, 320] x [320, 10240] shape on V620 and compare against rocBLAS's 13.1 ms/launch before
writing any epilogue.




## 2026-10-19 (late night) — S28: up-projection GEMM micro-bench on gfx1030 (gr_upmix gate)

`tests/hip/upmix_gemm_bench.cpp` (408 lines), bench commit pending. Purpose: decide whether a hand-rolled
SIMT fp16 dot2 GEMM can replace rocBLAS for the hc up-projection `[T,320] x [320,10240]`, which is the
prerequisite for porting `gr_upmix` (up+mix fused) to gfx1030.

Arm A = production `Gemm::bf16` with `set_f16_io(true)` (the exact `STRATA_HIP_PROMPT_F16=1` path, same
tuning table as the server). Arm B = hand-rolled SIMT fp16 `v_dot2_f32_f16` GEMM, two tile shapes:
64x128 tile, 4x8 per thread, BK=32, 256 thr, double-buffered LDS (B1) and 128x128 tile, 4x8 per thread,
BK=32, 256 thr (B2). Y written fp32 directly (no widen round-trip). Superblock swizzle (8x8 tiles per
superblock) keeps the L2 working set ~1 MB.

Hardware correction that matters for all earlier floor math: the V620 reports **36 CUs** (not 72 as
assumed in the S27 notes), 2570 MHz. Theoretical packed-f16 peak = 36 x 64 x 4 FLOP x 2.57 GHz =
**23.7 TFLOPS** (matches the measured kPeakV620 = 21.7e12 within 8%). The old "~26.5 TFLOPS dot2 peak"
and the 8.3 ms compute floor from the S27 notes were both based on 72 CUs and are wrong; the correct
compute floor for the 32k shape (2.197e11 FLOP) is **10.1 ms**, and the memory floor (X 21.9 MB + W
6.55 MB + Y 1.373 GB at ~512 GB/s) is ~2.7 ms, i.e. the shape is compute-bound.

Results (soak 12 reps discarded, then 3 counted passes of 12 reps, median):

    T=2048  (13.4 GFLOP):  A 0.580 ms / 23.1 TF   B1-kernel 0.590 / 22.7   B2-kernel 0.594 / 22.6
    T=8192  (53.7 GFLOP):  A 2.990 ms / 18.0 TF   B1-kernel 3.066 / 17.5   B2-kernel 2.933 / 18.3
    T=33516 (219.7 GFLOP): A 13.49 ms / 16.3 TF   B1-kernel 12.47 / 17.6   B2-kernel 12.32 / 17.8

At the production shape (T=33516) the dot2 kernel **beats production rocBLAS by ~8.5%** (12.35 ms with
per-call W convert, 12.32 ms kernel-only vs 13.49 ms for arm A, which includes bf16->f16 X convert +
rocBLAS + widen). At T=2048 rocBLAS is slightly faster (0.58 vs 0.59 ms) but that shape is launch-bound
(23 TF on 2.2e11 FLOP = 0.6 ms); at T=8192 they are even.

dot2 efficiency vs peak: 17.8 TF / 23.7 TF theoretical = **75%**; vs the measured kPeakV620 21.7 TF =
82%. The S28 gate was "~63-80% of peak to justify the port" — **the gate is met** (75-82%), and the
kernel also directly beats the thing it would replace.

Correctness: arm B vs CPU fp64 reference at T=2048: max rel 6.5e-07 (arm A, which writes fp16, sits at
4.6e-04, i.e. one fp16 ulp — expected). Tail-108-rows check at T=33516 (partial-M block, 33408..33515):
arm B max rel 8.9e-07, 0/108 rows with abs diff > 1e-3. Two real bugs were caught and fixed during
bring-up: (1) the grid was sized mb*nb instead of ceil(mb/8)*ceil(nb/8)*64 for the superblock swizzle —
silently skipped the last partial super-row (all 108 tail rows wrong), (2) a T=0 sample in the boundary
check read before the start of the captured buffer (host segfault).

rocBLAS note: the tuning table `gfx1030-rocblas-5.6.0.8d1ae90e.txt` is accepted but rocBLAS prints
`version mismatch: file=5.6.0.8d1ae90e runtime=5.1.0; rocBLAS's default kernels run` — same warning the
server gets, and arm A's 13.49 ms matches the server's measured 13.1 ms/launch, so the comparison is
faithful to production. Phase-2 re-tune (queued) may close part of the 8.5% gap; but the dot2 kernel
also writes fp32 Y directly and can take the gr_upmix epilogue for free, which rocBLAS cannot.

Verdict for the queue: the gr_upmix fusion port is **justified** — proceed with the B1/B2 mainloop as
the gr_upmix GEMM core for gfx1030, B2 (128x128) preferred at the production shape. Keep rocBLAS as the
fallback for T <= 2048 (or use B1 there too; the delta is one launch-length). Estimated win at 32k: up
13.49 -> ~12.3 ms kernel-only, plus the ~900-1160 ms fused-epilogue traffic saving estimated in S27,
i.e. roughly the full ~1 s ceiling minus ~1.2 ms x 96 launches of GEMM-mainloop delta ≈ **~1.05-1.2 s
per prefill pass (~3% PP)**.



## 2026-10-19 (cont.) — S28 next-step decision: lowest-effort first

Three options were on the table after the S28 gate passed:

1. **gr_upmix port** (highest effort/risk, biggest win ~1.0-1.2 s/32k). B2 mainloop proven by the
   bench; work = epilogue (read R, apply gate/mix, write mixed + mixed16/lo), parity harness,
   opt-in env flag, byte-identical default path.
2. **Phase-2 rocBLAS re-tune** (near-zero risk, unknown payoff). rocBLAS runs default kernels today
   (`version mismatch: file=5.6.0.8d1ae90e runtime=5.1.0` — same on the server), and K=320/N=10240
   is an odd shape likely badly auto-tuned. A real gfx1030 table covers every rocBLAS GEMM in the
   engine, not just up-proj. Can only shrink arm A (13.49 ms) and erode the 8.7% GEMM headroom, so
   it must be known BEFORE committing to the port.
3. **Elementwise fusion (inject + mix_r + norm/silu)** (~630 ms, ~1.8% PP). Largely subsumed by
   option 1; doing it first means partially throwing the work away.

**Decision (user): do the lowest effort first -> option 2, the rocBLAS re-tune.** Cheap data that
also de-risks the port decision. Option 1 stays next after that if the headroom survives.

Priority order from here:
1. rocBLAS gfx1030 tuning table for the hc shapes (up 320->10240 and the other engine GEMMs),
   verify server picks it up, re-measure arm A. (NOW)
2. Re-check the gr_upmix gate against the retuned arm A; port only if headroom survives.
3. Elementwise fusion only if the port is abandoned.
4. Housekeeping: commit tests/hip/upmix_gemm_bench.cpp (untracked on f-gfx103x-sh-stream).



## 2026-10-19 (cont.) — S29: rocBLAS re-tune at production T: no gains, but it exposed the real target

Correction to the S28 entry (wrong there): rocminfo says gfx1030 = **72 CUs** at 2570 MHz, not 36.
FP16 packed peak = 72 x 64 SP x 4 FLOP x 2.57 GHz = **47.5 TFLOPS**; FP32 FMA peak = 23.7 TFLOPS
(that is what kPeakV620 = 21.7e12 actually measured, 91% of FP32 peak). Consequence: the dot2 bench
kernel at 17.8 TF is only **38%** of packed peak, not 75-82%.

Ran tools/hip/tune_rocblas (built clean, 281 lines) over all big engine shapes at T=33516, 237
solutions each, TOP=8 full-measured. Every case: **default wins** (keep_default):

    N=10240 K=2560:  default 66.78 ms (26.3 TF)   best 72.91
    N=12288 K=2560:  default 84.41 ms (25.0 TF)   best 90.49
    N=6144  K=2560:  default 41.46 ms (25.4 TF)   best 45.20
    N=2560  K=6144:  default 47.24 ms (22.3 TF)   best 47.61
    N=2560  K=2560:  default 13.97 ms (31.4 TF)   best 17.13
    N=320   K=10240: default 10.29 ms (21.3 TF)   best 11.67
    N=10240 K=320:   default  6.03 ms (36.4 TF)   best  6.31  (up-proj, the S28 shape)

Tuning table unchanged (all rows would read `default` anyway). Phase-2 retune: **no gain available**.

The insight that matters: production up-launch is 13.1 ms (server) / 13.49 ms (bench arm A), but
rocBLAS's GEMM alone is **6.03 ms** (36.4 TF = 77% of packed peak - the most efficient shape). The
other ~7 ms is overhead in Gemm::f16_inplace (src/prefill/gemm.cu:1202-1216): the GEMM runs beta=0
writing FP16 into the low half of each FP32 slot of the [ldy*2, T] buffer, then widen_rows_f16
launches unconditionally (always, line 1216) and expands to FP32 in place - ~686 MB read + ~1.37 GB
write per launch, and it is 2x slower than a plain copy (in-place expansion). The S28 "B2 beats
production by 8.7%" was measuring B2 (which writes FP32 directly, no widen) against rocBLAS+widen,
NOT against rocBLAS's GEMM - the dot2 kernel itself is 2x slower than rocBLAS.

**Revised verdict: the gr_upmix GEMM port is dead** (a SIMT kernel would give up 6 ms/launch to
rocBLAS to save ~4 ms of widen). The real prize, in effort order:

1. **Skip the widen**: keep rocBLAS's FP16 output and have the consumer (mix_r / hc chain) read FP16
   directly. Widening FP16->FP32 is lossless, and FP32 accumulation of FP16 x FP16 products is exact
   (22 mantissa bits <= 24), so numerics are unchanged. Saves the widen (~4 ms) and halves mix_r's
   gated read (1.37 GB -> 686 MB, ~1.3 ms) ~= 5.3 ms/launch x 96 ~= **510 ms/32k (~1.5% PP)**.
   Needs: check what mix_r actually consumes (gated fp32 vs fp16) and whether any consumer needs FP32.
2. **Fused epilogue via a custom GEMM** - only if (1) is infeasible AND the 6 ms/launch GEMM penalty
   can be recovered, which the tuner says it cannot (38%-efficient SIMT vs 77% rocBLAS). Dead.
3. Elementwise fusion (inject+mix_r+norm/silu) as before (~630 ms) - still alive, orthogonal.

Next step: read the hc chain code (prefill.cpp ~2936-3060) to see what consumes the up GEMM output
and whether a direct-FP16 consumer is possible.

## F-2 (skip-widen, STRATA_HC_MIX_H16=1) - VERIFIED

Replaced the templated gr_mix_r_kernel (Packed16 bool) + gr_mix_r_h16 wrapper that was accidentally
dropping the sigmoid, which had been confusing parity. The default path's gr_mix_r now includes
sigm(gv) again.

**Parity harness (`tests/hip/mix_h16_parity.cu`, 21 shapes T=1..32768):**
- F-2 vs default mixed output: bitwise identical, max |diff| = 0 at every T.
- Both vs FP64 CPU reference (sampled): 4.5e-05 .. 1.17e-04 relative - FP16 rounding scale,
  consistent between default and F-2. 0 failures.

**Bench (`mix_h16_parity --bench`, medians of 20 alternating runs, gfx1030/V620):**
- T=2048:   default 1.111 ms (GEMM+widen 0.637 + gr_mix_r 0.474) | F-2 0.802 ms (GEMM 0.427 +
            gr_mix_r_h16 0.375) = 1.39x
- T=8192:   default 5.574 ms (3.035 + 2.539)                    | F-2 3.543 ms (1.723 + 1.820) = 1.57x
- T=32768:  default 21.815 ms (12.717 + 9.097)                  | F-2 14.261 ms (6.983 + 7.278) = 1.53x

Verdict: skipping widen_rows_f16 and having the consumer read packed FP16 halves directly is the
win predicted (GEMM itself ~6-7 ms is what rocBLAS gives us; widen is pure overhead). Gate behind
STRATA_HC_MIX_H16=1 (opt-in), default path byte-identical.

**Server startup (correct flags from ~/git/Strata/strata-iq3_s.json):**
- --pack /home/drewster/git/Strata-data/packs/iq3_s
- --native /var/lib/ollama/gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
- --ple-gguf /var/lib/ollama/gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
- --prefill auto --spec 4 --spec-min-p 0.5 --mtp /home/drewster/git/Strata-data/mtp/rt
- --max-context 262144 --kv int8 --kv-resident 32768 --ple-io ram --vram-reserve-mib 200
- env: STRATA_WATCHDOG_S=300 STRATA_PREFILL_TIMING=1 STRATA_PLE_BATCH=0 STRATA_HIP_PROMPT_F16=1

### F-2 server A/B — baseline arm (STRATA_HC_MIX_H16=0), T=32005, 2026-10-20
soak + 3 counted passes (b1..b3), med-of-3 (ms):
- b1: up 1202, inject 1266, mix 771 | b2: up 1216, inject 1226, mix 771 | b3: up 1226, inject 1230, mix 770
- median: up 1216, inject 1230, mix 771; hc read 4590; total GPU timeline 32871 ms (972 tok/s wall; passes 32550/32871/33100)
- in-server parity check (STRATA_HC_UPMIX_CHECK=2): "mixed rel RMS 0.000e+00, max |diff| 0.000e+00 (f16-skip)" — F-2 == default bitwise at T=8192 in-server

### F-2 server A/B — F-2 arm (STRATA_HC_MIX_H16=1), T=32005, 2026-10-20
soak + 3 counted passes (f1..f3), med-of-3 (ms):
- f1: up 739, inject 643, mix 622 | f2: up 741, inject 645, mix 624 | f3: up 738, inject 642, mix 621
- median: up 739, inject 643, mix 622; hc read 3355; total GPU timeline 31829 ms (1004 tok/s wall; passes 31507/31829/32080)
- in-server parity check: mixed rel RMS 0.000e+00, max |diff| 0.000e+00 (f16-skip) — held in all three passes

### F-2 server A/B — comparison (med-of-3, same 32005-token prompt, cache invalidated each pass)

| phase | baseline (ms) | F-2 (ms) | delta | % |
|---|---:|---:|---:|---:|
| **GPU timeline (total)** | **32871** | **31829** | **-1042** | **-3.2%** |
| hc read | 4590 | 3355 | -1235 | -26.9% |
| hc down | 1323 | 1326 | +3 | +0.2% |
| hc up | 1216 | 739 | -477 | -39.2% |
| hc inject | 1230 | 643 | -587 | -47.7% |
| hc mix | 771 | 622 | -149 | -19.3% |
| gemm gate/up | 5350 | 5396 | +46 | +0.9% |
| qsa attn | 4637 | 4631 | -6 | -0.1% |
| gdn (all) | 4386 | 4469 | +83 | +1.9% |

Summary:
- **Target phases (hc up+inj+mix)**: 3217 → 2004 ms, saving **1213 ms (-37.7%)**. Adding hc read
  (-1235 ms, reduced too), the gross hc-chain saving is **2448 ms**.
- **But the measured GPU timeline only improved by 1042 ms (-3.2%)**, not the full 2448 ms.
  Two reasons — secondary-phase drift and overlapped/hidden time — are quantified in the
  reconciliation section below.
- `hc down` is essentially unchanged (+3 ms, expected — it wasn't an F-2 target).
- Parity held throughout: rel RMS 0.000e+00, max |diff| 0.000e+00 for all three hc phases.
- Note on the baseline: an earlier draft of this table used the b1 first pass (32550 ms) as the
  baseline, which understated the win as -721 ms / -2.2%. The med-of-3 baseline is 32871 ms; the
  corrected end-to-end delta is **-1042 ms / -3.2%** (wall delta also -1042 ms, 972 → 1004 tok/s).

Verdict: F-2 is a real, reproducible ~3.2% GPU-timeline win at T=32005 with zero numerical
change (bitwise identical). It is gated behind STRATA_HC_MIX_H16=1 and the default path stays
byte-identical. Promotion is pending user decision — see "Open decisions & pending tasks
(as of 2026-10-20)" at the end of this file for the options.


### F-2 A/B — reconciliation of the unaccounted ~1200 ms (2026-10-20)

Problem: the per-phase deltas sum to ~-2242 ms, but the total GPU timeline only improved by
-1042 ms (med-of-3: baseline 32871, F-2 31829). Where did the other ~1200 ms go?

**Extraction (all 8 passes, counted set only, corrected baseline medians):** extracted EVERY phase
from the `strata prefill timing` lines for both arms. First: the earlier comparison table's
baseline medians were wrong — the doc's b1..b3 numbers came from the b1 first pass, and
"total GPU timeline 33100" was the b3 pass, not the median. Correct med-of-3 baseline (from the
raw logs): tl 32871, hc read 4590, up 1216, inject 1230, mix 771. F-2 medians were already right
(tl 31829, hc read 3355, up 739, inject 643, mix 622). So Δtl = 31829 − 32871 = **-1042 ms**.

Full phase-delta table (med-of-3 counted passes, ms):

| phase | base | F-2 | delta |
|---|---:|---:|---:|
| hc read | 4590 | 3355 | -1235 |
| hc inject | 1230 | 643 | -587 |
| hc up | 1216 | 739 | -477 |
| hc mix | 771 | 622 | -149 |
| hc down | 1323 | 1326 | +3 |
| hc norm / silu / q8 | 27 | 27 | 0 |
| embed+steps | 68 | 68 | 0 |
| gdn (total) | 4386 | 4469 | +83 |
| gemm gate/up | 5350 | 5396 | +46 |
| qsa proj | 2588 | 2614 | +26 |
| router+shared | 938 | 976 | +38 |
| gemm down | 2672 | 2690 | +18 |
| qsa attn | 4637 | 4631 | -6 |
| qsa select / indexer | 516 | 515 | -1 |
| combine / dequant | 2084 | 2084 | 0 |
| ple / kv stage | 164 | 164 | 0 |
| host grouping / wait copy / gather | 388 | 388 | 0 |
| **SUM of phase deltas** | | | **-2242** |
| **GPU timeline (measured)** | **32871** | **31829** | **-1042** |
| **unaccounted** | | | **+1200** |

**The unaccounted ~1200 ms is NOT in unextracted phases** — every phase is now extracted
(embed+steps, wait copy, ple, kv stage all delta ≈ 0). The explanation is **timer overlap**:

- The phase timers (`PfTimer` in prefill.cpp:~2304) are CUDA-event gap timers: `fold()` charges
  the elapsed time between consecutive event marks to the earlier mark's phase.
- But marks are recorded on **multiple streams** (main compute, the copy stream, kv_copy, PLE).
  Gaps measured across stream boundaries double-count concurrent work, so `sum(phases)` exceeds
  the true GPU timeline.
- Evidence: sum(phases) = 37435 ms vs tl 32871 in baseline (over-count **+4564 ms**), and
  35187 vs 31829 in F-2 (over-count **+3358 ms**). The engine's own timing line prints
  percentages that sum to ~113% — it has always acknowledged this overlap.
- So: Δtl = Δsum − Δover-count = -2242 − (-1206) ≈ **-1042 ms** ✓. The discrepancy is fully
  accounted for; there is no missing time and no hidden slow phase.

**What it means for F-2's honest end-to-end verdict:** the gross hc-phase saving is 2448 ms, but
1.2 s of that was time the hc kernels spent overlapping other streams' work (dequant/PLE/copy
running concurrently). Removing overlapped work shortens the phase sums by more than it
shortens the wall clock — the overlapped portion was already partially hidden. The true
end-to-end win is the -1042 ms / -3.2% GPU timeline (and -1042 ms wall, both medians; wall and
timeline move together, host staging excluded from wall in both arms).

Caveats:
- The over-count is an artifact of cross-stream gap charging, not of F-2's change; its exact
  composition per phase-pair isn't recoverable from the summary lines (would need per-event
  dumps, i.e. STRATA_PREFILL_TIMING with a stream-tagged variant).
- The earlier note "gemm gate/up +106, gdn +146, qsa attn +55" was computed against the bad
  baseline medians; corrected deltas are gdn +83, gemm gate/up +46, qsa attn -6 — the
  secondary-phase regression story mostly evaporates once the baseline is corrected.
- The three extra F-2 passes (T=32042, tl 32082/32213/32312, 24-token generation) in the log
  are a supplementary round from the first engine session; consistent with the counted arm
  (hc read 3390-3395, up 762-769, inj 624-637, mix 623). Not included in the medians.

Final verdict (unchanged in direction, now fully reconciled): F-2 = **-3.2% end-to-end GPU
timeline at T=32005 med-of-3, bitwise-identical output, zero secondary regression** (the
earlier +307 ms secondary-phase penalty was an artifact of comparing against wrong baseline
medians). Gated behind STRATA_HC_MIX_H16=1; promotion decision unchanged.


### F-2 committed (2026-10-20)
- Branch `f-hc-skip-widen-f2` (off `f-gfx103x-sh-stream` @ 09819e4c), commit `392586db`
  "F-2: hc skip-widen (STRATA_HC_MIX_H16=1) ...".
- 6 files: include/strata/prefill/gemm.hpp, include/strata/prefill/kernels.hpp,
  src/prefill/gemm.cu, src/prefill/kernels.cu, src/prefill/prefill.cpp (+475/-17) and
  tests/hip/upmix_gemm_bench.cpp (new).
- Left uncommitted/untracked: bench results dirs, other probes (gated_probe.cu, mix_probe.cu,
  mix_h16_parity.cu), config baks. Not part of the F-2 change; housekeeping for later.
- Open decision: promote STRATA_HC_MIX_H16=1 to the production config default, or first bound the
  secondary-phase drift (gdn/gemm gate/up, ~+200 ms net, likely noise) with a bracketing re-run
  (F-2 -> baseline -> F-2).

---

## Open decisions & pending tasks (as of 2026-10-20)

### Decision needed: promote F-2 to default?
- **Question:** turn `STRATA_HC_MIX_H16=1` on in the production config (`strata-iq3_s-*.json` env)
**RESOLVED 2026-10-24:** promoted — first `0.1.47+F-2` (`f2-0147`), re-pointed to `main+F-2` (`f2-0148`) the same day, then superseded by `wt-0155` on 2026-11-02; see the 2026-10-23/24 and 2026-11-01 sections.

  now, or first bound the secondary-phase drift?
- **Evidence:** the F-2 server A/B + reconciliation above — med-of-3 −1042 ms / −3.2% GPU timeline (and wall), bitwise-identical output, no real secondary regression.
- **Option A (promote now):** flip the env in the production config; keep the flag in the engine so
  the stock default stays byte-identical for upstream.
- **Option B (bracket first):** one re-run F-2 → baseline → F-2 (same protocol) to bound drift and
  firm up the -3.2% before promoting. Costs one server session; recommended if the number will be
  quoted in a PR or community report.

### Pending tasks
- [x] ~~**Recover the gfx1030 performance lost in the 0.1.48.1 → new-main jump**~~ **DONE 2026-11-01** — parking-equalized med-of-6: F-2 wins at every tier, new main + F-2 promoted as `wt-0155` (lead list and caveats live in the 2026-10-29 entry).
- [ ] **PR organization** (see branch inventory below): decide the commit split for the F-2 PR —
  `f-hc-skip-widen-f2` carries `09819e4c` (WIP hc sub-phase timing groundwork, S23) as a parent.
  Either submit the timing groundwork as its own small PR first, or rebase F-2 onto
  `wip/strata-main-base` (477ac13c) to keep the PR single-topic.
- [ ] Draft PR-A/PR-B write-ups (docs/COMMUNITY_BENCHMARKS.md style: measured numbers + hardware).
- [ ] Housekeeping commit on a dedicated branch: untracked `tests/hip/gated_probe.cu`,
  `mix_probe.cu`, `mix_h16_parity.cu`, bench result dirs (`2026-10-13-*`, `2026-10-16-*`,
  `2026-10-20-*f2`), `strata-iq4xs-spec5.json.bak`, `/tmp/f2_*` scripts → migrate into
  `tests/hip/` + `bench/results/` or delete.
- [ ] pi provider update (`~/.pi/agent/models.json`): add the F-2 config as a model entry if F-2
  becomes the default lane.
- [ ] Quality harness run on the F-2 build (should be unnecessary given bitwise parity, but cheap
  insurance before promotion).
- [ ] `--soak` opt-in patch: make the thermal-soak protocol (≥3 discarded warmup passes) a flag
  instead of a manual procedure.
- [ ] Elementwise fusion of the hc read chain (S23 target) — only if needed; F-2 already removed
  most of the widen traffic. Re-evaluate after promotion.
- [ ] Second V620 arrival → `--gpus 0,1` layer-split test (carried over from Open items #7).
- [ ] Optional: post V620 numbers on PRs #1167 / #1151 (docs/COMMUNITY_BENCHMARKS.md protocol).
- [ ] Optional: delete the 110.9 GiB ollama blob `sha256-89059c88…` (needs sudo / `ollama rm`;
  shared by two tags, one still served).

## Branch inventory (for PR organization)

```
main                    296b59d2  local worktree of /home/drewster/git/Strata — mirror of upstream
                                  f-1441 merge; 264 behind origin/main (STALE, do not build from here)
f-gfx103x-sh-stream     09819e4c  WORKING INTEGRATION BRANCH — all gfx1030 work lands here;
                                  HEAD = WIP "prefill hc sub-phase timing (STRATA_PREFILL_TIMING)"
                                  (the S23 timing groundwork that F-2's parity/timing work built on)
f-hc-skip-widen-f2      392586db  F-2 FEATURE BRANCH (current) — skip-widen commit on top of
                                  09819e4c; 6 files (+475/-17) + tests/hip/upmix_gemm_bench.cpp.
                                  The 09819e4c parent is timing groundwork, NOT part of F-2 — split
                                  or rebase before opening the PR (see pending tasks)
pr-2226                 6ce12b4a  upstream PR #2226 experiment worktree (hc persist results,
                                  0.1.43.1 interleaved pairs) — upstream-side, unrelated to F-2
wip/strata-main-base    477ac13c  /tmp worktree mirror of upstream main (tracking base for rebases)
wip/strata-old          48d6aa11  pre-wave-reduce snapshot (2026-10-14 abandon point)
wip/strata-pr1167       db649832  upstream PR #1167 mirror (rocBLAS table docs)
wip/strata-prA          01373a42  upstream PR-A mirror (gfx103x shared-expert stream default)
wip/strata-prB          faf35480  upstream PR-B mirror (native DPP wave reductions)
```

**PR plan:** (1) small PR: hc sub-phase timing instrumentation (`09819e4c` rebased onto
`wip/strata-main-base`); (2) main PR: F-2 skip-widen (`392586db` rebased onto that). Keep the
`wip/*` mirrors untouched as upstream references; do not submit from them.

## 2026-10-22 — best-ever tier record: upstream main 011c9e60 (0.1.47), fully calibrated

Context: PR-A (gfx1030 SH_STREAM default) dropped — upstream #1992 landed per-PC
calibration instead (`tools/calibrate.py`, AMD step 6: fork measured off-vs-on, kept
only if median gain > 3%, all rounds positive). V620 calibrate result: 59.9/60.0 vs
65.2/65.4 tok/s = +8.8% median, fork kept in config env. Recommended block applied:
STRATA_SH_STREAM=1, --pcie-frac 0.00 (+7.4% vs 0.55), --spec-min-p 0.70 (+5.3% vs 0.5).
Standard-v1 community report rerun calibrated: short 66.9, 4K 890.2/56.9, 16K 984.2/57.7
(bench/results/2026-10-22-standard-pro-v620-azure-c46c142d/).

Tier pass (same harness/fixtures as all rows above; upstream main 011c9e60 build
/tmp/wt-e0, med-of-3 counted passes, pass 1 soak; configs
.bench-configs-staging/strata-iq3_s-v620-calibrated.json; results
~/git/llm-benchmark/results/strata_tier_pass_20261022_{051112..052553}.json):

| Run | 16K PP/TG | 32K PP/TG | 64K PP/TG | 128K PP/TG |
|---|---|---|---|---|
| 0.1.47 calibrated (2026-10-22) | 891.9/89.3 | 914.0/85.3 | 921.2/84.6 | 913.6/84.5 |
| best prior: matrix IQ3_S spec-5 (10-16) | 802.6/82.0 | 819.5/78.6 | 826.6/79.3 | 818.7/75.5 |
| delta | +11.1/+8.9% | +11.5/+8.5% | +11.5/+6.7% | +11.6/+11.9% |

Best cell in every column this machine has recorded. Attribution caveats:
(1) PP +11% is upstream 0.1.44.1 -> 0.1.47 code, not config (today's PP is identical
across today's own runs; yesterday E0 on same build matched 892/913/921/914).
(2) TG gain = calibration recipe (+~13% measured vs uncalibrated same-build runs
78.7/75.5/77.0/74.4) plus spec-min-p 0.5 -> 0.70 recipe change vs the 10-16 row.
(3) evening/continuous-load epoch (~3h hot card) vs cool-start rows; settled-state
penalty applies, so cool-start build-vs-build gap is slightly wider than +11%.
Also: 16K TG 86.0-91.0 non-overlapping vs yesterday's PR-A arm (80.2-85.2) —
calibrated main beats the dropped PR-A default at every tier; #1992's per-PC
calibration > fixed default, confirmed with numbers.

**Addendum (2026-11-03) — which utility is `--calibrate`, and is the recipe still in the config.** The `--calibrate` flag belongs to setup's per-PC tuning: the wrapper `./setup.sh --calibrate` (Linux) / `START-HERE.bat --calibrate` (Windows) calls `tools/calibrate.py`, whose docstring says "Tune the engine's hardware-dependent settings on this PC (setup's --calibrate)". That is the utility used for the 10-22 session above. It measures `--pcie-frac`, `--spec-min-p`, `--pool-workers`, the `--adapt-*` tier knobs, `--prefill-pipe` (layer split only) and `STRATA_SH_STREAM` (AMD only) — each kept only when it wins by more than 3%. Verified today, the 10-22 recipe survives intact in the daily config v2 (`strata-v620-iq3_s-T9-draftvocab.json`): `--pcie-frac 0.00`, `--spec-min-p 0.70`, `env STRATA_SH_STREAM="1"`; no `--pool-workers` or `--adapt-*` overrides were adopted for this box (V620 + TR PRO 3975WX 32-core). The Terminal offers a separate interface (`strata launch --tune quick|--tune thorough`); we have not run it. Optional low-priority follow-up: re-run calibration on 0.1.55+F-2 — the tuned axes are hardware-bound (PCIe, CPU cores, stream overlap) and F-2/CPU_SHARE/HC_PERSIST do not touch them, so the 10-22 recipe is expected to hold. NOTE for any future re-run: the pool-worker step restarts the model repeatedly and the PC is busy for a minute or two per restart; never run it while the daily server is live.

## 2026-10-23 — spec5 decode sweep, adoption: T9 (--mtp-draft-vocab)

Sweep of friend's box config (tmp.json) vs our calibrated config, V620 / IQ3_S,
med-of-3, byte-identical env, hot-card re-anchored baseline (906.5/987.1/968.1 PP).

Adopted — the only real winner:

    --mtp-draft-vocab /home/drewster/git/Strata-main/data/draft_vocab_en.bin

  chat decode   63.40 -> 66.70 tok/s  (+5.2%)
  agent decode  65.90 -> 70.40 tok/s  (+6.8%)
  PP 4K/16K/32K 906.5/987.1/968.1 -> 906.2/986.8/968.1  (identical; zero prefill cost)
  TTFT          unchanged

Smaller draft vocab -> higher MTP acceptance -> faster decode. Zero downside measured.

Rejected arms (same protocol, all vs hot base): --spec 4 (worse), STRATA_ADAPT_OVERLAP=1
(chat -7.6%), --pcie-frac 0.35 (PP -2.5%, TTFT +0.1s), adapt trio (chat -4%),
--vram-reserve-mib 200 (agent -11%), no --mtp-window (agent +3.2% but weaker than T9 and
wash elsewhere), --max-context 131072 (no effect), T7+T9 combo (worse than T9 alone:
chat -3.3%, cache re-read 73 tok/session vs 3 — do not stack).
Removed --ple-io ram: 4K PP -4.5% — keep it.

Thermal note: session showed ~2.5% settled-state PP drift (base cold 924.7 -> hot 906.5
at 4K). Most of the ~3% PP gap vs friend's box is thermal epoch, not config.

Durable config saved to /home/drewster/strata-v620-iq3_s-T9-draftvocab.json
(bench dirs: bench/results/ab-spec5-vs-spec4/{base_rerun,draftvocab,t7t9_combo,...}).

## 2026-10-23/24 — PR-A on upstream main: E-series A/B + tier pass + decode gate

PR-A (`pr-a-gfx103x`, gfx103x Linux shared-expert stream default) rebased onto current
upstream main `011c9e60` (v0.1.47) as `a14ba922`, forked/pushed to andrewprock/Strata.
Measured on V620 / IQ3_S, med-of-3, byte-identical E0 config (untuned).

**E-series short bench (E0 main vs E1 PR-A):**

| metric | main `011c9e60` | PR-A `a14ba922` | Δ |
|---|---|---|---|
| PP 4K | 923.3 (921.4–924.6) | 900.1 (895.8–901.4) | −2.5% solid |
| PP 16K | 1004.0 (999.3–1009.2) | 980.7 (979.3–987.0) | −2.3% solid |
| PP 32K | 974.3 (971.6–977.7) | 966.4 (964.5–967.4) | −0.8% overlap |
| agent decode | 58.5 | 63.9 | +9% weak |
| chat decode | 55.7 | 59.1 | +6% weak |

**Attribution flip vs D-series (PR-A on 0.1.46.3 showed +2% PP / −8% decode).** Both
signs flipped on the new base — the commit message's +8% decode claim was measured on
0.1.44.1 (2026-10-14) and does not carry.

**Decode-gated PR-A (E2, `24f3e1cd`):** fork now fires only when `n == 1` (decode);
prefill reverts to single stream. PP recovers to baseline (32K 972.2, chat TTFT 0.53 s),
but chat decode came back **53.2** — the gate loses the decode win too. Branch work,
uncommitted; NOT adopted.

**Tier pass, main vs v0.1.48.1 upstream drop (`7ddc17a3`) — unpinned-tier benchmark, not PR-A:**

| tier | main PP/TG | 0.1.48.1 PP/TG | verdict |
|---|---|---|---|
| 16k | 890.8 / 74.5 | 866.2 / 78.4 | PP −2.8% solid, TG +5.2% |
| 32k | 911.8 / 74.9 | 906.8 / 77.3 | PP −0.5% wash, TG +3.2% |
| 64k | 919.8 / 74.6 | 918.5 / 74.3 | wash |
| 128k | 911.8 / 74.7 | 911.9 / 74.4 | wash |

Upstream 0.1.47→0.1.48.1: prefill regression bounded to ≤16k, decode gain directionally
positive but never non-overlapping. Candidate upstream report; culprit bisect not run.

**Standing:** PR-A as pushed (plain default) is NOT submittable — PP regression is the
measured truth on its target base; decode gate variant loses the win. Commit message

**2026-10-24 — config audit corrections (found while preparing the F-2 promotion):**

1. **`STRATA_HC_MIX_H16=1` is set in the daily config and every staging config, but is INERT in every binary those configs launch.** Verified: `wt-e0` (0.1.47, `011c9e60`) and `wt-v023` (0.1.48.1, `7ddc17a3`) binaries contain zero `HC_MIX` strings (7 other `STRATA_HC_*` vars present), whole-tree source grep finds none, and `merge-base --is-ancestor b5f10fe2` fails for both bases. The gate exists only at `f2-on-main-2026-10-21:src/prefill/prefill.cpp:486`. The Oct-22 "Option A (promote now)" was executed env-side only; the engine side never reached the daily binary. The Oct-20 A/B (−1042 ms / −3.2 %, bitwise identical) remains the only valid F-2 evidence — it was a same-binary flag A/B on the branch build.
2. **The Oct-23 `hcA`/`hcB` staging configs (flag 0 vs 1) both point at `wt-e0`** — a binary that ignores the flag. Any numbers from those arms are noise, not F-2 evidence. Do not cite them.
3. **The Oct-23 0.1.48.1 tier comparison (`v0231` config) ran UNCALIBRATED**: it drops `--pcie-frac 0.00`, reverts `--spec-min-p` to 0.5, drops `STRATA_SH_STREAM` and T9 vs the calibrated config. The "TG +5.2 % at 16k" verdict is untuned-vs-untuned; a calibrated+T9 0.1.48.1 has never been measured. Why `STRATA_SH_STREAM` was dropped from `v0231` is unrecorded — check before the rerun.
4. **`cal-vram200.json` is byte-identical to `calibrated.json`** (args and env), so the "rejected `--vram-reserve-mib 200` (agent −11 %)" line cannot be sourced from that staged file — edited after the run or mislabeled. The daily config keeps `--vram-reserve-mib 200`.
5. **All other env vars in the daily config are honored by both binaries** (strings check): `PLE_BATCH`, `HIP_PROMPT_F16`, `SELECT_SIMT`, `SH_STREAM`, `ROCBLAS_TUNING`, `WATCHDOG_S`. `STRATA_PREFILL_TIMING=1` was removed from the DAILY config this session (diagnostics; its timers over-count cross-stream overlap — keep it in bench configs only). The inert `STRATA_HC_MIX_H16=1` stays in the daily env: it goes live the moment the exe points at an F-2 build.
6. **/tmp fragility**: the daily binary lives in `/tmp/wt-e0/build` (tmpfs — wiped on reboot), and the server log at `/tmp/stdbench-server.log` likewise. Backups made this session: `~/engines/backup/strata-0.1.47-base`, `~/engines/backup/strata-0.1.48.1-base`. New F-2 builds go in persistent `/home/drewster/git/wt-f2-0147` and `wt-f2-0148`.
7. **Clean F-2 builds prepared this session**: `f2-0147` = `011c9e60` + F-2 cherry-pick (`c5b648f0`), `f2-0148` = `7ddc17a3` + F-2 cherry-pick (`ad634880`). Cherry-pick resolution: both conflicts in `src/prefill/prefill.cpp` resolved to F-2's side with the two `pt.mark(kPfHcUp/kPfHcInj, cs)` sub-phase lines dropped (those marks come from the separate timing-groundwork commit `3d34786f`, which upstream lacks — upstream defines only `kPfHc`/`kPfHcWrite`). `git cherry` confirms all six commits on `f2-on-main-2026-10-21` are absent from 0.1.47, so building that branch as-is would have confounded the A/B with the SH-default flip and DPP wave reductions.
numbers are stale. Next: decode-focused med-of-5 A/B on `011c9e60` base, or park the PR.

**2026-10-24 — F-2 promoted (steps 0-3 executed).** Built the clean F-2-only builds prepared in the config-audit section above (`wt-f2-0147`, `wt-f2-0148`; both Release/gfx1030 with setup's flags — canonical list in Environment/infra notes — tests ON). Cherry-pick resolution as recorded there.

Parity (`STRATA_HC_UPMIX_CHECK=2` + `STRATA_HC_UPMIX=1`, 16k prompt, both checks): `mixed rel RMS 0.000e+00, max |diff| 0.000e+00 (f16-skip)` on BOTH builds — matches the Oct-20 verification. Note: the check block only runs when `hc_upmix()` is true; on gfx1030 `STRATA_HC_UPMIX` is NOT an arch default (it is for gfx1103 iGPU only), so a parity run without `STRATA_HC_UPMIX=1` silently checks nothing. With `=1`, `gr_upmix` returns false on gfx1030 (gfx11-only kernel) and the flow falls into F-2's f16-skip path — which is exactly the path the check labels. F-2 is active without `STRATA_HC_UPMIX` too; the env only gates the check.

Calibration on 0.1.48.1+F-2 (`tools/calibrate.py`, 686 s): `--pcie-frac` best 0.00 (65.7 med vs 61.0 at default 0.55), `--spec-min-p` best 0.70 (69.3 vs 63.6 at 0.5), workers keep default 30 (20/15 only +1.7%, under the 3% gate), adapt tier keep default, `STRATA_SH_STREAM` keep 1 (68.7 vs 62.6, +9.7%). Result: the Oct-22 recipe transfers to 0.1.48.1 unchanged — `v0231`'s omission of all three was drift.

Tier A/B, one thermal epoch, 1 soak + 3 counted, med-of-3 (`run_tiers.sh`; arm configs `strata-iq3_s-ctl-e0.json` / `strata-iq3_s-f2-0147.json` / `strata-iq3_s-f2-0148.json`; all with T9, `STRATA_PREFILL_TIMING=1`, calibration recipe):

| tier | ctl (0.1.47) | f2m (0.1.47+F-2) | f2n (0.1.48.1+F-2) |
|---|---|---|---|
| 16k | 894.1 / 90.9 | 909.1 / 92.4 | 872.8 / 92.5 |
| 32k | 915.7 / 88.7 | 930.6 / 88.3 | 927.1 / 88.5 |
| 64k | 923.0 / 89.4 | 938.9 / 89.6 | 936.0 / 87.5 |
| 128k | 916.0 / 85.5 | 931.7 / 85.4 | 929.6 / 84.8 |

Deltas vs ctl: **f2m PP +1.7 % at every tier** (16k +1.7, 32k +1.6, 64k +1.7, 128k +1.7), TG wash. f2n PP +1.2-1.5 % at 32k+ but −2.4 % at 16k (the 0.1.48.1 ≤16k prefill regression persists even calibrated), TG wash. Control reproduced the Oct-22 baseline within noise (128k 916.0/85.5 vs recorded 913.6/84.5). The Oct-23 "TG +5.2 %" for 0.1.48.1 was an uncalibrated artifact: 0.1.48.1 ≈ 0.1.47 once calibrated.

**Promoted: the daily driver is now `0.1.47+F-2`.** `~/strata-v620-iq3_s-T9-draftvocab.json` exe → `/home/drewster/git/wt-f2-0147/build/strata`; `STRATA_HC_MIX_H16=1` is live in it for the first time. Verified: engine cmdline serves from the new binary, short prompt works, KV cache hits. The old binary is backed up at `~/engines/backup/strata-0.1.47-base`. The daily config has `STRATA_PREFILL_TIMING` removed (kept in bench arms to match prior baselines).

Not done yet: the 0.1.48.1 prefill-regression culprit bisect is now low priority (calibrated gap is 16k-only, −2.4 %, and 0.1.47+F-2 wins overall). Remaining from the plan: lane re-rank under the F-2 recipe, qsa-select 128k work, PR split (`f2-0147` is now exactly the single-topic PR branch: F-2 alone on `011c9e60` — rebase/submit per CONTRIBUTING), `--soak` flag, second-V620 trigger list.

**2026-10-24, later — daily driver re-pointed to main+F-2 (`ad634880`).** Decision: keep deployment on a single line of development that is exactly "upstream main + F-2" (`f2-0148`, `/home/drewster/git/wt-f2-0148/build/strata`) instead of the diverged 0.1.47+F-2 line (`f2-0147`, 131 commits behind main). Measured cost accepted, from the same-epoch tier A/B above: 16k PP −2.4 % vs ctl (−3.8 % vs f2m), PP +1.2–1.5 % at 32k/64k/128k, TG wash. Rationale: no divergence rot, PR and deployment are the same artifact (`f2-0148` is main + exactly one commit, `git cherry` clean). `strata-iq3_s-f2-0147.json` kept as the opt-in fallback config; the binary and worktree remain in place. Verified live before stopping: engine cmdline served from the f2-0148 build with the full calibrated recipe. Hygiene pending: local `main` branch still parked at `011c9e60` (0.1.47) — reset it to `origin/main` when convenient to match the new topology.

**2026-10-25 — the "16k regression" solved: it was the conversation cache, not code.** Full arc of the session:

1. *The apparent regression.* After promoting main+F-2 (`ad634880`), the counterbalanced 4-tier A/B (alternating arm order, med-of-2) showed 0.1.48.1+F-2 at 16k PP 880.7/882.3 vs 0.1.47+F-2 at 914.2/916.8 — −3.8 % at 16k, parity above. A git bisect was attempted; it died of noise (verdicts split on byte-identical source; ±1.3 % single-boot noise exceeds the effect) and was abandoned. Methodology: bisects at this effect size need interleaved candidate+reference pairs.
2. *A false lead was killed first.* The bisect's first "bad" binaries were traced to a build defect: the F-2 cherry-pick was built incrementally over the stale `wt-v023` build directory → mixed objects → slow prefill. Clean rebuild of the same commit recovered the deficit (paired −1.1 %, and a second clean build of the same source paired at +0.35 % = parity). Rule (second occurrence of the "missing 800 PP" class): **after any flag/base change, `rm -rf build` and rebuild clean**; never compare binaries from builds with different histories.
3. *Timing attribution.* With `STRATA_PREFILL_TIMING=1`, the 0.1.48.1 GPU prefill timeline at 16k is FASTER than 0.1.47's (17.2 vs 17.7 s), yet bench PP is slower → the deficit is ~0.8–1.2 s per request outside the prefill wall.
4. *Position probe.* Reversed tier order (128k→16k): the 16k deficit vanished (910.5 vs 919.2 ref, −0.9 %). The penalty tracks "the previous request was 128k", not "16k is slow".
5. *Root cause.* `f0010a60` (in `011c9e60..7ddc17a3`) changed the conversation cache default from off to sized-from-free-RAM (8192 MiB, up to 4 parked conversations). 0.1.48.1 parks each finished conversation's KV snapshot (log: `parked N tokens in ~210/343/601/1022 ms` after 16/32/64/128k, snapshot ≈ 2.9 GB after 128k); the next request with a different prompt must evict it before prefill → ~1 s. 0.1.47 never parks (no parking lines in its log) and its prompt path borrows 2303 expert slots (4.38 GiB) up front (0.1.48.1: 2300; expert fill 12467 vs 12394 slots — 0.1.48.1 reserves ~0.14 GiB less). Both engines run at the VRAM edge (82–224 MiB free warnings).
6. *Fix.* `--conversation-cache-mib 0` disables parking. Tier A/B with it, normal order: 16k 923.2 (beats ref 915.5), 32k 944.9, 64k 953.8, 128k 943.8; TG 92.2/91.5/88.7/86.2 — 0.1.48.1+F-2 wins every tier with parking off.
7. *Agent-loop verdict (the trade).* 5-turn loop on one ~25.9k-token conversation, parking on vs off: turns 2–4 run ~1.8–2.4 s in BOTH (reused ≈ 25.9k) — live-state prefix matching (`starts_with(live)`, `--kv-resident 32768`) already resumes a continuous conversation without the conversation cache. Parking matters only when *switching* conversations: ON = ~1 s eviction then fast restore of a parked session; OFF = full re-prefill of the other session (~18 s at 16k, ~100 s at 128k).

**Decision: keep parking ON (0.1.48.1 default) and keep `main`+F-2 (`ad634880`, `wt-f2-0148/build/strata`) as the daily driver — no flag added.** Rationale: single-conversation sessions are identical with or without it; session switches are 1 s (evict) vs 18–100 s (re-prefill), which dominates real agent usage. For batch/one-shot runs where conversations never repeat, flip `--conversation-cache-mib 0` for that run (pure win there). The tier bench's 16k column with parking on is a bench artifact (conversations never revisit) — note it when comparing future runs.

Upstream-worthy: (a) the eviction cost after a big conversation could be made async/overlapped with the next request's prefill; (b) the VRAM-edge state (82–224 MiB free) documented in the timing-sweep logs.

Session evidence: logs in `~/engines/logs/` (probe-rev, probe-nopark, agent-loop), scripts `/tmp/probe-rev.sh`, `/tmp/probe-nopark.sh`, `/tmp/agent-loop.py`, `/tmp/agent-loop-test.sh` (recreate in `bench/` if needed), JSONs in `~/git/llm-benchmark/results/`.

**2026-10-29 — new-main A/B (arms a/b), IQ3_S lane: old+F-2 stays the daily driver.** Fresh `origin/main` pulled (HEAD `46c152a1`, 1071 commits past the daily driver's 0.1.48.1 base). Two clean HIP builds with setup's flags (canonical list + `STRATA_ENABLE_HIP` gotcha in Environment/infra notes), fresh build dirs (clean-build rule respected): (a) new main as-is (`/tmp/strata-arm-a/build-hip/strata`), (b) new main + F-2 (`ad634880` cherry-picked clean onto `46c152a1` → `fd7a356e`, 6 files, `/tmp/strata-arm-b/build-hip/strata`). Both built in ~5 min. Lane: `packs/iq3_s` spec-5 draftvocab recipe, server configs `/tmp/iq3s-arm-{a,b}.json` (copies of the daily `strata-v620-iq3_s-T9-draftvocab.json`; only exe + model_name + log changed), 12,467 expert slots, bench = 1 soak (discarded) + 3 counted, med-of-3. Result JSONs: arm (a) `strata_tier_pass_20261029_{035708,040214,040723}.json`, arm (b) `…_{042009,042507,043005}.json`.

| Tier | (a) new main PP/TG | (b) new main + F-2 PP/TG | F-2 Δ | 10-25 baseline (0.1.48.1+F-2) PP/TG |
|---|---|---|---|---|
| 16k | 902.5 / 92.3 | 912.3 / 92.3 | +1.1 % / 0.0 % | 923.2 / 92.2 |
| 32k | 943.5 / 92.1 | 957.5 / 88.1 | +1.5 % / −4.3 % | 944.9 / 91.5 |
| 64k | 946.8 / 90.1 | 962.1 / 85.6 | +1.6 % / −5.0 % | 953.8 / 88.7 |
| 128k | 933.5 / 86.9 | 949.0 / 86.0 | +1.7 % / −1.0 % | 943.8 / 86.2 |

vs baseline: (a) PP −2.2/−0.1/−0.7/−1.1 %, TG +0.1/+0.7/+1.6/+0.8 %. (b) PP −1.2/+1.3/+0.9/+0.5 %, TG +0.1/−3.7/−3.5/−0.2 %.

Findings:

1. **F-2's TG cost is new on new main.** On the old base (10-24 A/B, f2m vs ctl) the same commit was a TG wash; here it costs −4.3/−5.0 % TG at 32k/64k while still giving +1.5/+1.6 % PP. Either main changed something F-2 now perturbs, or main's decode path moved under it.
2. **Arm (a) alone:** PP −0.1 to −2.2 % everywhere, TG +0.1 to +1.6 % everywhere vs the baseline row — decode-leaning, prefill-regressed.
3. **Arm (b):** PP wins at 32k/64k/128k (+0.5 to +1.3 %) but loses 16k PP (−1.2 %) and mid-tier TG (−3.5/−3.7 %).
4. **Neither arm dominates the old build → Decision: keep old+F-2 (`wt-f2-0148/build/strata`, 0.1.48.1+F-2) as the daily driver for now.** Both arm binaries stay servable from `/tmp/strata-arm-{a,b}/build-hip/strata` for the follow-up runs (worktrees are in `/tmp` — recreate from `origin/main` if the box reboots).

Comparability caveat: the arms ran the daily config as copied — conversation parking ON (no `--conversation-cache-mib 0` on the cmdline) — while the quoted 10-25 baseline row was measured parking-OFF. Per the 10-25 findings, the eviction after each tier's single request costs ~0.2–1.0 s (≈0.2–1.1 % of PP per tier, worst at 16k) and is a bench artifact (the bench never revisits a conversation). The true 16k code delta is therefore likely closer to parity than the raw −2.2/−1.2 %; TG columns are unaffected. A parking-equalized re-run is step 1 of the recovery TODO.

New TODO (user, 2026-10-29): **find out where the performance went and see if we can recover it for gfx1030.** Lead list, cheapest first: (1) parking-equalized re-run of arms (a)/(b); (2) `STRATA_PREFILL_TIMING=1` + decode-side attribution of the mid-tier TG gap (the bench reported `drafts None/None` on both arms — check whether these builds report MTP acceptance); (3) only if the gap survives: interleaved candidate+reference pairs across `011c9e60..46c152a1` (single-boot bisects die of noise at this effect size — established 10-25); (4) if the TG gap is F-2×new-main interaction, re-run the F-2 A/B on top of whatever recovery lands before opening the F-2 PR.

Side note (unconfirmed, not tested): new main carries upstream k-quant expert-kernel options (`STRATA_MMQ_KQUANTS`, `STRATA_Q6K_EXPERTS`) and docs add UD-Q5_K_XL/UD-Q6_K_XL quality tables (KL 0.033–0.038 vs IQ4_XS 0.101). Both arm builds had those options OFF (setup's IQ-lane flags), so whether the k-quant expert path actually serves on gfx1030 is unknown — a possible future quality ceiling above IQ4_XS if it does.

## 2026-11-01 — new-main round 2 (0.1.55): F-2 wins on med-of-6, promoted to `wt-0155`

Setup. Target `c62d5e3a` (v0.2.7.1-239, 0.1.55-era) pulled to `/tmp/strata-arm-{c,d}` from `origin/main` (the CWD worktree `/home/drewster/git/Strata-main` is parked at `c62d5e3a` via `parked/pr-a-gfx103x-default`). Two clean HIP builds, setup flags (canonical list in Environment/infra notes), clean build dirs (10-25 rule). Arm-c = new main as-is; arm-d = new main + F-2 (`ad634880` re-cherry-picked → `4bb6990e`; same prefill.cpp conflict as the 10-29 pick, resolved the same way: F-2's `if (!upmixed)` block with upstream's `pt.mark(kPfHcInj, cs)` after it, before the `if (checked)` debug block; `git cherry` clean).

**The 10-31 "server.py spawns a broken engine" mystery was my own bug.** Symptom had been: engine boots in seconds with no model loaded (`READY 32768`, no engine log, dies on first request). Actual cause: the server.py invocations omitted `--engine strata`, so 0.1.55's server.py starts a bare, argument-less engine. With the flag the boot is correct: full calibrated args, engine log written, rocBLAS tuning table loaded (`gfx1030-rocblas-5.6.0.8d1ae90e.txt`, absolute path already in the config env), `READY 262144`, **12,333 expert slots** (the cfa197e2 auto-cache shrink from 12,467, as predicted on 10-31). rocBLAS resolution: server.py does not touch the env value; the loader (`src/prefill/rocblas_tuning.hpp`) opens the env path verbatim — the daily config carries an absolute path, so arm configs (cloned) are fine. Boot ~100+ s (VRAM wait), engine=0.1.55.

Protocol. Same lane as 10-29 (IQ3_S, spec-5 draftvocab, batch 1). **Parking-OFF in both arm configs** (`--conversation-cache-mib 0`) — this closes the 10-29 comparability caveat and matches the 10-25 baseline row. **Interleaved A/B: 3 cycles × per boot (1 soak discarded + 2 counted) = 6 counted passes per arm, med-of-6.** Bench `~/git/llm-benchmark/bench_strata_tiers.py`, TIERS=16k,32k,64k,128k, result JSONs `strata_tier_pass_20261101_*.json`.

Results (med-of-6, PP/TG tok/s):

| Tier | arm-c (new main) | arm-d (new main + F-2) | F-2 Δ PP | F-2 Δ TG | 10-25 baseline (0.1.48.1+F-2) |
|---|---|---|---|---|---|
| 16k | 910.9 / 92.4 | 927.3 / 90.3 | +1.8 % | −2.3 % | 923.2 / 92.2 |
| 32k | 931.9 / 90.6 | 948.5 / 90.0 | +1.8 % | −0.7 % | 944.9 / 91.5 |
| 64k | 940.1 / 88.5 | 957.2 / 86.4 | +1.8 % | −2.4 % | 953.8 / 88.7 |
| 128k | 931.1 / 86.4 | 947.7 / 86.2 | +1.8 % | −0.2 % | 943.8 / 86.2 |

All six arm-d passes beat all six arm-c passes at every tier (6/6 sweep) — no overlap at any tier. Arm-c vs baseline: PP −1.3/−1.4/−1.4/−1.4 %, TG +0.2/−1.0/−0.2/+0.2 %. Arm-d vs baseline: PP +0.4 % at every tier, TG −2.1/−1.6/−2.6/0.0 %.

Findings and decision:

1. **F-2 wins on new main.** PP +1.8 % at every tier with a TG cost ≤2.4 % (max at 64k). The 10-29 finding ("F-2's TG cost is new on new main", −4.3/−5.0 % at 32k/64k) does not reproduce with parking equalized — the 10-29 TG gap was measured parking-ON in both arms, and the 10-25 findings say the eviction cost lands on the PP side of the *next* request, so the 10-29 TG deltas now look like a small-sample artifact (med-of-3, unbalanced soak).
2. **New main alone (arm-c) is PP-regressed ~1.4 % vs the 10-25 baseline at every tier** with TG roughly at baseline. F-2 more than recovers it: arm-d beats the old daily driver's PP everywhere while TG stays within ~2.6 %.
3. **Decision: promote `c62d5e3a` + F-2 as the daily driver.** `~/git/wt-0155` created from `c62d5e3a`, F-2 cherry-picked (same conflict resolution as arm-d, verified byte-identical in `src/prefill/prefill.cpp`), commit `ff9d98b0`, tree clean. Rebuild and live re-point of the daily config pending (build was interrupted mid-compile; cmake configured OK, `build-hip/strata` not yet produced).
4. Old arms a/b remain servable from `/tmp/strata-arm-{a,b}`; c/d from `/tmp/strata-arm-{c,d}` (all `/tmp` — recreate from `origin/main` on reboot). The 10-29 daily driver (`wt-f2-0148/build/strata`, 0.1.48.1+F-2) stays on disk as fallback until the 0.1.55 build is verified live.

Promotion DONE (2026-11-02): build `wt-0155/build-hip/strata` 0.1.55 verified (commit ff9d98b0, built 06:17 after the 01:32 commit) → engine boot check (READY 262144, 12,333 slots, engine log written, rocBLAS tuning loaded, ~30 s boot) → daily config `strata-v620-iq3_s-T9-draftvocab.json` exe re-pointed to `/home/drewster/git/wt-0155/build-hip/strata` (old exe preserved in `/tmp/strata-daily-config-backup-pre-0155.json`) → short-prompt smoke passed (real model reasoning, not the mock) → 0.1.48.1+F-2 binary backed up to `~/engines/backup/strata-0.1.48.1-f2`. Child env confirmed: `STRATA_HC_MIX_H16=1` (F-2 active), `STRATA_ROCBLAS_TUNING` absolute, HIP_PROMPT_F16/SIMT/PLE_BATCH/SH_STREAM all set. `learned_profile_args` inert (no `expert_profile_save` key) as before.
