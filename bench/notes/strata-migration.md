# Strata on the V620 — Working Record

What this is: the working notes for running Qwen3.8-Flash-Next locally on one AMD Radeon Pro V620,
covering the move from Ollama to Strata (October 2026) and every speed experiment since.
Started 2026-10-12. Rewritten for readability 2026-11-17.

**Where the fine detail went.** This file is the day-to-day reference: what is running, how to run a
benchmark, what was decided and how to redo any experiment. The fine detail (full logs, per-phase
timing tables, theories that went nowhere) lives in git history:
`git show c8d3f3b2:bench/notes/strata-migration.md` (the pre-cleanup original, 1,282 lines), plus the
benchmark result files in `~/git/llm-benchmark/results/` (named per experiment below).

---

## 1. Words used in this document

**Speed words**

- **Reading speed (PP)** — how fast the model reads a prompt, in tokens/second. The reliable number:
  varies only ±0.3% between identical runs. Older shorthand: "PP" (prompt processing).
- **Writing speed (TG)** — how fast the model writes its answer, in tokens/second. Noisy: identical
  runs vary ±3–4%, and even day to day on an unchanged build (seen: 92.6 one day, 87.9 the next).
  Older shorthand: "TG" (token generation). Never trust a single-run writing-speed comparison.
- **Tier** — a tested prompt length: 16k / 32k / 64k / 128k tokens. A benchmark "pass" runs all four.
- **med-of-3** — the middle value of 3 runs. One warm-up run (the "soak") is always discarded first,
  because a cold machine reads ~2–5% faster than a warm one.
- **A/B** — two configurations alternated and compared, ideally interleaved (boot A, boot B, boot A, …)
  so machine drift hits both equally.

**Model words**

- **Quantization level (IQ3_S, IQ2_XS, IQ1_M, Q4_K_M…)** — how strongly the model's weights were
  compressed. Roughly "bits per weight": Q4 ≈ 4-bit, IQ3_S ≈ 3.1-bit, IQ2_XS ≈ 2.5-bit, IQ1_M ≈ 1.6-bit.
  Lower = smaller and faster to read, but less accurate. **UD-** means Unsloth's "Dynamic" brand of
  mixed-precision compression.
- **Daily driver** — the setup used for real work (as opposed to test lanes).
- **Generation 1–4** — the four daily-driver setups over time, in order. (Older notes called them
  v1–v4.) All four are defined with dates and contents in §2.4.
- **MTP / draft head / 5-token drafting** — the model ships with a small extra head that guesses the
  next 5 tokens while the main model writes one; correct guesses are free speed. Strata calls the
  feature "spec" (speculative decoding); the knob `--spec 5` turns on 5-token drafting.

**Hardware and build words**

- **V620 / gfx1030 / RDNA2** — the GPU card; gfx1030 is its chip family code. **ROCm** is AMD's
  equivalent of CUDA. **HIP** is AMD's CUDA-like kernel language.
- **F-2** — our best engine change (2026-10-20): skip a float16-widening step inside the attention-mix
  path. Output is bit-identical, work is 3% less. Enabled with env flag `STRATA_HC_MIX_H16=1`.
- **T9 recipe** — the draft-vocabulary trick (2026-10-23): point the 5-token draft head at a smaller,
  English-only vocabulary file. Flag `--mtp-draft-vocab …/draft_vocab_en.bin`.
- **PLE** — "prompt-time expert batching" (env flag `STRATA_PLE_BATCH=1`): process the expert matrices
  in batches while reading a prompt instead of one token at a time. Flipped on 2026-10-19; worth
  +12–14% reading speed by itself. **All speed rows before that date were measured with the slow
  batching** — see the caveat above the table in §5.2.
- **MMQ patches** — our patches (2026-11-04) to the expert matrix-multiply kernels so they use AMD's
  packed-FP16 instructions (VOP3P). They live on a branch of the bundled llama.cpp:
  `v620-mmq-vop3p` on the GitHub fork `andrewprock/llama.cpp`.
- **CPU offload / persistent hc** — the two env flags added 2026-11-03: `STRATA_PREFILL_CPU_SHARE=1`
  (the CPU takes a share of prompt-processing work, helping short prompts) and `STRATA_HC_PERSIST=1`
  (keeps per-conversation computation state resident, cutting time-to-first-token ~12% on short prompts).
- **Parking** — keeping the conversation cache (KV) resident in VRAM between requests (8192 MiB,
  flag `--conversation-cache-mib`). ON for real use (switching conversations costs 1 s instead of
  18–100 s of re-reading), OFF for benchmarks (a benchmark never revisits a conversation, so parking
  would only charge it a small eviction tax and muddy the comparison).
- **Worktree (`wt-stack`, `wt-0155`, `wt-mmq`)** — a second checkout of the same git repo in another
  directory, so each build keeps its own folder. `wt-stack` is where the current daily build lives.
- **Fork** — our copy of a repo on GitHub (`andrewprock/Strata`, `andrewprock/llama.cpp`) where
  branches are pushed for safekeeping.
- **E483 rule** — if a setting changes *which* computation paths run, measure it on real text, never
  synthetic prompts (a synthetic-positive/real-negative lesson from ticket E483).

---

## 2. What is running right now (as of 2026-11-17)

### 2.1 The computer

- Azure NVv5-class virtual machine: Threadripper PRO 3975WX (32 cores), **one V620 with 32 GB VRAM**
  (gfx1030/RDNA2, 40 compute units), 207 GB RAM. ROCm 10.0 at `/opt/rocm`.
- Models live on a 984 GB disk mounted at `/var/lib/ollama` (~215 GB free).
- Power is deliberately capped at ~180 W (of the card's 265 W spec). Under load the chip settles at
  ~85 °C; clocks wander 1520–2470 MHz. `rocm-smi --setperfdeterminism` is not supported on this card.
- The OS memory-lock limit is 8 MiB, so the engine's big table mlock always fails — harmless but noisy;
  fixing it needs sudo (still pending, §8).

### 2.2 What is being served at this moment

- **The Unsloth 1-bit lane (UD-IQ1_M)** — a 1.6-bit compressed copy of Qwen3.8-Flash-Next, 68 GB on
  disk — serving on port 8080 since 2026-11-09 for a real-work trial. Model id
  `qwen3.8-flash-next-ud-iq1m`. Config file `/tmp/ud-iq1m-serve.json` (same knobs as the daily config,
  pointed at the 1-bit weights). It is the fastest lane ever measured on this box (§5.3); the accepted
  trade is accuracy (Unsloth's own ladder says ~79.7% of full-precision top-1 answers recovered).
- The wrapper `~/bin/strata-serve` manages it: `stop`, `logs`, `status`, or bare to run in foreground.
  Logs: `~/engines/logs/daily.log`, plus the per-config `"log"` file from the config's `log` key.

### 2.3 How to get back to the usual daily setup

- The daily config file is untouched: `~/strata-v620-iq3_s-T9-draftvocab.json` (the 3.1-bit IQ3_S
  weights with the draft-vocabulary recipe). Backups of older versions sit next to it:
  `.bak-pre-v2`, `.bak-pre-mmq`, `.bak-pre-v4` (those suffixes mean "before generation 2/3/4").
- Restoring = stop the current server and start one with that config instead (§3.4 has the exact command).
- Older binaries for instant fallback: `~/engines/backup/strata-0.1.48.1-f2` (generation 1) and
  `~/engines/backup/strata-0.1.47-base`.

### 2.4 The four generations of the daily driver (all defined)

| Name | Date | What it actually is | Where it lives |
|---|---|---|---|
| **Generation 1** | 2026-10-25 | Strata 0.1.48.1 + the F-2 change (skip-widen). The reference build for all A/Bs since. | binary `~/engines/backup/strata-0.1.48.1-f2` |
| **Generation 2** | 2026-11-03 | Newer base: Strata 0.1.55 + F-2 (promoted 11-02 after a two-round A/B), **plus two env flags** added to the config: CPU offload and persistent hc. | worktree `~/git/wt-0155`, commit `ff9d98b0`; the env flags live in the config file |
| **Generation 3** | 2026-11-04 | Generation-2 base **+ our patched expert kernels** (the MMQ/VOP3P work on the bundled llama.cpp). | worktree `~/git/wt-mmq`, branch `v620-daily-0155-f2` (on the fork) |
| **Generation 4 — current** | 2026-11-05 | **Strata 0.2.10** + F-2 (clean cherry-pick, commit `434dfe91`) + the patched expert kernels. | worktree `~/git/wt-stack`, branch `v620-stack-0210-f2-mmq` (on the fork) |

The patched-kernels part: 3 commits on the engine's exact llama.cpp pin (`3cf03257f`) — packed-FP16
dot chains for the q8_0 and IQ2-family kernels, plus a helper (`56ccab041`). The engine builds them in
via `FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP` pointing at that checkout (0.1.55 has no `STRATA_GGML_DIR`
option; 0.2.10 pins the same llama.cpp commit at CMakeLists line 2110, so the patches drop in unchanged).

### 2.5 Models on the disk (as of 11-17)

- **`IQ3_S/` (78 GB, 2 shards)** — the daily-driver weights (3.1-bit, "GSQ-RCO" build). These are the
  live daily shards: never delete while an engine is running.
- **`unsloth-iq1m/` (68 GB), `unsloth-iq2xs/` (78 GB)** — the two Unsloth lanes benched 2026-11-06;
  parked (not deleted). Packs for them: `~/git/Strata-data/packs/{iq1m,iq2xs}`; configs
  `/tmp/ud-iq{1m,2xs}.json`.
- **`IQ3_S-abl/` (80 GB)** — the abliterated (refusal-removed) copy of the daily weights; pack at
  `packs/iq3_s-abl`. **`maya-models/` (238 GB)** — user-managed, not Strata's. **`mtp-draft/` (2.8 GB)** —
  the draft head.
- The Ollama service (systemd) is separate software with its own 9 models (~162 GB); the user manages
  it (`ollama rm` is the clean way to remove those).
- Deleted lanes (kept in history): DeepSeek-V4.1-Flash at 2.5-bit (275 GB, deleted 11-06 at the user's
  request), the Unsloth 4-bit lane, the abliterated-v2 variant, IQ3_XXS, the coder 1-bit, and the whole
  Q4_K_XL/Q5/Q6/Q8 family (k-quant experts are a dead end on this engine — §4.2, §6.13).

---

## 3. Standing procedures (how we do things here)

### 3.1 Setting up a benchmark run

1. **Build clean, with the right flags.** The flags (setup's defaults — a hand-built binary missing
   them silently loses ~25% reading speed):
   `-DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_Q6K_EXPERTS=OFF
   -DCMAKE_HIP_ARCHITECTURES=gfx1030` plus Release mode. In words: AMD backend on, the new
   prompt-time matrix kernels on, k-quant expert kernels off (they don't exist for GPUs), gfx1030 target.
   Gotchas: the option is spelled `STRATA_ENABLE_HIP` (not `STRATA_HIP`) and `CMAKE_HIP_ARCHITECTURES`
   (not `HIP_ARCHITECTURES`). **After changing any flag or base commit, delete `build/` and rebuild
   from scratch** — incremental rebuilds over stale objects produced two false "regressions" (10-18, 10-25).
2. **Soak, then count.** One warm-up pass after every engine restart, discarded. Then 3 counted passes,
   take the middle value. For A/Bs: interleave boots when possible and reverse the order the second
   day-round; same day, same temperature state. Never compare numbers from different sessions without
   equalizing the warm-up.
3. **Parking off for benchmarks** (`--conversation-cache-mib 0`), parking on for real use — why: §1.
4. **The harness.** `~/git/llm-benchmark/bench_strata_tiers.py` — has NO command-line options; it is
   configured by environment variables: `STRATA_BASE` (server address), `STRATA_MODEL`, `STRATA_API_KEY`,
   `TIERS=16k,32k,64k,128k`, `NUM_CTX=135168`, `NUM_PREDICT=1024`, temperature 0. Output is auto-named
   `strata_tier_pass_<timestamp>.json` into `results/`. Note: draft-head acceptance is NOT exposed via
   `/v1/status` — read the `drafts accepted X of Y` line from the serve log instead.
5. **Starting the server.** From the worktree directory:
   `setsid nohup .venv/bin/python serve/server.py --engine strata --config <json> --port 8080 … & disown`.
   `--engine strata` is MANDATORY on 0.1.55+ — omitting it starts a bare engine without the server
   layer (cost us a day on 10-31). Readiness check: `grep ^ready:` in the log. Server logs are
   append-mode: to inspect only a fresh boot, record a byte offset first.

### 3.2 Rules about measuring (each one earned the hard way)

- **Noise floors.** Warm reading speed ±0.3%; writing speed ±3–4% within a day and worse across days.
  **Treat any single-pass writing-speed delta inside ±4% as noise.** The writing pass also varies in
  length (388–619 tokens per pass) because the model sometimes stops early.
- **Real text, not synthetic (the E483 rule).** For any switch that changes which computation paths
  run: synthetic prompts gave +2.8% while real text gave −1.6%/+9.9% for the same flag. Synthetic
  results are a hint, never a verdict.
- **Temperature epochs.** A cool machine reads ~2–3% faster than a settled one. Same-day A/Bs: compare
  first-pass to first-pass, or re-soak between arms (the 10-19 evening sweep drifted −0.3% to −4.1% across tiers within one session).
- **Phase timers over-count.** `STRATA_PREFILL_TIMING=1` reports sums ~113% of real time because it
  double-counts work running on parallel streams. Fine for "where did time go", never for totals.
  Keep it in bench configs, out of the daily config.
- **Trust the binary, not the plan.** Before quoting a flag's effect, confirm the binary contains the
  gate (`strings <binary> | grep <FLAG>`) — on 10-24 an env var was set against an older binary and the
  flag was silently inert.
- **`pkill -f` self-matches.** It kills the calling shell too. Use bracketed patterns (`server[.]py`)
  or `pkill -x`, and never put the pkill and the restart in the same shell command (the restart gets
  SIGTERM'd; this bit us twice).

### 3.3 Checking that a change didn't break anything

- **Byte-identical smoke.** With temperature 0, a fixed 96-token prompt must produce identical output
  vs the incumbent build. Or: the in-server parity check (`STRATA_HC_UPMIX_CHECK=2` **together with**
  `STRATA_HC_UPMIX=1` — without the second variable the check silently runs nothing on gfx1030).
- **Recall harness.** `~/git/llm-benchmark/halo_needle.py` — 15 needle questions in long context;
  a good change scores 15/15 (§6.12).

### 3.4 Stop / start of the daily server (exact commands)

Stop: `~/bin/strata-serve stop` (or `pkill -x strata`). Start from the build's worktree, e.g.
`cd ~/git/wt-stack && setsid nohup .venv/bin/python serve/server.py --engine strata --config
~/strata-v620-iq3_s-T9-draftvocab.json --port 8080 > ~/engines/logs/daily.log 2>&1 < /dev/null & disown`.
Wait for `^ready:` in the log before sending traffic.

---

## 4. What we adopted, what we rejected (one line each)

Numbers are med-of-3 tier results. Full detail: the matching entry in §6, or git history.

### 4.1 Adopted — all of these are in the current daily setup

| What | When | Result | Where it lives now |
|---|---|---|---|
| ROCm tuning table (upstream PR #1167) | 10-14 | Reading +1.2–2.7% on every tier, writing unchanged | env `STRATA_ROCBLAS_TUNING` (absolute path) + `lib_dirs: [/opt/rocm/lib]` in config |
| SIMT1 scorer (upstream PR #1151) | 10-14 | 64k +2.7%, 128k +8.6% reading speed (bigger than the author's +3.2% — single card, no split) | env `STRATA_SELECT_SIMT=1` |
| DPP wave-reduce (a small kernel fix) | 10-14 | Reading +0.3–1.0%, bit-identical output; now upstream | in the tree (commit `48d6aa11` lineage) |
| Batched expert reading (PLE) | 10-19 | Reading +11.6–14.0% every tier, writing +2.3–6% | env `STRATA_PLE_BATCH=1` (see the caveat above the §5.2 table) |
| Calibration recipe (upstream PR #1992) | 10-22 | Writing +~13% vs uncalibrated, same build | `--pcie-frac 0.00`, `--spec-min-p 0.70`, env `STRATA_SH_STREAM=1` |
| Draft-vocabulary recipe (T9) | 10-23 | Chat writing +5.2%, agent writing +6.8%, reading cost zero | `--mtp-draft-vocab /home/drewster/git/Strata-main/data/draft_vocab_en.bin` |
| 5-token drafting + window | 10-15 | Writing +6.2–8.5%, reading flat | `--spec 5 --spec-min-p … --mtp-window 8192` |
| F-2 (skip-widen) | 10-20, promoted 10-24 | −3.2% GPU timeline at 33k, output bit-identical | env `STRATA_HC_MIX_H16=1` |
| CPU offload + persistent hc | 11-03 | Short-prompt time-to-first-token −12%; writing +2.3–4.8% | env `STRATA_PREFILL_CPU_SHARE=1`, `STRATA_HC_PERSIST=1` |
| Patched expert kernels (MMQ/VOP3P) | 11-04 | Reading +0.9–1.4% every tier, writing a wash | llama.cpp branch `v620-mmq-vop3p` on the fork |
| Strata 0.2.10 base | 11-05 | Reading +0.4% vs generation 3 (8/8 interleaved pairs); +4.7% vs generation 1 | worktree `wt-stack` |

### 4.2 Measured and rejected — do not re-test without new evidence

| What | Verdict (when) | Why, in one line |
|---|---|---|
| SIMT2 scorer (`STRATA_SELECT_SIMT=2`) | retired 11-03 | loses reading speed on every tier, −9.9% at 128k, the loss grows with context |
| rocBLAS self-tune on this card | retired 10-14 / re-checked 11-03 | 56/112 kernel shapes pick differently than the shipped table, but tier deltas ±0.2% = noise |
| `STRATA_DENSE_MMQ`, `STRATA_TOPK_STREAM=0` | retired 11-03 | their apparent gains matched session drift; nothing real at any tier |
| `STRATA_IO_THREADS`, `STRATA_PIPED_CONCURRENT`, `STRATA_PREFILL_STREAM_MIN` | retired 11-03 | passed on synthetic text, failed on real text (E483 rule) |
| DOT2 attention score path | abandoned 10-19 | 1.9× faster in a micro-benchmark, 1.48× SLOWER in the engine (fixed per-block cost); opt-in flag `STRATA_ATTN_DOT2` remains in the tree, off |
| reduce12 warp_sum port | abandoned 10-19 | 1.71× slower on this chip; opt-in flag remains in the tree, off |
| fused_gr/gr kernel extension | abandoned 10-14 | only 2 of 19 call sites convert bit-identically; the other two produce hash diffs with no pinned cause |
| gr_upmix GEMM port (S28/S29) | dead 10-19 | the real waste was the widen step, not the GEMM → became F-2 |
| persistent hc for decode only (upstream PR #2226 alone) | neutral on one card 10-14 | later re-adopted as part of the 11-03 pair, which does help |
| PR-A making SH_STREAM a default | dropped 10-24 | reading −2.3–2.5% on its target base (the sign flipped vs an older base — attribution does not carry across bases) |
| k-quant experts (Q4_K_XL/Q5/Q6/Q8 weights) | dead end 10-15 | no GPU kernels exist for k-quant experts; the engine hard-fails; confirmed at scale in the 10-16 matrix |
| Ollama-side tweaks (`OLLAMA_KV_CACHE_TYPE=q8_0`, `use_mmap=false`, `--spec 6`) | rejected 10-14 | −27% writing speed; +53 GB RAM for no gain; worse cold-start |
| DFlash2 drafter | not adopted 11-05 | MTP wins the upstream head-to-heads; DFlash2 is a separate 1.2 GB draft model and Qwen-only |
| INT8 compute conversions | dead end 10-14 (ISA survey) | on this chip INT8 runs at quarter rate = same as FP32; packed FP16 is the only 2× path |

---

## 5. Speed record

### 5.1 How to read the tables

Reading speed (PP) is the trustworthy column; writing speed (TG) has the ±3–4% noise described in §1.
Every number is the middle of ≥3 counted passes after a discarded warm-up.

### 5.2 The story so far, in order

**⚠ One caveat before the table:** every row dated before 2026-10-19 was measured with the slow
one-token-at-a-time expert batching (a leftover workaround from the Ollama fork days). On 10-19 we
flipped it off (`STRATA_PLE_BATCH=1`) and reading speed jumped 12–14% from that alone. So the
pre-10-19 rows are not directly comparable to the later ones — the jump you see across that line is
mostly the batch flip, not the other changes listed.

| Date | What this row is, in words | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| 10-12 | Ollama 0.34.2, Q4_K_M weights (the starting point) | 168.3/15.3 | 171.3/14.8 | 160.5/14.1 | 140.4/11.4 |
| 10-13 | Ollama 0.40.3, same weights | 188.4/16.2 | 185.5/17.1 | 172.6/15.9 | 149.8/15.2 |
| 10-13 | First Strata build (the upstream fork), same weights — 2–4× Ollama | 290.7/45.3 | 288.8/47.4 | 286.0/48.9 | 280.3/46.4 |
| 10-14 | Strata 0.1.43 self-built, Q4_K_M | 527.7/51.1 | 540.5/49.1 | 536.9/50.8 | 516.3/49.6 |
| 10-15 | Same, weights switched to IQ3_S (3.1-bit) | 793.1/65.1 | 807.6/68.9 | 789.3/68.2 | 737.9/67.8 |
| 10-16 | Same, coder 1-bit variant (dropped same day) | 870.8/81.1 | 863.7/78.2 | 837.1/72.4 | 775.6/69.6 |
| 10-14 | 0.1.44.1 + ROCm table + SIMT1 scorer, IQ3_S | 823.0/73.9 | 834.0/76.8 | 834.7/76.5 | 822.6/72.9 |
| 10-14 | + the wave-reduce kernel fix | 828.8/72.7 | 841.6/75.5 | 842.3/74.0 | 827.5/74.2 |
| 10-15 | 0.1.44.1, Unsloth 4-bit lane + 5-token drafting | 835.5/73.2 | 857.8/73.7 | 869.4/73.3 | 861.0/70.0 |
| 10-25 | **Generation 1**: 0.1.48.1 + F-2, benchmark parking (the reference row) | 923.2/92.2 | 944.9/91.5 | 953.8/88.7 | 943.8/86.2 |
| 10-29 | 0.1.53 new main, without / with F-2 (first A/B round) | 902.5 / 912.3 | 943.5 / 957.5 | 946.8 / 962.1 | 933.5 / 949.0 |
| 11-01 | 0.1.55, without / with F-2 (second round, med-of-6) → F-2 wins, promoted | 910.9 / 927.3 | 931.9 / 948.5 | 940.1 / 957.2 | 931.1 / 947.7 |
| 11-03 | Same binary with the SIMT2 scorer (retired this day) | 918.7/91.0 | 932.0/88.2 | 914.4/87.2 | 858.2/86.3 |
| 11-03 | **Generation 2**: 0.1.55 + F-2 + CPU offload + persistent hc | 944.6/94.6 | 957.9/94.5 | 961.9/93.0 | 951.4/89.3 |
| 11-04 | **Generation 3**: same + patched expert kernels | 956.8/92.6 | 973.3/95.6 | 973.5/92.9 | 960.2/89.2 |
| 11-05 | **Generation 4 — current daily**: Strata 0.2.10 + F-2 + patched kernels | **966.2/90.4** | **986.0/90.7** | **982.3/94.0** | **965.8/89.4** |

### 5.3 The three quantizations compared (2026-11-06, same generation-4 binary, benchmark parking)

On this GPU, writing speed tracks how many expert bytes each token touches — the 1-bit set wins
monotonically. The 1-bit lane was the first time this box broke 1,000 reading and 100 writing.

| Lane (disk size, bits/weight) | 16K | 32K | 64K | 128K | Status |
|---|---|---|---|---|---|
| IQ3_S — the usual daily weights (78 GB, ~3.1 bits/weight) | 966.2/90.4 | 986.0/90.7 | 982.3/94.0 | 965.8/89.4 | daily |
| UD-IQ2_XS — Unsloth 2.5-bit (78 GB) | 979.7/96.0 | 984.5/96.6 | 980.6/94.3 | 964.6/93.0 | parked |
| **UD-IQ1_M — Unsloth 1-bit (68 GB)** | **1000.6/100.9** | **1004.4/98.1** | **999.7/97.6** | **982.2/92.1** | **serving since 11-09** |
| DeepSeek-V4.1-Flash 2.5-bit (deleted 11-06) | 209.4/19.2 | 231.7/19.2 | — (needs more VRAM than one card) | — | was: reading record-class, writing unusable |

Accuracy side (Unsloth's published ladder, not our measurement): the 1-bit set recovers ~79.7% of
full-precision top-1 answers (KLD 0.315); the 2.5-bit set has **no** published number; 3-bit-class
sets ≈ 85%+. One caveat on the 2.5-bit lane: its biggest expert layer is stored Q5_0 and that kernel
computed at 2.6–3.0% dot-product error on 0.2.10 (0.2.11 fixes it to ~1.2%). The 1-bit set has no
such layer.

### 5.4 Best numbers ever, and things worth remembering

- Best reading speed per tier: 1000.6 / 1004.4 / 999.7 / 982.2 (the 1-bit lane, 11-06). Best writing: 100.9 / 98.1 / 97.6 / 93.0.
- The "0.1.48 lost 16k" scare (10-24/25) was the conversation cache being evicted between tiers, not a
  code regression. The attempted bisect died of single-boot noise (±1.3% > the effect) — always interleave.
- 0.1.48.1's prefill regression vs 0.1.47 was bounded to ≤16k context, −2.4% calibrated (10-24); never
  bisected (low value).
- The 128k KV-compaction cycle went 16.4 min → 3.0 min across the record (5.5×).

---

## 6. Every experiment: what it was, what happened, how to redo it

Format per entry: **What we tried → what happened → verdict → how to redo it.** "Old doc" = the
pre-rewrite original in git (`c8d3f3b2`) which has the full logs and tables.

1. **Ollama → Strata move** (10-12/13). Tried the incumbent Ollama 0.34/0.40 against the first Strata
   build on the same weights. Result: Strata was 2–4× faster on every tier. Verdict: migration done.
   Redo: Ollama configs via `bench_ollama_tiers.py`; Strata via the v043 configs (old doc).
2. **ROCm tuning table (upstream PR #1167)** (10-14). Shipped a V620-tuned rocBLAS table.
   +1.2–2.7% reading speed. Adopted. Redo: env `STRATA_ROCBLAS_TUNING` pointing at
   `tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt` (shipped; the loader silently rejects tables whose
   version doesn't match — a system rocBLAS 5.1.1 table was rejected). Re-tune command:
   `cmake --build build-hip --target tune_rocblas && LD_LIBRARY_PATH=/opt/rocm/lib ./build-hip/tune_rocblas --tuning-out <out>.txt`. A/B configs: `strata-iq3_s-pr1167{,-noTable}.json` (old doc).
3. **SIMT1 scorer (upstream PR #1151)** (10-14). The author claimed +3.2%; on our single card it gave
   +8.6% reading at 128k. Adopted. Redo: env `STRATA_SELECT_SIMT=1`; the log line
   `qsa select: FP32 tiled scorer` confirms it took effect.
4. **DPP wave-reduce kernel fix** (10-14, commit `48d6aa11`). +0.3–1.0% reading, bit-identical.
   Adopted, now upstream. Redo: old doc "wave-reduce A/B" + result JSONs `…20261014_08*`.
5. **fused_gr/gr kernel extension** (10-14). Tried converting 19 call sites to DPP form: 17 converted
   bit-identically, 2 ternary sites in `fused_gr.cu` produce hash diffs with an unpinned cause.
   Abandoned. Revisit only with a hot-path profile AND an ISA diff of those two kernels. Old doc bisect section.
6. **rocBLAS self-tune on this card** (10-14). The shipped table was tuned on a 6900 XT; a V620
   self-tune picks different kernels for 56/112 shapes, but tier deltas were ±0.2% = noise. Kept the
   shipped table (portability datapoint). Redo: `tune_rocblas` command above; protocol = ≥3 soak
   passes per arm, med-of-3.
7. **Power/thermal study** (10-14). The ~180 W cap governs clocks; a cold machine reads up to 5%
   faster at 16k (2% at 64/128k) than a settled one. Knowledge → became the soak rule (§3.1).
   `setperfdeterminism` unsupported on this card.
8. **RDNA2 ISA survey** (10-14). Checked what the chip can actually do: INT8 runs at quarter rate
   (= FP32 parity, useless), packed FP16 (`v_pk_fma_f16`) is the only 2× path, no WMMA/MFMA on
   gfx1030; the dp4a shim routes correctly. Knowledge — this is what later justified the MMQ/VOP3P work.
9. **Unsloth 4-bit lane** (10-15). Reading +3–4% but writing −8–12% vs IQ3_S (the bigger weights cost
   memory bandwidth). Lane deleted 11-03. Redo: unsloth GGUF + `iq_pack.py --compat-bf16`, config
   clone with `--vram-reserve-mib 600`.
10. **5-token drafting A/B** (10-15). Writing +6.2–8.5%, reading flat; no acceptance decay at long
    context on prose. Adopted (`--spec 5 --mtp-window 8192`). Acceptance is only visible in the serve log.
11. **Halo ladder replication** (10-16). The Halo report's 128k acceptance collapse did NOT reproduce
    here (0.79–0.85 acceptance at 128k). Machine-specific upstream finding. Redo:
    `~/git/llm-benchmark/halo_ladder.py`, data `halo_ladder_runs.json`.
12. **Needle recall test** (10-16). 15/15 correct recall on both lanes tested (4-bit + 5-token, and
    IQ3_S + 4-token). This is our standing quality gate. Redo: `~/git/llm-benchmark/halo_needle.py`
    (parametrize per lane; needs `STRATA_API_KEY`).
13. **Six-lane keep/drop matrix** (10-16). Benchmarked six quantizations in one session. IQ3_XXS had
    the best writing speed; Q4_K_XL ran ~60% slower (k-quant dead end confirmed at scale). At the time
    we kept 4-bit/IQ3_XXS/abliterated and dropped the rest — note IQ3_S came back as the standard lane
    on 10-18, and the other lanes were deleted 11-03. Redo: `run_matrix.py` + configs `*-spec5.json` (old doc).
14. **huihui abliterated-v2** (10-18). Performance parity with the base lane; a byte-level diff proved
    the model card wrong — the expert down-projections ARE ablated in 43/48 layers. Recall 15/15,
    refusal probe 0/6. Lane deleted 11-03.
15. **Rebase + branch A/B matrix** (10-18/19). Rebased all local gfx1030 work onto upstream `477ac13c`;
    the +13% decode vs the fork is the fork default + wave-reduce; reading speed parity across arms.
    Redo: `tools/ab_engine.py --rounds 3`.
16. **Batched expert reading (PLE flip)** (10-19). Turned off a stale one-token-at-a-time workaround
    (upstream issue #541; no fix upstream, but zero stalls here since the 0.1.4x prefill rewrite).
    +10.8% reading at 33k probe, +11.6–14% on tiers, writing +2–6%. Adopted. If a stall ever
    reappears: set `STRATA_PF_STEP_SYNC=1` to diagnose; revert to `STRATA_PLE_BATCH=0` as last resort.
    The baseline caveat is §5.2.
17. **qsa-attention deep-dive (the "S-series")** (10-18/19). Profiled the attention scorer: it is
    instruction-issue-limited at ~23% of FP32 peak. Two attempted fixes both lost in the engine (fixed
    per-block overhead — micro-benchmarks do not transfer to small blocks). Opt-in flags remain in the
    tree, off by default. Redo: `tests/hip/qsa_rows_bench.cpp` + `tests/hip/dot_score_bench.cpp`
    (stop the server first — the benches need 30–150 MB VRAM).
18. **hc-chain analysis (S27–S29)** (10-19). Found the attention-mix path does a GEMM (6.03 ms/launch
    at 77% of packed peak via rocBLAS) and a wasteful widen step. Porting the GEMM was dead; skipping
    the widen was the prize → became F-2 (next entry). Redo: `tests/hip/upmix_gemm_bench.cpp`;
    `tools/hip/tune_rocblas` over engine shapes (all `keep_default`).
19. **F-2 skip-widen** (10-20 → promoted 10-24). Skip the float16-widen step in the attention-mix path:
    −1042 ms / −3.2% GPU timeline at T=32005, output bit-identical (parity: `tests/hip/mix_h16_parity.cu`,
    21 shapes + the in-server check). Reconciliation note: the per-phase timers sum to −2242 ms because
    they double-count cross-stream overlap (~4.6 s baseline over-count) — the difference is fully
    explained, there is no missing time. Adopted; committed as `392586db` lineage and cherry-picked
    onto every later base. Redo A/B: `STRATA_HC_MIX_H16=0/1` on the same binary.
20. **Calibration** (10-22 on 0.1.47; re-verified 10-24 on 0.1.48.1+F-2). Recipe: pcie-frac 0.00,
    spec-min-p 0.70, SH_STREAM=1. Run via `./setup.sh --calibrate` → `tools/calibrate.py` (686 s).
    Warning: never run it while the daily server is live (it restarts the model repeatedly).
    `strata launch --tune quick|thorough` is the Terminal equivalent (never used here).
21. **Draft-vocabulary recipe (T9)** (10-23). Point the 5-token draft head at a smaller English-only
    vocabulary: chat writing +5.2%, agent writing +6.8%, zero reading cost. Adopted. Rejected in the
    same sweep: spec 4, adapt-overlap, pcie-frac 0.35, T7+T9 combo, vram-reserve 200 on agent,
    no mtp-window. Redo: `--mtp-draft-vocab /home/drewster/git/Strata-main/data/draft_vocab_en.bin`.
22. **PR-A (making SH_STREAM a default)** (10-23/24). Reading −2.3–2.5% on the 0.1.47 base — opposite
    sign vs the same flag on 0.1.44.1. Dropped; upstream later shipped per-PC calibration (#1992)
    instead. Lesson: a flag's effect does not carry across base versions.
23. **Conversation-cache parking discovery** (10-25). A big session's cache eviction costs ~1 s on the
    next request; parking ON for daily use, OFF for benchmarks (§1). Redo: the `/tmp/probe-nopark.sh`
    and `/tmp/agent-loop.py` patterns (old doc).
24. **New-main A/Bs, rounds 1–2** (10-29, 11-01). Interleaved, parking-off. F-2 won the med-of-6
    (6/6 sweep) → the 0.1.55+F-2 binary was promoted (generation 2's binary). The missing
    `--engine strata` server gotcha was found here (§3.1.5).
25. **SIMT1 vs SIMT2 scorer** (11-03). SIMT2 loses reading speed on every tier and gets worse with
    context; the 128k row was identical across boot orders (so not thermal). Retired; SIMT1 stays.
26. **Six-knob screen** (11-03). One boot per candidate, soak + 2 counted passes. Adopted: CPU offload
    + persistent hc (generation 2's config). Retired: rocBLAS retune, `STRATA_DENSE_MMQ`,
    `STRATA_TOPK_STREAM=0`. Redo: the `/tmp/env_screen.py` pattern.
27. **Environment-variable screens** (11-03/04). `STRATA_IO_THREADS`, `STRATA_PIPED_CONCURRENT`,
    `STRATA_PREFILL_STREAM_MIN`: positive on synthetic text, negative on real text → retired. This is
    the origin of the E483 real-text rule (§3.2).
28. **Patched expert kernels (MMQ/VOP3P)** (11-04). Patches 6+7 on llama.cpp pin `3cf03257f` (+ helper
    commit `56ccab041`); the iq3_s kernel went 30.9→31.7 TFLOPS author-side; here reading +0.9–1.4% on
    every tier, writing a wash. Adopted (generation 3). Redo: llama.cpp branch `v620-mmq-vop3p` +
    `FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP` pointing at it.
29. **Strata 0.2.10 stack (generation 4)** (11-05). 0.2.10 + F-2 (clean cherry-pick `434dfe91`) +
    patched kernels: reading +0.4% vs generation 3 across 8/8 interleaved pairs; captures upstream's
    own +2–3% improvements. Adopted. Redo: worktree `wt-stack` + the build flags (§3.1.1).
30. **DFlash2 vs MTP drafter** (11-05). DFlash2 is a separate 1.2 GB drafter, Qwen-only; upstream
    head-to-heads favor MTP. Kept MTP. Redo: `--draft dflash2 --draft-model PACK` on 0.1.55+.
31. **DeepSeek-V4.1-Flash at 2.5-bit** (11-06). First gfx1030 run of this model anywhere: reading
    209–231 (ties the best documented single card) but writing 19.2 (it is a reasoning model, and its
    KV needs ~0.29 MB/token so 64k+ context cannot fit one card). Curiosity lane only; deleted 11-06
    at the user's request. Redo: the `dsv41` engine via config `"exe": strata-dsv41`,
    `-DSTRATA_BUILD_DSV41=ON`; pack recipe in the old doc.
32. **Unsloth 2.5-bit and 1-bit lanes** (11-06). Results in §5.3. Redo: configs
    `/tmp/ud-iq{2xs,1m}.json`, packs `packs/{iq2xs,iq1m}`; set `HF_HUB_DISABLE_XET=1` for fast downloads.
33. **Strata 0.2.11 review** (11-06). Read the release: no gfx1030-relevant changes (all new speed work
    is gated to NVIDIA, gfx12, or the split-engine path); llama.cpp pin unchanged; F-2 cherry-picks
    clean; server.py fixes worth having (streamed tool-args, branch races, 60 s no-progress disconnect).
    Also noted: the Unsloth sets were re-uploaded (v3.1) and 0.2.11 fixes the Q5_0 dot error (§5.3
    caveat). Rebase decision still open (§8.2).

---

## 7. If a second V620 arrives

Layer split is the expected win for reading speed (not helper mode — a helper-card experiment upstream
lost ≥1.5k tokens/s on PCIe 4.0). Evidence from other people's boxes, none measured here yet:
a 2×V620 report (`bench/results/2x-v620.txt`, untracked) shows 4k/16k reading 893.4/1,380.3 and writing
68.1/70.7; a 0.1.41 2×6900 XT report shows split ≈ +50% mid-tier reading and +32–39% writing; GitHub
issue #3930 (2×2080 Ti) shows writing 35.2→51.0 with the decode-timing window 70→38 ms.

1. Stop the daily server. In the config: `"gpu": [0,1]`, `"layer_split": "auto"`. Boot and WATCH the
   log's split lines (which layers went to which card).
2. Benchmark with parking off, compare against the generation-4 row (§5.2). Sanity check: if reading
   speed is flat from 4k to 16k, the second card is idle — stop and debug before continuing.
3. Then, one change at a time: `--pipeline-windows 2`, env `STRATA_SPLIT_OVERLAP=1`, and a fresh
   calibration run (never while serving the daily server). Re-check `STRATA_HC_PERSIST` — its
   author-side gains were measured on a split config.
4. DeepSeek-V4.1 is natively multi-GPU (a 2-card boot collapses `tensor parallel 2 -> 1`); 128k context
   fits two cards only without the drafter. The DSpark economics also flip at two cards (the drafter
   fits; ~18% expert-cache coverage).

---

## 8. Things still open

1. **Second V620** — runbook above (§7); this TODO closes when the card arrives.
2. **0.2.11 rebase** — F-2 is verified to apply cleanly; expected reading-speed delta ~0 on one card;
   the win would be the server.py fixes. Decision: build `0.2.11 + F-2 + patched kernels` and A/B vs
   generation 4, or hold.
3. **#3381 RDNA2 prompt-attention port** — the diff at `~/strata-upstream/pr-3381-prompt-attn-rdna2.diff`
   (373 lines) does NOT apply to 0.1.55 or 0.2.10 (drift at `prefill.cpp:3435`); needs a manual port.
   Author claims the QSA prompt chunk gets 29% faster.
4. **Memory-lock (memlock) raise** — needs interactive sudo; the PLE table mlock fails at 8 MiB (§2.1).
5. **Upstream write-ups** — two drafts sit in `~/strata-upstream/` (rocBLAS-retune-neutral,
   simt2-vs-simt1); PR descriptions pending for the three pushed fork branches.
6. **1-bit trial verdict** — keep serving UD-IQ1_M (current) or revert to the IQ3_S daily config,
   based on how the real-work trial feels (§2.2, §2.3).
7. **Decode-timing boot** — one benchmark boot with `STRATA_DECODE_TIMING=1` is still the candidate
   way to attribute split-config writing speed (see §7 and GitHub issue #3930).

---

## 9. Where everything lives

- **Worktrees**: `~/git/Strata` (main checkout, now at 0.2.11, commit `9f7d977d`), `~/git/Strata-main`
  (parked on branch `parked/pr-a-gfx103x-default`), `~/git/wt-0155` (generation 2 binary), `~/git/wt-mmq`
  (generation 3), `~/git/wt-stack` (generation 4 — this document's home).
- **GitHub forks** (backup copies): `andrewprock/Strata` — branches `v620-daily-0155-f2` (generation 3)
  and `v620-stack-0210-f2-mmq` (generation 4); `andrewprock/llama.cpp` — branch `v620-mmq-vop3p`
  (the patched kernels). The `gh` CLI is authenticated (https).
- **Configs**: daily `~/strata-v620-iq3_s-T9-draftvocab.json` (+ the three backups next to it); the
  Unsloth lanes `/tmp/ud-iq{1m,2xs}.json` (plus `-serve` variants); bench arms `/tmp/ab-*.json`,
  `/tmp/iq3s-*.json`.
- **Model packs**: `~/git/Strata-data/packs/{iq3_s,iq1m,iq2xs,iq3_s-abl}`; the MTP draft head at
  `~/git/Strata-data/mtp/rt`.
- **Benchmark results**: `~/git/llm-benchmark/results/strata_tier_pass_*.json` (the timestamp in the
  filename is the run time).
- **Binary backups**: `~/engines/backup/{strata-0.1.47-base,strata-0.1.48.1-f2}`. This document's
  earlier versions: `~/strata-migration.md.bak-20261117-pre-rewrite` and
  `~/strata-migration.md.bak-20261117-post-redundancy`, plus git history (`c8d3f3b2` original →
  `ba97795e` dedup → `08de0232` first rewrite → this one).
- **The 9 environment variables that every lane config carries** (what each does):
  `STRATA_WATCHDOG_S=300` (restart watchdog timer), `STRATA_PLE_BATCH=1` (batched expert reading),
  `STRATA_HIP_PROMPT_F16=1` (FP16 prompt path), `STRATA_SELECT_SIMT=1` (the fast scorer),
  `STRATA_SH_STREAM=1` (streamed short-context path), `STRATA_ROCBLAS_TUNING=<abs path>` (the tuning
  table), `STRATA_HC_MIX_H16=1` (F-2 skip-widen), `STRATA_PREFILL_CPU_SHARE=1` (CPU offload),
  `STRATA_HC_PERSIST=1` (persistent hc state).
