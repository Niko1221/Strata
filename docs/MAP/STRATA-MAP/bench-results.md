# Strata map: bench results (`bench/results/`)

Source: Strata commit `296b59d2`. Paths are relative to the Strata repo root. The 2,086 files under `bench/results/` are split into two parts by sorted path.

## Part 1: entries 1-1043

Entries 1..1043: first `bench/results/2026-09-24-final/estimates.json`, last `bench/results/2026-10-05-community-2x-rx-6900xt/helper-stock/benchmark.out`. 68 top-level folders.

## 1. Structure

Each folder = one dated packet `YYYY-MM-DD-<who/what>` (`community-*` = third-party submissions; others maintainer experiments, e.g. `layer-split`, `ctx-ladder`, `rope-scaling`, `windows-groups`, `dev-5090`, `final`). Typical packet: `README.md` (human table + caveats), `TRIMMED.md` (what was cut), raw `*.json` from the bench client, the client script itself (`benchmark.py`/`bench.py`), launch scripts (`.bat`, `.sh` and `.ps1` files), `PROVENANCE.txt`/`SHA256SUMS.txt`, sometimes `data/`, `logs-*/` subdirs.

Largest folders: rtx-pro-4500 174, rtx-5090-5950x 89, windows-groups 66, rtx3090-epyc-milan 60, rx6900xt-coder 52, dev-5090 50.

## 2. Formats (part 1 counts)

| ext | n | schema / meaning |
|---|---:|---|
| json | 699 | bench client output: top keys `started,url,model,engine_log,cap_tokens,reps,speed[],needles,notes,calibration,chars_per_token,warmup,finished`; each `speed[]` row: `ok,payload_variant,thinking_off,ttft_s,total_s,answer_len,reasoning_len,reasoning_text,finish_reason,usage{prompt_tokens,completion_tokens,total_tokens,prompt_tokens_details.cached_tokens},engine_prompt_tokens,reused_tokens,new_prompt_tokens,read_ms,read_tok_s,generated_tokens,gen_ms,decode_tok_s,target,rep,chars` (seen in bench/results/2026-10-03-community-r9700-windows/bench-output-standard.json:2-37; bench/results/2026-10-03-community-r9700-windows/bench-output-standard.json:280; bench/results/2026-10-03-community-r9700-windows/bench-output-standard.json:474-530). Also `results.json` from tools/strata_bench.py:912, runs.json, provenance/BUILD json |
| md | 113 | README / TRIMMED / METHOD reports with median+range tables |
| txt | 104 | provenance, sha sums, hipBLASLt tuning tables (e.g. bench/results/2026-10-04-community-rx7900xtx-hipblaslt-100500/gfx1100-hipblaslt-100500.txt:2), selftests |
| py | 61 | per-packet bench/monitor clients (benchmark.py; bench/results/2026-10-04-community-rx7900xtx-hipblaslt-100500/ab-driver.py:2; bench/results/2026-10-04-community-mi50/monitor_mi50.py:1) |
| csv | 47 | nvidia-smi style telemetry `time, MiB, W` (bench/results/2026-10-01-community-rtx-pro-4500/logs-2gpu/sec-iq3_s-128k.csv:1); comparison.csv |
| jsonl | 2 | telemetry `{epoch_s, memory{MemTotal_KiB..}, gpu:"..."}` (bench/results/2026-09-30-community-rtx-5090/telemetry.jsonl:1) |
| out/sh/bat/ps1 | 5/4/3/1 | raw stdout, launchers (`run-iq2_xs-*.bat`) |
| log/cu/(no extension) | 1/1/2 | other files |

Metrics: prompt read tok/s (`read_tok_s` = new_prompt_tokens/read_ms, engine log), decode tok/s (engine log), TTFT, total s, reused tokens (prefix cache), needle recall (FOUND at depths 10/50/90%), draft acceptance %, expert-cache hit % (bench/results/2026-10-02-community-rx9070xt-windows/README.md:50), VRAM/W telemetry.

```mermaid
---
config: {"theme":"base","themeVariables":{"darkMode":true,"background":"#0b1020","fontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif","fontSize":"14px","primaryColor":"#1e293b","primaryTextColor":"#e2e8f0","primaryBorderColor":"#3b82f6","secondaryColor":"#1e293b","secondaryTextColor":"#e2e8f0","secondaryBorderColor":"#3b82f6","tertiaryColor":"#111827","tertiaryTextColor":"#e2e8f0","tertiaryBorderColor":"#334155","mainBkg":"#1e293b","nodeBorder":"#3b82f6","textColor":"#e2e8f0","titleColor":"#93c5fd","lineColor":"#64748b","clusterBkg":"#0f172a","clusterBorder":"#334155","edgeLabelBackground":"#0b1020","noteBkgColor":"#172554","noteTextColor":"#dbeafe","noteBorderColor":"#3b82f6","actorBkg":"#1e293b","actorTextColor":"#e2e8f0","actorBorder":"#3b82f6","actorLineColor":"#475569","signalColor":"#94a3b8","signalTextColor":"#e2e8f0","labelBoxBkgColor":"#1e293b","labelBoxBorderColor":"#3b82f6","labelTextColor":"#e2e8f0","loopTextColor":"#93c5fd","activationBkgColor":"#334155","activationBorderColor":"#3b82f6"},"flowchart":{"curve":"basis","nodeSpacing":34,"rankSpacing":40,"padding":12,"wrappingWidth":260,"htmlLabels":true},"sequence":{"actorFontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif","messageFontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif","noteFontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"},"fontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"}
---
flowchart LR
  A[bench.sh / BENCH.bat] --> B[tools/strata_bench.py]
  C[community benchmark.py] --> D[/v1/chat/completions serve/server.py/]
  B --> D --> E[engine log: read ms, decode tok/s]
  E --> F[results.json / bench-output-standard.json]
  F --> G[README.md median tables + TRIMMED.md + PROVENANCE]
  G --> H[bench/results/DATE-community-HW/]
```

## 3. Hardware / models

**GPU name mentions** (also Arc Pro B60, RTX PRO 4500 and Titan RTX):

| GPU | Mentions |
|---|---|
| gfx1030 | 61 |
| RTX 3090 | 124 |
| RTX 5090 | 72 |
| gfx1100 | 43 |
| P40 | 41 |
| V100 | 35 |
| gfx1201 | 27 |
| RTX 4090 | 32 |
| RTX 5070 | 35 |
| 5060 Ti | 43 |
| MI50 | 16 |
| 2080 Ti | 28 |
| gfx1200 | 7 |

**Quant mentions:**

| Quant | Mentions |
|---|---|
| `iq3_s` | 894 |
| `iq3_xxs` | 798 |
| `iq2_xs` | 424 |
| `q4_k` | 420 |
| `iq1_m` | 405 |
| `iq4_xs` | 222 |

The model is always Qwen3.8-Flash-Next (plus the Coder variant, e.g. bench/results/2026-10-03-rx6900xt-coder/README.md:1, and dense 27B comparisons, e.g. bench/results/2026-10-04-community-4090-agents-vs-dense-27b/README.md:1-6). **NVFP4: no results.** The only text mention is at `bench/results/2026-10-02-community-rtx-5090-iq3s/README.md:95`.

## 4. AMD results (part 1)

| card / arch | quant, OS, engine | prompt tok/s | decode tok/s | file:line |
|---|---|---|---|---|
| R9700 gfx1201 | IQ2_XS, Win11, 0.1.38 prebuilt HIP, 9950X | 4K **855.8**, 32K **1,182.1**, 128K **1,231.7**, 248K **1,130.1** | 99.8 / 97.8 / 89.9 / 87.5 | bench/results/2026-10-03-community-r9700-windows/README.md:1-13 |
| R9700 needles | same | 33.5K 1,178.6 / 1,193.8 / 1,186.8; 131.4K 1,230.7 / 1,233.8 / 1,229.7 | 73.7–118.9 | bench/results/2026-10-03-community-r9700-windows/README.md:187-190 (6/6 FOUND); bench/results/2026-10-03-community-r9700-windows/README.md:1-4 |
| R9700 raw row | same | read_tok_s 855.8 (4195 tok, 4902 ms) | 99.8 | bench/results/2026-10-03-community-r9700-windows/bench-output-standard.json:27-34 (speed[0]); bench/results/2026-10-03-community-r9700-windows/README.md:1-4 |
| R9700 upstream ref (quoted) | Coder IQ1_M, Ubuntu VM | 4K 982, 16K 1,402; WMMA 1,784→2,427 | 45.5 / 48.3 | bench/results/2026-10-03-community-r9700-windows/README.md:232-237 (cites docs/AMD_HIP.md) |
| RX 9070 XT gfx1201 | IQ3_S, Win11, 0.1.35 prebuilt HIP | 4K 370.4, 32K 601.5 | 45.2 (draft acc 84.5%, hit 84.6%) | bench/results/2026-10-02-community-rx9070xt-windows/README.md:1-11; bench/results/2026-10-02-community-rx9070xt-windows/README.md:50-54 |
| RX 7900 XTX gfx1100 | nightly ROCm 10.2 + hipBLASLt table | 128K 1,687.0 | 62.26 | bench/results/2026-10-04-community-rx7900xtx-hipblaslt-100500/README.md:1-3; bench/results/2026-10-04-community-rx7900xtx-hipblaslt-100500/README.md:73-75 |
| RX 7900 XTX | ROCm 7.1.1 vs nightly | – | 80.4 vs 88.4 (+10%, nightly) | bench/results/2026-10-03-community-rx7900xtx-rocm102-nightly/README.md:1; bench/results/2026-10-03-community-rx7900xtx-rocm102-nightly/README.md:49-55 |
| RX 7900 GRE gfx1100 | IQ2_XS / IQ3_XXS / Coder IQ1_M, Linux | 1,140-1,216 (all three arms) | 65.3 / 59.1 / 54.7 | bench/results/2026-10-04-community-rx-7900-gre/README.md:1-11 |
| MI50 gfx906 | IQ2_XS, Linux, 0.1.38 | ~330 (4K/32K), ~305 (128K) | 36.0 @4K | bench/results/2026-10-04-community-mi50/README.md:1-10 |
| RX 6900 XT gfx1030 (Coder) | Coder IQ1_M, Win11, 0.1.38 HIP release, 31 workers (superseded selection; setup default 63 gave 6.1-12.5 decode) | 4K 312.5, 32K 352.1 | 45.3–48.0 | bench/results/2026-10-03-rx6900xt-coder/README.md:1-15; bench/results/2026-10-03-rx6900xt-coder/README.md:59-66 |

Also present, not detailed: rx-6800-windows, 2x-mi50, 2x-rx-6900xt, and the maintainer RX 6900 XT (gfx1030) packets rdna2-fp16-prompt, rdna2-helper-pcie-share, rdna2-pre75-attention, windows-groups (bench/results/2026-10-04-rdna2-fp16-prompt/README.md:1; bench/results/2026-10-04-rdna2-helper-pcie-share/README.md:1, 2x RX 6900 XT; bench/results/2026-10-04-rdna2-pre75-attention/README.md:1; bench/results/2026-10-03-windows-groups/README.md:3).

## 5. Gotchas

- R9700 medians drift ±10–15% between sessions (32K 1,182.1 vs 1,060.6) — bench/results/2026-10-03-community-r9700-windows/README.md:15-19; R9700 per bench/results/2026-10-03-community-r9700-windows/README.md:1.
- 128K/248K R9700 ran in one session only (bench/results/2026-10-03-community-r9700-windows/README.md:19-20; R9700 per bench/results/2026-10-03-community-r9700-windows/README.md:1); `bench/results/2026-10-03-community-r9700-windows/bench-output-standard-first-attempt-void.json` is a failed run (13/15 empty, bench/results/2026-10-03-community-r9700-windows/README.md:179-180) — excluded.
- `STRATA_PF_FUSED=1` no observable effect; `STRATA_PLE_BATCH=0` is a stall workaround, −12.8% — bench/results/2026-10-03-community-r9700-windows/README.md:238-245.
- R9700 run used `STRATA_HIPBLASLT_TUNING=tools/hip/gfx1201-hipblaslt-100500.txt` (bench/results/2026-10-03-community-r9700-windows/strata-iq2_xs.json:42; R9700 per bench/results/2026-10-03-community-r9700-windows/README.md:1); matrix-core path `STRATA_HIP_WMMA=1` not used → 855.8 not a ceiling (bench/results/2026-10-03-community-r9700-windows/README.md:236-237).
- Long ctx >16K on RDNA4 "not validated" per docs/AMD_HIP.md (quoted in bench/results/2026-10-03-community-r9700-windows/README.md:28).
- decode figures are engine-log values, not client-side; truncated (<256 tok) runs bias decode (bench/results/2026-10-03-community-r9700-windows/README.md:5-6; bench/results/2026-10-03-community-r9700-windows/README.md:228-231).

## Quick index

| Area | Kind | Name | Where | Notes |
|---|---|---|---|---|
| bench | tool | bench.sh | `bench.sh:7` | wraps .venv python tools/strata_bench.py |
| bench | function | results.json writer | `tools/strata_bench.py:912` | writes results.json |
| bench | behavior | R9700 best prefill 1231.7 tok/s 128K | `bench/results/2026-10-03-community-r9700-windows/README.md:1-12` | IQ2_XS Win 0.1.38 |
| bench | behavior | R9700 best decode 99.8 tok/s 4K | `bench/results/2026-10-03-community-r9700-windows/README.md:1-10` | IQ2_XS Win 0.1.38 |
| bench | env | STRATA_HIPBLASLT_TUNING | `bench/results/2026-10-03-community-r9700-windows/strata-iq2_xs.json:42` | gfx1201 table |
| bench | env | STRATA_HIP_WMMA | `bench/results/2026-10-03-community-r9700-windows/README.md:232-237` | 4K 1784->2427 per docs/AMD_HIP.md |
| bench | gotcha | session variance | `bench/results/2026-10-03-community-r9700-windows/README.md:17-18` | ±10-15% |
| bench | env | STRATA_PLE_BATCH=0 | `bench/results/2026-10-03-community-r9700-windows/README.md:243-245` | stall workaround, slower |
| bench | behavior | RX9070XT IQ3_S | `bench/results/2026-10-02-community-rx9070xt-windows/README.md:50-54` | decode 45.2 pf32K 601.5; IQ3_S per bench/results/2026-10-02-community-rx9070xt-windows/README.md:1 |
| bench | behavior | 7900XTX 128K prefill | `bench/results/2026-10-04-community-rx7900xtx-hipblaslt-100500/README.md:73-75` | 1687 tok/s; RX 7900 XTX per bench/results/2026-10-04-community-rx7900xtx-hipblaslt-100500/README.md:1 |
| bench | gotcha | no NVFP4 results | `bench/results/2026-10-02-community-rtx-5090-iq3s/README.md:95` | only text mention |

---

## Part 2: entries 1044-2086

Entries 1044..2086: first `bench/results/2026-10-05-community-2x-rx-6900xt/helper-stock/config.json`, last `bench/results/exchange-rotation-token-ids.json`. Paths below are relative to the Strata repo root.

## 1. Structure

Same packet convention as part 1: `YYYY-MM-DD-community-<hardware>[-variant]` (third-party) or `YYYY-MM-DD-<topic>` (maintainer: `llamacpp-0.6.0`, `q8-prefill8192-rtxpro`, `v100-decode-kernels`, `gfx906-qsa-query-swizzle`, `rope-yarn-long-context`, `rtx-5090-callback-affinity`, `p40-sm61-kernels`). Dates 2026-10-05 to 2026-10-12. Plus top-level `bench/results/COMMUNITY.md`, `bench/results/exchange-rotation-ab.json` and `bench/results/exchange-rotation-token-ids.json`.

Typical packet: `README.md` (tables + caveats), `REPRODUCE.md`/`METHODS.md`, `TRIMMED.md`, `SHA256SUMS`, strata server config `strata-<quant>.json`, bench-output JSON and Markdown files, engine `*.log`, client `benchmark.py`, sometimes `data/`, `raw/`, `scripts/`, `harness/`, `patches/*.patch`, `provenance/`, `analysis/`. Big sub-trees: 2x-rx-6900xt (stock 0.1.39 and 0.1.39 + PRs #835/#849/#854 × single/split/helper arms, plus helper-stock-pcie0; bench/results/2026-10-05-community-2x-rx-6900xt/README.md:3-6; bench/results/2026-10-05-community-2x-rx-6900xt/README.md:66), rtx5090-9800x3d-swift-1m (39 gzip request payloads plus requests/index.json, bench/results/2026-10-09-community-rtx5090-9800x3d-swift-1m/README.md:331; bench/results/2026-10-09-community-rtx5090-9800x3d-swift-1m/README.md:102), rx-9060-xt (fixtures + 18 coding `solutions/`).

```mermaid
---
config: {"theme":"base","themeVariables":{"darkMode":true,"background":"#0b1020","fontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif","fontSize":"14px","primaryColor":"#1e293b","primaryTextColor":"#e2e8f0","primaryBorderColor":"#3b82f6","secondaryColor":"#1e293b","secondaryTextColor":"#e2e8f0","secondaryBorderColor":"#3b82f6","tertiaryColor":"#111827","tertiaryTextColor":"#e2e8f0","tertiaryBorderColor":"#334155","mainBkg":"#1e293b","nodeBorder":"#3b82f6","textColor":"#e2e8f0","titleColor":"#93c5fd","lineColor":"#64748b","clusterBkg":"#0f172a","clusterBorder":"#334155","edgeLabelBackground":"#0b1020","noteBkgColor":"#172554","noteTextColor":"#dbeafe","noteBorderColor":"#3b82f6","actorBkg":"#1e293b","actorTextColor":"#e2e8f0","actorBorder":"#3b82f6","actorLineColor":"#475569","signalColor":"#94a3b8","signalTextColor":"#e2e8f0","labelBoxBkgColor":"#1e293b","labelBoxBorderColor":"#3b82f6","labelTextColor":"#e2e8f0","loopTextColor":"#93c5fd","activationBkgColor":"#334155","activationBorderColor":"#3b82f6"},"flowchart":{"curve":"basis","nodeSpacing":34,"rankSpacing":40,"padding":12,"wrappingWidth":260,"htmlLabels":true},"sequence":{"actorFontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif","messageFontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif","noteFontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"},"fontFamily":"Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"}
---
flowchart LR
  P[benchmark.py / bench_engine.py / strata_bench.py] -->|HTTP or native line protocol| S[strata serve / engine]
  S --> L[engine log: prompt N = reused + read in ms, decode tok/s]
  L --> J[bench-output-standard.json, telemetry.jsonl, requests.csv]
  J --> A[speed.csv, fresh-speed.csv, comparisons.csv]
  A --> R[README.md median+range tables]
  R --> C[COMMUNITY.md index]
```

## 2. Formats (part 2 counts)

| ext | n | schema / meaning |
|---|---:|---|
| json | 485 | 123 are lists (per-request rows); 35 keyed by length `{"4096","32768","128000"}`; 29 server state `activity,auto_load,cache_max_tokens,concurrency,context,dialects,engine,last_timings`; 14+7 launch configs `args,backend,cwd,env,exe,gpu,gpus_asked,lib_dirs\|layer_split`; 12 OpenAI request bodies `max_tokens,messages,model,reasoning_effort,stream,...`; 9 `env,label,memory,results,style` |
| md | 116 | READMEs / methods / TRIMMED |
| py | 104 | clients/harness (`benchmark.py`; `bench/results/2026-10-08-community-r9700-linux/harness/bench_engine.py:2`; `bench/results/2026-10-08-community-r9700-linux/harness/run_pairs.py:2`; `bench/results/2026-10-08-community-r9700-linux/harness/resource_sampler.py:1`) |
| txt | 85 | prompts (`bench/results/2026-10-06-llamacpp-0.6.0/prompt-4k.txt:1`), provenance, tuning tables |
| ts | 50 | rx-9060-xt coding fixtures/solutions (TypeScript), not benchmarks |
| gz | 42 | `measurements.json.gz`, `fixtures.json.gz` full records+logs |
| jsonl | 40 | per-request: P100 `target_tokens,prompt_tokens_tokenized,run,nonce,ttft_s,total_s,usage`; llama.cpp `wall_s,prompt_n,prompt_ms,prompt_tps,pred_n,pred_tps,draft_n,draft_acc,text` |
| log/out | 27/21 | engine stdout/session logs |
| csv | 23 | e.g. r9700 `fresh-speed.csv` (bench/results/2026-10-08-community-r9700-linux/data/fresh-speed.csv:1) cols `input_tokens,output_tokens,repeats,prefill_tokens_per_s_{median,min,max},decode_tokens_per_s_*,ttft_s_*,wall_s_*` |
| sh/patch/bat/cpp/cu/svg/png | 15/12/2/2/2/4/5 | launchers, archived PR patches, kernel experiments, plots |
| ids/(no extension)/sha256 | 4/3/1 | other files |

Metrics: prompt tok/s (= new tokens / engine prompt time), decode tok/s (engine), TTFT (native or client), total s, reused tokens, MTP drafts accepted, expert-cache hit %, needle recall, coding-task pass counts.

## 3. AMD gfx11/gfx12 results

| GPU / arch | Pack | Engine | Prompt tok/s | Decode tok/s | file:line |
|---|---|---|---|---|---|
| R9700 32GB gfx1201, Linux, 2×EPYC 9334 | IQ3_S | exp. 0.1.40.3 (d5ea713); the speed run at 25d2e33 reused the candidate (269a140) executable (bench/results/2026-10-08-community-r9700-linux/README.md:38-41) | 632.9 @1K, **768.1 @4K**, 750.5 @32K, 718.4 @128K | **101.9 @1K**, 96.4, 95.6, 87.6 @128K | bench/results/2026-10-08-community-r9700-linux/README.md:3-24; bench/results/2026-10-08-community-r9700-linux/README.md:83-88 |
| R9700 same, grouped gather candidate | IQ3_S | candidate 269a140 | TTFT −16.18% (512 new) / −14.07% (900) after 32K | IQ3_S decode time +3.5% to +4.7% at 4K/32K (bench/results/2026-10-08-community-r9700-linux/README.md:121) | bench/results/2026-10-08-community-r9700-linux/README.md:35-39; bench/results/2026-10-08-community-r9700-linux/README.md:98-109 |
| RX 9070 16GB gfx1201, Win11 | IQ3_XXS | 0.1.41 prebuilt | 586.2 @4.2K, 804.5 @33.5K, 832.9 @130K | 58.7 / 57.9 / 57.9 | bench/results/2026-10-10-community-rx-9070/README.md:1-4; bench/results/2026-10-10-community-rx-9070/README.md:106-110 |
| RX 9060 XT 16GB gfx1200, Win | IQ2_XS GSQ-RCO | 0.1.40.3 | 879.4 @8K (prefill 8192); 789.9 @113.5K | 56.3; 46.9 | bench/results/2026-10-10-community-rx-9060-xt/README.md:3-16; bench/results/2026-10-10-community-rx-9060-xt/README.md:51-59; bench/results/2026-10-10-community-rx-9060-xt/README.md:86-90 |
| Radeon 8060S gfx1151 (EVO-X2) | UD-IQ4_XS | 0.1.41 fb58e0d | fast: 1,165.4 @32K, 1,108.6 @128K | 49.6 / 47.5 | bench/results/2026-10-09-community-gfx1151-ud-iq4xs-0.1.41/README.md:1-3; bench/results/2026-10-09-community-gfx1151-ud-iq4xs-0.1.41/README.md:50-60 |
| 8060S gfx1151 laptop 70W | UD-IQ4_XS (fast cfg) | 0.1.41 | 969 @8K, 1,067 @32K, 1,031 @131K | 59.2 / 57.8 / 42.0 | bench/results/2026-10-09-community-strix-halo-laptop-70w/README.md:1-21; bench/results/2026-10-09-community-strix-halo-laptop-70w/README.md:49-53 |
| 8060S gfx1151, 0.1.39 | Q2_0 | 0.1.39 | tuning table 250.9 → 528.9 (2.11×) | 45.63 → 46.77 | bench/results/2026-10-05-community-gfx1151/README.md:1-18; bench/results/2026-10-05-community-gfx1151/README.md:46 |
| Radeon 890M gfx1150 | GSQ-RCO IQ3_XXS | 0.1.40.2 | 130.5 → 225.8 @7K (table+switches; table alone 219.4) | 15.2-15.3 | bench/results/2026-10-07-community-gfx1150/README.md:1-25; bench/results/2026-10-07-community-gfx1150/README.md:64-69 |
| RX 7900 XTX gfx1100, Win | IQ3_S | 0.1.40 | 1,612.0 @38K | 79.7 | bench/results/2026-10-06-community-rx-7900-xtx/README.md:3-7; bench/results/2026-10-06-community-rx-7900-xtx/README.md:81-84 |
| 4× RX 7900 XT | GSQ-RCO IQ3_S | 0.1.41 | opt-ins on 4,307 @128K (off 2,433) | 50.4 | bench/results/2026-10-09-community-4x-rx-7900-xt/README.md:1-4; bench/results/2026-10-09-community-4x-rx-7900-xt/README.md:93-100 |
| 2× RX 7900 GRE | Coder IQ1_M | 0.1.40 | fresh prefill 1,582.9 (stage-trim test, trim on; off 1,103.4; that test's decode 75.7 vs 67.6) | 69.3 (pipeline-windows 2) | bench/results/2026-10-06-community-2x-rx-7900-gre/README.md:3-4; bench/results/2026-10-06-community-2x-rx-7900-gre/README.md:14; bench/results/2026-10-06-community-2x-rx-7900-gre/README.md:17-18 |

Also present, not detailed: RX 7900 XTX gfx1100 primary + RX 6800 XT helper, 1M context (bench/results/2026-10-07-community-xtx-6800xt-1m-helper/README.md:1; bench/results/2026-10-07-community-xtx-6800xt-1m-helper/README.md:10-11).

R9700 Linux packet: Linux R9700 IQ3_S decode 101.9 tok/s @1K and prompt 768.1 tok/s @4K (bench/results/2026-10-08-community-r9700-linux/README.md:21-24; bench/results/2026-10-08-community-r9700-linux/README.md:76-86; bench/results/2026-10-08-community-r9700-linux/data/fresh-speed.csv:1-3). Config notably: Tensile BLAS, `STRATA_HIP_WMMA=0`, empty hipBLASLt tuning, `ROCBLAS_USE_HIPBLASLT=0` (bench/results/2026-10-08-community-r9700-linux/README.md:57-59; bench/results/2026-10-08-community-r9700-linux/README.md:70-72).

## 4. NVFP4 (5 files in part 2)

- `bench/results/2026-10-06-llamacpp-0.6.0/prompt-4k.txt:111` and `bench/results/2026-10-06-llamacpp-0.6.0/prompt-32k.txt:111` (1 hit): prompt filler text quoting "83 -> 109 tokens/s (RTX 5090, the NVFP4 fork)". Not a measurement.
- `bench/results/2026-10-09-community-rtx5090-9800x3d-swift-1m/data/overlap-prototype-runs.json:2246`, `bench/results/2026-10-09-community-rtx5090-9800x3d-swift-1m/data/supervisor-handoffs.json:6`, `bench/results/2026-10-09-community-rtx5090-9800x3d-swift-1m/data/overlap-prototype-manifest.json:121-149`: a *different* model `qwen3.8-27b-ninfer-awq-gptq-dflash2-262k-nvfp4kv` (NInfer, NVFP4 weights and NVFP4 **KV cache**, `qwen3_8_27b_nvfp4_awq.ninfer`) used as a co-resident/handoff model on RTX 5090; not Strata NVFP4 weights.
- **No NVFP4 Flash-Next Strata benchmark on any GPU** in part 2.

## 5. Gotchas

- R9700 Linux numbers are from a frozen experimental 0.1.40.3 build, not upstream HEAD (bench/results/2026-10-08-community-r9700-linux/README.md:13-15; bench/results/2026-10-08-community-r9700-linux/README.md:21-24); hipBLASLt numerical failures forced Tensile (bench/results/2026-10-08-community-r9700-linux/README.md:70-72).
- Two R9700 speed configs must not be combined into one speedup claim (bench/results/2026-10-08-community-r9700-linux/README.md:134; R9700 per bench/results/2026-10-08-community-r9700-linux/README.md:21); IQ3_S decode is 3.5–4.7% slower at 4K/32K with the grouped-gather candidate (bench/results/2026-10-08-community-r9700-linux/README.md:35-39; bench/results/2026-10-08-community-r9700-linux/README.md:120-121).
- R9700 packet uses a gfx1201 Q2_0 signed-zero fix (R9700 per bench/results/2026-10-08-community-r9700-linux/README.md:21; bench/results/2026-10-08-community-r9700-linux/README.md:34, bench/results/2026-10-08-community-r9700-linux/README.md:128).
- RX 9070: "128000" label produced 131,428 tokens and was rejected by context; refit to label 126,640 (129,970-129,973 tokens, bench/results/2026-10-10-community-rx-9070/README.md:110; RX 9070 per bench/results/2026-10-10-community-rx-9070/README.md:1) (bench/results/2026-10-10-community-rx-9070/README.md:91; bench/results/2026-10-10-community-rx-9070/README.md:131).
- gfx1151 "fast" arm = bundle of opt-in switches (STRATA_PF_FUSED, PF_GEMM, HC_UPMIX…), not one kernel (bench/results/2026-10-09-community-gfx1151-ud-iq4xs-0.1.41/README.md:62; bench/results/2026-10-09-community-strix-halo-laptop-70w/README.md:30).
- 4×7900 XT "opt-ins" give 1.8-2.7× prompt (bench/results/2026-10-09-community-4x-rx-7900-xt/README.md:1; bench/results/2026-10-09-community-4x-rx-7900-xt/README.md:105); cold first request 3,430 vs 963 (bench/results/2026-10-09-community-4x-rx-7900-xt/README.md:102-103).
- `.ts` files under `bench/results/2026-10-10-community-rx-9060-xt/solutions/` are model-generated coding outputs, not tooling (bench/results/2026-10-10-community-rx-9060-xt/README.md:82; bench/results/2026-10-10-community-rx-9060-xt/METHODS.md:114).
- Data contains `${HOME}`/`<STRATA_ROOT>` placeholders (bench/results/2026-10-08-community-r9700-linux/README.md:149; bench/results/2026-10-09-community-rtx5090-9800x3d-swift-1m/README.md:68).

## Quick index

| Area | Kind | Name | Where | Notes |
|---|---|---|---|---|
| bench | behavior | R9700 Linux best decode 101.9 tok/s @1K IQ3_S | `bench/results/2026-10-08-community-r9700-linux/README.md:52-85` | engine log, 256 tok, 0.1.40.3 (bench/results/2026-10-08-community-r9700-linux/README.md:13); R9700/Ubuntu per bench/results/2026-10-08-community-r9700-linux/README.md:21-24 |
| bench | behavior | R9700 Linux best prompt 768.1 tok/s @4K IQ3_S | `bench/results/2026-10-08-community-r9700-linux/README.md:76-86` | median of 3; R9700/Ubuntu per bench/results/2026-10-08-community-r9700-linux/README.md:21-24 |
| bench | behavior | R9700 Linux 128K 718.4/87.6 | `bench/results/2026-10-08-community-r9700-linux/README.md:83-88` | prompt/decode; R9700/Ubuntu per bench/results/2026-10-08-community-r9700-linux/README.md:21-24 |
| bench | env | STRATA_HIP_WMMA=0 + ROCBLAS_USE_HIPBLASLT=0 | `bench/results/2026-10-08-community-r9700-linux/README.md:70-72` | Tensile path; hipBLASLt numeric failures |
| bench | gotcha | R9700 build not upstream | `bench/results/2026-10-08-community-r9700-linux/README.md:13` | frozen 0.1.40.3 d5ea713; R9700 per bench/results/2026-10-08-community-r9700-linux/README.md:21 |
| bench | behavior | grouped gather TTFT -16% | `bench/results/2026-10-08-community-r9700-linux/README.md:103-106`, `bench/results/2026-10-08-community-r9700-linux/README.md:35-38` | 512 new tok after 32K, grouped gather |
| bench | tool | harness | `bench/results/2026-10-08-community-r9700-linux/harness/bench_engine.py:2` | native-protocol bench client |
| bench | behavior | RX 9070 gfx1201 IQ3_XXS | `bench/results/2026-10-10-community-rx-9070/README.md:106-110` | 832.9 prompt @130K, 57.9 decode; gfx1201 IQ3_XXS per bench/results/2026-10-10-community-rx-9070/README.md:1-4 |
| bench | env | STRATA_HIPBLASLT_TUNING gfx1201-100500 | `bench/results/2026-10-10-community-rx-9070/README.md:62` | 32 rows (bench/results/2026-10-10-community-rx-9070/README.md:67) |
| bench | behavior | RX 9060 XT gfx1200 | `bench/results/2026-10-10-community-rx-9060-xt/README.md:51-59` | 879.4 prompt @8K prefill 8192; gfx1200 per bench/results/2026-10-10-community-rx-9060-xt/README.md:11 |
| bench | behavior | gfx1151 fast 128K | `bench/results/2026-10-09-community-gfx1151-ud-iq4xs-0.1.41/README.md:50-60` | 1108.6/47.5; gfx1151 per bench/results/2026-10-09-community-gfx1151-ud-iq4xs-0.1.41/README.md:1 |
| bench | const | gfx1150 hipBLASLt table | `bench/results/2026-10-07-community-gfx1150/README.md:5` | tools/hip/gfx1150-hipblaslt-100401.txt, 1.5-1.7x prompt |
| bench | behavior | 4x7900XT opt-ins | `bench/results/2026-10-09-community-4x-rx-7900-xt/README.md:93-97` | 4307 prompt @128K; 4x RX 7900 XT per bench/results/2026-10-09-community-4x-rx-7900-xt/README.md:1 |
| bench | gotcha | NVFP4 only in another model (NInfer 27B, NVFP4 weights and KV) | `bench/results/2026-10-09-community-rtx5090-9800x3d-swift-1m/data/overlap-prototype-manifest.json:121-149` | NInfer 27B nvfp4kv; quant NVFP4 at bench/results/2026-10-09-community-rtx5090-9800x3d-swift-1m/data/overlap-prototype-manifest.json:148 |
| bench | gotcha | NVFP4 prompt filler | `bench/results/2026-10-06-llamacpp-0.6.0/prompt-4k.txt:111` | text, not measurement |
| bench | gotcha | RX9070 128K label overflow | `bench/results/2026-10-10-community-rx-9070/README.md:131` | 131428 tok (128000 label, bench/results/2026-10-10-community-rx-9070/README.md:91) rejected |
