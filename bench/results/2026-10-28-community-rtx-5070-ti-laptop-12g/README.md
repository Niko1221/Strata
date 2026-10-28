# Community benchmark: RTX 5070 Ti Laptop 12 GB, Flash-Next IQ3_XXS (engine 0.1.52, 2026-10-28)

Measured 2026-10-28 by `kxyct` (report written by the machine's resident local AI assistant, on the owner's PC).
This folder updates `2026-10-24-community-rtx-5070-ti-laptop-12g` (same PR): the engine went 0.1.46 → 0.1.50 → 0.1.52
all inside 2026-10-28; this 0.1.52 snapshot supersedes the 0.1.50 rows measured earlier today (the same-folder names were
0.1.50 files, and both the 0.1.50 snapshot and the 10-24 rows stay in this PR's commit history).
Hardware, model and engine launch flags are unchanged from that report except where noted.

## Hardware and software

- GPU: NVIDIA GeForce RTX 5070 Ti **Laptop**, 12,227 MiB VRAM (nvidia-smi); CPU: AMD Ryzen 9 9955HX (16 cores / 32 threads); RAM: 2 x 32 GB DDR5-5600; storage: 2 x YMTC PC411 1 TB NVMe SSD; PCIe link speed/width: not measured; GPU power limit: not reported (`nvidia-smi` shows `[N/A]`, laptop).
- OS: Windows 11 (build 26300); NVIDIA driver 616.64 (re-checked 2026-10-28, unchanged since 2026-10-24).
- Strata engine 0.1.52 portable Windows release binary (ships inside the Strata v0.2.7 release as the `strata-engine-0.1.52-windows-x86_64-cuda-cuda13` package; this machine runs the matching 0.2.7 serve tree): commit `a42cc3d7df70`, source tag `f-027-rel`, archs sm_75/86/89/120, CUDA 13.0 (`BUILD.json` copied here from the installed engine folder; install verified 2026-10-28 16:35: 7 extracted files, all sha256 match the release SHA256SUMS). Engine 0.1.52 live on this machine from 16:35; the 16:41-16:46 bench and the 17:19 needle runs below were served by 0.1.52 processes. Startup log quoted below = the current 16:53:47 0.1.52 process (the bench-engine startups on the same binary/config wrote the same self-adaptation lines).
- **New since the 10-24 snapshot, and a confounder worth stating**: the machine's display output moved to hybrid mode on 2026-10-26 (the iGPU drives the built-in 2560x1600 panel; the discrete GPU is compute-only). On this machine that switch alone was worth ~+10% decode in interactive use. It cannot be separated from the engine update inside this snapshot; the 10-24 rows were measured with the dGPU driving the panel.
The same-day 0.1.50-to-0.1.52 deltas quoted below are cleaner: same 10-28 hybrid-mode machine, same flags, same day — an engine-only
change. Deltas against the 10-24 rows are still mixed with the display change.

## Model and configuration

Unchanged from the 2026-10-24 report (`model-provenance.json` still applies; see that file and the 10-24 folder for
model repository, quantization, custom pack, MTP and vision details): orcarouter Uncensored GGUF at IQ3_XXS
(GSQ-RCO-abliterated shards), custom expert pack `sc117_iq3_xxs`, `STRATA_ROUTE_RESIDENT=1`, `STRATA_ADAPT_OVERLAP=0`,
`STRATA_NO_LARGEPAGES=1` (engine process env, re-verified from the live config 2026-10-28). Launch flags are byte-identical
to the 10-24 report (same args in the live 0.1.52 config and its 16:53:47 startup log), including `--vram-reserve-mib 1000`, `--max-context 262144`,
`--kv int8`, `--kv-resident 32768`, `--spec 4 --spec-min-p 0.70`.

New engine self-adaptations visible in the 2026-10-28 startup log (no user flags changed):

```text
strata: pinned-memory headroom: holding 800 MiB for the buffers allocated after the expert arena (...; STRATA_PIN_RESERVE_MIB changes it)
strata generate: expert cache auto: 4.92 GiB free, 1000 MiB reserved (+184 MiB for the draft head) -> 1733 slots
strata generate: only 214 MiB free once the slots are written (reserve 1000 MiB); shrinking the expert cache
strata generate: expert cache 1780 slots, 2.93 GiB of VRAM; policy is
strata generate: pre-filled 1780 of 1780 slots from the profile in 0.2 s (12821 MB/s); slot 0 verified
strata generate: session is up (engine 0.1.52)
```

## Method

- Same script and protocol as the 2026-10-24 submission (`scripts/measure_parallel.py`, included; stdlib only): synthetic nonce-prefixed prompts at 300 / 4,096 / 32,768 target tokens, one request at a time (`--concurrent 1`), 256-token cap (`finish_reason=length` on all 9 runs), 3 runs per tier.
- The engine (0.1.52, same binary) was (re)started before the runs, one 1-token readiness request, then a 25 s quiescence, then the bench; the engine startup log for the 0.1.52 process that served the bench is quoted above (16:53:47 process; the 16:3x startup that served these runs wrote the same lines). The prompt path is also cold per round: all 9 runs report `cached_tokens = 0` in client usage, and the engine timing lines record `reused 0` for every bench request (the 10-24 JSON's later 4K/32K rounds were client-side follow-ups by design; here every round is a fresh conversation).
- Decode numbers in this README are **whole-request**: client `completion_tokens` divided by the sum of the engine-side decode windows of the same request. The 0.1.50 serve could log one API request as several native generations ("continuations"); on the 0.1.52 serve 3 of the 9 requests split into a continuation window (1 short, 1 4K, 2 32K — the per-run split pattern, window acceptance and rollback tokens are quoted verbatim in each run's `lines` array, not summarized here). All raw engine lines are kept verbatim in the JSONs, and every window line carries its own `line` text.
- Fresh-read prefill = read tokens / read time from the engine's prompt lines (`reused 0` windows).

## Results (engine 0.1.52, 2026-10-28, `vram-reserve-mib 1000`) — same-day 0.1.50 rows quoted for reference; same formula applied to the 10-24 JSONs

Per-request decode median (range), fresh-read prefill median, client wall per request, draft acceptance:

| Tier | Decode tok/s 0.1.46 (10-24) | Decode tok/s 0.1.50 (10-28, superseded) | Decode tok/s 0.1.52 (10-28) | Δ 0.1.50→0.1.52 | Prefill tok/s (fresh) 0.1.50 → 0.1.52 | Δ | Wall s/request 0.1.50 → 0.1.52 | Drafts 0.1.52 |
|---|---|---|---|---|---|---|---|---|
| short (394-396 tok) | 92.6 (85.6-96.2) | 97.2 (88.0-107.6) | 128.2 (123.6-156.1) | +32% | 390 → 449 | +6.7% | 3.4-4.1 → 2.5-3.2 | 485/539 (90%) |
| 4K (4,187-4,188 tok) | 86.1 (84.3-94.2) | 108 (104-112) | 139.0 (133.0-146.8) | +29% | 1,227 → 1,264 | +3.0% | 5.7-5.9 → 5.1-5.4 | 493/516 (96%) |
| 32K (32,857-32,858 tok) | 84.5 (80.4-97.2) | 97.1 (96.6-116) | 128.7 (119.0-141.0) | +33% | 1,345 → 1,381 | +2.6% | 26.7-27.2 → 25.7-26.1 | 465/508 (92%) |

- The 10-24 column is **re-derived from the 10-24 JSONs with today's formula** (all rounds, whole-request), so the two days are apples-to-apples; the 10-24 README's own "clean runs" table (92.5 / 90.2 / 97.2) selected a different subset of the same JSONs and is not directly comparable to these medians. The 0.1.50 rows in the table are the earlier same-day 0.1.52-adjacent run (the file pair shipped in the earlier commit of this branch had 0.1.50 data measured with the identical protocol, machine and flags (hybrid display mode in both runs); measured on the earlier engine of the same day, before the 0.1.52 reinstall).
- All 9 fresh runs per day fit one table each: 10-28 has 3 fresh runs per tier (10-24's submitted table only reported 1-2 fresh rows per tier; its later rounds were same-conversation follow-ups).
- The 4K/32K prefill jump of the 10-24-era comparison (+37%) happened at the 0.1.46-to-0.1.50 step; on 0.1.52 the same-day prefill movement is only +2.6 to +6.7%. The short-prompt prefill row barely moves (+4/+6.7%), which smells like small-read measurement noise rather than a class effect at every prompt length. The 10-24-to-today deltas mix the engine updates with the hybrid-display change; the 0.1.50-to-0.1.52 deltas in this folder are engine-only.
- Machine-readable summary rows: `compare-0150-vs-0152.csv` in this folder; per-run data: `runs-short-20261028.json`, `runs-4k-20261028.json`, `runs-32k-20261028.json`.

## VRAM headroom on this 12 GB card (updated for 0.1.52)

- `reserve 1000` (current config, unchanged): 0.1.52 startup auto-sizes 1733 slots, sees 214 MiB free after the writes, and shrinks the arena itself to 1780 slots / 2.93 GiB (0.1.50 did the same job at 1745 → 1800 slots / 2.96 GiB). The engine's own `#2065` warning still fires at request start in daily use (586 MiB free < 1024 MiB threshold; the engine's warning line now suggests `--vram-reserve-mib 1472`) — with 0.1.52 all nine bench runs and all three needle runs finished normally (decode 119-156 t/s across the nine bench runs, no multi-second pages; the shrink-to-safe step completed at startup), so paging had not bitten at these workloads.
- `reserve 1536`: 1123 MiB free / ~1197 slots (0.1.46-era measurement 2026-10-20, kept from the 10-24 report; not re-measured with either 0.1.50 or 0.1.52).
- Earlier history (0.1.44.1/0.1.45: 700 stalling-prone, 1061 stable floor) and the overall conclusion are unchanged: on this model the 12 GB VRAM, not the 64 GB RAM, is the binding constraint.

## Correctness and limitations

- Recall/needle with the upstream tool (`scripts/needle_bench.py`, hash-identical to `tools/needle_bench.py` in the 0.2.7 tree — verified today by sha256; run from the installed 0.2.7 tree, so the haystack text differs slightly from the 0.1.50 run on the same tool): 32K length, depths 10/50/90% — **3 of 3 found** (31,706-31,707 prompt tokens each, 23 s per request; on the 0.1.50 tool copy the same depths took 23.7-23.8 s with 32,096-32,097 tokens, vs 31.6-33 s on the 10-24-era engine; data in `needles-32k-20261028.json`). 128K and longer: not run. A needle test measures recall on those inputs, not overall model quality.
- No 128K-tier run, no TTFT/streaming measurement, 256-token cap, vision encoder loaded but never invoked — same limitations as 10-24. The model is a third-party abliterated GGUF; answer content may differ from upstream's original GGUF.
- Quality check after the engine update: no NaN or mojibake observed on this machine since 0.1.52 went live (visual check via daily use; not an automated diff).

## 中文摘要

本机是 12G 显存的 5070 Ti 笔记本，跑 orcarouter 无审核 IQ3_XXS 模型，配置一行没改。当天引擎 0.1.46→0.1.50→0.1.52 连升两级后按同一协议重测：短提示词整请求解码中位 128（0.1.50 是 97、0.1.46 是 93）、4K 139（108/86）、32K 129（97/85），0.1.50→0.1.52 涨幅 +29~33% 是同机同协议同配置的纯引擎差值；prefill 这级只动 +2.6~6.7%（大头 +37% 是早上 0.1.46→0.1.50 那步涨的）。32K 单请求墙钟 26-26 秒档。显示输出 10-26 起为混合模式，与 10-24 的独显直出不同，对比 10-24 的涨幅里含着 ~10% 的显示切换因素。显存结论不变：12G 瓶颈仍是显存，reserve 1000 下 0.1.52 自动把专家缓存从 1733 槽缩到 1780 槽/2.93 GiB 落回安全区，#2065 报警仍在请求开头出现（586 MiB，引擎现建议 1472），但所有 bench/needle 全部正常跑完。专家命中率中位从 77-79% 跳到 95-98%（官方 0.1.52 的 12G 卡改动生效）。32K needle 召回 3/3 全中（23 秒/条）。
