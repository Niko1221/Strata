# Community benchmark: RTX 5070 Ti Laptop 12 GB, Flash-Next IQ3_XXS (engine 0.1.50, 2026-10-28)

Measured 2026-10-28 by `kxyct` (report written by the machine's resident local AI assistant, on the owner's PC).
This folder updates `2026-10-24-community-rtx-5070-ti-laptop-12g` (same PR): the engine was updated from 0.1.46 to
the 0.1.50 release package, so all speed rows are re-measured; the older snapshot stays in this PR's commit history.
Hardware, model and engine launch flags are unchanged from that report except where noted.

## Hardware and software

- GPU: NVIDIA GeForce RTX 5070 Ti **Laptop**, 12,227 MiB VRAM (nvidia-smi); CPU: AMD Ryzen 9 9955HX (16 cores / 32 threads); RAM: 2 x 32 GB DDR5-5600; storage: 2 x YMTC PC411 1 TB NVMe SSD; PCIe link speed/width: not measured; GPU power limit: not reported (`nvidia-smi` shows `[N/A]`, laptop).
- OS: Windows 11 (build 26300); NVIDIA driver 616.64 (re-checked 2026-10-28, unchanged since 2026-10-24).
- Strata engine 0.1.50 portable Windows release binary (shipped with the 0.2.5/0.2.6-era Qwen engine package): commit `1aff6d14117b`, source tag `f-0243-rel`, archs sm_75/86/89/120, CUDA 13.0 (`BUILD.json` copied here from the release engine folder). Engine process started 2026-10-28 12:48:24 (engine startup log).
- **New since the 10-24 snapshot, and a confounder worth stating**: the machine's display output moved to hybrid mode on 2026-10-26 (the iGPU drives the built-in 2560x1600 panel; the discrete GPU is compute-only). On this machine that switch alone was worth ~+10% decode in interactive use. It cannot be separated from the engine update inside this snapshot; the 10-24 rows were measured with the dGPU driving the panel.

## Model and configuration

Unchanged from the 2026-10-24 report (`model-provenance.json` still applies; see that file and the 10-24 folder for
model repository, quantization, custom pack, MTP and vision details): orcarouter Uncensored GGUF at IQ3_XXS
(GSQ-RCO-abliterated shards), custom expert pack `sc117_iq3_xxs`, `STRATA_ROUTE_RESIDENT=1`, `STRATA_ADAPT_OVERLAP=0`,
`STRATA_NO_LARGEPAGES=1` (engine process env, re-verified from the live config 2026-10-28). Launch flags are byte-identical
to the 10-24 report (same args in the 12:48:24 startup log), including `--vram-reserve-mib 1000`, `--max-context 262144`,
`--kv int8`, `--kv-resident 32768`, `--spec 4 --spec-min-p 0.70`.

New engine self-adaptations visible in the 2026-10-28 startup log (no user flags changed):

```text
strata: pinned-memory headroom: holding 800 MiB for the buffers allocated after the expert arena (...; STRATA_PIN_RESERVE_MIB changes it)
strata generate: expert cache auto: 4.94 GiB free, 1000 MiB reserved (+184 MiB for the draft head) -> 1745 slots
strata generate: only 220 MiB free once the slots are written (reserve 1000 MiB); shrinking the expert cache
strata generate: expert cache 1800 slots, 2.96 GiB of VRAM; policy is
strata generate: pre-filled 1800 of 1800 slots from the profile in 0.2 s (13407 MB/s); slot 0 verified
strata generate: session is up (engine 0.1.50)
```

## Method

- Same script and protocol as the 2026-10-24 submission (`scripts/measure_parallel.py`, included; stdlib only): synthetic nonce-prefixed prompts at 300 / 4,096 / 32,768 target tokens, one request at a time (`--concurrent 1`), 256-token cap (`finish_reason=length` on all 9 runs), 3 runs per tier.
- The engine was freshly (re)started at 12:48:24, one 1-token readiness request, then a 25 s quiescence, then the bench; the chat session that drives this machine was idle for the whole measurement window (single-slot engine; nothing else issued requests). The prompt path is also cold per round: all 9 runs report `cached_tokens = 0` in client usage, and the engine timing lines record `reused 0` for every bench request (the 10-24 JSON's later 4K/32K rounds were client-side follow-ups by design; here every round is a fresh conversation).
- Decode numbers in this README are **whole-request**: client `completion_tokens` divided by the sum of the engine-side decode windows of the same request. The 0.1.50 serve can log one API request as several native generations ("continuations", `serve/server.py:5637`, `:5685`); 3 of the 9 requests (short r2/r3, 32K r1) show a low-acceptance first window (52-65/69-79 drafts) followed by a full-acceptance continuation (84/84, re-reading 18-19 rollback tokens). All raw engine lines are kept verbatim in the JSONs, and every window line carries its own `line` text.
- Fresh-read prefill = read tokens / read time from the engine's prompt lines (`reused 0` windows).

## Results (engine 0.1.50, 2026-10-28, `vram-reserve-mib 1000`) — and the same formula applied to the 10-24 JSONs

Per-request decode median (range), fresh-read prefill median, client wall per request, draft acceptance:

| Tier | Decode tok/s 0.1.46 (10-24) | Decode tok/s 0.1.50 (10-28) | Δ | Prefill tok/s (fresh) 0.1.46 → 0.1.50 | Δ | Wall s/request 0.1.46 → 0.1.50 | Drafts 0.1.46 → 0.1.50 |
|---|---|---|---|---|---|---|---|
| short (394-396 tok) | 92.6 (85.6-96.2) | 97.2 (88.0-107.6) | +5% | 376 → 390 | +4% | 3.7-4.1 → 3.4-4.1 | 500/539 → 454/499 (93% → 91%) |
| 4K (4,187-4,188 tok) | 86.1 (84.3-94.2) | 108 (104-112) | +25% | 893 → 1,227 | +37% | 7.4-8.3 → 5.7-5.9 (−31%) | 470/506 → 507/547 (93% → 93%) |
| 32K (32,857-32,858 tok) | 84.5 (80.4-97.2) | 97.1 (96.6-116) | +15% | 979 → 1,345 | +37% | 36.3-36.8 → 26.7-27.2 (−26%) | 467/519 → 482/515 (90% → 94%) |

- The 10-24 column is **re-derived from the 10-24 JSONs with today's formula** (all rounds, whole-request), so the two days are apples-to-apples; the 10-24 README's own "clean runs" table (92.5 / 90.2 / 97.2) selected a different subset of the same JSONs and is not directly comparable to these medians.
- All 9 fresh runs per day fit one table each: 10-28 has 3 fresh runs per tier (10-24's submitted table only reported 1-2 fresh rows per tier; its later rounds were same-conversation follow-ups).
- The prefill jump at 4K/32K (+37%) is consistent across both long tiers; the short-prompt prefill row barely moves (+4%), which smells like small-read measurement noise rather than a 37%-class effect at every prompt length. Attributing the speedups between the engine update and the hybrid-display change is not possible from one before/after pair — both changed between 10-24 and 10-28.
- Machine-readable summary rows: `compare-0146-vs-0150.csv` in this folder; per-run data: `runs-short-20261028.json`, `runs-4k-20261028.json`, `runs-32k-20261028.json`.

## VRAM headroom on this 12 GB card (updated for 0.1.50)

- `reserve 1000` (current config): startup auto-sizes 1745 slots, then the engine itself shrinks the arena to 1800 slots / 2.96 GiB ("only 220 MiB free once the slots are written" — a self-adaptation step not present in the 10-24-era logs; it settles at a comfortable margin). The engine's own `#2065` warning still fires at request start in daily use (586 MiB free < 1024 MiB threshold right after the 12:48 restart; the 10-24 log showed 455 MiB at request start) — with 0.1.50 all nine bench runs and all needle runs finished normally (decode 84-131 t/s, no multi-second pages), so paging still had not bitten at these workloads.
- `reserve 1536`: 1123 MiB free / ~1197 slots (0.1.46-era measurement 2026-10-20, kept from the 10-24 report; not re-measured with 0.1.50).
- Earlier history (0.1.44.1/0.1.45: 700 stalling-prone, 1061 stable floor) and the overall conclusion are unchanged: on this model the 12 GB VRAM, not the 64 GB RAM, is the binding constraint.

## Correctness and limitations

- Recall/needle with the upstream tool (`scripts/needle_bench.py`, unmodified from `tools/needle_bench.py` at the same release commit `1aff6d14`; run from the installed 0.2.6-era tree, so the haystack text differs slightly from the 10-24 staging copy): 32K length, depths 10/50/90% — **3 of 3 found** (32,096-32,097 prompt tokens each, 23.7-23.8 s per request, vs 31.6-33 s on 10-24; data in `needles-32k-20261028.json`). 128K and longer: not run. A needle test measures recall on those inputs, not overall model quality.
- No 128K-tier run, no TTFT/streaming measurement, 256-token cap, vision encoder loaded but never invoked — same limitations as 10-24. The model is a third-party abliterated GGUF; answer content may differ from upstream's original GGUF.
- Quality check after the engine update: no NaN or mojibake observed on this machine since 0.1.50 went live (visual check via daily use; not an automated diff).

## 中文摘要

本机是 12G 显存的 5070 Ti 笔记本，跑 orcarouter 无审核 IQ3_XXS 模型，配置一行没改，引擎从 0.1.46 升到 0.1.50 后重测：短提示词整请求解码中位 97（上次 93）、4K 108（上次 86）、32K 97（上次 85），prefill 4K/32K 从 ~890/980 涨到 ~1230/1345，32K 单请求墙钟从 36 秒降到 27 秒。注意：10-26 晚本机把显示输出切到了核显（独显纯计算），这一开关此前单独实测值 ~10%，和引擎升级的功劳分不开，两个因素混在一起才有上面的涨幅。显存结论不变：12G 上瓶颈是显存，reserve 1000 时引擎 0.1.50 会自动把专家缓存从 1745 槽缩到 1800 槽/2.96 GiB 落回安全区，#2065 报警仍在请求开始时出现（586 MiB）但所有测试全部正常跑完。32K needle 召回 3/3 全中（23.7-23.8 秒/条，比上次的 31-33 秒快）。
