# RDNA3 deployment — measured production config for the RX 7900 XTX (gfx1100)

This fork's production deployment serves **Qwen3.8-Flash-Next (MoE, IQ3_S)** on a single
**Radeon RX 7900 XTX** (gfx1100 / RDNA3, 24 GiB). This page documents the exact knob set with
its paired-measurement provenance. The full experiment corpus behind every number here —
including ~80 measured negatives — lives in the companion ledger:
**https://github.com/xyzzing/rdna3-inference-ledger**

## Provenance (read before comparing numbers)

- **Branch `refresh-0155`, engine 0.1.55**, deployed 2026-11-07. The live serving config was
  recorded by content hash at deployment; the knobs below are that config, transcribed.
- hipBLASLt tuning table for gfx1100 (32 tuned rows — merged upstream in
  [#755](https://github.com/Niko1221/Strata/pull/755)), referenced via `STRATA_HIPBLASLT_TUNING`.
- Headline (this config): prefill **~3,400 tok/s @94K**, decode **~43–47 tok/s @94K depth** —
  **2.1–2.3× faster decode than llama.cpp** on the same GPU and weights.
- **Build vintage matters**: worktree builds measured **11–12% slower** than clean repo builds.
  Every number below was measured on its own vintage under paired interleaved arms; do not
  compare across vintages, and do not expect identical numbers on other cards or models.
- Measurement protocol: paired in-process baseline/candidate alternation, predeclared gates,
  uninstrumented wall time as the authority ([ledger methodology](https://github.com/xyzzing/rdna3-inference-ledger/blob/main/methodology.md)).

## Engine environment knobs

| Knob | What it does | Measured effect | Notes |
|---|---|---|---|
| `STRATA_HIPBLASLT_TUNING=<table>` | gfx1100 tuned GEMM table (was absent upstream) | 128K prefill 926→1,687 tok/s (**+82%**) | upstream [#755](https://github.com/Niko1221/Strata/pull/755) |
| `STRATA_SELECT_WMMA=1` | fp16 tensor-core attention block-scorer | **10–11×** kernel @128K; e2e +8.7–11.2% | fp16 arm measured 1.64× faster than the bf16 arm at 128K select |
| `STRATA_HIP_WMMA=1` | WMMA prompt attention (gfx12 kernel ported to gfx1100) | 128K prefill **+22.8%**; attention share 27%→9% | |
| `STRATA_PA_FAST=1` | prompt attention with single FP16 q/p | prefill-stack member | |
| `STRATA_PF_FUSED=1` | fused prefill expert path | prefill-stack member | an early variant of this knob hung at 128K and was disqualified before deployment; the stack as deployed measured **+13.82%** |
| `STRATA_PREFILL_RING=512` | prefill ring sized 512 | prefill-stack member | |
| **prefill stack combined** (`PA_FAST`+`PF_FUSED`+`PREFILL_RING=512`) | three knobs deployed together | +5.43% and +9.08% individually → **+13.82% combined** | super-additive; measure combinations, not just singles |
| `STRATA_PREFILL_AUTO_MAX=32768` | cap for chunked deep prefill (with `--prefill auto`) | chunking **+26.4%** deep prefill (2,365→2,983 tok/s) | |
| `STRATA_ROUTE_TAIL_SKIP=7` | skip never-routed experts in verify windows | **+14–19% decode** — the biggest single decode win | |
| `STRATA_ROUTE_PRIOR=0.5` | routing prior tuning | deployed as part of the routing config | |
| `STRATA_GDN_CHUNKED=1` + `STRATA_PF_GEMM=1` + `STRATA_PF_PAD=1` | chunked GDN prefill + the padded fp16 tensor-core dense GEMM path (production dense route) | +5.85% @32K, +4.41% @128K (combined row) | the padded WMMA dense route is ~4.5× faster than the opt-in MMQ dense arm — see the default-off list below |
| `STRATA_KV_RING_ADAPT=1` | adaptive KV ring sized to the request (not a fixed 32K) | **+9.4–10.6%** decode at depth | upstream [#3919](https://github.com/Niko1221/Strata/pull/3919), merge-ready |
| `STRATA_SH_STREAM=1` | shared expert on its own stream | deployed with `--spec 5`: **+11.3%** decode @1K (combined row) | default-on for some other gfx targets; on gfx1100 the defaults table gates it until measured — this fork measured it |

## Serve-layer flags

| Flag | Value | Measured effect / reason |
|---|---|---|
| `--spec` | `5` (with `--spec-min-p 0.70`) | +9.5%, then +11.3% decode @1K; depth 6 rejected (−6.0% @128K); `--spec-min-p 0.99` measured **−15%** — chains earn ~+17% at this shape |
| `--lookup-chain` | `2` | −5.6% *before* TAIL_SKIP → **+6.3% @1K / +19.9% @128K after** — re-pair old knobs when the stack changes |
| `--kv` | `int8` | KV streaming required at depth (all-VRAM KV exits at ≥64K: dense weights starve) |
| `--kv-resident` | `49152` | resident KV window; past the cliff, prompt reads drop 468→98 tok/s — size the window before the cliff |
| `--prefill` | `auto` | chunked deep prefill, capped by `STRATA_PREFILL_AUTO_MAX` |
| `--expert-cache` | `auto` (+ trained expert profile) | file-backed IQ experts replacing 47.9 GiB pinned RAM — fixed 3× OOM kills; cold-decode trade 14.3→7.6 tok/s |
| `--max-context` | `131072` | 262K also works (decode 75.6→69.3 at 262K) if VRAM allows |
| `--prompt-cache` / `--vram-reserve-mib` | `6` / `3072` | session-reuse machinery: turn resumption measured 13× turn-wall at agent depths |

## Host side

- **cgroup `MemoryHigh=62G` / `MemoryMax=66G`** on the serving unit — sheds file cache before
  the OOM killer acts (oom_kills 0 since).
- **The expert pack must live on NVMe**: mmap off hard storage measured 5.6–8 vs 25–33 tok/s.

## Default-off recorded arms (do not enable expecting wins)

In-tree but measured negative on this hardware; kept as opt-in records:

- `STRATA_DENSE_MMQ=1` — opt-in MMQ dense path: bitwise-clean, but the production padded
  fp16-WMMA route above is **~4.5× faster**.
- `STRATA_DOWN_WS=1` — SM60-style down-projection port: **2.4× slower** at the live shape;
  gfx1100's production down already reads 457–607 GB/s.

Further recorded arms (launch-ahead, draft-entropy gate, i8 dense, grouped GEMV, tree
verification, …) live in the [ledger's rejected corpus](https://github.com/xyzzing/rdna3-inference-ledger/tree/main/rejected) — several measured negative *after* being bitwise-perfect.
