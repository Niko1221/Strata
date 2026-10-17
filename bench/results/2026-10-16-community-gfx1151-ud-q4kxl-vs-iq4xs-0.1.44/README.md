# Community benchmark: EVO-X2, Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB UMA — UD-Q4_K_XL versus UD-IQ4_XS on 121 GiB, and which quant to pick when both will not fit

Measured 2026-10-16 (UD-Q4_K_XL) and 2026-10-11 (UD-IQ4_XS), on this machine. Adds a quantisation
choice the earlier gfx1151 reports did not cover: **[2026-10-09-community-gfx1151-ud-iq4xs-0.1.41](../2026-10-09-community-gfx1151-ud-iq4xs-0.1.41/README.md)**
ran UD-IQ4_XS only and declared 55 GiB as what is left for the expert cache; the host has since
become a multi-tenant box, which is what decides the answer here.

**Headline.** On 121 GiB of unified memory, UD-Q4_K_XL (103.69 GiB of GGUF) fits alongside a
desktop and a live inference server with `--resident-budget-gib 71`, and it beats UD-IQ4_XS on
**correctness** (96.7 % versus 95.6 % standard, 99.5 % versus 98.4 % hard) and on **prefill from
16 K to 64 K**, while **losing decode everywhere it was measured** (−6 % short code, −19 %
prose, −6 % at 128 K) and costing about 24 GiB of OS headroom. Take UD-IQ4_XS when throughput or
total RAM matters; take UD-Q4_K_XL when the answer has to be right.

## How to read this

- **Prompt** = prefill tok/s, **Output** = decode tok/s, from the engine's own reported counts
  identical to that report's convention.
- The two arms are **not same-day and not same-engine** (see [Caveats](#caveats-and-what-this-does-not-show)).
  Differences below about 5 % are inside the run-to-run spread of this box and are reported
  as "no clear difference" rather than as a delta.
- Correctness comes from the repository's own `bench/ctxbench` batteries, same seed
  (`20261006`), same levels, thinking off, in both arms. So the correctness comparison *is*
  controlled and is the stronger of the two findings.

## Hardware and OS

- EVO-X2, AMD Ryzen AI Max+ 395 (16 cores / 32 threads), Radeon 8060S / **gfx1151**;
  128 GB installed UMA, Linux `MemTotal` 127,098,452 KiB (121.2 GiB). GPU allocations consume
  the same RAM, so the memory numbers below are the actual constraint on quant choice.
- Ubuntu 26.04.1 LTS, kernel `7.0.0-38-generic`, in-tree kernel `amdgpu` driver (no DKMS).
  Power mode `performance`. Desktop and background processes kept running: this is a
  **production, multi-tenant measurement, not an isolated-OS benchmark** — deliberately, since
  the question being asked is what to run on a machine that is already doing something else.
- ROCm: the host's single install `/opt/rocm` → `/opt/rocm/core-7.14`, **7.14.1**, HSA/ROCr
  `1.21.0`, `libamdhip64.so.7.14.60850`. Both arms were launched with
  `lib_dirs: ["/opt/rocm/core-7.14/lib"]` and
  `STRATA_HIPBLASLT_TUNING=…/tools/hip/gfx1151-hipblaslt-100401.txt`. No
  `HSA_OVERRIDE_GFX_VERSION`, no BIOS or kernel parameter change.
- Note for anyone repeating this: `/opt/rocm/.info/version` does **not** exist in this packaging;
  the version is `/opt/rocm/core-7.14/.info/version`. Reading the first path returns nothing and
  invites a false "no ROCm installed" conclusion.

## Engines, models and configuration

| Arm | Engine | Model | GGUF bytes | `--resident-budget-gib` | Serving |
|---|---|---|---|---|---|
| A (UD-IQ4_XS) | Strata **v0.1.43**, HIP build (`/ai/strata/build-hip`) | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS) rev `38bb39ee`, 3 files | **93,682,584,224** (87.25 GiB) | 55 | direct on `:8091`, `llama-swap` stopped, nothing else resident |
| B (UD-Q4_K_XL) | Strata **v0.1.44.1**, HIP build (`/ai/strata-0.1.44.1/build-hip`) | same repo, [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/c8b5954a88c2775c546b92593eda40ea041d3176/UD-Q4_K_XL) rev `c8b5954a`, 4 files | **111,334,654,784** (103.69 GiB) | 71 | through `llama-swap` on `:8080` as model `deep` |

Shared arguments in both arms: `--expert-profile /ai/strata/data/expert-profile.bin`,
`--expert-cache auto`, `--prefill auto:16384`, `--mtp /ai/Strata-data/mtp/rt`,
`--max-context 262144`, `--kv int8`, `--kv-resident 32768`.

Arm-specific: A used `--spec 4 --spec-min-p 0.5` plus `STRATA_PF_SWITCH_MIN_T=4096`,
`STRATA_PREFILL_STREAM_MIN=128`; B used `--spec 6 --spec-min-p 0.6 --mtp-window 8192`.
Both are the "documented fast" configuration of their own release, per
[docs/STRIX_HALO.md](../../../docs/STRIX_HALO.md).

**The GGUF byte counts above are identical to the ones in the two committed gfx1151 reports**
(87.25 GiB in the 0.1.41 report), which is how the files were confirmed as the same revisions.
Both models are mixed-format: the quant name is a pack label, not a per-tensor statement — most
expert gate/up tensors are IQ3_S and most down tensors IQ4_NL, with selected Q8_0.

## Speed

Median of the repeated short-prompt rows, and the single long-prompt row per size. `avail` is
`/proc/meminfo` `MemAvailable` reported by the harness around the run.

| | A: UD-IQ4_XS (v0.1.43) | B: UD-Q4_K_XL (v0.1.44.1) |
|---|---|---|
| `avail` before load | 38.2 GiB | 13.9 GiB |
| `avail` after run | 37.2 GiB | 14.0 GiB |
| code decode (tg) | **69.3** t/s, draft accept 84 % | **67.0** t/s, draft accept 84 % |
| prose decode (tg) | **50.8** t/s, draft accept 61 % | **41.3** t/s, draft accept 54 % |
| prefill 4 K (4,058 tok) | 931.5 t/s | **1,067.4** t/s |
| prefill 16 K (15,504 tok) | 1,186.6 t/s | **1,225.4** t/s |
| prefill ~41 K (31,833 / 31,835 tok) | **1,261.2** t/s | 1,241.3 t/s |
| prefill 64 K (61,962 / 61,970 tok) | **1,264.5** t/s | 1,221.8 t/s |
| prefill 128 K (134,810 / 134,792 tok) | **1,211.6** t/s | 1,172.3 t/s |
| decode during 4 K prompt | 50.0 t/s | 41.2 t/s |
| decode during 16 K prompt | 56.5 t/s | 42.3 t/s |
| decode during 41 K prompt | 46.0 t/s | 41.7 t/s |
| decode during 64 K prompt | 40.5 t/s | 34.4 t/s |
| decode during 128 K prompt | 38.1 t/s | 35.8 t/s |
| 245 K prompt | refused: prompt 267,345 tok > 262,144 context | refused: prompt 267,318 tok > 262,144 |

What this shows:

- **Prefill does not fall off a cliff with the bigger quant; it trades shape.** B is 15 % ahead
  at 4 K and 3 % ahead at 16 K, then 2–4 % behind from 41 K to 128 K. Both arms stay in a
  0.93–1.26 K t/s band, so a 19 % larger model costs little prefill on this part; it matches the
  compute-bound prefill reading in [docs/STRIX_HALO.md](../../../docs/STRIX_HALO.md).
- **Decode is where the bigger quant costs, at every size measured.** Short code is a near-wash
  (69.3 vs 67.0, −3 %, inside spread). Everything else is worse: prose 50.8 → 41.3 (−19 %),
  4 K-prompt decode 50.0 → 41.2 (−18 %), 128 K-prompt decode 38.1 → 35.8 (−6 %). With 71 GiB of
  budget against 103.69 GiB of model, decode is paying for expert reads the smaller quant keeps
  resident. B's draft-accept rate is the same on code (84 %) but lower on prose (54 % vs 61 %),
  so part of the prose gap is speculation accepting less, not memory alone.
- **Both arms' decode falls with context length** (A: 56.5 → 38.1; B: 42.3 → 35.8), which is the
  expected KV/expert-cache shape and not a quant artefact.
- **The 245 K row failing in both arms is the context ceiling, not a quant or memory limit** —
  about 267.3 K prompt tokens in a 262,144 window, refused before inference. Do not read it as an
  OOM. The largest row that ran on both is 128 K.

## Correctness (controlled: same battery, seed and levels in both arms)

`bench/ctxbench`, seed `20261006`, thinking off, counting rows the harness itself marked
`ok` versus `BAD`/`FAIL`.

| Battery | A: UD-IQ4_XS | B: UD-Q4_K_XL |
|---|---|---|
| standard (182 scored tasks) | 174 ok, 8 bad — **95.6 %** | 176 ok, 6 bad — **96.7 %** |
| hard (185 scored tasks) | 182 ok, 3 bad — **98.4 %** | 184 ok, 1 bad — **99.5 %** |

This is the finding worth the most, because it is apples-to-apples: **the larger quant is more
correct on both batteries, at the cost of the decode numbers above.** Concretely, arm A failed
a level-0 reasoning/optimisation task with `got='None' want='102'` (and a rule violation on the
answer line) on a prompt with no context at all, while arm B answered it correctly in 50.3 s.
A failure at level 0 cannot be a long-context effect; it is the model, so it is a quantisation
effect. If your workload is reasoning-critical, take B and live with the decode cost.

## Memory headroom, which is the actual decision on this box

`--resident-budget-gib` is the model's claim on the unified memory; what is left is what the OS
and everything else gets. Both numbers below are the harness's own `avail` readings.

| | A: budget 55 | B: budget 71 |
|---|---|---|
| `MemAvailable` around the run | 37–38 GiB | 13.9–14.6 GiB |
| Desktop + browser + an app server usable alongside | comfortably | tight |

Arm A was measured with `llama-swap` stopped (the chain script aborts if it is running), so its
38 GiB is "model alone on a quiet box". Arm B was measured *with the inference server serving
live traffic through llama-swap*, which is why the budget went to 71 and the headroom fell to
about 14 GiB. This is the practical guidance:

- **Run B (Q4_K_XL) only if you can keep about 14 GiB free and accept it**, and if you are not
  also hosting a second resident model. On this host a second large resident model is exactly
  what pushed the budget around; a co-resident image model at ~8 GB was fine, another 70 GB
  LLM would not be.
- There is **no swap** on this machine, so an over-claim does not degrade, it OOMs. Keep the
  explicit `--resident-budget-gib` (never `auto`) for the same reason the 0.1.41 report reached
  for it, and for the reason in [docs/STRIX_HALO.md](../../../docs/STRIX_HALO.md):
  `auto` took memory the desktop needed and caused a global OOM on a 128 GB box (#1715).
- If you want both the correctness of B and the headroom of A, the honest answer from this data
  is that on 121 GiB you get to keep one of them.

## Reproduce

See [REPRODUCE.md](REPRODUCE.md) for the exact harness invocations, the per-arm engine JSON, and
the raw files' SHA-256.

## Caveats and what this does not show

- **Engine version is confounded with quant.** A is v0.1.43, B is v0.1.44.1, five weeks apart,
  and B additionally gained `--spec 6`/`--mtp-window 8192`. So the decode deltas are
  *quant-plus-engine-config*, not quant alone. The correctness comparison is **not** affected
  in the same way — but it is still two versions, so treat the 1.1-point standard gap as
  indicative and the level-0 task failure as the hard evidence.
- **Different harness paths**: A served directly on `:8091`, B through llama-swap on `:8080`.
  B's wall-clock numbers include llama-swap; the reported `pp`/`tg` are engine-reported token
  counts over engine-reported time, so llama-swap adds latency to `wall_s` but not materially
  to `pp`/`tg`. Unverified: exactly how much.
- **Single run per size**, unlike the 0.1.41 report's medians of 3. Short-prompt rows were
  repeated (3× code, 3× prose) and their medians are what the table shows; long rows were not.
- **Not an isolated OS.** Desktop, browser and Home Assistant traffic were live in arm B.
- GPU is an **iGPU on shared memory**: the harness's `avail` is a RAM reading, and there is no
  discrete PCIe link here to measure.
- 245 K never ran in either arm; nothing here characterises behaviour above 128 K.
