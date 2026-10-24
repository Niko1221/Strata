# Strix Halo (Ryzen AI Max, gfx1151): deploy on Windows

The ready-made Windows HIP build (`strata-windows-x64-hip.zip`) carries gfx1151
code, but the release notes and `docs/STRIX_HALO.md` describe the Linux path and
mark the Windows one as untested on a Strix Halo. This document is a working
Windows deployment with measured numbers, the switches that were worth setting,
and the traps that cost time. It complements `STRIX_HALO.md` (the Linux build)
rather than replacing it.

Two placeholders are used throughout: `<STRATA>` is the Strata install directory
(example `D:\Strata`) and `<MODELS>` is the model root (example `D:\Models`).
Substitute your own paths; keep the backslash form in JSON and batch files.

Measured on the Windows HIP build, Radeon 8060S (gfx1151), Ryzen AI Max+ 395,
128 GB LPDDR5X, Windows 11 x64. The first tuning rows were taken on 0.1.41, the
context and version rows after moving to 0.1.45; each row names its build. Every
number below was measured on this machine; numbers labelled *upstream* come from
the Strata documentation.

## 1. What this configuration reaches

Model IQ3_S (GSQ-RCO), native 262,144-token context, int8 KV. Fixed output cap,
three runs per configuration unless noted, medians. Rows 1-3 were measured on
0.1.41, rows 4-5 after moving to 0.1.45.

| Step | Change | Decode (zh) | Decode (long prompt) | Prompt (long) |
|---|---|---|---|---|
| start | defaults (auto expert cache, 320 slots, 33% hit rate) | 28.0 | not measured | not measured |
| 1 | `--mmap-experts` (cache grows to 24,576 slots) | 37.7 | 43.0 | 372 |
| 2 | the section 5 fast switches (they change bits) | 38.1 | 36.1 | **522 - 532** |
| 3 | `STRATA_SH_STREAM=0` | **39.6** | not measured | not measured |
| 4 | `--max-context` 131072 -> 262144 | 32.6 | not measured | not measured |
| 5 | engine 0.1.43 -> 0.1.45 (256K, same flags) | 34.0 | not measured | not measured |

- The same configuration with an English prompt decodes at **42.3 tok/s**
  (draft acceptance 71%), against 39.6 for Chinese. `STRIX_HALO.md` section 6
  reports 42.8 tok/s for IQ3_S at 128K on Linux, so the Windows build is within
  a few percent of the reference — the rest is language and Windows overhead.
- **Raising the context from 128K to 256K costs about 1.8 GiB of VRAM and
  nothing else.** 262,144 is the model's native trained length, so no rope
  scaling is involved. It is a GDN hybrid: only the QSA layers carry a growing
  KV cache (about 12.4 KB per token, so roughly 3.1 GiB at 256K). The expert
  cache stays at 24,576 slots, and free VRAM reads 45,907 MiB against 47,759 MiB
  at 128K. A back-to-back A/B measured **32.6 tok/s at 256K against 30.8 at
  128K** — noise, consistent with upstream's "decode is flat in context" (the
  GDN linear attention is O(1) per token).
- **The 0.1.43 -> 0.1.45 upgrade does not change speed here.** Measured 34.0
  against 32.6, both inside the same band. Upstream states the Flash-Next answers
  are byte-identical across these releases and that the 0.1.45 speed work
  "skipped Windows hardware"; the default speed-ups in 0.1.42 and 0.1.43 are
  CUDA and RDNA4 (gfx1200 / gfx1201) only, so gfx1151 keeps the earlier
  behaviour. What does change is the switch count: the startup line goes from
  17 to 18 gfx1151 defaults, the new one being `STRATA_TOPK_REG_MIN_BLOCKS`
  (the register top-k from 16K cells; upstream PR #2016, measured on gfx1151
  only).
- **Compare only back-to-back runs.** The same configuration measured 38.3 tok/s
  in one session and 32.6 about an hour later; that spread is machine state
  (page cache, temperature, background load), not a tuning effect.
- Reported "50+ tok/s" figures mostly come from smaller quants: the upstream
  table has IQ3_XXS at 50.9 and UD-IQ4_XS at 51.4 for 128K. If 50+ is the goal,
  switching quant is the direct route; IQ3_S measured at 42 tok/s at 128K here.
- Net effect of the three tuning steps: **28.0 -> 39.6 tok/s (+41%)**, with a
  larger gain on prompt ingestion.

## 2. BIOS and software

- Allocate a **32 GB fixed carve-out** to the iGPU in BIOS; the rest comes from
  shared memory. HIP then reports **104.5 GiB** usable (32 GB carve-out plus
  about 76.5 GB shared). No kernel parameters are involved on Windows, and
  `STRIX_HALO.md` also states that the engine changes no BIOS or kernel setting.
- Unzip the Windows HIP package into a directory with an **ASCII path and few
  spaces** (for example `<STRATA>`), then run `SETUP.bat` (which calls
  `setup.py`) to fetch the model and the derived data.
- Expected layout after setup:

```
<MODELS>\
├── ISTA-DASLab\IQ3_S\                 # original IQ3_S, 77.87 GiB
│   ├── ...-IQ3_S-00001-of-00002.gguf  # main model, 51.05 GiB
│   └── ...-IQ3_S-00002-of-00002.gguf  # the PLE table alone, 26.82 GiB
└── Strata-data\                       # derived data: packs, MTP draft layer
    ├── packs\iq3_s\
    └── mtp\rt\
```

**Trap:** once a model is moved out of `Strata-data\models\`, a later `setup.py`
run downloads it again, because the `.done` marker moved with the directory.
After reorganising the model tree, update `--native` and `--ple-gguf` in the
config to match.

## 3. The configuration

`strata-iq3_s.json`, with placeholders to substitute:

```json
{
 "exe": "<STRATA>\\engine\\strata.exe",
 "args": [
  "--pack",          "<MODELS>\\Strata-data\\packs\\iq3_s",
  "--native",        "<MODELS>\\ISTA-DASLab\\IQ3_S\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
  "--ple-gguf",      "<MODELS>\\ISTA-DASLab\\IQ3_S\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
  "--expert-profile","<STRATA>\\data\\expert-profile.bin",
  "--expert-cache",  "auto",
  "--prefill",       "16384",
  "--spec",          "4",
  "--spec-min-p",    "0.5",
  "--lookup-chain",  "3",
  "--mtp-q4",        "all",
  "--mtp-window",    "8192",
  "--mtp",           "<MODELS>\\Strata-data\\mtp\\rt",
  "--max-context",   "262144",
  "--kv",            "int8",
  "--mmap-experts"
 ],
 "backend": "hip",
 "env": {
  "STRATA_HIPBLASLT_TUNING": "<STRATA>\\tools\\hip\\gfx1151-hipblaslt-100500.txt",
  "STRATA_PF_FUSED": "1", "STRATA_PF_GEMM": "1", "STRATA_HC_UPMIX": "1",
  "STRATA_PA_FAST": "1", "STRATA_HIP_WMMA": "1", "STRATA_SELECT_WMMA": "1",
  "STRATA_HC_Q8": "1", "STRATA_PF_SWITCH_MIN_T": "4096",
  "STRATA_PREFILL_STREAM_MIN": "128",
  "STRATA_SH_STREAM": "0"
 }
}
```

### 3.1 `--mmap-experts` is the one that matters

Without it the other tuning is largely cancelled out. The default resident
expert arena pins about 46.84 GiB of experts into host memory; Windows counts
that memory against the process GPU budget (WDDM), so `device_free_bytes()`
reports only 2.25 GiB, the expert cache is squeezed to 320 slots (33% hit rate,
34% of experts read over PCIe) and decode lands at about 28 tok/s.

Evidence: while in that state the engine process reported
`Shared Usage = 48,723 MiB` against a `Dedicated Usage` of only 8,961 MiB — about
22.6 GiB of the 32 GB carve-out was still free.

With `--mmap-experts` the resident arena is dropped in favour of mmap over the
GGUF shards: the cache grows to **24,576 slots** (every expert resident in VRAM),
free VRAM reads 47,760 MiB, decode reaches **37.7 tok/s (+35%)** and PCIe expert
reads go to zero.

Cost: a cold start (first run after a reboot) reads the 51 GiB shard and spends
about 68 s filling the cache; with a warm page cache it is about 19 s.

### 3.2 The section 5 fast switches

`STRIX_HALO.md` section 5 lists the gfx1151 switches that round differently and
are therefore opt-in, gated by KL against the default. All of them are enabled
here — `STRATA_PF_FUSED`, `STRATA_PF_GEMM`, `STRATA_HC_UPMIX`, `STRATA_PA_FAST`,
`STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA`, `STRATA_HC_Q8` — together with
`STRATA_PF_SWITCH_MIN_T=4096` (keeps short prompts on the default numerics) and
`STRATA_PREFILL_STREAM_MIN=128`. Effect: **prompt ingestion +42% (372 -> 531
tok/s), reproduced twice**; decode essentially unchanged.

### 3.3 `STRATA_SH_STREAM=0` on Windows

Section 4 of `STRIX_HALO.md` documents `STRATA_SH_STREAM` (the shared expert on a
second stream) as **on by default on gfx1151**, and it does pay on Linux. On this
machine it costs: with the default (`1`) Chinese decode measured 34.1 tok/s, with
`0` it measured **39.6 tok/s (+13%)** — a clean A/B, far outside the noise. This
matches issue #816, where enabling it on Windows was 12-22% slower on an RX 6800.
**On Windows and gfx1151, set it to 0 explicitly.**

### 3.4 Decode flags

`--lookup-chain 3`, `--mtp-q4 all`, `--mtp-window 8192` and `--prefill 16384` are
the upstream fast configuration and are used as-is. `--expert-cache auto` reserves
2,560 MiB by default on Windows (the fix for the paging problem in #1376).

### 3.5 Leads that are not measured here

Three things look promising upstream but were not confirmed on this machine. The
first attempt to A/B the two switches below was invalidated rather than answered:
the three engine restarts took 57 s, 70 s and 141 s to fill the expert cache, so
the page cache was cooling between runs and the prompt-read numbers are not
comparable. They were re-run with the runs interleaved on engine 0.1.47; the
results are in section 3.6.

- `STRATA_PF_F16=1` (gfx11) — sends the covered prompt experts through the F16
  matrix cores instead of the int8 ones (`src/prefill/moe_f16_kq.cu`, on top of
  `STRATA_PF_FUSED=1`). `STRIX_HALO.md` section 5 reports one layer at 16,384
  tokens going 11.65 -> 9.87 ms for IQ3_S / IQ4_NL gate/up and 6.11 -> 4.71 ms
  for down. It rounds differently, so it is opt-in.
- `STRATA_REFILL_PARTS=1` — refills a lent expert-cache slot from the GGUF
  mapping instead of an assembled copy (`src/program/generate.cpp`). Upstream
  calls it exact and reports a 4K-8K prompt gaining 2.3% to 3.5%; it applies to
  a `--mmap-experts` configuration like this one.
- The other 16 gfx1151 default switches. `src/core/arch_defaults.cpp` notes that
  they are "Exact (bitwise) on Aurora" — tuned on Linux. `STRATA_SH_STREAM`
  already proved inverted on Windows (section 3.3), so the rest deserve the same
  back-to-back A/B; `STRATA_GFX1151_DEFAULTS=0` turns them all off at once.

### 3.6 Measured A/B on Windows (engine 0.1.47)

The three leads in section 3.5 were re-run after the 0.2.2 update: engine
0.1.47, ROCm 10.2.0a20260930, Windows 11, Radeon 8060S, 128 GB, the run config
in section 3 with `swift-abl-1.5-iq3_xxs` (IQ3_XXS), a 1,646-token prompt,
temperature 0, one warm-up request, then three measured requests per engine
start with a unique nonce. Each comparison was interleaved A/B/A/B/A/B and the
table reports the median of the three starts.

Only paired numbers are comparable. The same configuration's prefill moved from
467 to 600 tok/s (+28%) over the session as the machine warmed up, so a
before/after across hours means nothing here.

| Compared | prefill tok/s | decode tok/s | Result |
|---|---|---|---|
| 18 gfx1151 defaults on (the default) | 570.0 | 56.1 | baseline |
| `STRATA_GFX1151_DEFAULTS=0` (all off) | 534.5 (-6.2%) | 46.5 (-17%) | the defaults help on Windows too; keep them on |
| `STRATA_REFILL_PARTS=1` | 561.8 (+1.0%) | 43.1 | below the 3% bar; left off |
| `tools/hip/gfx1151-hipblaslt-100500.txt` | 594.5 | 45.8 | baseline |
| the package's `tuning/gfx1151-hipblaslt-100500.txt` | 576.6 (-3.0%) | 43.9 (-4.1%) | the table shipped with this guide stays |

Decode medians come from short completions (24-45 tokens) and are noisier than
the prefill numbers. Not measured: needle or correctness checks, TTFT, and
multi-turn agent workloads; the A/B above used IQ3_XXS, not IQ3_S.

## 4. Running it

The batch file matters mainly because `TEMP`/`TMP` are redirected to an ASCII
path:

```bat
@echo off
title Strata IQ3_S
cd /d "<STRATA>"
if not exist "<MODELS>\Strata-data\temp" mkdir "<MODELS>\Strata-data\temp"
set "TEMP=<MODELS>\Strata-data\temp"
set "TMP=<MODELS>\Strata-data\temp"
"<STRATA>\.venv\Scripts\python.exe" "<STRATA>\serve\server.py" ^
  "--engine" "strata" "--config" "<STRATA>\strata-iq3_s.json" "--port" "8080" "--open"
if errorlevel 1 pause
```

The server then serves the Web UI at `http://127.0.0.1:8080/` and exposes both
OpenAI-compatible and Anthropic-compatible endpoints.

Redirecting `TEMP` is not cosmetic: the server passes temporary file paths as
UTF-8 and the engine opens them as ANSI, so a Windows account whose profile path
contains non-ASCII characters makes the **vision encoder** fail to open its input.
Pointing `TEMP`/`TMP` at an ASCII path avoids it.

Readiness check:

```powershell
curl http://127.0.0.1:8080/health
# {"status":"ok","max_context":262144,"model":"...","images":true,"loaded":true,"service":"strata"}
```

`loaded: true` means the model is ready. `/metrics` exposes live state, tok/s,
free VRAM and expert cache slots.

## 5. Windows-specific traps

1. **The pinned expert arena eats the WDDM GPU budget** — the single trap with
   the largest effect here. 46.84 GiB of pinned experts are counted against the
   process GPU budget, leaving 2.25 GiB free and squeezing the cache to 320
   slots. Fix: `--mmap-experts` (+35%).
2. **`--kv-streaming on` is a `setup` option, not an engine option.** The engine
   only accepts `--kv-resident N` (minimum 20480). Putting `--kv-streaming` in the
   config `args` makes the engine print its help and exit 1.
3. **KV streaming is not usable at 128K here.** `--kv-resident 32768` pushes the
   expert cache to 0 slots and the engine refuses to serve with
   `no VRAM is left for the expert cache`. With `--mmap-experts` it is not needed.
4. **`STRATA_WDDM_BUDGET=0` has no effect in 0.1.41**: measured 320 slots and
   1,116 MiB free, identical to not setting it, and the log line the
   documentation describes (`Windows budgets N of this card's M MiB`) never appears.
5. **`--expert-cache auto` reserves 2,560 MiB by default on Windows**; when VRAM is
   short the engine shrinks the reservation to keep at least 250 slots, so a
   "0 MiB free" warning only appears when a large `--expert-cache N` is set.
6. **GGUF shard semantics differ between publishers — do not merge blindly.**
   Measured with `python tools/strata_inspect.py <shard>`: the original
   ISTA-DASLab shard 2 is a **deliberately separate PLE table** (100%
   `n-gram/per-layer embeddings`, a single 28.80 GB tensor) and **must not be
   merged**, because the engine keeps the PLE on its own pages. Other IQ3_XXS
   publishers split by size (shard 1 = PLE plus some experts, shard 2 = the
   rest), and those shards can be merged. A quick test: a shard for which
   `strata_inspect.py` reports `architecture None` is not the first shard.
   **`--ple-gguf` must point at the shard that contains the PLE** (shard 2 for the
   original, shard 1 for a size-split one).
7. **Moving a model out of `Strata-data\models\` makes `setup.py` download it
   again** (the `.done` marker moves with the directory). Update `--native` and
   `--ple-gguf` after reorganising.
8. **Windows and AMD expose no GPU load, VRAM or temperature readings.**
   `hardware_static.gpu_note` in `/metrics` says: "no GPU load or VRAM readings for
   AMD cards on Windows yet". For VRAM, use the engine's own `engine.vram_free_mib`,
   `engine.expert_cache_mib` and `engine.expert_slots` — those three are real data.
9. **`/metrics` mixes units, so one conversion helper is wrong.** `hardware.ram_used`
   and `ram_total` are **bytes** (divide by 2^30); `engine.vram_free_mib` and
   `expert_cache_mib` are **MiB** (divide by 2^10). The shipped `web/app.js` `gb()`
   divides by 2^30 everywhere, which only holds for the byte fields.
10. **A non-ASCII Windows profile path breaks the vision encoder** (see section 4:
    point `TEMP`/`TMP` at an ASCII path).
11. **PowerShell 5 parses a BOM-less `.ps1` as GBK**, so non-ASCII text in a script
    breaks the quoting. Keep benchmark scripts ASCII and read non-ASCII prompts
    from a separate UTF-8 file.
12. **`Start-Process -ArgumentList` does not quote paths containing spaces** —
    add the double quotes yourself.

## 6. Optional: vision

Add a `vision` block to the engine config; `--vision` is appended automatically:

```json
"vision": {
  "exe": "<STRATA>\\engine\\strata-vision.exe",
  "mmproj": "<MODELS>\\unsloth\\Qwen3.8-Flash-Next-GGUF\\mmproj-F16.gguf",
  "model": "<path to the shard that contains the PLE>",
  "gpu": false, "max_tokens": 300, "threads": 16
}
```

- `strata-vision.exe` is the **CPU vision encoder** and has to be built from source.
- `mmproj` must come from the **same base model** as the text weights (here a
  `mmproj-F16.gguf` published for the same base).
- `/health` then reports `"images": true`.

## 7. Optional: an uncensored model

An abliterated fine-tune (Swift-1.5-abliterated, IQ3_XXS, 70.62 GiB: shard 1
37.04 GiB including the PLE, shard 2 33.58 GiB) works with the same engine. Its
shards are split by size, so one pack can be built from both, and it needs
`--compat-bf16`. Decode measured **38-48 tok/s** (typically about 41, on par with
IQ3_S); the spread is driven by draft acceptance, not by the model — 73%
acceptance gave 48.5 tok/s, 57% gave 37.1. Note that IQ3_XXS is the smaller
quant, so the upstream 128K table lists 50.9 for it.

## 8. What did not help

Clean A/B runs, medians of three, same machine and workload:

- Smaller context: 8K measured 38.3, 32K measured 38.2 — no different from 128K.
  Larger context behaves the same way: 256K measured 32.6 against 30.8 for 128K in
  a back-to-back A/B, and 256K also left the expert cache at 24,576 slots.
- `--kv-grow` 38.3; `--vram-reserve-mib 512` 38.0; pool tasks 12 at 38.5.
- `--host-core last` 34.5 (worse); the section 5 switches all off, 35.3.
- `--spec 2` 33.5; `--spec 8` 30.6; `--spec-min-p 0.2` 31.1.
- `--kv q4` 35.4; `--mtp-q4 proj` 36.6.
- Head/norm/IO variants — `--mtp-q4 head` 42.5, `--mtp-hnorm stream` 42.9,
  `--ple-io ram` 41.4 (and 106 s to start), `--kv fp16` 42.8 — all inside the
  noise band.

**The engine will not start with these:** `--no-spec` and `--no-pool`. An IQ
quant needs `--spec>=2`, and with the CPU pool disabled the expert compute
exceeds VRAM.

## 9. Diagnostics worth reusing

`STRATA_DECODE_TIMING=1` together with `STRATA_VERIFY_PROFILE=1` prints a
per-window breakdown. On 0.1.41, for a 300-token Chinese request this machine
measured **50.67 ms/window, 1.91 tok/window, 37.7 tok/s**: verify 44.31 ms (87%,
of which GPU-reach wait 0.00, CPU experts 0.00 and jobs 0.00 — the CPU takes no
part, every expert is in VRAM), draft 6.22 ms, commit 0.14 ms. No single GPU
stage dominates; it is an evenly bandwidth-bound load. Raw verification
throughput is about 55.5 tok/s, so the 37.7 achieved is 68% efficiency, the loss
being draft acceptance.

## 10. Contributing these numbers

The measurements here map onto the report format in
`docs/COMMUNITY_BENCHMARKS.md`; a report would go in
`bench/results/YYYY-MM-DD-community-ryzen-ai-max-395-windows/`. Values not
measured on this machine (TTFT, per-run JSON, needle checks) should be published
as `not measured` rather than estimated. Engine logs are safe to attach: they
record token counts, timings and draft acceptance, not prompt or completion text.

## 11. References

- Strata: <https://github.com/Niko1221/Strata>
  - `docs/STRIX_HALO.md` — Strix Halo build and tuning on Linux (sections 4, 5, 6)
  - `docs/COMMUNITY_BENCHMARKS.md` — community report format and submission flow
  - `docs/AMD_HIP.md`, `docs/DETAILS.md`, `docs/TROUBLESHOOTING.md`
- Strix Halo community: the Strix Halo HomeLab wiki and its Discord
- hal0.dev forum (Profiles & Benchmarks)