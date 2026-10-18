# Community benchmark: RX 7800 XT 16 GB (gfx1101), Ryzen 5 9600X, 64 GB DDR5 (Windows 11, HIP)

Measured on 2026-10-16 and 2026-10-18 by durikim1-cpu on one native-Windows desktop. As far as I can see this is the first **Windows** report from a **gfx1101 / RX 7800 XT** card in `bench/results` (an earlier Linux validation of this card is #254 in `docs/AMD_HIP.md`). It has two parts, and they must not be read as one before/after pair:

- **A. Strata 0.1.41 as released** (prebuilt Windows HIP engine), Flash-Next IQ3_XXS, 64K context, fresh prompts of
  about 1K / 8K / 24K / 60K tokens, three runs each.
- **B vs C. A private build, same binary, one opt-in switch off/on.** A same-day A/B of a Windows file-read change
  (`STRATA_R6_FILE_READ_CACHE`, proposed separately in an issue) at a 60K fresh prompt, three runs each.
  **This build is not a Strata release**: it is 0.1.44.1 with local changes, listed below.

Main limitations: one machine, one model, mostly one long-prompt length for the A/B; desktop apps were open during the
runs; GPU clocks, driver and PCIe link were not captured per run; part A and part B/C differ in date, engine, KV type and
RAM budget, so the A→C difference is a history, not the effect of one change.

## Hardware and software

- GPU: AMD Radeon RX 7800 XT, 16 GB, gfx1101 (RDNA3). No NVIDIA GPU in the machine. PCIe link: not recorded.
- CPU: AMD Ryzen 5 9600X, 6 cores / 12 threads. Board: ASRock B650M Pro RS.
- RAM: 64 GB DDR5 (observed DDR5-5600); Windows reports about 61.6 GiB usable.
- Storage: SK hynix Gold P31 1 TB NVMe (model files on the same drive as the OS).
- OS: Windows 11 Home, build 26200. AMD driver 32.0.31041.1004 (version at report time; not captured per run).
- Part A engine: prebuilt Windows HIP **0.1.41** from setup, source tag `v0.1.41` (`fb58e0d`), `engine/BUILD.json`
  `"source": "prebuilt"`, ROCm `10.2.0a20260930`, hipBLASLt `100500`.
  `strata.exe` SHA-256 `5b5285c726e76ab307a6956fcefff0fcdde90e59d566d17c28e410471b9fd756`.
- Part B/C engine: **private source build** of 0.1.44.1 with local changes (HIP, Windows). One binary for both arms,
  `strata.exe` SHA-256 `6095a4f195c349414f48f7e6f7c6e750ce447a8128b8e99688835babe1a171aa`.
  Runtime DLLs observed: `amdhip64_7.dll` 10.0.3686.0, hipBLASLt 1.5.0.
- Background load: not idle. Ordinary desktop apps (a browser and chat clients) were open, about 10 GiB of working set
  at the start of part A. Close them for cleaner numbers.

### What the private build changes

The B/C arms run with these `STRATA_*` variables that do **not** exist in 0.1.44.1 (local changes):
`STRATA_MMQ_COMPACT`, `STRATA_PREFILL_LAYER_MAJOR`, `STRATA_PREFILL_LM_R`, `STRATA_PREFILL_LM_MARGIN_MIB`,
`STRATA_PREFILL_LM_TOKENS`, `STRATA_PREFILL_LM_RETAIN`, `STRATA_PREFILL_LM_CACHE_ARENA`,
`STRATA_PREFILL_LM_CACHE_GUARD`, `STRATA_R5_FETCH_TEAM`, `STRATA_CPU_IO_OVERLAP`, `STRATA_R6_FILE_READ_CACHE`.
They are local experiments: a compact MMQ scratch buffer, layer-major prompt reading (@vizdatom's
`layer-major-0.1.43.1` code from #1670, ported to this tree), a persistent expert-fetch team, CPU/I-O overlap, and the
file-read change. The build also carries libratechw's opt-in HIP prompt-attention patches (`STRATA_ATTN_GATE`,
`STRATA_QPREP`, not enabled in these runs). Only `STRATA_R6_FILE_READ_CACHE` differs between B and C. Everything else in
the configs is an existing Strata option. A gfx1101 hipBLASLt tuning table (70 shapes) is passed with
`STRATA_HIPBLASLT_TUNING` in B/C.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ3_XXS (`…-IQ3_XXS-00001-of-00002.gguf` and
  `…-00002-of-00002.gguf`), native pack prepared by setup, MTP from setup. Vision not installed.
- Context 65,536 in all arms. Thinking off, MTP on (`--spec 4 --spec-min-p 0.5`), `--suffix-draft 0`, temperature 0.
- Part A: `--expert-cache auto --prefill auto`, **FP16 KV** (the default), `--resident-budget-gib 24`,
  `STRATA_RESIDENT_PIN=0`, `STRATA_UNBUFFERED_LOAD=1`. The defaults alone stopped on this PC's memory guard (commit
  limit), so the budget and unbuffered load were set by hand.
- Part B/C: `--expert-cache 2200` (the engine grew it to 2,918 slots, about 4.77 GiB), `--prefill 4096`, **int8 KV**,
  `--resident-budget-gib 19`, `--adapt-every 16`, `STRATA_RESIDENT_PIN=0`, `STRATA_UNBUFFERED_LOAD=1` as in A, plus the variables above.
- Full launch arguments with local paths replaced: [`config-A-official-0.1.41.json`](config-A-official-0.1.41.json),
  [`config-B-private-file-cache-off.json`](config-B-private-file-cache-off.json),
  [`config-C-private-file-cache-on.json`](config-C-private-file-cache-on.json).
- Expert profile: each arm starts from a copy of the same profile file; the copy's hash was not recorded in part B/C.

## Method

- Own Python harness over the HTTP API, one request at a time, the model already loaded. Not `BENCH.bat`.
- Prompt: a synthetic English ledger (`Entry 00001: item 1, quantity 2, status pending.` repeated) ending with
  "Summarize this ledger and propose three data validation rules in approximately 250 words." The system message
  carries a unique run tag, so every request is a **fresh prompt: 0 reused tokens** in every run listed here.
- Output cap 384 tokens; the actual generated tokens are listed (the model sometimes ends earlier).
- Speeds are the engine's own timings (`prompt_per_second`, `predicted_per_second`). TTFT and total latency are
  measured by the client from sending the request to the first streamed token / the end.
- Three runs per configuration; median and range below. The OS file cache was **not** flushed between runs, and B ran
  before C on the same day (runs started 17:36 and 17:42 local time). This is not a cold-SSD test.
- Memory was watched by an external guard of the harness (stop below 4 GiB free RAM or free commit). No run in the
  results table hit it.
- Per-run data: [`runs.csv`](runs.csv) (one row per request; fields described at the end).

## Results

| Arm | Actual prompt tokens | Reused | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | --- | ---: | --- | --- | --- |
| A 0.1.41 release, FP16 KV, RAM 24 | 1,060 | 0 | 349 / 375 / 362 | 3 | 153.4 (153.3–165.3) | 28.3 (26.9–29.9) | 6.96 (6.46–6.96) |
| A 0.1.41 release, FP16 KV, RAM 24 | 8,041 | 0 | 384 / 384 / 384 | 3 | 409.7 (408.8–410.4) | 29.6 (29.6–30.8) | 19.68 (19.64–19.73) |
| A 0.1.41 release, FP16 KV, RAM 24 | 24,054 | 0 | 366 / 384 / 364 | 3 | 438.5 (437.9–440.9) | 29.8 (28.8–29.9) | 54.92 (54.62–55.01) |
| A 0.1.41 release, FP16 KV, RAM 24 | 60,047 | 0 | 353 / 337 / 341 | 3 | 437.0 (435.0–437.1) | 29.9 (29.6–30.4) | 137.48 (137.45–138.11) |
| B private build, file cache **off**, int8 KV, RAM 19 | 60,047 | 0 | 345 / 381 / 340 | 3 | 1,399.8 (1,383.3–1,403.8) | 28.5 (24.6–28.9) | 43.08 (42.91–43.50) |
| C private build, file cache **on**, int8 KV, RAM 19 | 60,047 | 0 | 345 / 381 / 340 | 3 | 1,476.4 (1,462.1–1,484.4) | 36.0 (32.2–36.3) | 40.74 (40.53–41.16) |

B→C, same binary and same day: prompt **+5.5%**, decode **+26.3%**, TTFT **−5.4%**, total request time
56.19 → 51.20 s (−8.9%). The generated text was identical pair by pair (345 / 381 / 340 tokens), so C is not faster
because it answered shorter.

Disk traffic, median of samples within each phase of the three B/C requests (whole `PhysicalDrive0`, so not all of it is
the model):

| Phase | Arm | Physical SSD read MiB/s | Engine-requested read MiB/s |
| --- | --- | ---: | ---: |
| Prompt | B off | 947.9 | 947.3 |
| Prompt | C on | 306.7 | 986.7 |
| Decode | B off | 678.4 | 678.3 |
| Decode | C on | 24.7 | 817.6 |

The engine's requested read rate does not drop; with the change most of those reads are served from the Windows file cache
instead of the SSD. The file cache size itself was not measured.

**Do not read A→C as one change.** Engine (0.1.41 release vs private 0.1.44.1 build), KV type (FP16 vs int8), RAM
budget (24 vs 19 GiB), expert cache and prefill size all differ, and the days differ. The 60K TTFT went from 137.5 s
(A) to 40.7 s (C) on this machine; that is the combined result of everything listed above.

### Failures and skipped cases

- Part A defaults (no `--resident-budget-gib`, buffered load) stopped on the memory guard before a result. Not a speed.
- In B/C, raising the GPU expert cache request to 2,600 (3,456 slots, about 5.64 GiB) loaded but warned LOW VRAM on the
  first request; WDDM's budget was 11,935 MiB with 12,030 MiB in use, so the engine saw 0 free. No valid speed.
- A 23 GiB RAM budget with the same build stopped on the commit guard (3.97 GiB) during a coding task, which is why
  B/C use 19 GiB.

## Correctness and limitations

Checks on arm C (file cache on), each a separate test; the denominators are not summed:

- Fixed suite: 8 short questions once + 4 long-context retrieval questions × 3 = 20 answers, **20/20**, no stop.
  Arm B's run of the same suite ended at request 14 (13/13 correct) when the engine waited on an MTP GPU fence and was
  stopped; that partial run is not counted as 20/20.
- Real coding agent (6 tasks × 3): functional **18/18** in both B and C; a stricter process check 15/18 in
  both (the three misses are the same task: the agent read the v2 file directly instead of hitting the planned v1
  error first). Median wall time over the 18 runs 61.45 → 50.31 s (B → C).
- MTP fence waits (engine stuck in an MTP GPU wait until restarted) during the coding runs: **3 in B, 1 in C**. Not
  fixed; C having fewer is an observation, not a cure.
- Long-context system-prompt precedence (the system says answer `ORBIT` for ALPHA; the 60K document says otherwise):
  ALPHA **0/3**, BETA 3/3 in C. The same failure appears on 0.1.41 as released with FP16 KV at RAM 23 GiB, so it is not
  new in this build. Repeating the system rule at the end of the prompt gave 3/3 and 3/3.
- Not tested: other models, contexts above 64K for the B/C settings, Thinking on, `BENCH.bat`, Linux/WSL2, cold SSD.

## Data fields (`runs.csv`)

`arm`, `label` (target length and run index), `target_prompt_tokens`, `prompt_tokens_fresh` (engine `prompt_n`),
`cache_reused_tokens` (engine `cache_n`), `generated_tokens` (engine `predicted_n`), `prompt_tok_s_engine`,
`decode_tok_s_engine`, `prompt_ms_engine`, `decode_ms_engine`, `draft_n`, `draft_accepted`, `ttft_s_client`,
`total_s_client`, `finish_reason`, `recorded_at` (local time, +09:00).
