# Community benchmark on RTX PRO 4500 Blackwell 32 GB with 32 GB DDR4

Measured on 2026-11-03 by Maxrio9. One request slot against the shipped four slots, on a RAM-limited host.
Main limitation: n=4 prefill and n=6 decode runs per configuration, one machine, desktop session open.
Part of this report was prepared with an AI assistant; every number comes from the attached raw rows
produced by the script on this machine.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 4500 Blackwell, 32 GB; CPU: AMD Ryzen 9 5900X (12 cores / 24 threads, AVX2, no AVX-512);
  installed RAM: 32 GB DDR4 (31 GiB reported, about 25 GiB usable with nothing loaded): 2 x 16 GB Corsair CMK32GX4M2E3200C16 (DDR4-3200 CL16),
  one module per channel (slots A1 and B1, dual channel), rated and configured speed 3200 MT/s (read from the firmware's memory table),
  XMP/DOCP profile on (inferred: the kit's default JEDEC speed is lower);
  storage: NVMe, Lexar NM990 2 TB (PCIe 4.0 x4 link), ext4, models and packs on the same disk; PCIe link: PCIe 4.0 x16 on an ASUS ROG STRIX X570-I GAMING board (the card itself is PCIe 5.0 capable)
  (the engine's own probe measured 28.1 to 28.4 GB/s host to device).
- OS: Arch Linux (Omarchy), kernel 7.2.15; NVIDIA driver 615.78; CUDA 13 (engine package `cuda13`).
- Strata: configurations A to D on app 0.2.9 / engine 0.1.55; E to H on app 0.2.10 / engine 0.1.56 (release binaries from the
  installer, `strata update`); commit: not known (release packages).
- Background workloads and power limits: a desktop session was open during the runs (compositor, browser, terminal,
  chat apps); no other GPU compute. GPU power limit: 200 W (default and maximum).

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, IQ3_XXS,
  `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (47,039,860,096 bytes, sha256 `219ea929...5856d15`) and `-00002-of-00002.gguf`
  (28,800,138,432 bytes, sha256 `316b46f3...161e113`); mmproj BF16 (0.91 GB).
- Vision: picture reader on the GPU (`--vision on`) except D and E (`--vision off`). No custom packs or profiles.
- Context 64K (`--context 64k`), KV int8, expert cache auto, `--tune skip`, access local.
- Calibration: **not enabled** in any measured configuration (an earlier `engine calibrate` result was rolled back before these runs).
  No experimental speed projection.

| | Strata / engine | `strata launch` flags added to the common set | Drafter that ran | Request slots | GPU expert-cache slots | VRAM free after load |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A | 0.2.9 / 0.1.55 | (none: shipped defaults) | MTP | 4 | 10,932 | 583 MiB |
| B | 0.2.9 / 0.1.55 | `--parallel 1 --drafting mtp` | MTP | 1 | 13,357 | 640 MiB |
| C | 0.2.9 / 0.1.55 | `--parallel 1 --drafting dflash2` | MTP (engine said "DFlash2 isn't available in this mode") | 1 | 13,346 | 638 MiB |
| D | 0.2.9 / 0.1.55 | as C plus `--vision off` | MTP (same fallback) | 1 | 14,169 | 657 MiB |
| E | 0.2.10 / 0.1.56 | `--parallel 1 --drafting dflash2 --vision off` | DFlash2 | 1 | 14,322 | 695 MiB |
| F | 0.2.10 / 0.1.56 | `--parallel 1 --drafting dflash2` | DFlash2 | 1 | 13,534 | 2,054 MiB |
| H | 0.2.10 / 0.1.56 | as F, started by a desktop launcher that runs the same `strata launch` | DFlash2 | 1 | 13,235 | 2,016 MiB |

```text
strata launch --family qwen --model iq3_xxs --model-path <models dir>/ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  --data-dir <models dir>/strata --context 64k --kv int8 --tune skip --access local --port 8080 --yes --no-tui \
  [flags from the table]
```

## Method

- Script: `strata_bench.py` (attached). Speeds are the engine's own per-request records from `GET /metrics`
  (`prompt_ms` with `prompt_read` for prompt tok/s; `decode_tok_s` for decode; `first_engine_token_s` for TTFT), not
  total time divided by tokens. Model loading is excluded; one unrecorded 16-token warm-up request runs first.
- Prefill: four prompts per configuration, each about 7,000 to 8,000 tokens (up to 10,468 in one run), made from random
  26,000-character slices of this repository's own `docs/*.md`, with a unique `Reference id <16 hex digits>` line first so
  nothing is reused; answer cap 8 tokens. Reused tokens: 0 in every run. (The 0.1.55 start-up log of configuration A said
  "conversation cache off: not enough free RAM to park conversations"; that line was not checked for the other configurations.)
- Decode: three fixed prompts (prose, code, JSON; 85 to 93 prompt tokens) twice each, greedy, cap 1,200 tokens, always
  reaching 1,200. Reused tokens: 0.
- A run counts only if the engine's request list shows exactly one new request and the engine is idle afterwards. None was discarded.
- The first prefill run after a start is often the slowest (cold expert cache): 1,526 (B), 1,886 (C), 2,006 (E), 1,516 (H) tok/s.
- Memory: engine-reported VRAM free after load (table above). Host RAM: MemAvailable was between about 2 and 5 GiB in spot
  checks with the model loaded (not logged continuously). In a tuner run on 0.1.56 the base run recorded a process RSS peak of 22.9 GiB and a minimum
  MemAvailable of 2.7 GiB. Paging or out-of-memory failures: none observed.

## Results

### Prefill (unique ~7K-token prompts, 8-token answer cap)

| Configuration | Actual prompt tokens (median) | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A: shipped defaults (4 slots, MTP), engine 0.1.55, picture reader on | 7150 | 0 | 8 (median) | 4 | 305 (283-342) | 21.88 (20.24-25.21) |
| B: `--parallel 1 --drafting mtp`, 0.1.55, reader on | 7297 | 0 | 8 (median) | 4 | 2651 (1526-2872) | 2.60 (2.23-6.95) |
| C: `--parallel 1 --drafting dflash2` (0.1.55 fell back to MTP), reader on | 8034 | 0 | 8 (median) | 4 | 2897 (1886-3041) | 3.33 (2.73-3.85) |
| D: as C with `--vision off` (MTP), 0.1.55 | 7187 | 0 | 8 (median) | 4 | 2916 (2840-3025) | 2.49 (2.44-2.56) |
| E: `--parallel 1 --drafting dflash2 --vision off`, 0.1.56 | 7278 | 0 | 8 (median) | 4 | 2853 (2006-3662) | 2.78 (1.94-3.72) |
| F: `--parallel 1 --drafting dflash2`, reader on, 0.1.56 | 7548 | 0 | 8 (median) | 4 | 2940 (2871-2965) | 2.60 (2.45-3.94) |
| H: as F, started by a desktop launcher, 0.1.56 | 7923 | 0 | 8 (median) | 4 | 2626 (1516-2822) | 2.91 (2.26-6.92) |

### Decode (three fixed prompts x 2, 1,200-token cap, greedy)

| Configuration | Prompt tokens | Generated tokens | Runs | Decode tok/s median (range) | prose / code / JSON medians | TTFT s median (range) |
| --- | ---: | ---: | ---: | --- | --- | --- |
| A | 85-93 | 1200 | 6 | 99.2 (93.1-101.0) | 97 / 97 / 101 | 2.24 (1.87-2.64) |
| B | 85-93 | 1200 | 6 | 149.6 (142.5-156.5) | 150 / 144 / 156 | 0.29 (0.28-0.33) |
| C | 85-93 | 1200 | 6 | 147.0 (136.6-155.0) | 146 / 145 / 152 | 0.29 (0.27-0.33) |
| D | 85-93 | 1200 | 6 | 155.3 (146.6-161.2) | 153 / 151 / 161 | 0.30 (0.28-0.33) |
| E | 85-93 | 1200 | 6 | 162.3 (151.8-167.0) | 158 / 158 / 165 | 0.29 (0.27-0.36) |
| F | 85-93 | 1200 | 6 | 160.4 (149.1-170.4) | 166 / 150 / 162 | 0.30 (0.28-0.31) |
| H | 85-93 | 1200 | 6 | 155.0 (151.1-160.3) | 156 / 155 / 156 | 0.29 (0.27-0.35) |

Per-run values (tok/s), prefill then decode (prose, code, JSON, prose, code, JSON):

- A: [288, 342, 283, 323]; [100.0, 94.8, 101.0, 93.1, 98.4, 100.7]
- B: [1526, 2758, 2545, 2872]; [154.2, 145.0, 156.2, 144.9, 142.5, 156.5]
- C: [1886, 2848, 2946, 3041]; [155.0, 144.8, 154.9, 136.6, 144.4, 149.2]
- D: [2840, 2891, 2942, 3025]; [159.7, 151.0, 160.3, 146.6, 150.1, 161.2]
- E: [2006, 3662, 2822, 2885]; [161.2, 151.8, 163.5, 155.6, 163.6, 167.0]
- F: [2952, 2965, 2871, 2929]; [170.4, 149.1, 159.9, 160.9, 150.6, 164.6]
- H: [1516, 2590, 2822, 2663]; [160.3, 152.7, 157.9, 151.1, 156.5, 153.5]

Per-run values are in `results.jsonl`; `prefill_tok_s` and `decode_tok_s` are the engine's numbers.

## Correctness and limitations

- No correctness, needle or tool-call check was run for these configurations. Speed only.
- One machine, one quant, 64K context, KV int8. Not tested: other quants, 128K or longer context, `--clients N`, concurrent requests.
- Slot count 4 to 1 changes two things together (the batch code path and the expert-cache size, 10,932 to 13,357 slots); they were not separated.
- B to D are engine 0.1.55 and E to H are 0.1.56; the drafter (MTP vs DFlash2) and the engine version were not separated because the
  MTP-on-0.1.56 control run was not done. D (MTP, reader off) to E (DFlash2, reader off) is +4 % decode (155.3 to 162.3), inside
  the spread; F (DFlash2, reader on) against C (MTP, reader on) is +9 % (147.0 to 160.4).
- A desktop session was open (the guide asks to close other programs). Run-to-run spread is about 3 to 7 %.
- The tuner: `strata launch --tune quick --retune` on 0.1.56 measured only per-request settings; its two restart
  candidates were skipped for lack of free RAM (reported separately as #5122; not part of this PR).
