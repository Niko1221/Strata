# One Radeon VII (gfx906, 16 GiB): ROCm 10.1 vs 7.14, and engine 0.1.63 vs 0.1.62

Five `standard-v1` runs on the same card, the same model, the same configuration and the same protocol
(2026-11-20, `strata bench run --suite standard-v1`, context 32768, KV int8, expert cache auto, prefill 2048).
Every run completed: 21 of 21 requests, no failures.

- **Card:** AMD Radeon VII, 16 GiB, gfx906, wave64. **Host:** Xeon E5-2690 v3, 85.9 GiB RAM, Debian 13 guest.
- **Model:** Qwen3.8-Flash-Next GSQ-RCO, IQ3_XXS (two GGUF shards), draft DFlash2 Q4_0, vision off (CPU).
- **Folders:**
  - `rocm-7.14-0.1.62/` — `mixa3607/rocm-gfx906:7.14-complete`, clang 23, engine 0.1.62 (v0.2.16), PLE table not locked.
  - `rocm-7.14-0.1.62-locked/` — same build, PLE table (26.8 GiB) locked in RAM (`memlock`), engine 0.1.62.
  - `rocm-10.1-0.1.62/` — same build, PLE table not locked.
  - `rocm-7.14-0.1.63/` — `mixa3607/rocm-gfx906:7.14-complete`, engine 0.1.63 (v0.2.17), PLE table not locked.
  - `rocm-7.14-0.1.63-locked/` — same as above, PLE table locked in RAM.

## Results (medians, n = 3 per case, min-max in the per-run reports)

| Case | 7.14 / 0.1.62 | 7.14 / 0.1.62, locked | 10.1 / 0.1.62 | 7.14 / 0.1.63 | 7.14 / 0.1.63, locked |
| --- | ---: | ---: | ---: | ---: | ---: |
| short, decode tok/s | 30.3 | 31.2 | 31.7 | 27.9 | 28.6 |
| short, TTFT s | 1.98 | 1.96 | 2.53 | 2.10 | 2.00 |
| prompt-4k, prompt tok/s | 340.1 | 338.5 | 317.8 | 334.8 | 339.0 |
| prompt-4k, decode tok/s | 26.4 | **29.9** | 28.0 | 23.5 | 26.0 |
| prompt-4k, TTFT s | 12.09 | 12.14 | 12.93 | 12.29 | 12.13 |
| prompt-16k, prompt tok/s | 339.7 | 340.3 | 335.0 | 339.1 | 339.9 |
| prompt-16k, decode tok/s | 28.2 | **30.6** | 27.0 | 25.1 | 26.1 |
| prompt-16k, TTFT s | 48.31 | 48.21 | 48.98 | 48.41 | 48.28 |
| sanity checks | 12/12 | 12/12 | 12/12 | 11/12 (R1) | 11/12 (I1) |

## What these runs say

1. **ROCm 10.1 is not faster than 7.14 on one gfx906 card.** Prompt reading is within ±1 % of 7.14 at 16K and
   about 7 % lower at 4K (one outlier aside: 240.9 in the worst of three repetitions), decode is within the
   spread. 10.1 (`10.1-complete`, clang 24) is therefore a matter of keeping the card on a maintained base,
   not of speed.
2. **Locking the PLE table helps this card.** With `memlock` raised so the engine can lock the 26.8 GiB table
   (`PLE table locked in RAM`), engine 0.1.62 reads 4K decode 26.4 → **29.9 tok/s (+13 %)** and 16K decode
   28.2 → **30.6 tok/s (+9 %)**; prompt speed and time to first token are unchanged, sanity stays 12/12.
   A single-card run with the table locked was not in the docs before, and it is worth having on a 16 GiB card:
   the table is bigger than the card's memory, so the pages it lives in are exactly the ones that get reclaimed.
3. **Engine 0.1.63 (v0.2.17) is slower here.** Decode drops 8-11 % without the lock and 6-7 % with it, prompt
   speed is unchanged, and one sanity check fails in each run — R1 (the answer hit the output cap) unlocked and
   I1 (a JSON value differed) locked. Engine 0.1.62 in two runs of the same day passed 12/12 twice. The log of
   0.1.63 shows the expert arena read through the file cache instead of RAM, which fits the drop.
4. **Building on `10.1-complete` needs a builder layer:** the image ships without cmake, ninja and the python
   packages the terminal needs (`yaml`, `jinja2`, `regex`, `numpy`, `PIL`, `requests`, `psutil`). The build
   itself runs clean: 157 of 157 targets, `strata 0.1.62` / `strata 0.1.63`.
5. **The two stream-priority names from #6246 are in `main`** (`65207788`): v0.2.17 builds for gfx906 with no
   local patch to `strata_hip.h`.

## Limits

One card, one host, one model, three repetitions per case. Cache and page state were not controlled beyond the
fixed configuration, so the decode differences of a few percent are indicative rather than exact. CUDA, SYCL,
Metal, Windows and multi-card behaviour were not tested.
