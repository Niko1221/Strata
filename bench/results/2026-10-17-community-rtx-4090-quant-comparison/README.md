# Community benchmark: three Flash-Next quantizations on one RTX 4090

Measured on 2026-10-16 and 2026-10-17 by hongxianzhi. Three quantizations of the same model on one
machine, each measured with the same tool, the same suite and the same context, after
`setup.py --calibrate` had tuned that quantization's engine settings for this PC. One session; the
three suite runs are within about two hours of each other.

This folder is the summary. The per-run reports and their `results.json` are in the three sibling
folders:

| Quantization | Run folder |
| --- | --- |
| IQ3_XXS | `2026-10-16-community-rtx-4090-qwen3-8-flash-next-iq3-xxs/` |
| IQ3_S | `2026-10-17-community-rtx-4090-qwen3-8-flash-next-iq3-s/` |
| Unsloth UD-Q4_K_XL | `2026-10-17-community-rtx-4090-qwen3-8-flash-next-unsloth-ud-q4-k-xl/` |

## Hardware and software

- NVIDIA GeForce RTX 4090, 24 GiB VRAM, PCIe 4.0 x16; driver 610.88; no power limit set (450 W card limit)
- 13th Gen Intel Core i7-13700K (16 cores, 24 threads, **no AVX-512**); 127.8 GiB RAM
- Windows 10 (10.0.26200); storage not measured
- Strata 0.1.44.1, project commit `296b59d2`, release binary; engine 0.1.44.1
  (`engine/strata.exe` sha256 `bd18453f65b4d03b15de6100b1a9226426266560779b51b9265de176ae8f34d4`,
  identical for all three runs)
- Background load and power draw not recorded

## Model and configuration

All three run Qwen3.8-Flash-Next (176.94 B parameters, vision encoder loaded) at 262,144 tokens of
context — the model's native window — with `--kv int8 --kv-resident 32768`, `--prefill auto`,
`--spec 4`, `--vision`, `--vram-reserve-mib 700`, one request at a time (`settings.parallel` is not
set), thinking off and temperature 0 for the suite. Each quantization was calibrated for this PC
first (`python setup.py --calibrate --model <NAME> --data-dir <DIR> --no-start --yes`), so the tuned
knobs differ; these numbers compare configured installations, not raw kernels.

| | IQ3_XXS | IQ3_S | Unsloth UD-Q4_K_XL |
| --- | ---: | ---: | ---: |
| Model repository | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | same | `unsloth/Qwen3.8-Flash-Next-GGUF` |
| Expert weights held in RAM | 39.97 GiB | 46.84 GiB | 71.73 GiB |
| Native pack | `packs/iq3_xxs`, 300 matrices, 1781 MiB | `packs/iq3_s`, 300, 2019 MiB | `packs/unsloth-ud-q4_k_xl`, 499, 3619 MiB |
| Output head | Q5_K, 437,043,200 B | Q5_K, 521,472,000 B | Q5_K, 675,430,400 B |
| Tuned settings | `--pcie-frac 0.35 --spec-min-p 0.70 --pool-workers 7 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97` | `--pcie-frac 0.20 --spec-min-p 0.70 --pool-workers 7 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97` | `--pcie-frac 0.20 --spec-min-p 0.70 --pool-workers 7 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97` |
| Environment | `STRATA_ROUTE_TAIL_SKIP=0` | `STRATA_ROUTE_TAIL_SKIP=0` | `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0` |

The full launch line of each run is in its own README. `STRATA_ROUTE_TAIL_SKIP=0` is this install's
setting: the engine computes every route the router picks, including the low-ranked tail of the route
list that the skip mode drops to save expert reads ([docs/QUANTS.md](../../../docs/QUANTS.md)
measures what that mode costs in quality). The numbers below are this install's quality-first
setting, not the fastest one.

## Method

`python tools/strata_bench.py --runs 3 --clients 4 --long` against the HTTP API of an already
running server; model loading is not timed. One warm-up chat and a 1K prompt are discarded first.
Every prompt carries a unique tag, so the fresh-prompt tests cannot reuse the conversation cache.
Between the quantizations the previous server was unloaded (`POST /unload?wait=300`), the next
quantization calibrated, its server started, and the suite run once on the fresh process.

## Results

Medians with the range of the runs in brackets. Speeds are the engine's own timings; first-token
times are measured by the client over HTTP.

| Quantization | chat decode tok/s | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | agent decode tok/s | 4 clients total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ3_XXS | 109.2 (98.4-121.4) | 2634.9 (2604.3-2637.0) | 4622.9 (4612.0-4634.1) | 4771.2 (4733.1-4850.1) | 132.3 (99.0-166.3) | 95.3 (85.5-96.6) |
| IQ3_S | 92.2 (85.9-106.2) | 2158.7 (2131.1-2161.8) | 4161.7 (4159.0-4171.2) | 4228.6 (4227.3-4234.6) | 119.8 (84.6-146.1) | 74.8 (69.8-77.5) |
| UD-Q4_K_XL | 62.8 (54.4-78.2) | 1310.9 (1301.1-1313.5) | 2510.6 (2509.7-2511.1) | 2498.1 (2493.5-2501.4) | 80.7 (54.8-109.7) | 47.8 (46.1-48.7) |

Time to first token (median seconds):

| Quantization | chat | 4K | 16K | 32K | agent turn |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ3_XXS | 0.28 | 1.54 | 3.50 | 6.77 | 1.56 |
| IQ3_S | 0.32 | 1.88 | 3.88 | 7.62 | 1.86 |
| UD-Q4_K_XL | 0.51 | 3.10 | 6.42 | 12.89 | 3.17 |

Agent-session conversation-cache hit over the follow-up turns: 68.6 % (IQ3_XXS), 68.8 % (IQ3_S),
68.7 % (UD-Q4_K_XL). Total suite wall time: 99 s, 116 s and 188 s.

### What the numbers show

- **IQ3_S is slower than IQ3_XXS at every length on this machine**: −15.6 % chat decode, −18.1 % 4K
  prompt, −10.0 % 16K, −11.4 % 32K, −9.4 % agent decode, −21.5 % with four clients; and it keeps
  6.9 GiB more expert weights in RAM (46.84 GiB vs 39.97 GiB). The registry note in `setup.py`
  ("about 15% slower than IQ3_XXS and 7 GB more RAM") had not been measured on this hardware before;
  this run is consistent with it. Independently, the calibration probe on this PC put IQ3_S at
  126.9 tok/s and IQ3_XXS at 142.4 tok/s (−10.9 %).
- **UD-Q4_K_XL runs at roughly half of IQ3_XXS's speed** here (chat 62.8 vs 109.2, 32K prompt 2498
  vs 4771, four clients 47.8 vs 95.3) and holds 71.7 GiB of expert weights in RAM.
- Decode follows the expert weights held in RAM in the order measured here — 39.97 GiB → 109.2 tok/s,
  46.84 GiB → 92.2, 71.73 GiB → 62.8 (chat) — while prefill sits at 1311-2635 tok/s at 4K and
  2498-4771 at 32K for all three. Prompt processing is 21-24× the decode rate at 4K and 40-46× at
  32K. This suite does not separate RAM traffic from the formats' own kernels as the cause.
- With four concurrent requests the summed rate falls below one request's decode rate — 87 %
  (IQ3_XXS), 81 % (IQ3_S) and 76 % (UD-Q4_K_XL) of it — while the tool reports that the server runs
  one request at a time, so four clients partly wait their turn. The ordering between quantizations
  does not change.

### Quality (not measured here)

These are speed measurements only. The repository's own quality numbers for the same quantizations
are in [docs/QUANTS.md](../../../docs/QUANTS.md), measured on an AMD EPYC 7452 with a Radeon AI PRO
R9700 (HIP, engine `0601072d`), teacher-forced over 4,995 positions:

| Quantization | Same top token as Q8_0 | Mean KL (Q8_0 ‖ quant) |
| --- | ---: | ---: |
| Unsloth UD-Q4_K_XL | 93.09 % | 0.056 |
| IQ3_S | 87.69 % | 0.144 |
| IQ3_XXS | 84.46 % | 0.202 |

## Correctness and limitations

- Recall, `tools/needle_bench.py`: IQ3_XXS found the needle at 128k and at 258,266 prompt tokens,
  depths 10 % and 90 % (4 of 4); IQ3_S found it at 258,266 prompt tokens, depths 10 % and 90 %
  (2 of 2). UD-Q4_K_XL was not needle-tested in this session.
- Vision was spot-checked after each server came up, not for UD-Q4_K_XL: `/v1/models` reports
  text+image and the vision path answered on IQ3_XXS (with "Red") and on IQ3_S (in thinking mode; the
  probe prints only the first 100 characters, so its final colour is not in the record). That shows
  the path works, not how well it reads an image.
- This is a speed suite: no quality, coding or long-conversation evaluation was run. The two quality
  columns above come from a different machine and were not measured here.
- Medians of three runs of a short suite: the chat test generates 100 tokens and the prompt tests
  24, so the ranges carry draft-acceptance noise. One machine, one GPU, warm expert cache,
  single-request server; background load and power draw were not recorded, and the load time of the
  model is not in any number.
- The three quantizations use different tuned settings, so a reader comparing engine kernels should
  not read these numbers as kernel-only differences.
