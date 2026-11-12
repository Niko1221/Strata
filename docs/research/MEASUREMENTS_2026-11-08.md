# Measurements 2026-11-08: stock upstream vs this fork, Q4_K_M experts, small-VRAM notes

Single RTX 3060 Ti, 8 GiB, Windows 11, CUDA 13.3 (sm_86). Prompt for every grid point is
`master.txt`, greedy, 256 response tokens. Primary sources are named in each section; the full
Russian/English grid is `docs/measurements/grid-2026-11-08.md` of the stand repository.

## 1. llama.cpp: KAT-Coder-V2.5 IQ4_XS, stock `3cf03257` vs this fork

Flags on both sides: `-ngl 99 --n-cpu-moe 30 -ctk q4_0 -ctv q4_0 -fa on -b 2048 -ub 2048 -t 6 -tb 8`.
Stock points were measured with a 64-token response and re-measured with 256; the table uses 256
(`stock-gpu-256` tag), because decode does not warm up on 64 tokens.

| window | stock pp | stock tg | fork pp | pp gain |
| ---: | ---: | ---: | ---: | ---: |
| 64k | 546.9 | 7.39 | 1050.5 | +92 % |
| 96k | 489.4 | 6.54 | 987.3 | +102 % |
| 128k | 477.4 | 6.78 | 890.8 | +87 % |

The fork has no 80k/160k point for this model in the 2026-11-08 grid: 160k is beyond the point
budget on 8 GiB and was not taken. Stock tg stays at 6.5–7.4 tok/s on both response lengths, so the
gap is not an artefact of a short answer.

Sources: `E:\tmp\queue-stock-gpu-256.log`, `grid-llama-2026-11-08.jsonl`,
`docs/measurements/grid-2026-11-08.md` §3.2 and §3.4.

## 2. Strata: KAT IQ4_XS, stock `146df3c4` vs this fork

Stock Strata has no `--kv-type`: KV is fp32 (~45 KiB/token), so it cannot reach long windows at all
and its ladder stops at 32k. The fork runs KV q4_0 + FA + super-chunk + split-K + pinned arena.

| window | stock pp | stock tg | fork pp | fork tg |
| ---: | ---: | ---: | ---: | ---: |
| 2k | 109.6 | 19.94 | — | — |
| 8k | 124.8 | 17.62 | — | — |
| 32k | 123.1 | 11.34 | — | — |
| 64k | not reachable (fp32 KV) | — | 1504.0 | 18.1 |
| 96k | not reachable | — | 1308.9 | 15.3 |
| 128k | not reachable | — | 1091.9 | 14.6 |
| 160k | not reachable | — | 959.9 | 13.6 |

The stock prefill sits at ~109–125 tok/s regardless of the window, i.e. ~10× below the fork at 96k.
That is the cost of fp32 KV plus the absence of the attention path, not a different model.

## 3. Qwen3.6-35B-A3B heretic Q4_K_M in Strata: before and after the Q4_K down fix

Before the patch the engine refuses the model: the down experts are Q6_K while gate/up are Q4_K,
and `is_iq()` does not instantiate Q4_K for the down projection — the server answers 503 with
"down-эксперты Q4_K не инстанцированы".

After `q36: down-эксперты в Q4_K — инстанцировать group-эксперты` (worktree `engine-q4kd`):

| window | max-context | prompt tokens | pp | tg |
| ---: | ---: | ---: | ---: | ---: |
| 64k | 65,536 | 64,660 | 1386.5 | 16.8 |
| 80k | 81,920 | 77,378 | 1273.0 | 16.5 |
| 96k | 98,304 | 86,589 | 1215.6 | 16.0 |
| 128k | 131,072 | 121,808 | 1026.1 | 14.3 |
| 160k | 163,840 | 139,293 | 932.4 | 13.1 |
| 176k | 180,224 | 139,276 | 942.0 | 14.1 |
| 192k | 196,608 | 139,276 | 940.7 | 13.9 |

Against KAT IQ4_XS on the same profiles the model is 3–8 % slower, i.e. the quant is served
correctly. Quality gate: needle at 96k — 6 records, 0 errors, 4/6 full-phrase hits, 6/6 when only
the number counts.

## 4. Hypotheses checked this day

* **HAGS on/off** (Windows hardware GPU scheduling): 96k KAT ×3 per state — pp 781.5 / 793.5 /
  545.8, tg 15.5 / 16.4 / 17.0 for off-before / on / off-after. No gain from the scheduler; the pp
  spread is page-cache warm-up. `HwSchMode` left at 1 (disabled).
* **`-ub` as the cause of the prefill collapse in llama Q4_K_M**: rejected — at `-b 2048` the
  ubatch is clamped, 64k with `-ub 8192` gives 87.3 pp against 92.0 with `-ub 2048`.
* **The 64k/80k/128k Q4_K_M collapse is not the machine**: measured on the same hour when HauhauCS
  IQ4_XS gave 838.9 pp at 96k and Q5_K_S at 160k gave 279.1, while Q5_K_S at 96k gave 47.0.
* **`-lv 4` is required for a usable primary source**: it adds `n_ubatch = 2048` and the KV buffer
  size (540 MiB at 96k q4_0), which `-lv 3` never prints.
* **VRAM headroom as the cause of the llama prefill collapse** (series J/K, 08-09.11): confirmed.
  At `--n-cpu-moe 30` some allocation layouts silently evict weights (60.65 pp at 86 558 tokens
  against 894.85 in a morning run of the same prompt); 31/32/33 are healthy (794.44 / 858.10 /
  843.41 pp), and free MiB alone does not predict the collapse (31 holds more VRAM than the
  collapsed 30). Full record with per-number sources: `VRAM_HEADROOM.md` and `VRAM_HEADROOM.ru.md`.

## 5. Reproduce

* llama fork grid: `harness/scripts/run_llama_fast_*.bat`, `llama_grid.py --family …`;
* stock llama: `run-stock-llama-gpu-256-r2.bat` (tag `stock-gpu-256`);
* Strata fork/stock: `harness/config/strata-*.json` + `live_strata.py`;
* heretic Q4_K_M: `strata-heretic-q4km-*.json` on the `engine-q4kd` worktree, needle gate
  `needle_run.py --cfg strata-heretic-q4km-96k --tokens 86589 --port 8099`.
