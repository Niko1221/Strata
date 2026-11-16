# VRAM headroom, not window length: the prefill collapse on an 8 GB card

Status: research, 2026-11-09, updated after series K of the same day. Every number carries a source; `origin` is `raw` (a measurement file), `report` (an
earlier report of this work) or `computed` (arithmetic on cited values). Machine data:
`2026-11-09_session/stageK-chart-data.json`, built by `glm/scripts/chart_data.py`
(149 numbers re-checked against the cited source lines, 0 mismatches; selftest 19/19).

## 1. Stand and subject

- RTX 3060 Ti 8192 MiB (sm_86), Windows 11, CUDA 13.3; i7-4790K, 32 GiB RAM.
- Model of the collapse: `Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf`, 19.78 GiB
  (`2026-11-08_session/grid-heretic-q4km-2026-11-08.jsonl:1:model_path`, size from
  `docs/measurements/grid-2026-11-08.md:223`).
- Binary `llama-master/build-master`, flags `-ngl 99 --n-cpu-moe N -ctk q4_0 -ctv q4_0 -fa on
  -b 2048 -t 6 -tb 8`, prompt `master.txt`, KV q4_0 (`stageJ-vram-cause-2026-11-08.md:3-5`).
- Subject: prefill at 58-141 tok/s where the same code gave 777-895 tok/s.

## 2. Claim, as it stands after series K

The collapse is driven by VRAM pressure at launch, not by prompt length, chunk size, quant or binary.
But the pressure is **not** a simple function of free MiB: section 11 shows a healthy run with less
free VRAM than a collapsed one. What the measurements support is "eviction happens on some
allocations layouts and not others", and `--n-cpu-moe` is the knob that changes the layout.

## 3. Length curve at ncmoe 30 (raw, 08.11)

| prompt tokens | pp | tg | wall, s | source |
| ---: | ---: | ---: | ---: | --- |
| 22 175 | 859.58 | 24.07 | 31.1 | `stageJ-ncmoe-ab.txt:3` |
| 42 441 | 448.01 | 23.79 | 100.1 | `stageJ-length-curve.txt:2` |
| 61 475 | 115.74 | 22.80 | 536.8 | `stageJ-length-curve.txt:5` |
| 86 558 | 60.65 | 13.52 | 1446.1 | `stageJ-A1-r2.txt:3` |

Computed: 44k/22k = 0.521, 66k/44k = 0.258, 86k/66k = 0.524. Decode holds 22.8-24.1 until the last
point, where it drops too (13.52).

The same 86 558-token prompt in the morning gave **894.85 / 21.24**
(`grid-heretic-q4km-2026-11-08.jsonl:3`, sha `53da63631d58958a`) - **x14.75** against the evening point
on one prompt. That pair is what rules out "the window is too long".

## 4. Step inside one run (report)

Inside the 66k run: 995 tok/s at 4k, 646 at 32k, 460 at 35k, 79 at 43k; a 2048-token chunk takes
26-38 s, read by the report as ~550 MB/s of weight traffic (`stageJ-vram-cause-2026-11-08.md:23-25`).
Origin `report`: the raw journal `E:\tmp\llama-J-*.log` is not in the primary list, so the per-chunk
numbers are quoted, not re-measured.

## 5. Checks that closed the alternatives

| check | result | source |
| --- | --- | --- |
| warm page cache (20.4 GiB standby), repeat of 66k | 114.83 against 115.74 cold = -0.8 % (computed) | `stageJ-length-curve-warm.txt:2` vs `stageJ-length-curve.txt:5` |
| second read of the model file | 1055 MiB/s disk, 4169 MiB/s cache (x3.95 computed) | `stageJ-vram-cause-2026-11-08.md:32` (report) |
| CPU during the run | 12 %, 4001 of 4001 MHz | same file `:33` (report) |
| GPU during the run | 100 % utilization, 63 W of 170, 5 % memory - waiting | same file `:34` (report) |
| pagefile | 277 MiB - no swap | same file `:35` (report) |
| repeat of 44k at ncmoe 30 | 450.68 against 448.01 = +0.6 % (computed) | `stageJ-control-and-windows.txt:2` |
| A/B ncmoe 30 vs 34 at 22 175 | 859.58/24.07 vs 827.17/21.88 = -3.8 % pp, -9.1 % tg | `stageJ-ncmoe-ab.txt:3,7` |

The last row is why a short-prompt A/B cannot see the effect: more host experts make a healthy short
prompt worse. The earlier conclusion "it is the length, not ncmoe"
(`stageJ-control-and-vram-2026-11-08.md:32`) was withdrawn by that same file at lines 39-42.

## 6. VRAM by ncmoe: what is raw and what is report

| ncmoe | used, MiB | free of 8192 (computed) | origin | source |
| ---: | ---: | ---: | --- | --- |
| 30 | 7738 | 454 | report | `docs/measurements/grid-2026-11-08.md:502` |
| 33 (08.11) | 6940 | 1252 | report | same line, `msg-14d2c6e.txt:30` |
| 36 | 5515 | 2677 | raw | `stageJ-vram-ncmoe36-monitor.csv` (max over the file) |
| 31 (09.11) | 7867 | 325 | raw | `stageK-series.txt:60` |
| 32 (09.11) | 7548 | 644 | raw | `stageK-series.txt:64` |
| 33 (09.11 repeat) | 6985 | 1207 | raw | `stageK-series.txt:52` |

Series K closed the gap for 31/32/33: those now have 2 s monitors. The 30 figure (7738) and the 08.11
33 figure (6940) are still report-level; the 09.11 repeat of 33 measured 6985, i.e. +45 MiB against
6940 on the same setting, so a single monitor value is not a constant of the profile.

## 7. Windows 64k-160k: morning (28/30) against 33 and 36 (08.11)

| window | prompt tokens | morning pp/tg | ncmoe 33 pp/tg | ncmoe 36 pp/tg | 33 vs morning | 36 vs 33 pp |
| --- | ---: | --- | --- | --- | ---: | ---: |
| 64k | 64 641 | 91.97 / 24.56 | 883.04 / 21.02 | 734.84 / 19.34 | x9.60 | -16.8 % |
| 80k | 77 350 | 58.09 / 21.63 | 871.05 / 20.49 | 722.51 / 18.73 | x15.00 | -17.1 % |
| 96k | 86 558 | 894.85 / 21.24 morning, 60.65 / 13.52 evening | 853.30 / 20.25 | 700.58 / 18.68 | x14.07 vs evening, -4.6 % vs morning | -17.9 % |
| 128k | 121 777 | 140.95 / 17.79 | 799.25 / 18.87 | 674.62 / 17.94 | x5.67 | -15.6 % |
| 160k | 139 262 | not measured | 777.40 / 18.20 | 657.98 / 17.28 | - | -15.4 % |

Sources: morning `grid-heretic-q4km-2026-11-08.jsonl:1-4` (no 160k row); 33
`stageJ-ncmoe33-windows.txt:2,5,8,11` and `stageJ-ncmoe33-96k.txt:2`; 36
`stageJ-ncmoe36-short.txt:2,5`, `stageJ-ncmoe36-long.txt:5`, `stageJ-control-and-windows.txt:5,8`.
Ratios computed. Decode cost of 36 against 33: -8.0 / -8.6 / -7.8 / -4.9 / -5.1 %.
Off-ladder 36 points: 42 441 -> 763.07/19.33 (`stageJ-vram-ncmoe36.txt:2`), 61 475 -> 737.77/19.33
(`stageJ-ncmoe36-long.txt:2`).

## 8. What was changed in the profiles

`--n-cpu-moe` 28/30 -> 33 in the five heretic llama launchers, `-ub 8192` -> `-ub 2048` in the 96k
launcher, `reports/profiles.md` rewritten with both series (commit `14d2c6e`, file list and numbers in
`msg-14d2c6e.txt:8-18,26-31`). No git operation was run here; the commit id comes from that artifact.

## 9. Discrepancies in the primaries (not resolved here)

1. Requested `-ub` against effective chunk: `stageJ-A1-r2.txt:3` and
   `grid-heretic-q4km-2026-11-08.jsonl:3` record `ub: 8192`, while `stageJ-vram-cause-2026-11-08.md:5`
   and `stageJ-control-and-vram-2026-11-08.md:4` state `-ub 2048`. The effective ubatch is clamped by
   `-b 2048` (`msg-14d2c6e.txt:11-12`), which explains the effective value, not the reports naming a
   different requested flag.
2. Two build identifiers: `build: a0039832b` (`stageJ-vram-cause-2026-11-08.md:4`) and sha
   `b561dcf0e1bb7456` (`stageJ-control-and-vram-2026-11-08.md:3`, `grid-2026-11-08.md:316`). Whether
   they are one build was not measured.
3. "12-18 % slower" for 36 (`grid-2026-11-08.md:507`) against 15.4-17.9 % computed over five windows.
4. `stageJ-vram-cause-2026-11-08.md:74-77` still recommends 36, while the later decision is 33
   (`grid-2026-11-08.md:515-516`). The cause document was not updated.
5. The series K report `stageK-vram-audit.md:39` marks 122k as still running, while the raw
   `stageK-series.txt:70` already carries its result (796.85 / 17.78). This draft takes numbers from
   the raw file, not from that report.

## 10. Limits

- One card, one driver; no second machine.
- The morning 96k point is a single run: 894.85 was headroom luck, not a property of the window
  (`stageJ-vram-cause-2026-11-08.md:68-69`).
- Per-chunk step numbers and the CPU/GPU/pagefile/disk checks are report-level; their raw journals are
  outside the primary list.
- The 7738 MiB figure for ncmoe 30 has no raw monitor in either session; 31/32/33 do (series K).

## 11. Series K (09.11): the threshold is not monotonic in free VRAM

Raw source for every row: `2026-11-09_session/stageK-series.txt` (2 s nvidia-smi monitor per point,
`VRAM max=... MiB строк=...` line). All points are 96k (ctx 98304, fill 260 000, prompt 86 558) unless
the last two rows say otherwise; response 128 tokens, `-b 2048 -ub 2048`.

| model | ncmoe | pp | tg | VRAM max | free (computed) | source line |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| heretic Q4_K_M | 30 (08.11 evening) | 60.65 | 13.52 | 7738 (report) | 454 | `stageJ-A1-r2.txt:3`, `grid-2026-11-08.md:502` |
| heretic Q4_K_M | 31 | 794.44 | 19.96 | 7867 | 325 | `stageK-series.txt:58,60` |
| heretic Q4_K_M | 32 | 858.10 | 20.04 | 7548 | 644 | `stageK-series.txt:62,64` |
| heretic Q4_K_M | 33 | 843.41 | 19.81 | 6985 | 1207 | `stageK-series.txt:50,52` |
| KAT IQ4_XS | 30 | 769.56 | 21.12 | 7412 | 780 | `stageK-series.txt:2,4` |
| HauhauCS IQ4_XS | 30 | 973.37 | 21.69 | 7405 | 787 | `stageK-series.txt:6,8` |
| llmfan Q5_K_S | 30 | run cut off (rc=1) | - | 7827 | 365 | `stageK-series.txt:10-48` |
| llmfan Q5_K_S | 33 | 678.73 | 18.04 | 7216 | 976 | `stageK-series.txt:54,56` |
| heretic, 110k (prompt 106 275) | 33 | 817.48 | 18.70 | 7127 | 1065 | `stageK-series.txt:66,68` |
| heretic, 122k (prompt 118 219) | 33 | 796.85 | 17.78 | 7242 | 950 | `stageK-series.txt:70,72` |

What this changes:

1. **Free MiB does not predict the collapse.** ncmoe 31 holds *more* VRAM (7867, free 325) than the
   collapsed 30 (7738, free 454) and is healthy at 794.44 pp. So a rule "free < 500 MiB -> eviction"
   is refuted by our own data; the layout of allocations matters, not only the remainder.
2. **The collapse is not universal at ncmoe 30.** KAT IQ4_XS (769.56 pp, free 780) and HauhauCS
   (973.37 pp, free 787) are healthy at the same setting on the same window. Model size and placement
   decide whether 30 is enough.
3. **The Q5_K_S hypothesis is now confirmed, not assumed.** At 30 the run was stopped by the operator
   once the monitor showed 7827 MiB over 175 samples (`stageK-vram-audit.md:17-18`); the raw block
   records `ConnectionResetError` and `rc=1` (`stageK-series.txt:10-48`), so no pp/tg exists for it.
   At 33 the same model gives 678.73 / 18.04 with 7216 MiB. The caveat "the confirming point has not
   been run" from the 08.11 write-up is closed by these two rows.
4. **Repeat spread of one setting.** 33 on 96k: 843.41 / 19.81 today against 853.30 / 20.25 on 08.11
   = -1.2 % pp and -2.2 % tg (computed), VRAM 6985 against 6940. The answer sha `e41cbd9333a67f80` is
   the same for today's 32, today's 33 and yesterday's 33 (`stageK-series.txt:50,62`,
   `stageJ-ncmoe33-96k.txt:2`), so the text did not change across those settings on this prompt.
5. **Practical choice.** 32 gives the highest pp (858.10) at 644 MiB free; 33 gives a guaranteed
   >= 1 GiB free (1207) for -1.7 % pp against 32 (computed). The profiles stay on 33; picking 32 for
   speed is a decision for the user, not for this draft.
