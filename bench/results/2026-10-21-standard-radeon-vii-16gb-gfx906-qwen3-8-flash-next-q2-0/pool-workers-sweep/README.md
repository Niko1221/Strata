# `--pool-workers` sweep (quick suite, same machine and configuration)

These ten runs use the community guide's quick suite (`tools/strata_bench.py`, the one `bench.sh` runs), not the
standard protocol above. They exist because the protocol fixes its prompts, thinking and sampling for cross-machine
comparison, while this question needs one setting changed at a time with everything else held fixed: on this 16-core
CPU the engine's default for `--pool-workers` is 15 workers, and the sweep asks whether a smaller pool decodes faster.

Same machine, model, engine and configuration as the standard run above (Radeon VII 16 GB, gfx906; Qwen3.8-Flash-Next
Q2_0 with MTP; `--expert-cache auto`; `--kv int8 --kv-resident 32768`; `--spec 4 --spec-min-p 0.5`; whole card, no
`--expert-vram` cap; release v0.2.1.4, engine 0.1.46.3, source build, no local patches). The quick suite sends only
`temperature: 0` with thinking off, and its numbers are **not** standard-protocol results.

## The six arms

| context | `--pool-workers` | chat decode | chat first token | 4K prompt | 16K prompt | agent decode | agent cache |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| 65536 | not passed (= 15) | 38.9 (32.1-42.6, n=9) | 0.79 s (0.67-0.98) | 494.4 | 519.0 | 45.1 (21.3-53.0, n=15) | 68.37 % |
| 65536 | 10 | 45.3 (40.4-53.7, n=9) | 0.66 s (0.55-0.74) | 494.8 | 506.2 | 47.1 (9.5-60.5, n=15) | 68.52 % |
| 65536 | 6 | 43.2 (37.1-52.5, n=9) | 0.69 s (0.57-0.80) | 489.4 | 504.9 | 48.9 (18.2-58.9, n=15) | 68.57 % |
| 262144 | not passed (= 15) | 37.3 (31.6-41.6, n=9) | 0.86 s (0.66-0.94) | 489.8 | 497.4 | 38.3 (21.8-51.6, n=15) | 68.53 % |
| 262144 | 10 | 45.8 (40.3-53.5, n=9) | 0.68 s (0.57-0.76) | 489.1 | 504.5 | 49.4 (16.8-57.3, n=15) | 68.57 % |
| 262144 | 6 | 44.7 (37.5-52.8, n=9) | 0.71 s (0.58-0.81) | 491.3 | 504.6 | 50.5 (27.9-61.1, n=15) | 68.58 % |

Files: `65536-default.json`, `65536-pool10.json`, `65536-pool6.json`, `262144-default.json`, `262144-pool10.json`,
`262144-pool6.json` — each the suite's own `results.json` for that arm. At 65536 the KV stays in VRAM and the expert
cache holds 8052 slots; at 262144 the KV is streamed (`32768 of 262144 cells per QSA layer in VRAM, the K/V in 3.09 GiB
of pinned RAM`) and the cache holds 7417 slots.

What the sweep shows: chat decode is 11-23 % faster with 10 or 6 workers than with the engine's own 15, and 10 and 6
are within noise of each other. Prompt processing does not move (489-519 tok/s at 4K and 16K in every arm), so the
difference is in decode only. The engine's log calls the pool PCIe-bound on this path (`the CPU pool takes 136 us per
missed expert (10.2 GB/s of expert bytes), the link 13.5 GB/s`).

## Repeats, and whether the gap is noise

The default arm was run three times at each context, so the effect can be compared with the run-to-run spread of one
unchanging configuration:

| context | default, run 1 | run 2 | run 3 | 10 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65536 | 38.9 | 40.0 | 39.0 | 45.3 | 43.2 |
| 262144 | 37.3 | 38.8 | 35.0 | 45.8 | 44.7 |

The three default runs span 2.8 % at 65536 and 10.2 % at 262144. Against the pooled default (27 requests per context)
the smaller pools are +16.2 % (10) and +10.8 % (6) at 65536, +22.8 % and +19.8 % at 262144; permutation tests over the
9-request medians give p < 0.0001 for 10 in both contexts and p = 0.0016 (65536) / 0.0001 (262144) for 6. At 65536 that
is 3.8-5.7x the replicate spread; at 262144 the spread is larger (10.2 %) and the effect 1.9-2.2x it — still outside
the noise, but the 65536 comparison is the cleaner one.

Files: `65536-default-r2.json`, `65536-default-r3.json`, `262144-default-r2.json`, `262144-default-r3.json`.

## Limits

- Quick-suite numbers, not standard-protocol results: `temperature: 0`, thinking off, no sanity checks.
- The two contexts are not comparable with each other (65536 keeps the KV in VRAM, 262144 streams it); only the arms
  within one context are.
- MTP is configured but the suite's requests trigger no drafts, so these are not MTP-active numbers.
- The agent test carries more run-to-run noise than the chat test (see the ranges).
