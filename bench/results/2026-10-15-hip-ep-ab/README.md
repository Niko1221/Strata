# HIP expert parallelism: served A/B, 2x RX 7900 XT

This is the served A/B for `--expert-parallel-device` (PR #5036), recorded on the original `ep-hip` branch
before it was rebased onto `c62d5e3a` (and later onto `ffa80d73`). The rebased head was built and its parity tests were run, but the served
A/B was not repeated on it.

**Machine and model.** 2x RX 7900 XT 20 GB (gfx1100), ROCm 7.2.4, Ryzen 7800X3D; Qwen3.8-Flash-Next 176B Q6K,
4,096 context, `--spec 4 --spec-min-p 0.5`, greedy.

**Arms.** Both arms are the same `strata-ep` build; only the split mechanism differs:

- pipe: `--layer-split auto --pipeline-windows 2`
- EP: `--expert-parallel-device 1`

The fixed four-request protocol, then a 400-token decode and a 128-token output capture. `run-clean.txt` is the
raw output of the clean window (2026-10-15 00:09-00:20). Values are `gen_tps`.

| arm | run 1 | run 2 | run 3 | run 4 | 400-token decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| pipe | 44.8 | 60.0 | 74.5 | 85.1 (still climbing) | 59.8 |
| EP | 84.8 | 90.3 | 90.3 | 90.4 | 80.4 |

The pipeline arm was still climbing at run 4; the EP arm settled from run 2. That is +6% over the pipeline's
best warm run (85.1) and +21% over the banked 74-78. The outputs are not byte-identical because the arms'
resident sets differ (the pipeline computes about 8% on the CPU with different rounding); the plan/row parity is
`ep_parity`.

Card busy/power during the 400-token decodes (12 s samples that include idle time, so lower bounds): pipe card0
24.0% / 51 W average / 178 W peak, card1 38.4% / 59 W / 152 W; EP card0 20.0% / 24 W / 63 W, card1 31.9% /
59 W / 160 W. The >80% busy target was not reached; the gain is overlap, not saturation.

A first attempt at the same window (2026-10-15 00:00-00:04) was contaminated by another engine loading on the
box: the pipeline 400-token run fell to 11.0 tok/s and the EP engine was OOM-killed during capture. Its partial
numbers are not a result and are not used above.
