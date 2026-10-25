# Community benchmark: 2x Quadro GV100 (sm_70), Strata 0.2.5 (engine 0.1.45), Unsloth UD-Q4_K_XL

Measured on 2026-10-25 with `tools/strata_bench.py` inside the official Docker
image. Same machine and power caps (150 W per card) as the 2026-10-18 report
(`../2026-10-18-community-2x-gv100/`).

Headline: **0.2.5 matches 0.1.45 on this hardware** — chat decode 69.1 tok/s
(vs 67.6–71.9 across the three older sessions). No measurable gain, no
regression.

## A note on the prompt numbers

The suite's short prompts make its `prompt tok/s` (~110) misleading: the
fixed per-request overhead dominates at ~50-token prompts. The engine's own
logs and `GET /metrics` show real prefill speed of **~1,500 tok/s** on large
prompts (46,610 tokens read in 31 s), and warm requests reuse almost the
whole prompt.

## Live metrics (real use, not the bench)

From `GET /metrics` during normal agent traffic the same day:

| Metric | Value |
| --- | --- |
| Decode (per request) | 66–90 tok/s, mean ~77 |
| Draft acceptance | 20,023 / 28,089 = 71% |
| Expert cache hit | 97–99%, PCIe spills ≤ 0.7% |
| Prompt reuse | 4,374,603 of 4,564,639 tokens (96%) |
| VRAM in use | ~32.1 GiB of 32 GiB per card |

Engine settings at the time: `--spec 6 --spec-min-p 0.5`, MTP runtime,
KV int8, context 131,072, auto expert cache and auto layer split.
