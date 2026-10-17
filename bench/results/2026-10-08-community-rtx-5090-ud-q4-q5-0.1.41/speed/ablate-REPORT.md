# Strata ablation 20261008-170929

Host big-office, Strata checkout `cb8e9be`, engine 0.1.39, production config `/home/chris/src/Strata/strata-iq3_s.json`.
Prompts: 4096, 32768 tokens x 3 runs per cell, greedy, reasoning off, 256-token cap, identical prompts in every cell, fresh engine per cell, one discarded 32K warm-up.
Noise thresholds used for verdicts: decode 6%, prefill 3% (mean change across prompt sizes, against the first baseline).

**Drift check (baseline end vs start):** 4096: decode -0.6% prefill -0.4%, 32768: decode -0.4% prefill -0.5%. Effects smaller than this are not distinguishable from run-to-run drift.
Same-config output repeatability: 6/6 identical greedy outputs (median common prefix 1162.5 chars). This is the floor for the 'same outputs' column.

## Results (medians; change vs first baseline)

| cell | status | decode 4096 | decode 32768 | prefill 4096 | prefill 32768 | hit @max | PCIe share @max | drafts | W | other CPU s | same outputs | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 188.4 (+0.0%) | 177.7 (+0.0%) | 5,288 (+0.0%) | 6,428 (+0.0%) | 0.982 | None | 79% | 439.1 | 0.035 | - |  |
| model=ud-q4_k_xl@b48@Strata-0.1.41 | ok | 109.7 (-41.8%) | 111.8 (-37.1%) | 2,230 (-57.8%) | 4,302 (-33.1%) | 0.901 | 0.0 | 83% | 400.8 | 1.98 | 0/6 | worse |
| model=ud-q4_k_xl@b48+vision@Strata-0.1.41 | ok | 101.2 (-46.3%) | 104.1 (-41.4%) | 2,335 (-55.8%) | 4,164 (-35.2%) | 0.875 | 0.0 | 82% | 366.7 | 0.405 | 0/6 | worse |
| model=ud-q5_k_xl@b48@Strata-0.1.41 | ok | 60.6 (-67.8%) | 64.4 (-63.8%) | 1,267 (-76.0%) | 2,528 (-60.7%) | 0.832 | 0.0 | 79% | 288.1 | 2.5 | 0/6 | worse |
| model=ud-q5_k_xl@b48+vision@Strata-0.1.41 | ok | 51.8 (-72.5%) | 60.9 (-65.7%) | 817.1 (-84.5%) | 2,421 (-62.3%) | 0.806 | 0.0 | 79% | 276.3 | 7.815 | 0/6 | worse |
| baseline-end | ok | 187.2 (-0.6%) | 177.0 (-0.4%) | 5,267 (-0.4%) | 6,394 (-0.5%) | 0.982 | None | 79% | 437.8 | 0.04 | 6/6 | within noise |

Decode ranges (min-max per size) and every per-run number are in `cells/*/result.json`; GPU/CPU samples in `cells/*/telemetry.csv`.

## Combination

Not needed: 0 winning knob(s).

## Suggested production change

Model cells: each changes the weights and the engine build, so outputs differ by design and the verdicts are speed only. Quality is measured separately (strata_quality.py). Model cells run without the image encoder; the baseline is production as configured.

## Validity checks

- Runs where the engine's prompt count differed from the local tokenizer: 36.
- Max reused (cached) prompt tokens in any measured run: 0 (should be 0).
- Production restore: answering on :8080 after 646s total.
