## Summary
All twelve requested model variants now have completed Tool-Eval-Bench results. The latest update retests Q8 on the same RTX PRO 6000 and pinned mainline binary as ISTA/Swift: **97/100 short; 88/100 standard**. Figures now use this reference, while the earlier three-P4 receipts remain available.

Benchmark created by **[SeraphimSerapis](https://github.com/SeraphimSerapis)**: **[Tool-Eval-Bench](https://github.com/SeraphimSerapis/tool-eval-bench)**. Scenarios and scoring are unchanged.

![All twelve variants: short and standard scores](https://raw.githubusercontent.com/CC-David-CC/Strata-a5500/work/tool-eval-variants-20261007/bench/results/tool-eval-20261007/figures/variant-scores.png)

![Category scores and paraphrased test examples](https://raw.githubusercontent.com/CC-David-CC/Strata-a5500/work/tool-eval-variants-20261007/bench/results/tool-eval-20261007/figures/category-scores-and-examples.png)

## Results included
| Variant | Short /100 | Standard /100 | Scored cases |
|---|---:|---:|---:|
| Reference (new Q8 run) | 97 | 88 | 69 |
| Gyro-M | 93 | 86 | 68 |
| AP-Q4_K_XL | 97 | 85 | 69 |
| AP-IQ3_XXS | 90 | 89 | 69 |
| AP-IQ2_S | 93 | 88 | 69 |

## What changed
Completed report, traces, portable deployment settings, resource samples, conversion records and reproducible PNG/SVG figures under `bench/results/tool-eval-20261007`. No engine or benchmark scoring changes in this PR.

## Interpretation
- Six ISTA/Swift runs share the mainline build and RTX PRO 6000. AP uses the same GPU and settings, with the existing Q6 build option enabled for its embedding.
- Gyro uses patched agentionai Strata rc1; TC-45 is excluded because its endpoint does not enforce required tool use. These are not results from the newer Gyro merge branch.
- Q8 now uses the same mainline engine and RTX PRO 6000 as ISTA/Swift, with memory-mapped experts and the required BF16 compatibility pack conversion. Both short and standard suites ran separately. Old P4 results are retained as history.
- Single trials; Swift is post-trained; AP is a mixed-precision recipe. Structured-output API restrictions, stale mock timestamps and background transfers are documented. Examples are paraphrases, not model quotes; all tool actions are mocks.

This benchmark report is independent of the Responses/cache integration in #1489 and the Responses implementation in #1446.
