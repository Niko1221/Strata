## Summary
Two views of the completed measurements in #1491: equal weight for all 15 categories, and the original preferences for agentic coding workflows. Q8 is labeled Reference. The reference now uses the Q8 retest on the same RTX PRO 6000 and mainline binary as ISTA/Swift. This analysis makes no engine changes and does not replace official benchmark scoring.

## Equal categories
Each of the 15 categories receives exactly 1/15 (6.6667%) of the score, including safety and structured output. Score = (100/15) * sum(earned category points / available category points), using exact fractions. Every category section has the same width; earned credit is filled and missed credit stays empty before the next section.

![Equal category weights](https://raw.githubusercontent.com/CC-David-CC/Strata-a5500/docs/agentic-code-weighting/bench/analysis/agentic-code-20261008/figures/equal-category-ranking.png)

ISTA IQ3_XXS 93.56; Gyro-S 93.05; Swift IQ3_XXS 92.46; Reference 92.11. This averages category fractions rather than the benchmark's overall point total. Gyro's instruction-following category uses four scored tests instead of five because TC-45 was excluded; it still receives 1/15. No new inference or test rescoring was performed.

## Original preference weighting
Tool selection 25%, error recovery 25%, context/state 20%, arguments 10%, planning 10%, multi-step chains 5%, code patterns 5%. Safety, structured output and all other categories have zero weight.

Score = sum(weight * earned category points / available category points), using exact fractions before rounding. These are preference weights, not a validated coding-success probability.

Each category keeps its full weight budget: earned points are colored, missed points stay empty, and the next category starts at the same position for every model.

![Weighted ranking](https://raw.githubusercontent.com/CC-David-CC/Strata-a5500/docs/agentic-code-weighting/bench/analysis/agentic-code-20261008/figures/weighted-ranking.png)

![Weights and examples](https://raw.githubusercontent.com/CC-David-CC/Strata-a5500/docs/agentic-code-weighting/bench/analysis/agentic-code-20261008/figures/weighting-explained.png)

## Model families
- ISTA: base-model quantization using GSQ refinement and RCO precision allocation.
- Swift 1.5: UkisAI post-training plus Swift-specific quantization; not simply another quant of identical weights.
- AP: Agention Precision, mixing standard quant formats by tensor group.
- Gyro: custom rotor-coded experts in a rotated basis, requiring compatible runtime support.
- Reference: Q8 retested on the same RTX PRO 6000 + mainline, with memory-mapped experts and a BF16 compatibility pack; not ground truth. The old P4 receipts remain historical evidence.

![Model-family guide](https://raw.githubusercontent.com/CC-David-CC/Strata-a5500/docs/agentic-code-weighting/bench/analysis/agentic-code-20261008/figures/model-families.png)

## Findings and limits
ISTA IQ3_XXS and Swift IQ3_XXS tie at 97.00; Gyro-S scores 96.33, Swift Q2_0 95.00, and Reference 91.21. One point in selection or recovery moves the score by 4.17, so the leaders are a shortlist rather than a statistically established order. All models saturated the small code-pattern group; repository-editing quality remains unmeasured.

Independent calculation checked all 12 scores against the original committed reports. Four figures are included as PNG/SVG with reproducible rendering scripts. Source counts, hashes, weights, formulas and the rendering script are included. The branch starts from main and adds only the analysis directory.

Benchmark credit: [Tool-Eval-Bench](https://github.com/SeraphimSerapis/tool-eval-bench) by [SeraphimSerapis](https://github.com/SeraphimSerapis). Examples are paraphrased. The report links each publisher model card and records differing runtime cohorts.

[Full analysis and publisher sources](https://github.com/CC-David-CC/Strata-a5500/blob/docs/agentic-code-weighting/bench/analysis/agentic-code-20261008/README.md)
