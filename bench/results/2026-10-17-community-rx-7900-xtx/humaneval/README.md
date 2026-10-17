# HumanEval correctness check

Companion to the [report above](../README.md). The speed suite does not measure answer quality, so the two
Flash-Next quantizations were also checked on a small, execution-graded coding task.

## Method

`coding_eval.py` runs the first 82 HumanEval problems (the official `HumanEval.jsonl.gz`, 164 problems total),
following the HumanEval convention: the model's completion is appended to the prompt, the canonical `check()`
tests are added, and the result is executed in a **subprocess** with a hard 10 s timeout. A candidate that raises,
hangs or fails an assertion is a failure.

- greedy (`temperature 0`), `max_tokens 512`, thinking off (`reasoning_effort: none`)
- markdown fences and leading prose are stripped; any `if __name__ == "__main__":` block is truncated
- `reasoning_content` is ignored; only `message.content` is graded
- 82 problems, sequential, one model at a time

`pass@1` here is single-sample accuracy over the first 82 problems, not the unbiased n-sample estimator. With 82
problems the 95 % confidence interval on a single score is roughly ±4 points.

## Results

| Model | pass@1 | Wall time | Completion tokens |
| --- | ---: | ---: | ---: |
| [Coder IQ1_M](coder_iq1_m.json) | **82/82 = 100 %** | 288 s | 14,854 |
| [Flash-Next IQ2_XS](iq2_xs.json) | 81/82 = 98.8 % | 284 s | 14,977 |

IQ2_XS's single failure was HumanEval/81 (assertion failure).

## Notes

- The two scores are statistically indistinguishable at this sample size. The useful reading is that neither
  quantization is materially crippled as a coder by its ~1-2 bit weights — which is not obvious a priori for IQ2_XS.
- [`qwen27b_q4.json`](qwen27b_q4.json) is included for reference only: a dense Qwen3.8-27B IQ4_XS served by
  **llama.cpp**, not Strata, scored 79/82 (96.3 %) in 619 s. Failures were HumanEval/32 and 81 (SyntaxError) and
  40 (AssertionError).
