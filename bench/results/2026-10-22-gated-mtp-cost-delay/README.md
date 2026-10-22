# Delayed draft-policy feedback: diagnostic measurements

These are the two completed diagnostic rounds used to investigate request-local window selection in PR #3416. They ran on the Windows RX 7900 XTX machine with the IQ1_M model, file-backed experts and a 10 GiB resident budget. U is the genuine v0.2.1 binary (SHA-256 `3aba1a97f6b12bcfd8d1064d43e5648df2a396ab2f1a9d8cafdce110448081a5`); A is the frozen-policy candidate at `470d558b`; C adds the delayed-feedback patch submitted in `5469c3e0`.

## Measurement conditions

Both rounds enabled `STRATA_DECODE_TIMING=1`. The final round's C/D binaries additionally contain window counters and a test-only fixed-lookup hook; `STRATA_IR_COST_DUMP=1` was enabled for C/D and the fixed-window arms, but not U. The fixed-window environment variable was set only for T4/T5/T6. These are diagnostic comparisons, not an uninstrumented production performance qualification.

The 16-block round runs English, repeat, English, repeat in each engine, alternating temperatures 0.6 and 1.0. The 8-block final round runs English, repeat, repeat. Engine order was randomized within blocks; the saved schedules retain seeds and request order. The runner appended prepared thinking completions to the prompts so the measurements cover continuation after that prefix. They therefore differ from the earlier `speed-v021` run, which included generated thinking tokens. Do not compare their absolute speeds across rounds.

`delay-measurements.json` and `final-measurements.json` preserve all 192 and 144 measured requests, their DONE lines, output token IDs, diagnostic output, schedules and binary hashes. Machine paths, build logs and runner internals are omitted from these extracts. The source-record hashes identify the original local records. All measured requests produced 256 tokens and stopped at the length cap.

Run `python3 recompute.py` to reproduce `summary.json`. Each block's repeated-text rate is the total tokens divided by total decode time for both repeat requests. The relative change is C/comparator minus one, computed per block, then averaged. Intervals are 95% Student-t intervals across blocks; ratios of overall mean speeds need not equal these mean paired ratios.

## Findings

| Comparison | Blocks | Repeat change, 95% CI | English change, 95% CI |
|---|---:|---:|---:|
| C vs U, first round | 16 | −0.46% [−1.93, +1.01] | +1.36% [−7.41, +10.12] |
| C vs A, first round | 16 | +1.78% [−1.41, +4.97] | +2.31% [−1.37, +5.99] |
| C vs U, final round | 8 | +0.74% [−3.51, +4.99] | −0.83% [−8.28, +6.62] |

These intervals do not establish performance equivalence. The request-level window measurements explain the frozen-policy behavior: A uses mean lookup windows of 5.87 and 3.38 for the two repeat requests, versus C's 4.31 and 4.05. The retained timing feedback lets the policy react within a request.

The fixed-window-5 diagnostic arm was faster than C on repetitive text: C/T5 − 1 = −9.55% [−12.58, −6.51]. This is an observed comparison for that workload, not a theoretical speed ceiling or grounds to change the default window. The alternative cost metric D did not show a clear improvement.

An earlier final-round attempt was stopped after one engine and the start of another because Windows commit pressure reduced the effective RAM residency budget to zero. It produced no complete cross-arm block and is not included in the completed-round estimates. The original aborted record is retained locally.

## Correctness and revision scope

The policy unit test at `5469c3e0` was independently compiled and run on macOS during this review; all 25 assertions passed. The supplied Windows records also report passing `draft_policy_test`, `gated_proposal_test` and spec-admission default/pick/top tests. The broader CUDA/HIP/SYCL/Metal build and numerical checks in the parent evidence directory apply to `470d558b`, before the delayed-feedback change.

The prior PR description additionally reported synthetic Markov distribution experiments. Their harness and raw outputs are not included in these measurement files, so this report does not independently certify those numerical claims. Finite statistical tests alone do not prove exactness. The code-level condition to review is that a cost observation cannot affect selection while any random counter on which it depends remains available to future windows.
