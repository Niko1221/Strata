# Code and non-code holdout transfer

Author: **Francisco Angulo de Lafuente** · [ORCID](https://orcid.org/0009-0001-1634-7063)

This follow-up makes the specialization tradeoff explicit: a code-trained profile
can improve code coverage while covering less routing from other workloads.
It responds to [ManfredCh's evaluation suggestions](https://github.com/Niko1221/Strata/pull/5668#issuecomment-6813531444)
with a reproducible comparison on this contribution's own logged requests.

All three profiles have **1,582 slots and identical slots per layer within each
fold**. Every evaluated fold is excluded from training. The same five published
folds, seed and 45 requests are retained. A single profile is applied across all
held-out requests; this tests transfer rather than choosing a profile using the
test request's domain label.

| Held-out cohort | Requests | Original mixed reference | Code-only training | Balanced code/non-code training |
| --- | ---: | ---: | ---: | ---: |
| Code | 12 | 20.07% | 32.21% | 28.57% |
| Non-code | 33 | 16.44% | 7.14% | 12.28% |
| All requests | 45 | 17.41% | 13.83% | 16.63% |

The metric is **mean static routing coverage**, with one held-out request per vote.
The original mixed reference gives every training request equal weight. The
balanced variant gives half the training weight to the code mean distribution and
half to the non-code mean distribution. Its ratio is fixed before this calculation;
no mixture was tuned against these results. Both candidates retain the reference's
per-layer capacity, rather than moving slots to different layers.

Code-only training gains 12.14 percentage points on code and loses 9.30 points on
non-code. The balanced variant gains 8.50 points on code and loses 4.16 points on
non-code. All three original non-code domains regress for both candidates; their
individual values and every request remain in [results.json](results.json).

## Reproduce and check

From the repository root, with Python and NumPy:

```sh
python bench/results/2026-11-09-domain-expert-profiles/holdout-transfer/analyze.py
python bench/results/2026-11-09-domain-expert-profiles/holdout-transfer/verify.py
```

[analyze.py](analyze.py) checks the pinned input hashes, full-fold exclusion, slot
counts and layer quotas. It reproduces all 45 published mixed-reference scores and
all 12 published quota-constrained code-profile scores to within 1e-14 coverage.
The output records each fold's training IDs, layer quotas and selected-set hashes.
The independent [check](verify.py) using Python ordering and integer sums reproduced all 15
profile selections, 135 per-request coverage values and 21 aggregate means.

[plan.json](plan.json) was fixed before this new calculation, after the corpus and
original domain results were already known. It is an exploratory follow-up, not a
preregistration made before seeing the data. The first invocation stopped at a
domain-label assertion before scoring: the manifest uses `stem`, rather than the
display label `STEM`. That label correction, its prior plan hash and unchanged
comparison choices are recorded in the plan's amendment.

Pinned text-input hashes normalize CRLF to LF so the same source checkout can be
checked on Windows or Linux. This portability correction was made after the first
calculation; the amendment records it. All profile selections, request scores and
aggregates were checked unchanged after that correction.

## Keeping a reference and reverting a trial

The predeclared static diagnostic allows **zero percentage-point mean loss in
either the code or non-code cohort**. Both candidate profiles fail it. This
diagnostic rejects a general-purpose substitution on these fixed traces; it does
not certify runtime quality or automatically choose, install or revert a profile.

This analyzer constructs fold-specific selections for evaluation and exports no
deployable candidate profile. The main tool's `--profiles-dir` writes its existing
global/domain bank; it does **not** export this follow-up's balanced profile. A
future live experiment needs a checked exporter for its declared weighting and
capacity before that candidate can be tested. No balanced-profile live trial is
provided or recommended here.

For a manual live trial with a separately verified deployable candidate, preserve
the known-good profile file, its SHA-256 and full configuration before starting.
Store the candidate separately without overwriting the reference. Select it
explicitly with the existing `--expert-profile` option. If its frozen code or
non-code acceptance criteria fail, stop that trial and restart with the saved
reference profile and configuration; verify the restored hash and repeat the
reference checks. This is an operator procedure, not a new hot-swap or automatic
rollback implementation.

Before any runtime promotion, freeze independent code and general-conversation
prompts, quality oracles, cold/warm timing protocol, repetition count and per-cohort
latency tolerance. Both cohorts must pass their declared criteria; a code gain
must not compensate for a general-cohort failure. Keep all runs and return to the
reference if either cohort fails. This follow-up does not supply those missing
live measurements or authorize a default/runtime change.

## Observations

- These are fixed-trace placement measurements, without model loading or GPU work.
  They do not measure tokens/s, generated-answer quality, copy costs or adaptation.
- The 33 non-code requests comprise STEM 11, languages 6 and other 16. This is not
  an independent general-conversation quality benchmark. The small corpus gives
  no population-level guarantee, confidence interval or optimal mixing ratio.
- ManfredCh's timings remain a separate contributor-reported experiment on a
  different model format, engine and device. They were not pooled with our data.
- The earlier [live pilot](../live-pilot/README.md) covers only coding tasks and
  retains its failed frozen output-parity/improvement gates. Its interpretation
  is unchanged. A runtime feature still needs maintainer agreement on the
  [proposed serial request-boundary warmup](https://github.com/Niko1221/Strata/issues/4978#issuecomment-6744443860).
  This analysis does not implement that feature or establish scope approval.
