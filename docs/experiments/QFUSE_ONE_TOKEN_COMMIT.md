# Supporting evidence for PR #1209: QFUSE one-token state commit

Independent regression evidence supporting the one-token commit eligibility guard in [PR #1209](https://github.com/Niko1221/Strata/pull/1209). The seven recorded cases used current upstream `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253` and an isolated equivalent guard correction, `e8b3c453e6685a7f1779177ee73f8b26c27201e0`. The reproduction patch has four added and two removed lines. This documentation branch changes no engine code. The full #1209 branch, its quantizer changes and concurrent-generation features were not tested by this evidence package.

## What went wrong

With `STRATA_QFUSE=1`, one-token graph capture disabled self-commit, but
`Verifier::commit` still skipped the explicit commit graph. The recurrent
state was not advanced. This also matters when MTP commits a one-token window.

Make the shared `one_token_self_commit()` decision return false under QFUSE,
so capture and commit agree. The existing explicit commit path then runs.
QFUSE-off and other backend behavior are preserved.

## Measured checks

RTX PRO 6000 Blackwell Workstation Edition 96GB, Ryzen 9 7950X, 128GB RAM,
Ubuntu 24.04.5, CUDA 13.2, full Unsloth Q8_0, FP16 KV, fixed expert placement.
Each of seven cases ran a 128-output pure Python arithmetic-function check
and a fresh 8,192-input / 512-output coding request. The long generated module
was not executed or scored.

| Comparison | Function check | Exact function tokens | Exact 512 coding tokens |
|---|---|---|---|
| Parent QFUSE off / fix QFUSE off | Both passed | Yes | Yes |
| Parent QFUSE on, `ONE_TOKEN_COMMIT=0` / fix QFUSE on, default commit setting | Both passed | Yes | Yes |
| Same workaround / fix with MTP T4 | Both passed | Yes | Yes |
| Parent QFUSE on, default commit setting | **Failed as expected: missing arithmetic function** | Negative control | Repeated code fences; invalid performance result |

The function prompt was:

> Write only a Python code block defining triangular(n). Use the formula n*(n+1)//2. No imports, function calls, tests, or explanation.

The checker accepts only a function made of arithmetic expressions, then checks
inputs 0, 1, 2, 10, 100 and 10000. It does not execute arbitrary model code.
Parent and fix used identical inputs, allocation, placement and ordinary
timing policy. Adaptive swaps, rotation, duplex, secondary cache, PDL and
DeepGEMM were disabled. There is no speedup claim: the broken path's apparent
throughput is invalid. These checks do not establish internal state-digest
equality, broad model quality or cross-hardware coverage.

[Compact source, binary, input/output hashes and results](qfuse-one-token-results.json).

## Scope of support

PR #1209 places the missing `!g_qfuse()` guard in `Verifier::commit`. The isolated test correction makes the shared eligibility predicate false under QFUSE, so capture and commit agree. Both restore the explicit commit on the affected one-token window. These results support that commit decision; they are not validation of the full PR.
