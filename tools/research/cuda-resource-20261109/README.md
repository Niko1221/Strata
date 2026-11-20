# DFlash compiler-resource candidate — not enabled

This research patch records the isolated candidate from
`H:/Strata/validation/cuda-resource-20261109/candidate-B-dflash2.diff`.
It is not applied to production, selected by CMake, or installed in the app.

The baseline reproduced all602 original resource records on CUDA13.3.73/MSVC
sm_120. Candidate B changes only the two failing DFlash rows variants:

| Kernel template | REG before/after | STACK before/after |
| --- | --- | --- |
| `<8,1,8>` | 80/90 | 24/0 |
| `<8,1,4>` | 80/79 | 16/0 |

The other81 DFlash entries retain their resource values. The unchanged guard
passes that candidate object. Higher register use may affect occupancy; no speed
or model-quality improvement is established. GPU parity and timings are NOT_RUN.
Both EXPDEC and QSA candidate experiments worsened spills and were rejected;
the production resource guard's four findings remain unresolved.

See `_docs/2026-11-09_cuda-resource-investigation_OpenCode.md` for frozen inputs,
PTXAS/SASS evidence, failures and the required independent parity/review gates.
