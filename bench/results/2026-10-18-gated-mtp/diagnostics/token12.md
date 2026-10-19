# First divergence in the truncated GSM8K answer

The partial GSM8K run includes one truncated answer, test index 119. Upstream `296b59d2` repeatedly reconsidered the salary calculation and eventually answered after 6,508 tokens. The tested patch at `96c3060d` continued reconsidering until the 8,192-token limit. Both used the recorded prompt, per-question seed, temperature 0.6, top-k 20, top-p 0.95 and min-p 0.05.

Four fresh 12-token runs reproduced the beginning of both historical outputs: unmodified upstream, unmodified patch, and an instrumented build of each. Instrumentation recompiled only `generate.cpp`; each arm retained its own engine, sampler and drafter libraries. The first 11 output IDs agree; output 12 is `1` upstream and `Let` in the patch. Instrumented and original outputs match within each arm, as do the accepted/offered counts (8/8).

## Failed admission chooses a different sampling path

Both runs reach row 2 of the same three-row verification window. Both draw `Step` from q, and both fail the 0.5 confidence gate:

| At output 12 | Upstream | Patch |
| --- | ---: | ---: |
| q(`Step`) | 0.12743318 | 0.08618915 |
| p(`1`) | 0.42061323 | 0.38435130 |
| p(`Let`) | 0.12889933 | 0.12903755 |
| Proposal rows supplied | 2 | 3 |
| Last-row behavior | Plain target sample | Explicit NULL residual |
| Uniform used | 0.11587322 | 0.96034712 |
| Actual GPU token | `1` | `Let` |

The patch's retained measure has positive mass only on `First`, q = 0.80097252. Its normalized residual is `1`: 0.45786955, `Ad`: 0.36571725, `Let`: 0.15371969, and `Given`: 0.02269351. The residual uniform falls in the `Let` interval. `Step` never reaches the acceptance-uniform test in either arm because it already failed admission.

All 24 target sampling decisions and 18 recorded draft draws reconstruct to the actual GPU outputs. Sixteen offline crossings of the two p distributions, two q distributions, plain/residual weights and plain/residual uniforms select `1` with the plain uniform and `Let` with the residual uniform. This isolates the counter-domain change as the determinant of this particular output split.

## Incremental target comparison remains open

The two incremental target distributions differ at this same emitted prefix: total variation is **0.03626194**. The first nonzero post-filter difference appears at output 6, with total variation 0.00744047; both still emit ` problem` there.

The patch changes native CPU expert arithmetic during probabilistic decode to singleton-equivalent evaluation. This trace has not isolated the numerical difference against a singleton reference. Earlier comparisons that re-prefilled a whole fixed prefix produced identical raw logits; those measurements concern recomputed states, not the incremental states captured here.

The 12-token experiment identifies the first output split. Its effect on the later long reasoning loop is a separate question. The original full assessment did not capture internal p/q traces; the probabilities here are from the short reproduction.

## Records

- [All 12 rows per arm](token12-trace.json): target distributions reconstructed from captured raw logits, exported q and retained measure, actual output IDs, counters, decision reconstructions and raw-logit SHA-256 values.
- [Counterfactuals and draft-draw checks](token12-counterfactuals.json): all 16 crossings, 18 draft checks and original/instrumented work counts.
- Related Coder-loop report: [#1015](https://github.com/Niko1221/Strata/issues/1015#issuecomment-6252449846).

The reproduction used Windows, Ryzen 5 7600X, RX 7900 XTX and Coder IQ1_M. It generated 48 tokens in total; model runs took 68.851 seconds and the isolated diagnostic builds took 33.942 seconds. Current memory pressure changed the RAM-source arena across starts, so these traces provide no throughput comparison. Full raw logits are retained in the local investigation archive; the public records above contain their checksums and derived distributions.
