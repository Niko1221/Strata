# Opt-in wider speculative batches

Review base: the QFUSE correctness branch on upstream v0.1.40.1. Upstream
already supports concurrent MTP. This adds wider target verification across
independent slots while retaining upstream admission, prefix reuse, request
cancellation, checkpoints, preemption, and return-to-solo behavior.

## Change audit

| Change | Why | Boundary / validation |
| --- | --- | --- |
| Causal slot segments, predecessor mapping and accepted-prefix commits | Verify several proposals per slot without committing rejected rows | 9840 CPU layouts; GPU recurrence: 16 layouts, 100 partial commits |
| Shared MTP weights with private slot state and residual staging | Avoid duplicate weights while keeping histories separate | Per-slot state is reset on admission; native lifecycle comparisons |
| Fixed physical layouts and bounded graph caching | Reduce repeated captures when proposal widths change | Padding is neither emitted nor committed; cache includes effective merged mode and residency |
| Independent draft-stream overlap | Reduce serial drafting waits | Capture precedes launch; synchronize before consuming results |
| 8/12/16 packed rows with bounded execution groups | Improve useful work per shared weight read | Slot segments remain intact; no claim that all 16 rows share one weight read |
| Merged expert work and optional parallel dense groups | Share expert execution while allowing independent projections to overlap | Private quantization and PLE scratch prevent buffer aliasing; QFUSE uses separate quantization in merged mode |
| CPU expert calls tiled beyond eight rows | Existing specialized kernels have bounded row capacities | AVX2 and AVX512 tails are processed; native wide parity fixtures |
| Per-slot suffix proposals alongside MTP | Reuse repeated prompt/output sequences | Only committed target tokens enter history; rejected/padded rows do not |
| Optional adaptation between speculative batches | Refresh residency during long concurrent decodes | Defers while prefill/borrowed buffers/pending transfers make it unsafe |
| Optional prefill row/chunk limits, row fairness and PCIe share | Expose tradeoffs between decode work and new-request latency | Opt-in; upstream settings remain the defaults |
| Effective-mode and timing diagnostics | Distinguish verified rows from emitted output tokens | Engine logs; Python monitoring is a separate PR |

## Compatibility and use

The new path is off by default (`--batch-spec 1`). It requires 2..4 slots,
one NVIDIA GPU, MTP, text, resident int8 KV and no helper expert cache. On
unsupported configurations it falls back to upstream batching; it does not
disable AMD or multi-GPU concurrency. Upstream `--batch-mtp` remains available
when the wide path is inactive. No AMD/multi-GPU runtime coverage is claimed.

For an existing suitable model config, an opt-in example is:

```
--batch 4 --batch-spec 4 --batch-rows 16 --batch-spec-fixed --batch-draft-overlap
```

`--batch-suffix` uses the existing suffix policy; `--suffix-draft 0` disables
lookup. `--batch-adapt`, `--batch-rows-fair`, `--batch-prefill-chunk`,
`--batch-prefill-rows`, and `--batch-pcie-frac` remain experimental controls.
`STRATA_BATCH_DENSE_PARALLEL=1` opts into parallel dense groups.
Consult `strata --help` / generate help for exact argument forms. These settings
are not inserted into setup defaults. No sampling defaults are changed.

## Reproduce the structural checks

Configure the usual CUDA/HIP build with STRATA_BUILD_TESTS=ON and
STRATA_NATIVE_EXPERTS=ON. Build batch_segments_test, batch_segment_gdn_test,
expert_multi_test, native_expert_parity and strata. Run the corresponding CTest
entries, including native_expert_parity_wide_iq2_xs and
native_expert_parity_wide_iq2_xs_q2_0. The GPU recurrence test needs GPU hardware.

The existing `tools/batch_test.py` and `tools/batch_interleave_test.py` accept
`--exe`, `--config`, and `--extra`. Use the same model/config for each arm and
pass the opt-in arguments above through `--extra`. Follow BATCHING.md's exact
comparison controls; additionally disable grouped CPU kernels and prefill
borrowing when reproducing the historical common-arithmetic controls. Preserve
normal kernels/adaptation for performance runs, and record those as a separate
comparison. Keep expert capacity, context, sampling and prompts matched.

## Fresh workload measurements

See [essay and tool-driven coding results](PR_WORKLOAD_BENCHMARKS.md). On the
undervolted RTX 5090 / 9950X3D / 48 GB DDR5-6000 setup, median four-stream
decode was 146.2 to 200.1 TPS for essays and 167.1 to 325.7 TPS for the coding
fixture. The baseline includes only the two-line shared-MTP-head metadata fix
already present here; untouched upstream MTP fails on this model. The report
includes whole-run rates, individual runs, failed checks and turn-limit counts.
These are throughput observations, not proof of equivalent task quality.

## Earlier evidence, interpretation and limits

Historical integrated-candidate tests used RTX 5090, Ryzen 9950X3D, Swift IQ2_XS,
98304 context/request and 2560 MiB reserve. This source split does not establish
a new speedup. A v040 integration screen measured 296.09 aggregate common-c4
decode TPS versus 158.19 with upstream MTP **inside the same integrated binary**.
It was not a pristine upstream-v040 A/B, adaptive outputs differed, and the
result does not establish equal-quality workflow speedup. A subsequent bounded
QFUSE-fix before/after screen measured 295.78/297.13 TPS and 47/47 ms p95 gaps;
that checks for an obvious regression, not a statistically significant gain.

Controlled c4 output matched 4096/4096 tokens from the corrected v039 candidate.
Wide and upstream-MTP lifecycle suites each matched eight transitions to solo.
Normal adaptive full-logit comparisons passed the recorded absolute screens
(mean KL <= .001, max KL <= .02, mean TV <= .01). Cross-build mean KL was
.000364..000666 for the general fixture and .000167..000182 for coding, comparable
to measured self-repeat variation. The stricter historical .0001 trigger still
flags pairs, including self-repeats: it is not claimed to pass. Two repetitions
are not a statistical equivalence study. The small answer screen scored 15/16
on both builds, with the same error; it is not a broad quality benchmark.

The complete earlier evidence remains at local tag
candidate-v040-numerics-20261006, under docs/evidence/v040-numerical-followup and
docs/evidence/v040-integration. The local review bundle maps those reports to
these extracted sources and records fresh preparation checks. Machine-specific
launchers, configs and large diagnostics are deliberately outside this patch.
Public summaries, provenance and fresh test logs are linked in
[PR_QUALIFICATION.md](PR_QUALIFICATION.md). AMD/multi-GPU testing remains unavailable; no new c=1 throughput gain is
asserted. The fresh baseline comparison and its minimal correctness fix are
documented separately in PR_WORKLOAD_BENCHMARKS.md.
