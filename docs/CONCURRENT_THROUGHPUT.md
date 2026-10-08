# Opt-in wider speculative batches

Review base: pristine upstream v0.1.40.4 (`6674a00`). Upstream
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
| Effective-mode and timing diagnostics | Distinguish verified rows from emitted output tokens | Engine logs; no Python server or telemetry changes in this PR |

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

## Upstream integration and evidence

See [V0404_REBASE.md](V0404_REBASE.md) for the conflict audit, exclusions and
fresh qualification. Historical PR #1209 performance ratios are not the
claims for this rebased submission. Performance is measured first, followed
by bounded correctness and lifecycle checks if the workload benefit remains.
