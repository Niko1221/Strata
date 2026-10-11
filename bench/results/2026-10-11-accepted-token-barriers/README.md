# Accepted-token adaptive barriers: current-main validation

Native source `4b8e0ee4e68a3dbe7f4b25e69a78abb5869901de`, based on upstream
`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`. The opt-in barrier holds expert
placement for a block of accepted tokens, counts accepted routing only, then
drains and applies blocking swaps before starting the next block. Rejected
speculative rows must not affect those swaps. Ordinary execution remains off.

The refresh fixes admission ordering: a positive `STRATA_ADAPT_BARRIER_TOKENS`
must resolve `--adapt-async 1` to blocking adaptation before current main checks
three-stage pipeline eligibility. The previous ordering disabled the pipeline
before the later override. The original setup failures remain in the campaign
checkpoint; the measurements below confirm actual pipeline controls and coverage.

## Fresh model evidence

Three Tesla P4s, CUDA sm61, ISTA IQ3_XXS, int8 KV, 8192 context, layer split 19/37,
MTP width 4, resident RAM experts, 27 pool workers, no PCIe sharing or prompt/
conversation reuse. Every compared process starts from the same recorded expert
profile and follows the same ordered code/prose/counting requests twice.
Inputs contain 28/26/31 tokens; output cap is 256, with actual finish/counts saved.

| Source / barrier policy | Requests | Full IDs and finish | Routing/residency hashes |
| --- | ---: | --- | --- |
| Barrier only, 64 accepted tokens / up to 80 swaps | 30 | Match | Match |
| Barrier only, 17 accepted tokens / up to 160 swaps | 30 | Match | Match |
| Barrier + CPU dispatch fix, 64 / 80 | 30 | Match | Match |
| Barrier + CPU dispatch fix, 17 / 160 | 30 | Match | Match |
| Current main versus feature off | 12 | Match | Not a barrier-hash comparison |

Each enabled matrix includes separate serial, pipeline, forced rollback,
no-guess and serial-repeat processes. Required nonzero speculative/rollback
coverage passes. All observed boundary positions, swap counts, accepted-routing
hashes and residency hashes match across the five arms. These hashes support the
measured comparisons; they are not a digest of every model/native-state byte.
`summary.json` retains coverage and executable hashes for all five campaigns.
The feature-off check preserves the ordinary routing policy and explicitly uses
barrier value 0. It is a finite serial serve comparison, not proof of all modes.

The combined source includes #1818 and a private optional route-row diagnostic;
`provenance.json` identifies it separately. Its trace is disabled in these full
matrices. The barrier-only matrices already pass with explicit IQ dispatch
control, so these results do not impose #1818 as a runtime dependency.

## Exact-routing controls and retained negative evidence

Enabled comparisons set `STRATA_IQ_MT_MIN=1` and `STRATA_ROUTE_TAIL_SKIP=0`.
The eligible CUDA rank-7 default is approximate: minimum expert rank across all
rows in a window determines whether a missed expert contributes. Adding draft
rows can change target routing contributions even with unchanged placement.
The barrier feature does not change that ordinary policy.

Two earlier same-binary rank-7 matrices are retained under
`default-tail-negative-controls/`. Re-auditing at matching request histories
still finds pipeline differences (for example tokens 191/30 at interval 64,
and 73/203 at interval 17), while the rank-0 replay matches. Their original
outputs/logs are not rewritten as successes. These observations distinguish
approximate window-dependent routing from the accepted-token boundary policy.

The original gate required every repeat to match the very first reference,
which is unsuitable for evolving adaptive placement. Original `result.json`
files retain that failed static audit. `matched-history-result.json` files are
independent re-audits with the explicit mode from #1813: compare the first arm
at the same `(repeat, prompt)`, reject reordered/duplicate histories, check input
hashes and finish/counts, forbid reuse, and preserve coverage requirements.
Both forms are retained. This changes the audit reference, not native outputs.

## Reproduce and scope

Use the standard-library gate from [#1813 at cd4d0670](https://github.com/CC-David-CC/Strata-a5500/blob/cd4d0670/tools/spec_window_parity.py).
For each campaign, substitute matching paths in `matched-history-manifest.json`,
including the expert profile described by `profiles.json`, and run against a new
output directory. The combined CPU source is a separate validation composition;
use the barrier-only source for the independent feature check.

```sh
python /path/spec_window_parity.py --audit barrier-full-exact-1779-64/matched-history-result.json
python /path/spec_window_parity.py --audit barrier-full-exact-1779-17/matched-history-result.json
python /path/spec_window_parity.py --audit 1779-current-default-off/result.json
# Expected nonzero: retained approximate-policy negative control.
python /path/spec_window_parity.py --audit default-tail-negative-controls/1779-barrier-64-admission-fixed/matched-history-result.json
```

The current CUDA source file hash matches the local source and HIP/SYCL exports.
HIP gfx1012 and SYCL builds passed; these are compatibility checks, not fresh
HIP/SYCL model-execution evidence. The separate SYCL engine does not implement
this option. Exact binary hashes are in each result; source provenance is in
`provenance.json`. Full model/pack revision hashes were not captured.

The four enabled matrices took 415.487/426.554/420.454/425.866 seconds of
unattended campaign time; the feature-off comparison took 120.022 seconds.
Loading and all requests are included in these campaign durations; engineering
setup time is separate and not inferred from them. No fresh general speedup is
claimed. Historical three-P4 throughput claims retain their old source/settings.

Not certified: other quantizations, arbitrary pipeline depth, suffix-supplied
windows, sampled decoding, remote/helper modes, cross-hardware numerical identity,
long production workloads or general answer quality. Unsupported runtime modes
remain refused rather than silently changing their semantics.
