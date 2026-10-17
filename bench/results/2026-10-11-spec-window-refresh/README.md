# Current-main speculative-window checks

Upstream source: `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`.
The gate adds no native runtime changes. Eleven CPU-only gate tests passed.
All runs use greedy 128-token requests and two repeats; manifests retain exact
input IDs, arguments, binary hashes and controlled expert placement.

| Campaign | Requests | Result |
| --- | ---: | --- |
| Current-main serial/pipeline/floor/rollback/no-guess | 48 | Full token IDs and finish reasons match; required speculative/rollback coverage passes |
| Current-main default CPU dispatch, differing floors | 12 | Exact outputs match |
| Copy suffix control | 8 | Outputs match, but pipeline suffix coverage is zero: gate correctly fails |
| Matched local/TCP on current #1748 | 24 | Exact outputs and finish reasons match |

Exact comparison requires `STRATA_ROUTE_TAIL_SKIP=0`. The eligible CUDA default
rank 7 is an intentional approximation: the minimum expert rank across a token
window determines whether a missed expert contributes. Different draft windows
can therefore change accepted-token outputs. Earlier current-main tests with
rank 7 differed at tokens 73/123/70; they must not be interpreted as proof of a
pipeline rollback bug. The rank-0 replay changes this routing policy explicitly;
normal engine defaults remain unchanged. CPU arithmetic and expert placement
are controlled separately (`STRATA_IQ_MT_MIN=1` for the pipeline matrix).

TCP is absent from plain current main. Its matched local/remote comparison uses
PR #1748 source `f2673bd25244546231fb98f9b7bb9f6b460c23e7`, rebased onto the
same upstream main. All processes use the same binary and expert profile, with
512 slots per stage, partition 16/32 on three P4s and loopback workers. Rank 0
is propagated consistently to workers. This validates serial MTP over TCP,
not remote overlapped pipeline windows or real LAN performance. Private worker
credentials are replaced with placeholders in the published manifests.

The suffix result is deliberately retained as a failing coverage gate. Enabling
suffix drafting does not establish that the pipeline supplied a suffix window.
The missing path is not certified by these measurements.

Recheck with `python tools/spec_window_parity.py --audit <result.json>`.
The pipeline, default and TCP results should pass. The suffix result should exit
nonzero with `copy-suffix-pipeline: suffix_windows=0 < required 1`.
Historical measurements remain separately dated under
`bench/results/2026-10-10-p4-spec-window-parity/`.

Not established: arbitrary prompts, dynamic adaptation, sampled decoding,
other formats/backends, CPU/GPU bit identity, quality or performance gains.
