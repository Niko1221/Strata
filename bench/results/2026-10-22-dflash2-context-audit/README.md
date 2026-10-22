# DFlash2 long-context proposal audit

The requested approximately 32% long-prompt advantage over MTP has not been achieved.
This audit found no first-proposal discrepancy against the published drafter on
these prompts. This is not a proof that every implementation path or
model family is correct.

## Proposal quality

RTX 4070 Ti SUPER 16 GiB, Ryzen 5900X, Linux, native NVFP4 Flash-Next target,
unchanged shared Q8_0 output head and BF16 embeddings. Context/resident 8192,
2900 actual expert slots, 11 workers, verifier capacity 8, prefill 512, short-read
64, prompt cache 0, no adaptive swaps/PCIe share/verifier splitting. The same
controlled prompts and genuine target-only continuations as the length diagnosis
were used. Original weights were not changed. No training or launch-parameter
search was performed.

The instrumented native executable saved all prompt and committed target features,
without gaps or overlaps, and logged official BF16 drafter proposals in the same
run at all 95 possible anchor positions of each 96-token target-only continuation.
It logged proposals before suppressing their submission to the verifier. All target tokens match the
common genuine no-drafter controls; no future token was supplied to the drafter.
These forced-single-token diagnostic runs are not throughput measurements.

An independent PyTorch execution of the published BF16 checkpoint consumed the
same captured features, anchor embeddings, absolute positions, local mask and
shared head. The Q8 head activation quantizer in the Python reference approximates
native ordering; this is not a claim of complete bitwise equivalence. The model
and reference source are pinned to Hugging Face revision
`a22a1744e85c17364dcc54a6a420af7135d4c7f6`.

| Prompt | Input tokens | Positions | Native first proposal correct | Reference first proposal correct | First-proposal disagreements |
|---|---:|---:|---:|---:|---:|
| Code short |27|95|90|90|0|
| Code with repeated notes |2427|95|81|81|0|
| Cache with repeated notes |2425|95|83|83|0|

Later proposal IDs differ at 36/665 and 4/665 positions respectively. Only the
available target continuation is used when scoring a prefix near its end.
For padded code, the reference shortens the correct prefix at two anchor positions
(native 5/reference 4; native 4/reference 3), improves none and matches the other
93. For cache, all 95 correct-prefix lengths agree. Simply reproducing reference
BF16 arithmetic therefore does not recover the missing acceptance on this fixture.
For short code, one correct prefix improves from 2 to 7 and the other 94 match;
19/665 later proposal IDs differ. Full rows are in `joint/reference-dense.json`.

A separate read-only replay of the already captured Q8 selector lattices finds
the true first token among the 16 candidates at 95/95 short-code anchors,
94/95 padded-code anchors and 95/95 cache anchors. Selector reranking corrects
two first proposals and spoils two in each code case; it changes neither result
in the cache case. Expanding the candidate list cannot explain most first errors.
The short and padded-code target-only responses themselves diverge at output
index 11 (zero based); only 12/96 output IDs agree at the same index. This is not
a comparison of identical target continuations.

Static checks against the published reference confirm the strict symmetric
`abs(key_position-query_position) < 2048` mask, true absolute RoPE positions and
target output-layer taps captured at the next input boundary. The
[llama.cpp DFlash2 PR](https://github.com/ggml-org/llama.cpp/pull/27342)
also selects from the predecessor-conditioned top-16 lattice and truncates by its
conditional softmax confidence. These source checks alone are not measurements.

## Diagnostic repair and regression coverage

The native stage-dump path incorrectly sized K/V downloads by logical context
pages. A local physical ring has fewer pages. Dump sizes now respect physical
slots and actual KV geometry, including the older prompt-pool diagnostic. The
NumPy stage oracle compares only live context cells; an overwritten historical
logical cell is not expected to retain its old K/V. This repairs diagnostics,
not the ordinary generation path or its acceptance.

The CUDA parity test now also checks a physical page ring at logical end 8197,
window 2048, 8 rows and the checkpoint's 32Q/8KV geometry against its FP64 oracle.
This complements the existing page-boundary, padded-launch and generic-geometry
cases. The physical ring aliases older logical pages: 34 pages for page size 64 and
515 pages for page size 4. The real stage capture uses page size 4.

## Context-work experiment

An opt-in prototype omits whole context-fusion batches before the first future
query's local window, preserving the remaining batch boundaries and arithmetic.
It applies only to fresh native-protocol requests with prompt cache disabled;
reference and diagnostic modes retain full processing. This cannot improve token
acceptance or steady-state decoding: it only removes unused prompt work.
One warmup and three measured repeats per arm, sequentially, on one immutable
executable. All original target-only controls pass. Remaining settings, official
Q8_0 drafter and head are fixed as above.

| Input | Full prompt median ms | Tail prompt median ms | Prompt time change |
|---|---:|---:|---:|
| Code, 27 tokens |439.5|439.4|−0.02%|
| Code, 2427 tokens |5927.4|5927.7|+0.01%|
| Cache, 2425 tokens |5908.7|5903.0|−0.10%|
| Code, 7227 tokens |16897.4|16759.0|−0.82%|

The first three tasks generate 96 tokens. The 7227-token fixture generates 32;
its full and tail continuations also match their own zero-proposal controls.
All output IDs and accepted/offered counts match between arms, including
request resets from a long prompt back to a short one. The larger fixture retains
20/28 acceptance. Three measured prompt times are 16906.9/16897.4/16896.1 ms
(full) and 16796.2/16759.0/16755.3 ms (tail). Its decode medians are
713.4 and 713.9 ms respectively. No steady-state decoding win is established.
The option remains off by default; the measured benefit is limited to prompt
processing on the larger fixture. See `tail-summary.json` and `long-tail.json`.

## Real small-page attention specialization

The native stage fixture confirms four-cell pages, 515 physical pages and
609 logical pages at proposal position 2427. Earlier canonical attention
specialization covered only 64-cell pages. The real four-cell runtime therefore
used generic integer division and four-warp value processing.

The new opt-in specializes four-cell addressing and uses the existing ordered
half2 value accumulation with two warps for long windows. Default execution is
unchanged. FP64 parity includes both page sizes and physical rings; unaligned
values retain the scalar-load path. Thirty attention microbenchmark outputs
(2/4/8 rows, five context lengths, page sizes 4/64) match the pre-change binary
bit for bit, both with the option unset and enabled. The enabled test includes
the newly added unaligned four-cell-page case.

At 8 rows, 32Q/8KV, 2053 logical cells and window 2048, the four-cell kernel
microbenchmark changes from 0.210514 to 0.153646 ms/pass (−27.0%). This uses
200 CUDA Graph replays, not end-to-end generation. Both numerical outputs and
original measurement logs are preserved. Whole-engine results follow the
separate fixed-setting comparison; a kernel improvement is not a +32% claim.

The repaired long-context stage capture also passes the independent NumPy
oracle for all five layers, live K/V, final normalization and full-vocabulary
conditioned selector. It uses the official BF16 checkpoint and tolerances in
`tools/dflash2_stage_parity.py`, not bitwise PyTorch parity. Its eight target
output IDs match the common target-only continuation.

## Evidence and limitations

Raw features, immutable diagnostic executables, their original instrumentation,
model files and full logs remain outside the checkout under `../Strata-data`.
The small analysis outputs and drivers are preserved here. The incomplete first
attempt at short-prompt capture is excluded: its old server instrumentation did
not capture prompt rows processed through the decoder path. The preliminary
long comparison also used separate capture and proposal trajectories. The final
joint capture fixes both limitations: prompt decoder rows are recorded, and
features plus proposals come from the same run. All three joint captures pass
complete-coverage checks and use one-token verifier windows.

HIP, SYCL, Metal, Windows, other target families, multi-GPU, vision and concurrent
requests were not tested. The existing PR remains a draft. No result here
establishes the requested +32% improvement over MTP.

## Whole-engine small-page comparison

One warmup and three measured repeats for each task and arm, sequentially using
one immutable executable, the same fixed launch settings above and genuine common
target-only controls. Prompt processing and startup are excluded from decode TPS.
All controls pass, and DFlash accepted/offered counters remain identical.

| Task | DFlash control tok/s | Small-page kernel tok/s | Change | MTP tok/s | DFlash vs MTP |
|---|---:|---:|---:|---:|---:|
| Code short |48.952|49.077|+0.26%|38.030|+29.05%|
| Code padded |40.759|40.915|+0.38%|36.813|+11.15%|
| Cache repeated |50.622|51.135|+1.01%|51.026|+0.21%|

The short-prompt ranges overlap. The three long-prompt samples do not overlap
between the kernel arms, but these remain small gains on one machine and three
repeats, not evidence of a universal gain. The +32% target is still unmet.
The small-page microbenchmark's later default-path control is 0.198943 ms/pass,
versus the earlier 0.210514 and enabled 0.153646. All three produce identical bits;
include this timing variation when interpreting the earlier 27% kernel figure.
See `small-page-summary.json` and raw per-arm JSON/logs.


## Reproduction and final checks

`joint-diagnostic-source.patch` adds diagnostic-only recording to the production
sources used here. It records full proposals, then zeros returned confidences to
force target-only verification. Do not use that executable for speed measurements.
The patch is not part of the production engine. Apply it with
`git apply --unidiff-zero`, build it, preserve the executable
as `../Strata-data/dflash2-context-audit-20261022/strata-joint-audit`, reverse the
patch, then run `joint_audit.py` and `joint_reference.py` with the recorded model
files and Python dependencies. Drivers require fresh output directories and do
not download models. Earlier raw captures remain outside the checkout. The
original production sources were restored immediately after preserving the
joint diagnostic executable.

The restored production executable SHA256 is identical to the executable used
for all three small-page engine arms:
`42a56c9a217ddb47c58fa6f42ad331c77ae2c296a532d84c53b3610fedb4a3f2`.
No diagnostic suppression is present in that binary.

One additional causal implementation hypothesis was rejected without native
integration: a per-request cache of unanimous, twice-observed two-token-context
successors, learned only from already verified output, could override a selector
candidate. Replay of captured full candidate lattices matches every baseline
accepted/offered count, but makes zero overrides on all three continuations.
It therefore does not improve acceptance. No launch parameters were searched,
and no such cache was added to the engine. `causal_cache_replay.py` records the
algorithm and causality; these are counterfactual diagnostics, not throughput.


Passed: CUDA build with DFlash2 enabled, CUDA build with DFlash2 disabled,
`dflash_load_test`, `dflash_policy_test`, default and enabled `dflash2_parity --bench`,
bitwise before/default/enabled attention-output comparisons, the long NumPy stage
oracle, all fresh native integration controls, joint feature/proposal capture,
and Python syntax compilation of the modified stage oracle. The diagnostic patch
also passes `git apply --unidiff-zero --check` against the restored production sources.

Failed experiments: the causal selector cache replay made no changes and was not
integrated. The initial short-prompt feature capture was incomplete and excluded;
the final joint capture passed. Neither optimization improves acceptance or
achieves +32% over MTP.

Not run: HIP/SYCL/Metal/Windows builds, other model families, multi-GPU, vision,
concurrent requests and fault-injected resource failures. Existing compiler
warnings remain; no new backend compatibility is claimed.
