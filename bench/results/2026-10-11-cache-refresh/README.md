# Current-main cache integration validation

Tested together: #1667 Responses history/summary/branch checkpoints and #1810
native disk caching, based on upstream `61b3fb5d`. This is integration evidence;
the two PRs retain distinct code responsibilities. #1667 preserves #1446's
Responses groundwork and #1489's live probe/historical reports; #1810 already
contains #1489's native snapshot work, including absent-MTP handling.

Native source: `5d8be3316c59bc16eb6b3f3b6c309dec183a8f59`. Engine SHA-256:
`151a1ea30b7ddd92782fb98809febfdbb7f7ff3977c19dc5f119c60985526202`. The Python-only checkpoint identity correction was
then tested in combined source `c11af10c59b91a3135ca23bf3d704e753669ccc6` with this same native binary.
It reads the first three fields of current engine restart metadata rather than
unpacking the older five-field tuple. The regression checks current metadata,
environment identity and fingerprints of both model shards.

Hardware/model: RTX PRO 6000 Blackwell 96 GB, CUDA sm120, ISTA IQ3_XXS.
Native conversation cache, split-failure, memory, file, spill and session-reclaim
tests passed. The combined Python serving suite passed 750 tests, with 12 skips;
the post-fix checkpoint/HTTP suite passed 30 tests. Windows #1667 serving checks
separately passed 748 tests, with 16 skips, before the identity correction.

## Exact native token checks

All six configurations passed: RAM, disk-only and RAM-to-disk fallback, each
greedy and seeded sampling (temperature 0.7, seed 12345), with MTP width 2.
Each compares a fresh process against the cache process across A-first, unrelated
B, A-return and A-repeat. A output IDs match across fresh, returned and repeated
requests; A-return confirms nonzero saved-prefix reuse. The JSON files preserve
every output ID and native timing/statistics field. These are short controlled
prompts, not broad sampling or model quality coverage.

HIP gfx1012 builds and cache/file/spill/validation native tests passed. SYCL
engine builds and host cache/file/spill tests passed. These are compile/host
checks, not fresh HIP or SYCL model-run parity claims.

## Responses and branch lifecycle

The isolated real-engine disk-only Responses probe passed 33/33 checks, with
MTP disabled. It covers durable history, conversation switching, summaries
on/off, deletion, cancellation, graceful/abrupt restart, corrupt-KV fallback,
and a disk budget smaller than a snapshot. It uses task-owned history and spill
paths; production state is not accessed.

The plain-server 32K checkpoint benchmark passed all 14 requests and marker
checks, including three fresh/restored pairs, branching before the answer,
bookmarked ancestry retained through shutdown, ordinary cache cleanup and
replay after deleting owned checkpoint blocks. Each restored 32K pair reused
32,761 prompt tokens. Median first-token time was 4.473s fresh and
1.244s restored (three observations each). Startup, model-artifact
identity hashing and shutdown are excluded from those request timings.
Filesystem caches were warm; this is a lifecycle/reuse demonstration on one
model, not a general performance estimate. No private profile was included.

The initial benchmark exposed the stale engine tuple assumption; its failure
was fixed, covered by a focused regression and rerun in a new scratch directory.
Earlier [integration evidence](../../../docs/RESPONSES_DISK_INTEGRATION.md)
retains its original date/source and is not presented as this refresh.

## Reproduction and machine time

Use `tools/conversation_token_parity.py` with an idle GPU and a local model
configuration; select each `--tier`, `--temperature 0` or `0.7`, `--seed 12345`,
`--spec 2`, a new `--output`, and `--run`. #1667 also retains
`tools/responses_disk_live.py` and `tools/bench_branch_checkpoints.py` for isolated
lifecycle probes. Follow their disposable-path checks.

Unattended machine time: CUDA build 85.072s, six native tests 1.322s, serving
suite 253.783s, six token-parity configurations 172.195s, Responses live probe
82.247s. The corrected checkpoint/HTTP plus 32K campaign took 189.478s.
Engineering/setup time was not instrumented and is excluded. These sequential
operation durations must not be treated as hands-on work time.
