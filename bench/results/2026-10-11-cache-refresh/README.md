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

The combined Responses lifecycle and 32K branch records are retained in
#1667's matching report. The API history store remains independent of native
KV cache retention: invalid/evicted snapshots replay saved history. Applications
using automatic disk switching should not also restore an application-owned
conversation snapshot before each request. Disk-only execution remains
single-device; cross-backend snapshot portability is not claimed.

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

## Strengthened reproducible policy follow-up (2026-10-11)

`--reproducible` now also forces `STRATA_ROUTE_TAIL_SKIP=0`. The ordinary eligible
CUDA default rank 7 is approximate and uses expert ranks across the token window;
changing draft rows can change a target row's missed-expert contribution. This
follow-up controls that source of variation without changing ordinary defaults.

Reproducible saved-state configuration fingerprints now include policy version
2. Older unversioned reproducible files are refused as an invalid configuration;
non-reproducible fingerprints remain unchanged. Real native SAVE/RESTORE tests
verified an ordinary old snapshot restores, an unversioned reproducible snapshot
is refused with the process still alive and able to generate again, and a new
versioned snapshot saves/restores successfully. The pre-version binary already
contains tail-skip disabling, isolating the fingerprint-version change.

The final combined native binary (source `c2ee849433c65524597ec2ad67ec91f20c4b5c0c`)
has SHA-256 `cd0efebf01f5a8058ece0e596f634f8cf8b701d27949cb4cd67008d5b5a33835`.
Snapshot records and both binary hashes are in `session-policy/`.
Seeded disk-only and RAM-to-disk fallback checks passed again after versioning
(temperature 0.7, seed 12345, MTP width 2); full IDs and restored-prefix counts
are retained. Before versioning, the same strengthened routing policy also
passed RAM, disk and fallback checks, plus a six-request mixed-placement floor
comparison with an explicit rank-7 environment override. HIP gfx1012 native
cache tests and SYCL build/host cache tests passed after the fingerprint change.
These checks do not establish model execution on HIP or SYCL.
