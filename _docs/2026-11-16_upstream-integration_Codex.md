# Hakua 0.6.0 upstream integration evidence

Status: implementation and validation in progress. No 0.6.0 release tag,
installer overwrite or public asset verification is claimed by this record.
The requested end state still includes those actions after the source gates.

## Frozen intake and scope

Fork: 6f13c7555a7d50289f0e3587ab780627c65a8000.
Official: 09b603886925994d0d4461857b01321ef3aa431c.
Base: f33c8c30588ff00e24b6b63bad0bf6a14da19081.
Worktree: H:/Strata/worktrees/upstream-sync-20261116 on
codex/upstream-sync-20261116. Primary main remains at the frozen fork input.
The merge is uncommitted. All original conflict paths have semantic resolutions;
snapshot data came from real fixture drivers, not manual JSON selection.

The first reviewed integration was committed as
394234d911fc2056af7a9bb2b38df75f94bf0a6a with the fork and original official
inputs as parents. Final source-cut checking found seven more official commits
through baf16363b4403ef0ed4f538f5d5ed11c43fb9535. A second Python-orchestrated
three-way intake covers their twelve files, including Mac Drafting Off avoiding
draft downloads, once-only offline notices and accurate preparation progress.
Fork versions stay 0.6.0. Official Terminal 0.2.14.1 and Control 0.1.60.1 are
recorded independently from app 0.2.14 and engine 0.1.60. The refreshed official
release API contains 39 assets; the earlier 36-asset snapshot remains historical.

The authoritative local evidence root is
H:/Strata/evidence/upstream-sync-20261116. Each guarded command has its own
immutable receipt and hashed receipts/NAME.command.txt output. The source
helpers in _docs/upstream-sync-20261116 preserve before/after digests and
the three immutable input identities. Historical failed and interrupted runs
are retained. A zero wrapper exit does not by itself prove a test count.

## Observed validation

Python core task: 294 passed/1 skipped; server: 523 passed/4 skipped;
setup slice: 392 passed/1 failed/2 skipped and the corrected spawn-policy slice:
18 passed. See the [Python report](2026-11-16_upstream-python-compatibility_Codex.md)
for retained earlier failures. Fresh snapshot replay passed 12 cases; the
snapshot full run finished with 59 passed/1 failed. Its earlier-collected
Strix comparison has a separate six-case passing correction; this does not
convert the historical full run to a pass. The orphan-process review finding
was corrected fail-closed and its 32-case focused run passed. Independent
scoped spec and quality re-review approved Task 1.

The subsequent complete snapshot-final-v3 run passed all 60 tests in
1328.28 seconds, with no skips and a drained owned Job. It finished before
the additional seven-commit source intake; that intake has separate regressions.

Host: updated Vite typecheck and sealed production build passed. The selected
32-file run has 285 passes and one environment failure; the latter was repaired
in the filtered test environment, then both Spin8 verifier tests passed under
host-resume-spin8-final-admitted. The unchanged file-drop assertion passed
under the full original filesystem/isolation runner. The broad suite hit its
900-second deadline; partial output is not a complete-suite pass. See the
[Host report](2026-11-16_upstream-host-compatibility_Codex.md).

After the Node security update, host-node243-security-regression-v1 passed
138 cases with three explicit optional-Harness skips under the original
filesystem isolation policy. Fresh typecheck and sealed public build passed.
The actual DeepSeek executable checked at admission is now also the canonical
absolute executable used at launch; saved configuration bytes are preserved.
Independent scoped spec and quality re-review approved this correction.

Source-bound CUDA spill tests: spill-tests-v3 has 31 passes and ten explicit
POSIX-script skips. Upstream known-instance diagnostics cannot relax the default
strict gate. Vendored ggml is bound by its actual content digest; external
ggml checkouts retain the clean commit requirement. Compiler/configuration
bindings include the new MMQ/window/quant options.

Native CPU initial build passed after correcting MSVC test allocation pairs.
Initial CTest ran 123 cases with two failures and three explicit skips. The
stack-overflow and invalid Windows fixture path causes were corrected; the two
focused cases passed. Fresh cpu-final-tests-v3 completed all 123 cases:
120 passed, three explicit skips, zero failures. The complete CPU rebuild passed.

CUDA configure succeeded with CUDA 13.3, MSVC 14.44, sm_120, portable static CRT,
Spin8/native-cache options and optional MMQ enabled. cuda-build-v1 was stopped
with RESOURCE_PRESSURE/125 when H free fell below the unchanged 10 GiB floor.
Its owned Job finished with active processes zero. Both engine executables
exist. cuda-build-v2 then completed all targets with five compiler jobs after
fresh capacity admission. Thirteen core tests passed. The first serial GPU
group had seven passes and two failures. The subsequent correction run passed
all four affected tests, including all six Q36 format pairs. Admission/minimum
free VRAM were 14,409/11,795 MiB against the unchanged 768 MiB reserve.
Independent spec and quality review approved the correction slice. Complete
engine executable relinking and authoritative build binding subsequently passed.
Compiler code-page warnings remain recorded.

The final nine-case serial GPU replay passed all nine, with minimum free VRAM
10,481 MiB against the unchanged reserve. Both executable version/help probes
passed all four checks. Compute Sanitizer reported zero errors for both KV
lifecycle cases, and Spin8 first-use/full synthetic checks passed. These are
compiled capability and numerical checks, not model-quality measurements.

The initial default CUDA resource check failed on three Q36 specializations.
The bounded scheduling correction reduces NVFP4 STACK from 32 to 16 bytes.
IQ2_S and IQ2_XS remain at STACK 24, LOCAL zero, against the unchanged default
16-byte limit. The default check therefore still fails on those two entries.
Mandatory WW4 reference tests now cover all three exact formats; both final
runs passed, and the registered Q36/original numerical group passed three cases.

Independent review approved the explicit SM120/CUDA 13.3/MSVC 14.44 measurement
profile. The controller accepts that profile only for its bound source,
compiler and vendored ggml identity. It records 451 measured nonzero entries;
unlisted entries retain zero allowance. Its SHA-256 is
a6a68113670400e151a37c92d8d7c3b158043d14d5e0081f0d89d8c39ae8ce67.
The cost remains 24 bytes per thread for those two IQ2 entries. This is not a
repair of the default limit, a blanket waiver, or a model/performance claim.
The final five-job product relink, configured-profile replay and authoritative
native identity passed. All nine measured library hashes remained unchanged.
The resulting strata.exe SHA-256 is
0d8bc027c649bfde788c72ba61be025dfa013679ceb48e3b49069d14c7d10ab4.

Hosted Linux checks at 4282fce exposed stale-request process ownership and a
shared-memory fixture assumption. The corrected server retains the initiating
process throughout STOP, drain, timeout, DONE and fatal-error handling. A stale
request keeps its own result without changing its successor's state. Focused
replacement tests reproduced two failures before correction, then passed 36
nearby cases and 13 final ownership/fatal-bookkeeping cases. Independent scoped
re-review approved the correction. Hosted replay of the final source remains
separate. The vision fixture observes the real write in controlled scratch;
production shared-memory placement and refusal rules remain unchanged.

The segmented-cache error contract intentionally remains fail-closed: a driver
error blocks further serving until successful cleanup and reopen. Accurate
mapped-prefix ownership is retained. Simulated fault hooks do not establish
recovery from an actual CUDA driver fault. Extracted-source CPU diagnostics
passed their focused case; the broader historical group remains ten passed,
one failed because its unchanged fixture requires a missing old executable.
No removed model or old executable was restored, and its input gate was not
weakened. See the [native report](2026-11-16_upstream-native-compatibility_Codex.md).

All 32 snapshot cells/128 entry records were generated to snapshot-all-v1 and
copied byte-for-byte after completeness checks. Twelve setup entries explicitly
name unsupported fixture drivers. snapshot-resolution/receipt.json binds all
three original input sets and generated output hashes. The first full gate was
resource-stopped and exposed Qwen-only drafting test assumptions applied to the
restored DeepSeek family. The follow-up now separates Qwen drafting fixtures
from the retained DeepSeek matrix; see the replay results above.

Installer/documentation focused checks: install-docs-tests-v2 has 24 passes and
ten platform skips. release-manifest-v2 passes. doc-links-v2 checks all 418
then-current pages and anchors successfully. Eight root README translations
are rewritten for the candidate, preserving explicit qualification limits.
Additional documentation needs a final link/version check after delivery.

## Security and resource records

Python runtime audit: 25 dependencies queried, zero skipped, zero known
vulnerabilities returned. Host lock remediation: Vite 6.4.3 to 6.4.6, npm audit
zero. Desktop npm audit: 12 dependencies, zero advisories. Cargo OSV audit and
targeted remediation are detailed in [UPSTREAM_0214](../docs/UPSTREAM_0214.md).
The Windows target graph is saved in cargo-tree-windows-v1.command.txt.

All 47 Windows Rust unit tests passed after the dependency and Node changes.
Their explicit unit-only empty-resource override does not qualify packaging;
the installer must use the unchanged product resource configuration.

The private Node runtime is updated from 24.22.2 to 24.22.3 for the official
November security release. The Windows archive and executable match published
SHA-256 values, and all four platform archive pins are updated. Only the new
worktree runtime was staged; the primary checkout and installed runtime remain
untouched. PGP verification was not performed. New vendored-ggml provenance
binds actual dependency bytes and the upstream pin rather than borrowing the
parent repository commit. Its eight focused integrity tests passed.

No floor was lowered: available physical RAM 3 GiB, commit headroom 4 GiB,
H free 10 GiB and C free 5 GiB. The runtime dynamic RAM stop and GPU reserve
remain independent. No deleted model was downloaded, normal profile edited,
or unowned process stopped in this integration phase.

Reversible NTFS compression examined only the new stopped CUDA build directory.
owned-build-compression-v1 inventories 6,200 files/2,787,415,115 logical bytes;
all bytes and hashes matched afterward, with zero deleted files. The cache
already used compression, so it did not restore admission capacity. The user
selected clearing H themselves. A fresh read then confirmed about 179 GiB free;
build/test admission resumed without changing the floor or moving data to C.

## Required completion evidence still outstanding

The root integration review found and corrected one malformed Metal CI step;
actionlint and scoped re-review passed. Hosted Metal compilation is separate.
The final snapshot replay passed all 60 cases and the final GPU group passed
all nine. Whole-branch review remains separate. Fresh frozen-source app/Control
packages, exact-source hosted checks, standard backed-up overwrite installation,
installed-app acceptance, immutable v0.6.0 tag, installer/tar.7z publication and
anonymous checksum readback remain required. Candidate README status must be
updated from actual delivery receipts. General model quality, full-window
inference and GPU concurrency remain unqualified: releaseQualified=false.

## Separately ordered Q38 work

The user selected implementing Q38 after delivery of 0.6.0, in a separate
workstream. The supplied Q38 plan and CPU admission prototype are read-only
inputs at this stage. Their ZIP SHA-256 is
d8f5e93a9ef282ca715092e30cb9e7dd87297f96d89ea0ecf04926066a4b0b94.
No attached prototype source was executed or integrated into this release.
Its externally reported checks are not local verification evidence.

The later mathematical supplement remains in the same separately ordered scope.
Its ZIP SHA-256 is
8657f353a5a782e9ababe940f32da9780bed0e52fead7135d1128eb3cfc8872f.
The supplied CPU script and claimed results have not been executed or adopted
as local verification. The document separates memory/ownership transformations,
floating-point order changes and approximations; these distinctions will be
assessed against the updated source after this release.
