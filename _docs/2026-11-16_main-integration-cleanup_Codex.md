# Main integration and authorised H-drive cleanup

Date: 2026-11-16 JST.

The user requested integration of the active worktree into main, followed by
commit and push. The user also authorised removal of earlier H:/Strata work
and validation folders while retaining incremental build caches, models,
signing keys, published distributions and backups.

## Integrated source

The active integration source was committed as
`00b26d2cbdb6041d8def535b04a0968042dcb4c2`. Its reviewed tree is
`7b4484256bb56a8e781d45013781776668e20433`.

The protected primary checkout was clean on main at
`6f13c7555a7d50289f0e3587ab780627c65a8000`. Integration produced exactly the
reviewed tree. The focused checks run from the actual primary checkout passed
27 cases. The merge commit is
`32c07800e0c384ab11cdede5dfd3b0be82fadb60`, with parents `6f13c755` and
`00b26d2c`. A normal push and remote readback confirmed origin/main at that
exact commit. The primary checkout was clean after the push and after cleanup.

The official source cut is `baf16363b4403ef0ed4f538f5d5ed11c43fb9535`.
The source remains an unreleased Hakua 0.6.0 candidate. Installation, packaging,
publication and qualification are separate stages. See the
[integration record](2026-11-16_upstream-integration_Codex.md) and the
[retained dependency correction](2026-11-16_upstream-retained-dependencies_Codex.md).

## Preservation and removal

The exact private plan has SHA-256
`9c8c3560395f33ad926745a06c0ab1082f3e941ee45881a0d7f492dfd8877fe3`.
Independent read-only review found no overlap between its removal targets and
protected paths. The executor rechecked source ownership, current worktree
state, backup hashes, the protected main identity and resource admission.

| Item | Recorded result |
|---|---|
| Exact removal targets | 1,260; all absent after execution |
| Planned regular-file count and logical size | 1,234,684 files; 245,213,318,437 bytes |
| Reparse points | 52 links unlinked without recursively following their targets |
| Local published-asset metadata matches retained | 237 files present after execution |
| Additional matching files preserved outside old work folders | 35 retained copies; all recorded SHA-256 values matched |
| Working-file preservation | 1,490 changed or untracked files; 4,442,479,064 bytes, with hashes and binary patches |
| Recovery references | All 13 recorded H worktree heads preserved and verified |
| Current proof archive | All 535 recorded file hashes matched after cleanup |

The two worktrees belonging to an independent H-side repository were preserved
in a complete Git bundle. Git verified its complete history; its SHA-256 is
`3196f0dc797e3ae46389bae28dec55c4fd2636fe3e05d920e773f93f28393cad`.
Small reference inputs were separately copied and verified in the private
backup. Old unmerged branch history and working files were retained for
recovery; their removal from disk does not merge those older changes into main.

All H-side registrations in the primary repository were removed by exact path.
The primary checkout and the four other C-side registrations were retained.
No branch deletion, force push, blanket prune, reset, clean or stash was used.

Models, packs, MTP data, private memory, signing keys, existing private backups,
dependency/toolchain caches and primary resource inputs under H:/Strata/stage
were protected. The selected existing 9B GGUF retained its recorded size, and
the signing key was checked by file metadata only. No private key content was
opened by cleanup or its verification.

The current native CPU/CUDA caches, Cargo cache, reviewed Host/Desktop
dependencies, pinned Python/wheel inputs and Node archive remain. The most
recently used older Cargo cache and CUDA JIT cache also remain. All nine native
library hashes and the native executable hash matched the pre-cleanup records.
The deleted H source path in CMake metadata must be migrated before a new
primary-checkout native build; an old receipt must not be relabelled as a C build.

The executor observed H free space increase from 67,061,723,136 to
270,525,485,056 bytes, approximately 62.5 to 251.9 GiB. These are volume
observations. Logical file size differs from allocated space because of
compression, links and filesystem accounting.

The RAM, commit and disk floors remained 3, 4, 10 and 5 GiB respectively.
The GPU reserve was unchanged; cleanup performed no GPU experiment or runtime
termination.

## Private receipts and remaining delivery work

The detailed inventory, WIP manifest, complete Git bundle, action journal,
completion receipt and post-verification receipt are under
`H:/Strata/private-backups/cleanup-20261116`. The previous integration proof
files are under its `source-evidence-v1` directory. These private files are
outside the public source and distribution inputs.

The old public-UI build input directory named `hakua033-public-assets` and its
receipt were removed with the historical work. Current source preparation
still referenced them. A fresh preparation using the pinned current generator
and a new input receipt is required; the old receipt will not be fabricated.
Current dist bytes matching an earlier manifest are historical comparison
evidence, rather than a fresh 0.6.0 build.

At the pushed source, hosted backend compilation, Metal, Docs, CPU preview,
monitor, Pages and minimal serving checks passed. The Linux snapshot passed
64 cases. Windows snapshot reached its 900-second subprocess limit after
52 progress marks. Broader Python checks recorded actual remaining failures:
Linux tests 4,212 passed / 18 failed, Windows tests 4,210 passed / 20 failed;
tools had three Linux and five macOS/Windows failures. Corrections and fresh
hosted replay remain in progress. The successful source merge and cleanup do
not constitute delivery acceptance. General model quality, full-window serving
and GPU concurrency remain unqualified; `releaseQualified=false`.

Q38 prototype and mathematical supplement work remains separately ordered
after 0.6.0 delivery. Cleanup did not execute the supplied prototype scripts or
modify the original Downloads attachments.
