# Adaptive feature history repair: source qualification

The repaired feature history retains `8689b1eefedcd5eadea1c819bd4554c88c7e238f`,
including the real allocator contribution and learned-profile identity fix from
medking82. It removes the accidental upstream-first release merge
`612c5514fecbc86aaafbe3135d400f87767cf168` from this feature branch's ancestry.
The resulting feature history remains based on v0.1.42. The separately tested
v0.1.43 runtime and its integration resolutions are preserved separately.

At inspection, current upstream already contained v0.1.43. Both old and repaired
three-dot feature comparisons therefore contained 113 changed files before
the documentation-only update. Removing the merge does not establish a smaller
current feature diff or current-main merge compatibility.

The only changes made above the retained feature revision in this repair are
documentation and removal of machine inventory from published result metadata.
Runtime, build and test source is unchanged from that retained revision. Existing
authors, implementation provenance, licenses and measured outcomes are retained.

Fresh local validation on Windows/CUDA, 13 October 2026:

| Check | Result | Boundary |
|---|---|---|
| Server unit/regression suite | 1,141 run, 9 skipped, no failures | Mock-engine server cases; not a model workload. Five skips need the pack tokenizer; two are Windows wrapper timeout fixtures, one POSIX permissions and one symlink privileges. |
| Resource-feedback broker suite | 54 passed | Contract/state-machine tests. |
| Native host fixtures | Seven passed | Conversation cache/memory/file, session-save reclaim, cache, serving window and coupled draft. |
| CUDA Release build | Passed | Native executable and the selected tests built for sm89. |
| Selected CUDA-build fixtures | Six passed | Live memory, prefill policy, background control, request stop, worker-pool background and real prefill relayout. Includes actual GPU VMM release/regrowth and allocator checks; not a full model run. |

The built native executable's SHA-256 is
`fa9eb4135edd3c529032f6b644ef550636f74d627ada5dc6efd1ad23d0d5eb64`.
Historical CI and media results retain their own recorded revisions. This repair
does not reclassify a failed historical workflow or qualify the additional local
checkpoint-bank/media composition as part of the public branch. HIP and SYCL
were not built; the feature remains a draft.
