# Adaptive Strata: implementation provenance

This document identifies sources actually used by this branch, as audited on
2026-10-08. It distinguishes inherited or adapted code, design inspiration, and
API/ABI references. A citation does not transfer a source's performance claims
to this implementation. Validation must use this branch's own test evidence.

## Code inherited and adapted

| Source | Concrete use in this branch | Attribution and license |
| --- | --- | --- |
| [medking82's Strata PR #726](https://github.com/Niko1221/Strata/pull/726), source snapshot [`15a59d4785f492b4df3fb78862373a5383696450`](https://github.com/medking82/Strata/commit/15a59d4785f492b4df3fb78862373a5383696450) | **Adapted code:** live RAM blocks, GPU VMM expert-cache resize, `MEMORY` protocol and acknowledgements, guarded prefill loans, memory policy, resource presets, their UI and tests. This source snapshot was ported onto the v0.1.40.3 base identified below; subsequent commits modify that port. The public source snapshot is the attribution reference, independently of the port's commit identity. | Original author **medking82** is retained on the port commit. This is substantial reused Strata code, not a newly invented allocator. Retain the repository's [MIT license and notices](../LICENSE). PR closure without merge is not represented as technical rejection. |
| [Strata v0.1.40.3 base, `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`](https://github.com/Niko1221/Strata/commit/d5ea7133741e67743c0e886bb426c0ce8d69cf6c) | **Existing project code extended:** the native generation loop, prefill execution, expert source/cache, asynchronous readers, worker pool, server FIFO, output parser, detokenizer, lifecycle and sampling. | Credit Niko1221 and Strata contributors; retain the existing MIT license. Fixes to this port's integration are not attributed to unrelated research. |

Specific existing mechanisms reused:

- [`ExpertPool`](https://github.com/Niko1221/Strata/blob/d5ea7133741e67743c0e886bb426c0ce8d69cf6c/src/kernels/cpu/pool.cpp):
  the existing epoch/condition-variable sleep path is extended with a background
  wait control. This contribution does not import a thread pool from another engine.
- [`Prefill`](https://github.com/Niko1221/Strata/blob/d5ea7133741e67743c0e886bb426c0ce8d69cf6c/src/prefill/prefill.cpp)
  and [`ExpertSource`](https://github.com/Niko1221/Strata/blob/d5ea7133741e67743c0e886bb426c0ce8d69cf6c/include/strata/core/expert_source.hpp):
  cooperative prompt returns preserve the current offset, drain existing readers,
  and return the PR #726 loan before a resize. Existing asynchronous read/future
  ownership remains authoritative; no external futures framework is introduced.
- [`platform/memory.cpp`](https://github.com/Niko1221/Strata/blob/d5ea7133741e67743c0e886bb426c0ce8d69cf6c/src/platform/memory.cpp):
  the existing non-local DXGI budget helper and CUDA-adapter LUID matching are
  refactored to support a local-segment query. The adapter enumeration is inherited.
- [`serve/server.py`](https://github.com/Niko1221/Strata/blob/d5ea7133741e67743c0e886bb426c0ce8d69cf6c/serve/server.py)
  and [`serve/frontend.py`](https://github.com/Niko1221/Strata/blob/d5ea7133741e67743c0e886bb426c0ce8d69cf6c/serve/frontend.py):
  active-request parking retains the existing FIFO owner, incremental parser,
  streamed tool-call identity, UTF-8 detokenizer, and cancellation contract. It uses
  exact-prefix re-prefill; it does not import another project's checkpoint format.

## Design principles actually used

| Source | Specific influence | What was not reused |
| --- | --- | --- |
| Liu, Ye, Li and Li, [ATSInfer, *Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices*, arXiv:2607.10183v2, sections 4.3–4.4](https://arxiv.org/html/2607.10183v2) | **Design inspiration:** `serve/routing_costs.py` compares measured CPU/GPU choices and exposed transfer/completion cost under changing load. The citation is also beside that implementation. | No ATSInfer source, tensor-placement algorithm, learned estimator, benchmark or claimed speedup is incorporated. This branch only selects among already-supported Strata request-level routing choices using qualified matched samples. |
| [StarPU performance models and data-aware task scheduling](https://starpu.gitlabpages.inria.fr/features.html) | **Design inspiration:** the same optional routing-cost gate considers expected completion cost and data movement rather than utilization alone. | No StarPU runtime, scheduler source, task graph, out-of-core subsystem or dependency was imported. Its source license is not being used to license this original routing gate. |

These are the two research/system principles explicitly used in the implemented
routing gate. The fresh-sample requirements, thresholds, hysteresis, lease bounds,
pressure admission and bounded retry policy are original choices in this branch;
they are not presented as implementations of ATSInfer or StarPU.

## API and ABI references used by original integration code

| Source | Implemented use | Reuse classification |
| --- | --- | --- |
| Microsoft [`QueryVideoMemoryInfo`](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo) and [`DXGI_QUERY_VIDEO_MEMORY_INFO`](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info) | `src/platform/memory.cpp` and native `CAPACITY` report local process budget and usage. Admission uses the smaller of CUDA free bytes and nonnegative local budget headroom; a successful zero budget is pressure, not a missing reading. | **API contract reference**, extending existing Strata code. No Microsoft sample implementation was copied. Budget-change event registration was researched but is not implemented. |
| NVIDIA [CUDA runtime memory-management API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html) | Existing `cudaMemGetInfo` supplies the allocator view at safe native boundaries; the result is not a reservation or a guarantee that a later allocation succeeds. | **API contract reference**. No new NVIDIA sample, DMA engine, or Unified Memory implementation was imported. |
| Microsoft [`GlobalMemoryStatusEx`](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex) and [`MEMORYSTATUSEX`](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex) | `serve/telemetry.py`, `memory_policy.py` and parking admission treat available physical RAM and commit as distinct constraints. `ullAvailPageFile` is the calling process's available commit bound; it is not free disk capacity. | **API/structure reference**. No sample program was copied. Volatile readings cannot promise immunity to an external allocation race. |
| Microsoft [`NtQuerySystemInformation`](https://learn.microsoft.com/en-us/windows/win32/api/winternl/nf-winternl-ntquerysysteminformation) and psutil's [`ntextapi.h`, pinned at `2408579876aeb8b42b15d1de3934fd072828c495`](https://github.com/giampaolo/psutil/blob/2408579876aeb8b42b15d1de3934fd072828c495/psutil/arch/windows/ntextapi.h) | `serve/windows_process_snapshot.py` makes one bounded Windows x64 process-counter snapshot. Field offsets were cross-checked against the NT ABI; pointers and record lengths are bounds-checked before reading. | **ABI facts/reference only**. No psutil declarations or implementation were copied. psutil is an existing runtime dependency with its own [BSD-3-Clause license](https://github.com/giampaolo/psutil/blob/2408579876aeb8b42b15d1de3934fd072828c495/LICENSE); it was not vendored by this change. Unsupported/malformed snapshots fall back conservatively. |
| Microsoft [`CreateDirectoryW`](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createdirectoryw), [SDDL format](https://learn.microsoft.com/en-us/windows/win32/secauthz/security-descriptor-string-format), and [`GetTokenInformation`](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-gettokeninformation) | `serve/request_parking.py` creates a private directory with an explicit protected Windows DACL for the current user and SYSTEM, instead of assuming POSIX permission bits protect Windows files. | **API references** for original ctypes code. No security sample or third-party ACL implementation was copied. |
| Microsoft [`GetVolumePathNameW`](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getvolumepathnamew) and [`GetVolumeInformationW`](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getvolumeinformationw) | The Windows journal verifies `FILE_PERSISTENT_ACLS` before creating its private request directory; unsupported filesystems are rejected. | **API references**, not copied source. This closes the documented case where a filesystem can ignore a requested security descriptor. |
| Python [`os.fsync`](https://docs.python.org/3/library/os.html#os.fsync) and [`os.replace`](https://docs.python.org/3/library/os.html#os.replace) | The request journal writes an exclusively created temporary file, flushes it, then replaces the owned journal. Serialization and integrity use standard `json`, `secrets` and `hashlib` APIs. | **Standard-library API use** by original journal code, not a copied journaling package. This does not claim survival of every filesystem/controller/power failure. |

## Original changes and attribution boundaries

The short-lived `BACKGROUND` lease, request STOP epoch helper, cooperative prefill
pause integration, current acknowledgement reconciliation, lifecycle worker,
exact-output journal/admission policy and their new regression tests are original
extensions of the inherited Strata implementation. Internal correctness fixes
such as retaining the last valid context during a pre-READY retry do not derive
from a research paper and are not cited as such.

`ADAPTIVE_STRATA.md` also contains a broader research survey. Senpai/TMO, DAMON,
XSched, SGLang HiCache, vLLM KV offloading, FlexLLMGen and the other listed projects
were evaluated as related work; their code or algorithms are not incorporated
here. The same distinction applies to overlapping Strata PRs #1117, #1324, #1461,
#1471 and #1480: checking their scope is compatibility review, not implementation
reuse. They should not appear as borrowed implementation credits unless a later
change actually uses them.

For a pull request, retain medking82's author credit and the MIT notices, link this
document, and report only measured results for the submitted revision. Keep
design inspiration separate from code attribution and from local validation.
