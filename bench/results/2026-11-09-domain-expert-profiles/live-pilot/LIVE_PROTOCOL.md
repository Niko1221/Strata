# Prospective live profile comparison

Author: Francisco Angulo de Lafuente. Target: a small coding pilot using the profile bank already built from the 37-request training corpus. The code domain was selected because it had a positive held-out signal in that exploratory study. This is a targeted pilot, not a claim about all domains.

| Item | Frozen value |
| --- | --- |
| Sequence | A1 global, B code-domain, A2 global |
| Engine and weights | Existing release engine0.1.59 and the pinned IQ3_XXS model used by the trace collection |
| Profiles | profiles-37-live1587/global_profile and domains/code, copied with SHA-256; matched layer quotas at1,587 slots |
| Capacity | Same --expert-cache1200 native-byte budget; CPU/source prediction is exactly1,587 slots in every leg,2,794,931,200bytes; reject a mismatch before the three measured requests |
| Shared settings | Same baseline args; prompt cache0, PCIe fraction0, route prior0, ADAPT_NOWAIT0, ADAPT_LAG1; same settings for all legs |
| Workload | Three new deterministic coding requests: clamp, half-open interval overlap, positive-amount SQL aggregation |
| Generation | Greedy, thinking off, max96 new tokens per request; same order in every leg |
| Warmup | One identical two-token OK request; raw warmup and its cost retained |
| Outputs | Keep full raw responses, timings and logs; run fixed Python/SQLite correctness checks |
| Comparison gate | Every quality check passes; output text and token count identical across all three legs; A1/A2 total request time differs by at most10% |
| Improvement gate | B total request wall time at least5% lower than each A leg; report per-request data, startup separately and all failures |
| Scope | Existing profile loading at engine start, not hot switching or a new engine feature. No CUDA/HIP/SYCL source change |
| Budget | One managed servidor-ia job, timeout25min,1GPU14GiB estimatedVRAM14GiB RAM; no local GPU and no Kaggle |

The runtime report must include elapsed startup, warmup, each request's wall time and engine timing, profile/bin hashes and exact cache size. Greedy output parity is a gate for interpreting the speed comparison, not a general proof of equal quality. A1/B/A2 is a single interleaved pilot; neither a production-default recommendation nor a statistical population estimate follows from it.

The reference quality checks use independent cases, including interval touch/containment/nonoverlap and negative/zero SQL values. They test actual generated answers. No answer tokens are forced. The prompt texts, code, profiles and protocol are hashed before execution; later changes require a new annotated protocol.

Before any live result: source and native-expert metadata showed that the native pack turns the nominal cache count into a byte budget. The shipped profile fits1,582 slots at1200 nominal blobs, but the learned global profile fits1,587. The live bank is rebuilt at1,587 and keeps its per-layer capacities for the domain control. The capacity plan records the exact byte calculation and its calibration against the original engine logs. SQL checks also include a category containing only zero amounts. An inner24min signal leaves cleanup time before the managed25min limit, and cleanup targets the created process group even if its parent exited.
