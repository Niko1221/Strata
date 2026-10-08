# Pressure-triggered idle unloading: bounded Windows acceptance

The final idle-admission implementation passed a real Hermes tool-feedback cycle,
including full native/vision process release, capacity waiting, reload and normal
agent completion. This is a correctness and resource-release result, not a speedup.
The experimental branch includes upstream v0.1.40.4 (`6674a006`); production settings
were not changed. The earlier [background-control packet](../2026-10-08-adaptive-background)
retains its original revisions and failed attempts.

## Tested setup and identity

Windows x64; Ryzen 9 7940HS; RTX 4070 Laptop 8 GiB; 64 GiB dual-channel DDR5-5600,
about 59.72 GiB exposed to Windows; model files on a PCIe 4.0 NVMe SSD.
ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S, two GGUF shards with PLE, 65,536
configured context, high main-request reasoning, int8 KV, MTP and CPU vision.
The existing experimental projection was held constant, not evaluated as a quality gain.

The common settings are recorded in the earlier [runtime fixture](../2026-10-08-adaptive-background/runtime-settings.json):
only its common model/native settings apply here, not its historical experiment overrides.
32 GiB resident budget, expert-cache argument 264, seven pool workers, fixed PCIe
fraction 0.37, 448 MiB startup fitting reserve and 320 MiB running headroom target.
The hard test guards were 3 GiB global available RAM and 250 MiB fresh native/DXGI
headroom. Startup required 42 GiB available RAM. No clocks, power plan or drivers changed.

- Native SHA-256: `af83775b5fc32cae7595e16a749317590a4a75b0e795dc17282c02f7aaac9b37`.
- Final server SHA-256: `a97599d38ed421576cafab1b8a5757384ab9fa5fdd05321c6ce50b77600da1a5`.
- [results.json](results.json) records exact measured source-byte hashes and selected results.

## Final-revision results

| Check | Result |
|---|---|
| Complete Python suite | 893 run: 886 passed, 7 skipped |
| Targeted native tests on the rebuilt v0.1.40.4 native code | Five passed: live memory, background control, request STOP, background pool and platform memory |
| Real idle pressure | Automatic unload under a separate touched 3 GiB allocation; original native **and vision** identities exited |
| Streaming wait and disconnect | Actual capacity heartbeat; preparation released about 0.514 s after disconnect, without loading |
| Admission hold | Another request remained blocked for 5.817 s while the allocation was held |
| Recovery | After release and recovery dwell, both native and vision had new process identities; arithmetic probe returned exactly 42 |
| Hermes feedback | Same session completed normally after two main model calls; final answer corroborated against the real tool result and a fresh tool-only nonce |
| Real C++ build | Four concurrent jobs, 2,096 passing cases; 104 additional independent checks passed |
| Agent wall time | 190.442 s, including the controlled tool wait, extra admission probe and reload |
| Minimum sampled headroom in that cycle | 5.623 GiB available RAM; 294 MiB native/DXGI VRAM |
| HTTP compatibility with idle policy enabled | Real red-square image answered correctly in 12.498 s; actual function call/result returned 42 in 13.674 s |
| Minimum sampled headroom in HTTP check | 9.186 GiB RAM; 342 MiB native/DXGI VRAM |
| Cleanup | No recorded owned model/helper process remained after the Hermes campaign |

The build itself took **2.296 s and finished before unloading**. A separate real
RAM holder supplied pressure, and the compiler tool's return was deliberately
delayed. This exercises continuity between model calls during a controlled tool
wait; it does **not** show faster compilation or a compiler requiring 3 GiB.
The C++ implementation was already correct; this test checks orchestration and
feedback, not the model's ability to invent or repair that algorithm. Its compiler
used `-march=znver4`; that flag alone does not prove every instruction used AVX-512.

The trigger was fixed once before allocation at 6.398 GiB available RAM, using
measured current/reload footprints to create a safe test window. Pressure dwell
was 6 s and recovery dwell 5 s. Capacity readings, allocation and process release
were real; thresholds were elevated to avoid exhausting the host. The three extra
API records include the admission probes and an auxiliary request; they are not
three more completed Hermes turns. Full unloading discards the in-memory prompt
cache and requires rereading supplied history. No matched no-unload agent arm was run.

The first full suite exposed four old process fakes whose `wait`/`poll` behavior
did not represent confirmed death. Those fakes were corrected without weakening
runtime teardown. Review then caught image-download/FIFO admission and heartbeat
races, plus missing guards after active-only cancellation and timed/manual unload.
The final suite covers these fixes. Earlier logs and successful earlier runs were
retained rather than silently replaced; only the hashes above identify this final run.

## Additional feasibility work on the earlier server revision

Using the same v0.1.40.4 native binary and earlier server hash
`c465e86d53e0bc082a96f7b079f6c5ee86fae1bed5531f3f5ddb0fd2c3601ec8`, the baseline,
active parking/resume and parked cancellation regressions all passed. Separate
static 32/20 GiB profiles passed matched arithmetic, vision and tool-call checks:

| Resident cap | Peak process-tree RSS | Peak private commit | Minimum available RAM | Minimum sampled native/DXGI VRAM |
|---|---:|---:|---:|---:|
| 32 GiB | 39.184 GiB | 46.599 GiB | 8.907 GiB | 324 MiB |
| 20 GiB | 27.033 GiB | 34.368 GiB | 21.070 GiB | 324 MiB |

The smaller profile was slower in these short single trials. This establishes
feasibility of freeing roughly 12 GiB, not an automatic reload policy or causal
speed estimate. These probes configure 64K but do not use near-64K input. Qualified
long-context envelopes, strict model/resource identities, recovery and failure
coverage are required before automatic profile selection. The current controller
still admits the full measured startup profile.

The separate footprint observer missed native fields because it checked the outer
Windows virtual-environment Python launcher for the listening socket, while the
server ran in its child. Those fields remain missing. The VRAM minima above come
from the runner's separate fresh native/DXGI samples, corroborated with owned
process-tree samples. The historical listener PID was not recorded. Global memory
and process RSS/private-commit observations are separate, valid measurements.

## Reproduction and scope

Run the server tests with `python -m unittest discover -s serve -p "test_*.py"`.
Configuration and lifecycle semantics are in [ADAPTIVE_STRATA.md](../../../docs/ADAPTIVE_STRATA.md).
For a real pressure trial, first measure a safe load, choose fixed thresholds,
bound the external allocation, record both process identities and capacity ages,
and independently validate tool output after reload. Never induce physical
exhaustion to reproduce this test. This packet supplies selected measurements and
an acceptance protocol; it is not a portable one-click model/test bundle.

Policies are opt-in. Sampled minima and past working-set peaks cannot guarantee
that a racing allocation, device failure or client timeout will never interrupt
work. CPU scheduling, multi-GPU placement, SSD selection and automatic smaller
reload profiles remain outside this result. Source attribution is in
[ADAPTIVE_PROVENANCE.md](../../../docs/ADAPTIVE_PROVENANCE.md), including PR #726's
allocator foundation and PR #1093's specific stale-counter reporting idea.
