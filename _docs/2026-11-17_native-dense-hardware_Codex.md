# Bounded native CUDA allocation and synchronization pass

Base is d10d1a0461ded703eaecb0dfa1f8e1e08648ed00 on
`codex/native-dense-own-cuda13`, in the isolated H: worktree. The task brief is
`C:/Users/downl/Documents/ChatGPT/Strata/.superpowers/sdd/2026-11-16_upstream-sync-plan_Codex/native-dense-hardware-brief.md`.
Only the owned CUDA13/SM120 runner, directly covering tests and documentation
change. No original checkpoint, app, package, primary checkout source or
published artifact was edited. Root owns independent review and actual models.

## Requirements and implementation

R1 replaces per-tensor owners with `weight_arena.hpp`: checked sizes and
256-byte alignment, a single RAII allocation, non-owning exact spans, original
compressed payloads, F32 auxiliary destination spans and unchanged host embedding.
The production admission payload is asserted equal to the planned unpadded
payload. Admission uses the padded total and one large-buffer allowance;
per-tensor allocator margins are removed only because those allocations no
longer exist. The original 16 MiB small-pool allowance remains. The bounded
16 MiB pageable reader still checks resource floors for each chunk. Partial
construction unwinds the owner. Allocation/upload reserve messages carry
stage, canonical required tensor name, free, needed and reserve bytes.

R2 changes `kv.cu`, `ops.cu`, their interfaces and `engine.cu`. Input and storage
finite checks stay enabled. Device status is sticky until reset. Encoders and
attention readers refuse failed storage; output scratch is initialized even
on failure. Host causal positions are checked before enqueue. A detected error
poisons the session until whole reset. Deferred host length advances are not
valid output after a failed fence. `LayerFence` checks STOP before each enqueue
and after no more than eight layers. At each group boundary the Engine reads
activation and KV status, synchronizes, checks unchanged resource floors and
STOP. Final logits remain validated before publication. Diagnostic
`--conservative-sync` retains intermediate KV fences and per-layer fences.
No new product UI control is introduced.

R3 accounts for initial runtime residency before selection. `startup.cuh`
queries kernel attributes, then runs tiny actual KV kernels for all six modes
and the production GDN conv/SiLU step with zero synthetic inputs. There is no
checkpoint or user prompt in warmup. Before/after VRAM snapshots are logged.
An additional conservative 128 MiB covers launch-size-dependent residency
beyond this tiny warmup. This is not a proven universal maximum. RAM 3 GiB,
commit 4 GiB, H: 10 GiB, C: 5 GiB and GPU reserve 768 MiB remain unchanged.
No GPU setting, stack limit, FTZ, fast-math or register cap is changed.

Packed matvec compile-time specialization was intentionally deferred: the
bounded pass already changes allocation and host execution, and no controlled
candidate measurement established a benefit for a third optimization. Existing
FP32 accumulation and every weight codec, including IQ2_S/IQ3_S, are unchanged.
Batched prefill, MTP, CUDA graphs and full-window work remain deferred.

## Fresh verification and measurements

All receipts below are under
`H:/Strata/evidence/upstream-sync-20261116/receipts/`. The command log beside each
JSON contains the exact argument vector's complete output. The established
`_docs/upstream-sync-20261116/check.py` wrapper used Python 3.12, an owned Windows
Job without a diagnostic memory cap and five requested build jobs. No fixture
or compiler ran concurrently with root GPU work. Every named command exited 0,
reported Job active-after 0 and zero nominal RAM floor breaches.

`q38-hardware-configure-r1` configured Visual Studio 17 2022 x64, CUDA 13.3.73,
MSVC 14.44.35207 / compiler 19.44.35229.0 and SM120. Build receipts r1, r2 and
`q38-hardware-build-final` passed. The latter minimum available RAM was
13351440384 bytes. `q38-hardware-ctest-r1` and `q38-hardware-ctest-final` passed
12/12 groups; final minimum available RAM was 15190388736 bytes. The final
`q38-hardware-cli-final` passed 50/50 CPU admission/CLI cases without loading a
model. The GPU suite includes original numerics, IQ2_S/IQ3_S, 257-position KV
reference/byte tests and pinned dialect regression (11013 checks, zero failures).

No numerical tolerance was relaxed. Existing attention tolerance is 3e-5
absolute plus 4e-5 relative; pure Spin8 inverse is 5e-7 plus 5e-7. IQ matvec
retains 0.005 absolute plus 0.0001 relative, exact decoded values. The added
hardware fixture requires bit-exact conservative/grouped outputs, all six KV
modes and all three Spin8 views at capacities 9 and 33 with changing inputs.
It covers NaN and finite encoding overflow, sticky failure, reset/exact reuse,
8-layer cancellation boundaries, exact/below admission, layout overflow,
alignment, neighboring-span sentinels, production auxiliary decode/matvec,
out-of-range views and exception cleanup. The cancellation fixture exercises
the shared production scheduling helper; full Engine/resident cancellation is
still root-owned actual-model acceptance.

The final startup snapshot decreased free VRAM from 15908995072 to 15906897920
bytes (2097152 bytes). This observation does not establish the cause of the
historical failed IQ upload or an upper bound for real-model first use.

For 129 irregular approximately 1 MiB weight spans, payload was 135332217 bytes,
arena size 135349850 and padding 17633. Actual CUDA-used deltas were 251658240
bytes for individual owners and 136314880 for the arena, a difference of
115343360 bytes. The 31-small-span and 513-small-span cases each had equal
individual/arena deltas (2097152 and 18874368 bytes respectively). Do not infer
a universal saving or stable near-budget model loading from these fixtures.

Timings use two warmups and seven bounded runs, eight positions by sixteen
attention layers; they include host scheduling gaps between CUDA events.
F16 conservative/grouped medians were 27.613920/16.223072 ms, with ranges
26.070688..31.905697 / 15.286976..19.314625 ms. Spin8 medians were
39.978046/27.883295 ms, with ranges 37.385441..71.346596 /
24.607616..28.991680 ms. Turbo4 medians were 35.435490/22.124832 ms, with ranges
33.356064..39.387745 / 20.907808..25.618624 ms. Other mode details are retained
in the final verbose receipt. Explicit host fence counts were 512 to 64
(Spin8 640 to 64). Pageable D2H operations may also synchronize internally;
these counters represent explicit fences. The GPU was reserved for this task,
but desktop graphics were not disabled or clocks clamped; variation remains.
These synthetic observations do not qualify real-model tokens per second.

`q38-hardware-resources-final` captures `cuobjdump --dump-resource-usage`.
Matvec uses 47 registers, zero stack; KV aggregate 38 registers, zero stack;
Turbo store 36 registers/592 stack bytes; inverse Turbo 40/512; GDN SiLU
128 registers, zero stack, 6208 shared bytes. These are compiler resource
observations, not a claim about spill instructions. `q38-hardware-deps-final`
records only Windows/MSVC/Universal CRT PE imports, no external inference or
CUDA runtime DLL. Existing NVIDIA header C4819 warnings remain unsuppressed.

Frozen executable is
`H:/Strata/evidence/native-dense-own-cuda13-20261117/build-hardware/Release/strata-q38.exe`,
646144 bytes, SHA-256
`9a68375e5d90c66506de31dbb0b7af09818b702d3af4a145d215d800d74a66f4`.
The prior IQ executable remains SHA-256
`f17cbd0adf91d53cc48a3cb8ffaa34cb9a0484f058bdce084b933a9c9381dc62`.
`BUILD_MANIFEST-hardware.json` in that evidence directory freezes a conservative
superset of all local source/header inputs, including the complete include tree,
Gguf split/family/geometry/checks, sampling/sampler, generated profile, CMake
options, binaries, compiler binaries, dependency/resource and test receipts.
The wider hash inventory deliberately replaces the historical narrow list.

## Controls, limitations and handoff

Canonical prompt, PC/project AGENTS, MILSPEC Codex skill, common engineering,
start gate, skill selection, security, LLMOps, CPP and Python SOPs were read.
The Python custom-wrapper exception uses the established interpreter; metadata
inspection uses the same interpreter without dependency changes. No network
inputs, credentials or original model contents were sent to another service.
Git whitespace check passed. clang-format/clang-tidy were unavailable on PATH;
no package was installed, so those checks are NOT_RUN. Scoped manual ownership
and arithmetic review and actual MSVC/CUDA builds cover the changed code.
No formal MILSPEC certification is claimed.

Self-review-event: ticket_completed. Applied agent-self-review-evolution in
review-only mode. Source review checked sole ownership/unwind, sticky status,
guarded reads, bounded cancellation, unchanged math and complete source binding.
No governance/memory files changed. A shell alias invocation of `python` failed
before running an edit; the established absolute interpreter succeeded. All
build/test attempts passed; earlier receipts remain preserved.

The compiler/GPU/formal slot was returned after all commands drained. Root must
independently review, compare the same original PQ checkpoint/prompt/Spin8 mode,
check forced logits if required, Japanese/retrieval/cancel/reset and replay
the failed IQ case plus the mixed-IQ checkpoint. No 27B model was loaded here.
Near-budget stability, 98k/131k/300s, full-window quality, actual model speed,
installation and publication remain NOT_RUN here; releaseQualified=false.
Rollback is the retained base commit and unchanged prior executable. No cleanup,
reset, force push, merge or publication was performed by this worker.
