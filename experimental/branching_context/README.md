# Immutable branching contexts — draft

Continue two different futures from the same consumed prefix while keeping the
prefix and every saved future immutable. Rewind a branch reference without
deleting history, and rebuild from that history when numerical checkpoints have
been evicted.

This is a local Python API and an experimental native CUDA backend. Its public
boundary is a consumed token: selecting or streaming a token alone does not
commit a snapshot. The mathematical contract is in
[spec/branching-context-api.md](spec/branching-context-api.md).

This draft provides a foundation for
[#1938: retaining useful conversation prefixes](https://github.com/Niko1221/Strata/issues/1938)
and [#1962: sharing identical prefix blocks](https://github.com/Niko1221/Strata/issues/1962).
The latest [corrected-driver validation](evidence/dispatch-correction-validation.json)
records eight supporting test methods and 34 exact full-vocabulary comparisons
on Q8_0 weights with FP16 KV. G0 passed 51/51 and all four reference mutants
were detected. Full G0-G5 certification remains incomplete.

Both private drivers previously omitted resident expert contributions. Their
old agreement therefore did not establish ordinary-model correctness. The
corrected drivers bind the actual expert dispatch plan; four one-token checks
now match ordinary execution exactly, and 30 branching comparisons match a
fresh full-history oracle. Earlier numerical evidence below is historical and
must be read with that limitation.

The prototype now has owner-affine execution leases, a FIFO slot queue, and
native host-RAM snapshots with exact-history reuse. Execution is serialized on
one backend workspace. Conversation-server integration, concurrent batch
execution and image-input compatibility remain open. Neither issue is closed.

Batched-prefill equivalence remains unresolved; help comparing ordinary
prefill and token-level execution is welcome. The preserved
[diagnostic evidence](evidence/normal-prefill-validation.json) records the
mismatch without relaxing tolerances. The current numerical profile is still
unqualified for gate admission.

## Implementation

- `src/context_time_travel/store.py` owns the single canonical parent-linked
  history and controller records. Named branches are separate revisioned
  references with CAS and retry receipts.
- `real.py` materializes optional checkpoints behind leases and publishes
  consumed-token frontiers. RNG draw state, grammar, penalties, stopping and
  decoding state belong to the canonical controller, rather than being assumed
  to exist in a native state export.
- `strata_backend.py` adapts the private `strata-branch` pipe. CUDA VMM pages
  share retained allocations and copy writable tails. Recurrent state and
  frontiers have explicit ownership; activation can still copy those buffers.
- `strata-fresh` starts an empty process/session for each full token history.
  The current oracle is separately linked and accepts no branch checkpoint or
  retained handle; its source and binary hashes are in the corrected evidence.
- Native page hashes, ownership, allocation/release, forward-work and copy
  counters are read directly. Image handles are removed from a live-entry list
  when released, and page integrity is checked before activation.

Positions and RoPE/YaRN settings stay bound to the model contract. Sharing a
prefix does not permit moving its keys to a different logical position or
changing its positional semantics.

This upstream branch is based on `main` revision `7695859a`. It brings across
the native branching patch and local API from the tested development branch,
without its earlier Responses/server changes. The original tested native
revision is `d702b68d`; its measurements below are historical evidence and do
not automatically certify this port. The Python API has no HTTP service,
database server or GUI dependency and adds no serving route. It currently
requires POSIX `fcntl` locks and does not import on native Windows Python.

## Validation on upstream main

The port at `ed5c4557` built `strata`, `strata-branch`, `strata-fresh` and eleven
native support targets on Linux with CUDA 13.3, GCC 15.2 and an RTX 4090. G0
passed 51/51, the demo passed, all four reference mutants were rejected, and six
calibration-input tests passed. Nine native support executables and both digest
tests passed, including 14728 CUDA lifecycle checks and 434 session checks
covering all four KV layouts with pattern data. The first build exposed two
missing no-MTP snapshot overloads; the port adds those overloads and retains
the existing reference-taking callers.

The unchanged registered G3-004 method also passed against a separately linked
fresh-process oracle using Q8_0 weights and FP16 KV: eight full-vocabulary
comparisons were exact, 28 retained-byte audits passed, and each of six hot
appends computed one token with zero prefix recomputation. Each initial child
shared 24 physical allocations with the root. The run measured 216 COW pages,
452984832 copied bytes and a sampled tracked-VMM peak of 704643072 bytes. All
eight image/handle owners were released and the native process closed. The
case took 74.80 seconds on the shared host; this is not a throughput benchmark.

[The port evidence](evidence/upstream-validation.json) records the executed
test IDs, direct counters, source/model/build fingerprints and original-report
hashes. Its machine profile remains unqualified for numerical gate admission;
G3-004 compares exact float tuples without applying a tolerance. One registered
case is not a cumulative G3 PASS. HIP/SYCL builds and default serving-path parity
were not verified, and full G1–G5 certification remains incomplete.

## Original executed validation

The tested native revision is `d702b68d`, using Qwen3.8-Flash-Next Q8_0 weights,
FP16 KV and the pinned `Verifier.run(T=1)` operator on Linux/CUDA.

| Check | Observed result |
| --- | --- |
| G0 logical/reference suite | 51 passed; no failures or skips |
| Model-free demo and mutation smoke | Demo passed; four mutants rejected |
| Original-build G3-004 | One unchanged registered method passed |
| Forced A/B futures | Eight fresh full-vocabulary comparisons; maximum absolute logit error 0.0 |
| Immutable retained state | Seven images; 28 physical-byte audits passed |
| Hot append work | Six appends; one new forward token each; zero prefix recomputation |
| Actual ownership | Each initial child shared 24 physical allocations with the root |
| Resource scope | 216 COW pages / 452984832 copied bytes; sampled VMM peak 704643072 bytes |
| Release | Eight image/handle owners released; tracked payload/handle bytes returned to zero; native process closed |

The short case consumed 24 SUT tokens and 84 independent-oracle tokens. The
initial run took 88.0105 seconds; the published-source rerun passed the same
case in 70.1252 seconds. These are shared-host observations, not throughput benchmarks.
VMM counters exclude model arenas and temporary workspaces. Final native
counters precede native teardown; process closure is checked separately.

Earlier evidence on native `2622c3a5` passed all 27 registered G1 methods with
136 exact comparisons. That earlier result does not certify the combined current
source. Native lifecycle support at `d702b68d` passed 14728 CUDA checks across
four KV layouts using pattern data, plus 24605 host handle-registry checks and
sanitizer support. Pattern-data checks are separate from model equivalence.

All 16 registered G3 method bodies are included; the short draft run executes
only G3-004. Exhaustive page offsets, the remaining current-build model cases,
cumulative G1/G2/G3 and the FP16/INT8 execution matrix remain incomplete. INT8
has earlier isolated support but no completed independent baseline. No Windows
backend, HIP backend, power-loss, full-capacity stress or production guarantee
is asserted here.

Normal batched-prefill comparisons previously reached 1.5612177848815918 maximum
absolute logit error and were rejected. The measured draft uses the separately
pinned token-level operator; that result is not a tolerance for batched prefill.

## Run the reference checks

From this directory on Linux with Python 3:

```sh
python3 scripts/verify.py --gate G0
python3 examples/branching_demo.py
python3 scripts/mutation_smoke.py
```

These commands use the standard library and do not load a model. The acceptance
registry and unchanged required assertions are in `docs/acceptance.json` and
`docs/test-matrix.md`; missing tests, skips and expected failures cannot qualify
a gate.

## Run a model case

With an existing configured CUDA build, the native targets are `strata-branch`
and `strata-fresh`. Keep the independently built oracle separate from the SUT.
The native `branch_*` test targets provide the physical ownership and integrity
support checks; device-pressure programs are deliberately excluded from ordinary
parallel CTest runs.

Set `CTT_PROFILE` to a locally measured Strata execution profile that binds model
file hashes, executable hashes, hardware, execution arguments and the oracle.
Profiles are intentionally specific to a machine and build. The published
measurement summary is not an executable profile. Existing measured admission,
model-hash and calibration receipts can be checked with
`scripts/build_strata_profile.py`; profile provisioning is still a development
workflow.

```sh
CTT_PROFILE=/absolute/path/to/measured-profile.json \
  python3 scripts/run_ticket.py draft-branch-check \
  --gate G3 --pattern test_immutability.py
```

That unchanged case forces two different tokens through three consumed
generation boundaries per branch, compares each frontier to a completely fresh
full-history forward, audits every retained parent/sibling, and checks direct
forward counters. A one-case PASS is not a cumulative G3 PASS.

The source package preserves the tested API, registered tests, reference
equations and numerical assertions byte-for-byte. Host-specific orchestration
is excluded. One calibration helper now requires its existing `--native-root`
argument instead of supplying a private machine path; its mathematical rules
are unchanged. [The source manifest](evidence/source-manifest.json) records that
packaging boundary. Public evidence omits private paths and host identifiers;
the original artifact hashes and measurement scopes are retained.
