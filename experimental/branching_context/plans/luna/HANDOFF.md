# Immutable branching contexts: Luna execution handoff

## Start here

This is an implementation handoff with bounded draft validation. Work through
`queue.json` one small ticket at a time. The 85 registered real-backend cases
remain authoritative. A ticket state is never a certification gate result.
The final `CERT-LONG` ticket is disabled; do not enable or launch it.

Repository: `C:\Users\dflanag3\Documents\fleet\strata-immutable-branching-upstream`.
Local branch: `feat/branching-completion`, created from published commit
`6dd5f9df7e1aa7f00045cf72e997fd8595175847`. Its published predecessor remains
`feat/immutable-branching-upstream`. Inspect Git before acting; do not reset
changes, force-push, publish, merge, submit a PR, or alter another worktree.

Read, in order:

1. The fleet parent `AGENTS.md`, repository `AGENTS.md`, and package `AGENTS.md`.
2. All of `spec/branching-context-api.md`, then all package `docs/` and
   `plans/tickets.md`. The mathematical contract wins.
3. Package `README.md`, `evidence/upstream-validation.json`, `evidence/STATUS.md`
   and `evidence/DECISIONS.md`, then this directory's queue and runner.
4. The native conversation-cache/session/prefill sources named by the next
   ticket. Read upstream `docs/DETAILS.md` for integration and the relevant
   backend build instructions before touching shared headers or implementations.

Use `python` on Windows. Do not invoke the Microsoft Store `python3` alias.
Remote interpreters must be independently inspected. Favor simple queues,
lists, explicit ownership and state machines. No Docker, database server, GUI
or web service. Validate native conversation/request paths with local drivers.

## Current evidence and limits

The upstream port's last numerical evidence is for native commit
`ed5c45574877ccb17f791ccf394bc3103717ee6f`, tree
`61e6d4fed182b288f6c4eb6c775987504df262da`. The later publication commit changed
documentation/evidence. Verify source and binary fingerprints again; old PASS
profiles do not certify subsequent changes.

Historical current-port observations, not new harness measurements:

| Check | Executed evidence |
| --- | --- |
| G0 | Linux 51/51, demo passed, four reference mutants rejected |
| CUDA support | Nine native executables and two digest tests passed |
| Registered G3-004 | One case, eight exact full-vocabulary comparisons, max error 0.0, 28 byte audits |
| Hot appends | Six appends, one computed token each, zero prefix recomputation |
| Ownership/copies | 24 root allocations shared per initial child; 216 COW pages / 452984832 bytes |
| Sampled tracked VMM peak | 704643072 bytes, excluding model arenas and other memory |

There are 62 existing registered methods: G1 27, G2 19, G3 16. G4's 11 and G5's
12 are absent at handoff creation. This is code inventory, not 62 current passes.
The current-port profile is **not numerically qualified**. G3-004 asserts exact
tuples without a tolerance; it cannot establish a baseline for other profiles.
Normal batched-prefill previously reached 1.5612177848815918 absolute logit
error. Keep that failure; do not reinterpret it as a percentage or widen limits.

The new harness measurements are in `evidence/harness-validation.json`. The
latest short case also reports 24 shared root allocations after each initial
child append, but none of those allocations reports filled-prefix bytes.
Do not count unused allocated capacity as populated prefix payload. Prove
populated full-prefix sharing and bounded tail copies with longer real fixtures
and direct ownership/copy counters; the short case only establishes its recorded
immutability, numerical and work observations.

The store currently imports POSIX `fcntl` and fails to import on native Windows.
Host-RAM prefix deduplication, ordinary cache/batch integration, complete G4,
G5 product hardening, Windows support, HIP/SYCL builds, default-path parity and
overall latency/RAM-benefit qualification remain outstanding.

## Backend and owned remote workspace

Initial model profile: Qwen3.8-Flash-Next **Q8_0 weights / FP16 KV** on Linux CUDA.
Use Q4 only with separately measured evidence; do not use Q2 weights.
Positions, visibility, model/input identity and RoPE/YaRN are part of the
semantic contract. No rephasing or changed positional contract is assumed.

The previously used host is SSH alias `llm-49`, hostname `r4090`, RTX 4090.
It was rechecked during harness creation: Python 3.14.4 at `/usr/bin/python3`,
driver 595.99.02, 24564 MiB VRAM. This does not reserve the machine or prove it
will remain idle. Inspect current processes and utilization before each run.
Do not automatically use llm-60, r730, or another machine.

Existing artifacts to inspect, not overwrite:

```text
/home/dflanag3/branching-g3/upstream-draft-20261012/
  source/                       tested native source
  build/strata-branch            SUT executable
  build/strata-fresh             independent fresh-process oracle
  native-source.json             native source hashes
  results-v2/execution-profile.json
  results-v2/{measurements,native,oracle}.jsonl
  results-v2/test-report.json
```

Use a new owned scratch directory for changed source and new evidence. Transfer
an exact source manifest and changes, verify them before execution, and run the
harness on the execution host. Never describe a Windows command wrapping SSH
as a locally measured GPU run. Keep host-specific bindings and raw private
paths out of public evidence. Do not rerun the old provisioning scripts: they
write existing validation directories and assume specific old source commits.

Verify every model file against its actual hash or a still-valid executed hash
receipt with unchanged file identity/size/timestamps, and record which was used.
Fingerprint the actual SUT and oracle binaries. Inspect oracle symbols/source
and execution arguments: it must start empty for each full exact history and
must not read SUT checkpoints. Checkpoint export does not provide the controller
by assumption: the canonical controller owns RNG draws, grammar, penalties,
stops, output bytes and generation scope.

## Drive one ticket

Run these commands from `experimental/branching_context`:

```powershell
python plans/luna/drive.py status
python plans/luna/drive.py next
python plans/luna/drive.py run H00 harness
python plans/luna/drive.py record H00 VALIDATED --note "Executed harness checks; see ledger"
python plans/luna/drive.py run B00 inventory
python plans/luna/drive.py record B00 VALIDATED --note "Static inventory only; no model tests executed"
```

Check current status first: a command with current successful evidence must not
be duplicated. The next runnable ticket may already be B01 or B02.
The initial baseline was executed on the owned Linux scratch copy recorded in
`evidence/runs/luna/remote/location.json`; its next ticket is C01. The Windows
ledger only has H00/B00 because the remaining baseline needs Linux. Consult the
Linux ledger before rerunning those steps.

For an implementation ticket, write its named adversarial unittest first.
Run `red`, inspect the actual assertion failure, implement the smallest fix,
then record `IMPLEMENTED` with a factual note. Run its `green`, `registered`
(where present), `G0` and `mutants` commands. Record `VALIDATED` only after
all required current evidence exists. New tests go in the paths in the queue;
do not rewrite existing registered assertions, reference equations or tolerances.

```text
python plans/luna/drive.py run C01 red
python plans/luna/drive.py record C01 IMPLEMENTED --note "Describe the actual fix"
python plans/luna/drive.py --bindings /absolute/private/bindings.json run C01 green
python plans/luna/drive.py run C01 G0
python plans/luna/drive.py run C01 mutants
python plans/luna/drive.py --bindings /absolute/private/bindings.json record C01 VALIDATED --note "Describe executed evidence"
```

Use the same binding set for all evidence and transitions for a ticket, including
red and regressions when the ticket uses model artifacts. Binding changes
invalidate validation. A private bindings file has this shape:

```json
{
  "artifacts": {
    "profile": "/absolute/path/to/executed-profile.json",
    "model_manifest": "/absolute/path/to/verified-model-manifest.json",
    "native_source": "/absolute/path/to/native-source.json",
    "sut": "/absolute/path/to/strata-branch",
    "oracle": "/absolute/path/to/strata-fresh"
  }
}
```

Add native support executables and build configurations to `artifacts` whenever
used. Hash bindings are receipts, not model-file verification or numerical
admission. The runner does not invent numerical/work/memory measurements;
attach the actual test/engine measurement artifacts and review them.

Ticket dependencies mean implementation readiness. Previously validated
tickets become stale as sources/builds change; that is visible in `status`.
Historical implementation readiness can permit the next ticket to start, but
final regression checks must run on the resulting build. A `BLOCKED` dependency
cannot be bypassed. One unfinished active ticket is permitted.

## Resume, budgets and blockers

The append-only hash-chained ledger is `evidence/runs/luna/events.jsonl`.
Each run keeps stdout, stderr, a test report and before/after fingerprints.
`check` verifies artifacts and kernel process identity; a live child prevents
duplicate execution. A confirmed vanished child is recorded as interrupted,
not PASS. An uncertain launch or inaccessible process remains unresolved;
inspect the original host/process before repairing anything. Never delete the
ledger or infer termination merely from a timeout observing SSH.

Windows subprocess trees use an owned job object; Linux subprocesses use an
owned process group. Timeouts stop only that tree. Kernel identity observation
is supported on Windows and Linux, not claimed for other operating systems.
An abrupt Linux driver death can leave its child live; poll the recorded child
and do not launch a replacement. The runner does not automatically kill it.

Defaults are 1800 seconds per command and 7200 validation seconds per ticket.
Preserve failures and partial evidence. Do not restart an unchanged timed-out
or interrupted job. Failed unchanged commands require `--retry-note` with a
reason. New independent work may proceed after an active ticket is explicitly
blocked with an existing evidence artifact:

```text
python plans/luna/drive.py record C01 BLOCKED --note "Exact missing primitive or failed invariant" --evidence /absolute/path/to/run.json
```

The implementation queue may be refined into smaller tickets when source
inspection justifies it. Keep the same obligations, original protected-file
hashes and disabled certification ticket. A command edit invalidates prior
evidence; explain the change in DECISIONS. Do not extend budgets to recreate
the long campaign.

## Acceptance and final handoff

Each branch's next logits must agree with a completely fresh forward of its
own exact `P || suffix`, before any suffix and after every consumed token.
Controller bytes, history, positions, CAS and authorization are exact checks.
Numerical limits come from independent frozen baselines. Actual saved bytes,
physical sharing, direct work counters, cleanup and durability need separate
measurements. Process kill is not power-loss certification.

Implement all 23 absent registered methods with the existing required names
and assertions. Preserve >=1000 programs in G5-002 and real long-soak assertions
in G5-004; their exhaustive runs stay deferred. Never mark missing/skipped/
xfail cases, mocked GPU, absent instrumentation or absent oracle as gate PASS.
Evaluator subprocesses and storage permissions only establish the isolation or
at-rest guarantees actually enforced and tested. Block unsupported stronger
claims; do not create flags or alternate histories to hide failures.

Update STATUS and DECISIONS after each ticket with observed facts. Final output
must list completed tickets, remaining implementation/blockers, unexecuted
certification cases, actual numerical maxima and work/memory measurements,
backend/build/model fingerprints, artifact locations and restart commands.
Do not stop after reading or after a plan: implement the next available ticket
and continue through independent work within the stated limits.
