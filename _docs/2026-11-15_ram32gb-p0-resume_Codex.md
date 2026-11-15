# RAM32GB P0 baseline resume

The user explicitly requested a fresh RAM check, preservation of all protection
conditions, continuation from the P0 real-machine baseline, and P1 onward only
after measurement passes. Starting source is ecee62341d982444511dafe106e31226d371ce97;
the checkout was clean. The original plan and the prior PR00 record remain the
requirements and history. This record does not qualify a context or model.

Read the PC-wide/project instructions, canonical pre-implementation prompt and
MILSPEC skill again. The common/start gate/skill selection/application/LLMOps/
security/Python/CPP/TypeScript SOPs and incremental/performance skills already
read in this conversation continue to apply. No governance, dependency,
installed-app, original-model selection or protection weakening is authorized
by this resume.

Fresh observation: available physical RAM 9,356,648 KiB (about 8.9 GiB),
available commit 31,349,912 KiB; C: free 42,509,758,464 bytes and H: free
196,805,931,008 bytes. nvidia-smi reported 14,142 MiB GPU free and 1,911 MiB used.
No Strata/llama/q36/ds4/Hakua executable matched the scoped process inventory.
These are point observations, not memory peaks or inference measurements.

P0 diagnostic refinement: reuse the existing Windows process_memory_reader on
the retained native Popen handle. Sampled process private bytes/working set must
remain separate from system available RAM and whole-Job peak commit. Persist
samples while startup has no output so a resource stop does not erase them.
Unknown/failed samples stay unknown. This does not add CUDA calls or alter the
3 GiB RAM, 4 GiB commit, 5-8 GiB dynamic stop, 768 MiB GPU reserve or disk floors.

Measurement sequence: verify the current native binary/inputs, examine startup
under the existing guard, then run the selected Q36 baseline settings with
capacity and actual consumed input recorded separately. A 4096 startup smoke is
not the 65536 baseline. Preserve failures and raw artifacts. P1 remains gated
on actual P0 evidence; no successful result is inferred from compilation.

## Results and current gate

The user confirmed that the Q36 model had been deleted, then explicitly chose
the existing 9B model for a separate P0 track instead of downloading Q36 again.
The Q36 directory was empty. No replacement download, repack, model selection,
profile update, installation or publication was performed.

The isolated 9B admission used the existing Qwen3.5 dense Q4_K_M file,
5,629,111,584 bytes, variant local-cbca9a0eca0f30752f82f2ce. It selected the
existing installed PrismML 0.4.1 runtime. Its approved manifest digest matched
9c729c73eb050dc466203fe149a27acd5be48763a33ea2c5d2b89f78e4ddcbaa;
payload verification and full-model hashing were not run because this was an
admission-only measurement. The raw model SHA remains unknown in this result;
the previously imported metadata signature is identified as a declaration.

With fully offloaded weights and mmap disabled (the prior bounded 9B load
policy), the unchanged production admitLocalMemory calculation requires more
than 10,997,820,704 bytes of available RAM: the 5 GiB stop allowance plus the
5,629,111,584-byte model staging allowance. The final admission sample was
7,922,085,888 bytes, about 7.38 GiB. Admission was rejected. Twenty-one physical
RAM samples were saved over five seconds. No inference engine or request was
started by this measurement. Reducing context would not remove this unchanged
weight-staging admission requirement. No runtime setting was silently reduced.

P0 generation, quality, cold/warm prefill, cancellation/reload and full-window
qualification remain NOT_RUN. P1 and later implementation have not started:
the user's prerequisite of a passing real-machine baseline has not been met.
The 9B admission result is not a Q36 baseline or proof of Q36 optimizations.

Private evidence root: H:/Strata/evidence/ram32gb-p0-resume-20261115.
The dense9b-admission-r1 folder contains run_manifest.json,
phase_metrics.jsonl and report.md. The four original profile files had matching
before/after SHA-256 values during this measurement; no secret setting values
were copied into these artifacts. The outer owned Job finished with zero active
descendants and termination_confirmed=true. Its successful command receipt
means admission measurement completion, not model inference success.

## P0 instrumentation increment

The opt-in Windows bounded-run observer now samples the actual native process
through its retained Popen handle, reusing the existing process_memory_reader.
It persists private bytes and working-set samples separately from system RAM
and whole-Job commit, including during silent startup. These are sampled maxima,
not exact peaks. Failed/absent observations remain unknown. Ordinary runs do not
activate this observer, and no CUDA synchronization or numerical behavior changed.

Requested prefill=auto is accepted without inserting a numerical chunk into the
engine arguments. Comparison still requires a positive numeric effective chunk
from the actual run and the same effective chunk across both arms. Unknown or
different effective values fail comparison. Missing/unreadable artifacts now
retain a FAILED/artifact_unavailable preflight result and all three observation
artifacts without launching an engine; digest mismatches remain validation errors.

The native executable metadata probe metadata-r1 completed with observed native
exit 0 and strata-q36 0.5.9. Fourteen retained-handle samples recorded maximum
private bytes 268,017,664 and working set 177,676,288. The existing host dynamic
guard recorded no pressure, with minimum available RAM 6,127,255,552 bytes.
This is executable startup evidence only; no model was loaded by the probe.

The prepared Q36 run baseline-capacity65536-r1 requested capacity 65,536, FP32
KV, MTP2, parallel1, auto prefill, bounded5 expert slots and three short public
requests. Default warmup/graph policies were retained. It failed while opening
the deleted model, before native engine launch. The controller's exit 1 is
observed; the inference engine exit is unknown because that engine was never
started. Existing raw failure output and host guard receipt remain immutable.
The newly corrected artifact-unavailable path was tested afterward; it is not
misrepresented as having generated the earlier failed run's artifacts.

## Verification and limits

The final focused suite passed 90 tests without skips in the bounded-run,
comparison and phase-metrics modules. It includes real Windows Job containment
around CPU fake native processes, retained-handle memory samples, auto-prefill
comparability and missing-model rejection without child launch. These are not
real-model generation tests. Receipt: resume-final-pytests-r2, under the prior
PR00 evidence receipts directory H:/Strata/evidence/ram32gb-pr00-20261115/receipts.

Black and flake8 passed for the changed Python implementation/tests and the
initial measurement adapters. The first final Black check found one formatting
issue in the new exception path; it was corrected and resume-black-final passed.
The environment had pytest 8.3.5 rather than the repository's 9.1.1 pin; no
dependency changes were made. All final owned test/check Jobs drained to zero.
No C++/CUDA source changed in this resume, so the existing candidate was not
rebuilt. HIP, SYCL, hosted CI, actual 9B inference and general quality were not
run. Existing resource floors and the production guard source remain unchanged.

Task adapters live in _docs/ram32gb-p0-resume. Run them only through the existing
_docs/ram32gb-pr00/check.py owned-Job wrapper, with a fresh receipt/run name.
The finish-evidence adapter binds the receipts, final passive environment and
the incomplete gate in RESUME_RESULT.json. Raw paths/logs stay private on H:;
no performance or release qualification is claimed.
