# Native build migration to primary main

The Windows CUDA cache was reconfigured for the physical primary checkout at
`C:/Users/downl/Documents/ChatGPT/Strata`. The deleted H: worktree's old native
receipt is retained as historical evidence; it is not relabelled as a C: build.

The five-job full build passed at the C: source in 1388.98 seconds. Its owned
Job exited 0 and drained. Available physical RAM remained at least
8,873,119,744 bytes, above the unchanged 3 GiB floor. The compiler/configuration
and actual vendored ggml identities remain unchanged. CUDA is 13.3.73, MSVC is
19.44.35224.0 and the selected architecture is sm_120. The measured GPU is an
RTX 5060 Ti 16 GB with driver 621.14; this build does not establish general GPU
compatibility or concurrent inference quality.

After the reviewed scalar-float RoPE correction at
`c297453fe1dfe800a71a038f6445c10eeb5234eb`, another five-job build passed in
100.60 seconds, with exit 0 and no remaining owned processes. Relative to the
initial integrated main, the bound native source changes are confined to
`src/image/te/qwen2_tok_main.cpp` and `src/image/dit/qi21_rope.cpp`. The former
handles CRLF text merges; the latter retains separately evaluated float trig
calls. See the separate tools and Windows RoPE records for validation limits.

The final resource capture read nine actual rebuilt CUDA libraries. It found
451 nonzero STACK/LOCAL entries. Moving the source changes 445 mangled anonymous
namespace identifiers. Demangling both complete entry sets and keying by library,
object, SM and full function signature produces a one-to-one match for all 451.
There are no added or missing entries and no changed STACK/LOCAL amounts. The
comparison rejects duplicate normalized keys. This is a comparison of measured
resource amounts, not a numerical or performance benchmark.

The explicit profile is rebound to source digest
`8bc0448a35e217052eb0537285c7e38e724f0422a02f33f30554d2fb07d6a848`.
Its compiler key remains
`3d02d9aa3a340546a098972927f76942ce6b1bb54348a6699d26fc9dc9195d65`,
and its ggml identity remains
`vendored-sha256:00d79121e61703168b5f335234d8b391edd3192560b5bc98216a9c8fe8401af1`.
The old raw profile is preserved privately alongside the new measurements and
the full bijective comparison.

The default ALLOWED checker remains unchanged and still fails two IQ entries:
T22 (IQ2_S) and T17 (IQ2_XS), each STACK 24 / LOCAL 0 against the default STACK
16 / LOCAL 0 allowance. These are retained findings. The explicitly selected,
source/compiler/ggml-bound Windows profile retains their previously reviewed
24-byte measurements. No generic allowance, default, or unlisted-entry budget
was raised. Unlisted entries in the bound profile still permit zero bytes.

Evidence is retained under
`H:/Strata/private-backups/cleanup-20261116/native-primary-profile-v1` and
`H:/Strata/evidence/upstream-sync-20261116/receipts`. The full build, incremental
build and final capture receipts are `native-primary-build-v1`,
`native-primary-rope-rebuild-v1` and `native-primary-profile-measure-v2`.
`native-primary-default-strict-v1` records the expected failed default gate.
The raw capture is measurement only. `native-primary-profile-replay-v2` then
passed against the selected profile, current source, compiler key and vendored
ggml identity, with exit 0 and no remaining owned processes. Independent profile
review remains required before accepting this migration.

NOT_RUN by this migration: current-source GPU numerical replay, model inference,
installer creation, installation and publication. Full-window generation,
general model quality and GPU concurrency remain unqualified;
`releaseQualified=false` is unchanged.
