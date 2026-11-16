# Explicit operation RAM exception preparation

The local release controllers now accept an explicit --allow-low-ram option.
Without it, operation admission and monitoring keep the 3 GiB RAM floor. The
option selects a zero RAM floor only for that invocation. It does not change
shipped model protection, the dynamic guard or saved user settings.

The controller binds the choice to the reviewed source, command, reservation
and actual owned worker identity. Consumers require those bindings and the raw
successful drained Job together. Receipts retain actual sample counts, the
minimum available RAM and the number of samples below the nominal 3 GiB floor.
The aggregate includes successful parent, initial worker and additional install
admission observations as well as every monitor sample. Child-only admissions
remain bound to the immutable worker context and required install artifact.
An arbitrary JSON flag cannot grant the exception. Commit headroom, disk
reserves, compiler/test concurrency, containment, deadlines, output limits and
GPU reserves remain unchanged.

Focused synthetic tests cover allowed below-floor execution, default refusal,
forged or changed provenance, missing samples, non-RAM floor failures and honest
nominal breach reporting. The initial five-module test attempt completed with
276 passed and one failed runtime fixture. That failure was an atomic session
save racing a durable-log scan. The repaired lifecycle and complete-snapshot
reader passed the 46-case replay and subsequent 10-case collector/consumer
replay. Canonical-log disappearance, invalid input, denied access, reparse
links and exhausted retries continue to fail. Static syntax, fatal lint and
scoped source checks passed. The failed attempt remains retained separately.
Independent review also found omitted child-only admission samples. The bound
admission correction passed 16 focused producer/consumer cases, including
worker and install minima below every parent/monitor observation and rejected
missing or forged admission evidence. Prior evidence remains retained.

This is preparation evidence. It does not establish application build,
installation, genuine model inference, model quality, release approval or
publication. releaseQualified=false remains unchanged.
