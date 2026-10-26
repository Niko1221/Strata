# Parked conversation cache deduplication

## Enable or disable

Set `STRATA_PARKED_DEDUP=1` in the engine's environment and restart it to enable.
Unset the variable, or set it to `0`, and restart to disable. Both modes use the
same binary. The default is disabled. Changing the environment of a running
process does not convert its existing cache. A restart starts an empty cache.

The startup log and `/metrics` under `conversation_cache.dedup.mode` report the
mode. Enabled diagnostics also report `logical_bytes`, `physical_bytes`,
`saved_bytes`, `metadata_bound`, `accounted_bytes`, `allocation_peak`, `hits`,
`cow` and `index_entries`. Payload counters measure allocation capacities across
live snapshot references, including retained, active-checkpoint and temporary
state. They are not process RSS; measure RSS separately to include allocator
retention and other engine memory. With the feature enabled, the cache's
`bytes` figure is this process-wide ledger plus metadata, not the sum of the
parked entries.

## Ownership and scope

Enabled mode stores parked KV in fixed 1 MiB blocks and shares immutable block
and checkpoint payloads. A hash selects candidates; a complete byte comparison
is required before independently captured payloads share storage. Equal token
IDs alone never establish captured-state equality. Mutable access detaches
before writing; a shrinking detach copies only the bytes it keeps. Active device
execution stays private.

Checkpoint identity, image/steering matching and retention metadata stay private.
Borrowed donor references preserve the existing dirty-token, page, index-row and
speculative-tail boundaries; `STRATA_SNAPSHOT_FULL_CAPTURE=1` disables them as it
disables other retention. The intern index holds weak references and every
payload removes its own entry when released, so the index never holds dead
entries and needs no sweep or size cap.

Enabled serving is restricted to a single GPU and serial requests. Batch and
layer-split serving are rejected. Disabled mode preserves the original segmented
storage, capture estimate and eviction policy and avoids hashing/interning.

## Admission

The enabled budget is checked against the payload ledger, which already holds a
taken (held) image, retained K/V and active checkpoints. Admission therefore:

- ignores `held`, which the disabled policy adds separately;
- estimates a capture as the storage it adds: blocks from the one holding the
  retained prefix onwards, new directories and the live running state.
  Checkpoint copies of frozen payloads are references and cost no payload;
- plans oldest-first evictions from what each entry would actually release
  under the ledger's own rules: payload (a shared block only with its last
  owner), entry metadata and the per-wrapper and per-index-node allowances;
  retained index buckets stay charged. It evicts nothing when the image cannot
  fit, the same refusal the disabled policy makes up front;
- in `put()`, plans the superseded copies of the incoming conversation as the
  first removals and drops them only once the image is admitted. A refused
  replacement keeps them. (The serving path still drops superseded copies before
  its capture estimate, as in the original policy.)

Pinned prefixes and a guarded borrow donor are never evicted, as before.

The physical-RAM gate uses the same addition. glibc raises its mmap threshold
after freeing large chunks, so freed 1 MiB blocks can stay in the heap and not
show up in `MemAvailable`; enabled mode calls `malloc_trim(0)` before the gate's
second sample when the first one fails (budget evictions just before the gate
may still sit in the heap), after each gate eviction and in session-save
reclaim, before re-reading memory. This costs roughly 80 ms per GB released and
only runs when the gate would refuse.

## Portable checks

From the repository root:

```sh
cmake -S . -B build -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build build --target conversation_storage_test conversation_cache_test
ctest --test-dir build -R "conversation_storage|conversation_cache"
```

ctest runs the ownership fixture in disabled mode (`conversation_storage_test`)
and in enabled mode (`conversation_storage_test_dedup`).
`conversation_cache_test` checks the original segmented layout and is a
disabled-mode suite. To additionally force all candidate hashes equal, build
the fixture with `-DSTRATA_DEDUP_TEST_HASH_COLLISION` and run it with
`STRATA_PARKED_DEDUP=1`; never define that macro in a production build. The
fixture covers byte-identical captures, checkpoints, private writes across
block boundaries, random append/rewind, donor eviction, last-reference
release, cache pressure, external references, injected allocation failures,
the held/retained admission order of a park, refusal without eviction,
shared-payload eviction planning, metadata allowances at the budget boundary,
superseded copies kept on refusal, guarded donors, COW-free inspection and
index release.

The GPU snapshot and host validation suites run in both modes. In enabled mode
their capture-estimate checks compare the ledger's growth, including its
transient peak, with the estimate; disabled mode keeps the image-size check.
Partial immutable blocks may relocate during growth; full-vs-incremental byte
equality, donor isolation and restored device-state checks are the contract.

## Validation

Qualified on one Linux single-GPU CUDA deployment (Release build): the
ownership fixture passes in disabled mode, in enabled mode, with forced hash
collisions, and under ASan+UBSan; the original cache-policy suite passes with
the feature disabled, and the GPU snapshot and host validation suites pass in
both modes. Controlled full-model probe sets with identical prompts against
the same engine with the feature off and on produced identical outputs and
identical internal cache-state fingerprints, including rewinds and
conversation switches. Under a tight cache budget, a synthetic workload of
many conversations sharing one long prefix retained more parked histories
with the feature enabled at unchanged park/restore latency.

Not yet exercised: other backends (HIP/SYCL/Metal), Windows, and
maximum-context histories. Batch and layer-split serving are rejected by
design.
