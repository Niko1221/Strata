# Conversation snapshots and RAM cache

[Issue #57](https://github.com/Niko1221/Strata/issues/57) preserves reusable state
when independent agent conversations alternate on one server. Requests remain
serial, require no client session markers, and restore into existing allocations.
The shared core and RAM policy land first; the optional NVMe tier is a dependent
integration, following the [maintainer's requested order](https://github.com/Niko1221/Strata/pull/52#issuecomment-5898176448).

## Ownership and matching

- `conversation_snapshot.hpp`, `conversation_snapshot.cpp`, and
  `conversation_state.cpp` define complete capture, validation and restoration.
  They contain no retention policy or filesystem operations.
- `ConversationCache` owns inactive snapshots, selects the longest exact token
  prefix with matching image/grid identity and steering state, and evicts the
  least recently active entries. Active state wins ties.
- The serve loop parks before a switch or checkpoint rewind overwrites state.
  A continuing request does not trigger a snapshot. Checkpoint root/LRU behavior
  from upstream is retained inside each conversation.

Snapshots include live and checkpoint GDN/PLE state, page-rounded used main KV,
indexer pooled rows/tail/spare key/block position, and used draft KV. They exclude
scratch recomputed before use. Restoring an early checkpoint reconstructs the
spare pooled row and PLE token history. Streamed KV uses authoritative host pools;
restore invalidates replaceable VRAM maps and refills the draft ring. Device
addresses remain stable for captured graphs.

FP16, INT8, Q4 and identity-layout K8V4 are represented explicitly. K8V4 stores
INT8 K/scales separately from Q4 V; it does not use the block movers' format enum.
Hybrid streaming/ring layouts and whole-conversation layer-split parking are
rejected. Ordinary upstream layer-split checkpoints remain available.

## Capacity and failure handling

`--conversation-cache-mib` defaults to zero (disabled),
`--conversation-cache-slots` to four, and `--conversation-cache-min-free-mib` to
2560. `--prompt-cache 0` disables parking and produces a warning when requested.

The byte budget counts vector capacity, retained checkpoints, and incoming
snapshots held during an exchange; it is additional to the active session and
is not an exact RSS limit. Estimate before allocation, then check physical RAM
using Linux `MemAvailable` or Windows `GlobalMemoryStatusEx`. Unknown telemetry
rejects admission. Check the free-RAM floor again after capture. These samples
are not reservations or cgroup/job-limit enforcement.

Oversized entries and allocation/admission failures skip parking. Validate the
entire incoming image before modifying the outgoing session; invalid images
are discarded and ordinary prefix selection/prefill remains available. Transfer
or synchronization failures may leave partial GPU state, so the engine reports
an error and exits rather than continuing inference. Publish resume metadata
only after successful restore. Validation is not a transactional GPU rollback.

## NVMe integration contract

The disk adapter must consume this representation and restore through this core,
without another state-copy implementation. Spill immutable snapshots on RAM
eviction; disabled persistence performs no filesystem operations. A disk hit
competes with active/RAM prefixes and must reserve an explicit bounded staging
allocation plus physical RAM headroom before reading.

Persist a versioned portable envelope, not native C++ objects. Bind exact weights,
tokenizer, steering configuration, geometry, KV format (including K8V4) and state
schema. Read, integrity-check and validate the same staged bytes before applying
them. Atomic publication must reject interrupted, truncated, corrupt and foreign
entries. A failed spill may drop the already-evicted entry; it must not exceed
RAM limits or stop inference. Eviction-only persistence does not promise that
the latest active turn survives a crash. Files contain conversation content.

The adapter is being integrated from @maedoc's #52; full-model disk restart,
compatibility and eviction/promotion tests remain required.
Windows admission coverage includes @midhatn's contribution, preserved with its
original authorship. See the [validation record](shared-conversation-upstream-validation.md)
for results, commands, and hardware limits.

The dependent NVMe branch now provides `conversation_file.hpp`: a little-endian
envelope with a SHA-256 footer covering header and payload, and a full-content
asset/settings identity. It rejects #52's experimental native-struct v3 files.
It writes the shared image directly and decodes into one admitted image; it has
no CUDA apply walk. Prefix lookup seeks past payloads and returns an untrusted
candidate, which must be decoded, integrity-checked and core-validated before use.
The codec follows the streaming-envelope approach contributed by Marmaduke
Woodman (@maedoc); his original branch remains preserved for attribution.

`ConversationStore` owns `strata-conversations-v1` inside the configured directory.
An exclusive process-held lock prevents concurrent writers and releases on crash.
Files are flushed and synced before atomic rename; Linux also syncs the directory.
Byte and entry quotas cover all identities, including the in-progress file.
Oldest-use eviction makes room before writing, while a selected promotion candidate
can be protected from eviction. Failed spills are dropped. Reopening removes
interrupted temporary files and applies changed quotas. Unmanaged files are left
alone. Linux creates owner-only files/directories; Windows inherits directory ACLs
and its implementation remains untested here.

The RAM policy provides a synchronous eviction callback and a staging reservation
that shares its byte budget without consuming a parked slot. Neither promotion nor
cache destruction calls the spill callback. A protected disk hit may cause a spill
to be declined when disk space is tight; RAM eviction must still proceed.

Build these optional components with `STRATA_ENABLE_CONVERSATION_DISK=ON` (requires
OpenSSL Crypto); default builds have no new dependency. Runtime identity collection
and serve-loop promotion wiring are still pending, so this branch does not yet
provide a usable disk cache. The staging bound covers vector storage and a 64 KiB
codec allowance, not allocator or process RSS overhead.
