# Conversation cache on disk (`--conversation-cache-dir`)

Status: **draft, not built or measured yet.** The code is in `include/strata/core/conversation_disk.hpp`,
`src/core/conversation_disk.cpp`, the hooks in `include/strata/core/conversation_cache.hpp` and the
`--serve` loop of `src/program/generate.cpp`; tests in `src/core/conversation_disk_test.cpp`.

## Why

`--conversation-cache-mib` parks the conversations a request switches away from in host RAM. On
Qwen3.8-Flash-Next (IQ3_S, `--kv int8`, `--mtp`) a parked conversation costs about **19 KB per token plus
about 0.5 GB** (running state and checkpoints): 80K tokens are 2.1 GB, 114K tokens 2.7 GB (measured from
the `parked ... snapshot_bytes=` log lines). On a 62 GB host that also holds 40 GB of expert weights,
RAM fits one or two such conversations. An agent workload with more than that many long sessions
alternates and every switch reads 30-120K tokens of prompt again: 30-70 s.

A local NVMe drive reads 2.5-7 GB/s. Keeping the parked images there makes a switch back cost about one
second of file reading plus the ~0.2 s RAM-to-GPU restore that the cache already does, and lets tens of
conversations stay restorable. Because the files outlive the process, conversations also **survive an
engine restart** (same model, same settings, same layer split).

## What a parked conversation holds

`SavedConversation` (`conversation_cache.hpp`), captured by `conversation_snapshot_save`:

- `live`: the running state at the conversation's end - the GDN recurrences and conv histories of the
  linear-attention layers, each QSA layer's indexer tail (`tails`, `dead`, `block_pos`) and the PLE
  history - plus its token ids and picture keys.
- `checkpoints`: the same running state at earlier points of the chain (the system-prompt root, the last
  turn boundary, every `--prompt-cache-every` tokens), ~118 MB each on this model.
- `kv`: the positional state - each full-attention (QSA) layer's K/V pages in identity layout with its
  scales and pooled indexer rows, then the MTP draft layer's K/V.
- With a layer split: `stage_images`, one image per later stage holding **that stage's** layers
  (`layer_lo`, `layer_hi`), its own running state and checkpoints and its K/V. The drafter's K/V is
  saved once, with the image of the stage whose GPU runs the drafter.

Everything is plain host memory (`ConversationBuffer`, 16 MiB segments). Nothing in a parked image
refers to device addresses.

## Why the file SAVE / RESTORE refused a layer split

The manual session files (`POST /slots/0?action=save|restore`, engine `SAVE`/`RESTORE`) refuse
`!stages.empty() || multi_gpu`. Format v1 (`conversation_file.cpp`) serializes one image: the payload
has one layer range, and `stage_images` is not part of it; a checkpoint that still carries `stage_parts`
is refused outright. The SAVE path also streams the K/V straight from the main session's pools
(`conversation_snapshot_sources`) and the RESTORE path writes back into the main session only.

The disk tier does not need a new payload format: a parked split image is already **one v1-compatible
image per stage** (the checkpoints were split apart into the stages by
`conversation_checkpoints_split` before the capture). So each stage image is written as its own v1
session file, and a small index file binds them into one conversation. Reading returns the same
`SavedConversation` with `stage_images` filled in, and the existing request path validates and
restores it stage by stage, as it does for one taken from RAM.

## Design

### Files

```
<--conversation-cache-dir>/<model fp>-<config fp>-<layout key>/
    <id>.s0.strses     stage 0 (the main session; the drafter's K/V when it runs there)
    <id>.s1.strses     stage 1 ... one per stage, session file format v1
    <id>.idx           the index: written last, it commits the conversation
```

- **Subfolder per identity and layout.** The model fingerprint and the config fingerprint are the ones the
  session files use (`session_identity` in `generate.cpp`: sampled model files by role; rope, K/V format,
  engine version, control vector, arithmetic switches). The layout key hashes each stage's layer range.
  Another model, engine version or `--layer-split` carve gets its own folder and never parses foreign
  files. Other folders are reported at start-up and left alone.
- **Stage files** are written by `session_file_write`: aligned 16 MiB blocks, `O_DIRECT` where the
  filesystem accepts it (no page-cache pollution of the mmap'd model), a payload hash, a hidden temporary
  name renamed over the target, file and folder flush, a free-space preflight
  (`--session-min-free-mib`). They are read by `session_file_read` with every count checked against the
  runtime's limits before allocation (`conversation_session_read_limits`) and a RAM preflight.
- **Index file** (`STRCDX` v1, little-endian): identity, layout key, id, LRU stamp, control-vector mode,
  every resume point (`tokens`, a 64-bit hash of its token ids, its picture keys, live or checkpoint) and
  each stage file's size, then a hash of all of it. It is written last with the same temp + flush +
  rename + folder flush sequence.

### Matching

The tier keeps only the index in RAM (a few hundred bytes per conversation). `best()` takes the lengths
of all points whose pictures match the prompt's, hashes the prompt once in a streaming pass, and returns
the deepest point whose hash matches - the rule of `conversation_prefix` (the last prompt token is never
covered; pictures must match in order; the control-vector mode must match). A hash match is confirmed
after the read: the loaded live ids are compared with the index, and the image then passes the ordinary
`conversation_snapshot_validate` of every stage before any device write. A collision or a stale file
costs one wasted read and falls back to reading the prompt; it can never restore a wrong state.

In the request path (`generate.cpp`, after `conversations.best`), a disk match is used when it reaches at
least `--conversation-cache-disk-min-gain` (default 4096) tokens deeper than everything else on offer
(the live session, an idle batch slot, a RAM-parked conversation). It is loaded into an
`incoming` image and from there on follows the existing "take" path: the outgoing conversation is parked
with the incoming image counted as `held`, the image is restored stage by stage, its K/V is retained for
the next park. Disk matches are never borrows: the files stay on disk and are not changed by a read.

### When it writes (disk wear)

- **On eviction.** `ConversationCache::evict_oldest()` calls the tier's hook before it drops an entry
  (eviction for room or for the physical-RAM floor; not `take`, `clear` or `drop_superseded`). The hook
  writes the image unless the tier already holds it.
- **When idle.** With `--conversation-cache-disk-idle-s N` (default 20), the serve loop's `next_line`
  waits for a request at most N seconds; on timeout it writes the oldest RAM-parked conversation the
  tier does not hold yet, marks it (`SavedConversation::disk_id`) and waits again. This is what makes
  conversations survive a restart, and it moves the write cost out of the next switch. A request that
  arrives during a write waits for that one write (about a second for 2-3 GB). `0` writes on eviction only.
- **No duplicate writes.** `holding()` recognizes an image identical to a kept conversation (same points,
  same mode); `drop_superseded()` removes the copies a newer turn of the same conversation replaces
  (the rule of `ConversationCache::drop_superseded`). So a conversation alternating with others costs
  one write per park at most; a burst of switches inside the idle window writes only what RAM evicts.

### Retention

LRU by the stamp in the index (rewritten on every read, persisted across restarts), bounded by
`--conversation-cache-disk-gib` (default 64) and 256 conversations. The conversation being read is
never the victim. After non-finite logits (#879) every conversation on disk is removed, like the RAM ones.

### Concurrency and crash safety

- Everything runs on the engine's request thread: the idle writer runs only while no request, batch
  window or pipelined group is in flight (`next_line` is reached only then); an eviction write runs
  inside the park that caused it; a read runs before the outgoing conversation is parked. No device
  state is read or written by the tier, so it cannot race the GPU.
- A write is announced to the watchdog like a SAVE (`progress_allow` with `session_phase_limit_s` for
  the flushes, a beat per 16 MiB block).
- A crash between stage files and the index leaves stage files without an index; a crash during a file
  leaves a hidden `.tmp`. `open()` removes both. A damaged or foreign index drops its conversation.
- A stage file that fails its hash or the runtime's limits on read drops the conversation; a RAM or I/O
  refusal keeps it.

### RAM

RAM then only needs the active conversation(s): e.g. `--conversation-cache-mib 3072
--conversation-cache-slots 1` keeps one parked image in RAM (the one most likely to come back next) and
everything else on disk. A read needs the image's size in RAM above `--conversation-cache-min-free-mib`;
the read's RAM preflight evicts RAM-parked conversations (to disk) first when it has to.

## Flags

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--conversation-cache-dir PATH` | off | the folder (local SSD; the tier makes a 0700 subfolder) |
| `--conversation-cache-disk-gib N` | 64 | byte budget of the files |
| `--conversation-cache-disk-idle-s N` | 20 | write parked conversations after N s without a request (0 = on eviction only) |
| `--conversation-cache-disk-min-gain N` | 4096 | read from disk only for at least N more reused tokens |

The tier needs the RAM cache on (`--conversation-cache-mib`, `--conversation-cache-slots`,
`--prompt-cache`). With `--layer-split auto` the carve depends on the free VRAM at start-up; for the
files to survive restarts, give the split explicitly (`--layer-split 19`), or a changed carve starts an
empty folder.

## Log lines

```
strata serve: conversation disk: <folder>: N conversations kept from earlier runs, X of Y GiB; ...
strata serve: conversation disk: wrote N tokens (evicted|idle) in T ms, R GB/s; entries=E bytes=B evictions=V
strata serve: conversation disk: read N tokens (M reused, live|checkpoint) in T ms, R GB/s; entries=E bytes=B
strata serve: conversation disk: not written (...) / cannot read a conversation (...)
```

The server's Monitor (`ConvCacheLog`) counts them as `disk_writes`, `disk_reads`, `disk_entries`,
`disk_bytes`.

## Open points

- Reads are synchronous `pread`-style 16 MiB blocks followed by a full hash and then the existing
  host-to-device restore. Overlapping the file read with the upload (pinned double buffer per stage)
  would hide most of the restore time; not done in this draft.
- The idle writer copies nothing but holds the engine thread for one write; a request arriving then waits.
- The manual SAVE/RESTORE still refuses a layer split; it could reuse the per-stage file set.
- The idle writer writes whole conversations; a conversation that grows by a few hundred tokens per turn
  is rewritten whole each idle period (there is no delta format).
