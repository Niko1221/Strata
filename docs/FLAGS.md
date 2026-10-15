# Flags of the disk layer

This page lists every `strata --serve` flag this tree adds, the two functions they belong to, and the upstream
flags they are most often confused with. The names and the defaults are the exact strings the engine parses and
prints in `--help`. A flag the engine does not know is **fatal** (`unknown argument`, then `usage`, exit code 2),
so a config for an older engine must not carry these flags.

The functions themselves are described in [SPILL_AND_PROMPT_CACHE.md](SPILL_AND_PROMPT_CACHE.md). The behaviour in
prose is also in [DETAILS.md](DETAILS.md) ("Disk tier for the parked conversations" and "System-prompt prefill
cache").

## How to read the "Since" column

- **`layer/base`** — this tree's base: upstream `v0.1.41` (`fb58e0d`) plus PR #1271 (the disk tier) and PR #1269, cherry-picked onto it.
- **`layer/delta1`** — this tree's own work on top of that base (the spill tier as overflow).
- **`layer/delta2`** — the mirror at the park, compaction detection and cancellation (on top of delta 1).
- **`layer/delta3`** — the batch-MTP slots across a layer split, and the device that runs the head (on top of delta 2).
- **`layer/delta5a`** — the archive: the parking copy the tier would discard is **moved** into a subdirectory instead of removed, opt-in with the default off (on top of delta 4).
- **`aporte/cascada-t5`** — the **cascade**: an optional WARM level between the RAM cache and the disk tier, so a capacity eviction **degrades** down a level instead of deleting; the disk tier of always becomes the COLD level, the last one and the only one that removes. Opt-in by the mere presence of the warm folder, so a run that does not name it is byte-for-byte the tier above.
- **`engine <version>`** — the flag came from upstream and is unchanged here.

## Three caches, not one

Do not mix these. They share the word "cache" and little else:

1. **The parked conversations in RAM** (`--conversation-cache-mib`, `--conversation-cache-slots`,
   `--conversation-cache-min-free-mib`; upstream engine). Conversations the engine keeps in host RAM between
   requests, so a returning chat does not read its prompt again. RAM only: lost on restart.
2. **The disk tier** (`--conversation-cache-spill-dir` + `--conversation-cache-disk-mib` + the flags beside
   them). A conversation is written to a folder as an ordinary session file and read back later or after a restart.
   By default (`--conversation-cache-spill-on park`) it is a **mirror**: the state parked at the end of a request
   is written then. With `--conversation-cache-spill-on evict` it is **overflow**: only a conversation the RAM cache
   evicts reaches disk.
3. **The system-prompt prefill cache** (`--system-prompt-cache*`). The checkpoint root that ends the system prompt
   (`--prompt-cache-root`, in RAM) is persisted to its own folder as a session file and reloaded at start.

Two nearby names are **not** caches: `--kv-resident` decides **where** the context's K/V lives (VRAM or host RAM),
and `slot_save_path` is the server's manual save/restore API (the same session files, written only when a client
asks). The RAM-side prefill checkpoints (`--prompt-cache`, `--prompt-cache-every`, `--prompt-cache-tail`) are what
the system-prompt cache persists; `--prompt-cache-root` is the threshold for the root checkpoint it targets.

## A. The disk tier (conversations on disk)

The disk tier has two modes, selected by `--conversation-cache-spill-on` and detailed in the table below: the
default **mirror** (`park`, when the tier is on) writes the conversation at the park, and the **overflow** opt-out
(`evict`) reproduces delta 1 byte for byte. The folder no longer has to be named by hand: with the RAM cache on
(`--conversation-cache-mib > 0`) and a nonzero disk budget (the default 8192 MiB), the engine resolves a spill
directory on its own and turns the tier on, so the two modes act without `--conversation-cache-spill-dir`;
naming the folder only overrides the resolved one.

| Flag | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `--conversation-cache-spill-dir DIR` | empty (`<Strata dir>\slots`, then `<user profile>\.strata\slots`) | Names the disk tier's folder. A conversation the RAM cache evicts is written there as an ordinary session file (`.sess`) plus a small sidecar (`.meta`). When the flag is absent the engine resolves a folder of its own (see Scope) and logs which rule won. | `--serve` only. The tier opens when the RAM cache is on (`--conversation-cache-mib > 0`, `--prompt-cache > 0`, `--conversation-cache-slots > 0`) and `--conversation-cache-disk-mib` is nonzero; naming no folder no longer keeps it off. With the flag absent the resolved folder is `<Strata dir>\slots` (the engine's own directory) when that folder is off the system drive (`C:` on Windows), else `<user profile>\.strata\slots` - and the same profile fallback when the Strata folder cannot be created; if neither path resolves the tier stays off with a warning. The explicit flag always wins. Works with `--layer-split` (one file per stage). | No authentication: the files hold the conversation's token IDs and state, so the folder must stay private. A file from another model or another configuration (including another KV quant) is **rejected, not reused**. | `layer/base` (`v0.1.41` + PR #1271); the resolved default in `aporte/serie-integrada-t4` |
| `--conversation-cache-disk-mib N` | 8192 | The spill folder's byte budget, in MiB. The GC removes stored conversations over it, oldest first. | `--serve`; inside the disk tier. `0` disables the disk tier. This is the tier's own folder, not `--conversation-cache-mib`, which is the RAM budget. | A budget, not a disk reservation or a quota. A stored conversation larger than the whole budget is kept anyway (removing it could not bring the folder under the budget) and counted as `oversized`. | `layer/base` |
| `--conversation-cache-cold-dir DIR` | empty (= `--conversation-cache-spill-dir`) | **Canonical name of the cold level**: the same field, and therefore the same folder, as `--conversation-cache-spill-dir`. With a warm level configured this folder is the last level of the cascade — the only one that removes (FIFO by budget and, if asked, by age), and the destination a degradation from the warm level lands in. | `--serve` only. Alias of `--conversation-cache-spill-dir`: naming **both** with two different folders is an error at start (exit 2), and naming it writes the same field, so it does **not** add capacity. Without it and without `-spill-dir` the resolved default applies exactly as before. | An alias, not a second level. The engine logs which of the two names was used. Naming it does **not** turn the cascade on: that is `--conversation-cache-warm-dir`. | `aporte/cascada-t5` |
| `--conversation-cache-cold-mib N` | 0 (= `--conversation-cache-disk-mib`, default 8192) | **Canonical name of the cold level's byte budget** (MiB), the same field as `--conversation-cache-disk-mib`. | `--serve`; alias of `--conversation-cache-disk-mib`. Two different values in the two names are an error at start. `0` keeps the disk tier off (and with no cold level there is no cascade). | A budget, not a reservation. In the cold level the GC **removes** (it is the last level); a failed degradation from the warm level leaves the entry **in the warm level**, never lost. | `aporte/cascada-t5` |
| `--conversation-cache-warm-dir DIR` | empty (**off**) | Names the **warm** level of the cascade: the level the park writer and the RAM cache's evictions write to, and the level whose own capacity GC (budget, and age if set) **moves** the oldest conversation down to the cold level instead of deleting it. It is what makes the cascade exist. | `--serve`; only with `--conversation-cache-warm-mib > 0` and with the cold level open. **With the flag absent there is no warm level and the disk tier is byte-for-byte what it was.** There is no default: the operator names the folder. | A level, not an independent cache: it does not take part in the RAM match nor in the batch slots. It is a folder of ordinary session files with the same model/config identity as the cold level, so a file of another model is **rejected, not reused**. | `aporte/cascada-t5` |
| `--conversation-cache-warm-mib N` | 0 (**off**) | The warm level's byte budget (MiB). Its capacity GC **degrades** the oldest conversation to the cold level instead of removing it. | `--serve`; inside the warm level. `0` (or an empty folder) = the warm level does not open, and then there is no cascade. | A budget, not a reservation or a quota. A conversation larger than the whole budget is left alone (the existing `oversized` rule) and counted. With `--conversation-cache-spill-when-full reject` the warm level **refuses the new copy and degrades nothing**. | `aporte/cascada-t5` |
| `--conversation-cache-spill-when-full MODE` | `evict-oldest` | What the GC does at the budget. `evict-oldest` drops the oldest stored conversation to make room (what the tier always did) — or, in the warm level of the cascade, **degrades** it to the cold level. `reject` stores nothing new and removes nothing, in either level. | `--serve`; only inside a disk level. An unknown mode is refused at start. | `reject` does not bound the folder in any other way: a folder already over budget stops accepting spills and keeps what is there. In the warm level `reject` also blocks the degradation (a degradation is an eviction by another name). The age lever below is independent of this. | `layer/delta1`; the warm-level reading in `aporte/cascada-t5` |
| `--conversation-cache-spill-max-age-days N` | 0 (off) | Optional age pruning. With a positive N, the GC removes stored conversations older than N days, oldest first, counted separately from the budget GC. In the **warm** level of the cascade it **degrades** them instead: time is a retention rule of the last level, so an aged warm copy is moved down (and its stamp re-sealed on arrival), never deleted. | `--serve`; only inside a disk level. `0` means **no deletion by time at all**: this lever is off unless set. | Off by default. Independent of `--conversation-cache-spill-when-full`: both can act, each with its own counter in the log. **Behaviour change with the cascade on**: an operator who relied on this flag to bound the disk sees the warm level stop deleting by time — the cold level does it, one level later. | `layer/delta1`; the warm-level reading in `aporte/cascada-t5` |
| `--conversation-cache-spill-on MODE` | **`park`** | What the tier is there for. `park` (the default **when the tier is on**: a spill directory and a nonzero budget) also writes a conversation the moment it is parked at the end of a request, so a close loses at most the request in flight. `evict` is the delta-1 behaviour: only what the RAM cache evicts reaches disk. | `--serve`; inside the disk tier. **The default changes with the tier**: `park` needs a folder, and the engine resolves one when `--conversation-cache-spill-dir` is absent, so the mode acts as soon as the RAM cache is on (see the `--conversation-cache-spill-dir` row). An unknown mode is refused at start. | The mirror writes at the end of each request, never mid-generation, and collapses a burst of parks of one conversation (one cell per conversation holding only the newest state; the async writer drains it). The disk is bounded by `--conversation-cache-disk-mib` exactly as before. | `layer/delta2` |
| `--conversation-cache-spill-divergence-tokens N` | 4096 | A stored copy whose common prefix with the incoming prompt is shorter than N tokens, while its header (system prompt + first turn) still matches, is read as a **rewritten tail** — a compaction or an edited history — and is **discarded** (index and file) instead of being kept as an orphan. Counted in `compacted`. | `--serve`; inside the disk tier. Needs a turn token (`--turn-token`); without one nothing is ever called a rewrite. A copy the prompt **extends** is kept; one whose header differs is another conversation and is left alone. | A threshold, not a measurement: a compaction that keeps more than N tokens of common prefix is not detected and the copy is left for the GC. Prevents a compacted copy from occupying GB while it can never be a hit. | `layer/delta2` |
| `--conversation-cache-spill-park-throttle-s N` | 0 | In `park` mode, do not rewrite the same conversation inside an N-second window; the park is postponed to a later one. | `--serve`; inside the disk tier, `park` mode only. `0` = write on every park. | A rate limiter, not a correctness lever: the newest state still ends up on disk, just on the next park outside the window. Counted in `throttled`. | `layer/delta2` |
| `--conversation-cache-archive-mode MODE` | `off` | What the tier does with the parking copy it would otherwise **discard** — a compaction (`--conversation-cache-spill-divergence-tokens`), the copy a re-park supersedes, and the provisional state of a cancelled request. `off` (default) removes it, byte for byte as delta 4. `ids` **moves** only the sidecar (the token/image ids, no K/V). `state` moves the session file (the K/V included) with the sidecar. A cancelled request archives the state **before** the revert (the tokens it actually produced), and always as the **sidecar only** (ids + images, no K/V) even in `state` mode. | `--serve`; only inside the disk tier, and only beside a live tier. An unknown mode is refused at start. With `off` no directory is created and no byte is written. | **The archive is never a reuse candidate**: `best()` never returns one (the attention is causal, so an archived tail can never be a hit). It is evidence and recovery, not a cache, and it does not make the post-compaction read any faster. `state` is ~30 KB/token (a 45 455-token conversation measured 1 333 MiB), so bound it with `--conversation-cache-archive-mib`. The **cancellation never stores K/V** (only the pre-revert ids), even in `state` mode. | `layer/delta5a` |
| `--conversation-cache-archive-dir DIR` | empty (`<spill-dir>\archive`) | The archive's own folder. Empty with a mode other than `off` resolves to `<spill-dir>\archive`. | `--serve`; inside the disk tier. | A subdirectory of the spill folder is fine: the tier's scan skips directories, so the archive is never counted as a stale spill file, and it is never indexed for reuse. It must **not** be the spill folder itself (refused at start). | `layer/delta5a` |
| `--conversation-cache-archive-mib N` | 2048 | The archive's **own** byte budget (MiB), separate from `--conversation-cache-disk-mib`. The archive's GC removes the oldest archived copies over it. | `--serve`; inside the archive. `0` = no byte budget (the per-conversation cap still applies). Never touches the tier's `bytes_`, `disk_evictions` or age lever. | A separate budget, not the tier's, and it is **strict**: a copy that alone exceeds the whole budget is **not archived** (it is discarded as before), so the archive never holds more than this. The archive's GC has its own counter in the log. | `layer/delta5a` |
| `--conversation-cache-archive-keep N` | 4 | How many copies of **one** conversation the archive keeps, newest first. The archive's GC removes the older ones. | `--serve`; inside the archive. `0` = no per-conversation cap (only `--conversation-cache-archive-mib` bounds it). | A count of archived copies, not of live ones: the live tier still holds exactly one copy per conversation. | `layer/delta5a` |
| `--conversation-cache-archive-max-age-days N` | 0 (off) | Optional **age retention** of the archive: with a positive N the archive's own GC removes archived copies older than N days, oldest first. Mirrors the tier's `--conversation-cache-spill-max-age-days`, but for the archive. | `--serve`; inside the archive. `0` means **no retention by time at all** (the archive is then bounded only by `-mib` and `-keep`). | The **only** time-based retention of the archive; independent of the budget/keep GC, with its own counter (`by age`) in the log. | `layer/delta5a` |
| `--conversation-cache-similarity F` | 0.0 | The least fraction of the new prompt that a stored (RAM or disk) conversation must share as a common prefix before it may be reused. | `--serve`; applies to the RAM cache and the disk tier's match. A finite number in `[0, 1)`; anything else is refused. | 0 accepts any match. It filters weak hits only: a candidate that passes still has to be an exact token/image prefix to be restored. | `layer/base` |
| `--conversation-cache-n-min N` | 0 | The least number of common-prefix tokens a stored conversation must offer before it may be reused. | `--serve`; same place as `--conversation-cache-similarity` (RAM and disk). | 0 accepts any length. A hit that offers fewer tokens than a live slot or the RAM cache already reaches is not used. | `layer/base` |

The scan never deletes: at start any file it does not understand (another model/config identity, a broken or
missing sidecar, an orphan session file, a leftover temporary) is **ignored and counted**, not removed. Only the GC
removes, by budget and (if enabled) by age, oldest first, with a counter and a log line.

### The cascade (RAM → warm → cold)

The disk tier is a **hierarchy**, not one folder, and the names say which level is which:

- **RAM** (`--conversation-cache-mib`) evicts to...
- the **warm** level (`--conversation-cache-warm-dir` / `-mib`), which is where the parking writes and **degrades** to...
- the **cold** level (`--conversation-cache-cold-dir` / `-mib`, the disk tier of always), which is the last level and
  the only one that **removes**.

An entry travels **down** the hierarchy and only on a **capacity** decision — the budget GC or the age GC. Every
**correction** discard (a compaction, a superseded copy, a cancelled request's provisional state, an image the
engine cannot read) still discards, wherever it is: a copy that can never be a hit must not be pushed into a level
that *is* a reuse candidate.

**FIFO at every frontier**, by the order the entry was *written* to that level — the same rule `evict-oldest` always
used (position in the level's own list, never last access; no reading reorders it). A degraded conversation is the
**newest** entry of the level that receives it, whatever its age above, and its stamp is re-sealed to the moment it
arrived (in memory and on the file), so the cold level's own age retention does not prune it on sight.

**Degrading is not deleting, and it is all-or-nothing.** The copy is written into the cold level *first* and removed
from the warm level *after*; if the cold level cannot take it (`reject`, no room, a failed move, no space) the entry
**stays in the warm level**. The worst case is therefore exactly the one the flat tier had (the copy the GC would
have deleted), never worse.

**Reading takes the copy from the level that has it** — no promotion on the read path, because a promotion would be a
second full write while the decode is stopped. It happens by itself: a conversation that was restored is live again,
and its next park (or the next RAM eviction) writes it back to the **warm** level.

Two consequences worth stating: the same conversation **can be in both levels at once** (it is a hierarchy, not a
set), so the bytes are read and exposed **per level and never summed**; and with `--conversation-cache-segments on`
the per-segment store lives in the **cold** folder, because it shares its directory with the tier
(`SegmentStoreConfig.directory`) — the cascade does not move it.

**Compatibility.** The cascade is opt-in by the **mere presence** of `--conversation-cache-warm-dir` with a nonzero
`--conversation-cache-warm-mib`. With no warm level: the cold level's GC removes as it always did, reads come from
the single disk tier, the `/metrics` rows are RAM + `SSD 1`, and every log line is the one of the tier above. The
**cold level inherits what the old folder already holds without moving a byte** — the engine recognises its files by
the name prefix and the sidecar's model/config identity, neither of which the cache flags take part in. It is also
why the engine version must not change in the same step: `STRATA_VERSION` **is** part of that identity, so a bump
turns the parked folder into `foreign` files (kept, never deleted, but never matched) — check `foreign kept=0` in the
start log before trusting the inheritance.

The **archive** (`--conversation-cache-archive-mode`, delta 5a) is the exception that keeps instead of removing:
where the tier would discard the parking copy — a compaction, a superseded copy, the provisional state of a
cancelled request — it **moves** it into a subdirectory. It is opt-in and **off by default**, so with the mode off
the tier is byte-for-byte delta 4. It has its **own** budget (`--conversation-cache-archive-mib`) and its **own**
GC (`--conversation-cache-archive-keep`, then the budget, oldest first, plus `--conversation-cache-archive-max-age-days`
if set), neither of which touches the live tier's `bytes_`, `disk_evictions` or age lever. It is **never a reuse
candidate** (its entries are not in the tier's index, so `best()` cannot return one): archiving recovers work and
lets a client that reverts a compaction restore it, it does **not** speed up any read.

**Risk and retention of the archive (operational note).** The archive keeps what the tier would have deleted, so it
is a **larger and longer-lived exposure surface** than the tier:

- **It is plaintext.** The stored `.meta` holds the conversation's token ids and image keys, and in `state` mode
  the `.sess` holds the whole K/V. There is **no encryption** and **no secure delete**: removal (tier GC, archive
  GC, or `--conversation-cache-archive-mode off` after the fact) is an ordinary file unlink. Treat the archive
  folder exactly like the spill folder: keep it on a private volume with restrictive permissions (ISO/IEC 27040 /
  ISO 15489-1 — records with access control and a retention rule).
- **Retention is explicit and bounded only by what you set:** `--conversation-cache-archive-keep` (copies per
  conversation), `--conversation-cache-archive-mib` (the archive's byte total) and, optionally,
  `--conversation-cache-archive-max-age-days`. With `-max-age-days 0` (the default) there is **no deletion by
  time**: an archived copy persists as long as it fits the byte budget. There is no other retention policy.
- **The byte budget cannot be exceeded by a single copy:** a conversation whose archived copy alone would exceed
  `--conversation-cache-archive-mib` is **not archived** (it is discarded exactly as without the archive), and the
  budget GC is strict on reopen. Size `-mib` for the largest conversation you want recoverable.

## B. The system-prompt prefill cache

| Flag | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `--system-prompt-cache` | off | Turns the function on. The checkpoint root that ends the system prompt — already built in RAM by `--prompt-cache-root` — is written to disk as an ordinary session file (one per stage under a layer split, with a joint sidecar), and reloaded at start so a NEW chat of the same client reads only the tokens after the root. | `--serve` only. Needs `--system-prompt-cache-dir`, `--prompt-cache > 0`, `--prompt-cache-root > 0`, a turn token and `--mtp` (without MTP it reports itself off rather than capturing a different artifact). Works with `--layer-split`, one session file per stage, and a variant stored while part of the context lived in host RAM loads back too. Off unless set: no directory is created and no byte is written. | It persists a prefix, never a conversation tail: a stored variant opens only a prompt that **begins with exactly those tokens**. It changes what is read, not what the model computes. | `layer/delta1` |
| `--system-prompt-cache-dir DIR` | empty | The cache's own folder, separate from the spill folder. | `--serve`; with `--system-prompt-cache`. Without it the function warns and stays off. | Use one folder per model/configuration: the identity check refuses a foreign file, and a folder shared across models just accumulates refused files. | `layer/delta1` |
| `--system-prompt-cache-mib N` | 2048 | The folder's byte budget. The GC trims variants over it, oldest first. | `--serve`; inside the system-prompt cache. | Budget, not a reservation. `0` means no byte budget (the variant count still applies). | `layer/delta1` |
| `--system-prompt-cache-slots N` | 2 | How many variants coexist. A variant is one exact system-prompt prefix (plus `--system-prompt-cache-key` and the model/config identity). The GC removes the oldest over this count. | `--serve`; inside the system-prompt cache. `0` means no variant cap (only the byte budget bounds it). | **A variant is never removed because the system prompt changed.** The old variant is kept and is a hit again if the client returns to it; only the GC removes one. `2` lets two prompts (for example two clients) alternate without re-reading each. | `layer/delta1` |
| `--system-prompt-cache-max-age-days N` | 0 (off) | Optional age pruning. A positive N removes variants older than N days, oldest first. | `--serve`; inside the system-prompt cache. `0` means no deletion by time at all. | Off by default. Independent of the count/byte GC; each has its own counter. | `layer/delta1` |
| `--system-prompt-cache-key STR` | empty | An optional declared identity, added to the variant key beside the prefix hash. | `--serve`; inside the system-prompt cache. | It only gives two identical prefixes distinct variants (for example two deployments that must not share one). It does not make a **different** system prompt reusable. | `layer/delta1` |

## C. Upstream flags these are confused with

| Flag | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `--conversation-cache-mib N` | 0 (off) | The RAM budget for parked conversations. The disk tier is an extension of this RAM cache, not a replacement. | `--serve`. The disk tier needs it nonzero. It bounds host RAM only. | Upstream parking is RAM-only: with the disk tier off (`--conversation-cache-disk-mib 0`, or the RAM cache off altogether), a parked conversation does not survive a restart. | engine 0.1.39 (upstream) |
| `--conversation-cache-slots N` | 4 | At most N parked conversations in RAM. | `--serve`; the RAM cache. The disk tier needs it nonzero. | Not the count of conversations stored on disk: the disk tier's index holds its own set (up to 256), bounded by `--conversation-cache-disk-mib`. | engine 0.1.39 (upstream) |
| `--conversation-cache-min-free-mib N` | 2560 | The physical-RAM floor when parking or restoring a session file. | `--serve`; the RAM cache and every disk hit, which is read into host RAM before it is restored. | A RAM park or a session save that does not leave this floor available is refused. A copy destined to disk - a conversation disk hit, the segment-tier and system-prompt-cache hits, a streamed session restore and a session save - reads under the smaller `--conversation-cache-park-min-free-mib` floor instead (section D) when this one does not hold; only when neither floor holds is it skipped and the prompt read normally. It is a preflight, not a reservation. | engine 0.1.39 (upstream); the transient-floor fallback in `aporte/serie-integrada-t4` |
| `--kv-resident N` | 0 / setup | Where the context's K/V lives: keep N cells of each QSA layer in VRAM and the rest in host RAM (KV streaming, minimum 20480). | `--serve`; the KV cache, not any conversation cache. | **Not a cache and not part of this layer.** It changes the per-turn read cost and VRAM residency; the disk tier does not. | engine 0.1.5 (upstream) |
| `--prompt-cache-root N` | 2048 | The minimum system-prompt length (tokens) at which the first turn boundary is checkpointed as the root, in RAM. | `--serve`; the RAM prefill checkpoints. The system-prompt cache persists this root. `0` = no system-prompt checkpoint. | RAM only by itself: it does not survive a restart. The system-prompt cache is what makes the root durable. | engine 0.1.39 (upstream) |
| `slot_save_path` (config; `--slot-save-path DIR` on the server) | empty (off) | The folder for the manual `POST /slots/0?action=save` and `?action=restore` API. A saved file is the same session format a spill writes. | The server, not the engine. Needs the model loaded, a single slot, and no `--batch`. | An explicit client action, not a cache: nothing is written unless a client asks, and nothing deletes the files. | engine 0.1.40.1 (upstream server) |

## D. The parking destination (`--conversation-cache-park`, `--conversation-cache-park-min-free-mib`)

`--conversation-cache-mib` makes the engine park a conversation in host RAM between requests. These two flags
decide **where a parking capture goes** when the physical-RAM admission gate is read, and the smaller floor a copy
destined to disk may take. The rule is one pure function (`conversation_park_placement` in
`include/strata/core/conversation_memory.hpp`): the capture is kept in RAM while the parking floor
(`--conversation-cache-min-free-mib`) holds, spilled to disk while the transient floor
(`--conversation-cache-park-min-free-mib`) holds and the disk tier is on, and **discarded** - and counted
(`discards=N` on the refusal line) - only when neither holds. Before this tree a capture that met neither floor was
discarded even with the disk tier on, so a prompt already read was paid for and thrown away.

| Flag | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `--conversation-cache-park MODE` | `ram` | Where a parked conversation goes. `ram` (the default, the behaviour before the flag): the capture stays in host RAM and, with `--conversation-cache-spill-on park`, is also mirrored to disk. `disk`: the capture is handed to the asynchronous spill writer and the RAM copy is freed at once, so the RAM cache does not grow for it and the park takes none of `--conversation-cache-slots`. `auto`: resolved once at start, after the arena and the model are resident, from the free RAM (`conversation_park_fits_ram` - one snapshot per slot beside the parking floor); the choice is logged as `park target auto -> ram\|disk`. | `--serve`; the parking capture. The CLI accepts exactly `ram`, `disk` and `auto`: the engine's internal enumerator for the third is named `automatic`, but that string is not accepted and is refused at start. `disk` needs the disk tier (`--conversation-cache-spill-dir` with `--conversation-cache-spill-on park`); without it the parking stays in RAM, with a warning. A provisional park (a cancelled request) is a RAM park whatever the mode. | An operator who does not set it keeps the pre-flags behaviour (the `ram` default). In `disk` the writer takes the captured image itself, so no second copy is made; a capture the writer already holds at least as new - or one the park throttle postponed - is not queued, and is counted. The `ram` and `auto` modes were **not** run live when this was added; the measured configuration ran `disk`. | `aporte/serie-integrada-t4` |
| `--conversation-cache-park-min-free-mib N` | 512 | The **transient** physical-RAM floor a copy destined to disk may take, as opposed to `--conversation-cache-min-free-mib` (the floor a copy that stays in RAM must leave). It is also the floor a rejected capture falls back to before it is discarded. `0` requires only the snapshot's own size. | `--serve`; every disk-destined path: a parking capture that spills, the conversation disk hit, the segment-tier and system-prompt-cache hits, a streamed session restore and a session save. | A second floor, not a second budget: it goes through the same admission helper, so a capture is dropped only when neither floor holds, and the drop is counted. Refusing a disk-destined capture does not save RAM - the next turn re-reads the same K/V - which is why the smaller floor exists. It is a preflight, not a reservation. | `aporte/serie-integrada-t4` |

## E. Upstream mechanisms this layer lives beside (coexistence, not fusion)

Three upstream things touch this layer without being part of it. Each keeps its own meaning here: no flag of this
layer bends to it, and it does not bend to them.

- **`pin=N` (the runtime pin) and `--system-prompt-cache*`.** `pin=N` retains a conversation in RAM - a runtime
  mark upstream does not persist; the system-prompt cache writes a durable session file with its own
  model/configuration identity. Both hook the same prompt boundary (the root position), and neither flag changes
  the other's behaviour. The pin is runtime-only: **it does not survive a restart or a restore** - after either,
  send `pin=N` again.
- **`--peer-device` and `--head-device`.** The peer expert tier (upstream `v0.1.41`) takes the trailing cards for
  expert streaming; `--head-device` reorders the split's stages so the chosen card runs the head. They are two
  claims about the card order, validated independently on every start - with a split, with a peer tier, or with
  neither - and neither moves for the other in this tree.
- **`STRATA_KV_GROW_HOLD` and the disk tier.** The hold grows the KV pool from the longest conversation the RAM
  cache still holds; the spill tier removes conversations from that cache, so it changes what the hold sees. The
  interaction is **untested** - the host-only tests do not cover it.

What has been verified against this base (build, flags, tests, and a live `--serve` run of the four scenarios the
battery covers, with the layer on) and what has not (a run with the layer **off** to compare against, and the flags
that battery never reached - `--head-device`, a layer split, `--batch-mtp`, cancellation, the GC's age and budget
levers, compaction): [SPILL_AND_PROMPT_CACHE.md](SPILL_AND_PROMPT_CACHE.md), "Validation environment", "Measured
benefit" and "Non-claims".
## F. The continuous agenda (delta 4 / node D4-7, phase 1)

Node D4-7 of the Delta-4 plan. These are **env knobs, not CLI flags**: the engine reads the environment, so a
config for an older engine never carries an unknown argument and nothing here needs an anchor in
`upstream.lock`. All of them are opt-in and **default off**: with `STRATA_BATCH_AGENDA` unset the read, the
interleave and the `strata batch:` line are the base tag's, byte for byte (the pure decisions live in
`include/strata/core/agenda.hpp` and are exercised by `tests/core/agenda_test.cpp`).

What it changes, in one paragraph: a prompt read while slots are decoding used to advance in full `--prefill`
chunks (up to 8192 tokens) and, between them, run the slots' windows for a share of the **time** the chunk
took (`STRATA_BATCH_DECODE_SHARE`, a fraction of wall time, default 0.5). With the agenda on, the read advances
one **fine** chunk per turn, the decode window of the active slots goes **first**, and the per-step budget is
by **tokens**: the decode rows are reserved first (one per active slot, two under `--batch-mtp`, capped at
`kVerifyMaxT` = 8) and the read takes the remainder. A client that goes away is noticed every admission chunk
instead of every full chunk, and an aging floor keeps a busy slot from starving the read.

| Knob | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `STRATA_BATCH_AGENDA` | unset / `0` (**off**) | Turns the continuous agenda on. With it on the admission read advances one fine chunk per turn, the decode window runs first, and the step budget is by tokens. | `--serve` with `--batch`. Inert without `--batch`, under `--batch-groups` (the pipelined path), and when a layer split has no active slot (nothing to decode-first: the read stays one pass). | It is **not** a mid-window row insert: `batch_step` is untouched and a window is rebuilt from the active slots every step (the verifier's "a window keeps every row" invariant is unmoved). It does **not** change the prompt arithmetic and does **not** raise the slot count (still `--batch 4`, rows capped at `kVerifyMaxT`). | `layer/delta4-d7` (phase 1) |
| `STRATA_BATCH_AGENDA_CHUNK` | `512` | The admission chunk `C_adm`: the read advances this many tokens per turn, so `should_stop` is honoured every `C_adm` tokens (down from the full `--prefill` chunk). | With `STRATA_BATCH_AGENDA` on. `0` keeps the full chunk (the base granularity). | A granularity, not a budget: it does not change what the model reads, only how often the read yields. It is applied to the read's chunking; the chunks still partition the same prompt. | `layer/delta4-d7` (phase 1) |
| `STRATA_BATCH_AGENDA_BUDGET` | `520` = `C_adm` + `kVerifyMaxT` | The per-step **token** budget: the decode rows are reserved first and the read gets the remainder. Replaces the time share when the agenda is on. | With `STRATA_BATCH_AGENDA` on and `STRATA_BATCH_DECODE_SHARE` **not** set. | A token budget, not a time share. At the default it never binds (the read keeps its full chunk); a value at or below the decode rows makes the read rely on the aging floor. | `layer/delta4-d7` (phase 1) |
| `STRATA_BATCH_AGENDA_AGE` | `4` | The aging threshold: after this many consecutive turns with no read advance, the read is granted its full admission chunk and that turn's window is skipped. | With `STRATA_BATCH_AGENDA` on. `0` disables the aging *grant* (a 1-token floor still keeps the loop live). | A **floor of progress**, not a scheduler: it bounds a starvation the token budget could cause; it does not change the window's composition. | `layer/delta4-d7` (phase 1) |
| `STRATA_BATCH_DECODE_SHARE` (existing) | `0.5` | Unchanged. When it is **set explicitly**, the engine keeps the base tag's time-share reparto **even with the agenda on** (the compatibility path the plan asks for). | `--serve`. With it set, `STRATA_BATCH_AGENDA_BUDGET` does not apply. | Still a fraction of the chunk's wall time, applied between chunks - it is not a token budget. | `layer/base` (unchanged) |

## G. The per-segment tier (layer/segmentos-t3, opt-in)

The segment tier replaces "one growing `.sess` per conversation" with independent, content-addressed segment files:
each phase of a turn - the systemprompt(s), a file read, a file write/edit, a tool call and its result, the thinking
block, the answer, the user turn - is its **own file**, named after the content address of the segment, and the
conversation is the **ordered list of references** (the manifest `strata-conv-<serial>.segs`). The state is the
**sum of the segments**, not a file that grows. An identical segment (the systemprompt, above all) is stored **once**
and shared, with a **refcount** so the GC never removes a segment another conversation still references. The address
folds in the segment that precedes it (`seg_id = H(previous ‖ kind ‖ domain ‖ tokens ‖ role_token ‖ model ‖ config)`),
which is what makes the dedup correct under causal attention: a segment is reused **only** when its whole prefix chain
matches.

**Status / the D-7 inertia.** The module (`include/strata/core/conversation_segments.hpp`,
`src/core/conversation_segments.cpp`) is implemented and covered by the host-only test
`src/core/conversation_segment_test.cpp`. The engine **parses the three flags below** and, with
`--conversation-cache-segments on` beside a live spill tier, opens the store on the **same** spill directory as the
`.sess` tier. With every flag absent the tree behaves exactly as the ported layer, byte for byte (D-7). It is now a
**writer and a reader**: at the point the disk tier writes a conversation (the eviction, and the park mirror) the
parked image is split into its phase segments by `segment_plan_split()` (a `sys` segment at the first `--turn-token`,
a prefix/`Other` segment, and the `effort+answer` tail at the `<|im_start|>` opener of the last `--tail-role-token`
turn - the frontier is anchored at the opener, not the role token, because the engine checkpoints at the opener
(measured: opener `248045`, role `system` `8678` at opener + 1)), and `store_image()` writes
each NEW segment as its own content-addressed `<seg_id>.seg` and the manifest `strata-conv-<serial>.segs` last. The
restore path consults `best()`/`load()` exactly as the `.sess` tier does and reads only the segments `0..k`. The
writer slices **one segment at a time** and skips any segment already on disk (idempotent), so a growing chat pays
only for its new tail and a shared systemprompt is written once; the whole K/V is never copied on the write path.

| Flag | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `--conversation-cache-segments MODE` | `off` | Turns the per-segment tier on for a spill directory. `off` (default) is the ported layer byte for byte; `on` reads and writes the content-addressed `<seg_id>.seg` files and the manifests `<stem>.segs` beside the `.sess` copies. | `--serve`, inside the disk tier. Needs `--conversation-cache-spill-dir`; an unknown mode is refused at start. | **Off by default (D-7).** With `on` the engine splits each parked image into phase segments and writes them (`store_image`), replacing the conversation's manifest in place. A segment file is an ordinary session file: the identity rules that guard the `.sess` tier guard it too. The phase cuts are a heuristic over the token ids (`--turn-token`/`--tail-role-token`); with neither, one whole-prompt segment is written. | `layer/prod2-0441` / `layer/segwriter-t3` |
| `--conversation-cache-segment-budget-mib N` | 0 | The segment tier's **own** byte budget (MiB), separate from `--conversation-cache-disk-mib`. `0` = no own budget. The GC removes a segment only when its refcount is ≤ 1 (orphan first, then `think`/`answer`/`tool` before `read`/`write`); a shared segment is never removed. It runs at shutdown. | inside the segment tier. | A **separate** budget and GC: it never touches the `.sess` tier's `bytes_` or `disk_evictions_`, and it does not bound the `.sess` copies. It is not a disk reservation. With `0` nothing is ever removed by this GC. | `layer/prod2-0441` |
| `--conversation-cache-segment-min-tokens N` | 0 | The materialization threshold: a non-system segment shorter than N tokens is fused with the one before it (as an `Other` segment), so a short chat does not pay one file per message. | inside the segment tier, on the **write** path (`store`/`store_image`). `0` = materialize every segment. | A **policy**, not a correctness lever: it only decides how many files the chain is cut into. | `layer/prod2-0441` |

## H. The restore ladder (`STRATA_RESTORE_LADDER`, opt-in)

An **env knob, not a CLI flag** (the engine reads the environment, so an older config never carries an unknown
argument). Off by default: with the variable unset the reject path is the one that shipped - a parked image that does
not validate is discarded and the slot's own prefix is the fallback. With it on, the engine descends the image's own
running-state checkpoints largest first and restores the largest one that validates, re-reading only the tokens after
it; if none validates it falls back to the parked RAM copy. The leaf is the same validation and transfer as the
whole-image restore, so a chosen rung cannot fail later. See [SPILL_AND_PROMPT_CACHE.md](SPILL_AND_PROMPT_CACHE.md),
"The restore ladder".

| Knob | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `STRATA_RESTORE_LADDER` | unset / `0` (**off**) | Enables the prefix restore ladder: when the parked image the serve loop picked fails validation, restore the largest of the image's own checkpoints that validates (whole K/V, the checkpoint's running state, `resume` at its length) instead of discarding the image. | `--serve`, on the restore path. Inert when the image validates (no rung is needed). | Off by default. It salvages a **prefix**, never a different carve: a stage-count mismatch is logged and falls through to the RAM copy (restoring some stages and re-reading others is refused). It does **not** fix the root cause that produced the bad image. | `layer/ladder-t3` |

## I. Windows chunked large pages (`STRATA_ARENA_LP_CHUNKS`, `STRATA_ARENA_LP_CHUNK_MIB`, `STRATA_NO_LARGEPAGES`)

**Env knobs, not CLI flags.** The expert arena is reserved as **one** `VirtualAlloc(MEM_LARGE_PAGES)` of the packed
layout; on Windows that single contiguous request fails with `1450` (`ERROR_NO_SYSTEM_RESOURCES`) once the
large-page pool cannot give one piece (the pool fragments), even though the same total fits in several pieces. These
knobs change only the Windows reserve path in `src/core/pinned.cu`; nothing changes when they are unset. See
[DETAILS.md](DETAILS.md) (the "arena reserved first" note) for the numbers.

| Knob | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `STRATA_ARENA_LP_CHUNKS` | unset / `0` (**off**) | After the one-piece `MEM_LARGE_PAGES` attempt is refused with `1450` (contiguity, not the privilege `1314`), retry the arena as **chunks**: one `VirtualAlloc(MEM_RESERVE\|MEM_COMMIT\|MEM_LARGE_PAGES)` per chunk at an explicit 2 MiB-aligned, contiguous base. The arena stays one contiguous virtual block; a chunk that cannot get large pages is halved, then falls back to 4 KB in its own slot. | Windows only (`#ifdef _WIN32`). Only after the one-piece attempt fails with `1450`; never when the privilege is missing. | Off by default: unset is byte-for-byte the old path. It is a Windows-only mechanism - Linux uses `mmap(MAP_HUGETLB)`, and the shared expert arena (#2271) is a separate `#ifdef`. The result is reported (`all N chunks … are 2 MiB`, or `partial: M/N chunks … rest 4 KB`); a chunked reserve that gets no large page at all releases and falls back to 4 KB. | `layer/arenachunks-t3` |
| `STRATA_ARENA_LP_CHUNK_MIB` | `8192` | The chunk size (MiB) the chunked reserve splits the arena into. | With `STRATA_ARENA_LP_CHUNKS` on. `0` or unset = 8192. | A sizing hint, not a correctness lever: finer chunks fit a fragmented pool (measured 8 GiB × 7 = 56 GiB beats 24 GiB × 2 = 48 GiB), but the pieces do **not** add up - 48/56 GiB are alternative totals of the same pool. | `layer/arenachunks-t3` |
| `STRATA_NO_LARGEPAGES` | unset (**off**) | Skips the large-page attempt entirely and reserves 4 KB pages, for a same-run A/B (`STRATA_NO_LARGEPAGES=1`). | Windows and Linux. | A measurement arm, not a feature: it reproduces the pre-large-pages behaviour on one machine so the 2 MiB vs. 4 KB difference can be measured without a rebuild. | `layer/arenachunks-t3` |
